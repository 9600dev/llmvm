import asyncio
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Callable, List, Optional, cast

import async_timeout
import click
import httpx
import jsonpickle
import nest_asyncio
import openai
import pandas as pd
import rich
from click_default_group import DefaultGroup
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import (Completer, Completion, PathCompleter,
                                       WordCompleter, merge_completers)
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from pydantic.type_adapter import TypeAdapter
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax

from helpers.helpers import Helpers
from helpers.logging_helpers import setup_logging, suppress_logging
from objects import (Assistant, AstNode, Content, DownloadItem, Message,
                     MessageModel, Response, SessionThread, StreamNode, System,
                     TokenStopNode, User)

nest_asyncio.apply()

invoke_context = None
logging = setup_logging()

global thread_id
thread_id = 0


def parse_command_string(s, command):
    parts = s.split()
    tokens = []
    skip_next = False

    for i, part in enumerate(parts):
        if skip_next:
            skip_next = False
            continue

        if i == 0 and part == command.name:
            continue

        # Check if this part is an option in the given click.Command
        option = next((param for param in command.params if part in param.opts), None)

        # If the option is found and it's not a flag, consume the next value.
        if option and not option.is_flag:
            tokens.append(part)
            if i + 1 < len(parts):
                tokens.append(parts[i + 1])
                skip_next = True
        elif option and option.is_flag:
            tokens.append(part)
        else:
            message = '"' + ' '.join(parts[i:]) + '"'
            tokens.append(message)
            break
    return tokens


async def stream_gpt_response(response, print_lambda: Callable):
    async with async_timeout.timeout(280):
        try:
            async for chunk in response:
                if chunk['choices'][0]['delta']:
                    print_lambda(chunk['choices'][0]['delta']['content'])
            print_lambda('\n')
        except asyncio.TimeoutError as ex:
            logging.exception(ex)
            raise ex


async def stream_response(response, print_lambda: Callable):
    def strip_string(str):
        if str.startswith('"'):
            str = str[1:]
        if str.endswith('"'):
            str = str[:-1]
        return str

    def decode(content) -> bool:
        try:
            data = jsonpickle.decode(content)

            # tokens
            if isinstance(data, Content):
                print_lambda(str(cast(Content, data)))
            elif isinstance(data, TokenStopNode):
                print_lambda(str(cast(TokenStopNode, data)))
            elif isinstance(data, StreamNode):
                print_lambda(cast(StreamNode, data))
            elif isinstance(data, AstNode):
                response_objects.append(data)
            elif isinstance(data, (dict, list)):
                response_objects.append(data)
            else:
                print_lambda(strip_string(data))
            return True
        except json.decoder.JSONDecodeError:
            return False

    response_objects = []
    async with async_timeout.timeout(280):
        try:
            buffer = ''
            async for raw_bytes in response.aiter_raw():
                content = raw_bytes.decode('utf-8')
                content = content.replace('data: ', '').strip()

                if content == '[DONE]':
                    pass
                elif content == '':
                    pass
                else:
                    result = decode(content)
                    if not result:
                        buffer += content
                        result = decode(buffer)
                        if result:
                            buffer = ''
                    else:
                        buffer = ''

        except asyncio.TimeoutError as ex:
            logging.exception(ex)
            # await response.aclose()
            raise ex
    return response_objects


async def get_thread(
    api_endpoint: str,
    id: int,
):
    params = {
        'id': id,
    }
    response: httpx.Response = httpx.get(f'{api_endpoint}/v1/chat/get_thread', params=params)
    thread = SessionThread.model_validate(response.json())
    return thread


async def get_threads(
    api_endpoint: str,
):
    response: httpx.Response = httpx.get(f'{api_endpoint}/v1/chat/get_threads')
    thread = cast(List[SessionThread], TypeAdapter(List[SessionThread]).validate_python(response.json()))
    return thread


async def execute_llm_call_direct(
    message: User,
    context_messages: list[Message] = []
) -> SessionThread:
    message_response = ''
    printer = StreamPrinter('')

    def chained_printer(s: str):
        nonlocal message_response
        message_response += s
        printer.write(s)

    messages_list = [Message.to_dict(m) for m in context_messages + [User(Content(message))]]
    response = await openai.ChatCompletion.acreate(
        model='gpt-3.5-turbo-16k-0613',
        temperature=0.0,
        messages=messages_list,
        stream=True,
    )
    asyncio.run(stream_gpt_response(response, chained_printer))
    return SessionThread(id=-1, messages=[MessageModel(role='assistant', content=message_response)])


async def execute_llm_call(
    api_endpoint: str,
    id: int,
    message: User,
    direct: bool,
    context_messages: list[Message] = [],
) -> SessionThread:
    thread = await get_thread(api_endpoint, id)
    thread.messages.append(MessageModel.from_message(message=message))
    thread.current_mode = 'tool' if not direct else 'direct'

    async with httpx.AsyncClient(timeout=280.0) as client:
        async with client.stream('POST', f'{api_endpoint}/v1/chat/tools_completions', json=thread.model_dump()) as response:
            objs = await stream_response(response, StreamPrinter('').write)

    await response.aclose()

    if objs:
        session_thread = SessionThread.model_validate(objs[-1])
        return session_thread
    return thread


def llm(
    message: Optional[str],
    id: int,
    direct: bool,
    endpoint: str,
) -> SessionThread:
    context_messages: List[Message] = []

    if not sys.stdin.isatty():
        if not message: message = ''
        file_content = sys.stdin.read()
        context_messages = [User(Content(file_content))]

    if not message:
        message = ''

    return asyncio.run(execute_llm_call(endpoint, id, User(Content(message)), direct, context_messages))


class StreamPrinter():
    def __init__(self, role: str):
        self.buffer = ''
        self.console = Console(file=sys.stderr)
        self.markdown_mode = False
        self.role = role
        self.started = False

    def display_image(self, image_bytes):
        try:
            # Create a temporary file to store the output from kitty icat
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                # Use kitty icat to save its output to the temporary file
                process = subprocess.Popen(['kitty', 'icat', '--transfer-mode', 'file'], stdin=subprocess.PIPE, stdout=temp_file)
                process.communicate(input=image_bytes)
                process.wait()
                # Now cat the temporary file to stderr
                subprocess.run(['cat', temp_file.name], stdout=sys.stderr)
        except Exception as e:
            pass

    def write_string(self, string: str):
        if logging.level <= 20:  # INFO
            self.console.print(f'[bright_black]{string}[/bright_black]', end='')

    def write(self, node: AstNode):
        if logging.level <= 20:  # INFO
            if not self.started and self.role:
                self.console.print(f'[bold green]{self.role}[/bold green]: ', end='')
                self.started = True

            string = ''

            if isinstance(node, Content):
                string = str(node)
            elif isinstance(node, TokenStopNode):
                string = '\n'
            elif isinstance(node, StreamNode):
                if isinstance(node.obj, bytes):
                    self.display_image(node.obj)
                    return
            else:
                string = str(node)

            self.buffer += string
            self.console.print(f'[bright_black]{string}[/bright_black]', end='')


def markdown__rich_console__(
    self,
    console: Console,
    options: ConsoleOptions,
) -> RenderResult:
    code = str(self.text).rstrip()
    syntax = Syntax(
        code, self.lexer_name, theme=self.theme, word_wrap=True, padding=0
    )
    yield syntax


def print_response(messages: List[Message], suppress_role: bool = False):
    def contains_token(s, tokens):
        return any(token in s for token in tokens)

    def pprint(prepend: str, s: str):
        markdown_tokens = ['###', '* ', '](', '```']
        console = Console()

        if contains_token(s, markdown_tokens):
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(s))
        else:
            console.print(f'{prepend}{s}')

    def fire_helper(string: str):
        if 'digraph' and 'edge' and 'node' in string:
            # fire up graphvis.
            graphvis_code = 'digraph' + Helpers.in_between(string, 'digraph', '}') + '}\n\n'
            temp_file = tempfile.NamedTemporaryFile(mode='w+')
            temp_file.write(graphvis_code)
            temp_file.flush()
            cmd = 'dot -Tx11 {}'.format(temp_file.name)
            subprocess.run(cmd, text=True, shell=True, env=os.environ)

    for message in messages:
        if message.role() == 'assistant':
            if not suppress_role:
                pprint('[bold cyan]Assistant[/bold cyan]: ', str(message))
            else:
                pprint('', str(message))
            fire_helper(str(message))
        elif message.role() == 'system':
            if not suppress_role:
                pprint('[bold red]System[/bold red]: ', str(message))
            else:
                pprint('', str(message))
        elif message.role() == 'user':
            if not suppress_role:
                pprint('[bold cyan]User[/bold cyan]: ', str(message))
            else:
                pprint('', str(message))


def print_thread(thread: SessionThread, suppress_role: bool = False):
    print_response([MessageModel.to_message(message) for message in thread.messages], suppress_role)


def invoke_context_wrapper(ctx):
    global invoke_context
    invoke_context = ctx



class Repl():
    def __init__(
        self,
    ):
        pass

    def open_editor(self, editor: str, initial_text: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            temp_file.write(initial_text)
            temp_file.flush()

            if editor == 'vim' or editor == 'nvim':
                cmd = '{} -c "normal G" -c "normal A" {}'.format(editor, temp_file.name)
                subprocess.run(cmd, text=True, shell=True, env=os.environ)
            else:
                pass
                subprocess.run([editor, temp_file.name], env=os.environ)

            temp_file.seek(0)
            edited_text = temp_file.read()
        return edited_text

    def open_default_editor(self, initial_text: str) -> str:
        return self.open_editor(os.environ.get('EDITOR', 'vim'), initial_text)  # type: ignore

    def help(self):
        ctx = click.Context(cli)
        commands = {
            cmd_name: ctx.command.get_command(ctx, cmd_name).get_short_help_str()  # type: ignore
            for cmd_name in ctx.command.list_commands(ctx)  # type: ignore
        }

        rich.print()
        rich.print('[red]Commands:[/red]')
        for key, value in commands.items():
            if key == 'message':
                key = 'message (default)'
            rich.print(f'  [green]{key.ljust(20)}[/green]  {value}')
            for option in [param for param in ctx.command.get_command(ctx, key).params if isinstance(param, click.Option)]:  # type: ignore
                rich.print(f'    {str(", ".join(option.opts)).ljust(20)} {option.help if option.help else ""}')

        rich.print()
        rich.print()
        rich.print('[white](Ctrl-c or "exit" to exit, Ctrl-e to open $EDITOR for multiline input, Ctrl-r search prompt history)[/white]')  # noqa: E501
        rich.print('[bold]I am a helpful assistant.[/bold]')
        rich.print()

    def repl(
        self,
    ):
        global thread_id
        ctx = click.Context(cli)
        console = Console()
        history = FileHistory(os.path.expanduser('~/.local/share/llmvm/.repl_history'))
        kb = KeyBindings()
        edit = False
        mode = 'tool'
        debug = False
        stream = True

        @kb.add('c-e')
        def _(event):
            editor = os.environ.get('EDITOR', 'vim')
            text = self.open_editor(editor, event.app.current_buffer.text)
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text) - 1

        custom_style = Style.from_dict({
            'suggestion': 'bg:#888888 #444444'
        })

        commands = {
            cmd_name: ctx.command.get_command(ctx, cmd_name).get_short_help_str()  # type: ignore
            for cmd_name in ctx.command.list_commands(ctx)  # type: ignore
        }

        command_completer = WordCompleter(list(commands.keys()), ignore_case=True, display_dict=commands)
        path_completer = PathCompleter()
        combined_completer = merge_completers([command_completer, path_completer])

        session = PromptSession(
            completer=combined_completer,
            auto_suggest=AutoSuggestFromHistory(),
            style=custom_style,
            history=history,
            enable_history_search=True,
            vi_mode=True,
            key_bindings=kb,
            complete_while_typing=True,
        )

        self.help()

        while True:
            try:
                ctx = click.Context(cli)

                query = session.prompt(
                    f'[{thread_id}] query>> ',
                )

                # see if the first argument is a command
                args = query.split(' ')

                if args[0] in commands:
                    command = ctx.command.get_command(ctx, args[0])  # type: ignore
                    tokens = parse_command_string(query, command)

                    result = command.invoke(ctx, **{
                        param.name: value
                        for param, value in zip(command.params, command.parse_args(ctx, tokens))  # args[1:]))
                    })
                else:
                    # default message command
                    command = ctx.command.get_command(ctx, 'message')  # type: ignore
                    tokens = parse_command_string(query, command)

                    # wire up the session to the server side thread id
                    if '--id' not in tokens and '-i' not in tokens:
                        tokens = ['--id', str(thread_id)] + tokens

                    thread = command.invoke(ctx, **{
                        param.name: value
                        for param, value in zip(command.params, command.parse_args(ctx, tokens))
                    })
                    if thread:
                        thread_id = thread.id

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                break

            except Exception:
                console.print_exception(max_frames=10)


@click.group(
    cls=DefaultGroup,
    default='message',
    default_if_no_args=False,
)
@click.pass_context
def cli(ctx):
    global invoke_context
    global renderer

    if ctx.invoked_subcommand is None:
        ctx.invoke(invoke_context)
    elif not invoke_context:
        commands = {
            cmd_name: ctx.command.get_command(ctx, cmd_name).get_short_help_str()  # type: ignore
            for cmd_name in ctx.command.list_commands(ctx)  # type: ignore
        }

        # see if the first argument is a command
        query = ' '.join(sys.argv[1:])
        args = query.split(' ')

        if args[0] in commands:
            command = ctx.command.get_command(ctx, args[0])  # type: ignore
            tokens = parse_command_string(query, command)

            command.invoke(ctx, **{
                param.name: value
                for param, value in zip(command.params, command.parse_args(ctx, tokens))  # args[1:]))
            })
            ctx.exit(0)
        else:
            # default message command
            command = ctx.command.get_command(ctx, 'message')  # type: ignore
            tokens = parse_command_string(query, command)

            command.invoke(ctx, **{
                param.name: value
                for param, value in zip(command.params, command.parse_args(ctx, tokens))
            })
            ctx.exit(0)


@cli.command('status')
def status():
    rich.print('status')


@cli.command('exit', hidden=True)
def exit():
    os._exit(os.EX_OK)


@cli.command('clear', hidden=True)
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


@cli.command('help', hidden=True)
def help():
    Repl().help()


@cli.command('act')
@click.argument('actor', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to retrieve.')
@click.option('--direct', '-d', type=bool, is_flag=True, required=False, default=True,
              help='Send messages directly to LLM without using the Starlark runtime.')
@click.option('--endpoint', '-e', type=str, required=False, default='http://127.0.0.1:8000',
              help='llmvm endpoint to use. Default is http://127.0.0.1:8000')
@click.option('--suppress_role', '-s', type=bool, is_flag=True, required=False, default=False)
def act(
    actor: str,
    id: int,
    direct: bool,
    endpoint: str,
    suppress_role: bool,
):
    df = pd.read_csv('prompts/awesome_prompts.csv')
    if not actor:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        for column in df.columns:
            table.add_column(column)
        for _, row in df.iterrows():  # type: ignore
            table.add_row(*row)

        console.print(table)
    else:
        prompt_result = Helpers.tfidf_similarity(actor, (df.act + ' ' + df.processed_prompt).to_list())

        rich.print()
        rich.print('[bold red]Setting actor mode.[/bold red]')
        rich.print()
        rich.print('Prompt: {}'.format(prompt_result))
        rich.print()

        with click.Context(message) as ctx:
            ctx.ensure_object(dict)
            ctx.params['message'] = prompt_result
            ctx.params['id'] = id
            ctx.params['direct'] = direct
            ctx.params['suppress_role'] = suppress_role
            ctx.params['endpoint'] = endpoint
            message.invoke(ctx)


@cli.command('download')
@click.argument('url', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to download and push the content to. Default is last thread.')
@click.option('--endpoint', '-e', type=str, required=False, default='http://127.0.0.1:8000',
              help='llmvm endpoint to use. Default is http://127.0.0.1:8000')
def download(
    url: str,
    id: int,
    endpoint: str,
):
    item = DownloadItem(url=url, id=id)
    global thread_id

    async def download_helper():
        async with httpx.AsyncClient(timeout=280.0) as client:
            async with client.stream('POST', f'{endpoint}/download', json=item.model_dump()) as response:
                objs = await stream_response(response, StreamPrinter('').write)
        await response.aclose()

        session_thread = SessionThread.model_validate(objs[-1])
        return session_thread

    thread: SessionThread = asyncio.run(download_helper())
    thread_id = thread.id
    return thread


@cli.command('search')
@click.argument('query', type=str, required=False, default='')
@click.option('--endpoint', '-e', type=str, required=False, default='http://127.0.0.1:8000',
              help='llmvm endpoint to use. Default is http://127.0.0.1:8000')
def search(
    query: str,
    endpoint: str,
):
    console = Console()

    response: httpx.Response = httpx.get(f'{endpoint}/search/{query}', timeout=280.0)
    if response.status_code == 200:
        results = response.json()

        if len(results) == 0:
            console.print(f'No results found for query: {query}')
            return

        for result in results:
            title = result['title']
            snippet = result['snippet']
            link = result['link']
            score = result['score']
            metadata = result['metadata']

            if not link.startswith('http'):
                link = f'file:///{link}'

            console.print(f'[bold]Title: {title[0:80]}[/bold]')
            console.print(f'    [link={link}]{link}[/link]')
            wrapped_text = textwrap.wrap(snippet, width=console.width - 6)
            wrapped_lines = '\n'.join('    ' + line for line in wrapped_text[:4])
            console.print(f'{wrapped_lines}')
            console.print(f'    [italic]Score: {score}[/italic]')
            console.print('\n\n')


@cli.command('injest')
@click.argument('filename', type=str, required=False, default='')
@click.option('--endpoint', '-e', type=str, required=False, default='http://127.0.0.1:8000',
              help='llmvm endpoint to use. Default is http://127.0.0.1:8000')
def injest(
    filename: str,
    endpoint: str,
):
    async def upload_helper():
        async with httpx.AsyncClient(timeout=280.0) as client:
            with open(filename, 'rb') as f:
                files = {'file': (filename, f)}
                response = await client.post(f'{endpoint}/injest', files=files)
                return response.text

    response = asyncio.run(upload_helper())
    rich.print(response)


@cli.command('threads')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to retrieve.')
@click.option('--endpoint', '-e', type=str, required=False, default='http://127.0.0.1:8000',
              help='llmvm endpoint to use. Default is http://127.0.0.1:8000')
@click.option('--suppress_role', '-s', type=bool, is_flag=True, required=False, default=False)
def threads(
    id: int,
    endpoint: str,
    suppress_role: bool,
):
    global thread_id
    if id <= 0:
        threads = asyncio.run(get_threads(endpoint))
        for thread in threads:
            if len(thread.messages) > 0:
                message_content = str(thread.messages[-1].content).replace('\n', ' ')[0:75]
                rich.print(f'[{thread.id}]: {message_content}')
        active_threads = [t for t in threads if len(t.messages) > 0]
        thread_id = active_threads[-1].id
        return active_threads[-1]
    else:
        thread = asyncio.run(get_thread(endpoint, id))
        print_thread(thread=thread, suppress_role=suppress_role)
        thread_id = thread.id
        return thread


@cli.command('messages')
@click.option('--endpoint', '-e', type=str, required=False, default='http://127.0.0.1:8000',
              help='llmvm endpoint to use. Default is http://127.0.0.1:8000')
@click.option('--suppress_role', '-s', type=bool, is_flag=True, required=False, default=False)
def messages(
    endpoint: str,
    suppress_role: bool,
):
    thread = asyncio.run(get_thread(endpoint, thread_id))
    print_thread(thread=thread, suppress_role=suppress_role)


@cli.command('new')
@click.option('--endpoint', '-e', type=str, required=False, default='http://127.0.0.1:8000',
              help='llmvm endpoint to use. Default is http://127.0.0.1:8000')
def new(
    endpoint: str,
):
    global thread_id
    thread = asyncio.run(get_thread(endpoint, 0))
    thread_id = thread.id


@cli.command('message')
@click.argument('message', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to send message to. Default is last thread.')
@click.option('--direct', '-d', type=bool, is_flag=True, required=False, default=False,
              help='Send messages directly to LLM without using the Starlark runtime.')
@click.option('--endpoint', '-e', type=str, required=False, default='http://127.0.0.1:8000',
              help='llmvm endpoint to use. Default is http://127.0.0.1:8000')
@click.option('--suppress_role', '-s', type=bool, is_flag=True, required=False, default=False)
def message(
    message: Optional[str],
    id: int,
    direct: bool,
    endpoint: str,
    suppress_role: bool,
):
    if message:
        if message.startswith('"') and message.endswith('"'):
            message = message[1:-1]

        thread = llm(message, id, direct, endpoint)
        if not suppress_role: StreamPrinter('').write_string('\n')
        print_response([MessageModel.to_message(thread.messages[-1])], suppress_role)
        if not suppress_role: StreamPrinter('').write_string('\n')
        return thread


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        repl_inst = Repl()
        repl_inst.repl()
    else:
        cli()
