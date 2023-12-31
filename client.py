import asyncio
import base64
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import async_timeout
import click
import httpx
import jsonpickle
import nest_asyncio
import pandas as pd
import pyperclip
import rich
from anthropic.lib.streaming._messages import (AsyncMessageStream,
                                               AsyncMessageStreamManager)
from anthropic.types.completion import Completion
from click import MissingParameter
from click_default_group import DefaultGroup
from PIL import Image
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import (Completer, PathCompleter, WordCompleter,
                                       merge_completers)
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from pydantic.type_adapter import TypeAdapter
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax

from anthropic_executor import AnthropicExecutor
from container import Container
from helpers.helpers import Helpers
from helpers.logging_helpers import (disable_timing, get_timer, setup_logging,
                                     suppress_logging)
from helpers.pdf import PdfHelpers
from objects import (Assistant, AstNode, Content, DownloadItem, Executor,
                     FileContent, ImageContent, Message, MessageModel,
                     PdfContent, Response, SessionThread, StreamNode, System,
                     TokenStopNode, User)
from openai_executor import OpenAIExecutor

nest_asyncio.apply()

invoke_context = None
logging = setup_logging(enable_timing=True)

global thread_id
global current_mode
global last_thread
global suppress_role
global timing


thread_id: int = 0
current_mode = 'auto'
suppress_role = False
timing = get_timer()


def parse_message_thread(message: str):
    def create_message(type) -> Message:
        MessageClass = globals()[type]
        return MessageClass('')

    messages = []
    roles = ['Assistant: ', 'System: ', 'User: ']

    while any(message.startswith(role) for role in roles):
        role = next(role for role in roles if message.startswith(role))
        parsed_message = create_message(role.replace(': ', ''))
        message_content = Helpers.in_between_ends(message, role, roles)
        content = message_content

        if 'ImageContent(' in content:
            content = Helpers.in_between(content, 'ImageContent(', ')')
            with open(content, 'rb') as file_content:
                parsed_message.message = ImageContent(file_content.read(), url=content)
                messages.append(parsed_message)
        elif 'PdfContent(' in content:
            logging.debug('PdfContent found, but not implemented')
            parsed_message.message = Content(content)
        else:
            parsed_message.message = Content(content)
            messages.append(parsed_message)

        message = message[len(role) + len(message_content):]
    return messages


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


def parse_path(ctx, param, value) -> List[str]:
    def is_glob_pattern(s):
        return any(char in s for char in "*?[]")

    if not value:
        return []

    if (
        (value.startswith('"') or value.startswith("'"))
        and (value.endswith("'") or value.endswith('"'))
    ):
        value = value[1:-1]

    files = []

    recursive = ctx.params.get('recursive', False)

    if not value:
        return files

    if ',' in value:
        value = value.split(',')

    if isinstance(value, str):
        value = [value]

    for item in value:
        if os.path.isdir(item):
            # If it's a directory, add all files within
            for dirpath, dirnames, filenames in os.walk(item):
                for filename in filenames:
                    files.append(os.path.join(dirpath, filename))

                if not recursive:
                    dirnames.clear()
        elif os.path.isfile(item):
            # If it's a file, add it to the list
            files.append(item)
        # check for glob
        elif is_glob_pattern(item):
            for filepath in glob.glob(item, recursive=recursive):
                files.append(filepath)
        else:
            raise click.BadParameter(f'Path {item} is not a valid file or directory')
    return files


def get_files_as_messages(
    path: List[str],
    upload: bool = False,
    allowed_file_types: List[str] = []
) -> List[User]:

    files: Sequence[User] = []
    for file_path in path:
        if allowed_file_types and not any(file_path.endswith(parsable_file_type) for parsable_file_type in allowed_file_types):
            continue

        if file_path.endswith('.pdf'):
            if upload:
                with open(file_path, 'rb') as f:
                    files.append(User(PdfContent(f.read(), url=os.path.abspath(file_path))))
            else:
                files.append(User(PdfContent(b'', url=os.path.abspath(file_path))))
        elif file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
            if upload:
                with open(file_path, 'rb') as f:
                    files.append(User(ImageContent(f.read(), url=os.path.abspath(file_path))))
            else:
                files.append(User(ImageContent(b'', url=os.path.abspath(file_path))))
        else:
            # assume it's a text file
            try:
                with open(file_path, 'r') as f:
                    if upload:
                        file_content = f.read().encode('utf-8')
                    else:
                        file_content = b''
                    files.append(User(FileContent(file_content, url=os.path.abspath(file_path))))
            except UnicodeDecodeError:
                raise ValueError(f'File {file_path} is not a valid text file, pdf or image.')
    return files


async def stream_gpt_response(response, print_lambda: Callable):
    async with async_timeout.timeout(280):
        # anthropic new messages API
        if isinstance(response, AsyncMessageStreamManager):
            async with response as stream_async:
                async for text in stream_async.text_stream:  # type: ignore
                    print_lambda(text)

            _ = await stream_async.get_final_message()
            return
        try:
            async for chunk in response:
                # anthropic completion prior to messages API introduction
                if isinstance(chunk, Completion):
                    print_lambda(chunk.completion)
                else:
                    if chunk.choices[0].delta.content:
                        print_lambda(chunk.choices[0].delta.content)
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
        except KeyboardInterrupt as ex:
            await response.aclose()
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

async def set_thread(
    api_endpoint: str,
    thread: SessionThread,
) -> SessionThread:
    async with httpx.AsyncClient(timeout=280.0) as client:
        response = await client.post(
            f'{api_endpoint}/v1/chat/set_thread',
            json=thread.model_dump()
        )
        session_thread = SessionThread.model_validate(response.json())
        return session_thread

async def get_threads(
    api_endpoint: str,
):
    response: httpx.Response = httpx.get(f'{api_endpoint}/v1/chat/get_threads')
    thread = cast(List[SessionThread], TypeAdapter(List[SessionThread]).validate_python(response.json()))
    return thread


async def __execute_llm_call_direct(
    message: Message,
    api_key: str,
    executor_name: str,
    model_name: str,
    context_messages: Sequence[Message] = [],
) -> SessionThread:
    global timing

    message_response = ''
    printer = StreamPrinter('')

    def chained_printer(s: str):
        timing.save_intermediate('first_token')
        nonlocal message_response
        message_response += s
        printer.write(s)  # type: ignore

    messages_list = [Message.to_dict(m) for m in list(context_messages) + [message]]
    executor: Optional[Executor] = None

    if executor_name == 'openai':
        executor = OpenAIExecutor(
            api_key=api_key,
            default_model=model_name,
        )
    elif executor_name == 'anthropic':
        executor = AnthropicExecutor(
            api_key=api_key,
            default_model=model_name,
        )
    else:
        raise ValueError('No executor specified.')

    timing.start()

    response = await executor.aexecute_direct(messages_list)  # type: ignore
    asyncio.run(stream_gpt_response(response, chained_printer))
    result = SessionThread(id=-1, messages=[MessageModel(role='assistant', content=message_response)])
    timing.end()
    return result


async def execute_llm_call(
    api_endpoint: str,
    id: int,
    message: Message,
    executor: str,
    model: str,
    mode: str,
    context_messages: Sequence[Message] = [],
    cookies: List[Dict[str, Any]] = [],
    append_to_server_thread: bool = True,
) -> SessionThread:

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f'{api_endpoint}/health')
            response.raise_for_status()

        thread = await get_thread(api_endpoint, id)

        if append_to_server_thread:
            for context_message in context_messages:
                thread.messages.append(MessageModel.from_message(message=context_message))

        thread.messages.append(MessageModel.from_message(message=message))
        thread.current_mode = mode
        thread.cookies = cookies
        thread.executor = executor
        thread.model = model

        if mode == 'direct' or mode == 'tool' or mode == 'auto':
            endpoint = 'tools_completions'
        else:
            endpoint = 'code_completions'

        async with httpx.AsyncClient(timeout=280.0) as client:
            async with client.stream(
                'POST',
                f'{api_endpoint}/v1/chat/{endpoint}',
                json=thread.model_dump(),
            ) as response:
                objs = await stream_response(response, StreamPrinter('').write)

        await response.aclose()

        if objs:
            session_thread = SessionThread.model_validate(objs[-1])
            return session_thread
        return thread

    except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
        if mode == 'tool' or mode == 'code':
            logging.debug('LLMVM server is down, but we are in tool mode. Cannot execute directly')
            raise ex

    # server is down, go direct. this means that executor and model can't be nothing
    if not executor and not model:
        if Container.get_config_variable('LLMVM_EXECUTOR', 'executor'):
            executor = Container.get_config_variable('LLMVM_EXECUTOR')

        if Container.get_config_variable('LLMVM_MODEL', 'model'):
            model = Container.get_config_variable('LLMVM_MODEL')

    if executor and model:
        if executor == 'openai' and Container.get_config_variable('OPENAI_API_KEY'):
            return await __execute_llm_call_direct(
                message,
                Container.get_config_variable('OPENAI_API_KEY'),
                'openai',
                model,
                context_messages
            )
        elif executor == 'anthropic' and Container.get_config_variable('ANTHROPIC_API_KEY'):
            return await __execute_llm_call_direct(
                message,
                Container.get_config_variable('ANTHROPIC_API_KEY'),
                'anthropic',
                model,
                context_messages
            )
        else:
            raise ValueError(f'Executor {executor} and model {model} are set, but no API key is set.')
    elif Container.get_config_variable('OPENAI_API_KEY'):
        return await __execute_llm_call_direct(
            message,
            Container.get_config_variable('OPENAI_API_KEY'),
            'openai',
            'gpt-4-vision-preview',
            context_messages
        )
    elif os.environ.get('ANTHROPIC_API_KEY'):
        return await __execute_llm_call_direct(
            message,
            Container.get_config_variable('ANTHROPIC_API_KEY'),
            'anthropic',
            'claude-2.1',
            context_messages
        )
    else:
        logging.warning('Neither OPENAI_API_KEY or ANTHROPIC_API_KEY is set. Unable to execute direct call to LLM.')
        raise ValueError('Neither OPENAI_API_KEY or ANTHROPIC_API_KEY is set. Unable to execute direct call to LLM.')

def llm(
    message: Optional[str | bytes | Message],
    id: int,
    mode: str,
    endpoint: str,
    executor: str,
    model: str,
    context_messages: Sequence[Message] = [],
    cookies: List[Dict[str, Any]] = [],
) -> SessionThread:
    user_message = User(Content(''))
    if isinstance(message, str):
        user_message = User(Content(message))
    elif isinstance(message, bytes):
        user_message = User(Content(message.decode('utf-8')))
    elif isinstance(message, Message):
        user_message = message

    context_messages_list = list(context_messages)

    if not sys.stdin.isatty():
        # input is coming from a pipe, could be binary or text
        if not message: message = ''

        file_content = sys.stdin.buffer.read()

        with io.BytesIO(file_content) as bytes_buffer:
            if Helpers.is_image(bytes_buffer):
                output = io.BytesIO()
                with Image.open(io.BytesIO(bytes_buffer.read())) as img:
                    img.save(output, format='JPEG')
                    StreamPrinter('user').display_image(output.getvalue())
                    bytes_buffer.seek(0)
                context_messages_list.insert(0, User(ImageContent(bytes_buffer.read(), url='cli')))
            elif Helpers.is_pdf(bytes_buffer):
                context_messages_list.insert(0, User(PdfContent(bytes_buffer.read(), url='cli')))
            else:
                context_messages_list.insert(0, User(Content(bytes_buffer.read().decode('utf-8'))))

    append_to_server_thread = True

    # if the incoming message is a thread, parse it and send it through
    role_strings = ['Assistant: ', 'System: ', 'User: ']
    if isinstance(message, str) and any(role_string in message for role_string in role_strings):
        all_messages = parse_message_thread(message)
        user_message = all_messages[-1]
        context_messages_list += all_messages[:-1]
        append_to_server_thread = False

    return asyncio.run(
        execute_llm_call(
            endpoint,
            id,
            user_message,
            executor,
            model,
            mode,
            context_messages_list,
            cookies,
            append_to_server_thread,
        )
    )


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
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                if (
                    Helpers.is_emulator('kitty')
                    and (
                        shutil.which('kitty')
                        or os.path.exists('/Applications/kitty.app/Contents/MacOS/kitty')
                    )
                    or (
                        Helpers.is_emulator('tmux')
                        and (
                            shutil.which('kitty')
                            or os.path.exists('/Applications/kitty.app/Contents/MacOS/kitty')
                        )
                        and Helpers.is_running('kitty')
                    )
                ):
                    # Use kitty icat to save its output to the temporary file
                    cmd_path = shutil.which('kitty') or '/Applications/kitty.app/Contents/MacOS/kitty'
                    process = subprocess.Popen(
                        [cmd_path, 'icat', '--transfer-mode', 'file'],
                        stdin=subprocess.PIPE,
                        stdout=temp_file
                    )
                    process.communicate(input=image_bytes)
                    process.wait()
                    # Now cat the temporary file to stderr
                    subprocess.run(['cat', temp_file.name], stdout=sys.stderr)
                elif (
                    shutil.which('viu')
                ):
                    temp_file.write(image_bytes)
                    temp_file.flush()
                    subprocess.run(['viu', temp_file.name], stdout=sys.stderr)
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
    # if we're being piped or redirected, we probably want to keep the
    # original output of the LLM, rather than render markdown.
    def contains_token(s, tokens):
        return any(token in s for token in tokens)

    def pprint(prepend: str, content: Content):
        markdown_tokens = ['###', '* ', '](', '```', '## ']
        console = Console()

        if isinstance(content, ImageContent):
            StreamPrinter('user').display_image(content.sequence)
        elif isinstance(content, PdfContent):
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown('PdfContent: ' + content.url))
        elif contains_token(str(content), markdown_tokens) and sys.stdout.isatty():
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(str(content)))
        else:
            console.print(f'{prepend}{content}')

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
                pprint('[bold cyan]Assistant[/bold cyan]: ', message.message)
            else:
                pprint('', message.message)
            fire_helper(str(message))
        elif message.role() == 'system':
            if not suppress_role:
                pprint('[bold red]System[/bold red]: ', message.message)
            else:
                pprint('', message.message)
        elif message.role() == 'user':
            if not suppress_role:
                pprint('[bold cyan]User[/bold cyan]: ', message.message)
            else:
                pprint('', message.message)


def print_thread(thread: SessionThread, suppress_role: bool = False):
    print_response([MessageModel.to_message(message) for message in thread.messages], suppress_role)


def get_string_thread_with_roles(thread: SessionThread):
    string_result = ''
    for message in [MessageModel.to_message(message) for message in thread.messages]:
        if message.role() == 'assistant':
            string_result += 'Assistant: '
        elif message.role() == 'system':
            string_result += 'System: '
        elif message.role() == 'user':
            string_result += 'User: '

        if isinstance(message.message, ImageContent):
            if message.message.url.startswith('data:image'):
                decoded_base64 = base64.b64decode(message.message.url.split(',')[1])
                with tempfile.NamedTemporaryFile(mode='w+b', suffix='.jpg', delete=False) as temp_file:
                    temp_file.write(decoded_base64)
                    temp_file.flush()
                    string_result += f'[ImageContent({temp_file.name})]\n\n'
            else:
                string_result += f'[ImageContent({message.message.url})]\n\n'
        elif isinstance(message.message, FileContent):
            string_result += f'[FileContent({message.message.url})]\n\n'
        else:
            string_result += str(message) + '\n\n'
    return string_result


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
            cmd_name: ctx.command.get_command(ctx, cmd_name).help or ''  # type: ignore
            for cmd_name in ctx.command.list_commands(ctx)  # type: ignore
        }

        rich.print()
        rich.print('[red]Commands:[/red]')
        for key, value in commands.items():
            if key == 'message':
                key = 'message [red](default)[/red]'
            rich.print(f' [green]{key.ljust(23)}[/green]  {value}')
            for option in [param for param in ctx.command.get_command(ctx, key).params if isinstance(param, click.Option)]:  # type: ignore  # NOQA: E501
                rich.print(f'  {str(", ".join(option.opts)).ljust(25)} {option.help if option.help else ""}')

        rich.print()
        rich.print(f'$LLMVM_EXECUTOR: {Container.get_config_variable("LLMVM_EXECUTOR", default="(not set)")}')
        rich.print(f'$LLMVM_MODEL: {Container.get_config_variable("LLMVM_MODEL", default="(not set)")}')
        rich.print()
        rich.print()
        rich.print('[bold]Keys:[/bold]')
        rich.print('[white](Ctrl-c or "exit" to exit)[/white]')
        rich.print('[white](Ctrl-n to clear the current line)[/white]')
        rich.print('[white](Ctrl-e to open $EDITOR for multi-line User prompt)[/white]')
        rich.print('[white](Ctrl-g to open $EDITOR for multi messages edit)[/white]')
        rich.print('[white](Ctrl-r search prompt history)[/white]')
        rich.print('[white](If the LLMVM server is not running, messages are executed directly)[/white]')
        rich.print('[white]("message" is the default command, so you can omit it)[/white]')
        rich.print()
        rich.print('[bold]I am a helpful assistant that has access to tools. Use "mode" to switch tools on and off.[/bold]')
        rich.print()

    def repl(
        self,
    ):
        global thread_id
        global current_mode
        global last_thread

        ctx = click.Context(cli)
        console = Console()
        history = FileHistory(os.path.expanduser('~/.local/share/llmvm/.repl_history'))
        kb = KeyBindings()
        current_mode = 'auto'

        @kb.add('c-e')
        def _(event):
            editor = os.environ.get('EDITOR', 'vim')
            text = self.open_editor(editor, event.app.current_buffer.text)
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text) - 1

        @kb.add('c-n')
        def _(event):
            global thread_id
            thread = asyncio.run(get_thread(Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'), 0))
            thread_id = thread.id
            rich.print('New thread created.')

        @kb.add('c-g')
        def _(event):
            editor = os.environ.get('EDITOR', 'vim')
            thread = asyncio.run(
                get_thread(Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'), thread_id)
            )
            thread_text = get_string_thread_with_roles(thread)

            current_text = event.app.current_buffer.text
            if len(current_text) > 0:
                thread_text = thread_text + 'User:  ' + current_text
            else:
                thread_text = thread_text + 'User:  '

            text = self.open_editor(editor, thread_text)
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

        def has_option(option_str, command):
            if isinstance(command, click.core.Command):
                return any(option_str in param.opts for param in command.params if isinstance(param, click.Option))
            elif isinstance(command, list):
                return any(option_str in token for token in command)
            return False

        def repl_default_option(command, tokens):
            if (
                thread_id > 0
                and (has_option('--id', command) or has_option('-i', command))
                and ('--id' not in tokens or '-i' not in tokens)
            ):
                tokens = ['--id', str(thread_id)] + tokens
            if (
                (has_option('--mode', command))
                and ('--mode' not in tokens)
            ):
                tokens = ['--mode', current_mode] + tokens
            return tokens

        command_executing = False

        while True:
            try:
                ctx = click.Context(cli)

                query = session.prompt(
                    f'[{thread_id}] query>> ',
                )

                # there are a few special commands that aren't 'clickified'
                if query == 'yy':
                    # copy the last assistant message to the clipboard
                    last_thread_t: SessionThread = last_thread
                    pyperclip.copy(str(last_thread_t.messages[-1].content))
                    rich.print('Last message copied to clipboard.')
                    continue

                # see if the first argument is a command
                args = query.split(' ')

                if args[0] in commands:
                    command = ctx.command.get_command(ctx, args[0])  # type: ignore
                    tokens = parse_command_string(query, command)
                    tokens = repl_default_option(command, tokens)

                    command_executing = True
                    thread = command.invoke(ctx, **{
                        param.name: value
                        for param, value in zip(command.params, command.parse_args(ctx, tokens))  # args[1:]))
                    })
                    command_executing = False
                    if thread and isinstance(thread, SessionThread):
                        thread_id = thread.id
                else:
                    # default message command
                    command = ctx.command.get_command(ctx, 'message')  # type: ignore
                    tokens = parse_command_string(query, command)
                    tokens = repl_default_option(command, tokens)

                    command_executing = True
                    thread = command.invoke(ctx, **{
                        param.name: value
                        for param, value in zip(command.params, command.parse_args(ctx, tokens))
                    })
                    command_executing = False
                    if thread and isinstance(thread, SessionThread):
                        thread_id = thread.id

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                if command_executing:
                    command_executing = False
                    continue
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

        query = ' '.join(sys.argv[1:])
        args = query.split(' ')

        # see if the first argument is a command
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
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def status(
    endpoint: str,
):
    async def status_helper():
        async with httpx.AsyncClient(timeout=2.0) as client:
            try:
                response = await client.get(f'{endpoint}/health')
                return response.json()
            except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
                return {'status': f'LLMVM server not available at {endpoint}'}

    rich.print(asyncio.run(status_helper()))


@cli.command('mode', help='Switch between "auto", "tool", "direct" and "code" mode.')
@click.argument('mode', type=str, required=False, default='')
def mode(
    mode: str,
):
    global current_mode
    old_mode = current_mode

    if mode.startswith('"') and mode.endswith('"') or mode.startswith("'") and mode.endswith("'"):
        mode = mode[1:-1]

    if not mode:
        if current_mode == 'auto': current_mode = 'tool'
        elif current_mode == 'tool': current_mode = 'direct'
        elif current_mode == 'direct': current_mode = 'code'
        elif current_mode == 'code': current_mode = 'auto'
    elif mode == 'auto' or mode == 'tool' or mode == 'direct' or mode == 'code':
        current_mode = mode
    else:
        rich.print(f'Invalid mode: {mode}')
        return

    rich.print(f'Switching mode from {old_mode} to {current_mode}')


@cli.command('exit', hidden=True)
def exit():
    os._exit(os.EX_OK)


@cli.command('clear', hidden=True)
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


@cli.command('help', hidden=True)
def help():
    Repl().help()


@cli.command('ls', hidden=True)
def ls():
    os.system('ls')

@cli.command('cookies', help='Set cookies for a message thread so that the tooling is able to access authenticated content.')
@click.option('--sqlite', '-s', type=str, required=False,
              help='location of Firefox/Chrome cookies sqlite file.')
@click.option('--file_location', '-l', type=str, required=False,
              help='location of Firefox/Chrome cookies txt file in Netscape format.')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread id to attach cookies to')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def cookies(
    sqlite: str,
    file_location: str,
    id: int,
    endpoint: str,
):
    global thread_id

    if id <= 0:
        id = thread_id

    cookies = []

    if file_location:
        with open(file_location, 'r') as f:
            cookies = Helpers.read_netscape_cookies(f.read())
    else:
        if not sqlite:
            # find either firefox or chrome cookies
            start_locations = []

            if os.path.exists(os.path.expanduser('~/.mozilla/firefox')):
                # find any cookies.sqlite file and print the directory
                for root, _, files in os.walk(os.path.expanduser('~/.mozilla/firefox')):
                    for file in files:
                        if file == 'cookies.sqlite':
                            start_locations.append(os.path.join(root, file))

            if os.path.exists(os.path.expanduser('~/.config/google-chrome')):
                # find any cookies.sqlite file and print the directory
                for root, dirs, files in os.walk(os.path.expanduser('~/.config/google-chrome')):
                    for file in files:
                        if file == 'Cookies':
                            start_locations.append(os.path.join(root, file))

            if len(start_locations) == 0:
                rich.print('No cookies files found.')
                return

            # print the list of cookies files and ask the user to pick one
            rich.print('Select a cookies file:')
            for i, location in enumerate(start_locations):
                rich.print(f'[{i}]: {location}')

            selection = int(input('Selection: '))
            if selection < 0 or selection >= len(start_locations):
                rich.print('Invalid selection.')
                return

            sqlite = start_locations[selection]

        # we have a location, now extract and upload
        # run the scripts/extract_cookies.py script and capture the text output
        browser = 'firefox' if 'sqlite' in sqlite else 'chrome'
        cmd = f'scripts/extract_{browser}_cookies.sh {sqlite}'
        result = subprocess.run(cmd, text=True, shell=True, env=os.environ, capture_output=True)
        if result.returncode != 0:
            rich.print(result.stderr)
            return
        cookies = Helpers.read_netscape_cookies(result.stdout)

    async def cookies_helper():
        async with httpx.AsyncClient(timeout=280.0) as client:
            response = await client.post(f'{endpoint}/v1/chat/cookies', json={'id': id, 'cookies': cookies})  # type: ignore
            session_thread = SessionThread.model_validate(response.json())
        return session_thread

    thread = asyncio.run(cookies_helper())
    rich.print(f'Set cookies for thread [{thread.id}]')
    return thread


@cli.command('act', help='Use a prompt from awesome_prompts to set a character, actor or persona.')
@click.argument('actor', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to retrieve.')
@click.option('--mode', type=click.Choice(['auto', 'direct', 'tool'], case_sensitive=False), required=False, default='auto',
              help='ode to use "auto", "tool" or "direct". Default is "auto".')
@click.option('--executor', '-x', type=str, required=False, default=Container.get_config_variable('LLMVM_EXECUTOR', default=''),
              help='model to use. Default is $LLMVM_EXECUTOR or LLMVM server default.')
@click.option('--model', '-m', type=str, required=False, default=Container.get_config_variable('LLMVM_MODEL', default=''),
              help='model to use. Default is $LLMVM_MODEL or LLMVM server default.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),  # type: ignore
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
@click.option('--suppress_role', '-s', type=bool, is_flag=True, required=False, default=False)
def act(
    actor: str,
    id: int,
    mode: str,
    executor: str,
    model: str,
    endpoint: str,
    suppress_role: bool,
):
    client_directory = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(f'{client_directory}/prompts/awesome_prompts.csv')
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
            ctx.params['path'] = ''
            ctx.params['upload'] = False
            ctx.params['mode'] = mode
            ctx.params['endpoint'] = endpoint
            ctx.params['cookies'] = ''
            ctx.params['executor'] = executor
            ctx.params['model'] = model
            ctx.params['suppress_role'] = suppress_role
            return message.invoke(ctx)

@cli.command('file', help='Insert a file (image or pdf) into the message thread.')
@click.argument('filename', type=str, required=True)
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to attach content to. Default is last thread.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def file(
    filename: str,
    id: int,
    endpoint: str,
):
    global thread_id
    if id <= 0:
        id = thread_id

    if filename.startswith('"') and filename.endswith('"'):
        filename = filename[1:-1]

    if not os.path.exists(filename):
        rich.print(f'File {filename} does not exist.')
        return
    else:
        user_message = User(Content(''))
        try:

            with open(filename, 'rb') as file_content:
                bytes_content = file_content.read()
                file_content.seek(0)

                if Helpers.is_image(bytes_content):
                    output = io.BytesIO()
                    with Image.open(io.BytesIO(file_content.read())) as img:
                        img.save(output, format='JPEG')
                        StreamPrinter('user').display_image(output.getvalue())
                        file_content.seek(0)
                    user_message = User(ImageContent(bytes_content, url='cli'))
                elif Helpers.is_pdf(file_content):
                    user_message = User(PdfContent(bytes_content, url='cli'))
                else:
                    user_message = User(Content(bytes_content.decode('utf-8')))
        except Exception as ex:
            pass

        thread = asyncio.run(get_thread(endpoint, id))
        thread.messages.append(MessageModel.from_message(message=user_message))
        rich.print(f'Inserting file {filename} into thread [{thread.id}]')
        result = asyncio.run(set_thread(endpoint, thread))
        thread_id = result.id
        return result


@cli.command('url', help='Download a url and insert the content into the message thread.')
@click.argument('url', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to download and push the content to. Default is last thread.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def url(
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


@cli.command('search', help='Perform a search on ingested content using the LLMVM search engine.')
@click.argument('query', type=str, required=False, default='')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
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
            if link.endswith('.py'):
                markdown_snippet = f'```python\n{snippet}\n```'
                CodeBlock.__rich_console__ = markdown__rich_console__
                console.print(Markdown(markdown_snippet))
            else:
                wrapped_text = textwrap.wrap(snippet, width=console.width - 6)
                wrapped_lines = '\n'.join('    ' + line for line in wrapped_text[:4])
                console.print(f'{wrapped_lines}')
            console.print(f'    [italic]Score: {score}[/italic]')
            console.print('\n\n')


@cli.command('ingest', help='Ingest a file into the LLMVM search engine.')
@click.argument('filename', type=str, required=True)
@click.option('--recursive', '-r', type=bool, required=True, is_flag=True, help='Walk the directory recursively.')
@click.option('--project', '-p', type=str, required=True, default='', help='Assign the ingested content to project.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def ingest(
    filename: str,
    recursive: bool,
    project: str,
    endpoint: str,
):
    files = []

    if filename.startswith('"') and filename.endswith('"'):
        filename = filename[1:-1]

    filename_directory = os.path.abspath(filename)

    if not os.path.exists(filename_directory):
        rich.print(f'File or directory {filename_directory} does not exist.')
        return

    if os.path.isdir(filename_directory):
        if recursive:
            for root, _, filenames in os.walk(filename_directory):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        else:
            files += [os.path.join(filename_directory, f)
                      for f in os.listdir(filename_directory) if os.path.isfile(os.path.join(filename_directory, f))]

    async def upload_helper():
        async with httpx.AsyncClient(timeout=280.0) as client:
            responses = []
            for filename_str in files:
                with open(filename_str, 'rb') as f:
                    data = {
                        'file': (filename_str, f),
                        'project': project,
                    }
                    response = await client.post(f'{endpoint}/ingest', files=data)
                    responses.append(response.text)
                    rich.print(response.text)
            return responses

    return asyncio.run(upload_helper())


@cli.command('threads', help='List all message threads and set to last.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def threads(
    endpoint: str,
):
    global thread_id
    global last_thread

    threads = asyncio.run(get_threads(endpoint))

    for thread in threads:
        if len(thread.messages) > 0:
            message_content = str(thread.messages[-1].content).replace('\n', ' ')[0:75]
            rich.print(f'[{thread.id}]: {message_content}')

    active_threads = [t for t in threads if len(t.messages) > 0]

    if thread_id == 0 and len(active_threads) > 0:
        thread_id = active_threads[-1].id
        last_thread = active_threads[-1]
        return active_threads[-1]


@cli.command('thread', help='List all message threads.')
@click.argument('id', type=str, required=True)
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def thread(
    id: str,
    endpoint: str,
):
    global thread_id

    if id.startswith('"') and id.endswith('"'):
        int_id = int(id[1:-1])
    else:
        int_id = int(id)

    thread = asyncio.run(get_thread(endpoint, int_id))
    print_thread(thread=thread)
    thread_id = thread.id
    return thread


@cli.command('messages', help='List all messages in a message thread.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
@click.option('--suppress_role', '-s', type=bool, is_flag=True, required=False, default=False)
def messages(
    endpoint: str,
    suppress_role: bool,
):
    global thread_id
    thread = asyncio.run(get_thread(endpoint, thread_id))
    print_thread(thread=thread, suppress_role=suppress_role)


@cli.command('new', help='Create a new message thread.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
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
@click.option('--path', '-p', callback=parse_path, required=False,
              help='Path to a single file, multiple files, directory of files, or a glob, to add to User message stack.')
@click.option('--recursive', '-r', type=bool, required=False, default=False, is_flag=True,
              help='When using the --path option, recursively walk the directory.')
@click.option('--upload', '-u', is_flag=True, required=True, default=False,
              help='Upload the files to the LLMVM server. If False, LLMVM server must be run locally. Default is False.')
@click.option('--mode', '-o', type=click.Choice(['auto', 'direct', 'tool', 'code'], case_sensitive=False),
              required=False, default='auto',
              help='Mode to use "auto", "tool", "code", or "direct". Default is "auto".')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is $LLMVM_ENDPOINT or http://127.0.0.1:8011')
@click.option('--cookies', '-e', type=str, required=False, default=Container.get_config_variable('LLMVM_COOKIES', default=''),
              help='cookies.txt file (Netscape) for the request. Default is $LLMVM_COOKIES or empty.')
@click.option('--executor', '-x', type=str, required=False, default=Container.get_config_variable('LLMVM_EXECUTOR', default=''),
              help='model to use. Default is $LLMVM_EXECUTOR or LLMVM server default.')
@click.option('--model', '-m', type=str, required=False, default=Container.get_config_variable('LLMVM_MODEL', default=''),
              help='model to use. Default is $LLMVM_MODEL or LLMVM server default.')
@click.option('--suppress_role', '-s', type=bool, is_flag=True, required=False)
def message(
    message: Optional[str | bytes | Message],
    id: int,
    path: List[str],
    recursive: bool,
    upload: bool,
    mode: str,
    endpoint: str,
    cookies: str,
    executor: str,
    model: str,
    suppress_role: bool,
):
    global thread_id
    global last_thread
    context_messages: List[User] = []

    if mode == 'code' and not path:
        raise MissingParameter('path')

    if not suppress_role and not sys.stdin.isatty():
        suppress_role = True

    if model:
        if (model.startswith('"') and model.endswith('"')) or (model.startswith("'") and model.endswith("'")):
            model = model[1:-1]

    if executor:
        if (executor.startswith('"') and executor.endswith('"')) or (executor.startswith("'") and executor.endswith("'")):
            executor = executor[1:-1]

    if path:
        allowed_extensions = ['.py', '.md', 'Dockerfile', '.sh', '.txt'] if mode == 'code' else []
        context_messages = get_files_as_messages(path, upload, allowed_extensions)

    # if we have files, but no message, grab the last file and use it as the message
    if not message and context_messages:
        message = context_messages[-1]
        context_messages = context_messages[:-1]

    if message:
        if isinstance(message, str) and (
            (message.startswith('"') and message.endswith('"'))
            or (message.startswith("'") and message.endswith("'"))
        ):
            message = message[1:-1]

        if id <= 0:
            id = thread_id  # type: ignore

        cookies_list = []
        if cookies:
            with open(cookies, 'r') as f:
                cookies_list = Helpers.read_netscape_cookies(f.read())

        thread = llm(
            message=message,
            id=id,
            mode=mode,
            endpoint=endpoint,
            executor=executor,
            model=model,
            context_messages=context_messages,
            cookies=cookies_list
        )
        if not suppress_role: StreamPrinter('').write_string('\n')
        print_response([MessageModel.to_message(thread.messages[-1])], suppress_role)
        if not suppress_role: StreamPrinter('').write_string('\n')
        last_thread = thread
        return thread


if __name__ == '__main__':
    # special case the hijacking of --help
    if len(sys.argv) == 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        Repl().help()
        sys.exit(0)

    if len(sys.argv) <= 1:
        repl_inst = Repl()
        repl_inst.repl()
    else:
        cli()
