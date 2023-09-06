import ast
import os
import subprocess
import sys
import tempfile
import textwrap
from gc import enable
from re import I
from typing import Callable, Dict, List, Optional, cast

import click
import pandas as pd
import rich
from langchain import OpenAI
from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import (Completer, Completion, PathCompleter,
                                       WordCompleter, merge_completers)
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax

from bcl import BCL
from container import Container
from helpers.edgar import EdgarHelpers
from helpers.email_helpers import EmailHelpers
from helpers.helpers import Helpers
from helpers.logging_helpers import setup_logging, suppress_logging
from helpers.market import MarketHelpers
from helpers.webhelpers import WebHelpers
from objects import (Answer, Assistant, AstNode, Content, Executor,
                     FunctionCall, Message, Statement, System, User)
from openai_executor import OpenAIExecutor
from persistent_cache import PersistentCache
from starlark_execution_controller import StarlarkExecutionController
from vector_store import VectorStore

logging = setup_logging()


# we have to monkeypatch this method to remove the 'padding' from the markdown
# console output, so copy and paste "just works(tm)"
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


def print_response(statements: List[Statement | AstNode], suppress_user: bool = False):
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

    for statement in statements:
        if isinstance(statement, Assistant):
            # todo, this is a hack, Assistant not a Statement
            if not suppress_user:
                rich.print()
                pprint('[bold green]Assistant[/bold green]: ', str(statement.message))
            else:
                pprint('', str(statement.message))
            fire_helper(str(statement.message))
        elif isinstance(statement, Content):
            pprint('', str(statement).strip())
        elif isinstance(statement, System):
            if not suppress_user:
                pprint('[bold red]System[/bold red]: ', str(statement.message))
            else:
                pprint('', str(statement.message))
        elif isinstance(statement, User):
            if not suppress_user:
                pprint('[bold green]User[/bold green]: ', str(statement.message))
            else:
                pprint('', str(statement.message))
        elif isinstance(statement, Answer):
            if isinstance(statement.result(), Statement):
                print_response([cast(Statement, statement.result())])
            else:
                if not suppress_user:
                    rich.print('[bold green]Assistant[/bold green]:')
                # print_response(statement.conversation)
                pprint('', str(statement.result()))
        elif isinstance(statement, FunctionCall):
            logging.debug('FunctionCall: {}({})'.format(statement.name, str(statement.args)))
            if 'search_internet' in statement.name:
                pprint('[bold yellow]FunctionCall[/bold yellow]: ', statement.to_code_call())
            else:
                pprint('[bold yellow]FunctionCall[/bold yellow]: ', statement.to_code_call())
                pprint('', f'  {statement.result()}')
        else:
            pprint('', str(statement))


def is_binary_file(filepath):
    """Check if a file is binary or text by reading a chunk of the file."""
    with open(filepath, 'rb') as file:
        CHUNKSIZE = 8192  # Read 8KB to see if the file has a null byte.
        chunk = file.read(CHUNKSIZE)
    return b'\x00' in chunk


def parse_content(text) -> List[Message]:
    lines = text.strip().split("\n")

    result: List[Message] = []
    current_key = None
    current_content = []

    for line in lines:
        if line.startswith("User:") or line.startswith("Assistant:"):
            if current_key:
                if current_key == 'User':
                    result.append(User(Content("\n".join(current_content).strip())))
                else:
                    result.append(Assistant(Content("\n".join(current_content).strip())))
                current_content = []

            current_key = "User" if line.startswith("User:") else "Assistant"
            current_content.append(line.split(":", 1)[-1].strip())
        else:
            current_content.append(line.strip())

    if current_key and current_content:
        if current_key == 'User':
            result.append(User(Content("\n".join(current_content).strip())))
        else:
            result.append(Assistant(Content("\n".join(current_content).strip())))
    return result


class StreamPrinter():
    def __init__(self, role: str):
        self.buffer = ''
        self.console = Console(file=sys.stderr)
        self.markdown_mode = False
        self.role = role
        self.started = False

    def write(self, string: str):
        if logging.level <= 20:  # INFO
            if not self.started and self.role:
                self.console.print(f'[bold green]{self.role}[/bold green]: ', end='')
                self.started = True
            self.buffer += string
            self.console.print(f'[bright_black]{string}[/bright_black]', end='')


class Repl():
    def __init__(
        self,
        executor: Executor,
        controller: StarlarkExecutionController,
        agents: List[Callable]
    ):
        self.executor = executor
        self.agents: List[Callable] = agents
        self.controller = controller

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

    def help(self, commands):
        rich.print()
        rich.print('[white](Ctrl-c or "/exit" to exit, Ctrl-e to open $EDITOR for multiline input, Ctrl-r search prompt history)[/white]')  # noqa: E501
        rich.print('[bold]I am a helpful assistant.[/bold]')
        rich.print()

        rich.print('Commands:')
        for command, description in commands.items():
            rich.print('  [bold]{}[/bold] - {}'.format(command, description))

    def repl(
        self,
    ):
        console = rich.console.Console()
        history = FileHistory(os.path.expanduser('~/.local/share/llmvm/.repl_history'))
        kb = KeyBindings()
        edit = False
        mode = 'tool'
        debug = False
        stream = True

        session_cache_file = Container().get('cache_directory') + '/session.db'

        if os.path.exists(Container().get('firefox_download_dir') + '/mozilla.pdf'):
            os.remove(Container().get('firefox_download_dir') + '/mozilla.pdf')

        @kb.add('c-e')
        def _(event):
            editor = os.environ.get('EDITOR', 'vim')
            text = self.open_editor(editor, event.app.current_buffer.text)
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text) - 1

        # todo: this is a hack, we need to refactor the execution controller
        message_history: List[Message] = []

        commands = {
            '/act': 'load an acting prompt and set to "actor" mode. This will similarity search on awesome_prompts.',
            '/agents': 'list the available helper functions',
            '/clear': 'clear the message history',
            '/cls': 'clear the screen',
            '/compile': 'ask LLM to compile query into AST and print to screen',
            '/continuation': 'run in interpreted continuation passing style',
            '/debug': 'toggle debug mode',
            '/delcache': 'delete all persistence caches: session cache, message history, vector store etc',
            '/direct': 'sets mode to "direct" sending messages stack directly to LLM',
            '/download': 'download content from the specified url into the message history',
            '/edit': 'edit the message history',
            '/edit_ast': 'edit any tool AST result in $EDITOR',
            '/edit_last': 'edit the last Assitant message in $EDITOR',
            '/exit': 'exit the repl',
            '/file': 'load the contents of a file into the message history',
            '/last': 'clear the conversation except for the last Assistant message',
            '/load': 'load the current stack and message history from disk',
            '/local': 'set openai.api_base url and model to local settings in config.yaml',
            '/messages': 'show message history',
            '/mode': 'show the current mode',
            '/model': 'get/set the default model',
            '/model_tools': 'get/set the tools model',
            '/openai_api_base': 'set the openai api base url (e.g https://api.openai.com/v1 or http://127.0.0.1:8000/v1)',
            '/save': 'serialize the current stack and message history to disk',
            '/stream': 'toggle stream mode',
            '/sysprompt': 'set the System prompt mode to the supplied prompt.',
            '/tool': '[default mode] sets mode to Starlark runtime tool mode, where the LLM is asked to interact with tools',
            '/y': 'yank the last Assistant message to the clipboard using xclip',
        }

        self.help(commands)

        custom_style = Style.from_dict({
            'suggestion': 'bg:#888888 #444444'
        })

        cmds = [str(command)[1:] for command in commands.keys()]
        command_completer = WordCompleter(cmds, ignore_case=True, display_dict=commands)
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

        while True:
            try:
                query = session.prompt(
                    'prompt>> ',
                )

                if query is None or query == '':
                    continue

                elif query.startswith('/help'):
                    self.help(commands)
                    continue

                elif query.startswith('/openai_api_base'):
                    base_url = Helpers.in_between(query, '/openai_api_base', '\n').strip()
                    if base_url == '':
                        rich.print('No base url specified.')
                        continue
                    import openai
                    openai.api_base = base_url
                    rich.print('Setting openai.api_base to {}'.format(base_url))
                    continue

                elif query.startswith('/file'):
                    filename = Helpers.in_between(query, '/file ', '\n').strip()
                    if filename == '':
                        rich.print('No filename specified.')
                        continue
                    if not os.path.exists(filename):
                        rich.print('File does not exist.')
                        continue

                    contents = ''
                    if is_binary_file(filename) and 'pdf' in filename:
                        from helpers.pdf import PdfHelpers
                        contents = PdfHelpers.parse_pdf(filename)
                    else:
                        with open(filename, 'r') as f:
                            contents = f.read()
                    message_history.append(User(Content(contents)))
                    rich.print(f'File {filename} loaded into message history.')
                    continue

                elif query.startswith('/local'):
                    api_base = Container().get('local_openai_api_base')
                    local_model = Container().get('local_model')
                    local_tools_model = Container().get('local_tools_model')
                    local_model_max_tokens = Container().get('local_model_max_tokens')

                    import openai
                    openai.api_base = api_base
                    self.executor.set_default_model(local_model)
                    self.controller.tools_model = local_tools_model
                    self.executor.set_default_max_tokens(int(local_model_max_tokens))

                    rich.print('Setting openai.api_base to {}'.format(api_base))
                    rich.print('Setting StarlarkExecutionController model to {}'.format(local_model))
                    rich.print('Setting StarlarkExecutionController tools model to {}'.format(local_tools_model))
                    rich.print('Setting OpenAIExecutor default max tokens to {}'.format(local_model_max_tokens))
                    continue

                elif query.startswith('/model_tools'):
                    model = Helpers.in_between(query, '/model_tools ', '\n').strip()
                    if model == '':
                        rich.print(f'Tools model: {self.controller.tools_model}')
                        continue
                    self.controller.tools_model = model
                    rich.print('Setting StarlarkExecutionController tools model to {}'.format(model))
                    continue

                elif query.startswith('/model'):
                    model = Helpers.in_between(query, '/model ', '\n').strip()
                    if model == '':
                        rich.print(f'Default model: {self.executor.get_default_model()}')
                        continue
                    self.executor.set_default_model(model)
                    rich.print('Setting StarlarkExecutionController model to {}'.format(model))
                    continue

                elif query.startswith('/continuation'):
                    if self.controller.continuation_passing_style:
                        self.controller.continuation_passing_style = False
                        rich.print('Disabling continuation passing style.')
                    else:
                        self.controller.continuation_passing_style = True
                        rich.print('Enabling continuation passing style.')
                    continue

                elif query.startswith('/tool'):
                    rich.print('Setting tool mode.')
                    mode = 'tool'
                    continue

                elif query.startswith('/direct') or query.startswith('/d '):
                    if query.startswith('/d '): query = query.replace('/d ', '/direct ')

                    if query.startswith('/direct '):
                        # just execute the query, don't change the mode
                        direct_query = query.replace('/direct ', '')
                        assistant = self.controller.execute_llm_call(
                            message=User(Content(direct_query)),
                            context_messages=message_history,
                            query='',
                            original_query='',
                            lifo=True,
                            stream_handler=StreamPrinter('').write if stream else None,
                        )

                        rich.print()
                        print_response([assistant])
                        message_history.append(User(Content(query)))
                        message_history.append(assistant)
                        rich.print()
                        continue
                    else:
                        mode = 'direct'
                        rich.print('Setting direct to LLM mode')
                        continue

                elif query.startswith('/exit') or query == 'exit':
                    sys.exit(0)

                elif query.startswith('/clear'):
                    message_history = []
                    continue

                elif query.startswith('/cls'):
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue

                elif query.startswith('/delcache'):
                    os.remove(session_cache_file)
                    os.remove(Container().get('cache_directory') + '/cache.db')
                    cache = PersistentCache(session_cache_file)
                    cache.set('message_history', [])
                    continue

                elif query.startswith('/messages') or query.startswith('/m '):
                    print_response(message_history)  # type: ignore
                    continue

                elif query.startswith('/last'):
                    rich.print('Clearning conversation except last message: {}'.format(message_history[-1]))
                    message_history = message_history[:-1]

                elif query.startswith('/edit_last'):
                    if len(message_history) > 0:
                        message_history[-1].message = Content(self.open_default_editor(str(message_history[-1].message)))
                    continue

                elif query.startswith('/edit_ast'):
                    if not edit:
                        rich.print('Enabling AST edit mode')
                        edit = True
                        self.controller.edit_hook = self.open_default_editor
                    else:
                        rich.print('Disabling AST edit mode')
                        edit = False
                        self.controller.edit_hook = None
                    continue

                elif query.startswith('/edit'):
                    def role(message: Message):
                        if isinstance(message, User):
                            return 'User'
                        elif isinstance(message, Assistant):
                            return 'Assistant'
                        elif isinstance(message, System):
                            return 'System'
                        else:
                            return 'Unknown'
                    rich.print('Editing message history')
                    txt_message_history = '\n\n'.join(
                        [f"{role(message)}: {str(message.message)}" for message in message_history]
                    )
                    new_message_content = self.open_default_editor(txt_message_history)
                    new_message_history = parse_content(new_message_content)
                    message_history = new_message_history
                    continue

                elif query.startswith('/agents'):
                    rich.print('Agents:')
                    for agent in self.agents:
                        rich.print('  [bold]{}[/bold]'.format(agent))
                    continue

                elif query == '/mode':
                    rich.print('Mode: {}'.format(mode))
                    continue

                elif query.startswith('/compile') or query.startswith('/c'):
                    rich.print()
                    compilation_query = Helpers.in_between(query, '/context', '\n').strip()

                    response = self.controller.execute_with_agents(
                        messages=[User(Content(compilation_query))],
                        agents=agents,
                        temperature=0.0,
                    )
                    rich.print()
                    rich.print()
                    rich.print('Executor execute_with_agents() result:')
                    rich.print()
                    rich.print(str(response.message))
                    rich.print()
                    try:
                        tree = ast.parse(str(response.message))
                        rich.print('Python Abstract Syntax Tree:')
                        rich.print(ast.dump(tree))
                    except Exception as ex:
                        rich.print('Failed to compile AST: {}'.format(ex))
                        rich.print()
                        rich.print(str(response.message))
                    continue

                elif query.startswith('/save'):
                    cache = PersistentCache(session_cache_file)
                    cached_history = []
                    if cache.get('message_history'):
                        cached_history = cast(List[Message], cache.get('message_history'))
                    cache.set('message_history', cached_history + message_history)
                    continue

                elif query.startswith('/load'):
                    cache = PersistentCache(session_cache_file)
                    if cache.get('message_history'):
                        cached_history = cast(List[Message], cache.get('message_history'))
                        message_history = cached_history
                    continue

                elif query.startswith('/y'):
                    process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                    process.communicate(str(message_history[-1].message).encode('utf-8'))
                    continue

                elif query.startswith('/act'):
                    if query.startswith('/act'):
                        df = pd.read_csv('prompts/awesome_prompts.csv')

                        actor = Helpers.in_between(query, '/act', '\n').strip()
                        if actor == '':
                            from rich.console import Console
                            from rich.table import Table
                            console = Console()
                            table = Table(show_header=True, header_style="bold magenta")
                            for column in df.columns:
                                table.add_column(column)
                            for _, row in df.iterrows():  # type: ignore
                                table.add_row(*row)

                            console.print(table)
                            continue

                        prompt_result = Helpers.tfidf_similarity(actor, (df.act + ' ' + df.processed_prompt).to_list())

                        rich.print()
                        rich.print('[bold red]Setting actor mode.[/bold red]')
                        rich.print()
                        rich.print('Prompt: {}'.format(prompt_result))
                        rich.print()

                        assistant = self.controller.execute_llm_call(
                            message=User(Content(prompt_result)),
                            context_messages=[System(Content(prompt_result))] + message_history,
                            query='',
                            original_query='',
                            lifo=True,
                            stream_handler=StreamPrinter('').write if stream else None,
                        )
                        print_response([assistant])

                        message_history.append(System(Content(prompt_result)))
                        message_history.append(User(Content(prompt_result)))
                        message_history.append(assistant)
                        rich.print()
                        mode = 'actor'
                        continue

                elif query.startswith('/sysprompt'):
                    mode = 'actor'
                    sys_prompt = Helpers.in_between(query, '/sysprompt', '\n').strip()
                    if sys_prompt == '':
                        rich.print('No System prompt specified.')
                        continue

                    rich.print('Setting sysprompt mode.')
                    rich.print('Prompt: {}'.format(sys_prompt))
                    rich.print()
                    assistant = self.controller.execute_llm_call(
                        message=User(Content(sys_prompt)),
                        context_messages=[System(Content(sys_prompt))] + message_history,
                        query='',
                        original_query='',
                        lifo=True,
                    )
                    print_response([assistant])
                    message_history.append(System(Content(sys_prompt)))
                    message_history.append(assistant)
                    rich.print()
                    continue

                elif query.startswith('/debug'):
                    if debug:
                        debug = False
                        rich.print('Disabling debug mode.')
                        suppress_logging()
                    else:
                        debug = True
                        rich.print('Enabling debug mode.')
                        setup_logging()
                    continue

                elif query.startswith('/stream'):
                    if stream:
                        stream = False
                        self.controller.stream_handler = None
                        rich.print('Disabling streaming mode.')
                    else:
                        stream = True
                        rich.print('Enabling streaming mode.')
                        self.controller.stream_handler = StreamPrinter('').write
                    continue

                elif query.startswith('/download'):
                    from bcl import ContentDownloader
                    url = Helpers.in_between(query, '/download', '\n').strip()
                    downloader = ContentDownloader(
                        url,
                        self.agents,
                        message_history,
                        self.controller.starlark_runtime,
                        original_code='',
                        original_query=''
                    )
                    content_message = User(Content(downloader.get()))
                    message_history.append(content_message)
                    rich.print()
                    rich.print('Content downloaded into message successfully.')
                    rich.print()
                    continue

                # execute the query in either tool mode (default) or direct/actor mode
                results = None
                if mode == 'tool':
                    results = self.controller.execute(
                        messages=message_history + [User(Content(query))],
                    )
                    if results:
                        message_history.append(User(Content(query)))
                        message_history.append(Assistant(Content(str(results[-1].result()))))
                    else:
                        rich.print('Something went wrong in the execution controller and no results were returned.')
                else:
                    assistant_result = self.controller.execute_llm_call(
                        message=User(Content(query)),
                        context_messages=message_history,
                        query='',
                        original_query='',
                        lifo=True,
                        stream_handler=StreamPrinter('').write if stream else None,
                    )
                    message_history.append(User(Content(query)))
                    message_history.append(assistant_result)
                    results = [assistant_result]

                rich.print()
                rich.print('[bold green]User:[/bold green] {}'.format(query))
                rich.print()
                print_response(results)  # type: ignore
                rich.print()

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                break

            except Exception:
                console.print_exception(max_frames=10)

agents = [
    BCL.datetime,
    WebHelpers.search_linkedin_profile,
    WebHelpers.get_linkedin_profile,
    EdgarHelpers.get_report,
    MarketHelpers.get_stock_price,
    MarketHelpers.get_current_market_capitalization,
    EmailHelpers.send_email,
    EmailHelpers.send_calendar_invite,
]

def start(
    executor: Executor,
    message: Optional[str],
    context_messages: List[Message],
    system: Optional[str],
    direct: bool,
    tools_model: str,
    quiet: bool,
):
    if quiet:
        suppress_logging()

    stream_handler = StreamPrinter('').write

    if not context_messages:
        context_messages = []

    controller = StarlarkExecutionController(
        executor=executor,
        agents=agents,
        vector_store=VectorStore(),
        stream_handler=stream_handler,
        tools_model=tools_model
    )

    if not message:
        repl = Repl(executor=executor, controller=controller, agents=agents)
        repl.repl()
    else:
        if direct:
            results = controller.execute_llm_call(
                message=User(Content(message)),  # type; ignore
                context_messages=[System(Content(system))] + context_messages if system else context_messages,  # type: ignore
                query='',
                original_query='',
                lifo=True,
                stream_handler=stream_handler,
            )
            print_response([results], True)  # type: ignore
        else:
            results = controller.execute(
                messages=context_messages + [User(Content(message))],
            )
            print_response(results, True)  # type: ignore


@click.command()
@click.argument('message', type=str, required=False, default='')
@click.option('--system', '-s', type=str, required=False, default='',
              help='System prompt to use. Default is "You are a helpful assistant."')
@click.option('--direct', '-d', type=bool, is_flag=True, required=False, default=False,
              help='Send messages directly to LLM without using the Starlark runtime.')
@click.option('--local', '-l', type=bool, is_flag=True, required=False, default=False, help='Uses local LLM endpoint')
@click.option('--quiet', '-q', type=bool, is_flag=True, default=False, help='Suppress logging')
@click.option('--model', '-m', type=str, required=False, default='')
@click.option('--tools_model', '-t', type=str, required=False, default='')
@click.option('--api_endpoint', type=str, required=False, default='')
def main(
    message: Optional[str],
    system: Optional[str],
    direct: bool,
    local: bool,
    quiet: bool,
    model: str,
    tools_model: str,
    api_endpoint: str,
):
    if not os.environ.get('OPENAI_API_KEY'):
        raise Exception('OPENAI_API_KEY environment variable not set')

    if not os.path.exists(os.path.expanduser('~/.local/share/llmvm')):
        os.makedirs(os.path.expanduser('~/.local/share/llmvm'))
        os.makedirs(os.path.expanduser('~/.local/share/llmvm/download'))
        os.makedirs(os.path.expanduser('~/.local/share/llmvm/cache'))

    if not os.path.exists(os.path.expanduser('~/.config/llmvm')):
        os.makedirs(os.path.expanduser('~/.config/llmvm'))

    if not os.path.exists(os.path.expanduser('~/.config/llmvm/config.yaml')):
        with open(os.path.expanduser('~/.config/llmvm/config.yaml'), 'w') as f:
            f.write(textwrap.dedent("""
                firefox_profile: '~/.mozilla/firefox/cp6sgb0s.selenium'
                firefox_marionette_port: 2828
                firefox_download_dir: '~/.local/share/llmvm/download'
                firefox_cookies: '~/.local/share/llmvm/cookies.txt'
                smtp_server: 'localhost'
                smtp_port: 1025
                smtp_username: 'user@domain.com'
                smtp_password: ''
                cache_directory: '~/.local/share/llmvm/cache'
                openai_api_base: 'https://api.openai.com/v1'
                openai_model: 'gpt-3.5-turbo-16k-0613'
                openai_tools_model: 'gpt-4-0613'
                local_openai_api_base: 'http://localhost:8000/v1'
                local_model: 'llongorca.gguf'
                local_tools_model: 'llongorca.gguf'
                local_model_max_tokens: 16385
            """))

    if not model:
        model = Container().get('openai_model')

    if not tools_model:
        tools_model = Container().get('openai_tools_model')

    def openai_executor():
        openai_executor = OpenAIExecutor(
            openai_key=openai_key,
            default_model=model,
            api_endpoint=api_endpoint if api_endpoint else Container().get('openai_api_base'),
            cache=PersistentCache(Container().get('cache_directory') + '/cache.db'),
        )
        return openai_executor

    openai_key = str(os.environ.get('OPENAI_API_KEY'))
    executor_inst: OpenAIExecutor = openai_executor()
    context_messages = []

    if not system:
        # only used for the /direct case, otherwise Repl sets it.
        system = 'You are a helpful assistant.'

    if not sys.stdin.isatty():
        if not message: message = ''
        file_content = sys.stdin.read()
        context_messages = [User(Content(file_content))]

    if local:
        import openai
        openai.api_base = api_endpoint if api_endpoint else Container().get('local_openai_api_base')
        local_model = model if model else Container().get('local_model')
        local_tools_model = tools_model if tools_model else Container().get('local_tools_model')
        local_model_max_tokens = Container().get('local_model_max_tokens')
        executor_inst.set_default_model(local_model)
        executor_inst.set_default_max_tokens(int(local_model_max_tokens))
        tools_model = local_tools_model

    start(
        executor_inst,
        message,
        context_messages,  # type: ignore
        system,
        direct,
        tools_model,
        quiet,
    )

if __name__ == '__main__':
    main()
