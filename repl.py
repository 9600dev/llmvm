import ast
import os
import subprocess
import sys
import tempfile
import textwrap
from typing import Callable, Dict, List, Optional, cast

import click
import pandas as pd
import rich
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown

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

def print_response(statements: List[Statement | AstNode]):
    def contains_token(s, tokens):
        return any(token in s for token in tokens)

    def pprint(prepend: str, s: str):
        markdown_tokens = ['###', '* ', '](', '```']
        console = Console()

        if contains_token(s, markdown_tokens):
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
            rich.print()
            pprint('[bold green]Assistant[/bold green]: ', str(statement.message))
            fire_helper(str(statement.message))
        elif isinstance(statement, Content):
            pprint('', str(statement).strip())
        elif isinstance(statement, System):
            pprint('[bold red]System[/bold red]: ', str(statement.message))
        elif isinstance(statement, User):
            pprint('[bold green]User[/bold green]: ', str(statement.message))
        elif isinstance(statement, Answer):
            if isinstance(statement.result(), Statement):
                print_response([cast(Statement, statement.result())])
            else:
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


class StreamPrinter():
    def __init__(self, role: str):
        self.buffer = ''
        self.console = Console()
        self.markdown_mode = False
        self.role = role
        self.started = False

    def write(self, string: str):
        if not self.started and self.role:
            self.console.print(f'[bold green]{self.role}[/bold green]: ', end='')
            self.started = True
        self.buffer += string
        self.console.print(f'[bright_black]{string}[/bright_black]', end='')


class Repl():
    def __init__(
        self,
        executor: Executor,
        agents: List[Callable]
    ):
        self.executor = executor
        self.agents: List[Callable] = agents

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

    def repl(self):
        console = rich.console.Console()
        history = FileHistory(".repl_history")
        kb = KeyBindings()
        edit = False
        mode = 'tool'
        debug = False
        stream = True
        continuation = False

        if os.path.exists(Container().get('firefox_download_dir') + '/mozilla.pdf'):
            os.remove(Container().get('firefox_download_dir') + '/mozilla.pdf')

        @kb.add('c-e')
        def _(event):
            editor = os.environ.get('EDITOR', 'vim')
            text = self.open_editor(editor, event.app.current_buffer.text)
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text) - 1

        # todo: this is a hack, we need to refactor the execution controller
        controller = StarlarkExecutionController(
            executor=self.executor,
            agents=agents,
            cache=PersistentCache('cache/cache.db'),
            edit_hook=None,
            stream_handler=StreamPrinter('').write if stream else None,
        )

        message_history: List[Message] = []

        commands = {
            '/act': 'load an acting prompt and set to "actor" mode. This will similarity search on awesome_prompts.',
            '/agents': 'list the available helper functions',
            '/clear': 'clear the message history',
            '/cls': 'clear the screen',
            '/compile': 'ask LLM to compile query into AST and print to screen',
            '/continuation': 'run in interpreted continuation passing style',
            '/debug': 'toggle debug mode',
            '/delcache': 'delete the persistence cache',
            '/direct': 'sets mode to "direct" sending messages stack directly to LLM',
            '/download': 'download content from the specified url into the message history',
            '/edit': 'edit the message history',
            '/edit_ast': 'edit any tool AST result in $EDITOR',
            '/edit_last': 'edit the last Assitant message in $EDITOR',
            '/exit': 'exit the repl',
            '/last': 'clear the conversation except for the last Assistant message',
            '/load': 'load the current stack and message history from disk',
            '/local': 'set openai.api_base and openai.api_version to local settings in config.yaml',
            '/messages': 'show message history',
            '/mode': 'show the current mode',
            '/openai_api_base': 'set the openai api base url (e.g https://api.openai.com/v1 or http://127.0.0.1:8000/v1)',
            '/openai_api_version': 'set the openai api model version (e.g. gpt-3.5 gpt-4, vicuna)',
            '/save': 'serialize the current stack and message history to disk',
            '/stream': 'toggle stream mode',
            '/sysprompt': 'set the System prompt mode to the supplied prompt.',
            '/tool': '[default mode] sets mode to Starlark runtime tool mode, where the LLM is asked to interact with tools',
            '/y': 'yank the last Assistant message to the clipboard using xclip',
        }

        self.help(commands)

        while True:
            try:
                query = prompt(
                    'prompt>> ',
                    history=history,
                    enable_history_search=True,
                    vi_mode=True,
                    key_bindings=kb,
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

                elif query.startswith('/local'):
                    api_base = Container().get('local_openai_api_base')
                    api_version = Container().get('local_openai_api_version')
                    import openai
                    openai.api_base = api_base
                    openai.api_version = api_version
                    self.executor.set_default_model(api_version)
                    rich.print('Setting openai.api_base to {}'.format(api_base))
                    rich.print('Setting openai.api_version to {}'.format(api_version))
                    continue

                elif query.startswith('/continuation'):
                    if controller.continuation_passing_style:
                        controller.continuation_passing_style = False
                        rich.print('Disabling continuation passing style.')
                    else:
                        controller.continuation_passing_style = True
                        rich.print('Enabling continuation passing style.')
                    continue

                elif query.startswith('/openai_api_version'):
                    version = Helpers.in_between(query, '/openai_api_version', '\n').strip()
                    if version == '':
                        rich.print('No version specified.')
                        continue
                    import openai
                    openai.api_version = version
                    self.executor.set_default_model(version)
                    rich.print('Setting openai.api_version to {}'.format(version))
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
                        assistant = controller.execute_llm_call(
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
                    os.remove('cache/session.db')
                    cache = PersistentCache('cache/session.db')
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
                        controller.edit_hook = self.open_default_editor
                    else:
                        rich.print('Disabling AST edit mode')
                        edit = False
                        controller.edit_hook = None
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
                    self.open_default_editor(txt_message_history)
                    continue

                elif query.startswith('/agents'):
                    rich.print('Agents:')
                    for agent in self.agents:
                        rich.print('  [bold]{}[/bold]'.format(agent))
                    continue

                elif query.startswith('/mode'):
                    rich.print('Mode: {}'.format(mode))
                    continue

                elif query.startswith('/compile') or query.startswith('/c'):
                    rich.print()
                    compilation_query = Helpers.in_between(query, '/context', '\n').strip()

                    response = controller.execute_with_agents(
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
                    cache = PersistentCache('cache/session.db')
                    cached_history = []
                    if cache.get('message_history'):
                        cached_history = cast(List[Message], cache.get('message_history'))
                    cache.set('message_history', cached_history + message_history)
                    continue

                elif query.startswith('/load'):
                    cache = PersistentCache('cache/session.db')
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

                        assistant = controller.execute_llm_call(
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
                    assistant = controller.execute_llm_call(
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
                        controller.stream_handler = None
                        rich.print('Disabling streaming mode.')
                    else:
                        stream = True
                        rich.print('Enabling streaming mode.')
                        controller.stream_handler = StreamPrinter('').write
                    continue

                elif query.startswith('/download'):
                    from bcl import ContentDownloader
                    url = Helpers.in_between(query, '/download', '\n').strip()
                    downloader = ContentDownloader(
                        url,
                        self.agents,
                        message_history,
                        controller.starlark_runtime,
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
                    results = controller.execute(
                        messages=message_history + [User(Content(query))],
                    )
                    if results:
                        message_history.append(User(Content(query)))
                        message_history.append(Assistant(Content(str(results[-1].result()))))
                    else:
                        rich.print('Something went wrong in the execution controller and no results were returned.')
                else:
                    assistant_result = controller.execute_llm_call(
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
    prompt: Optional[str],
    verbose: bool,
):
    if not prompt:
        repl = Repl(executor, agents=agents)
        repl.repl()
    else:
        controller = StarlarkExecutionController(
            executor=executor,
            agents=agents,
            vector_store=VectorStore(),
            stream_handler=StreamPrinter('').write,
        )

        results = controller.execute(
            messages=[User(Content(prompt))]
        )
        print_response(results)  # type: ignore


@click.command()
@click.option('--executor', type=click.Choice(['openai', 'local']), required=False, default='openai')
@click.option('--prompt', type=str, required=False, default='')
@click.option('--verbose', type=bool, default=True)
@click.option('--local_openai_base', type=str, required=False, default='https://127.0.0.1:8000/v1')
@click.option('--local_openai_version', type=str, required=False, default='')
def main(
    executor: Optional[str],
    prompt: Optional[str],
    verbose: bool,
    local_openai_base: Optional[str],
    local_openai_version: Optional[str],
):
    if not os.environ.get('OPENAI_API_KEY'):
        raise Exception('OPENAI_API_KEY environment variable not set')

    openai_key = str(os.environ.get('OPENAI_API_KEY'))

    def openai_executor():
        openai_executor = OpenAIExecutor(openai_key, verbose=verbose, cache=PersistentCache('cache/cache.db'))
        return openai_executor

    if executor == 'local':
        import openai
        openai.api_base = local_openai_base
        openai.api_version = local_openai_version

    start(
        openai_executor(),
        prompt,
        verbose)

if __name__ == '__main__':
    main()
