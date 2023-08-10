import os
import subprocess
import sys
import tempfile
from typing import Callable, Dict, List, Optional, cast

import click
import rich
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from helpers.edgar import EdgarHelpers
from helpers.email_helpers import EmailHelpers
from helpers.helpers import Helpers, PersistentCache
from helpers.logging_helpers import setup_logging
from helpers.market import MarketHelpers
from helpers.pdf import PdfHelpers
from helpers.vector_store import VectorStore
from helpers.websearch import WebHelpers
from objects import (Answer, Assistant, AstNode, Content, ExecutionFlow,
                     Executor, FunctionCall, LambdaVisitor, LLMCall, Message,
                     Program, StackNode, Statement, System, User,
                     tree_traverse)
from openai_executor import OpenAIExecutor
from runtime import ExecutionController, Parser
from starlark_runtime import StarlarkExecutionController

logging = setup_logging()

def print_response(statements: List[Statement | AstNode]):
    def contains_token(s, tokens):
        return any(token in s for token in tokens)

    def pprint(prepend: str, s: str):
        markdown_tokens = ['###', '* ', '](', '```']
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()

        if contains_token(s, markdown_tokens):
            console.print(f'{prepend}', end='')
            console.print(Markdown(s))
        else:
            console.print(f'{prepend}{s}')

    for statement in statements:
        if isinstance(statement, Assistant):
            # todo, this is a hack, Assistant not a Statement
            rich.print()
            pprint('[bold green]Assistant[/bold green]: ', str(statement.message))
        elif isinstance(statement, StackNode):
            continue
        elif isinstance(statement, Content):
            pprint('', str(statement).strip())
        elif isinstance(statement, System):
            pprint('[bold red]System[/bold red]: ', str(statement.message))
        elif isinstance(statement, User):
            pprint('[bold blue]User[/bold blue]: ', str(statement.message))
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
        elif isinstance(statement, LLMCall):
            pprint(f'[bold green]{statement.message.role().capitalize()}[/bold green]: ', str(statement.message.message))
            for message in statement.supporting_messages:
                pprint(f'[bold green]{message.role().capitalize()}[/bold green]: ', str(message.message))
            if statement.result():
                print_response([cast(AstNode, statement.result())])
        else:
            pprint('', str(statement))


class Repl():
    def __init__(
        self,
        executors: List[Executor],
        agents: List[Callable]
    ):
        self.executors: List[Executor] = executors
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

    def repl(self):
        console = rich.console.Console()
        history = FileHistory(".repl_history")
        kb = KeyBindings()
        edit = False

        @kb.add('c-e')
        def _(event):
            editor = os.environ.get('EDITOR', 'vim')
            text = self.open_editor(editor, event.app.current_buffer.text)
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text) - 1

        rich.print()
        rich.print('[white](Ctrl-c or "exit" to exit, Ctrl-e to open $EDITOR for multiline input, Ctrl-r search prompt history)[/white]')
        rich.print('[bold]I am a helpful assistant.[/bold]')
        rich.print()

        executor_contexts = self.executors
        executor_names = [executor.name() for executor in executor_contexts]

        current_context = 'openai'

        # todo: this is a hack, we need to refactor the execution controller
        execution_controller = StarlarkExecutionController(
            execution_contexts=executor_contexts,
            agents=agents,
            cache=PersistentCache('cache/cache.db'),
            edit_hook=None,
        )

        message_history: List[Message] = []

        commands = {
            '/exit': 'exit the repl',
            '/context': 'change the current context',
            '/agents': 'list the available agents',
            '/any': 'execute the query in all contexts',
            '/clear': 'clear the message history',
            '/delcache': 'delete the persistence cache',
            '/messages': 'show message history',
            '/edit': 'edit any tool AST result in $EDITOR',
            '/compile': 'ask LLM to compile query into AST and print to screen',
            '/last': 'clear the conversation except for the last Assistant message',
            '/direct': 'execute the query in the current context',
            '/save': 'serialize the current stack and message history to disk',
            '/load': 'load the current stack and message history from disk',
        }

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
                    rich.print('Commands:')
                    for command, description in commands.items():
                        rich.print('  [bold]{}[/bold] - {}'.format(command, description))
                    continue

                elif query.startswith('/exit') or query == 'exit':
                    sys.exit(0)

                elif query.startswith('/clear') or query.startswith('/cls'):
                    message_history = []
                    continue

                elif query.startswith('/delcache'):
                    cache = PersistentCache('cache/session.db')
                    cache.set('message_history', [])
                    continue

                elif query.startswith('/messages') or query.startswith('/conversations') or query.startswith('/m '):
                    print_response(message_history)  # type: ignore
                    continue

                elif query.startswith('/last'):
                    rich.print('Clearning conversation except last message: {}'.format(message_history[-1]))
                    message_history = message_history[:-1]

                elif query.startswith('/context'):
                    context = Helpers.in_between(query, '/context', '\n').strip()

                    if context in executor_names:
                        current_context = context
                        executor_contexts = [executor for executor in self.executors if executor.name() == current_context]
                        rich.print('Current context: {}'.format(current_context))
                    elif context == '':
                        rich.print([e.name() for e in self.executors])
                    else:
                        rich.print('Invalid context: {}'.format(current_context))
                    continue

                elif query.startswith('/edit'):
                    if not edit:
                        rich.print('Enabling AST edit mode')
                        edit = True
                        execution_controller.edit_hook = self.open_default_editor
                    else:
                        rich.print('Disabling AST edit mode')
                        edit = False
                        execution_controller.edit_hook = None
                    continue

                elif query.startswith('/agents'):
                    rich.print('Agents:')
                    for agent in self.agents:
                        rich.print('  [bold]{}[/bold]'.format(agent))
                    continue

                elif query.startswith('/compile') or query.startswith('/c'):
                    rich.print()
                    compilation_query = Helpers.in_between(query, '/context', '\n').strip()

                    response = executor_contexts[0].execute_with_agents(
                        call=LLMCall(message=User(Content(compilation_query))),
                        agents=agents,
                        temperature=0.0,
                    )
                    rich.print()
                    rich.print()
                    rich.print('LLM execute_with_agents result:')
                    rich.print(str(response.message))
                    rich.print()
                    rich.print('Parser() output:')
                    program = Parser().parse_program(str(response.message), agents, executor_contexts[0])
                    tree_traverse(
                        program,
                        LambdaVisitor(lambda node: print('{}: {}'.format(type(node), node))),
                        post_order=False,
                    )
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

                elif query.startswith('/any'):
                    executor_contexts = self.executors
                    continue

                elif query.startswith('/direct') or query.startswith('/d '):
                    if query.startswith('/d '): query = query.replace('/d ', '/direct ')
                    query = query[8:].strip()
                    assistant = execution_controller.execute_chat(
                        messages=message_history + [User(Content(query))],
                    )
                    print_response([assistant])
                    message_history.append(User(Content(query)))
                    message_history.append(assistant)
                    rich.print()
                    continue

                # execute the query
                results = execution_controller.execute(
                    LLMCall(
                        message=User(Content(query)),
                        supporting_messages=message_history,
                    )
                )

                if results:
                    message_history.append(User(Content(query)))
                    message_history.append(Assistant(Content(str(results[-1].result()))))
                else:
                    rich.print('Something went wrong in the execution controller')

                rich.print()
                rich.print('[bold green]User:[/bold green] {}'.format(query))
                rich.print()
                print_response(results)  # type: ignore
                rich.print()

                # add the user message to the conversation

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                break

            except Exception:
                console.print_exception(max_frames=10)

agents = [
    WebHelpers.get_url,
    WebHelpers.get_url_firefox,
    WebHelpers.get_news_url,
    WebHelpers.get_content_by_search,
    WebHelpers.get_news_by_search,
    WebHelpers.search_linkedin_profile,
    WebHelpers.get_linkedin_profile,
    EdgarHelpers.get_latest_form_text,
    PdfHelpers.parse_pdf,
    MarketHelpers.get_stock_price,
    MarketHelpers.get_market_capitalization,
    EmailHelpers.send_email,
    EmailHelpers.send_calendar_invite,
]

def start(
    context: Optional[str],
    prompt: Optional[str],
    verbose: bool
):
    openai_key = str(os.environ.get('OPENAI_API_KEY'))
    execution_environments = []

    # def langchain_executor():
    #    openai_executor = LangChainExecutor(openai_key, verbose=verbose)
    #    return openai_executor

    def openai_executor():
        openai_executor = OpenAIExecutor(openai_key, verbose=verbose, cache=PersistentCache('cache/cache.db'))
        return openai_executor

    executors = {
        'openai': openai_executor(),
        # 'langchain': langchain_executor(),
    }

    if context:
        execution_environments.append(executors[context])
    else:
        execution_environments.append(list(executors.values()))

    if not prompt:
        repl = Repl(execution_environments, agents=agents)
        repl.repl()
    else:
        controller = ExecutionController(
            execution_environments,
            agents=agents,
            vector_store=VectorStore(),
        )

        results = controller.execute(
            LLMCall(
                message=User(Content(prompt))
            ))

        print_response(results)  # type: ignore


@click.command()
@click.option('--context', type=click.Choice(['openai', 'langchain', 'local']), required=False, default='openai')
@click.option('--prompt', type=str, required=False, default='')
@click.option('--verbose', type=bool, default=True)
def main(
    context: Optional[str],
    prompt: Optional[str],
    verbose: bool,
):
    if not os.environ.get('OPENAI_API_KEY'):
        raise Exception('OPENAI_API_KEY environment variable not set')

    if not verbose:
        import logging as logging_library
        logging_library.getLogger().setLevel(logging_library.ERROR)

    start(
        context,
        prompt,
        verbose)

if __name__ == '__main__':
    main()
