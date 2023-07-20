import os
import sys
from typing import Dict, List, Optional, cast

import click
import rich
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory

from helpers.edgar import EdgarHelpers
from helpers.email_helpers import EmailHelpers
from helpers.helpers import Helpers, PersistentCache
from helpers.logging_helpers import setup_logging
from helpers.market import MarketHelpers
from helpers.pdf import PdfHelpers
from helpers.vector_store import VectorStore
from helpers.websearch import WebHelpers
from objects import (Agent, Answer, Assistant, AstNode, Content, ExecutionFlow,
                     Executor, FunctionCall, LLMCall, Message, Program,
                     Statement, System, User)
from openai_executor import OpenAIExecutor
from runtime import ExecutionController

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
            pprint('[bold green]Assistant[/bold green]: ', str(statement.message))
        elif isinstance(statement, Content):
            pprint('', str(statement).strip())
        elif isinstance(statement, System):
            pprint('[bold red]System[/bold red]: ', str(statement.message))
        elif isinstance(statement, User):
            pprint('[bold blue]User[/bold blue]: ', str(statement.message))
        elif isinstance(statement, Assistant):
            pprint('[bold green]Assistant[/bold green]: ', str(statement.message))
        elif isinstance(statement, Answer):
            if isinstance(statement.result(), Statement):
                print_response([cast(Statement, statement.result())])
            else:
                rich.print('[bold green]Assistant[/bold green]:')
                print_response(statement.conversation)
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
        executors: List[Executor]
    ):
        self.executors: List[Executor] = executors
        self.agents: List[Agent] = []

    def repl(self):
        console = rich.console.Console()
        history = FileHistory(".repl_history")

        rich.print()
        rich.print('[bold]I am a helpful assistant.[/bold]')
        rich.print()

        executor_contexts = self.executors
        executor_names = [executor.name() for executor in executor_contexts]

        current_context = 'openai'
        execution_controller = ExecutionController(
            execution_contexts=executor_contexts,
            agents=agents,
            cache=PersistentCache('cache/cache.db')
        )

        messages: List[Message] = []

        commands = {
            'exit': 'exit the repl',
            '/context': 'change the current context',
            '/agents': 'list the available agents',
            '/any': 'execute the query in all contexts',
        }

        while True:
            try:
                query = prompt('prompt>> ', history=history, enable_history_search=True, vi_mode=True)

                if query is None or query == '':
                    continue

                elif '/help' in query:
                    rich.print('Commands:')
                    for command, description in commands.items():
                        rich.print('  [bold]{}[/bold] - {}'.format(command, description))
                    continue

                elif 'exit' in query:
                    sys.exit(0)

                elif '/clear' in query:
                    messages = []
                    continue

                elif '/messages' in query or '/conversations' in query:
                    print_response(messages)  # type: ignore
                    continue

                elif '/context' in query:
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

                elif '/agents' in query:
                    rich.print('Agents:')
                    for agent in self.agents:
                        rich.print('  [bold]{}[/bold]'.format(agent.__class__.__name__))
                        rich.print('    {}'.format(agent.instruction()))
                    continue

                elif '/any' in query:
                    executor_contexts = self.executors
                    continue

                elif query.startswith('/direct'):
                    query = Helpers.in_between(query, '/direct', '\n').strip()
                    statement = execution_controller.execute_statement(
                        statement=LLMCall(message=User(Content(query))),
                        executor=executor_contexts[0],
                        program=Program(executor_contexts[0]),
                    )
                    print_response([statement])
                    rich.print()
                    continue

                # execute the query
                results = execution_controller.execute(
                    LLMCall(
                        message=User(Content(query)),
                        supporting_messages=messages
                    )
                )

                print_response(results)  # type: ignore

                # add the user message to the conversation
                messages.append(User(Content(query)))
                rich.print()

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                break

            except Exception:
                console.print_exception(max_frames=10)

agents = [
    WebHelpers.get_url,
    WebHelpers.get_news,
    WebHelpers.get_url_firefox,
    WebHelpers.search_news,
    WebHelpers.search_internet,
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
        repl = Repl(execution_environments)
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
