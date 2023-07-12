import datetime as dt
import inspect
import os
import sys
import time
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

import click
import rich
from docstring_parser import parse as docstring_parse
from guidance.llms.transformers import LLaMA, Vicuna
from langchain.agents import initialize_agent, load_tools
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI as langchain_OpenAI
from langchain.text_splitter import (MarkdownTextSplitter,
                                     PythonCodeTextSplitter, TokenTextSplitter)
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from selenium.webdriver.firefox.options import Options as FirefoxOptions

from eightbitvicuna import VicunaEightBit
from helpers.edgar import EdgarHelpers
from helpers.email_helpers import EmailHelpers
from helpers.helpers import Helpers
from helpers.logging_helpers import setup_logging
from helpers.market import MarketHelpers
from helpers.pdf import PdfHelpers
from helpers.vector_store import VectorStore
from helpers.websearch import WebHelpers
from objects import (Agent, Answer, Assistant, AstNode, Content, Continuation,
                     ExecutionFlow, Executor, ForEach, FunctionCall, Message,
                     NaturalLanguage, Order, Program, Statement, System, Text,
                     UncertainOrError, User)
from openai_executor import OpenAIExecutor

logging = setup_logging()


# https://glean.com/product/ai-search
# https://dust.tt/
# https://support.apple.com/guide/automator/welcome/mac


def vector_store():
    from langchain.vectorstores import FAISS

    from helpers.vector_store import VectorStore

    return VectorStore(openai_key=os.environ.get('OPENAI_API_KEY'), store_filename='faiss_index')  # type: ignore

def load_vicuna():
    return VicunaEightBit('models/applied-vicuna-7b', device_map='auto')


def invokeable(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} ran in {end_time - start_time} seconds")
        return result
    return wrapper


class Parser():
    def __init__(
        self,
        message_type=User,
    ):
        self.message: str = ''
        self.remainder: str = ''
        self.index = 0
        self.agents: List[Callable] = []
        self.message_type: type = message_type

    def consume(self, token: str):
        if token in self.remainder:
            self.remainder = self.remainder[self.remainder.index(token) + len(token):]
            self.remainder = self.remainder.strip()

    def to_function_call(self, call_str: str) -> Optional[FunctionCall]:
        function_description = Helpers.parse_function_call(call_str, self.agents)
        if function_description:
            name = function_description['name']
            arguments = []
            types = []
            for arg_name, metadata in function_description['parameters']['properties'].items():
                # todo if we don't have an argument here, we should ensure that
                # the function has a default value for the parameter
                if 'argument' in metadata:
                    arguments.append({arg_name: metadata['argument']})
                    types.append({arg_name: metadata['type']})

            return FunctionCall(
                name=name,
                args=arguments,
                types=types
            )
        return None

    def parse_function_call(
        self,
    ) -> Optional[FunctionCall]:
        text = self.remainder

        sequence: List[AstNode] = []

        if (
                ('[[' not in text and ']]' not in text)
                and ('```python' not in text)
                and ('[' not in text and ']]' not in text)
        ):
            return None

        while (
            ('[[' in text and ')]]' in text)
            and ('[' in text and ')]' in text)
            or ('```python' in text and '```\n' in text)
        ):
            start_token = ''
            end_token = ''

            match text:
                case _ if '```python' in text:
                    start_token = '```python'
                    end_token = '```\n'
                case _ if '[[' and ']]' in text:
                    start_token = '[['
                    end_token = ']]'
                case _ if '[' and ')]' in text:
                    start_token = '['
                    end_token = ']'

            function_call_str = Helpers.in_between(text, start_token, end_token)
            function_call: Optional[FunctionCall] = self.to_function_call(function_call_str)
            if function_call:
                function_call.context = Content(Text(text))

            # remainder is the stuff after the end_token
            self.remainder = text[text.index(end_token) + len(end_token):]
            return function_call

        return None

    def parse_ast_node(
        self,
    ) -> AstNode:
        re = self.remainder

        while re.strip() != '':
            if re.startswith('"') and re.endswith('"'):
                self.remainder = self.remainder[:1]
                self.consume('"')
                return Text(re.strip('"'))
            else:
                result = Text(self.remainder)
                self.remainder = ''
                return result
        return Text('')

    def parse_statement(
        self,
        stack: List[Statement],
    ) -> Statement:
        def parse_continuation():
            lhs = stack.pop()
            continuation = Continuation(lhs=lhs, rhs=Statement())

            stack.append(continuation)

            continuation.rhs = self.parse_statement(stack)
            return continuation

        re = self.remainder.strip()

        while re != '':
            if re.startswith('Output:'):
                self.consume('Output:')
                return self.parse_statement(stack)

            if re.startswith('answer(') and ')' in re:
                answer = Helpers.in_between(re, 'answer(', ')')
                self.consume(')')
                return Answer(conversation=[Text(answer)])

            if re.startswith('function_call(') and ')' in re:
                function_call = self.parse_function_call()
                if not function_call:
                    # push past the function call and keep parsing
                    self.consume(')')
                    re = self.remainder
                else:
                    self.consume(')')
                    return function_call

            if re.startswith('natural_language(') and ')' in re:
                language = Helpers.in_between(re, 'natural_language(', ')')
                self.consume(')')
                return NaturalLanguage(messages=[User(Content(language))])  # type: ignore

            if re.startswith('continuation(') and ')' in re:
                continuation = parse_continuation()
                self.consume(')')
                return continuation

            if re.startswith('[[=>]]'):
                self.consume('[[=>]]')
                return parse_continuation()

            if re.startswith('[[FOREACH]]'):
                self.consume('[[FOREACH]]')
                fe = ForEach(lhs=stack, rhs=Statement())
                fe.rhs = self.parse_statement(stack)
                return fe

            # we have no idea, so return something the LLM can figure out
            if re.startswith('"'):
                message = self.message_type(self.parse_ast_node())
                return NaturalLanguage(messages=[message])

            result = NaturalLanguage(messages=[self.message_type(Text(self.remainder))])
            self.remainder = ''
            return result
        return Statement()

    def parse_program(
        self,
        message: str,
        agents: List[Callable],
        executor: Executor,
        execution_flow: ExecutionFlow[Statement],
    ) -> Program:
        self.message = message
        self.agents = agents
        self.remainder = message

        program = Program(executor, execution_flow)
        stack: List[Statement] = []

        while self.remainder.strip() != '':
            statement = self.parse_statement(stack)
            program.statements.append(statement)
            stack.append(statement)

        self.message = ''
        self.agents = []

        return program


class ExecutionController():
    def __init__(
        self,
        execution_contexts: List[Executor],
        agents: List[Callable] = [],
        vector_store: Optional[VectorStore] = None,
    ):
        self.execution_contexts: List[Executor] = execution_contexts
        self.agents = agents
        self.parser = Parser()
        self.messages: List[Message] = []
        self.vector_store = vector_store

    def classify_tool_or_direct(
        self,
        prompt: str,
    ) -> Dict[str, float]:
        def parse_result(result: str) -> Dict[str, float]:
            if ',' in result:
                first = result.split(',')[0]
                second = result.split(',')[1]
                try:
                    if first.startswith('tool') or first.startswith('"tool"'):
                        return {'tool': float(second)}
                    elif first.startswith('direct') or first.startswith('"direct"'):
                        return {'direct': float(second)}
                    else:
                        return {'tool': 1.0}
                except ValueError as ex:
                    return {'tool': 1.0}
            return {'tool': 1.0}

        executor = self.execution_contexts[0]

        # assess the type of task
        function_list = [Helpers.get_function_description_flat(f) for f in self.agents]
        query_understanding = Helpers.load_and_populate_prompt(
            prompt_filename='prompts/query_understanding.prompt',
            template={
                'functions': '\n'.join(function_list),
                'user_input': prompt,
            }
        )

        assistant: Assistant = executor.execute(
            system_message=System(Text(query_understanding['system_message'])),
            user_messages=[User(Content(Text(query_understanding['user_message'])))],
        )

        if assistant.error or not parse_result(str(assistant.message)):
            return {'tool': 1.0}
        return parse_result(str(assistant.message))

    def execute_statement(
        self,
        statement: Statement,
        executor: Executor,
        callee: Optional[Statement] = None,
    ) -> Statement:

        # we have an answer to the query, return it
        if isinstance(statement, Answer):
            return statement

        elif isinstance(statement, FunctionCall) and not statement.response():
            # unpack the args, call the function
            function_call = statement
            function_args_desc = statement.args
            function_args = {}

            # Note: the JSON response from the model may not be valid JSON
            func: Callable | None = Helpers.first(lambda f: f.__name__ in function_call.name, self.agents)

            if not func:
                logging.error('Could not find function {}'.format(function_call.name))
                return UncertainOrError(conversation=[Text('I could find the function {}'.format(function_call.name))])

            # check for enum types and marshal from string to enum
            counter = 0
            for p in inspect.signature(func).parameters.values():
                if p.annotation != inspect.Parameter.empty and p.annotation.__class__.__name__ == 'EnumMeta':
                    function_args[p.name] = p.annotation(function_args_desc[counter][p.name])
                elif counter < len(function_args_desc):
                    function_args[p.name] = function_args_desc[counter][p.name]
                else:
                    function_args[p.name] = p.default
                counter += 1

            try:
                function_response = func(**function_args)
            except Exception as e:
                logging.error(e)
                return UncertainOrError(
                    conversation=[Text('The function could not execute. It raised an exception: {}'.format(e))]
                )

            function_call.call_response = function_response
            return function_call

        elif (
            isinstance(statement, NaturalLanguage)
            and not statement.response()
            and callee
        ):
            # callee provides context for the natural language statement
            messages: List[User] = []
            messages.extend(statement.messages)

            system_message = statement.system if statement.system else System(Text('You are a helpful assistant.'))
            response = executor.execute(system_message=system_message, user_messages=messages)
            statement.call_response = str(response.message)
            return statement

        elif isinstance(statement, NaturalLanguage) and statement.response():
            return statement

        elif isinstance(statement, ForEach):
            return UncertainOrError()

        elif isinstance(statement, Continuation):
            return UncertainOrError()

        return UncertainOrError()

    def execute_program(
        self,
        program: Program,
        execution: ExecutionFlow[Statement]
    ) -> List[Statement]:
        answers: List[Statement] = []

        for s in reversed(program.statements):
            execution.push(s)

        while statement := execution.pop():
            answers.append(self.execute_statement(statement, program.executor))

        return answers

    def execute_simple(
        self,
        system_message: str,
        user_message: str,
    ) -> Assistant:
        executor = self.execution_contexts[0]
        return executor.execute(
            system_message=System(Text(system_message)),
            user_messages=[User(Content(Text(user_message)))])

    def execute(
        self,
        prompt: str,
    ) -> List[Statement]:
        results: List[Statement] = []

        # pick the right execution context that will get the task done
        # for now, we just grab the first
        executor = self.execution_contexts[0]

        # create an execution flow
        execution: ExecutionFlow[Statement] = ExecutionFlow(Order.QUEUE)

        # assess the type of task
        classification = self.classify_tool_or_direct(prompt)

        # if it requires tooling, hand it off to the AST execution engine
        if 'tool' in classification:
            # call the LLM, asking it to hand back an AST
            llm_call = NaturalLanguage(
                messages=[User(Content(Text(prompt)))],
                executor=executor
            )

            response = executor.execute_with_agents(call=llm_call, agents=agents)
            assistant_response = str(response.message)

            program = Parser().parse_program(assistant_response, self.agents, executor, execution)
            answers: List[Statement] = self.execute_program(program, execution)
            results.extend(answers)
        else:
            assistant_reply: Assistant = self.execute_simple(
                system_message='You are a helpful assistant.',
                user_message=prompt
            )
            results.append(Answer(
                conversation=[Text(str(assistant_reply.message))],
                result=assistant_reply
            ))

        return results


class Repl():
    def __init__(
        self,
        executors: List[Executor]
    ):
        self.executors: List[Executor] = executors
        self.agents: List[Agent] = []

    def print_response(self, statements: List[Statement]):
        for statement in statements:
            rich.print(str(statement))

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
        )

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

                if '/help' in query:
                    rich.print('Commands:')
                    for command, description in commands.items():
                        rich.print('  [bold]{}[/bold] - {}'.format(command, description))
                    continue

                if 'exit' in query:
                    sys.exit(0)

                if '/context' in query:
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

                if '/agents' in query:
                    rich.print('Agents:')
                    for agent in self.agents:
                        rich.print('  [bold]{}[/bold]'.format(agent.__class__.__name__))
                        rich.print('    {}'.format(agent.instruction()))
                    continue

                if '/any' in query:
                    executor_contexts = self.executors
                    continue

                results = execution_controller.execute(prompt=query)

                self.print_response(results)
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
        openai_executor = OpenAIExecutor(openai_key, verbose=verbose)
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
        execution_environments[0].execute(prompt, '')


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
