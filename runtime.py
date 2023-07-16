import datetime as dt
import inspect
import math
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
from selenium.webdriver.firefox.options import Options as FirefoxOptions

from eightbitvicuna import VicunaEightBit
from helpers.edgar import EdgarHelpers
from helpers.email_helpers import EmailHelpers
from helpers.helpers import Helpers, PersistentCache
from helpers.logging_helpers import setup_logging
from helpers.market import MarketHelpers
from helpers.pdf import PdfHelpers
from helpers.vector_store import VectorStore
from helpers.websearch import WebHelpers
from objects import (Agent, Answer, Assistant, AstNode, Content, Continuation,
                     ExecutionFlow, Executor, ForEach, FunctionCall, Message,
                     NaturalLanguage, Order, Program, Statement, System,
                     UncertainOrError, User, tree_map)
from openai_executor import OpenAIExecutor

logging = setup_logging()
def response_writer(callee, message):
    with (open('logs/ast.log', 'a')) as f:
        f.write(f'{str(dt.datetime.now())} {callee}: {message}\n')


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
        parsed_function = Helpers.parse_function_call(call_str, self.agents)
        if parsed_function:
            func, function_description = parsed_function
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
                types=types,
                func=func,
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
                function_call.context = Content(text)

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
                return Content(re.strip('"'))
            else:
                result = Content(self.remainder)
                self.remainder = ''
                return result
        return Content('')

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
                if answer.startswith('"') and answer.endswith('"'):
                    answer = answer[1:-1]
                if answer.startswith('\'') and answer.endswith('\''):
                    answer = answer[1:-1]
                self.consume(')')
                return Answer(conversation=[Content(answer)])

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

            if re.startswith('[[') and not re.startswith('[[=>]]') and ')]]' in re:  # this is not great, it's not ebnf
                function_call = self.parse_function_call()
                if not function_call:
                    # push past the function call and keep parsing
                    self.consume(']]')
                    re = self.remainder
                else:
                    return function_call

            # might be a string
            if re.startswith('"'):
                message = self.message_type(self.parse_ast_node())
                return NaturalLanguage(messages=[message])

            # we have no idea, so start packaging tokens into a Natural Language call until we see something we know
            known_tokens = ['[[=>]]', '[[FOREACH]]', 'function_call', 'natural_language', 'answer', 'continuation', 'Output']
            tokens = re.split(' ')
            consumed_tokens = []
            while len(tokens) > 0:
                if tokens[0] in known_tokens:
                    self.remainder = ' '.join(tokens)
                    return NaturalLanguage(messages=[self.message_type(Content(' '.join(consumed_tokens)))])

                consumed_tokens.append(tokens[0])
                self.remainder = self.remainder[len(tokens[0]) + 1:]
                tokens = tokens[1:]

            self.remainder = ''
            return NaturalLanguage(messages=[self.message_type(Content(' '.join(consumed_tokens)))])
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
        vector_store: VectorStore = VectorStore(),
        cache: PersistentCache = PersistentCache(),
    ):
        self.execution_contexts: List[Executor] = execution_contexts
        self.agents = agents
        self.parser = Parser()
        self.vector_store: VectorStore = vector_store
        self.cache = cache

    def classify_tool_or_direct(
        self,
        prompt: str,
    ) -> Dict[str, float]:
        def parse_result(result: str) -> Dict[str, float]:
            if ',' in result:
                if result.startswith('Assistant: '):
                    result = result[len('Assistant: '):].strip()

                first = result.split(',')[0].strip()
                second = result.split(',')[1].strip()
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
            messages=[
                System(Content(query_understanding['system_message'])),
                User(Content(query_understanding['user_message']))
            ],
        )

        response_writer('query_understanding.prompt', assistant)

        if assistant.error or not parse_result(str(assistant.message)):
            return {'tool': 1.0}
        return parse_result(str(assistant.message))

    def __chunk_messages(
        self,
        query: Any,
        messages: List[Message],
        max_token_count: int,
    ) -> List[Message]:
        # force conversion to str
        query = str(query)
        total_message = '\n'.join([str(m.message) for m in messages])

        if Helpers.calculate_tokens(total_message) + Helpers.calculate_tokens(query) < max_token_count:
            return messages

        total_tokens_per_message = math.floor(max_token_count / len(messages))

        chunked_messages: List[Message] = []

        for message in messages:
            ranked: List[Tuple[str, float]] = self.vector_store.chunk_and_rank(
                query,
                str(message.message),
                chunk_token_count=math.floor(total_tokens_per_message / 5),
                max_tokens=total_tokens_per_message,
            )
            ranked_str = '\n\n'.join([r[0] for r in ranked])

            while Helpers.calculate_tokens(ranked_str) + Helpers.calculate_tokens(query) > max_token_count:
                ranked_str = ranked_str[0:len(ranked_str) - 16]

            chunked_messages.append(message.__class__(Content(ranked_str)))
        return chunked_messages

    def __chunk_content(
        self,
        query: Any,
        current_str: Any,
        max_token_count: int,
    ) -> str:
        # force conversion to str
        query = str(query)
        current_str = str(current_str)

        if Helpers.calculate_tokens(current_str) + Helpers.calculate_tokens(query) < max_token_count:
            return str(current_str)

        ranked: List[Tuple[str, float]] = self.vector_store.chunk_and_rank(
            query,
            current_str,
            chunk_token_count=math.floor((max_token_count / 10)),
        )

        ranked_str = '\n\n'.join([r[0] for r in ranked])
        while Helpers.calculate_tokens(ranked_str) + Helpers.calculate_tokens(query) > max_token_count:
            ranked_str = ranked_str[0:len(ranked_str) - 16]

        return ranked_str

    def __lhs_to_str(self, lhs: List[Statement]) -> str:
        results = [r for r in lhs if r.result()]
        # unique
        results = list(set(results))
        return '\n'.join([str(r) for r in results])

    def __summarize_to_messages(self, node: AstNode) -> List[Message]:
        def summarize_conversation(ast_node: AstNode) -> Optional[Message]:
            if isinstance(ast_node, FunctionCall) and ast_node.result():
                return User(Content(str(ast_node.result())))
            elif isinstance(ast_node, NaturalLanguage):
                return ast_node.messages[0]
            else:
                return None

        # we probably need to do stuff to fit inside the context window etc
        messages: List[Message] = []
        context_messages = cast(List[Message], tree_map(node, summarize_conversation))
        context_messages = [m for m in context_messages if m is not None]
        messages.extend(context_messages)

        return messages

    def execute_statement(
        self,
        statement: Statement,
        executor: Executor,
        callee_or_context: Optional[Statement] = None,
    ) -> Statement:

        # we have an answer to the query, return it
        if isinstance(statement, Answer):
            return statement

        elif isinstance(statement, FunctionCall) and not statement.result():
            # unpack the args, call the function
            function_call = statement
            function_args_desc = statement.args
            function_args = {}

            # Note: the JSON response from the model may not be valid JSON
            func: Callable | None = Helpers.first(lambda f: f.__name__ in function_call.name, self.agents)

            if not func:
                logging.error('Could not find function {}'.format(function_call.name))
                return UncertainOrError(conversation=[Content('I could find the function {}'.format(function_call.name))])

            def marshal(value: object, type: str) -> Any:
                def strip_quotes(value: str) -> str:
                    if value.startswith('\'') or value.startswith('"'):
                        value = value[1:]
                    if value.endswith('\'') or value.endswith('"'):
                        value = value[:-1]
                    return value

                if type == 'str':
                    result = str(value)
                    return strip_quotes(result)
                elif type == 'int':
                    return int(strip_quotes(str(value)))
                elif type == 'float':
                    return float(strip_quotes(str(value)))
                else:
                    return value

            # check for enum types and marshal from string to enum
            counter = 0
            for p in inspect.signature(func).parameters.values():
                if p.annotation != inspect.Parameter.empty and p.annotation.__class__.__name__ == 'EnumMeta':
                    function_args[p.name] = p.annotation(function_args_desc[counter][p.name])
                elif counter < len(function_args_desc):
                    function_args[p.name] = marshal(function_args_desc[counter][p.name], p.annotation.__name__)
                else:
                    function_args[p.name] = p.default
                counter += 1

            try:
                # let's check the cache first
                if self.cache.has_key(function_args):
                    function_response = self.cache.get(function_args)
                else:
                    function_response = func(**function_args)
                    self.cache.set(function_args, function_response)

            except Exception as e:
                logging.error(e)
                return UncertainOrError(
                    conversation=[Content('The function could not execute. It raised an exception: {}'.format(e))]
                )

            function_call._result = function_response
            return function_call

        elif (
            isinstance(statement, NaturalLanguage)
            and not statement.result()
            and callee_or_context
        ):
            # callee provides context for the natural language statement
            messages: List[Message] = []
            messages.extend(statement.messages)

            if isinstance(callee_or_context, FunctionCall):
                function_call = callee_or_context
                function_args_desc = function_call.args
                function_args = {}
                function_result = function_call.result()

                continuation_function = Helpers.load_and_populate_prompt(
                    prompt_filename='prompts/continuation_function.prompt',
                    template={
                        'function_name': function_call.to_code_call(),
                        'function_result': self.__chunk_content(
                            query=statement.messages[0],
                            current_str=function_result,
                            max_token_count=executor.max_tokens()
                        ),
                        'natural_language': str(statement.messages[0]),
                    }
                )

                messages.append(User(Content(continuation_function['user_message'])))
                assistant: Assistant = executor.execute(messages)
                response_writer('continuation_function.prompt', assistant)
                statement._result = assistant
                return statement

            elif isinstance(callee_or_context, ForEach):
                # the result() could have a list of things
                if isinstance(callee_or_context.result(), list):
                    result_list = []
                    for foreach_result in cast(list, callee_or_context.result()):
                        assistant_result = self.execute_statement(statement, executor, callee_or_context=foreach_result).result()
                        # we have to unset the result so that the next iteration works properly
                        statement._result = None
                        # todo: we can remove this type cast, after execute_statement is just Statement
                        assistant_result = cast(Assistant, assistant_result)
                        # todo: not sure if "NaturalLanguage" is the right thing to return here.
                        result_list.append(NaturalLanguage([assistant_result], executor=executor))

                    statement._result = result_list

            else:
                # generic continuation response
                continuation_generic = Helpers.load_and_populate_prompt(
                    prompt_filename='prompts/continuation_generic.prompt',
                    template={
                        'context': str(callee_or_context.result()),
                        'query': str(statement.messages[0]),
                    }
                )
                messages.append(User(Content(continuation_generic['user_message'])))
                assistant: Assistant = executor.execute(messages)
                response_writer('continuation_generic.prompt', assistant)
                statement._result = assistant
                return statement

        elif (
            isinstance(statement, NaturalLanguage)
            and not statement.result()
            and not callee_or_context
        ):
            messages: List[Message] = []
            system_message = statement.system if statement.system else System(Content('You are a helpful assistant.'))
            messages.append(system_message)
            messages.extend(statement.messages)

            result = executor.execute(self.__chunk_messages(
                query=messages[-1],
                messages=messages,
                max_token_count=executor.max_tokens()
            ))
            response_writer('natural_language.prompt', result)

            statement._result = result
            return statement

        elif isinstance(statement, NaturalLanguage) and statement.result():
            return statement

        elif isinstance(statement, ForEach):
            messages: List[Message] = []

            if isinstance(statement.rhs, FunctionCall):
                foreach_function = Helpers.load_and_populate_prompt(
                    prompt_filename='prompts/foreach_functioncall.prompt',
                    template={
                        'function_call': statement.rhs.to_code_call(),  # statement.rhs.to_definition(),
                        'list': self.__lhs_to_str(statement.lhs),
                    }
                )

                def summarize_conversation(ast_node: AstNode) -> Optional[Message]:
                    if isinstance(ast_node, FunctionCall) and ast_node.result():
                        return User(Content(str(ast_node.result())))
                    elif isinstance(ast_node, NaturalLanguage):
                        return ast_node.messages[0]
                    else:
                        return None

                # I need to add supporting context so that the LLM can fill in the callsite args
                if callee_or_context:
                    messages.extend(self.__summarize_to_messages(callee_or_context))

                messages.append(User(Content(foreach_function['user_message'])))
                result = executor.execute(messages)

                # debug output
                response_writer('foreach_functioncall.prompt', result)

                # todo there needs to be some sort of error handling/backtracking here
                if '"missing"' in str(result.message):
                    # try pushing the LLM to re-write the correct function call
                    function_call_rewrite = Helpers.load_and_populate_prompt(
                        'prompts/functioncall_correction.prompt',
                        template={
                            'function_calls_missing': str(result.message),
                            'function_call_signatures': '\n'.join(
                                [Helpers.get_function_description_flat_extra(f) for f in self.agents]
                            ),
                            'previous_messages': '\n'.join([str(m.message) for m in messages[0:-1]])
                        }
                    )

                    result = executor.execute(messages=[User(Content(function_call_rewrite['user_message']))])
                    response_writer('functioncall_correction.prompt', result)

                # parse the result
                foreach_parser = Parser()
                flow = ExecutionFlow(Order.QUEUE)
                program = foreach_parser.parse_program(str(result.message), self.agents, executor, flow)

                # execute the program
                program_result = self.execute_program(program, flow)

                if len(program_result) > 0:
                    statement._result = program_result

                return statement

            elif isinstance(statement.rhs, NaturalLanguage):
                foreach_functioncall = Helpers.load_and_populate_prompt(
                    prompt_filename='prompts/foreach_functioncall.prompt',
                    template={
                        'list': '\n'.join([str(s.result()) for s in statement.lhs]),
                        'natural_language': str(statement.rhs.messages[0]),
                    }
                )
                messages.append(User(Content(foreach_functioncall['user_message'])))
                result = executor.execute(messages)
                response_writer('foreach_functioncall.prompt', result)
                statement._result = result
                return statement

        elif isinstance(statement, Continuation):
            result = self.execute_statement(statement.rhs, executor, statement.lhs)
            statement._result = result
            return statement

        else:
            raise ValueError('shouldnt be here')

    def execute_program(
        self,
        program: Program,
        execution: ExecutionFlow[Statement]
    ) -> List[Statement]:
        answers: List[Statement] = []

        for s in reversed(program.statements):
            execution.push(s)

        while statement := execution.pop():
            callee_context = None
            if len(answers) > 0:
                callee_context = answers[-1]

            answers.append(self.execute_statement(statement, program.executor, callee_or_context=callee_context))

        return answers

    def execute_chat(
        self,
        messages: List[Message],
    ) -> Assistant:
        executor = self.execution_contexts[0]
        return executor.execute(messages)

    def execute(
        self,
        call: NaturalLanguage
    ) -> List[Statement]:
        # pick the right execution context that will get the task done
        # for now, we just grab the first
        results: List[Statement] = []
        executor = self.execution_contexts[0]

        # create an execution flow
        execution: ExecutionFlow[Statement] = ExecutionFlow(Order.QUEUE)

        # assess the type of task
        classification = self.classify_tool_or_direct(str(call.messages[-1].message))

        # if it requires tooling, hand it off to the AST execution engine
        if 'tool' in classification:
            response = executor.execute_with_agents(call=call, agents=self.agents)
            assistant_response = str(response.message)

            # debug output
            response_writer('llm_call', assistant_response)

            # parse the response
            program = Parser().parse_program(assistant_response, self.agents, executor, execution)

            # execute the program
            answers: List[Statement] = self.execute_program(program, execution)
            results.extend(answers)
        else:
            assistant_reply: Assistant = self.execute_chat(call.messages)
            results.append(Answer(conversation=[Content(str(assistant_reply.message))]))

        return results

    def execute_completion(
        self,
        user_message: str,
        system_message: str = 'You are a helpful assistant.'
    ) -> List[Statement]:
        call = NaturalLanguage(messages=[
            System(Content(system_message)),
            User(Content(user_message))
        ])

        return self.execute(call)


