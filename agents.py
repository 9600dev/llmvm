import datetime as dt
import inspect
import json
import os
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

import click
import openai
import requests
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

logging = setup_logging()


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

T = TypeVar('T')

class Visitor(ABC):
    @abstractmethod
    def visit(self, node: 'AstNode') -> 'AstNode':
        pass


class Executor(ABC):
    @abstractmethod
    def execute(
        self,
        system_message: 'System',
        user_messages: List['User'],
    ) -> 'Assistant':
        pass

    @abstractmethod
    def execute_with_tools(
        self,
        call: 'NaturalLanguage',
    ) -> 'Assistant':
        pass

    @abstractmethod
    def execute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: str = 'gpt-3.5-turbo-16k',
        max_completion_tokens: int = 256,
        temperature: float = 1.0,
        chat_format: bool = True,
    ) -> Dict:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def max_tokens(self) -> int:
        pass


class Agent(ABC):
    @abstractmethod
    def is_task(self, query: str) -> bool:
        pass

    @abstractmethod
    def perform_task(self, task: str, **kwargs) -> str:
        pass

    @abstractmethod
    def invoked_by(self) -> str:
        pass

    @abstractmethod
    def instruction(self) -> str:
        pass


class AstNode(ABC):
    def __init__(
        self
    ):
        self.original_text: str = ''

    def accept(self, visitor: Visitor) -> 'AstNode':
        return visitor.visit(self)


class Text(AstNode):
    def __init__(
        self,
        text: str = '',
    ):
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return f'Text({self.text})'


class Content(AstNode):
    def __init__(
        self,
        sequence: AstNode | List[AstNode],
    ):
        if type(sequence) is Content:
            self.sequence = sequence.sequence  # type: ignore
        if type(sequence) is AstNode:
            self.sequence = [sequence]
        else:
            self.sequence = cast(List[AstNode], sequence)

    def __str__(self):
        if isinstance(self.sequence, list):
            return ' '.join([str(n) for n in self.sequence])
        else:
            return str(self.sequence)

    def __repr__(self):
        return f'Content({self.sequence})'


class Message(AstNode):
    def __init__(
        self,
        message: Content,
    ):
        self.message: Content = message

    @abstractmethod
    def role(self) -> str:
        pass

    @staticmethod
    def from_dict(message: Dict[str, str]) -> 'Message':
        role = message['role']
        content = message['content']
        if role == 'user':
            return User(Content(Text(content)))
        elif role == 'system':
            return System(Text(content))
        elif role == 'assistant':
            return Assistant(Content(Text(content)))
        raise ValueError('role not found supported')

    def __getitem__(self, key):
        return {'role': self.role(), 'content': self.message}

    @staticmethod
    def to_dict(message: 'Message') -> Dict[str, str]:
        return {'role': message.role(), 'content': str(message.message)}


class User(Message):
    def __init__(
        self,
        message: Content
    ):
        super().__init__(message)

    def role(self) -> str:
        return 'user'

    def __str__(self):
        return str(self.message)

    def __repr__(self):
        return f'Message({self.message})'


class System(Message):
    def __init__(
        self,
        message: Text = Text('''
            You are a helpful assistant.
            Dont make assumptions about what values to plug into functions.
            Ask for clarification if a user request is ambiguous.
        ''')
    ):
        super().__init__(Content(message))

    def role(self) -> str:
        return 'system'

    def __str__(self):
        return str(self.message)

    def __repr__(self):
        return f'SystemPrompt({self.message})'


class Assistant(Message):
    def __init__(
        self,
        message: Content,
        error: bool = False,
        messages_context: List[Message] = [],
        system_context: object = None,
        llm_call_context: object = None,
    ):
        super().__init__(message)
        self.error = error
        self._llm_call_context: object = llm_call_context
        self._system_context = system_context,
        self._messages_context: List[Message] = messages_context

    def role(self) -> str:
        return 'assistant'

    def __str__(self):
        return f'Assistant({self.message}) {self.error})'

    def __repr__(self):
        return f'Assistant({self.message} {self.error})'


class Statement(AstNode):
    pass


class Call(Statement):
    def __init__(
        self,
    ):
        self.call_response: Optional[str] = None

    def response(self) -> str:
        return self.call_response or ''


class NaturalLanguage(Call):
    def __init__(
        self,
        messages: List[User],
        system: Optional[System] = None,
        executor: Optional[Executor] = None,
    ):
        self.messages: List[User] = messages
        self.system = system
        self.executor = executor


class Continuation(Statement):
    def __init__(
        self,
        lhs: Statement,
        rhs: Statement,
    ):
        self.lhs = lhs
        self.rhs = rhs


class ForEach(Statement):
    def __init__(
        self,
        lhs: List[Statement],
        rhs: Statement,
    ):
        self.lhs = lhs
        self.rhs = rhs


class FunctionCall(Call):
    def __init__(
        self,
        name: str,
        args: List[Dict[str, object]],
        types: List[Dict[str, object]],
        context: Content = Content(Text('')),
    ):
        self.name = name
        self.args = args
        self.types = types
        self.context = context
        self.result: Optional[Text] = None


class Answer(Statement):
    def __init__(
        self,
        conversation: List[AstNode] = [],
        result: object = None,
        error: object = None,
    ):
        self.conversation = conversation,
        self.result = result
        self.error = error

    def __str__(self):
        result = f'Answer({self.error}, {self.result})\n'
        result += 'Conversation:\n'
        result += '\n'.join([str(n) for n in self.conversation])
        return result


class UncertainOrError(Statement):
    def __init__(
        self,
        conversation: List[AstNode] = [],
        result: object = None,
        error: object = None,
    ):
        self.conversation = conversation,
        self.result = result
        self.error = error

    def __str__(self):
        result = f'uncertain_or_error({self.error}, {self.result})\n'
        result += 'Conversation:\n'
        result += '\n'.join([str(n) for n in self.conversation])
        return result


class Program(AstNode):
    def __init__(
        self,
        executor: Executor,
        flow: 'ExecutionFlow',
    ):
        self.statements: List[Statement] = []
        self.flow = flow
        self.executor: Executor = executor


class PromptStrategy(Enum):
    THROW = 'throw'
    SEARCH = 'search'
    SUMMARIZE = 'summarize'

class Order(Enum):
    STACK = 'stack'
    QUEUE = 'queue'


class ExecutionFlow(Generic[T]):
    def __init__(self, order: Order):
        self.flow: List[T] = []
        self.order = order

    def push(self, item: T):
        if self.order == Order.QUEUE:
            self.flow.insert(0, item)
        else:
            self.flow.append(item)

    def pop(self) -> Optional[T]:
        if len(self.flow) == 0:
            return None

        if self.order == Order.QUEUE:
            return self.flow.pop(0)
        else:
            return self.flow.pop()

    def peek(self, index: int = 0) -> Optional[T]:
        if len(self.flow) == 0:
            return None

        if index > 0:
            logging.warning('ExecutionFlow.peek index must be zero or negative')

        if self.order == Order.QUEUE:
            if len(self.flow) <= abs(index):
                return None
            return self.flow[abs(index)]
        else:
            if len(self.flow) <= abs(index):
                return None
            return self.flow[-1 + index]

    def is_empty(self) -> bool:
        return len(self.flow) == 0

    def count(self) -> int:
        return len(self.flow)

    def __getitem__(self, index):
        return self.flow[index]


def tree_map(node: AstNode, call: Callable[[AstNode], Any]) -> List[Any]:
    visited = []
    visited.extend([call(node)])

    if isinstance(node, Content):
        if isinstance(node.sequence, list):
            for n in node.sequence:
                visited.extend(tree_map(n, call))
        else:
            visited.extend(tree_map(node.sequence, call))
    elif isinstance(node, Text):
        pass
    elif isinstance(node, User):
        visited.extend(tree_map(node.message, call))
    elif isinstance(node, Assistant):
        visited.extend(tree_map(node.message, call))
    elif isinstance(node, ForEach):
        for statement in node.lhs:
            visited.extend(tree_map(statement, call))
        visited.extend(tree_map(node.rhs, call))
    elif isinstance(node, Continuation):
        visited.extend(tree_map(node.lhs, call))
        visited.extend(tree_map(node.rhs, call))
    elif isinstance(node, FunctionCall):
        visited.extend(tree_map(node.context, call))
    elif isinstance(node, NaturalLanguage):
        for n in node.messages:
            visited.extend(tree_map(n, call))
        if node.system:
            visited.extend(tree_map(node.system, call))
    elif isinstance(node, Program):
        for statement in node.statements:
            visited.extend(tree_map(statement, call))
    else:
        raise ValueError('not implemented')
    return visited


def tree_traverse(node, visitor: Visitor):
    if isinstance(node, Content):
        if isinstance(node.sequence, list):
            node.sequence = Helpers.flatten([cast(AstNode, tree_traverse(child, visitor)) for child in node.sequence])
        elif isinstance(node, AstNode):
            node.sequence = [cast(AstNode, tree_traverse(node.sequence, visitor))]  # type: ignore
    elif isinstance(node, Assistant):
        node.message = cast(Content, tree_traverse(node.message, visitor))
    elif isinstance(node, Message):
        node.message = cast(Content, tree_traverse(node.message, visitor))
    elif isinstance(node, Text):
        pass
    elif isinstance(node, NaturalLanguage):
        node.messages = [cast(User, tree_traverse(child, visitor)) for child in node.messages]
        if node.system:
            node.system = cast(System, tree_traverse(node.system, visitor))
    elif isinstance(node, FunctionCall):
        node.context = cast(Content, tree_traverse(node.context, visitor))
    elif isinstance(node, Continuation):
        node.lhs = cast(Statement, tree_traverse(node.lhs, visitor))
        node.rhs = cast(Statement, tree_traverse(node.rhs, visitor))
    elif isinstance(node, Program):
        node.statements = [cast(Statement, tree_traverse(child, visitor)) for child in node.statements]
    return node.accept(visitor)


class ReplacementVisitor(Visitor):
    def __init__(
        self,
        node_lambda: Callable[[AstNode], bool],
        replacement_lambda: Callable[[AstNode], AstNode]
    ):
        self.node_lambda = node_lambda
        self.replacement = replacement_lambda

    def visit(self, node: AstNode) -> AstNode:
        if self.node_lambda(node):
            return self.replacement(node)
        else:
            return node


class OpenAIExecutor(Executor):
    def __init__(
        self,
        openai_key: str,
        max_function_calls: int = 5,
        model: str = 'gpt-3.5-turbo-16k',
        agents: List = [],
        verbose: bool = True,
    ):
        self.openai_key = openai_key
        self.verbose = verbose
        self.model = model
        self.agents = agents
        self.max_function_calls = max_function_calls

    def max_tokens(self) -> int:
        match self.model:
            case 'gpt-3.5-turbo-16k':
                return 16385
            case _:
                return 4096

    def name(self) -> str:
        return 'openai'

    def execute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: str = 'gpt-3.5-turbo-16k',
        max_completion_tokens: int = 256,
        temperature: float = 1.0,
        chat_format: bool = True,
    ) -> Dict:
        total_tokens = Helpers.calculate_tokens(messages)
        if total_tokens + max_completion_tokens > self.max_tokens():
            raise Exception(
                'Prompt too long, calculated user tokens: {}, completion tokens: {}'
                .format(total_tokens, max_completion_tokens)
            )

        if not chat_format and len(functions) > 0:
            raise Exception('Functions are not supported in non-chat format')

        if chat_format:
            if functions:
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                    functions=functions,
                    messages=messages,
                )
            else:
                # for whatever reason, [] functions generates an InvalidRequestError
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                    messages=messages,
                )
            return response  # type: ignore
        else:
            response = openai.Completion.create(
                model=model,
                temperature=temperature,
                max_tokens=max_completion_tokens,
                messages=messages,
            )
            return response  # type: ignore

    def __execute_tool_response(
        self,
        call: NaturalLanguage,
        response: str,
        tool_str: str,
    ) -> Assistant:
        logging.debug('__execute_tool_response')

        tool_prompt_message = '''
            You asked me to invoke a helper function {}.
            Here is the helper function response: {}.
            Please perform the task that was required using this helper function response.
            If there are still outstanding helper function requests, I'll send the results of those next.
        '''

        tool_prompt_message = tool_prompt_message.format(tool_str, response)

        messages = [Message.to_dict(m) for m in call.messages]
        messages.append({'role': 'user', 'content': tool_prompt_message})

        chat_response = self.execute_direct(
            model=self.model,
            temperature=1.0,
            max_completion_tokens=1024,
            messages=messages,
            chat_format=True,
        )

        chat_response = chat_response['choices'][0]['message']  # type: ignore
        messages.append(chat_response)

        if len(chat_response) == 0:
            return Assistant(Content(Text('The model could not execute the query.')), error=True)
        else:
            logging.debug('OpenAI Assistant Response: {}'.format(chat_response['content']))
            return Assistant(
                message=Content(Text(chat_response['content'])),
                error=False,
                messages_context=[Message.from_dict(m) for m in messages],
                llm_call_context=call,
            )

    def execute_with_tools(
        self,
        call: NaturalLanguage,
    ) -> Assistant:
        logging.debug('execute_with_tools')

        user_message: User = cast(User, call.messages[-1])
        messages = []
        message_results = []

        functions = [Helpers.get_function_description_flat(f) for f in agents]

        prompt = Helpers.load_and_populate_prompt(
            prompt_filename='prompts/tool_execution_prompt.prompt',
            template={
                'functions': '\n'.join(functions),
                'user_input': str(user_message),
            }
        )

        # todo, we probably need to figure out if we should pass in the
        # entire message history or not.
        messages.append({'role': 'system', 'content': prompt['system_message']})
        messages.append({'role': 'user', 'content': prompt['user_message']})

        chat_response = self.execute_direct(
            model=self.model,
            temperature=1.0,
            max_completion_tokens=2048,  # todo: calculate this properly
            messages=messages,
            chat_format=True,
        )

        chat_response = chat_response['choices'][0]['message']  # type: ignore
        message_results.append(chat_response)

        if len(chat_response) == 0:
            return Assistant(Content(Text('The model could not execute the query.')), error=True)
        else:
            logging.debug('OpenAI Assistant Response: {}'.format(chat_response['content']))
            return Assistant(
                message=Content(Text(chat_response['content'])),
                error=False,
                messages_context=[Message.from_dict(m) for m in messages],
                system_context=prompt['system_message'],
                llm_call_context=call,
            )

    def execute(
        self,
        system_message: System,
        user_messages: List[User],
    ) -> Assistant:
        logging.debug('OpenAIExecutor.execute system_message={} user_messages={}'
                      .format(system_message, user_messages))

        messages: List[Dict[str, str]] = []

        messages.append(Message.to_dict(system_message))
        for message in user_messages:
            messages.append(Message.to_dict(message))

        chat_response = self.execute_direct(
            messages,
            max_completion_tokens=2048,  # tood: calculate this properly
            chat_format=True,
        )

        if len(chat_response) == 0:
            return Assistant(
                message=Content(Text('The model could not execute the query.')),
                error=True,
                messages_context=[Message.from_dict(m) for m in messages],
                system_context=system_message,
            )

        messages.append(chat_response['choices'][0]['message'])

        conversation: List[Message] = [Message.from_dict(m) for m in messages]

        return Assistant(
            message=conversation[-1].message,
            messages_context=conversation
        )


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
            program.statements.append(self.parse_statement(stack))

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
                else:
                    function_args[p.name] = function_args_desc[counter][p.name]
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

            response = executor.execute_with_tools(llm_call)
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
        openai_executor = OpenAIExecutor(openai_key, verbose=verbose, agents=agents)
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
