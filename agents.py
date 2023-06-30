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
    def execute(self, query: Union[str, List[Dict]], data: str) -> 'Assistant':
        pass

    @abstractmethod
    def chat_execute(self, messages: List['Message']) -> 'Assistant':
        pass

    def execute_one_shot(
        self,
        system_message: str,
        user_message: str,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def can_execute(self, query: Union[str, List[Dict]], data: str) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chat_context(self, chat: bool):
        pass

    @abstractmethod
    def max_tokens(self):
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
            self.sequence = sequence.sequence
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
        message: Text = Text('Don\'t make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.')  # type: ignore
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
        messages: List[Message],
        system: Optional[System] = None,
        executor: Optional[Executor] = None,
    ):
        self.messages: List[Message] = messages
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
        flow: 'ExecutionFlow'['AstNode'],
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
        node.messages = [cast(Message, tree_traverse(child, visitor)) for child in node.messages]
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
        chat: bool = False,
        max_function_calls: int = 5,
        max_tokens: int = 4096,
        model: str = 'gpt-3.5-turbo-16k',
        agents: List = [],
        verbose: bool = True,
    ):
        self.openai_key = openai_key
        self.verbose = verbose
        self.model = model
        self.agents = agents
        self.chat = chat
        self.messages: List[Dict] = []
        self.max_tokens = max_tokens
        self.max_function_calls = max_function_calls

    def name(self) -> str:
        return 'openai'

    def chat_context(self, chat: bool):
        self.chat = chat

    def execute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: str = 'gpt-3.5-turbo-16k',
        max_tokens: int = 4096,
        temperature: float = 1.0,
        chat_format: bool = True,
    ) -> Dict:
        total_tokens = Helpers.calculate_tokens(messages)
        if total_tokens > max_tokens:
            raise Exception('Prompt too long, max_tokens: {}, calculated tokens: {}'.format(max_tokens, total_tokens))

        if not chat_format and len(functions) > 0:
            raise Exception('Functions are not supported in non-chat format')

        if chat_format:
            if functions:
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    functions=functions,
                    messages=messages,
                )
            else:
                # for whatever reason, [] functions generates an InvalidRequestError
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages,
                )
            return response  # type: ignore
        else:
            response = openai.Completion.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response  # type: ignore

    def execute_tool_response(
        self,
        call: LLMCall,
        response: str,
        tool_str: str,
    ) -> Assistant:
        logging.debug('execute_tool_response')

        user_message = call.messages[-1]['content']

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
            max_tokens=self.max_tokens,
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

    def execute_with_tools_2(
        self,
        call: NaturalLanguage,
    ) -> Assistant:
        logging.debug('execute_query_with_tools')

        user_message = call.messages[-1]['content']
        messages = []
        message_results = []

        prompt = Helpers.get_prompt('prompts/tool_execution_prompt.prompt')

        function_system_message = prompt['system_message']
        tool_prompt_message = prompt['user_message']
        functions = [Helpers.get_function_description_flat(f) for f in agents]

        tool_prompt_message.replace('{{functions}}', '\n'.join(functions))
        tool_prompt_message.replace('{{user_input}}', user_message)

        messages.append({'role': 'user', 'content': tool_prompt_message})

        chat_response = self.execute_direct(
            model=self.model,
            temperature=1.0,
            max_tokens=self.max_tokens,
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
                system_context=function_system_message,
                llm_call_context=call,
            )

    def execute_with_tools(
        self,
        call: LLMCall,
    ) -> Assistant:
        logging.debug('execute_query_with_tools')

        user_message = call.messages[-1]['content']

        messages = []
        message_results = []

        # functions = [Helpers.get_function_description(f, False, True) for f in self.agents]
        functions = [Helpers.get_function_description(f, True, False) for f in self.agents]
        function_dict = {}
        function_dict.update({'functions': functions})

        function_system_message = '''
            You are a helpful assistant with access to helper functions.
            Don't make assumptions about what values to plug into functions.
            Ask for clarification if a user request is ambiguous.
        '''

        tool_prompt_message = '''
            As a helpful assistant with access to API helper functions,
            I will give you a json of helper functions under "Functions:",
            then I will give you a question or task under "Input:".
            Re-write the Question and inject function calls if required to complete the task.
            Only use the helper functions specified under "Functions:".

            You can use the [[=>]] token to chain the results of function calls together.
            You can use the [[FOREACH]] token to ensure that a function call specified on the right
            hand side of the [[FOREACH]] is repeatedly executed for each list element on the left hand
            side of the [[FOREACH]] token. The [[END]] token can be used to close the [[FOREACH]] block.

            List of functions:
        '''

        functions_message = json.dumps(function_dict)
        # functions_message = '\n'.join(functions)

        example_message = '''
        Here are examples of calling the APIs:

        Example:
        Input: Search for and summarize the profile of Jane Doe from Alphabet, John James from Facebook, and Jeff Dean from Google.
        Output:

        Summarize the profile of Jane Doe from Alphabet. [[Helpers.search_linkedin_profile('Jane', 'Doe', 'Alphabet')]]
        Summarize the profile of John James from Facebook. [[Helpers.search_linkedin_profile('John', 'James', 'Facebook')]]
        Summarize the profile of Jeff Dean from Google. [[Helpers.search_linkedin_profile('Jeff', 'Dean', 'Google')]]

        Example:
        Input: Who is the current CEO of AMD?
        Output: Who is the current CEO of AMD? [[Helpers.search_internet('current CEO of AMD')]] ?

        Example:
        Input: What is the latest strategy updates from NVDA?
        Output: What is the latest strategy updates from NVDA [[EdgarHelpers.get_latest_form_text('NVDA')]] ?

        Example:
        Input: Build a profile of the leadership team of NVDA. Include education credentials and the last
        company each executive worked at.

        Output:
        Generate a list of the leadership team of NVDA: [[Helpers.search_internet('leadership team of NVDA')]] [[=>]]
        For each executive in the list, summarize their profile:
        [[FOREACH]] Summarize the profile of [[Helpers.search_linkedin_profile(executive, company)]]
        [[END]]

        '''

        user_message = f'Now here is your task:\n\nInput: {user_message}'

        prompt_message = f'{tool_prompt_message}\n\n{functions_message}\n\n{example_message}\n\n{user_message}\nOutput: '

        messages.append({'role': 'user', 'content': prompt_message})

        chat_response = self.execute_direct(
            model=self.model,
            temperature=1.0,
            max_tokens=self.max_tokens,
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
                system_context=function_system_message,
                llm_call_context=call,
            )

    def __chat_completion_request(
        self,
        messages: List[Dict],
        functions: List[Dict] = [],
    ) -> List[Dict[str, str]]:
        message_results: List[Dict[str, str]] = []

        response = self.execute_direct(
            model=self.model,
            messages=messages,
            functions=functions,
        )

        message = response['choices'][0]['message']  # type: ignore
        message_results.append(message)
        counter = 1

        # loop until function calls are all resolved
        while message.get('function_call') and counter < self.max_function_calls:
            function_name = message['function_call']['name']
            function_args = json.loads(message['function_call']['arguments'])
            logging.debug('__chat_completion_request function_name={} function_args={}'.format(function_name, function_args))

            # Step 3, call the function
            # Note: the JSON response from the model may not be valid JSON
            func: Callable | None = Helpers.first(lambda f: f.__name__ == function_name, self.agents)

            if not func:
                return []

            # check for enum types and marshal from string to enum
            for p in inspect.signature(func).parameters.values():
                if p.annotation != inspect.Parameter.empty and p.annotation.__class__.__name__ == 'EnumMeta':
                    function_args[p.name] = p.annotation(function_args[p.name])

            try:
                function_response = func(**function_args)
            except Exception as e:
                function_response = 'The function could not execute. It raised an exception: {}'.format(e)

            # Step 4, send model the info on the function call and function response
            message_results.append({
                "role": "function",
                "name": function_name,
                "content": function_response,
            })

            second_response = self.execute_direct(
                model=self.model,
                messages=messages + message_results,
            )

            message = second_response['choices'][0]['message']  # type: ignore
            message_results.append(message)
            counter += 1

        return message_results

    def execute_query(
        self,
        query: Union[str, List[Dict]],
        data: str,
    ) -> ExprNode:
        logging.debug('execute_query query={}'.format(query))

        functions = [Helpers.get_function_description(f, True) for f in self.agents]

        function_system_message = '''
            Dont make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
            If a function returns a value that does not address the users request, you should call a different function.
        '''

        self.messages.append({'role': 'system', 'content': function_system_message})
        self.messages.append({'role': 'user', 'content': query})
        if data:
            self.messages.append({'role': 'user', 'content': data})

        chat_response = self.__chat_completion_request(
            self.messages,
            functions,
        )

        if len(chat_response) == 0:
            return Result(error='The model could not execute the query.')

        for message in chat_response:
            self.messages.append(message)

        if not self.chat:
            self.messages = []

        conversation: List[AstNode] = []
        for message in self.messages:
            if message['role'] == 'user':
                conversation.append(User(message['content']))
            elif message['role'] == 'system':
                conversation.append(System(message['content']))
            elif message['role'] == 'assistant':
                conversation.append(Assistant(message['content']))

        return Result(result={'answer': chat_response[-1]['content']}, conversation=conversation)

    def can_execute(self, query: Union[str, List[Dict]]) -> bool:
        return True

    def execute(self, query: Union[str, List[Dict]], data: str) -> ExprNode:
        return self.execute_query(query, data)


class Parser():
    def __init__(
        self,
        message_type = User,
    ):
        self.message: str = ''
        self.remainder: str = ''
        self.index = 0
        self.agents: List[Callable] = []
        self.message_type: type = message_type

    def consume(self, token: str):
        if token in self.remainder:
            self.remainder = self.remainder[:self.remainder.index(token) + len(token)]

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
            function_context = text  # Helpers.extract_context(text, start_token, end_token, stop_tokens=['\n'])

            split_text = Helpers.split_between(text, start_token, end_token)
            sequence.append(Text(split_text[0]))
            if function_call:
                function_call.context = Content(Text(function_context))
                sequence.append(function_call)
            else:
                sequence.append(
                    Text(function_call_str)
                )
            text = split_text[1]
            self.remainder = text

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

        re = self.remainder

        while re.strip() != '':
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
        execution_flow: ExecutionFlow[AstNode],
    ) -> Program:
        self.message = message
        self.agents = agents

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

        assistant: Assistant = executor.execute_one_shot(
            system_message=query_understanding['system_message'],
            user_message=query_understanding['user_message'],
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
                return UncertainOrError(conversation=[Text('The function could not execute. It raised an exception: {}'.format(e))])

            function_call.call_response = function_response
            return function_call

        elif isinstance(statement, NaturalLanguage) and not statement.response() and callee:
            # callee provides context for the natural language statement
            messages: List[Message] = []
            messages.extend(statement.messages)

            response = executor.chat_execute(messages=messages)
            statement.call_response = str(response.message)
            return statement

        elif isinstance(statement, NaturalLanguage) and statement.response():
            #



    def execute_program(
        self,
        program: Program,
        execution: ExecutionFlow[Statement]
    ):
        executor = self.execution_contexts[0]
        answers: List[Statement] = []

        for s in reversed(program.statements):
            execution.push(s)

        while statement := execution.pop():
            answers.append(self.execute_statement(statement))



    def execute(
        self,
        prompt: str,
    ):
        # pick the right execution context that will get the task done
        # for now, we just grab the first
        executor = self.execution_contexts[0]

        # create an execution flow
        execution: ExecutionFlow[Statement] = ExecutionFlow(Order.QUEUE)

        # assess the type of task
        classification = self.classify_tool_or_direct(prompt)

        if 'tool' in classification:
            program = self.parse(prompt)
            self.execute_program(program)
        else:
            self.execute_simple(prompt)


    def rewrite(self, node: AstNode) -> AstNode:
        def function_call_rewriter(node: AstNode) -> AstNode:
            if isinstance(node, Text):
                text = node.text
                sequence: List[AstNode] = []

                if (
                        ('[[' not in text and ']]' not in text)
                        and ('```python' not in text)
                        and ('[' not in text and ']]' not in text)
                ):
                    return node

                while (
                    ('[[' in text and ']]' in text)
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
                    function_call: Optional[FunctionCall] = self.parser.to_function_call(function_call_str)
                    function_context = text  # Helpers.extract_context(text, start_token, end_token, stop_tokens=['\n'])

                    split_text = Helpers.split_between(text, start_token, end_token)
                    sequence.append(Text(split_text[0]))
                    if function_call:
                        function_call.context = Content(Text(function_context))
                        sequence.append(function_call)
                    else:
                        sequence.append(
                            Text(function_call_str)
                        )
                    text = split_text[1]

                # add the left over text
                sequence.append(Text(text))
                return Content(sequence)
            else:
                return node

        rewriter = ReplacementVisitor(
            node_lambda=lambda node: isinstance(node, Text),
            replacement_lambda=function_call_rewriter
        )

        return tree_traverse(node, rewriter)

    def parse(
        self,
        prompt: str,
        max_api_calls: int = 5,
    ) -> Program:
        executor = self.execution_contexts[0]

        execution: ExecutionFlow[ExprNode] = ExecutionFlow(Order.QUEUE)

        if Helpers.calculate_tokens(prompt) > max_tokens and prompt_strategy == PromptStrategy.THROW:
            raise Exception('Prompt and data too long: {}, {}'.format(prompt, data))

        elif Helpers.calculate_tokens(prompt) > max_tokens and prompt_strategy == PromptStrategy.SEARCH:
            sections = Helpers.chunk_and_rank(prompt, data, max_chunk_length=max_tokens)
            calls: List[Call] = []
            for section in sections:
                calls.append(
                    LLMCall(
                        messages=[User(Content(Text(prompt)))],
                        data=Data(data),
                        system=System(),
                        executor=executor,
                    )
                )
            execution.push(ChainedCall(calls))

        elif Helpers.calculate_tokens(prompt) > max_tokens and prompt_strategy == PromptStrategy.SUMMARIZE:
            data_chunks = Helpers.split_text_into_chunks(data)
            if max_api_calls == 0:
                max_api_calls = len(data_chunks)

            calls = []
            for chunk in data_chunks[0:max_api_calls]:
                calls.append(
                    LLMCall(
                        messages=[User(Content(Text(prompt)))],
                        data=Data(chunk),
                        system=System(),
                        executor=executor,
                    )
                )
            execution.push(ChainedCall(calls))

        else:
            execution.push(
                LLMCall(
                    messages=[User(Content(Text(prompt)))],
                    data=Data(data),
                    system=System(),
                    executor=executor,
                )
            )

        return execution

    def parse_tree_execute(self, execution: ExecutionFlow[AstNode]) -> List[AstNode]:
        def __increment_counter():
            nonlocal counter
            nonlocal node
            counter += 1
            if counter >= execution.count():
                node = None
            else:
                node = execution[counter]

        if execution.count() <= 0:
            raise ValueError('No nodes to execute')

        results: List[AstNode] = []

        counter = 0
        node: Optional[AstNode] = execution.peek()
        execution_backup = node.executor

        while node is not None:
            if isinstance(node, LLMCall):
                assistant_result: Assistant = node.executor.execute_with_tools(node)  # str(node.user.prompt), node.data.data.text)
                rewritten_result: Assistant = cast(Assistant, self.rewrite(assistant_result))

                if any(tree_map(rewritten_result, lambda n: isinstance(n, Call))):
                    # we still have work to do
                    execution.push(rewritten_result)
                else:
                    results.append(Result(conversation=[assistant_result.message]))
                    __increment_counter()

            elif isinstance(node, ChainedCall):
                for call in node.calls:
                    if type(call) is LLMCall:
                        execute_node = call.executor.execute([Message.to_dict(m) for m in call.messages], call.data.text)
                        if type(execute_node) is Result:
                            results.append(execute_node)
                            __increment_counter()
                        elif type(execute_node) is ExprNode:
                            execution.push(execute_node)

            elif isinstance(node, FunctionCall):
                # unpack the args, call the function
                function_call = node
                function_args_desc = node.args
                function_args = {}

                # Note: the JSON response from the model may not be valid JSON
                func: Callable | None = Helpers.first(lambda f: f.__name__ in function_call.name, self.agents)

                if not func:
                    raise ValueError('Could not find function {}'.format(function_call.name))

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
                    function_response = 'The function could not execute. It raised an exception: {}'.format(e)

                def match_function(node: AstNode, name: str, args: Dict[str, object]) -> bool:
                    return (
                        isinstance(node, FunctionCall)
                        and node.name == function_call.name
                        and node.args == function_call.args
                    )

                callee = execution.peek(-1)

                # remove the function call
                rewriter = ReplacementVisitor(
                    node_lambda=lambda n: match_function(n, function_call.name, function_args),
                    replacement_lambda=lambda n: Content(Text(''))
                )

                tree_traverse(callee, rewriter)

                tool_result = execution_backup.execute_tool_response(
                    callee._llm_call_context,
                    function_response,
                    '{}({})'.format(function_call.name, ','.join(function_args.values()))
                )

                results.append(tool_result)

                # todo: this is the wrong way to do this, but for now it works
                # grab the callee
                # callee = execution.peek(-1)

                # if isinstance(callee, Assistant):
                #     # assistant probably shouldn't have all the system/user prompt stuff
                #     # attached to it, as this can be captured by the execution stack
                #     messages = callee._messages_context
                #     messages.append(User(Content(Text(function_response))))
                #     callee_executor = cast(OpenAIExecutor, callee._llm_call_context.executor)
                #     call = LLMCall(messages=messages, system=callee._system_context, executor=callee_executor)
                #     response_message = call.executor.execute_direct(messages=messages, functions=[], chat_format=True)

                #     chat_response = response_message['choices'][0]['message']  # type: ignore
                #     results.append(Result(conversation=[Text(chat_response)]))

                #     # pop the FunctionCall
                #     execution.pop()

            elif isinstance(node, Assistant) and any(tree_map(node.message, lambda n: isinstance(n, Call))):
                # there's still work to do
                execution.push(node)

                for ast_node in tree_map(node.message, lambda n: n):
                    if isinstance(ast_node, Call):
                        execution.push(ast_node)

            # we're done
            elif isinstance(node, Assistant):
                results.append(Result([node.message]))
                __increment_counter()
            elif isinstance(node, Result):
                results.append(node)
                __increment_counter()

            node = execution.pop()
        return results


class Repl():
    def __init__(
        self,
        executors: List[Executor]
    ):
        self.executors: List[Executor] = executors
        self.agents: List[Agent] = []

    def print_response(self, result: Union[str, Result]):
        if type(result) is str:
            rich.print(f'[bold cyan]{result}[/bold cyan] ')

        elif type(result) is Result:
            rich.print('[bold cyan]Conversation:[/bold cyan] ')
            for message in result.conversation:
                rich.print('  ' + str(message))
            rich.print(f'[bold cyan]Answer:[/bold cyan]')
            if type(result.result) is str:
                rich.print(f'{result.result}')
            elif type(result.result) is dict and 'answer' in result.result:
                rich.print('{}'.format(cast(dict, result.result)['answer']))
            else:
                rich.print('{}'.format(str(result.result)))
        else:
            rich.print(str(result))

    def repl(self):
        console = rich.console.Console()
        history = FileHistory(".repl_history")

        rich.print()
        rich.print('[bold]I am a helpful assistant.[/bold]')
        rich.print()

        executor_contexts = self.executors
        executor_names = [executor.name() for executor in executor_contexts]

        current_context = 'openai'
        parser = ParserController(
            execution_contexts=executor_contexts,
            chat_mode=False,
            current_chat_context=[],
            agents = agents,
        )

        commands = {
            'exit': 'exit the repl',
            '/context': 'change the current context',
            '/chat': 'change to chat mode',
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

                if '/chat' in query:
                    chat = Helpers.in_between(query, '/chat', '\n').strip()
                    if chat == 'true':
                        for executor in executor_contexts:
                            executor.chat_context(True)
                            parser.chat_mode = True
                    elif chat == 'false':
                        for executor in executor_contexts:
                            executor.chat_context(False)
                            parser.chat_mode = False
                    else:
                        rich.print('Invalid chat value: {}'.format(chat))
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

                # for agent in self.agents:
                #     if agent.is_task(query):
                #         result = agent.perform_task(query)

                # result = self.execute(query=query, executors=executor_contexts)
                # self.print_response(result)

                data = prompt('data>> ', history=history, enable_history_search=True, vi_mode=True)

                expr_tree = parser.parse(prompt=query, data=data)
                results = parser.parse_tree_execute(expr_tree)

                for result in results:
                    self.print_response(str(result))
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

    def langchain_executor():
        openai_executor = LangChainExecutor(openai_key, verbose=verbose)
        return openai_executor

    def openai_executor():
        openai_executor = OpenAIExecutor(openai_key, verbose=verbose, agents=agents)
        return openai_executor

    executors = {
        'openai': openai_executor(),
        'langchain': langchain_executor(),
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
