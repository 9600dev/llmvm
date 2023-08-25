from abc import ABC, abstractmethod
from enum import Enum
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

import pandas as pd
import pandas_gpt

T = TypeVar('T')

class Visitor(ABC):
    @abstractmethod
    def visit(self, node: 'AstNode') -> 'AstNode':
        pass


class Executor(ABC):
    @abstractmethod
    def execute(
        self,
        messages: List['Message'],
        max_completion_tokens: int = 2048,
        temperature: float = 1.0,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def execute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: str = 'gpt-3.5-turbo-16k-0613',
        max_completion_tokens: int = 2048,
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

    @abstractmethod
    def max_prompt_tokens(self, completion_token_count: int = 2048) -> int:
        pass

    @abstractmethod
    def calculate_tokens(
        self,
        messages: List['Message'] | str,
        extra_str: str = '',
    ) -> int:
        pass


class Controller():
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def execute_llm_call(
        self,
        message: 'Message',
        context_messages: List['Message'],
        query: str,
        original_query: str,
        prompt_filename: Optional[str] = None,
        completion_tokens: int = 2048,
        temperature: float = 0.0,
        lifo: bool = False,
    ) -> 'Assistant':
        pass


class AstNode(ABC):
    def __init__(
        self
    ):
        pass

    def accept(self, visitor: Visitor) -> 'AstNode':
        return visitor.visit(self)


class Content(AstNode):
    def __init__(
        self,
        sequence: Optional[AstNode | List[AstNode] | str] = None,
    ):
        if sequence is None:
            self.sequence = ''
            return

        if isinstance(sequence, str):
            self.sequence = [sequence]
        elif isinstance(sequence, Content):
            self.sequence = sequence.sequence  # type: ignore
        elif isinstance(sequence, AstNode):
            self.sequence = [sequence]
        elif isinstance(sequence, list) and len(sequence) > 0 and isinstance(sequence[0], AstNode):
            self.sequence = sequence
        else:
            raise ValueError('not supported')

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
            return User(Content(content))
        elif role == 'system':
            return System(Content(content))
        elif role == 'assistant':
            return Assistant(Content(content))
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
        message: Content = Content('''
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
        return f'{self.message}'

    def __add__(self, other):
        other_message = str(other)

        assistant = Assistant(
            message=Content(str(self.message) + other_message),
            messages_context=self._messages_context,
            system_context=self._system_context,
            llm_call_context=self._llm_call_context,
        )
        return assistant

    def __repr__(self):
        return f'Assistant({self.message} {self.error})'


class Statement(AstNode):
    def __init__(
        self,
        ast_text: Optional[str] = None,
    ):
        self._result: object = None
        self._ast_text: Optional[str] = ast_text

    def __str__(self):
        if self._result:
            return str(self._result)
        else:
            return str(type(self))

    def result(self):
        return self._result

    def token(self):
        return 'statement'


class DataFrame(Statement):
    def __init__(
        self,
        elements: List,
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)
        self.elements = elements

    def token(self):
        return 'dataframe'


class Call(Statement):
    def __init__(
        self,
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)


class FunctionCallMeta(Call):
    def __init__(
        self,
        callsite: str,
        func: Callable,
        result: object,
        lineno: Optional[int],
    ):
        self.callsite = callsite
        self.func = func
        self._result = result
        self.lineno = lineno

    def result(self) -> object:
        return self._result

    def token(self):
        return 'functioncallmeta'

    def __str__(self):
        return str(self._result)


class PandasMeta(Call):
    def __init__(
        self,
        expr_str: str,
        pandas_df: pd.DataFrame,
    ):
        self.expr_str = expr_str
        self.df = pandas_df

    def result(self) -> object:
        return self._result

    def token(self):
        return 'pandasmeta'

    def __str__(self):
        return str(self.df)

    def ask(self, *args, **kwargs) -> object:
        self._result = self.df.ask(*args, **kwargs)  # type: ignore
        return self._result


class FunctionCall(Call):
    def __init__(
        self,
        name: str,
        args: List[Dict[str, object]],
        types: List[Dict[str, object]],
        context: Content = Content(),
        func: Optional[Callable] = None,
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)
        self.name = name
        self.args = args
        self.types = types
        self.context = context
        self._result: Optional[Content] = None
        self.func: Optional[Callable] = func

    def to_code_call(self):
        arguments = []
        for arg in self.args:
            for k, v in arg.items():
                arguments.append(v)

        str_args = ', '.join([str(arg) for arg in arguments])
        return f'{self.name}({str_args})'

    def to_definition(self):
        definitions = []
        for arg in self.types:
            for k, v in arg.items():
                definitions.append(f'{k}: {v}')

        str_args = ', '.join([str(t) for t in definitions])
        return f'{self.name}({str_args})'

    def token(self):
        return 'function_call'

class Answer(Statement):
    def __init__(
        self,
        conversation: List[Message] = [],
        result: object = None,
        error: object = None,
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)
        self.conversation: List[Message] = conversation
        self._result = result
        self.error = error

    def __str__(self):
        ret_result = f'Answer({self.result})\n'
        ret_result = f'Error: {self.error}\n'
        ret_result += '  Conversation:\n'
        ret_result += '\n  '.join([str(n) for n in self.conversation])
        return ret_result

    def token(self):
        return 'answer'


class UncertainOrError(Statement):
    def __init__(
        self,
        error_message: Content,
        supporting_conversation: List[AstNode] = [],
        supporting_result: object = None,
        supporting_error: object = None,
    ):
        super().__init__()
        self.error_message = error_message,
        self.supporting_conversation = supporting_conversation
        self._result = supporting_result
        self.supporting_error = supporting_error

    def __str__(self):
        ret_result = f'UncertainOrError({str(self.error_message)} {self.supporting_error}, {self.result})\n'
        ret_result += '  Conversation:\n'
        ret_result += '\n  '.join([str(n) for n in self.supporting_conversation])
        return ret_result

    def token(self):
        return 'uncertain_or_error'


class LambdaVisitor(Visitor):
    def __init__(
        self,
        node_lambda: Callable[[AstNode], Any],
    ):
        self.node_lambda = node_lambda

    def visit(self, node: AstNode) -> AstNode:
        if self.node_lambda(node):
            self.node_lambda(node)
            return node
        else:
            return node
