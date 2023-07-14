from abc import ABC, abstractmethod
from enum import Enum
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

from helpers.helpers import Helpers
from helpers.logging_helpers import setup_logging

logging = setup_logging()


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
    ) -> 'Assistant':
        pass

    @abstractmethod
    def execute_with_agents(
        self,
        call: 'NaturalLanguage',
        agents: List[Callable],
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
        if isinstance(sequence, Content):
            self.sequence = sequence.sequence  # type: ignore
        elif isinstance(sequence, Text):
            self.sequence = sequence
        elif isinstance(sequence, AstNode):
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
        return f'{self.message}'

    def __repr__(self):
        return f'Assistant({self.message} {self.error})'


class Statement(AstNode):
    def __init__(
        self,
    ):
        self._result: object = None

    def __str__(self):
        if self._result:
            return str(self._result)
        else:
            return str(type(self))

    def result(self):
        return self._result


class Call(Statement):
    def __init__(
        self,
    ):
        super().__init__()


class NaturalLanguage(Call):
    def __init__(
        self,
        messages: List[Message],
        system: Optional[System] = None,
        executor: Optional[Executor] = None,
    ):
        super().__init__()
        self.messages: List[Message] = messages
        self.system = system
        self.executor = executor


class Continuation(Statement):
    def __init__(
        self,
        lhs: Statement,
        rhs: Statement,
    ):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def result(self) -> object:
        return self.rhs.result()


class ForEach(Statement):
    def __init__(
        self,
        lhs: List[Statement],
        rhs: Statement,
    ):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs

    def result(self) -> object:
        return self._result
        # return self.rhs.result()


class DataFrame(Statement):
    def __init__(
        self,
        columns: List[str],
        rows: List[List[str]],
    ):
        super().__init__()
        self.columns = columns
        self.rows = rows


class FunctionCall(Call):
    def __init__(
        self,
        name: str,
        args: List[Dict[str, object]],
        types: List[Dict[str, object]],
        context: Content = Content(Text('')),
        func: Optional[Callable] = None,
    ):
        super().__init__()
        self.name = name
        self.args = args
        self.types = types
        self.context = context
        self._result: Optional[Text] = None
        self.func: Optional[Callable] = func

    def to_code_call(self):
        arguments = []
        for arg in self.args:
            for k, v in arg.items():
                arguments.append(v)

        str_args = ', '.join([str(arg) for arg in arguments])
        return f'{self.name}({str_args})'

class Answer(Statement):
    def __init__(
        self,
        conversation: List[AstNode] = [],
        result: object = None,
        error: object = None,
    ):
        super().__init__()
        self.conversation = conversation,
        self._result = result
        self.error = error

    def __str__(self):
        ret_result = f'Answer({self.result})\n'
        ret_result = f'Error: {self.error}\n'
        ret_result += '  Conversation:\n'
        ret_result += '\n  '.join([str(n) for n in self.conversation])
        return ret_result


class UncertainOrError(Statement):
    def __init__(
        self,
        conversation: List[AstNode] = [],
        result: object = None,
        error: object = None,
    ):
        super().__init__()
        self.conversation = conversation,
        self._result = result
        self.error = error

    def __str__(self):
        ret_result = f'UncertainOrError({self.error}, {self.result})\n'
        ret_result += '  Conversation:\n'
        ret_result += '\n  '.join([str(n) for n in self.conversation])
        return ret_result


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
