from abc import ABC, abstractmethod
from enum import Enum
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

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
        temperature: float = 1.0,
        max_completion_tokens: int = 2048,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def execute_with_agents(
        self,
        call: 'LLMCall',
        agents: List[Callable],
        temperature: float = 1.0,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def execute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: str = 'gpt-3.5-turbo-16k',
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
        messages: List['Message'],
        extra_str: str = '',
    ) -> int:
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


class StackNode(Statement):
    def __init__(
        self,
        value: int,
    ):
        self.value = value

    def __str__(self):
        if self.value == 0:
            return 'stack()'
        else:
            return f'stack({self.value})'

    def token(self):
        return 'stack'


class Get(Statement):
    def __init__(
        self,
        variable: str,
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)
        self.variable = variable

    def __str__(self):
        return f'get("{self.variable}")'

    def token(self):
        return 'get'


class Set(Statement):
    def __init__(
        self,
        variable: str,
        name: str = '',
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)
        self.variable = variable
        self.name = name

    def __str__(self):
        if self.name:
            return f'set("{self.variable}, {self.name}")'
        else:
            return f'set("{self.variable}")'

    def token(self):
        return 'set'


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


class LLMCall(Call):
    def __init__(
        self,
        context: Optional[StackNode] = None,
        message: Message = User(Content('')),
        supporting_messages: List[Message] = [],
        supporting_system: Optional[System] = None,
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)
        self.context: Optional[StackNode] = context
        self.message = message

        self.supporting_messages: List[Message] = supporting_messages
        self.supporting_system: Optional[System] = supporting_system

    def token(self):
        return 'llm_call'


class ForEach(Statement):
    def __init__(
        self,
        lhs: Statement,
        rhs: Statement,
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)
        self.lhs = lhs
        self.rhs = rhs
        self._result = []

    def result(self) -> object:
        return self._result

    def token(self):
        return 'foreach'

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
        conversation: List[AstNode] = [],
        result: object = None,
        error: object = None,
        ast_text: Optional[str] = None,
    ):
        super().__init__(ast_text)
        self.conversation: List[AstNode] = conversation
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


class Program(AstNode):
    def __init__(
        self,
        executor: Executor,
    ):
        self.executor: Executor = executor
        self.statements: List[Statement] = []
        self.conversation: List[Message] = []
        self.runtime_stack: Stack[Statement] = Stack[Statement]()
        self.runtime_registers: Dict[str, Tuple[str, Statement]] = {}
        self.answers: List[Answer] = []

        self.executed_stack: Stack[Statement] = Stack[Statement]()
        self.original_query: str = ''
        self.errors: List[UncertainOrError] = []


class PromptStrategy(Enum):
    THROW = 'throw'
    SEARCH = 'search'
    SUMMARIZE = 'summarize'

class Order(Enum):
    STACK = 'stack'
    QUEUE = 'queue'


class Stack(Generic[T]):
    def __init__(self):
        self.stack: List[T] = []

    def push(self, item: T):
        self.stack.append(item)

    def pop(self) -> Optional[T]:
        if len(self.stack) == 0:
            return None
        return self.stack.pop()

    def peek(self, n: int = 1) -> List[T]:
        if len(self.stack) == 0:
            return []

        if n > 0:
            result = []
            for i in range(n):
                result.append(self.stack[-1 - i])
            return result
        elif n == 0:
            return self.stack
        else:
            return []

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def count(self) -> int:
        return len(self.stack)

    def __getitem__(self, index):
        return self.stack[index]


class ExecutionFlow(Generic[T]):
    def __init__(self, order: Order = Order.QUEUE):
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
                if isinstance(n, AstNode):
                    visited.extend(tree_map(n, call))
    elif isinstance(node, User):
        visited.extend(tree_map(node.message, call))
    elif isinstance(node, Assistant):
        visited.extend(tree_map(node.message, call))
    elif isinstance(node, DataFrame):
        for n in node.elements:
            visited.extend(tree_map(n, call))
    elif isinstance(node, ForEach):
        visited.extend(tree_map(node.lhs, call))
        visited.extend(tree_map(node.rhs, call))
    elif isinstance(node, FunctionCall):
        visited.extend(tree_map(node.context, call))
    elif isinstance(node, LLMCall):
        if node.context:
            visited.extend(tree_map(node.context, call))
        visited.extend(tree_map(node.message, call))
        for n in node.supporting_messages:
            visited.extend(tree_map(n, call))
        if node.supporting_system:
            visited.extend(tree_map(node.supporting_system, call))
    elif isinstance(node, Program):
        for statement in node.statements:
            visited.extend(tree_map(statement, call))
    else:
        raise ValueError('not implemented')
    return visited


def tree_traverse(node, visitor: Visitor, post_order: bool = True):
    def flatten(lst):
        def __has_list(lst):
            for item in lst:
                if isinstance(item, list):
                    return True
            return False

        def __inner_flatten(lst):
            return [item for sublist in lst for item in sublist]

        while __has_list(lst):
            lst = __inner_flatten(lst)
        return lst

    accept_result = None
    if not post_order:
        accept_result = node.accept(visitor)

    if node is None:
        return node
    elif isinstance(node, Content):
        if isinstance(node.sequence, list):
            node.sequence = flatten(
                [cast(AstNode, tree_traverse(child, visitor)) for child in node.sequence if isinstance(child, AstNode)]
            )
        elif isinstance(node, AstNode):
            node.sequence = [cast(AstNode, tree_traverse(node.sequence, visitor))]  # type: ignore
    elif isinstance(node, Assistant):
        node.message = cast(Content, tree_traverse(node.message, visitor))
    elif isinstance(node, Message):
        node.message = cast(Content, tree_traverse(node.message, visitor))
    elif isinstance(node, DataFrame):
        node.elements = [cast(AstNode, tree_traverse(child, visitor)) for child in node.elements]
    elif isinstance(node, ForEach):
        node.lhs = cast(Statement, tree_traverse(node.lhs, visitor))
        node.rhs = cast(Statement, tree_traverse(node.rhs, visitor))
    elif isinstance(node, LLMCall):
        node.context = cast(StackNode, tree_traverse(node.context, visitor))
        node.message = cast(Message, tree_traverse(node.message, visitor))
        node.supporting_messages = [cast(User, tree_traverse(child, visitor)) for child in node.supporting_messages]
        if node.supporting_system:
            node.supporting_system = cast(System, tree_traverse(node.supporting_system, visitor))
    elif isinstance(node, FunctionCall):
        node.context = cast(Content, tree_traverse(node.context, visitor))
    elif isinstance(node, Program):
        node.statements = [cast(Statement, tree_traverse(child, visitor)) for child in node.statements]

    if post_order:
        return node.accept(visitor)
    else:
        return accept_result


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
