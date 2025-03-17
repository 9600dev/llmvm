import base64
import copy
import datetime as dt
import importlib
import json
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from importlib import resources
from typing import (Any, Awaitable, Callable, Optional, OrderedDict, TextIO, Type,
                    TypedDict, TypeVar, Union, cast)

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from llmvm.common.container import Container
from llmvm.common.logging_helpers import setup_logging

logging = setup_logging()


T = TypeVar('T')


class DownloadParams(TypedDict):
    url: str
    goal: str
    search_term: str


class TokenPriceCalculator():
    def __init__(
        self,
        price_file: str = 'model_prices_and_context_window.json',
    ):
        self.price_file = resources.files('llmvm') / price_file
        self.prices = self.__load_prices()
        self.absolute_defaults = {
            'anthropic': {
                'max_input_tokens': 200000,
                'max_output_tokens': 4096,
            },
            'openai': {
                'max_input_tokens': 128000,
                'max_output_tokens': 4096,
            },
            'gemini': {
                'max_input_tokens': 2000000,
                'max_output_tokens': 4096,
            },
            'deepseek': {
                'max_input_tokens': 64000,
                'max_output_tokens': 4096,
            },
            'bedrock': {
                'max_input_tokens': 300000,
                'max_output_tokens': 4096,
            },
        }

    def __load_prices(self):
        with open(self.price_file, 'r') as f:  # type: ignore
            json_prices = json.load(f)
            return json_prices

    def __absolute_default(self, executor: Optional[str], key: str) -> int | None:
        if not executor:
            return None

        executor_value = None
        if executor:
            executor_defaults = self.absolute_defaults.get(executor)
            if executor_defaults:
                executor_value = executor_defaults.get(key)
                if executor_value: return executor_value

    def get(self, model: str, key: str, executor: Optional[str] = None) -> Optional[Any]:
        if model in self.prices and key in self.prices[model]:
            return self.prices[model][key]
        elif executor and f'{executor}/{model}' in self.prices and key in self.prices[f'{executor}/{model}']:
            return self.prices[f'{executor}/{model}'][key]
        return None

    def input_price(
        self,
        model: str,
        executor: Optional[str] = None
    ) -> float:
        return self.get(model, 'input_cost_per_token', executor) or 0.0

    def output_price(
        self,
        model: str,
        executor: Optional[str] = None
    ) -> float:
        return self.get(model, 'output_cost_per_token', executor) or 0.0

    def max_input_tokens(
        self,
        model: str,
        default: int,
        executor: Optional[str] = None,
    ) -> int:
        max_input_tokens = self.get(model, 'max_input_tokens', executor) or default or self.__absolute_default(executor, 'max_input_tokens')

        if not max_input_tokens:
            raise ValueError(f'max_input_tokens not found for model {model} and executor {executor} and no default provided.')

        return cast(int, max_input_tokens)

    def max_output_tokens(
        self,
        model: str,
        default: int,
        executor: Optional[str] = None,
    ) -> int:
        max_output_tokens = self.get(model, 'max_output_tokens', executor) or default or self.__absolute_default(executor, 'max_output_tokens')

        if not max_output_tokens:
            raise ValueError(f'max_output_tokens not found for model {model} and executor {executor} and no default provided.')

        return cast(int, max_output_tokens)

def bcl(module_or_path):
    def decorator(cls):
        class NewClass(cls):
            def __init__(self, *args, **kwargs):
                try:
                    if '.' in module_or_path:
                        # Treat it as a module name
                        module = importlib.import_module(module_or_path)
                        self.arg_string = getattr(module, 'arg_string', None)
                    else:
                        # Treat it as a file path
                        self.arg_string = resources.files(module_or_path).read_text()
                except (ImportError, AttributeError, FileNotFoundError):
                    self.arg_string = None

                super(NewClass, self).__init__(*args, **kwargs)

            def print_arg_string(self):
                if self.arg_string:
                    print(f"Decorator argument: {self.arg_string}")
                else:
                    print("Decorator argument not found.")

        NewClass.__name__ = cls.__name__
        NewClass.__doc__ = cls.__doc__
        return NewClass
    return decorator


async def awaitable_none(a: 'AstNode') -> None:
    pass


def none(a: 'AstNode') -> None:
    pass


class ContentEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json') and callable(getattr(obj, 'to_json')):
            return obj.to_json()
        return super().default(obj)


class TokenCountCache:
    _instance = None

    def __new__(cls, max_size: int = 500):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.max_size = max_size
            cls._instance.cache = OrderedDict()
        return cls._instance

    def __init__(self, max_size: int = 500):
        # The initialization will only happen once
        # subsequent calls will not modify max_size
        if not hasattr(self, 'cache'):
            self.max_size = max_size
            self.cache = OrderedDict()

    def _generate_key(self, messages: list[dict[str, Any]]) -> str:
        return str(hash(str(messages)))

    def get(self, messages: list[dict[str, Any]]) -> Optional[int]:
        key = self._generate_key(messages)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, messages: list[dict[str, Any]], token_count: int) -> None:
        key = self._generate_key(messages)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = token_count
            return
        self.cache[key] = token_count
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)


class TokenPerf:
    def __init__(
        self,
        name: str,
        executor_name: str,
        model_name: str,
        prompt_len: int = 0,
        enabled: bool = Container.get_config_variable('profiling', 'LLMVM_PROFILING', default=False),
        log_file: str = Container.get_config_variable(
            'profiling_file',
            'LLMVM_PROFILING_FILE',
            default='~/.local/share/llmvm/trace.log'
        ),
        request_id: str = '',
        total_tokens: int = 0,
    ):
        self._name: str = name
        self._executor: str = executor_name
        self._model: str = model_name
        self._start: float = 0.0
        self._stop: float = 0.0
        self._prompt_len: int = prompt_len
        self._completion_len: int = 0
        self._ticks: list[float] = []
        self.enabled = enabled
        self.log_file = log_file
        self.calculator = TokenPriceCalculator()
        self.request_id = request_id
        self.stop_reason = ''
        self.stop_token = ''
        self.total_tokens = total_tokens
        self.object = None

    def start(self):
        if self.enabled:
            self._start = time.perf_counter()

    def stop(self):
        if self.enabled:
            self._stop = time.perf_counter()

        return self.result()

    def reset(self):
        self._ticks = []

    def result(self):
        if self.enabled:
            ttlt = self._stop - self._start
            ttft = self._ticks[0] - self._start if self._ticks else 0
            completion_time = ttlt - ttft
            try:
                s_tok_sec = len(self._ticks) / ttlt
            except ZeroDivisionError:
                s_tok_sec = 0.0
            try:
                p_tok_sec = self._prompt_len / ttft
            except ZeroDivisionError:
                p_tok_sec = 0.0
            return {
                'name': self._name,
                'executor': self._executor,
                'model': self._model,
                'ttlt': ttlt,
                'ttft': ttft,
                'completion_time': completion_time,
                'prompt_len': self._prompt_len,
                'completion_len': self._completion_len if self._completion_len > 0 else len(self._ticks),
                's_tok_sec': s_tok_sec,
                'p_tok_sec': p_tok_sec,
                'p_cost': self._prompt_len * self.calculator.input_price(self._model, self._executor),
                's_cost': len(self._ticks) * self.calculator.output_price(self._model, self._executor),
                'request_id': self.request_id,
                'stop_reason': self.stop_reason,
                'stop_token': self.stop_token,
                'total_tokens': self.total_tokens,
                'ticks': self.ticks()
            }
        else:
            return {}

    def tick(self):
        if self.enabled:
            self._ticks.append(time.perf_counter())

    def ticks(self):
        if self.enabled:
            return [self._ticks[i] - self._ticks[i - 1] for i in range(1, len(self._ticks))]
        else:
            return []

    def __str__(self):
        if self.enabled:
            res = self.result()
            result = f'{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")},{res["name"]},{res["executor"]},{res["model"]},{res["ttlt"]},{res["ttft"]},{res["completion_time"]},{res["prompt_len"]},{res["completion_len"]},{res["p_tok_sec"]},{res["s_tok_sec"]},{res["request_id"]},{res["stop_reason"]},{res["stop_token"]},{res["total_tokens"]},{",".join([f"{t:.8f}" for t in res["ticks"]])}'
            return result
        else:
            return ''

    def debug(self):
        if self.enabled:
            res = self.result()
            # output \n to the debug stream without using logging.debug
            import sys
            sys.stderr.write('\n')
            logging.debug(f"ttft: {res['ttft']:.2f} ttlt: {res['ttlt']:.2f} completion_time: {res['completion_time']:.2f}")
            logging.debug(f"prompt_len: {res['prompt_len']} completion_len: {res['completion_len']} model: {res['model']}")
            logging.debug(f"p_tok_sec: {res['p_tok_sec']:.2f} s_tok_sec: {res['s_tok_sec']:.2f} stop_reason: {res['stop_reason']}")
            logging.debug(f"p_cost: ${res['p_cost']:.5f} s_cost: ${res['s_cost']:.5f} request_id: {res['request_id']}")

    def log(self):
        if self.enabled:
            self.debug()
            if not os.path.exists(os.path.expanduser(self.log_file)):
                with open(os.path.expanduser(self.log_file), 'w') as f:
                    f.write('name,executor,model,ttlt,ttft,prompt_tokens,completion_time,prompt_len,completion_len,p_tok_sec,s_tok_sec,p_cost,s_cost,request_id,stop_reason,stop_token,total_tokens,ticks\n')
            with open(os.path.expanduser(self.log_file), 'a') as f:
                result = str(self)
                f.write(result + '\n')
                return self.result()
        else:
            return {
                'name': self._name,
                'executor': self._executor,
                'ttlt': 0.0,
                'ttft': 0.0,
                'completion_time': 0.0,
                'prompt_len': 0,
                'completion_len': 0,
                'p_tok_sec': 0.0,
                's_tok_sec': 0.0,
                'p_cost': 0.0,
                's_cost': 0.0,
                'request_id': '',
                'stop_reason': '',
                'stop_token': '',
                'total_tokens': 0,
                'ticks': []
            }


########################################################################################
## Model classes
########################################################################################
class Visitor(ABC):
    @abstractmethod
    def visit(self, node: 'AstNode') -> 'AstNode':
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
        sequence: str | bytes | list['Content'],
        content_type: str = '',
        url: str = '',
    ):
        self.sequence = sequence
        self.url = url
        self.content_type = content_type
        self.original_sequence: object = None

    def __repr__(self):
        return f'Content({self.sequence.__repr__()})'

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_str(self) -> str:
        pass

    def to_json(self) -> dict:
        sequence_json = ''
        if isinstance(self.sequence, bytes):
            sequence_json = base64.b64encode(self.sequence).decode('utf-8')
        elif isinstance(self.sequence, str):
            sequence_json = self.sequence
        elif isinstance(self.sequence, list):
            sequence_json = [c.to_json() for c in self.sequence]

        return {
            'type': self.__class__.__name__,
            'sequence': sequence_json,
            'content_type': self.content_type,
            'url': self.url,
            'original_sequence': self.original_sequence,
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Content':
        content_type = data.get('content_type', '')
        sequence = data['sequence']
        url = data.get('url', '')

        if content_type == 'text':
            return TextContent(sequence, url)
        elif content_type == 'image':
            image_type = cast(str, data.get('image_type', 'image/png'))
            return ImageContent(
                base64.b64decode(sequence),
                url,
                image_type,
            )
        elif content_type == 'pdf':
            return PdfContent(
                base64.b64decode(sequence),
                url
            )
        elif content_type == 'file':
            return FileContent(
                base64.b64decode(sequence),
                url,
            )
        # container types
        elif content_type == 'browser':
            # Handle BrowserContent's list of Content
            sequence_contents = [Content.from_json(content_data) for content_data in sequence]
            return BrowserContent(sequence_contents, url)
        elif content_type == 'markdown':
            # Handle MarkdownContent's list of Content
            sequence_contents = [Content.from_json(content_data) for content_data in sequence]
            return MarkdownContent(sequence_contents, url)
        else:
            raise ValueError(f"Unknown content type: {content_type}")


class SupportedMessageContent(Content):
    pass

class BinaryContent(Content):
    def __init__(
        self,
        sequence: bytes,
        content_type: str = '',
        url: str = '',
    ):
        if not isinstance(sequence, bytes):
            raise ValueError('sequence must be a bytes object')

        super().__init__(sequence, content_type, url)

    @abstractmethod
    def get_str(self) -> str:
        pass

    @abstractmethod
    def get_bytes(self) -> bytes:
        pass


class TextContent(SupportedMessageContent):
    def __init__(
        self,
        sequence: str,
        url: str = '',
    ):
        if not isinstance(sequence, str):
            raise ValueError('sequence must be a string')

        super().__init__(sequence, 'text', url)
        self.sequence = sequence

    def __str__(self):
        return self.get_str()

    def __repr__(self):
        return f'TextContent({self.sequence})'

    def get_str(self) -> str:
        return self.sequence


class ImageContent(BinaryContent, SupportedMessageContent):
    def __init__(
        self,
        sequence: bytes,
        url: str = '',
        image_type: str = '',
    ):
        super().__init__(sequence, 'image', url)
        self.sequence = sequence
        self.image_type = image_type

    def __str__(self):
        representation = self.url if self.url else f'{len(self.sequence)} bytes'
        return f'ImageContent({representation})'

    def __repr__(self):
        representation = self.url if self.url else f'{len(self.sequence)} bytes'
        return f'ImageContent({representation})'

    def get_str(self) -> str:
        return self.__str__()

    def get_bytes(self) -> bytes:
        return self.sequence


class PdfContent(BinaryContent):
    def __init__(
        self,
        sequence: bytes,
        url: str = '',
    ):

        super().__init__(sequence, 'pdf', url)
        self.sequence = sequence

    def __str__(self):
        return f'PdfContent({self.url})'

    def is_local(self):
        return os.path.isfile(self.url)

    def get_str(self) -> str:
        logging.debug('PdfContent.get_str() called, [PdfContent] string returned')
        return self.__str__()

    def get_bytes(self) -> bytes:
        return self.sequence


class FileContent(BinaryContent):
    def __init__(
        self,
        sequence: bytes,
        url: str = '',
    ):
        super().__init__(sequence, 'file', url)
        self.sequence = sequence

    def __str__(self):
        return f'FileContent({self.url.__str__()} is_local: {self.is_local()})'

    def __repr__(self):
        return f'FileContent({self.url.__str__()} is_local: {self.is_local()})'

    def is_local(self):
        return os.path.isfile(self.url)

    def get_str(self) -> str:
        if self.is_local():
            with open(self.url, 'r') as f:
                # return f.read()
                return f"<file url={self.url}>\n{f.read()}\n</file>"
        elif isinstance(self.sequence, bytes):
            # convert the bytes to a string and return
            return self.sequence.decode('utf-8')
        else:
            raise ValueError('FileContent.get_str() called on non-local file')

    def get_bytes(self) -> bytes:
        return super().get_bytes()


class ContainerContent(Content):
    def __init__(
        self,
        sequence: list[Content],
        content_type: str,
        url: str = '',
    ):
        if not isinstance(sequence, list):
            raise ValueError('sequence must be a list of Content objects')

        super().__init__(sequence, content_type, url)

    def to_json(self) -> dict:
        return {
            "content_type": self.content_type,
            "url": self.url,
            "sequence": [cast(Content, content).to_json() for content in self.sequence]
        }

class BrowserContent(ContainerContent):
    def __init__(
        self,
        sequence: list[Content],
        url: str = '',
    ):
        # browser sequence usually ImageContent, MarkdownContent
        super().__init__(sequence, 'browser', url)
        self.sequence = sequence

    def __str__(self):
        return f'BrowserContent({self.url}) {self.sequence}'

    def __repr__(self):
        return f'BrowserContent({self.url})'

    def get_str(self):
        return '\n'.join([c.get_str() for c in self.sequence])


class MarkdownContent(ContainerContent):
    def __init__(
        self,
        sequence: list[Content],
        url: str = '',
    ):
        if len(sequence) > 2:
            raise ValueError('MarkdownContent sequence must be a list of length 2')
        super().__init__(sequence, 'markdown', url)
        self.sequence = sequence

    def __str__(self):
        return f'MarkdownContent({self.url})'

    def __repr__(self):
        return f'MarkdownContent({self.url.__str__()} sequence: {self.sequence})'

    def get_str(self) -> str:
        return '\n'.join([c.get_str() for c in self.sequence])

class Message(AstNode):
    def __init__(
        self,
        message: list[Content],
        hidden: bool = False,
    ):
        if not isinstance(message, list):
            raise ValueError('message must be a list of Content objects')

        self.message: list[Content] = message
        self.pinned: int = 0  # 0 is not pinned, -1 is pinned last, anything else is pinned
        self.prompt_cached: bool = False
        self.hidden: bool = hidden

    @abstractmethod
    def role(self) -> str:
        pass

    @abstractmethod
    def get_str(self) -> str:
        pass

    def to_json(self) -> dict:
        return {
            "role": self.role(),
            "message": [cast(Content, content).to_json() for content in self.message],
            "pinned": self.pinned,
            "prompt_cached": self.prompt_cached,
            "hidden": self.hidden,
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Message':
        role = data.get('role')
        messages = [Content.from_json(content_data) for content_data in data.get('message', [])]
        prompt_cached = data.get('prompt_cached', False)
        hidden = data.get('hidden', False)
        pinned = data.get('pinned', 0)
        if role == 'user':
            user = User(messages, hidden)
            user.prompt_cached = prompt_cached
            user.pinned = pinned
            return user
        elif role == 'system':
            system = System(messages[0].get_str())
            system.prompt_cached = prompt_cached
            system.pinned = pinned
            return system
        elif role == 'assistant':
            assistant = Assistant(messages[0], hidden)
            assistant.prompt_cached = prompt_cached
            assistant.pinned = pinned
            return assistant
        else:
            raise ValueError(f'Role type not supported {role}, from {data}')

class User(Message):
    def __init__(
        self,
        message: Content | list[Content],
        hidden: bool = False,
    ):
        if not isinstance(message, list):
            message = [message]

        # check to see if all elements are Content
        if not all(isinstance(m, Content) for m in message):
            raise ValueError('User message must be a Content object or list of Content objects')

        super().__init__(message, hidden)

    def role(self) -> str:
        return 'user'

    def __str__(self):
        return self.get_str()

    def get_str(self):
        def content_str(content) -> str:
            if isinstance(content, Content):
                return content.get_str()
            elif isinstance(content, list):
                return '\n'.join([content_str(c) for c in content])
            elif isinstance(content, AstNode):
                return str(content)
            else:
                raise ValueError(f'Unsupported content type for User.get_str(): {type(content)}')

        if isinstance(self.message, Content):
            return self.message.get_str()

        return '\n'.join([content_str(c) for c in self.message])

    def __repr__(self):
        return f'User({self.message.__repr__()})'

    def __add__(self, other):
        a, b = coerce_types(str(self), other)
        return a + b  # type: ignore

    def __radd__(self, other):
        a, b = coerce_types(other, str(self))
        return a + b  # type: ignore


class System(Message):
    def __init__(
        self,
        message: str = '''
            You are a helpful assistant.
            Dont make assumptions about what values to plug into functions.
            Ask for clarification if a user request is ambiguous.
        '''
    ):
        if not isinstance(message, str):
            raise ValueError('System message must be a string')

        super().__init__([TextContent(message)])

    def role(self) -> str:
        return 'system'

    def __str__(self):
        return self.get_str()

    def __repr__(self):
        return f'System({self.message.__repr__()})'

    def get_str(self) -> str:
        return self.message[0].get_str()


class Assistant(Message):
    def __init__(
        self,
        message: Content | list[Content],
        error: bool = False,
        system_context: object = None,
        llm_call_context: object = None,
        stop_reason: str = '',
        stop_token: str = '',
        perf_trace: object = None,
        hidden: bool = False,
        total_tokens: int = 0,
    ):
        if isinstance(message, list):
            super().__init__(message, hidden)
        else:
            super().__init__([message], hidden)
        self.error = error
        self._system_context = system_context,
        self._llm_call_context: object = llm_call_context
        self.stop_reason: str = stop_reason
        self.stop_token: str = stop_token
        self.perf_trace: object = perf_trace
        self.total_tokens: int = total_tokens

    def role(self) -> str:
        return 'assistant'

    def __str__(self):
        return self.get_str()

    def get_str(self):
        return ' '.join([str(m.get_str()) for m in self.message])

    def __add__(self, other):
        def str_str(x):
            if hasattr(x, 'get_str'):
                return x.get_str()
            return str(x)

        assistant = Assistant(
            message=TextContent(str_str(self.message) + str_str(other)),
            system_context=self._system_context,
            llm_call_context=self._llm_call_context,
            stop_reason=self.stop_reason,
            stop_token=self.stop_token,
        )
        return assistant

    def __repr__(self):
        if self.error:
            return f'Assistant({self.message[0].__repr__()} {self.error})'
        else:
            return f'Assistant({self.message[0].__repr__()})'

    def to_json(self):
        json_result = super().to_json()
        json_result['error'] = self.error
        json_result['system_context'] = self._system_context
        json_result['stop_reason'] = self.stop_reason
        json_result['stop_token'] = self.stop_token
        json_result['total_tokens'] = self.total_tokens
        return json_result

    @classmethod
    def from_json(cls, data: dict) -> 'Assistant':
        assistant = cast(Assistant, super().from_json(data))
        assistant.error = data.get('error')
        assistant._system_context = data.get('system_context')
        assistant.stop_reason = cast(str, data.get('stop_reason'))
        assistant.stop_token = cast(str, data.get('stop_token'))
        assistant.total_tokens = cast(int, data.get('total_tokens'))
        return assistant


########################################################################################
## Interface classes
########################################################################################
class Executor(ABC):
    def __init__(
        self,
        default_model: str,
        api_endpoint: str,
        default_max_input_len: int,
        default_max_output_len: int,
    ):
        self._default_model = default_model
        self.api_endpoint = api_endpoint
        self.default_max_input_len = default_max_input_len
        self.default_max_output_len = default_max_output_len

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def aexecute(
        self,
        messages: list['Message'],
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        stop_tokens: list[str] = [],
        model: Optional[str] = None,
        thinking: int = 0,
        stream_handler: Optional[Callable[['AstNode'], Awaitable[None]]] = None,
    ) -> 'Assistant':
        pass

    @property
    def default_model(
        self,
    ) -> str:
        return self._default_model

    @default_model.setter
    def default_model(
        self,
        default_model: str,
    ) -> None:
        self._default_model = default_model

    def max_input_tokens(
        self,
        model: Optional[str] = None,
    ) -> int:
        if model: return TokenPriceCalculator().max_input_tokens(model=model, default=self.default_max_input_len, executor=self.name())
        else: return self.default_max_input_len

    def max_output_tokens(
        self,
        model: Optional[str] = None,
    ) -> int:
        if model: return TokenPriceCalculator().max_output_tokens(model=model, default=self.default_max_output_len, executor=self.name())
        else: return self.default_max_output_len

    @abstractmethod
    def execute(
        self,
        messages: list['Message'],
        max_output_tokens: int = 2048,
        temperature: float = 1.0,
        stop_tokens: list[str] = [],
        model: Optional[str] = None,
        stream_handler: Optional[Callable[['AstNode'], None]] = None,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def to_dict(self, message: 'Message') -> dict:
        pass

    @abstractmethod
    def from_dict(self, message: dict) -> 'Message':
        pass

    @abstractmethod
    async def count_tokens(
        self,
        messages: list['Message'],
    ) -> int:
        pass

    @abstractmethod
    async def count_tokens_dict(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        pass

    @abstractmethod
    def user_token(
        self
    ) -> str:
        pass

    @abstractmethod
    def assistant_token(
        self
    ) -> str:
        pass

    @abstractmethod
    def append_token(
        self
    ) -> str:
        pass

    @abstractmethod
    def scratchpad_token(
        self
    ) -> str:
        pass

    @abstractmethod
    def unpack_and_wrap_messages(self, messages: list[Message], model: Optional[str] = None) -> list[dict[str, str]]:
        pass

def coerce_types(a, b):
    # Function to check if a string can be converted to an integer or a float
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_float(x):
        return np.isscalar(x) and isinstance(x, (float, np.floating))

    def is_aware(dt):
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

    if isinstance(a, FunctionCallMeta):
        a = a.result()

    if isinstance(b, FunctionCallMeta):
        b = b.result()

    if isinstance(a, Assistant):
        a = a.get_str()

    if isinstance(b, Assistant):
        b = b.get_str()

    # If either operand is a string and represents a number, convert it
    if isinstance(a, str) and is_number(a):
        a = int(a) if '.' not in a else float(a)
    if isinstance(b, str) and is_number(b):
        b = int(b) if '.' not in b else float(b)

    if isinstance(a, dt.date):
        a = dt.datetime(a.year, a.month, a.day)

    if isinstance(b, dt.date):
        b = dt.datetime(b.year, b.month, b.day)

    if isinstance(a, dt.datetime) and isinstance(b, dt.datetime):
        if is_aware(a) and is_aware(b):
            return a, b
        elif not is_aware(a) and not is_aware(b):
            return a, b
        else:
            a = a.replace(tzinfo=None)
            b = b.replace(tzinfo=None)

    # If either operand is a string now, convert both to strings
    if isinstance(a, str) or isinstance(b, str):
        return str(a), str(b)

    # If they are of the same type, return them as-is
    if type(a) is type(b):
        return a, b

    # numpy and python floats
    if is_float(a) and is_float(b):
        return float(a), float(b)  # type: ignore

    # If one is a float and the other an int, convert the int to float
    if isinstance(a, float) and isinstance(b, int):
        return a, float(b)
    if isinstance(b, float) and isinstance(a, int):
        return float(a), b

    if isinstance(a, dt.datetime) and isinstance(b, dt.timedelta):
        return a, b
    if isinstance(b, dt.datetime) and isinstance(a, dt.timedelta):
        return a, b

    raise TypeError(f"Cannot coerce types {type(a)} and {type(b)} to a common type")

def coerce_to(a: Any, type_var: Type[T]) -> Any:
    # Helper functions
    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    if isinstance(a, type_var):
        return a

    if isinstance(a, FunctionCallMeta):
        a = str(a.result())
    if isinstance(a, User):
        a = a.get_str()
    if isinstance(a, Assistant):
        a = a.get_str()

    if isinstance(a, str):
        if type_var == bool:
            return a.lower() in ('true', 'yes', '1', 'on')
        elif type_var in (int, float) and is_number(a):
            return type_var(a)
        elif type_var == dt.datetime:
            try:
                return dt.datetime.fromisoformat(a)
            except ValueError:
                pass  # If it fails, we'll raise TypeError at the end

    if isinstance(a, dt.date) and type_var == dt.datetime:
        return dt.datetime(a.year, a.month, a.day)

    if isinstance(a, dt.datetime) and type_var == dt.date:
        return dt.datetime(a.year, a.month, a.day)

    if type_var == str:
        if isinstance(a, dt.datetime):
            return a.isoformat()
        elif isinstance(a, dt.date):
            return a.isoformat()
        elif isinstance(a, list):
            return ' '.join([str(n) for n in a])
        elif isinstance(a, dict):
            return ' '.join([f'{k}: {v}' for k, v in a.items()])
        return str(a)

    if type_var in (int, float, np.floating, np.number):
        if isinstance(a, (int, float, np.number, np.floating)):
            return type_var(a)

    if type_var == bool:
        if isinstance(a, (int, float, np.number, bool)):
            return bool(a)

    raise TypeError(f"Cannot coerce type {type(a)} with value {str(a)[0:50]} to {type_var}")

class TokenCompressionMethod(Enum):
    AUTO = 0
    LIFO = 1
    SIMILARITY = 2
    MAP_REDUCE = 3
    SUMMARY = 4


def compression_enum(input_str):
    normalized_str = input_str.upper().replace('MAPREDUCE', 'MAP_REDUCE')
    try:
        return TokenCompressionMethod[normalized_str]
    except KeyError:
        raise ValueError(f"Unknown compression method: {input_str}")


class LLMCall():
    def __init__(
        self,
        user_message: 'Message',
        context_messages: list['Message'],
        executor: Executor,
        model: str,
        temperature: float,
        max_prompt_len: int,
        completion_tokens_len: int,
        prompt_name: str,
        stop_tokens: list[str] = [],
        thinking: int = 0,
        stream_handler: Callable[['AstNode'], Awaitable[None]] = awaitable_none
    ):
        self.user_message = user_message
        self.context_messages = context_messages
        self.executor = executor
        self.model = model
        self.temperature = temperature
        self.max_prompt_len = max_prompt_len
        self.completion_tokens_len = completion_tokens_len
        self.prompt_name = prompt_name
        self.stop_tokens = stop_tokens
        self.thinking = thinking
        self.stream_handler = stream_handler

    def copy(self):
        return LLMCall(
            user_message=copy.deepcopy(self.user_message),
            context_messages=copy.deepcopy(self.context_messages),
            executor=self.executor,
            model=self.model,
            temperature=self.temperature,
            max_prompt_len=self.max_prompt_len,
            completion_tokens_len=self.completion_tokens_len,
            prompt_name=self.prompt_name,
            stop_tokens=self.stop_tokens,
            thinking=self.thinking,
            stream_handler=self.stream_handler,
        )


class Controller():
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def aexecute_llm_call(
        self,
        llm_call: LLMCall,
        query: str,
        original_query: str,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def execute_llm_call(
        self,
        llm_call: LLMCall,
        query: str,
        original_query: str,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    ) -> 'Assistant':
        pass

    @abstractmethod
    def get_executor() -> Executor:
        pass


class TokenNode(AstNode):
    def __init__(
        self,
        token: str,
    ):
        super().__init__()
        self.token = token

    def __str__(self):
        return self.token

    def __repr__(self):
        return f'TokenNode({self.token})'


class TokenThinkingNode(TokenNode):
    def __init__(
        self,
        token: str,
    ):
        super().__init__(token)

    def __repr__(self):
        return f'TokenThinkingNode({self.token})'


class TokenStopNode(AstNode):
    def __init__(
        self,
        print_str: str = '',
    ):
        super().__init__()
        self.print_str = print_str

    def __str__(self):
        return self.print_str

    def __repr__(self):
        return f'TokenStopNode(print_str={self.print_str!r})'


class StreamingStopNode(AstNode):
    def __init__(
        self,
        print_str: str = '\n',
    ):
        super().__init__()
        self.print_str = print_str

    def __str__(self):
        return self.print_str

    def __repr__(self):
        return f'StreamingStopNode(print_str={self.print_str!r})'


class QueueBreakNode(AstNode):
    def __init__(
        self,
    ):
        super().__init__()

    def __str__(self):
        return '\n'

    def __repr__(self):
        return 'QueueBreakNode()'



class StreamNode(AstNode):
    def __init__(
        self,
        obj: object,
        type: str,
        metadata: object = None,
    ):
        super().__init__()
        self.obj = obj
        self.type = type
        self.metadata = metadata

    def __str__(self):
        return f'StreamNode{str(self.obj)}'

    def __repr__(self):
        return 'StreamNode()'


class DebugNode(AstNode):
    def __init__(
        self,
        debug_str: str,
    ):
        super().__init__()
        self.debug_str = debug_str

    def __str__(self):
        return f'DebugNode({self.debug_str})'

    def __repr__(self):
        return 'DebugNode()'


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

    def token(self):
        return 'statement'


class DataFrame(Statement):
    def __init__(
        self,
        elements: list,
    ):
        super().__init__()
        self.elements = elements

    def token(self):
        return 'dataframe'


class Call(Statement):
    def __init__(
        self,
    ):
        super().__init__()


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

    def __enter__(self):
        return self._result.__enter__()  # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._result.__exit__(exc_type, exc_val, exc_tb)  # type: ignore

    def __float__(self):
        return float(self._result)  # type: ignore

    def __getattr__(self, name):
        if self._result is not None:
            return getattr(self._result, name)

        raise AttributeError(f"'self._result isn't set, and {self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        if self._result is not None and hasattr(self._result, '__getitem__'):
            return self._result[key]  # type: ignore

        raise AttributeError(f"{type(self._result)} is not subscriptable")

    def __setstate__(self, state):
        # Directly set _data without going through __getattr__
        self._result = state.get('_result')

    def __getstate__(self):
        # Return a dictionary representing the object's state
        return {'_result': self._result}

    def __str__(self):
        return str(self._result)

    def __repr__(self):
        return self._result.__repr__()

    def get_str(self):
        if hasattr(self._result, 'get_str'):
            return self._result.get_str()  # type: ignore
        else:
            return str(self._result)

    def __add__(self, other):
        a, b = coerce_types(self._result, other)
        return a + b  # type: ignore

    def __sub__(self, other):
        a, b = coerce_types(self._result, other)
        return a - b  # type: ignore

    def __mul__(self, other):
        a, b = coerce_types(self._result, other)
        return a * b  # type: ignore

    def __div__(self, other):
        a, b = coerce_types(self._result, other)
        return a / b  # type: ignore

    def __truediv__(self, other):
        a, b = coerce_types(self._result, other)
        return a / b  # type: ignore

    def __rtruediv__(self, other):
        a, b = coerce_types(other, self._result)
        return a / b  # type: ignore

    def __radd__(self, other):
        a, b = coerce_types(other, self._result)
        return a + b  # type: ignore

    def __rsub__(self, other):
        a, b = coerce_types(other, self._result)
        return a - b  # type: ignore

    def __rmul__(self, other):
        a, b = coerce_types(other, self._result)
        return a * b  # type: ignore

    def __rdiv__(self, other):
        a, b = coerce_types(other, self._result)
        return a / b  # type: ignore

    def __gt__(self, other):
        a, b = coerce_types(self._result, other)
        return a > b  # type: ignore

    def __lt__(self, other):
        a, b = coerce_types(self._result, other)
        return a < b  # type: ignore

    def __ge__(self, other):
        a, b = coerce_types(self._result, other)
        return a >= b  # type: ignore

    def __le__(self, other):
        a, b = coerce_types(self._result, other)
        return a <= b  # type: ignore

    def __rgt__(self, other):
        # Note the order in coerce_types is reversed
        a, b = coerce_types(other, self._result)
        return a > b  # type: ignore

    def __rlt__(self, other):
        a, b = coerce_types(other, self._result)
        return a < b  # type: ignore

    def __rge__(self, other):
        a, b = coerce_types(other, self._result)
        return a >= b  # type: ignore

    def __rle__(self, other):
        a, b = coerce_types(other, self._result)
        return a <= b  # type: ignore

    def __format__(self, format_spec):
        return format(self._result, format_spec)


class PandasMeta(Call):
    def __init__(
        self,
        expr_str: str,
        pandas_df,
    ):
        self.expr_str = expr_str
        self.df: pd.DataFrame = pandas_df

    def result(self) -> object:
        return self._result

    def token(self):
        return 'pandasmeta'

    def __str__(self):
        str_acc = ''
        if self.df is not None:
            str_acc += f'info()\n'
            str_acc += f'{self.df.info()}\n\n'  # type: ignore
            str_acc += f'describe()\n'
            str_acc += f'{self.df.describe()}\n\n'  # type: ignore
            str_acc += f'head()\n'
            str_acc += f'{self.df.head()}\n\n'  # type: ignore
            str_acc += '\n'
            str_acc += f'call "to_string()" to get the entire DataFrame as a string\n'
            return str_acc
        else:
            return '[]'

    def __gt__(self, other):
        return self.df > other

    def __lt__(self, other):
        return self.df < other

    def __ge__(self, other):
        return self.df >= other

    def __le__(self, other):
        return self.df <= other

    def __rgt__(self, other):
        return self.df > other

    def __rlt__(self, other):
        return self.df < other

    def __rge__(self, other):
        return self.df >= other

    def __rle__(self, other):
        return self.df <= other

    def __add__(self, other):
        return self.df + other

    def __len__(self):
        return len(self.df)

    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self.df, name)
        elif object.__getattribute__(self, 'df') is not None:
            return getattr(self.df, name)
        raise AttributeError(f"'self.df isn't set, and {self.__class__.__name__}' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self.df.__getitem__(key)  # type: ignore

    def __setitem__(self, key, value):
        self.df.__setitem__(key, value)

    def __iter__(self):
        return iter(self.df)

    def __format__(self, format_spec):
        return format(self.pandas_df, format_spec)

    def to_string(self):
        return self.df.to_string()


class FunctionCall(Call):
    def __init__(
        self,
        name: str,
        args: list[dict[str, object]],
        types: list[dict[str, object]],
        context: 'Content' = TextContent(''),
        func: Optional[Callable] = None,
    ):
        super().__init__()
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
        result: object = None,
        error: object = None,
    ):
        super().__init__()
        self._result = result
        self.error = error

    def __str__(self):
        if not self.error:
            return str(self.result())
        else:
            return str(self.error)

    def get_str(self):
        return str(self)

    def token(self):
        return 'answer'


########################################################################################
## Pydantic classes
########################################################################################
class DownloadItemModel(BaseModel):
    id: int
    url: str


class ContentModel(BaseModel):
    sequence: Union[list[dict], str, bytes]
    content_type: str
    original_sequence: Optional[Union[list[dict], str, bytes]] = None
    url: str

    class Config:
        from_attributes = True

    def to_content(self) -> Content:
        return Content.from_json(data=self.model_dump())

    @classmethod
    def from_content(cls, content: Content) -> 'ContentModel':
        return cls.model_validate(content.to_json())


class MessageModel(BaseModel):
    role: str
    content: list[ContentModel]
    pinned: int = 0
    prompt_cached: bool = False
    total_tokens: int = 0  # only used on Assistant messages

    def to_message(self) -> Message:
        content_objects = [c.to_content() for c in self.content]

        if self.role == 'user':
            msg = User(content_objects)
        elif self.role == 'system':
            msg = System(content_objects[0].get_str())
        elif self.role == 'assistant':
            msg = Assistant(content_objects[0], total_tokens=self.total_tokens)
        else:
            raise ValueError(f"MessageModel.to_message() Unsupported role: {self.role}")

        msg.pinned = self.pinned
        msg.prompt_cached = self.prompt_cached
        return msg

    @classmethod
    def from_message(cls, message: Message) -> 'MessageModel':
        content_models = [ContentModel.from_content(content) for content in message.message]

        return cls(
            role=message.role(),
            content=content_models,
            pinned=message.pinned,
            prompt_cached=message.prompt_cached,
            total_tokens=message.total_tokens if isinstance(message, Assistant) else 0,
        )


class SessionThreadModel(BaseModel):
    id: int = -1
    executor: str = ''
    model: str = ''
    compression: str = ''
    temperature: float = 0.0
    stop_tokens: list[str] = []
    output_token_len: int = 0
    current_mode: str = 'tools'
    thinking: int = 0
    cookies: list[dict[str, Any]] = []
    messages: list[MessageModel] = []
    locals_dict: dict[str, Any] = Field(default_factory=dict, exclude=True)