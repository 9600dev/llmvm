from enum import Enum
import httpx
import asyncio
import nest_asyncio

from pydantic import TypeAdapter

from llmvm.common.objects import Message, User, Content, Assistant, Executor, AstNode, SessionThread, MessageModel
from llmvm.common.container import Container
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.gemini_executor import GeminiExecutor
from llmvm.common.logging_helpers import setup_logging
from llmvm.client.parsing import parse_message_thread, parse_message_actions
from llmvm.client.printing import StreamPrinter, stream_response
from typing import Awaitable, Callable, Optional, List, Dict, Any, Sequence, Union, cast


logging = setup_logging()
nest_asyncio.apply()


class Mode(Enum):
    DIRECT = 'direct'
    TOOL = 'tool'


_printer = StreamPrinter('')
async def default_stream_handler(node: AstNode):
    await _printer.write(node)  # type: ignore


def llm(
    messages: list[Message] | list[str] | str,
    executor: Optional[Executor] = None,
    model: Optional[str] = None,
    output_token_len: int = 4096,
    temperature: float = 0.0,
    stop_tokens: List[str] = [],
    stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = default_stream_handler,
    template_args: Optional[Dict[str, Any]] = None,
) -> Assistant:
    if (
        not isinstance(messages, list)
        and not isinstance(messages, str)
    ):
        raise ValueError('messages must be a list of Message objects or a string')

    if isinstance(messages, str):
        messages = cast(list[Message], [User(Content(messages))])
    elif isinstance(messages, list) and all(isinstance(message, str) for message in messages):
        messages = cast(list[Message], [User(Content(message)) for message in messages])

    return asyncio.run(
        LLMVMClient(
            api_endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://localhost:8011'),
            default_executor_name=Container.get_config_variable('LLMVM_EXECUTOR'),
            default_model_name=Container.get_config_variable('LLMVM_MODEL'),
            api_key='',
        ).call_direct(
            messages=cast(list[Message], messages),
            executor=executor,
            model=model,
            output_token_len=output_token_len,
            temperature=temperature,
            stop_tokens=stop_tokens,
            stream_handler=stream_handler,
            template_args=template_args,
        )
    )


def llmvm(
    messages: list[Message] | list[str] | str,
    executor_name: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    output_token_len: int = 4096,
    stop_tokens: List[str] = [],
    cookies: List[Dict[str, Any]] = [],
    compression: str = 'auto',
    mode: str = 'auto',
    stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = default_stream_handler,
    template_args: Optional[Dict[str, Any]] = None,
) -> SessionThread:
    if (
        not isinstance(messages, list)
        and not isinstance(messages, str)
    ):
        raise ValueError('messages must be a list of Message objects or a string')

    if isinstance(messages, str):
        messages = cast(list[Message], [User(Content(messages))])
    elif isinstance(messages, list) and all(isinstance(message, str) for message in messages):
        messages = cast(list[Message], [User(Content(message)) for message in messages])

    return asyncio.run(
        LLMVMClient(
            api_endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://localhost:8011'),
            default_executor_name=executor_name or Container.get_config_variable('LLMVM_EXECUTOR'),
            default_model_name=model_name or Container.get_config_variable('LLMVM_MODEL'),
            api_key='',
        ).call(
            thread=-1,
            messages=cast(list[Message], messages),
            executor_name=executor_name,
            model_name=model_name,
            temperature=temperature,
            output_token_len=output_token_len,
            stop_tokens=stop_tokens,
            cookies = cookies,
            compression=compression,
            mode=mode,
            stream_handler=stream_handler,
            template_args=template_args,
        )
    )


class LLMVMClient():
    def __init__(
        self,
        api_endpoint: str,
        default_executor_name: str,
        default_model_name: str,
        api_key: str,
        throw_if_server_down: bool = False,
        default_stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = default_stream_handler
    ):
        self.api_endpoint = api_endpoint
        self.throw_if_server_down = throw_if_server_down
        self.default_stream_handler = default_stream_handler
        self.executor = default_executor_name
        self.model = default_model_name
        self.__set_defaults(default_executor_name, default_model_name, api_key)
        self.role_strings = ['Assistant: ', 'System: ', 'User: ']
        self.action_strings = ['[ImageContent(', '[PdfContent(', '[FileContent(']

    def __set_defaults(self, executor: str, model: str, api_key: str):
        executor_instance = self.get_executor(executor, model, api_key)
        self.default_executor = executor_instance
        self.model = executor_instance.default_model

    def __parse_messages(self, messages: list[Message]) -> list[Message]:
        thread_messages = messages
        thread_messages_copy = []
        for message in thread_messages:
            if (
                type(message.message) is Content
                and any(role_string in message.message.get_str() for role_string in self.role_strings)
            ):
                parsed_messages = parse_message_thread(message.message.get_str(), self.action_strings)
                thread_messages_copy += parsed_messages
            # if the incoming message has actions [ImageContent(...)], [PdfContent(...)], [FileContent(...)] etc
            elif (
                type(message.message) is Content
                and any(action_string in message.message.get_str() for action_string in self.action_strings)
            ):
                parsed_messages = parse_message_actions(User, message.message.get_str(), self.action_strings)
                thread_messages_copy += parsed_messages
            else:
                thread_messages_copy.append(message)
        return thread_messages_copy

    def get_executor(self, executor_name: str, model_name: Optional[str], api_key: Optional[str]) -> Executor:
        if executor_name == 'anthropic':
            if Container.get_config_variable('ANTHROPIC_API_KEY') or api_key:
                return AnthropicExecutor(
                    api_key=api_key or Container.get_config_variable('ANTHROPIC_API_KEY'),
                    default_model=cast(str, model_name) if model_name else 'claude-3-5-sonnet-20240620'
                )
            else:
                raise ValueError('anthropic executor requested, but unable to find Anthropic API key.')

        elif executor_name == 'openai':
            if Container.get_config_variable('OPENAI_API_KEY') or api_key:
                return OpenAIExecutor(
                    api_key=api_key or Container.get_config_variable('OPENAI_API_KEY'),
                    default_model=cast(str, model_name) if model_name else 'gpt-4o-2024-08-06'
                )
            else:
                raise ValueError('openai executor requested, but unable to find OpenAI API key.')

        elif executor_name == 'gemini':
            if Container.get_config_variable('GEMINI_API_KEY') or api_key:
                return GeminiExecutor(
                    api_key=api_key or Container.get_config_variable('GEMINI_API_KEY'),
                    default_model=cast(str, model_name) if model_name else 'gemini-pro'
                )
            else:
                raise ValueError('GeminiExecutor requires a model name and API key.')

        else:
            if Container.get_config_variable('ANTHROPIC_API_KEY'):
                return AnthropicExecutor(
                    api_key=Container.get_config_variable('ANTHROPIC_API_KEY'),
                    default_model='claude-3-5-sonnet-20240620'
                )
            elif Container.get_config_variable('OPENAI_API_KEY'):
                return OpenAIExecutor(
                    api_key=Container.get_config_variable('OPENAI_API_KEY'),
                    default_model='gpt-4o-2024-08-06'
                )
            elif Container.get_config_variable('GEMINI_API_KEY'):
                return GeminiExecutor(
                    api_key=Container.get_config_variable('GEMINI_API_KEY'),
                    default_model='gemini-pro'
                )
            raise ValueError('No API key is set for any executor in ENV. Unable to set default executor.')

    async def call_direct(
        self,
        messages: list[Message],
        executor: Optional[Executor] = None,
        model: Optional[str] = None,
        output_token_len: int = 4096,
        temperature: float = 0.0,
        stop_tokens: List[str] = [],
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = None,
        template_args: Optional[Dict[str, Any]] = None,
    ) -> Assistant:
        async def null_handler(node: AstNode):
            pass

        if not stream_handler:
            stream_handler = null_handler

        if not executor:
            executor = self.default_executor

        if not model:
            model = self.model

        thread_messages: List[Message] = self.__parse_messages(messages)

        assistant = await executor.aexecute(
            messages=thread_messages,
            max_output_tokens=output_token_len,
            temperature=temperature,
            stop_tokens=stop_tokens,
            model=model,
            stream_handler=stream_handler,
            template_args=template_args,
        )
        return assistant

    async def get_thread(
        self,
        id: int,
    ) -> SessionThread:
        params = {
            'id': id,
        }
        response: httpx.Response = httpx.get(f'{self.api_endpoint}/v1/chat/get_thread', params=params)
        thread = SessionThread.model_validate(response.json())
        return thread

    async def set_thread(
        self,
        thread: SessionThread,
    ) -> SessionThread:
        async with httpx.AsyncClient(timeout=400.0) as client:
            response = await client.post(
                f'{self.api_endpoint}/v1/chat/set_thread',
                json=thread.model_dump()
            )
            session_thread = SessionThread.model_validate(response.json())
            return session_thread

    async def get_threads(
        self,
    ) -> List[SessionThread]:
        response: httpx.Response = httpx.get(f'{self.api_endpoint}/v1/chat/get_threads')
        threads = cast(List[SessionThread], TypeAdapter(List[SessionThread]).validate_python(response.json()))

        return threads

    async def call_with_session(
        self,
        session_thread: SessionThread,
    ) -> SessionThread:
        return await self.call(thread=session_thread)

    async def status(self) -> Dict[str, str]:
        async with httpx.AsyncClient(timeout=2.0) as client:
            try:
                response = await client.get(f'{self.api_endpoint}/health')
                return response.json()
            except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
                return {'status': f'LLMVM server not available at {self.api_endpoint}. Set endpoint using $LLMVM_ENDPOINT.'}

    async def call(
        self,
        thread: int | SessionThread,
        messages: Union[List[Message], None] = None,
        executor_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        output_token_len: int = 4096,
        stop_tokens: List[str] = [],
        cookies: List[Dict[str, Any]] = [],
        compression: str = '',
        mode: str = '',
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = default_stream_handler,
        template_args: Optional[Dict[str, Any]] = None,
    ) -> SessionThread:
        if (
            (isinstance(messages, list) and len(messages) > 0 and not isinstance(messages[0], Message))
            and not isinstance(messages, type(None))
        ):
            raise ValueError('the messages argument must be a list of Message objects')

        # deal with weird message types and inputs
        thread_messages: List[Message] = []

        if isinstance(thread, SessionThread):
            thread_messages = [MessageModel.to_message(session_message) for session_message in thread.messages]
        elif isinstance(messages, list):
            thread_messages = messages

        thread_messages = self.__parse_messages(thread_messages)

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f'{self.api_endpoint}/health')
                response.raise_for_status()

            if isinstance(thread, int):
                thread = await self.get_thread(thread)
                # server thread has messages, so we append thread_messages to it
                if thread.messages:
                    server_thread_messages = [MessageModel.to_message(session_message) for session_message in thread.messages]
                    thread_messages = server_thread_messages + thread_messages

            if not executor_name: executor_name = self.default_executor.name()
            if not model_name: model_name = self.model

            if not thread.executor: thread.executor = executor_name
            if not thread.model: thread.model = model_name
            if not thread.output_token_len: thread.output_token_len = output_token_len
            if temperature:
                thread.temperature = temperature
            if cookies:
                thread.cookies = cookies
            if stop_tokens:
                thread.stop_tokens = stop_tokens
            if compression:
                thread.compression = compression
            if mode:
                thread.current_mode = mode

            # attach the messages to the thread
            thread.messages = [MessageModel.from_message(message) for message in thread_messages]

            if mode == 'direct' or mode == 'tool' or mode == 'auto':
                endpoint = '/tools/completions'

            async with httpx.AsyncClient(timeout=400.0) as client:
                async with client.stream(
                    'POST',
                    f'{self.api_endpoint}/v1{endpoint}',
                    json=thread.model_dump(),
                ) as response:
                    objs = await stream_response(response, StreamPrinter('').write)

            await response.aclose()

            if objs:
                session_thread = SessionThread.model_validate(objs[-1])
                return session_thread
            return thread

        except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
            if self.throw_if_server_down:
                logging.debug('LLMVM server is not running and throw_if_server_down is set to True.')
                raise ex

            if mode == 'tool' or mode == 'code':
                logging.debug('LLMVM server is down, but we are in tool mode. Cannot execute directly')
                raise ex

        executor = self.default_executor
        if executor_name:
            executor = self.get_executor(executor_name, model_name, None)

        # server is down, go direct. this means that executor and model can't be nothing
        assistant = await self.call_direct(
            messages=thread_messages,
            executor=executor,
            model=model_name,
            temperature=temperature,
            output_token_len=output_token_len,
            stop_tokens=stop_tokens,
            stream_handler=stream_handler,
            template_args=template_args,
        )
        return SessionThread(
            id=-1,
            messages=[MessageModel.from_message(message) for message in thread_messages + [assistant]],
            current_mode='direct',
            executor=executor.name(),
            model=model_name if model_name else self.model,
        )

    def search(
        self,
        query: str,
    ):
        response: httpx.Response = httpx.get(f'{self.api_endpoint}/search/{query}', timeout=400.0)
        return response.json()
