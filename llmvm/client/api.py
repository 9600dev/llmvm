from enum import Enum
import os
import httpx
import asyncio

from pydantic import TypeAdapter

from llmvm.common.objects import Message, User, Content, Assistant, Executor, AstNode, SessionThread, MessageModel
from llmvm.common.container import Container
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.gemini_executor import GeminiExecutor
from llmvm.common.logging_helpers import setup_logging
from llmvm.client.parsing import parse_message_thread, parse_message_actions
from llmvm.client.printing import StreamPrinter, stream_response
from typing import Optional, List, Dict, Any, Sequence, cast
from functools import singledispatch


logging = setup_logging()


class Mode(Enum):
    DIRECT = 'direct'
    TOOL = 'tool'


class LLMVMClient():
    def __init__(
        self,
        api_endpoint: str,
        default_executor: str,
        default_model: str,
        throw_if_server_down: bool = False
    ):
        self.api_endpoint = api_endpoint
        self.throw_if_server_down = throw_if_server_down
        self.default_executor = default_executor
        self.default_model = default_model

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

    async def call_session(
        self,
        session_thread: SessionThread,
    ) -> SessionThread:
        return await self.call(thread=session_thread)

    async def call(
        self,
        thread: int | SessionThread,
        messages: List[Message] = [],
        cookies: List[Dict[str, Any]] = [],
        compression: str = 'auto',
        executor: str = '',
        model: str = '',
        mode: str = 'auto',
    ) -> SessionThread:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f'{self.api_endpoint}/health')
                response.raise_for_status()

            if isinstance(thread, int):
                thread = await self.get_thread(thread)

            if not executor: executor = self.default_executor
            if not model: model = self.default_model

            if not thread.executor: thread.executor = executor
            if not thread.model: thread.model = model
            if not thread.cookies: thread.cookies = cookies
            if not thread.compression: thread.compression = compression
            if not thread.current_mode: thread.current_mode = mode

            thread_messages: List[Message] = [MessageModel.to_message(session_message) for session_message in thread.messages]

            for message in messages:
                if message not in thread_messages:
                    thread.messages.append(MessageModel.from_message(message))

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
            if mode == 'tool' or mode == 'code':
                logging.debug('LLMVM server is down, but we are in tool mode. Cannot execute directly')
                raise ex

        # server is down, go direct. this means that executor and model can't be nothing
        if not executor and not model:
            if Container.get_config_variable('LLMVM_EXECUTOR', 'executor'):
                executor = Container.get_config_variable('LLMVM_EXECUTOR')

            if Container.get_config_variable('LLMVM_MODEL', 'model'):
                model = Container.get_config_variable('LLMVM_MODEL')

        if executor and model:
            if executor == 'openai' and Container.get_config_variable('OPENAI_API_KEY'):
                assistant = await execute_llm_call_direct(
                    message,
                    Container.get_config_variable('OPENAI_API_KEY'),
                    'openai',
                    model,
                    context_messages
                )
                return SessionThread(
                    id=-1,
                    messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
                )
            elif executor == 'anthropic' and Container.get_config_variable('ANTHROPIC_API_KEY'):
                assistant = await execute_llm_call_direct(
                    message,
                    Container.get_config_variable('ANTHROPIC_API_KEY'),
                    'anthropic',
                    model,
                    context_messages
                )
                return SessionThread(
                    id=-1,
                    messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
                )
            elif executor == 'gemini' and Container.get_config_variable('GOOGLE_API_KEY'):
                assistant = await execute_llm_call_direct(
                    message,
                    Container.get_config_variable('GOOGLE_API_KEY'),
                    'gemini',
                    model,
                    context_messages
                )
                return SessionThread(
                    id=-1,
                    messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
                )
            else:
                raise ValueError(f'Executor {executor} and model {model} are set, but no API key is set.')
        elif Container.get_config_variable('OPENAI_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('OPENAI_API_KEY'),
                'openai',
                'gpt-4o',
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        elif os.environ.get('ANTHROPIC_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('ANTHROPIC_API_KEY'),
                'anthropic',
                'claude-2.1',
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        elif os.environ.get('GOOGLE_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('GOOGLE_API_KEY'),
                'gemini',
                'gemini-pro',
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        else:
            logging.warning('Neither OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY is set. Unable to execute direct call to LLM.')  # noqa
            raise ValueError('Neither OPENAI_API_KEY, ANTHROPIC_API_KEY or GOOGLE_API_KEY is set. Unable to execute direct call to LLM.')  # noqa










async def execute_llm_call_direct(
    message: Message,
    api_key: str,
    executor_name: str,
    model_name: str,
    context_messages: Sequence[Message] = [],
    api_endpoint: str = ''
) -> Assistant:
    printer = StreamPrinter('')

    async def __stream_handler(node: AstNode):
        printer.write(node)  # type: ignore

    executor: Optional[Executor] = None

    if executor_name == 'openai':
        executor = OpenAIExecutor(
            api_key=api_key,
            default_model=model_name,
            api_endpoint=api_endpoint or Container.get_config_variable('LLMVM_API_BASE', default='https://api.openai.com/v1')
        )
    elif executor_name == 'anthropic':
        executor = AnthropicExecutor(
            api_key=api_key,
            default_model=model_name,
            api_endpoint=api_endpoint or Container.get_config_variable('LLMVM_API_BASE', default='https://api.anthropic.com')
        )
    elif executor_name == 'gemini':
        executor = GeminiExecutor(
            api_key=api_key,
            default_model=model_name,
        )
    else:
        raise ValueError('No executor specified.')

    messages = list(context_messages) + [message]
    assistant = await executor.aexecute(
        messages=messages,
        stream_handler=__stream_handler,
    )
    return assistant


async def execute_llm_call(
    api_endpoint: str,
    id: int,
    message: Message,
    executor: str,
    model: str,
    mode: str,
    context_messages: Sequence[Message] = [],
    cookies: List[Dict[str, Any]] = [],
    compression: str = 'auto',
    clear_thread: bool = False,
) -> SessionThread:

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f'{api_endpoint}/health')
            response.raise_for_status()

        thread = await get_thread(api_endpoint, id)

        if clear_thread:
            thread.messages = []

        for context_message in context_messages:
            thread.messages.append(MessageModel.from_message(message=context_message))

        thread.messages.append(MessageModel.from_message(message=message))
        thread.current_mode = mode
        thread.cookies = cookies
        thread.executor = executor
        thread.model = model
        thread.compression = compression

        if mode == 'direct' or mode == 'tool' or mode == 'auto':
            endpoint = '/tools/completions'

        async with httpx.AsyncClient(timeout=400.0) as client:
            async with client.stream(
                'POST',
                f'{api_endpoint}/v1{endpoint}',
                json=thread.model_dump(),
            ) as response:
                objs = await stream_response(response, StreamPrinter('').write)

        await response.aclose()

        if objs:
            session_thread = SessionThread.model_validate(objs[-1])
            return session_thread
        return thread

    except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
        if mode == 'tool' or mode == 'code':
            logging.debug('LLMVM server is down, but we are in tool mode. Cannot execute directly')
            raise ex

    # server is down, go direct. this means that executor and model can't be nothing
    if not executor and not model:
        if Container.get_config_variable('LLMVM_EXECUTOR', 'executor'):
            executor = Container.get_config_variable('LLMVM_EXECUTOR')

        if Container.get_config_variable('LLMVM_MODEL', 'model'):
            model = Container.get_config_variable('LLMVM_MODEL')

    if executor and model:
        if executor == 'openai' and Container.get_config_variable('OPENAI_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('OPENAI_API_KEY'),
                'openai',
                model,
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        elif executor == 'anthropic' and Container.get_config_variable('ANTHROPIC_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('ANTHROPIC_API_KEY'),
                'anthropic',
                model,
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        elif executor == 'gemini' and Container.get_config_variable('GOOGLE_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('GOOGLE_API_KEY'),
                'gemini',
                model,
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        else:
            raise ValueError(f'Executor {executor} and model {model} are set, but no API key is set.')
    elif Container.get_config_variable('OPENAI_API_KEY'):
        assistant = await execute_llm_call_direct(
            message,
            Container.get_config_variable('OPENAI_API_KEY'),
            'openai',
            'gpt-4o',
            context_messages
        )
        return SessionThread(
            id=-1,
            messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
        )
    elif os.environ.get('ANTHROPIC_API_KEY'):
        assistant = await execute_llm_call_direct(
            message,
            Container.get_config_variable('ANTHROPIC_API_KEY'),
            'anthropic',
            'claude-2.1',
            context_messages
        )
        return SessionThread(
            id=-1,
            messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
        )
    elif os.environ.get('GOOGLE_API_KEY'):
        assistant = await execute_llm_call_direct(
            message,
            Container.get_config_variable('GOOGLE_API_KEY'),
            'gemini',
            'gemini-pro',
            context_messages
        )
        return SessionThread(
            id=-1,
            messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
        )
    else:
        logging.warning('Neither OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY is set. Unable to execute direct call to LLM.')  # noqa
        raise ValueError('Neither OPENAI_API_KEY, ANTHROPIC_API_KEY or GOOGLE_API_KEY is set. Unable to execute direct call to LLM.')  # noqa


def llm(
    message: Optional[str | bytes | Message],
    id: int,
    mode: str,
    endpoint: str,
    executor: str,
    model: str,
    context_messages: Sequence[Message] = [],
    cookies: List[Dict[str, Any]] = [],
    compression: str = 'auto',
) -> SessionThread:
    user_message = User(Content(''))
    if isinstance(message, str):
        user_message = User(Content(message))
    elif isinstance(message, bytes):
        user_message = User(Content(message.decode('utf-8')))
    elif isinstance(message, Message):
        user_message = message

    context_messages_list = list(context_messages)

    clear_thread = False
    # if the incoming message is a thread, parse it and send it through
    role_strings = ['Assistant: ', 'System: ', 'User: ']
    action_strings = ['ImageContent(', 'PdfContent(', 'FileContent(']

    if isinstance(message, str) and any(role_string in message for role_string in role_strings):
        all_messages = parse_message_thread(message)
        user_message = all_messages[-1]
        context_messages_list += all_messages[:-1]
        clear_thread = True
    # if the incoming message has actions [ImageContent(...), PdfContent(...), FileContent(...)] etc
    # parse those actions
    elif isinstance(message, str) and any(action_string in message for action_string in action_strings):
        all_messages = parse_message_actions(User, message)
        user_message = all_messages[-1]
        context_messages_list += all_messages[:-1]

    return asyncio.run(
        execute_llm_call(
            endpoint,
            id,
            user_message,
            executor,
            model,
            mode,
            context_messages_list,
            cookies,
            compression,
            clear_thread,
        )
    )

