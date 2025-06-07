import asyncio
from enum import Enum
import os
import time
from typing import Any, Awaitable, Callable, Optional, Union, cast
import uuid

import httpx
import nest_asyncio
from pydantic import TypeAdapter

from llmvm.client.parsing import parse_message_actions, parse_message_thread
from llmvm.client.printing import StreamPrinter, stream_response
from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.bedrock_executor import BedrockExecutor
from llmvm.common.container import Container
from llmvm.common.deepseek_executor import DeepSeekExecutor
from llmvm.common.gemini_executor import GeminiExecutor
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Assistant, AstNode, Content, Executor, TokenCompressionMethod,
                                  Message, MessageModel, SessionThreadModel,
                                  TextContent, User)
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.common.openai_tool_translator import OpenAIFunctionTranslator

logging = setup_logging()
nest_asyncio.apply()


class Mode(Enum):
    DIRECT = 'direct'
    TOOL = 'tool'


_printer = StreamPrinter()
async def default_stream_handler(node: AstNode):
    await _printer.write(node)  # type: ignore


def llm(
    messages: list[Message] | list[str] | str,
    executor: Optional[Executor] = None,
    model: Optional[str] = None,
    output_token_len: int = 8192,
    temperature: float = 0.0,
    stop_tokens: list[str] = [],
    thinking: int = 0,
    stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = default_stream_handler,
    template_args: Optional[dict[str, Any]] = None,
) -> Assistant:
    if (
        not isinstance(messages, list)
        and not isinstance(messages, str)
    ):
        raise ValueError('messages must be a list of Message objects or a string')

    if isinstance(messages, str):
        messages = cast(list[Message], [User(TextContent(messages))])
    elif isinstance(messages, list) and all(isinstance(message, str) for message in messages):
        messages = cast(list[str], messages)
        messages = cast(list[Message], [User(TextContent(message)) for message in messages])

    return asyncio.run(
        LLMVMClient(
            api_endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://localhost:8011'),
            default_executor_name=executor.name() if executor else Container.get_config_variable('LLMVM_EXECUTOR'),
            default_model_name=model if model else Container.get_config_variable('LLMVM_MODEL'),
            api_key='',
        ).call_direct(
            messages=cast(list[Message], messages),
            executor=executor,
            model=model,
            output_token_len=output_token_len,
            temperature=temperature,
            stop_tokens=stop_tokens,
            thinking=thinking,
            stream_handler=stream_handler,
            template_args=template_args,
        )
    )


def llmvm(
    messages: list[Message] | list[str] | str,
    executor_name: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    output_token_len: int = 8192,
    stop_tokens: list[str] = [],
    cookies: list[dict[str, Any]] = [],
    compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    mode: str = 'auto',
    thinking: int = 0,
    stream_handler: Callable[[AstNode], Awaitable[None]] = default_stream_handler,
    template_args: Optional[dict[str, Any]] = None,
) -> SessionThreadModel:
    if (
        not isinstance(messages, list)
        and not isinstance(messages, str)
    ):
        raise ValueError('messages must be a list of Message objects or a string')

    if isinstance(messages, str):
        messages = cast(list[Message], [User(TextContent(messages))])
    elif isinstance(messages, list) and all(isinstance(message, str) for message in messages):
        messages = cast(list[str], messages)
        messages = cast(list[Message], [User(TextContent(message)) for message in messages])

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
            thinking=thinking,
            stream_handler=stream_handler,
            template_args=template_args,
        )
    )


def get_executor(
    executor_name: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Executor:
    return get_client(executor_name, model_name, api_key).get_executor(executor_name, model_name, api_key)


def get_client(
    executor_name: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> 'LLMVMClient':
    if executor_name == 'anthropic' and not model_name:
        model_name = 'claude-sonnet-4-20250514'
    elif executor_name == 'openai' and not model_name:
        model_name = 'gpt-4.1'
    elif executor_name == 'gemini' and not model_name:
        model_name = 'gemini-2.5-pro-preview-05-06'
    elif executor_name == 'deepseek' and not model_name:
        model_name = 'deepseek-chat'
    elif executor_name == 'bedrock' and not model_name:
        model_name = 'amazon.nova-pro-v1:0'

    if executor_name == 'anthropic' and not api_key:
        api_key = Container().get_config_variable('ANTHROPIC_API_KEY', '')
    elif executor_name == 'openai' and not api_key:
        api_key = Container().get_config_variable('OPENAI_API_KEY', '')
    elif executor_name == 'gemini' and not api_key:
        api_key = Container().get_config_variable('GEMINI_API_KEY', '')
    elif executor_name == 'deepseek' and not api_key:
        api_key = Container().get_config_variable('DEEPSEEK_API_KEY', '')

    if not executor_name or not model_name:
        raise ValueError('executor_name, model_name, and api_key must be set')

    return LLMVMClient(
        api_endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://localhost:8011'),
        default_executor_name=executor_name,
        default_model_name=model_name,
        api_key=api_key if api_key else ''
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

    def __parse_template(self, message: Message, template_args: Optional[dict[str, Any]]) -> Message:
        if not template_args:
            return message
        # {{templates}}
        # check to see if any of the content nodes have {{templates}} in them, and if so, replace
        # with template_args.
        for content in message.message:
            if isinstance(content, TextContent):
                message_text = content.get_str()
                for key, value in template_args.items():
                    key_replace = '{{' + key + '}}'
                    if key_replace in message_text:
                        content.sequence = message_text.replace(key_replace, value)
        return message

    def __parse_messages(self, messages: list[Message]) -> list[Message]:
        thread_messages = messages
        thread_messages_copy = []
        for message in thread_messages:
            for content in message.message:
                if (
                    type(content) is TextContent
                    and any(role_string in content.get_str() for role_string in self.role_strings)
                ):
                    parsed_messages = parse_message_thread(content.get_str(), self.action_strings)
                    thread_messages_copy += parsed_messages
                # if the incoming message has actions [ImageContent(...)], [PdfContent(...)], [FileContent(...)] etc
                elif (
                    type(content) is TextContent
                    and any(action_string in content.get_str() for action_string in self.action_strings)
                ):
                    parsed_messages = parse_message_actions(User, content.get_str(), self.action_strings)
                    thread_messages_copy += parsed_messages
                else:
                    thread_messages_copy.append(message)
        return thread_messages_copy

    def get_executor(self, executor_name: str, model_name: Optional[str], api_key: Optional[str]) -> Executor:
        if executor_name == 'anthropic':
            if Container.get_config_variable('ANTHROPIC_API_KEY') or api_key:
                return AnthropicExecutor(
                    api_key=api_key or Container.get_config_variable('ANTHROPIC_API_KEY'),
                    default_model=cast(str, model_name) if model_name else 'claude-sonnet-4-20250514'
                )
            else:
                raise ValueError('anthropic executor requested, but unable to find Anthropic API key.')

        elif executor_name == 'openai':
            if Container.get_config_variable('OPENAI_API_KEY') or api_key:
                return OpenAIExecutor(
                    api_key=api_key or Container.get_config_variable('OPENAI_API_KEY'),
                    default_model=cast(str, model_name) if model_name else 'gpt-4.1'
                )
            else:
                raise ValueError('openai executor requested, but unable to find OpenAI API key.')

        elif executor_name == 'gemini':
            if Container.get_config_variable('GEMINI_API_KEY') or api_key:
                return GeminiExecutor(
                    api_key=api_key or Container.get_config_variable('GEMINI_API_KEY'),
                    default_model=cast(str, model_name) if model_name else 'gemini-2.5-pro-preview-05-06'
                )
            else:
                raise ValueError('GeminiExecutor requires a model name and API key.')

        elif executor_name == 'deepseek':
            if Container.get_config_variable('DEEPSEEK_API_KEY') or api_key:
                return DeepSeekExecutor(
                    api_key=api_key or Container.get_config_variable('DEEPSEEK_API_KEY'),
                    default_model=cast(str, model_name) if model_name else 'deepseek-chat'
                )
            else:
                raise ValueError('DeepSeekExecutor requires a model name and API key.')

        elif executor_name == 'bedrock':
            return BedrockExecutor(
                api_key=api_key or Container.get_config_variable('BEDROCK_API_KEY'),
                default_model=cast(str, model_name) if model_name else 'amazon.nova-pro-v1:0',
                region_name=Container.get_config_variable('BEDROCK_API_BASE', default='us-east-1'),
            )

        else:
            if Container.get_config_variable('ANTHROPIC_API_KEY'):
                return AnthropicExecutor(
                    api_key=Container.get_config_variable('ANTHROPIC_API_KEY'),
                    default_model='claude-sonnet-4-20250514'
                )
            elif Container.get_config_variable('OPENAI_API_KEY'):
                return OpenAIExecutor(
                    api_key=Container.get_config_variable('OPENAI_API_KEY'),
                    default_model='gpt-4.1'
                )
            elif Container.get_config_variable('GEMINI_API_KEY'):
                return GeminiExecutor(
                    api_key=Container.get_config_variable('GEMINI_API_KEY'),
                    default_model='gemini-2.5-pro-preview-05-06'
                )
            elif Container.get_config_variable('DEEPSEEK_API_KEY'):
                return DeepSeekExecutor(
                    api_key=Container.get_config_variable('DEEPSEEK_API_KEY'),
                    default_model='deepseek-chat'
                )
            raise ValueError('No API key is set for any executor in ENV. Unable to set default executor.')

    async def call_direct(
        self,
        messages: list[Message],
        executor: Optional[Executor] = None,
        model: Optional[str] = None,
        output_token_len: int = 8192,
        temperature: float = 0.0,
        stop_tokens: list[str] = [],
        thinking: int = 0,
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = None,
        template_args: Optional[dict[str, Any]] = None,
    ) -> Assistant:
        async def null_handler(node: AstNode):
            pass
        # todo: deal with template_args

        if not stream_handler:
            stream_handler = null_handler

        if not executor:
            executor = self.default_executor

        if not model:
            model = self.model

        thread_messages: list[Message] = self.__parse_messages(messages)
        thread_messages: list[Message] = [self.__parse_template(m, template_args) for m in thread_messages]

        assistant = await executor.aexecute(
            messages=thread_messages,
            max_output_tokens=output_token_len,
            temperature=temperature,
            stop_tokens=stop_tokens,
            model=model,
            thinking=thinking,
            stream_handler=stream_handler,
        )
        return assistant

    async def get_thread(
        self,
        id: int,
    ) -> SessionThreadModel:
        params = {
            'id': id,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_endpoint}/v1/chat/get_thread", params=params)
            return SessionThreadModel.model_validate(response.json())

    async def get_program(
        self,
        id: Optional[int],
        program_name: Optional[str],
    ) -> SessionThreadModel:
        params = {
            **({'id': id} if id else {}),
            **({'program_name': program_name} if program_name else {}),
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_endpoint}/v1/chat/get_program", params=params)
            result = SessionThreadModel.model_validate(response.json())
            return result

    async def set_thread(
        self,
        thread: SessionThreadModel,
    ) -> SessionThreadModel:
        async with httpx.AsyncClient(timeout=400.0) as client:
            response = await client.post(
                f'{self.api_endpoint}/v1/chat/set_thread',
                json=thread.model_dump()
            )
            session_thread = SessionThreadModel.model_validate(response.json())
            return session_thread

    async def set_thread_title(
        self,
        id: int,
        title: str,
    ) -> SessionThreadModel:
        async with httpx.AsyncClient(timeout=400.0) as client:
            response = await client.post(f'{self.api_endpoint}/v1/chat/set_thread_title', json={'id': id, 'title': title})
            session_thread = SessionThreadModel.model_validate(response.json())
            return session_thread

    async def get_threads(
        self,
    ) -> list[SessionThreadModel]:
        response: httpx.Response = httpx.get(f'{self.api_endpoint}/v1/chat/get_threads')
        threads = cast(list[SessionThreadModel], TypeAdapter(list[SessionThreadModel]).validate_python(response.json()))

        return threads

    async def call_with_session(
        self,
        session_thread: SessionThreadModel,
    ) -> SessionThreadModel:
        return await self.call(thread=session_thread)

    async def status(self) -> dict[str, str]:
        async with httpx.AsyncClient(timeout=2.0) as client:
            try:
                response = await client.get(f'{self.api_endpoint}/health')
                return response.json()
            except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
                return {'status': f'LLMVM server not available at {self.api_endpoint}. Set endpoint using $LLMVM_ENDPOINT.'}

    async def count_tokens(
        self,
        messages: list[Message],
        executor: Optional[Executor] = None,
    ) -> int:
        if not executor:
            executor = self.default_executor

        result = await executor.count_tokens(messages)
        return result

    async def compile(
        self,
        thread: int,
        program_name: str,
        compile_instructions: str = '',
        executor_name: Optional[str] = None,
        model_name: Optional[str] = None,
        cookies: list[dict[str, Any]] = [],
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
        thinking: int = 0,
        stream_handler: Callable[[AstNode], Awaitable[None]] = default_stream_handler,
    ) -> SessionThreadModel:
        thread_model: SessionThreadModel = await self.get_thread(thread)
        # user might want to override the executor, model, compression, thinking, or cookies
        # we're just compiling, so this matters less.
        thread_model.executor = executor_name or thread_model.executor
        thread_model.model = model_name or thread_model.model
        thread_model.compression = TokenCompressionMethod.get_str(compression)
        thread_model.thinking = thinking
        thread_model.cookies = cookies or thread_model.cookies
        thread_model.title = program_name
        thread_model.compile_prompt = compile_instructions

        try:
            async with httpx.AsyncClient(timeout=400.0) as client:
                async with client.stream(
                    'POST',
                    f'{self.api_endpoint}/v1/tools/compile',
                    json=thread_model.model_dump(),
                ) as response:
                    objs = await stream_response(response, stream_handler)

            await response.aclose()

            if objs:
                session_thread = SessionThreadModel.model_validate(objs[-1])
                return session_thread
            else:
                raise ValueError('compile() no result from server')

        except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
            logging.debug('compile() LLMVM server is down, cannot compile thread')
            raise ex


    async def call(
        self,
        thread: int | SessionThreadModel,
        messages: Union[list[Message], None] = None,
        executor_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 1.0,
        output_token_len: int = 8192,
        stop_tokens: list[str] = [],
        cookies: list[dict[str, Any]] = [],
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
        mode: str = '',
        thinking: int = 0,
        stream_handler: Callable[[AstNode], Awaitable[None]] = default_stream_handler,
        template_args: Optional[dict[str, Any]] = None,
    ) -> SessionThreadModel:
        if (
            (isinstance(messages, list) and len(messages) > 0 and not isinstance(messages[0], Message))
            and not isinstance(messages, type(None))
        ):
            raise ValueError('the messages argument must be a list of Message objects')

        # deal with weird message types and inputs
        thread_messages: list[Message] = []

        if isinstance(thread, SessionThreadModel):
            thread_messages = [MessageModel.to_message(session_message) for session_message in thread.messages]
        elif isinstance(messages, list):
            thread_messages = messages

        thread_messages: list[Message] = self.__parse_messages(thread_messages)
        thread_messages: list[Message] = [self.__parse_template(m, template_args) for m in thread_messages]

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
                thread.compression = TokenCompressionMethod.get_str(compression)
            if mode:
                thread.current_mode = mode
            thread.thinking = thinking

            # attach the messages to the thread
            thread.messages = [MessageModel.from_message(message) for message in thread_messages]

            async with httpx.AsyncClient(timeout=400.0) as client:
                async with client.stream(
                    'POST',
                    f'{self.api_endpoint}/v1/tools/completions',
                    json=thread.model_dump(),
                ) as response:
                    objs = await stream_response(response, stream_handler)

            await response.aclose()

            if objs:
                session_thread = SessionThreadModel.model_validate(objs[-1])
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
            thinking=thinking,
            template_args=template_args,
        )
        return SessionThreadModel(
            id=-1,
            messages=[MessageModel.from_message(message) for message in thread_messages + [assistant]],
            current_mode='direct',
            executor=executor.name(),
            model=model_name if model_name else self.model,
        )

    async def openai_tool_call(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | list[Callable],
        executor: Optional[Executor] = None,
        model: Optional[str] = None,
        temperature: float = 1.0,
        stream_handler: Callable[[AstNode], Awaitable[None]] = default_stream_handler,
        output_token_len: int = 4096,
        thinking: bool = False,
    ) -> dict:
        python_types: list[str] = []

        if not executor:
            executor = self.default_executor

        if not model:
            model = self.model

        if len(tools) == 0:
            raise ValueError('tools must be a non-empty list of functions or callables')

        tool_python_desc: list[str] = []
        if isinstance(tools[0], dict):
            # openai function description
            tool_python_desc = [OpenAIFunctionTranslator.generate_python_function_signature_from_oai_description(cast(dict[str, Any], tool)) for tool in tools]
        else:
            # callable
            tool_python_desc = [Helpers.get_function_description_flat(cast(Callable, tool)) for tool in tools]

        system_message, tools_message = Helpers.prompts(
            prompt_name='tool_call.prompt',
            template={
                'functions': '\n'.join(tool_python_desc),
                'task': messages[-1].get_str(),
                'types': '\n'.join(python_types),
            },
            user_token=executor.user_token(),
            assistant_token=executor.assistant_token(),
            scratchpad_token=executor.scratchpad_token(),
            append_token=executor.append_token(),
        )

        assistant = await self.call_direct(
            messages=[system_message] + messages + [tools_message],
            executor=executor,
            model=model,
            temperature=temperature,
            output_token_len=output_token_len,
            stop_tokens=[],
            stream_handler=stream_handler,
            thinking=thinking,
        )

        logging.debug(f'LLMVMClient.openai_tool_call tool_call_prompt result: {assistant.get_str()}')

        # assistant should return [python_call()]
        tool_calls = OpenAIFunctionTranslator.parse_python_tool_call_result_to_oai_choices(assistant.get_str().strip())

        assistant_result = None

        if not tool_calls:
            non_tool_result = await self.call_direct(
                messages=[system_message] + messages,
                executor=executor,
                model=model,
                temperature=temperature,
                output_token_len=output_token_len,
                stop_tokens=[],
                stream_handler=stream_handler,
                thinking=thinking,
            )
            assistant_result = non_tool_result.get_str().strip()

        openai_response = {}
        openai_response['id'] = f"chatcmpl-{uuid.uuid4().hex}"
        openai_response['object'] = "chat.completion"
        openai_response['created'] = int(time.time())
        openai_response['model'] = model
        openai_response['choices'] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_result,
                "refusal": None,
                "annotations": [],
                **({"tool_calls": tool_calls} if tool_calls else {})
            },
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "logprobs": None,
        }]

        openai_response['usage'] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        }
        if tool_calls:
            openai_response['status'] = "requires_action"
        return openai_response
