import asyncio
import base64
import os
from io import BytesIO
from typing import Awaitable, Callable, Dict, List, Optional, cast

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import Function
from PIL import Image

from llmvm.common.logging_helpers import messages_trace, setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Assistant, AstNode, Content, Executor, MarkdownContent,
                                  Message, PdfContent, System, TokenStopNode, User,
                                  awaitable_none)
from llmvm.common.perf import TokenPerf, TokenStreamManager

logging = setup_logging()

class OpenAIExecutor(Executor):
    def __init__(
        self,
        api_key: str = cast(str, os.environ.get('OPENAI_API_KEY')),
        default_model: str = 'gpt-4o',
        api_endpoint: str = 'https://api.openai.com/v1',
        default_max_token_len: int = 128000,
        default_max_output_len: int = 4096,
    ):
        super().__init__(
            default_model=default_model,
            api_endpoint=api_endpoint,
            default_max_token_len=default_max_token_len,
            default_max_output_len=default_max_output_len,
        )
        self.openai_key = api_key
        self.aclient = AsyncOpenAI(api_key=self.openai_key, base_url=self.api_endpoint)

    def user_token(self) -> str:
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

    def __calculate_image_tokens(self, width: int, height: int):
        from math import ceil

        h = ceil(height / 512)
        w = ceil(width / 512)
        n = w * h
        total = 85 + 170 * n
        return total

    def count_tokens(
        self,
        messages: List[Message] | List[Dict[str, str]] | str,
        model: Optional[str] = None,
    ) -> int:
        model_str = model if model else self.default_model

        # obtained from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        def num_tokens_from_messages(messages, model: str):
            """Return the number of tokens used by a list of messages."""
            encoding = tiktoken.get_encoding('cl100k_base')
            if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-3.5-turbo-0125",
                "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-vision-preview",
                "gpt-4o",
                "gpt-4-1106-preview",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
            }:
                tokens_per_message = 3
                tokens_per_name = 1
            elif model == "gpt-3.5-turbo-0301":
                tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif "gpt-3.5-turbo" in model:
                return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
            elif "gpt-4" in model:
                return num_tokens_from_messages(messages, model="gpt-4-0613")
            else:
                logging.error(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")  # noqa: E501
                tokens_per_message = 3
                tokens_per_name = 1
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message

                for key, value in message.items():
                    if isinstance(value, list):
                        for list_item in value:
                            if 'type' in list_item and list_item['type'] == 'image_url' and 'image_url' in list_item:
                                if 'detail' in list_item['image_url'] and list_item['image_url']['detail'] == 'high':
                                    try:
                                        with Image.open(BytesIO(base64.b64decode(list_item['image_url']['url'].split(',')[1]))) as img:  # NOQA: E501
                                            width, height = img.size
                                            num_tokens += self.__calculate_image_tokens(width=width, height=height)
                                    except Exception as e:
                                        num_tokens += 85
                                else:
                                    num_tokens += 85
                    else:
                        num_tokens += len(encoding.encode(value))
                        if key == "name":
                            num_tokens += tokens_per_name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens

        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], Message):
            dict_messages = [Message.to_dict(m) for m in messages]  # type: ignore
            return num_tokens_from_messages(dict_messages, model=model_str)
        elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            return num_tokens_from_messages(messages, model=model_str)
        elif isinstance(messages, str):
            return num_tokens_from_messages([Message.to_dict(User(Content(messages)))], model=model_str)
        else:
            raise ValueError('cannot calculate tokens for messages: {}'.format(messages))

    def name(self) -> str:
        return 'openai'

    async def __aexecute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        stop_tokens: List[str] = [],
        model: Optional[str] = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> TokenStreamManager:
        model = model if model else self.default_model

        # only works if profiling or LLMVM_PROFILING is set to true
        message_tokens = self.count_tokens(messages, model=model)
        if message_tokens > self.max_input_tokens(max_output_tokens, model=model):
            raise Exception('Prompt too long. prompt tokens: {}, completion tokens: {}, total: {}, max context window: {}'
                            .format(message_tokens,
                                    max_output_tokens,
                                    message_tokens + max_output_tokens,
                                    self.max_tokens(model)))

        messages_cast = cast(List[ChatCompletionMessageParam], messages)
        functions_cast = cast(List[Function], functions)

        token_trace = TokenPerf('__aexecute_direct', 'openai', model, prompt_len=message_tokens)  # type: ignore
        token_trace.start()

        base_params = {
            "model": model if model else self.default_model,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "messages": messages_cast,
            "stop": stop_tokens if stop_tokens else None,
            "functions": functions_cast if functions else None,
            "stream": True
        }

        params = {k: v for k, v in base_params.items() if v is not None}
        response = await self.aclient.chat.completions.create(**params)

        return TokenStreamManager(response, token_trace)  # type: ignore

    async def aexecute(
        self,
        messages: List[Message],
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        stop_tokens: List[str] = [],
        model: Optional[str] = None,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
    ) -> Assistant:
        model = model if model else self.default_model

        def last(predicate, iterable):
            result = [x for x in iterable if predicate(x)]
            if result:
                return result[-1]
            return None

        # find the system message and append to the front
        system_message = last(lambda x: x.role() == 'system', messages)

        if not system_message:
            system_message = System(Content('You are a helpful assistant.'))

        expanded_messages = []
        for message in messages:
            if isinstance(message, User) and isinstance(message.message, PdfContent):
                expanded_messages.extend(ObjectTransformers.transform_pdf_content(message.message, self))
            elif isinstance(message, User) and isinstance(message.message, MarkdownContent):
                expanded_messages.extend(ObjectTransformers.transform_markdown_content(message.message, self))
            else:
                expanded_messages.append(message)

        # fresh message list
        messages_list: List[Dict[str, str]] = []

        messages_list.append(Message.to_dict(system_message))
        for message in [m for m in expanded_messages if m.role() != 'system']:
            messages_list.append(Message.to_dict(message))

        stream = self.__aexecute_direct(
            messages_list,
            max_output_tokens=max_output_tokens,
            model=model if model else self.default_model,
            temperature=temperature,
            stop_tokens=stop_tokens,
        )

        text_response = ''
        perf = None

        async with await stream as stream_async:  # type: ignore
            async for text in stream_async:  # type: ignore
                await stream_handler(Content(text))
                text_response += text
            await stream_handler(TokenStopNode())
            perf = stream_async.perf

        messages_list.append({'role': 'assistant', 'content': text_response})
        conversation: List[Message] = [Message.from_dict(m) for m in messages_list]

        await stream_async.get_final_message()

        # todo, stashing this in 'perf' isn't great, should probably fix that.
        assistant = Assistant(
            message=conversation[-1].message,
            messages_context=conversation,
            stop_reason=stream_async.perf.stop_reason,
            stop_token=stream_async.perf.stop_token,
        )

        perf.log()

        messages_trace(messages_list)

        return assistant

    def execute(
        self,
        messages: List[Message],
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        stop_tokens: List[str] = [],
        model: Optional[str] = None,
        stream_handler: Optional[Callable[[AstNode], None]] = None,
    ) -> Assistant:
        async def stream_pipe(node: AstNode):
            if stream_handler:
                stream_handler(node)

        return asyncio.run(self.aexecute(messages, max_output_tokens, temperature, stop_tokens, model, stream_pipe))
