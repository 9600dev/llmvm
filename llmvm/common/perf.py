import datetime as dt
import inspect
import json
import os
import time
from typing import Any, List, Optional, cast

import openai
from botocore.eventstream import EventStream
from anthropic import AsyncMessageStream, AsyncMessageStreamManager
from anthropic import AsyncStream as AnthropicAsyncStream
from anthropic.types import Completion as AnthropicCompletion
from anthropic.types import Message as AnthropicMessage
from openai.types.chat.chat_completion_chunk import \
    ChatCompletionChunk as OAICompletionChunk
from openai.types.chat.chat_completion import ChatCompletion as OAICompletion

from llmvm.common.container import Container
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import TokenPriceCalculator, TokenPerf
from llmvm.common.helpers import Helpers

logging = setup_logging()


class AsyncIteratorWrapper:
    def __init__(self, sync_iterator):
        self.sync_iterator = iter(sync_iterator)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.sync_iterator)
        except StopIteration:
            raise StopAsyncIteration


class O1AsyncIterator:
    def __init__(self, item):
        self.item = item
        self.done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.done:
            self.done = True
            return self.item
        else:
            raise StopAsyncIteration


class LoggingAsyncIterator:
    def __init__(self, original_iterator, token_perf: TokenPerf):
        if Helpers.is_sync_iterator(original_iterator) and not Helpers.is_async_iterator(original_iterator):
            self.original_iterator = AsyncIteratorWrapper(original_iterator)
            self.perf = token_perf
        else:
            self.original_iterator = original_iterator.__aiter__()
            self.perf = token_perf

    async def __anext__(self) -> str:
        try:
            result = await self.original_iterator.__anext__()

            if self.perf.enabled:
                self.perf.tick()

            if isinstance(result, AnthropicCompletion):
                return cast(str, result.completion or '')
            elif isinstance(result, OAICompletionChunk):
                if result.choices and len(result.choices) > 0 and result.choices[0].finish_reason:
                    self.perf.stop_reason = result.choices[0].finish_reason
                if result.usage:
                    self.perf.total_tokens = result.usage.total_tokens  # type: ignore
                if result.choices and len(result.choices) > 0:
                    return cast(str, result.choices[0].delta.content or '')  # type: ignore
                else:
                    return ''
            elif isinstance(result, OAICompletion):  # o1 mini and o1 preview don't do streaming
                if result.choices and len(result.choices) > 0 and result.choices[0].finish_reason:
                    self.perf.stop_reason = result.choices[0].finish_reason
                if 'usage' in result and 'total_tokens' in result['usage']:  # type: ignore
                    self.perf.total_tokens = result.usage.total_tokens  # type: ignore
                if result.choices and len(result.choices) > 0:
                    return cast(str, result.choices[0].message.content or '')  # type: ignore
                else:
                    return ''
            # amazon nova
            elif isinstance(result, dict) and 'chunk' in result:
                chunk = result['chunk']
                chunk_json = json.loads(chunk.get("bytes").decode())
                content_block_delta = chunk_json.get("contentBlockDelta")
                if content_block_delta:
                    return cast(str, content_block_delta.get("delta").get("text"))
                else:
                    return ''
            elif isinstance(result, str):
                return result
            else:
                raise ValueError(f'Unknown completion type: {type(result)}, stream type: {type(self.original_iterator)}')
        except StopAsyncIteration:
            if self.perf.enabled:
                self.perf.stop()
            raise


class TokenStreamWrapper:
    def __init__(
        self,
        original_stream,
        token_perf: 'TokenPerf'
    ):
        self.stream = original_stream
        self.perf = token_perf
        self.object: Optional[Any] = None  # used for Anthropic's AsyncMessageStream needed for get_final_message()

    def __aiter__(self):
        return LoggingAsyncIterator(self.stream, self.perf)

    # act as a proxy to self.original_stream for any other methods that are called
    def __getattr__(self, name):
        if name == 'text_stream':
            return self
        else:
            return getattr(self.stream, name)

    # we're proxying the .text_stream, which doesn't have get_final_message()
    # but we don't need it as we're streaming
    async def get_final_message(self) -> Optional[AnthropicMessage]:
        if isinstance(self.object, AsyncMessageStream):
            final_message = await self.object.get_final_message()
            self.perf._prompt_len = final_message.usage.input_tokens
            self.perf._completion_len = final_message.usage.output_tokens
            self.perf.object = final_message  # type: ignore
            self.perf.stop_reason = str(final_message.stop_reason) if final_message.stop_reason else ''
            self.perf.stop_token = final_message.stop_sequence if final_message.stop_sequence else ''
            self.perf.total_tokens = final_message.usage.input_tokens + final_message.usage.output_tokens + (final_message.usage.cache_read_input_tokens or 0)
            await self.object.close()
            return final_message
        else:
            return None


class TokenStreamManager:
    def __init__(
        self,
        stream: AsyncMessageStreamManager | AnthropicAsyncStream[AnthropicCompletion] | openai.AsyncStream | EventStream,  # type: ignore
        token_perf: 'TokenPerf',
        stop_tokens: List[str] = [],
        response_object: Any = None
    ):
        self.stream = stream
        self.perf = token_perf
        self.token_perf_wrapper = None
        self.stop_tokens = stop_tokens
        self.response_object = response_object

    async def __aenter__(self) -> TokenStreamWrapper:
        self.perf.start()

        if isinstance(self.stream, AsyncMessageStreamManager):
            result: AsyncMessageStream = await self.stream.__aenter__()
            # special anthropic debugging
            self.perf.request_id = result.response.headers['request-id']
            if Container().get_config_variable('LLMVM_SHARE', default=''):
                with open(os.path.expanduser(Container().get_config_variable('LLMVM_SHARE') + f'/{self.perf.request_id}.json'), 'w') as f:
                    f.write(result.response._request._content.decode('utf-8'))  # type: ignore
            self.token_perf_wrapper = TokenStreamWrapper(result.text_stream, self.perf)  # type: ignore
            self.token_perf_wrapper.object = result

            return self.token_perf_wrapper
        elif isinstance(self.stream, openai.AsyncStream) or isinstance(self.stream, O1AsyncIterator):
            self.token_perf_wrapper = TokenStreamWrapper(self.stream, self.perf)
            return self.token_perf_wrapper
        elif isinstance(self.stream, AnthropicAsyncStream):
            self.token_perf_wrapper = TokenStreamWrapper(self.stream, self.perf)
            return self.token_perf_wrapper
        elif isinstance(self.stream, EventStream):
            self.token_perf_wrapper = TokenStreamWrapper(self.stream.__iter__(), self.perf)
            self.token_perf_wrapper.object = self.response_object
            return self.token_perf_wrapper
        elif inspect.isasyncgen(self.stream):
            self.token_perf_wrapper = TokenStreamWrapper(self.stream, self.perf)
            return self.token_perf_wrapper

        logging.error(f'Unknown stream type: {type(self.stream)}')
        return TokenStreamWrapper(self.stream, self.perf)

    async def __aexit__(self, exc_type, exc, tb):
        if isinstance(self.stream, AsyncMessageStreamManager):
            await self.stream.__aexit__(exc_type, exc, tb)

    async def close(self):
        return

    # act as a proxy to self.original_stream for any other methods that are called
    def __getattr__(self, name):
        return getattr(self.stream, name)
