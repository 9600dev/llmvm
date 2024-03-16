import datetime as dt
import inspect
import os
import time
from typing import List, cast

import openai
from anthropic import AsyncMessageStream, AsyncMessageStreamManager
from anthropic import AsyncStream as AnthropicAsyncStream
from anthropic.types import Completion as AnthropicCompletion
from google.generativeai.types.generation_types import \
    AsyncGenerateContentResponse as AsyncGeminiStream
from google.generativeai.types.generation_types import \
    GenerateContentResponse as GeminiCompletion
from mistralai.models.chat_completion import \
    ChatCompletionStreamResponse as MistralCompletion
from openai.types.chat.chat_completion_chunk import \
    ChatCompletionChunk as OAICompletion

from llmvm.common.calculator import TokenPriceCalculator
from llmvm.common.container import Container
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import Content

logging = setup_logging()


class TokenPerf:
    # class to measure time taken between start() and stop() for a given task
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
        request_id: str = ''
    ):
        self._name: str = name
        self._executor: str = executor_name
        self._model: str = model_name
        self._start: float = 0.0
        self._stop: float = 0.0
        self._prompt_len: int = prompt_len
        self._ticks: List[float] = []
        self.enabled = enabled
        self.log_file = log_file
        self.calculator = TokenPriceCalculator()
        self.request_id = request_id

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
        def avg(list):
            return sum(list) / len(list)

        if self.enabled:
            total_time = self._stop - self._start
            prompt_time = self._ticks[0] - self._start if self._ticks else 0
            sample_time = total_time - prompt_time
            try:
                s_tok_sec = len(self._ticks) / total_time
            except ZeroDivisionError:
                s_tok_sec = 0.0
            try:
                p_tok_sec = self._prompt_len / prompt_time
            except ZeroDivisionError:
                p_tok_sec = 0.0
            return {
                'name': self._name,
                'executor': self._executor,
                'model': self._model,
                'total_time': total_time,
                'prompt_time': prompt_time,
                'sample_time': sample_time,
                'prompt_len': self._prompt_len,
                'sample_len': len(self._ticks),
                's_tok_sec': s_tok_sec,
                'p_tok_sec': p_tok_sec,
                'p_cost': self._prompt_len * self.calculator.prompt_price(self._model, self._executor),
                's_cost': len(self._ticks) * self.calculator.sample_price(self._model, self._executor),
                'request_id': self.request_id,
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
            result = f'{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")},{res["name"]},{res["executor"]},{res["model"]},{res["total_time"]},{res["prompt_time"]},{res["sample_time"]},{res["prompt_len"]},{res["sample_len"]},{res["p_tok_sec"]},{res["s_tok_sec"]},{res["request_id"]},{",".join([f"{t:.8f}" for t in res["ticks"]])}'
            return result
        else:
            return ''

    def debug(self):
        if self.enabled:
            res = self.result()
            # output \n to the debug stream without using logging.debug
            import sys
            sys.stderr.write('\n')
            logging.debug(f"total_time: {res['total_time']:.2f} prompt_time: {res['prompt_time']:.2f} sample_time: {res['sample_time']:.2f}")
            logging.debug(f"prompt_len: {res['prompt_len']} sample_len: {len(res['ticks'])}")
            logging.debug(f"p_tok_sec: {res['p_tok_sec']:.2f} s_tok_sec: {res['s_tok_sec']:.2f}")
            logging.debug(f"p_cost: ${res['p_cost']:.5f} s_cost: ${res['s_cost']:.5f} request_id: {res['request_id']}")

    def log(self):
        if self.enabled:
            self.debug()
            if not os.path.exists(os.path.expanduser(self.log_file)):
                with open(os.path.expanduser(self.log_file), 'w') as f:
                    f.write('name,executor,model,total_time,prompt_time,prompt_tokens,sample_time,prompt_len,sample_len,p_tok_sec,s_tok_sec,p_cost,s_cost,request_id,ticks\n')
            with open(os.path.expanduser(self.log_file), 'a') as f:
                result = str(self)
                f.write(result + '\n')
                return self.result()
        else:
            return {
                'name': self._name,
                'executor': self._executor,
                'total_time': 0.0,
                'prompt_time': 0.0,
                'sample_time': 0.0,
                'prompt_len': 0,
                'sample_len': 0,
                'p_tok_sec': 0.0,
                's_tok_sec': 0.0,
                'p_cost': 0.0,
                's_cost': 0.0,
                'request_id': '',
                'ticks': []
            }


class LoggingAsyncIterator:
    def __init__(self, original_iterator, token_perf):
        self.original_iterator = original_iterator.__aiter__()
        self.perf = token_perf

    async def __anext__(self) -> str:
        try:
            result = await self.original_iterator.__anext__()

            if self.perf.enabled:
                self.perf.tick()

            if isinstance(result, AnthropicCompletion):
                return cast(str, result.completion or '')
            elif isinstance(result, OAICompletion):
                return cast(str, result.choices[0].delta.content or '')  # type: ignore
            elif isinstance(result, MistralCompletion):
                return cast(str, result.choices[0].delta.content or '')  # type: ignore
            elif isinstance(result, GeminiCompletion):
                return cast(str, result.text or '')
            elif isinstance(result, str):
                return result
            else:
                raise ValueError(f'Unknown completion type: {type(result)}, stream type: {type(self.original_iterator)}')
        except StopAsyncIteration:
            if self.perf.enabled:
                self.perf.stop()
                self.perf.log()
            raise


class TokenStreamWrapper:
    def __init__(
        self,
        original_stream,
        token_perf: 'TokenPerf'
    ):
        self.stream = original_stream
        self.perf = token_perf

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
    async def get_final_message(self):
        return ''


class TokenStreamManager:
    def __init__(
        self,
        stream: AsyncMessageStreamManager | AnthropicAsyncStream[AnthropicCompletion] | openai.AsyncStream,  # type: ignore
        token_perf: 'TokenPerf'
    ):
        self.stream = stream
        self.perf = token_perf
        self.token_perf_wrapper = None

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
            return self.token_perf_wrapper
        elif isinstance(self.stream, openai.AsyncStream):
            self.token_perf_wrapper = TokenStreamWrapper(self.stream, self.perf)
            return self.token_perf_wrapper
        elif isinstance(self.stream, AsyncGeminiStream):
            self.token_perf_wrapper = TokenStreamWrapper(self.stream, self.perf)
            return self.token_perf_wrapper
        elif isinstance(self.stream, AnthropicAsyncStream):
            self.token_perf_wrapper = TokenStreamWrapper(self.stream, self.perf)
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
