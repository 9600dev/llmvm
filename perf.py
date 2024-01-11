import datetime as dt
import os
import time
from typing import Any, Dict, List, Optional, Type

from anthropic import (AsyncMessageStream, AsyncMessageStreamManager,
                       AsyncMessageStreamT)
from anthropic.types.beta import MessageStreamEvent
from typing_extensions import override

from container import Container
from helpers.logging_helpers import setup_logging

logging = setup_logging()


class LoggingAsyncIterator:
    def __init__(self, original_iterator, token_perf):
        self.original_iterator = original_iterator.__aiter__()
        self.perf = token_perf

    async def __anext__(self):
        try:
            result = await self.original_iterator.__anext__()
            self.perf.tick()
            return result
        except StopAsyncIteration:
            self.perf.stop()
            self.perf.log()
            raise


class TokenPerfWrapper:
    def __init__(self, original_stream, token_perf: Optional['TokenPerf']):
        self.original_stream = original_stream
        self.perf = token_perf

    def __aiter__(self):
        if self.perf and self.perf.enabled:
            return LoggingAsyncIterator(self.original_stream, self.perf)
        else:
            return self.original_stream


class MyAnthropicStream(AsyncMessageStream):
    def __init__(self, token_perf: Optional['TokenPerf']):
        self.perf = token_perf

    @override
    async def on_stream_event(self, event: MessageStreamEvent) -> None:
        if self.perf: self.perf.tick()


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
            )
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
            prompt_time = self._ticks[0] - self._start
            sample_time = total_time - prompt_time
            average_token = avg(self.ticks())
            avg_token_sec = len(self._ticks) / total_time
            return {
                'name': self._name,
                'executor': self._executor,
                'model': self._model,
                'total_time': total_time,
                'prompt_time': prompt_time,
                'prompt_len': self._prompt_len,
                'sample_time': sample_time,
                'avg_token': average_token,
                'avg_token_sec': avg_token_sec,
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
            result = f'{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")},{res["name"]},{res["executor"]},{res["model"]},{res["total_time"]},{res["prompt_time"]},{res["prompt_len"]},{res["sample_time"]},{res["avg_token"]},{res["avg_token_sec"]},{",".join([f"{t:.8f}" for t in res["ticks"]])}'
            return result
        else:
            return ''

    def debug(self):
        if self.enabled:
            res = self.result()
            logging.debug(f"total_time: {res['total_time']:.4} prompt_time: {res['prompt_time']:.3} sample_time: {res['sample_time']:.3} prompt_len: {res['prompt_len']} sample_len: {len(res['ticks'])} avg_token: {res['avg_token']:.3} avg_tok_sec: {res['avg_token_sec']:.3}")

    def log(self):
        if self.enabled:
            self.debug()
            if not os.path.exists(os.path.expanduser(self.log_file)):
                with open(os.path.expanduser(self.log_file), 'w') as f:
                    f.write('name,executor,model,total_time,prompt_time,prompt_tokens,sample_time,sample_tokens,avg_token,avg_token_sec,ticks\n')
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
                'avg_token': 0.0,
                'avg_token_sec': 0.0,
                'ticks': []
            }
