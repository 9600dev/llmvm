import datetime as dt
import os
import time
from typing import List

from anthropic import AsyncMessageStream, AsyncMessageStreamManager

from llmvm.common.container import Container
from llmvm.common.logging_helpers import setup_logging

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
            result = f'{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")},{res["name"]},{res["executor"]},{res["model"]},{res["total_time"]},{res["prompt_time"]},{res["sample_time"]},{res["prompt_len"]},{res["sample_len"]},{res["p_tok_sec"]},{res["s_tok_sec"]},{",".join([f"{t:.8f}" for t in res["ticks"]])}'
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
            logging.debug(f"p_tok_sec {res['p_tok_sec']:.2f} s_tok_sec: {res['s_tok_sec']:.2f}")

    def log(self):
        if self.enabled:
            self.debug()
            if not os.path.exists(os.path.expanduser(self.log_file)):
                with open(os.path.expanduser(self.log_file), 'w') as f:
                    f.write('name,executor,model,total_time,prompt_time,prompt_tokens,sample_time,prompt_len,sample_len,p_tok_sec,s_tok_sec,ticks\n')
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
                'ticks': []
            }


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
    def __init__(self, original_stream, token_perf: 'TokenPerf'):
        self.stream = original_stream
        self.perf = token_perf

    def __aiter__(self):
        if self.perf.enabled:
            return LoggingAsyncIterator(self.stream, self.perf)
        else:
            return self.stream

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


class TokenPerfWrapperAnthropic:
    def __init__(self, stream_manager: AsyncMessageStreamManager, token_perf: 'TokenPerf'):
        self.stream_manager: AsyncMessageStreamManager = stream_manager
        self.perf = token_perf
        self.token_perf_wrapper = None

    async def __aenter__(self):
        self.perf.start()
        result: AsyncMessageStream = await self.stream_manager.__aenter__()
        self.token_perf_wrapper = TokenPerfWrapper(result.text_stream, self.perf)
        return self.token_perf_wrapper

    async def __aexit__(self, exc_type, exc, tb):
        await self.stream_manager.__aexit__(exc_type, exc, tb)

    # act as a proxy to self.original_stream for any other methods that are called
    def __getattr__(self, name):
        return getattr(self.stream, name)
