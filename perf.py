import datetime as dt
import os
import time
from typing import List

from container import Container


class TokenPerf:
    # class to measure time taken between start() and stop() for a given task
    def __init__(
            self,
            name: str,
            executor_name: str,
            model_name: str,
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
        self._ticks: List[float] = []
        self.enabled = enabled
        self.log_file = log_file

    def start(self):
        if self.enabled:
            self._start = time.perf_counter()

    def stop(self):
        if self.enabled:
            self._stop = time.perf_counter()
            return self.log()
        else:
            return (0.0, 0.0, 0.0, 0.0, [])

    def reset(self):
        self._ticks = []

    def result(self):
        return self._stop - self._start

    def tick(self):
        if self.enabled:
            self._ticks.append(time.perf_counter())

    def ticks(self):
        if self.enabled:
            return [self._ticks[i] - self._ticks[i - 1] for i in range(1, len(self._ticks))]
        else:
            return []

    def log(self):
        def avg(l):
            return sum(l) / len(l)

        if self.enabled:
            if not os.path.exists(os.path.expanduser(self.log_file)):
                with open(os.path.expanduser(self.log_file), 'w') as f:
                    f.write('name,executor,model,total_time,prompt_time,avg_token,avg_token_sec,ticks\n')
            with open(os.path.expanduser(self.log_file), 'a') as f:
                total_time = self.result()
                prompt_time = self._ticks[0] - self._start
                average_token = avg(self.ticks())
                avg_token_sec = len(self._ticks) / total_time
                csv = ','.join([f"{t:.8f}" for t in self.ticks()])
                result = f'{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")},{self._name},{self._executor},{self._model},{total_time},{prompt_time},{average_token},{avg_token_sec},{csv}'
                f.write(result + '\n')
                return (total_time, prompt_time, average_token, avg_token_sec, self.ticks())
        else:
            return (0.0, 0.0, 0.0, 0.0, [])
