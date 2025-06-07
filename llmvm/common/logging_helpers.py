import datetime as dt
import json
import logging
import os
import sys
import time
from logging import Logger
from typing import Dict, List, Any

import rich
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from llmvm.common.container import Container


def __trace(content):
    try:
        if Container.get_config_variable('LLMVM_EXECUTOR_TRACE', default=''):
            with open(os.path.expanduser(Container.get_config_variable('LLMVM_EXECUTOR_TRACE')), 'a+') as f:
                f.write(content)
    except Exception as e:
        rich.print(f"Error tracing: {e}")


def messages_trace(messages: List[Dict[str, Any]]):
    if Container.get_config_variable('LLMVM_EXECUTOR_TRACE', default=''):
        for m in messages:
            if 'content' in m:
                __trace(f"<{m['role'].capitalize()}:>{m['content']}</{m['role'].capitalize()}>\n\n")
            elif 'parts' in m and isinstance(m['parts'], list) and isinstance(m['parts'][0], dict) and 'inline_data' in m['parts'][0]:
                # ImageContent todo fix properly
                __trace(f"<{m['role'].capitalize()}:>[ImageContent()]</{m['role'].capitalize()}>\n\n")
            elif 'parts' in m:
                content = ' '.join(m['parts'])
                __trace(f"<{m['role'].capitalize()}:>{content}</{m['role'].capitalize()}>\n\n")


def serialize_messages(messages):
    if Container.get_config_variable('LLMVM_SERIALIZE', default=''):
        result = json.dumps([m.to_json() for m in messages], indent=2)
        file_path = os.path.expanduser(Container.get_config_variable('LLMVM_SERIALIZE'))
        with open(file_path, 'a+') as f:
            f.write(result + '\n\n')
            f.flush()


class TimedLogger(logging.Logger):
    def __init__(self, name='timing', level=logging.NOTSET):
        super().__init__(name, level)
        self._start_time = None
        self._intermediate_timings = {}
        self._prepend = ''

    def start(self, prepend=''):
        self._start_time = time.time()
        self._intermediate_timings.clear()  # Clear previous intermediate timings
        self._prepend = prepend

    def save_intermediate(self, label):
        if self._start_time is None:
            self.warning("Timer was not started!")
            return

        if label in self._intermediate_timings:
            return

        current_time = time.time()
        elapsed_time = (current_time - self._start_time) * 1000  # Convert to milliseconds
        self._intermediate_timings[label] = elapsed_time
        self.debug(f"'{label}' timing: {elapsed_time:.2f} ms {self._prepend}")

    def end(self, message="Elapsed time"):
        if self._start_time is None:
            self.warning("Timer was not started!")
            return
        elapsed_time = (time.time() - self._start_time) * 1000  # Convert to milliseconds
        self.debug(f"{message}: {elapsed_time:.2f} ms {self._prepend}")
        # Optionally, log intermediate timings at the end
        for label, timing in self._intermediate_timings.items():
            self.debug(f"'{label}' timing: {timing:.2f} ms {self._prepend}")
        self._start_time = None
        self._intermediate_timings.clear()


timing = TimedLogger()
global_loggers: Dict[str, Logger] = {}
handler = RichHandler()

if not os.path.exists(Container.get_config_variable('log_directory', default='~/.local/share/llmvm/logs')):
    os.makedirs(Container.get_config_variable('log_directory', default='~/.local/share/llmvm/logs'))


def no_indent_debug(logger, message) -> None:
    if logger.level <= logging.DEBUG:
        console = Console(file=sys.stderr)
        console.print(message)


def role_debug(logger, callee, role, message) -> None:
    def split_string_by_width(input_string, width=20):
        result = ''
        width_counter = 0

        for i in range(0, len(input_string)):
            if width_counter >= width:
                width_counter = 0
                result += '\n'

            if input_string[i] == '\n':
                result += input_string[i]
                width_counter = 0
            elif width_counter < width:
                result += input_string[i]
                width_counter += 1
        return result.split('\n')

    if logger.level <= logging.DEBUG:
        if callee.startswith('prompts/'):
            callee = callee.replace('prompts/', '')

        console = Console(file=sys.stderr)
        width, _ = console.size
        callee_column = 20
        role_column = 10
        text_column = width - callee_column - role_column - 4

        # message_lines = message.split('\n')
        message_lines = split_string_by_width(message, width=text_column)
        header = True
        counter = 1
        max_lines = 20
        try:
            for message in message_lines:
                if header:
                    console.print('[orange]{}[/orange][green]{}[/green][grey]{}[/grey]'.format(
                        callee[0:callee_column - 1].ljust(callee_column)[:callee_column],
                        role.ljust(role_column)[:role_column],
                        message.ljust(text_column)[:text_column]
                    ))
                    header = False
                elif counter < max_lines or counter >= len(message_lines) - 5:
                    console.print('{}{}{}'.format(
                        ''.ljust(callee_column),
                        ''.ljust(role_column),
                        message.ljust(text_column)[:text_column]
                    ))
                elif counter == max_lines:
                    console.print('{}{}{}'.format(
                        ''.ljust(callee_column),
                        ''.ljust(role_column),
                        '...'
                    ))
                counter += 1
        except Exception as _:
            pass


def setup_logging(
    module_name='root',
    default_level=logging.DEBUG,
    enable_timing=False,
):
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('markdown_it').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('pdfminer').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('parso.python.diff').disabled = True
    logging.getLogger('parso').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.CRITICAL)
    logging.getLogger('PIL').setLevel(logging.CRITICAL)
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('grpc').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('dateparser').setLevel(logging.CRITICAL)
    logging.getLogger('tesseract').setLevel(logging.CRITICAL)
    logging.getLogger('pytesseract').setLevel(logging.CRITICAL)
    logging.getLogger('tzlocal').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.WARNING)

    logger: Logger = logging.getLogger()

    handlers_to_remove = [handler for handler in logger.handlers if isinstance(handler, logging.StreamHandler)]
    for handler in handlers_to_remove:
        logger.removeHandler(handler)

    if module_name in global_loggers:
        return global_loggers[module_name]

    install(show_locals=False, max_frames=20, suppress=["importlib, site-packages"])
    handler = RichHandler(console=Console(file=sys.stderr))
    handler.setLevel(default_level)

    logger.setLevel(default_level)
    logger.addHandler(handler)

    if enable_timing:
        handlers_to_remove = [h for h in timing.handlers if isinstance(handler, logging.StreamHandler)]
        for h in handlers_to_remove:
            timing.removeHandler(h)

        timing.setLevel(default_level)
        timing.addHandler(handler)
    else:
        timing.setLevel(logging.CRITICAL)

    global_loggers[module_name] = logger
    return logger


def suppress_logging():
    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger('root').setLevel(logging.CRITICAL)
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    logging.getLogger('markdown_it').setLevel(logging.CRITICAL)
    logging.getLogger('numexpr').setLevel(logging.CRITICAL)
    logging.getLogger('rich').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('httpcore').setLevel(logging.CRITICAL)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('anthropic').setLevel(logging.WARNING)
    logging.getLogger('grpc').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def get_timer():
    return timing


def disable_timing(name='timing'):
    timing.setLevel(logging.DEBUG)


def response_writer(callee, message):
    with (open(f"{Container().get('log_directory')}/ast.log", 'a')) as f:
        f.write(f'{str(dt.datetime.now())} {callee}: {message}\n')