import datetime as dt
import inspect
import logging
import os
import sys
import types
from logging import Logger
from typing import Any, Dict, Protocol

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

from container import Container

global_loggers: Dict[str, Logger] = {}
handler = RichHandler()
if not os.path.exists(Container().get('log_directory')):
    os.makedirs(Container().get('log_directory'))

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


def setup_logging(
    module_name='root',
    default_level=logging.DEBUG,
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
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    logger: Logger = logging.getLogger()

    handlers_to_remove = [handler for handler in logger.handlers if isinstance(handler, logging.StreamHandler)]
    for handler in handlers_to_remove:
        logger.removeHandler(handler)

    if module_name in global_loggers:
        return global_loggers[module_name]

    install(show_locals=True)
    handler = RichHandler(console=Console(file=sys.stderr))
    handler.setLevel(default_level)

    logger.setLevel(default_level)
    logger.addHandler(handler)

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


def response_writer(callee, message):
    with (open(f"{Container().get('log_directory')}/ast.log", 'a')) as f:
        f.write(f'{str(dt.datetime.now())} {callee}: {message}\n')
