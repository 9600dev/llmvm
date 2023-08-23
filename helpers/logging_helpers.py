import datetime as dt
import logging
import types
from logging import Logger
from typing import Dict

import rich
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

global_loggers: Dict[str, Logger] = {}
debug_log = True
handler = RichHandler()

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

    logger = logging.getLogger()

    handlers_to_remove = [handler for handler in logger.handlers if isinstance(handler, logging.StreamHandler)]
    for handler in handlers_to_remove:
        logger.removeHandler(handler)

    if module_name in global_loggers:
        return global_loggers[module_name]

    install(show_locals=True)
    handler = RichHandler()
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

def console_debug(callee, role, message):
    def split_string_by_width(input_string, width=20):
        splits = [input_string[i:i + width] for i in range(0, len(input_string), width)]
        result = []
        for r in splits:
            if '\n' in r:
                result.extend(r.split('\n'))
            else:
                result.append(r)
        return result

    def split_string_by_width_2(input_string, width=20):
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

    if debug_log:
        if callee.startswith('prompts/'):
            callee = callee.replace('prompts/', '')

        console = Console()
        width, _ = console.size
        callee_column = 20
        role_column = 10
        text_column = width - callee_column - role_column - 4

        # message_lines = message.split('\n')
        message_lines = split_string_by_width_2(message, width=text_column)
        header = True
        counter = 1
        max_lines = 20
        for message in message_lines:
            if header:
                rich.print('[orange]{}[/orange][green]{}[/green][grey]{}[/grey]'.format(
                    callee[0:callee_column - 1].ljust(callee_column)[:callee_column],
                    role.ljust(role_column)[:role_column],
                    message.ljust(text_column)[:text_column]
                ))
                header = False
            elif counter < max_lines or counter >= len(message_lines) - 5:
                rich.print('{}{}{}'.format(
                    ''.ljust(callee_column),
                    ''.ljust(role_column),
                    message.ljust(text_column)[:text_column]
                ))
            elif counter == max_lines:
                rich.print('{}{}{}'.format(
                    ''.ljust(callee_column),
                    ''.ljust(role_column),
                    '...'
                ))
            counter += 1

def response_writer(callee, message):
    with (open('logs/ast.log', 'a')) as f:
        f.write(f'{str(dt.datetime.now())} {callee}: {message}\n')
