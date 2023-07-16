import logging
from logging import Logger
from typing import Dict

from rich.logging import RichHandler
from rich.traceback import install

global_loggers: Dict[str, Logger] = {}

def setup_logging(
    module_name='root',
    default_level=logging.DEBUG,
):
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('markdown_it').setLevel(logging.WARNING)
    logging.getLogger('markdown_it').setLevel(logging.WARNING)

    if module_name in global_loggers:
        return global_loggers[module_name]

    install(show_locals=True)
    handler = RichHandler()
    handler.setLevel(default_level)

    logger = logging.getLogger()
    logger.setLevel(default_level)
    logger.addHandler(handler)
    global_loggers[module_name] = logger
    return logger
