from __future__ import annotations

import ast
import asyncio
import datetime as dt
import re
import time
from datetime import timedelta
from typing import Any, Callable, Dict, Generator, List, Optional, cast
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import astunparse
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Assistant, Content, FunctionCall, LLMCall,
                                  Message, System, TokenCompressionMethod,
                                  User)
from llmvm.server.starlark_runtime import StarlarkRuntime
from llmvm.server.tools.firefox import FirefoxHelpers
from llmvm.server.tools.pdf import PdfHelpers
from llmvm.server.tools.search import SerpAPISearcher
from llmvm.server.tools.webhelpers import WebHelpers
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()

class BCL():
    def datetime(self, expr, timezone: Optional[str] = None) -> dt.datetime:
        """
        Returns a datetime object from a string using datetime.strftime().
        Examples: datetime("2020-01-01"), datetime("now"), datetime("-1 days"), datetime("now", "Australia/Brisbane")
        """
        return Helpers.parse_relative_datetime(str(expr), timezone)
