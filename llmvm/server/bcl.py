from __future__ import annotations

import datetime as dt
from typing import Optional

from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging

logging = setup_logging()

class BCL():
    def datetime(self, expr, timezone: Optional[str] = None) -> dt.datetime:
        """
        Returns a datetime object from a string using datetime.strftime().
        Examples: datetime("2020-01-01"), datetime("now"), datetime("-1 days"), datetime("now", "Australia/Brisbane")
        """
        return Helpers.parse_relative_datetime(str(expr), timezone)
