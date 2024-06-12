from __future__ import annotations

import datetime as dt
import numpy as np
from typing import Optional, Any


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

    def sample_list(self, data: list) -> Any:
        """
        Returns a random sample from a list.
        Examples: sample_list([1, 2, 3]), sample_list(["a", "b", "c"])
        """
        return np.random.choice(data)

    def sample_normal(self, mean: float = 0.0, std_dev: float = 1.0) -> float:
        """
        Returns a random sample from a normal distribution with the given mean and standard deviation.
        Examples: sample_normal(0, 1), sample_normal(10, 2)
        """
        return np.random.normal(mean, std_dev)

    def sample_binomial(self, n: int, p: float) -> float:
        """
        Returns a random sample from a binomial distribution with the given number of trials and probability of success.
        Examples: sample_binomial(10, 0.5), sample_binomial(100, 0.1)
        """
        return np.random.binomial(n, p)

    def sample_lognormal(self, mean: float = 0.0, std_dev: float = 1.0) -> float:
        """
        Returns a random sample from a lognormal distribution with the given mean and standard deviation.
        Examples: sample_lognormal(0, 1), sample_lognormal(10, 2)
        """
        return np.random.lognormal(mean, std_dev)