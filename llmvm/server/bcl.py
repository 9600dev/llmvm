from __future__ import annotations

import datetime as dt
import io
import numpy as np
from typing import Dict, Optional, Any


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

    def generate_graph_image(self, data: Dict, title: str, x_label: str, y_label: str) -> None:
        """
        Generates a graph image from the given data and returns it as bytes.

        :param data: The data to plot
        :type data: Dict
        :param title: The title of the graph
        :type title: str
        :param x_label: The label for the x-axis
        :type x_label: str
        :param y_label: The label for the y-axis
        :type y_label: str
        :return: None

        Examples:
        generate_graph_image({"x": [1, 2, 3], "y": [4, 5, 6]}, "Title", "X Label", "Y Label")
        """

        data_dict = {}

        if isinstance(data, list):
            list_data = [float(item) for item in data if isinstance(item, (int, float))]
            data_dict['x'] = list(range(len(list_data)))
            data_dict['y'] = list_data
        elif 'dates' in data and 'prices' in data:
            # dates is Timestamp, prices is float
            data_dict['x'] = [timestamp.strftime('%Y-%m-%d') for timestamp in data['dates']]
            data_dict['y'] = data['prices']
        elif 'x' not in data:
            data_dict['x'] = data.keys()
            data_dict['y'] = data.values()
        else:
            data_dict = data

        from matplotlib import pyplot as plt
        plt.figure(figsize=(10.24, 7.68))
        plt.plot(data_dict['x'], data_dict['y'])  # type: ignore
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Create a bytes buffer
        buffer = io.BytesIO()

        # Save the plot to the buffer in PNG format
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')

        # Close the plot to free up memory
        plt.close()

        # Get the contents of the buffer
        image_bytes = buffer.getvalue()

        # Close the buffer
        buffer.close()

        write_client_stream(image_bytes)
