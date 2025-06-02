from __future__ import annotations
import asyncio
import subprocess
from bs4 import BeautifulSoup
from typing import List, cast
import datetime as dt
import io
import os
import httpx
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import io
from functools import wraps
from contextlib import contextmanager

from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import FunctionCallMeta, ImageContent, StreamNode, TextContent
from llmvm.server.base_library.source import Source

logging = setup_logging()


class BCL():
    @staticmethod
    def datetime(expr, timezone: Optional[str] = None) -> dt.datetime:
        """
        Returns a datetime object from a string using datetime.strftime().
        Examples: BCL.datetime("2020-01-01"), BCL.datetime("now"), BCL.datetime("-1 days"), BCL.datetime("now", "Australia/Brisbane")

        :param expr: The string to parse as a datetime.
        :type expr: str
        :param timezone: The timezone to use for the datetime. If not provided, the current timezone is used.
        :type timezone: Optional[str]
        :return: The parsed datetime object.
        :rtype: datetime.datetime
        """
        return Helpers.parse_relative_datetime(str(expr), timezone)

    @staticmethod
    def address_lat_lon(address: str) -> Tuple[float, float]:
        """
        Returns the latitude and longitude of an address as a Tuple of floats.
        Examples: lat, lon = BCL.address_lat_lon(address="1600 Pennsylvania Avenue NW, Washington, DC")

        :param address: The address to get the latitude and longitude for.
        :type address: str
        :return: The latitude and longitude of the address as a tuple of floats.
        :rtype: Tuple[float, float]
        """
        logging.debug(f"BCL.address_lat_lon() address: {address}")
        address = address.replace(" ", "+")
        url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&addressdetails=1"
        json_data = httpx.get(url).json()
        try:
            lat = json_data[0]["lat"]
            lon = json_data[0]["lon"]
            return float(lat), float(lon)
        except Exception as ex:
            return (0.0, 0.0)

    @staticmethod
    def get_central_bank_rates() -> str:
        """
        Returns the most popular central bank interest rates as a natural language string. Dates are in American format.
        Example: rates = BCL.get_central_bank_rates()
        result(rates)
            American Central Bank: 5.00 % as of 09-19-2024
            ...

        :return: central bank interest rates as a string
        :rtype: str
        """
        html_content = httpx.get("https://www.global-rates.com/en/interest-rates/central-banks/").content
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the table with central bank data
        table = soup.find('table', class_='table')
        if not table:
            return 'The API errored out as it did not return any data.'

        results: List[Dict] = []

        # Process each row in the table body
        for row in table.find('tbody').find_all('tr'):  # type: ignore
            cells = row.find_all('td')  # type: ignore
            if len(cells) >= 6:  # Ensure we have enough cells
                name = cells[0].find('a').text.strip()  # type: ignore
                rate = cells[2].div.text.strip().replace(' %', '')  # type: ignore
                date = cells[5].div.text.strip()  # type: ignore

                try:
                    rate_float = float(rate)
                    results.append({
                        'name': name,
                        'current_interest_rate': rate_float,
                        'change_date': date,
                    })
                except ValueError:
                    # Skip if we can't parse the rate as a float
                    continue

        if results:
            str_result = f"Central bank interest rates:\n"
            for result in results:
                str_result += f"{result['name']}: {result['current_interest_rate']} %, as of {result['change_date']}\n"
            return str_result
        else:
            return 'The API errored out as it did not return any data.'

    @staticmethod
    def get_currency_rates(currency_code: str) -> str:
        """
        Returns the most popular currency rates for a given currency code as a natural language string.
        Example:
        rates = BCL.get_currency_rates(currency_code="AUD")
        result(rates)

        result:
        ""AUD":1,"AED":2.340945,"AFN":46.158158,"ALL":55.758944,"AMD":248.942953,"ANG":1.140992, ..."

        :param currency_code: The currency code to get the rates for.
        :type currency_code: str
        :return: currency rates as a string
        :rtype: str
        """
        result = httpx.get(f"https://open.er-api.com/v6/latest/{currency_code}").json()
        str_result = f"Currency rates for {currency_code}:\n"
        if 'rates' in result:
            str_result += str(result['rates'])
        else:
            str_result += str(result)
        return str_result

    @staticmethod
    def get_gold_silver_price_in_usd() -> str:
        """
        Returns the current gold and silver spot prices in USD as a natural language string.
        Example: prices = BCL.get_gold_silver_price_in_usd()
        result(prices)

        :return: gold and silver prices as a string
        :rtype: str
        """
        result = asyncio.run(Helpers.download("https://data-asg.goldprice.org/dbXRates/USD"))
        str_result = f"Gold price in USD:\n"
        str_result += str(result)
        return str_result

    @staticmethod
    def get_bitcoin_prices_in_usd() -> str:
        """
        Returns the most popular bitcoin rates as a natural language string. Try and use at least 3 decimal places.
        Example: rates = BCL.get_bitcoin_rates()
        result(rates)

        :return: bitcoin prices as a string
        :rtype: str
        """
        result = httpx.get("https://api.binance.com/api/v1/ticker/price?symbol=BTCUSDT").json()
        str_result = f"Bitcoin rates:\n"
        str_result += str(result)
        return str_result

    @staticmethod
    def get_tvshow_ratings_and_details(tvshow_name: str) -> str:
        """
        Returns the ratings and details for a given TV show as a natural language string.
        Examples: ratings = BCL.get_tvshow_ratings_and_details(tvshow_name="Breaking Bad")
        result(ratings)
        """
        tvshow_name = tvshow_name.replace(' ', '+')
        result = httpx.get(f"https://api.tvmaze.com/search/shows?q={tvshow_name}").json()
        str_result = f"TV show ratings and details for {tvshow_name}:\n"
        str_result += str(result)
        return str_result

    @staticmethod
    def get_weather(location: str) -> str:
        """
        Returns the weather forecast for a location as natural language string.
        Examples: weather = BCL.get_weather(location="New York, NY")
        """
        logging.debug(f"BCL.get_weather() location: {location}")
        lat, lon = BCL.address_lat_lon(location)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "is_day", "precipitation", "rain", "showers", "snowfall", "cloud_cover", "wind_speed_10m", "wind_direction_10m"],
            "forecast_days": 1
        }
        result = httpx.post(url, data=params).json()
        current_weather = result["current"]
        str_result = f"Weather for {location}:\n"
        str_result += f"Temperature: {current_weather['temperature_2m']} °C\n"
        str_result += f"Relative Humidity: {current_weather['relative_humidity_2m']} %\n"
        str_result += f"Is Day: {bool(current_weather['is_day'])}\n"
        str_result += f"Precipitation: {current_weather['precipitation']} mm\n"
        str_result += f"Rain: {current_weather['rain']} mm\n"
        str_result += f"Showers: {current_weather['showers']} mm\n"
        str_result += f"Snowfall: {current_weather['snowfall']} mm\n"
        str_result += f"Cloud Cover: {current_weather['cloud_cover']} %\n"
        str_result += f"Wind Speed: {current_weather['wind_speed_10m']} m/s\n"
        str_result += f"Wind Direction: {current_weather['wind_direction_10m']} °\n"
        return str_result

    @staticmethod
    def sample_list(data: list) -> Any:
        """
        Returns a random sample from a list.
        Examples: BCL.sample_list([1, 2, 3]), BCL.sample_list(["a", "b", "c"])
        """
        return np.random.choice(data)

    @staticmethod
    def sample_normal(mean: float = 0.0, std_dev: float = 1.0) -> float:
        """
        Returns a random sample from a normal distribution with the given mean and standard deviation.
        Examples: BCL.sample_normal(0, 1), BCL.sample_normal(10, 2)
        """
        return np.random.normal(mean, std_dev)

    @staticmethod
    def sample_binomial(n: int, p: float) -> float:
        """
        Returns a random sample from a binomial distribution with the given number of trials and probability of success.
        Examples: BCL.sample_binomial(10, 0.5), BCL.sample_binomial(100, 0.1)
        """
        return np.random.binomial(n, p)

    @staticmethod
    def sample_lognormal(mean: float = 0.0, std_dev: float = 1.0) -> float:
        """
        Returns a random sample from a lognormal distribution with the given mean and standard deviation.
        Examples: BCL.sample_lognormal(0, 1), BCL.sample_lognormal(10, 2)
        """
        return np.random.lognormal(mean, std_dev)

    @staticmethod
    def matplotlib_to_image(func=None, figsize=(28.0, 18.0), dpi=130):
        """
        A context manager that converts any matplotlib figure to an ImageContent object which will be sent to the user.
        You must use this method if you want to use matplotlib in a helper function.
        figsize's of (28.0, 18.0) or larger, and dpi of 130 or greater is required.
        use fontsize > 24 for labels, titles and so on.
        the context manager argument (like 'fig') is a matplotlib figure object,
        you do not need to call plt.figure() to create it.

        Example:
        with BCL.matplotlib_to_image(figsize=(28.0, 18.0), dpi=130) as fig:
            plt.plot([1, 2, 3], [4, 5, 6])
            plt.title("My Plot")

        :param func: The function to decorate (optional)
        :param figsize: Figure size in inches (width, height)
        :param dpi: Resolution in dots per inch
        :returns: a matplotlib Figure object
        """
        # Set non-interactive backend before importing pyplot
        import matplotlib
        matplotlib.use('Agg')  # This prevents any window from appearing

        from matplotlib import pyplot as plt
        import io
        from functools import wraps

        class MatplotlibImageContext:
            def __init__(self, figsize, dpi):
                self.figsize = figsize
                self.dpi = dpi
                self.image_bytes: bytes | None = None

            def __enter__(self):
                plt.figure(figsize=self.figsize)
                return plt.gcf()

            def __exit__(self, exc_type, exc_val, exc_tb):
                fig = plt.gcf()
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)
                self.image_bytes = buffer.getvalue()
                buffer.close()
                write_client_stream(self.image_bytes)
                return False

            def get_image_content(self) -> ImageContent:
                return ImageContent(cast(bytes, self.image_bytes))
        return MatplotlibImageContext(figsize, dpi)

    @staticmethod
    def generate_graph_image(x_y_data_dict: Dict[str, Any], title: str, x_label: str, y_label: str) -> ImageContent:
        """
        Generates a graph image from the given x_y_data_dict Dictionary, which has two keys: 'x' and 'y' and a list of int/floats
        and prints it to the client's screen. It returns None.

        Example:
        image_content = BCL.generate_graph_image(x_y_data_dict={"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}, title="My Graph Title", x_label="X Label", y_label="Y Label")

        :param x_y_data_dict: The data to plot
        :type x_y_data_dict: Dict[str, Any]
        :param title: The title of the graph
        :type title: str
        :param x_label: The label for the x-axis
        :type x_label: str
        :param y_label: The label for the y-axis
        :type y_label: str
        :return: ImageContent which can be observed by both the client and the LLM
        """
        data = x_y_data_dict
        data_dict = {}

        if isinstance(data, FunctionCallMeta):
            data: dict = cast(dict, data.result())

        if isinstance(data, list):
            list_data = [float(item) for item in data if isinstance(item, (int, float))]
            data_dict['x'] = list(range(len(list_data)))
            data_dict['y'] = list_data
        elif isinstance(data, dict) and 'dates' in data and 'prices' in data:
            # dates is Timestamp, prices is float
            data_dict['x'] = [timestamp.strftime('%Y-%m-%d') for timestamp in data['dates']]
            data_dict['y'] = data['prices']
        elif 'x' not in data:
            data_dict['x'] = data.keys()
            data_dict['y'] = data.values()
        else:
            data_dict = data

        from matplotlib import pyplot as plt
        plt.figure(figsize=(28.0, 18.0))
        plt.plot(data_dict['x'], data_dict['y'])  # type: ignore
        plt.title(title, fontsize=28)
        plt.xlabel(x_label, fontsize=24)
        plt.ylabel(y_label, fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Create a bytes buffer
        buffer = io.BytesIO()

        # Save the plot to the buffer in PNG format
        plt.savefig(buffer, format='png', dpi=130, bbox_inches='tight')

        # Close the plot to free up memory
        plt.close()

        # Get the contents of the buffer
        image_bytes = buffer.getvalue()

        # Close the buffer
        buffer.close()

        write_client_stream(StreamNode(image_bytes, type='bytes'))
        return ImageContent(image_bytes)

    @staticmethod
    def find(
        pattern: str,
        paths: Union[str, List[str]] = ".",
        file_patterns: Optional[Union[str, List[str]]] = None,
        ignore_case: bool = False,
        word_regexp: bool = False,
        max_depth: Optional[int] = None,
        max_results: Optional[int] = 10,
    ) -> list[dict[str, Any]]:
        """
        Execute ripgrep search and return results as a dictionary that can be converted to JSON.

        :param pattern: The search pattern (regular expression)
        :type pattern: str
        :param paths: Directory path(s) to search in
        :type paths: str or list
        :param file_patterns: File pattern(s) to include (e.g., "*.py"). Use a glob if possible.
        :type file_patterns: str or list, optional
        :param ignore_case: Whether to perform case-insensitive search
        :type ignore_case: bool
        :param word_regexp: Whether to match the pattern as a whole word
        :type word_regexp: bool
        :param max_depth: Maximum depth to recursively search
        :type max_depth: int, optional
        :param max_results: Maximum number of results to return
        :type max_results: int, optional

        :returns: dict[str, Any] which can be converted to json via json.dumps():
        [
            {
                "file": "/path/to/project/src/main.py",
                "line_number": 42,
                "content": "def process_data(input_file, output_format='json'):",
                "submatches": [
                    {
                        "match": "process_data",
                        "start": 4,
                        "end": 16
                    }
                ]
            },
            ...
        ]
        """
        # Initialize ripgrep command
        cmd = ["rg"]

        # Add basic arguments
        if ignore_case:
            cmd.append("-i")
        if word_regexp:
            cmd.append("-w")

        # Add optional depth limit
        if max_depth is not None:
            cmd.extend(["--max-depth", str(max_depth)])

        # Add max results limit
        if max_results is not None:
            cmd.extend(["--max-count", str(max_results)])

        # Always include line numbers
        cmd.append("--line-number")

        # Handle file patterns correctly
        if file_patterns:
            if isinstance(file_patterns, str):
                if file_patterns.startswith('.'): file_patterns = "*" + file_patterns
                cmd.extend(["--glob", file_patterns])
            else:
                for glob_pattern in file_patterns:
                    if glob_pattern.startswith('.'): glob_pattern = "*" + glob_pattern
                    cmd.extend(["--glob", glob_pattern])

        cmd.append("--json")

        # Fix for empty pattern - use "." to match any character if pattern is empty
        search_pattern = pattern if pattern else "."
        cmd.append(search_pattern)

        # Add the search paths
        if isinstance(paths, str):
            cmd.append(os.path.expanduser(paths))
        else:
            for path in paths:
                cmd.append(os.path.expanduser(path))

        try:
            # Execute ripgrep command
            logging.debug('BCL.find() cmd: {}'.format(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0 and result.returncode != 1:  # 1 means no matches, which is valid
                if "not found" in result.stderr:
                    raise RuntimeError("ripgrep (rg) is not installed or not in PATH")
                else:
                    raise RuntimeError(f"ripgrep error: {result.stderr} called via cmd: {cmd}")

            # Process output based on requested format
            output = []
            for line in result.stdout.strip().split('\n'):
                if line:  # Skip empty lines
                    try:
                        match_data = json.loads(line)
                        output.append({
                            'file': match_data.get('data', {}).get('path', {}).get('text', ''),
                            'line_number': match_data.get('data', {}).get('line_number', 0),
                            'content': match_data.get('data', {}).get('lines', {}).get('text', ''),
                            'submatches': [
                                {
                                    'match': m.get('match', {}).get('text', ''),
                                    'start': m.get('start', 0),
                                    'end': m.get('end', 0)
                                }
                                for m in match_data.get('data', {}).get('submatches', [])
                            ]
                        })
                    except json.JSONDecodeError:
                        continue
            return output
        except FileNotFoundError:
            raise RuntimeError("ripgrep (rg) is not installed or not in PATH")

    @staticmethod
    def search_and_replace(text: str, search: str, replace: str) -> str:
        """
        Performs a simple string replacement on the given text.

        :param text: The input text
        :type text: str
        :param search: The search pattern
        :type search: str
        :param replace: The replacement string
        :type replace: str
        :return: The modified text
        :rtype: str
        """
        if isinstance(text, TextContent):
            text = text.get_str()

        return text.replace(search, replace)

    @staticmethod
    def __source_paths(source_file_paths: Union[List[str], str]) -> List[str]:
        def find_python_files(path):
            python_files = []
            for root, _, files in os.walk(os.path.expanduser(path)):
                for file in files:
                    if file.endswith('.py') and '.venv' not in root:
                        python_files.append(os.path.join(root, file))
            return python_files

        if isinstance(source_file_paths, str):
            expanded_path = os.path.expanduser(source_file_paths)
            if os.path.isdir(expanded_path):
                return find_python_files(expanded_path)
            elif os.path.isfile(expanded_path) and expanded_path.endswith('.py') and '.venv' not in expanded_path:
                return [expanded_path]
            else:
                raise ValueError(f"Invalid source file path: {source_file_paths}. Must be a directory or a .py file.")

        elif isinstance(source_file_paths, list):
            all_python_files = []
            for path in source_file_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.isdir(expanded_path):
                    all_python_files.extend(find_python_files(expanded_path))
                elif os.path.isfile(expanded_path) and expanded_path.endswith('.py') and '.venv' not in expanded_path:
                    all_python_files.append(expanded_path)
                elif expanded_path.endswith('.py') and not os.path.isfile(expanded_path):
                    raise ValueError(f"Invalid source file path: {expanded_path}. File does not exist.")
                else:
                    raise ValueError(f"Invalid source file path: {expanded_path}. Must be a directory or a .py file.")
            return all_python_files

        else:
            raise ValueError(f"Invalid source file paths: {source_file_paths}. Must be a list of file paths or a directory path.")

    @staticmethod
    def get_source_code_structure_summary(source_file_paths: Union[List[str], str]) -> str:
        """
        Gets all class names, method names, and docstrings for each of the source code files listed in source_files.
        This method does not return any source code, only class names, method names and their docstrings.

        :param source_file_paths: List of file paths to the source code files, or the source code directory root.
        :type source_file_paths: List[str] | str
        :return: A string containing the class names, method names, and docstrings for each of the source code files.
        """
        source_file_paths = BCL.__source_paths(source_file_paths)

        logging.debug(f"Getting code structure summary for {len(source_file_paths)} files.")
        structure = ''
        for source_file in source_file_paths:
            source = Source(os.path.expanduser(source_file))

            structure += f'File Path: {source_file}\n'
            for class_def in source.get_classes():
                structure += f'{class_def.class_definition()}\n'
                if class_def.docstring:
                    structure += f'    """{class_def.docstring}"""\n'
                    structure += '\n'
                for method_def in source.get_methods(class_def.name):
                    structure += f'    {method_def.method_definition()}\n'
            structure += '\n\n'
        return structure

    @staticmethod
    def get_source_code(source_file_path) -> str:
        """
        Gets the source code from the file at the given file path.
        :param source_file_path: The path to the file.
        :return: The source code from the file.
        """
        logging.debug(f"Getting source code for file at {source_file_path}")
        if os.path.exists(os.path.expanduser(source_file_path)):
            with open(os.path.expanduser(source_file_path), 'r') as file:
                return file.read()
        else:
            raise ValueError(f"File {source_file_path} not found")

    @staticmethod
    def find_all_references_to_method(
            source_file_paths: List[str],
            method_name: str
        ) -> str:
        """
        Find's all references to the given method in the source code files listed in source_files.
        :param source_file_paths: List of file paths to the source code files.
        :param method_name: The name of the method to find references to.
        :return: A string containing the references to the given method.
        """
        source_file_paths = BCL.__source_paths(source_file_paths)

        # open each source file, and grep for the method name
        references = []
        sources = [Source(os.path.expanduser(source_file)) for source_file in source_file_paths]

        for source in sources:
            references.extend(Source.get_references(source.tree, method_name))

        if references:
            return '\n'.join([str(r) for r in references])
        else:
            return f"No references found for method {method_name}"
