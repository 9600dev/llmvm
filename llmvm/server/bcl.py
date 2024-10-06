from __future__ import annotations

import datetime as dt
import io
import os
from urllib.parse import urlencode
import httpx
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union


from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.server.base_library.source import Source

logging = setup_logging()

class BCL():
    @staticmethod
    def datetime(expr, timezone: Optional[str] = None) -> dt.datetime:
        """
        Returns a datetime object from a string using datetime.strftime().
        Examples: BCL.datetime("2020-01-01"), BCL.datetime("now"), BCL.datetime("-1 days"), BCL.datetime("now", "Australia/Brisbane")
        """
        return Helpers.parse_relative_datetime(str(expr), timezone)

    @staticmethod
    def address_lat_lon(address: str) -> Tuple[float, float]:
        """
        Returns the latitude and longitude of an address as a Tuple of floats.
        Examples: lat, lon = BCL.address_lat_lon("1600 Pennsylvania Avenue NW, Washington, DC")
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
    def get_weather(location: str) -> str:
        """
        Returns the weather forecast for a location as natural language string.
        Examples: weather = BCL.get_weather("New York, NY")
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
    def generate_graph_image(x_y_data_dict: Dict[str, Any], title: str, x_label: str, y_label: str) -> None:
        """
        Generates a graph image from the given x_y_data_dict Dictionary, which has two keys: 'x' and 'y' and a list of int/floats
        and prints it to the client's screen. It returns None.

        Example:
        BCL.generate_graph_image(x_y_data_dict={"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}, title="My Graph Title", x_label="X Label", y_label="Y Label")

        :param x_y_data_dict: The data to plot
        :type data: Dict[str, Any]
        :param title: The title of the graph
        :type title: str
        :param x_label: The label for the x-axis
        :type x_label: str
        :param y_label: The label for the y-axis
        :type y_label: str
        :return: None
        """
        data = x_y_data_dict
        data_dict = {}

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

    @staticmethod
    def __source_paths(source_file_paths: Union[List[str], str]) -> List[str]:
        def find_python_files(path):
            python_files = []
            for root, _, files in os.walk(os.path.expanduser(path)):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            return python_files

        if isinstance(source_file_paths, str):
            expanded_path = os.path.expanduser(source_file_paths)
            if os.path.isdir(expanded_path):
                return find_python_files(expanded_path)
            elif os.path.isfile(expanded_path) and expanded_path.endswith('.py'):
                return [expanded_path]
            else:
                raise ValueError(f"Invalid source file path: {source_file_paths}. Must be a directory or a .py file.")

        elif isinstance(source_file_paths, list):
            all_python_files = []
            for path in source_file_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.isdir(expanded_path):
                    all_python_files.extend(find_python_files(expanded_path))
                elif os.path.isfile(expanded_path) and expanded_path.endswith('.py'):
                    all_python_files.append(expanded_path)
                else:
                    raise ValueError(f"Invalid source file path: {expanded_path}. Must be a directory or a .py file.")
            return all_python_files

        else:
            raise ValueError(f"Invalid source file paths: {source_file_paths}. Must be a list of file paths or a directory path.")

    @staticmethod
    def get_code_structure_summary(source_file_paths: Union[List[str], str]) -> str:
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
    def find_all_references(
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

