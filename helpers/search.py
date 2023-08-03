
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Generator, List

import rich
from serpapi import BingSearch, GoogleSearch

from helpers.logging_helpers import setup_logging

logging = setup_logging()

class Searcher(ABC):
    @abstractmethod
    def search_internet(self, query: str) -> Generator[Dict, None, None]:
        pass

    @abstractmethod
    def search_news(self, query: str) -> Generator[str, None, None]:
        pass


class SerpAPISearcher(Searcher):
    def __init__(
        self,
        api_key: str = os.environ.get('SERPAPI_API_KEY'),  # type: ignore
        country_code: str = 'US',
        location: str = 'United States',
        interval: int = 7,
        page_limit: int = 1,
    ):
        self.api_key = api_key
        self.country_code = country_code
        self.location = location
        self.interval = interval
        self.page_limit = page_limit
        self.search_count = 0

    def search_internet(
        self,
        query: str,
    ) -> Generator[Dict, None, None]:
        logging.debug('SerpAPISearcher.search() type={} query={}'.format(
            str(type),
            query
        ))
        location = 'San Jose, California, United States'

        params = {
            'q': query,
            'location': location,
            'hl': 'en',
            'gl': 'us',
            'api_key': self.api_key,
            'engine': 'google',
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            yield {}

        page_count = 0
        self.search_count += 1

        while 'error' not in results and self.page_limit > page_count:
            page_count += 1
            results = search.get_dict()
            organic_results = results.get('organic_results', [])
            for result in organic_results:
                yield result

    def search_internet_bing(
        self,
        query: str,
    ) -> Generator[Dict, None, None]:
        logging.debug('SerpAPISearcher.search() type={} query={}'.format(str(type), query))

        params = {
            'api_key': self.api_key,
            'q': query,
            'engine': 'bing',
        }

        search = BingSearch(params)
        results = search.get_dict()

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            yield {}

        page_count = 0
        self.search_count += 1

        while 'error' not in results and self.page_limit > page_count:
            page_count += 1
            results = search.get_dict()
            organic_results = results.get('organic_results', [])
            for result in organic_results:
                yield result

    def search_news(
        self,
        query: str,
    ) -> Generator[Dict, None, None]:
        logging.debug('SerpAPISearcher.search_news() query={}'.format(query))
        search_results = []

        params = {
            'api_key': self.api_key,
            'q': query,
            'engine': 'bing_news',
        }

        bing_params = {
            'cc': self.country_code,
            'location': self.location,
            'first': 1,
            'count': 50,
            'qft': 'interval={}'.format(self.interval)
        }
        params = {**params, **bing_params}  # type: ignore
        search = BingSearch(params)

        results = search.get_dict()

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            yield {}

        page_count = 0
        self.search_count += 1

        while 'error' not in results and self.page_limit > page_count:
            page_count += 1
            results = search.get_dict()
            organic_results = results.get('organic_results', [])
            for result in organic_results:
                yield result
