
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import rich
from serpapi import BingSearch, GoogleSearch

from helpers.logging_helpers import setup_logging

logging = setup_logging()

class Searcher(ABC):
    @abstractmethod
    def search_internet(self, query: str):
        pass

    @abstractmethod
    def search_news(self, query: str):
        pass


class SerpAPISearcher(Searcher):
    def __init__(
        self,
        api_key: str = os.environ.get('SERPAPI_API_KEY'),  # type: ignore
        country_code: str = 'US',
        location: str = 'United States',
        interval: int = 7,
        page_limit: int = 1,
        link_limit: int = 4,
    ):
        self.api_key = api_key
        self.country_code = country_code
        self.location = location
        self.interval = interval
        self.page_limit = page_limit
        self.search_count = 0
        self.link_limit = link_limit

    def search_internet(
        self,
        query: str,
    ) -> List[Dict]:
        logging.debug('SerpAPISearcher.search() type={} link_limit={} query={}'.format(str(type), str(self.link_limit), query))
        search_results = []
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
            return []

        page_count = 0
        self.search_count += 1

        while 'error' not in results and self.page_limit > page_count:
            page_count += 1
            results = search.get_dict()
            search_results.extend(results.get('organic_results', []))

        return search_results[:self.link_limit]

    def search_internet_bing(
        self,
        query: str,
    ) -> List[Dict]:
        logging.debug('SerpAPISearcher.search() type={} link_limit={} query={}'.format(str(type), str(self.link_limit), query))
        search_results = []

        params = {
            'api_key': self.api_key,
            'q': query,
            'engine': 'bing',
        }

        search = BingSearch(params)
        results = search.get_dict()

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            return []

        page_count = 0
        self.search_count += 1

        while 'error' not in results and self.page_limit > page_count:
            page_count += 1
            results = search.get_dict()
            search_results.extend(results.get('organic_results', []))

        return search_results[:self.link_limit]

    def search_news(
        self,
        query: str,
    ) -> List[Dict]:
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
            return []

        page_count = 0
        self.search_count += 1

        while 'error' not in results and self.page_limit > page_count:
            page_count += 1
            results = search.get_dict()
            search_results.extend(results.get('organic_results', []))

        return search_results[:self.link_limit]
