
import datetime as dt
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Generator, List

import rich
from serpapi import BingSearch, GoogleSearch

from helpers.helpers import write_client_stream
from helpers.logging_helpers import setup_logging
from helpers.search_hn import SearchHN

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
        api_key: str = os.environ.get('SERPAPI_API_KEY', ''),  # type: ignore
        country_code: str = 'US',
        location: str = 'United States',
        interval: int = 7,
        page_limit: int = 1,
    ):
        if not api_key:
            raise ValueError('SerpAPISearcher requires a valid api_key')

        self.api_key = api_key
        self.country_code = country_code
        self.location = location
        self.interval = interval
        self.page_limit = page_limit
        self.search_count = 0

    def __search_hackernews_helper(
        self,
        query: str,
    ) -> List[Dict]:
        hn = SearchHN()
        # clean up the query
        if 'site:' in query:
            query = query.split('site:')[0].strip()

        return hn.search(query).created_before(dt.datetime.now() - dt.timedelta(days=365)).stories().get()  # type: ignore

    def search_hackernews(
        self,
        query: str,
    ) -> List[Dict]:
        stories = self.__search_hackernews_helper(query)
        results = []
        for story in stories:
            results.append({
                'title': story['title'],
                'url': story['url'],
                'author': story['author'],
                'created_at': story['created_at'],
            })
        return results

    def search_hackernews_comments(
        self,
        query: str,
        token_length: int = 8192,
    ) -> List[Dict]:
        stories = self.__search_hackernews_helper(query)
        result = []
        token_count = 0
        if stories:
            # only get the top result.
            for story in stories:
                comments = story.get_story_comments()  # type: ignore
                for comment in comments:
                    token_count += len(comment.comment_text.split(' '))
                    result.append({
                        'title': stories[0].title,  # type: ignore
                        'url': stories[0].url,  # type: ignore
                        'author': comment.author,
                        'comment_text': comment.comment_text,
                        'created_at': comment.created_at,
                    })
                if token_count > token_length:
                    break
        return result

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

        # {
        #    "position": 1,
        #    "title": "SerpApi - Google Search API",
        #    "link": "https://serpapi.com/",
        #    "snippet": "SerpApi is a real-time API ..."
        # }

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

    def search_yelp(
        self,
        query: str,
        location: str,
    ) -> Dict:
        params = {
            'find_desc': query,
            'find_loc': location,
            'engine': 'yelp',
            'api_key': self.api_key,
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            return {}

        page_count = 0
        counter = 3
        self.search_count += 1

        place_id = ''
        title = ''
        link = ''
        neighborhood = ''
        snippet = ''

        while 'error' not in results and counter > page_count:
            page_count += 1
            results = search.get_dict()
            organic_results = results.get('organic_results', [])
            for result in organic_results:
                if result.get('place_ids'):
                    place_id = result.get('place_ids')[0]
                    title = result.get('title')
                    link = result.get('link')
                    neighborhood = result.get('neighborhoods')
                    snippet = result.get('snippet')
                    break

        reviews_text = []
        page_count = 0

        if place_id:
            # get the reviews
            params = {
                'engine': 'yelp_reviews',
                'api_key': self.api_key,
                'place_id': place_id,
            }

            search = GoogleSearch(params)
            results = search.get_dict()

            while 'error' not in results and counter > page_count:
                results = search.get_dict()
                page_count += 1
                reviews = results.get('reviews', [])
                for review in reviews:
                    if 'comment' in review and 'text' in review['comment']:
                        name = review['user']['name']
                        address = review['user']['address']
                        reviews_text.append(f"{name} from {address} had this review: {review['comment']['text']}")

            return {
                'title': title,
                'link': link,
                'neighborhood': neighborhood,
                'snippet': snippet,
                'reviews': '\n\n'.join(reviews_text)
            }

        return {}

    def search_news(
        self,
        query: str,
    ) -> Generator[Dict, None, None]:
        logging.debug('SerpAPISearcher.search_news() query={}'.format(query))

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
