
import datetime as dt
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Generator, List

import rich
import serpapi

from llmvm.common.helpers import write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.server.tools.search_hn import SearchHN

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
        self.api_key = api_key
        self.country_code = country_code
        self.location = location
        self.interval = interval
        self.page_limit = page_limit
        self.client = serpapi.Client(api_key=self.api_key)

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
            'engine': 'google',
        }

        results = self.client.search(params)

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            yield {}

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
        }

        results = self.client.search(params)

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            return {}

        place_id = ''
        title = ''
        link = ''
        neighborhood = ''
        snippet = ''

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

        if place_id:
            # get the reviews
            params = {
                'engine': 'yelp_reviews',
                'place_id': place_id,
            }

            results = self.client.search(params)

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

    def search_research(
        self,
        query: str,
    ) -> Generator[Dict, None, None]:
        logging.debug('SerpAPISearcher.search_research() query={}'.format(query))

        params = {
            'q': query,
            'engine': 'google_scholar',
        }

        google_params = {
            'hl': 'en',
        }
        params = {**params, **google_params}  # type: ignore
        results = self.client.search(params)

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            yield {}

        organic_results = results.get('organic_results', [])
        for result in organic_results:
            yield result

    def search_news(
        self,
        query: str,
    ) -> Generator[Dict, None, None]:
        logging.debug('SerpAPISearcher.search_news() query={}'.format(query))

        params = {
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
        results = self.client.search(params)

        if results.get('error'):
            rich.print_json(json.dumps(results, default=str))
            yield {}

        organic_results = results.get('organic_results', [])
        for result in organic_results:
            yield result
