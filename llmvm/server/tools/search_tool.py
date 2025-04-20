
from typing import Iterator, cast

from llmvm.common.container import Container
from llmvm.common.helpers import write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import HackerNewsResult, MarkdownContent, SearchResult, TextContent, YelpResult
from llmvm.server.tools.search import SerpAPISearcher
from llmvm.server.tools.webhelpers import WebHelpers

logging = setup_logging()


class Search():
    def __init__(
        self,
    ):
        pass

    @staticmethod
    def google_search(
        expr: str,
        total_links_to_return: int = 2,
        exclude_results: list[SearchResult] = [],
    ) -> list[SearchResult]:
        """
        Searches the internet using the Google search engine.

        returns a list of SearchResult objects:

        @dataclass
        class SearchResult(TextContent):
            url: str = ""
            title: str = ""
            snippet: str = ""
            engine: str = ""

        :param expr: The search query
        :type expr: str
        :param total_links_to_return: The number of search results to return, max is 5, default is 2
        :type total_links_to_return: int
        :param exclude_results: A list of SearchResult objects to exclude from the search results
        :type exclude_results: list[SearchResult]
        :return: A list of SearchResult objects
        """
        logging.debug(f'PythonRuntime.google_search({str(expr)})')

        def python_library_google_search(query: str):
            from googlesearch import search as google_search, SearchResult as GoogleSearchResult
            write_client_stream(
                TextContent("$SERPAPI_API_KEY not found, using googlesearch-python to talk to Google Search.\n")
            )

            results = cast(Iterator[GoogleSearchResult], google_search(query, advanced=True))

            return_results: list[SearchResult] = []
            for result in results:
                return_results.append(SearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.description,
                    engine='Google',
                ))

            return return_results[0:total_links_to_return]


        if not Container().get_config_variable('SERPAPI_API_KEY'):
            return python_library_google_search(query=expr)
        else:
            results = SerpAPISearcher().search_internet(expr)
            search_results = []
            counter = 0
            for result in results:
                search_results.append(SearchResult(url=result['link'], title=result['title'], snippet=result['snippet'], engine='Google'))
                counter += 1
                if counter >= total_links_to_return:
                    break

            # try the python library
            if total_links_to_return > 0 and len(search_results) == 0:
                return python_library_google_search(query=expr)

            return search_results

    @staticmethod
    def google_patent_search(
        query: str,
        total_links_to_return: int = 3,
        exclude_results: list[SearchResult] = [],
    ) -> list[SearchResult]:
        """
        Searches the internet using the Google Patent search engine.

        returns a list of SearchResult objects:

        @dataclass
        class SearchResult(TextContent):
            url: str = ""
            title: str = ""
            snippet: str = ""
            engine: str = ""

        :param expr: The search query
        :type expr: str
        :param total_links_to_return: The number of search results to return, max is 5, default is 2
        :type total_links_to_return: int
        :param exclude_results: A list of SearchResult objects to exclude from the search results
        :type exclude_results: list[SearchResult]
        :return: A list of SearchResult objects
        """
        logging.debug(f'PythonRuntime.google_patent_search({str(query)})')
        search_results: list[SearchResult] = []
        results = SerpAPISearcher().search_research(query)
        counter = 0
        for result in results:
            search_results.append(SearchResult(url=result['link'], title=result['title'], snippet=result['snippet'], engine='GooglePatent'))
            counter += 1
            if counter >= total_links_to_return:
                break
        return search_results

    @staticmethod
    def bluesky_search(
        query: str,
    ) -> MarkdownContent:
        """
        Searches the internet using the BlueSky search engine.

        :param query: The search query
        :type query: str
        :return: A MarkdownContent object containing the search results
        """
        logging.debug(f'PythonRuntime.bluesky_search({str(query)})')
        result: MarkdownContent = cast(MarkdownContent, WebHelpers.get_url(f'https://bsky.app/search?q={query.replace(" ", "+")}'))
        return result

    @staticmethod
    def yelp_search(
        query: str,
        location: str,
        total_links_to_return: int = 5,
        total_reviews_to_return: int = 20,
    ) -> list[YelpResult]:
        """
        Searches all of Yelp (www.yelp.com) for the query at the specified location.

        Returns a YelpResult object:

        @dataclass
        class YelpResult(TextContent):
            title: str = ''
            link: str = ''
            neighborhood: str = ''
            snippet: str = ''
            reviews: str = ''

        :param query: The search query, e.g. "best burgers in SF"
        :type query: str
        :param location: The location to search in, e.g. "San Francisco, CA"
        :type location: str
        :param total_links_to_return: The number of search results to return, default = 5
        :type total_links_to_return: int
        :param total_reviews_to_return: The number of reviews to return, default = 20
        :type total_reviews_to_return: int
        :return: A list of YelpResult objects
        """
        logging.debug(f'PythonRuntime.yelp_search({str(query)})')

        results = SerpAPISearcher().search_yelp(
            query,
            location,
            total_links_to_return=total_links_to_return,
            total_reviews_to_return=total_reviews_to_return
        )
        return_results: list[YelpResult] = []
        for result in results:
            return_results.append(YelpResult(
                title=result['title'],
                link=result['link'],
                neighborhood=result['neighborhood'],
                snippet=result['snippet'],
                reviews=result['reviews'],
            ))
        return return_results

    @staticmethod
    def hackernews_search(
        query: str,
    ) -> list[HackerNewsResult]:
        """
        Searches Hacker News for the query.

        Returns a HackerNewsResult object:

        @dataclass
        class HackerNewsResult(TextContent):
            title: str = ""
            url: str = ""
            author: str = ""
            comment_text: str = ""
            created_at: str = ""

        :param query: The search query, e.g. "AMD graphics card"
        :type query: str
        :return: A list of HackerNewsResult objects
        """
        logging.debug(f'PythonRuntime.hackernews_search({str(query)})')
        results = SerpAPISearcher().search_hackernews_comments(query)
        return_results: list[HackerNewsResult] = []
        for result in results:
            return_results.append(HackerNewsResult(
                title=result['title'],
                url=result['url'],
                author=result['author'],
                comment_text=result['comment_text'],
                created_at=result['created_at'],
            ))
        return return_results