import re
from typing import Any, Callable, Dict, List

from googlesearch import search as google_search

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Content, LLMCall, Message,
                                  TokenCompressionMethod, User, bcl)
from llmvm.server.starlark_execution_controller import ExecutionController
from llmvm.server.tools.search import SerpAPISearcher
from llmvm.server.tools.webhelpers import WebHelpers
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()

class Searcher():
    def __init__(
        self,
        expr,
        controller: ExecutionController,
        original_code: str,
        original_query: str,
        vector_search: VectorSearch,
        total_links_to_return: int = 3,
    ):
        self.query = expr
        self.original_code = original_code
        self.original_query = original_query
        self.controller = controller
        self.query_expansion = 2

        self.parser = WebHelpers.get_url
        self.ordered_snippets: List = []
        self.index = 0
        self._result = None
        self.total_links_to_return: int = total_links_to_return
        self.vector_search = vector_search

        if self.query.startswith('"') and self.query.endswith('"'):
            self.query = self.query[1:-1]

    def search_hook(self, url: str, query: str):
        write_client_stream(
            Content("$SERPAPI_API_KEY not found, using googlesearch-python to talk to Google Search.\n")
        )

        results = list(google_search(query, advanced=True))
        return_results: List[Dict] = []
        for result in results:
            return_results.append({
                'title': result.title,
                'link': result.url,
                'snippet': result.description,
            })

        return return_results

    def search_google_hook(self, query: str):
        if not Container().get_config_variable('SERPAPI_API_KEY'):
            return self.search_hook('https://www.google.com/search?q=', query)
        else:
            return SerpAPISearcher().search_internet(query)

    def search_news_hook(self, query: str):
        if not Container().get_config_variable('SERPAPI_API_KEY'):
            return self.search_hook('https://news.google.com/search?q=', query)
        else:
            return SerpAPISearcher().search_news(query)

    def search_research_hook(self, query: str):
        if not Container().get_config_variable('SERPAPI_API_KEY'):
            return self.search_hook('https://www.google.com/search?q=', query)
        else:
            # likely want more thorough answers so we'll return more results
            self.total_links_to_return = self.total_links_to_return * 2
            return SerpAPISearcher().search_research(query)

    def search(
        self,
        titles_seen: List[str] = [],
    ) -> List[Content]:
        # todo: we should probably return the Search instance, so we can futz with it later on.
        query_expander = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='search_expander.prompt',
                    template={
                        'query': self.query,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='search_expander.prompt',
            ),
            query=self.query,
            original_query=self.original_query,
        )

        queries = []

        # double shot try
        try:
            queries = eval(str(query_expander.message))[:self.query_expansion]
        except SyntaxError as ex:
            logging.debug(f"search() query expansion parsing SyntaxError: {ex}")
            logging.debug("search() trying again with regex list extractor")

            # try and extract a list
            pattern = r'\[\s*("(?:\\.|[^"\\])*"\s*,\s*)*"(?:\\.|[^"\\])*"\s*\]'
            match = re.search(pattern, str(query_expander.message))

            if match:
                try:
                    queries = eval(match.group(0))[:self.query_expansion]
                except SyntaxError as ex:
                    logging.debug(f"search() failed regex extractor with SyntaxError: {ex}")
                    pass

        # the first query is the original query
        if self.query not in queries:
            queries.insert(0, self.query)
            queries = queries[:self.query_expansion]

        def url_to_text(result: Dict[str, Any]) -> Content:
            if 'link' in result and isinstance(result['link'], Dict) and 'link' in result['link']:
                return WebHelpers.get_url(result['link']['link'])
            elif 'link' in result:
                return WebHelpers.get_url(result['link'])  # type: ignore
            else:
                return Content()

        def yelp_to_text(reviews: Dict[Any, Any]) -> Content:
            return_str = f"{reviews['title']} in {reviews['neighborhood']}."
            return_str += '\n\n'
            return_str += f"{reviews['reviews']}\n"
            return Content(return_str)

        def local_to_text(document: Dict[Any, Any]) -> Content:
            return_str = f"Title: \"{document['title']}\".\n"
            return_str += f"Link: {document['link']}\n"
            return_str += '\n\n'
            return_str += f"Snippet: \"{document['snippet']}\"\n"
            return Content(return_str)

        def hackernews_comments_to_text(results: List[Dict[str, str]], num_comments: int = 100) -> Content:
            if not results:
                return Content()

            title = results[0]['title']
            url = results[0]['url']
            return_str = f'For the Hacker News article: {title} which has a url of: {url}, the comments are as follows:\n\n'
            for comment in results[:num_comments]:
                return_str += f"{comment['author']} said {comment['comment_text']}.\n"
            return Content(return_str)

        engines = {
            'Google Search': {'searcher': self.search_google_hook, 'parser': url_to_text, 'description': 'Google Search is a general web search engine that is good at answering questions, finding knowledge and information, and has a complete scan of the Internet.'},  # noqa:E501
            'Google Patent Search': {'searcher': self.search_google_hook, 'parser': url_to_text, 'description': 'Google Patent Search is a search engine that is exceptional at findind matching patents for a given query.'},  # noqa:E501
            'Yelp Search': {'searcher': SerpAPISearcher().search_yelp, 'parser': yelp_to_text, 'description': 'Yelp is a search engine dedicated to finding geographically local establishments, restaurants, stores etc and extracing their user reviews.'},  # noqa:E501
            'Local Files Search': {'searcher': self.vector_search.search, 'parser': local_to_text, 'description': 'Local file search engine. Searches the users hard drive for content in pdf, csv, html, doc and docx files.'},  # noqa:E501
            'Hacker News Search': {'searcher': SerpAPISearcher().search_hackernews_comments, 'parser': hackernews_comments_to_text, 'description': 'Hackernews (or hacker news) is search engine dedicated to technology, programming and science. This search engine finds and returns commentary from smart individuals about news, technology, programming and science articles. Rank this engine first if the search query specifically asks for "hackernews".'},  # noqa:E501
            'Google Scholar Search': {'searcher': self.search_research_hook, 'parser': url_to_text, 'description': 'Google Scholar Search is a search engine to help find and summarize academic papers, studies, and research about particular topics'},  # noqa:E501
        }  # noqa:E501

        # classify the search engine
        engine_rank = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='search_classifier.prompt',
                    template={
                        'query': '\n'.join(queries),
                        'engines': '\n'.join([f'* {key}: {value["description"]}' for key, value in engines.items()]),
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='search_classifier.prompt',
            ),
            query=self.query,
            original_query=self.original_query,
        )

        engine = str(engine_rank.message).split('\n')[0]
        searcher = self.search_google_hook

        for key, _ in engines.items():
            if key in engine:
                self.parser = engines[key]['parser']
                searcher = engines[key]['searcher']

        write_client_stream(Content(f"I think the {engine} engine is best to perform a search for {self.query}\n"))

        # perform the search
        search_results = []

        # deal especially for yelp.
        if 'Yelp' in engine:
            # take the first query, and figure out the location
            location = self.controller.execute_llm_call(
                llm_call=LLMCall(
                    user_message=Helpers.prompt_message(
                        prompt_name='search_location.prompt',
                        template={
                            'query': queries[0],
                        },
                        user_token=self.controller.get_executor().user_token(),
                        assistant_token=self.controller.get_executor().assistant_token(),
                        append_token=self.controller.get_executor().append_token(),
                    ),
                    context_messages=[],
                    executor=self.controller.get_executor(),
                    model=self.controller.get_executor().get_default_model(),
                    temperature=0.0,
                    max_prompt_len=self.controller.get_executor().max_input_tokens(),
                    completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                    prompt_name='search_location.prompt',
                ),
                query=self.query,
                original_query=self.original_query,
                compression=TokenCompressionMethod.SUMMARY,
            )

            query_result, location = eval(str(location.message))
            yelp_result = SerpAPISearcher().search_yelp(query_result, location)
            return [yelp_to_text(yelp_result)]

        if 'Hacker' in engine:
            result = SerpAPISearcher().search_hackernews_comments(queries[0])
            return [hackernews_comments_to_text(result)]

        for query in queries:
            search_results.extend(list(searcher(query))[:10])

        import random

        snippets = {
            str(random.randint(0, 100000)): {
                'title': result['title'],
                'snippet': result['snippet'] if 'snippet' in result else '',
                'link': result['link']
            }
            for result in search_results if 'title' in result
        }

        result_rank = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='search_ranker.prompt',
                    template={
                        'queries': '\n'.join(queries),
                        'snippets': '\n'.join(
                            [f'* {str(key)}: {value["title"]} {value["snippet"]}' for key, value in snippets.items()]
                        ),
                        'seen_list': '\n'.join(titles_seen),
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='search_ranker.prompt',
            ),
            query=self.query,
            original_query=self.original_query,
            compression=TokenCompressionMethod.SUMMARY,
        )

        # double shot try
        ranked_results = []
        try:
            ranked_results = eval(str(result_rank.message))
        except SyntaxError as ex:
            logging.debug(f"SyntaxError: {ex}")
            pattern = r'\[\s*(-?\d+\s*,\s*)*-?\d+\s*\]'
            match = re.search(pattern, str(result_rank.message))

            if match:
                try:
                    ranked_results = eval(match.group(0))
                except SyntaxError as ex:
                    logging.debug(f"SyntaxError: {ex}")
                    pass

        # anthropic doesn't follow instructions, and coerces the result into a list of ints
        ranked_results = [str(result) for result in ranked_results]

        self.ordered_snippets = [snippets[key] for key in ranked_results if key in snippets]

        write_client_stream(Content(f"I found and ranked #{len(self.ordered_snippets)} results. Returning the top {self.total_links_to_return}:\n\n"))  # noqa:E501
        for snippet in self.ordered_snippets[0:self.total_links_to_return]:
            write_client_stream(Content(f"{snippet['title']}\n{snippet['link']}\n\n"))

        return self.results()

    def results(self) -> List[Content]:
        return_results = []

        while len(return_results) < self.total_links_to_return and self.index < len(self.ordered_snippets):
            for result in self.ordered_snippets[self.index:]:
                self.index += 1
                try:
                    parser_content: Content = self.parser(result)
                    # if parser_content:
                    #   return_results.append(f"The following content is from: {result['link']} and has the title: {result['title']} \n\n{parser_content}")  # noqa:E501
                    if parser_content:
                        return_results.append(parser_content)

                    if len(return_results) >= self.total_links_to_return:
                        break
                except Exception as e:
                    logging.error(e)
                    pass
        return return_results

    def result(self) -> Content:
        return Content('\n\n\n'.join([str(result) for result in self.results()]))
