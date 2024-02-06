from __future__ import annotations

import ast
import asyncio
import datetime as dt
import re
import time
from datetime import timedelta
from typing import Any, Callable, Dict, Generator, List, Optional, cast
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import astunparse
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from googlesearch import search as google_search

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Assistant, Content, FunctionCall, LLMCall,
                                  Message, System, TokenCompressionMethod,
                                  User)
from llmvm.server.ast_parser import Parser
from llmvm.server.source import Source
from llmvm.server.starlark_runtime import StarlarkRuntime
from llmvm.server.tools.firefox import FirefoxHelpers
from llmvm.server.tools.pdf import PdfHelpers
from llmvm.server.tools.search import SerpAPISearcher
from llmvm.server.tools.webhelpers import WebHelpers
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()

class BCL():
    @staticmethod
    def __last_day_of_quarter(year, quarter):
        start_month = 3 * quarter - 2
        end_month = start_month + 2

        if end_month > 12:
            end_month = 12

        last_day = (dt.datetime(year, end_month, 1) + timedelta(days=31)).replace(day=1) - timedelta(days=1)
        return last_day

    @staticmethod
    def __parse_relative_datetime(relative_expression: str, timezone: Optional[str] = None) -> dt.datetime:
        if relative_expression.startswith('Q'):
            quarter = int(relative_expression[1:])
            return BCL.__last_day_of_quarter(dt.datetime.now().year, quarter)

        tz = dt.datetime.now().astimezone().tzinfo

        if timezone:
            tz = ZoneInfo(timezone)

        if 'now' in relative_expression:
            return dt.datetime.now(tz)

        parts = relative_expression.split()

        if len(parts) != 2:
            return parse(relative_expression)  # type: ignore

        value = int(parts[0])
        unit = parts[1].lower()

        if unit == "days":
            return dt.datetime.now(tz) + timedelta(days=value)
        elif unit == "months":
            return dt.datetime.now(tz) + relativedelta(months=value)
        elif unit == "years":
            return dt.datetime.now(tz) + relativedelta(years=value)
        elif unit == "hours":
            return dt.datetime.now(tz) + timedelta(hours=value)
        else:
            return parse(relative_expression)  # type: ignore

    @staticmethod
    def datetime(expr, timezone: Optional[str] = None) -> dt.datetime:
        """
        Returns a datetime object from a string using datetime.strftime().
        Examples: datetime("2020-01-01"), datetime("now"), datetime("-1 days"), datetime("now", "Australia/Brisbane")
        """
        return BCL.__parse_relative_datetime(str(expr), timezone)


class ContentDownloader():
    def __init__(
        self,
        expr,
        agents: List[Callable],
        messages: List[Message],
        starlark_runtime: StarlarkRuntime,
        original_code: str,
        original_query: str,
    ):
        self.expr = expr
        self.agents = agents
        self.messages: List[Message] = messages
        self.starlark_runtime = starlark_runtime
        self.original_code = original_code
        self.original_query = original_query
        self.firefox_helper = FirefoxHelpers()

        # the client can often send through urls with quotes around them
        if self.expr.startswith('"') and self.expr.endswith('"'):
            self.expr = self.expr[1:-1]

    def parse_pdf(self, filename: str) -> str:
        content = PdfHelpers.parse_pdf(filename)

        query_expander = self.starlark_runtime.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='pdf_content.prompt',
                    template={},
                    user_token=self.starlark_runtime.controller.get_executor().user_token(),
                    assistant_token=self.starlark_runtime.controller.get_executor().assistant_token(),
                    append_token=self.starlark_runtime.controller.get_executor().append_token(),
                ),
                context_messages=[self.starlark_runtime.statement_to_message(content)],
                executor=self.starlark_runtime.controller.get_executor(),
                model=self.starlark_runtime.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.starlark_runtime.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.starlark_runtime.controller.get_executor().max_completion_tokens(),
                prompt_name='pdf_content.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )

        if 'SUCCESS' in str(query_expander.message):
            return content
        else:
            return PdfHelpers.parse_pdf_image(filename)

    def get(self) -> str:
        logging.debug('ContentDownloader.get: {}'.format(self.expr))

        # deal with files
        result = urlparse(self.expr)
        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                return self.parse_pdf(result.path)
            if '.htm' in result.path or '.html' in result.path:
                return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read())

        # deal with pdfs
        elif (result.scheme == 'http' or result.scheme == 'https') and '.pdf' in result.path:
            loop = asyncio.get_event_loop()
            task = loop.create_task(self.firefox_helper.pdf_url(self.expr))

            pdf_filename = loop.run_until_complete(task)
            return self.parse_pdf(pdf_filename)

        # deal with websites
        elif result.scheme == 'http' or result.scheme == 'https':
            loop = asyncio.get_event_loop()
            task = loop.create_task(self.firefox_helper.get_url(self.expr))
            result = loop.run_until_complete(task)

            return WebHelpers.convert_html_to_markdown(result)
        return ''


class Searcher():
    def __init__(
        self,
        expr,
        agents: List[Callable],
        messages: List[Message],
        starlark_runtime: StarlarkRuntime,
        original_code: str,
        original_query: str,
        vector_search: VectorSearch,
        total_links_to_return: int = 2,
    ):
        self.query = expr
        self.messages: List[Message] = messages
        self.agents = agents
        self.original_code = original_code
        self.original_query = original_query
        self.starlark_runtime = starlark_runtime
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

    def search(
        self,
    ) -> str:
        # todo: we should probably return the Search instance, so we can futz with it later on.
        query_expander = self.starlark_runtime.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='search_expander.prompt',
                    template={
                        'query': self.query,
                    },
                    user_token=self.starlark_runtime.controller.get_executor().user_token(),
                    assistant_token=self.starlark_runtime.controller.get_executor().assistant_token(),
                    append_token=self.starlark_runtime.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.starlark_runtime.controller.get_executor(),
                model=self.starlark_runtime.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.starlark_runtime.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.starlark_runtime.controller.get_executor().max_completion_tokens(),
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

        def url_to_text(result: Dict[Any, Any]) -> str:
            return WebHelpers.get_url(result['link'])

        def yelp_to_text(reviews: Dict[Any, Any]) -> str:
            return_str = f"{reviews['title']} in {reviews['neighborhood']}."
            return_str += '\n\n'
            return_str += f"{reviews['reviews']}\n"
            return return_str

        def local_to_text(document: Dict[Any, Any]) -> str:
            return_str = f"Title: \"{document['title']}\".\n"
            return_str += f"Link: {document['link']}\n"
            return_str += '\n\n'
            return_str += f"Snippet: \"{document['snippet']}\"\n"
            return return_str

        def hackernews_comments_to_text(results: List[Dict[str, str]], num_comments: int = 100) -> str:
            if not results:
                return ''

            title = results[0]['title']
            url = results[0]['url']
            return_str = f'For the Hacker News article: {title} which has a url of: {url}, the comments are as follows:\n\n'
            for comment in results[:num_comments]:
                return_str += f"{comment['author']} said {comment['comment_text']}.\n"
            return return_str

        engines = {
            'Google Search': {'searcher': self.search_google_hook, 'parser': url_to_text, 'description': 'Google Search is a general web search engine that is good at answering questions, finding knowledge and information, and has a complete scan of the Internet.'},  # noqa:E501
            'Google News': {'searcher': self.search_news_hook, 'parser': url_to_text, 'description': 'Google News Search is a news search engine. This engine is excellent at finding news about particular topics, people, companies and entities.'},  # noqa:E501
            'Google Patent Search': {'searcher': self.search_google_hook, 'parser': url_to_text, 'description': 'Google Patent Search is a search engine that is exceptional at findind matching patents for a given query.'},  # noqa:E501
            'Yelp Search': {'searcher': SerpAPISearcher().search_yelp, 'parser': yelp_to_text, 'description': 'Yelp is a search engine dedicated to finding geographically local establishments, restaurants, stores etc and extracing their user reviews.'},  # noqa:E501
            'Local Files Search': {'searcher': self.vector_search.search, 'parser': local_to_text, 'description': 'Local file search engine. Searches the users hard drive for content in pdf, csv, html, doc and docx files.'},  # noqa:E501
            'Hacker News Search': {'searcher': SerpAPISearcher().search_hackernews_comments, 'parser': hackernews_comments_to_text, 'description': 'Hackernews (or hacker news) is search engine dedicated to technology, programming and science. This search engine finds and returns commentary from smart individuals about news, technology, programming and science articles. Rank this engine first if the search query specifically asks for "hackernews".'},  # noqa:E501
        }  # noqa:E501

        # classify the search engine
        engine_rank = self.starlark_runtime.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='search_classifier.prompt',
                    template={
                        'query': '\n'.join(queries),
                        'engines': '\n'.join([f'* {key}: {value["description"]}' for key, value in engines.items()]),
                    },
                    user_token=self.starlark_runtime.controller.get_executor().user_token(),
                    assistant_token=self.starlark_runtime.controller.get_executor().assistant_token(),
                    append_token=self.starlark_runtime.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.starlark_runtime.controller.get_executor(),
                model=self.starlark_runtime.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.starlark_runtime.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.starlark_runtime.controller.get_executor().max_completion_tokens(),
                prompt_name='search_classifier.prompt',
            ),
            query=self.query,
            original_query=self.original_query,
        )

        engine = str(engine_rank.message).split('\n')[0]
        searcher = self.search_google_hook

        for key, value in engines.items():
            if key in engine:
                self.parser = engines[key]['parser']
                searcher = engines[key]['searcher']

        write_client_stream(Content(f"I'm using the {engine} engine to perform search.\n"))

        # perform the search
        search_results = []

        # deal especially for yelp.
        if 'Yelp' in engine:
            # take the first query, and figure out the location
            location = self.starlark_runtime.controller.execute_llm_call(
                llm_call=LLMCall(
                    user_message=Helpers.prompt_message(
                        prompt_name='search_location.prompt',
                        template={
                            'query': queries[0],
                        },
                        user_token=self.starlark_runtime.controller.get_executor().user_token(),
                        assistant_token=self.starlark_runtime.controller.get_executor().assistant_token(),
                        append_token=self.starlark_runtime.controller.get_executor().append_token(),
                    ),
                    context_messages=[],
                    executor=self.starlark_runtime.controller.get_executor(),
                    model=self.starlark_runtime.controller.get_executor().get_default_model(),
                    temperature=0.0,
                    max_prompt_len=self.starlark_runtime.controller.get_executor().max_prompt_tokens(),
                    completion_tokens_len=self.starlark_runtime.controller.get_executor().max_completion_tokens(),
                    prompt_name='search_location.prompt',
                ),
                query=self.query,
                original_query=self.original_query,
                token_compression_method=TokenCompressionMethod.SUMMARY,
            )

            query_result, location = eval(str(location.message))
            yelp_result = SerpAPISearcher().search_yelp(query_result, location)
            return yelp_to_text(yelp_result)

        if 'Hacker' in engine:
            result = SerpAPISearcher().search_hackernews_comments(queries[0])
            return hackernews_comments_to_text(result)

        for query in queries:
            search_results.extend(list(searcher(query))[:10])

        import random

        snippets = {
            str(random.randint(0, 100000)): {
                'title': result['title'],
                'snippet': result['snippet'] if 'snippet' in result else '',
                'link': result['link']
            }
            for result in search_results
        }

        result_rank = self.starlark_runtime.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='search_ranker.prompt',
                    template={
                        'queries': '\n'.join(queries),
                        'snippets': '\n'.join(
                            [f'* {str(key)}: {value["title"]} {value["snippet"]}' for key, value in snippets.items()]
                        ),
                    },
                    user_token=self.starlark_runtime.controller.get_executor().user_token(),
                    assistant_token=self.starlark_runtime.controller.get_executor().assistant_token(),
                    append_token=self.starlark_runtime.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.starlark_runtime.controller.get_executor(),
                model=self.starlark_runtime.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.starlark_runtime.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.starlark_runtime.controller.get_executor().max_completion_tokens(),
                prompt_name='search_ranker.prompt',
            ),
            query=self.query,
            original_query=self.original_query,
            token_compression_method=TokenCompressionMethod.SUMMARY,
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

        return self.result()

    def result(self) -> str:
        return_results = []

        while len(return_results) < self.total_links_to_return and self.index < len(self.ordered_snippets):
            for result in self.ordered_snippets[self.index:]:
                self.index += 1
                try:
                    parser_result = self.parser(result).strip()
                    if parser_result:
                        return_results.append(f"The following content is from: {result['link']} and has the title: {result['title']} \n\n{parser_result}")  # noqa:E501
                    if len(return_results) >= self.total_links_to_return:
                        break
                except Exception as e:
                    logging.error(e)
                    pass
        return '\n\n\n'.join(return_results)


class FunctionBindable():
    def __init__(
        self,
        expr,
        func: str,
        agents: List[Callable],
        messages: List[Message],
        lineno: int,
        expr_instantiation,
        scope_dict: Dict[Any, Any],
        original_code: str,
        original_query: str,
        starlark_runtime: StarlarkRuntime,
    ):
        self.expr = expr
        self.expr_instantiation = expr_instantiation
        self.messages: List[Message] = messages
        self.func = func.replace('"', '')
        self.agents = agents
        self.lineno = lineno
        self.scope_dict = scope_dict
        self.original_code = original_code
        self.original_query = original_query
        self.starlark_runtime = starlark_runtime
        self.bound_function: Optional[Callable] = None
        self._result = None

    def __call__(self, *args, **kwargs):
        if self._result:
            return self._result

    def __bind_helper(
        self,
        func: str,
    ) -> Message:
        # if we have a list, we need to use a different prompt
        if isinstance(self.expr, list):
            raise ValueError('llm_bind() does not support lists. You should rewrite the code to use a for loop instead.')

        # get a function definition fuzzy binding
        function_str = Helpers.in_between(func, '', '(')
        function_callable = [f for f in self.agents if function_str in str(f)]
        if not function_callable:
            raise ValueError('could not find function: {}'.format(function_str))

        function_callable = function_callable[0]
        function_definition = Helpers.get_function_description_flat_extra(cast(Callable, function_callable))

        message = Helpers.prompt_message(
            prompt_name='llm_bind_global.prompt',
            template={
                'function_definition': function_definition,
            },
            user_token=self.starlark_runtime.controller.get_executor().user_token(),
            assistant_token=self.starlark_runtime.controller.get_executor().assistant_token(),
            append_token=self.starlark_runtime.controller.get_executor().append_token(),
        )
        return message

    def binder(
        self,
        expr,
        func: str,
    ) -> Generator['FunctionBindable', None, None]:
        bound = False
        global_counter = 0
        messages: List[Message] = []
        extra: List[str] = []
        goal = ''
        bindable = ''
        function_call: Optional[FunctionCall] = None

        def find_string_instantiation(target_string, source_code):
            parsed_ast = ast.parse(source_code)

            for node in ast.walk(parsed_ast):
                # Check for direct assignment
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(node.value, ast.Str):
                            if node.value.s == target_string:
                                return (node, None)
                        # Check for string instantiation in a list
                        elif isinstance(node.value, ast.List):
                            for element in node.value.elts:
                                if isinstance(element, ast.Str) and element.s == target_string:
                                    return (element, node)
            return None, None

        # the following code determines the progressive scope exposure to the LLM
        # to help it determine the binding
        expr_instantiation_message = User(Content())
        if isinstance(expr, str) and find_string_instantiation(expr, self.original_code):
            node, parent = find_string_instantiation(expr, self.original_code)
            if parent:
                expr_instantiation_message.message = Content(
                    f"The data in the next message was instantiated via this Starlark code: {astunparse.unparse(parent)}"
                )
            elif node:
                expr_instantiation_message.message = Content(
                    f"The data in the next message was instantiated via this Starlark code: {astunparse.unparse(node.value)}"
                )

        messages.append(System(Content(
            '''You are a Starlark compiler and code generator. You generate parsable Starlark code.'''
        )))

        # get the binder prompt message
        messages.append(self.__bind_helper(
            func=func,
            # context_messages=messages[:counter + assistant_counter][::-1],  # reversing the list using list slicing
        ))
        # start with just the expression binding
        messages.append(self.starlark_runtime.statement_to_message(expr))
        # instantiation
        if str(expr_instantiation_message.message):
            messages.append(expr_instantiation_message)
        # goal
        messages.append(User(Content(
            f"""The overall goal of the Starlark program is to: {self.original_query}."""
        )))
        messages.append(User(Content(
            f"""The Starlark code that is currently being executed is: {self.original_code}"""
        )))
        # program scope
        scope = '\n'.join(['{} = "{}"'.format(key, value) for key, value in self.scope_dict.items() if key.startswith('var')])
        messages.append(User(Content(
            f"""The Starlark program's running global scope for all variables is:

            {scope}

            You might find data you need to bind function arguments in the values of these variables.
            """
        )))

        counter = 3  # we start at 3 because we've already added the system, expr and the function definition prompt
        assistant_counter = 0

        while global_counter < 3:
            # try and bind the callsite without executing
            while not bound and counter < 8:

                llm_bind_result = self.starlark_runtime.controller.execute_llm_call(
                    llm_call=LLMCall(
                        user_message=User(Content()),  # we can pass an empty message here and the context_messages contain everything  # noqa:E501
                        context_messages=messages[:counter + assistant_counter][::-1],  # reversing the list using list slicing
                        executor=self.starlark_runtime.controller.get_executor(),
                        model=self.starlark_runtime.controller.get_executor().get_default_model(),
                        temperature=0.0,
                        max_prompt_len=self.starlark_runtime.controller.get_executor().max_prompt_tokens(),
                        completion_tokens_len=self.starlark_runtime.controller.get_executor().max_completion_tokens(),
                        prompt_name=''
                    ),
                    query=self.original_query,
                    original_query=self.original_query,
                )

                bindable = str(llm_bind_result.message)

                # the LLM can get confused and generate a function definition instead of a callsite
                # or enclose the result in ```python ... ``` code blocks.
                if 'def ' in bindable:
                    bindable = bindable.replace('def ', '')

                if '```python' in bindable:
                    match = re.search(r'```python([\s\S]*?)```', bindable)
                    if match:
                        bindable = match.group(1).replace('python', '').strip()

                if '```starlark' in bindable:
                    match = re.search(r'```starlark([\s\S]*?)```', bindable)
                    if match:
                        bindable = match.group(1).replace('starlark', '').strip()

                # get function definition
                parser = Parser()
                parser.agents = self.agents
                function_call = parser.get_callsite(bindable)

                if 'None' in str(bindable):
                    # move forward a stage and add the latest assistant response
                    # as the assistant response will have a # based question in it
                    # which will help bind the unbindable arguments.
                    counter += 1
                    assistant_counter += 1

                    if '#' in bindable:
                        question = bindable.split('#')[1].strip()
                        prompt = f'''
                        Using the data found in previous messages, answer the question "{question}", and then bind the callsite
                        using the same reply rules as in previous messages. Reply with only Starlark code.
                        '''
                        messages.insert(0, User(Content(prompt)))
                    else:
                        # todo figure this out
                        messages.insert(0, Assistant(message=Content(bindable)))
                    if counter > len(messages) - assistant_counter:
                        # we've run out of messages, so we'll just use the original code
                        break

                elif 'None' not in str(bindable) and function_call:
                    break
                else:
                    # no function_call result, so bump the counter
                    messages.insert(0, Assistant(message=Content(bindable)))
                    messages.insert(0, User(message=Content(
                        """Please try harder to bind the callsite.
                        Look thoroughly through the previous messages for data and then reply with your best guess at the bounded
                        callsite. Reply only with Starlark code that can be parsed by the Starlark compiler.
                        Do not apologize. Do not explain yourself. If you have previously replied with natural language,
                        it's likely I could not compile it. Please reply with only Starlark code.
                        """
                    )))
                    assistant_counter += 1
                    counter += 1

            # Using the previous messages, What is the company name associated with Steve Baxter?
            # If you've answered the question above, can you rebind the callsite?
            # def search_linkedin_profile(first_name: str, last_name: str, company_name: str) -> str
            # # Searches for the LinkedIn profile of a given first name and last name and optional
            # company name and returns the LinkedIn profile. If you use this method you do not need
            # to call get_linkedin_profile.

            if not function_call:
                raise ValueError('couldn\'t bind function call for func: {}, expr: {}'.format(func, expr))

            # todo need to properly parse this.
            if ' = ' not in bindable:
                # no identifier, so we'll create one to capture the result
                identifier = 'result_{}'.format(str(time.time()).replace('.', ''))
                bindable = '{} = {}'.format(identifier, bindable)
            else:
                identifier = bindable.split(' = ')[0].strip()

            # execute the function
            # todo: figure out what to do when you get back None, or ''
            starlark_code = bindable
            globals_dict = self.scope_dict.copy()
            globals_result = {}

            try:
                global_counter += 1

                globals_result = StarlarkRuntime(
                    controller=self.starlark_runtime.controller,
                    agents=self.agents,
                    vector_search=self.starlark_runtime.vector_search,
                ).run(starlark_code, '')

                self._result = globals_result[identifier]
                yield self

                # if we're here, it's because we've been next'ed() and it was the wrong binding
                # reset the binding parameters and try again.
                counter = 0
                bound = False

            except Exception as ex:
                logging.debug('Error executing function call: {}'.format(ex))
                counter += 1
                starlark_code = self.starlark_runtime.rewrite_starlark_error_correction(
                    query=self.original_query,
                    starlark_code=starlark_code,
                    error=str(ex),
                    globals_dictionary=globals_dict,
                )

        # we should probably return uncertain_or_error here.
        # raise ValueError('could not bind and or execute the function: {} expr: {}'.format(func, expr))
        self._result = 'could not bind and or execute the function: {} expr: {}'.format(func, expr)
        yield self

    def bind(
        self,
        expr,
        func,
    ) -> 'FunctionBindable':
        for bindable in self.binder(expr, func):
            return bindable

        raise ValueError('could not bind and or execute the function: {} expr: {}'.format(func, expr))


class SourceProject:
    def __init__(
        self,
        starlark_runtime: StarlarkRuntime
    ):
        # todo: this is weirdly circular - Controller -> Runtime -> BCL -> Controller. Weird.. Fix.
        self.starlark_runtime = starlark_runtime
        self.sources: List[Source] = []
        self.other_files: List[str] = []

    def set_files(self, files):
        for source_path in files:
            source = Source(source_path)
            if source.tree:
                self.sources.append(source)
            else:
                self.other_files.append(source_path)

    def get_source_structure(self) -> str:
        """
        gets all class names, method names, and docstrings for all classes and methods in all files
        listed in "Files:". This method does not return any source code.
        """
        structure = ''
        for source in self.sources:
            structure += f'File: {source.file_path}\n'
            for class_def in source.get_classes():
                structure += f'class {class_def.name}:\n'
                structure += f'    """{class_def.docstring}"""\n'
                structure += '\n'
                for method_def in source.get_methods(class_def.name):
                    structure += f'    def {method_def.name}:\n'
                    structure += f'        """{method_def.docstring}"""\n'
                    structure += '\n'
            structure += '\n\n'

        return structure

    def get_source_summary(self, file_path: str) -> str:
        """
        gets all class names, method names and natural language descriptions of class and method names
        for a given source file. The file_name must be in the "Files:" list. It does not return any source code.
        """
        def _summary_helper(source: Source) -> str:
            write_client_stream(Content(f"Asking LLM to summarize file: {source.file_path}\n\n"))
            summary = ''
            for class_def in source.get_classes():
                summary += f'class {class_def.name}:\n'
                summary += f'    """{class_def.docstring}"""\n'
                summary += '\n'

                method_definition = ''
                for method_def in source.get_methods(class_def.name):
                    method_definition += f'    def {method_def.name}({", ".join([param[0] for param in method_def.params])})\n'
                    method_definition += f'        """{method_def.docstring}"""\n'
                    method_definition += '\n'
                    method_definition += f'        {source.get_method_source(method_def.name)}\n\n'
                    method_definition += f'Summary of method {method_def.name}:\n'

                    # get the natural language definition
                    assistant = self.starlark_runtime.controller.execute_llm_call(
                        llm_call=LLMCall(
                            user_message=Helpers.prompt_message(
                                prompt_name='code_method_definition.prompt',
                                template={},
                                user_token=self.starlark_runtime.controller.get_executor().user_token(),
                                assistant_token=self.starlark_runtime.controller.get_executor().assistant_token(),
                                append_token=self.starlark_runtime.controller.get_executor().append_token(),
                            ),
                            context_messages=[User(Content(method_definition))],
                            executor=self.starlark_runtime.controller.get_executor(),
                            model=self.starlark_runtime.controller.get_executor().get_default_model(),
                            temperature=0.0,
                            max_prompt_len=self.starlark_runtime.controller.get_executor().max_prompt_tokens(),
                            completion_tokens_len=self.starlark_runtime.controller.get_executor().max_completion_tokens(),
                            prompt_name='code_method_definition.prompt',
                        ),
                        query='',
                        original_query='',
                    )

                    method_definition += str(assistant.message) + '\n\n'
                    summary += method_definition
                    write_client_stream(Content(method_definition))

                summary += '\n\n'
            return summary

        for source in self.sources:
            if source.file_path == file_path:
                return _summary_helper(source)
        raise ValueError(f"Source file not found: {file_path}")

    def get_files(self):
        return self.sources

    def get_source(self, file_path):
        for source in self.sources:
            if source.file_path == file_path:
                return source.source_code

        for source in self.other_files:
            if source == file_path:
                with open(source, 'r') as file:
                    return file.read()

        raise ValueError(f"Source file not found: {file_path}")

    def get_methods(self, class_name) -> List['Source.Symbol']:
        methods = []
        for source in self.sources:
            methods.extend(source.get_methods(class_name))
        return methods

    def get_classes(self) -> List['Source.Symbol']:
        classes = []
        for source in self.sources:
            classes.extend(source.get_classes())
        return classes

    def get_references(self, method_name) -> List['Source.Callsite']:
        references = []
        for source in self.sources:
            references.extend(Source.get_references(source.get_tree(), method_name))
        return references
