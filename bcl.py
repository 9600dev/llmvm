from __future__ import annotations

import ast
import re
import time
from typing import Any, Callable, Dict, Generator, List, Optional, cast
from urllib.parse import urlparse

import astunparse

from ast_parser import Parser
from helpers.firefox import FirefoxHelpers
from helpers.helpers import Helpers
from helpers.logging_helpers import setup_logging
from helpers.pdf import PdfHelpers
from helpers.search import SerpAPISearcher
from helpers.webhelpers import WebHelpers
from objects import (Assistant, Content, Executor, FunctionCall, Message,
                     System, User)
from starlark_runtime import StarlarkRuntime

logging = setup_logging()

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

    def __parse_pdf(self, filename: str) -> str:
        content = PdfHelpers.parse_pdf(filename)

        query_expander = self.starlark_runtime.executor.execute_llm_call(
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/pdf_content.prompt',
                template={
                }),
            context_messages=[self.starlark_runtime.statement_to_message(content)],
            query=self.original_query,
            original_query=self.original_query,
        )

        if 'SUCCESS' in str(query_expander.message):
            return content
        else:
            return PdfHelpers.parse_pdf_image(filename)

    def get(self) -> str:
        logging.debug('ContentDownloader.download: {}'.format(self.expr))

        result = urlparse(self.expr)
        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                return self.__parse_pdf(result.path)
            if '.htm' in result.path or '.html' in result.path:
                return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read())

        elif (result.scheme == 'http' or result.scheme == 'https') and '.pdf' in result.path:
            pdf_filename = self.firefox_helper.pdf_url(self.expr)
            return self.__parse_pdf(pdf_filename)

        elif result.scheme == 'http' or result.scheme == 'https':
            return WebHelpers.convert_html_to_markdown(self.firefox_helper.get_url(self.expr))
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
        total_links_to_return: int = 4,
    ):
        self.query = expr
        self.messages: List[Message] = messages
        self.agents = agents
        self.original_code = original_code
        self.original_query = original_query
        self.starlark_runtime = starlark_runtime

        self.parser = WebHelpers.get_url
        self.ordered_snippets: List = []
        self.index = 0
        self._result = None
        self.total_links_to_return: int = total_links_to_return

    def search(
        self,
    ) -> str:
        # todo: we should probably return the Search instance, so we can futz with it later on.
        query_expander = self.starlark_runtime.executor.execute_llm_call(
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/search_expander.prompt',
                template={
                    'query': self.query,
                }),
            context_messages=[],
            query=self.query,
            original_query=self.original_query,
        )

        queries = eval(str(query_expander.message))[:3]

        def yelp_to_text(reviews: Dict[Any, Any]) -> str:
            return_str = f"{reviews['title']} in {reviews['neighborhood']}."
            return_str += '\n\n'
            return_str += f"{reviews['reviews']}"
            return return_str

        engines = {
            'Google Search': {'searcher': SerpAPISearcher().search_internet, 'parser': WebHelpers.get_url, 'description': 'a general web search engine that is good at answering questions, finding knowledge and information, and has a complete scan of the Internet.'},  # noqa:E501
            'Google News': {'searcher': SerpAPISearcher().search_news, 'parser': WebHelpers.get_news_url, 'description': 'a news search engine. This engine is excellent at finding news about particular topics, people, companies and entities.'},  # noqa:E501
            'Google Product Search': {'searcher': SerpAPISearcher().search_internet, 'parser': WebHelpers.get_url, 'description': 'a product search engine that is excellent at finding the prices of products, finding products that match descriptions of products, and finding where to buy a particular product.'},  # noqa:E501
            'Google Patent Search': {'searcher': SerpAPISearcher().search_internet, 'parser': WebHelpers.get_url, 'description': 'a search engine that is exceptional at findind matching patents for a given query.'},  # noqa:E501
            'Yelp Search': {'searcher': SerpAPISearcher().search_yelp, 'parser': yelp_to_text, 'description': 'a search engine dedicated to finding geographically local establishments, restaurants, stores etc and extracing their user reviews.'},  # noqa:E501
            'Hacker News Search': {'searcher': SerpAPISearcher().search_internet, 'parser': WebHelpers.get_url, 'description': 'a search engine dedicated to technology, programming and science. This search engine finds and returns commentary from smart individuals about news, technology, programming and science articles.'},  # noqa:E501
        }  # noqa:E501

        # classify the search engine
        engine_rank = self.starlark_runtime.executor.execute_llm_call(
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/search_classifier.prompt',
                template={
                    'query': '\n'.join(queries),
                    'engines': '\n'.join([f'* {key}: {value["description"]}' for key, value in engines.items()]),
                }),
            context_messages=[],
            query=self.query,
            original_query=self.original_query,
        )
        engine = str(engine_rank.message).split('\n')[0]
        searcher = SerpAPISearcher().search_internet

        for key, value in engines.items():
            if key in engine:
                self.parser = engines[key]['parser']
                searcher = engines[key]['searcher']

        # perform the search
        search_results = []

        # deal especially for yelp.
        if 'Yelp' in engine:
            # take the first query, and figure out the location
            location = self.starlark_runtime.executor.execute_llm_call(
                message=Helpers.load_and_populate_message(
                    prompt_filename='prompts/starlark/search_location.prompt',
                    template={
                        'query': queries[0],
                    }),
                context_messages=[],
                query=self.query,
                original_query=self.original_query,
                prompt_filename='prompts/starlark/search_location.prompt',
            )
            query_result, location = eval(str(location.message))
            yelp_result = SerpAPISearcher().search_yelp(query_result, location)
            return yelp_to_text(yelp_result)

        for query in queries:
            search_results.extend(list(searcher(query))[:10])

        import random

        snippets = {
            str(random.randint(0, 100000)): {'title': result['title'], 'snippet': result['snippet'], 'link': result['link']}
            for result in search_results
        }

        result_rank = self.starlark_runtime.executor.execute_llm_call(
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/search_ranker.prompt',
                template={
                    'queries': '\n'.join(queries),
                    'snippets': '\n'.join(
                        [f'* {str(key)}: {value["title"]} {value["snippet"]}' for key, value in snippets.items()]
                    ),
                }),
            context_messages=[],
            query=self.query,
            original_query=self.original_query,
        )

        ranked_results = eval(str(result_rank.message))
        self.ordered_snippets = [snippets[key] for key in ranked_results if key in snippets]
        return self.result()

    def result(self) -> str:
        return_results = []

        while len(return_results) < self.total_links_to_return and self.index < len(self.ordered_snippets):
            for result in self.ordered_snippets[self.index:]:
                self.index += 1
                try:
                    parser_result = self.parser(result['link']).strip()
                    if parser_result:
                        return_results.append(f"The following content is from: {result['link']} with the title: {result['title']} \n\n{parser_result}")  # noqa:E501
                    if len(return_results) >= 4:
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

        message = Helpers.load_and_populate_message(
            prompt_filename='prompts/starlark/llm_bind_global.prompt',
            template={
                'function_definition': function_definition,
            }
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

                llm_bind_result = self.starlark_runtime.executor.execute_llm_call(
                    message=User(Content()),  # we can pass an empty message here and the context_messages contain everything
                    context_messages=messages[:counter + assistant_counter][::-1],  # reversing the list using list slicing
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
                    self.starlark_runtime.executor,
                    self.agents,
                    self.starlark_runtime.vector_store
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
