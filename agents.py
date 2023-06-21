import asyncio
import datetime as dt
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from enum import Enum
from itertools import cycle, islice
from typing import (Any, Callable, Dict, Generator, List, Optional, Sequence,
                    Tuple, Union, cast)

import click
import faiss
import guidance
import html2text
import nest_asyncio
import newspaper
import nltk
import openai
import requests
import rich
import tiktoken
from docstring_parser import parse as docstring_parse
from guidance.llms import LLM, OpenAI
from guidance.llms.transformers import LLaMA, Vicuna
from langchain.agents import initialize_agent, load_tools
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI as langchain_OpenAI
from langchain.text_splitter import (MarkdownTextSplitter,
                                     PythonCodeTextSplitter, TokenTextSplitter)
from langchain.vectorstores import FAISS
from llama_cpp import LogitsProcessorList, StoppingCriteriaList
from newspaper import Article
from newspaper.configuration import Configuration
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.logging import RichHandler
from rich.traceback import install
from sec_api import ExtractorApi, QueryApi
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from sentence_transformers import SentenceTransformer, util
from serpapi import BingSearch, GoogleSearch
from torch import Tensor

from eightbitvicuna import VicunaEightBit
from helpers.edgar import EdgarHelpers
from helpers.helpers import Helpers
from helpers.logging_helpers import setup_logging
from helpers.pdf import PdfHelpers
from helpers.search import Searcher, SerpAPISearcher
from helpers.vector_store import VectorStore
from helpers.websearch import WebHelpers

# https://www.blocktool.com/earnings-call/META/1/2023
# https://github.com/operand/agency langchain equivalent, looks pretty good
# https://github.com/lvwerra/trl
# langchain trained local llm https://huggingface.co/ausboss/llama-30b-supercot
# https://github.com/whitead/paper-qa  (q&a over pdfs)
# https://prompts.chat/
# Automatic prompt optimization: https://arxiv.org/pdf/2305.03495v1.pdf
# Toolformer: Language models that can teach themselves to use tools https://arxiv.org/pdf/2302.04761.pdf
# https://twitter.com/dr_cintas/status/1646871072220823552
# tree of thought
# auto prompt

# https://github.com/kyrolabs/awesome-langchain
# https://twitter.com/Jorisdejong4561/status/1660372052468015105
# https://betterprogramming.pub/creating-my-first-ai-agent-with-vicuna-and-langchain-376ed77160e3

# python -m llama_cpp.server --model models/wizard-mega-13B-GGML/wizard-mega-13B.ggmlv3.q5_1.bin --n_gpu_layers 44

# /v1/chat/completions: gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
# /v1/completions:    text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001

# https://github.com/Josh-XT/AGiXT


logging = setup_logging()
vector_store = VectorStore(openai_key=os.environ.get('OPENAI_API_KEY'), store_filename='faiss_index')  # type: ignore


def load_vicuna():
    return VicunaEightBit('models/applied-vicuna-7b', device_map='auto')


def invokeable(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} ran in {end_time - start_time} seconds")
        return result
    return wrapper


class Executor(ABC):
    @abstractmethod
    def execute(self, query: Union[str, List[Dict]], data: str) -> 'ExprNode':
        pass

    @abstractmethod
    def can_execute(self, query: Union[str, List[Dict]], data: str) -> bool:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chat_context(self, chat: bool):
        pass


class Agent(ABC):
    @abstractmethod
    def is_task(self, query: str) -> bool:
        pass

    @abstractmethod
    def perform_task(self, task: str, **kwargs) -> str:
        pass

    @abstractmethod
    def invoked_by(self) -> str:
        pass

    @abstractmethod
    def instruction(self) -> str:
        pass


class AstNode(ABC):
    def __init__(
        self
    ):
        pass

class Data(AstNode):
    def __init__(
        self,
        data: str,
    ):
        self.data = data

    def __str__(self):
        return f'Data({self.data})'

class Prompt(AstNode):
    def __init__(
        self,
        prompt: Union[str, List[Dict]],
    ):
        self.prompt = prompt

    def __str__(self):
        return f'Prompt({self.prompt})'

class SystemPrompt(AstNode):
    def __init__(
        self,
        prompt: str = 'Don\'t make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.',  # type: ignore
    ):
        self.prompt = prompt

    def __str__(self):
        return f'SystemPrompt({self.prompt})'

class Assistant(AstNode):
    def __init__(
        self,
        content: str,
    ):
        self.content = content

    def __str__(self):
        return f'Assistant({self.content})'

class ExprNode(AstNode):
    pass

class Call(ExprNode):
    pass

class LLMCall(Call):
    def __init__(
        self,
        prompt: Prompt,
        data: Data,
        system: SystemPrompt,
        executor: Executor,
    ):
        self.prompt: Prompt = prompt
        self.data: Data = data
        self.system: SystemPrompt = system
        self.executor: Executor = executor

class FunctionCall(Call):
    def __init__(
        self,
        name: str,
        args: List[Dict[str, object]],
    ):
        self.name = name
        self.args = args

class ChainedCall(Call):
    def __init__(
        self,
        calls: Sequence[Call],
    ):
        self.calls = calls

class Result(ExprNode):
    def __init__(
        self,
        conversation: List[AstNode] = [],
        result: object = None,
        error: object = None,
    ):
        self.conversation = conversation,
        self.result = result
        self.error = error


class DataProvider(Agent):
    def __init__(
        self,
    ):
        pass

    def is_task(self, query: str) -> bool:
        if 'using(' in query and ')' in query:
            return True
        return False

    def perform_task(self, task: str, **kwargs) -> str:
        return WebHelpers.get_url(Helpers.in_between(task, 'using(', ')'))

    def invoked_by(self) -> str:
        return 'using(url)'

    def instruction(self) -> str:
        return 'From now on, whenever your response depends on external data, please call {} before responding.'.format(self.invoked_by())


class LocalSearchAgent(Agent):
    def __init__(
        self,
        searcher: Searcher,
        llm: Callable[[], LLM],
        chunk_limit: int = 4,
    ):
        self.searcher = searcher
        self.llm_call: Callable[[], LLM] = llm
        self.llm_instance: Optional[LLM] = None
        self.chunk_limit = chunk_limit

    def llm(self):
        if not self.llm_instance:
            self.llm_instance = self.llm_call()
        return self.llm_instance

    @invokeable
    def search(self, query: str) -> str:
        """Search the web for a query and return the top textual results.

        Args:
            query (str): The query to search the web for.
        Returns:
            str: The top results for the query.
        """
        query = query.replace('<search>', '').replace('</search>', '').strip()

        results = self.searcher.search_internet(query)
        chunked = []
        for result in results:
            chunked.append(Helpers.chunk_and_rank(query, result)[:self.chunk_limit])
        items = list(Helpers.roundrobin(*chunked))
        result_list = []
        for i, item in enumerate(items):
            result_list.append({'text': item, 'index': i})

        return ' '.join([result['text'] for result in result_list])

    def is_task(self, completion):
        return '<search>' in completion

    def invoked_by(self):
        return '<search>query</search>'

    def instruction(self) -> str:
        return 'From now on, whenever your response depends on factual information, please search the web using the function {} before responding.'.format(self.invoked_by())

    def perform_task(self, query: str, **kwargs) -> Dict[str, str]:
        logging.debug('SearchAgent.perform_task() query={}'.format(query))
        # https://github.com/andysalerno/guidance-GPTQ-for-LLaMa/blob/triton/guidance_playground.ipynb
        demo_results = [
            {'text': 'OpenAI systems run on the fifth most powerful supercomputer in the world. [5] [6] [7] The organization was founded in San Francisco in 2015 by Sam Altman, Reid Hoffman, Jessica Livingston, Elon Musk, Ilya Sutskever, Peter Thiel and others, [8] [1] [9] who collectively pledged US$ 1 billion. Musk resigned from the board in 2018 but remained a donor.'},
            {'text': 'About OpenAI is an AI research and deployment company. Our mission is to ensure that artificial general intelligence benefits all of humanity. Our vision for the future of AGI Our mission is to ensure that artificial general intelligence—AI systems that are generally smarter than humans—benefits all of humanity. Read our plan for AGI'},
            {'text': 'Samuel H. Altman ( AWLT-mən; born April 22, 1985) is an American entrepreneur, investor, and programmer. [2] He is the CEO of OpenAI and the former president of Y Combinator. [3] [4] Altman is also the co-founder of Loopt (founded in 2005) and Worldcoin (founded in 2020). Early life and education [ edit]'}
        ]

        practice_template = '''
            {{#user~}}
            Who are the founders of OpenAI?
            {{~/user}}
            {{#assistant~}}
            <search>OpenAI founders</search>
            {{~/assistant}}
            {{#user~}}
            Search results:
            {{~#each results}}
            <result>
            {{this.text}}
            </result>{{/each}}
            {{~/user}}
            {{#assistant~}}
            The founders of OpenAI are Sam Altman, Reid Hoffman, Jessica Livingston, Elon Musk, Ilya Sutskever, Peter Thiel and others.
            {{~/assistant}}
        '''

        practice_round = guidance(practice_template, self.llm())  # type: ignore
        practice_round = practice_round(results=demo_results)  # type: ignore

        template = '''
            {{#system~}}
            You are a helpful assistant.
            {{~/system}}
            {{#user~}}
            From now on, whenever your response depends on factual information, please search the web using the function <search>query</search> before responding.
            I will then paste web results in, and you can respond. But please respond as if I have not seen the results.
            {{~/user}}
            {{#assistant~}}
            Ok, I will do that. Let's do a practice round.
            {{~/assistant}}
            {{>practice_round}}
            {{#user~}}
            That was great, now let's do another one.
            {{~/user}}
            {{#assistant~}}
            Ok, I'm ready.
            {{~/assistant}}
            {{#user~}}
            {{user_query}}
            {{~/user}}
            {{#assistant~}}
            {{gen "query" stop="</search"}}{{#if (is_search query)}}</search>{{/if}}
            {{~/assistant}}
            {{#if (is_search query)}}
            {{#user~}}
            Search results: {{#each (search query)}}
            <result>
            {{this.text}}
            </result>{{/each}}
            {{~/user}}
            {{#assistant~}}
            {{gen "answer"}}
            {{~/assistant}}
            {{/if}}
        '''

        return Helpers.execute_llm_template(
            template,
            self.llm(),
            practice_round=practice_round,
            search=self.search,
            is_search=self.is_task,
            user_query=query
        )


class NewsAgent(Agent):
    def __init__(
        self,
        searcher: Searcher,
        llm: Callable[[], LLM],
    ):
        self.searcher = searcher
        self.llm_call: Callable[[], LLM] = llm
        self.llm_instance: Optional[LLM] = None

    def llm(self):
        if not self.llm_instance:
            self.llm_instance = self.llm_call()
        return self.llm_instance

    def perform_task(self, task: str, **kwargs) -> str:
        logging.debug('NewsAgent.perform_task() query={}'.format(task))
        results = self.searcher.search_news(task)
        for result in results:
            rich.print('[bold]Result[/bold]: {}'.format(result['title']))

        return ' '.join([result['title'] for result in results])

    def invoked_by(self) -> str:
        return '<news>query</news>'

    def instruction(self) -> str:
        return 'From now on, whenever your response depends on the latest news of the world, please search news using the function {} before responding.'.format(self.invoked_by())

    def is_task(self, task: str):
        if '<news>' in task:
            return True
        else:
            return False

class EdgarAgent(Agent):
    def perform_task(self, task: str, **kwargs) -> str:
        logging.debug('EdgarAgent.perform_task() query={}'.format(task))
        return EdgarHelpers.get_latest_form_text(
            sec_api_key=os.environ.get('SEC_API_KEY'),  # type: ignore
            symbol=task,
            form_type=EdgarHelpers.FormType.TENQ
        )

    def invoked_by(self) -> str:
        return '<edgar>symbol</edgar>'

    def instruction(self) -> str:
        return 'From now on, whenever your response depends on the latest financial statement of a company, please get that using the function {} before responding.'.format(self.invoked_by())

    def is_task(self, task: str):
        if '<edgar>' in task:
            return True
        else:
            return False


class GenericAgent(Agent):
    def __init__(
            self,
            llm: Callable[[], LLM],
    ):
        self.llm_call: Callable[[], LLM] = llm
        self.llm_instance: Optional[LLM] = None

    def llm(self):
        if not self.llm_instance:
            self.llm_instance = self.llm_call()
        return self.llm_instance

    def perform_task(self, query: str, **kwargs) -> str:
        logging.debug('GenericAgent.perform_task() query={}'.format(query))
        template = '''
            {{llm.default_system_prompt}}
            {{#user~}}
            {{query}}
            {{~/user}}
            {{#assistant~}}
            {{gen 'answer' temperature=0}}
            {{~/assistant~}}
        '''
        return Helpers.execute_llm_template(template, self.llm(), query=query)['answer']

    def is_task(self, task: str):
        return not ('<' in task and '>' in task)

    def invoked_by(self) -> str:
        return 'I have a generic query for you.'


class LangChainExecutor(Executor):
    def __init__(
            self,
            openai_key: str,
            temperature: float = 0.6,
            verbose: bool = True,
    ):
        self.openai_key = openai_key
        self.temperature = temperature
        self.verbose = verbose
        self.gpt = langchain_OpenAI(temperature=temperature)  # type: ignore
        # Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
        self.tools = load_tools(
            ["serpapi", "llm-math", "news-api"],
            llm=self.gpt,
            news_api_key='ecdb70595c9c4464ac70d338610c9390'
        )
        self.chat = False

        self.context: Any = None

    def name(self) -> str:
        return 'langchain'

    def chat_context(self, chat: bool):
        self.chat = chat

    def parse_action(self, query: str) -> Tuple[str, Callable]:
        if '.pdf' in query:
            url = Helpers.extract_token(query, '.pdf')
            from urllib.parse import urlparse

            from langchain.document_loaders import PyPDFLoader

            documents = []

            result = urlparse(url)
            if result.scheme == '' or result.scheme == 'file':
                logging.debug('LangChainExecutor.parse_action loading and splitting {}'.format(result))
                loader = PyPDFLoader(result.path)
                documents = loader.load_and_split()

                if len(documents) == 0:
                    # use tesseract to do the parsing instead
                    import pdf2image
                    import pytesseract
                    from langchain.document_loaders import TextLoader
                    from pytesseract import Output, TesseractError

                    text: List[str] = []
                    images = pdf2image.convert_from_path(result.path)  # type: ignore
                    for pil_im in images:
                        ocr_dict = pytesseract.image_to_data(pil_im, output_type=Output.DICT)
                        text.append(' '.join(ocr_dict['text']))

                    with tempfile.NamedTemporaryFile('w') as temp:
                        temp.write('\n'.join(text))
                        temp.flush()

                        loader = TextLoader(temp.name)
                        documents = loader.load()

            elif result.scheme == 'https' or result.scheme == 'http':
                import io
                response = requests.get(url=url, timeout=20)
                with tempfile.NamedTemporaryFile(suffix='.pdf', mode='wb', delete=True) as temp_file:
                    temp_file.write(response.content)
                    loader = PyPDFLoader(temp_file.name)
                    documents = loader.load_and_split()

            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(
                model='text-embedding-ada-002',
                openai_api_key=self.openai_key,
            )  # type: ignore

            logging.debug('LangChainExecutor.parse_action FAISS.from_documents {}'.format(result))
            docsearch = FAISS.from_documents(texts, embeddings)
            llm = ChatOpenAI(
                openai_api_key=self.openai_key,
                model_name='gpt-3.5-turbo',  # type: ignore
                temperature=self.temperature,
            )  # type: ignore

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=docsearch.as_retriever()
            )

            self.context = qa
            return (query, qa.run)
        elif 'edgar(' in query:
            symbol = Helpers.in_between(query, 'edgar(', ')')
            logging.debug('loading firefox and getting the latest 10Q for {}'.format(symbol))
            report_text = EdgarAgent().perform_task(symbol)
            with open('edgar.text', 'w') as f:
                f.write(report_text)

            query = Helpers.strip_between(query, 'edgar(', ')')

            documents = []

            from langchain.document_loaders.text import TextLoader

            with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=True) as t:
                t.write(report_text)
                t.seek(0)
                logging.debug('parsing in BeautifulSoup')
                html_loader = TextLoader(t.name)
                data = html_loader.load()
                documents = html_loader.load_and_split()

            logging.debug('token splitting')
            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(
                model='text-embedding-ada-002',
                openai_api_key=self.openai_key,
            )  # type: ignore

            logging.debug('LangChainExecutor.parse_action FAISS.from_documents {}'.format(symbol))
            docsearch = FAISS.from_documents(texts, embeddings)
            llm = ChatOpenAI(
                openai_api_key=self.openai_key,
                model_name='gpt-3.5-turbo',  # type: ignore
                temperature=self.temperature,
            )  # type: ignore

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=docsearch.as_retriever()
            )

            self.context = qa
            return (query, qa.run)
        else:
            # generic langchain
            agent = initialize_agent(self.tools, self.gpt, agent='zero-shot-react-description', verbose=self.verbose)  # type: ignore
            self.context = agent

            def execute_agent(query: str):
                result = agent({'input': query})
                return result['output']
            return (query, execute_agent)

    def execute(self, query: Union[str, List[Dict]], data: str) -> ExprNode:
        logging.debug('LangChainExecutor.execute_query query={}'.format(query))

        # todo check to see if we've got repl context
        prompt, action = self.parse_action(query)
        return Result(result=action(prompt))

    def can_execute(self, query: Union[str, List[Dict]], data: str) -> bool:
        return True

class PromptStrategy(Enum):
    THROW = 'throw'
    SEARCH = 'search'
    SUMMARIZE = 'summarize'

class OpenAIExecutor(Executor):
    def __init__(
        self,
        openai_key: str,
        chat: bool = False,
        verbose: bool = True,
        max_function_calls: int = 5,
    ):
        self.openai_key = openai_key
        # self.agents = agents
        self.verbose = verbose
        self.model = 'gpt-3.5-turbo-0613'
        self.agents = [
            WebHelpers.get_url,
            WebHelpers.get_news,
            WebHelpers.get_url_firefox,
            WebHelpers.search_news,
            WebHelpers.search_internet,
            WebHelpers.get_linkedin_profile,
            EdgarHelpers.get_latest_form_text,
            PdfHelpers.parse_pdf
        ]
        self.chat = chat
        self.messages: List[Dict] = []
        self.max_tokens = 4000
        self.max_function_calls = max_function_calls

    def name(self) -> str:
        return 'openai'

    def chat_context(self, chat: bool):
        self.chat = chat

    def __chat_completion_request(
        self,
        messages: List[Dict],
        functions: List[Dict] = [],
    ) -> List[Dict]:
        message_results: List[Dict] = []

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=functions,
        )

        message = response['choices'][0]['message']  # type: ignore
        message_results.append(message)
        counter = 1

        # loop until function calls are all resolved
        while message.get('function_call') and counter < self.max_function_calls:
            function_name = message['function_call']['name']
            function_args = json.loads(message['function_call']['arguments'])
            logging.debug('__chat_completion_request function_name={} function_args={}'.format(function_name, function_args))

            # Step 3, call the function
            # Note: the JSON response from the model may not be valid JSON
            func: Callable | None = Helpers.first(lambda f: f.__name__ == function_name, self.agents)

            if not func:
                return []

            # check for enum types and marshal from string to enum
            for p in inspect.signature(func).parameters.values():
                if p.annotation != inspect.Parameter.empty and p.annotation.__class__.__name__ == 'EnumMeta':
                    function_args[p.name] = p.annotation(function_args[p.name])

            try:
                function_response = func(**function_args)
            except Exception as e:
                function_response = 'The function could not execute. It raised an exception: {}'.format(e)

            # Step 4, send model the info on the function call and function response
            message_results.append({
                "role": "function",
                "name": function_name,
                "content": function_response,
            })

            second_response = openai.ChatCompletion.create(  # type: ignore
                model=self.model,
                messages=messages + message_results,
            )

            message = second_response['choices'][0]['message']  # type: ignore
            message_results.append(message)
            counter += 1

        return message_results

    def execute_query(
        self,
        query: Union[str, List[Dict]],
        data: str,
        prompt_strategy: PromptStrategy = PromptStrategy.THROW
    ) -> ExprNode:
        logging.debug('execute_query query={}'.format(query))

        functions = [Helpers.get_function_description(f, True) for f in self.agents]

        function_system_message = '''
            Dont make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
            If a function returns a value that does not address the users request, you should call a different function.
        '''

        self.messages.append({'role': 'system', 'content': function_system_message})
        self.messages.append({'role': 'user', 'content': query})
        if data:
            self.messages.append({'role': 'user', 'content': data})

        chat_response = self.__chat_completion_request(
            self.messages,
            functions,
        )

        if len(chat_response) == 0:
            return Result(error='The model could not execute the query.')

        for message in chat_response:
            self.messages.append(message)

        if not self.chat:
            self.messages = []

        conversation: List[AstNode] = []
        for message in self.messages:
            if message['role'] == 'user':
                conversation.append(Prompt(message['content']))
            elif message['role'] == 'system':
                conversation.append(SystemPrompt(message['content']))
            elif message['role'] == 'assistant':
                conversation.append(Assistant(message['content']))

        return Result(result={'answer': chat_response[-1]['content']}, conversation=conversation)

    def can_execute(self, query: Union[str, List[Dict]]) -> bool:
        return True

    def execute(self, query: Union[str, List[Dict]], data: str) -> ExprNode:
        return self.execute_query(query, data)


class LocalLLMExecutor(Executor):
    def __init__(
        self,
        agents: List[Agent],
        llm: Callable[[], LLM],
        verbose: bool = True,
    ):
        self.agents = agents
        self.llm: Optional[LLM] = None
        self.llm_call = llm
        self.verbose = verbose
        self.chat = False

    def name(self) -> str:
        return 'local'

    def chat_context(self, chat: bool):
        self.chat = chat

    def init_llm(self):
        if not self.llm:
            self.llm = self.llm_call()
        return self.llm

    def execute_raw(self, prompt: str, post_prompt: str):
        logging.debug('execute_raw prompt={}, post_prompt={}'.format(prompt, post_prompt))
        template = '''
            {{#system~}}
            {{llm.default_system_prompt}}
                               {{/system~}}
            {{#user~}}
            {{prompt}}
            {{post_prompt}}
            {{~/user}}

            {{#assistant~}}
            {{gen 'answer' temperature=0.7}}
            {{~/assistant~}}
        '''
        return Helpers.execute_llm_template(template, self.init_llm(), prompt=prompt, post_prompt=post_prompt)

    def execute_query(self, query: str):
        logging.debug('execute_query query={}'.format(query))
        agent_invoke_strings = [
            {'agent': agent, 'invoke_string': agent.invoked_by()} for agent in self.agents
        ]

        template = '''
            {{llm.default_system_prompt}}
            {{#user~}}
            I'd like you to answer a query for me. You have access to helper agents to help you answer the question. You can invoke these agents by calling the following functions:
            {{~#each agents}}
            {{this.invoke_string}}
            {{/each}}
            Are you ready for my query?
            {{~/user}}
            {{#assistant~}}
            {{gen 'ready'}}
            {{~/assistant~}}
            {{#user~}}
            '''

    def execute(self, query: Union[str, List[Dict]]) -> ExprNode:
        return Result()

    def can_execute(self, query: str | List[Dict], data: str) -> bool:
        return super().can_execute(query, data)

class WizardMegaExecutor(Executor):
    def __init__(
        self,
        agents: List[Agent],
        verbose: bool = True,
    ):
        self.agents = agents
        self.verbose = verbose

    def name(self) -> str:
        return 'wizardmega'

    def execute_raw(self, prompt: str, post_prompt: str):
        logging.debug('execute_raw prompt={}, post_prompt={}'.format(prompt, post_prompt))
        template = '''
            {{#system~}}
            {{llm.default_system_prompt}}
            {{/system~}}
            {{#user~}}
            {{prompt}}
            {{post_prompt}}
            {{~/user}}

            {{#assistant~}}
            {{gen 'answer' temperature=0.7}}
            {{~/assistant~}}
        '''
        return Helpers.execute_llm_template(template, self.init_llm(), prompt=prompt, post_prompt=post_prompt)

    def execute_query(self, query: str):
        logging.debug('execute_query query={}'.format(query))
        agent_invoke_strings = [
            {'agent': agent, 'invoke_string': agent.invoked_by()} for agent in self.agents
        ]

        template = '''
            {{llm.default_system_prompt}}
            {{#user~}}
            I'd like you to answer a query for me. You have access to helper agents to help you answer the question. You can invoke these agents by calling the following functions:
            {{~#each agents}}
            {{this.invoke_string}}
            {{/each}}
            Are you ready for my query?
            {{~/user}}
            {{#assistant~}}
            {{gen 'ready'}}
            {{~/assistant~}}
            {{#user~}}
            {{query}}
            {{~/user}}
            {{#assistant~}}
            {{gen 'answer'}}
            {{~/assistant~}}
        '''
        answer = Helpers.execute_llm_template(template, self.init_llm(), agents=agent_invoke_strings, query=query)
        return answer

    def can_execute(self, query: Union[str, List[Dict]]) -> bool:
        return True

    def execute(self, query: str) -> Result:
        if 'raw: ' in query:
            return Result(result=self.execute_raw(query.replace('raw: ', ''), '')['answer'])
        else:
            return Result(result=self.execute_query(query)['answer'])


class Parser():
    def __init__(
        self,
        execution_contexts: List[Executor],
        chat_mode: bool = False,
        current_chat_context: List[str] = [],
    ):
        self.execution_contexts = execution_contexts
        self.chat_context: List[str] = current_chat_context
        self.chat_mode = chat_mode

    def parse(
        self,
        prompt: str,
        data: str,
        max_tokens: int = 4000,
        prompt_strategy: PromptStrategy = PromptStrategy.SEARCH,
        max_api_calls: int = 5,
    ) -> List[ExprNode]:
        executor = self.execution_contexts[0]

        results: List[ExprNode] = []

        if Helpers.calculate_tokens(prompt) > max_tokens and prompt_strategy == PromptStrategy.THROW:
            raise Exception('Prompt and data too long: {}, {}'.format(prompt, data))

        elif Helpers.calculate_tokens(prompt) > max_tokens and prompt_strategy == PromptStrategy.SEARCH:
            sections = Helpers.chunk_and_rank(prompt, data, max_chunk_length=max_tokens)
            calls = []
            for section in sections:
                calls.append(
                    LLMCall(
                        prompt=Prompt(prompt),
                        data=Data(data),
                        system=SystemPrompt(),
                        executor=executor,
                    )
                )
            results.append(ChainedCall(calls))

        elif Helpers.calculate_tokens(prompt) > max_tokens and prompt_strategy == PromptStrategy.SUMMARIZE:
            data_chunks = Helpers.split_text_into_chunks(data)
            if max_api_calls == 0:
                max_api_calls = len(data_chunks)

            calls = []
            for chunk in data_chunks[0:max_api_calls]:
                calls.append(
                    LLMCall(
                        prompt=Prompt(prompt),
                        data=Data(chunk),
                        system=SystemPrompt(),
                        executor=executor,
                    )
                )
            results.append(ChainedCall(calls))

        else:
            results.append(
                LLMCall(
                    prompt=Prompt(prompt),
                    data=Data(data),
                    system=SystemPrompt(),
                    executor=executor,
                )
            )

        return results

    def parse_tree_execute(self, nodes: List[ExprNode]) -> List[Result]:
        def __increment_counter():
            nonlocal counter
            nonlocal node
            counter += 1
            if counter >= len(nodes):
                node = None
            else:
                node = nodes[counter]

        if len(nodes) <= 0:
            raise ValueError('No nodes to execute')

        results: List[Result] = []

        counter = 0
        node: Optional[ExprNode] = nodes[counter]

        while node is not None and isinstance(node, ExprNode):
            if isinstance(node, LLMCall):
                execute_node: ExprNode = node.executor.execute(node.prompt.prompt, node.data.data)
                if type(execute_node) is Result:
                    results.append(execute_node)
                    __increment_counter()
                elif type(execute_node) is ExprNode:
                    node = execute_node

            elif isinstance(node, ChainedCall):
                for call in node.calls:
                    if type(call) is LLMCall:
                        execute_node = call.executor.execute(call.prompt.prompt, call.data.data)
                        if type(execute_node) is Result:
                            results.append(execute_node)
                            __increment_counter()
                        elif type(execute_node) is ExprNode:
                            node = execute_node
            # todo: ensure function call works here etc.
            elif isinstance(node, Result):
                results.append(node)
                __increment_counter()

        return results


class Repl():
    def __init__(
        self,
        executors: List[Executor]
    ):
        self.executors: List[Executor] = executors
        self.agents: List[Agent] = [DataProvider()]

    def print_response(self, result: Union[str, Result]):
        if type(result) is str:
            rich.print(f'[bold cyan]{result}[/bold cyan] ')

        elif type(result) is Result:
            rich.print('[bold cyan]Conversation:[/bold cyan] ')
            for message in result.conversation:
                rich.print('  ' + str(message))
            rich.print(f'[bold cyan]Answer:[/bold cyan]')
            if type(result.result) is str:
                rich.print(f'{result.result}')
            elif type(result.result) is dict and 'answer' in result.result:
                rich.print('{}'.format(cast(dict, result.result)['answer']))
            else:
                rich.print('{}'.format(str(result.result)))
        else:
            rich.print(str(result))

    def repl(self):
        console = rich.console.Console()
        history = FileHistory(".repl_history")

        rich.print()
        rich.print('[bold]I am a helpful assistant.[/bold]')
        rich.print()

        executor_contexts = self.executors
        executor_names = [executor.name() for executor in executor_contexts]

        current_context = 'openai'
        parser = Parser(
            execution_contexts=executor_contexts,
            chat_mode=False,
            current_chat_context=[]
        )

        while True:
            try:
                query = prompt('prompt>> ', history=history, enable_history_search=True, vi_mode=True)
                data = prompt('data>> ', history=history, enable_history_search=True, vi_mode=True)

                if query is None or query == '':
                    continue

                if 'exit' in query:
                    sys.exit(0)

                if '/context' in query:
                    context = Helpers.in_between(query, '/context', '\n').strip()
                    print(context)
                    if context in executor_names:
                        current_context = context
                        executor_contexts = [executor for executor in self.executors if executor.name() == current_context]
                        rich.print('Current context: {}'.format(current_context))
                    elif context == '':
                        rich.print([e.name() for e in self.executors])
                    else:
                        rich.print('Invalid context: {}'.format(current_context))
                    continue

                if '/chat' in query:
                    chat = Helpers.in_between(query, '/chat', '\n').strip()
                    if chat == 'true':
                        for executor in executor_contexts:
                            executor.chat_context(True)
                            parser.chat_mode = True
                    elif chat == 'false':
                        for executor in executor_contexts:
                            executor.chat_context(False)
                            parser.chat_mode = False
                    else:
                        rich.print('Invalid chat value: {}'.format(chat))
                    continue

                if '/agents' in query:
                    rich.print('Agents:')
                    for agent in self.agents:
                        rich.print('  [bold]{}[/bold]'.format(agent.__class__.__name__))
                        rich.print('    {}'.format(agent.instruction()))
                    continue

                if '/any' in query:
                    executor_contexts = self.executors
                    continue

                # for agent in self.agents:
                #     if agent.is_task(query):
                #         result = agent.perform_task(query)

                # result = self.execute(query=query, executors=executor_contexts)
                # self.print_response(result)

                expr_tree = parser.parse(prompt=query, data=data)
                results = parser.parse_tree_execute(expr_tree)
                for result in results:
                    self.print_response(result)
                rich.print()

            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                break

            except Exception:
                console.print_exception(max_frames=10)


def start(context: Optional[str], verbose: bool):
    openai_key = os.environ.get('OPENAI_API_KEY') or ''
    serpapi_key = os.environ.get('SERPAPI_API_KEY') or ''

    def local():
        s = SerpAPISearcher(api_key=serpapi_key)
        search_agent = LocalSearchAgent(s, llm=load_vicuna)
        news_agent = NewsAgent(s, llm=load_vicuna)
        local_executor = LocalLLMExecutor(agents=[search_agent, news_agent], llm=load_vicuna, verbose=verbose)
        return local_executor

    def langchain_executor():
        openai_executor = LangChainExecutor(openai_key, verbose=verbose)
        return openai_executor

    def openai_executor():
        openai_executor = OpenAIExecutor(openai_key, verbose=verbose)
        return openai_executor

    executors = {
        'openai': openai_executor(),
        'local': local(),
        'langchain': langchain_executor(),
    }

    if context:
        executor = executors[context]
        repl = Repl([executor])
        repl.repl()
    else:
        repl = Repl(list(executors.values()))
        repl.repl()


@click.command()
@click.option('--context', type=str, required=False)
@click.option('--openai-key', type=str, required=False)
@click.option('--serpapi-key', type=str, required=False)
@click.option('--verbose', type=bool, default=True)
def main(
    context: Optional[str],
    openai_key: Optional[str],
    serpapi_key: Optional[str],
    verbose: bool,
):
    if openai_key is None:
        openai_key = os.environ.get('OPENAI_API_KEY')
    if serpapi_key is None:
        serpapi_key = os.environ.get('SERPAPI_API_KEY')

    if openai_key is None:
        rich.print('[red]Error[/red]: OpenAI API key not found. Please set it as an environment variable or pass it in with --openai-key.')
        sys.exit(1)
    if serpapi_key is None:
        rich.print('[red]Error[/red]: SerpAPI API key not found. Please set it as an environment variable or pass it in with --serpapi-key.')
        sys.exit(1)

    if not verbose:
        import logging as logging_library
        logging_library.getLogger().setLevel(logging_library.ERROR)

    start(context, verbose)

if __name__ == '__main__':
    main()
