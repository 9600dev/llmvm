import os
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from itertools import cycle, islice
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import nltk
import requests
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
from pypdf import PdfReader
from rich.logging import RichHandler
from rich.traceback import install
from sec_api import ExtractorApi, QueryApi
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from sentence_transformers import SentenceTransformer, util
from serpapi import BingSearch, GoogleSearch
from torch import Tensor

from helpers.logging_helpers import setup_logging
from helpers.pdf import PdfHelpers
from helpers.search import SerpAPISearcher

logging = setup_logging()


class WebHelpers():
    @staticmethod
    def convert_html_to_markdown(html: str) -> str:
        """Converts html to markdown using pandoc"""
        logging.debug('convert_html_to_markdown')
        with tempfile.NamedTemporaryFile(suffix='.html', mode='w', delete=True) as temp_file:
            temp_file.write(html)

            command = "pandoc -s -i "
            command += temp_file.name
            command += " -t markdown | grep -v '^:' | grep -v '^```' | grep -v '<!-- --->' | sed -e ':again' -e N -e '$!b again' -e 's/{[^}]*}//g' | grep -v 'data:image'"
            result = (os.popen(command).read())

            lines = []
            for line in result.splitlines():
                stripped = line.strip()
                if stripped != '':
                    if stripped == '<div>' or stripped == '</div>':
                        continue

                    if stripped == '[]' or stripped == '[[]]':
                        continue

                    if stripped.startswith('![]('):
                        continue

                    if stripped.startswith('[]') and stripped.replace('[]', '').strip() == '':
                        continue

                    if stripped.startswith('[\\') and stripped.replace('[\\', '').strip() == '':
                        continue

                    if stripped.startswith(']') and stripped.replace(']', '').strip() == '':
                        continue

                    if stripped.startswith('[') and stripped.replace('[', '').strip() == '':
                        continue

                    lines.append(stripped)
            return '\n'.join(lines)

    @staticmethod
    def __search_helper(
        query: str,
        searcher: Callable[[str], List[Dict[str, str]]],
        parser: Callable[[str], str],
    ) -> str:
        return_results = []
        search_results = searcher(query)

        for result in search_results:
            return_results.append(parser(result['link']))

        return ' '.join(return_results)

    @staticmethod
    def search_internet(query: str, total_links_to_return: int = 3) -> str:
        '''Searches the internet for a query and returns the text of the top results'''
        searcher = SerpAPISearcher(link_limit=total_links_to_return)
        return WebHelpers.__search_helper(query, searcher.search_internet, WebHelpers.get_url)

    @staticmethod
    def search_news(query: str, total_links_to_return: int = 3) -> str:
        '''Searches the current and historical news for a query and returns the text of the top results'''
        searcher = SerpAPISearcher(link_limit=total_links_to_return)
        return WebHelpers.__search_helper(query, searcher.search_news, WebHelpers.get_news)

    @staticmethod
    def get_url_firefox(url: str) -> str:
        """
        Extracts the text from a url using the Firefox browser.
        This is useful for hard to extract text, an exception thrown by the other functions,
        or when searching/extracting from sites that require logins liked LinkedIn, Facebook, Gmail etc.
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.wait import WebDriverWait

        logging.debug('WebHelpers.get_url_firefox: {}'.format(url))
        profile_directory = '/home/joelp/.mozilla/firefox/cp6sgb0s.selenium'

        options = FirefoxOptions()
        options.headless = False

        profile = webdriver.FirefoxProfile(profile_directory)
        service_args = ['--marionette-port', '2828']

        # options.add_argument("--headless")
        with webdriver.Firefox(firefox_profile=profile, options=options, service_args=service_args) as driver:
            driver.get(url)
            driver.implicitly_wait(5)
            wait = WebDriverWait(driver, 5)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'p')))

            driver.execute_script('setTimeout(function() { return; }, 0);')
            return driver.page_source

    @staticmethod
    def pdf_url_firefox(url: str) -> str:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.wait import WebDriverWait

        logging.debug('WebHelpers.get_url_firefox: {}'.format(url))
        profile_directory = '/home/joelp/.mozilla/firefox/cp6sgb0s.selenium'

        options = FirefoxOptions()
        options.headless = False
        options.set_preference("print.always_print_silent", True)
        options.set_preference("print.printer_Mozilla_Save_to_PDF.print_to_file", True)
        options.set_preference("print_printer", "Mozilla Save to PDF")

        profile = webdriver.FirefoxProfile(profile_directory)
        service_args = ['--marionette-port', '2828']

        # options.add_argument("--headless")
        with webdriver.Firefox(firefox_profile=profile, options=options, service_args=service_args) as driver:
            driver.get(url)
            driver.implicitly_wait(5)
            wait = WebDriverWait(driver, 5)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'p')))

            driver.execute_script('setTimeout(function() { return; }, 0);')
            driver.execute_script('window.print();')
            driver.execute_script('setTimeout(function() { return; }, 0);')
            import time
            time.sleep(5)

        if os.path.exists('mozilla.pdf'):
            return os.path.abspath('mozilla.pdf')
        else:
            logging.debug('WebHelpers.get_url_firefox: pdf not found')
            return ''

    @staticmethod
    def get_linkedin_profile(linkedin_url: str) -> str:
        """Extracts the career information from a person's LinkedIn profile from a given LinkedIn url"""
        logging.debug('WebHelpers.get_linkedin_profile: {}'.format(linkedin_url))
        pdf_file = WebHelpers.pdf_url_firefox(linkedin_url)
        data = PdfHelpers.parse_pdf(pdf_file)
        return data

    @staticmethod
    def search_linkedin_profile(first_name: str, last_name: str, company_name: str) -> str:
        """Searches for the LinkedIn profile of a given person name and optional company name and returns the profile text"""
        searcher = SerpAPISearcher(link_limit=1)
        links: List[Dict] = searcher.search_internet('{} {} {} LinkedIn profile'.format(first_name, last_name, company_name))
        if len(links) > 0:
            return WebHelpers.get_linkedin_profile(links[0]['link'])
        else:
            return ''

    @staticmethod
    def get_news(url: str) -> str:
        """Extracts the text from a news article"""
        logging.debug('WebHelpers.get_news: {}'.format(url))
        nltk.download('punkt')

        config = Configuration()
        config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        article = Article(url=url, config=config)
        article.download()
        article.parse()
        return article.text

    @staticmethod
    def get_url(url: str) -> str:
        """Extracts the text from a url. Url can be a file, web url, or a pdf"""
        logging.debug('WebHelpers.get_url: {}'.format(url))

        text = ''

        result = urlparse(url)
        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                return PdfHelpers.parse_pdf(url)
            if '.htm' in result.path or '.html' in result.path:
                return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read())

        if result.scheme == 'http' or result.scheme == 'https':
            text = requests.get(url, timeout=10).text
            if text:
                return WebHelpers.convert_html_to_markdown(text)
            else:
                return WebHelpers.convert_html_to_markdown(WebHelpers.get_url_firefox(url))

        return ''
