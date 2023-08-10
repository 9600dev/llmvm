import os
import tempfile
import unicodedata
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import nltk
import requests
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from newspaper import Article
from newspaper.configuration import Configuration
from selenium.webdriver.firefox.options import Options as FirefoxOptions

from helpers.firefox import FirefoxHelpers
from helpers.logging_helpers import setup_logging
from helpers.pdf import PdfHelpers
from helpers.search import SerpAPISearcher

logging = setup_logging()

class IgnoringScriptConverter(MarkdownConverter):
    def convert_script(self, el, text, convert_as_inline):
        return ''


class WebHelpers():
    @staticmethod
    def clean_markdown(markdown_text: str) -> str:
        lines = []
        blank_counter = 0
        for line in markdown_text.splitlines():
            if line == '' and blank_counter == 0:
                blank_counter += 1
                lines.append(line)

            elif line == '' and blank_counter >= 1:
                continue

            elif line == '<div>' or line == '</div>':
                continue

            elif line == '[]' or line == '[[]]':
                continue

            elif line == '*' or line == '* ' or line == ' *':
                continue

            elif line == '&starf;' or line == '&star;' or line == '&nbsp;':
                continue

            else:
                lines.append(line)
                blank_counter = 0
        return '\n'.join(lines)

    @staticmethod
    def convert_html_to_markdown(html: str) -> str:
        logging.debug('convert_html_to_markdown_soup')
        soup = BeautifulSoup(html, features='lxml')

        for data in soup(['style', 'script']):
            data.decompose()

        result = IgnoringScriptConverter().convert_soup(soup)
        cleaned_result = WebHelpers.clean_markdown(result)
        return unicodedata.normalize('NFKD', cleaned_result)

    # @staticmethod
    # def convert_html_to_markdown_old(html: str) -> str:
    #     """Converts html to markdown using pandoc"""
    #     logging.debug('convert_html_to_markdown')
    #     with tempfile.NamedTemporaryFile(suffix='.html', mode='w', delete=True) as temp_file:
    #         temp_file.write(html)

    #         command = "pandoc -s -i "
    #         command += temp_file.name
    #         command += " -t markdown | grep -v '^:' | grep -v '^```' | grep -v '<!-- --->' | sed -e ':again' -e N -e '$!b again' -e 's/{[^}]*}//g' | grep -v 'data:image'"
    #         result = (os.popen(command).read())

    #         lines = []
    #         for line in result.splitlines():
    #             stripped = line.strip()
    #             if stripped != '':
    #                 if stripped == '<div>' or stripped == '</div>':
    #                     continue

    #                 if stripped == '[]' or stripped == '[[]]':
    #                     continue

    #                 if stripped.startswith('![]('):
    #                     continue

    #                 if stripped.startswith('[]') and stripped.replace('[]', '').strip() == '':
    #                     continue

    #                 if stripped.startswith('[\\') and stripped.replace('[\\', '').strip() == '':
    #                     continue

    #                 if stripped.startswith(']') and stripped.replace(']', '').strip() == '':
    #                     continue

    #                 if stripped.startswith('[') and stripped.replace('[', '').strip() == '':
    #                     continue

    #                 lines.append(stripped)
    #         return '\n'.join(lines)

    @staticmethod
    def __search_helper(
        query: str,
        searcher: Callable[[str], Generator[Dict[str, str], None, None]],
        parser: Callable[[str], str],
        total_links_to_return: int,
    ) -> str:
        return_results = []
        search_results = searcher(query)

        for result in search_results:
            try:
                parser_result = parser(result['link']).strip()
                if parser_result:
                    return_results.append(parser_result)

                if len(return_results) >= total_links_to_return:
                    break

            except Exception as e:
                logging.error(e)
                pass

        return ' '.join(return_results)

    @staticmethod
    def get_content_by_search(query: str, pages_to_include: int = 4) -> str:
        '''Searches the internet for a query and returns a string with the entire text of all the top results'''
        searcher = SerpAPISearcher()
        return WebHelpers.__search_helper(query, searcher.search_internet, WebHelpers.get_url, pages_to_include)

    @staticmethod
    def get_news_by_search(query: str, pages_to_include: int = 4) -> str:
        '''Searches the current and historical news for a query and returns the entire text of the top results'''
        searcher = SerpAPISearcher()
        return WebHelpers.__search_helper(query, searcher.search_news, WebHelpers.get_news_url, pages_to_include)

    @staticmethod
    def get_url_firefox(url: str) -> str:
        """
        Extracts the text from a url using the Firefox browser.
        This is useful for hard to extract text, an exception thrown by the other functions,
        or when searching/extracting from sites that require logins liked LinkedIn, Facebook, Gmail etc.
        """
        return FirefoxHelpers.get_url(url)

    @staticmethod
    def get_url_firefox_via_pdf(url: str) -> str:
        """Extracts the career information from a person's LinkedIn profile from a given LinkedIn url"""
        logging.debug('WebHelpers.get_url_firefox_via_pdf: {}'.format(url))
        from selenium.webdriver.common.by import By

        firefox = FirefoxHelpers()
        firefox.goto(url)
        firefox.wait()
        pdf_file = firefox.print_pdf()
        data = PdfHelpers.parse_pdf(pdf_file)
        return data

    @staticmethod
    def pdf_url_firefox(url: str) -> str:
        """Gets a pdf version of the url using the Firefox browser."""
        return FirefoxHelpers().pdf_url(url)

    @staticmethod
    def get_linkedin_profile(linkedin_url: str) -> str:
        """Extracts the career information from a person's LinkedIn profile from a given LinkedIn url"""
        logging.debug('WebHelpers.get_linkedin_profile: {}'.format(linkedin_url))
        from selenium.webdriver.common.by import By

        firefox = FirefoxHelpers()
        firefox.goto(linkedin_url)
        firefox.wait_until_func(lambda driver: driver.find_elements(By.XPATH, "//*[contains(text(), 'Experience')]"))
        pdf_file = firefox.print_pdf()
        data = PdfHelpers.parse_pdf(pdf_file)
        return data

    @staticmethod
    def search_linkedin_profile(first_name: str, last_name: str, company_name: str) -> str:
        """
        Searches for the LinkedIn profile of a given first name and last name and optional company name and returns the
        LinkedIn profile. If you use this method you do not need to call get_linkedin_profile.
        """
        searcher = SerpAPISearcher()
        links = searcher.search_internet('{} {}, {} linkedin profile site:linkedin.com/in/'.format(
            first_name, last_name, company_name
        ))

        link_counter = 0
        # search for linkedin urls
        for link in links:
            if 'link' not in link:
                break

            if link_counter > 5:
                break

            if 'linkedin.com' in link['link']:
                return WebHelpers.get_linkedin_profile(link['link'])
            link_counter += 1
        return ''

    @staticmethod
    def get_news_url(url: str) -> str:
        """Extracts the news text from a given url"""
        logging.debug('WebHelpers.get_news_article: {}'.format(url))
        nltk.download('punkt', quiet=True)

        config = Configuration()
        config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        article = Article(url=url, config=config)
        article.download()
        article.parse()
        normalized_return = unicodedata.normalize('NFKD', article.text)
        return normalized_return

    @staticmethod
    def get_url(url: str, force_firefox: bool = False) -> str:
        """
        Connects to and downloads the text content from a url and returns the text content.
        Url can be a http or https web url or a filename and directory location.
        """
        logging.debug('WebHelpers.get_url: {}'.format(url))

        text = ''

        result = urlparse(url)
        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                return PdfHelpers.parse_pdf(url)
            if '.htm' in result.path or '.html' in result.path:
                return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read())

        elif (result.scheme == 'http' or result.scheme == 'https') and '.pdf' in result.path:
            return PdfHelpers.parse_pdf(url)

        elif result.scheme == 'http' or result.scheme == 'https':
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  # type: ignore
            }

            if force_firefox:
                return WebHelpers.convert_html_to_markdown(WebHelpers.get_url_firefox(url))

            text = requests.get(url, headers=headers, timeout=10, allow_redirects=True).text
            if text:
                return WebHelpers.convert_html_to_markdown(text)
            else:
                return WebHelpers.convert_html_to_markdown(WebHelpers.get_url_firefox(url))

        return ''
