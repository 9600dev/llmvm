import asyncio
import os
import re
import unicodedata
from typing import Callable, Dict, Generator
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

from llmvm.common.helpers import write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.server.tools.firefox import FirefoxHelpers
from llmvm.server.tools.pdf import PdfHelpers
from llmvm.server.tools.search import SerpAPISearcher

logging = setup_logging()

class IgnoringScriptConverter(MarkdownConverter):
    def convert_script(self, el, text, convert_as_inline):
        return ''

firefox_helpers = FirefoxHelpers()

class WebHelpers():
    @staticmethod
    def convert_html_to_markdown(html: str) -> str:
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

                elif '(data:image' in line and ')' in line:
                    # remove data:image
                    lines.append(re.sub(r'\(data:image[^\)]+\)', '', line))
                else:
                    lines.append(line)
                    blank_counter = 0
            return '\n'.join(lines)

        logging.debug(f'WebHelpers.convert_html_to_markdown_soup: {html[:25]}')
        soup = BeautifulSoup(html, features='lxml')

        for data in soup(['style', 'script']):
            data.decompose()

        result = IgnoringScriptConverter().convert_soup(soup)
        cleaned_result = clean_markdown(result)
        return unicodedata.normalize('NFKD', cleaned_result)

    @staticmethod
    def search_helper(
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
        '''
        Searches the internet for a query and returns a string with the markdown text results.
        Returns the top 'pages_to_include' results.
        '''
        searcher = SerpAPISearcher()
        return WebHelpers.search_helper(query, searcher.search_internet, WebHelpers.get_url, pages_to_include)

    @staticmethod
    def pdf_url_firefox(url: str) -> str:
        """Gets a pdf version of the url using the Firefox browser."""
        return asyncio.run(firefox_helpers.pdf_url(url))

    @staticmethod
    def get_linkedin_profile(linkedin_url: str) -> str:
        """Extracts the career information from a person's LinkedIn profile from a given LinkedIn url"""
        logging.debug('WebHelpers.get_linkedin_profile: {}'.format(linkedin_url))

        asyncio.run(firefox_helpers.goto(linkedin_url))
        asyncio.run(firefox_helpers.wait_until_text('Experience'))
        pdf_file = asyncio.run(firefox_helpers.pdf())
        data = PdfHelpers.parse_pdf(pdf_file)
        os.remove(pdf_file)
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
        logging.debug('WebHelpers.get_news_url: {}'.format(url))
        raise ValueError('Not implemented')

    @staticmethod
    def get_url(url: str) -> str:
        """
        Connects to and downloads the text content from a url and returns the text content.
        Url can be a http or https web url or a filename and directory location.
        """
        logging.debug('WebHelpers.get_url: {}'.format(url))

        result = urlparse(url)
        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                return PdfHelpers.parse_pdf(url)
            if '.htm' in result.path or '.html' in result.path:
                return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read())

        elif (result.scheme == 'http' or result.scheme == 'https') and '.pdf' in result.path:
            return PdfHelpers.parse_pdf(url)

        elif result.scheme == 'http' or result.scheme == 'https':
            return WebHelpers.convert_html_to_markdown(asyncio.run(firefox_helpers.get_url(url)))

        return ''
