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
from llmvm.common.objects import Content, MarkdownContent
from llmvm.server.tools.firefox import FirefoxHelpers
from llmvm.server.tools.chrome import ChromeHelpers
from llmvm.server.tools.search import SerpAPISearcher

logging = setup_logging()

class VerboseConverter(MarkdownConverter):
    def convert_script(self, el, text, convert_as_inline):
        return ''

    def convert_input(self, el, text, convert_as_inline):
        return str(el)

    def convert_textarea(self, el, text, convert_as_inline):
        return str(el)

    def convert_label(self, el, text, convert_as_inline):
        return str(el)

class WebHelpers():
    @staticmethod
    def convert_html_to_markdown(html: str, url: str = '') -> MarkdownContent:
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

        result = VerboseConverter().convert_soup(soup)
        cleaned_result = clean_markdown(result)
        return MarkdownContent(sequence=unicodedata.normalize('NFKD', cleaned_result), url=url)

    @staticmethod
    def get_linkedin_profile(linkedin_url: str) -> str:
        """
        Extracts the career information from a person's LinkedIn profile from a given LinkedIn url and returns
        the career information as a string.
        """
        from llmvm.common.pdf import PdfHelpers
        logging.debug('WebHelpers.get_linkedin_profile: {}'.format(linkedin_url))

        firefox_helpers = ChromeHelpers()
        asyncio.run(firefox_helpers.goto(linkedin_url))
        asyncio.run(firefox_helpers.wait_until_text('Experience'))
        pdf_file = asyncio.run(firefox_helpers.pdf())
        data = PdfHelpers.parse_pdf(pdf_file)
        os.remove(pdf_file)
        asyncio.run(firefox_helpers.close())
        return data

    @staticmethod
    def search_linkedin_profile(first_name: str, last_name: str, company_name: str) -> str:
        """
        Searches for the LinkedIn profile of a given first name and last name and optional company name and returns the
        LinkedIn profile information as a string. If you call this method you do not need to call get_linkedin_profile().
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
    def get_url(url: str) -> Content:
        """
        Connects to and downloads the text content from a url and returns the text content.
        Url can be a http or https web url or a filename and directory location.
        """
        # todo this is redundant with ContentDownloader
        from llmvm.server.base_library.content_downloader import \
            ContentDownloader
        logging.debug('WebHelpers.get_url: {}'.format(url))
        downloader = ContentDownloader(url)
        return downloader.download()
