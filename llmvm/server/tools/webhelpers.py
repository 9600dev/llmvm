import asyncio
import os
import re
from typing import cast
import unicodedata
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import warnings
import re

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from markdownify import MarkdownConverter

from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import Content, DownloadParams, MarkdownContent, TextContent
from llmvm.server.tools.chrome import ChromeHelpers
from llmvm.server.tools.search import SerpAPISearcher
from llmvm.common.helpers import Helpers

logging = setup_logging()

class VerboseConverter(MarkdownConverter):
    def chomp(self, text):
        """
        If the text in an inline tag like b, a, or em contains a leading or trailing
        space, strip the string and return a space as suffix of prefix, if needed.
        This function is used to prevent conversions like
        <b> foo</b> => ** foo**
        """
        prefix = ' ' if text and text[0] == ' ' else ''
        suffix = ' ' if text and text[-1] == ' ' else ''
        text = text.strip()
        return (prefix, suffix, text)

    def convert_script(self, el, text, convert_as_inline):
        return ''

    def convert_input(self, el, text, convert_as_inline):
        return str(el)

    def convert_textarea(self, el, text, convert_as_inline):
        return str(el)

    def convert_label(self, el, text, convert_as_inline):
        return str(el)

    def convert_a(self, el, text, convert_as_inline):
        prefix, suffix, text = self.chomp(text)
        if not text:
            return ''
        href = el.get('href')
        title = el.get('title')

        if href:
            href = Helpers.clean_tracking(href)
            href = Helpers.clean_url_params(href, limit=50)

        # For the replacement see #29: text nodes underscores are escaped
        if (self.options['autolinks']
                and text.replace(r'\_', '_') == href
                and not title
                and not self.options['default_title']):
            # Shortcut syntax
            return '<%s>' % href
        if self.options['default_title'] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r'\"') if title else ''
        result = '%s[%s](%s%s)%s' % (prefix, text, href, title_part, suffix) if href else text
        return result


class WebHelpers():
    @staticmethod
    def convert_xml_to_text(xml_string: str) -> str:
        xml_string = re.sub(r'\sxmlns[^"]+"[^"]+"', '', xml_string)

        root = ET.fromstring(xml_string)

        def get_text(element):
            text_parts = []

            if element.text and element.text.strip():
                text_parts.append(element.text.strip())

            for child in element:
                text_parts.extend(get_text(child))

                if child.tail and child.tail.strip():
                    text_parts.append(child.tail.strip())

            return text_parts

        all_text = get_text(root)

        cleaned_text = '\n'.join(all_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

        return cleaned_text

    @staticmethod
    def convert_html_to_markdown(html: str, url: str = '') -> MarkdownContent:
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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

            lines = [line.strip() for line in lines]
            return '\n'.join([line for line in lines if line])

        logging.debug(f'WebHelpers.convert_html_to_markdown_soup: {html[:25]}')

        if 'parlinfo.aph.gov.au' in url and '.xml' in url:
            logging.info('WebHelpers.convert_html_to_markdown: parlinfo.aph.gov.au')
            return MarkdownContent(sequence=[TextContent(WebHelpers.convert_xml_to_text(html))], url=url)

        soup = BeautifulSoup(html, features='lxml')

        for data in soup(['style', 'script']):
            data.decompose()

        result = VerboseConverter().convert_soup(soup)
        cleaned_result = clean_markdown(result)
        return MarkdownContent(sequence=[TextContent(unicodedata.normalize('NFKD', cleaned_result))], url=url)

    @staticmethod
    def get_linkedin_profile(linkedin_url: str) -> str:
        """
        Extracts the career information from a person's LinkedIn profile from a given LinkedIn url and returns
        the career information as a string.
        """
        from llmvm.common.pdf import PdfHelpers
        logging.debug('WebHelpers.get_linkedin_profile: {}'.format(linkedin_url))

        chrome_helpers = ChromeHelpers()
        asyncio.run(chrome_helpers.goto(linkedin_url))
        asyncio.run(chrome_helpers.wait_until_text('Experience'))
        pdf_file = asyncio.run(chrome_helpers.pdf())
        data = PdfHelpers.parse_pdf(pdf_file)
        os.remove(pdf_file)
        asyncio.run(chrome_helpers.close())
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
            WebAndContentDriver

        logging.debug('WebHelpers.get_url: {}'.format(url))
        downloader = WebAndContentDriver()
        download_params: DownloadParams = {'url': url, 'goal': '', 'search_term': ''}
        return downloader.download(download_params)

    @staticmethod
    def get_hackernews_latest() -> Content:
        """
        Returns the latest Hacker News articles as a big blob of string text (a loose json format).
        """
        from llmvm.server.tools.search_hn import SearchHN
        return TextContent('\n\n'.join([str(story) for story in cast(list, SearchHN().get_latest_stories())]))