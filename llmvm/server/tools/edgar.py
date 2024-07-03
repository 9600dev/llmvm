import datetime as dt
import os
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup
from sec_api import ExtractorApi, QueryApi, RenderApi

from llmvm.common.logging_helpers import setup_logging
from llmvm.server.tools.webhelpers import WebHelpers

logging = setup_logging()


class EdgarHelpers():
    """Helper functions for working with the SEC Edgar API"""

    # class FormType(Enum):
    #     TENK = '10-K'
    #     TENQ = '10-Q'
    #     EIGHTK = '8-K'
    #     ANY = 'ANY'

    #     def __str__(self):
    #         return self.value

    @staticmethod
    def get_form_urls(
        ticker: str,
        form_type: Optional[str] = None,
        start_date: dt.datetime = dt.datetime.now() - dt.timedelta(days=180),
        end_date: dt.datetime = dt.datetime.now(),
    ):
        sec_api_key = os.environ.get('SEC_API_KEY')

        logging.debug('get_form_urls {}'.format(ticker))
        query_api = QueryApi(sec_api_key)
        str_start_date = start_date.strftime('%Y-%m-%d')
        str_end_date = end_date.strftime('%Y-%m-%d')

        q = 'ticker:' + ticker + ' '
        q += 'AND filedAt:{' + str_start_date + ' TO ' + str_end_date + '} '
        if form_type:
            q += 'AND formType:"' + str(form_type) + '"'

        query = {
            "query": {
                "query_string": {
                    "query": q
                }
            },
            "from": "0",
            "size": "10",
            "sort": [{"filedAt": {"order": "desc"}}]
        }

        result = query_api.get_filings(query)

        filings: List[Tuple[str, str, str]] = []

        for filing in result['filings']:  # type: ignore
            for document in filing['documentFormatFiles']:  # type: ignore
                if (
                    'description' in document
                    and (
                        '10-Q' in document['description']
                        or '10-K' in document['description']
                        or '8-K' in document['description']
                    )
                ):
                    filings.append((filing['periodOfReport'], document['description'], document['documentUrl']))
        return filings

    @staticmethod
    def extract_form_text(
        url: str,
        sections: List[str],
        type: str = 'text'
    ):
        logging.debug('extract_form_text{}'.format(url))
        sec_api_key = os.environ.get('SEC_API_KEY')

        parts = {}

        extractor = ExtractorApi(api_key=sec_api_key)
        for part in sections:
            logging.debug('extract_form_text part: {}'.format(part))
            parts[part] = extractor.get_section(url, part, type)
        return parts

    @staticmethod
    def get_form_filing(
        url: str,
    ):
        url = url.replace('ix?doc=/Archives', 'Archives')
        logging.debug('get_form_filing{}'.format(url))
        sec_api_key = os.environ.get('SEC_API_KEY')

        render = RenderApi(api_key=sec_api_key)
        return render.get_filing(url) or ''

    @staticmethod
    def get_form_text(
        url: str,
        form_type: str,
    ):
        sections_10q = [
            'part1item1', 'part1item2', 'part1item3', 'part1item4',
            'part2item1', 'part2item1a', 'part2item2', 'part2item3', 'part2item4',
            'part2item5', 'part2item6'
        ]

        sections_10k = [
            '1', '1A', '1B', '2', '3', '4', '5', '6', '7', '7A', '8', '9', '9A',
            '9B', '10', '11', '12', '13', '14'
        ]

        logging.debug('get_form_text: {}'.format(url))

        if form_type == '10-Q':
            sections = EdgarHelpers.extract_form_text(url, sections_10q)
            return '\n'.join([section for section in sections.values()])
        elif form_type == '10-K':
            sections = EdgarHelpers.extract_form_text(url, sections_10k)
            return '\n'.join([section for section in sections.values()])
        elif form_type == '8-K':
            soup = BeautifulSoup(EdgarHelpers.get_form_filing(url), 'html.parser')
            full_ref = url[0:url.rfind('/')]
            # find the exhibit links
            hrefs = [full_ref + '/' + a['href'] for a in soup.find_all('a')]
            forms = [EdgarHelpers.get_form_filing(href) for href in hrefs]
            parsed = [WebHelpers.convert_html_to_markdown(form, url=url).get_str() for form in forms]
            return '\n\n'.join(parsed)
        else:
            return ''

    @staticmethod
    def get_report(
        symbol: str,
        form_type: str,
        date: dt.datetime,
    ) -> str:
        """
        Gets the 10-Q, 10-K or 8-K report text for a given company symbol/ticker for a given date.
        This is useful to get financial information for a company,
        their current strategy, investments and risks. Use form_type = '' to
        get the latest form of any type. form_type can be '10-Q', '10-K' or '8-K'.
        date is a Python datetime.
        """
        logging.debug('get_report: {} {} {}'.format(symbol, form_type, str(date)))

        if not isinstance(date, dt.datetime) or not date:
            date = dt.datetime.now()

        urls = EdgarHelpers.get_form_urls(
            symbol,
            form_type,
            end_date=date + dt.timedelta(days=1)
        )
        if not urls:
            logging.debug('No urls found for {}'.format(symbol))
            return ''

        return EdgarHelpers.get_form_text(urls[0][2], form_type)

    @staticmethod
    def get_latest_form_text(
        symbol: str,
        form_type: str,
    ) -> str:
        """
        Gets the latest 10-Q, 10-K or 8-K text for a given company symbol/ticker.
        This is useful to get the latest financial information for a company,
        their current strategy, investments and risks. Use form_type = '' to
        get the latest form of any type. form_type can be '10-Q', '10-K' or '8-K'.
        """
        logging.debug('get_latest_form_text: {}'.format(symbol))

        urls = EdgarHelpers.get_form_urls(symbol, form_type)
        if not urls:
            logging.debug('No urls found for {}'.format(symbol))
            return ''

        return EdgarHelpers.get_form_text(urls[0][2], form_type)

    @staticmethod
    def get_latest_filing(
        symbol: str,
    ):
        urls = EdgarHelpers.get_form_urls(symbol)
        latest_form = urls[-1][2]
        form_type = urls[-1][1]
        return EdgarHelpers.get_form_text(latest_form, form_type)
