import datetime as dt
import os
from enum import Enum
from itertools import cycle, islice
from typing import List, Tuple

from sec_api import ExtractorApi, QueryApi

from helpers.logging_helpers import setup_logging

logging = setup_logging()


class EdgarHelpers():
    """Helper functions for working with the SEC Edgar API"""

    class FormType(Enum):
        TENK = '10-K'
        TENQ = '10-Q'

        def __str__(self):
            return self.value

    @staticmethod
    def get_form_urls(
        ticker: str,
        form_type: FormType,
        start_date: dt.datetime = dt.datetime(2023, 1, 1),
        end_date: dt.datetime = dt.datetime.now(),
    ):
        sec_api_key = os.environ.get('SEC_API_KEY')

        logging.debug('get_form_urls {}'.format(ticker))
        query_api = QueryApi(sec_api_key)
        str_start_date = start_date.strftime('%Y-%m-%d')
        str_end_date = end_date.strftime('%Y-%m-%d')

        q = 'ticker:' + ticker + ' '
        q += 'AND filedAt:{' + str_start_date + ' TO ' + str_end_date + '} '
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
        filings: List[Tuple[str, str]] = []

        for filing in result['filings']:  # type: ignore
            for document in filing['documentFormatFiles']:  # type: ignore
                if 'description' in document and (document['description'] == '10-Q' or document['description'] == '10-K'):
                    filings.append((filing['periodOfReport'], document['documentUrl']))
        return filings

    @staticmethod
    def extract_form_text(
        url: str,
        sections: List[str],
    ):
        logging.debug('extract_form_text{}'.format(url))
        sec_api_key = os.environ.get('SEC_API_KEY')

        parts = {}

        extractor = ExtractorApi(api_key=sec_api_key)
        for part in sections:
            logging.debug('extract_form_text part: {}'.format(part))
            parts[part] = extractor.get_section(url, part, 'text')
        return parts

    @staticmethod
    def get_latest_form_text(
        symbol: str,
        form_type: FormType,
    ):
        """Gets the latest 10Q or 10K for a given NASDAQ or NYSE market symbol"""
        logging.debug('get_latest_form_text: {}'.format(symbol))

        sections_10q = [
            'part1item1', 'part1item2', 'part1item3', 'part1item4',
            'part2item1', 'part2item1a', 'part2item2', 'part2item3', 'part2item4',
            'part2item5', 'part2item6'
        ]

        sections_10k = [
            '1', '1A', '1B', '2', '3', '4', '5', '6', '7', '7A', '8', '9', '9A',
            '9B', '10', '11', '12', '13', '14'
        ]

        urls = EdgarHelpers.get_form_urls(symbol, form_type)
        if not urls:
            logging.debug('No urls found for {}'.format(symbol))
            return ''

        latest_url = urls[0][1]
        logging.debug('get_latest_form_text latest_url: {}'.format(latest_url))

        if form_type == EdgarHelpers.FormType.TENQ:
            sections = EdgarHelpers.extract_form_text(latest_url, sections_10q)
            return '\n'.join([section for section in sections.values()])
        elif form_type == EdgarHelpers.FormType.TENK:
            sections = EdgarHelpers.extract_form_text(latest_url, sections_10k)
            return '\n'.join([section for section in sections.values()])
        else:
            raise ValueError('not implemented')
