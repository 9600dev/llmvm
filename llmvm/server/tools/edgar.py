import datetime as dt
import os
from typing import Optional, cast

from bs4 import BeautifulSoup
from sec_api import ExtractorApi, QueryApi, RenderApi
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import ImageContent, MarkdownContent, PdfContent, TextContent
from llmvm.server.tools.webhelpers import WebHelpers
from llmvm.common.container import Container

logging = setup_logging()


class EdgarHelpers():
    @staticmethod
    def get_filings(
        ticker: str,
        form_type: Optional[str] = None,
        items: Optional[str] = None,
        start_date_str: str = (dt.datetime.now() - dt.timedelta(days=180)).strftime('%Y-%m-%d'),
        end_date_str: str = dt.datetime.now().strftime('%Y-%m-%d'),
    ) -> dict:
        """
        :param ticker: The NASDAQ/NYSE company symbol/ticker to get the filing reports for.
        :type ticker: str
        :param form_type: The type of form to get. '' for all forms, or '10-Q', '10-K' or '8-K', '8-K/A', 'D', 'D/A', 'ABS-15G', 'ABS-15G/A', '1-U', '1-U/A'.
        :type form_type: str
        :param items: Items represents an array of item strings as reported on the forms in form_type. For example: ["Item 3.02: Unregistered Sales of Equity Securities", "Item 9.01: Financial Statements and Exhibits", "9.02"]
        :type items: list[str]
        :param start_date_str: The start date to use to find the closest report for. Defaults to 180 days ago.
        :param end_date_str: The end date to use to find the closest report for. Defaults to today.
        :return: dict response from the SEC Edgar API, presented to you as a json string, which is all the filings for the ticker/company with the constraints you've specified.
        :rtype: dict
        """
        sec_api_key = Container().get_config_variable('SEC_API_KEY', default=None)
        if not sec_api_key:
            raise ValueError('SEC_API_KEY not found in config file or environment')

        logging.debug('EdgarHelpers._get_form_urls {}'.format(ticker))
        query_api = QueryApi(sec_api_key)

        from dateutil import parser

        if isinstance(end_date_str, dt.datetime):
            end_date_str = cast(dt.datetime, end_date_str).strftime('%Y-%m-%d')
        if isinstance(start_date_str, dt.datetime):
            start_date_str = cast(dt.datetime, start_date_str).strftime('%Y-%m-%d')

        str_start_date = parser.parse(start_date_str).strftime('%Y-%m-%d')
        str_end_date = parser.parse(end_date_str).strftime('%Y-%m-%d')

        # build the query
        q = f'ticker:{ticker} AND filedAt:[{str_start_date} TO {str_end_date}]'
        if form_type: q += f' AND formType:"{str(form_type)}"'

        if items:
            for item in items:
                q += f' AND items:{item}'

        query = {
            "query": q,
            "from": "0",
            "size": "10",
            "sort": [{ "filedAt": {"order": "desc"} }]
        }

        result = query_api.get_filings(query)
        return cast(dict, result)

    @staticmethod
    def get_form_filing_or_item_url_as_markdown(
        document_urls: list[str],
    ) -> MarkdownContent:
        """
        Gets the documents, xml or images from the SEC Edgar API and returns them as a MarkdownContent object.
        You can obtain these from a call to EdgarHelpers.get_filings() inside the 'documentUrl' key.

        :param document_urls: The urls of the documents (typically .htm or .html, .xml or an images) to get from the SEC Edgar API. .htm and .html files are highly preferred.
        :type document_urls: list[str]
        :return: The documents text, or images, embedded in a single MarkdownContent object.
        :rtype: MarkdownContent
        """
        sec_api_key = Container().get_config_variable('SEC_API_KEY', default=None)
        if not sec_api_key:
            raise ValueError('SEC_API_KEY not found in config file or environment')

        markdown = MarkdownContent(sequence=[TextContent(f'Filing documents')])

        render_api = RenderApi(sec_api_key)
        for document_url in document_urls:
            if document_url.endswith('.htm') or document_url.endswith('.html'):
                text_content = cast(str, render_api.get_file(document_url))
                # get rid of the xml cruft at the top
                if '<?xml' in text_content and '</div>' in text_content:
                    text_content = text_content[text_content.find('</div>') + len('</div>'):]

                markdown_content = WebHelpers.convert_html_to_markdown(text_content, document_url)
                for content in markdown_content.sequence:
                    markdown.sequence.append(content)
            elif document_url.endswith('.xml'):
                render_api.get_file(document_url)
                markdown.sequence.append(TextContent(WebHelpers.convert_xml_to_text(cast(str, render_api.get_file(document_url)))))
            elif document_url.endswith('.pdf'):
                logging.debug('EdgarHelpers.get_form_or_item_as_markdown() skipping pdf')
            elif document_url.endswith('.jpg') or document_url.endswith('.png'):
                markdown.sequence.append(ImageContent(cast(bytes, render_api.get_file(document_url, return_binary=True))))
            elif document_url.endswith('.xlsx'):
                logging.debug('EdgarHelpers.get_form_or_item_as_markdown() skipping xlsx')
            elif document_url.endswith('.txt'):
                markdown.sequence.append(TextContent(cast(str, render_api.get_file(document_url))))
            else:
                logging.debug(f'EdgarHelpers.get_form_or_item_as_markdown() skipping unknown document type: {document_url}')
        return markdown

    @staticmethod
    def get_latest_filing_as_markdown(
        ticker: str,
        form_type: str,
    ) -> MarkdownContent:
        """
        Gets the latest 10-Q, 10-K or 8-K filing for a given company symbol/ticker.
        This is useful to get the latest financial information for a company,
        their current strategy, investments and risks. Use form_type = '' to
        get the latest form of any type. form_type can be '10-Q', '10-K' or '8-K'.
        You should use an llm_call() right after this call so that you don't fill your context window with the result.

        Example: "what is the latest earnings for APPL?"
        <helpers>
        apple_ten_10k = EdgarHelpers.get_latest_filing_as_markdown("AAPL", "10-K")
        earnings = llm_call([apple_ten_10k], "extract the latest earnings for AAPL")
        result(earnings)
        </helpers>

        :param ticker: The NASDAQ/NYSE company symbol/ticker to get the latest filing for.
        :type ticker: str
        :param form_type: The type of form to get: '10-Q', '10-K' or '8-K'.
        :return: The latest filing (10-Q, 10-K or 8-K) for the given company as MarkdownContent.
        """
        filings = EdgarHelpers.get_filings(ticker, form_type=form_type)
        document_url = ''

        if form_type not in ['10-Q', '10-K', '8-K']:
            raise ValueError(f'form_type must be one of 10-Q, 10-K, 8-K, given {form_type}')

        for filing in filings['filings']:  # type: ignore
            for document in filing['documentFormatFiles']:  # type: ignore
                if (
                    'description' in document
                    and (
                        '10-Q' in document['description']
                        or '10-K' in document['description']
                        or '8-K' in document['description']
                    )
                ):
                    if form_type in document['description']:
                        document_url = document['documentUrl']
                        break
        if document_url:
            return EdgarHelpers.get_form_filing_or_item_url_as_markdown([document_url])
        else:
            return MarkdownContent(sequence=[TextContent(f'No filings found for {ticker} and form type {form_type}')])
