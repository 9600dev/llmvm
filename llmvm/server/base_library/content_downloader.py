import asyncio
from typing import Callable, Dict, List
from urllib.parse import urlparse

from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import Content, PdfContent
from llmvm.server.tools.webhelpers import FirefoxHelpers, WebHelpers

logging = setup_logging()


class ContentDownloader():
    def __init__(
        self,
        expr,
        cookies: List[Dict] = [],
    ):
        self.expr = expr
        self.cookies = cookies

        # the client can often send through urls with quotes around them
        if self.expr.startswith('"') and self.expr.endswith('"'):
            self.expr = self.expr[1:-1]

    def download(self) -> Content:
        logging.debug('ContentDownloader.download: {}'.format(self.expr))

        # deal with files
        result = urlparse(self.expr)
        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                return PdfContent(sequence=b'', url=str(result.path))
            if '.htm' in result.path or '.html' in result.path:
                return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read(), url=self.expr)

        # deal with pdfs
        elif (result.scheme == 'http' or result.scheme == 'https') and '.pdf' in result.path:
            firefox_helper = FirefoxHelpers(cookies=self.cookies)
            loop = asyncio.get_event_loop()
            # downloads the pdf and gets a local file url
            task = loop.create_task(firefox_helper.pdf_url(self.expr))

            pdf_filename = loop.run_until_complete(task)
            _ = loop.run_until_complete(loop.create_task(firefox_helper.close()))
            return PdfContent(sequence=b'', url=pdf_filename)

        # deal with websites
        elif result.scheme == 'http' or result.scheme == 'https':
            firefox_helper = FirefoxHelpers(cookies=self.cookies)
            loop = asyncio.get_event_loop()
            task = loop.create_task(firefox_helper.get_url(self.expr))
            result = loop.run_until_complete(task)
            _ = loop.run_until_complete(loop.create_task(firefox_helper.close()))
            return WebHelpers.convert_html_to_markdown(result, url=self.expr)

        # else, nothing
        return Content()
