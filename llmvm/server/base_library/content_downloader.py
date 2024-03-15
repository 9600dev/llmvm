import asyncio
from typing import Callable, Dict, List
from urllib.parse import urlparse

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import Content, LLMCall, Message, User, bcl
from llmvm.server.starlark_execution_controller import ExecutionController
from llmvm.server.tools.pdf import PdfHelpers
from llmvm.server.tools.webhelpers import FirefoxHelpers, WebHelpers

logging = setup_logging()


class ContentDownloader():
    def __init__(
        self,
        expr,
        agents: List[Callable],
        messages: List[Message],
        controller: ExecutionController,
        original_code: str,
        original_query: str,
        cookies: List[Dict] = [],
    ):
        self.expr = expr
        self.agents = agents
        self.messages: List[Message] = messages
        self.controller = controller
        self.original_code = original_code
        self.original_query = original_query
        self.cookies = cookies

        # the client can often send through urls with quotes around them
        if self.expr.startswith('"') and self.expr.endswith('"'):
            self.expr = self.expr[1:-1]

    def parse_pdf(self, filename: str) -> str:
        content = PdfHelpers.parse_pdf(filename)

        query_expander = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='pdf_content.prompt',
                    template={},
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=self.controller.statement_to_message(content),
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='pdf_content.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )

        if 'SUCCESS' in str(query_expander.message):
            return content
        else:
            return PdfHelpers.parse_pdf_image(filename)

    def download(self) -> Content:
        logging.debug('ContentDownloader.download: {}'.format(self.expr))

        # deal with files
        result = urlparse(self.expr)
        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                return Content(self.parse_pdf(result.path))
            if '.htm' in result.path or '.html' in result.path:
                return WebHelpers.convert_html_to_markdown(open(result.path, 'r').read(), url=self.expr)

        # deal with pdfs
        elif (result.scheme == 'http' or result.scheme == 'https') and '.pdf' in result.path:
            firefox_helper = FirefoxHelpers(cookies=self.cookies)
            loop = asyncio.get_event_loop()
            task = loop.create_task(firefox_helper.pdf_url(self.expr))

            pdf_filename = loop.run_until_complete(task)
            _ = loop.run_until_complete(loop.create_task(firefox_helper.close()))

            return Content(self.parse_pdf(pdf_filename))

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
