import asyncio
import re
from typing import Any, Callable, Dict, List, Tuple

from googlesearch import search as google_search
from playwright.async_api import ElementHandle, Error, Page, async_playwright

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (BrowserContent, Content, ImageContent, LLMCall, MarkdownContent, Message,
                                  TokenCompressionMethod, User, bcl)
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.python_runtime import PythonRuntime
from llmvm.server.tools.chrome import ChromeHelpers
from llmvm.server.tools.webhelpers import WebHelpers

logging = setup_logging()

class Browser():
    def __init__(
        self,
        task: str,
        controller: ExecutionController,
        runtime: PythonRuntime,
        original_code: str,
        original_query: str,
        cookies: List[Dict[str, str]] = [],
    ):
        """
        Controls a Chrome browser instance to interact with web pages.
        """
        self.task = task
        self.original_code = original_code
        self.original_query = original_query
        self.controller = controller
        self.cookies = cookies
        self.runtime = PythonRuntime
        self.query = original_query

        self.browser = ChromeHelpers(cookies=cookies)
        self.current_page: Page
        self.current_screenshot: ImageContent
        self.current_markdown: MarkdownContent

    def close(self):
        """
        Closes the browser instance.
        """
        logging.debug("Closing browser")
        asyncio.run(self.browser.close())

    def goto(self, url: str) -> BrowserContent:
        """
        Opens a Chrome browser instance at the url specified and returns the webpage and webpage markdown for you to interact with.
        At the start of the Markdown, a list of clickable elements and their selector ids are provided.
        This allows you to see what is on the current web page and either click or fill in text boxes.

        :param url: The url to open in the browser
        :type url: str
        :return: A tuple containing an object and a MarkdownContent object
        """
        logging.debug(f"Browser.goto() getting verbose markdown for {url}")
        self.current_page = asyncio.run(self.browser.goto(url))
        self.current_markdown = self.element_markdown(url)
        self.current_screenshot = ImageContent(asyncio.run(self.browser.screenshot()))
        return BrowserContent([self.current_screenshot, self.current_markdown])

    def input_boxes(self) -> List[ElementHandle]:
        logging.debug("Getting input elements")
        return asyncio.run(self.browser.get_input_elements())

    def clickable(self) -> List[ElementHandle]:
        logging.debug("Getting clickable elements")
        return asyncio.run(self.browser.get_clickable_elements())

    def element_markdown(self, url: str = '') -> MarkdownContent:
        html = asyncio.run(self.browser.get_html())

        clickable_elements = self.clickable()
        input_elements = self.input_boxes()
        append_str = ''
        if clickable_elements:
            append_str += '\n\nClickable Elements:\n'
            for index, element in enumerate(clickable_elements):
                append_str += f"{index}: {element}\n"

        if input_elements:
            append_str += '\n\nInput Elements:\n'
            for index, element in enumerate(input_elements):
                append_str += f"{index}: {element}\n"

        markdown = WebHelpers.convert_html_to_markdown(html, url)
        markdown.sequence += '\n\n\n' + append_str  # type: ignore
        return markdown

    def click(
        self,
        selector: str,
    ):
        """
        Clicks on the element specified by the selector. Selectors are ids of elements on the page.
        """
        logging.debug(f"Clicking {selector}")
        asyncio.run(self.browser.click(selector))

    def fill(
        self,
        selector: str,
        text: str,
    ):
        """
        Inserts text into the element specified by the selector. Selectors are ids of elements on the page.
        """
        logging.debug(f"Inserting {text} into {selector}")
        asyncio.run(self.browser.fill(selector, text))
