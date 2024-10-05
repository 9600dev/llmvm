import asyncio
import re
from typing import Any, Callable, Dict, List, Tuple

from googlesearch import search as google_search
from playwright.async_api import ElementHandle, Error, Page, async_playwright

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (BrowserContent, Content, ImageContent, LLMCall, MarkdownContent, Message, StreamNode,
                                  TokenCompressionMethod, User, bcl)
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.python_runtime import PythonRuntime
from llmvm.server.tools.chrome import ChromeHelpers
from llmvm.server.tools.webhelpers import WebHelpers

logging = setup_logging()

class Browser():
    def __init__(
        self,
        controller: ExecutionController,
        runtime: PythonRuntime,
        cookies: List[Dict[str, str]] = [],
    ):
        """
        Controls a Chrome browser instance to interact with web pages.
        """
        self.controller = controller
        self.runtime = runtime
        self.cookies = cookies

        self.browser = ChromeHelpers(cookies=cookies)
        self.current_page: Page
        self.current_url: str = ''
        self.current_screenshot: ImageContent
        self.current_markdown: MarkdownContent

    def __get_state(self) -> BrowserContent:
        url: str = asyncio.run(self.browser.url())
        self.current_markdown: MarkdownContent = self.__element_markdown(url)
        screenshot: bytes = asyncio.run(self.browser.screenshot())

        write_client_stream(
            StreamNode(
                obj=screenshot,
                type='bytes',
                metadata={'type': 'bytes', 'url': url}
            ))

        self.current_screenshot = ImageContent(screenshot)
        return BrowserContent([self.current_screenshot, self.current_markdown])

    async def __resolve_selector(self, selector: str) -> ElementHandle:
        locator = self.current_page.locator(selector)
        return await locator.element_handle()

    def __input_boxes(self) -> List[ElementHandle]:
        logging.debug("Getting input elements")
        return asyncio.run(self.browser.get_input_elements())

    def __clickable(self) -> List[ElementHandle]:
        logging.debug("Getting clickable elements")
        return asyncio.run(self.browser.get_clickable_elements())  # type: ignore

    def __element_markdown(self, url: str = '') -> MarkdownContent:
        html = asyncio.run(self.browser.get_html())

        clickable_elements = self.__clickable()
        input_elements = self.__input_boxes()
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

    def close(self):
        """
        Closes and disposes of the browser instance.
        """
        logging.debug("Closing browser")
        asyncio.run(self.browser.close())

    def get_selector(self, expression: str) -> str:
        """
        Returns the best matching element selector for the expression. It can be
        an <a href...> link (/path/link.html), a html id (#id), or any other selector that uniquely
        identifies the element on the page.

        Example:
        <code>
        browser = Browser()
        page_content = browser.goto("https://formula1.com")
        answer(page_content)
        </code>
        Now find the selector for the Race Results button:
        <code>
        race_results_id = browser.get_selector("Race Results")
        </code>

        :param expression: The expression of the element to click
        :return: The best matching element selector
        """

        PROMPT = f"""
        The previous messages contain a screenshot of a Markdown page, and a Markdown page.
        The Markdown page contains a list of clickable elements.

        You are also given a textual description of an element on the Markdown page at the bottom
        under "Expression".

        Your task is to find the best matching element selector for the expression. It can be
        an <a href...> link (/path/link.html), a html id (#id), or any other selector that uniquely
        identifies the element on the page.

        Return the best matching element selector as a string, and nothing else. Don't apologize.
        Just return the best matching element selector as a string.

        Expression: {expression}
        """

        result = asyncio.run(self.controller.aexecute_llm_call(
            llm_call=LLMCall(
                user_message=User(Content(PROMPT)),
                context_messages=[User(self.current_screenshot), User(self.current_markdown)],
                executor=self.controller.executor,
                model=self.controller.executor.default_model,
                temperature=1.0,
                max_prompt_len=self.controller.executor.max_input_tokens(),
                completion_tokens_len=self.controller.executor.max_output_tokens(),
                prompt_name='click_on',
            ),
            query=expression,
            original_query=expression,
            compression=TokenCompressionMethod.AUTO,
        ))

        result = result.message.get_str().strip()
        return result

    def click_on(self, expression: str) -> BrowserContent:
        """
        Clicks on the element specified by the expression. Expressions are textual descriptions
        of the element you want to click on.

        Example:
        <code>
        browser = Browser()
        page_content = browser.goto("https://formula1.com")
        answer(page_content)
        </code>
        Now clicking the Race Results button:
        <code>
        race_results_content= browser.click_on("Race Results")
        answer(race_results_content)
        </code>

        :param expression: The expression of the element to click
        :return: The current state of the browser
        """
        logging.debug(f'Browser.click_on({expression}')

        result = self.get_selector(expression)

        # check to see if the result is a valid CSS selector
        if result.startswith('#') or result.startswith('.') or result.startswith(':'):
            return self.click(result)
        elif result.startswith('http'):
            return self.goto(result)
        elif result.startswith('/'):
            return self.goto(f'{self.current_url}{result}')
        else:
            return self.click(result)

    def goto(self, url: str) -> BrowserContent:
        """
        Opens a Chrome browser instance at the url specified and returns the webpage and webpage markdown for you to interact with.
        At the start of the Markdown, a list of clickable elements and their selector ids are provided.
        This allows you to see what is on the current web page and either click or fill in text boxes.
        Returns the current state of the browser which you can use in an answer() call.
        Example:
        browser = Browser()
        page_content = browser.goto("https://google.com")
        answer(page_content)

        :param url: The url to open in the browser
        :type url: str
        :return: The current state of the browser, including a screenshot and all clickable elements and their ids
        """
        logging.debug(f"Browser.goto() getting verbose markdown for {url}")
        self.current_url = url
        asyncio.run(self.browser.goto(url))
        self.current_page = asyncio.run(self.browser.page())
        asyncio.run(self.browser.wait(500))
        return self.__get_state()

    def click(
        self,
        selector: str,
    ) -> BrowserContent:
        """
        Clicks on the element specified by the selector. Selectors are ids of elements on the page.

        Example:
        <code>
        browser = Browser()
        page_content = browser.goto("https://google.com")
        selector_id = llm_call([page_content], "What is the id of the search button?")
        answer(selector_id)
        </code>
        Now clicking the button:
        <code>
        click_result = browser.click(selector_id)
        answer(click_result)
        </code>

        :param selector: The selector of the element to click
        :type selector: str
        :return: The current state of the browser
        """

        # check to see if the result is a valid CSS selector
        if selector.startswith('http'):
            return self.goto(selector)
        elif selector.startswith('/'):
            return self.goto(f'{self.current_url}{selector}')

        logging.debug(f"Clicking {selector}")
        element_handle = asyncio.run(self.__resolve_selector(selector))
        asyncio.run(self.browser.click(element_handle))
        asyncio.run(self.browser.wait(500))
        return self.__get_state()

    def type_into(
        self,
        selector: str,
        text: str,
    ) -> BrowserContent:
        """
        Inserts text into the element specified by the selector. Selectors are ids of elements on the page.

        Example:
        User: search on google for "vegemite"
        <code>
        browser = Browser()
        page_content = browser.goto("https://google.com")
        selector_id = llm_call([page_content], "What is the id of the search box?")
        answer(selector_id)
        </code>
        Now typing into the search box:
        <code>
        type_into_result = browser.type_into(selector_id, "vegemite")
        answer(type_into_result)
        </code>

        :param selector: The selector of the element to insert text into
        :type selector: str
        :param text: The text to insert
        :type text: str
        :return: The current state of the browser
        """
        # check to see if the result is a valid CSS selector
        if selector.startswith('http'):
            return self.goto(selector)
        elif selector.startswith('/'):
            return self.goto(f'{self.current_url}{selector}')

        logging.debug(f"Inserting {text} into {selector}")
        asyncio.run(self.browser.fill(asyncio.run(self.__resolve_selector(selector)), text))
        asyncio.run(self.browser.wait(500))
        return self.__get_state()