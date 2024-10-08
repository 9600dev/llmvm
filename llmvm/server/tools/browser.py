import asyncio
import re
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urljoin, urlparse

from googlesearch import search as google_search
from playwright.async_api import ElementHandle, Error, Page, async_playwright, TimeoutError

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (BrowserContent, Content, ImageContent, LLMCall, MarkdownContent, Message, StreamNode,
                                  TokenCompressionMethod, User, bcl)
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.python_runtime import PythonRuntime
from llmvm.server.tools.chrome import ChromeHelpers, ClickableElementHandle
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

    async def __find_html_element_handle(self, html_string: str) -> ElementHandle | None:
        # Extract attributes from the HTML string
        page = self.current_page
        attr_pattern = r'(\w+)=["\']([^"\']*)["\']'
        attributes = dict(re.findall(attr_pattern, html_string))

        tag_match = re.match(r'<(\w+)', html_string)
        tag_name = tag_match.group(1) if tag_match else None

        if not tag_name:
            return None

        methods = [
            ('id', lambda v: f'#{v}'),
            ('name', lambda v: f'[name="{v}"]'),
            ('class', lambda v: f'.{v.replace(" ", ".")}'),
            ('type', lambda v: f'{tag_name}[type="{v}"]'),
            ('value', lambda v: f'{tag_name}[value="{v}"]'),
            ('placeholder', lambda v: f'{tag_name}[placeholder="{v}"]'),
        ]

        for attr, selector_func in methods:
            if attr in attributes:
                selector = selector_func(attributes[attr])
                element = await page.query_selector(selector)
                if element:
                    print(f"Element found using {attr}: {selector}")
                    return element

        combined_selector = tag_name
        for attr, value in attributes.items():
            combined_selector += f'[{attr}="{value}"]'

        element = await page.query_selector(combined_selector)
        if element:
            return element

        xpath = f"//{tag_name}"
        for attr, value in attributes.items():
            xpath += f'[@{attr}="{value}"]'

        element = await page.query_selector(f'xpath={xpath}')
        if element:
            return element

        return None

    async def __resolve_selector(self, selector: str) -> None | ElementHandle:
        logging.debug(f"Browser.__resolve_selector() resolving {selector}")
        try:
            async with asyncio.timeout(2):
                if selector.startswith('<'):
                    return await self.__find_html_element_handle(selector)

                locator = self.current_page.locator(selector)
                if locator:
                    return await locator.element_handle()
                else:
                    return None
        except asyncio.TimeoutError:
            logging.debug(f"Browser.__resolve_selector() TimeoutError resolving {selector}")
            raise Exception(f"Timeout resolving selector {selector}")
        except Exception as ex:
            logging.debug(f"Browser.__resolve_selector() Exception resolving {selector}: {ex}")
            raise Exception(f"Exception resolving selector {selector}: {ex}")

    def __handle_navigate_expression(self, selector: str) -> BrowserContent:
        def __internal_click(selector: str) -> BrowserContent:
            logging.debug(f"Browser.__handle_navigate_expression() clicking {selector}")
            element_handle = asyncio.run(self.__resolve_selector(selector))
            asyncio.run(self.browser.click(element_handle))
            asyncio.run(self.browser.wait(500))
            return self.__get_state()

        result = selector.strip()

        # Check if it's a valid CSS selector
        if result.startswith('#') or result.startswith('.') or result.startswith(':'):
            return __internal_click(result)

        # Check if it's a URL or part of a URL
        url_pattern = re.compile(r'^(https?://)?[\w\-]+(\.[\w\-]+)+[/#?]?.*$')
        if url_pattern.match(result):
            # If it's a full URL
            if result.startswith(('http://', 'https://')):
                return self.goto(result)
            # If it's a protocol-relative URL
            elif result.startswith('//'):
                return self.goto(f"{urlparse(self.current_url).scheme}:{result}")
            # If it's a domain or partial URL
            else:
                return self.goto(urljoin(self.current_url, '//' + result))

        # Check if it's a relative path
        elif result.startswith('/'):
            return self.goto(urljoin(self.current_url, result))

        else:
            return __internal_click(result)

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
        return BrowserContent(sequence=[self.current_screenshot, self.current_markdown], url=url)

    def __input_boxes(self) -> List[ClickableElementHandle]:
        logging.debug("Getting input elements")
        return asyncio.run(self.browser.get_input_elements())

    def __clickable(self) -> List[ClickableElementHandle]:
        logging.debug("Getting clickable elements")
        return asyncio.run(self.browser.get_clickable_elements())

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
        <helpers>
        browser = Browser()
        page_content = browser.goto("https://formula1.com")
        answer(page_content)
        </helpers>
        Now find the selector for the Race Results button:
        <helpers>
        race_results_id = browser.get_selector("Race Results")
        </helpers>

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
                prompt_name='get_selector',
            ),
            query=expression,
            original_query=expression,
            compression=TokenCompressionMethod.AUTO,
        ))

        result = result.message.get_str().strip()
        return result

    def find_and_click_on(self, expression: str) -> BrowserContent:
        """
        Clicks on an element that matches the textual description in expression.

        Example:
        <helpers>
        browser = Browser()
        page_content = browser.goto("https://formula1.com")
        answer(page_content)
        </helpers>
        Now clicking the Race Results button:
        <helpers>
        race_results_content = browser.click_on("Race Results button")
        answer(race_results_content)
        </helpers>

        :param expression: The natural language description of the element to click
        :return: The current state of the browser
        """
        logging.debug(f'Browser.find_and_click_on({expression}')
        result = self.get_selector(expression)
        return self.__handle_navigate_expression(selector=result)

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
        <helpers>
        browser = Browser()
        page_content = browser.goto("https://google.com")
        selector_id = browser.get_selector("What is the id of the search button?")
        answer(selector_id)
        </helpers>
        Now clicking the button:
        <helpers>
        click_result = browser.click(selector_id)
        answer(click_result)
        </helpers>

        :param selector: The selector of the element to click
        :type selector: str
        :return: The current state of the browser
        """
        logging.debug(f"Browser.click() clicking {selector}")
        return self.__handle_navigate_expression(selector=selector)

    def type_into(
        self,
        selector: str,
        text: str,
        hit_enter: bool = False,
    ) -> BrowserContent:
        """
        Inserts text into the element specified by the selector, and hits the enter key if required. Selectors are ids of elements on the page.

        Example:
        User: search on google for "vegemite"
        <helpers>
        browser = Browser()
        page_content = browser.goto("https://google.com")
        selector_id = browser.get_selector("What is the id of the search box?")
        answer(selector_id)
        </helpers>
        Now typing into the search box and hit enter to search:
        <helpers>
        type_into_result = browser.type_into(selector_id, "vegemite", hit_enter=True)
        answer(type_into_result)
        </helpers>

        :param selector: The selector of the element to insert text into
        :type selector: str
        :param text: The text to insert
        :type text: str
        :param hit_enter: Whether to hit enter after inserting text
        :type hit_enter: bool
        :return: The current state of the browser
        """

        logging.debug(f"Inserting {text} into {selector}")
        element_handle: ElementHandle | None = asyncio.run(self.__resolve_selector(selector))
        if element_handle:
            asyncio.run(self.browser.fill(element=element_handle, value=text))
            asyncio.run(self.browser.wait(500))
            if hit_enter:
                asyncio.run(element_handle.press('Enter', timeout=1000))
                asyncio.run(self.browser.wait(500))
            return self.__get_state()
        else:
            logging.debug(f"Browser.type_into() Element not found for selector {selector}")
            # todo: this is wrong, we should probably return an error with the current state
            return self.__get_state()