import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from googlesearch import search as google_search
from playwright.async_api import (ElementHandle, Error, Page, Locator, TimeoutError,
                                  async_playwright)

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (BrowserContent, Content, ImageContent,
                                  LLMCall, MarkdownContent, Message,
                                  StreamNode, TextContent,
                                  TokenCompressionMethod, User, bcl)
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.tools.chrome import ChromeHelpers, ClickableElementHandle
from llmvm.server.tools.webhelpers import WebHelpers

logging = setup_logging()

class Browser():
    def __init__(
        self,
        controller: ExecutionController,
        cookies: List[Dict[str, str]] = [],
    ):
        """
        Controls a Playwright based Chrome browser instance to interact with web pages.
        You typically only need one instance of this class per user session:

        browser = Browser()
        ...
        """
        self.controller = controller
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

        # Handle coordinate-based selector
        if selector.startswith('(') and selector.endswith(')') and ',' in selector:
            selector = selector[1:-1]
            x, y = selector.split(',')
            return await self.browser.get_element_at_position(int(x), int(y))

        async def __locator(selector: str) -> Optional[ElementHandle]:
            count = 0
            async with asyncio.timeout(3):
                locator = self.current_page.locator(selector)
                count = await locator.count()

            if count > 0:
                return await locator.first.element_handle(timeout=4000)
            else: return None

        original_selector = selector

        try:
            if selector.startswith('<'):
                return await self.__find_html_element_handle(selector)

            result = await __locator(selector)
            if result: return result

            if not selector.startswith('#') and not selector.startswith('.') and not selector.startswith(':'):
                result = await __locator(f'#{selector}')
                if result: return result
                result = await __locator(f'.{selector}')
                if result: return result
            elif selector.startswith('#'):
                result = await __locator(f'.selector[1:]')
                if result: return result
            elif selector.startswith('.'):
                result = await __locator(f'#{selector[1:]}')
                if result: return result

            logging.debug(f"Browser.__resolve_selector() failed to find selector {original_selector}")
            return None
        except asyncio.TimeoutError:
            logging.debug(f"Browser.__resolve_selector() TimeoutError resolving {original_selector}")
            raise Exception(f"Timeout resolving selector {selector}")
        except Exception as ex:
            logging.debug(f"Browser.__resolve_selector() Exception resolving {original_selector}: {ex}")
            raise Exception(f"Exception resolving selector {selector}: {ex}")

    def __handle_navigate_expression(self, selector: str) -> BrowserContent:
        def __internal_click(selector: str) -> BrowserContent:
            if 'current_page' not in self.__dict__:
                raise ValueError('The browser instance is either closed, or has not been opened yet. Call goto() first.')

            logging.debug(f"Browser.__handle_navigate_expression() clicking {selector}")
            element_handle = asyncio.run(self.__resolve_selector(selector))
            asyncio.run(self.browser.click(element_handle))
            asyncio.run(self.browser.wait(500))
            return self.__get_state()

        result = selector.strip()

        if result.startswith('(') and result.endswith(')') and ',' in result:
            result = result[1:-1]
            x, y = result.split(',')
            return self.mouse_move_x_y_and_click(int(x), int(y))

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
            for _, element in enumerate(clickable_elements):
                append_str += f"{element}\n"

        if input_elements:
            append_str += '\n\nInput Elements:\n'
            for _, element in enumerate(input_elements):
                append_str += f"{element}\n"

        markdown = WebHelpers.convert_html_to_markdown(html, url)
        markdown.sequence.append(TextContent('\n\n\n' + append_str))
        return markdown

    def close(self):
        """
        Closes and disposes of the browser instance.
        """
        logging.debug("Closing browser")
        asyncio.run(self.browser.close())

    def find_selector_or_mouse_x_y(self, expression: str) -> str:
        """
        Returns the best matching element selector for the expression, or the x, y coordinates for a mouse click.
        It can be either a 1) <a href...> link (/path/link.html), 2) a html id (#id), 3) x, y coordinates for a mouse click in the format of (x,y),
        or 4) or any other selector that uniquely identifies the element on the page.

        Assistant:
        <helpers>
        race_results_id = browser.find_selector_or_mouse_x_y("Race Results")
        result(race_results_id)
        </helpers>

        The selector_id is #race-results-button-id

        :param expression: The expression of the element to click
        :return: The best matching element selector
        """
        PROMPT = f"""
        The previous messages contain a screenshot of a Markdown page, and a Markdown page.
        The Markdown page contains a list of clickable elements.

        You are also given a textual description of an element on the Markdown page at the bottom
        under "Expression".

        Your task is to find the best matching element selector for the expression. The selector
        can be either 1) a <a href...> link (/path/link.html), 2) a html id (#id), 3) x, y coordinates for a mouse click, or
        4) any other css selector that can uniquely identify the element on the page.

        Return the best matching element selector or mouse x, y coordinate as a string, and nothing else. Don't apologize.
        Example: "(140,200)"
        Example: "#search-button-id"

        Expression: {expression}
        """
        if 'current_page' not in self.__dict__:
            raise ValueError('The browser instance is either closed, or has not been opened yet. Call goto() first.')

        result = asyncio.run(self.controller.aexecute_llm_call(
            llm_call=LLMCall(
                user_message=User(TextContent(PROMPT)),
                context_messages=[User([self.current_screenshot, self.current_markdown])],
                executor=self.controller.executor,
                model=self.controller.executor.default_model,
                temperature=1.0,
                max_prompt_len=self.controller.executor.max_input_tokens(),
                completion_tokens_len=self.controller.executor.max_output_tokens(),
                prompt_name='find_selector_or_mouse_x_y',
            ),
            query=expression,
            original_query=expression,
            compression=TokenCompressionMethod.AUTO,
        ))

        result = result.get_str().strip()
        return result

    def find_and_click_on_expression(self, expression: str) -> BrowserContent:
        """
        Finds and then clicks on an element that matches the natural languagedescription in expression.

        Example:
        User: open formula1.com and find the latest race results

        Assistant:
        <helpers>
        browser = Browser()
        page_content = browser.goto("https://formula1.com")
        result(page_content)
        </helpers>

        I can see a "Race Results" button in the page. Let's click the "Race Results" button:

        <helpers>
        race_results_content = browser.find_and_click_on_expression("Race Results button")
        result(race_results_content)
        </helpers>

        :param expression: The natural language description of the element to click
        :return: The current state of the browser
        """
        logging.debug(f'Browser.find_and_click_on_expression({expression}')
        result = self.find_selector_or_mouse_x_y(expression)
        return self.__handle_navigate_expression(selector=result)

    def goto(self, url: str) -> BrowserContent:
        """
        Opens a Chrome browser instance at the url specified and returns the webpage and webpage markdown for you to interact with.
        At the start of the Markdown, a list of clickable elements and their selector ids are provided.
        This allows you to see what is on the current web page and either click or fill in text boxes.
        Returns the current state of the browser which you can use in an result() call.
        You should not try and complete the page if its returned to you. If you see BrowserContent() in a helper result
        then you should just stop the completion and allow the user to recommend next steps.
        If the page you are returning contains news, data, or other content, start with a Markdown link [Link](http://example.com)
        The resolution of the page is 1920x1080 by default.

        Example:
        User: open google.com
        Assistant:
        <helpers>
        browser = Browser()
        page_content = browser.goto("https://google.com")
        result(page_content)
        </helpers>

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

    def click_selector(
        self,
        selector: str,
    ) -> BrowserContent:
        """
        Clicks on the element specified by the selector. Selectors are ids of elements on the page.
        You should wrap the result of this call in an result() call, i.e. result(browser.click(selector_id))

        Example:
        <helpers>
        browser = Browser()
        page_content = browser.goto("https://google.com")
        selector_id = browser.find_selector_or_mouse_x_y("What is the id of the search button?")
        result(selector_id)
        </helpers>

        The selector_id is #search-button-id
        Now clicking the button using click_selector:

        <helpers>
        click_result = browser.click_selector(selector_id)
        result(click_result)
        </helpers>

        :param selector: The selector of the element to click
        :type selector: str
        :return: The current state of the browser
        """
        logging.debug(f"Browser.click() clicking {selector}")
        return self.__handle_navigate_expression(selector=selector)

    def type_into_selector_or_mouse_x_y(
        self,
        selector: str,
        text: str,
        hit_enter: bool = True,
    ) -> BrowserContent:
        """
        Inserts text into the element specified by the selector, and hits the enter key if required. Selectors are ids of elements on the page
        or x, y coordinates for a mouse click. You should wrap the result of this call in an result() call,
        i.e. result(browser.type_into(selector_id, "vegemite", hit_enter=True))

        Example:
        User: search on google for "vegemite"

        Assistant:
        ...

        selector_id is "(180,10)" for the search box
        Now typing into the search box and hit enter to search:

        <helpers>
        type_into_result = browser.type_into_selector_or_mouse_x_y(selector_id, "vegemite", hit_enter=True)
        result(type_into_result)
        </helpers>

        :param selector: The selector of the element to insert text into
        :type selector: str
        :param text: The text to insert
        :type text: str
        :param hit_enter: Whether to hit enter after inserting text
        :type hit_enter: bool
        :return: The current state of the browser
        """
        if 'current_page' not in self.__dict__:
            raise ValueError('The browser instance is either closed, or has not been opened yet. Call goto() first.')

        logging.debug(f"Inserting {text} into {selector}")
        try:
            element_handle: ElementHandle | None = asyncio.run(self.__resolve_selector(selector))
        except Exception as ex:
            logging.debug(f"Browser.type_into() Exception resolving selector {selector}: {ex}")
            # try again with a # id selector
            if not selector.startswith('#'):
                selector = f'#{selector}'
                element_handle: ElementHandle | None = asyncio.run(self.__resolve_selector(selector))
            else:
                selector = f'{selector}[id="{selector}"]'
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

    def mouse_move_x_y_and_click(
        self,
        x: int,
        y: int,
    ) -> BrowserContent:
        """
        Moves the mouse to the specified coordinates and clicks the left mouse button.
        If you encounter errors with the other Browser helper functions, you should try this one as it usually works.
        The page is 1920x1080 by default. 0,0 is the top left corner. 1920,1080 is the bottom right corner.

        User: open google travel and find accomodation in "currumbin, qld", december 15th

        Assistant:
        ... (Example browser page has the search input filled in with "currumbin, qld" and has a date selector button at x=185, y=78:)

        <helpers>
        click_result = browser.mouse_move_x_y_and_click(x=190, y=80)
        result(click_result)
        </helpers>

        :param x: The x coordinate of the mouse pointer.
        :type x: int
        :param y: The y coordinate of the mouse pointer.
        :type y: int
        :return: The current state of the browser
        """
        if 'current_page' not in self.__dict__:
            raise ValueError('The browser instance is either closed, or has not been opened yet. Call goto() first.')

        asyncio.run(self.browser.mouse_move_x_y_and_click(x, y))
        asyncio.run(self.browser.wait(500))
        return self.__get_state()

