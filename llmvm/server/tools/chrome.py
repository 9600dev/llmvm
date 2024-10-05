import asyncio
import base64
import concurrent
import concurrent.futures
import datetime as dt
import os
import threading
import tempfile
from typing import Dict, List, Tuple, cast
from urllib.parse import urlparse

import aiofiles
from bs4 import BeautifulSoup
import httpx
import nest_asyncio
from playwright.async_api import ElementHandle, Error, Page, async_playwright

from llmvm.common.container import Container
from llmvm.common.helpers import write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import StreamNode
from llmvm.common.singleton import Singleton

nest_asyncio.apply()
logging = setup_logging()

class ClickableElementHandle():
    def __init__(
        self,
        element_handle: ElementHandle,
        scope_id: int,
        id: str,
        outer_html: str,
        html: str,
    ):
        self.element_handle = element_handle
        self.scope_id = scope_id
        self.id = id
        self.outer_html = outer_html
        self.html = html

    def __str__(self):
        return f'ClickableElementHandle(scope_id={self.scope_id}, id={self.id}, html={self.html})'

    def __repr__(self) -> str:
        return f'ClickableElementHandle(scope_id={self.scope_id}, id={self.id})'

def read_netscape_cookies(cookies_txt_filename: str):
    cookies = []
    with open(cookies_txt_filename, 'r') as file:
        for line in file.readlines():
            if not line.startswith('#') and line.strip():  # Ignore comments and empty lines
                try:
                    domain, _, path, secure, expires_value, name, value = line.strip().split('\t')

                    if int(expires_value) != -1 and int(expires_value) < 0:
                        continue  # Skip invalid cookies

                    dt_object = dt.datetime.fromtimestamp(int(expires_value))
                    if dt_object.date() < dt.datetime.now().date():
                        continue

                    cookies.append({
                        "name": name,
                        "value": value,
                        "domain": domain,
                        "path": path,
                        "expires": int(expires_value),
                        "httpOnly": False,
                        "secure": secure == "TRUE"
                    })
                except Exception as ex:
                    logging.warn(f'Malformatted cookies line in {cookies_txt_filename}: {line}')
                    pass
    return cookies


class ChromeHelpers():
    def __init__(self, cookies: List[Dict] = []):
        self.loop = asyncio.SelectorEventLoop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        self.chrome = ChromeHelpersInternal(cookies=cookies)

    @staticmethod
    async def check_installed() -> bool:
        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            await browser.close()
            return True
        except Exception as ex:
            logging.debug(f'ChromeHelpers.check_installed() failed with: {ex}')
            return False

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_in_loop(self, coro):
        future = concurrent.futures.Future()

        async def wrapped():
            try:
                result = await coro
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        asyncio.run_coroutine_threadsafe(wrapped(), self.loop)
        return future

    def set_cookies(self, cookies: List[Dict]):
        self.chrome.set_cookies(cookies)

    async def url(self) -> str:
        return self.run_in_loop(self.chrome.url()).result()

    async def page(self) -> Page:
        return self.run_in_loop(self.chrome.page()).result()

    async def close(self) -> None:
        return self.run_in_loop(self.chrome.close()).result()

    async def goto(self, url: str):
        return self.run_in_loop(self.chrome.goto(url)).result()

    async def wait(self, milliseconds: int) -> None:
        return self.run_in_loop(self.chrome.wait(milliseconds)).result()

    async def wait_until(self, selector: str) -> None:
        return self.run_in_loop(self.chrome.wait_until(selector)).result()

    async def wait_until_text(self, selector: str) -> None:
        return self.run_in_loop(self.chrome.wait_until_text(selector)).result()

    async def get_html(self) -> str:
        return self.run_in_loop(self.chrome.get_html()).result()

    async def screenshot(self) -> bytes:
        return self.run_in_loop(self.chrome.screenshot()).result()

    async def get_url(self, url: str):
        result = self.run_in_loop(self.chrome.get_url(url)).result()
        write_client_stream(
            StreamNode(
                obj=await self.screenshot(),
                type='bytes',
                metadata={'type': 'bytes', 'url': url}
            ))
        return result

    async def pdf(self) -> str:
        return self.run_in_loop(self.chrome.pdf()).result()

    async def pdf_url(self, url: str) -> str:
        return self.run_in_loop(self.chrome.pdf_url(url)).result()

    async def download(self, url: str) -> str:
        return self.run_in_loop(self.chrome.download(url)).result()

    async def clickable(self) -> List[str]:
        return self.run_in_loop(self.chrome.clickable()).result()

    async def click_str(self, str_element) -> bool:
        return self.run_in_loop(self.chrome.click_str(str_element)).result()

    async def click(self, element) -> bool:
        return self.run_in_loop(self.chrome.click(element)).result()

    async def get_ahrefs(self) -> List[Tuple[str, str]]:
        return self.run_in_loop(self.chrome.get_ahrefs()).result()

    async def get_input_elements(self) -> List[ClickableElementHandle]:
        return self.run_in_loop(self.chrome.get_input_elements()).result()

    async def get_clickable_elements(self) -> List[ClickableElementHandle]:
        return self.run_in_loop(self.chrome.get_clickable_elements()).result()

    async def fill(self, element: ElementHandle, value: str) -> None:
        return self.run_in_loop(self.chrome.fill(element, value)).result()

class ChromeHelpersInternal():
    def __init__(self, cookies: List[Dict] = []):
        self.args = [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-infobars',
            '--disable-dev-shm-usage',
            '--disable-blink-features=AutomationControlled',
            '--ignore-certificate-errors',
            '--no-first-run',
            '--no-service-autorun',
            '--password-store=basic',
            '--use-mock-keychain',
        ]

        self.cookies = cookies
        self._context = None
        self._page = None
        self.playwright = None
        self.browser = None
        self.is_closed = True
        self.wait_fors = {
            'twitter.com': lambda page: self.wait(1500),
            'x.com': lambda page: self.wait(1500),
            'techmeme.com': lambda page: self.wait(1500),
            'arxiv.org': lambda page: self.wait(1000),
        }

    async def safe_click_element(self, element: ElementHandle, timeout=5000):
        try:
            # Ensure the element is in the viewport
            await element.scroll_into_view_if_needed(timeout=timeout)

            # Check if the element is truly visible and clickable
            is_visible = await element.is_visible()
            is_enabled = await element.is_enabled()

            if not is_visible or not is_enabled:
                logging.debug("Element is not visible or not enabled")
                return False

            # Get the bounding box of the element
            box = await element.bounding_box()
            if not box:
                logging.debug("Could not get bounding box for element")
                return False

            # Get the page associated with this element
            page = await element.owner_frame() or await element.page()

            # Click in the center of the element
            x = box['x'] + box['width'] / 2
            y = box['y'] + box['height'] / 2
            await page.mouse.click(x, y)

            logging.debug("Successfully clicked element")
            return True

        except Exception as e:
            logging.error(f"Error interacting with element: {str(e)}")
            return False

    def set_cookies(self, cookies: List[Dict]):
        self.cookies = cookies

    async def __new_page(self, cookies: List[Dict] = []) -> Page:
        if self.is_closed:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=Container().get('chromium_headless', default=True),
                args=self.args
            )

        self._context = await self.browser.new_context(viewport={'width': 1920, 'height': 1080}, accept_downloads=True)  # type: ignore
        cookie_file = Container().get('chromium_cookies', '')
        if os.path.exists(cookie_file):
            result = read_netscape_cookies(cookie_file)
            await self._context.add_cookies(result)

        if self.cookies:
            await self._context.add_cookies(self.cookies)  # type: ignore
        page = await self._context.new_page()
        await page.set_extra_http_headers({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
        self.is_closed = False
        return page

    async def page(self) -> Page:
        if self._page is None:
            self._page = await self.__new_page()
            return self._page
        else:
            return cast(Page, self._page)

    async def url(self) -> str:
        return (await self.page()).url

    async def close(self) -> None:
        if self._page is not None:
            await (await self.page()).close()
        if self.browser is not None:
            await self.browser.close()
        self.is_closed = True

    async def goto(self, url: str):
        try:
            if (await self.page()).url != url:
                await (await self.page()).goto(url, wait_until='load')

                domain = urlparse(url).netloc

                if domain in self.wait_fors:
                    wait_for_lambda = self.wait_fors[domain]
                    await wait_for_lambda(await self.page())

                # wait some fraction of a second for some rendering
                await self.wait(250)

        except Error as _:
            # try new page
            self._page = await self.__new_page()
            await (await self.page()).goto(url)

    async def wait(self, milliseconds: int) -> None:
        await (await self.page()).evaluate("() => { setTimeout(function() { return; }, 0); }")
        await (await self.page()).wait_for_timeout(milliseconds)

    async def wait_until(self, selector: str) -> None:
        element = await (await self.page()).wait_for_selector(selector)

    async def wait_until_text(self, selector: str) -> None:
        element = await (await self.page()).wait_for_selector(f'*:has-text("{selector}")', timeout=5000)

    async def get_html(self) -> str:
        result = await (await self.page()).content()
        if (
            '<embed ' in result
            and ('application/pdf' in result or 'x-google-chrome-pdf' in result)
        ):
            return await self.download((await self.page()).url)
        return result

    async def screenshot(self) -> bytes:
        try:
            return await asyncio.wait_for((await self.page()).screenshot(type='png'), timeout=2)
        except asyncio.TimeoutError as ex:
            logging.debug(f'screenshot timed out with: {ex}')
            return b''

    async def get_url(self, url: str):
        await self.goto(url)

        try:
            html = await self.get_html()
            return html
        except Error as ex:
            # try new page
            self._page = await self.__new_page()
            await self.goto(url)
            html = await self.get_html()
            return html

    async def pdf(self) -> str:
        pdf_path = 'output.pdf'
        await (await self.page()).pdf(path=pdf_path)
        return os.path.abspath(pdf_path)

    async def download(self, url: str) -> str:
        logging.debug(f'ChromeHelpersInternal.download({url})')
        async def requests_download(url, filename: str):
            logging.debug(f'ChromeHelpersInternal.download({url}): using requests to download')
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                async with aiofiles.open(filename, 'wb') as file:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await file.write(chunk)
            return os.path.abspath(filename)

        parsed_url = urlparse(url)
        _, file_extension = os.path.splitext(parsed_url.path)
        if not file_extension:
            file_extension = '.tmp'

        try:
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_filename = temp_file.name
                logging.debug(f'ChromeHelpersInternal.download({url}): force download with html file: {temp_filename}')

            # chrome renders the pdf, and you can't download it. So we force it.
            html = '<html><head></head><body><a href="' + url + '">Download</a></body></html>'
            data_uri = f"data:text/html;base64,{base64.b64encode(html.encode()).decode()}"

            await self.goto(data_uri)

            async with (await self.page()).expect_download(timeout=10000) as download_info:
                link = await (await self.page()).query_selector('a')
                await link.click(modifiers=['Alt'])  # type: ignore
                download = await download_info.value

                await download.save_as(temp_filename)
                return os.path.abspath(temp_filename)
        except Exception as ex:
            logging.debug(f'ChromeHelpersInternal.download({url}) failed with: {ex}, trying requests')
            return await requests_download(url, temp_filename)

    async def pdf_url(self, url: str) -> str:
        if '.pdf' in url:
            return await self.download(url)
        elif '.csv' in url:
            return await self.download(url)
        else:
            await self.goto(url)
            return await self.pdf()

    async def clickable(self) -> List[str]:
        clickable_elements = await (await self.page()).query_selector_all(
            "a, button, input[type='submit'], input[type='button']"
        )
        unique_ids = set()

        for element in clickable_elements:
            unique_ids.add(element)
        return list(unique_ids)

    async def click_str(self, str_element) -> bool:
        # can either be a <a href ...> or a <button> or a straight up id
        # figure out what, then click it
        element = await (await self.page()).query_selector(str_element)
        if element:
            await element.click()
            return True
        else:
            element = await (await self.page()).query_selector(f'#{str_element}')
            if element:
                await element.click()
                return True
        return False

    async def click(self, element: ElementHandle) -> bool:
        await self.safe_click_element(element)
        return True

    async def get_ahrefs(self) -> List[Tuple[str, str]]:
        ahrefs = await (await self.page()).query_selector_all('a')
        result = [(await ahref.evaluate('(el) => el.outerHTML'), await ahref.get_attribute('href') or '') for ahref in ahrefs]
        return result

    async def get_clickable_elements(self) -> List[ClickableElementHandle]:
        page = await self.page()
        clickable_elements = await page.query_selector_all(
            "a, button, input[type='submit'], input[type='button']"
        )
        elements = []
        counter = 0

        for element in clickable_elements:
            element_id = await element.get_attribute('id')
            element_html = await element.evaluate('''(el) => {
                        const clone = el.cloneNode(false);
                        return clone.outerHTML;
                    }''')
            outer_html = await element.evaluate('(el) => el.outerHTML')
            elements.append(ClickableElementHandle(
                element_handle=element,
                scope_id=counter,
                id=element_id or '',
                html=element_html,
                outer_html=outer_html
            ))
            counter+=1
        return elements

    async def get_input_elements(self) -> List[ClickableElementHandle]:
        input_elements = await (await self.page()).query_selector_all(
            "input[type='text'], input[type='email'], input[type='password'], textarea"
        )
        elements: List[ClickableElementHandle] = []
        counter = 0

        for element in input_elements:
            element_id = await element.get_attribute('id')
            element_html = await element.evaluate('''(el) => {
                        const clone = el.cloneNode(false);
                        return clone.outerHTML;
                    }''')
            outer_html = await element.evaluate('(el) => el.outerHTML')
            elements.append(ClickableElementHandle(
                element_handle=element,
                scope_id=counter,
                id=element_id or '',
                html=element_html,
                outer_html=outer_html
            ))
            counter+=1
        return elements

    async def fill(self, element: ElementHandle, value: str) -> None:
        if element is None or not isinstance(element, ElementHandle):
            logging.debug(f'ChromeHelpersInternal.fill() Element {element} is not a valid ElementHandle')
            return
        if not await element.is_editable():
            logging.debug(f'ChromeHelpersInternal.fill() Element {element} is not editable')
            return
        await element.fill(value, timeout=2000)

