import asyncio
import concurrent
import concurrent.futures
import datetime as dt
import os
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, cast

import aiofiles
import httpx
import nest_asyncio
from PIL import Image
from playwright.async_api import Error, Page, async_playwright
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from yfinance import download

from container import Container
from helpers.helpers import Helpers, write_client_stream
from helpers.logging_helpers import setup_logging
from helpers.singleton import Singleton
from objects import StreamNode

nest_asyncio.apply()
logging = setup_logging()


class BrowserPageInternal():
    def __init__(self, internals: 'BrowserHelperInternal', page: Page):
        self.internals = internals
        self._page = page

    async def page(self) -> Page:
        return self._page  # type: ignore

    async def goto(self, url: str):
        try:
            if (await self.page()).url != url:
                await (await self.page()).goto(url)
        except Error as ex:
            # try new page
            self._page = await self.internals.page()
            await (await self.page()).goto(url)

    async def set_cookies(self, cookies: List[Dict[str, Any]]):
        await (await self.page()).context.add_cookies(cookies)  # type: ignore

    async def wait(self, milliseconds: int) -> None:
        await (await self.page()).evaluate("() => { setTimeout(function() { return; }, 0); }")
        await (await self.page()).wait_for_timeout(milliseconds)

    async def wait_until(self, selector: str) -> None:
        element = await (await self.page()).wait_for_selector(selector)

    async def wait_until_text(self, selector: str) -> None:
        element = await (await self.page()).wait_for_selector(f'*:has-text("{selector}")', timeout=5000)

    async def get_html(self) -> str:
        return await (await self.page()).content()

    async def screenshot(self) -> bytes:
        screenshot_data = await (await self.page()).screenshot(type='png')
        return screenshot_data

    async def get_url(self, url: str):
        await self.goto(url)

        try:
            html = await self.get_html()
            return html
        except Error as ex:
            # try new page
            self._page = await self.internals.page()
            await self.goto(url)
            await self._page.wait_for_load_state('networkidle')
            html = await self.get_html()
            return html

    async def pdf(self) -> str:
        await (await self.page()).evaluate("() => { setTimeout(function() { return; }, 0); }")
        await (await self.page()).evaluate("() => { window.print(); }")
        await (await self.page()).evaluate("() => { setTimeout(function() { return; }, 0); }")

        # we have to wait for the pdf to be produced
        counter = 0
        while not os.path.exists('mozilla.pdf') and counter < 7:
            time.sleep(1)
            counter += 1

        if os.path.exists('mozilla.pdf'):
            return os.path.abspath('mozilla.pdf')
        else:
            logging.debug('pdf: pdf not found')
            return ''

    async def pdf_url(self, url: str) -> str:
        async def requests_download(url, filename: str):
            logging.debug('pdf_url: using requests to download')
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                async with aiofiles.open(filename, 'wb') as file:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await file.write(chunk)

            return os.path.abspath(filename)

        if '.pdf' in url:
            if os.path.exists('mozilla.pdf'):
                os.remove('mozilla.pdf')
            if os.path.exists('/tmp/mozilla.pdf'):
                os.remove('/tmp/mozilla.pdf')
            try:
                if 'arxiv.org' in url:
                    return await requests_download(url, '/tmp/mozilla.pdf')

                await (await self.page()).set_content(f"""
                        <html>
                        <body>
                        <a href="{url}" download id="downloadLink">Download</a>
                        </body>
                        </html>
                        """)

                async with (await self.page()).expect_download() as download_info:
                    await (await self.page()).click('#downloadLink')
                await (await self.page()).wait_for_timeout(2000)

                d = await download_info.value
                await d.save_as('mozilla.pdf')
                end_time = time.time() + 8
                while time.time() < end_time:
                    if os.path.exists('mozilla.pdf'):
                        return os.path.abspath('mozilla.pdf')
                    else:
                        await asyncio.sleep(1)

                logging.debug(f'pdf_url({url}) failed, trying requests')
                return await requests_download(url, '/tmp/mozilla.pdf')
            except Exception as ex:
                logging.debug(f'pdf_url({url}) failed with: {ex}, trying requests')
                return await requests_download(url, '/tmp/mozilla.pdf')
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

    async def click(self, element) -> None:
        await element.click()

    async def close(self) -> None:
        await (await self.page()).close()
        self._page = None


class BrowserHelperInternal(metaclass=Singleton):
    def __init__(self):
        self.prefs = {
            "print.always_print_silent": True,
            "print.printer_Mozilla_Save_to_PDF.print_to_file": True,
            "print_printer": "Mozilla Save to PDF",
            "browser.download.dir": Container().get('browser_download_directory', default=os.path.expanduser('~')),
            "browser.download.folderList": 2,
            "browser.helperApps.neverAsk.saveToDisk": "text/plain, application/vnd.ms-excel, text/csv, text/comma-separated-values, application/octet-stream",
        }

        self._context = None
        self.playwright = None
        self.browser = None
        self.init_lock = asyncio.Lock()

    async def initialize_browser(self):
        async with self.init_lock:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.firefox.launch(
                headless=Container().get('browser_headless', default=False),
                firefox_user_prefs=self.prefs
            )

            self._context = await self.browser.new_context(accept_downloads=True)
            cookie_file = Container().get('browser_cookies', '')
            if os.path.exists(cookie_file):
                result = Helpers.read_netscape_cookies(open(cookie_file, 'r').read())
                await self._context.add_cookies(result)  # type: ignore

    async def page(self) -> Page:
        await self.initialize_browser()
        context = await self.browser.new_context(accept_downloads=True)  # type: ignore
        page = await context.new_page()
        return page

    async def browser_page(self) -> 'BrowserPageInternal':
        await self.initialize_browser()
        return BrowserPageInternal(self, await self.page())


class BrowserHelper(metaclass=Singleton):
    _is_initialized = False

    def __init__(self):
        if not self._is_initialized:
            self.loop = asyncio.SelectorEventLoop()
            self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.thread.start()
            self.firefox = BrowserHelperInternal()
            BrowserHelper._is_initialized = True

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

    def browser_page(self, cookies_file_content: str = '') -> 'BrowserPage':
        return BrowserPage(self)


class BrowserPage():
    def __init__(self, browser_helper: BrowserHelper):
        self.helper = browser_helper
        self.browser_page = BrowserPageInternal(self.helper.firefox, self.helper.run_in_loop(self.helper.firefox.page()).result())

    async def goto(self, url: str):
        return self.helper.run_in_loop(self.browser_page.goto(url)).result()

    async def wait(self, milliseconds: int) -> None:
        return self.helper.run_in_loop(self.browser_page.wait(milliseconds)).result()

    async def wait_until(self, selector: str) -> None:
        return self.helper.run_in_loop(self.browser_page.wait_until(selector)).result()

    async def wait_until_text(self, selector: str) -> None:
        return self.helper.run_in_loop(self.browser_page.wait_until_text(selector)).result()

    async def get_html(self) -> str:
        return self.helper.run_in_loop(self.browser_page.get_html()).result()

    async def screenshot(self) -> bytes:
        return self.helper.run_in_loop(self.browser_page.screenshot()).result()

    async def set_cookies(self, cookies: List[Dict[str, Any]]):
        return self.helper.run_in_loop(self.browser_page.set_cookies(cookies)).result()

    async def get_url(self, url: str):
        result = self.helper.run_in_loop(self.browser_page.get_url(url)).result()
        write_client_stream(
            StreamNode(
                obj=await self.screenshot(),
                type='bytes',
                metadata={'type': 'bytes', 'url': url}
            ))
        return result

    async def pdf(self) -> str:
        return self.helper.run_in_loop(self.browser_page.pdf()).result()

    async def pdf_url(self, url: str) -> str:
        return self.helper.run_in_loop(self.browser_page.pdf_url(url)).result()

    async def clickable(self) -> List[str]:
        return self.helper.run_in_loop(self.browser_page.clickable()).result()

    async def click(self, element) -> None:
        return self.helper.run_in_loop(self.browser_page.click(element)).result()

    async def close(self) -> None:
        return self.helper.run_in_loop(self.browser_page.close()).result()