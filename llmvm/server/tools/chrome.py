import asyncio
import concurrent
import concurrent.futures
import datetime as dt
import os
import threading
import time
from typing import Dict, List, cast
from urllib.parse import urlparse

import aiofiles
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

    async def clickable(self) -> List[str]:
        return self.run_in_loop(self.chrome.clickable()).result()

    async def click(self, element) -> None:
        return self.run_in_loop(self.chrome.click(element)).result()

    async def get_input_elements(self) -> List[ElementHandle]:
        return self.run_in_loop(self.chrome.get_input_elements()).result()

    async def get_clickable_elements(self) -> List[ElementHandle]:
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
        self.wait_fors = {
            'twitter.com': lambda page: self.wait(1500),
            'techmeme.com': lambda page: self.wait(1500)
        }

    def set_cookies(self, cookies: List[Dict]):
        self.cookies = cookies

    async def __new_page(self, cookies: List[Dict] = []) -> Page:
        if self.playwright is None or self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=Container().get('chrome_headless', default=True),
                args=self.args
            )

        self._context = await self.browser.new_context(viewport={'width': 1920, 'height': 1080}, accept_downloads=True)
        cookie_file = Container().get('chrome_cookies', '')
        if os.path.exists(cookie_file):
            result = read_netscape_cookies(cookie_file)
            await self._context.add_cookies(result)

        if self.cookies:
            await self._context.add_cookies(self.cookies)  # type: ignore
        return await self._context.new_page()

    async def page(self) -> Page:
        if self._page is None:
            self._page = await self.__new_page()
            return self._page
        else:
            return cast(Page, self._page)

    async def close(self) -> None:
        if self._page is not None:
            await (await self.page()).close()
        if self.browser is not None:
            await self.browser.close()

    async def goto(self, url: str):
        try:
            if (await self.page()).url != url:
                await (await self.page()).goto(url)

                domain = urlparse(url).netloc
                if domain in self.wait_fors:
                    wait_for_lambda = self.wait_fors[domain]
                    await wait_for_lambda(await self.page())

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
        return await (await self.page()).content()

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
            await self._page.wait_for_load_state('networkidle')
            html = await self.get_html()
            return html

    async def pdf(self) -> str:
        pdf_path = 'output.pdf'
        await (await self.page()).pdf(path=pdf_path)
        return os.path.abspath(pdf_path)

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
            pdf_path = '/tmp/output.pdf'
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            try:
                if 'arxiv.org' in url:
                    return await requests_download(url, pdf_path)

                await self.goto(url)
                await (await self.page()).pdf(path=pdf_path)
                return os.path.abspath(pdf_path)
            except Exception as ex:
                logging.debug(f'pdf_url({url}) failed with: {ex}, trying requests')
                return await requests_download(url, pdf_path)
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

    async def get_clickable_elements(self) -> List[ElementHandle]:
        clickable_elements = await (await self.page()).query_selector_all(
            "a, button, input[type='submit'], input[type='button']"
        )
        return clickable_elements

    async def get_input_elements(self) -> List[ElementHandle]:
        input_elements = await (await self.page()).query_selector_all(
            "input[type='text'], input[type='email'], input[type='password'], textarea"
        )
        return input_elements

    async def fill(self, element: ElementHandle, value: str) -> None:
        await element.fill(value)

