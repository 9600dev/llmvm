import asyncio
import concurrent
import concurrent.futures
import datetime as dt
import os
import threading
import time
from typing import List, cast

import aiofiles
import httpx
import nest_asyncio
from playwright.async_api import Error, Page, async_playwright

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
                    pass
    return cookies


class FirefoxHelpers(metaclass=Singleton):
    def __init__(self):
        self.loop = asyncio.SelectorEventLoop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        self.firefox = FirefoxHelpersInternal()

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

    async def goto(self, url: str):
        return self.run_in_loop(self.firefox.goto(url)).result()

    async def wait(self, milliseconds: int) -> None:
        return self.run_in_loop(self.firefox.wait(milliseconds)).result()

    async def wait_until(self, selector: str) -> None:
        return self.run_in_loop(self.firefox.wait_until(selector)).result()

    async def wait_until_text(self, selector: str) -> None:
        return self.run_in_loop(self.firefox.wait_until_text(selector)).result()

    async def get_html(self) -> str:
        return self.run_in_loop(self.firefox.get_html()).result()

    async def screenshot(self) -> bytes:
        return self.run_in_loop(self.firefox.screenshot()).result()

    async def get_url(self, url: str):
        result = self.run_in_loop(self.firefox.get_url(url)).result()
        write_client_stream(
            StreamNode(
                obj=await self.screenshot(),
                type='bytes',
                metadata={'type': 'bytes', 'url': url}
            ))
        return result

    async def pdf(self) -> str:
        return self.run_in_loop(self.firefox.pdf()).result()

    async def pdf_url(self, url: str) -> str:
        return self.run_in_loop(self.firefox.pdf_url(url)).result()

    async def clickable(self) -> List[str]:
        return self.run_in_loop(self.firefox.clickable()).result()

    async def click(self, element) -> None:
        return self.run_in_loop(self.firefox.click(element)).result()


class FirefoxHelpersInternal(metaclass=Singleton):
    def __init__(self):
        self.prefs = {
            "print.always_print_silent": True,
            "print.printer_Mozilla_Save_to_PDF.print_to_file": True,
            "print_printer": "Mozilla Save to PDF",
            "browser.download.dir": Container().get('firefox_download_directory', default=os.path.expanduser('~')),
            "browser.download.folderList": 2,
            "browser.helperApps.neverAsk.saveToDisk": "text/plain, application/vnd.ms-excel, text/csv, text/comma-separated-values, application/octet-stream",
        }

        profile_directory = Container().get('firefox_profile_directory', '')
        if os.path.exists(profile_directory):
            self.prefs.update({"profile": profile_directory})

        self._context = None
        self._page = None
        self.playwright = None
        self.browser = None

    async def __new_page(self):
        if self.playwright is None or self.browser is None:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.firefox.launch(
                headless=Container().get('firefox_headless', default=True),
                firefox_user_prefs=self.prefs
            )

        self._context = await self.browser.new_context(accept_downloads=True)
        cookie_file = Container().get('firefox_cookies', '')
        if os.path.exists(cookie_file):
            result = read_netscape_cookies(cookie_file)
            await self._context.add_cookies(result)
        return await self._context.new_page()

    async def page(self) -> Page:
        if self._page is None:
            self._page = await self.__new_page()
            return self._page
        else:
            return cast(Page, self._page)

    async def goto(self, url: str):
        try:
            if (await self.page()).url != url:
                await (await self.page()).goto(url)
        except Error as ex:
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
        screenshot_data = await (await self.page()).screenshot(type='png')
        return screenshot_data

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
