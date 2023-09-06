import asyncio
import datetime as dt
import os
import time
from typing import Any, Callable, Dict, List

import nest_asyncio
import requests
from playwright.sync_api import Error, sync_playwright
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from yfinance import download

from container import Container
from helpers.logging_helpers import setup_logging
from helpers.singleton import Singleton

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
        profile_directory = Container().get('firefox_profile')
        self.prefs = {
            "profile": profile_directory,
            "print.always_print_silent": True,
            "print.printer_Mozilla_Save_to_PDF.print_to_file": True,
            "print_printer": "Mozilla Save to PDF",
            "browser.download.dir": Container().get('firefox_download_dir'),
            "browser.download.folderList": 2,
            "browser.helperApps.neverAsk.saveToDisk": "text/plain, application/vnd.ms-excel, text/csv, text/comma-separated-values, application/octet-stream",
        }
        self._context = None
        self._page = None
        self.playwright = None
        self.browser = None

    def __new_page(self):
        if self.playwright is None or self.browser is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.firefox.launch(
                headless=False,
                firefox_user_prefs=self.prefs
            )

        self._context = self.browser.new_context(accept_downloads=True)
        cookie_file = Container().get('firefox_cookies')
        if cookie_file:
            result = read_netscape_cookies(cookie_file)
            self._context.add_cookies(result)
        return self._context.new_page()

    @property
    def page(self):
        if self._page is None:
            self._page = self.__new_page()
        return self._page

    def goto(self, url: str):
        try:
            if self.page.url != url:
                self.page.goto(url)
        except Error as ex:
            # try new page
            self._page = self.__new_page()
            self.page.goto(url)

    def wait(self, milliseconds: int) -> None:
        self.page.evaluate("() => { setTimeout(function() { return; }, 0); }")
        self.page.wait_for_timeout(milliseconds)

    def wait_until(self, selector: str) -> None:
        element = self.page.wait_for_selector(selector)

    def wait_until_text(self, selector: str) -> None:
        element = self.page.wait_for_selector(f'*:has-text("{selector}")', timeout=5000)

    def get_html(self) -> str:
        return self.page.content()

    def get_url(self, url: str):
        self.goto(url)
        try:
            html = self.get_html()
            return html
        except Error as ex:
            # try new page
            self._page = self.__new_page()
            self.goto(url)
            html = self.get_html()
            return html

    def pdf(self) -> str:
        self.page.evaluate("() => { setTimeout(function() { return; }, 0); }")
        self.page.evaluate("() => { window.print(); }")
        self.page.evaluate("() => { setTimeout(function() { return; }, 0); }")

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

    def pdf_url(self, url: str) -> str:
        def requests_download(url, filename: str):
            logging.debug('pdf_url: using requests to download')
            response = requests.get(url, allow_redirects=True, stream=True)
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return os.path.abspath(filename)

        if '.pdf' in url:
            if os.path.exists('mozilla.pdf'):
                os.remove('mozilla.pdf')
            if os.path.exists('/tmp/mozilla.pdf'):
                os.remove('/tmp/mozilla.pdf')
            try:
                if 'arxiv.org' in url:
                    return requests_download(url, '/tmp/mozilla.pdf')

                self.page.set_content(f"""
                        <html>
                        <body>
                        <a href="{url}" download id="downloadLink">Download</a>
                        </body>
                        </html>
                        """)

                with self.page.expect_download() as download_info:
                    self.page.click('#downloadLink')
                self.page.wait_for_timeout(2000)

                d = download_info.value
                d.save_as('mozilla.pdf')
                end_time = time.time() + 8
                while time.time() < end_time:
                    if os.path.exists('mozilla.pdf'):
                        return os.path.abspath('mozilla.pdf')
                    else:
                        time.sleep(1)

                logging.debug(f'pdf_url({url}) failed, trying requests')
                return requests_download(url, '/tmp/mozilla.pdf')
            except Exception as ex:
                logging.debug(f'pdf_url({url}) failed with: {ex}, trying requests')
                return requests_download(url, '/tmp/mozilla.pdf')
        else:
            self.goto(url)
            return self.pdf()

    def clickable(self) -> List[str]:
        clickable_elements = self.page.query_selector_all(
            "a, button, input[type='submit'], input[type='button']"
        )
        unique_ids = set()

        for element in clickable_elements:
            unique_ids.add(element)
        return list(unique_ids)

    def click(self, element) -> None:
        element.click()
