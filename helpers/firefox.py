import datetime as dt
import os
import time
from typing import Any, Callable, Dict, List

import nest_asyncio
from playwright.sync_api import sync_playwright
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from helpers.container import Container
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
        self._page = None

    @property
    def page(self):
        if self._page is None:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.firefox.launch(
                headless=False,
                firefox_user_prefs=self.prefs
            )

            context = self.browser.new_context()
            cookie_file = Container().get('firefox_cookies')
            if cookie_file:
                result = read_netscape_cookies(cookie_file)
                context.add_cookies(result)
            self._page = context.new_page()

        return self._page

    def goto(self, url: str):
        if self.page.url != url:
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
        return self.get_html()

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
        self.goto(url)
        self.wait(1000)
        return self.pdf()

    def clickable(self) -> List[str]:
        clickable_elements = self.page.query_selector_all(
            "a, button, input[type='submit'], input[type='button'], input[type='reset']"
        )
        unique_ids = set()

        for element in clickable_elements:
            element_id = element.get_attribute("id")
            if element_id:  # Check if the element has an id attribute
                unique_ids.add(element_id)

        return list(unique_ids)

    def click(self, selector: str) -> None:
        element_to_click = self.page.query_selector(selector)
        if element_to_click:
            element_to_click.click()
