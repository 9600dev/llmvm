import os
import time
from typing import Dict, List

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

from helpers.container import Container
from helpers.logging_helpers import setup_logging
from helpers.singleton import Singleton

logging = setup_logging()

class FirefoxHelpers(metaclass=Singleton):
    def __init__(self):
        logging.debug('FirefoxHelpers()')

        profile_directory = Container().get('firefox_profile')
        profile = webdriver.FirefoxProfile(profile_directory)
        options = FirefoxOptions()
        options.headless = False
        options.set_preference("print.always_print_silent", True)
        options.set_preference("print.printer_Mozilla_Save_to_PDF.print_to_file", True)
        options.set_preference("print_printer", "Mozilla Save to PDF")
        options.set_preference("browser.download.dir", Container().get('firefox_download_dir'))
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/plain, application/vnd.ms-excel, text/csv, text/comma-separated-values, application/octet-stream")
        options.profile = profile
        service_args = ['--marionette-port', str(Container().get('firefox_marionette_port'))]
        self.web_driver = webdriver.Firefox(options=options)

    def driver(self):
        return self.web_driver

    @staticmethod
    def __wait() -> None:
        driver = FirefoxHelpers().driver()
        driver.implicitly_wait(5)
        wait = WebDriverWait(driver, 5)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'div')))
        driver.execute_script('setTimeout(function() { return; }, 0);')

    @staticmethod
    def __goto(url: str):
        driver = FirefoxHelpers().driver()
        if driver.current_url != url:
            driver.get(url)
            FirefoxHelpers.__wait()
        return driver

    @staticmethod
    def __get_download_file(wait_time_seconds: int = 5) -> str:
        import time
        driver = FirefoxHelpers().driver()
        driver.get('about:downloads')

        end_time = time.time() + wait_time_seconds
        filename = ''
        while True:
            try:
                filename = driver.execute_script("return document.querySelector('#contentAreaDownloadsView .downloadMainArea .downloadContainer description:nth-of-type(1)').value")
            except Exception as ex:
                pass
            time.sleep(1)
            if time.time() > end_time:
                break

        if os.path.exists(os.path.join(Container().get('firefox_download_dir'), filename)):
            return os.path.abspath(os.path.join(Container().get('firefox_download_dir'), filename))
        else:
            return ''

    @staticmethod
    def pdf_url(url: str) -> str:
        driver = FirefoxHelpers().__goto(url)
        driver.execute_script('window.print();')
        driver.execute_script('setTimeout(function() { return; }, 0);')

        time.sleep(5)

        if os.path.exists('mozilla.pdf'):
            return os.path.abspath('mozilla.pdf')
        else:
            logging.debug('WebHelpers.get_url_firefox: pdf not found')
            return ''

    @staticmethod
    def get_url(url: str) -> str:
        """
        Extracts the text from a url using the Firefox browser.
        This is useful for hard to extract text, an exception thrown by the other functions,
        or when searching/extracting from sites that require logins liked LinkedIn, Facebook, Gmail etc.
        """
        try:
            driver = FirefoxHelpers().__goto(url)
            return driver.page_source
        except Exception as e:
            logging.debug(e)
            return str(e)

    @staticmethod
    def __clickable() -> List[WebElement]:
        driver = FirefoxHelpers().driver()
        a = driver.find_elements(By.TAG_NAME, 'a')
        buttons = driver.find_elements(By.TAG_NAME, 'button')
        inputs = driver.find_elements(By.TAG_NAME, 'input')

        return a + buttons + inputs

    @staticmethod
    def get_clickable(url: str) -> List[str]:
        """
        Extracts and returns the text for all links, buttons and inputs on a given page url.
        You can pass the text of any of these clickable elements to the click(url, link_or_button_text) function to click on it.
        """
        try:
            driver = FirefoxHelpers().__goto(url)
            clickable = FirefoxHelpers.__clickable()
            links = set([item.text for item in clickable])
            return list(links)
        except Exception as e:
            logging.debug(e)
            return []

    @staticmethod
    def click(url: str, link_or_button_text: str) -> str:
        """
        Goes to the url and clicks on the link or button with the specified text and returns
        the page source of the resulting page.
        """
        def first(predicate, iterable):
            try:
                result = next(x for x in iterable if predicate(x))
                return result
            except StopIteration as ex:
                return None

        driver = FirefoxHelpers().driver()
        try:
            driver = FirefoxHelpers().__goto(url)
            FirefoxHelpers().wait_until(url, link_or_button_text)
            clickable = FirefoxHelpers.__clickable()

            if first(lambda n: n.text == link_or_button_text, clickable):
                node = first(lambda n: n.text == link_or_button_text, clickable)
                node.click()  # type: ignore
                FirefoxHelpers().__wait()
                return driver.page_source
            else:
                return ''
        except Exception as e:
            logging.debug(e)
            return ''

    @staticmethod
    def wait_until(url: str, link_or_button_text: str, duration: int = 10) -> bool:
        """
        Goes to the url and waits until the link or button with the specified text is clickable and returns
        the page source of the resulting page.
        """
        def wait_for_text(text: str):
            # This function will be used as a wait condition
            def condition(driver):
                elements = driver.find_elements(By.CSS_SELECTOR, 'a, input, button')
                return any(element.text == text for element in elements)
            return condition

        try:
            driver = FirefoxHelpers().__goto(url)
            wait = WebDriverWait(driver, duration)
            return wait.until(wait_for_text(link_or_button_text))
        except Exception as ex:
            logging.debug(ex)
            return False

    @staticmethod
    def get_downloaded_file_text() -> str:
        filename = FirefoxHelpers().__get_download_file()
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return f.read()
        else:
            return ''

    @staticmethod
    def get_downloaded_filename() -> str:
        return FirefoxHelpers().__get_download_file()
