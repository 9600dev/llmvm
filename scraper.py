import datetime as dt
import hashlib
import re
from turtle import setup
from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup
from numpy import dtype

from container import Container
from helpers.firefox import FirefoxHelpers
from helpers.logging_helpers import setup_logging
from helpers.webhelpers import WebHelpers
from persistent_cache import PersistentCache

logging = setup_logging()

class ScraperMeta():
    def __init__(
        self,
        url: str,
        content: str,
        html_content: str,
        parent: str,
        last_scraped: dt.datetime,
    ):
        self.url = url
        self.content = content
        self.html_content = html_content
        self.parent = parent
        self.last_scraped = last_scraped

    def __hash__(self):
        content = f"{self.content}".encode()
        return int(hashlib.md5(content).hexdigest(), 16)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

class Scraper():
    def __init__(
        self
    ):
        self.browser = FirefoxHelpers()
        self.cache = PersistentCache(Container().get('cache_directory') + '/scraper.db')

    def normalize_url(self, url):
        parsed = urlparse(url)

        # Decode percent-encoded paths
        path = parsed.path
        if path:
            path = path.rstrip('/')

        # Sort query parameters
        query = sorted(parse_qsl(parsed.query))

        # Filter out unwanted query parameters
        query = [(k, v) for k, v in query if k not in ["utm_source", "utm_medium"]]  # example unwanted parameters

        # Construct the normalized URL
        normalized = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower().split(':')[0],  # Remove port
            path=path,
            query=urlencode(query),
            fragment=''
        )
        return str(urlunparse(normalized))

    def get(self, url: str) -> Optional[ScraperMeta]:
        normalized_url = self.normalize_url(url)
        if self.cache.has_key(normalized_url):
            return self.cache.get(normalized_url)  # type: ignore
        else:
            return None

    def get_site(self, url_regex: str) -> List[ScraperMeta]:
        results = []
        for url in self.cache.keys():  # type: ignore
            if re.search(url_regex, url):
                results.append(self.cache.get(url))
        return results

    def crawl(
        self,
        start_url: str,
        max_depth: int = 2,
        include_url_regex: List[str] = [],
        exclude_url_regex: List[str] = [],
    ):
        def __get_helper(url: str, parent: str, max_depth: int, depth: int):
            if depth > max_depth:
                return

            normalized_url = self.normalize_url(url)

            # check to see if the url is in the excluded regex list
            for regex in exclude_url_regex:
                if re.search(regex, normalized_url):
                    return

            # check to see if the url is in the included regex list
            if include_url_regex:
                for regex in include_url_regex:
                    if re.search(regex, normalized_url):
                        break
                else:
                    return

            logging.debug('crawling: %s', normalized_url)
            html_content = self.browser.get_url(self.normalize_url(url))
            soup = BeautifulSoup(html_content, 'html.parser')
            scraped = dt.datetime.now()

            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    normalized_href = self.normalize_url(href)
                    if not self.cache.has_key(normalized_href):
                        __get_helper(normalized_href, normalized_url, max_depth, depth + 1)

            text_content = WebHelpers.convert_html_to_markdown(html_content)

            self.cache.set(normalized_url, ScraperMeta(
                url=normalized_url,
                content=text_content,
                html_content=html_content,
                parent=parent,
                last_scraped=scraped,
            ))

        __get_helper(start_url, '', max_depth, 0)
