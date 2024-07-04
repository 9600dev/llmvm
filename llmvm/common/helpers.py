import asyncio
import base64
import datetime as dt
import glob
import importlib
import inspect
import io
import itertools
import math
import os
import re
import traceback
import dateparser
import typing
from collections import Counter
from enum import Enum, IntEnum
from functools import reduce
from importlib import resources
from itertools import cycle, islice
from logging import Logger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse
import zlib
from zoneinfo import ZoneInfo

import httpx
import nest_asyncio
import psutil
from dateutil.relativedelta import relativedelta
from docstring_parser import parse
from PIL import Image

from llmvm.common.objects import (Content, ImageContent, MarkdownContent,
                                  Message, StreamNode, System, User)


def write_client_stream(obj):
    if isinstance(obj, bytes):
        obj = StreamNode(obj, type='bytes')

    frame = inspect.currentframe()
    while frame:
        # Check if 'self' exists in the frame's local namespace
        if 'stream_handler' in frame.f_locals:
            asyncio.run(frame.f_locals['stream_handler'](obj))
            return

        instance = frame.f_locals.get('self', None)
        if hasattr(instance, 'stream_handler'):
            asyncio.run(instance.stream_handler(obj))
            return
        frame = frame.f_back


class Helpers():
    @staticmethod
    def remove_duplicates(lst, key_func=lambda x: x):
        seen = set()
        result = list(filter(lambda x: key_func(x) not in seen and not seen.add(key_func(x)), lst))
        try:
            # check to see if any of the strings in the list are substrings of another list item, and if so, remove that
            sub_dups = [a for a in result if not any(key_func(a) in key_func(b) for b in result if a != b)]
            return sub_dups
        except Exception:
            return result

    @staticmethod
    def anthropic_image_tok_count(base64_encoded: str):
        # go from base64 encoded to bytes
        image = base64.b64decode(base64_encoded)
        # open the image
        img = Image.open(io.BytesIO(image))
        return (img.width * img.height) // 750

    @staticmethod
    def anthropic_resize(image_bytes: bytes) -> bytes:
        image_type = Helpers.classify_image(image_bytes)
        pil_extension = 'JPEG'
        if image_type == 'image/png': pil_extension = 'PNG'
        if image_type == 'image/webp': pil_extension = 'WEBP'

        image = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = image.size

        # Determine the aspect ratio and corresponding max dimensions
        aspect_ratio = original_width / original_height
        max_dimensions = {
            (1, 1): (1092, 1092),
            (3, 4): (951, 1268),
            (2, 3): (896, 1344),
            (9, 16): (819, 1456),
            (1, 2): (784, 1568)
        }

        # Find the closest aspect ratio and its max dimensions
        closest_ratio = min(max_dimensions.keys(), key=lambda x: abs((x[0]/x[1]) - aspect_ratio))
        max_width, max_height = max_dimensions[closest_ratio]

        # Check if the image exceeds the maximum dimensions
        if original_width > max_width or original_height > max_height:
            # Resize the image
            image.thumbnail((max_width, max_height), Image.LANCZOS)

        # Save or return the image
        resized = io.BytesIO()
        image.save(resized, format=pil_extension)
        return resized.getvalue()

    @staticmethod
    async def download_bytes(url_or_file: str) -> bytes:
        url_result = urlparse(url_or_file)
        headers = {  # type: ignore
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  # NOQA
        }
        stream = b''

        if url_result.scheme == 'http' or url_result.scheme == 'https':
            async with httpx.AsyncClient() as client:
                response = await client.get(url_or_file, headers=headers, follow_redirects=True, timeout=10)
                stream = io.BytesIO(response.content)
        else:
            try:
                with open(url_or_file, 'rb') as file:
                    stream = io.BytesIO(file.read())
            except FileNotFoundError:
                raise ValueError('The supplied argument url_or_file: {} is not a correct filename or url.'.format(url_or_file))

        return stream.getvalue()

    @staticmethod
    async def download(url_or_file: str) -> str:
        stream = await Helpers.download_bytes(url_or_file)
        if stream:
            return stream.decode('utf-8')
        return ''

    @staticmethod
    async def get_image_fuzzy_url(logging, url: str, image_url: str, min_width: int, min_height: int) -> bytes:
        parsed_url = urlparse(url)
        path = parsed_url.path
        without_file = parsed_url.scheme + "://" + parsed_url.netloc + os.path.dirname(path)

        try:
            if image_url.startswith('http'):
                result = await Helpers.download_bytes(image_url)
                width, height = Helpers.image_size(result)
                if width >= min_width and height >= min_height:
                    return result

            elif image_url.startswith('/'):
                result = await Helpers.download_bytes(parsed_url.scheme + "://" + parsed_url.netloc + image_url)
                width, height = Helpers.image_size(result)
                if width >= min_width and height >= min_height:
                    return result

            else:
                result = await Helpers.download_bytes(without_file + image_url)
                width, height = Helpers.image_size(result)
                if width >= min_width and height >= min_height:
                    return result
        except Exception as e:
            logging.debug(f"Error downloading image {url} {image_url}: {e}")
            return b''

        return b''

    @staticmethod
    async def markdown_content_to_messages(
        logging,
        markdown_content: MarkdownContent,
        min_width: int,
        min_height: int
    ) -> List[Content]:
        pattern = r'(.*?)!\[(?:.*?)\]\((.*?)\)|(.+?)$'
        chunks = []
        tasks = []
        last_end = 0
        content = markdown_content.get_str()
        content_list: List[Content] = []
        idx = 0

        for match in re.finditer(pattern, content, re.DOTALL):
            before, url, after = match.groups()

            if before and before != content[last_end:match.start()] and last_end != match.start():
                content_list.append(Content(content[last_end:match.start()]))
                idx += 1

            if before:
                content_list.append(Content(before + '![]('))
                idx += 1

            if url:
                task = asyncio.create_task(
                    Helpers.get_image_fuzzy_url(logging, markdown_content.url, url, min_width, min_height)
                )
                tasks.append((idx, task, url))
                content_list.append(Content())
                idx += 1

            if after:
                content_list.append(Content(after))
                idx += 1

            last_end = match.end()

        if last_end < len(content):
            content_list.append(Content(content[last_end:]))
            idx += 1

        for idx, task, image_url in tasks:
            image_bytes = await task
            if image_bytes:
                content_list[idx] = ImageContent(image_bytes, image_url)  # type: ignore

        # we're going to collapse the content list here. if there is two text items in a row, we're going to combine them
        # first, drop all empty content
        collapsed_content_list = [c for c in content_list if c.sequence]
        combined_content_list = []
        current_text = ""

        for c in collapsed_content_list:
            if not isinstance(c, ImageContent):
                # If it's a Content instance, add its text to current_text
                current_text += c.get_str()
            else:
                # If current_text has accumulated content, add it to the list as a new Content instance
                if current_text:
                    combined_content_list.append(Content(current_text))
                    current_text = ""  # Reset current_text for the next batch of text
                # Add the non-text content directly to the list
                combined_content_list.append(c)

        # If there's any remaining text in current_text, add it to the list as a new Content instance
        if current_text:
            combined_content_list.append(Content(current_text))

        return combined_content_list

    @staticmethod
    async def markdown_content_to_messages_deprecated(
        logging,
        markdown_content: MarkdownContent,
        min_width: int,
        min_height: int
    ) -> List[Content]:
        markdown_str = markdown_content.get_str()
        tasks = []
        pattern = r"!\[(.*?)\]\((.*?)\)|([^!]+)"

        content_list: List[Content] = [Content()] * sum(1 for _ in re.finditer(pattern, markdown_str))

        matches = re.finditer(pattern, markdown_str)

        for idx, match in enumerate(matches):
            if match.group(1) is not None:
                # This is an image
                image_url = str(match.group(2))
                task = asyncio.create_task(
                    Helpers.get_image_fuzzy_url(logging, markdown_content.url, image_url, min_width, min_height)
                )
                tasks.append((idx, task, image_url))
            elif match.group(3) is not None:
                # This is text
                text = match.group(3).strip()
                if text:
                    content_list[idx] = Content(text)  # type: ignore

        for idx, task, image_url in tasks:
            image_bytes = await task
            if image_bytes:
                content_list[idx] = ImageContent(image_bytes, image_url)  # type: ignore

        # should we be collapsing the content list here?
        results = [c for c in content_list if c.sequence]
        return results

    @staticmethod
    def last_day_of_quarter(year, quarter):
        start_month = 3 * quarter - 2
        end_month = start_month + 2

        if end_month > 12:
            end_month = 12

        last_day = (dt.datetime(year, end_month, 1) + dt.timedelta(days=31)).replace(day=1) - dt.timedelta(days=1)
        return last_day

    @staticmethod
    def parse_relative_datetime(relative_expression: str, timezone: Optional[str] = None) -> dt.datetime:
        if relative_expression.startswith('Q'):
            quarter = int(relative_expression[1:])
            return Helpers.last_day_of_quarter(dt.datetime.now().year, quarter)

        tz = dt.datetime.now().astimezone().tzinfo

        if timezone:
            tz = ZoneInfo(timezone)

        if 'now' in relative_expression:
            return dt.datetime.now(tz)

        parts = relative_expression.split()

        if len(parts) != 2:
            return dateparser.parse(relative_expression)  # type: ignore

        value = int(parts[0])
        unit = parts[1].lower()

        if unit == "days":
            return dt.datetime.now(tz) + dt.timedelta(days=value)
        elif unit == "months":
            return dt.datetime.now(tz) + relativedelta(months=value)
        elif unit == "years":
            return dt.datetime.now(tz) + relativedelta(years=value)
        elif unit == "hours":
            return dt.datetime.now(tz) + dt.timedelta(hours=value)
        else:
            return dateparser.parse(relative_expression)  # type: ignore

    @staticmethod
    def load_resize_save(raw_data: bytes, output_format='PNG', max_size=5 * 1024 * 1024) -> bytes:
        if output_format not in ['PNG', 'JPEG', 'WEBP']:
            raise ValueError('Invalid output format')

        temp_output = io.BytesIO()
        result: bytes
        with Image.open(io.BytesIO(raw_data)) as im:
            # convert to the required format
            im.save(temp_output, format=output_format)
            temp_output.seek(0)
            result = temp_output.getvalue()

            # check to see if larger than 5MB
            if len(raw_data) >= max_size:
                # Reduce the image size
                for quality in range(95, 10, -5):
                    temp_output.seek(0)
                    temp_output.truncate(0)
                    im.save(temp_output, format=output_format, quality=quality)
                    reduced_data = temp_output.getvalue()
                    if len(reduced_data) <= max_size:
                        result = reduced_data
                        raw_data = result
                        break
                else:
                    # If the image is still too large, resize the image
                    while len(raw_data) > max_size:
                        im = im.resize((int(im.width * 0.9), int(im.height * 0.9)))
                        temp_output.seek(0)
                        temp_output.truncate(0)
                        im.save(temp_output, format=output_format)
                        result = temp_output.getvalue()
                        raw_data = result
        return result

    @staticmethod
    def classify_image(raw_data):
        if raw_data:
            if raw_data[:8] == b'\x89PNG\r\n\x1a\n': return 'image/png'
            elif raw_data[:2] == b'\xff\xd8': return 'image/jpeg'
            elif raw_data[:4] == b'RIFF' and raw_data[-4:] == b'WEBP': return 'image/webp'
        return 'image/unknown'

    @staticmethod
    def log_exception(logger, e, message=None):
        exc_traceback = e.__traceback__

        while exc_traceback.tb_next:
            exc_traceback = exc_traceback.tb_next
        frame = exc_traceback.tb_frame
        lineno = exc_traceback.tb_lineno
        filename = frame.f_code.co_filename

        log_message = traceback.format_exception(type(e), e, e.__traceback__)
        if message:
            log_message += f": {message}"

        logger.error(log_message)

    @staticmethod
    def glob_exclusions(pattern):
        if not pattern.startswith('!'):
            return []

        pattern = pattern.replace('!', '')
        # Find files matching exclusion patterns
        excluded_files = set()
        excluded_files.update(glob.glob(pattern, recursive=True))
        return excluded_files

    @staticmethod
    def is_glob_pattern(s):
        return any(char in s for char in "*?[]{}!")

    @staticmethod
    def is_glob_recursive(s):
        return '**' in s

    @staticmethod
    def glob_brace(pattern):
        parts = pattern.split('{')
        if len(parts) == 1:
            # No brace found, use glob directly
            return glob.glob(pattern)

        pre = parts[0]
        post = parts[1].split('}', 1)[1]
        options = parts[1].split('}', 1)[0].split(',')

        # Create individual patterns
        patterns = [pre + option + post for option in options]

        # Apply glob to each pattern and combine results
        files = set(itertools.chain.from_iterable(glob.glob(pat) for pat in patterns))
        return list(files)

    @staticmethod
    def late_bind(module_name, class_name, method_name, *args, **kwargs):
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            method = getattr(cls, method_name)

            if isinstance(method, staticmethod):
                return method.__func__(*args, **kwargs)
            else:
                instance = cls()
                return getattr(instance, method_name)(*args, **kwargs)
        except Exception as e:
            pass

    @staticmethod
    def is_running(process_name):
        for proc in psutil.process_iter():
            try:
                if process_name.lower() in proc.name().lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False

    @staticmethod
    def __find_terminal_emulator(process):
        try:
            if process.parent():
                # Check if the process name matches known terminal emulators
                name = process.parent().name()
                if 'Terminal' in name:
                    return 'Terminal'
                elif 'iTerm' in name:
                    return 'iTerm2'
                elif 'alacritty' in name:
                    return 'alacritty'
                elif 'kitty' in name:
                    return 'kitty'
                elif 'tmux' in name:
                    return 'tmux'
                # If no match, check the next parent
                return Helpers.__find_terminal_emulator(process.parent())
            else:
                # No more parents, terminal emulator not found
                return 'Unknown'
        except Exception as e:
            return str(e)

    @staticmethod
    def is_emulator(emulator: str):
        current_process = psutil.Process(os.getpid())
        result = Helpers.__find_terminal_emulator(current_process)
        return emulator == result

    @staticmethod
    def is_pdf(byte_stream):
        # PDF files start with "%PDF-" (hex: 25 50 44 46 2D)
        pdf_signature = b'%PDF-'
        # Read the first 5 bytes to check the signature
        first_bytes = byte_stream.read(5)
        # Reset the stream position to the beginning if possible
        if hasattr(byte_stream, 'seek'):
            byte_stream.seek(0)
        # Return True if the signature matches
        return first_bytes == pdf_signature

    @staticmethod
    def is_image(byte_stream):
        try:
            if isinstance(byte_stream, io.BytesIO):
                byte_stream = byte_stream.getvalue()

            with Image.open(io.BytesIO(byte_stream)) as im:
                return True
        except Exception:
            return False

    @staticmethod
    def decompress_if_compressed(byte_stream):
        try:
            # Attempt to decompress to see if it's zlib compressed
            decompressed_data = zlib.decompress(byte_stream)
            return True, decompressed_data
        except zlib.error as e:
            # If decompression fails, it's likely not zlib compressed or the data is corrupted/incomplete
            return False, byte_stream

    @staticmethod
    def image_size(byte_stream: bytes) -> tuple[int, int]:
        try:
            if isinstance(byte_stream, io.BytesIO):
                byte_stream = byte_stream.getvalue()

            with Image.open(io.BytesIO(byte_stream)) as im:
                return im.size
        except Exception:
            return (0, 0)

    @staticmethod
    def is_base64_encoded(s):
        import binascii

        try:
            if len(s) % 4 == 0:
                base64.b64decode(s, validate=True)
                return True
        except (ValueError, binascii.Error):
            return False
        return False

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def read_netscape_cookies(cookies_file_content: str) -> List[Dict[str, Any]]:
        cookies = []
        for line in cookies_file_content.splitlines():
            if not line.startswith('#') and line.strip():  # Ignore comments and empty lines
                try:
                    domain, _, path, secure, expires_value, name, value = line.strip().split('\t')

                    if not expires_value.isnumeric():
                        if expires_value == 'Session':
                            expires_value = 1999999999
                        else:
                            import time
                            expiration_datetime = dt.datetime.strptime(expires_value, '%Y-%m-%dT%H:%M:%S.%fZ')
                            expires_value = int(time.mktime(expiration_datetime.timetuple()))

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

    @staticmethod
    def get_callable(logging: Logger, method_str) -> Optional[Callable]:
        parts = method_str.split(".")
        class_name = ''

        if len(parts) == 1:
            # If it's a function in the current module
            module = importlib.import_module("__main__")
            method_name = parts[0]
        else:
            # If it's a method in some other module or class
            module_name = ".".join(parts[:-2]) if len(parts) > 2 else ".".join(parts[:-1])
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                logging.error(f"Module '{module_name}' not found")
                return None

            if len(parts) > 2:
                class_name = parts[-2]
                class_obj = getattr(module, class_name, None)
                if class_obj is None:
                    raise ValueError(f"Class '{class_name}' not found in module '{module_name}'")
                method_name = parts[-1]
                method = getattr(class_obj, method_name, None)
            else:
                method_name = parts[-1]
                method = getattr(module, method_name, None)

            if method is None:
                logging.error(f"Method '{method_name}' not found in {'class ' + class_name + ' in ' if len(parts) > 2 else ''}module '{module_name}'")  # noqa: E501
            return method

    @staticmethod
    def tfidf_similarity(query: str, text_list: list[str]) -> str:
        def tokenize(text: str) -> List[str]:
            # Simple tokenizer, can be enhanced
            return text.lower().split()

        def compute_tf(text_tokens: List[str]) -> dict:
            # Count the occurrences of each word in the text
            tf = Counter(text_tokens)
            # Divide by the total number of words for Term Frequency
            tf = {word: count / len(text_tokens) for word, count in tf.items()}
            return tf

        def compute_idf(documents: List[List[str]]) -> dict:
            # Number of documents
            N = len(documents)
            # Count the number of documents that contain each word
            idf = {}
            for document in documents:
                for word in set(document):
                    idf[word] = idf.get(word, 0) + 1
            # Calculate IDF
            idf = {word: math.log(N / df) for word, df in idf.items()}
            return idf

        def compute_tfidf(tf: dict, idf: dict) -> dict:
            # Multiply TF by IDF
            tfidf = {word: tf_value * idf.get(word, 0) for word, tf_value in tf.items()}
            return tfidf

        def cosine_similarity(vec1: dict, vec2: dict) -> float:
            # Compute the dot product
            dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in set(vec1.keys()) | set(vec2.keys()))
            # Compute the magnitudes
            mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
            mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
            # Compute cosine similarity
            if mag1 * mag2 == 0:
                return 0
            else:
                return dot_product / (mag1 * mag2)

        # Tokenize and prepare documents
        documents = [tokenize(text) for text in text_list]
        query_tokens = tokenize(query)
        documents.append(query_tokens)

        # Compute IDF for the whole corpus
        idf = compute_idf(documents)

        # Compute TF-IDF for query and documents
        query_tfidf = compute_tfidf(compute_tf(query_tokens), idf)
        documents_tfidf = [compute_tfidf(compute_tf(doc), idf) for doc in documents[:-1]]  # Exclude query

        # Compute similarity and find the most similar document
        similarities = [cosine_similarity(query_tfidf, doc_tfidf) for doc_tfidf in documents_tfidf]
        max_index = similarities.index(max(similarities))

        return text_list[max_index]

    @staticmethod
    def flatten(lst):
        def __has_list(lst):
            for item in lst:
                if isinstance(item, list):
                    return True
            return False

        def __inner_flatten(lst):
            return [item for sublist in lst for item in sublist]

        while __has_list(lst):
            lst = __inner_flatten(lst)
        return lst

    @staticmethod
    def extract_token(s, ident):
        if s.startswith(ident) and ' ' in s:
            return s[0:s.index(' ')]

        if ident in s:
            parts = s.split(ident)
            start = parts[0][parts[0].rfind(' ') + 1:] if ' ' in parts[0] else parts[0]
            end = parts[1][:parts[1].find(' ')] if ' ' in parts[1] else parts[1]
            return start + ident + end
        return ''

    @staticmethod
    def in_between(s, start, end):
        if end == '\n' and '\n' not in s:
            return s[s.find(start) + len(start):]

        after_start = s[s.find(start) + len(start):]
        part = after_start[:after_start.find(end)]
        return part

    @staticmethod
    def in_between_ends(s, start, end_strs: List[str]):
        # get the text from s between start and any of the end_strs strings.
        possibilities = []
        for end in end_strs:
            if end == '\n' and '\n' not in s:
                result = s[s.find(start) + len(start):]
                possibilities.append(result)
            elif end in s:
                after_start = s[s.find(start) + len(start):]
                part = after_start[:after_start.find(end)]
                if part:
                    possibilities.append(part)

        # return the shortest one
        return min(possibilities, key=len)

    @staticmethod
    def extract_code_blocks(markdown_text):
        # Pattern to match code blocks with or without specified language
        pattern = r'```(\w+\n)?(.*?)```'

        # Using re.DOTALL to make the '.' match also newlines
        matches = re.findall(pattern, markdown_text, re.DOTALL)

        # Extracting just the code part (ignoring the optional language part)
        code_blocks = [match[1].strip() for match in matches]
        return code_blocks

    @staticmethod
    def extract_context(s, start, end, stop_tokens=['\n', '.', '?', '!']):
        def capture(s, stop_tokens, backwards=False):
            if backwards:
                for i in range(len(s) - 1, -1, -1):
                    if s[i] in stop_tokens:
                        return s[i + 1:]
                return s
            else:
                for i in range(0, len(s)):
                    if s[i] in stop_tokens:
                        return s[:i]
                return s

        if end == '\n' and '\n' not in s:
            s += '\n'

        left_of_start = s.split(start)[0]
        right_of_end = s.split(end)[-1]
        return str(capture(left_of_start, stop_tokens, backwards=True)) + str(capture(right_of_end, stop_tokens))

    @staticmethod
    def strip_between(s: str, start: str, end: str):
        first = s[:s.find(start)]
        rest = s[s.find(start) + len(start):]
        return first + rest[rest.find(end) + len(end):]

    @staticmethod
    def split_between(s: str, start: str, end: str):
        first = s[:s.find(start)]
        rest = s[s.find(start) + len(start):]
        return (first, rest[rest.find(end) + len(end):])

    @staticmethod
    def first(predicate, iterable, default=None):
        try:
            result = next(x for x in iterable if predicate(x))
            return result
        except StopIteration as ex:
            return default

    @staticmethod
    def filter(predicate, iterable):
        return [x for x in iterable if predicate(x)]

    @staticmethod
    def last(predicate, iterable, default=None):
        result = [x for x in iterable if predicate(x)]
        if result:
            return result[-1]
        return default

    @staticmethod
    def resize_image(screenshot_data, base_width=500):
        # Load the image from the in-memory data
        image = Image.open(io.BytesIO(screenshot_data))

        # Calculate the height maintaining the aspect ratio
        w_percent = base_width / float(image.size[0])
        h_size = int(float(image.size[1]) * float(w_percent))

        # Resize the image
        image = image.resize((base_width, h_size), Image.NEAREST)  # type: ignore

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def find_string_between_tokens(text, start_token, end_token):
        start_index = text.rfind(start_token)
        if start_index == -1:
            return ''

        end_index = text.rfind(end_token, start_index)
        if end_index == -1:
            return ''

        result = text[start_index + len(start_token):end_index]
        return result.strip()

    @staticmethod
    def roundrobin(*iterables):
        num_active = len(iterables)
        nexts = cycle(iter(it).__next__ for it in iterables)
        while num_active:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                # Remove the iterator we just exhausted from the cycle.
                num_active -= 1
                nexts = cycle(islice(nexts, num_active))

    @staticmethod
    def iter_over_async(ait, loop):
        ait = ait.__aiter__()

        async def get_next():
            try:
                obj = await ait.__anext__()
                return False, obj
            except StopAsyncIteration:
                return True, None
        while True:
            done, obj = loop.run_until_complete(get_next())
            if done:
                break
            yield obj

    @staticmethod
    def split_text_into_chunks_eol_boundary_aligned(text: str, max_chunk_length: int = 400) -> List[str]:
        lines = text.splitlines()
        sentences: List[str] = []

        for line in lines:
            parts = line.split('.')
            parts_with_period = [bel + '.' for bel in parts if bel]
            sentences.extend(parts_with_period)

        combined: List[str] = []

        for sentence in sentences:
            if not combined:
                combined.append(sentence)
                continue

            prev = combined[-1]
            if len(prev) + len(sentence) < max_chunk_length:
                combined[-1] = f'{prev.strip()} {sentence.strip()}'
            else:
                combined.append(sentence)
        return combined

    @staticmethod
    def split_text_into_chunks(text: str, max_chunk_length: int = 400) -> List[str]:
        words = text.split()
        result = []
        for i in range(0, len(words), max_chunk_length):
            result.append(' '.join(words[i:i + max_chunk_length]))
        return result

    @staticmethod
    def find_closest_sections(query: str, sections: list[str]):
        raise NotImplementedError('This is not implemented yet')
        # from sentence_transformers import SentenceTransformer, util
        # from torch import Tensor

        # model = SentenceTransformer('all-mpnet-base-v2')
        # corpus_embeddings: List[Tensor] | Any = model.encode(sections, convert_to_tensor=True)
        # query_embedding: List[Tensor] | Any = model.encode([query], convert_to_tensor=True)

        # cosine_scores = util.cos_sim(corpus_embeddings, query_embedding)  # type: ignore

        # scored_sections = list(zip(sections, cosine_scores))
        # scored_sections = sorted(scored_sections, key=lambda x: x[1], reverse=True)  # type: ignore

        # scores = [{'text': text, 'score': score.cpu().item()} for text, score in scored_sections]
        # return scores

    @staticmethod
    def chunk_and_rank(query: str, data: str, max_chunk_length=400) -> List[str]:
        """Chunks the data into sections and ranks them based on the query"""
        sections = Helpers.split_text_into_chunks(data, max_chunk_length=max_chunk_length)
        if sections:
            results = Helpers.find_closest_sections(query, sections)
        else:
            return []
        return [a['text'] for a in results]

    @staticmethod
    def prompt_data_iterable(
        prompt: str,
        data: str,
        max_tokens=4000,
        prompt_at_end: bool = False,
    ) -> Generator[str, None, None]:
        """Ensures that prompt and data are under the max token length, repeats prompt and data if necessary"""
        prompt_words = prompt.split()
        sections = Helpers.split_text_into_chunks(data, max_chunk_length=max_tokens - len(prompt_words))
        for section in sections:
            if prompt_at_end:
                yield f'{section} {prompt}'
            else:
                yield f'{prompt} {section}'

    @staticmethod
    def calculate_prompt_cost(content: str, max_chunk_length=4000):
        words = content.split()

        # confirm with user
        est_tokens = len(words) / 0.75
        cost_per_token = 0.0002 / 1000
        est_cost = est_tokens * cost_per_token
        num_chunks = round(len(words) / max_chunk_length)
        est_time = est_tokens / 4000 * 1.5  # around 1.5 mins per 4000 tokens
        return {
            'est_tokens': est_tokens,
            'cost_per_token': cost_per_token,
            'est_cost': est_cost,
            'num_chunks': num_chunks,
            'est_time': est_time,
        }

    @staticmethod
    def messages_to_str(messages: List[Dict[str, str]]) -> str:
        words = []
        for m in messages:
            words.append([w.split() for w in m.values()])
        return ' '.join(Helpers.flatten(words))

    @staticmethod
    async def generator_for_new_tokens(program, *args, **kwargs):
        future = program(*args, **kwargs, silent=True, async_mode=True)
        starting_text = future.text
        while not future._execute_complete.is_set():
            await asyncio.sleep(0.2)
            snapshot = future.text
            yield snapshot[len(starting_text):]
            starting_text = snapshot
        yield future.text[len(starting_text):]

    @staticmethod
    def run_and_stream(program, *args, **kwargs):
        try:
            other_loop = asyncio.get_event_loop()
            nest_asyncio.apply(other_loop)
        except RuntimeError:
            pass
        loop = asyncio.new_event_loop()

        full_text = ""
        for new_text in Helpers.iter_over_async(Helpers.generator_for_new_tokens(program, *args, **kwargs), loop):
            if new_text:
                full_text += new_text
                yield new_text

    @staticmethod
    def strip_roles(text: str) -> str:
        text = text.replace('{{llm.default_system_prompt}}', '')
        result = text.replace('{{#system~}}', '') \
            .replace('{{~/system}}', '') \
            .replace('{{#user~}}', '') \
            .replace('{{~/user}}', '') \
            .replace('{{#assistant~}}', '') \
            .replace('{{~/assistant}}', '')
        return result

    @staticmethod
    def __get_class_of_func(func):
        if inspect.ismethod(func):
            for cls in inspect.getmro(func.__self__.__class__):
                if cls.__dict__.get(func.__name__) is func:
                    return cls
            func = func.__func__  # fallback to __qualname__ parsing
        if inspect.isfunction(func):
            cls = getattr(
                inspect.getmodule(func),
                func.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
            )
            if isinstance(cls, type):
                return cls
        return getattr(func, '__objclass__', None)  # handle special descriptor objects

    @staticmethod
    def get_function_description(func, openai_format: bool) -> Dict[str, Any]:
        def parse_type(t):
            if t is str:
                return 'string'
            elif t is int:
                return 'integer'
            elif t is IntEnum:
                return 'integer'
            elif t is Enum:
                return 'string'
            elif t is float:
                return 'number'
            else:
                return 'object'

        import inspect

        description = ''
        if func.__doc__ and parse(func.__doc__).short_description:
            description = parse(func.__doc__).short_description
        if func.__doc__ and parse(func.__doc__).long_description:
            description += ' ' + str(parse(func.__doc__).long_description).replace('\n', ' ')  # type: ignore

        func_name = func.__name__
        func_class = Helpers.__get_class_of_func(func)
        invoked_by = f'{func_class.__name__}.{func_name}' if func_class else func_name

        params = {}

        for p in inspect.signature(func).parameters:
            param = inspect.signature(func).parameters[p]
            parameter = {
                param.name: {
                    'type': parse_type(param.annotation) if param.annotation is not inspect._empty else 'object',
                    'description': '',
                }
            }

            if param.annotation and isinstance(param.annotation, type) and issubclass(param.annotation, Enum):
                values = [v.value for v in param.annotation.__members__.values()]
                parameter[param.name]['enum'] = values  # type: ignore

            params.update(parameter)

            # if it's got doc comments, use those instead
            for p in parse(func.__doc__).params:  # type: ignore
                params.update({
                    p.arg_name: {  # type: ignore
                        'type': parse_type(p.type_name) if p.type_name is not None else 'string',  # type: ignore
                        'description': p.description,  # type: ignore
                    }  # type: ignore
                })

        def required_params(func):
            parameters = inspect.signature(func).parameters
            return [
                name for name, param in parameters.items()
                if param.default == inspect.Parameter.empty and param.kind != param.VAR_KEYWORD
            ]

        function = {
            'name': invoked_by,
            'description': description,
            'parameters': {
                'type': 'object',
                'properties': params
            },
            'required': required_params(func),
        }

        if openai_format:
            return function
        else:
            return {
                'invoked_by': invoked_by,
                'description': description,
                'parameters': list(params.keys()),
                'types': [p['type'] for p in params.values()],
                'return_type': typing.get_type_hints(func).get('return')
            }

    @staticmethod
    def get_function_description_simple(function: Callable) -> str:
        description = Helpers.get_function_description(function, openai_format=False)
        return (f'{description["invoked_by"]}({", ".join(description["parameters"])})  # {description["description"] or "No docstring"}')

    @staticmethod
    def get_function_description_flat(function: Callable) -> str:
        description = Helpers.get_function_description(function, openai_format=False)
        parameter_type_list = [f"{param}: {typ}" for param, typ in zip(description['parameters'], description['types'])]
        return_type = description['return_type'].__name__ if description['return_type'] else 'Any'
        return (f'def {description["invoked_by"]}({", ".join(parameter_type_list)}) -> {return_type}  # {description["description"] or "No docstring"}')  # noqa: E501

    @staticmethod
    def load_resources_prompt(prompt_name: str, module: str = 'llmvm.server.prompts.starlark') -> Dict[str, Any]:
        prompt_file = resources.files(module) / prompt_name

        with open(prompt_file, 'r') as f:  # type: ignore
            prompt = f.read()

            if '[system_message]' not in prompt:
                raise ValueError('Prompt file must contain [system_message]')

            if '[user_message]' not in prompt:
                raise ValueError('Prompt file must contain [user_message]')

            system_message = Helpers.in_between(prompt, '[system_message]', '[user_message]').strip()
            user_message = prompt[prompt.find('[user_message]') + len('[user_message]'):].strip()
            templates = []

            temp_prompt = prompt
            while '{{' and '}}' in temp_prompt:
                templates.append(Helpers.in_between(temp_prompt, '{{', '}}'))
                temp_prompt = temp_prompt.split('}}', 1)[-1]

            return {
                'system_message': system_message,
                'user_message': user_message,
                'templates': templates
            }

    @staticmethod
    def get_prompts(
        prompt_text: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        append_token: str = '',
    ) -> Tuple[System, User]:
        if '[system_message]' not in prompt_text:
            raise ValueError('Prompt file must contain [system_message]')

        if '[user_message]' not in prompt_text:
            raise ValueError('Prompt file must contain [user_message]')

        system_message = Helpers.in_between(prompt_text, '[system_message]', '[user_message]').strip()
        user_message = prompt_text[prompt_text.find('[user_message]') + len('[user_message]'):].strip()
        templates = []

        temp_prompt = prompt_text
        while '{{' and '}}' in temp_prompt:
            templates.append(Helpers.in_between(temp_prompt, '{{', '}}'))
            temp_prompt = temp_prompt.split('}}', 1)[-1]

        prompt = {
            'system_message': system_message,
            'user_message': user_message,
            'templates': templates
        }

        if not template.get('user_token'):
            template['user_token'] = user_token
            template['user_colon_token'] = user_token + ':'
        if not template.get('assistant_token'):
            template['assistant_token'] = assistant_token
            template['assistant_colon_token'] = assistant_token + ':'

        for key, value in template.items():
            prompt['system_message'] = prompt['system_message'].replace('{{' + key + '}}', value)
            prompt['user_message'] = prompt['user_message'].replace('{{' + key + '}}', value)

        # deal with exec() statements to inject things like datetime
        import datetime
        for message_key in ['system_message', 'user_message']:
            message = prompt[message_key]
            while '{{' in message and '}}' in message:
                start = message.find('{{')
                end = message.find('}}', start)
                if end == -1:  # No closing '}}' found
                    break

                key = message[start+2:end]
                replacement = ''

                if key.startswith('exec('):
                    try:
                        replacement = str(eval(key[5:-1]))
                    except Exception as e:
                        pass
                else:
                    replacement = key

                message = message[:start] + replacement + message[end+2:]

            prompt[message_key] = message
        return (System(Content(prompt['system_message'])), User(Content(prompt['user_message'] + append_token)))

    @staticmethod
    def load_and_populate_prompt(
        prompt_name: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        append_token: str = '',
        module: str = 'llmvm.server.prompts.starlark'
    ) -> Dict[str, Any]:
        prompt: Dict[str, Any] = Helpers.load_resources_prompt(prompt_name, module)

        if not template.get('user_token'):
            template['user_token'] = user_token
            template['user_colon_token'] = user_token + ':'
        if not template.get('assistant_token'):
            template['assistant_token'] = assistant_token
            template['assistant_colon_token'] = assistant_token + ':'

        for key, value in template.items():
            prompt['system_message'] = prompt['system_message'].replace('{{' + key + '}}', value)
            prompt['user_message'] = prompt['user_message'].replace('{{' + key + '}}', value)

        # deal with exec() statements to inject things like datetime
        import datetime
        for message_key in ['system_message', 'user_message']:
            message = prompt[message_key]
            while '{{' in message and '}}' in message:
                start = message.find('{{')
                end = message.find('}}', start)
                if end == -1:  # No closing '}}' found
                    break

                key = message[start+2:end]
                replacement = ''

                if key.startswith('exec('):
                    try:
                        replacement = str(eval(key[5:-1]))
                    except Exception as e:
                        pass
                else:
                    replacement = key

                message = message[:start] + replacement + message[end+2:]

            prompt[message_key] = message

        prompt['user_message'] += f'{append_token}'
        prompt['prompt_name'] = prompt_name
        return prompt

    @staticmethod
    def populate_prompts(
        prompt_str: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        append_token: str = '',
        module: str = 'llmvm.server.prompts.starlark'
    ) -> Tuple[System, User]:
        prompt = Helpers.load_and_populate_prompt(prompt_str, template, user_token, assistant_token, append_token, module)
        return (prompt['system_message'], prompt['user_message'])

    @staticmethod
    def prompt_message(
        prompt_name: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        append_token: str = '',
        module: str = 'llmvm.server.prompts.starlark'
    ) -> Message:
        prompt = Helpers.load_and_populate_prompt(prompt_name, template, user_token, assistant_token, append_token, module)
        return User(Content(prompt['user_message']))

    @staticmethod
    def prompts(
        prompt_name: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        append_token: str = '',
        module: str = 'llmvm.server.prompts.starlark'
    ) -> Tuple[System, User]:
        prompt = Helpers.load_and_populate_prompt(prompt_name, template, user_token, assistant_token, append_token, module)
        return (System(Content(prompt['system_message'])), User(Content(prompt['user_message'])))
