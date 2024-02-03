import tempfile
from cgitb import text
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

import pdf2image
import pytesseract
import requests
from pdfminer.high_level import extract_text_to_fp
from pypdf import PdfReader
from pytesseract import Output

from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import StreamNode

logging = setup_logging()


class PdfHelpers():
    @staticmethod
    def __page_image(pdf_bytes: bytes):
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
        if len(images) > 0:
            byte_stream = BytesIO()
            images[0].save(byte_stream, format='PNG')
            return Helpers.resize_image(byte_stream.getvalue())

    @staticmethod
    def __get_pdf(url_or_file: str) -> Optional[BytesIO]:
        url_result = urlparse(url_or_file)
        headers = {  # type: ignore
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  # type: ignore
        }
        stream = None

        if url_result.scheme == 'http' or url_result.scheme == 'https':
            content = requests.get(url_or_file, headers=headers, allow_redirects=True, timeout=10).content
            stream = BytesIO(content)
        else:
            try:
                with open(url_or_file, 'rb') as file:
                    stream = BytesIO(file.read())
            except FileNotFoundError:
                raise ValueError('The supplied argument url_or_file: {} is not a correct filename or url.'.format(url_or_file))
        return stream

    @staticmethod
    def parse_pdf_bytes(stream: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp:
            temp.write(stream)
            temp.seek(0)
            return PdfHelpers.parse_pdf(temp.name)

    @staticmethod
    def parse_pdf_image(url_or_file: str) -> str:
        result = urlparse(url_or_file)

        text_chunks: List[str] = []
        images = pdf2image.convert_from_path(result.path)  # type: ignore
        for pil_im in images:
            ocr_dict = pytesseract.image_to_data(pil_im, output_type=Output.DICT)
            text_chunks.append(' '.join(ocr_dict['text']))

        return '\n'.join(text_chunks)

    @staticmethod
    def parse_pdf(url_or_file: str) -> str:
        """
        Downloads a pdf file from the url_or_file argument and returns the text.
        You can only use either a url or a path to a pdf file.

        Args:
            url_or_file (str): url or path to pdf file
        Returns:
            str: text from pdf
        """
        logging.debug('parse_pdf: extracting text from pdf')

        stream = PdfHelpers.__get_pdf(url_or_file)
        if not stream:
            return ''

        escape_chars = ['\x0c', '\x0b', '\x0a']
        text_stream = BytesIO()

        # text_result = extract_text(stream)

        extract_text_to_fp(stream, text_stream, output_type='text')
        text_stream.seek(0)
        text_result = text_stream.read().decode('utf-8')

        for char in escape_chars:
            text_result = text_result.replace(char, ' ').strip()

        if text_result == '':
            text_result = PdfHelpers.parse_pdf_image(url_or_file)

        if stream:
            write_client_stream(
                StreamNode(
                    PdfHelpers.__page_image(stream.getvalue()),
                    type='bytes',
                    metadata={'type': 'image/png', 'url': url_or_file}
                )
            )

        return text_result

    @staticmethod
    def parse_pdf_deprecated(url_or_file: str) -> str:
        """
        Downloads a pdf file from the url_or_file argument and returns the text.
        You can only use either a url or a path to a pdf file.

        Args:
            url_or_file (str): url or path to pdf file
        Returns:
            str: text from pdf
        """
        url_result = urlparse(url_or_file)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  # type: ignore
        }
        stream = None
        text_result = ''

        if url_result.scheme == 'http' or url_result.scheme == 'https':
            content = requests.get(url_or_file, headers=headers, allow_redirects=True, timeout=10).content
            stream = BytesIO(content)
        else:
            try:
                with open(url_or_file, 'rb') as file:
                    stream = BytesIO(file.read())
            except FileNotFoundError:
                raise ValueError('The supplied argument url_or_file: {} is not a correct filename or url.'.format(url_or_file))

        reader = PdfReader(stream)

        logging.debug('get_url: extracting text from pdf')
        for i in range(0, len(reader.pages)):
            page = reader.pages[i].extract_text()
            if page:
                text_result += ' ' + page

        text_result = text_result.strip()

        if len(text_result) == 0:
            stream.seek(0)
            texts = []
            # try tesselation of pdf
            images = pdf2image.convert_from_bytes(stream.read())  # type: ignore
            # images = pdf2image.convert(url_result.path)  # type: ignore
            for pil_im in images:
                ocr_dict = pytesseract.image_to_data(pil_im, output_type=Output.DICT)
                texts.append(' '.join(ocr_dict['text']))

            text_result = ' '.join(texts)

        return text_result
