import tempfile
from io import BytesIO
from typing import List
from urllib.parse import urlparse

import pdf2image
import pytesseract
import requests
from pypdf import PdfReader
from pytesseract import Output

from helpers.logging_helpers import setup_logging

logging = setup_logging()


class PdfHelpers():
    @staticmethod
    def parse_pdf_image(url_or_file: str) -> str:
        result = urlparse(url_or_file)

        text: List[str] = []
        images = pdf2image.convert_from_path(result.path)  # type: ignore
        for pil_im in images:
            ocr_dict = pytesseract.image_to_data(pil_im, output_type=Output.DICT)
            text.append(' '.join(ocr_dict['text']))

        return '\n'.join(text)

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
        url_result = urlparse(url_or_file)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'  # type: ignore
        }
        stream = None
        text = ''

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
                text += ' ' + page

        text = text.strip()

        if len(text) == 0:
            stream.seek(0)
            texts = []
            # try tesselation of pdf
            images = pdf2image.convert_from_bytes(stream.read())  # type: ignore
            # images = pdf2image.convert(url_result.path)  # type: ignore
            for pil_im in images:
                ocr_dict = pytesseract.image_to_data(pil_im, output_type=Output.DICT)
                texts.append(' '.join(ocr_dict['text']))

            text = ' '.join(texts)

        return text
