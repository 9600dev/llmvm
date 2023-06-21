from io import BytesIO
from urllib.parse import urlparse

import pdf2image
import pytesseract
import requests
from langchain.document_loaders import TextLoader
from pypdf import PdfReader
from pytesseract import Output, TesseractError

from helpers.logging_helpers import setup_logging

logging = setup_logging()


class PdfHelpers():
    @staticmethod
    def parse_pdf(url_or_file: str) -> str:
        """Parse a pdf file and return the text

        Args:
            url_or_file (str): url or path to pdf file
        Returns:
            str: text from pdf
        """
        url_result = urlparse(url_or_file)
        stream = None
        text = ''

        if url_result.scheme == 'http' or url_result.scheme == 'https':
            stream = BytesIO(requests.get(url_or_file).content)
        else:
            with open(url_or_file, 'rb') as file:
                stream = BytesIO(file.read())

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

