import asyncio
import io
import tempfile
from io import BytesIO
from typing import cast
from urllib.parse import urlparse

import pdf2image
import pdfplumber
import pytesseract
from pdfminer.high_level import extract_text_to_fp
from pdfminer.image import ImageWriter
from PIL import Image
from pytesseract import Output

from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Content, Executor, ImageContent, LLMCall,
                                  PdfContent, StreamNode, TextContent, User)

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
    def parse_pdf_bytes(stream: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix='.pdf') as temp:
            temp.write(stream)
            temp.seek(0)
            return PdfHelpers.parse_pdf(temp.name)

    @staticmethod
    def parse_pdf_image_to_text(url_or_file: str) -> str:
        result = urlparse(url_or_file)

        text_chunks: list[str] = []
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
        logging.debug('PdfHelpers.parse_pdf: extracting text from pdf simple')

        stream = BytesIO(asyncio.run(Helpers.download_bytes(url_or_file)))
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
            text_result = PdfHelpers.parse_pdf_image_to_text(url_or_file)

        if stream:
            write_client_stream(
                StreamNode(
                    PdfHelpers.__page_image(stream.getvalue()),
                    type='bytes',
                    metadata={'type': 'image/png', 'url': url_or_file}
                )
            )

        return text_result


class Pdf():
    def __init__(
        self,
        executor: Executor,
    ):
        self.executor = executor

    async def __check_rendering(self, text: str) -> bool:
        prompt = f"""
        I have some text content from a PDF file. Sometimes it extracts incorrectly
        and the words do not have spaces between them. Can you tell me if the text
        looks correctly formatted with spaces between the words? Answer "yes" or "no" only.
        Do not explain yourself. Just answer "yes" or "no".

        Content: {text}
        """
        logging.debug(f'Pdf.__check_rendering({text[:50]})')

        text = text[:1500]

        assistant = await self.executor.aexecute([User(TextContent(prompt))])
        return 'yes' in str(assistant.message).lower()

    def parse_pdf(self, byte_stream: bytes, url_or_file: str) -> list[Content]:
        """
        Downloads a pdf file from the url_or_file argument and returns the text.
        You can only use either a url or a path to a pdf file.

        Args:
            url_or_file (str): url or path to pdf file
        Returns:
            str: text from pdf
        """
        logging.debug(f'Pdf.parse_pdf: extracting text and images from pdf using executor: {self.executor.name()}')

        stream = BytesIO(byte_stream)
        if not stream:
            return []

        pdf = pdfplumber.open(stream)
        pages_count = len(pdf.pages)
        content: list[Content] = []

        if pages_count <= 0:
            # try the old way
            return [TextContent(PdfHelpers.parse_pdf_image_to_text(url_or_file))]

        # determine the the space tolerance
        first_page = pdf.pages[0]
        x_tolerance = 3

        while x_tolerance >= 1:
            text = first_page.extract_text(x_tolerance=x_tolerance)
            if asyncio.run(self.__check_rendering(text)):
                break
            else:
                x_tolerance -= 1

        if x_tolerance == 0:
            return [TextContent(PdfHelpers.parse_pdf_image_to_text(url_or_file))]

        original = first_page.to_image(resolution=150, antialias=True).original
        return_image = BytesIO()
        original.save(return_image, format='PNG')

        # parse each page
        for i in range(0, pages_count):
            page_content: list[Content] = []
            page = pdf.pages[i]
            text = page.extract_text(x_tolerance=x_tolerance)
            if text:
                page_content.append(TextContent(text))

            images = page.images
            for img in images:
                raw_data = img['stream'].get_rawdata()
                _, raw_data = Helpers.decompress_if_compressed(raw_data)
                if Helpers.is_image(raw_data):
                    img_stream = BytesIO(img['stream'].get_rawdata())
                    im = Image.open(img_stream)
                    buf = io.BytesIO()
                    im.save(buf, format='PNG')
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        im.save(temp_file.name, format='PNG', optimize=False, compression_level=0)
                        page_content.append(ImageContent(buf.getvalue(), url=temp_file.name))
                else:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        image_bbox = (img['x0'], page.height - img['y1'], img['x1'], page.height - img['y0'])
                        page.crop(image_bbox).to_image(resolution=150, antialias=True).save(temp_file.name, format='PNG', optimize=False, compression_level=0)
                        im = Image.open(temp_file.name)
                        buf = io.BytesIO()
                        im.save(buf, format='PNG')
                        page_content.append(ImageContent(buf.getvalue(), url=temp_file.name))

            content.extend(page_content)

        write_client_stream(
            StreamNode(
                return_image.getvalue(),
                type='bytes',
                metadata={'type': 'image/png', 'url': url_or_file}
            )
        )
        return content

    def get_pdf(self, url_or_file: str) -> list[Content]:
        bytes = asyncio.run(Helpers.download_bytes(url_or_file))
        if bytes:
            return self.parse_pdf(bytes, url_or_file)
        return []

    def get_pdf_content(self, content: PdfContent) -> list[Content]:
        if isinstance(content.sequence, bytes):
            return self.parse_pdf(cast(bytes, content.sequence), content.url)
        if content.url:
            return self.get_pdf(content.url)
        return []
