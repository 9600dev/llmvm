import asyncio
from collections import OrderedDict
import os
import re
from typing import List, cast

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (BrowserContent, Content, Executor, FileContent, HTMLContent, ImageContent, LLMCall,
                                  MarkdownContent, Message, PdfContent,
                                  StreamNode, SupportedMessageContent, TextContent, User)
from llmvm.common.pdf import Pdf, PdfHelpers

logging = setup_logging()


class ObjectCache:
    _instance = None
    _cache = OrderedDict()
    _max_size = 100

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ObjectCache, cls).__new__(cls)
        return cls._instance

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(False)

    def clear(self):
        self._cache.clear()


# there's a case where you pass in a -p argument, and the code that is generated
# is download(file_url), and you end up re-transforming the object again.
cache = ObjectCache()

class ObjectTransformers():
    @staticmethod
    def transform_inline_markdown_to_image_content_list(content: Content) -> list[Content]:
        def split_markdown_images(text: str) -> list[SupportedMessageContent]:
            result: list[SupportedMessageContent] = []

            while text:
                # Pattern to match markdown images: ![alt_text](url)
                pattern = r'!\[(.*?)\]\((.*?)\)'
                match = re.search(pattern, text)

                if not match:
                    if not result:
                        return [TextContent(text)]

                    # No more images found, add remaining text as final result
                    if text:
                        result.append(TextContent(text))
                    break

                # Get the full matched string and its components
                full_match = match.group(0)
                alt_text = match.group(1)
                url = match.group(2)

                # Split text into before and after
                start, end = match.span()
                before = text[:start]
                after = text[end:]

                before += full_match + '\n'

                result.append(TextContent(before))
                if os.path.exists(url):
                    with open(url, 'rb') as f:
                        image_data = f.read()
                        if Helpers.is_image(image_data):
                            result.append(ImageContent(image_data, url=url))
                        else:
                            result.append(TextContent(f'Image at {url} is not an image.'))
                elif url.startswith('http'):
                    bytes_result = asyncio.run(Helpers.download_bytes(url, throw=False))
                    if bytes_result and Helpers.is_image(bytes_result):
                        result.append(ImageContent(bytes_result, url=url))
                    else:
                        result.append(TextContent(f'Image at {url} is not an image.'))

                text = after
            return result

        # todo: hack
        if isinstance(content, TextContent) and not content.get_str().startswith('BrowserContent('):
            return cast(list[Content], split_markdown_images(content.get_str()))

        return [content]

    @staticmethod
    def transform_pdf_to_content(content: PdfContent, executor: Executor, xml_wrapper: bool = False) -> list[SupportedMessageContent]:
        if content.original_sequence is not None and isinstance(content.sequence, list):
            return content.sequence

        if cache.get(content.url):
            return ObjectTransformers.transform_pdf_to_content(cast(PdfContent, cache.get(content.url)), executor, xml_wrapper)

        if not content.sequence and content.is_local():
            content.sequence = open(content.url, 'rb').read()

        if content.original_sequence is None and Container.get_config_variable('LLMVM_FULL_PROCESSING', default=False):
            # avoid circular import
            pdf = Pdf(executor=executor)
            result: list[content] = pdf.get_pdf_content(content)
            content.original_sequence = content.sequence
            content.sequence = result
            cache.set(content.url, content)
            if xml_wrapper:
                return [TextContent(f'<pdf url={content.url}>')] + result + [TextContent('</pdf>')]
            else:
                return result
        else:
            text_result: str = PdfHelpers.parse_pdf_bytes(cast(bytes, content.sequence))
            if xml_wrapper:
                return cast(list[SupportedMessageContent], [TextContent(f'<pdf url={content.url}>')] + [TextContent(text_result)] + [TextContent('</pdf>')])
            else:
                return [TextContent(text_result)]

    @staticmethod
    def transform_markdown_to_content(content: MarkdownContent, executor: Executor, xml_wrapper: bool = False) -> list[SupportedMessageContent]:
        if content.original_sequence is not None and isinstance(content.sequence, list) and all([isinstance(c, SupportedMessageContent) for c in content.sequence]):
            return cast(list[SupportedMessageContent], content.sequence)

        if cache.get(content.url):
            return ObjectTransformers.transform_markdown_to_content(cast(MarkdownContent, cache.get(content.url)), executor, xml_wrapper)

        if Container.get_config_variable('LLMVM_FULL_PROCESSING', default=False):
            result: list[SupportedMessageContent] = asyncio.run(Helpers.markdown_content_to_supported_content(logging, content, 150, 150))
            content.original_sequence = content.sequence
            content.sequence = cast(list[Content], result)
            if xml_wrapper:
                return [TextContent(f'<markdown url={content.url}>')] + result + [TextContent('</markdown>')]
            else:
                return result
        else:
            # turns out, if there are embedded images in the markdown, we should probably strip them out
            # because otherwise, we're just uploading these things for no reason
            markdown_content = Helpers.remove_embedded_images(content.get_str())
            if xml_wrapper:
                return cast(list[SupportedMessageContent], [TextContent(f'<markdown url={content.url}>')] + [TextContent(markdown_content)] + [TextContent('</markdown>')])
            else:
                return [TextContent(markdown_content)]

    @staticmethod
    def transform_html_to_content(content: HTMLContent, executor: Executor, xml_wrapper: bool = False) -> list[SupportedMessageContent]:
        if content.original_sequence is not None and isinstance(content.sequence, list) and all([isinstance(c, SupportedMessageContent) for c in content.sequence]):
            return cast(list[SupportedMessageContent], content.sequence)

        if cache.get(content.url):
            return ObjectTransformers.transform_html_to_content(cast(HTMLContent, cache.get(content.url)), executor, xml_wrapper)

        html_content = content.get_str()
        if xml_wrapper:
            return cast(list[SupportedMessageContent], [TextContent(f'<html_content url={content.url}>')] + [TextContent(html_content)] + [TextContent('</html_content>')])
        else:
            return [TextContent(html_content)]

    @staticmethod
    def transform_browser_to_content(content: BrowserContent, executor: Executor, xml_wrapper: bool = False) -> list[SupportedMessageContent]:
        if content.original_sequence is not None and isinstance(content.sequence, list) and all([isinstance(c, SupportedMessageContent) for c in content.sequence]):
            return cast(list[SupportedMessageContent], content.sequence)

        if cache.get(content.url):
            return ObjectTransformers.transform_browser_to_content(cast(BrowserContent, cache.get(content.url)), executor)

        if Container.get_config_variable('LLMVM_FULL_PROCESSING', default=False):
            result: list[SupportedMessageContent] = []
            content.original_sequence = content.sequence
            supported_contents = Helpers.flatten([ObjectTransformers.transform_to_supported_content(c, executor, xml_wrapper) for c in content.sequence])
            content.sequence = cast(list[Content], supported_contents)
            return supported_contents
        else:
            non_image_content = [content for content in content.sequence if not isinstance(content, ImageContent)]
            return Helpers.flatten([ObjectTransformers.transform_to_supported_content(c, executor, xml_wrapper) for c in non_image_content])

    @staticmethod
    def transform_file_to_content(content: FileContent, executor: Executor, xml_wrapper: bool = False) -> list[SupportedMessageContent]:
        if xml_wrapper:
            return cast(list[SupportedMessageContent], [TextContent(f'<file url={content.url}>')] + [TextContent(content.get_str())] + [TextContent('</file>')])
        else:
            return [TextContent(content.get_str())]

    @staticmethod
    def transform_content_to_string(content: list[Content], executor: Executor, xml_wrapper: bool = False) -> str:
        result = Helpers.flatten([ObjectTransformers.transform_to_supported_content(c, executor, xml_wrapper) for c in content])
        return '\n'.join([c.get_str() for c in result])

    @staticmethod
    def transform_to_supported_content(content: Content, executor: Executor, xml_wrapper: bool = False) -> list[SupportedMessageContent]:
        if isinstance(content, PdfContent):
            return ObjectTransformers.transform_pdf_to_content(cast(PdfContent, content), executor, xml_wrapper)
        elif isinstance(content, MarkdownContent):
            return ObjectTransformers.transform_markdown_to_content(cast(MarkdownContent, content), executor, xml_wrapper)
        elif isinstance(content, BrowserContent):
            return ObjectTransformers.transform_browser_to_content(cast(BrowserContent, content), executor)
        elif isinstance(content, FileContent):
            return ObjectTransformers.transform_file_to_content(cast(FileContent, content), executor, xml_wrapper)
        elif isinstance(content, ImageContent):
            return [content]
        elif isinstance(content, HTMLContent):
            return ObjectTransformers.transform_html_to_content(cast(HTMLContent, content), executor, xml_wrapper)
        elif isinstance(content, TextContent):
            return [content]
        else:
            raise ValueError(f"Unknown content type: {type(content)}")