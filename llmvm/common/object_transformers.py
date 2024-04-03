import asyncio
from collections import OrderedDict
from typing import List, cast

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Content, Executor, ImageContent, LLMCall,
                                  MarkdownContent, Message, PdfContent,
                                  StreamNode, User)
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
    def transform_pdf_content(content: PdfContent, executor: Executor) -> List[Message]:
        if content.original_sequence is not None and isinstance(content.sequence, list):
            return [User(c) for c in content.sequence]

        if cache.get(content.url):
            return ObjectTransformers.transform_pdf_content(cast(PdfContent, cache.get(content.url)), executor)

        if not content.sequence and content.is_local():
            content.sequence = open(content.url, 'rb').read()

        if content.original_sequence is None and Container.get_config_variable('LLMVM_FULL_PROCESSING', default=False):
            # avoid circular import
            pdf = Pdf(executor=executor)
            result = pdf.get_pdf_content(content)
            content.original_sequence = content.sequence
            content.sequence = result
            cache.set(content.url, content)

            return [User(content) for content in result]
        else:
            result = PdfHelpers.parse_pdf_bytes(cast(bytes, content.sequence))
            return [User(Content(result))]

    @staticmethod
    def transform_markdown_content(content: MarkdownContent, executor: Executor) -> List[Message]:
        if content.original_sequence is not None and isinstance(content.sequence, list):
            return [User(content) for content in content.sequence]

        if cache.get(content.url):
            return ObjectTransformers.transform_markdown_content(cast(MarkdownContent, cache.get(content.url)), executor)

        if Container.get_config_variable('LLMVM_FULL_PROCESSING', default=False):
            result = asyncio.run(Helpers.markdown_content_to_messages(logging, content, 150, 150))
            content.original_sequence = content.sequence
            content.sequence = result
            return [User(content) for content in result]
        else:
            return [User(Content(content.get_str()))]

    @staticmethod
    def transform_str(content: Content, executor: Executor) -> str:
        if isinstance(content, PdfContent):
            result = ObjectTransformers.transform_pdf_content(cast(PdfContent, content), executor)
            '\n'.join([c.message.get_str() for c in result])
        if isinstance(content, MarkdownContent):
            result = ObjectTransformers.transform_markdown_content(cast(MarkdownContent, content), executor)
            '\n'.join([c.message.get_str() for c in result])
        return content.get_str()
