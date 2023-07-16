import math
import os
import tempfile
from typing import List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (CharacterTextSplitter, Language,
                                     MarkdownTextSplitter,
                                     PythonCodeTextSplitter,
                                     RecursiveCharacterTextSplitter,
                                     SpacyTextSplitter, TextSplitter,
                                     TokenTextSplitter)
from langchain.vectorstores import FAISS

from helpers.helpers import Helpers
from helpers.logging_helpers import setup_logging

logging = setup_logging()


class VectorStore():
    def __init__(
        self,
        openai_key: str = os.environ.get('OPENAI_API_KEY'),  # type: ignore
        store_filename: str = 'faiss_index',
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.openai_key = openai_key
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=self.openai_key,
        )  # type: ignore

        self.store_filename: str = store_filename

        if not os.path.exists(self.store_filename):
            self.store: FAISS = FAISS.from_texts([''], self.embeddings)
            self.store.save_local(self.store_filename)

        self.store = FAISS.load_local(
            self.store_filename,
            self.embeddings
        ) if self.store_filename else FAISS.from_texts([''], self.embeddings)  # type: ignore
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text(self, text: str):
        documents = []

        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=True) as t:
            t.write(text)
            t.seek(0)
            text_loader = TextLoader(t.name)
            data = text_loader.load()
            documents = text_loader.load_and_split()

        text_splitter = SpacyTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_texts = text_splitter.split_documents(documents)
        self.store.add_documents(split_texts)
        self.store.save_local(self.store_filename)

    def search_document(self, query: str, max_results: int = 4) -> List[Document]:
        return self.store.similarity_search(query, k=max_results)

    def search(self, query: str, max_results: int = 4) -> List[str]:
        result = self.store.similarity_search(query, k=max_results)
        return [a.page_content for a in result]

    def search_with_similarity(self, query: str, max_results: int = 4) -> List[Document]:
        result = self.store.similarity_search(query, k=max_results)
        return result

    def chunk_and_rank(
        self,
        query: str,
        s: str,
        chunk_token_count: int = 256,
        chunk_overlap: int = 0,
        max_tokens: int = 8196,
        splitter: Optional[TextSplitter] = None,
    ) -> List[Tuple[str, float]]:
        def contains_token(s, tokens):
            return any(token in s for token in tokens)

        if splitter:
            text_splitter = splitter
        else:
            html_tokens = ['<html>', '<body>', '<div>', '<script>', '<style>']
            markdown_tokens = ['###', '* ', '](']
            if contains_token(s, html_tokens):
                text_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.HTML,
                    chunk_size=chunk_token_count,
                    chunk_overlap=chunk_overlap
                )
            elif contains_token(s, markdown_tokens):
                text_splitter = MarkdownTextSplitter(chunk_size=chunk_token_count, chunk_overlap=chunk_overlap)
            else:
                text_splitter = SpacyTextSplitter(chunk_size=chunk_token_count, chunk_overlap=chunk_overlap)

        logging.debug('VectorStore.chunk_and_rank splitting documents')

        split_texts = text_splitter.split_text(s)

        token_chunk_cost = Helpers.calculate_tokens(split_texts[0])

        logging.debug('VectorStore.chunk_and_rank document length: {} split_texts: {}'.format(len(s), len(split_texts)))
        chunk_faiss = FAISS.from_texts(split_texts, self.embeddings)

        chunk_k = math.floor(max_tokens / token_chunk_cost)
        result = chunk_faiss.similarity_search_with_relevance_scores(query, k=chunk_k * 5)

        total_tokens = Helpers.calculate_tokens(query)
        return_results = []

        def half_str(s):
            mid = len(s) // 2
            return s[:mid]

        for doc, rank in result:
            if total_tokens + Helpers.calculate_tokens(doc.page_content) < max_tokens:
                return_results.append((doc.page_content, rank))
                total_tokens += Helpers.calculate_tokens(doc.page_content)
            elif total_tokens + Helpers.calculate_tokens(half_str(doc.page_content)):
                return_results.append((half_str(doc.page_content)[0], rank))
                total_tokens += Helpers.calculate_tokens(half_str(doc.page_content))
            else:
                break

        return return_results

