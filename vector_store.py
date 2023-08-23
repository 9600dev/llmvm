import math
import os
import tempfile
from typing import List, Optional, Tuple

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (Language, MarkdownTextSplitter,
                                     RecursiveCharacterTextSplitter,
                                     SpacyTextSplitter, TextSplitter,
                                     TokenTextSplitter)

from helpers.logging_helpers import setup_logging
from objects import Executor
from openai_executor import OpenAIExecutor

logging = setup_logging()


class VectorStore():
    def __init__(
        self,
        executor: Executor = OpenAIExecutor(),
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

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store_filename: str = store_filename
        self.executor = executor

        if not os.path.exists(self.store_filename):
            from langchain.vectorstores import FAISS

            self.store: FAISS = FAISS.from_texts([''], self.embeddings)
            self.store.save_local(self.store_filename)

    def load_store(self):
        from langchain.vectorstores import FAISS

        if not self.store:
            self.store = FAISS.load_local(
                self.store_filename,
                self.embeddings
            )

        return self.store

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
        self.load_store().add_documents(split_texts)
        self.load_store().save_local(self.store_filename)

    def search_document(self, query: str, max_results: int = 4) -> List[Document]:
        return self.load_store().similarity_search(query, k=max_results)

    def search(self, query: str, max_results: int = 4) -> List[str]:
        result = self.load_store().similarity_search(query, k=max_results)
        return [a.page_content for a in result]

    def search_with_similarity(self, query: str, max_results: int = 4) -> List[Document]:
        result = self.load_store().similarity_search(query, k=max_results)
        return result

    def chunk(
        self,
        content: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        _chunk_size = chunk_size if chunk_size else self.chunk_size
        _overlap = overlap if overlap else self.chunk_overlap
        text_splitter = TokenTextSplitter(chunk_size=_chunk_size, chunk_overlap=_overlap)
        return text_splitter.split_text(content)

    def chunk_and_rank(
        self,
        query: str,
        content: str,
        chunk_token_count: int = 256,
        chunk_overlap: int = 0,
        max_tokens: int = 8196,
        splitter: Optional[TextSplitter] = None,
    ) -> List[Tuple[str, float]]:
        from langchain.vectorstores import FAISS

        if not content:
            return []

        def contains_token(s, tokens):
            return any(token in s for token in tokens)

        if splitter:
            text_splitter = splitter
        else:
            text_splitter = TokenTextSplitter(chunk_size=chunk_token_count, chunk_overlap=chunk_overlap)

        logging.debug('VectorStore.chunk_and_rank splitting documents')

        split_texts = text_splitter.split_text(content)

        token_chunk_cost = self.executor.calculate_tokens(split_texts[0])

        logging.debug('VectorStore.chunk_and_rank document length: {} split_texts: {}'.format(len(content), len(split_texts)))
        chunk_faiss = FAISS.from_texts(split_texts, self.embeddings)

        chunk_k = math.floor(max_tokens / token_chunk_cost)
        result = chunk_faiss.similarity_search_with_relevance_scores(query, k=chunk_k * 5)

        total_tokens = self.executor.calculate_tokens(query)
        return_results = []

        def half_str(s):
            mid = len(s) // 2
            return s[:mid]

        for doc, rank in result:
            if total_tokens + self.executor.calculate_tokens(doc.page_content) < max_tokens:
                return_results.append((doc.page_content, rank))
                total_tokens += self.executor.calculate_tokens(doc.page_content)
            elif (
                half_str(doc.page_content)
                and total_tokens + self.executor.calculate_tokens(half_str(doc.page_content)) < max_tokens
            ):
                return_results.append((half_str(doc.page_content)[0], rank))
                total_tokens += self.executor.calculate_tokens(half_str(doc.page_content))
            else:
                break
        return return_results
