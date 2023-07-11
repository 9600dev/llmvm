import os
import tempfile
from typing import List, Optional

import faiss
import openai
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (CharacterTextSplitter,
                                     MarkdownTextSplitter,
                                     PythonCodeTextSplitter, TokenTextSplitter)
from langchain.vectorstores import FAISS


class VectorStore():
    def __init__(
        self,
        openai_key: str = os.environ.get('OPENAI_API_KEY'),  # type: ignore
        store_filename: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.openai_key = openai_key
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=self.openai_key,
        )  # type: ignore

        self.store_filename: Optional[str] = store_filename

        if self.store_filename and not os.path.exists(self.store_filename):
            self.store = FAISS.from_texts([''], self.embeddings)
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

        text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_texts = text_splitter.split_documents(documents)

        self.store.merge_from(FAISS.from_documents(split_texts, self.embeddings))
        if self.store_filename:
            self.store.save_local(self.store_filename)

    def search_document(self, query: str, max_results: int = 4) -> List[Document]:
        return self.store.similarity_search(query, k=max_results)

    def search(self, query: str, max_results: int = 4) -> List[str]:
        result = self.store.similarity_search(query, k=max_results)
        return [a.page_content for a in result]
