import math
import os
import tempfile
from typing import Callable, List, Optional, Tuple

import torch
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import TextSplitter, TokenTextSplitter

from helpers.logging_helpers import setup_logging

logging = setup_logging()

class VectorStore():
    def __init__(
        self,
        token_calculator: Callable[[str], int],
        store_directory: str,
        index_name: str,
        embedding_model: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.token_calculator = token_calculator
        self._embeddings = None
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store_directory: str = store_directory
        self.index_name: str = index_name

        if not os.path.exists(self.store_directory):
            os.makedirs(self.store_directory)

        if not os.path.exists(os.path.join(self.store_directory, self.index_name + '.faiss')):
            from langchain.vectorstores import FAISS
            self.store: FAISS = FAISS.from_texts([''], self.embeddings())
            self.store.save_local(folder_path=self.store_directory, index_name=self.index_name)

    def embeddings(self):
        if not self._embeddings:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cuda:0'} if torch.cuda.is_available() else {'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings

    def __metadata_str(self, document: Document):
        if document.metadata:
            return ', '.join([f'{str(k)}: {str(v)}' for k, v in document.metadata.items()])
        else:
            return ''

    def __document_str(self, document: Document):
        return f'{self.__metadata_str(document)} {document.page_content}'

    def __load_store(self):
        from langchain.vectorstores import FAISS
        if not hasattr(self, 'store') or not self.store:
            self.store = FAISS.load_local(
                folder_path=self.store_directory,
                embeddings=self.embeddings(),
                index_name=self.index_name
            )
        return self.store

    def ingest_documents(
        self,
        documents: List[Document],
    ):
        self.__load_store().add_documents(documents)
        self.__load_store().save_local(folder_path=self.store_directory, index_name=self.index_name)

    def ingest_text(self, text: str, metadata: Optional[dict] = None):
        documents = []

        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=True) as t:
            t.write(text)
            t.seek(0)
            text_loader = TextLoader(t.name)
            data = text_loader.load()
            documents = text_loader.load_and_split()

        if metadata:
            for d in documents:
                d.metadata = metadata

        text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        split_texts = text_splitter.split_documents(documents)
        self.__load_store().add_documents(split_texts)
        self.__load_store().save_local(folder_path=self.store_directory, index_name=self.index_name)

    def search_document(self, query: str, max_results: int = 4) -> List[Document]:
        documents = self.__load_store().similarity_search_with_relevance_scores(query, k=max_results)
        for doc, score in documents:
            doc.metadata['score'] = score
        return [doc for doc, _ in documents if doc.page_content]

    def search(self, query: str, max_results: int = 4) -> List[str]:
        result = self.__load_store().similarity_search(query, k=max_results)
        return [f'{self.__metadata_str(a)} {a.page_content}' for a in result if a.page_content]

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

        token_chunk_cost = self.token_calculator(split_texts[0])

        logging.debug('VectorStore.chunk_and_rank document length: {} split_texts: {}'.format(len(content), len(split_texts)))
        chunk_faiss = FAISS.from_texts(split_texts, self.embeddings())

        chunk_k = math.floor(max_tokens / token_chunk_cost)
        result = chunk_faiss.similarity_search_with_relevance_scores(query, k=chunk_k * 5)

        total_tokens = self.token_calculator(query)
        return_results = []

        def half_str(s):
            mid = len(s) // 2
            return s[:mid]

        for doc, rank in result:
            if total_tokens + self.token_calculator(doc.page_content) < max_tokens:
                return_results.append((self.__document_str(doc), rank))
                total_tokens += self.token_calculator(self.__document_str(doc))
            elif (
                half_str(doc.page_content)
                and total_tokens + self.token_calculator(half_str(doc.page_content)) < max_tokens
            ):
                return_results.append((half_str(self.__document_str(doc))[0], rank))
                total_tokens += self.token_calculator(half_str(self.__document_str(doc)))
            else:
                break
        return return_results
