import ast
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter, TokenTextSplitter

from helpers.logging_helpers import setup_logging
from helpers.pdf import PdfHelpers
from helpers.webhelpers import WebHelpers
from objects import Message
from vector_store import VectorStore

logging = setup_logging()


class VectorSearch():
    def __init__(
        self,
        vector_store: VectorStore,
    ):
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        max_results: int = 4,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        documents = self.vector_store.search_document(query, max_results)
        search_results = [
            {
                'title': document.metadata['title'],
                'link': document.metadata['url'],
                'snippet': document.page_content,
                'score': document.metadata['score'],
                'metadata': document.metadata,
            }
            for document in documents
            if 'score' in document.metadata
            and document.metadata['score'] >= min_score
            and 'url' in document.metadata
            and 'title' in document.metadata
        ]

        # now we need to merge the results by link
        merged_results = {}
        for result in search_results:
            if result['link'] not in merged_results:
                merged_results[result['link']] = result

        return list(merged_results.values())

    def chunk(
        self,
        content: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        return self.vector_store.chunk(content, chunk_size, overlap)

    def chunk_and_rank(
        self,
        query: str,
        content: str,
        chunk_token_count: int = 256,
        chunk_overlap: int = 0,
        max_tokens: int = 8196,
        splitter: Optional[TextSplitter] = None,
    ) -> List[Tuple[str, float]]:
        return self.vector_store.chunk_and_rank(
            query,
            content,
            chunk_token_count,
            chunk_overlap,
            max_tokens,
            splitter
        )

    def injest_messages(
        self,
        messages: List[Message],
        title: str,
        url: str,
        metadata: dict
    ) -> None:
        metadata.update({'url': url})
        metadata.update({'title': title})
        for m in messages:
            logging.debug('injesting message: {}'.format(str(m.message)[0:25]))
            self.vector_store.ingest_text(str(m.message), url, metadata)

    def injest_text(
        self,
        text: str,
        title: str,
        url: str,
        metadata: dict
    ) -> None:
        metadata.update({'type': 'text'})
        metadata.update({'url': url})
        metadata.update({'title': title})
        self.vector_store.ingest_text(text, url, metadata)

    def injest_file(
        self,
        filename: str,
        url: str,
        metadata: dict
    ) -> None:
        def extract_classes_methods_and_docstrings(python_code):
            node = ast.parse(python_code)
            classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
            result = {}

            for cls in classes:
                class_name = cls.name
                result[class_name] = {
                    'docstring': ast.get_docstring(cls),
                    'methods': {}
                }

                methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
                for method in methods:
                    result[class_name]['methods'][method.name] = ast.get_docstring(method)
            return result

        logging.debug('injesting file: {}'.format(filename))
        if filename.endswith('.pdf'):
            text = PdfHelpers.parse_pdf(filename)
            metadata.update({'type': 'pdf'})
            metadata.update({'url': url})
            metadata.update({'sitle': text[0:100]})
            self.vector_store.ingest_text(text, url, metadata)
        elif filename.endswith('.csv'):
            columns = []
            try:
                columns = list(pd.read_csv(filename).columns)
            except Exception as ex:
                logging.error(ex)
                pass
            metadata.update({'type': 'csv'})
            metadata.update({'url': url})
            metadata.update({'columns': columns})
            metadata.update({'title': ','.join(columns)})
            self.vector_store.ingest_text(str(pd.read_csv(filename)), url, metadata)
        elif filename.endswith('.txt'):
            with open(filename, 'r') as f:
                text = f.read()
                metadata.update({'type': 'txt'})
                metadata.update({'url': url})
                metadata.update({'title': text[0:100]})
                self.vector_store.ingest_text(text, url, metadata)
        elif filename.endswith('.html') or filename.endswith('.htm'):
            with open(filename, 'r') as f:
                html = f.read()
                text = WebHelpers.convert_html_to_markdown(html)
                metadata.update({'type': 'html'})
                metadata.update({'url': url})
                metadata.update({'title': text[0:100]})
                self.vector_store.ingest_text(text, url, metadata)
        elif filename.endswith('.py'):
            with open(filename, 'r') as f:
                code = f.read()
                metadata.update({'type': 'python'})
                metadata.update({'title': filename})
                metadata.update({'url': url})
                metadata.update({'classes': []})
                metadata.update({'methods': []})
                metadata.update({'docstrings': []})

                classes = extract_classes_methods_and_docstrings(code)
                for _class, value in classes.items():
                    metadata['classes'].append(_class)
                    for _method, value in value['methods'].items():
                        metadata['methods'].append(_method)
                        metadata['docstrings'].append(value)
                self.vector_store.ingest_text(code, url, metadata)
        else:
            logging.debug('file not supported for injestion: {}'.format(filename))

