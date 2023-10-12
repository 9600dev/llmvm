import ast
import datetime as dt
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import spacy
from dateutil.parser import parse
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter, TokenTextSplitter

from helpers.logging_helpers import setup_logging
from helpers.pdf import PdfHelpers
from helpers.webhelpers import WebHelpers
from objects import Message
from vector_store import VectorStore

logging = setup_logging()


class EntityMetadata():
    def __init__(
        self,
    ):
        self.title: str = ''
        self.url: str = ''
        self.ingest_datetime: str = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.type: str = ''
        self.names: List[str] = []
        self.locations: List[str] = []
        self.organizations: List[str] = []
        self.dates: List[dt.datetime] = []
        self.events: List[str] = []
        self.parent: str = ''
        self.extra: Dict[str, Any] = {}

    def to_dict(self):
        d = {
            'title': self.title,
            'url': self.url,
            'ingest_datetime': self.ingest_datetime,
            'type': self.type,
            'names': self.names,
            'locations': self.locations,
            'organizations': self.organizations,
            'dates': self.dates,
            'events': self.events,
            'parent': self.parent,
        }
        for k, v in self.extra.items():
            if k not in d:
                d[k] = v

class VectorSearch():
    def __init__(
        self,
        vector_store: VectorStore,
    ):
        self.vector_store = vector_store
        self.nlp = spacy.load('en_core_web_sm')

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

    def parse_metadata(
        self,
        content: str,
        title: Optional[str] = None,
        url: Optional[str] = None,
        type: Optional[str] = None,
        ingest_datetime: Optional[str] = None,
        parent: Optional[str] = None,
        extra_metdata: Optional[dict] = None,
    ) -> EntityMetadata:
        doc = self.nlp(content)
        e = EntityMetadata()

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                e.names.append(ent.text)
            elif ent.label_ == "EVENT":
                e.events.append(ent.text)
            elif ent.label_ == "DATE":
                try:
                    e.dates.append(parse(ent.text))
                except Exception as ex:
                    logging.debug(ex)
            elif ent.label_ == "GPE" or ent.label_ == "LOC":
                e.locations.append(ent.text)
            elif ent.label_ == "ORG":
                e.organizations.append(ent.text)

        if title:
            e.title = title
        if url:
            e.url = url
        if type:
            e.type = type
        if ingest_datetime:
            e.ingest_datetime = ingest_datetime
        if parent:
            e.parent = parent
        if extra_metdata:
            e.extra = extra_metdata
        return e

    def ingest_messages(
        self,
        messages: List[Message],
        title: str,
        url: str,
        metadata: dict
    ) -> None:
        for m in messages:
            logging.debug('ingesting message: {}'.format(str(m.message)[0:25]))
            entity = self.parse_metadata(
                content=str(m.message),
                title=title,
                url=url,
                type='message',
                ingest_datetime=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                parent='',
                extra_metdata=metadata
            )
            self.vector_store.ingest_text(str(m.message), metadata)

    def ingest_text(
        self,
        text: str,
        title: str,
        url: str,
        metadata: dict
    ) -> None:
        entity = self.parse_metadata(
            content=text,
            title=title,
            url=url,
            type='text',
            ingest_datetime=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            parent='',
            extra_metdata=metadata
        )
        self.vector_store.ingest_text(text, entity.to_dict())

    def ingest_file(
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

        logging.debug('ingesting file: {}'.format(filename))

        if filename.endswith('.pdf'):
            text = PdfHelpers.parse_pdf(filename)
            entity = self.parse_metadata(
                content=text,
                title='',
                url=url,
                type='pdf',
                ingest_datetime=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                parent='',
                extra_metdata=metadata
            )
            self.vector_store.ingest_text(text, entity.to_dict())
        elif filename.endswith('.csv'):
            columns = []
            try:
                columns = list(pd.read_csv(filename).columns)
            except Exception as ex:
                logging.error(ex)
                pass
            metadata.update({'columns': columns})
            content = str(pd.read_csv(filename))
            entity = self.parse_metadata(
                content=content,
                title='',
                url=url,
                type='csv',
                ingest_datetime=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                parent='',
                extra_metdata=metadata
            )
            self.vector_store.ingest_text(content, entity.to_dict())
        elif filename.endswith('.txt'):
            with open(filename, 'r') as f:
                text = f.read()
                entity = self.parse_metadata(
                    content=text,
                    title='',
                    url=url,
                    type='txt',
                    ingest_datetime=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    parent='',
                    extra_metdata=metadata
                )
                self.vector_store.ingest_text(text, entity.to_dict())
        elif filename.endswith('.html') or filename.endswith('.htm'):
            with open(filename, 'r') as f:
                html = f.read()
                text = WebHelpers.convert_html_to_markdown(html)
                entity = self.parse_metadata(
                    content=text,
                    title='',
                    url=url,
                    type='html',
                    ingest_datetime=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    parent='',
                    extra_metdata=metadata,
                )
                self.vector_store.ingest_text(text, entity.to_dict())
        elif filename.endswith('.py'):
            with open(filename, 'r') as f:
                code = f.read()
                metadata.update({'classes': []})
                metadata.update({'methods': []})
                metadata.update({'docstrings': []})

                classes = extract_classes_methods_and_docstrings(code)
                for _class, value in classes.items():
                    metadata['classes'].append(_class)
                    for _method, value in value['methods'].items():
                        metadata['methods'].append(_method)
                        metadata['docstrings'].append(value)
                entity = self.parse_metadata(
                    content=code,
                    title=filename,
                    url=url,
                    type='python',
                    ingest_datetime=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    parent='',
                    extra_metdata=metadata,
                )
                self.vector_store.ingest_text(code, entity.to_dict())
        else:
            logging.debug('file not supported for ingestion: {}'.format(filename))
