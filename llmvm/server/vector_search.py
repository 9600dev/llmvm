import datetime as dt
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain.text_splitter import TextSplitter

from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import Message
from llmvm.common.pdf import PdfHelpers
from llmvm.server.base_library.source import Source
from llmvm.server.tools.webhelpers import WebHelpers
from llmvm.server.vector_store import VectorStore

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
        return d

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
        token_calculator: Callable[[str], int],
        chunk_token_count: int = 256,
        chunk_overlap: int = 0,
        max_tokens: int = 8196,
        splitter: Optional[TextSplitter] = None,
    ) -> List[Tuple[str, float]]:
        return self.vector_store.chunk_and_rank(
            query,
            content,
            token_calculator,
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
        e = EntityMetadata()

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

    def parse_python_file(
        self,
        filename: str,
        url: str,
        metadata: dict
    ) -> None:
        metadata.update({'classes': []})
        metadata.update({'methods': []})
        metadata.update({'docstrings': []})
        sourcer = Source(filename)

        classes = sourcer.get_classes()
        for _class in classes:
            metadata['classes'].append(_class)
            for _method in sourcer.get_methods(_class):
                metadata['methods'].append(_method.name)
                metadata['docstrings'].append(_method.docstring)

        entity = self.parse_metadata(
            content=sourcer.source_code,
            title=filename,
            url=url,
            type='python',
            ingest_datetime=dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            parent='',
            extra_metdata=metadata,
        )
        self.vector_store.ingest_text(sourcer.source_code, entity.to_dict())
        logging.debug('ingested python file: {}'.format(filename))

    def ingest_file(
        self,
        filename: str,
        project: str,
        url: str,
        metadata: dict
    ) -> None:
        import pandas as pd

        logging.debug(f'ingesting file: {filename} with url {url} into project: {project}')

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
            logging.debug('ingested pdf file: {}'.format(filename))
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
            logging.debug('ingested csv file: {}'.format(filename))
        elif filename.endswith('.txt') or filename.endswith('.md'):
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
                logging.debug('ingested text file: {}'.format(filename))
        elif filename.endswith('.html') or filename.endswith('.htm'):
            with open(filename, 'r') as f:
                html = f.read()
                text = WebHelpers.convert_html_to_markdown(html, url=url).get_str()
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
                logging.debug('ingested html file: {}'.format(filename))
        elif filename.endswith('.py'):
            self.parse_python_file(filename, url, metadata)
        else:
            logging.debug('file not supported for ingestion: {}'.format(filename))
