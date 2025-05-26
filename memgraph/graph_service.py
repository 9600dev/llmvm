from typing import Optional, Sequence
import openai
from graphdocs import GraphDocs
from llmvm.common.objects import Content, ContentContent, ImageContent, HTMLContent, TextContent, FileContent


class GraphService:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        model: str = "text-embedding-3-small",
        dims: int = 1536,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        def embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
            embeddings = []
            for text in texts:
                text = text.replace("\n", " ")
                embeddings.append(openai.embeddings.create(input = [text], model=model).data[0].embedding)
            return embeddings

        self.gd = GraphDocs(embedder=embed, dims=dims)

    def upsert(
        self,
        content: ContentContent,
        doc_id: Optional[str] = None,
        parse: bool = False
    ) -> None:
        if isinstance(content, )


