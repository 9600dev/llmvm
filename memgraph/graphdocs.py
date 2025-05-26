import uuid
import json
from typing import Callable, Sequence, List, Dict, Any, Optional

from gqlalchemy import Memgraph
from markdown_it import MarkdownIt

_CHUNK_HEADERS = ["h1", "h2", "h3"]


class GraphDocs:
    """Store Markdown documents + embeddings in Memgraph with vector search.

    Parameters
    ----------
    embedder : Callable[[Sequence[str]], Sequence[Sequence[float]]]
        Function that converts a list of strings to a list/array of vectors.
    dims : int
        Dimensionality of the vectors your embedder returns.
    host, port, user, password : Memgraph connection details.
    """
    def __init__(
        self,
        *,
        embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]],
        dims: int,
        host: str = "localhost",
        port: int = 7687,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        if embedder is None:
            raise ValueError("You must supply an embedding function.")
        self.embedder = embedder
        self.dims = dims
        self.mg = Memgraph(host=host, port=port, username=user, password=password)
        self._ensure_vector_index()

    def add_markdown(
        self,
        md_text: str,
        *,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        doc_id = doc_id or f"doc-{uuid.uuid4()}"
        chunks = self._split_markdown(md_text)
        embeddings = list(self.embedder(chunks))

        with self.mg.transaction() as tx:
            tx.execute(
                """
                MERGE (d:Document {id:$doc_id})
                SET d.raw = $raw, d.meta = $meta
                """,
                {"doc_id": doc_id, "raw": md_text, "meta": json.dumps(metadata or {})},
            )
            for idx, (chunk_txt, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}::chunk{idx}"
                tx.execute(
                    """
                    MERGE (c:Chunk {id:$cid})
                    SET c.text = $txt, c.embedding = $emb
                    WITH c
                    MATCH (d:Document {id:$doc_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    {"cid": chunk_id, "txt": chunk_txt, "emb": list(map(float, emb)), "doc_id": doc_id},
                )
        return doc_id

    def update_markdown(self, doc_id: str, new_md: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.delete(doc_id)
        self.add_markdown(new_md, doc_id=doc_id, metadata=metadata)

    def search(self, query: str, *, k: int = 5) -> List[Dict[str, Any]]:
        qvec = list(self.embedder([query])[0])
        rows = self.mg.execute(
            """
            CALL vector_search.search('Chunk','embedding',$vec,$k,'COSINE')
            YIELD id, distance
            MATCH (c:Chunk) WHERE id(c)=id
            MATCH (d:Document)-[:HAS_CHUNK]->(c)
            RETURN d.id AS doc_id, c.text AS chunk, distance
            ORDER BY distance ASC
            """,
            {"vec": qvec, "k": k},
        )
        return [{"doc_id": r["doc_id"], "score": r["distance"], "chunk": r["chunk"]} for r in rows]

    def delete(self, doc_id: str) -> None:
        self.mg.execute(
            """MATCH (d:Document {id:$doc_id})-[:HAS_CHUNK]->(c) DETACH DELETE d, c""",
            {"doc_id": doc_id},
        )

    def _ensure_vector_index(self) -> None:
        self.mg.execute(
            """
            CALL vector_search.show_index_info() YIELD label, property
            WITH collect(label + property) AS info
            WHERE NOT 'Chunkembedding' IN info
            CALL {
                CREATE VECTOR INDEX ON :Chunk(embedding) OPTIONS {distance:'COSINE', dims:$dims}
            } IN TRANSACTIONS OF 1 ROWS
            """,
            {"dims": self.dims},
        )

    def _split_markdown(self, md_text: str) -> List[str]:
        md = MarkdownIt()
        tokens = md.parse(md_text)
        chunks: List[str] = []
        current: List[str] = []
        for tok in tokens:
            if tok.tag in _CHUNK_HEADERS and tok.type == "heading_open":
                if current:
                    chunks.append("\n".join(current).strip())
                    current = []
            if tok.type == "inline":
                current.append(tok.content)
        if current:
            chunks.append("\n".join(current).strip())
        return [c for c in chunks if c]
