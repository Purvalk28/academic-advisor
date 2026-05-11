"""Vector similarity retrieval over doc_chunks.

Embeds the query, runs cosine-similarity search via pgvector, returns ranked
results with similarity scores. Includes a confidence threshold for the agent
to decide whether to trust the retrieval or flag the answer as uncertain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.db.connection import get_connection
from src.rag.embedder import embed_text


CONFIDENCE_THRESHOLD = 0.5  # below this, retrieval is unreliable for our corpus


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from retrieval, with similarity score."""
    source: str
    chunk_index: int
    content: str
    similarity: float

    @property
    def is_confident(self) -> bool:
        """True if similarity is above the trust threshold."""
        return self.similarity >= CONFIDENCE_THRESHOLD


def retrieve(query: str, top_k: int = 3) -> list[RetrievedChunk]:
    """Find the top-k most similar chunks to the query.

    Uses cosine distance via pgvector's <=> operator. Returns chunks ordered by
    decreasing similarity (most relevant first). Similarity is computed as
    1 - cosine_distance, so 1.0 = identical, 0.0 = orthogonal.
    """
    if not query.strip():
        return []

    query_embedding = embed_text(query)

    search_sql = """
        SELECT source, chunk_index, content,
               1 - (embedding <=> %s::vector) AS similarity
        FROM doc_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(search_sql, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            source=row[0],
            chunk_index=row[1],
            content=row[2],
            similarity=float(row[3]),
        )
        for row in rows
    ]


def format_results(results: Iterable[RetrievedChunk]) -> str:
    """Pretty-print retrieval results for CLI inspection."""
    lines = []
    for i, chunk in enumerate(results, 1):
        confidence_marker = "✓" if chunk.is_confident else "?"
        lines.append(
            f"\n[{i}] {confidence_marker} similarity={chunk.similarity:.3f}  "
            f"({chunk.source}, chunk {chunk.chunk_index})"
        )
        lines.append("    " + chunk.content[:200].replace("\n", " "))
        if len(chunk.content) > 200:
            lines.append("    ...")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.rag.retrieve 'your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Query: {question}")
    results = retrieve(question, top_k=3)

    if not results:
        print("No results found.")
    else:
        print(format_results(results))