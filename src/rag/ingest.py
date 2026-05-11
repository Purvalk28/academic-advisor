"""Ingest documents into the doc_chunks table.

Pipeline: read markdown files → chunk → embed → bulk-insert into PostgreSQL
with pgvector. Idempotent: TRUNCATEs doc_chunks before inserting.
"""
from __future__ import annotations

from pathlib import Path

from src.db.connection import get_connection
from src.rag.chunker import Chunk, chunk_markdown_file
from src.rag.embedder import embed_batch


DOCS_DIR = Path(__file__).parent.parent.parent / "docs"


def collect_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Read every .md file in docs/ and produce chunks."""
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files found in {docs_dir}")

    all_chunks: list[Chunk] = []
    for md_file in md_files:
        chunks = chunk_markdown_file(md_file)
        all_chunks.extend(chunks)
        print(f"  {md_file.name}: {len(chunks)} chunks")

    return all_chunks


def ingest() -> int:
    """End-to-end: chunk all docs, embed them, insert into doc_chunks.

    Returns the number of chunks inserted.
    """
    print("Collecting chunks from docs/...")
    chunks = collect_chunks()
    print(f"Total chunks: {len(chunks)}\n")

    print("Embedding chunks...")
    embeddings = embed_batch([c.content for c in chunks])
    print(f"Generated {len(embeddings)} embeddings\n")

    print("Inserting into doc_chunks...")
    insert_sql = """
        INSERT INTO doc_chunks (source, chunk_index, content, embedding)
        VALUES (%s, %s, %s, %s)
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE doc_chunks RESTART IDENTITY")
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    insert_sql,
                    (chunk.source, chunk.chunk_index, chunk.content, embedding),
                )
        conn.commit()

    print(f"✓ Inserted {len(chunks)} chunks into doc_chunks")
    return len(chunks)


if __name__ == "__main__":
    ingest()