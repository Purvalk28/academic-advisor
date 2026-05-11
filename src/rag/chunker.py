"""Document chunker for RAG ingestion.

Splits long markdown documents into smaller, overlapping pieces sized for
embedding. Overlap preserves context that would otherwise be lost at chunk
boundaries — e.g. a sentence split across two chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text with metadata for retrieval."""
    source: str          # filename or section identifier
    chunk_index: int     # position within the source document
    content: str         # the actual text


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Args:
        text: The full document text.
        source: Identifier for the source (typically the filename).
        chunk_size: Target chunk length in characters.
        overlap: Number of characters to repeat between adjacent chunks.

    Returns:
        Ordered list of Chunk objects.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_content = text[start:end].strip()

        # Skip empty trailing chunks
        if chunk_content:
            chunks.append(Chunk(
                source=source,
                chunk_index=chunk_index,
                content=chunk_content,
            ))
            chunk_index += 1

        if end == len(text):
            break
        start += step

    return chunks


def chunk_markdown_file(path: Path, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Read a markdown file and chunk its contents."""
    text = path.read_text(encoding="utf-8")
    return chunk_text(text, source=path.name, chunk_size=chunk_size, overlap=overlap)