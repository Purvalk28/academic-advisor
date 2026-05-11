"""Embedding generation using sentence-transformers.

Wraps the all-MiniLM-L6-v2 model with lazy loading and a clean interface.
This model produces 384-dimensional embeddings, matches our doc_chunks
vector column, runs entirely on CPU, and is fast enough for our corpus size.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once and cache it.

    First call downloads ~90 MB of model weights to ~/.cache/huggingface/.
    Subsequent calls are instant.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> list[float]:
    """Embed a single piece of text."""
    model = get_embedder()
    vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return vec.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many pieces of text in a single batch.

    Batching is significantly faster than calling embed_text in a loop.
    """
    model = get_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
    return [vec.tolist() for vec in vecs]