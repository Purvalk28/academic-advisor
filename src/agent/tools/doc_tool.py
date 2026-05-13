"""Document retrieval tool.

Wraps src.rag.retrieve so the agent treats RAG retrieval and SQL query as
siblings with the same interface shape. Also logs every invocation to
query_log for evaluation, matching the sql_tool pattern.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from src.db.connection import get_connection
from src.rag.retrieve import RetrievedChunk, retrieve


@dataclass(frozen=True)
class DocToolResult:
    """Outcome of a document retrieval call."""
    question: str
    success: bool
    chunks: list[RetrievedChunk]
    top_similarity: float
    is_confident: bool
    latency_ms: int = 0
    error: str = ""


def run_doc_tool(question: str, top_k: int = 3) -> DocToolResult:
    """Retrieve top-k chunks for a question."""
    start = time.monotonic()
    try:
        chunks = retrieve(question, top_k=top_k)
    except Exception as e:
        latency = int((time.monotonic() - start) * 1000)
        _log_query(question, success=False, error=str(e), latency_ms=latency)
        return DocToolResult(
            question=question,
            success=False,
            chunks=[],
            top_similarity=0.0,
            is_confident=False,
            latency_ms=latency,
            error=str(e),
        )

    latency = int((time.monotonic() - start) * 1000)
    top_sim = chunks[0].similarity if chunks else 0.0
    confident = bool(chunks) and chunks[0].is_confident
    _log_query(question, success=True, confidence=top_sim, latency_ms=latency)

    return DocToolResult(
        question=question,
        success=True,
        chunks=chunks,
        top_similarity=top_sim,
        is_confident=confident,
        latency_ms=latency,
    )


def _log_query(
    question: str,
    *,
    success: bool,
    confidence: float = 0.0,
    error: str = "",
    latency_ms: int = 0,
) -> None:
    """Log retrieval to query_log. Failures here are non-fatal."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_log
                        (user_query, tool_used, confidence, success, error_message, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (question, "doc_search", confidence, success, error or None, latency_ms),
                )
            conn.commit()
    except Exception:
        pass


def format_result(result: DocToolResult) -> str:
    """Pretty-print a DocToolResult for CLI inspection."""
    lines = [
        f"Question: {result.question}",
        f"Latency: {result.latency_ms}ms",
        f"Top similarity: {result.top_similarity:.3f} ({'confident' if result.is_confident else 'low confidence'})",
    ]
    if not result.success:
        lines.append(f"FAILED: {result.error}")
        return "\n".join(lines)

    for i, chunk in enumerate(result.chunks, 1):
        lines.append(f"\n[{i}] {chunk.source} (chunk {chunk.chunk_index}, sim={chunk.similarity:.3f})")
        lines.append("    " + chunk.content[:180].replace("\n", " "))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.agent.tools.doc_tool 'your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = run_doc_tool(question)
    print(format_result(result))