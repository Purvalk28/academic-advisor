"""Anthropic Claude client wrapper.

Centralizes API calls so we have one place to:
  - Read the API key from environment
  - Choose models (Haiku for tools, Sonnet for orchestration)
  - Apply consistent timeouts and error handling
  - Add observability hooks later (LangSmith etc.)
"""
from __future__ import annotations

import os
from functools import lru_cache

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


# Model selection — Haiku for fast structured outputs (SQL generation),
# Sonnet for reasoning-heavy orchestration (router, synthesizer).
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"


@lru_cache(maxsize=1)
def get_client() -> Anthropic:
    """Return a singleton Anthropic client.

    Reads ANTHROPIC_API_KEY from environment. Fails loudly if missing.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to your .env file."
        )
    return Anthropic(api_key=key)


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str = MODEL_HAIKU,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Send a single-turn message to Claude and return the text response.

    Args:
        prompt: The user message.
        system: Optional system prompt to constrain behavior.
        model: Which Claude model to use.
        max_tokens: Cap on response length.
        temperature: 0.0 for deterministic structured output (SQL generation),
                     higher for creative tasks.

    Returns:
        The model's text response, stripped of leading/trailing whitespace.
    """
    client = get_client()

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)

    # response.content is a list of content blocks; we expect text-only
    text_parts = [block.text for block in response.content if hasattr(block, "text")]
    return "".join(text_parts).strip()


if __name__ == "__main__":
    # Smoke test: does the API key work?
    result = complete(
        "Reply with exactly the words 'connection works' and nothing else.",
        max_tokens=20,
    )
    print(f"Claude says: {result}")