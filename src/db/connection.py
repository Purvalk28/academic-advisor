"""Database connection utilities.

Single source of truth for how the app connects to PostgreSQL. All other
modules import from here rather than reading env vars directly — makes
testing and configuration changes trivial.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def get_database_url() -> str:
    """Read DATABASE_URL from environment, fail loudly if missing."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Copy .env.example to .env and configure it."
        )
    return url


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Yield a psycopg connection with the pgvector adapter registered.

    Use as a context manager so connections are always closed,
    even on exceptions:

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    conn = psycopg.connect(get_database_url())
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
