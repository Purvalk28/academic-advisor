"""End-of-Day-1 sanity check. Verifies the DB is healthy and seeded correctly.

Run this any time you suspect something has drifted — after pulling new
changes, after restarting Docker, or before starting work for the day.
"""
from __future__ import annotations

from src.db.connection import get_connection


CHECKS: list[tuple[str, str]] = [
    ("pgvector extension installed",
     "SELECT 1 FROM pg_extension WHERE extname = 'vector'"),
    ("courses table populated",
     "SELECT COUNT(*) >= 25 FROM courses"),
    ("multiple departments represented",
     "SELECT COUNT(DISTINCT department) >= 4 FROM courses"),
    ("prereq generated column working",
     "SELECT COUNT(*) > 0 FROM courses WHERE has_prereqs = true"),
    ("array column working",
     "SELECT COUNT(*) > 0 FROM courses WHERE 'winter' = ANY(terms_offered)"),
    ("doc_chunks table exists",
     "SELECT 1 FROM information_schema.tables WHERE table_name = 'doc_chunks'"),
]


def run_checks() -> bool:
    """Run all checks and print results. Returns True if all passed."""
    all_passed = True
    with get_connection() as conn:
        with conn.cursor() as cur:
            for description, query in CHECKS:
                cur.execute(query)
                result = cur.fetchone()
                passed = bool(result and result[0])
                marker = "✓" if passed else "✗"
                print(f"  {marker} {description}")
                if not passed:
                    all_passed = False
    return all_passed


if __name__ == "__main__":
    print("Running Day 1 sanity checks:\n")
    if run_checks():
        print("\n✓ All checks passed — Day 1 complete.")
    else:
        print("\n✗ Some checks failed. Review the output above.")
        exit(1)
