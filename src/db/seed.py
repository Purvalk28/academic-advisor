"""Seed the courses table from CSV.

Idempotent: TRUNCATEs and reloads on every run. Simple and reliable for
dev seed data — production would use a migration tool like Alembic.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.db.connection import get_connection

SEED_PATH = Path(__file__).parent.parent.parent / "data" / "seed" / "courses.csv"


def _parse_array(value: str) -> list[str]:
    """Parse PostgreSQL array literal '{fall,winter}' into a Python list."""
    return [v.strip() for v in value.strip("{}").split(",") if v.strip()]


def _to_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "t", "1", "yes")


def seed_courses() -> int:
    """Load courses from CSV. Returns number of rows inserted."""
    with SEED_PATH.open() as f:
        rows = list(csv.DictReader(f))

    insert_sql = """
        INSERT INTO courses (
            course_code, title, department, level, credits, description,
            prerequisites, terms_offered, instructor, is_thesis, is_project
        ) VALUES (
            %(course_code)s, %(title)s, %(department)s, %(level)s, %(credits)s,
            %(description)s, %(prerequisites)s, %(terms_offered)s,
            %(instructor)s, %(is_thesis)s, %(is_project)s
        )
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE courses RESTART IDENTITY CASCADE")
            for row in rows:
                row["level"] = int(row["level"])
                row["credits"] = float(row["credits"])
                row["terms_offered"] = _parse_array(row["terms_offered"])
                row["prerequisites"] = row["prerequisites"] or None
                row["instructor"] = row["instructor"] or None
                row["is_thesis"] = _to_bool(row["is_thesis"])
                row["is_project"] = _to_bool(row["is_project"])
                cur.execute(insert_sql, row)
        conn.commit()

    return len(rows)


if __name__ == "__main__":
    n = seed_courses()
    print(f"✓ Seeded {n} courses")
