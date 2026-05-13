"""Text-to-SQL tool.

Given a natural-language question about courses, prompts Claude with the
schema, parses the generated SQL, validates it through sql_guard, executes
it, and returns results. Logs every invocation to query_log for evaluation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from src.agent.llm import MODEL_HAIKU, complete
from src.agent.sql_guard import check_sql
from src.db.connection import get_connection


# Schema description fed to the LLM. Keep this in sync with src/db/schema.sql.
# We include type info and brief column descriptions because the model can't
# read the database directly — this prompt IS its view of the schema.
SCHEMA_DESCRIPTION = """
Table: courses

Columns:
  id              SERIAL PRIMARY KEY
  course_code     TEXT NOT NULL UNIQUE     -- e.g. 'COMP 5900', 'SERG 5101'
  title           TEXT NOT NULL            -- e.g. 'Database Management Systems'
  department      TEXT NOT NULL            -- one of: 'COMP', 'SYSC', 'ELEC', 'SERG'
  level           INT NOT NULL             -- 5000-6999 (graduate-only)
  credits         NUMERIC(2,1) NOT NULL    -- typically 0.5; thesis = 5.0; co-op = 0
  description     TEXT                     -- prose course description
  prerequisites   TEXT                     -- free text, may be NULL
  has_prereqs     BOOLEAN                  -- GENERATED; true if prerequisites is set
  terms_offered   TEXT[] NOT NULL          -- e.g. {'fall'}, {'fall','winter'}
  instructor      TEXT                     -- may be NULL
  is_thesis       BOOLEAN                  -- true for thesis courses
  is_project      BOOLEAN                  -- true for project courses (SERG 5908)

Notes:
  - Use 'winter' = ANY(terms_offered) to filter by term
  - Use has_prereqs = false to find courses without prerequisites
  - department values are uppercase: 'COMP' not 'comp'
  - Course codes contain a space: 'COMP 5900' not 'COMP5900'
"""

SYSTEM_PROMPT = f"""You are a SQL generation assistant for a PostgreSQL database
containing a graduate course catalog.

Given a user's natural-language question, generate exactly one SELECT statement
that answers it. Output ONLY the SQL. No explanation, no markdown, no comments.

If the question cannot be answered with the available schema, output exactly:
CANNOT_ANSWER

Constraints:
  - Only SELECT statements. Never INSERT, UPDATE, DELETE, DROP, or any DDL.
  - Only the 'courses' table. No other tables exist.
  - Use case-sensitive department values: 'COMP', 'SYSC', 'ELEC', 'SERG'
  - Limit results to 20 rows unless the question requires aggregation.

Schema:
{SCHEMA_DESCRIPTION}
"""


@dataclass(frozen=True)
class SqlToolResult:
    """Outcome of running the Text-to-SQL tool."""
    question: str
    sql: str
    success: bool
    rows: list[dict[str, Any]]
    column_names: list[str]
    error: str = ""
    latency_ms: int = 0


def _extract_sql(raw_response: str) -> str:
    """Pull SQL out of the model's response.

    Handles cases where Claude returns SQL wrapped in markdown code fences
    despite being instructed not to.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    return text


def run_sql_tool(question: str) -> SqlToolResult:
    """Convert question to SQL, validate, execute, and return results."""
    start = time.monotonic()

    # 1. Ask Claude for SQL
    raw = complete(
        prompt=question,
        system=SYSTEM_PROMPT,
        model=MODEL_HAIKU,
        max_tokens=512,
        temperature=0.0,
    )
    sql = _extract_sql(raw)

    # 2. Handle the explicit refusal case
    if sql == "CANNOT_ANSWER":
        latency = int((time.monotonic() - start) * 1000)
        _log_query(question, sql, success=False, error="model refused", latency_ms=latency)
        return SqlToolResult(
            question=question,
            sql=sql,
            success=False,
            rows=[],
            column_names=[],
            error="The model could not answer this question with the available schema.",
            latency_ms=latency,
        )

    # 3. Validate
    check = check_sql(sql)
    if not check.is_safe:
        latency = int((time.monotonic() - start) * 1000)
        _log_query(question, sql, success=False, error=f"guardrail: {check.reason}", latency_ms=latency)
        return SqlToolResult(
            question=question,
            sql=sql,
            success=False,
            rows=[],
            column_names=[],
            error=f"SQL rejected by guardrail: {check.reason}",
            latency_ms=latency,
        )

    # 4. Execute
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                column_names = [desc[0] for desc in cur.description] if cur.description else []
                raw_rows = cur.fetchall()
                rows = [dict(zip(column_names, row)) for row in raw_rows]
    except Exception as e:
        latency = int((time.monotonic() - start) * 1000)
        _log_query(question, sql, success=False, error=f"db error: {e}", latency_ms=latency)
        return SqlToolResult(
            question=question,
            sql=sql,
            success=False,
            rows=[],
            column_names=[],
            error=f"Database error: {e}",
            latency_ms=latency,
        )

    latency = int((time.monotonic() - start) * 1000)
    _log_query(question, sql, success=True, latency_ms=latency)

    return SqlToolResult(
        question=question,
        sql=sql,
        success=True,
        rows=rows,
        column_names=column_names,
        latency_ms=latency,
    )


def _log_query(
    question: str,
    sql: str,
    *,
    success: bool,
    error: str = "",
    latency_ms: int = 0,
) -> None:
    """Insert a row into query_log. Failures here are non-fatal."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO query_log
                        (user_query, tool_used, generated_sql, success, error_message, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (question, "sql_query", sql, success, error or None, latency_ms),
                )
            conn.commit()
    except Exception:
        # Logging failure must never break the tool
        pass


def format_result(result: SqlToolResult) -> str:
    """Pretty-print a SqlToolResult for CLI inspection."""
    lines = [
        f"Question: {result.question}",
        f"Generated SQL: {result.sql}",
        f"Latency: {result.latency_ms}ms",
    ]
    if not result.success:
        lines.append(f"FAILED: {result.error}")
        return "\n".join(lines)

    lines.append(f"Returned {len(result.rows)} row(s)")
    if result.rows:
        lines.append("")
        # Print first 5 rows compactly
        for row in result.rows[:5]:
            preview = ", ".join(f"{k}={v}" for k, v in row.items() if k in result.column_names[:4])
            lines.append(f"  - {preview}")
        if len(result.rows) > 5:
            lines.append(f"  ... and {len(result.rows) - 5} more")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.agent.tools.sql_tool 'your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = run_sql_tool(question)
    print(format_result(result))