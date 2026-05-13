"""SQL safety guardrail.

The agent generates SQL, but we never trust it blindly. This module enforces
an allow-list: only SELECT statements against expected tables are permitted.
Any other statement (DROP, DELETE, UPDATE, INSERT) is rejected before it
touches the database.

Block-list approaches are brittle — attackers find creative phrasings. An
allow-list says "exactly these patterns are allowed, everything else is no."
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import DML, Keyword


# Only these tables are queryable. Add more here as the schema grows.
ALLOWED_TABLES = {"courses"}


@dataclass(frozen=True)
class SqlCheckResult:
    """Outcome of safety validation."""
    is_safe: bool
    reason: str = ""


def check_sql(sql: str) -> SqlCheckResult:
    """Validate a SQL string against the safety policy.

    Returns SqlCheckResult with is_safe=True only when:
      1. The string parses as exactly one statement
      2. The statement is a SELECT (no DML mutations, no DDL)
      3. Every table reference is in the allow-list
      4. No semicolons mid-query (no stacked statements)
    """
    if not sql or not sql.strip():
        return SqlCheckResult(False, "Empty SQL")

    # Reject stacked statements via semicolons
    # (trailing semicolons are fine; we strip then check)
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        return SqlCheckResult(False, "Multiple statements not allowed")

    parsed = sqlparse.parse(stripped)
    if len(parsed) != 1:
        return SqlCheckResult(False, f"Expected 1 statement, got {len(parsed)}")

    stmt: Statement = parsed[0]

    # Statement type must be SELECT
    stmt_type = stmt.get_type()
    if stmt_type != "SELECT":
        return SqlCheckResult(False, f"Only SELECT allowed, got {stmt_type}")

    # Walk tokens to find any forbidden DML/DDL keywords
    forbidden = {"INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
                 "ALTER", "CREATE", "GRANT", "REVOKE"}
    for token in stmt.flatten():
        if token.ttype in (DML, Keyword) and token.value.upper() in forbidden:
            return SqlCheckResult(False, f"Forbidden keyword: {token.value.upper()}")

    # Verify every table reference is in the allow-list
    tables = _extract_table_names(stmt)
    unknown = tables - ALLOWED_TABLES
    if unknown:
        return SqlCheckResult(False, f"Unknown table(s): {sorted(unknown)}")

    return SqlCheckResult(True)


def _extract_table_names(stmt: Statement) -> set[str]:
    """Pull table identifiers out of a parsed SELECT statement.

    sqlparse doesn't have a clean AST for this, so we walk tokens looking
    for identifiers that follow FROM or JOIN keywords.
    """
    tables: set[str] = set()
    tokens = list(stmt.flatten())
    expecting_table = False

    for token in tokens:
        if token.ttype is Keyword and token.value.upper() in ("FROM", "JOIN"):
            expecting_table = True
            continue

        if expecting_table:
            # Skip whitespace and punctuation
            if token.is_whitespace or token.ttype in (None,):
                if token.value.strip() and not token.is_whitespace:
                    # An identifier — extract base name (handle schema.table)
                    name = token.value.strip().lower().split(".")[-1]
                    # Strip quotes/aliases
                    name = name.split()[0].strip('"').strip("'")
                    if name:
                        tables.add(name)
                    expecting_table = False
                continue

            # Direct identifier match
            name = token.value.strip().lower().split(".")[-1]
            name = name.split()[0].strip('"').strip("'")
            if name and not token.is_whitespace:
                tables.add(name)
                expecting_table = False

    return tables


if __name__ == "__main__":
    # Quick self-test
    tests = [
        ("SELECT * FROM courses", True),
        ("SELECT course_code FROM courses WHERE department = 'COMP'", True),
        ("DROP TABLE courses", False),
        ("DELETE FROM courses", False),
        ("SELECT * FROM courses; DROP TABLE courses;", False),
        ("UPDATE courses SET title = 'hacked'", False),
        ("SELECT * FROM secret_table", False),
        ("", False),
        ("not even sql", False),
    ]
    for sql, expected_safe in tests:
        result = check_sql(sql)
        marker = "✓" if result.is_safe == expected_safe else "✗ FAIL"
        print(f"  {marker} safe={result.is_safe}  ({result.reason or 'ok'})  -- {sql[:60]}")