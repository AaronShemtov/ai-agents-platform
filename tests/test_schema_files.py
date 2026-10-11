"""The schema file and the migration have to say the same thing about thread_id.

`sql/schema.sql` is the final state, for a database built from scratch; the migration
is how an existing one got there. If they drift, a rebuilt database is not the live
one, and nothing else in CI reads either file. The DEFAULT 0 in particular is what lets
code from before the column keep inserting, so it is pinned here rather than left to a
comment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

MIGRATION = "sql/migrations/2026-10-11-conversations-thread-id.sql"
DROP_OLD_INDEX = "sql/migrations/2026-10-11-conversations-thread-id-drop-old-index.sql"
INDEX = "create index conversations_by_thread on conversations (agent, chat_id, thread_id, id)"
COLUMN = "alter table conversations add (thread_id number default 0 not null)"


def statements(text: str) -> list[str]:
    """The SQL statements of a file: comments dropped, whitespace normalised."""
    code = "\n".join(line for line in text.splitlines() if not line.strip().startswith("--"))
    return [" ".join(part.split()) for part in code.split(";") if part.strip()]


@pytest.fixture
def schema(repo_root: Path) -> str:
    return (repo_root / "sql" / "schema.sql").read_text(encoding="utf-8")


def test_the_schema_file_has_the_column_with_its_default(schema: str) -> None:
    assert re.search(r"thread_id\s+number default 0 not null", schema)


def test_the_schema_file_has_the_index_the_migration_creates(
    schema: str, repo_root: Path
) -> None:
    assert INDEX in statements(schema)
    assert INDEX in statements((repo_root / MIGRATION).read_text(encoding="utf-8"))


def test_the_migration_adds_the_same_column_the_schema_file_defines(repo_root: Path) -> None:
    assert COLUMN in statements((repo_root / MIGRATION).read_text(encoding="utf-8"))


def test_the_migration_holds_only_the_two_steps_that_come_before_the_deploy(
    repo_root: Path,
) -> None:
    """Pasted or posted whole, a script is split on its semicolons and every statement
    runs. The drop of the old index must not be in the file that runs first."""
    assert statements((repo_root / MIGRATION).read_text(encoding="utf-8")) == [COLUMN, INDEX]


def test_the_old_index_is_dropped_by_a_file_of_its_own(schema: str, repo_root: Path) -> None:
    assert statements((repo_root / DROP_OLD_INDEX).read_text(encoding="utf-8")) == [
        "drop index conversations_by_chat"
    ]
    # And the final state no longer has it.
    assert "conversations_by_chat" not in schema
