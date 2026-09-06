"""Durable memory: conversation history and facts about the user.

Why this exists
---------------
`memory.py` keeps history in a dict, so a rollout of a single-replica Deployment
with `strategy: Recreate` — which is every deploy — loses every conversation.
The complaint that produced this module was the practical one: having to explain
the same context again after each restart.

How it reaches the database
---------------------------
Oracle Autonomous Database, through ORDS' REST-Enabled SQL endpoint: plain HTTPS
with Basic Auth, no Oracle client library, no wallet, no ACL change. Verified
against the live instance before this file was written, which is worth knowing
because none of it is obvious:

  * binds require a ``data_type``. Without it ORDS answers 400 "Problem
    recognizing JSON" and gives no hint that the shape is the problem.
  * a bare INSERT does not commit. Sending ``commit`` as a second request would
    leave a window where the row exists uncommitted and the process dies, so
    writes go as one PL/SQL block that commits inside itself.
  * a hostile string survives intact — quotes, semicolons, newlines, Cyrillic,
    and a ``drop table`` in the payload, which stays a payload.

Which schema, and why not ADMIN
-------------------------------
Its own schema, ``AGENTS``, with no grants on anything else. The same database
holds the URL shortener's data, and an agent with the ADMIN password could
delete it by mistake. Confirmed by trying: ``select from admin.urls`` as this
user is ORA-00942, table does not exist.

No vectors yet
--------------
Deliberately. Facts number in the dozens and go into the prompt whole, so there
is nothing to search; history will want search eventually, and the database
supports it (``VECTOR``, 26ai, checked), but choosing where embeddings come from
is a separate decision worth making when there is enough history to search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx2

log = logging.getLogger(__name__)

# Oracle's VARCHAR2 bind limit in this context. Longer values go as CLOB, which
# is why the choice is made per value rather than per column.
_VARCHAR_MAX = 4000

# A single message can be enormous — a file the agent read, a tool result — and
# the transcript is trimmed by the caller anyway. Storing a 2 MB row helps
# nobody and makes the next read slow.
MAX_CONTENT_CHARS = 200_000


class StoreUnavailable(RuntimeError):
    """The database could not be reached or refused the statement."""


def _bind(name: str, value: Any) -> dict[str, Any]:
    """One bind in the shape ORDS insists on.

    `data_type` is not optional: omitting it fails the whole request with a
    parse error that says nothing about binds.
    """
    if value is None:
        return {"name": name, "data_type": "VARCHAR2", "value": None}
    if isinstance(value, bool):
        # Before the int branch: bool is an int in Python, and Oracle has no
        # boolean bind type here.
        return {"name": name, "data_type": "NUMBER", "value": int(value)}
    if isinstance(value, int | float):
        return {"name": name, "data_type": "NUMBER", "value": value}
    text = str(value)
    kind = "CLOB" if len(text) > _VARCHAR_MAX else "VARCHAR2"
    return {"name": name, "data_type": kind, "value": text}


@dataclass(frozen=True)
class Fact:
    key: str
    fact: str
    scope: str
    updated_at: str


class AdbStore:
    """Conversation history and facts, in Oracle Autonomous Database."""

    def __init__(
        self,
        *,
        base_url: str,
        username: str,
        password: str,
        timeout_seconds: float = 10.0,
    ) -> None:
        # `/ords/agents/_/sql` — the schema's own REST-Enabled SQL endpoint.
        self._url = base_url.rstrip("/")
        self._auth = (username, password)
        self._timeout = timeout_seconds

    # -- transport ----------------------------------------------------------

    async def _run(self, statement: str, binds: dict[str, Any] | None = None) -> list[dict]:
        """Execute one statement and return its rows (empty for writes)."""
        payload: dict[str, Any] = {"statementText": statement}
        if binds:
            payload["binds"] = [_bind(k, v) for k, v in binds.items()]

        try:
            async with httpx2.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload, auth=self._auth)
        except httpx2.HTTPError as exc:
            raise StoreUnavailable(f"cannot reach the database: {exc}") from exc

        if response.status_code != 200:
            # ORDS puts the useful part in the body; the status alone is 400 for
            # everything from a typo to a bad bind shape.
            raise StoreUnavailable(
                f"ORDS returned HTTP {response.status_code}: {response.text[:300]}"
            )

        items = response.json().get("items") or []
        rows: list[dict] = []
        for item in items:
            if item.get("errorMessage"):
                raise StoreUnavailable(f"SQL error: {item['errorMessage'][:300]}")
            result_set = item.get("resultSet")
            if result_set:
                rows.extend(result_set.get("items") or [])
        return rows

    async def ping(self) -> bool:
        try:
            await self._run("select 1 as ok from dual")
            return True
        except StoreUnavailable as exc:
            log.warning("memory store unreachable: %s", exc)
            return False

    # -- conversation history ----------------------------------------------

    async def append(self, *, agent: str, chat_id: int, message: dict[str, Any]) -> None:
        """Store one message. Best effort: a failure must not lose the answer
        the user is waiting for, so it is logged rather than raised."""
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if content and len(content) > MAX_CONTENT_CHARS:
            kept = MAX_CONTENT_CHARS
            content = content[:kept] + f"\n… [{len(content) - kept} characters not stored]"

        tool_calls = message.get("tool_calls")
        statement = (
            "begin insert into conversations "
            "(agent, chat_id, role, content, tool_calls, tool_call_id) "
            "values (:agent, :chat_id, :role, :content, :tool_calls, :tool_call_id); "
            "commit; end;"
        )
        try:
            await self._run(
                statement,
                {
                    "agent": agent,
                    "chat_id": chat_id,
                    "role": message.get("role", "user"),
                    "content": content,
                    "tool_calls": (
                        json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
                    ),
                    "tool_call_id": message.get("tool_call_id"),
                },
            )
        except StoreUnavailable as exc:
            log.warning("could not store a message: %s", exc)

    async def history(self, *, agent: str, chat_id: int, limit: int) -> list[dict[str, Any]]:
        """The most recent `limit` messages, oldest first.

        Ordered newest-first in SQL and reversed here: `fetch first N` has to
        take the newest rows, but the transcript has to be replayed in the order
        it happened.
        """
        statement = (
            "select role, content, tool_calls, tool_call_id from conversations "
            "where agent = :agent and chat_id = :chat_id "
            "order by id desc fetch first :lim rows only"
        )
        try:
            rows = await self._run(
                statement, {"agent": agent, "chat_id": chat_id, "lim": limit}
            )
        except StoreUnavailable as exc:
            log.warning("could not read history, starting empty: %s", exc)
            return []

        messages: list[dict[str, Any]] = []
        for row in reversed(rows):
            message: dict[str, Any] = {"role": row["role"]}
            # Only set what is present. An empty tool_call_id on an assistant
            # message, or an empty string where the API expects a list, is
            # rejected by the model endpoint — and a NULL column may come back
            # as either None or "" depending on the type.
            if row.get("content"):
                message["content"] = row["content"]
            else:
                message["content"] = None
            if row.get("tool_calls"):
                try:
                    message["tool_calls"] = json.loads(row["tool_calls"])
                except (TypeError, ValueError):
                    log.warning("dropping unparseable tool_calls on a stored message")
            if row.get("tool_call_id"):
                message["tool_call_id"] = row["tool_call_id"]
            messages.append(message)
        return messages

    async def clear(self, *, agent: str, chat_id: int) -> None:
        """Forget one conversation — what /new means once history is durable."""
        statement = (
            "begin delete from conversations "
            "where agent = :agent and chat_id = :chat_id; commit; end;"
        )
        try:
            await self._run(statement, {"agent": agent, "chat_id": chat_id})
        except StoreUnavailable as exc:
            log.warning("could not clear history: %s", exc)

    # -- facts about the user ----------------------------------------------

    async def facts(self, *, scope: str | None = None) -> list[Fact]:
        """Every fact, because they all go into the prompt.

        There is no search here on purpose: a few dozen facts cost about a
        thousand tokens, and retrieving a subset would mean the agent knowing
        things only sometimes, which is worse than not knowing them.
        """
        if scope:
            statement = (
                "select fact_key, fact, scope, to_char(updated_at, 'YYYY-MM-DD') as updated_at "
                "from facts where scope = :scope order by fact_key"
            )
            binds: dict[str, Any] = {"scope": scope}
        else:
            statement = (
                "select fact_key, fact, scope, to_char(updated_at, 'YYYY-MM-DD') as updated_at "
                "from facts order by fact_key"
            )
            binds = {}
        try:
            rows = await self._run(statement, binds)
        except StoreUnavailable as exc:
            log.warning("could not read facts: %s", exc)
            return []
        return [
            Fact(
                key=row["fact_key"],
                fact=row["fact"],
                scope=row.get("scope") or "user",
                updated_at=row.get("updated_at") or "",
            )
            for row in rows
        ]

    async def remember(
        self, *, key: str, fact: str, scope: str = "user", chat_id: int | None = None
    ) -> None:
        """Add or replace one fact.

        Keyed rather than appended, so learning the same thing twice updates it
        instead of stacking near-duplicates the agent then has to reconcile.
        `merge` keeps that a single statement — read-then-write would race two
        agents against each other.
        """
        statement = (
            "begin "
            "merge into facts f using (select :key as fact_key from dual) s "
            "on (f.fact_key = s.fact_key) "
            "when matched then update set fact = :fact, scope = :scope, "
            "updated_at = systimestamp "
            "when not matched then insert (fact_key, fact, scope, source_chat) "
            "values (:key, :fact, :scope, :chat_id); "
            "commit; end;"
        )
        await self._run(
            statement, {"key": key, "fact": fact, "scope": scope, "chat_id": chat_id}
        )

    async def forget(self, key: str) -> bool:
        """Remove one fact. Returns whether there was anything to remove.

        Actually deletes. Being able to say "forget that about me" and have it
        be true is the reason this lives in a database rather than in git, where
        a deleted line stays in the history for good.
        """
        rows = await self._run(
            "select count(*) as n from facts where fact_key = :key", {"key": key}
        )
        existed = bool(rows) and int(rows[0]["n"]) > 0
        if existed:
            await self._run(
                "begin delete from facts where fact_key = :key; commit; end;", {"key": key}
            )
        return existed
