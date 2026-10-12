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

Which conversation
------------------
A row belongs to an agent, a chat and a topic (`thread_id`: the forum topic of the
chat, 0 for none). Without the topic every topic of a forum group replayed one shared
history, and the agent answered each of them with the others mixed in. `thread_id` is
a required keyword on everything that reads or deletes history, so a call site that
forgot it is a TypeError rather than a silent read or delete of the whole chat.
Rows from before the column existed carry the default, 0, and were deliberately left
there. In the private chat that is simply its history. In a forum group it is every
topic's messages mixed together, which is what the group had: they are replayed as
the General topic's, and the General topic's /new deletes them. Nothing records which
topic an old message came from, so nothing could have sorted them.

No vectors yet
--------------
Deliberately. Facts number in the dozens and go into the prompt whole, so there
is nothing to search. History is searched — the other conversations are reached
that way, see recall.py — but by plain substring, `lower()` and `dbms_lob.instr`
over every row, with no index: about 250 ms at two thousand rows, measured on the
live instance. The database supports vectors (``VECTOR``, 26ai, checked); choosing
where embeddings come from is a separate decision worth making when the scan is
too slow or the words people search with stop matching the words they used.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
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

# How much of a conversation's first and last user message the listing carries.
PREVIEW_CHARS = 200

# The search excerpt: this many characters, starting this far before the match.
EXCERPT_CHARS = 360
EXCERPT_LEAD = 120

# Past every id the table will hold: "the newest messages" is "everything up to here".
_NEWEST = 10**15


class StoreUnavailable(RuntimeError):
    """The database could not be reached or refused the statement."""


class StoreSqlError(StoreUnavailable):
    """The database answered, and the statement failed: a missing column, a constraint.

    Apart from the transport failures so that a caller can tell a database that is
    down, which comes back by itself, from a schema that is wrong, which does not.
    """


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


def replayable(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim a restored window down to something an endpoint will accept.

    `history` takes the newest N rows, and N counts messages rather than turns,
    so the window routinely begins in the middle of one. A window that starts on
    a `tool` message is not merely untidy — /chat/completions rejects the whole
    request with 400 "messages with role 'tool' must be a response to a
    preceeding message with 'tool_calls'", and since the same window is replayed
    on every following message, the conversation stays broken until it is
    cleared. Observed exactly that on a 322-message chat whose 40-message window
    opened on an orphaned tool result.

    It only surfaced when the lead moved to a /chat/completions model. The
    Responses API had been tolerating the same malformed history, which is worth
    knowing: this was latent for as long as the default was a codex or astra
    deployment, and a model switch is what made it visible.

    Three rules, all about turn boundaries:

      * Drop everything before the first `user` message. That is the only
        unambiguous start of a turn, and it is already how memory._split_into_turns
        defines one.
      * Drop a `tool` message that no preceding assistant asked for. A turn
        persisted halfway — the process died between two appends — can leave one.
      * Drop a trailing assistant whose tool calls were never answered, for the
        same reason in the other direction.
    """
    first_user = next(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), None
    )
    if first_user is None:
        # Nothing but a fragment: replaying it can only produce a 400, and an
        # agent with no history answers fine.
        return []

    out: list[dict[str, Any]] = []
    awaiting: set[str] = set()
    for message in messages[first_user:]:
        role = message.get("role")
        if role == "tool":
            if message.get("tool_call_id") not in awaiting:
                log.warning("dropping a tool message no assistant asked for")
                continue
            awaiting.discard(message["tool_call_id"])
        elif role == "assistant":
            awaiting = {
                call.get("id")
                for call in (message.get("tool_calls") or [])
                if call.get("id")
            }
        else:
            awaiting = set()
        out.append(message)

    if awaiting:
        # Cut back to before the assistant whose calls went unanswered. Popping
        # only the last message is not enough: with two calls answered once, the
        # tail is `assistant, tool` and a partial set of results is rejected just
        # as an empty one is.
        for index in range(len(out) - 1, -1, -1):
            if out[index].get("role") == "assistant" and out[index].get("tool_calls"):
                log.warning("dropping an assistant turn whose tool calls went unanswered")
                del out[index:]
                break

    return out


@dataclass(frozen=True)
class Fact:
    key: str
    fact: str
    scope: str
    updated_at: str


@dataclass(frozen=True)
class ConversationSummary:
    """One conversation of an agent, as the index and the recall tools describe it.

    `first_user` and `last_user` are the opening characters of the person's first and
    latest message in it, empty when there is none. Times are the database's, UTC.
    """

    chat_id: int
    thread_id: int
    messages: int
    first_day: str
    last_at: str
    first_user: str
    last_user: str


@dataclass(frozen=True)
class StoredText:
    """Words from a conversation, the person's or the agent's, as recall shows them.

    `text` is a window of the message rather than all of it: it begins at `start`
    (1-based, the way Oracle counts) and the whole message is `length` characters,
    so whoever shows it can say what was left out.
    """

    id: int
    chat_id: int
    thread_id: int
    role: str
    at: str
    text: str
    start: int
    length: int


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
            # Two shapes, checked against the live instance. A plain statement that
            # fails (a select from a missing table) carries `errorMessage`. A PL/SQL
            # block that raises (every write here is one: `begin insert ...; commit;
            # end;`) carries only `errorDetails`. Looking for the first alone let a
            # failed insert, delete or merge come back as an empty result: /new
            # reported the database cleared when it was not, and remember() confirmed
            # a fact it had not stored.
            error = item.get("errorMessage") or item.get("errorDetails")
            if error:
                raise StoreSqlError(f"SQL error: {str(error)[:300]}")
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

    async def check_schema(self) -> bool:
        """Whether `conversations` has the columns this version writes.

        Asked once at startup and logged loudly rather than raised: a pod that will
        not start is worse than an agent that forgets, but an agent that forgets
        without a word is what the first line of this module exists to prevent.
        Without this, code that meets a table without `thread_id` answers every
        message normally, reads no history and stores nothing, each failure a single
        warning among the others.
        """
        try:
            await self._run("select thread_id from conversations where 1 = 0")
        except StoreSqlError as exc:
            # The database answered and refused: the column is not there. This is the
            # one case that sends the reader to the migration.
            log.error(
                "memory is not usable: the conversations table does not answer a "
                "query for thread_id (%s). Apply sql/migrations/ before this version "
                "runs; until then no history is read or stored.",
                exc,
            )
            return False
        except StoreUnavailable as exc:
            # Down, slow or refusing the login: says nothing about the schema, and
            # the first message after it recovers will find out for itself.
            log.warning("could not check the memory schema: %s", exc)
            return False
        except Exception:
            # Runs as a task nobody awaits; an exception escaping it would only
            # surface as "Task exception was never retrieved" at garbage collection.
            log.exception("could not check the memory schema")
            return False
        log.info("memory schema ok: conversations has thread_id")
        return True

    # -- conversation history ----------------------------------------------

    async def append(
        self, *, agent: str, chat_id: int, thread_id: int, message: dict[str, Any]
    ) -> None:
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
            "(agent, chat_id, thread_id, role, content, tool_calls, tool_call_id) "
            "values (:agent, :chat_id, :thread_id, :role, :content, :tool_calls, "
            ":tool_call_id); "
            "commit; end;"
        )
        try:
            await self._run(
                statement,
                {
                    "agent": agent,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
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

    async def history(
        self, *, agent: str, chat_id: int, thread_id: int, limit: int
    ) -> list[dict[str, Any]]:
        """The most recent `limit` messages of one conversation, oldest first.

        Ordered newest-first in SQL and reversed here: `fetch first N` has to
        take the newest rows, but the transcript has to be replayed in the order
        it happened.
        """
        statement = (
            "select role, content, tool_calls, tool_call_id from conversations "
            "where agent = :agent and chat_id = :chat_id and thread_id = :thread_id "
            "order by id desc fetch first :lim rows only"
        )
        try:
            rows = await self._run(
                statement,
                {"agent": agent, "chat_id": chat_id, "thread_id": thread_id, "lim": limit},
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
        return replayable(messages)

    async def clear(self, *, agent: str, chat_id: int, thread_id: int) -> bool:
        """Forget one conversation — what /new means once history is durable.

        Returns whether the rows are gone, so that /new can say so only when it is
        true: the next restart replays whatever a failed delete left behind.

        One topic, not the chat: /new typed in a topic must not reach into the
        others. In the General topic, or a chat without topics, that is thread 0 —
        which is also where everything stored before topics were told apart sits.
        """
        statement = (
            "begin delete from conversations "
            "where agent = :agent and chat_id = :chat_id and thread_id = :thread_id; "
            "commit; end;"
        )
        try:
            await self._run(
                statement, {"agent": agent, "chat_id": chat_id, "thread_id": thread_id}
            )
        except StoreUnavailable as exc:
            log.warning("could not clear history: %s", exc)
            return False
        return True

    # -- the other conversations ---------------------------------------------
    #
    # What recall.py and the tools in localtools.py read. Unlike history(), these raise
    # StoreUnavailable instead of answering empty: a search that failed must reach the
    # model as a failure, because "nothing was found" is an answer it would pass on as
    # a fact. Times come back as the database keeps them, UTC.

    async def conversations(self, *, agent: str) -> list[ConversationSummary]:
        """Every conversation `agent` has had, the most recently active first.

        One statement: the counts per conversation, joined to the first and the latest
        message the person wrote in it. Tool results are counted but never shown.
        """
        statement = (
            "select o.chat_id, o.thread_id, o.n, o.first_day, o.last_at, "
            "f.preview as first_user, l.preview as last_user "
            "from (select chat_id, thread_id, count(*) as n, max(id) as last_id, "
            "to_char(min(created_at), 'YYYY-MM-DD') as first_day, "
            "to_char(max(created_at), 'YYYY-MM-DD HH24:MI') as last_at "
            "from conversations where agent = :agent group by chat_id, thread_id) o "
            "left join (select chat_id, thread_id, dbms_lob.substr(content, :width, 1) as preview, "
            "row_number() over (partition by chat_id, thread_id order by id) as rk "
            "from conversations where agent = :agent and role = 'user' "
            "and content is not null) f "
            "on f.chat_id = o.chat_id and f.thread_id = o.thread_id and f.rk = 1 "
            "left join (select chat_id, thread_id, dbms_lob.substr(content, :width, 1) as preview, "
            "row_number() over (partition by chat_id, thread_id order by id desc) as rk "
            "from conversations where agent = :agent and role = 'user' "
            "and content is not null) l "
            "on l.chat_id = o.chat_id and l.thread_id = o.thread_id and l.rk = 1 "
            "order by o.last_id desc"
        )
        rows = await self._run(statement, {"agent": agent, "width": PREVIEW_CHARS})
        return [
            ConversationSummary(
                chat_id=int(row["chat_id"]),
                thread_id=int(row["thread_id"]),
                messages=int(row["n"]),
                first_day=row.get("first_day") or "",
                last_at=row.get("last_at") or "",
                first_user=row.get("first_user") or "",
                last_user=row.get("last_user") or "",
            )
            for row in rows
        ]

    async def search(
        self, *, agent: str, terms: Sequence[str], limit: int
    ) -> list[StoredText]:
        """Messages of `agent`'s conversations that contain any of `terms`.

        The person's words and the agent's answers, not tool results: those are most of
        the table and most of them are file contents and logs, which is noise when the
        question is what was said. Case-insensitive substring, so a stem finds every
        form of a word. Ranked by how many of the terms a message has, then newest.

        The terms are binds, never part of the statement text: they come from the model.
        Only their number shapes the statement. The excerpt starts a little before the
        first term that matched, because the start of a long answer is rarely the part
        that was searched for.
        """
        terms = [term.lower() for term in terms if term]
        if not terms:
            return []
        binds: dict[str, Any] = {
            "agent": agent,
            "lim": limit,
            "width": EXCERPT_CHARS,
            "lead": EXCERPT_LEAD,
        }
        found: list[str] = []
        first: list[str] = []
        for index, term in enumerate(terms):
            name = f"t{index}"
            binds[name] = term
            found.append(f"sign(dbms_lob.instr(lc, :{name}))")
            first.append(f"nullif(dbms_lob.instr(lc, :{name}), 0)")
        statement = (
            "select id, chat_id, thread_id, role, at, len, "
            "greatest(pos - :lead, 1) as start_at, "
            "dbms_lob.substr(content, :width, greatest(pos - :lead, 1)) as excerpt "
            "from (select id, chat_id, thread_id, role, content, "
            "to_char(created_at, 'YYYY-MM-DD HH24:MI') as at, "
            "dbms_lob.getlength(content) as len, "
            f"{' + '.join(found)} as score, "
            f"coalesce({', '.join(first)}, 1) as pos "
            "from (select id, chat_id, thread_id, role, created_at, content, "
            "lower(content) as lc from conversations "
            "where agent = :agent and role in ('user', 'assistant') "
            "and content is not null)) "
            "where score > 0 order by score desc, id desc fetch first :lim rows only"
        )
        rows = await self._run(statement, binds)
        return [
            StoredText(
                id=int(row["id"]),
                chat_id=int(row["chat_id"]),
                thread_id=int(row["thread_id"]),
                role=row["role"],
                at=row.get("at") or "",
                text=row.get("excerpt") or "",
                start=int(row.get("start_at") or 1),
                length=int(row.get("len") or 0),
            )
            for row in rows
        ]

    async def read(
        self,
        *,
        agent: str,
        chat_id: int,
        thread_id: int,
        around: int | None,
        limit: int,
        width: int,
        focus_width: int,
    ) -> list[StoredText]:
        """Up to `limit` messages of one conversation, oldest first.

        Without `around`, the newest ones. With it, the messages on both sides of that
        id, the message itself included and given `focus_width` characters instead of
        `width`: it is the one a search pointed at. The person's words and the agent's
        answers only, as in search().
        """
        if around is None:
            focus, before, after = -1, limit, 0
            upto = _NEWEST
        else:
            focus, upto = around, around
            before = (limit + 1) // 2
            after = limit - before
        messages = (
            "select id, role, to_char(created_at, 'YYYY-MM-DD HH24:MI') as at, "
            "dbms_lob.substr(content, case when id = :focus then :focus_width "
            "else :width end, 1) as text, "
            "dbms_lob.getlength(content) as len "
            "from conversations "
            "where agent = :agent and chat_id = :chat_id and thread_id = :thread_id "
            "and role in ('user', 'assistant') and content is not null "
        )
        statement = (
            f"select * from ({messages} and id <= :upto "
            "order by id desc fetch first :before rows only)"
        )
        binds: dict[str, Any] = {
            "agent": agent,
            "chat_id": chat_id,
            "thread_id": thread_id,
            "focus": focus,
            "focus_width": focus_width,
            "width": width,
            "upto": upto,
            "before": before,
        }
        if after:
            statement += (
                f" union all select * from ({messages} and id > :upto "
                "order by id fetch first :after rows only)"
            )
            binds["after"] = after
        rows = await self._run(statement, binds)
        return sorted(
            (
                StoredText(
                    id=int(row["id"]),
                    chat_id=chat_id,
                    thread_id=thread_id,
                    role=row["role"],
                    at=row.get("at") or "",
                    text=row.get("text") or "",
                    start=1,
                    length=int(row.get("len") or 0),
                )
                for row in rows
            ),
            key=lambda message: message.id,
        )

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
