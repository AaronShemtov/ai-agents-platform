"""What the durable store sends, and what it does when the database says no.

Tested against a real HTTP server rather than a mocked client, because most of
what went wrong while writing this module was the shape of the request — binds
without `data_type` come back as a 400 that blames the SQL — and a mock would
have accepted whatever shape it was handed.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentcore.store import MAX_CONTENT_CHARS, AdbStore, StoreUnavailable

# What the fake ORDS answers next, and what it was asked. Module-level because
# the handler class cannot easily carry state.
_replies: list[tuple[int, str]] = []
_requests: list[dict] = []


class FakeOrds(BaseHTTPRequestHandler):
    # Name fixed by BaseHTTPRequestHandler, not a style choice.
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        _requests.append(json.loads(self.rfile.read(length) or b"{}"))
        status, body = _replies.pop(0) if _replies else (200, '{"items":[]}')
        payload = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        return  # keep pytest output readable


@pytest.fixture
def ords() -> Iterator[str]:
    _replies.clear()
    _requests.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOrds)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/ords/agents/_/sql"
    server.shutdown()
    server.server_close()


def store(url: str) -> AdbStore:
    return AdbStore(base_url=url, username="AGENTS", password="pw", timeout_seconds=5)


def rows(*items: dict) -> str:
    return json.dumps({"items": [{"resultSet": {"items": list(items)}}]})


def binds_of(request: dict) -> dict[str, dict]:
    return {b["name"]: b for b in request.get("binds", [])}


# -- the request shape ORDS insists on ---------------------------------------


async def test_every_bind_carries_a_data_type(ords: str) -> None:
    """The mistake that cost the most: without data_type, ORDS answers 400
    "Problem recognizing JSON" and says nothing about binds."""
    await store(ords).append(
        agent="lead", chat_id=7, message={"role": "user", "content": "hi"}
    )
    for name, bind in binds_of(_requests[0]).items():
        assert "data_type" in bind, f"bind {name} has no data_type"


async def test_numbers_bind_as_numbers_and_text_as_text(ords: str) -> None:
    await store(ords).append(
        agent="lead", chat_id=7, message={"role": "user", "content": "hi"}
    )
    binds = binds_of(_requests[0])
    assert binds["chat_id"]["data_type"] == "NUMBER"
    assert binds["agent"]["data_type"] == "VARCHAR2"


async def test_a_long_message_binds_as_a_clob(ords: str) -> None:
    """VARCHAR2 cannot hold it, and the failure would only show up on the one
    long message somebody happens to send."""
    await store(ords).append(
        agent="lead", chat_id=7, message={"role": "user", "content": "x" * 5000}
    )
    assert binds_of(_requests[0])["content"]["data_type"] == "CLOB"


async def test_an_absent_value_binds_as_null(ords: str) -> None:
    await store(ords).append(
        agent="lead", chat_id=7, message={"role": "user", "content": "hi"}
    )
    tool_call_id = binds_of(_requests[0])["tool_call_id"]
    assert tool_call_id["value"] is None


async def test_writes_commit_in_the_same_statement(ords: str) -> None:
    """Sending `commit` as a second request leaves a window in which the row
    exists uncommitted and the process can die."""
    await store(ords).append(
        agent="lead", chat_id=7, message={"role": "user", "content": "hi"}
    )
    text = _requests[0]["statementText"]
    assert text.startswith("begin ")
    assert "commit;" in text


# -- history -----------------------------------------------------------------


async def test_history_comes_back_in_the_order_it_happened(ords: str) -> None:
    """SQL has to take the newest rows, so it sorts descending; a transcript
    replayed newest-first would be nonsense."""
    _replies.append(
        (
            200,
            rows(
                {"role": "assistant", "content": "second", "tool_calls": None,
                 "tool_call_id": None},
                {"role": "user", "content": "first", "tool_calls": None,
                 "tool_call_id": None},
            ),
        )
    )
    history = await store(ords).history(agent="lead", chat_id=7, limit=10)
    assert [m["content"] for m in history] == ["first", "second"]


async def test_tool_calls_come_back_as_structure_not_text(ords: str) -> None:
    _replies.append(
        (
            200,
            rows(
                {"role": "assistant", "content": None,
                 "tool_calls": '[{"id": "1", "name": "github__get_me"}]',
                 "tool_call_id": None}
            ),
        )
    )
    history = await store(ords).history(agent="lead", chat_id=7, limit=10)
    assert history[0]["tool_calls"] == [{"id": "1", "name": "github__get_me"}]


async def test_an_empty_tool_call_id_is_left_off_entirely(ords: str) -> None:
    """The model endpoint rejects a transcript carrying tool_call_id on a
    message that is not a tool result."""
    _replies.append(
        (200, rows({"role": "user", "content": "hi", "tool_calls": None, "tool_call_id": ""}))
    )
    history = await store(ords).history(agent="lead", chat_id=7, limit=10)
    assert "tool_call_id" not in history[0]


async def test_unparseable_tool_calls_lose_the_field_not_the_message(ords: str) -> None:
    _replies.append(
        (200, rows({"role": "assistant", "content": "still useful",
                    "tool_calls": "{not json", "tool_call_id": None}))
    )
    history = await store(ords).history(agent="lead", chat_id=7, limit=10)
    assert history[0]["content"] == "still useful"
    assert "tool_calls" not in history[0]


# -- what happens when the database is not there -----------------------------


async def test_reading_history_survives_the_database_being_down(ords: str) -> None:
    """An agent with no memory is worse than one with memory, and far better
    than one that cannot answer at all."""
    _replies.append((500, "the database is having a moment"))
    assert await store(ords).history(agent="lead", chat_id=7, limit=10) == []


async def test_storing_a_message_survives_the_database_being_down(ords: str) -> None:
    _replies.append((500, "nope"))
    # Must not raise: the user is waiting for an answer that is already written.
    await store(ords).append(
        agent="lead", chat_id=7, message={"role": "user", "content": "hi"}
    )


async def test_remembering_a_fact_does_not_fail_silently(ords: str) -> None:
    """The opposite call: this one is an explicit instruction from the user, so
    swallowing the failure would have the agent confirm it stored something it
    did not."""
    _replies.append((500, "nope"))
    with pytest.raises(StoreUnavailable):
        await store(ords).remember(key="k", fact="f")


async def test_a_sql_error_is_reported_not_hidden(ords: str) -> None:
    _replies.append(
        (200, json.dumps({"items": [{"errorMessage": "ORA-00942: table does not exist"}]}))
    )
    with pytest.raises(StoreUnavailable, match="ORA-00942"):
        await store(ords).remember(key="k", fact="f")


async def test_ping_reports_rather_than_raises(ords: str) -> None:
    _replies.append((500, "down"))
    assert await store(ords).ping() is False


# -- facts -------------------------------------------------------------------


async def test_all_facts_are_returned_because_they_all_go_in_the_prompt(ords: str) -> None:
    _replies.append(
        (
            200,
            rows(
                {"fact_key": "git-identity", "fact": "personal email in personal repos",
                 "scope": "user", "updated_at": "2026-09-06"},
                {"fact_key": "no-local-builds", "fact": "never build images locally",
                 "scope": "user", "updated_at": "2026-09-06"},
            ),
        )
    )
    facts = await store(ords).facts()
    assert [f.key for f in facts] == ["git-identity", "no-local-builds"]
    assert facts[0].scope == "user"
    # No limit and no filter in the statement: retrieving a subset would have
    # the agent know things only sometimes.
    assert "fetch first" not in _requests[0]["statementText"]


async def test_remembering_the_same_key_twice_updates_it(ords: str) -> None:
    """Appending instead would stack near-duplicates for the agent to
    reconcile, and it has no way to tell which is current."""
    await store(ords).remember(key="git-identity", fact="v2")
    assert "merge into facts" in _requests[0]["statementText"]


async def test_forgetting_something_absent_says_so(ords: str) -> None:
    _replies.append((200, rows({"n": 0})))
    assert await store(ords).forget("never-knew-this") is False
    # And it must not go on to issue a pointless delete.
    assert len(_requests) == 1


async def test_forgetting_something_present_deletes_it(ords: str) -> None:
    _replies.append((200, rows({"n": 1})))
    _replies.append((200, '{"items":[]}'))
    assert await store(ords).forget("git-identity") is True
    assert "delete from facts" in _requests[1]["statementText"]


# -- size --------------------------------------------------------------------


async def test_an_enormous_message_is_trimmed_before_storing(ords: str) -> None:
    """A tool result can be megabytes. Storing it whole slows every later read
    of the conversation for no benefit — the transcript is trimmed anyway."""
    await store(ords).append(
        agent="lead",
        chat_id=7,
        message={"role": "tool", "content": "y" * (MAX_CONTENT_CHARS + 5000)},
    )
    stored = binds_of(_requests[0])["content"]["value"]
    assert len(stored) < MAX_CONTENT_CHARS + 200
    assert "not stored" in stored


async def test_structured_content_is_stored_as_json(ords: str) -> None:
    await store(ords).append(
        agent="lead",
        chat_id=7,
        message={"role": "user", "content": [{"type": "text", "text": "hi"}]},
    )
    assert json.loads(binds_of(_requests[0])["content"]["value"]) == [
        {"type": "text", "text": "hi"}
    ]
