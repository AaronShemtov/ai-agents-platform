"""What the durable store sends, and what it does when the database says no.

Tested against a real HTTP server rather than a mocked client, because most of
what went wrong while writing this module was the shape of the request — binds
without `data_type` come back as a 400 that blames the SQL — and a mock would
have accepted whatever shape it was handed.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentcore.store import (
    MAX_CONTENT_CHARS,
    AdbStore,
    ConversationSummary,
    StoredText,
    StoreSqlError,
    StoreUnavailable,
)

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
        agent="lead", chat_id=7, thread_id=0, message={"role": "user", "content": "hi"}
    )
    for name, bind in binds_of(_requests[0]).items():
        assert "data_type" in bind, f"bind {name} has no data_type"


async def test_numbers_bind_as_numbers_and_text_as_text(ords: str) -> None:
    await store(ords).append(
        agent="lead", chat_id=7, thread_id=0, message={"role": "user", "content": "hi"}
    )
    binds = binds_of(_requests[0])
    assert binds["chat_id"]["data_type"] == "NUMBER"
    assert binds["agent"]["data_type"] == "VARCHAR2"


async def test_a_long_message_binds_as_a_clob(ords: str) -> None:
    """VARCHAR2 cannot hold it, and the failure would only show up on the one
    long message somebody happens to send."""
    await store(ords).append(
        agent="lead", chat_id=7, thread_id=0, message={"role": "user", "content": "x" * 5000}
    )
    assert binds_of(_requests[0])["content"]["data_type"] == "CLOB"


async def test_an_absent_value_binds_as_null(ords: str) -> None:
    await store(ords).append(
        agent="lead", chat_id=7, thread_id=0, message={"role": "user", "content": "hi"}
    )
    tool_call_id = binds_of(_requests[0])["tool_call_id"]
    assert tool_call_id["value"] is None


async def test_writes_commit_in_the_same_statement(ords: str) -> None:
    """Sending `commit` as a second request leaves a window in which the row
    exists uncommitted and the process can die."""
    await store(ords).append(
        agent="lead", chat_id=7, thread_id=0, message={"role": "user", "content": "hi"}
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
    history = await store(ords).history(agent="lead", chat_id=7, thread_id=0, limit=10)
    assert [m["content"] for m in history] == ["first", "second"]


async def test_tool_calls_come_back_as_structure_not_text(ords: str) -> None:
    _replies.append(
        (
            200,
            rows(
                # Newest first, the way the descending query returns them, and
                # a whole turn rather than a fragment: history() now refuses a
                # window that opens or closes mid-turn, because an endpoint
                # rejects both an orphaned tool result and a tool call with no
                # result — and the replay would 400 on every message after.
                {"role": "tool", "content": "you are Aaron",
                 "tool_calls": None, "tool_call_id": "1"},
                {"role": "assistant", "content": None,
                 "tool_calls": '[{"id": "1", "name": "github__get_me"}]',
                 "tool_call_id": None},
                {"role": "user", "content": "кто я?", "tool_calls": None,
                 "tool_call_id": None},
            ),
        )
    )
    history = await store(ords).history(agent="lead", chat_id=7, thread_id=0, limit=10)
    assert history[1]["tool_calls"] == [{"id": "1", "name": "github__get_me"}]


async def test_an_empty_tool_call_id_is_left_off_entirely(ords: str) -> None:
    """The model endpoint rejects a transcript carrying tool_call_id on a
    message that is not a tool result."""
    _replies.append(
        (200, rows({"role": "user", "content": "hi", "tool_calls": None, "tool_call_id": ""}))
    )
    history = await store(ords).history(agent="lead", chat_id=7, thread_id=0, limit=10)
    assert "tool_call_id" not in history[0]


async def test_unparseable_tool_calls_lose_the_field_not_the_message(ords: str) -> None:
    _replies.append(
        (200, rows(
            # Newest first, as the descending query returns them.
            {"role": "assistant", "content": "still useful",
             "tool_calls": "{not json", "tool_call_id": None},
            {"role": "user", "content": "спроси", "tool_calls": None,
             "tool_call_id": None},
        ))
    )
    history = await store(ords).history(agent="lead", chat_id=7, thread_id=0, limit=10)
    assert history[1]["content"] == "still useful"
    assert "tool_calls" not in history[1]


# -- what happens when the database is not there -----------------------------


async def test_reading_history_survives_the_database_being_down(ords: str) -> None:
    """An agent with no memory is worse than one with memory, and far better
    than one that cannot answer at all."""
    _replies.append((500, "the database is having a moment"))
    assert await store(ords).history(agent="lead", chat_id=7, thread_id=0, limit=10) == []


async def test_storing_a_message_survives_the_database_being_down(ords: str) -> None:
    _replies.append((500, "nope"))
    # Must not raise: the user is waiting for an answer that is already written.
    await store(ords).append(
        agent="lead", chat_id=7, thread_id=0, message={"role": "user", "content": "hi"}
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


async def test_an_error_raised_inside_a_block_is_reported_not_hidden(ords: str) -> None:
    """The shape ORDS gives a PL/SQL block that raises, read off the live instance:
    no `errorMessage`, only `errorDetails`. Every write here is such a block, and
    looking for the other key alone had a failed insert come back as an empty,
    successful result."""
    _replies.append(
        (
            200,
            json.dumps(
                {
                    "items": [
                        {
                            "statementType": "plsql",
                            "result": "0",
                            "errorCode": "12899",
                            "errorLine": "1",
                            "errorDetails": "ORA-12899: value too large for column",
                        }
                    ]
                }
            ),
        )
    )
    with pytest.raises(StoreUnavailable, match="ORA-12899"):
        await store(ords).remember(key="k", fact="f")


async def test_storing_a_message_survives_an_error_inside_the_block(
    ords: str, caplog: pytest.LogCaptureFixture
) -> None:
    _replies.append((200, json.dumps({"items": [{"errorDetails": "ORA-00904: invalid"}]})))
    # Best effort, as with a database that is down: the answer is already written.
    with caplog.at_level("WARNING"):
        await store(ords).append(
            agent="lead", chat_id=7, thread_id=0, message={"role": "user", "content": "hi"}
        )
    # And said, not swallowed: the failure is in the log.
    assert any("could not store a message" in r.getMessage() for r in caplog.records)


async def test_a_sql_error_is_told_apart_from_a_database_that_is_down(ords: str) -> None:
    """One is a schema that is wrong and stays wrong; the other comes back by itself."""
    _replies.append((200, json.dumps({"items": [{"errorDetails": "ORA-00904: invalid"}]})))
    with pytest.raises(StoreSqlError):
        await store(ords).remember(key="k", fact="f")
    _replies.append((500, "down"))
    with pytest.raises(StoreUnavailable) as info:
        await store(ords).remember(key="k", fact="f")
    assert not isinstance(info.value, StoreSqlError)


async def test_clearing_says_whether_the_rows_are_gone(ords: str) -> None:
    """/new tells the owner the database is clear only when it is."""
    assert await store(ords).clear(agent="lead", chat_id=7, thread_id=0) is True
    _replies.append((200, json.dumps({"items": [{"errorDetails": "ORA-01031: privileges"}]})))
    assert await store(ords).clear(agent="lead", chat_id=7, thread_id=0) is False
    _replies.append((500, "down"))
    assert await store(ords).clear(agent="lead", chat_id=7, thread_id=0) is False


async def test_ping_reports_rather_than_raises(ords: str) -> None:
    _replies.append((500, "down"))
    assert await store(ords).ping() is False


# -- the schema check --------------------------------------------------------


async def test_the_schema_check_passes_when_the_column_answers(ords: str) -> None:
    assert await store(ords).check_schema() is True
    assert "thread_id" in _requests[0]["statementText"]


async def test_the_schema_check_says_so_when_it_passes(
    ords: str, caplog: pytest.LogCaptureFixture
) -> None:
    """After a good rollout one line proves the check ran; silence proves nothing."""
    with caplog.at_level("INFO"):
        assert await store(ords).check_schema() is True
    assert any("memory schema ok" in r.getMessage() for r in caplog.records)


async def test_a_database_that_is_down_does_not_send_anyone_to_the_migration(
    ords: str, caplog: pytest.LogCaptureFixture
) -> None:
    _replies.append((503, "unavailable"))
    with caplog.at_level("WARNING"):
        assert await store(ords).check_schema() is False
    messages = [r.getMessage() for r in caplog.records]
    assert any("could not check the memory schema" in m for m in messages)
    assert not any("migrations" in m for m in messages)
    assert not any(r.levelname == "ERROR" for r in caplog.records)


async def test_an_answer_that_is_not_json_does_not_escape_the_check(
    ords: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The check runs as a task nobody awaits: an exception out of it would only show
    up later, as "Task exception was never retrieved"."""
    _replies.append((200, "<html>not json</html>"))
    with caplog.at_level("ERROR"):
        assert await store(ords).check_schema() is False
    assert any("could not check the memory schema" in r.getMessage() for r in caplog.records)


async def test_the_schema_check_says_so_when_the_column_is_missing(
    ords: str, caplog: pytest.LogCaptureFixture
) -> None:
    """New code on a table without thread_id reads nothing and stores nothing, and
    every statement fails as one warning among others. This is the one loud line."""
    _replies.append(
        (200, json.dumps({"items": [{"errorMessage": "ORA-00904: invalid identifier"}]}))
    )
    with caplog.at_level("ERROR"):
        assert await store(ords).check_schema() is False
    assert any("thread_id" in r.getMessage() and r.levelname == "ERROR" for r in caplog.records)


# -- topics ------------------------------------------------------------------


async def test_the_topic_goes_in_with_the_message(ords: str) -> None:
    await store(ords).append(
        agent="lead", chat_id=-100123, thread_id=7, message={"role": "user", "content": "hi"}
    )
    request = _requests[0]
    assert "thread_id" in request["statementText"]
    assert binds_of(request)["thread_id"]["data_type"] == "NUMBER"
    assert binds_of(request)["thread_id"]["value"] == 7


async def test_history_is_read_for_one_topic_only(ords: str) -> None:
    """The whole point: a topic replays its own messages, not the group's."""
    await store(ords).history(agent="lead", chat_id=-100123, thread_id=7, limit=40)
    request = _requests[0]
    assert "thread_id = :thread_id" in request["statementText"]
    assert binds_of(request)["thread_id"]["value"] == 7
    assert binds_of(request)["chat_id"]["value"] == -100123


async def test_clearing_one_topic_leaves_the_others(ords: str) -> None:
    """/new typed in a topic must not reach into the other topics of the chat."""
    await store(ords).clear(agent="lead", chat_id=-100123, thread_id=7)
    request = _requests[0]
    assert "thread_id = :thread_id" in request["statementText"]
    assert binds_of(request)["thread_id"]["value"] == 7


async def test_no_topic_is_zero_and_still_a_number(ords: str) -> None:
    """0, not NULL: the column is NOT NULL, and 0 is where every old row sits."""
    await store(ords).append(
        agent="lead", chat_id=7, thread_id=0, message={"role": "user", "content": "hi"}
    )
    bind = binds_of(_requests[0])["thread_id"]
    assert bind["data_type"] == "NUMBER"
    assert bind["value"] == 0


async def test_every_bind_is_used_by_its_statement_and_the_other_way_round(ords: str) -> None:
    """Oracle refuses a bind the statement does not name and a name it has no bind
    for, and nothing in CI runs Oracle. A statement and its binds drifting apart is
    otherwise found on the first message after a deploy."""
    s = store(ords)
    await s.append(
        agent="lead", chat_id=1, thread_id=2, message={"role": "user", "content": "hi"}
    )
    await s.history(agent="lead", chat_id=1, thread_id=2, limit=5)
    await s.clear(agent="lead", chat_id=1, thread_id=2)
    for request in _requests:
        named = set(re.findall(r":([a-z_][a-z0-9_]*)", request["statementText"]))
        assert named == set(binds_of(request)), request["statementText"]


async def test_the_topic_is_not_optional_on_anything_that_reads_or_deletes(ords: str) -> None:
    """A default would let a forgotten call site read, or delete, the whole chat."""
    with pytest.raises(TypeError):
        await store(ords).history(agent="lead", chat_id=7, limit=10)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await store(ords).clear(agent="lead", chat_id=7)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await store(ords).append(  # type: ignore[call-arg]
            agent="lead", chat_id=7, message={"role": "user", "content": "hi"}
        )


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
        thread_id=0,
        message={"role": "tool", "content": "y" * (MAX_CONTENT_CHARS + 5000)},
    )
    stored = binds_of(_requests[0])["content"]["value"]
    assert len(stored) < MAX_CONTENT_CHARS + 200
    assert "not stored" in stored


async def test_structured_content_is_stored_as_json(ords: str) -> None:
    await store(ords).append(
        agent="lead",
        chat_id=7,
        thread_id=0,
        message={"role": "user", "content": [{"type": "text", "text": "hi"}]},
    )
    assert json.loads(binds_of(_requests[0])["content"]["value"]) == [
        {"type": "text", "text": "hi"}
    ]


# -- the other conversations -------------------------------------------------


def bound(request: dict) -> dict:
    return {name: bind["value"] for name, bind in binds_of(request).items()}


async def test_the_conversations_of_one_agent_are_listed_newest_first(ords: str) -> None:
    _replies.append(
        (
            200,
            rows(
                {
                    "chat_id": -100123,
                    "thread_id": 78,
                    "n": 4,
                    "first_day": "2026-10-11",
                    "last_at": "2026-10-11 21:42",
                    "first_user": "начало темы",
                    "last_user": None,
                },
                {
                    "chat_id": 555,
                    "thread_id": 0,
                    "n": 505,
                    "first_day": "2026-09-06",
                    "last_at": "2026-10-06 19:25",
                    "first_user": None,
                    "last_user": "",
                },
            ),
        )
    )
    got = await store(ords).conversations(agent="lead")
    assert got == [
        ConversationSummary(
            chat_id=-100123,
            thread_id=78,
            messages=4,
            first_day="2026-10-11",
            last_at="2026-10-11 21:42",
            first_user="начало темы",
            last_user="",
        ),
        ConversationSummary(
            chat_id=555,
            thread_id=0,
            messages=505,
            first_day="2026-09-06",
            last_at="2026-10-06 19:25",
            first_user="",
            last_user="",
        ),
    ]
    statement = _requests[0]["statementText"]
    assert statement.count("agent = :agent") == 3  # the counts and both previews
    assert statement.endswith("order by o.last_id desc")
    assert bound(_requests[0])["agent"] == "lead"


async def test_a_listing_says_where_a_conversation_began_and_what_its_previews_cut(
    ords: str,
) -> None:
    _replies.append(
        (
            200,
            rows(
                {
                    "chat_id": -100123,
                    "thread_id": 78,
                    "n": 4,
                    "first_id": 1852,
                    "first_day": "2026-10-11",
                    "last_at": "2026-10-11 21:42",
                    "first_user": "x" * 200,
                    "first_len": 3000,
                    "last_user": "short",
                    "last_len": 5,
                }
            ),
        )
    )
    [got] = await store(ords).conversations(agent="lead")
    assert got.first_id == 1852
    assert got.first_user_cut is True
    assert got.last_user_cut is False


async def test_a_listing_that_fails_is_an_error_rather_than_no_conversations(ords: str) -> None:
    """history() answers [] when the database is down, because a turn goes on without
    it. A tool cannot: "there are no other conversations" is a claim about the past."""
    _replies.append((503, "down"))
    with pytest.raises(StoreUnavailable):
        await store(ords).conversations(agent="lead")
    _replies.append((200, json.dumps({"items": [{"errorMessage": "ORA-00904: invalid"}]})))
    with pytest.raises(StoreSqlError):
        await store(ords).conversations(agent="lead")


async def test_search_terms_are_binds_and_never_part_of_the_statement(ords: str) -> None:
    """They come from the model, which is reading whatever the conversation contained."""
    await store(ords).search(agent="lead", terms=["Кластер", "x' or 1=1 --"], limit=5)
    request = _requests[0]
    values = bound(request)
    assert values["t0"] == "кластер"  # lower case, to meet lower(content)
    assert values["t1"] == "x' or 1=1 --"
    assert "кластер" not in request["statementText"]
    assert "1=1" not in request["statementText"]
    assert (values["agent"], values["lim"]) == ("lead", 5)


async def test_search_reads_what_was_said_and_not_what_tools_returned(ords: str) -> None:
    await store(ords).search(agent="lead", terms=["oracle"], limit=1)
    statement = _requests[0]["statementText"]
    assert "role in ('user', 'assistant')" in statement
    assert "agent = :agent" in statement


async def test_search_ranks_by_how_many_terms_matched_then_newest(ords: str) -> None:
    await store(ords).search(agent="lead", terms=["a1", "b2"], limit=3)
    statement = _requests[0]["statementText"]
    assert "sign(dbms_lob.instr(lc, :t0)) + sign(dbms_lob.instr(lc, :t1)) as score" in statement
    assert statement.endswith("then 1 else 0 end, score desc, id desc fetch first :lim rows only")


async def test_search_puts_the_conversation_the_turn_is_in_after_the_others(ords: str) -> None:
    """Its newest messages are already in the transcript, and being the newest they
    filled every slot a search had."""
    await store(ords).search(agent="lead", terms=["ords"], limit=8, current=(-100123, 7))
    request = _requests[0]
    values = bound(request)
    assert (values["cur_chat"], values["cur_thread"]) == (-100123, 7)
    assert (
        "order by case when chat_id = :cur_chat and thread_id = :cur_thread then 1 else 0 end"
        in request["statementText"]
    )


async def test_search_without_a_conversation_puts_nothing_last(ords: str) -> None:
    await store(ords).search(agent="lead", terms=["ords"], limit=8)
    values = bound(_requests[0])
    assert (values["cur_chat"], values["cur_thread"]) == (0, -1)  # matches no row


async def test_search_reads_yo_as_ye_on_both_sides(ords: str) -> None:
    await store(ords).search(agent="lead", terms=["Отчёт"], limit=1)
    request = _requests[0]
    assert bound(request)["t0"] == "отчет"
    assert "translate(lower(content), 'ё', 'е') as lc" in request["statementText"]


async def test_no_terms_means_no_query_at_all(ords: str) -> None:
    assert await store(ords).search(agent="lead", terms=[], limit=5) == []
    assert await store(ords).search(agent="lead", terms=[""], limit=5) == []
    assert _requests == []


async def test_a_hit_says_where_its_excerpt_sits_in_the_message(ords: str) -> None:
    _replies.append(
        (
            200,
            rows(
                {
                    "id": 1853,
                    "chat_id": -100123,
                    "thread_id": 78,
                    "role": "user",
                    "at": "2026-10-11 21:42",
                    "len": 900,
                    "start_at": 401,
                    "excerpt": "загадал число 7",
                }
            ),
        )
    )
    [hit] = await store(ords).search(agent="lead", terms=["число"], limit=5)
    assert hit == StoredText(
        id=1853,
        chat_id=-100123,
        thread_id=78,
        role="user",
        at="2026-10-11 21:42",
        text="загадал число 7",
        start=401,
        length=900,
    )


async def test_reading_the_newest_messages_asks_for_nothing_after_them(ords: str) -> None:
    await store(ords).read(
        agent="lead", chat_id=-100123, thread_id=78, around=None, limit=12, width=600,
        focus_width=2500,
    )
    request = _requests[0]
    values = bound(request)
    assert "union all" not in request["statementText"]
    assert values["before"] == 12
    assert "after" not in values
    assert values["focus"] == -1  # no message is singled out
    assert values["upto"] >= 10**15


async def test_reading_around_a_message_takes_both_sides_and_widens_that_one(ords: str) -> None:
    await store(ords).read(
        agent="lead", chat_id=-100123, thread_id=78, around=1853, limit=5, width=600,
        focus_width=2500,
    )
    request = _requests[0]
    values = bound(request)
    assert "union all" in request["statementText"]
    assert (values["before"], values["after"]) == (3, 2)  # the message itself is in before
    assert values["upto"] == values["focus"] == 1853
    assert (values["width"], values["focus_width"]) == (600, 2500)
    assert (values["agent"], values["chat_id"], values["thread_id"]) == ("lead", -100123, 78)
    assert request["statementText"].count("thread_id = :thread_id") == 2


async def test_a_read_comes_back_oldest_first_in_whatever_order_it_arrives(ords: str) -> None:
    _replies.append(
        (
            200,
            rows(
                {"id": 12, "role": "assistant", "at": "t", "text": "b", "len": 1},
                {"id": 10, "role": "user", "at": "t", "text": "a", "len": 1},
                {"id": 14, "role": "user", "at": "t", "text": "c", "len": 5},
            ),
        )
    )
    got = await store(ords).read(
        agent="lead", chat_id=1, thread_id=0, around=12, limit=3, width=10, focus_width=20
    )
    assert [m.id for m in got] == [10, 12, 14]
    assert all((m.chat_id, m.thread_id, m.start) == (1, 0, 1) for m in got)
    assert got[-1].length == 5


async def test_every_recall_statement_names_exactly_the_binds_it_is_sent(ords: str) -> None:
    s = store(ords)
    await s.conversations(agent="lead")
    await s.search(agent="lead", terms=["a1", "b2", "c3"], limit=4)
    await s.read(agent="lead", chat_id=1, thread_id=2, around=None, limit=3, width=10, focus_width=20)
    await s.read(agent="lead", chat_id=1, thread_id=2, around=7, limit=3, width=10, focus_width=20)
    assert len(_requests) == 4
    for request in _requests:
        named = set(re.findall(r":([a-z_][a-z0-9_]*)", request["statementText"]))
        assert named == set(binds_of(request)), request["statementText"]
        for name, bind in binds_of(request).items():
            assert "data_type" in bind, name
