"""A restored window has to be something an endpoint will accept.

`history` takes the newest N rows and N counts messages, not turns, so the window
routinely opens in the middle of one. /chat/completions answers 400 to a `tool`
message with no assistant in front of it and refuses the entire request, and
because the same window is replayed on every following message the conversation
stays broken until it is cleared. The Responses API had been tolerating it, so
this was latent until the lead moved to a chat-completions model.
"""

from __future__ import annotations

from agentcore.store import replayable


def call(cid: str) -> dict:
    return {"id": cid, "type": "function", "function": {"name": "t", "arguments": "{}"}}


def user(text="привет"):
    return {"role": "user", "content": text}


def assistant(content="ok", calls=None):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [call(c) for c in calls]
    return msg


def tool(cid="c1", content="result"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


# -- the bug that was actually observed --------------------------------------


def test_a_window_opening_on_an_orphaned_tool_message():
    """The exact shape seen in production: 322 messages, window of 40."""
    window = [tool("c1"), assistant("итог"), user(), assistant("ответ")]
    out = replayable(window)
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[0]["content"] == "привет"


def test_a_complete_conversation_is_untouched():
    window = [
        user("посчитай"),
        assistant(None, ["c1"]),
        tool("c1"),
        assistant("готово"),
        user("спасибо"),
        assistant("пожалуйста"),
    ]
    assert replayable(window) == window


def test_everything_before_the_first_user_is_dropped():
    window = [assistant("хвост прошлого хода"), tool("c9"), user(), assistant("ok")]
    out = replayable(window)
    assert [m["role"] for m in out] == ["user", "assistant"]


def test_a_window_with_no_user_at_all_is_refused():
    """A fragment can only produce a 400; no history answers fine."""
    assert replayable([tool("c1"), assistant("итог")]) == []
    assert replayable([]) == []


# -- partially persisted turns ----------------------------------------------


def test_a_tool_result_nobody_asked_for_is_dropped():
    # The assistant row never made it to the database.
    window = [user(), tool("c1"), assistant("ответ")]
    out = replayable(window)
    assert [m["role"] for m in out] == ["user", "assistant"]


def test_a_tool_result_with_the_wrong_id_is_dropped():
    window = [user(), assistant(None, ["c1"]), tool("SOMETHING-ELSE"), assistant("ok")]
    out = replayable(window)
    assert [m["role"] for m in out] == ["user", "assistant", "assistant"]


def test_an_unanswered_assistant_turn_is_cut_off_the_end():
    window = [user(), assistant("ответ"), user("ещё"), assistant(None, ["c1"])]
    out = replayable(window)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]


def test_a_partially_answered_assistant_turn_is_cut_whole():
    """One of two results is as invalid as none, so the turn goes entirely."""
    window = [user(), assistant(None, ["c1", "c2"]), tool("c1")]
    out = replayable(window)
    assert [m["role"] for m in out] == ["user"]


def test_a_fully_answered_multi_call_turn_survives():
    window = [user(), assistant(None, ["c1", "c2"]), tool("c1"), tool("c2"), assistant("ok")]
    assert replayable(window) == window


# -- invariants --------------------------------------------------------------


def test_every_tool_message_has_an_owner_afterwards():
    """The property that matters, on a window built to violate it repeatedly."""
    window = [
        tool("orphan-1"),
        user(),
        tool("orphan-2"),
        assistant(None, ["c1"]),
        tool("c1"),
        assistant("ok"),
        tool("orphan-3"),
        user("ещё"),
        assistant(None, ["c2"]),
        tool("c2"),
        assistant("done"),
    ]
    out = replayable(window)

    awaiting: set[str] = set()
    for message in out:
        if message["role"] == "assistant":
            awaiting = {c["id"] for c in message.get("tool_calls", [])}
        elif message["role"] == "tool":
            assert message["tool_call_id"] in awaiting, "orphan survived"
            awaiting.discard(message["tool_call_id"])
        else:
            awaiting = set()
    assert not awaiting, "the window ends on an unanswered assistant"
    assert out[0]["role"] == "user"
