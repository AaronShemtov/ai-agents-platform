"""Where the history starts must hold still, or no prompt cache can match it.

Trimming to exactly what fits moves the first message on every turn once the
budget is reached. Measured on the local box at a 1,200-token budget: turns
where the window held still cost 2.9s and 11.4s, the two where it slid cost
30.1s and 29.7s — the same context each time, the difference being 1,100 tokens
re-read from scratch. On Azure the same instability is money rather than
seconds, a cached input token costing a tenth of a fresh one.
"""

from __future__ import annotations

from itertools import pairwise

from agentcore.memory import TRIM_STEP, ChatMemory, _split_into_turns, estimate_tokens

SYSTEM = "system prompt"


def chat_with(turns: int, words: int = 60) -> ChatMemory:
    chat = ChatMemory(chat_id=1)
    for i in range(turns):
        chat.append({"role": "user", "content": f"вопрос {i} " + "слово " * words})
        chat.append({"role": "assistant", "content": f"ответ {i} " + "слово " * words})
    return chat


def first_message(messages: list[dict]) -> str:
    """What the cache keys on: the first thing after the system prompt."""
    return str(messages[1]["content"])[:40] if len(messages) > 1 else ""


# -- the property that matters -----------------------------------------------


def test_the_window_holds_still_for_several_turns():
    """One expensive re-read should buy a run of cheap turns, not one."""
    chat = chat_with(40)
    budget = 1500

    starts = []
    for i in range(TRIM_STEP * 3):
        # Turns the size of the ones already there. With tiny additions nothing
        # is evicted and the test would measure the padding, not the trimming.
        chat.append({"role": "user", "content": f"новый вопрос {i} " + "слово " * 60})
        starts.append(first_message(chat.transcript(SYSTEM, max_tokens=budget)))
        chat.append({"role": "assistant", "content": f"ответ {i} " + "слово " * 60})

    moves = sum(1 for a, b in pairwise(starts) if a != b)
    assert moves <= len(starts) // TRIM_STEP + 1, (
        f"the window moved {moves} times in {len(starts)} turns: {starts}"
    )


def test_without_snapping_the_window_moves_every_turn():
    """The behaviour being replaced, pinned so the comparison is not folklore."""
    chat = chat_with(40)
    budget = 1500

    starts = []
    for i in range(8):
        chat.append({"role": "user", "content": f"новый вопрос {i} " + "слово " * 60})
        starts.append(first_message(chat.transcript(SYSTEM, max_tokens=budget, step=1)))
        chat.append({"role": "assistant", "content": f"ответ {i} " + "слово " * 60})

    moves = sum(1 for a, b in pairwise(starts) if a != b)
    assert moves >= 6, f"expected the unsnapped window to slide constantly, saw {moves}"


# -- correctness is unchanged ------------------------------------------------


def test_the_budget_is_still_respected():
    chat = chat_with(40)
    for budget in (800, 1500, 4000):
        messages = chat.transcript(SYSTEM, max_tokens=budget)
        assert estimate_tokens(messages) <= budget, budget


def test_the_newest_turn_always_survives():
    chat = chat_with(40)
    chat.append({"role": "user", "content": "самый свежий вопрос"})
    messages = chat.transcript(SYSTEM, max_tokens=200)
    assert messages[-1]["content"] == "самый свежий вопрос"


def test_history_is_still_cut_only_at_turn_boundaries():
    """An orphaned tool result is a 400, whatever the snapping does."""
    chat = ChatMemory(chat_id=1)
    for i in range(12):
        chat.append({"role": "user", "content": f"вопрос {i} " + "слово " * 60})
        chat.append({"role": "assistant", "content": None,
                     "tool_calls": [{"id": f"c{i}", "type": "function",
                                     "function": {"name": "t", "arguments": "{}"}}]})
        chat.append({"role": "tool", "tool_call_id": f"c{i}", "content": "результат"})
        chat.append({"role": "assistant", "content": f"ответ {i}"})

    messages = chat.transcript(SYSTEM, max_tokens=1200)
    awaiting: set[str] = set()
    for message in messages[1:]:
        if message["role"] == "assistant":
            awaiting = {c["id"] for c in message.get("tool_calls") or []}
        elif message["role"] == "tool":
            assert message["tool_call_id"] in awaiting, "orphaned tool result"
            awaiting.discard(message["tool_call_id"])
        else:
            awaiting = set()


def test_a_short_history_is_untouched():
    """Snapping must not throw away history that fits comfortably."""
    chat = chat_with(3, words=5)
    messages = chat.transcript(SYSTEM, max_tokens=10_000)
    assert len(messages) == 1 + len(chat.messages)


def test_step_one_is_the_old_behaviour():
    chat = chat_with(40)
    snapped = chat.transcript(SYSTEM, max_tokens=1500)
    exact = chat.transcript(SYSTEM, max_tokens=1500, step=1)
    # The snapped one carries the same turns or fewer, never more.
    assert len(snapped) <= len(exact)
    assert len(_split_into_turns(snapped[1:])) <= len(_split_into_turns(exact[1:]))
