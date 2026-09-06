"""The in-process memory tools: the catalogue, the policy and the dispatch.

The point of these is that the model calls them itself, so what is worth testing
is everything between the model naming `memory__remember` and a row existing:
that the tool is offered at all, that policy does not stop to ask a human, and
that the loop runs it here rather than looking for an MCP server named `memory`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agentcore.config import Settings
from agentcore.llm.base import LLMResponse, ToolCall, Usage
from agentcore.localtools import FORGET, REMEMBER, call_local, is_local, memory_tools
from agentcore.loop import AgentLoop
from agentcore.mcp.toolset import build_catalog
from agentcore.memory import ChatMemory
from agentcore.policy import Decision, Policy
from agentcore.store import StoreUnavailable


class FakeStore:
    """Enough AdbStore to answer the two calls these tools make."""

    def __init__(self, *, existed: bool = True, fails: bool = False):
        self.remembered: list[dict] = []
        self.forgotten: list[str] = []
        self._existed = existed
        self._fails = fails

    async def remember(self, *, key, fact, scope="user", chat_id=None):
        if self._fails:
            raise StoreUnavailable("ORDS returned HTTP 503")
        self.remembered.append(
            {"key": key, "fact": fact, "scope": scope, "chat_id": chat_id}
        )

    async def forget(self, key):
        if self._fails:
            raise StoreUnavailable("ORDS returned HTTP 503")
        self.forgotten.append(key)
        return self._existed


# -- what is offered ---------------------------------------------------------


def test_no_memory_tools_without_a_store():
    """A tool that cannot work costs two steps to discover that."""
    assert memory_tools(None) == []


def test_both_memory_tools_are_offered_with_a_store():
    names = [t.qualified_name for t in memory_tools(FakeStore())]
    assert names == [REMEMBER, FORGET]


def test_local_names_are_recognised():
    assert is_local(REMEMBER)
    assert is_local(FORGET)
    assert not is_local("github__create_issue")
    # Not namespaced at all: must not raise on its way to False.
    assert not is_local("remember")


def test_memory_tools_reach_the_catalogue_and_resolve_back():
    catalog = build_catalog(memory_tools(FakeStore()), is_allowed=lambda _n: True)
    assert REMEMBER in catalog.names()
    assert catalog.resolve(REMEMBER) == REMEMBER


def test_a_profile_can_deny_memory_like_any_other_tool():
    catalog = build_catalog(
        memory_tools(FakeStore()),
        is_allowed=lambda name: not name.startswith("memory__"),
    )
    assert catalog.names() == []


# -- policy ------------------------------------------------------------------


def test_remembering_does_not_stop_to_ask_a_human():
    """Approving every act of remembering would defeat the point of it."""
    verdict = Policy(Settings()).evaluate(REMEMBER, {"key": "k", "fact": "f"})
    assert verdict.decision is Decision.ALLOW


def test_an_unknown_namespace_is_still_gated():
    verdict = Policy(Settings()).evaluate("something__else", {})
    assert verdict.decision is Decision.REQUIRE_APPROVAL


# -- dispatch ----------------------------------------------------------------


def test_remember_stores_the_fact_and_where_it_came_from():
    store = FakeStore()
    result = asyncio.run(
        call_local(
            REMEMBER,
            {"key": "git-identity", "fact": "Коммитит как Aaron Shemtov"},
            store=store,
            chat_id=42,
        )
    )
    assert result.ok
    assert store.remembered == [
        {
            "key": "git-identity",
            "fact": "Коммитит как Aaron Shemtov",
            "scope": "user",
            "chat_id": 42,
        }
    ]


@pytest.mark.parametrize(
    "arguments", [{"key": "k"}, {"key": "k", "fact": "   "}, {"fact": "f"}, {}]
)
def test_remember_refuses_incomplete_arguments_without_writing(arguments):
    store = FakeStore()
    result = asyncio.run(call_local(REMEMBER, arguments, store=store))
    assert not result.ok
    assert store.remembered == []


def test_forget_says_whether_there_was_anything_to_forget():
    present = asyncio.run(
        call_local(FORGET, {"key": "k"}, store=FakeStore(existed=True))
    )
    absent = asyncio.run(
        call_local(FORGET, {"key": "k"}, store=FakeStore(existed=False))
    )
    assert present.ok and "забыто" in present.text
    assert absent.ok and "нечего забывать" in absent.text


def test_a_failed_write_comes_back_as_a_failure_rather_than_an_exception():
    """The store raises so a caller cannot claim a fact was kept when it was not.

    Here that has to become something the model reads, or one unreachable
    database would end the whole turn over a single fact.
    """
    result = asyncio.run(
        call_local(REMEMBER, {"key": "k", "fact": "f"}, store=FakeStore(fails=True))
    )
    assert not result.ok
    assert "не сохранён" in result.text


def test_without_a_store_the_call_fails_instead_of_crashing():
    result = asyncio.run(call_local(REMEMBER, {"key": "k", "fact": "f"}, store=None))
    assert not result.ok


# -- through the loop --------------------------------------------------------


class FakeAudit:
    def turn(self, **_kwargs):
        pass

    def tool_call(self, **_kwargs):
        pass

    def denied(self, **_kwargs):
        pass


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def complete(self, **_kwargs):
        return next(self.responses)


class ExplodingPool:
    """Any call here means the loop went looking for an MCP server it should not."""

    async def list_tools(self, **_kwargs):
        return []

    async def call_tool(self, name, _arguments):
        raise AssertionError(f"{name} should have been dispatched in-process")


def _loop(store, responses):
    return AgentLoop(
        profile=SimpleNamespace(
            max_steps=5,
            max_tokens_per_turn=10_000,
            system_prompt="test",
            tool_allowed=lambda _name: True,
        ),
        settings=Settings(max_billable_tokens_per_turn=200_000),
        llm=FakeLLM(responses),
        pool=ExplodingPool(),
        policy=Policy(Settings()),
        store=store,
    )


def test_the_loop_runs_the_memory_tool_itself_and_records_the_chat():
    store = FakeStore()
    loop = _loop(
        store,
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="1",
                        name=REMEMBER,
                        arguments={"key": "deploys", "fact": "Собирает только в CI"},
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=1),
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="запомнил",
                usage=Usage(prompt_tokens=12, completion_tokens=1),
                finish_reason="stop",
            ),
        ],
    )

    result = asyncio.run(
        loop.run(
            chat=ChatMemory(chat_id=77),
            user_text="я собираю только в CI",
            model="test-model",
            audit=FakeAudit(),
        )
    )

    assert result.tool_calls == 1
    assert store.remembered == [
        {
            "key": "deploys",
            "fact": "Собирает только в CI",
            "scope": "user",
            "chat_id": 77,
        }
    ]


def test_the_loop_offers_no_memory_tools_when_memory_is_off():
    catalog = asyncio.run(_loop(None, []).catalog())
    assert catalog.names() == []
