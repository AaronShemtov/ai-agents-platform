"""Names in AI stats describe actual per-turn executions, not proposals."""

import html
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentcore.config import Settings
from agentcore.llm.base import LLMResponse, ToolCall, Usage
from agentcore.loop import AgentLoop, LoopResult
from agentcore.mcp.client import ToolResult
from agentcore.mcp.toolset import ToolCatalog
from agentcore.memory import ChatMemory
from agentcore.policy import Decision
from agentcore.ui.telegram import MAX_MESSAGE, TelegramUI
from agentcore.ui.usage import format_usage_footer as ui_footer
from agentcore.usage_display import format_usage_footer

FORMATTERS = [format_usage_footer, ui_footer]


def names_from_footer(text):
    return json.loads(text.splitlines()[-1].split(") ", 1)[1])


@pytest.mark.parametrize("formatter", FORMATTERS)
def test_empty_names_and_legacy_keywords(formatter):
    footer = formatter(model="m", steps=3, usage=Usage(), duration_ms=10)
    assert "tools=0 (0ms) []" in footer
    assert "duration=10ms" in footer
    assert names_from_footer(footer) == []
    legacy = formatter(model="m", steps=3, usage=Usage(), duration_ms=10, tool_calls=2)
    assert "tools=2 (0ms) []" in legacy


@pytest.mark.parametrize("formatter", FORMATTERS)
def test_order_duplicates_and_json_escaping(formatter):
    names = ["github__read", "github__read", 'a"b\\c\n', "память__запомнить", "a<b>&c"]
    footer = formatter(
        model="m", steps=2, usage=Usage(), duration_ms=2450,
        tool_calls=len(names), tool_duration_ms=310, tool_names=names,
    )
    assert "duration=2.5s · tools=5 (310ms)" in footer
    assert names_from_footer(footer) == names
    assert "память__запомнить" in footer


@pytest.mark.parametrize("formatter", FORMATTERS)
def test_long_list_is_not_silently_truncated(formatter):
    names = tuple(f"github__read_{i:04d}" for i in range(300))
    footer = formatter(
        model="m", steps=2, usage=Usage(), duration_ms=0,
        tool_calls=len(names), tool_names=names,
    )
    assert len(footer) > 4096
    assert names_from_footer(footer) == list(names)


@pytest.mark.asyncio
async def test_telegram_sends_complete_long_array_with_html_escaping():
    names = [f"test__read_{i:04d}<tag>&" for i in range(300)]
    footer = format_usage_footer(
        model="m", steps=2, usage=Usage(), duration_ms=0,
        tool_calls=len(names), tool_names=names,
    )
    reply = AsyncMock()
    update = SimpleNamespace(effective_message=SimpleNamespace(reply_text=reply))
    await TelegramUI._send_long(None, update, footer)
    assert reply.await_count > 1
    decoded = []
    for call in reply.await_args_list:
        rendered = call.args[0]
        assert "<tag>" not in rendered
        chunk = html.unescape(rendered.removeprefix("<pre>").removesuffix("</pre>"))
        assert len(chunk) <= MAX_MESSAGE
        decoded.append(chunk)
    reconstructed = "".join(decoded)
    assert reconstructed == footer
    assert names_from_footer(reconstructed) == names


class FakeAudit:
    agent = "test"

    def turn(self, **_kwargs):
        pass

    def tool_call(self, **_kwargs):
        pass

    def denied(self, **_kwargs):
        pass


def make_loop(aliases, *, mapping=None, decision=Decision.ALLOW, failure=False):
    mapping = mapping if mapping is not None else {"alias": "test__tool"}
    response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=str(i), name=name, arguments={"secret": "never-in-stats"})
                    for i, name in enumerate(aliases)],
        finish_reason="tool_calls",
    )
    llm = SimpleNamespace(complete=AsyncMock(side_effect=[response, LLMResponse(content="done")]))
    pool = SimpleNamespace(call_tool=AsyncMock(return_value=ToolResult(ok=not failure, text="private-result")))
    policy = SimpleNamespace(evaluate=lambda _name, _args: SimpleNamespace(decision=decision, reason="test"))
    loop = AgentLoop(
        profile=SimpleNamespace(max_steps=5, max_tokens_per_turn=10000, system_prompt="test", tool_allowed=lambda _: True),
        settings=Settings(), llm=llm, pool=pool, policy=policy,
    )
    loop.catalog = AsyncMock(return_value=ToolCatalog(
        specs=[{"type": "function", "function": {"name": name}} for name in mapping],
        _by_openai_name=mapping, skipped=[],
    ))
    return loop, pool, llm


async def run(loop, **kwargs):
    return await loop.run(chat=ChatMemory(chat_id=1), user_text="x", model="m", audit=FakeAudit(), **kwargs)


@pytest.mark.asyncio
async def test_qualified_names_local_remote_delegation_and_duplicates(monkeypatch):
    mapping = {"alias": "github__read", "local": "memory__remember", "delegate": "coder__ask"}
    loop, pool, _llm = make_loop(["alias", "local", "delegate", "alias"], mapping=mapping)
    local = AsyncMock(return_value=ToolResult(ok=True, text="private-local-result"))
    monkeypatch.setattr("agentcore.loop.call_local", local)
    result = await run(loop)
    expected = ["github__read", "memory__remember", "coder__ask", "github__read"]
    assert result.tool_names == expected
    assert result.tool_calls == len(expected)
    assert names_from_footer(result.text) == expected
    assert pool.call_tool.await_count == 3
    assert local.await_count == 1
    assert "never-in-stats" not in result.text
    assert "private-result" not in result.text
    assert "private-local-result" not in result.text


@pytest.mark.asyncio
async def test_unknown_calls_are_not_executions():
    loop, pool, _llm = make_loop(["unknown", "alias"])
    result = await run(loop)
    assert result.tool_names == ["test__tool"]
    assert result.tool_calls == 1
    assert pool.call_tool.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["denied", "rejected", "unavailable"])
async def test_nonexecuted_calls_are_excluded(mode):
    decision = Decision.DENY if mode == "denied" else Decision.REQUIRE_APPROVAL
    loop, pool, _llm = make_loop(["alias"], decision=decision)
    kwargs = {"approver": AsyncMock(return_value=False)} if mode == "rejected" else {}
    result = await run(loop, **kwargs)
    assert result.tool_calls == 0
    assert result.tool_names == []
    assert names_from_footer(result.text) == []
    pool.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_executed_failure_is_counted():
    loop, pool, _llm = make_loop(["alias"], failure=True)
    result = await run(loop)
    assert result.tool_calls == 1
    assert result.tool_names == ["test__tool"]
    assert names_from_footer(result.text) == ["test__tool"]
    pool.call_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_names_do_not_leak_across_turns():
    loop, _pool, llm = make_loop(["alias"])
    first = await run(loop)
    llm.complete.side_effect = [LLMResponse(content="next answer")]
    second = await run(loop)
    assert first.tool_names == ["test__tool"]
    assert second.tool_names == []
    assert second.tool_calls == 0
    assert names_from_footer(second.text) == []


def test_result_lists_are_not_shared():
    first = LoopResult(text="a")
    second = LoopResult(text="b")
    first.tool_names.append("test__tool")
    assert second.tool_names == []
