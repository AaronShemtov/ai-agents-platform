"""What an agent is told when a call needs a human and there is none.

The coder profile runs with no UI: it is invoked by the lead agent over MCP, so
there is no chat to put a button in. Before this, such a call came back saying the
user had pressed Reject — and the coder duly reported a refusal that never
happened, about a question nobody was asked.
"""

import asyncio
from types import SimpleNamespace

from unittest.mock import AsyncMock

from agentcore.config import Settings
from agentcore.llm.base import LLMResponse, ToolCall, Usage
from agentcore.loop import AgentLoop, ApprovalUnavailable, _deny_by_default
from agentcore.mcp.client import ToolResult
from agentcore.mcp.toolset import ToolCatalog
from agentcore.memory import ChatMemory
from agentcore.policy import Decision


class FakeAudit:
    agent = "coder"

    def turn(self, **_kwargs):
        pass

    def tool_call(self, **kwargs):
        self.last_decision = kwargs.get("decision")

    def denied(self, **_kwargs):
        pass


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def complete(self, **_kwargs):
        return next(self.responses)


class FakePool:
    def __init__(self):
        self.called = False

    async def call_tool(self, _name, _arguments):
        self.called = True
        return ToolResult(ok=True, text="merged")


class NeedsApproval:
    def evaluate(self, _name, _arguments):
        return SimpleNamespace(
            decision=Decision.REQUIRE_APPROVAL,
            reason="слияние PR в personal-k8s: Flux применит эту ветку к живому кластеру",
        )


def run_with_no_approver():
    pool = FakePool()
    audit = FakeAudit()
    loop = AgentLoop(
        profile=SimpleNamespace(
            max_steps=5,
            max_tokens_per_turn=10_000,
            system_prompt="test",
            tool_allowed=lambda _name: True,
        ),
        settings=Settings(max_steps=5, max_tokens_per_turn=10_000),
        llm=FakeLLM(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                    usage=Usage(prompt_tokens=10, completion_tokens=1),
                    finish_reason="tool_calls",
                ),
                LLMResponse(
                    content="не смог слить, нужен человек",
                    usage=Usage(prompt_tokens=12, completion_tokens=2),
                    finish_reason="stop",
                ),
            ]
        ),
        pool=pool,
        policy=NeedsApproval(),
    )
    loop.catalog = AsyncMock(
        return_value=ToolCatalog(
            specs=[{"type": "function", "function": {"name": "test_tool"}}],
            _by_openai_name={"test_tool": "test__tool"},
            skipped=[],
        )
    )
    chat = ChatMemory(chat_id=1)
    asyncio.run(loop.run(chat=chat, user_text="merge it", model="m", audit=audit))
    tool_messages = [m for m in chat.messages if m.get("role") == "tool"]
    return tool_messages, pool, audit


def test_the_default_approver_says_it_cannot_ask_rather_than_saying_no():
    """The distinction the whole change rests on."""
    with_no_ui = _deny_by_default("t", {}, "r")
    try:
        asyncio.run(with_no_ui)
    except ApprovalUnavailable:
        return
    raise AssertionError("_deny_by_default answered instead of declaring itself unable")


def test_the_agent_is_told_nobody_could_be_asked():
    tool_messages, _pool, _audit = run_with_no_approver()
    assert tool_messages, "the loop returned no tool result at all"
    text = tool_messages[0]["content"]
    assert "спросить некого" in text
    # And specifically not the thing that was wrong before.
    assert "Отклонить" not in text
    assert "пользователь" not in text


def test_the_call_is_not_made():
    """An unanswerable approval must stop the call, not fall through to it."""
    _messages, pool, _audit = run_with_no_approver()
    assert not pool.called, "the tool ran despite needing an approval nobody gave"


def test_it_is_recorded_as_unavailable_not_as_a_rejection():
    """A rejection is a person exercising judgement and worth counting as such.
    Folding these in with it would make the approval numbers describe a human who
    was never there."""
    _messages, _pool, audit = run_with_no_approver()
    assert audit.last_decision == "unavailable"


def test_the_message_tells_it_not_to_retry():
    """Without this it burns its whole step budget on the same blocked call."""
    tool_messages, _pool, _audit = run_with_no_approver()
    assert "Повторять бессмысленно" in tool_messages[0]["content"]
