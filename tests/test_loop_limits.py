import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agentcore.config import Settings
from agentcore.llm.base import LLMResponse, ToolCall, Usage
from agentcore.loop import AgentLoop, MAX_TOOL_RESULT_CHARS, _clip
from agentcore.mcp.client import ToolResult
from agentcore.mcp.toolset import ToolCatalog
from agentcore.memory import ChatMemory
from agentcore.policy import Decision


class FakeAudit:
    def __init__(self):
        self.stop_reason = None

    def turn(self, **kwargs):
        self.stop_reason = kwargs["stopped_because"]

    def tool_call(self, **_kwargs):
        pass

    def denied(self, **_kwargs):
        pass


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def complete(self, **_kwargs):
        self.calls += 1
        return next(self.responses)


class FakePool:
    async def call_tool(self, _name, _arguments):
        return ToolResult(ok=True, text="ok")


class FakePolicy:
    def evaluate(self, _name, _arguments):
        return SimpleNamespace(decision=Decision.ALLOW, reason="")


def make_loop(responses, *, max_tokens=10, max_steps=5):
    profile = SimpleNamespace(
        max_steps=max_steps,
        max_tokens_per_turn=max_tokens,
        system_prompt="test",
        tool_allowed=lambda _name: True,
    )
    loop = AgentLoop(
        profile=profile,
        settings=Settings(max_steps=max_steps, max_tokens_per_turn=max_tokens),
        llm=FakeLLM(responses),
        pool=FakePool(),
        policy=FakePolicy(),
    )
    catalog = ToolCatalog(
        specs=[{"type": "function", "function": {"name": "test_tool"}}],
        _by_openai_name={"test_tool": "test__tool"},
        skipped=[],
    )
    loop.catalog = AsyncMock(return_value=catalog)
    return loop


def run(loop):
    audit = FakeAudit()
    result = asyncio.run(
        loop.run(
            chat=ChatMemory(chat_id=1),
            user_text="do it",
            model="test-model",
            audit=audit,
        )
    )
    return result, audit


def test_cumulative_usage_does_not_stop_a_multistep_turn():
    loop = make_loop(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                usage=Usage(prompt_tokens=100, completion_tokens=20),
                finish_reason="tool_calls",
            ),
            LLMResponse(
                content="done",
                usage=Usage(prompt_tokens=120, completion_tokens=5),
                finish_reason="stop",
            ),
        ],
        max_tokens=10,
    )

    result, audit = run(loop)

    assert result.text == "done"
    assert result.steps == 2
    assert result.stopped_because == "completed"
    assert audit.stop_reason == "completed"


def test_provider_length_finish_reason_is_reported_exactly():
    loop = make_loop(
        [
            LLMResponse(
                content=None,
                usage=Usage(prompt_tokens=10, completion_tokens=10),
                finish_reason="length",
            )
        ]
    )

    result, audit = run(loop)

    assert result.stopped_because == "model_output_limit"
    assert "лимита ответа" in result.text
    assert audit.stop_reason == "model_output_limit"


def test_max_steps_has_a_neutral_message():
    calls = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id=str(i), name="test_tool", arguments={})],
            finish_reason="tool_calls",
        )
        for i in range(2)
    ]
    loop = make_loop(calls, max_steps=2)

    result, audit = run(loop)

    assert result.stopped_because == "max_steps"
    assert "лимит шагов агента" in result.text
    assert "лимит токенов" not in result.text
    assert audit.stop_reason == "max_steps"


def test_tool_result_is_clipped_with_explicit_count():
    text = "x" * (MAX_TOOL_RESULT_CHARS + 37)

    clipped = _clip(text)

    assert len(clipped) < len(text)
    assert clipped.startswith("x" * MAX_TOOL_RESULT_CHARS)
    assert "обрезано 37 символов" in clipped
