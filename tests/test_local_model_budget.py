"""A locally served model gets a different deal from a hosted one.

Measured on the box before writing any of this: with the lead's 61-tool
catalogue attached, qwen3.5:0.8b never answered at all — Cloudflare returned 524
because prefill outlasts the tunnel's 100s origin timeout. With no tools but the
full 30k transcript it took 19s and invented work it had not done. So the loop
withholds the catalogue and shrinks the transcript, and these tests pin that to
the routing decision rather than to a model name spelled out somewhere.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agentcore.config import Settings
from agentcore.llm.base import LLMResponse, Usage
from agentcore.loop import AgentLoop
from agentcore.mcp.client import RemoteTool
from agentcore.memory import ChatMemory
from agentcore.policy import Decision

LOCAL = "qwen3.5:0.8b"
HOSTED = "gpt-5.6-luna"


class FakeAudit:
    def turn(self, **_kwargs):
        pass

    def tool_call(self, **_kwargs):
        pass

    def denied(self, **_kwargs):
        pass


class RecordingLLM:
    """Answers immediately and remembers what it was handed."""

    def __init__(self):
        self.calls = []

    async def complete(self, *, model, messages, tools=None):
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        return LLMResponse(
            content="ok", usage=Usage(prompt_tokens=10, completion_tokens=1),
            finish_reason="stop",
        )


class FakePool:
    def __init__(self):
        self.listed = 0

    async def list_tools(self, **_kwargs):
        self.listed += 1
        return [
            RemoteTool(
                server="cluster",
                name=f"list_{i}",
                qualified_name=f"cluster__list_{i}",
                description="d",
                input_schema={"type": "object", "properties": {}},
            )
            for i in range(3)
        ]

    async def call_tool(self, name, _arguments):
        raise AssertionError(f"{name} should not have been called")


class FakePolicy:
    def evaluate(self, _name, _arguments):
        return SimpleNamespace(decision=Decision.ALLOW, reason="")


def build(settings=None):
    llm, pool = RecordingLLM(), FakePool()
    loop = AgentLoop(
        profile=SimpleNamespace(
            max_steps=3,
            max_tokens_per_turn=100_000,
            system_prompt="system",
            tool_allowed=lambda _n: True,
        ),
        settings=settings
        or Settings(models_ollama=LOCAL, ollama_base_url="http://box/v1/",
                    max_billable_tokens_per_turn=500_000),
        llm=llm,
        pool=pool,
        policy=FakePolicy(),
    )
    return loop, llm, pool


def run(loop, model, chat=None):
    return asyncio.run(
        loop.run(
            chat=chat or ChatMemory(chat_id=1),
            user_text="привет",
            model=model,
            audit=FakeAudit(),
        )
    )


# -- routing -----------------------------------------------------------------


def test_a_local_model_is_recognised_by_configuration_not_by_name():
    s = Settings(models_ollama=LOCAL, ollama_base_url="http://box/v1/")
    assert s.is_local_model(LOCAL)
    assert not s.is_local_model(HOSTED)
    # Nothing about the string decides it — an unconfigured qwen is not local.
    assert not Settings().is_local_model(LOCAL)


# -- what each model is handed -----------------------------------------------


def test_a_local_model_is_given_no_tools():
    """61 tool definitions cost ~10k tokens and a 0.8B model cannot use them."""
    loop, llm, pool = build()
    run(loop, LOCAL)
    assert llm.calls[0]["tools"] is None
    # And the catalogue is not even fetched, so no MCP round trip is paid for.
    assert pool.listed == 0


def test_a_hosted_model_still_gets_the_catalogue():
    loop, llm, pool = build()
    run(loop, HOSTED)
    assert llm.calls[0]["tools"], "hosted models must keep their tools"
    assert pool.listed == 1


def test_local_tools_can_be_turned_back_on():
    """The withholding is a default, not a law: a bigger local model may earn them."""
    loop, llm, pool = build(
        Settings(models_ollama=LOCAL, ollama_base_url="http://box/v1/",
                 local_tools=True, max_billable_tokens_per_turn=500_000)
    )
    run(loop, LOCAL)
    assert llm.calls[0]["tools"]
    assert pool.listed == 1


# -- how much transcript ------------------------------------------------------


def _long_chat() -> ChatMemory:
    chat = ChatMemory(chat_id=1)
    for i in range(400):
        chat.append({"role": "user", "content": f"вопрос {i} " + "слово " * 40})
        chat.append({"role": "assistant", "content": f"ответ {i} " + "слово " * 40})
    return chat


def test_a_local_model_gets_a_much_shorter_transcript():
    settings = Settings(models_ollama=LOCAL, ollama_base_url="http://box/v1/",
                        max_billable_tokens_per_turn=500_000)
    loop, llm, _ = build(settings)
    run(loop, LOCAL, _long_chat())
    local_messages = len(llm.calls[0]["messages"])

    loop, llm, _ = build(settings)
    run(loop, HOSTED, _long_chat())
    hosted_messages = len(llm.calls[0]["messages"])

    assert local_messages < hosted_messages, (
        f"local got {local_messages} messages, hosted got {hosted_messages}"
    )


def test_the_local_budget_is_configurable():
    tiny = Settings(models_ollama=LOCAL, ollama_base_url="http://box/v1/",
                    local_max_tokens=500, max_billable_tokens_per_turn=500_000)
    roomy = Settings(models_ollama=LOCAL, ollama_base_url="http://box/v1/",
                     local_max_tokens=20_000, max_billable_tokens_per_turn=500_000)

    loop, llm, _ = build(tiny)
    run(loop, LOCAL, _long_chat())
    small = len(llm.calls[0]["messages"])

    loop, llm, _ = build(roomy)
    run(loop, LOCAL, _long_chat())
    large = len(llm.calls[0]["messages"])

    assert small < large


def test_the_most_recent_turn_survives_even_a_tiny_budget():
    """Dropping it would mean answering with no idea what was asked."""
    loop, llm, _ = build(
        Settings(models_ollama=LOCAL, ollama_base_url="http://box/v1/",
                 local_max_tokens=1, max_billable_tokens_per_turn=500_000)
    )
    run(loop, LOCAL, _long_chat())
    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "привет"
