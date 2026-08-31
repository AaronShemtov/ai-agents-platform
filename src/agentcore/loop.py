"""The agent loop: LLM proposes tool calls, we execute them, repeat until an answer.

Approval design — worth reading before changing anything here:

The loop never suspends itself. When policy says a call needs a human, it simply
`await`s an `approver` callback supplied by the UI. The Telegram layer implements
that by sending a message with Approve/Reject buttons and awaiting a Future that the
callback-query handler resolves. So the whole turn stays one coroutine and there is no
serialised "pending state" to resume.

The cost is that a pending approval dies with the process. With one replica and
`strategy: Recreate` that means a rollout mid-approval loses it, and the user has to
ask again. Same limitation the existing SRE agent has; a durable approval store is
phase 2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agentcore.audit import AuditLog, Timer
from agentcore.config import Settings
from agentcore.llm.base import LLMClient, LLMError, ToolCall, Usage
from agentcore.mcp.client import MCPPool
from agentcore.mcp.toolset import ToolCatalog, build_catalog
from agentcore.memory import ChatMemory
from agentcore.policy import Decision, Policy
from agentcore.profiles import AgentProfile

logger = logging.getLogger(__name__)

# A single tool result can be enormous (a whole file, a full pod list). Truncate so one
# greedy call cannot evict the rest of the conversation from the context window.
MAX_TOOL_RESULT_CHARS = 20_000

ProgressFn = Callable[[str], Awaitable[None]]
ApproverFn = Callable[[str, dict[str, Any], str], Awaitable[bool]]


@dataclass
class LoopResult:
    text: str
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    stopped_because: str = "completed"


async def _noop_progress(_: str) -> None:
    return None


async def _deny_by_default(_tool: str, _args: dict[str, Any], _reason: str) -> bool:
    """Without a UI able to ask, an approval-required call is refused."""
    return False


class AgentLoop:
    def __init__(
        self,
        *,
        profile: AgentProfile,
        settings: Settings,
        llm: LLMClient,
        pool: MCPPool,
        policy: Policy,
    ) -> None:
        self._profile = profile
        self._settings = settings
        self._llm = llm
        self._pool = pool
        self._policy = policy

    @property
    def max_steps(self) -> int:
        return self._profile.max_steps or self._settings.max_steps

    @property
    def max_tokens(self) -> int:
        return self._profile.max_tokens_per_turn or self._settings.max_tokens_per_turn

    async def catalog(self, *, force_refresh: bool = False) -> ToolCatalog:
        tools = await self._pool.list_tools(force_refresh=force_refresh)
        return build_catalog(tools, is_allowed=self._profile.tool_allowed)

    async def run(
        self,
        *,
        chat: ChatMemory,
        user_text: str,
        model: str,
        audit: AuditLog,
        progress: ProgressFn = _noop_progress,
        approver: ApproverFn = _deny_by_default,
        cancel: asyncio.Event | None = None,
    ) -> LoopResult:
        started = time.monotonic()
        chat.append({"role": "user", "content": user_text})

        catalog = await self.catalog()
        usage = Usage()
        steps = 0
        stopped = "completed"
        answer = ""

        while steps < self.max_steps:
            if cancel is not None and cancel.is_set():
                stopped = "cancelled"
                answer = answer or "Отменено."
                break

            if usage.total > self.max_tokens:
                stopped = "token_budget"
                answer = answer or (
                    "Достиг лимита токенов на один запрос. Скажи, продолжать ли, "
                    "и я возьмусь за оставшуюся часть отдельно."
                )
                break

            steps += 1
            messages = chat.transcript(self._profile.system_prompt, max_tokens=self.max_tokens)

            try:
                response = await self._llm.complete(
                    model=model, messages=messages, tools=catalog.specs or None
                )
            except LLMError as exc:
                logger.exception("llm call failed")
                stopped = "llm_error"
                answer = f"Ошибка модели: {exc}"
                break

            usage = usage + response.usage

            chat.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    **(
                        {"tool_calls": [_serialise_call(c) for c in response.tool_calls]}
                        if response.tool_calls
                        else {}
                    ),
                }
            )

            if not response.tool_calls:
                answer = response.content or ""
                stopped = "completed"
                break

            if response.content:
                await progress(response.content)

            for call in response.tool_calls:
                result_text = await self._execute(
                    call, catalog=catalog, audit=audit, progress=progress, approver=approver
                )
                chat.append(
                    {"role": "tool", "tool_call_id": call.id, "content": _clip(result_text)}
                )
        else:
            stopped = "max_steps"
            answer = (
                f"Остановился после {self.max_steps} шагов, не дойдя до конца. "
                "Опиши задачу поменьше или уточни, что делать дальше."
            )

        audit.turn(
            model=model,
            steps=steps,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            duration_ms=int((time.monotonic() - started) * 1000),
            stopped_because=stopped,
        )
        return LoopResult(text=answer, steps=steps, usage=usage, stopped_because=stopped)

    # -- one tool call -------------------------------------------------------

    async def _execute(
        self,
        call: ToolCall,
        *,
        catalog: ToolCatalog,
        audit: AuditLog,
        progress: ProgressFn,
        approver: ApproverFn,
    ) -> str:
        qualified = catalog.resolve(call.name)
        if qualified is None:
            # The model invented a tool, or called one filtered out by the profile.
            audit.denied(tool=call.name, reason="unknown tool")
            return f"error: инструмента {call.name} не существует. Доступные: {catalog.names()}"

        verdict = self._policy.evaluate(qualified, call.arguments)

        if verdict.decision is Decision.DENY:
            audit.denied(tool=qualified, reason=verdict.reason)
            audit.tool_call(
                tool=qualified, arguments=call.arguments, decision="deny", ok=False
            )
            return f"denied by local policy, not by the remote service: {verdict.reason}"

        if verdict.decision is Decision.REQUIRE_APPROVAL:
            await progress(f"⏸ Жду подтверждения: {qualified}")
            approved = await approver(qualified, call.arguments, verdict.reason)
            if not approved:
                audit.tool_call(
                    tool=qualified, arguments=call.arguments, decision="rejected", ok=False
                )
                return (
                    "denied by local policy: пользователь нажал «Отклонить». "
                    "Запрос к внешнему сервису НЕ отправлялся, ничего не изменилось. "
                    "Это решение пользователя, а не сбой — не предлагай повторить "
                    "тот же вызов с теми же аргументами."
                )

        await progress(f"🔧 {qualified}")

        with Timer() as timer:
            result = await self._pool.call_tool(qualified, call.arguments)

        audit.tool_call(
            tool=qualified,
            arguments=call.arguments,
            decision=verdict.decision.value,
            ok=result.ok,
            duration_ms=timer.ms,
            error=None if result.ok else result.text[:200],
        )
        return result.for_model()


def _serialise_call(call: ToolCall) -> dict[str, Any]:
    """Rebuild the tool_call entry for the transcript.

    Built explicitly rather than echoing the provider's raw message: the raw object
    carries extra fields (refusal, annotations, audio) that some endpoints reject when
    handed straight back.
    """
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments, ensure_ascii=False),
        },
    }


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    dropped = len(text) - MAX_TOOL_RESULT_CHARS
    return text[:MAX_TOOL_RESULT_CHARS] + f"\n… (обрезано {dropped} символов)"
