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

from agentcore import metrics
from agentcore.audit import AuditLog, Timer
from agentcore.config import Settings
from agentcore.llm.azure import RAW_OUTPUT_KEY
from agentcore.llm.base import LLMClient, LLMError, ToolCall, Usage
from agentcore.mcp.client import MCPPool
from agentcore.mcp.toolset import ToolCatalog, build_catalog
from agentcore.memory import ChatMemory
from agentcore.policy import Decision, Policy
from agentcore.profiles import AgentProfile
from agentcore.usage_display import append_usage_footer, format_usage_footer

logger = logging.getLogger(__name__)

# Tool output is paid for again as prompt input on every following step. Keep enough
# detail to act on, but do not let a full patch or log consume the whole transcript.
MAX_TOOL_RESULT_CHARS = 12_000

ProgressFn = Callable[[str], Awaitable[None]]
ApproverFn = Callable[[str, dict[str, Any], str], Awaitable[bool]]


@dataclass
class LoopResult:
    text: str
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    stopped_because: str = "completed"
    duration_ms: int = 0
    tool_calls: int = 0
    tool_duration_ms: int = 0


async def _noop_progress(_: str) -> None:
    return None


class ApprovalUnavailable(RuntimeError):
    """There is a human to ask in principle, but not from here.

    Raised instead of answering "no", because the two are different facts and the
    agent reports whichever it is told. The coder profile runs with no UI at all —
    it is called by the lead agent, not by a person — so telling it the user
    pressed Reject would have it report a refusal that never happened.
    """


async def _deny_by_default(_tool: str, _args: dict[str, Any], _reason: str) -> bool:
    """Without a UI able to ask, an approval-required call cannot proceed."""
    raise ApprovalUnavailable


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
        """Input-context budget used when trimming chat history.

        NOT a cumulative per-turn spend cap. Prompt tokens are counted again on every
        step of the loop, so summing them and comparing against this value stopped
        healthy multi-step tasks with a false token-limit message. `max_billable` is
        the limit that answers the cost question.
        """
        return self._profile.max_tokens_per_turn or self._settings.max_tokens_per_turn

    @property
    def max_billable(self) -> int:
        """Spend ceiling for one turn, in tokens that actually cost money.

        Excludes the cached prefix — see Usage.billable. A thirty-step turn at a 95%
        cache hit rate lands around 33k, so this is a backstop against a pathological
        turn (huge tool outputs, a cold cache), not a limit ordinary work should meet.
        """
        return self._settings.max_billable_tokens_per_turn

    def _provider_for(self, model: str) -> str:
        """Provider label for the GenAI metrics.

        Optional on the LLMClient protocol: a client predating it, or a test double,
        simply does not answer, and the metric says "unknown" rather than a turn failing
        over a label.
        """
        resolve = getattr(self._llm, "provider_for", None)
        return resolve(model) if callable(resolve) else "unknown"

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
        extra_system: str = "",
    ) -> LoopResult:
        started = time.monotonic()
        chat.append({"role": "user", "content": user_text})

        # What the agent knows about the person it is talking to, supplied by the
        # caller rather than baked into the profile: the profile is a file in git
        # and these facts are rows in a database that changes between turns.
        system_prompt = self._profile.system_prompt
        if extra_system:
            system_prompt = system_prompt + "\n\n" + extra_system

        catalog = await self.catalog()
        usage = Usage()
        steps = 0
        stopped = "completed"
        answer = ""
        tool_calls = 0
        tool_duration_ms = 0

        while steps < self.max_steps:
            if cancel is not None and cancel.is_set():
                stopped = "cancelled"
                answer = answer or "Отменено."
                break

            # A cost ceiling, measured against max_billable rather than max_tokens:
            # the two limits answer different questions. See both properties.
            if usage.billable > self.max_billable:
                stopped = "token_budget"
                answer = answer or (
                    f"Остановился на лимите бюджета: {usage.billable} оплачиваемых "
                    f"токенов за {steps} шагов при лимите {self.max_billable}. "
                    "Задача оказалась объёмнее ожидаемого. Разбей её на части "
                    "или скажи продолжать — возьмусь за остаток отдельным запросом."
                )
                break

            steps += 1
            messages = chat.transcript(system_prompt, max_tokens=self.max_tokens)

            # Timed apart from the turn on purpose: a slow turn is either a slow model
            # or slow tools, and only these two clocks tell the two apart.
            call_started = time.monotonic()
            provider = self._provider_for(model)
            try:
                response = await self._llm.complete(
                    model=model, messages=messages, tools=catalog.specs or None
                )
            except LLMError as exc:
                logger.exception("llm call failed")
                metrics.record_llm_call(
                    provider=provider, model=model,
                    duration_seconds=time.monotonic() - call_started,
                    prompt_tokens=0, completion_tokens=0,
                    cached_tokens=0, reasoning_tokens=0,
                    error_type=type(exc).__name__,
                )
                stopped = "llm_error"
                answer = f"Ошибка модели: {exc}"
                break

            metrics.record_llm_call(
                provider=provider, model=model,
                duration_seconds=time.monotonic() - call_started,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=response.usage.cached_tokens,
                reasoning_tokens=response.usage.reasoning_tokens,
            )
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
                    # A Responses model returns reasoning items alongside its tool call,
                    # and they have to be replayed on the next request or the model loses
                    # its own chain of thought between steps. Carried verbatim; stripped
                    # again before anything reaches /chat/completions.
                    **(
                        {RAW_OUTPUT_KEY: response.raw_message[RAW_OUTPUT_KEY]}
                        if RAW_OUTPUT_KEY in response.raw_message
                        else {}
                    ),
                }
            )

            # This is the provider's real stop signal. Do not execute a possibly
            # truncated tool call and do not let the model invent a different reason.
            if response.finish_reason == "length":
                stopped = "model_output_limit"
                answer = response.content or (
                    "Модель достигла своего лимита ответа до завершения задачи. "
                    "Продолжи отдельным сообщением."
                )
                break

            if not response.tool_calls:
                answer = response.content or ""
                stopped = "completed"
                break

            if response.content:
                await progress(response.content)

            for call in response.tool_calls:
                result_text, called, duration_ms = await self._execute(
                    call, catalog=catalog, audit=audit, progress=progress, approver=approver
                )
                if called:
                    tool_calls += 1
                    tool_duration_ms += duration_ms
                chat.append(
                    {"role": "tool", "tool_call_id": call.id, "content": _clip(result_text)}
                )
        else:
            stopped = "max_steps"
            answer = (
                f"Остановился после {self.max_steps} шагов: достигнут лимит шагов агента. "
                "Скажи продолжать — продолжу из текущего контекста."
            )

        duration_ms = int((time.monotonic() - started) * 1000)
        answer = append_usage_footer(
            answer,
            format_usage_footer(
                model=model,
                steps=steps,
                usage=usage,
                duration_ms=duration_ms,
                tool_calls=tool_calls,
                tool_duration_ms=tool_duration_ms,
                stopped_because=stopped,
            ),
        )

        audit.turn(
            model=model,
            steps=steps,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            billable_tokens=usage.billable,
            duration_ms=duration_ms,
            stopped_because=stopped,
            tool_calls=tool_calls,
        )
        return LoopResult(
            text=answer,
            steps=steps,
            usage=usage,
            stopped_because=stopped,
            duration_ms=duration_ms,
            tool_calls=tool_calls,
            tool_duration_ms=tool_duration_ms,
        )

    # -- one tool call -------------------------------------------------------

    async def _execute(
        self,
        call: ToolCall,
        *,
        catalog: ToolCatalog,
        audit: AuditLog,
        progress: ProgressFn,
        approver: ApproverFn,
    ) -> tuple[str, bool, int]:
        qualified = catalog.resolve(call.name)
        if qualified is None:
            # The model invented a tool, or called one filtered out by the profile.
            audit.denied(tool=call.name, reason="unknown tool")
            # Also recorded as a tool_call so that every attempt appears exactly once in
            # both the audit log and the metrics. Without it, invented names — the ones
            # most worth counting — would be the only attempts missing from the counter.
            audit.tool_call(
                tool=call.name, arguments=call.arguments, decision="unknown_tool", ok=False
            )
            return (
                f"error: инструмента {call.name} не существует. Доступные: {catalog.names()}",
                False,
                0,
            )

        verdict = self._policy.evaluate(qualified, call.arguments)

        if verdict.decision is Decision.DENY:
            audit.denied(tool=qualified, reason=verdict.reason)
            audit.tool_call(
                tool=qualified, arguments=call.arguments, decision="deny", ok=False
            )
            return (
                f"denied by local policy, not by the remote service: {verdict.reason}",
                False,
                0,
            )

        if verdict.decision is Decision.REQUIRE_APPROVAL:
            await progress(f"⏸ Жду подтверждения: {qualified}")
            # How long a person takes to answer is the one latency here that no amount of
            # engineering shortens, and on an agent that asks before it changes anything
            # it dominates every other number in the turn.
            asked_at = time.monotonic()
            try:
                approved = await approver(qualified, call.arguments, verdict.reason)
            except ApprovalUnavailable:
                metrics.record_approval(
                    agent=audit.agent,
                    tool=qualified,
                    outcome="unavailable",
                    waited_seconds=time.monotonic() - asked_at,
                )
                audit.tool_call(
                    tool=qualified, arguments=call.arguments, decision="unavailable", ok=False
                )
                return (
                    "denied by local policy, not by the remote service: этот вызов "
                    f"требует подтверждения человека ({verdict.reason}), а здесь "
                    "спросить некого — ты вызван другим агентом, кнопку показать "
                    "негде. Повторять бессмысленно. Заверши работу и напиши в ответе, "
                    "что осталось подтвердить: спросит тот агент, у которого есть "
                    "человек.",
                    False,
                    0,
                )
            metrics.record_approval(
                agent=audit.agent,
                tool=qualified,
                outcome="approved" if approved else "rejected",
                waited_seconds=time.monotonic() - asked_at,
            )
            if not approved:
                audit.tool_call(
                    tool=qualified, arguments=call.arguments, decision="rejected", ok=False
                )
                return (
                    "denied by local policy, not by the remote service: пользователь "
                    "нажал «Отклонить». Запрос никуда не отправлялся, ничего не "
                    "изменилось. Это решение пользователя, а не сбой — не предлагай "
                    "повторить тот же вызов с теми же аргументами.",
                    False,
                    0,
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
        return result.for_model(), True, timer.ms


def _serialise_call(call: ToolCall) -> dict[str, Any]:
    """Rebuild the tool_call entry for the transcript.

    Built explicitly rather than echoing the provider's raw message: the raw object
    carries extra fields (refusal, annotations, audio) that some endpoints reject
    when handed straight back.
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
