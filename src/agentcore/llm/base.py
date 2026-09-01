"""Provider-agnostic LLM interface.

Deliberately thin. The point is that swapping Azure for Anthropic or Gemini later
touches one file, not the agent loop — not to build an abstraction layer with its own
concepts. If a provider needs something this protocol cannot express, widen it then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str  # the OpenAI-side (sanitised) name; resolve via ToolCatalog
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Subset of prompt_tokens the provider served from its prompt cache. Worth
    # tracking: the system prompt plus the tool definitions form a stable prefix that
    # is re-sent on every step of the loop, so from the second step onward most of the
    # input should be cache hits. A hit rate near zero means something is perturbing
    # the prefix and every step is being paid for in full.
    cached_tokens: int = 0
    # Hidden thinking, billed as output. Measured 2026-09-01 on this resource: the
    # gpt-5.6 family spends ~10-14 on a routine tool-selection step, gpt-5-mini ~128.
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def billable(self) -> int:
        """Tokens that carry real cost, for budgeting a turn.

        `total` is the wrong measure for a per-turn budget. The system prompt and the
        tool definitions — some 12k tokens — are re-sent on every step of the loop, and
        at a 95% cache hit rate they are charged at roughly a tenth of the input rate.
        Counting them at full weight made the budget trip after nine or ten steps while
        max_steps allowed thirty, which read to the user as a mysterious "token limit".
        """
        return max(0, self.prompt_tokens - self.cached_tokens) + self.completion_tokens

    @property
    def cache_hit_pct(self) -> int:
        return round(100 * self.cached_tokens / self.prompt_tokens) if self.prompt_tokens else 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    # The assistant message exactly as the provider returned it, to be appended to the
    # transcript verbatim. Providers are picky about the shape of tool-call turns.
    raw_message: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    pass


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...
