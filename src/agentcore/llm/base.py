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

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_pct(self) -> int:
        return round(100 * self.cached_tokens / self.prompt_tokens) if self.prompt_tokens else 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
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
