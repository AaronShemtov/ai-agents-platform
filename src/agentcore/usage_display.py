"""Compact, provider-neutral usage statistics for user-facing replies.

`cached_tokens` is the number of prompt tokens the provider reports as served from
its prompt cache. Providers do not expose their internal KV-cache allocation, GPU
memory, cache evictions, or similar runtime internals, so those must not be inferred
from this value.
"""

from __future__ import annotations

from agentcore.llm.base import Usage


def format_usage_footer(
    *,
    model: str,
    steps: int,
    usage: Usage,
    duration_ms: int,
    tool_calls: int = 0,
    tool_duration_ms: int = 0,
    stopped_because: str = "completed",
) -> str:
    """Return a compact plain-text footer safe for Telegram HTML rendering."""
    return (
        "📊 AI stats\n"
        f"model={model} · steps={steps} · stop={stopped_because}\n"
        f"input={usage.prompt_tokens:,} · output={usage.completion_tokens:,} · "
        f"cached={usage.cached_tokens:,} ({usage.cache_hit_pct}%)\n"
        f"reasoning={usage.reasoning_tokens:,} · billable={usage.billable:,}\n"
        f"duration={_duration(duration_ms)} · tools={tool_calls} ({_duration(tool_duration_ms)})"
    )


def append_usage_footer(text: str, footer: str) -> str:
    """Append stats to an answer without changing the answer's own formatting."""
    return f"{text.rstrip()}\n\n{footer}" if text.strip() else footer


def _duration(milliseconds: int) -> str:
    if milliseconds < 1_000:
        return f"{milliseconds}ms"
    return f"{milliseconds / 1_000:.1f}s"
