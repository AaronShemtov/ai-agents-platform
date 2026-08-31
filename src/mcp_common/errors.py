"""Uniform tool-error shaping.

An MCP tool that raises gets wrapped by the SDK into an `UnexpectedToolError` and
reaches the model as a stack-trace-ish blob. That wastes tokens and tells the model
nothing it can act on. Every tool here instead returns a small dict describing what
went wrong, so the agent can decide whether to retry, ask, or give up.
"""

from __future__ import annotations

from typing import Any


class ToolError(Exception):
    """Raised inside a tool when the *caller* did something wrong.

    Carries a hint that is meant to be read by the model, not by a human operator.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


def tool_error(message: str, *, hint: str | None = None, **extra: Any) -> dict[str, Any]:
    """Build the error payload a tool returns instead of raising.

    Keep it flat and short: this goes straight into the model's context.
    """
    payload: dict[str, Any] = {"ok": False, "error": message}
    if hint:
        payload["hint"] = hint
    payload.update(extra)
    return payload
