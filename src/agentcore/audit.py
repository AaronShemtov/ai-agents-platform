"""Structured audit logging.

Every tool call the agent makes is emitted as one JSON line on stdout. In the cluster
that is picked up by Alloy and lands in Loki, so "what did the agent actually do
today" is a Grafana query rather than an archaeology exercise.

Arguments are hashed rather than logged verbatim: they routinely contain whole source
files, and occasionally contain things that should not be sitting in a log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from typing import Any

import structlog

_AUDIT = "audit"


def configure_logging(level: str = "INFO") -> None:
    """JSON logs on stdout, for both stdlib logging and structlog."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def _digest(arguments: dict[str, Any]) -> str:
    try:
        blob = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # any argument shape must still produce a digest
        blob = repr(arguments)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]  # sha256 here is a dedup key, not a security control


class AuditLog:
    def __init__(self, *, agent: str, chat_id: int | str) -> None:
        self._log = structlog.get_logger(_AUDIT).bind(agent=agent, chat_id=str(chat_id))

    def tool_call(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        decision: str,
        ok: bool | None = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        self._log.info(
            "tool_call",
            tool=tool,
            args_sha=_digest(arguments),
            args_keys=sorted(arguments)[:20],
            decision=decision,
            ok=ok,
            duration_ms=duration_ms,
            error=error,
        )

    def turn(
        self,
        *,
        model: str,
        steps: int,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
        stopped_because: str,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        billable_tokens: int = 0,
    ) -> None:
        self._log.info(
            "turn",
            model=model,
            steps=steps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            billable_tokens=billable_tokens,
            cache_hit_pct=round(100 * cached_tokens / prompt_tokens) if prompt_tokens else 0,
            total_tokens=prompt_tokens + completion_tokens,
            duration_ms=duration_ms,
            stopped_because=stopped_because,
        )

    def denied(self, *, tool: str, reason: str) -> None:
        self._log.warning("tool_denied", tool=tool, reason=reason)


class Timer:
    """Millisecond timer for audit records."""

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.ms = int((time.monotonic() - self._start) * 1000)
