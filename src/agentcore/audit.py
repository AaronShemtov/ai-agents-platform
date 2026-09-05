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
import re
import sys
import time
from typing import Any

import structlog

from agentcore import metrics

_AUDIT = "audit"


# Anything shaped like a credential, redacted before a line can be written.
#
# This exists because it already went wrong. httpx logs every request at INFO as
# "HTTP Request: POST https://api.telegram.org/bot<TOKEN>/getUpdates" — the bot long-polls,
# so that line was written about once a second, went to Loki, and Loki is reachable
# through a Grafana that anonymous users can query. The bot token was world-readable.
#
# Two independent measures, because one was clearly not enough: the noisy loggers are
# silenced at source, and this filter catches whatever else ever puts a secret in a
# message. Patterns are deliberately shape-based rather than value-based — a filter that
# had to be told the current secrets would miss the next one.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Telegram: the token sits in the path, its halves split by a colon. The prefix is
    # kept so the line still says which call it was.
    (re.compile(r"(api\.telegram\.org/bot)\d+:[A-Za-z0-9_-]+"), r"\1<REDACTED>"),
    (re.compile(r"\b(bot)\d{6,}:[A-Za-z0-9_-]{20,}"), r"\1<REDACTED>"),
    # GitHub personal access tokens, classic and fine-grained.
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "<REDACTED>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "<REDACTED>"),
    # Bearer and api-key headers, however they happen to be spelled.
    (
        re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?bearer\s+)\S+"),
        r"\1<REDACTED>",
    ),
    (
        re.compile(
            r"(?i)((?:api[-_]?key|x-agent-key)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"
        ),
        r"\1<REDACTED>",
    ),
    # Credentials inside a URL: https://user:secret@host — user and host are kept.
    (re.compile(r"(://[^/\s:@]+:)[^/\s@]+(@)"), r"\1<REDACTED>\2"),
)


class SecretRedactionFilter(logging.Filter):
    """Rewrite a record's message so no credential reaches a handler.

    Installed on the handlers rather than on the root logger on purpose. A filter
    attached to a logger only sees records logged through that logger — records
    propagating up from httpx or httpcore never pass it. A handler's filters see
    everything that handler writes, which is the property actually wanted here.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # a broken format string must not lose the line entirely
            return True
        cleaned = message
        for pattern, replacement in _REDACTIONS:
            cleaned = pattern.sub(replacement, cleaned)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def configure_logging(level: str = "INFO") -> None:
    """JSON logs on stdout, for both stdlib logging and structlog."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # First measure: the source. httpx and httpcore log the full request URL at INFO,
    # which is where the Telegram token was leaking from. Nothing of value is lost —
    # every call the agent makes is already recorded here with its outcome.
    for noisy in ("httpx", "httpcore", "httpx2", "telegram.request", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Second measure: everything else, forever. On the handlers, not the root logger.
    redaction = SecretRedactionFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(redaction)

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
        self._agent = agent
        self._log = structlog.get_logger(_AUDIT).bind(agent=agent, chat_id=str(chat_id))

    @property
    def agent(self) -> str:
        """Which agent this log belongs to, for callers recording their own metrics."""
        return self._agent

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
        metrics.record_tool_call(
            agent=self._agent, tool=tool, decision=decision, ok=ok, duration_ms=duration_ms
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
        tool_calls: int = 0,
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
        # Tokens are deliberately not passed on: they are recorded against the model
        # calls that consumed them, and counting them again per turn would double any
        # sum taken over the token histogram.
        metrics.record_turn(
            agent=self._agent,
            model=model,
            steps=steps,
            tool_calls=tool_calls,
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
