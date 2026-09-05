"""Credentials must not survive a trip through the log handler.

This file exists because it already went wrong. httpx logs every request at INFO as
"HTTP Request: POST https://api.telegram.org/bot<TOKEN>/getUpdates". The bot long-polls,
so that line was written about once a second, went to Loki, and Loki answers
unauthenticated queries through a deliberately public Grafana. The bot token was
world-readable, and a query from outside the cluster with no credentials returned it.

The tests are written against shapes rather than any real value, so they keep working
after the secrets are rotated — which they had to be.
"""

from __future__ import annotations

import logging

import pytest

from agentcore.audit import SecretRedactionFilter, configure_logging


def scrub(message: str) -> str:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, message, (), None)
    SecretRedactionFilter().filter(record)
    return record.getMessage()


# -- the one that actually leaked --------------------------------------------


def test_a_telegram_token_does_not_survive() -> None:
    leaked = (
        "HTTP Request: POST "
        "https://api.telegram.org/bot1234567890:AAFakeTokenValueForTestsOnly_xyz/getUpdates "
        '"HTTP/1.1 200 OK"'
    )
    out = scrub(leaked)
    assert "AAFakeTokenValueForTestsOnly_xyz" not in out
    assert "1234567890" not in out
    # The line stays useful: you can still see which call it was.
    assert "api.telegram.org/bot<REDACTED>/getUpdates" in out
    assert "200 OK" in out


def test_a_bare_bot_token_anywhere_in_a_line_is_caught() -> None:
    """Not every mention arrives inside a URL."""
    out = scrub("using bot1234567890:AAFakeTokenValueForTestsOnly_xyz to poll")
    assert "AAFakeTokenValueForTestsOnly_xyz" not in out


# -- everything else that could leak the same way -----------------------------


@pytest.mark.parametrize(
    "secret",
    [
        "ghp_0123456789abcdefghijklmnopqrstuvwxyz",
        "github_pat_11ABCDEFG0123456789_abcdefghijklmnop",
    ],
)
def test_github_tokens_do_not_survive(secret: str) -> None:
    assert secret not in scrub(f"cloning with {secret} now")


def test_a_bearer_header_keeps_its_name_and_loses_its_value() -> None:
    out = scrub('headers={"Authorization": "Bearer sk-abcdefghijklmnopqrstuvwxyz012345"}')
    assert "sk-abcdefghijklmnopqrstuvwxyz012345" not in out
    assert "Authorization" in out


def test_the_ollama_shared_secret_does_not_survive() -> None:
    out = scrub('headers={"X-Agent-Key": "0123456789abcdef0123456789abcdef"}')
    assert "0123456789abcdef0123456789abcdef" not in out


def test_a_password_in_a_url_loses_only_the_password() -> None:
    """The host is what makes the line worth keeping."""
    out = scrub("connecting to https://ADMIN:hunter2hunter2@db.example.com/ords/admin")
    assert "hunter2hunter2" not in out
    assert "db.example.com" in out
    assert "ADMIN" in out


# -- the filter must never make things worse ----------------------------------


def test_an_ordinary_line_is_left_exactly_alone() -> None:
    line = "tool catalog: 59 allowed, 1 filtered by profile"
    assert scrub(line) == line


def test_a_broken_format_string_does_not_lose_the_record() -> None:
    """A filter that raises would drop the line entirely, which is worse than a leak
    it was not going to catch anyway."""
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "%s %s", ("only-one",), None)
    assert SecretRedactionFilter().filter(record) is True


# -- and the source is silenced too -------------------------------------------


def test_the_clients_that_log_full_urls_are_quietened() -> None:
    """The filter is the second line of defence. The first is not logging it at all."""
    configure_logging("INFO")
    for noisy in ("httpx", "httpcore", "openai"):
        assert logging.getLogger(noisy).level >= logging.WARNING


def test_the_filter_is_installed_on_the_handler_not_the_logger() -> None:
    """A filter on the root logger never sees records propagating up from httpx —
    logger filters apply only to records logged through that logger. Handler filters
    see everything the handler writes, which is what is wanted."""
    configure_logging("INFO")
    handlers = logging.getLogger().handlers
    assert handlers, "root logger has no handler to filter on"
    assert any(
        isinstance(f, SecretRedactionFilter) for h in handlers for f in h.filters
    )
