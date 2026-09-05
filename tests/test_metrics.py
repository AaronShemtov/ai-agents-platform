"""What the metrics record, and what they refuse to record.

The refusals matter more than the counts here. A Prometheus deployment is destroyed by
unbounded label values, and one of these labels — the tool name — arrives from the
model, which is free to invent it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from prometheus_client import REGISTRY

from agentcore import metrics


@pytest.fixture(autouse=True)
def clean_tool_labels() -> Iterator[None]:
    """The seen-tools set is module state; a test that fills it must not leak."""
    saved = set(metrics._seen_tools)
    yield
    metrics._seen_tools.clear()
    metrics._seen_tools.update(saved)


def value(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# -- what a tool name is allowed to be ---------------------------------------


def test_a_real_tool_name_is_kept() -> None:
    assert metrics._tool_label("github__create_pull_request") == "github__create_pull_request"


def test_invented_names_stop_creating_series_once_the_cap_is_reached() -> None:
    """Otherwise every hallucination is a permanent time series."""
    for i in range(metrics.MAX_TOOL_LABELS):
        metrics._tool_label(f"tool_{i}")
    assert metrics._tool_label("one_too_many") == "other"


def test_a_name_already_seen_survives_the_cap() -> None:
    """The cap must not start discarding tools that are genuinely in use."""
    metrics._tool_label("cluster__list_pods")
    for i in range(metrics.MAX_TOOL_LABELS * 2):
        metrics._tool_label(f"junk_{i}")
    assert metrics._tool_label("cluster__list_pods") == "cluster__list_pods"


def test_the_server_is_taken_from_the_prefix() -> None:
    assert metrics.server_of("cloudflare__purge_cache") == "cloudflare"
    assert metrics.server_of("coder__ask") == "coder"


def test_an_unqualified_name_does_not_become_its_own_server() -> None:
    """An invented name has no prefix; it must not mint a server label."""
    assert metrics.server_of("just_made_this_up") == "unknown"


# -- counting ----------------------------------------------------------------


def test_a_successful_call_is_counted_and_timed() -> None:
    before = value("agent_tool_calls_total", agent="t1", tool="github__x",
                   decision="allow", ok="true")
    metrics.record_tool_call(agent="t1", tool="github__x", decision="allow",
                             ok=True, duration_ms=1500)
    after = value("agent_tool_calls_total", agent="t1", tool="github__x",
                  decision="allow", ok="true")
    assert after == before + 1
    assert value("agent_tool_duration_seconds_sum", agent="t1", server="github") >= 1.5


def test_a_denied_call_reports_no_outcome_rather_than_failure() -> None:
    """`ok=false` would read as the tool failing; nothing was sent at all."""
    metrics.record_tool_call(agent="t2", tool="github__y", decision="deny",
                             ok=None, duration_ms=None)
    assert value("agent_tool_calls_total", agent="t2", tool="github__y",
                 decision="deny", ok="n/a") == 1
    assert value("agent_tool_calls_total", agent="t2", tool="github__y",
                 decision="deny", ok="false") == 0


def test_a_call_that_never_ran_is_not_timed() -> None:
    metrics.record_tool_call(agent="t3", tool="github__z", decision="deny",
                             ok=None, duration_ms=None)
    assert value("agent_tool_duration_seconds_count", agent="t3", server="github") == 0


def test_a_turn_records_its_tokens_by_kind() -> None:
    metrics.record_turn(
        agent="t4", model="gpt-5.6-sol", steps=4, duration_ms=8000,
        stopped_because="completed", prompt_tokens=16000, completion_tokens=200,
        cached_tokens=15000, reasoning_tokens=12, billable_tokens=1200,
    )
    assert value("agent_turns_total", agent="t4", model="gpt-5.6-sol",
                 stopped_because="completed") == 1
    assert value("agent_tokens_total", agent="t4", model="gpt-5.6-sol", kind="billable") == 1200
    assert value("agent_tokens_total", agent="t4", model="gpt-5.6-sol", kind="cached") == 15000
    assert value("agent_turn_steps_sum", agent="t4", model="gpt-5.6-sol") == 4


def test_a_zero_token_kind_creates_no_series() -> None:
    """A model that never reasons should not carry an always-zero reasoning series."""
    metrics.record_turn(
        agent="t5", model="qwen3.5:0.8b", steps=1, duration_ms=3000,
        stopped_because="completed", prompt_tokens=300, completion_tokens=30,
        cached_tokens=0, reasoning_tokens=0, billable_tokens=330,
    )
    assert REGISTRY.get_sample_value(
        "agent_tokens_total",
        {"agent": "t5", "model": "qwen3.5:0.8b", "kind": "reasoning"},
    ) is None


# -- exposition --------------------------------------------------------------


def test_render_returns_something_prometheus_will_accept() -> None:
    metrics.record_tool_call(agent="t6", tool="cluster__list_pods", decision="allow",
                             ok=True, duration_ms=40)
    payload, content_type = metrics.render()
    assert "text/plain" in content_type
    body = payload.decode()
    assert "agent_tool_calls_total" in body
    assert "# TYPE agent_tool_duration_seconds histogram" in body


def test_nothing_anyone_said_is_in_the_payload() -> None:
    """Arguments are hashed in the audit log and never reach this module at all."""
    metrics.record_tool_call(agent="t7", tool="github__get_file_contents",
                             decision="allow", ok=True, duration_ms=100)
    body = metrics.render()[0].decode()
    assert "args" not in body
    assert "chat_id" not in body
