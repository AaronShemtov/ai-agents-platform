"""What the metrics record, and what they refuse to record.

The refusals matter most. A Prometheus deployment is killed by unbounded label values,
and one of these labels — the tool name — arrives from the model, which is free to
invent it.

The names and bucket boundaries are the OpenTelemetry GenAI conventions', not ours, so
a few tests pin them: they are the part that silently stops matching a backend built for
the convention if someone improves them.
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


# -- the convention ----------------------------------------------------------


def test_the_metric_names_are_the_conventions_ones() -> None:
    """Dots become underscores and durations gain _seconds; everything else is theirs."""
    body = metrics.render()[0].decode()
    for name in (
        "gen_ai_client_token_usage",
        "gen_ai_client_operation_duration_seconds",
        "gen_ai_invoke_agent_duration_seconds",
        "gen_ai_invoke_agent_inference_calls",
        "gen_ai_invoke_agent_tool_calls",
        "gen_ai_execute_tool_duration_seconds",
    ):
        assert f"# TYPE {name} histogram" in body


def test_the_duration_buckets_are_the_specifications() -> None:
    """Copied, not chosen. Matching them is what makes the histograms portable."""
    assert metrics._DURATION_BUCKETS[:4] == (0.01, 0.02, 0.04, 0.08)
    assert metrics._DURATION_BUCKETS[-1] == 81.92
    assert metrics._TOKEN_BUCKETS[0] == 1
    assert metrics._TOKEN_BUCKETS[-1] == 67108864
    assert metrics._CALL_COUNT_BUCKETS == (1, 2, 4, 8, 16, 32, 64, 128)


def test_a_model_call_records_tokens_by_the_conventions_types() -> None:
    metrics.record_llm_call(
        provider="azure.ai.openai", model="gpt-5.6-sol", duration_seconds=2.5,
        prompt_tokens=16000, completion_tokens=200, cached_tokens=15000, reasoning_tokens=12,
    )
    for token_type, expected in (("input", 16000), ("output", 200),
                                 ("cached", 15000), ("reasoning", 12)):
        assert value(
            "gen_ai_client_token_usage_sum",
            gen_ai_operation_name="chat", gen_ai_provider_name="azure.ai.openai",
            gen_ai_request_model="gpt-5.6-sol", gen_ai_token_type=token_type,
        ) == expected


def test_a_failed_call_is_timed_but_charges_no_tokens() -> None:
    """An error consumed no tokens; recording zeros would flatten every average."""
    metrics.record_llm_call(
        provider="ollama", model="qwen3.5:0.8b", duration_seconds=100.0,
        prompt_tokens=0, completion_tokens=0, cached_tokens=0, reasoning_tokens=0,
        error_type="APIStatusError",
    )
    assert value(
        "gen_ai_client_operation_duration_seconds_count",
        gen_ai_operation_name="chat", gen_ai_provider_name="ollama",
        gen_ai_request_model="qwen3.5:0.8b", error_type="APIStatusError",
    ) == 1
    assert REGISTRY.get_sample_value(
        "gen_ai_client_token_usage_sum",
        {"gen_ai_operation_name": "chat", "gen_ai_provider_name": "ollama",
         "gen_ai_request_model": "qwen3.5:0.8b", "gen_ai_token_type": "input"},
    ) is None


def test_a_turn_records_its_shape_but_not_its_tokens() -> None:
    """Tokens belong to the calls that consumed them; counting them again would double
    every sum taken over the token histogram."""
    metrics.record_turn(agent="lead", model="gpt-5.6-sol", steps=4, tool_calls=3,
                        duration_ms=12000, stopped_because="completed")
    assert value("agent_turns_total", agent="lead", model="gpt-5.6-sol",
                 stopped_because="completed") == 1
    assert value("gen_ai_invoke_agent_inference_calls_sum",
                 gen_ai_agent_name="lead", gen_ai_request_model="gpt-5.6-sol") == 4
    assert value("gen_ai_invoke_agent_tool_calls_sum",
                 gen_ai_agent_name="lead", gen_ai_request_model="gpt-5.6-sol") == 3


def test_running_out_of_steps_is_not_an_error() -> None:
    """It is the limit doing its job. Only a model or transport failure is an error."""
    metrics.record_turn(agent="a1", model="m", steps=30, tool_calls=20,
                        duration_ms=1000, stopped_because="max_steps")
    assert value("gen_ai_invoke_agent_duration_seconds_count",
                 gen_ai_agent_name="a1", gen_ai_request_model="m", error_type="") == 1


def test_a_model_failure_is_an_error() -> None:
    metrics.record_turn(agent="a2", model="m", steps=1, tool_calls=0,
                        duration_ms=1000, stopped_because="llm_error")
    assert value("gen_ai_invoke_agent_duration_seconds_count",
                 gen_ai_agent_name="a2", gen_ai_request_model="m",
                 error_type="llm_error") == 1


# -- our own extensions ------------------------------------------------------


def test_every_attempted_call_is_counted_by_what_policy_decided() -> None:
    for decision in ("allow", "deny", "rejected", "unknown_tool"):
        metrics.record_tool_call(agent="p1", tool="github__x", decision=decision,
                                 ok=None, duration_ms=None)
        assert value("agent_policy_decisions_total", agent="p1", tool="github__x",
                     decision=decision) == 1


def test_a_call_that_never_ran_is_not_timed() -> None:
    """A refusal took no time in the tool; timing it would poison the latency."""
    metrics.record_tool_call(agent="p2", tool="github__y", decision="deny",
                             ok=None, duration_ms=None)
    assert value("gen_ai_execute_tool_duration_seconds_count",
                 gen_ai_agent_name="p2", gen_ai_tool_name="github__y", error_type="") == 0


def test_a_failed_tool_call_is_timed_and_marked() -> None:
    metrics.record_tool_call(agent="p3", tool="cluster__list_pods", decision="allow",
                             ok=False, duration_ms=2000)
    assert value("gen_ai_execute_tool_duration_seconds_count",
                 gen_ai_agent_name="p3", gen_ai_tool_name="cluster__list_pods",
                 error_type="tool_error") == 1


def test_waiting_for_a_human_is_measured() -> None:
    metrics.record_approval(agent="h1", tool="cloudflare__create_dns_record",
                            outcome="approved", waited_seconds=42.0)
    assert value("agent_approvals_total", agent="h1",
                 tool="cloudflare__create_dns_record", outcome="approved") == 1
    assert value("agent_approval_wait_seconds_sum", agent="h1",
                 tool="cloudflare__create_dns_record") == 42.0


# -- what a tool name is allowed to be ---------------------------------------


def test_invented_names_stop_creating_series_once_the_cap_is_reached() -> None:
    for i in range(metrics.MAX_TOOL_LABELS):
        metrics._tool_label(f"tool_{i}")
    assert metrics._tool_label("one_too_many") == "other"


def test_a_name_already_seen_survives_the_cap() -> None:
    metrics._tool_label("cluster__list_pods")
    for i in range(metrics.MAX_TOOL_LABELS * 2):
        metrics._tool_label(f"junk_{i}")
    assert metrics._tool_label("cluster__list_pods") == "cluster__list_pods"


def test_the_server_is_taken_from_the_prefix() -> None:
    assert metrics.server_of("cloudflare__purge_cache") == "cloudflare"
    assert metrics.server_of("just_made_this_up") == "unknown"


# -- exposition --------------------------------------------------------------


def test_nothing_anyone_said_is_in_the_payload() -> None:
    metrics.record_tool_call(agent="x1", tool="github__get_file_contents",
                             decision="allow", ok=True, duration_ms=100)
    body = metrics.render()[0].decode()
    assert "args" not in body
    assert "chat_id" not in body


def test_render_returns_something_prometheus_will_accept() -> None:
    payload, content_type = metrics.render()
    assert "text/plain" in content_type
    assert payload.startswith(b"#")
