from agentcore.llm.base import Usage
from agentcore.ui.usage import append_usage_footer, format_usage_footer


def test_footer_reports_provider_and_agent_metrics() -> None:
    footer = format_usage_footer(
        model="gpt-test",
        steps=3,
        usage=Usage(
            prompt_tokens=10_000,
            completion_tokens=500,
            cached_tokens=8_000,
            reasoning_tokens=125,
        ),
        duration_ms=2_450,
        tool_calls=2,
        tool_duration_ms=310,
        stopped_because="completed",
    )

    assert "model=gpt-test" in footer
    assert "steps=3" in footer
    assert "input=10,000" in footer
    assert "output=500" in footer
    assert "cached=8,000 (80%)" in footer
    assert "reasoning=125" in footer
    assert "billable=2,500" in footer
    assert "duration=2.5s" in footer
    assert "tools=2 (310ms)" in footer
    assert "KV" not in footer


def test_footer_handles_empty_usage() -> None:
    footer = format_usage_footer(model="m", steps=0, usage=Usage(), duration_ms=0)
    assert "cached=0 (0%)" in footer
    assert "duration=0ms" in footer


def test_footer_is_appended_after_the_answer() -> None:
    result = append_usage_footer("answer\n", "stats")
    assert result == "answer\n\nstats"
