"""Token accounting.

Two things are pinned here. `Usage.__add__` builds a new Usage positionally, so a field
inserted in the wrong order would silently swap counters — the kind of bug that only
shows up as puzzling numbers on a Grafana panel weeks later.

The other is the per-turn budget, which shipped wrong. It compared `total` — every
token of every request — against the limit, but the system prompt and the 58 tool
definitions, some 12k tokens, are re-sent on every step and mostly served from cache at
a fraction of the price. A turn hit the ceiling after nine or ten steps while max_steps
allowed thirty, and the user saw an unexplained "token limit" message on ordinary work.
"""

from __future__ import annotations

from agentcore.llm.base import Usage

# One step of a real turn, measured on the live agent: ~12k prompt of which ~95% is the
# cached system-prompt-plus-tools prefix, and a few hundred output tokens.
STEP = Usage(prompt_tokens=12_000, completion_tokens=500, cached_tokens=11_400)


def test_total_counts_everything_sent() -> None:
    usage = Usage(prompt_tokens=1000, completion_tokens=50, cached_tokens=900)
    assert usage.total == 1050


def test_billable_excludes_the_cached_prefix() -> None:
    usage = Usage(prompt_tokens=1000, completion_tokens=50, cached_tokens=900)
    assert usage.billable == 150


def test_billable_never_goes_negative_if_a_provider_reports_oddly() -> None:
    assert Usage(prompt_tokens=100, cached_tokens=500).billable == 0


def test_a_thirty_step_turn_stays_within_budget_on_billable_but_not_on_total() -> None:
    """The regression. Same turn, two measures, opposite verdicts."""
    limit = 120_000
    turn = Usage()
    for _ in range(30):
        turn = turn + STEP

    assert turn.total > limit, "total blows the budget on an ordinary 30-step turn"
    assert turn.billable < limit, "billable leaves room, which is the point"


def test_the_old_measure_tripped_around_step_ten() -> None:
    """Why it looked like a mystery: the ceiling arrived a third of the way in."""
    limit = 120_000
    turn = Usage()
    tripped_at = None
    for step in range(1, 31):
        turn = turn + STEP
        if tripped_at is None and turn.total > limit:
            tripped_at = step
    assert tripped_at is not None
    assert 8 <= tripped_at <= 11


def test_addition_keeps_each_counter_in_its_own_field() -> None:
    total = Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40)
    assert (
        total.prompt_tokens,
        total.completion_tokens,
        total.cached_tokens,
        total.reasoning_tokens,
    ) == (11, 22, 33, 44)


def test_reasoning_tokens_default_to_zero_for_providers_that_omit_them() -> None:
    assert Usage(prompt_tokens=10, completion_tokens=5).reasoning_tokens == 0


def test_cache_hit_pct() -> None:
    assert Usage(prompt_tokens=1000, cached_tokens=900).cache_hit_pct == 90
    assert Usage(prompt_tokens=1000, cached_tokens=0).cache_hit_pct == 0


def test_cache_hit_pct_does_not_divide_by_zero() -> None:
    assert Usage().cache_hit_pct == 0
