"""Token accounting.

`Usage.__add__` builds a new Usage positionally, so a field inserted in the wrong
order would silently swap counters — the kind of bug that only shows up as puzzling
numbers on a Grafana panel weeks later.
"""

from __future__ import annotations

from agentcore.llm.base import Usage


def test_total_excludes_cached_which_is_a_subset_of_prompt() -> None:
    # cached_tokens counts part of prompt_tokens, so adding it in would double-count.
    usage = Usage(prompt_tokens=1000, completion_tokens=50, cached_tokens=900)
    assert usage.total == 1050


def test_addition_keeps_each_counter_in_its_own_field() -> None:
    total = Usage(1, 2, 3) + Usage(10, 20, 30)
    assert (total.prompt_tokens, total.completion_tokens, total.cached_tokens) == (11, 22, 33)


def test_cache_hit_pct() -> None:
    assert Usage(prompt_tokens=1000, cached_tokens=900).cache_hit_pct == 90
    assert Usage(prompt_tokens=1000, cached_tokens=0).cache_hit_pct == 0


def test_cache_hit_pct_does_not_divide_by_zero() -> None:
    assert Usage().cache_hit_pct == 0
