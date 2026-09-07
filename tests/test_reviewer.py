"""The reviewer agent: a third profile the lead reaches as one more tool.

There is no new machinery here — `python -m mcp_agent` already turns a profile
into an MCP server — so what these tests pin down is the part that a typo would
turn into a review nobody notices is missing: that the profile cannot write, that
the lead can reach it, and that reaching it does not stop to ask a human.
"""

from __future__ import annotations

from pathlib import Path

from agentcore.config import Settings
from agentcore.policy import Decision, Policy
from agentcore.profiles import load_profile

# Every verb that changes something on GitHub. None of these may be reachable.
WRITING = (
    "github__create_or_update_file",
    "github__create_pull_request",
    "github__merge_pull_request",
    "github__push_files",
    "github__update_issue",
    "github__delete_file",
)

READING = (
    "github__get_file_contents",
    "github__list_commits",
    "github__search_code",
    "github__pull_request_read",
    "cluster__list_pods",
)


def test_the_reviewer_runs_on_a_different_model_than_the_coder(agents_dir: Path) -> None:
    """Same model as the author would inherit the author's blind spots."""
    reviewer = load_profile("reviewer", agents_dir)
    coder = load_profile("coder", agents_dir)
    assert reviewer.model
    assert reviewer.model != coder.model


def test_the_reviewer_cannot_write_anything(agents_dir: Path) -> None:
    reviewer = load_profile("reviewer", agents_dir)
    for tool in WRITING:
        assert not reviewer.tool_allowed(tool), tool


def test_the_reviewer_can_still_read_the_code_it_reviews(agents_dir: Path) -> None:
    reviewer = load_profile("reviewer", agents_dir)
    for tool in READING:
        assert reviewer.tool_allowed(tool), tool


def test_the_reviewer_has_no_one_to_delegate_to(agents_dir: Path) -> None:
    """It reviews; it does not hand work onward. Nor can it call itself."""
    reviewer = load_profile("reviewer", agents_dir)
    assert "coder" not in reviewer.mcp_servers
    assert "reviewer" not in reviewer.mcp_servers


def test_the_lead_may_call_the_reviewer(agents_dir: Path) -> None:
    lead = load_profile("lead", agents_dir)
    assert lead.tool_allowed("reviewer__ask")


def test_asking_for_review_does_not_wait_on_a_human() -> None:
    """A review that costs a button press is a review that gets skipped."""
    verdict = Policy(Settings()).evaluate("reviewer__ask", {"task": "review this"})
    assert verdict.decision is Decision.ALLOW


def test_the_reviewer_endpoint_is_optional() -> None:
    """Unset means this deployment has no reviewer, not a broken one."""
    assert "reviewer" not in Settings().mcp_endpoints()


def test_a_configured_reviewer_endpoint_is_offered() -> None:
    s = Settings(mcp_reviewer_url="http://agent-reviewer.agents.svc.cluster.local:8080/mcp")
    assert s.mcp_endpoints()["reviewer"].endswith("/mcp")
