"""Guardrail tests.

The case that matters most is `test_protected_repo_*`: a direct push to the GitOps
repository reaches the live cluster within about ten minutes, so it must be refused
even when the write mode is the most permissive one.
"""

from __future__ import annotations

import pytest

from agentcore.config import Settings, WriteMode
from agentcore.policy import Decision, Policy


def make_policy(**overrides: object) -> Policy:
    defaults: dict[str, object] = {
        "github_write_mode": WriteMode.DIRECT_PUSH,
        "cloudflare_write_mode": WriteMode.APPROVE,
        # Comma-separated, exactly as a ConfigMap supplies it — see the note in
        # agentcore.config about why these are strings and not lists.
        "protected_repos": "personal-k8s",
    }
    defaults.update(overrides)
    return Policy(Settings(**defaults))  # type: ignore[arg-type]


# -- read tools --------------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        "github__get_file_contents",
        "github__list_commits",
        "github__search_code",
        "cloudflare__list_dns_records",
        "cluster__get_pod_logs",
        "cluster__list_pods",
    ],
)
def test_read_tools_always_allowed(tool: str) -> None:
    assert make_policy().evaluate(tool, {}).decision is Decision.ALLOW


def test_read_only_check_is_on_the_bare_name_not_the_namespace() -> None:
    # "cluster__" must not accidentally match a read prefix.
    assert make_policy().evaluate("cluster__delete_pod", {}).decision is Decision.DENY


# -- GitHub ------------------------------------------------------------------


def test_github_read_only_mode_blocks_writes() -> None:
    policy = make_policy(github_write_mode=WriteMode.READ_ONLY)
    verdict = policy.evaluate("github__create_or_update_file", {"repo": "some-site"})
    assert verdict.decision is Decision.DENY
    assert "read-only" in verdict.reason


def test_direct_push_allows_ordinary_repo() -> None:
    verdict = make_policy().evaluate(
        "github__create_or_update_file", {"repo": "my-new-site", "branch": "main"}
    )
    assert verdict.decision is Decision.ALLOW


def test_protected_repo_blocked_on_default_branch_even_in_direct_push() -> None:
    verdict = make_policy().evaluate(
        "github__create_or_update_file", {"repo": "personal-k8s", "branch": "main"}
    )
    assert verdict.decision is Decision.DENY
    assert "pull request" in verdict.reason.lower()


def test_protected_repo_blocked_when_branch_is_omitted() -> None:
    # No branch argument means the tool acts on the default branch.
    verdict = make_policy().evaluate("github__push_files", {"repo": "personal-k8s"})
    assert verdict.decision is Decision.DENY


def test_protected_repo_case_insensitive() -> None:
    verdict = make_policy().evaluate("github__push_files", {"repo": "Personal-K8s"})
    assert verdict.decision is Decision.DENY


def test_protected_repo_writable_on_a_feature_branch() -> None:
    verdict = make_policy().evaluate(
        "github__create_or_update_file",
        {"repo": "personal-k8s", "branch": "agent/add-site"},
    )
    assert verdict.decision is Decision.ALLOW


def test_protected_repo_allows_opening_the_pull_request() -> None:
    # create_pull_request names main as its base; that must not be mistaken for a push.
    verdict = make_policy().evaluate(
        "github__create_pull_request", {"repo": "personal-k8s", "base": "main"}
    )
    assert verdict.decision is Decision.ALLOW


def test_pull_request_mode_blocks_default_branch_on_any_repo() -> None:
    policy = make_policy(github_write_mode=WriteMode.PULL_REQUEST)
    verdict = policy.evaluate(
        "github__create_or_update_file", {"repo": "my-new-site", "branch": "master"}
    )
    assert verdict.decision is Decision.DENY


def test_pull_request_mode_allows_feature_branch() -> None:
    policy = make_policy(github_write_mode=WriteMode.PULL_REQUEST)
    verdict = policy.evaluate(
        "github__create_or_update_file", {"repo": "my-new-site", "branch": "feature/x"}
    )
    assert verdict.decision is Decision.ALLOW


# -- Cloudflare --------------------------------------------------------------


def test_cloudflare_defaults_to_requiring_approval() -> None:
    verdict = make_policy().evaluate("cloudflare__delete_dns_record", {"zone": "1ms.my"})
    assert verdict.decision is Decision.REQUIRE_APPROVAL


def test_cloudflare_auto_mode_allows() -> None:
    policy = make_policy(cloudflare_write_mode=WriteMode.AUTO)
    assert policy.evaluate("cloudflare__create_dns_record", {}).decision is Decision.ALLOW


def test_cloudflare_read_only_denies() -> None:
    policy = make_policy(cloudflare_write_mode=WriteMode.READ_ONLY)
    assert policy.evaluate("cloudflare__create_dns_record", {}).decision is Decision.DENY


# -- cluster and unknown -----------------------------------------------------


def test_cluster_writes_are_refused_with_a_gitops_explanation() -> None:
    verdict = make_policy().evaluate("cluster__restart_deployment", {})
    assert verdict.decision is Decision.DENY
    assert "GitOps" in verdict.reason


def test_unknown_namespace_requires_approval_rather_than_being_allowed() -> None:
    verdict = make_policy().evaluate("something__do_things", {})
    assert verdict.decision is Decision.REQUIRE_APPROVAL


# -- merging into the GitOps repository --------------------------------------


def test_protected_repo_merge_asks_instead_of_refusing() -> None:
    """Merging is a decision to put in front of a person, not one to refuse.

    Denying it did not make anything safer — the pull request still existed and
    still got merged, just by hand on github.com, which is a worse place to judge
    a diff than a phone with an Approve button.
    """
    verdict = make_policy().evaluate(
        "github__merge_pull_request", {"repo": "personal-k8s", "pullNumber": 7}
    )
    assert verdict.decision is Decision.REQUIRE_APPROVAL
    # The reason has to say what approving actually does, because it is read on a
    # phone by someone who was not watching the agent work.
    assert "Flux" in verdict.reason


def test_merging_still_asks_when_the_base_branch_is_named_explicitly() -> None:
    verdict = make_policy().evaluate(
        "github__merge_pull_request", {"repo": "personal-k8s", "base": "main"}
    )
    assert verdict.decision is Decision.REQUIRE_APPROVAL


def test_direct_writes_to_the_protected_default_branch_are_still_refused() -> None:
    """The concession is merging, and only merging. Pushing to main is still out."""
    for tool in ("github__create_or_update_file", "github__push_files"):
        verdict = make_policy().evaluate(tool, {"repo": "personal-k8s", "branch": "main"})
        assert verdict.decision is Decision.DENY, tool


def test_merging_an_ordinary_repo_needs_no_approval() -> None:
    """Nothing there reaches the cluster, and asking about every one of them is
    what made the button worth avoiding in the first place."""
    verdict = make_policy().evaluate(
        "github__merge_pull_request", {"repo": "urlshortener-backend"}
    )
    assert verdict.decision is Decision.ALLOW
