import pytest

from agentcore.config import Settings, WriteMode
from agentcore.policy import Decision, Policy, is_read_only_tool


@pytest.mark.parametrize("tool", ["pull_request_read", "issue_read"])
def test_dispatcher_is_classified_as_read_only(tool):
    assert is_read_only_tool(tool)


@pytest.mark.parametrize("tool", ["pull_request_read", "issue_read"])
@pytest.mark.parametrize("mode", [WriteMode.DIRECT_PUSH, WriteMode.PULL_REQUEST, WriteMode.READ_ONLY])
def test_read_dispatcher_is_allowed_on_protected_repo(tool, mode):
    policy = Policy(
        Settings(
            github_write_mode=mode,
            protected_repos="personal-k8s",
        )
    )

    verdict = policy.evaluate(
        f"github__{tool}",
        {
            "owner": "AaronShemtov",
            "repo": "personal-k8s",
            "pullNumber": 3,
            "method": "get_check_runs",
        },
    )

    assert verdict.decision is Decision.ALLOW


def test_write_to_protected_default_branch_is_still_denied():
    policy = Policy(
        Settings(
            github_write_mode=WriteMode.DIRECT_PUSH,
            protected_repos="personal-k8s",
        )
    )

    verdict = policy.evaluate(
        "github__push_files",
        {
            "owner": "AaronShemtov",
            "repo": "personal-k8s",
            "branch": "main",
            "files": [],
        },
    )

    assert verdict.decision is Decision.DENY
    assert "GitOps" in verdict.reason
