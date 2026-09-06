"""Guardrails evaluated before every tool call.

This is defence in depth, not the primary control. The real boundaries live outside
this process and cannot be argued with by a language model:

  * the GitHub PAT is fine-grained and scoped to exactly two repositories
  * `mcp-github` runs with `--toolsets` (and optionally `--read-only`) fixed in git
  * `mcp-cluster` has a read-only ClusterRole; the agent pod has no ClusterRole at all
  * `mcp-cloudflare` only exposes the ~13 tools we wrote

What this module adds is the part that depends on *arguments* rather than identity —
above all: never push straight to the GitOps repository. A bad manifest committed to
`personal-k8s` reaches the live cluster within about ten minutes, so that one is
enforced regardless of how permissive the write mode is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agentcore.config import Settings, WriteMode
from agentcore.localtools import MEMORY_SERVER
from agentcore.mcp.client import split_qualified

# Verbs that only read. Anything not matching these is treated as mutating — unknown
# tools should be gated, not waved through.
READ_PREFIXES = (
    "get_",
    "list_",
    "search_",
    "read_",
    "show_",
    "describe_",
    "fetch_",
    "find_",
    "check_",
    "download_",
)

# GitHub MCP has dispatcher tools whose operation is selected by a required `method`
# argument. Despite not starting with a read verb, these expose read operations only.
# Keep this explicit rather than allowing every name containing "read".
READ_ONLY_TOOLS = frozenset(
    {
        "issue_read",
        "pull_request_read",
    }
)

# Branch names we treat as "the branch Flux and production read from".
DEFAULT_BRANCHES = frozenset({"main", "master"})

# Argument keys different tools use for the same concepts.
_REPO_KEYS = ("repo", "repository", "repo_name")
_BRANCH_KEYS = ("branch", "ref", "head", "base")


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


def is_read_only_tool(bare_name: str) -> bool:
    return bare_name in READ_ONLY_TOOLS or bare_name.startswith(READ_PREFIXES)


def _arg(arguments: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class Policy:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._protected = settings.protected_repo_set()

    def evaluate(self, qualified_name: str, arguments: dict[str, Any]) -> Verdict:
        server, bare = split_qualified(qualified_name)

        if is_read_only_tool(bare):
            return Verdict(Decision.ALLOW)

        if server == "github":
            return self._github(bare, arguments)
        if server == "cloudflare":
            return self._cloudflare(bare)
        if server == "cluster":
            # The ClusterRole makes writes impossible anyway; refusing here gives the
            # model a clear explanation instead of an opaque 403 from the apiserver.
            return Verdict(
                Decision.DENY,
                "кластер только на чтение: он управляется через GitOps. "
                "Чтобы изменить кластер, отредактируй манифесты в personal-k8s и открой PR.",
            )

        if server == MEMORY_SERVER:
            # The agent's own two tables, under a database user with no grants
            # on anything else — `select from admin.urls` as it is ORA-00942.
            # There is nothing here worth gating: the worst case is a wrong
            # fact about the person, which they can see with /memories and drop
            # with /forget. Requiring a button would mean approving every act of
            # remembering, which is the opposite of what memory is for.
            return Verdict(Decision.ALLOW)

        # Unknown namespace: gate rather than guess.
        return Verdict(
            Decision.REQUIRE_APPROVAL,
            f"инструмент {qualified_name} не относится к известным пространствам имён",
        )

    # -- per-namespace rules -------------------------------------------------

    def _github(self, bare: str, arguments: dict[str, Any]) -> Verdict:
        mode = self._s.github_write_mode

        if mode is WriteMode.READ_ONLY:
            return Verdict(Decision.DENY, "GITHUB_WRITE_MODE=read-only: запись в GitHub запрещена")

        repo = _arg(arguments, _REPO_KEYS)
        branch = _arg(arguments, _BRANCH_KEYS)
        # A missing branch means the tool acts on the repository default branch.
        targets_default = branch is None or branch.lower() in DEFAULT_BRANCHES

        # Protected repos: writable, but only through a pull request. This
        # survives direct-push mode on purpose.
        if repo and repo.lower() in self._protected and targets_default:
            if bare.startswith(("create_pull_request", "create_branch")):
                return Verdict(Decision.ALLOW)
            # Merging is not the same act as writing to main, even though both end
            # up there. A pull request has a diff someone can look at, and refusing
            # to merge it did not make anything safer — it only moved the click to
            # github.com, which is a worse place to make the decision than a phone.
            # So this asks instead of refusing: one button, same human, same diff.
            if bare.startswith("merge_pull_request"):
                return Verdict(
                    Decision.REQUIRE_APPROVAL,
                    f"слияние PR в {repo}: Flux применит эту ветку к живому кластеру "
                    "в течение ~10 минут",
                )
            return Verdict(
                Decision.DENY,
                f"{repo} — GitOps-репозиторий, прямая запись в основную ветку запрещена. "
                "Создай ветку и открой pull request.",
            )

        if mode is WriteMode.PULL_REQUEST and targets_default:
            return Verdict(
                Decision.DENY,
                "GITHUB_WRITE_MODE=pull-request: пиши в отдельную ветку и открывай PR, "
                "а не в main/master.",
            )

        return Verdict(Decision.ALLOW)

    def _cloudflare(self, bare: str) -> Verdict:
        mode = self._s.cloudflare_write_mode
        if mode is WriteMode.READ_ONLY:
            return Verdict(
                Decision.DENY, "CLOUDFLARE_WRITE_MODE=read-only: изменения в Cloudflare запрещены"
            )
        if mode is WriteMode.AUTO:
            return Verdict(Decision.ALLOW)
        # Default: APPROVE. A bad commit is revertible; a deleted DNS record takes
        # 1ms.my offline until someone notices.
        return Verdict(
            Decision.REQUIRE_APPROVAL,
            f"изменение в Cloudflare ({bare}) требует подтверждения",
        )
