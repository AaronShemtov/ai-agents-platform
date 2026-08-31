"""Agent profiles: the whole reason this is a monorepo rather than a repo per agent.

An agent is defined by four things — system prompt, tool allowlist, model, limits.
All four are data. Adding an architect / coder / reviewer means adding a YAML file
under `agents/`, not writing Python.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class AgentProfile(BaseModel):
    id: str
    name: str
    description: str = ""

    system_prompt: str

    # None means "use MODEL_DEFAULT from the ConfigMap", so a profile does not have to
    # be edited when the deployment switches models.
    model: str | None = None

    # Logical MCP server names, resolved against Settings.mcp_endpoints().
    mcp_servers: list[str] = Field(default_factory=list)

    # Glob patterns matched against namespaced tool names, e.g. "github__create_*".
    # Empty allow_tools means "everything the configured servers expose".
    allow_tools: list[str] = Field(default_factory=list)
    deny_tools: list[str] = Field(default_factory=list)

    # None means "inherit the process-wide limit from Settings".
    max_steps: int | None = None
    max_tokens_per_turn: int | None = None

    @field_validator("system_prompt")
    @classmethod
    def _non_empty_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("system_prompt must not be empty")
        return v.strip()

    def tool_allowed(self, tool_name: str) -> bool:
        """Deny wins over allow, and an empty allow list means allow-all."""
        if any(fnmatch.fnmatch(tool_name, pat) for pat in self.deny_tools):
            return False
        if not self.allow_tools:
            return True
        return any(fnmatch.fnmatch(tool_name, pat) for pat in self.allow_tools)


class ProfileNotFound(Exception):
    pass


def load_profile(profile_id: str, agents_dir: Path) -> AgentProfile:
    """Load `<agents_dir>/<profile_id>.yaml`."""
    path = agents_dir / f"{profile_id}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in agents_dir.glob("*.yaml")) if agents_dir.is_dir() else []
        raise ProfileNotFound(
            f"no profile {profile_id!r} at {path}; available: {available or '(none)'}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("id", profile_id)
    return AgentProfile.model_validate(data)


def list_profiles(agents_dir: Path) -> list[str]:
    if not agents_dir.is_dir():
        return []
    return sorted(p.stem for p in agents_dir.glob("*.yaml"))
