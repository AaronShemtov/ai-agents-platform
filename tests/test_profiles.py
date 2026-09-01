"""Profile loading and tool-glob matching.

`test_shipped_profiles_are_valid` is the one that earns its keep in CI: a typo in a
YAML profile would otherwise only surface as a crash loop after deploy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentcore.profiles import AgentProfile, ProfileNotFound, list_profiles, load_profile


def make_profile(**overrides: object) -> AgentProfile:
    data: dict[str, object] = {
        "id": "t",
        "name": "Test",
        "system_prompt": "do things",
    }
    data.update(overrides)
    return AgentProfile.model_validate(data)


# -- shipped files -----------------------------------------------------------


def test_shipped_profiles_are_valid(agents_dir: Path) -> None:
    names = list_profiles(agents_dir)
    assert names, "no agent profiles found"
    for name in names:
        profile = load_profile(name, agents_dir)
        assert profile.id == name
        assert profile.system_prompt


def test_lead_profile_wires_up_the_three_servers(agents_dir: Path) -> None:
    lead = load_profile("lead", agents_dir)
    assert set(lead.mcp_servers) == {"github", "cloudflare", "cluster"}


def test_missing_profile_names_the_alternatives(agents_dir: Path) -> None:
    with pytest.raises(ProfileNotFound, match="lead"):
        load_profile("nope", agents_dir)


# -- tool matching -----------------------------------------------------------


def test_empty_allow_list_means_allow_everything() -> None:
    profile = make_profile()
    assert profile.tool_allowed("github__anything")


def test_allow_glob_scopes_to_a_server() -> None:
    profile = make_profile(allow_tools=["github__*"])
    assert profile.tool_allowed("github__create_issue")
    assert not profile.tool_allowed("cloudflare__create_dns_record")


def test_deny_beats_allow() -> None:
    profile = make_profile(allow_tools=["github__*"], deny_tools=["github__delete_*"])
    assert profile.tool_allowed("github__create_issue")
    assert not profile.tool_allowed("github__delete_repository")


def test_deny_applies_even_with_an_empty_allow_list() -> None:
    profile = make_profile(deny_tools=["*__delete_*"])
    assert not profile.tool_allowed("cloudflare__delete_dns_record")
    assert profile.tool_allowed("cloudflare__create_dns_record")


def test_empty_system_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        make_profile(system_prompt="   ")


def test_coder_has_no_cloudflare(agents_dir: Path) -> None:
    """Cloudflare's write mode is `approve` and a worker agent has no one to ask.

    Giving it those tools would only produce refusals at the point of use, so the
    absence is deliberate and worth pinning.
    """
    coder = load_profile("coder", agents_dir)
    assert "cloudflare" not in coder.mcp_servers
    assert not coder.tool_allowed("cloudflare__create_dns_record")


def test_coder_pins_its_own_model(agents_dir: Path) -> None:
    """The lead inherits MODEL_DEFAULT; the coder is chosen on measurement."""
    coder = load_profile("coder", agents_dir)
    assert coder.model == "gpt-5.3-codex"
    assert load_profile("lead", agents_dir).model is None


def test_lead_can_reach_the_coder(agents_dir: Path) -> None:
    lead = load_profile("lead", agents_dir)
    assert "coder" in lead.mcp_servers
    assert lead.tool_allowed("coder__ask")
