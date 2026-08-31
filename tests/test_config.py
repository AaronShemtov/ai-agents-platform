"""Settings parsing.

These pin down a startup crash that is easy to reintroduce: pydantic-settings
json.loads() the raw environment value of any complex-typed field *before* field
validators run. Typing `telegram_allowed_users` as `list[int]` therefore blows up on
`TELEGRAM_ALLOWED_USERS=139236889` — which is exactly the shape a ConfigMap supplies —
and no `mode="before"` validator would ever be reached to fix it up.

So the fields stay strings and the parsing lives in accessors. These tests exist to
stop someone "tidying" them back into lists.
"""

from __future__ import annotations

import pytest

from agentcore.config import Settings, WriteMode


def test_single_telegram_id_parses() -> None:
    # The exact ConfigMap shape that broke the list[int] version.
    assert Settings(telegram_allowed_users="139236889").allowed_user_ids() == [139236889]


def test_several_telegram_ids_parse() -> None:
    assert Settings(telegram_allowed_users="1, 2,3 ").allowed_user_ids() == [1, 2, 3]


def test_empty_allowlist_means_nobody() -> None:
    # Fail-closed: this is what makes an unset allowlist refuse everyone.
    assert Settings(telegram_allowed_users="").allowed_user_ids() == []


def test_junk_ids_are_skipped_not_fatal() -> None:
    assert Settings(telegram_allowed_users="1,oops,3").allowed_user_ids() == [1, 3]


def test_model_allowlist_is_comma_separated() -> None:
    settings = Settings(model_default="gpt-5-mini", model_allowlist="gpt-5-mini,gpt-5")
    assert settings.allowed_models() == ["gpt-5-mini", "gpt-5"]


def test_default_model_is_always_selectable_even_if_absent_from_the_allowlist() -> None:
    settings = Settings(model_default="gpt-5-mini", model_allowlist="gpt-5")
    assert settings.allowed_models() == ["gpt-5-mini", "gpt-5"]


def test_protected_repos_parse_lowercased() -> None:
    assert Settings(protected_repos="personal-k8s, Other").protected_repo_set() == {
        "personal-k8s",
        "other",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://r.openai.azure.com/openai/v1", "https://r.openai.azure.com/openai/v1/"),
        ("https://r.openai.azure.com/openai/v1/", "https://r.openai.azure.com/openai/v1/"),
        ("", ""),
    ],
)
def test_base_url_gets_exactly_one_trailing_slash(raw: str, expected: str) -> None:
    assert Settings(azure_openai_base_url=raw).base_url() == expected


def test_write_modes_accept_the_configmap_strings() -> None:
    settings = Settings(github_write_mode="pull-request", cloudflare_write_mode="read-only")
    assert settings.github_write_mode is WriteMode.PULL_REQUEST
    assert settings.cloudflare_write_mode is WriteMode.READ_ONLY


def test_unset_mcp_servers_are_omitted() -> None:
    settings = Settings(
        mcp_github_url="http://mcp-github:8082/",
        mcp_cloudflare_url="",
        mcp_cluster_url="http://mcp-cluster:8080/mcp",
    )
    assert set(settings.mcp_endpoints()) == {"github", "cluster"}
