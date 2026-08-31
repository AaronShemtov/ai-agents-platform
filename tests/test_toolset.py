"""Tests for the MCP -> OpenAI tool translation.

Both limits checked here cause a 400 from Azure that reads as "the model is broken"
rather than "one tool description was too long", so they are worth pinning down.
"""

from __future__ import annotations

from agentcore.mcp.client import RemoteTool
from agentcore.mcp.toolset import (
    MAX_DESCRIPTION,
    MAX_NAME,
    build_catalog,
    sanitize_name,
    truncate_description,
)


def tool(name: str, *, server: str = "github", description: str = "d", schema=None) -> RemoteTool:
    return RemoteTool(
        server=server,
        name=name,
        qualified_name=f"{server}__{name}",
        description=description,
        input_schema=schema if schema is not None else {"type": "object", "properties": {}},
    )


# -- descriptions ------------------------------------------------------------


def test_short_description_untouched() -> None:
    assert truncate_description("hello world") == "hello world"


def test_description_is_truncated_to_the_api_limit() -> None:
    assert len(truncate_description("word " * 1000)) <= MAX_DESCRIPTION


def test_truncation_prefers_a_word_boundary() -> None:
    result = truncate_description("word " * 1000)
    assert result.endswith("…")
    assert not result[:-1].endswith(" ")


def test_multiline_prose_is_collapsed() -> None:
    # MCP servers ship descriptions with newlines and runs of spaces.
    assert truncate_description("a\n\n  b\tc") == "a b c"


def test_real_world_long_description_fits() -> None:
    catalog = build_catalog([tool("create_or_update_file", description="x" * 5000)])
    assert len(catalog.specs[0]["function"]["description"]) <= MAX_DESCRIPTION


# -- names -------------------------------------------------------------------


def test_namespaced_name_survives_intact() -> None:
    assert sanitize_name("github__create_issue") == "github__create_issue"


def test_illegal_characters_replaced() -> None:
    assert sanitize_name("cf__dns.record/create") == "cf__dns_record_create"


def test_over_long_name_is_shortened_and_stays_within_limit() -> None:
    name = sanitize_name("github__" + "a" * 200)
    assert len(name) <= MAX_NAME


def test_two_different_long_names_do_not_collapse_into_one() -> None:
    first = sanitize_name("github__" + "a" * 200)
    second = sanitize_name("github__" + "a" * 199 + "b")
    assert first != second


def test_catalog_names_are_unique_even_after_sanitising() -> None:
    catalog = build_catalog([tool("a.b"), tool("a/b")])
    assert len(set(catalog.names())) == len(catalog.names()) == 2


# -- filtering ---------------------------------------------------------------


def test_profile_filter_removes_tools_before_the_model_sees_them() -> None:
    tools = [tool("create_issue"), tool("delete_repository")]
    catalog = build_catalog(tools, is_allowed=lambda name: not name.endswith("delete_repository"))
    assert catalog.names() == ["github__create_issue"]
    assert catalog.skipped == ["github__delete_repository"]


def test_resolve_maps_back_to_the_mcp_qualified_name() -> None:
    catalog = build_catalog([tool("create_issue")])
    assert catalog.resolve("github__create_issue") == "github__create_issue"
    assert catalog.resolve("nope") is None


def test_resolve_works_for_a_renamed_long_tool() -> None:
    long_tool = tool("z" * 200)
    catalog = build_catalog([long_tool])
    openai_name = catalog.names()[0]
    assert catalog.resolve(openai_name) == long_tool.qualified_name


# -- schemas -----------------------------------------------------------------


def test_empty_schema_becomes_a_valid_object_schema() -> None:
    spec = build_catalog([tool("x", schema={})]).specs[0]
    assert spec["function"]["parameters"] == {"type": "object", "properties": {}}


def test_existing_schema_is_preserved() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    spec = build_catalog([tool("x", schema=schema)]).specs[0]
    assert spec["function"]["parameters"]["required"] == ["a"]
