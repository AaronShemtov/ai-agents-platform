"""Header-bound tool parameters, and readable transport errors.

Both of these come straight out of a live failure — see docs/mcp-header-params.md.
Every GitHub tool taking a repository was unusable, and the error the agent reported
was "unhandled errors in a TaskGroup", which named the wrapper and hid the cause.
"""

from __future__ import annotations

from agentcore.mcp.client import explain_exception, header_params

# Shape the GitHub MCP server actually returns for list_branches.
GITHUB_SCHEMA = {
    "type": "object",
    "properties": {
        "owner": {"type": "string", "x-mcp-header": "owner"},
        "repo": {"type": "string", "x-mcp-header": "repo"},
        "perPage": {"type": "number"},
        "page": {"type": "number"},
    },
    "required": ["owner", "repo"],
}

# search_repositories — the one tool that worked, because nothing is header-bound.
PLAIN_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "perPage": {"type": "number"}},
}


def test_header_bound_params_become_mcp_param_headers() -> None:
    got = header_params(GITHUB_SCHEMA, {"owner": "AaronShemtov", "repo": "personal-k8s"})
    assert got == {"Mcp-Param-owner": "AaronShemtov", "Mcp-Param-repo": "personal-k8s"}


def test_plain_params_are_not_promoted_to_headers() -> None:
    got = header_params(GITHUB_SCHEMA, {"owner": "a", "repo": "b", "perPage": 30, "page": 1})
    assert set(got) == {"Mcp-Param-owner", "Mcp-Param-repo"}


def test_schema_without_header_bindings_yields_nothing() -> None:
    assert header_params(PLAIN_SCHEMA, {"query": "kubernetes", "perPage": 30}) == {}


def test_absent_arguments_are_skipped() -> None:
    # A header must not be sent for a parameter the model did not supply.
    assert header_params(GITHUB_SCHEMA, {"owner": "a"}) == {"Mcp-Param-owner": "a"}


def test_none_is_treated_as_absent() -> None:
    assert header_params(GITHUB_SCHEMA, {"owner": "a", "repo": None}) == {"Mcp-Param-owner": "a"}


def test_values_are_stringified() -> None:
    schema = {"properties": {"n": {"x-mcp-header": "n"}}}
    assert header_params(schema, {"n": 42}) == {"Mcp-Param-n": "42"}


def test_empty_and_malformed_schemas_do_not_raise() -> None:
    assert header_params({}, {"a": 1}) == {}
    assert header_params({"properties": None}, {"a": 1}) == {}
    assert header_params({"properties": {"a": "not-a-dict"}}, {"a": 1}) == {}


# -- error flattening --------------------------------------------------------


def test_plain_exception_names_its_type_and_message() -> None:
    assert explain_exception(ValueError("bad")) == "ValueError: bad"


def test_exception_group_is_unwrapped_to_the_real_cause() -> None:
    group = ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("header mismatch")])
    assert explain_exception(group) == "RuntimeError: header mismatch"


def test_nested_groups_are_flattened() -> None:
    inner = ExceptionGroup("inner", [ValueError("a")])
    outer = ExceptionGroup("outer", [inner, KeyError("b")])
    out = explain_exception(outer)
    assert "ValueError: a" in out
    assert "KeyError" in out


def test_group_wrapper_text_never_leaks_alone() -> None:
    group = ExceptionGroup("unhandled errors in a TaskGroup", [OSError("connection reset")])
    assert "TaskGroup" not in explain_exception(group)
