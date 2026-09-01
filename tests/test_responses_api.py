"""Translation between our Chat-shaped transcript and the Responses API.

Two endpoints exist because the codex models refuse /chat/completions — HTTP 400 "The
requested operation is unsupported", by design across four model generations. The
translation below is what lets the coder role use the model actually built for code.

The case that matters most is `_raw_output`: a reasoning model returns reasoning items
next to its function_call, and those must be replayed on the following request or the
model loses its own chain of thought between steps. Rebuilding an assistant turn from
our own fields silently drops them, which is exactly the bug these tests exist to catch.
"""

from __future__ import annotations

from types import SimpleNamespace

from agentcore.llm.azure import (
    RAW_OUTPUT_KEY,
    from_responses,
    strip_internal,
    to_responses_input,
    to_responses_tools,
)

CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "list_pods",
        "description": "List pods",
        "parameters": {"type": "object", "properties": {"ns": {"type": "string"}}},
    },
}


# -- tools -------------------------------------------------------------------


def test_tools_are_flattened_out_of_the_function_wrapper() -> None:
    (out,) = to_responses_tools([CHAT_TOOL])
    assert out == {
        "type": "function",
        "name": "list_pods",
        "description": "List pods",
        "parameters": {"type": "object", "properties": {"ns": {"type": "string"}}},
    }


def test_a_tool_without_parameters_still_gets_a_valid_schema() -> None:
    (out,) = to_responses_tools([{"type": "function", "function": {"name": "ping"}}])
    assert out["parameters"] == {"type": "object", "properties": {}}


# -- internal keys never leave the process -----------------------------------


def test_internal_keys_are_stripped_before_chat_completions() -> None:
    messages = [{"role": "assistant", "content": "hi", RAW_OUTPUT_KEY: [{"type": "reasoning"}]}]
    assert strip_internal(messages) == [{"role": "assistant", "content": "hi"}]


def test_stripping_leaves_ordinary_messages_alone() -> None:
    messages = [{"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    assert strip_internal(messages) == messages


# -- transcript -> Responses input -------------------------------------------


def test_system_prompt_becomes_instructions_not_an_input_item() -> None:
    instructions, items = to_responses_input(
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    )
    assert instructions == "be terse"
    assert items == [{"role": "user", "content": "hi"}]


def test_tool_results_become_function_call_output() -> None:
    _, items = to_responses_input(
        [{"role": "tool", "tool_call_id": "call_1", "content": "3 pods"}]
    )
    assert items == [{"type": "function_call_output", "call_id": "call_1", "output": "3 pods"}]


def test_reasoning_items_are_replayed_verbatim() -> None:
    """The whole point: a reasoning item must survive the round trip."""
    raw = [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "function_call", "call_id": "call_1", "name": "list_pods", "arguments": "{}"},
    ]
    _, items = to_responses_input(
        [{"role": "assistant", "content": None, RAW_OUTPUT_KEY: raw}]
    )
    assert items == raw


def test_a_chat_shaped_assistant_turn_is_rebuilt_when_there_are_no_native_items() -> None:
    """After a /model switch the history can be in the other endpoint's shape."""
    _, items = to_responses_input(
        [
            {
                "role": "assistant",
                "content": "looking",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_pods", "arguments": '{"ns":"cv"}'},
                    }
                ],
            }
        ]
    )
    assert items == [
        {"role": "assistant", "content": "looking"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "list_pods",
            "arguments": '{"ns":"cv"}',
        },
    ]


def test_a_full_multi_step_transcript_round_trips_in_order() -> None:
    raw = [{"type": "reasoning", "id": "rs_1"}]
    instructions, items = to_responses_input(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "check cv"},
            {"role": "assistant", "content": None, RAW_OUTPUT_KEY: raw},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {"role": "user", "content": "and now?"},
        ]
    )
    assert instructions == "sys"
    assert [i.get("type") or i.get("role") for i in items] == [
        "user",
        "reasoning",
        "function_call_output",
        "user",
    ]


# -- Responses output -> LLMResponse -----------------------------------------


def fake_response(output, *, status="completed", incomplete_reason=None, usage=None):
    return SimpleNamespace(
        output=output,
        status=status,
        incomplete_details=SimpleNamespace(reason=incomplete_reason),
        usage=usage,
    )


def test_a_function_call_is_parsed_with_call_id_as_the_tool_call_id() -> None:
    resp = fake_response(
        [{"type": "function_call", "call_id": "call_9", "name": "list_pods", "arguments": '{"ns":"cv"}'}]
    )
    parsed = from_responses(resp)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("call_9", "list_pods", {"ns": "cv"})


def test_message_text_is_joined_and_reasoning_items_are_kept_for_replay() -> None:
    resp = fake_response(
        [
            {"type": "reasoning", "id": "rs_1"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ]
    )
    parsed = from_responses(resp)
    assert parsed.content == "done"
    assert parsed.raw_message[RAW_OUTPUT_KEY][0]["type"] == "reasoning"


def test_malformed_arguments_become_an_empty_dict_rather_than_crashing() -> None:
    resp = fake_response(
        [{"type": "function_call", "call_id": "c", "name": "x", "arguments": "{not json"}]
    )
    assert from_responses(resp).tool_calls[0].arguments == {}


def test_usage_maps_onto_the_shared_shape() -> None:
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=200,
        input_tokens_details=SimpleNamespace(cached_tokens=900),
        output_tokens_details=SimpleNamespace(reasoning_tokens=150),
    )
    parsed = from_responses(fake_response([], usage=usage))
    assert parsed.usage.prompt_tokens == 1000
    assert parsed.usage.completion_tokens == 200
    assert parsed.usage.cached_tokens == 900
    assert parsed.usage.reasoning_tokens == 150
    # Same definition as the chat path: the cached prefix does not count as spend.
    assert parsed.usage.billable == 300


def test_missing_usage_does_not_raise() -> None:
    assert from_responses(fake_response([])).usage.prompt_tokens == 0


def test_truncation_arrives_as_length_so_the_loop_stops() -> None:
    """The loop keys off finish_reason == 'length' to avoid acting on a partial call."""
    resp = fake_response([], status="incomplete", incomplete_reason="max_output_tokens")
    assert from_responses(resp).finish_reason == "length"


def test_a_completed_response_reads_as_stop() -> None:
    assert from_responses(fake_response([])).finish_reason == "stop"


def test_an_unfamiliar_incomplete_reason_is_passed_through_not_mistaken_for_length() -> None:
    resp = fake_response([], status="incomplete", incomplete_reason="content_filter")
    assert from_responses(resp).finish_reason == "content_filter"
