"""Provider-side tools — web search above all.

These are not in the tool catalogue and the loop never dispatches them: Azure
runs them and returns the result inside its own output items. So what is worth
testing is the boundary — that they reach /responses alongside our functions,
that they never reach /chat/completions, and that an output item we do not
recognise neither breaks the parse nor gets dropped from the transcript.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agentcore.config import Settings
from agentcore.llm.azure import RAW_OUTPUT_KEY, AzureFoundryClient, from_responses

WEB = {"type": "web_search_preview"}

FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "cluster__list_pods",
        "description": "List pods",
        "parameters": {"type": "object", "properties": {}},
    },
}


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output=[], usage=None, status="completed")


class FakeChat:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )


def _client(hosted, responses_models):
    c = AzureFoundryClient(
        base_url="https://example.invalid/openai/v1/",
        api_key="k",
        responses_models=responses_models,
        hosted_tools=hosted,
    )
    responses, chat = FakeResponses(), FakeChat()
    c._client = SimpleNamespace(
        responses=responses,
        chat=SimpleNamespace(completions=chat),
    )
    return c, responses, chat


def _run(c, model, tools):
    return asyncio.run(
        c.complete(model=model, messages=[{"role": "user", "content": "hi"}], tools=tools)
    )


# -- configuration -----------------------------------------------------------


def test_hosted_tools_are_parsed_into_type_objects():
    s = Settings(hosted_tools="web_search_preview, code_interpreter")
    assert s.hosted_tool_specs() == [
        {"type": "web_search_preview"},
        {"type": "code_interpreter"},
    ]


def test_no_hosted_tools_by_default():
    assert Settings().hosted_tool_specs() == []


# -- the responses path ------------------------------------------------------


def test_hosted_tools_ride_alongside_our_functions():
    c, responses, _ = _client([WEB], {"gpt-6-astra"})
    _run(c, "gpt-6-astra", [FUNCTION_TOOL])

    sent = responses.kwargs["tools"]
    assert {"type": "web_search_preview"} in sent
    # Ours is still there, flattened into the Responses shape.
    assert any(t.get("name") == "cluster__list_pods" for t in sent)
    assert responses.kwargs["tool_choice"] == "auto"


def test_a_hosted_tool_is_sent_even_with_no_functions_of_our_own():
    """An agent with an empty catalogue can still be asked to look something up."""
    c, responses, _ = _client([WEB], {"gpt-6-astra"})
    _run(c, "gpt-6-astra", None)
    assert responses.kwargs["tools"] == [WEB]


def test_nothing_is_sent_when_there_is_nothing_to_send():
    """An empty tools array is rejected by the API, so the key must be absent."""
    c, responses, _ = _client([], {"gpt-6-astra"})
    _run(c, "gpt-6-astra", None)
    assert "tools" not in responses.kwargs


# -- the chat path -----------------------------------------------------------


def test_hosted_tools_never_reach_chat_completions():
    """They do not exist on that endpoint, and astra answers 400 to tools there."""
    c, _, chat = _client([WEB], set())  # not a responses model -> chat path
    _run(c, "gpt-5.6-sol", [FUNCTION_TOOL])

    assert chat.kwargs["tools"] == [FUNCTION_TOOL]
    assert WEB not in chat.kwargs["tools"]


# -- parsing what comes back -------------------------------------------------


def test_a_web_search_call_is_carried_but_not_mistaken_for_a_tool_call():
    """The loop must not try to execute it, and the next request must replay it.

    Azure has already run the search; the item exists so the model can see its own
    step. Dropping it from the transcript is what would make the model repeat the
    search on the following turn.
    """
    resp = SimpleNamespace(
        status="completed",
        usage=None,
        output=[
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "reasoning", "id": "rs_1"},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Example Domain"}],
            },
        ],
    )

    parsed = from_responses(resp)

    assert parsed.tool_calls == []
    assert parsed.content == "Example Domain"
    kinds = [item.get("type") for item in parsed.raw_message[RAW_OUTPUT_KEY]]
    assert kinds == ["web_search_call", "reasoning", "message"]
