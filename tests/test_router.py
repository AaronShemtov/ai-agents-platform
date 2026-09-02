"""Model-to-backend routing, and the misconfigurations Settings refuses up front."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agentcore.config import Settings
from agentcore.llm.azure import AzureFoundryClient
from agentcore.llm.base import LLMResponse
from agentcore.llm.router import USER_AGENT, Backend, ModelRouter, build_llm

AZURE = {
    "azure_openai_base_url": "https://res.services.ai.azure.com/openai/v1/",
    "azure_openai_api_key": "k",
}


class RecordingClient:
    """Stands in for a provider and remembers what it was asked for."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append(model)
        return LLMResponse(content=self.name)


# -- routing -----------------------------------------------------------------


async def test_a_listed_model_goes_to_its_backend() -> None:
    hosted, local = RecordingClient("hosted"), RecordingClient("local")
    router = ModelRouter(
        default=Backend(label="azure", client=hosted),
        extra=[Backend(label="ollama", client=local, models=frozenset({"qwen3:4b"}))],
    )

    answer = await router.complete(model="qwen3:4b", messages=[])

    assert answer.content == "local"
    assert local.calls == ["qwen3:4b"]
    assert hosted.calls == []


async def test_an_unlisted_model_falls_back_to_the_default() -> None:
    hosted, local = RecordingClient("hosted"), RecordingClient("local")
    router = ModelRouter(
        default=Backend(label="azure", client=hosted),
        extra=[Backend(label="ollama", client=local, models=frozenset({"qwen3:4b"}))],
    )

    answer = await router.complete(model="gpt-5.6-sol", messages=[])

    assert answer.content == "hosted"
    assert local.calls == []


def test_the_backend_can_be_named_without_calling_it() -> None:
    """The label exists so a log line can say where a turn actually ran."""
    router = ModelRouter(
        default=Backend(label="azure", client=RecordingClient("hosted")),
        extra=[
            Backend(label="ollama", client=RecordingClient("local"), models=frozenset({"qwen3:4b"}))
        ],
    )
    assert router.backend_for("qwen3:4b").label == "ollama"
    assert router.backend_for("gpt-5.6-sol").label == "azure"


# -- construction ------------------------------------------------------------


def test_without_ollama_the_call_path_is_unchanged() -> None:
    """No router at all when there is nothing to route — not a router of one."""
    llm = build_llm(Settings(**AZURE))
    assert isinstance(llm, AzureFoundryClient)


def test_with_ollama_configured_a_router_is_built() -> None:
    llm = build_llm(
        Settings(**AZURE, ollama_base_url="https://ollama.example/v1", models_ollama="qwen3:4b")
    )
    assert isinstance(llm, ModelRouter)
    assert llm.backend_for("qwen3:4b").label == "ollama"


def test_the_local_backend_is_never_asked_to_speak_responses() -> None:
    """Ollama implements /chat/completions only; an empty set is what keeps it there."""
    llm = build_llm(
        Settings(
            **AZURE,
            ollama_base_url="https://ollama.example/v1",
            models_ollama="qwen3:4b",
            models_responses_api="gpt-5.3-codex",
        )
    )
    assert isinstance(llm, ModelRouter)
    local = llm.backend_for("qwen3:4b").client
    assert isinstance(local, AzureFoundryClient)
    assert not local.uses_responses("qwen3:4b")


# -- refused configurations --------------------------------------------------


def test_local_models_without_an_endpoint_are_refused() -> None:
    """Otherwise the name reaches Azure and comes back as an unknown deployment."""
    with pytest.raises(ValidationError, match="OLLAMA_BASE_URL"):
        Settings(**AZURE, models_ollama="qwen3:4b")


def test_a_model_cannot_be_both_local_and_responses_only() -> None:
    with pytest.raises(ValidationError, match="chat/completions"):
        Settings(
            **AZURE,
            ollama_base_url="https://ollama.example/v1",
            models_ollama="qwen3:4b",
            models_responses_api="qwen3:4b,gpt-5.3-codex",
        )


# -- the gate in front of Ollama ---------------------------------------------


def test_the_shared_secret_becomes_a_header() -> None:
    settings = Settings(**AZURE, ollama_base_url="https://x/v1", ollama_auth_token="s3cret")
    assert settings.ollama_request_headers() == {"X-Agent-Key": "s3cret"}


def test_no_token_means_no_header() -> None:
    """An empty value must not be sent — the gate would refuse it and the cause would
    read as a network problem rather than a missing secret."""
    settings = Settings(**AZURE, ollama_base_url="https://x/v1")
    assert settings.ollama_request_headers() == {}


def test_the_header_name_is_configurable() -> None:
    settings = Settings(
        **AZURE,
        ollama_base_url="https://x/v1",
        ollama_auth_header="X-Other",
        ollama_auth_token="s3cret",
    )
    assert settings.ollama_request_headers() == {"X-Other": "s3cret"}


def test_the_pair_and_the_dict_are_merged() -> None:
    """A Cloudflare Access token needs two headers; a shared secret needs one."""
    settings = Settings(
        **AZURE,
        ollama_base_url="https://x/v1",
        ollama_headers={"CF-Access-Client-Id": "abc.access"},
        ollama_auth_token="s3cret",
    )
    assert settings.ollama_request_headers() == {
        "CF-Access-Client-Id": "abc.access",
        "X-Agent-Key": "s3cret",
    }


def test_a_real_token_wins_over_a_blank_one_in_the_dict() -> None:
    settings = Settings(
        **AZURE,
        ollama_base_url="https://x/v1",
        ollama_headers={"X-Agent-Key": ""},
        ollama_auth_token="s3cret",
    )
    assert settings.ollama_request_headers()["X-Agent-Key"] == "s3cret"


def test_the_local_backend_identifies_itself() -> None:
    """Not as the OpenAI SDK — Cloudflare blocks that User-Agent outright, and it is
    not what is talking anyway."""
    llm = build_llm(
        Settings(**AZURE, ollama_base_url="https://x/v1", models_ollama="qwen3.5:0.8b")
    )
    assert isinstance(llm, ModelRouter)
    local = llm.backend_for("qwen3.5:0.8b").client
    sent = local._client._custom_headers or {}  # type: ignore[attr-defined]
    assert sent.get("User-Agent") == USER_AGENT
    assert "OpenAI" not in sent.get("User-Agent", "")


def test_the_secret_survives_the_user_agent() -> None:
    """Both are set in one dict literal; an ordering slip would drop the secret."""
    llm = build_llm(
        Settings(
            **AZURE,
            ollama_base_url="https://x/v1",
            models_ollama="qwen3.5:0.8b",
            ollama_auth_token="s3cret",
        )
    )
    assert isinstance(llm, ModelRouter)
    sent = llm.backend_for("qwen3.5:0.8b").client._client._custom_headers  # type: ignore[attr-defined]
    assert sent.get("X-Agent-Key") == "s3cret"
    assert sent.get("User-Agent") == USER_AGENT


def test_the_client_is_built_with_those_headers() -> None:
    llm = build_llm(
        Settings(
            **AZURE,
            ollama_base_url="https://x/v1",
            models_ollama="qwen3.5:0.8b",
            ollama_auth_token="s3cret",
        )
    )
    assert isinstance(llm, ModelRouter)
    local = llm.backend_for("qwen3.5:0.8b").client
    sent = local._client._custom_headers or {}  # type: ignore[attr-defined]
    assert sent.get("X-Agent-Key") == "s3cret"


def test_headers_are_json_because_a_secret_may_contain_a_comma() -> None:
    """The gate in front of Ollama needs credentials; JSON keeps them intact."""
    settings = Settings(
        **AZURE,
        ollama_base_url="https://ollama.example/v1",
        ollama_headers={"CF-Access-Client-Id": "abc.access", "CF-Access-Client-Secret": "s,e=c"},
    )
    assert settings.ollama_headers["CF-Access-Client-Secret"] == "s,e=c"


# -- thinking ----------------------------------------------------------------
#
# These reach into _extra_params on purpose. What is being pinned is that the parameter
# is sent at all: it is the only way to stop a reasoning model on Ollama from spending a
# minute per step, and nothing else in the suite would notice if it silently stopped
# being passed.


def test_a_local_reasoning_model_is_told_not_to_think() -> None:
    llm = build_llm(
        Settings(**AZURE, ollama_base_url="https://ollama.example/v1", models_ollama="qwen3.5:0.8b")
    )
    assert isinstance(llm, ModelRouter)
    local = llm.backend_for("qwen3.5:0.8b").client
    assert local._extra_params == {"reasoning_effort": "none"}  # type: ignore[attr-defined]


def test_thinking_can_be_turned_back_on_per_deployment() -> None:
    llm = build_llm(
        Settings(
            **AZURE,
            ollama_base_url="https://ollama.example/v1",
            models_ollama="qwen3.5:0.8b",
            ollama_reasoning_effort="low",
        )
    )
    assert isinstance(llm, ModelRouter)
    local = llm.backend_for("qwen3.5:0.8b").client
    assert local._extra_params == {"reasoning_effort": "low"}  # type: ignore[attr-defined]


def test_an_empty_effort_sends_nothing_at_all() -> None:
    """Not the string "none" — no key, so the model's own default stands."""
    llm = build_llm(
        Settings(
            **AZURE,
            ollama_base_url="https://ollama.example/v1",
            models_ollama="qwen3.5:0.8b",
            ollama_reasoning_effort="",
        )
    )
    assert isinstance(llm, ModelRouter)
    local = llm.backend_for("qwen3.5:0.8b").client
    assert local._extra_params == {}  # type: ignore[attr-defined]


def test_azure_requests_are_left_alone() -> None:
    """Ollama's knob must not follow the hosted models around."""
    llm = build_llm(Settings(**AZURE))
    assert isinstance(llm, AzureFoundryClient)
    assert llm._extra_params == {}  # type: ignore[attr-defined]


def test_the_endpoint_always_ends_in_a_slash() -> None:
    settings = Settings(**AZURE, ollama_base_url="http://box:11434/v1")
    assert settings.ollama_url() == "http://box:11434/v1/"
    assert Settings(**AZURE).ollama_url() == ""
