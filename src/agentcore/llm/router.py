"""Which backend serves which model.

The loop asks for a model by name and does not care where it runs. Keeping that
decision here means a second provider is a table entry rather than a branch inside the
loop — which was the point of the LLMClient protocol in the first place.

Routing is by exact model name, taken from configuration rather than guessed from the
name. Guessing would be wrong the first time a local deployment is called something that
looks like a hosted one, and a wrong guess here surfaces as a 404 from the wrong
provider.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentcore.config import Settings
from agentcore.llm.azure import AzureFoundryClient
from agentcore.llm.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Backend:
    """One place models can run. `models` is empty for the fallback backend."""

    label: str
    client: LLMClient
    models: frozenset[str] = frozenset()


class ModelRouter:
    """Sends each `complete()` to the backend that serves the requested model."""

    def __init__(self, *, default: Backend, extra: Sequence[Backend] = ()) -> None:
        self._default = default
        self._by_model: dict[str, Backend] = {
            model: backend for backend in extra for model in backend.models
        }

    def backend_for(self, model: str) -> Backend:
        return self._by_model.get(model, self._default)

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        backend = self.backend_for(model)
        logger.debug("model %s -> backend %s", model, backend.label)
        return await backend.client.complete(model=model, messages=messages, tools=tools)


def build_llm(settings: Settings) -> LLMClient:
    """The LLM client for this process: Azure, and Ollama when it is configured.

    With no Ollama models configured this returns the Azure client itself rather than a
    router wrapping it, so the ordinary deployment keeps exactly the call path it had.
    """
    azure = AzureFoundryClient(
        base_url=settings.base_url(),
        api_key=settings.azure_openai_api_key,
        responses_models=settings.responses_api_models(),
    )

    local = settings.ollama_models()
    if not local:
        return azure

    ollama = Backend(
        label="ollama",
        client=AzureFoundryClient(
            base_url=settings.ollama_url(),
            api_key=settings.ollama_api_key,
            # Empty on purpose: Ollama has no /responses. Settings refuses a model
            # listed in both MODELS_OLLAMA and MODELS_RESPONSES_API, so this cannot
            # silently disagree with configuration.
            responses_models=set(),
            headers=settings.ollama_request_headers(),
            # Empty string means "send nothing and take the model's default", which is
            # how a deployment opts back into thinking.
            extra_params=(
                {"reasoning_effort": settings.ollama_reasoning_effort}
                if settings.ollama_reasoning_effort
                else {}
            ),
            timeout=settings.ollama_timeout_seconds,
            label="OLLAMA",
        ),
        models=frozenset(local),
    )
    logger.info("routing %s to ollama at %s", sorted(local), settings.ollama_base_url)
    return ModelRouter(default=Backend(label="azure", client=azure), extra=[ollama])
