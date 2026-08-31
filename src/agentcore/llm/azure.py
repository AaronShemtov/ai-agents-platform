"""Azure AI Foundry via the OpenAI-compatible endpoint.

Uses the plain `openai` SDK (`AsyncOpenAI`), not `AzureOpenAI`, pointed at

    https://<resource>.openai.azure.com/openai/v1/

That route uses implicit versioning, so no `api-version` query parameter, and the
**deployment name goes in the `model` field**. That is the whole reason model
switching is a config change here: nothing else in the code knows which model it is.

The older `<resource>.services.ai.azure.com/models` endpoint (Azure AI Inference SDK)
is retired — do not migrate back to it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import APIError, AsyncOpenAI

from agentcore.llm.base import LLMError, LLMResponse, ToolCall, Usage

logger = logging.getLogger(__name__)


class AzureFoundryClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        if not base_url:
            raise LLMError("AZURE_OPENAI_BASE_URL is not set")
        if not api_key:
            raise LLMError("AZURE_OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        # An empty `tools` array is rejected by the API — omit the key entirely when
        # the agent has no tools rather than sending [].
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except APIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        choice = resp.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=_parse_arguments(call.function.arguments, call.function.name),
                )
            )

        # prompt_tokens_details.cached_tokens is present on Azure but absent on some
        # deployments and older API shapes, hence the defensive getattr chain.
        details = getattr(resp.usage, "prompt_tokens_details", None)
        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
        )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "",
            raw_message=message.model_dump(exclude_none=True),
        )


def _parse_arguments(raw: str | None, tool_name: str) -> dict[str, Any]:
    """Tool arguments arrive as a JSON *string* and models do occasionally emit junk.

    A malformed blob becomes an empty dict; the tool then fails its own schema
    validation and the model gets a readable error it can correct, which is far better
    than crashing the turn.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("tool %s: arguments were not valid JSON: %r", tool_name, raw[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}
