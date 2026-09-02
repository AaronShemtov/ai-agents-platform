"""Azure AI Foundry, over both endpoints it exposes.

Uses the plain `openai` SDK (`AsyncOpenAI`), not `AzureOpenAI`, pointed at

    https://<resource>.services.ai.azure.com/openai/v1/

That route uses implicit versioning — no `api-version` — and the **deployment name goes
in the `model` field**, which is the whole reason switching models here is a config
change. The older `/models` path (Azure AI Inference SDK) is retired; the host is fine,
it is the path that went away.

Two endpoints, not one
----------------------
Deployments do not all answer on the same endpoint, and the split is not a preference.
Measured on this resource on 2026-08-31:

    gpt-5-mini, gpt-5.6-sol / terra / luna, o3   ->  /chat/completions  (and /responses)
    gpt-5.1-codex-max, gpt-5.3-codex             ->  /responses ONLY

The codex line rejects /chat/completions with HTTP 400 "The requested operation is
unsupported", and both Microsoft and OpenAI document that as a property of the family
across four generations — not an Azure gap waiting to be closed. Since those are the
models actually built for writing code (gpt-5.3-codex produced a correct Dockerfile in
4.0s where gpt-5-mini took 28.8s and got a requirement wrong), speaking Responses is the
price of putting the right model on the coder role.

Which deployments need Responses is configuration — MODELS_RESPONSES_API — rather than a
guess from the model name, so a new deployment is a ConfigMap edit, not a release.

Reasoning items
---------------
The awkward part. A reasoning model returns `reasoning` items alongside its
`function_call`, and those must be handed back on the next request together with the
tool results. Drop them and the model loses its own chain of thought between steps; some
models refuse a `function_call_output` whose matching reasoning item is missing.

Our transcript is in Chat Completions shape, so the provider's raw output items ride
along on the assistant message under `_raw_output` and are replayed verbatim. The key is
stripped before anything reaches /chat/completions, which rejects unknown fields — and
that matters, because /model can switch a chat between the two endpoints mid-conversation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import APIError, AsyncOpenAI

from agentcore.llm.base import LLMError, LLMResponse, ToolCall, Usage

logger = logging.getLogger(__name__)

# Key under which provider-native output items ride along in our own transcript.
RAW_OUTPUT_KEY = "_raw_output"


class AzureFoundryClient:
    """Azure AI Foundry, and anything else speaking the same wire protocol.

    The name says who the main user is, not what the class is limited to: everything
    below is the OpenAI wire protocol, so a self-hosted Ollama at <host>:11434/v1/ is
    served by this same class with a different base_url and an empty `responses_models`
    — Ollama implements /chat/completions and not /responses. The Azure-specific
    knowledge is in the module docstring above, which is why it stays in this file.

    `label` only names the settings in the two startup errors, so a misconfigured
    Ollama does not report a missing AZURE_OPENAI_BASE_URL.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        responses_models: set[str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        label: str = "AZURE_OPENAI",
    ) -> None:
        if not base_url:
            raise LLMError(f"{label}_BASE_URL is not set")
        if not api_key:
            raise LLMError(f"{label}_API_KEY is not set")
        self._responses_models = responses_models or set()
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers or None,
            timeout=timeout,
            max_retries=max_retries,
        )

    def uses_responses(self, model: str) -> bool:
        return model in self._responses_models

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self.uses_responses(model):
            return await self._complete_responses(model=model, messages=messages, tools=tools)
        return await self._complete_chat(model=model, messages=messages, tools=tools)

    # -- /chat/completions ---------------------------------------------------

    async def _complete_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": model, "messages": strip_internal(messages)}
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
        out_details = getattr(resp.usage, "completion_tokens_details", None)
        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            # Billed as output and invisible in the reply, so without this the cost of a
            # reasoning model cannot be told apart from the cost of a chatty one.
            reasoning_tokens=getattr(out_details, "reasoning_tokens", 0) or 0,
        )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason or "",
            raw_message=message.model_dump(exclude_none=True),
        )

    # -- /responses ----------------------------------------------------------

    async def _complete_responses(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        instructions, items = to_responses_input(messages)
        kwargs: dict[str, Any] = {"model": model, "input": items}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = to_responses_tools(tools)
            kwargs["tool_choice"] = "auto"

        try:
            resp = await self._client.responses.create(**kwargs)
        except APIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        return from_responses(resp)


# -- transcript translation --------------------------------------------------


def strip_internal(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop our own bookkeeping keys before a message goes to the provider.

    /chat/completions rejects unknown fields, and a chat can be carrying `_raw_output`
    from an earlier turn on a Responses model if /model was used to switch.
    """
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def to_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten Chat-Completions tool specs into the Responses shape.

    Chat nests the definition under "function"; Responses puts name, description and
    parameters at the top level of the tool object.
    """
    flattened: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function") or {}
        flattened.append(
            {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return flattened


def to_responses_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert our Chat-shaped transcript into (instructions, input items).

    The system prompt becomes `instructions` rather than an input item — it is the
    stable cached prefix and Responses treats it as such.

    Where an assistant turn carries `_raw_output`, those provider-native items are
    replayed verbatim instead of being rebuilt. That is what preserves reasoning items
    across steps; rebuilding from our own fields would silently drop them.
    """
    instructions = ""
    items: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")

        if role == "system":
            instructions = message.get("content") or ""
            continue

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": message.get("content") or "",
                }
            )
            continue

        if role == "assistant":
            raw = message.get(RAW_OUTPUT_KEY)
            if raw:
                items.extend(raw)
                continue
            # No native items: this turn came from /chat/completions, so there are no
            # reasoning items to lose. Rebuild what the shape allows.
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        content = message.get("content")
        if content:
            items.append({"role": role or "user", "content": content})

    return instructions, items


def from_responses(resp: Any) -> LLMResponse:
    """Parse a Responses reply into the provider-neutral shape."""
    output = list(getattr(resp, "output", None) or [])

    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    raw_items: list[dict[str, Any]] = []

    for item in output:
        kind = getattr(item, "type", None) if not isinstance(item, dict) else item.get("type")
        raw_items.append(_dump(item))

        if kind == "function_call":
            tool_calls.append(
                ToolCall(
                    id=_attr(item, "call_id") or "",
                    name=_attr(item, "name") or "",
                    arguments=_parse_arguments(_attr(item, "arguments"), _attr(item, "name") or "?"),
                )
            )
        elif kind == "message":
            for block in _attr(item, "content") or []:
                text = _attr(block, "text")
                if text:
                    text_parts.append(text)

    usage_obj = getattr(resp, "usage", None)
    in_details = getattr(usage_obj, "input_tokens_details", None)
    out_details = getattr(usage_obj, "output_tokens_details", None)
    usage = Usage(
        prompt_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
        cached_tokens=getattr(in_details, "cached_tokens", 0) or 0,
        reasoning_tokens=getattr(out_details, "reasoning_tokens", 0) or 0,
    )

    return LLMResponse(
        content="\n".join(text_parts) or None,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=_finish_reason(resp),
        # Carried into the transcript so the next request can replay reasoning items.
        raw_message={RAW_OUTPUT_KEY: raw_items},
    )


def _finish_reason(resp: Any) -> str:
    """Map Responses' status onto the finish_reason the loop already understands.

    The loop stops on "length" so it never acts on a truncated tool call, so a response
    cut short by max_output_tokens has to arrive under that name.
    """
    status = getattr(resp, "status", None)
    if status == "incomplete":
        detail = getattr(resp, "incomplete_details", None)
        reason = getattr(detail, "reason", None) or ""
        return "length" if "token" in reason else (reason or "incomplete")
    return "stop"


def _attr(obj: Any, name: str) -> Any:
    """Read a field from either a provider object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _dump(item: Any) -> dict[str, Any]:
    """Provider objects back to plain dicts, so they can be replayed as input."""
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return {"type": getattr(item, "type", "unknown")}


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
