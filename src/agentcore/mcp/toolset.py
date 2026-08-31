"""Translate MCP tools into OpenAI/Azure tool definitions.

Two limits here are not cosmetic — violating either makes Azure reject the whole
request with a 400, which looks like "the model is broken" and is tedious to trace
back to one over-long tool description:

  * a tool/function description may be at most 1024 characters
  * a tool name must match ^[a-zA-Z0-9_-]{1,64}$

The GitHub MCP server ships descriptions well over 1024 characters, so truncation is
mandatory, not defensive. Names are namespaced (`github__create_or_update_file`),
which eats 8-11 characters of the 64 before the server's own name is even counted.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from agentcore.mcp.client import RemoteTool

logger = logging.getLogger(__name__)

MAX_DESCRIPTION = 1024
MAX_NAME = 64
_ELLIPSIS = "…"


def truncate_description(text: str, limit: int = MAX_DESCRIPTION) -> str:
    """Trim to `limit` characters, preferring a word boundary."""
    text = " ".join(text.split())  # collapse the multi-line prose MCP servers ship
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    # Only back off to a space if that does not throw away most of the text.
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut + _ELLIPSIS


def sanitize_name(name: str) -> str:
    """Force a name into ^[a-zA-Z0-9_-]{1,64}$, keeping it recognisable."""
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)
    if not cleaned:
        cleaned = "tool"
    if len(cleaned) <= MAX_NAME:
        return cleaned
    # Keep the readable head, append a short digest so distinct long names stay distinct.
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]  # sha1 as a short disambiguator, not a security control
    return cleaned[: MAX_NAME - 7] + "_" + digest


@dataclass(frozen=True)
class ToolCatalog:
    """OpenAI tool specs plus the mapping back to MCP qualified names."""

    specs: list[dict[str, Any]]
    _by_openai_name: dict[str, str]
    skipped: list[str]

    def resolve(self, openai_name: str) -> str | None:
        """Map the name the model called back to `<server>__<tool>`."""
        return self._by_openai_name.get(openai_name)

    def __len__(self) -> int:
        return len(self.specs)

    def names(self) -> list[str]:
        return [s["function"]["name"] for s in self.specs]


def build_catalog(
    tools: list[RemoteTool],
    *,
    is_allowed: object = None,
) -> ToolCatalog:
    """Build the catalog handed to the LLM.

    `is_allowed` is an optional callable taking the qualified tool name; it is where
    the agent profile's allow/deny globs are applied. Filtering happens here, before
    the model ever sees a tool, so a denied tool cannot be called at all rather than
    being refused after the fact.
    """
    specs: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    skipped: list[str] = []
    used: set[str] = set()

    for tool in tools:
        if callable(is_allowed) and not is_allowed(tool.qualified_name):
            skipped.append(tool.qualified_name)
            continue

        name = sanitize_name(tool.qualified_name)
        if name in used:
            # Two different tools collapsed onto one name after sanitising. Rare, but
            # silently dropping one would make a tool mysteriously uncallable.
            digest = hashlib.sha1(tool.qualified_name.encode()).hexdigest()[:6]  # short disambiguator only
            name = f"{name[: MAX_NAME - 7]}_{digest}"
        used.add(name)

        mapping[name] = tool.qualified_name
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": truncate_description(tool.description),
                    "parameters": _normalise_schema(tool.input_schema),
                },
            }
        )

    if skipped:
        logger.info("tool catalog: %d allowed, %d filtered by profile", len(specs), len(skipped))
    return ToolCatalog(specs=specs, _by_openai_name=mapping, skipped=skipped)


def _normalise_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make sure the schema is something the API will accept as `parameters`."""
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    normalised = dict(schema)
    normalised.setdefault("type", "object")
    if normalised["type"] == "object":
        normalised.setdefault("properties", {})
    return normalised
