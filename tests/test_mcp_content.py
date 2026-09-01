"""What a tool's content blocks turn into for the model.

The case that matters here is the embedded resource. The GitHub server answers
get_file_contents with a text block saying the download succeeded and the file itself as
a resource, so a flattener that only reads text blocks gives the model a receipt and no
file — and the model then answers about the file anyway.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

from agentcore.mcp.client import _flatten_content


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def resource_block(**resource: object) -> SimpleNamespace:
    """An embedded resource: the payload hangs off `.resource`, not off the block."""
    return SimpleNamespace(type="resource", resource=SimpleNamespace(**resource))


def test_text_blocks_pass_through() -> None:
    assert _flatten_content([text_block("hello"), text_block("world")]) == "hello\nworld"


def test_a_text_resource_reaches_the_model() -> None:
    blocks = [
        text_block("successfully downloaded text file (SHA: abc123)"),
        resource_block(uri="repo://f.yaml", mimeType="text/yaml", text="model: gpt-5.3-codex"),
    ]
    out = _flatten_content(blocks)
    assert "model: gpt-5.3-codex" in out
    assert "(resource content block)" not in out


def test_a_base64_resource_is_decoded() -> None:
    payload = "line one\nline two"
    blocks = [resource_block(uri="repo://f.txt", blob=base64.b64encode(payload.encode()).decode())]
    assert _flatten_content(blocks) == payload


def test_a_binary_resource_is_described_not_mangled() -> None:
    blob = base64.b64encode(bytes([0xFF, 0xFE, 0x00, 0x01])).decode()
    out = _flatten_content([resource_block(uri="repo://x.png", mimeType="image/png", blob=blob)])
    assert "binary resource" in out
    assert "image/png" in out
    assert "4 bytes" in out


def test_an_undecodable_payload_says_so_rather_than_raising() -> None:
    out = _flatten_content([resource_block(uri="repo://x", blob="not base64 at all!!")])
    assert "undecodable" in out


def test_an_unknown_block_is_still_summarised() -> None:
    """Whatever we cannot read, the model should at least know arrived."""
    out = _flatten_content([SimpleNamespace(type="image", data="...")])
    assert out == "(image content block)"


def test_empty_content_is_empty() -> None:
    assert _flatten_content([]) == ""
    assert _flatten_content(None) == ""
