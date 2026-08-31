"""Shared skeleton for the MCP servers we write ourselves.

`mcp_cloudflare` and `mcp_cluster` both boot through `serve()` here, so transport
setup, health endpoints and error shaping live in exactly one place.
"""

from mcp_common.errors import ToolError, tool_error
from mcp_common.server import build_server, serve

__all__ = ["ToolError", "build_server", "serve", "tool_error"]
