"""An agent exposed as an MCP server, so other agents can delegate to it.

Telegram will not carry this. A bot never receives another bot's messages — Telegram
drops them outright to stop two bots looping on each other — so delegation cannot go
through the chat even if every agent had its own bot. It goes over the cluster network
instead, and the natural shape for that here is MCP: the lead already speaks it to
three servers, so a sub-agent that speaks it too arrives as one more tool, subject to
the same policy checks and the same audit log as `github__create_pull_request`.

A worker agent therefore needs no bot and no Telegram token. Only an agent you talk to
directly needs those.
"""
