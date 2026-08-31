# `x-mcp-header`: tool parameters that must also travel as HTTP headers

Diagnosed 2026-08-31 against `ghcr.io/github/github-mcp-server:v1.11.0` and
`mcp` Python SDK 2.1.1.

## Symptom

Some GitHub tool calls failed and some succeeded, with nothing obviously different
about them:

| tool | arguments the model sent | result |
|---|---|---|
| `search_repositories` | `query, page, perPage, minimal_output` | worked |
| `list_branches` | `owner, repo, perPage` | failed |
| `get_file_contents` | `owner, repo, path, ref` | failed |
| `create_branch` | `owner, repo, branch, from_branch` | failed |

In the pod log the shape was always the same — `initialize` returned 200, the next
POST returned **400 Bad Request** — and the agent reported only:

    tool transport error: unhandled errors in a TaskGroup (1 sub-exception)

That string is what `str()` of an anyio `ExceptionGroup` produces. The real cause sits
in `.exceptions` and was being thrown away.

## Cause

Unwrapping the group gives:

    MCPError: header mismatch: missing Mcp-Param-owner header for parameter "owner"

The server marks individual parameters in its JSON Schema:

```json
"owner": { "type": "string", "description": "Repository owner", "x-mcp-header": "owner" },
"repo":  { "type": "string", "description": "Repository name",  "x-mcp-header": "repo"  }
```

A parameter carrying `x-mcp-header: <name>` must be sent **both** in the JSON-RPC
`arguments` object **and** as an HTTP header `Mcp-Param-<name>`. The point is that an
intermediary — a gateway, a proxy, an audit layer — can then see *which repository* a
call touches without parsing the request body, and allow or refuse it there.

`search_repositories` declares no `x-mcp-header` on any property, which is exactly why
it was the one tool that worked.

Confirmed by hand from inside the pod: a raw `httpx2.post` of `tools/call` with
`{"owner": ..., "repo": ...}` and **no** such headers returns 200 and real data. So the
400 is not the server rejecting the body — the client-side SDK refuses to send the
request once it notices the mismatch.

## Fix

In `agentcore/mcp/client.py`, at call time:

1. Look up the `RemoteTool` for the qualified name (the pool already caches schemas).
2. Walk `input_schema["properties"]`; collect every property that has `x-mcp-header`
   **and** is present in the arguments for this call.
3. Send `Mcp-Param-<header name>: <value>` alongside the server's base headers.

The long-lived `httpx2.AsyncClient` per server cannot carry these, since they vary per
call — so the client for a call that needs them is built per call. Calls with no
header-bound parameters keep using the pooled client.

## Lesson for the error path

`ToolResult` used to surface `str(exc)`, which for an `ExceptionGroup` names the
container and hides the contents. Transport errors are now flattened recursively
before they reach the log or the model, so the next protocol-level surprise reads as
itself rather than as "unhandled errors in a TaskGroup".
