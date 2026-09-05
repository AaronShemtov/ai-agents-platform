"""Prometheus metrics for the agents.

Emitted from `AuditLog`, right beside the log line describing the same event. That is
the whole design: one call site per fact, so a counter and its log line cannot drift
apart, and the loop needs no instrumentation of its own.

What goes where
---------------
Metrics answer *how many, how fast, how much*; logs answer *what exactly*. The split is
not stylistic — it is forced by how a time-series database stores data. Series count is
the product of label value counts, and every series costs memory and index space for as
long as it is retained. So a tool name is a fine label (there are 59) and a repository
path, a chat id or the text of a question is not.

Nothing here records what anyone said. Arguments stay hashed in the audit log; this
module never sees them.

Cardinality budget
------------------
Worst case with two agents, 59 tools, 7 models: roughly 1200 series, and in practice a
few hundred because only combinations that actually occur are created. Measured against
a head block holding 31,617 series, so this is a percent or two.

The one genuinely unbounded input is the tool name, because a model can invent one. That
is what `_tool_label` is for.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Far above the real catalog (59 today), far below anything that would hurt.
MAX_TOOL_LABELS = 128
_seen_tools: set[str] = set()

TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Tool calls attempted, by what local policy decided and whether the call succeeded.",
    ["agent", "tool", "decision", "ok"],
)

TOOL_DURATION = Histogram(
    "agent_tool_duration_seconds",
    "Wall time of a tool call that was actually made.",
    # Deliberately labelled by server rather than by tool: a per-tool histogram would be
    # 59 tools times a dozen buckets, which is most of the cardinality budget spent on
    # detail nobody reads. The server is what differs — a GitHub call is a second, a
    # delegated coder run is minutes.
    ["agent", "server"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 300, 900),
)

TURNS = Counter(
    "agent_turns_total",
    "Completed turns, by why the loop stopped.",
    ["agent", "model", "stopped_because"],
)

TURN_DURATION = Histogram(
    "agent_turn_duration_seconds",
    "Wall time from the user's message to the answer.",
    ["agent", "model"],
    buckets=(1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600),
)

TURN_STEPS = Histogram(
    "agent_turn_steps",
    "How many model calls a turn took.",
    ["agent", "model"],
    buckets=(1, 2, 3, 5, 8, 13, 21, 30, 40),
)

TOKENS = Counter(
    "agent_tokens_total",
    "Tokens by kind. `billable` is prompt minus cached plus completion — the one that "
    "corresponds to spend, since a cached prompt token costs about a tenth of a fresh one.",
    ["agent", "model", "kind"],
)


def _tool_label(name: str) -> str:
    """Bound the tool label, because the name can come from the model.

    A hallucinated tool name reaches this module the same way a real one does. Without a
    cap, every invention would mint a permanent time series, which is precisely the way
    a Prometheus deployment is destroyed. Once the cap is reached, further unseen names
    collapse into "other" — the count stays correct, only the breakdown stops growing.
    """
    if name in _seen_tools:
        return name
    if len(_seen_tools) < MAX_TOOL_LABELS:
        _seen_tools.add(name)
        return name
    return "other"


def server_of(qualified: str) -> str:
    """The MCP server a tool belongs to: `github__create_branch` -> `github`."""
    return qualified.split("__", 1)[0] if "__" in qualified else "unknown"


def record_tool_call(
    *,
    agent: str,
    tool: str,
    decision: str,
    ok: bool | None,
    duration_ms: int | None,
) -> None:
    """One attempted tool call. `ok` is None when nothing was sent."""
    TOOL_CALLS.labels(
        agent=agent,
        tool=_tool_label(tool),
        decision=decision,
        # A denied call has no outcome to report, and "false" would read as a failure of
        # the tool rather than a refusal to call it.
        ok={True: "true", False: "false", None: "n/a"}[ok],
    ).inc()
    if duration_ms is not None:
        TOOL_DURATION.labels(agent=agent, server=server_of(tool)).observe(duration_ms / 1000)


def record_turn(
    *,
    agent: str,
    model: str,
    steps: int,
    duration_ms: int,
    stopped_because: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
    billable_tokens: int,
) -> None:
    """One completed turn, however it ended."""
    TURNS.labels(agent=agent, model=model, stopped_because=stopped_because).inc()
    TURN_DURATION.labels(agent=agent, model=model).observe(duration_ms / 1000)
    TURN_STEPS.labels(agent=agent, model=model).observe(steps)
    for kind, value in (
        ("prompt", prompt_tokens),
        ("completion", completion_tokens),
        ("cached", cached_tokens),
        ("reasoning", reasoning_tokens),
        ("billable", billable_tokens),
    ):
        if value:
            TOKENS.labels(agent=agent, model=model, kind=kind).inc(value)


def render() -> tuple[bytes, str]:
    """The exposition payload and its content type, for whatever serves /metrics."""
    return generate_latest(), CONTENT_TYPE_LATEST
