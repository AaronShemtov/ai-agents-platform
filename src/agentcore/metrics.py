"""Metrics for the agents, following the OpenTelemetry GenAI semantic conventions.

Why these names and not ours
----------------------------
The GenAI conventions are what every LLM observability product consumes, so an agent
that emits them can be pointed at one later without re-instrumenting. Names, units and
bucket boundaries below are copied from the specification rather than chosen, because
the buckets are the part that silently stops matching if you improvise.

Worth knowing what is being adopted, though: as of September 2026 every `gen_ai.*`
attribute and metric still carries the "Development" stability badge, there is no 1.0,
and in June 2026 the whole set moved out of the main semantic-conventions repository
into `open-telemetry/semantic-conventions-genai`. So this is the direction the industry
is pointing, not a settled contract. Expect churn, and prefer the spec's spelling over
inventing our own when a concept exists in both.

OTel names use dots; Prometheus does not allow them. The mapping applied here is the
standard one — dots become underscores, and duration metrics take a `_seconds` suffix.

What the standard does not cover
--------------------------------
Three things this platform does that the conventions have no place for, kept under our
own `agent_` prefix so the two sets never look like one:

  * what local policy decided — the guardrail is the point of an agent with write access
  * how long a human took to approve, and whether they did
  * cached and reasoning tokens, added as extra `gen_ai.token.type` values

Nothing here records what anyone said. Arguments stay hashed in the audit log; this
module never sees them.

Cardinality
-----------
Series count is the product of label value counts, and every series is paid for until
it falls out of retention. The expensive one is `gen_ai_execute_tool_duration_seconds`:
the spec requires the tool name, and 59 tools times fourteen buckets times two agents is
about 1650 series if every tool is used. In practice a handful are. Measured against a
head block holding 31,617 series, that is affordable — but it is the line to watch if
the catalog grows.

The one genuinely unbounded input is the tool name, because a model can invent one.
`_tool_label` bounds it.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Far above the real catalog (59 today), far below anything that would hurt.
MAX_TOOL_LABELS = 128
_seen_tools: set[str] = set()

# Bucket boundaries are the specification's, verbatim. They look arbitrary because they
# are shared: matching them is what lets a dashboard or a backend built for the
# convention read these histograms without being told anything.
_TOKEN_BUCKETS = (
    1, 4, 16, 64, 256, 1024, 4096, 16384, 65536,
    262144, 1048576, 4194304, 16777216, 67108864,
)
_DURATION_BUCKETS = (
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28,
    2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
)
_AGENT_DURATION_BUCKETS = (
    0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6,
)
_CALL_COUNT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)

# -- the convention ----------------------------------------------------------

TOKEN_USAGE = Histogram(
    "gen_ai_client_token_usage",
    "Tokens used by a model call, by kind.",
    ["gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model", "gen_ai_token_type"],
    buckets=_TOKEN_BUCKETS,
)

OPERATION_DURATION = Histogram(
    "gen_ai_client_operation_duration_seconds",
    "Duration of a single model call.",
    ["gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model", "error_type"],
    buckets=_DURATION_BUCKETS,
)

AGENT_DURATION = Histogram(
    "gen_ai_invoke_agent_duration_seconds",
    "Duration of a whole agent turn, from the message to the answer.",
    ["gen_ai_agent_name", "gen_ai_request_model", "error_type"],
    buckets=_AGENT_DURATION_BUCKETS,
)

AGENT_INFERENCE_CALLS = Histogram(
    "gen_ai_invoke_agent_inference_calls",
    "Model calls made during one agent turn.",
    ["gen_ai_agent_name", "gen_ai_request_model"],
    buckets=_CALL_COUNT_BUCKETS,
)

AGENT_TOOL_CALLS = Histogram(
    "gen_ai_invoke_agent_tool_calls",
    "Tool calls made during one agent turn.",
    ["gen_ai_agent_name", "gen_ai_request_model"],
    buckets=_CALL_COUNT_BUCKETS,
)

EXECUTE_TOOL_DURATION = Histogram(
    "gen_ai_execute_tool_duration_seconds",
    "Duration of one tool call that was actually made.",
    ["gen_ai_agent_name", "gen_ai_tool_name", "error_type"],
    buckets=_DURATION_BUCKETS,
)

# -- our own, for what the convention has no place for ------------------------

POLICY_DECISIONS = Counter(
    "agent_policy_decisions_total",
    "Tool calls by what local policy decided. The whole reason an agent with write "
    "access is safe to run, so it is counted separately rather than folded into an "
    "error rate: a refusal is a correct outcome, not a failure.",
    ["agent", "tool", "decision"],
)

APPROVALS = Counter(
    "agent_approvals_total",
    "Confirmations asked of a human, by what they answered.",
    ["agent", "tool", "outcome"],
)

APPROVAL_WAIT = Histogram(
    "agent_approval_wait_seconds",
    "How long the agent waited for a human to answer a confirmation.",
    ["agent", "tool"],
    # Human timescales, not machine ones: a person glances at a phone in seconds or
    # gets to it in minutes, and the tail matters more than the middle.
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600),
)

TURNS = Counter(
    "agent_turns_total",
    "Completed turns by why the loop stopped. Complements the convention's duration "
    "histogram, which has no place to say whether a turn ran out of steps or budget.",
    ["agent", "model", "stopped_because"],
)


def _tool_label(name: str) -> str:
    """Bound the tool label, because the name can come from the model.

    A hallucinated tool name arrives here exactly like a real one. Uncapped, every
    invention would mint a permanent time series — the standard way a Prometheus
    deployment dies. Past the cap, unseen names collapse to "other": the totals stay
    right, only the breakdown stops growing.
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


def record_llm_call(
    *,
    provider: str,
    model: str,
    duration_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
    error_type: str = "",
) -> None:
    """One model call. `error_type` is empty on success, per the convention."""
    OPERATION_DURATION.labels(
        gen_ai_operation_name="chat",
        gen_ai_provider_name=provider,
        gen_ai_request_model=model,
        error_type=error_type,
    ).observe(duration_seconds)

    if error_type:
        return

    for token_type, value in (
        # The two the specification defines.
        ("input", prompt_tokens),
        ("output", completion_tokens),
        # Ours. `cached` is the one that decides what a turn costs — a cached prompt
        # token is billed at roughly a tenth — and the convention has nowhere to put it.
        ("cached", cached_tokens),
        ("reasoning", reasoning_tokens),
    ):
        if value:
            TOKEN_USAGE.labels(
                gen_ai_operation_name="chat",
                gen_ai_provider_name=provider,
                gen_ai_request_model=model,
                gen_ai_token_type=token_type,
            ).observe(value)


def record_tool_call(
    *,
    agent: str,
    tool: str,
    decision: str,
    ok: bool | None,
    duration_ms: int | None,
) -> None:
    """One attempted tool call. `ok` is None when nothing was sent."""
    POLICY_DECISIONS.labels(agent=agent, tool=_tool_label(tool), decision=decision).inc()
    if duration_ms is None:
        # Denied, rejected, or invented: there was no call to time.
        return
    EXECUTE_TOOL_DURATION.labels(
        gen_ai_agent_name=agent,
        gen_ai_tool_name=_tool_label(tool),
        error_type="" if ok else "tool_error",
    ).observe(duration_ms / 1000)


def record_approval(*, agent: str, tool: str, outcome: str, waited_seconds: float) -> None:
    """One confirmation put to a human. `outcome` is approved or rejected."""
    APPROVALS.labels(agent=agent, tool=_tool_label(tool), outcome=outcome).inc()
    APPROVAL_WAIT.labels(agent=agent, tool=_tool_label(tool)).observe(waited_seconds)


def record_turn(
    *,
    agent: str,
    model: str,
    steps: int,
    tool_calls: int,
    duration_ms: int,
    stopped_because: str,
) -> None:
    """One completed turn, however it ended.

    Tokens are not recorded here: they belong to the model calls that consumed them, and
    counting them twice would make any sum wrong.
    """
    TURNS.labels(agent=agent, model=model, stopped_because=stopped_because).inc()
    AGENT_DURATION.labels(
        gen_ai_agent_name=agent,
        gen_ai_request_model=model,
        # A turn that hit max_steps or the spend ceiling did not error — it was stopped
        # on purpose. Only a model or transport failure is an error here.
        error_type="llm_error" if stopped_because == "llm_error" else "",
    ).observe(duration_ms / 1000)
    AGENT_INFERENCE_CALLS.labels(gen_ai_agent_name=agent, gen_ai_request_model=model).observe(steps)
    AGENT_TOOL_CALLS.labels(gen_ai_agent_name=agent, gen_ai_request_model=model).observe(tool_calls)


def render() -> tuple[bytes, str]:
    """The exposition payload and its content type, for whatever serves /metrics."""
    return generate_latest(), CONTENT_TYPE_LATEST
