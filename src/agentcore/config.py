"""Runtime configuration, entirely from environment variables.

Everything is lazy on purpose. In `gemini-sre-agent` the equivalent modules call
`os.environ[...]`, `genai.configure()` and `CoreV1Api()` at *import* time, which makes
them unimportable without a live kubeconfig and a full env — and therefore untestable.
Here nothing touches the environment until `get_settings()` is called, so tests can
import any module and inject their own Settings.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WriteMode(StrEnum):
    """How far a tool namespace is allowed to go without a human in the loop."""

    READ_ONLY = "read-only"
    APPROVE = "approve"  # mutation requires an explicit Telegram confirmation
    AUTO = "auto"  # mutate freely
    DIRECT_PUSH = "direct-push"  # GitHub-specific alias for AUTO
    PULL_REQUEST = "pull-request"  # GitHub-specific: may write, but only via a PR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # `model_default` / `model_allowlist` would otherwise trip pydantic v2's
        # protected `model_` namespace and warn on every construction.
        protected_namespaces=(),
    )

    # NOTE on the comma-separated string fields below.
    #
    # They are deliberately NOT typed as list[...]. pydantic-settings tries to
    # json.loads() the raw environment value for any complex-typed field *before*
    # field validators run, so `TELEGRAM_ALLOWED_USERS=139236889` or
    # `MODEL_ALLOWLIST=gpt-5-mini,gpt-5` — which is how a ConfigMap supplies them —
    # would fail at startup and no `mode="before"` validator would ever be reached.
    # Keeping them as strings and parsing in accessors sidesteps that entirely.

    # --- which agent this process is ---------------------------------------
    agent_profile: str = "lead"
    agents_dir: Path = Path("agents")

    # --- LLM (Azure AI Foundry via the OpenAI-compatible endpoint) ----------
    # NOTE: must be the /openai/v1/ route. The older .services.ai.azure.com/models
    # endpoint (Azure AI Inference SDK) is retired.
    azure_openai_base_url: str = ""
    azure_openai_api_key: str = ""
    model_default: str = "gpt-5-mini"
    model_allowlist: str = ""  # comma-separated
    # Deployments that answer only on /responses. Configuration rather than a guess
    # from the model name, so a new deployment is a ConfigMap edit and not a release.
    # The codex line rejects /chat/completions with HTTP 400 by design — see
    # agentcore.llm.azure.
    models_responses_api: str = ""  # comma-separated

    # --- Ollama (self-hosted, OpenAI-compatible) ---------------------------
    # Ollama serves the OpenAI wire protocol at <host>:11434/v1/, so it needs no client
    # of its own. What it does not serve is /responses — see the validator below.
    ollama_base_url: str = ""
    models_ollama: str = ""  # comma-separated
    # Ollama ignores the key, but the OpenAI SDK refuses to construct without one.
    # Overridable in case something that does check it sits in front.
    ollama_api_key: str = "ollama"
    # JSON, unlike the comma-separated fields above, and deliberately so: these are
    # credentials for whatever gates the endpoint — Cloudflare Access sends two headers
    # — and a secret may well contain a comma. A dict field is precisely the case
    # pydantic-settings' JSON parsing is for.
    ollama_headers: dict[str, str] = {}
    # Local inference is slow, and on CPU the whole prompt must be ingested before the
    # first token appears. Azure's 180s would time out on a long input.
    ollama_timeout_seconds: float = 600.0
    # Reasoning models on Ollama think by default, and on four CPU cores that is the
    # difference between a second and a minute. Measured 2026-09-03 on qwen3.5:0.8b,
    # "say ready": thinking on, 11s and 427 completion tokens; off, 1s and 28.
    #
    # reasoning_effort is the ONLY lever that reaches it over /v1. Also measured, all
    # on the same box: Ollama silently ignores `think: false` there (it works on the
    # native /api/chat), ignores `chat_template_kwargs.enable_thinking` — which made it
    # worse, 45s — rejects `PARAMETER think` in a Modelfile outright, and Qwen3.5 no
    # longer honours `/no_think` in the prompt the way Qwen3 did.
    #
    # Raise it to low/medium per deployment for a role that genuinely benefits.
    ollama_reasoning_effort: str = "none"

    # --- Telegram ----------------------------------------------------------
    telegram_bot_token: str = ""
    # Fail-closed: an empty allowlist refuses everyone rather than allowing everyone.
    telegram_allowed_users: str = ""  # comma-separated numeric ids

    # --- MCP servers -------------------------------------------------------
    # Empty URL == that server is not wired up in this deployment.
    mcp_github_url: str = ""
    mcp_cloudflare_url: str = ""
    mcp_cluster_url: str = ""
    # A worker agent exposed as an MCP server — see src/mcp_agent. Empty means this
    # deployment has no one to delegate to.
    mcp_coder_url: str = ""

    # The official GitHub MCP server in `http` mode has NO token field: it requires
    # `Authorization: Bearer <PAT>` on every request. So the PAT lives here, in the
    # agent pod, not in the mcp-github pod.
    github_pat: str = ""

    # --- policy ------------------------------------------------------------
    github_write_mode: WriteMode = WriteMode.DIRECT_PUSH
    cloudflare_write_mode: WriteMode = WriteMode.APPROVE
    # Repos that may never be pushed to directly, whatever github_write_mode says.
    # A bad manifest landing in the GitOps repo reaches the cluster in about a minute:
    # source-controller polls git on a 1m interval and kustomize-controller reconciles on
    # the source-revision change rather than waiting for its own 10m interval.
    protected_repos: str = "personal-k8s"  # comma-separated

    # --- limits ------------------------------------------------------------
    max_steps: int = 30
    # How much transcript is sent on a single step.
    max_tokens_per_turn: int = 120_000
    # Spend ceiling for one turn, counting only tokens outside the cached prefix.
    # A thirty-step turn costs roughly 33k of these, so this is a backstop, not a
    # limit ordinary work should ever meet.
    max_billable_tokens_per_turn: int = 200_000
    tool_timeout_seconds: float = 120.0
    # A delegated agent runs a whole tool loop of its own before answering, so it needs
    # far longer than an ordinary tool call.
    agent_tool_timeout_seconds: float = 900.0

    # --- process -----------------------------------------------------------
    health_port: int = 8080
    log_level: str = "INFO"

    # -- parsed accessors ---------------------------------------------------

    def base_url(self) -> str:
        """Azure endpoint, always with a trailing slash."""
        return self.azure_openai_base_url.rstrip("/") + "/" if self.azure_openai_base_url else ""

    def allowed_user_ids(self) -> list[int]:
        """Telegram ids permitted to talk to the bot. Junk entries are ignored."""
        ids: list[int] = []
        for item in _split(self.telegram_allowed_users):
            try:
                ids.append(int(item))
            except ValueError:
                continue
        return ids

    def protected_repo_set(self) -> set[str]:
        return {r.lower() for r in _split(self.protected_repos)}

    def mcp_endpoints(self) -> dict[str, str]:
        """Configured MCP servers as {logical name: url}, skipping unset ones."""
        candidates = {
            "github": self.mcp_github_url,
            "cloudflare": self.mcp_cloudflare_url,
            "cluster": self.mcp_cluster_url,
            "coder": self.mcp_coder_url,
        }
        return {name: url for name, url in candidates.items() if url}

    def responses_api_models(self) -> set[str]:
        """Deployments to call over /responses instead of /chat/completions."""
        return set(_split(self.models_responses_api))

    def allowed_models(self) -> list[str]:
        """Models selectable via /model, always including the default."""
        models = _split(self.model_allowlist)
        if self.model_default and self.model_default not in models:
            models.insert(0, self.model_default)
        return models

    def ollama_url(self) -> str:
        """Ollama's OpenAI-compatible endpoint, always with a trailing slash."""
        return self.ollama_base_url.rstrip("/") + "/" if self.ollama_base_url else ""

    def ollama_models(self) -> set[str]:
        """Models to route to Ollama rather than to Azure."""
        return set(_split(self.models_ollama))

    @model_validator(mode="after")
    def _ollama_routing_is_coherent(self) -> Settings:
        """Refuse the two misconfigurations that would otherwise fail mid-conversation.

        Both produce errors that point nowhere near the cause: an unrouted local model
        reaches Azure as an unknown deployment, and an Ollama model asked for /responses
        gets a bare 404 from a path Ollama does not implement.
        """
        local = set(_split(self.models_ollama))
        if local and not self.ollama_base_url:
            raise ValueError(
                f"MODELS_OLLAMA lists {sorted(local)} but OLLAMA_BASE_URL is empty, "
                "so those names would be sent to Azure as deployment names"
            )
        clash = local & set(_split(self.models_responses_api))
        if clash:
            raise ValueError(
                f"{sorted(clash)} appear in both MODELS_OLLAMA and MODELS_RESPONSES_API; "
                "Ollama serves /chat/completions only"
            )
        return self


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached; call `get_settings.cache_clear()` in tests."""
    return Settings()
