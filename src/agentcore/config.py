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

    # --- Telegram ----------------------------------------------------------
    telegram_bot_token: str = ""
    # Fail-closed: an empty allowlist refuses everyone rather than allowing everyone.
    telegram_allowed_users: str = ""  # comma-separated numeric ids

    # --- MCP servers -------------------------------------------------------
    # Empty URL == that server is not wired up in this deployment.
    mcp_github_url: str = ""
    mcp_cloudflare_url: str = ""
    mcp_cluster_url: str = ""

    # The official GitHub MCP server in `http` mode has NO token field: it requires
    # `Authorization: Bearer <PAT>` on every request. So the PAT lives here, in the
    # agent pod, not in the mcp-github pod.
    github_pat: str = ""

    # --- policy ------------------------------------------------------------
    github_write_mode: WriteMode = WriteMode.DIRECT_PUSH
    cloudflare_write_mode: WriteMode = WriteMode.APPROVE
    # Repos that may never be pushed to directly, whatever github_write_mode says.
    # A bad manifest landing in the GitOps repo reaches the cluster within ~10 minutes.
    protected_repos: str = "personal-k8s"  # comma-separated

    # --- limits ------------------------------------------------------------
    max_steps: int = 30
    max_tokens_per_turn: int = 120_000
    tool_timeout_seconds: float = 120.0

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
        }
        return {name: url for name, url in candidates.items() if url}

    def allowed_models(self) -> list[str]:
        """Models selectable via /model, always including the default."""
        models = _split(self.model_allowlist)
        if self.model_default and self.model_default not in models:
            models.insert(0, self.model_default)
        return models


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached; call `get_settings.cache_clear()` in tests."""
    return Settings()
