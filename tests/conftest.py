from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def agents_dir(repo_root: Path) -> Path:
    return repo_root / "agents"


@pytest.fixture(autouse=True)
def isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient environment out of every Settings() a test constructs.

    Settings is a BaseSettings: it reads the process environment for any field a test
    does not pass. Run the suite somewhere that exports MCP_CLUSTER_URL — inside the
    agent pod, say — and a test asserting on the exact set of configured servers fails
    for reasons that have nothing to do with the code. Clearing the whole field set
    makes the suite say the same thing wherever it runs.
    """
    from agentcore.config import Settings

    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
