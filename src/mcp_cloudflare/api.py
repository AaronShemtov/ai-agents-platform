"""Thin async client for the Cloudflare API v4.

The API token lives only in this process. Unlike the GitHub MCP server — which takes
a bearer per request and therefore forces the PAT to sit in the agent pod — this
server holds its own credential and never hands it out.

Every v4 response is wrapped as {success, errors, messages, result}; `request()`
unwraps it and turns `success: false` into a readable exception so tools do not each
have to re-check.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import httpx2

logger = logging.getLogger(__name__)

BASE_URL = "https://api.cloudflare.com/client/v4"


class CloudflareError(Exception):
    def __init__(self, message: str, *, status: int | None = None, codes: list[int] | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.codes = codes or []


class CloudflareAPI:
    def __init__(self, token: str, *, account_id: str = "", timeout: float = 30.0) -> None:
        if not token:
            raise CloudflareError("CLOUDFLARE_API_TOKEN is not set")
        self._token = token
        self._account_id = account_id
        self._client = httpx2.AsyncClient(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        self._zone_cache: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        response = await self._client.request(method, path, json=json, params=params)
        try:
            payload = response.json()
        except Exception as exc:
            raise CloudflareError(
                f"non-JSON response ({response.status_code})", status=response.status_code
            ) from exc

        if not payload.get("success", False):
            errors = payload.get("errors") or []
            detail = "; ".join(
                f"{e.get('code', '?')}: {e.get('message', '')}" for e in errors
            ) or f"HTTP {response.status_code}"
            raise CloudflareError(
                detail,
                status=response.status_code,
                codes=[e.get("code") for e in errors if isinstance(e.get("code"), int)],
            )
        return payload.get("result")

    # -- identifiers ---------------------------------------------------------

    async def account_id(self) -> str:
        """The account to act on, auto-detected when not pinned by env.

        Auto-detection needs the token to carry `Account Resources: Read`.
        """
        if self._account_id:
            return self._account_id
        accounts = await self.request("GET", "/accounts")
        if not accounts:
            raise CloudflareError(
                "token sees no accounts; add the 'Account Resources: Read' permission "
                "or set CLOUDFLARE_ACCOUNT_ID"
            )
        if len(accounts) > 1:
            names = ", ".join(a.get("name", a.get("id", "?")) for a in accounts)
            raise CloudflareError(
                f"token sees several accounts ({names}); set CLOUDFLARE_ACCOUNT_ID explicitly"
            )
        self._account_id = accounts[0]["id"]
        return self._account_id

    async def zone_id(self, zone: str) -> str:
        """Resolve a zone name like `1ms.my` to its id. Accepts an id unchanged."""
        # Zone ids are 32 hex characters; treat anything matching that as already resolved.
        if len(zone) == 32 and all(c in "0123456789abcdef" for c in zone.lower()):
            return zone
        if zone in self._zone_cache:
            return self._zone_cache[zone]
        zones = await self.request("GET", "/zones", params={"name": zone})
        if not zones:
            raise CloudflareError(f"zone {zone!r} not found on this account")
        zone_id = zones[0]["id"]
        self._zone_cache[zone] = zone_id
        return zone_id


@lru_cache(maxsize=1)
def get_api() -> CloudflareAPI:
    """Lazy singleton — nothing touches the environment at import time."""
    return CloudflareAPI(
        token=os.environ.get("CLOUDFLARE_API_TOKEN", ""),
        account_id=os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""),
    )
