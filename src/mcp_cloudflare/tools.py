"""Cloudflare tools: DNS records, WAF custom rules, Tunnel configuration.

No official self-hostable Cloudflare MCP server exists — Cloudflare's own are Workers
applications requiring platform-only bindings, and the community ones are unmaintained
— so this wraps the ~13 REST calls actually needed.

Two things here break if implemented naively, and both are covered by unit tests:

1. **WAF custom rules are the Rulesets API**, not the old firewall/rules endpoint.
   They live in the *entrypoint ruleset* of the `http_request_firewall_custom` phase,
   and the only way to change one rule is to PUT the whole rules array back.

2. **Tunnel ingress is ordered and ends in a catch-all.** The last entry is always
   `{"service": "http_status:404"}` with no hostname. Appending a new hostname after
   it means the hostname never matches, so new rules must be inserted *before* it.
   `update_tunnel_config` also replaces the config wholesale, so everything here is
   read-modify-write.

Tool names follow the read/write verb convention the policy layer keys off:
`list_*` and `get_*` are treated as read-only, everything else as mutating.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp_cloudflare.api import CloudflareAPI, CloudflareError, get_api
from mcp_common.errors import tool_error

logger = logging.getLogger(__name__)

WAF_PHASE = "http_request_firewall_custom"
CATCH_ALL_SERVICE = "http_status:404"

VALID_WAF_ACTIONS = frozenset(
    {"block", "challenge", "js_challenge", "managed_challenge", "log", "skip"}
)


# -- pure helpers (unit-tested without network) ------------------------------


def is_catch_all(rule: dict[str, Any]) -> bool:
    """The terminating ingress rule: no hostname, matches everything left over."""
    return not rule.get("hostname")


def insert_before_catchall(
    ingress: list[dict[str, Any]], new_rule: dict[str, Any]
) -> list[dict[str, Any]]:
    """Place a hostname rule ahead of the catch-all, preserving order.

    Cloudflare evaluates ingress rules top to bottom and stops at the first match, so
    anything placed after the catch-all is dead configuration.
    """
    rules = [r for r in ingress if not is_catch_all(r)]
    tail = [r for r in ingress if is_catch_all(r)]
    rules.append(new_rule)
    # A config with no catch-all is invalid; supply one rather than PUTting a broken config.
    if not tail:
        tail = [{"service": CATCH_ALL_SERVICE}]
    return rules + tail[:1]


def find_hostname(ingress: list[dict[str, Any]], hostname: str) -> dict[str, Any] | None:
    for rule in ingress:
        if rule.get("hostname") == hostname:
            return rule
    return None


# -- readiness ---------------------------------------------------------------


async def readiness() -> tuple[bool, str]:
    try:
        result = await get_api().request("GET", "/user/tokens/verify")
    except CloudflareError as exc:
        return False, f"cloudflare token check failed: {exc.message}"
    except Exception as exc:  # noqa: BLE001
        return False, f"cloudflare unreachable: {exc}"
    return True, f"token {result.get('status', 'unknown')}"


# -- WAF helpers -------------------------------------------------------------


async def _read_waf_rules(api: CloudflareAPI, zone_id: str) -> list[dict[str, Any]]:
    try:
        ruleset = await api.request("GET", f"/zones/{zone_id}/rulesets/phases/{WAF_PHASE}/entrypoint")
    except CloudflareError as exc:
        # No custom ruleset has ever been created for this zone yet.
        if exc.status == 404:
            return []
        raise
    return list(ruleset.get("rules") or [])


async def _write_waf_rules(
    api: CloudflareAPI, zone_id: str, rules: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = await api.request(
        "PUT",
        f"/zones/{zone_id}/rulesets/phases/{WAF_PHASE}/entrypoint",
        json={"rules": rules},
    )
    return list(result.get("rules") or [])


def _slim_waf_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": rule.get("id"),
        "description": rule.get("description"),
        "expression": rule.get("expression"),
        "action": rule.get("action"),
        "enabled": rule.get("enabled", True),
    }


# -- registration ------------------------------------------------------------


def register(server: Any) -> None:  # noqa: C901 - a flat list of tool definitions
    api = get_api

    # ---- zones ------------------------------------------------------------

    @server.tool(description="List zones (domains) this token can see, with their ids.")
    async def list_zones() -> dict[str, Any]:
        try:
            zones = await api().request("GET", "/zones")
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {
            "ok": True,
            "zones": [
                {"id": z["id"], "name": z["name"], "status": z.get("status")} for z in zones
            ],
        }

    # ---- DNS --------------------------------------------------------------

    @server.tool(
        description=(
            "List DNS records in a zone. zone accepts a name like '1ms.my' or a zone id. "
            "Optionally filter by record type (A, CNAME, TXT...) or exact name."
        )
    )
    async def list_dns_records(
        zone: str, type: str | None = None, name: str | None = None
    ) -> dict[str, Any]:
        try:
            zone_id = await api().zone_id(zone)
            params: dict[str, Any] = {"per_page": 100}
            if type:
                params["type"] = type.upper()
            if name:
                params["name"] = name
            records = await api().request("GET", f"/zones/{zone_id}/dns_records", params=params)
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {
            "ok": True,
            "zone": zone,
            "records": [
                {
                    "id": r["id"],
                    "type": r["type"],
                    "name": r["name"],
                    "content": r["content"],
                    "proxied": r.get("proxied"),
                    "ttl": r.get("ttl"),
                }
                for r in records
            ],
        }

    @server.tool(
        description=(
            "Create a DNS record. For a site served through the Cloudflare Tunnel use "
            "type=CNAME, content=<tunnel-id>.cfargotunnel.com and proxied=true."
        )
    )
    async def create_dns_record(
        zone: str,
        type: str,
        name: str,
        content: str,
        proxied: bool = True,
        ttl: int = 1,
        comment: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": type.upper(),
            "name": name,
            "content": content,
            "ttl": ttl,  # 1 means "automatic"
            "proxied": proxied,
        }
        if comment:
            body["comment"] = comment
        try:
            zone_id = await api().zone_id(zone)
            record = await api().request("POST", f"/zones/{zone_id}/dns_records", json=body)
        except CloudflareError as exc:
            return tool_error(exc.message, hint="проверь, что запись с таким именем ещё не существует")
        return {"ok": True, "record": {"id": record["id"], "name": record["name"]}}

    @server.tool(description="Change fields of an existing DNS record. Only pass what changes.")
    async def update_dns_record(
        zone: str,
        record_id: str,
        content: str | None = None,
        name: str | None = None,
        type: str | None = None,
        proxied: bool | None = None,
        ttl: int | None = None,
    ) -> dict[str, Any]:
        body = {
            k: v
            for k, v in {
                "content": content,
                "name": name,
                "type": type.upper() if type else None,
                "proxied": proxied,
                "ttl": ttl,
            }.items()
            if v is not None
        }
        if not body:
            return tool_error("нечего менять: не передано ни одного поля")
        try:
            zone_id = await api().zone_id(zone)
            record = await api().request(
                "PATCH", f"/zones/{zone_id}/dns_records/{record_id}", json=body
            )
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {"ok": True, "record": {"id": record["id"], "name": record["name"]}}

    @server.tool(description="Delete a DNS record by id. Get the id from list_dns_records first.")
    async def delete_dns_record(zone: str, record_id: str) -> dict[str, Any]:
        try:
            zone_id = await api().zone_id(zone)
            await api().request("DELETE", f"/zones/{zone_id}/dns_records/{record_id}")
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {"ok": True, "deleted": record_id}

    # ---- WAF --------------------------------------------------------------

    @server.tool(description="List custom WAF (firewall) rules for a zone, in evaluation order.")
    async def list_waf_rules(zone: str) -> dict[str, Any]:
        try:
            zone_id = await api().zone_id(zone)
            rules = await _read_waf_rules(api(), zone_id)
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {"ok": True, "zone": zone, "rules": [_slim_waf_rule(r) for r in rules]}

    @server.tool(
        description=(
            "Append a custom WAF rule. expression uses Cloudflare's filter syntax, e.g. "
            '(ip.src eq 1.2.3.4) or (http.request.uri.path contains "/admin"). '
            "action is one of block, challenge, managed_challenge, js_challenge, log, skip."
        )
    )
    async def create_waf_rule(
        zone: str, expression: str, action: str, description: str = ""
    ) -> dict[str, Any]:
        if action not in VALID_WAF_ACTIONS:
            return tool_error(
                f"недопустимое действие {action!r}",
                hint=f"допустимые: {', '.join(sorted(VALID_WAF_ACTIONS))}",
            )
        try:
            zone_id = await api().zone_id(zone)
            rules = await _read_waf_rules(api(), zone_id)
            rules.append(
                {
                    "expression": expression,
                    "action": action,
                    "description": description or "created by agent",
                    "enabled": True,
                }
            )
            written = await _write_waf_rules(api(), zone_id, rules)
        except CloudflareError as exc:
            return tool_error(exc.message, hint="частая причина — синтаксическая ошибка в expression")
        return {"ok": True, "rules": [_slim_waf_rule(r) for r in written]}

    @server.tool(description="Change an existing custom WAF rule, identified by its id.")
    async def update_waf_rule(
        zone: str,
        rule_id: str,
        expression: str | None = None,
        action: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        if action is not None and action not in VALID_WAF_ACTIONS:
            return tool_error(f"недопустимое действие {action!r}")
        try:
            zone_id = await api().zone_id(zone)
            rules = await _read_waf_rules(api(), zone_id)
            target = next((r for r in rules if r.get("id") == rule_id), None)
            if target is None:
                return tool_error(
                    f"правило {rule_id} не найдено",
                    hint="сначала вызови list_waf_rules и возьми id оттуда",
                )
            for key, value in (
                ("expression", expression),
                ("action", action),
                ("description", description),
                ("enabled", enabled),
            ):
                if value is not None:
                    target[key] = value
            written = await _write_waf_rules(api(), zone_id, rules)
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {"ok": True, "rules": [_slim_waf_rule(r) for r in written]}

    @server.tool(description="Delete a custom WAF rule by id.")
    async def delete_waf_rule(zone: str, rule_id: str) -> dict[str, Any]:
        try:
            zone_id = await api().zone_id(zone)
            rules = await _read_waf_rules(api(), zone_id)
            remaining = [r for r in rules if r.get("id") != rule_id]
            if len(remaining) == len(rules):
                return tool_error(f"правило {rule_id} не найдено")
            written = await _write_waf_rules(api(), zone_id, remaining)
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {"ok": True, "deleted": rule_id, "rules": [_slim_waf_rule(r) for r in written]}

    # ---- Tunnels ----------------------------------------------------------

    @server.tool(description="List Cloudflare Tunnels on the account, with ids and connection state.")
    async def list_tunnels() -> dict[str, Any]:
        try:
            account = await api().account_id()
            tunnels = await api().request(
                "GET", f"/accounts/{account}/cfd_tunnel", params={"is_deleted": "false"}
            )
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {
            "ok": True,
            "tunnels": [
                {
                    "id": t["id"],
                    "name": t.get("name"),
                    "status": t.get("status"),
                    "connections": len(t.get("connections") or []),
                }
                for t in tunnels
            ],
        }

    @server.tool(
        description=(
            "Read a tunnel's ingress configuration: the ordered list of hostname -> service "
            "rules, ending in a catch-all."
        )
    )
    async def get_tunnel_config(tunnel_id: str) -> dict[str, Any]:
        try:
            account = await api().account_id()
            result = await api().request(
                "GET", f"/accounts/{account}/cfd_tunnel/{tunnel_id}/configurations"
            )
        except CloudflareError as exc:
            return tool_error(exc.message)
        config = result.get("config") or {}
        return {
            "ok": True,
            "tunnel_id": tunnel_id,
            "version": result.get("version"),
            "ingress": config.get("ingress") or [],
        }

    @server.tool(
        description=(
            "Add a public hostname to a tunnel, routing it to an in-cluster service. "
            "Every site on this cluster uses the Envoy Gateway as its service: "
            "http://envoy-envoy-gateway-system-homelab-gateway-00f55f79.envoy-gateway-system.svc.cluster.local:80 "
            "(copy it verbatim — the -00f55f79 suffix is generated by Envoy Gateway and "
            "the name does not resolve without it). Use tunnel homelab-tunnel; the other "
            "tunnels on this account serve different things. "
            "The rule is inserted before the catch-all so it actually matches. "
            "Note this only makes the tunnel route the hostname — a DNS record is still needed."
        )
    )
    async def add_tunnel_hostname(
        tunnel_id: str, hostname: str, service: str
    ) -> dict[str, Any]:
        try:
            account = await api().account_id()
            current = await api().request(
                "GET", f"/accounts/{account}/cfd_tunnel/{tunnel_id}/configurations"
            )
            config = dict(current.get("config") or {})
            ingress = list(config.get("ingress") or [])

            if find_hostname(ingress, hostname) is not None:
                return tool_error(
                    f"hostname {hostname} уже есть в конфиге туннеля",
                    hint="используй update_tunnel_config, если нужно поменять его service",
                )

            config["ingress"] = insert_before_catchall(ingress, {
                "hostname": hostname,
                "service": service,
            })
            await api().request(
                "PUT",
                f"/accounts/{account}/cfd_tunnel/{tunnel_id}/configurations",
                json={"config": config},
            )
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {"ok": True, "hostname": hostname, "ingress": config["ingress"]}

    @server.tool(
        description=(
            "Replace a tunnel's whole ingress list. Destructive: whatever you send becomes "
            "the entire configuration. Read get_tunnel_config first and send the full list "
            "back with your change applied. The last entry must be the catch-all "
            '{"service": "http_status:404"}. Prefer add_tunnel_hostname for adding a host.'
        )
    )
    async def update_tunnel_config(
        tunnel_id: str, ingress: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not ingress:
            return tool_error("ingress пустой — это снесёт всю маршрутизацию туннеля")
        if not is_catch_all(ingress[-1]):
            return tool_error(
                "последним правилом должен быть catch-all без hostname",
                hint='добавь в конец {"service": "http_status:404"}',
            )
        try:
            account = await api().account_id()
            await api().request(
                "PUT",
                f"/accounts/{account}/cfd_tunnel/{tunnel_id}/configurations",
                json={"config": {"ingress": ingress}},
            )
        except CloudflareError as exc:
            return tool_error(exc.message)
        return {"ok": True, "tunnel_id": tunnel_id, "ingress": ingress}
