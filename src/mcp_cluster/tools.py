"""Read-only Kubernetes tools.

Every response is deliberately summarised rather than dumped. A raw V1Pod is several
kilobytes of JSON that is almost entirely irrelevant to "why is this pod unhealthy",
and ten of them would evict the actual conversation from the context window.

There are no write tools here, and there is no way to add one that would work: the
ServiceAccount this server runs under has a get/list/watch-only ClusterRole. Cluster
changes go through the GitOps repository.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from mcp_common.errors import tool_error

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _apis() -> tuple[client.CoreV1Api, client.AppsV1Api]:
    """Lazy, cached client construction.

    Deliberately not done at import time: that is exactly what makes the equivalent
    module in gemini-sre-agent impossible to import (and therefore to test) without a
    live kubeconfig.
    """
    try:
        config.load_incluster_config()
        logger.info("using in-cluster kubernetes config")
    except config.ConfigException:
        config.load_kube_config()
        logger.info("using local kubeconfig")
    return client.CoreV1Api(), client.AppsV1Api()


def readiness() -> tuple[bool, str]:
    try:
        core, _ = _apis()
        core.list_namespace(limit=1, _request_timeout=5)
    except Exception as exc:  # noqa: BLE001
        return False, f"kube-apiserver unreachable: {exc}"
    return True, "kube-apiserver reachable"


def _api_error(exc: ApiException, what: str) -> dict[str, Any]:
    if exc.status == 403:
        return tool_error(
            f"нет прав на {what}",
            hint="у mcp-cluster намеренно только read-only ClusterRole",
            status=403,
        )
    if exc.status == 404:
        return tool_error(f"{what} не найден", status=404)
    return tool_error(f"ошибка API при {what}: {exc.reason}", status=exc.status)


def register(server: Any) -> None:
    """Attach the tools to an MCPServer."""

    @server.tool(description="List all namespaces in the cluster.")
    def list_namespaces() -> dict[str, Any]:
        core, _ = _apis()
        try:
            items = core.list_namespace().items
        except ApiException as exc:
            return _api_error(exc, "чтении namespaces")
        return {
            "ok": True,
            "namespaces": [
                {"name": ns.metadata.name, "phase": ns.status.phase} for ns in items
            ],
        }

    @server.tool(
        description=(
            "List pods in a namespace with their health: phase, ready containers, "
            "restart count, node, and the reason for any waiting container."
        )
    )
    def list_pods(namespace: str, only_unhealthy: bool = False) -> dict[str, Any]:
        core, _ = _apis()
        try:
            items = core.list_namespaced_pod(namespace).items
        except ApiException as exc:
            return _api_error(exc, f"чтении подов в {namespace}")

        pods = []
        for pod in items:
            statuses = pod.status.container_statuses or []
            ready = sum(1 for c in statuses if c.ready)
            restarts = sum(c.restart_count or 0 for c in statuses)
            reasons = [
                c.state.waiting.reason
                for c in statuses
                if c.state and c.state.waiting and c.state.waiting.reason
            ]
            healthy = pod.status.phase in {"Running", "Succeeded"} and ready == len(statuses)
            if only_unhealthy and healthy:
                continue
            pods.append(
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "ready": f"{ready}/{len(statuses)}",
                    "restarts": restarts,
                    "node": pod.spec.node_name,
                    "reasons": reasons or None,
                    "images": [c.image for c in pod.spec.containers],
                }
            )
        return {"ok": True, "namespace": namespace, "count": len(pods), "pods": pods}

    @server.tool(
        description=(
            "Read the tail of a pod's logs. Use container= when the pod has several. "
            "previous=true reads the logs of the last crashed instance."
        )
    )
    def get_pod_logs(
        namespace: str,
        pod: str,
        container: str | None = None,
        tail_lines: int = 100,
        previous: bool = False,
    ) -> dict[str, Any]:
        core, _ = _apis()
        try:
            text = core.read_namespaced_pod_log(
                name=pod,
                namespace=namespace,
                container=container,
                tail_lines=max(1, min(tail_lines, 500)),
                previous=previous,
                timestamps=False,
            )
        except ApiException as exc:
            return _api_error(exc, f"чтении логов {namespace}/{pod}")
        return {"ok": True, "pod": pod, "namespace": namespace, "logs": text or "(пусто)"}

    @server.tool(
        description="Recent events in a namespace, newest last. Good for scheduling and image-pull failures."
    )
    def list_events(namespace: str, limit: int = 40) -> dict[str, Any]:
        core, _ = _apis()
        try:
            items = core.list_namespaced_event(namespace).items
        except ApiException as exc:
            return _api_error(exc, f"чтении событий в {namespace}")

        items.sort(key=lambda e: e.last_timestamp or e.event_time or "")
        events = [
            {
                "type": e.type,
                "reason": e.reason,
                "object": f"{e.involved_object.kind}/{e.involved_object.name}",
                "message": (e.message or "")[:300],
                "count": e.count,
                "at": str(e.last_timestamp or e.event_time or ""),
            }
            for e in items[-max(1, min(limit, 100)) :]
        ]
        return {"ok": True, "namespace": namespace, "events": events}

    @server.tool(description="Describe a Deployment: replicas, strategy, images and conditions.")
    def get_deployment(namespace: str, name: str) -> dict[str, Any]:
        _, apps = _apis()
        try:
            dep = apps.read_namespaced_deployment(name=name, namespace=namespace)
        except ApiException as exc:
            return _api_error(exc, f"чтении deployment {namespace}/{name}")
        return {
            "ok": True,
            "name": name,
            "namespace": namespace,
            "replicas": {
                "desired": dep.spec.replicas,
                "ready": dep.status.ready_replicas or 0,
                "available": dep.status.available_replicas or 0,
                "updated": dep.status.updated_replicas or 0,
            },
            "strategy": dep.spec.strategy.type if dep.spec.strategy else None,
            "selector": (dep.spec.selector.match_labels if dep.spec.selector else None),
            "images": [c.image for c in dep.spec.template.spec.containers],
            "conditions": [
                {"type": c.type, "status": c.status, "reason": c.reason}
                for c in (dep.status.conditions or [])
            ],
        }

    @server.tool(
        description=(
            "Describe a Service and its Endpoints. An empty endpoint list usually means "
            "the selector does not match any ready pod."
        )
    )
    def get_service(namespace: str, name: str) -> dict[str, Any]:
        core, _ = _apis()
        try:
            svc = core.read_namespaced_service(name=name, namespace=namespace)
        except ApiException as exc:
            return _api_error(exc, f"чтении service {namespace}/{name}")

        addresses: list[str] = []
        try:
            endpoints = core.read_namespaced_endpoints(name=name, namespace=namespace)
            for subset in endpoints.subsets or []:
                addresses.extend(a.ip for a in (subset.addresses or []))
        except ApiException:
            # A Service with no Endpoints object at all is itself the finding.
            pass

        return {
            "ok": True,
            "name": name,
            "namespace": namespace,
            "type": svc.spec.type,
            "cluster_ip": svc.spec.cluster_ip,
            "selector": svc.spec.selector,
            "ports": [
                {"port": p.port, "target_port": str(p.target_port), "protocol": p.protocol}
                for p in (svc.spec.ports or [])
            ],
            "endpoint_count": len(addresses),
            "endpoints": addresses[:20],
        }
