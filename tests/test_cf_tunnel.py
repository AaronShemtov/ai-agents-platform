"""Tunnel ingress ordering.

Cloudflare evaluates ingress rules top to bottom and stops at the first match. The
list always ends with a catch-all that has no hostname, so a rule appended after it
is dead configuration — the new site would return 404 while the config *looks* right
in the dashboard. Hence these tests.
"""

from __future__ import annotations

from mcp_cloudflare.tools import (
    CATCH_ALL_SERVICE,
    find_hostname,
    insert_before_catchall,
    is_catch_all,
)

GATEWAY = "http://envoy.envoy-gateway-system.svc.cluster.local:80"
CATCH_ALL = {"service": CATCH_ALL_SERVICE}


def test_catch_all_is_recognised_by_the_absence_of_a_hostname() -> None:
    assert is_catch_all(CATCH_ALL)
    assert is_catch_all({"service": "http_status:404"})
    assert not is_catch_all({"hostname": "a.1ms.my", "service": GATEWAY})


def test_new_rule_lands_before_the_catch_all() -> None:
    ingress = [{"hostname": "cv.1ms.my", "service": GATEWAY}, CATCH_ALL]
    result = insert_before_catchall(ingress, {"hostname": "new.1ms.my", "service": GATEWAY})
    assert result[-1] == CATCH_ALL
    assert result[-2]["hostname"] == "new.1ms.my"


def test_existing_hostnames_keep_their_order() -> None:
    ingress = [
        {"hostname": "a.1ms.my", "service": GATEWAY},
        {"hostname": "b.1ms.my", "service": GATEWAY},
        CATCH_ALL,
    ]
    result = insert_before_catchall(ingress, {"hostname": "c.1ms.my", "service": GATEWAY})
    assert [r.get("hostname") for r in result] == ["a.1ms.my", "b.1ms.my", "c.1ms.my", None]


def test_missing_catch_all_is_supplied() -> None:
    # A config without a terminating rule is rejected by Cloudflare, so rather than
    # PUT something invalid we add the catch-all ourselves.
    result = insert_before_catchall([], {"hostname": "a.1ms.my", "service": GATEWAY})
    assert result[-1] == CATCH_ALL
    assert len(result) == 2


def test_only_one_catch_all_survives() -> None:
    ingress = [CATCH_ALL, {"hostname": "a.1ms.my", "service": GATEWAY}, CATCH_ALL]
    result = insert_before_catchall(ingress, {"hostname": "b.1ms.my", "service": GATEWAY})
    assert sum(1 for r in result if is_catch_all(r)) == 1
    assert result[-1] == CATCH_ALL


def test_a_hostname_stranded_after_the_catch_all_is_pulled_back_in_front() -> None:
    # If a previous naive write left a hostname after the catch-all, it was dead.
    # Rebuilding the list repairs it instead of preserving the broken order.
    ingress = [CATCH_ALL, {"hostname": "stranded.1ms.my", "service": GATEWAY}]
    result = insert_before_catchall(ingress, {"hostname": "new.1ms.my", "service": GATEWAY})
    assert [r.get("hostname") for r in result] == ["stranded.1ms.my", "new.1ms.my", None]


def test_find_hostname() -> None:
    ingress = [{"hostname": "a.1ms.my", "service": GATEWAY}, CATCH_ALL]
    assert find_hostname(ingress, "a.1ms.my") is not None
    assert find_hostname(ingress, "b.1ms.my") is None
