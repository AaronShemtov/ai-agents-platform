"""What a purge request is allowed to be.

The guard is against the caller, not the API: Cloudflare accepts "purge everything"
without complaint, and the consequence — every page on the zone fetched from the origin
again — lands on a two-node Always Free cluster.
"""

from __future__ import annotations

from mcp_cloudflare.tools import MAX_PURGE_URLS, purge_body


def test_specific_urls_become_a_file_list() -> None:
    assert purge_body(["https://1ms.my/a", "https://1ms.my/b"], False) == {
        "files": ["https://1ms.my/a", "https://1ms.my/b"]
    }


def test_everything_is_its_own_form() -> None:
    assert purge_body(None, True) == {"purge_everything": True}


def test_asking_for_both_is_refused() -> None:
    """Ambiguous, and the two differ enormously in blast radius."""
    result = purge_body(["https://1ms.my/a"], True)
    assert isinstance(result, str)
    assert "одно" in result


def test_asking_for_neither_is_refused() -> None:
    result = purge_body(None, False)
    assert isinstance(result, str)
    assert "нечего чистить" in result


def test_an_empty_list_is_not_a_silent_purge_of_everything() -> None:
    assert isinstance(purge_body([], False), str)


def test_paths_are_refused_because_cloudflare_wants_full_urls() -> None:
    result = purge_body(["/index.html"], False)
    assert isinstance(result, str)
    assert "URL" in result


def test_the_thirty_url_cap_is_enforced_here_rather_than_by_a_400() -> None:
    ok = [f"https://1ms.my/{i}" for i in range(MAX_PURGE_URLS)]
    assert purge_body(ok, False) == {"files": ok}

    too_many = [*ok, "https://1ms.my/one-more"]
    result = purge_body(too_many, False)
    assert isinstance(result, str)
    assert str(MAX_PURGE_URLS) in result
