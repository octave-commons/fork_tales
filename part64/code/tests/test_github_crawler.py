from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from code.world_web.github_extract import canonical_github_url, extract_github_atoms
from code.world_web.github_presence import GithubPresence
from code.world_web.graph_queries import build_facts_snapshot, run_named_graph_query


def _write_config(part_root: Path, payload: dict[str, Any]) -> None:
    config_dir = part_root / "world_state" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "github_seeds.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        "utf-8",
    )


def _success_result(url: str, payload: Any) -> dict[str, Any]:
    canonical = canonical_github_url(url)
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "ok": True,
        "url": url,
        "canonical_url": canonical,
        "status": 200,
        "duration_ms": 3,
        "rate_limit_remaining": 5000,
        "rate_limit_reset": 0,
        "payload": payload,
        "resource": {
            "kind": "github:unknown",
            "content_hash": hashlib.sha256(body).hexdigest(),
            "text_excerpt_hash": "",
        },
    }


def test_canonical_github_url_strips_fragments_and_normalizes_api_paths() -> None:
    html_url = "https://github.com/acme/repo/pull/7?utm=1#files"
    api_url = "https://api.github.com/repos/acme/repo/pulls/7?per_page=1#frag"
    raw_url = "https://raw.githubusercontent.com/acme/repo/main/src/app.py#L1"

    assert canonical_github_url(html_url) == "https://github.com/acme/repo/pull/7"
    assert canonical_github_url(api_url) == "https://github.com/acme/repo/pull/7"
    assert (
        canonical_github_url(raw_url)
        == "https://raw.githubusercontent.com/acme/repo/main/src/app.py"
    )
    assert canonical_github_url(html_url) == canonical_github_url(
        "https://github.com/acme/repo/pull/7"
    )


def test_extract_github_atoms_is_bounded_and_deterministic() -> None:
    keywords = [f"kw{i}" for i in range(80)]
    payload = {
        "title": "security update",
        "body": " ".join(keywords),
        "labels": [{"name": "security"}],
        "number": 12,
        "state": "closed",
        "merged_at": "2026-02-27T00:00:00Z",
        "filenames_touched": [f"src/file_{i}.py" for i in range(40)],
    }
    config = {
        "keywords": keywords,
        "file_patterns": ["package.json", "requirements.txt"],
    }
    canonical = "https://github.com/acme/repo/pull/12"

    atoms_a = extract_github_atoms(canonical, payload, config)
    atoms_b = extract_github_atoms(canonical, payload, config)

    assert len(atoms_a) == 50
    assert atoms_a == atoms_b
    assert atoms_a == sorted(
        atoms_a,
        key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True),
    )


def test_extract_github_atoms_reads_conversation_and_commit_context() -> None:
    payload = {
        "title": "refactor",
        "body": "minor cleanup",
        "conversation_rows": [
            {
                "channel": "issue-comment",
                "author": "security-review",
                "body": "Please address oauth token leak before merge.",
            }
        ],
        "commit_rows": [
            {
                "sha": "abcdef012345",
                "author": "octocat",
                "message": "fix oauth token handling and add CVE-2026-3333 note",
            }
        ],
    }
    config = {"keywords": ["oauth", "leak"]}
    canonical = "https://github.com/acme/repo/pull/99"

    atoms = extract_github_atoms(canonical, payload, config)
    kinds = {str(row.get("kind", "")) for row in atoms}
    terms = {
        str(row.get("term", "")).lower()
        for row in atoms
        if str(row.get("kind", "")) == "mentions"
    }
    cves = {
        str(row.get("cve_id", ""))
        for row in atoms
        if str(row.get("kind", "")) == "references_cve"
    }

    assert "mentions" in kinds
    assert "oauth" in terms
    assert "leak" in terms
    assert "CVE-2026-3333" in cves


def test_extract_github_atoms_captures_dependency_version_delta() -> None:
    payload = {
        "title": "Bump parser dependency",
        "files": [
            {
                "filename": "package.json",
                "patch": '- "parser": "1.0.0"\n+ "parser": "1.0.1"',
            }
        ],
    }
    config = {"keywords": [], "file_patterns": ["package.json"]}
    canonical = "https://github.com/acme/repo/pull/7"

    atoms = extract_github_atoms(canonical, payload, config)
    dependency_atom = next(
        row for row in atoms if str(row.get("kind", "")).strip() == "changes_dependency"
    )

    assert dependency_atom.get("dep_name") == "package.json"
    assert dependency_atom.get("from_ver") == "1.0.0"
    assert dependency_atom.get("to_ver") == "1.0.1"


def test_extract_github_atoms_reads_advisory_identifiers_and_description() -> None:
    payload = {
        "summary": "Advisory for token parsing bug",
        "description": "This release fixes CVE-2026-7777 and improves auth checks.",
        "severity": "high",
        "ghsa_id": "GHSA-abcd-efgh-ijkl",
        "identifiers": [
            {"type": "GHSA", "value": "GHSA-abcd-efgh-ijkl"},
            {"type": "CVE", "value": "CVE-2026-7777"},
        ],
    }
    config = {"keywords": ["auth", "token"]}
    canonical = "https://github.com/acme/repo/security-advisories/GHSA-abcd-efgh-ijkl"

    atoms = extract_github_atoms(canonical, payload, config)
    terms = {
        str(row.get("term", "")).lower()
        for row in atoms
        if str(row.get("kind", "")) == "mentions"
    }
    cves = {
        str(row.get("cve_id", ""))
        for row in atoms
        if str(row.get("kind", "")) == "references_cve"
    }

    assert "auth" in terms
    assert "token" in terms
    assert "CVE-2026-7777" in cves


def test_github_presence_builds_graph_nodes_edges_and_events() -> None:
    calls: list[str] = []

    def fake_fetcher(url: str, _config: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if "/pulls?" in url:
            payload = [
                {
                    "html_url": "https://github.com/acme/repo/pull/7",
                    "url": "https://api.github.com/repos/acme/repo/pulls/7",
                    "number": 7,
                    "title": "Bump parser dependency",
                    "body": "Fix CVE-2026-12345 in parser",
                    "labels": [{"name": "security"}],
                    "updated_at": "2026-02-27T00:00:00Z",
                    "user": {"login": "octocat"},
                    "comments": 9,
                    "state": "closed",
                }
            ]
            return _success_result(url, payload)

        if url.endswith("/pulls/7"):
            payload = {
                "html_url": "https://github.com/acme/repo/pull/7",
                "number": 7,
                "title": "Bump parser dependency",
                "body": "See https://github.com/acme/repo/issues/9 and CVE-2026-12345",
                "labels": [{"name": "security"}],
                "updated_at": "2026-02-27T00:00:05Z",
                "user": {"login": "octocat"},
                "state": "closed",
                "merged_at": "2026-02-27T00:00:06Z",
                "comments": 12,
            }
            return _success_result(url, payload)

        if "/pulls/7/files" in url:
            payload = [
                {
                    "filename": "package.json",
                    "patch": '- "parser": "1.0.0"\n+ "parser": "1.0.1"',
                },
                {
                    "filename": "src/auth/token_parser.py",
                    "patch": "token validation tightened",
                },
            ]
            return _success_result(url, payload)

        if "/issues/7/comments" in url:
            payload = [
                {
                    "user": {"login": "reviewer-a"},
                    "created_at": "2026-02-27T00:00:07Z",
                    "body": "Please harden token validation in parser path.",
                }
            ]
            return _success_result(url, payload)

        if "/pulls/7/reviews" in url:
            payload = [
                {
                    "user": {"login": "security-bot"},
                    "submitted_at": "2026-02-27T00:00:08Z",
                    "body": "LGTM after CVE note and release mention.",
                }
            ]
            return _success_result(url, payload)

        if "/pulls/7/comments" in url:
            payload = [
                {
                    "user": {"login": "reviewer-b"},
                    "created_at": "2026-02-27T00:00:09Z",
                    "body": "Patch line near token parser should include tests.",
                }
            ]
            return _success_result(url, payload)

        if "/pulls/7/commits" in url:
            payload = [
                {
                    "sha": "abc1234567890deadbeef",
                    "commit": {
                        "author": {"name": "octocat"},
                        "message": "tighten parser token checks for CVE path",
                    },
                }
            ]
            return _success_result(url, payload)

        return {
            "ok": False,
            "url": url,
            "canonical_url": canonical_github_url(url),
            "status": 404,
            "duration_ms": 1,
            "error": "http_error",
        }

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        _write_config(
            part_root,
            {
                "enabled": True,
                "repos": ["acme/repo"],
                "keywords": ["cve", "token", "parser", "security"],
                "file_patterns": ["package.json", "requirements.txt"],
                "max_repos": 1,
                "max_items_per_repo": 3,
                "max_github_fetches_per_tick": 10,
                "repo_cooldown_s": 0,
                "url_cooldown_s": 1200,
                "endpoint_order": ["pulls"],
            },
        )

        presence = GithubPresence(part_root, fetcher=fake_fetcher)
        touched = presence.tick(slack_ms=80.0, now_ts=1_700_000_000.0)
        assert touched
        assert len(calls) == 7

        graph = presence.graph_snapshot()
        nodes = graph.get("crawler_nodes", [])
        edges = graph.get("edges", [])
        events = graph.get("events", [])

        assert any(
            isinstance(node, dict) and str(node.get("web_node_role", "")) == "web:url"
            for node in nodes
        )
        pr_resource = next(
            node
            for node in nodes
            if isinstance(node, dict)
            and str(node.get("web_node_role", "")) == "web:resource"
            and str(node.get("kind", "")) == "github:pr"
        )
        assert pr_resource.get("number") == 7
        assert "package.json" in pr_resource.get("filenames_touched", [])
        assert int(pr_resource.get("importance_score", 0)) >= 4
        assert int(pr_resource.get("conversation_comment_count", 0)) >= 3
        assert int(pr_resource.get("commit_count", 0)) == 1
        assert "parser" in str(pr_resource.get("text_excerpt", "")).lower()
        assert (
            "conversation" in str(pr_resource.get("conversation_markdown", "")).lower()
        )

        edge_kinds = {
            str(row.get("kind", "")).strip().lower()
            for row in edges
            if isinstance(row, dict)
        }
        assert "web:source_of" in edge_kinds
        assert "web:links_to" in edge_kinds
        assert "crawl:triggered_fetch" in edge_kinds
        assert any(
            isinstance(row, dict)
            and str(row.get("event", "")).strip() == "github_extract_atoms"
            for row in events
        )


def test_github_presence_collects_repo_security_advisories() -> None:
    calls: list[str] = []

    def fake_fetcher(url: str, _config: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if "/security-advisories?" in url:
            payload = [
                {
                    "ghsa_id": "GHSA-abcd-efgh-ijkl",
                    "summary": "Token verification bypass",
                    "url": "https://api.github.com/repos/acme/repo/security-advisories/GHSA-abcd-efgh-ijkl",
                    "html_url": "https://github.com/advisories/GHSA-abcd-efgh-ijkl",
                    "severity": "high",
                    "published_at": "2026-02-27T00:00:00Z",
                    "updated_at": "2026-02-27T00:00:01Z",
                }
            ]
            return _success_result(url, payload)

        if "/security-advisories/GHSA-abcd-efgh-ijkl" in url:
            payload = {
                "ghsa_id": "GHSA-abcd-efgh-ijkl",
                "summary": "Token verification bypass",
                "description": "Fixes CVE-2026-7777 in auth token parser.",
                "severity": "high",
                "identifiers": [
                    {"type": "GHSA", "value": "GHSA-abcd-efgh-ijkl"},
                    {"type": "CVE", "value": "CVE-2026-7777"},
                ],
                "updated_at": "2026-02-27T00:00:02Z",
                "published_at": "2026-02-27T00:00:00Z",
            }
            return _success_result(url, payload)

        return {
            "ok": False,
            "url": url,
            "canonical_url": canonical_github_url(url),
            "status": 404,
            "duration_ms": 1,
            "error": "http_error",
        }

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        _write_config(
            part_root,
            {
                "enabled": True,
                "repos": ["acme/repo"],
                "keywords": ["token", "auth", "security"],
                "max_repos": 1,
                "max_items_per_repo": 2,
                "max_github_fetches_per_tick": 2,
                "repo_cooldown_s": 0,
                "url_cooldown_s": 0,
                "endpoint_order": ["advisories"],
                "include_conversation_context": False,
            },
        )

        presence = GithubPresence(part_root, fetcher=fake_fetcher)
        touched = presence.tick(slack_ms=80.0, now_ts=1_700_000_050.0)
        assert touched
        assert len(calls) == 2

        graph = presence.graph_snapshot()
        resources = [
            row
            for row in graph.get("crawler_nodes", [])
            if isinstance(row, dict)
            and str(row.get("web_node_role", "")) == "web:resource"
            and str(row.get("kind", "")).strip().lower() == "github:advisory"
        ]
        assert len(resources) == 1
        first = resources[0]
        assert str(first.get("repo", "")).strip().lower() == "acme/repo"
        cves = {
            str(atom.get("cve_id", ""))
            for atom in first.get("atoms", [])
            if isinstance(atom, dict)
            and str(atom.get("kind", "")).strip().lower() == "references_cve"
        }
        assert "CVE-2026-7777" in cves


def test_github_presence_enforces_cooldown_and_backoff_growth() -> None:
    calls: list[str] = []

    def failing_fetcher(url: str, _config: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        return {
            "ok": False,
            "url": url,
            "canonical_url": canonical_github_url(url),
            "status": 429,
            "duration_ms": 2,
            "error": "http_error",
            "rate_limit_reset": 0,
        }

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        _write_config(
            part_root,
            {
                "enabled": True,
                "repos": ["acme/repo"],
                "max_repos": 1,
                "max_items_per_repo": 1,
                "max_github_fetches_per_tick": 1,
                "repo_cooldown_s": 0,
                "url_cooldown_s": 10,
                "endpoint_order": ["pulls"],
            },
        )

        presence = GithubPresence(part_root, fetcher=failing_fetcher)
        now_a = 1_700_000_000.0
        presence.tick(slack_ms=50.0, now_ts=now_a)

        list_url = "https://api.github.com/repos/acme/repo/pulls?state=all&sort=updated&direction=desc&per_page=1"
        canonical = canonical_github_url(list_url)
        url_row_a = presence.state.get("url_state", {}).get(canonical, {})
        assert int(url_row_a.get("fail_count", 0)) == 1
        backoff_a = float(url_row_a.get("next_allowed_fetch_ts", 0.0)) - now_a
        assert backoff_a >= 800.0

        now_b = float(url_row_a.get("next_allowed_fetch_ts", 0.0)) + 1.0
        presence.tick(slack_ms=50.0, now_ts=now_b)
        url_row_b = presence.state.get("url_state", {}).get(canonical, {})
        backoff_b = float(url_row_b.get("next_allowed_fetch_ts", 0.0)) - now_b

        assert int(url_row_b.get("fail_count", 0)) == 2
        assert backoff_b > backoff_a
        assert len(calls) == 2


def test_github_presence_counts_pr_files_subfetch_in_budget() -> None:
    calls: list[str] = []

    def fake_fetcher(url: str, _config: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if "/pulls?" in url:
            payload = [
                {
                    "html_url": "https://github.com/acme/repo/pull/7",
                    "url": "https://api.github.com/repos/acme/repo/pulls/7",
                    "number": 7,
                    "title": "First PR",
                    "updated_at": "2026-02-27T00:00:03Z",
                    "state": "open",
                },
                {
                    "html_url": "https://github.com/acme/repo/pull/8",
                    "url": "https://api.github.com/repos/acme/repo/pulls/8",
                    "number": 8,
                    "title": "Second PR",
                    "updated_at": "2026-02-27T00:00:02Z",
                    "state": "open",
                },
                {
                    "html_url": "https://github.com/acme/repo/pull/9",
                    "url": "https://api.github.com/repos/acme/repo/pulls/9",
                    "number": 9,
                    "title": "Third PR",
                    "updated_at": "2026-02-27T00:00:01Z",
                    "state": "open",
                },
            ]
            return _success_result(url, payload)

        if (
            url.endswith("/pulls/7")
            or url.endswith("/pulls/8")
            or url.endswith("/pulls/9")
        ):
            pr_number = int(url.rsplit("/", 1)[-1])
            payload = {
                "html_url": f"https://github.com/acme/repo/pull/{pr_number}",
                "number": pr_number,
                "title": f"PR {pr_number}",
                "body": "No file patch metadata",
                "updated_at": "2026-02-27T00:00:10Z",
                "state": "open",
                "user": {"login": "octocat"},
            }
            return _success_result(url, payload)

        if "/files?per_page=200" in url:
            return _success_result(url, [])

        return {
            "ok": False,
            "url": url,
            "canonical_url": canonical_github_url(url),
            "status": 404,
            "duration_ms": 1,
            "error": "http_error",
        }

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        _write_config(
            part_root,
            {
                "enabled": True,
                "repos": ["acme/repo"],
                "keywords": ["security"],
                "max_repos": 1,
                "max_items_per_repo": 3,
                "max_github_fetches_per_tick": 4,
                "repo_cooldown_s": 0,
                "url_cooldown_s": 0,
                "endpoint_order": ["pulls"],
                "include_conversation_context": False,
            },
        )

        presence = GithubPresence(part_root, fetcher=fake_fetcher)
        touched = presence.tick(slack_ms=80.0, now_ts=1_700_000_000.0)
        assert touched

        # list fetch + first detail + first files subfetch + second detail
        assert len(calls) == 4

        graph = presence.graph_snapshot()
        resources = [
            row
            for row in graph.get("crawler_nodes", [])
            if isinstance(row, dict)
            and str(row.get("web_node_role", "")) == "web:resource"
            and str(row.get("kind", "")).startswith("github:pr")
        ]
        assert len(resources) == 2


def test_github_presence_suppresses_near_duplicate_issue_resources() -> None:
    calls: list[str] = []

    def fake_fetcher(url: str, _config: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if "/issues?" in url:
            payload = [
                {
                    "html_url": "https://github.com/acme/repo/issues/7",
                    "url": "https://api.github.com/repos/acme/repo/issues/7",
                    "number": 7,
                    "title": "token leak in auth parser path",
                    "body": "security incident",
                    "updated_at": "2026-02-27T00:00:03Z",
                    "state": "open",
                    "labels": [{"name": "security"}],
                    "user": {"login": "octocat"},
                },
                {
                    "html_url": "https://github.com/acme/repo/issues/8",
                    "url": "https://api.github.com/repos/acme/repo/issues/8",
                    "number": 8,
                    "title": "token leak in auth parser path follow-up",
                    "body": "security incident follow up",
                    "updated_at": "2026-02-27T00:00:04Z",
                    "state": "open",
                    "labels": [{"name": "security"}],
                    "user": {"login": "octocat"},
                },
            ]
            return _success_result(url, payload)

        if url.endswith("/issues/7"):
            payload = {
                "html_url": "https://github.com/acme/repo/issues/7",
                "number": 7,
                "title": "token leak in auth parser path",
                "body": "Rotate oauth secret now. token leak in auth parser path detected.",
                "updated_at": "2026-02-27T00:00:08Z",
                "state": "open",
                "labels": [{"name": "security"}],
                "user": {"login": "octocat"},
                "comments": 4,
            }
            return _success_result(url, payload)

        if url.endswith("/issues/8"):
            payload = {
                "html_url": "https://github.com/acme/repo/issues/8",
                "number": 8,
                "title": "token leak in auth parser path follow-up",
                "body": "Rotate oauth secret now! token leak in auth parser path detected today.",
                "updated_at": "2026-02-27T00:00:09Z",
                "state": "open",
                "labels": [{"name": "security"}],
                "user": {"login": "octocat"},
                "comments": 1,
            }
            return _success_result(url, payload)

        return {
            "ok": False,
            "url": url,
            "canonical_url": canonical_github_url(url),
            "status": 404,
            "duration_ms": 1,
            "error": "http_error",
        }

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        _write_config(
            part_root,
            {
                "enabled": True,
                "repos": ["acme/repo"],
                "keywords": ["token", "auth", "security", "oauth"],
                "max_repos": 1,
                "max_items_per_repo": 2,
                "max_github_fetches_per_tick": 3,
                "repo_cooldown_s": 0,
                "url_cooldown_s": 0,
                "endpoint_order": ["issues"],
                "include_conversation_context": False,
                "near_duplicate_dedupe_enabled": True,
                "near_duplicate_hamming_max": 12,
            },
        )

        presence = GithubPresence(part_root, fetcher=fake_fetcher)
        touched = presence.tick(slack_ms=80.0, now_ts=1_700_000_100.0)
        assert touched
        assert len(calls) == 3

        graph = presence.graph_snapshot()
        resources = [
            row
            for row in graph.get("crawler_nodes", [])
            if isinstance(row, dict)
            and str(row.get("web_node_role", "")) == "web:resource"
            and str(row.get("kind", "")).strip().lower() == "github:issue"
        ]
        assert len(resources) == 1
        assert int(graph.get("status", {}).get("duplicate_suppressed", 0) or 0) == 1
        assert any(
            isinstance(row, dict)
            and str(row.get("event", "")).strip() == "github_resource_deduped"
            and str(row.get("action", "")).strip() == "suppressed"
            for row in graph.get("events", [])
        )


def test_github_presence_frontier_plan_balances_priority_and_exploration() -> None:
    calls: list[str] = []

    def fake_fetcher(url: str, _config: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if "/pulls?" in url:
            return _success_result(url, [])
        return {
            "ok": False,
            "url": url,
            "canonical_url": canonical_github_url(url),
            "status": 404,
            "duration_ms": 1,
            "error": "http_error",
        }

    def _repo_from_list_url(url: str) -> str:
        marker = "/repos/"
        if marker not in url:
            return ""
        tail = url.split(marker, 1)[1]
        return tail.split("/pulls", 1)[0].strip().lower()

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        _write_config(
            part_root,
            {
                "enabled": True,
                "repos": ["acme/high", "acme/low", "acme/new"],
                "max_repos": 2,
                "max_items_per_repo": 1,
                "max_github_fetches_per_tick": 2,
                "repo_cooldown_s": 0,
                "url_cooldown_s": 0,
                "endpoint_order": ["pulls"],
                "include_conversation_context": False,
                "frontier_exploration_ratio": 0.5,
                "frontier_value_window_ticks": 7200,
            },
        )

        presence = GithubPresence(part_root, fetcher=fake_fetcher)
        now_a = 1_700_000_200.0
        presence.state["resources"] = {
            "res:high": {
                "id": "res:high",
                "repo": "acme/high",
                "kind": "github:advisory",
                "canonical_url": "https://github.com/acme/high/security/advisories/GHSA-high",
                "importance_score": 10,
                "fetched_ts": now_a - 8.0,
                "atoms": [{"kind": "references_cve", "cve_id": "CVE-2026-9001"}],
            },
            "res:low": {
                "id": "res:low",
                "repo": "acme/low",
                "kind": "github:issue",
                "canonical_url": "https://github.com/acme/low/issues/1",
                "importance_score": 1,
                "fetched_ts": now_a - 900.0,
                "atoms": [{"kind": "mentions", "term": "docs"}],
            },
        }

        touched_a = presence.tick(slack_ms=80.0, now_ts=now_a)
        assert touched_a
        list_calls_a = [
            _repo_from_list_url(url)
            for url in calls
            if "/repos/" in url and "/pulls?" in url
        ]
        assert len(list_calls_a) == 2
        assert list_calls_a[0] == "acme/high"
        assert set(list_calls_a) == {"acme/high", "acme/low"}

        calls.clear()
        touched_b = presence.tick(slack_ms=80.0, now_ts=now_a + 61.0)
        assert touched_b
        list_calls_b = [
            _repo_from_list_url(url)
            for url in calls
            if "/repos/" in url and "/pulls?" in url
        ]
        assert len(list_calls_b) == 2
        assert list_calls_b[0] == "acme/high"
        assert set(list_calls_b) == {"acme/high", "acme/new"}

        graph = presence.graph_snapshot()
        status = (
            graph.get("status", {}) if isinstance(graph.get("status", {}), dict) else {}
        )
        assert float(status.get("frontier_exploration_ratio", 0.0) or 0.0) > 0.0
        selected = status.get("frontier_selected_repos", [])
        assert isinstance(selected, list)
        assert "acme/high" in selected
        assert any(
            isinstance(row, dict)
            and str(row.get("event", "")).strip() == "github_frontier_plan"
            for row in graph.get("events", [])
        )


def test_github_presence_applies_regime_policy_to_budgets(monkeypatch: Any) -> None:
    calls: list[str] = []

    def fake_fetcher(url: str, _config: dict[str, Any]) -> dict[str, Any]:
        calls.append(url)
        if "/pulls?" in url:
            return _success_result(url, [])
        return {
            "ok": False,
            "url": url,
            "canonical_url": canonical_github_url(url),
            "status": 404,
            "duration_ms": 1,
            "error": "http_error",
        }

    def _repo_from_list_url(url: str) -> str:
        marker = "/repos/"
        if marker not in url:
            return ""
        tail = url.split(marker, 1)[1]
        return tail.split("/pulls", 1)[0].strip().lower()

    from code.world_web import graph_queries as graph_queries_module

    def _fake_run_named_graph_query(
        _graph: dict[str, Any],
        query_name: str,
        args: dict[str, Any] | None = None,
        simulation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = (args, simulation)
        if str(query_name).strip().lower() == "cyber_regime_state":
            return {
                "query": "cyber_regime_state",
                "snapshot_hash": "snap:regime-policy",
                "result": {
                    "state": "active_exploitation_wave",
                    "posterior": {
                        "baseline": 0.10,
                        "elevated_chatter": 0.12,
                        "active_exploitation_wave": 0.70,
                        "supply_chain_campaign": 0.05,
                        "geopolitical_targeting_shift": 0.03,
                    },
                    "policy": {
                        "risk_score_threshold": 6,
                        "crawl_budget_multiplier": 2.0,
                        "query_expansion_multiplier": 2.0,
                        "pressure": 0.77,
                    },
                },
            }
        return {
            "query": str(query_name),
            "snapshot_hash": "snap:noop",
            "result": {},
        }

    monkeypatch.setattr(
        graph_queries_module,
        "run_named_graph_query",
        _fake_run_named_graph_query,
    )

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        _write_config(
            part_root,
            {
                "enabled": True,
                "repos": ["acme/high", "acme/medium", "acme/low"],
                "max_repos": 1,
                "max_items_per_repo": 1,
                "max_github_fetches_per_tick": 1,
                "repo_cooldown_s": 0,
                "url_cooldown_s": 0,
                "endpoint_order": ["pulls"],
                "include_conversation_context": False,
                "frontier_exploration_ratio": 0.0,
                "regime_budget_enabled": True,
                "regime_policy_refresh_s": 1,
            },
        )

        presence = GithubPresence(part_root, fetcher=fake_fetcher)
        touched = presence.tick(slack_ms=80.0, now_ts=1_700_000_360.0)
        assert touched

        list_calls = [
            _repo_from_list_url(url)
            for url in calls
            if "/repos/" in url and "/pulls?" in url
        ]
        assert len(list_calls) == 2

        graph = presence.graph_snapshot()
        status = (
            graph.get("status", {}) if isinstance(graph.get("status", {}), dict) else {}
        )
        assert str(status.get("regime_state", "")) == "active_exploitation_wave"
        assert int(status.get("regime_base_fetch_budget", 0) or 0) == 1
        assert int(status.get("regime_effective_fetch_budget", 0) or 0) == 2
        assert int(status.get("regime_base_max_repos", 0) or 0) == 1
        assert int(status.get("regime_effective_max_repos", 0) or 0) == 2
        assert any(
            isinstance(row, dict)
            and str(row.get("event", "")).strip() == "github_regime_budget_applied"
            for row in graph.get("events", [])
        )


def test_github_queries_and_facts_snapshot() -> None:
    simulation = {
        "nexus_graph": {
            "nodes": [
                {
                    "id": "url:gh1",
                    "role": "web:url",
                    "label": "acme repo pr",
                    "extension": {
                        "canonical_url": "https://github.com/acme/repo/pull/7",
                        "next_allowed_fetch_ts": 0.0,
                        "last_fetch_ts": 120.0,
                        "fail_count": 0,
                        "last_status": "ok",
                        "source_hint": "github",
                    },
                },
                {
                    "id": "res:gh1",
                    "role": "web:resource",
                    "label": "Bump parser dependency",
                    "extension": {
                        "canonical_url": "https://github.com/acme/repo/pull/7",
                        "fetched_ts": 120.0,
                        "content_hash": "hash-gh-pr",
                        "title": "Bump parser dependency",
                        "source_url_id": "url:gh1",
                        "kind": "github:pr",
                        "repo": "acme/repo",
                        "number": 7,
                        "labels": ["security"],
                        "authors": ["octocat"],
                        "updated_at": "2026-02-27T00:00:05Z",
                        "importance_score": 8,
                        "atoms": [
                            {
                                "kind": "references_cve",
                                "repo": "acme/repo",
                                "cve_id": "CVE-2026-12345",
                            },
                            {
                                "kind": "changes_dependency",
                                "repo": "acme/repo",
                                "dep_name": "package.json",
                            },
                        ],
                        "state": "closed",
                        "merged_at": "2026-02-27T00:00:06Z",
                    },
                },
            ],
            "edges": [
                {
                    "source": "res:gh1",
                    "target": "url:gh1",
                    "kind": "web:source_of",
                }
            ],
        },
        "crawler_graph": {
            "status": {
                "github": {
                    "monitored_repos": ["acme/repo"],
                    "queue_length": 1,
                    "active_fetches": 0,
                    "cooldown_blocks": 2,
                }
            },
            "events": [
                {
                    "id": "evt:github-1",
                    "event": "github_fetch_completed",
                    "kind": "github_fetch_completed",
                    "ts": "2026-02-27T00:00:07Z",
                }
            ],
        },
        "presence_dynamics": {
            "tick": 9,
            "daimoi_outcome_summary": {"food": 0, "death": 0, "total": 0},
            "daimoi_outcome_trails": [],
            "field_particles": [],
            "resource_heartbeat": {"devices": {}},
        },
    }

    nexus = simulation["nexus_graph"]

    status = run_named_graph_query(nexus, "github_status", simulation=simulation)
    assert status.get("result", {}).get("resource_node_count") == 1
    assert status.get("result", {}).get("monitored_repos") == ["acme/repo"]

    summary = run_named_graph_query(
        nexus,
        "github_repo_summary",
        args={"repo": "acme/repo"},
        simulation=simulation,
    )
    assert summary.get("result", {}).get("repo") == "acme/repo"
    assert summary.get("result", {}).get("count") == 1

    found = run_named_graph_query(
        nexus,
        "github_find",
        args={"term": "cve", "repo": "acme/repo", "limit": 8},
        simulation=simulation,
    )
    assert found.get("result", {}).get("count", 0) >= 1

    recent = run_named_graph_query(
        nexus,
        "github_recent_changes",
        args={"window_ticks": 60, "limit": 8},
        simulation=simulation,
    )
    assert recent.get("result", {}).get("count", 0) >= 1

    with tempfile.TemporaryDirectory() as td:
        facts = build_facts_snapshot(simulation, part_root=Path(td))
        github = facts.get("github", {})
        assert github.get("monitored_repos") == ["acme/repo"]
        assert len(github.get("recent_resources", [])) == 1
        assert len(github.get("top_atoms", [])) >= 1
