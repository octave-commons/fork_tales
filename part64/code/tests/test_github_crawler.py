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
                "max_github_fetches_per_tick": 3,
                "repo_cooldown_s": 0,
                "url_cooldown_s": 1200,
                "endpoint_order": ["pulls"],
            },
        )

        presence = GithubPresence(part_root, fetcher=fake_fetcher)
        touched = presence.tick(slack_ms=80.0, now_ts=1_700_000_000.0)
        assert touched
        assert len(calls) == 3

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
