from __future__ import annotations

import tempfile
from pathlib import Path

from code.world_web.graph_queries import build_facts_snapshot, run_named_graph_query


def _nexus_graph() -> dict:
    return {
        "nodes": [
            {
                "id": "url:aaaa",
                "role": "web:url",
                "label": "A",
                "canonical_url": "https://example.org/a",
                "next_allowed_fetch_ts": 0.0,
                "fail_count": 0,
                "last_fetch_ts": 10.0,
                "last_status": "ok",
            },
            {
                "id": "url:bbbb",
                "role": "web:url",
                "label": "B",
                "canonical_url": "https://example.org/b",
                "next_allowed_fetch_ts": 20.0,
                "fail_count": 1,
                "last_fetch_ts": 12.0,
                "last_status": "error",
            },
            {
                "id": "res:1111",
                "role": "web:resource",
                "label": "Doc A",
                "canonical_url": "https://example.org/a",
                "fetched_ts": 10.0,
                "content_hash": "hash-a",
            },
            {"id": "file:a", "role": "file", "label": "A.md", "importance": 0.7},
        ],
        "edges": [
            {"source": "res:1111", "target": "url:aaaa", "kind": "web:source_of"},
            {"source": "res:1111", "target": "url:bbbb", "kind": "web:links_to"},
            {"source": "file:a", "target": "url:aaaa", "kind": "mentions"},
        ],
    }


def _simulation_payload() -> dict:
    return {
        "nexus_graph": _nexus_graph(),
        "presence_dynamics": {
            "tick": 77,
            "daimoi_outcome_summary": {"food": 2, "death": 1, "total": 3},
            "daimoi_outcome_trails": [
                {
                    "seq": 1,
                    "tick": 76,
                    "daimoi_id": "field:witness_thread:001",
                    "outcome": "food",
                    "graph_node_id": "url:aaaa",
                    "reason": "resource_consumed",
                    "trail_steps": 6,
                    "intensity": 0.44,
                    "ts": "2026-02-27T00:00:00+00:00",
                },
                {
                    "seq": 2,
                    "tick": 77,
                    "daimoi_id": "field:witness_thread:001",
                    "outcome": "death",
                    "graph_node_id": "url:bbbb",
                    "reason": "timeout",
                    "trail_steps": 7,
                    "intensity": 0.52,
                    "ts": "2026-02-27T00:00:01+00:00",
                },
            ],
            "field_particles": [
                {
                    "id": "field:witness_thread:001",
                    "presence_id": "witness_thread",
                    "top_job": "invoke_graph_crawl",
                    "graph_node_id": "url:bbbb",
                    "route_node_id": "url:aaaa",
                    "message_probability": 0.61,
                    "collision_count": 2,
                    "age": 77,
                }
            ],
            "resource_heartbeat": {
                "devices": {
                    "cpu": {"utilization": 32.0},
                    "gpu1": {"utilization": 18.0},
                }
            },
        },
        "crawler_graph": {
            "status": {"queue_length": 2},
            "stats": {"event_count": 3, "web_role_counts": {"web:resource": 1}},
            "nodes": [
                {
                    "id": "url:arxiv-1",
                    "role": "web:url",
                    "canonical_url": "https://arxiv.org/abs/2602.23342",
                    "url": "https://arxiv.org/abs/2602.23342",
                    "title": "[2602.23342] Sample Paper One",
                    "last_status": "ok",
                    "last_fetch_ts": 120.0,
                },
                {
                    "id": "url:arxiv-2",
                    "role": "web:url",
                    "canonical_url": "https://arxiv.org/abs/2602.23344",
                    "url": "https://arxiv.org/abs/2602.23344",
                    "title": "[2602.23344] Sample Paper Two",
                    "last_status": "queued",
                    "last_fetch_ts": 0.0,
                },
                {
                    "id": "url:web-1",
                    "role": "web:url",
                    "canonical_url": "https://example.org/a",
                    "last_status": "ok",
                    "last_fetch_ts": 100.0,
                },
            ],
            "events": [
                {
                    "id": "evt:web",
                    "kind": "web_fetch_completed",
                    "ts": 100.0,
                    "url_id": "url:aaaa",
                    "reason": "",
                    "status": "ok",
                }
            ],
        },
    }


def test_run_named_graph_query_overview_neighbors_and_search() -> None:
    overview = run_named_graph_query(_nexus_graph(), "overview")
    assert overview.get("query") == "overview"
    assert overview.get("result", {}).get("node_count") == 4
    assert overview.get("result", {}).get("edge_count") == 3
    assert len(str(overview.get("snapshot_hash", ""))) == 64

    neighbors = run_named_graph_query(
        _nexus_graph(),
        "neighbors",
        args={"node_id": "url:aaaa"},
    )
    assert neighbors.get("result", {}).get("node_id") == "url:aaaa"
    assert neighbors.get("result", {}).get("neighbor_count", 0) >= 1

    search = run_named_graph_query(
        _nexus_graph(), "search", args={"q": "example.org/b"}
    )
    assert search.get("result", {}).get("count") == 1
    assert search.get("result", {}).get("nodes", [])[0].get("id") == "url:bbbb"

    touched = run_named_graph_query(
        _nexus_graph(), "recently_touched", args={"limit": 2}
    )
    assert touched.get("query") == "recently_updated"
    assert len(touched.get("result", {}).get("nodes", [])) == 2


def test_run_named_graph_query_url_status_and_resource_for_url() -> None:
    status = run_named_graph_query(
        _nexus_graph(),
        "url_status",
        args={"target": "https://example.org/a"},
    )
    found = status.get("result", {}).get("found", {})
    assert found.get("id") == "url:aaaa"
    assert found.get("last_status") == "ok"

    resource = run_named_graph_query(
        _nexus_graph(),
        "resource_for_url",
        args={"target": "url:aaaa"},
    )
    assert resource.get("result", {}).get("count") == 1
    assert resource.get("result", {}).get("resources", [])[0].get("id") == "res:1111"


def test_run_named_graph_query_named_menu_uses_simulation_context() -> None:
    simulation = _simulation_payload()
    graph = simulation.get("nexus_graph", {})

    explain = run_named_graph_query(
        graph,
        "explain_daimoi",
        args={"daimoi_id": "field:witness_thread:001"},
        simulation=simulation,
    )
    assert explain.get("result", {}).get("status") in {"alive", "death", "food"}
    assert explain.get("result", {}).get("last_outcome", {}).get("outcome") == "death"

    outcomes = run_named_graph_query(
        graph,
        "recent_outcomes",
        args={"window_ticks": 20, "limit": 8},
        simulation=simulation,
    )
    assert outcomes.get("result", {}).get("count") == 2

    crawler = run_named_graph_query(
        graph,
        "crawler_status",
        simulation=simulation,
    )
    assert crawler.get("result", {}).get("queue_length") == 2
    assert crawler.get("result", {}).get("arxiv_abs_count") == 2
    assert crawler.get("result", {}).get("arxiv_abs_fetched") == 1

    arxiv = run_named_graph_query(
        graph,
        "arxiv_papers",
        args={"limit": 5},
        simulation=simulation,
    )
    assert arxiv.get("result", {}).get("count_total") == 2
    assert arxiv.get("result", {}).get("count_fetched") == 1
    first = arxiv.get("result", {}).get("papers", [])[0]
    assert first.get("canonical_url") == "https://arxiv.org/abs/2602.23342"

    resource_summary = run_named_graph_query(
        graph,
        "web_resource_summary",
        args={"target": "url:aaaa"},
        simulation=simulation,
    )
    assert (
        resource_summary.get("result", {}).get("found", {}).get("res_id") == "res:1111"
    )

    summary = run_named_graph_query(
        graph,
        "graph_summary",
        args={"scope": "all", "n": 5},
        simulation=simulation,
    )
    assert summary.get("result", {}).get("node_count") == 4


def test_run_named_graph_query_github_threat_radar_ranks_security_rows() -> None:
    graph = {
        "nodes": [
            {
                "id": "res:gh-issue",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 42,
                "canonical_url": "https://github.com/openai/codex/issues/42",
                "title": "CVE-2026-1234 token leak in auth flow",
                "state": "open",
                "fetched_ts": 200.0,
                "importance_score": 7,
                "atoms": [
                    {
                        "kind": "references_cve",
                        "repo": "openai/codex",
                        "cve_id": "CVE-2026-1234",
                    },
                    {
                        "kind": "changes_dependency",
                        "repo": "openai/codex",
                        "dep_name": "package-lock.json",
                    },
                    {
                        "kind": "has_label",
                        "repo": "openai/codex",
                        "label": "security",
                    },
                ],
            },
            {
                "id": "res:gh-pr",
                "role": "web:resource",
                "kind": "github:pr",
                "repo": "openai/codex",
                "number": 99,
                "canonical_url": "https://github.com/openai/codex/pull/99",
                "title": "Refactor docs",
                "state": "open",
                "fetched_ts": 199.0,
                "importance_score": 1,
                "atoms": [],
            },
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    result = payload.get("result", {})
    assert result.get("count", 0) >= 1
    first = (result.get("threats", []) or [{}])[0]
    assert first.get("repo") == "openai/codex"
    assert first.get("number") == 42
    assert first.get("risk_score", 0) >= 8
    assert "references_cve" in (first.get("signals", []) or [])


def test_build_facts_snapshot_includes_web_sections_and_writes_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        simulation = _simulation_payload()
        facts = build_facts_snapshot(simulation, part_root=root)
        assert facts.get("record") == "eta-mu.facts-snapshot.v1"
        assert facts.get("counts", {}).get("nodes_by_role", {}).get("web:url") == 2
        assert len(facts.get("web", {}).get("urls", [])) == 2
        assert len(facts.get("web", {}).get("resources", [])) == 1
        assert facts.get("dynamics", {}).get("daimoi_outcomes", {}).get("food") == 2
        assert len(str(facts.get("snapshot_hash", ""))) == 64
        snapshot_path = str(facts.get("snapshot_path", ""))
        assert snapshot_path
        assert Path(snapshot_path).exists()
        tables = facts.get("tables", {})
        assert isinstance(tables, dict)
        assert int(tables.get("web_url", {}).get("row_count", 0)) == 2
        assert int(tables.get("web_resource", {}).get("row_count", 0)) == 1
        assert int(tables.get("food", {}).get("row_count", 0)) == 1
        assert int(tables.get("death", {}).get("row_count", 0)) == 1
        assert Path(str(tables.get("node", {}).get("path", ""))).exists()
        assert Path(str(tables.get("event_collision", {}).get("path", ""))).exists()
