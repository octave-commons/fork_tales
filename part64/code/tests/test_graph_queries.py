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


def test_build_facts_snapshot_includes_web_sections_and_writes_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        simulation = {
            "nexus_graph": _nexus_graph(),
            "presence_dynamics": {
                "daimoi_outcome_summary": {"food": 2, "death": 1, "total": 3},
                "daimoi_outcome_trails": [
                    {
                        "seq": 1,
                        "outcome": "food",
                        "graph_node_id": "url:aaaa",
                        "reason": "resource_consumed",
                        "ts": "2026-02-27T00:00:00+00:00",
                    }
                ],
            },
            "crawler_graph": {
                "events": [
                    {
                        "id": "evt:web",
                        "kind": "web_fetch_completed",
                        "ts": 100.0,
                        "url_id": "url:aaaa",
                        "reason": "",
                        "status": "ok",
                    }
                ]
            },
        }
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
