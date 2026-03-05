from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from code.world_web import graph_queries as graph_queries_module
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


def test_run_named_graph_query_hormuz_threat_radar_scores_maritime_atoms() -> None:
    graph = {
        "nodes": [
            {
                "id": "res:ukmto",
                "role": "web:resource",
                "kind": "maritime:ukmto_advisory",
                "canonical_url": "https://www.ukmto.org/advisory/003-26",
                "title": "UKMTO advisory update",
                "fetched_ts": 220.0,
                "atoms": [
                    {
                        "kind": "hazard",
                        "region": "hormuz",
                        "label": "military_activity",
                    },
                    {
                        "kind": "hazard",
                        "region": "hormuz",
                        "label": "electronic_interference",
                    },
                ],
            },
            {
                "id": "res:marad",
                "role": "web:resource",
                "kind": "maritime:marad_advisory",
                "canonical_url": "https://www.maritime.dot.gov/msci/2026-001-persian-gulf-strait-hormuz-and-gulf-oman-iranian-illegal-boarding-detention-seizure",
                "title": "MSCI 2026-001",
                "fetched_ts": 215.0,
                "atoms": [
                    {
                        "kind": "hazard",
                        "region": "hormuz",
                        "label": "boarding_seizure_risk",
                    }
                ],
            },
            {
                "id": "res:noise",
                "role": "web:resource",
                "kind": "github:issue",
                "canonical_url": "https://github.com/octocat/hello-world/issues/1",
                "fetched_ts": 300.0,
                "atoms": [
                    {
                        "kind": "references_cve",
                        "cve_id": "CVE-2026-9999",
                    }
                ],
            },
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "hormuz_threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    result = payload.get("result", {})
    assert result.get("count") == 2
    assert result.get("high_count", 0) >= 1
    assert result.get("low_count", 0) >= 1

    first = (result.get("threats", []) or [{}])[0]
    assert first.get("kind") == "maritime:ukmto_advisory"
    assert first.get("risk_score") == 5
    assert sorted(first.get("labels", [])) == [
        "electronic_interference",
        "military_activity",
    ]


def test_run_named_graph_query_hormuz_threat_radar_returns_watchlist_sources(
    monkeypatch: Any,
) -> None:
    def _fake_watchlist_sources() -> list[dict[str, Any]]:
        return [
            {
                "url": "https://www.ukmto.org/advisory/003-26",
                "kind": "maritime:ukmto_advisory",
                "title": "UKMTO Advisory 003-26",
                "source_type": "website",
                "domain_id": "hormuz",
            },
            {
                "url": "https://api.gdeltproject.org/api/v2/doc/doc?query=%22Strait%20of%20Hormuz%22&mode=ArtList&maxrecords=50&format=json",
                "kind": "maritime:dataset",
                "title": "GDELT Document API - Strait of Hormuz",
                "source_type": "dataset",
                "domain_id": "hormuz",
            },
        ]

    monkeypatch.setattr(
        graph_queries_module,
        "_load_world_watchlist_sources",
        _fake_watchlist_sources,
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "hormuz_threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    result = payload.get("result", {})
    assert result.get("count") == 0
    assert result.get("source_count") == 2
    sources = result.get("sources", [])
    assert isinstance(sources, list)
    assert len(sources) == 2
    assert str(sources[0].get("url", "")).startswith("https://")
    assert str(sources[1].get("source_type", "")).strip() in {"website", "dataset"}


def test_run_named_graph_query_hormuz_threat_radar_receives_compute_budget(
    monkeypatch: Any,
) -> None:
    captured_args: dict[str, Any] = {}

    def _fake_hormuz_threat_radar(
        _nodes: list[dict[str, Any]],
        args: dict[str, Any],
    ) -> dict[str, Any]:
        captured_args.update(dict(args))
        return {
            "kind": "",
            "window_ticks": 1440,
            "count": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "hot_kinds": [],
            "scoring": {"mode": "deterministic", "llm_enabled": False},
            "source_count": 0,
            "sources": [],
            "threats": [],
        }

    monkeypatch.setattr(
        graph_queries_module,
        "_query_hormuz_threat_radar",
        _fake_hormuz_threat_radar,
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "hormuz_threat_radar",
        args={"limit": 8},
        simulation={
            "presence_dynamics": {
                "resource_consumption": {
                    "control_budget": {"mode": "reduced", "ratio": 0.15},
                    "queue_ratio": 0.8,
                    "cpu_sentinel_burn_active": False,
                },
                "compute_jobs_180s": 12,
            }
        },
    )

    assert payload.get("query") == "hormuz_threat_radar"
    budget = captured_args.get("_threat_compute_budget", {})
    assert isinstance(budget, dict)
    assert budget.get("bound") is True
    assert budget.get("mode") == "reduced"


def test_run_named_graph_query_geopolitical_news_radar_surfaces_frontier_and_sources() -> (
    None
):
    graph = {
        "nodes": [
            {
                "id": "res:ukmto",
                "role": "web:resource",
                "kind": "maritime:ukmto_advisory",
                "canonical_url": "https://www.ukmto.org/advisory/003-26",
                "title": "UKMTO advisory update",
                "summary": "military activity in Strait of Hormuz",
                "fetched_ts": 220.0,
                "importance_score": 4,
                "atoms": [
                    {
                        "kind": "hazard",
                        "region": "hormuz",
                        "label": "military_activity",
                    }
                ],
            },
            {
                "id": "res:reuters",
                "role": "web:resource",
                "kind": "maritime:news_report",
                "canonical_url": "https://www.reuters.com/world/middle-east/shipping-disruption-2026-03-01/",
                "title": "Shipping disruption in Red Sea",
                "summary": "drone strike disrupts shipping corridor",
                "fetched_ts": 219.0,
                "importance_score": 3,
                "atoms": [],
            },
            {
                "id": "res:noise",
                "role": "web:resource",
                "kind": "github:issue",
                "canonical_url": "https://github.com/octocat/hello-world/issues/1",
                "title": "ignore github row",
                "fetched_ts": 230.0,
                "atoms": [],
            },
            {
                "id": "res:gh-api",
                "role": "web:resource",
                "kind": "web:article",
                "canonical_url": "https://api.github.com/repos/openai/codex/issues/9001",
                "title": "shipping disruption and missile strike",
                "summary": "red sea conflict and maritime advisory",
                "fetched_ts": 229.0,
                "atoms": [],
            },
            {
                "id": "url:ukmto",
                "role": "web:url",
                "canonical_url": "https://www.ukmto.org/advisory/003-26",
                "last_status": "ok",
                "fail_count": 0,
            },
            {
                "id": "url:reuters",
                "role": "web:url",
                "canonical_url": "https://www.reuters.com/world/middle-east/shipping-disruption-2026-03-01/",
                "last_status": "ok",
                "fail_count": 0,
            },
            {
                "id": "url:frontier",
                "role": "web:url",
                "canonical_url": "https://www.bbc.com/news/world-middle-east-20260301",
                "last_status": "queued",
                "fail_count": 0,
                "next_allowed_fetch_ts": 0.0,
            },
        ],
        "edges": [
            {"source": "res:ukmto", "target": "url:ukmto", "kind": "web:source_of"},
            {
                "source": "res:reuters",
                "target": "url:reuters",
                "kind": "web:source_of",
            },
            {"source": "res:ukmto", "target": "url:frontier", "kind": "web:links_to"},
            {
                "source": "res:reuters",
                "target": "url:frontier",
                "kind": "web:links_to",
            },
        ],
    }

    payload = run_named_graph_query(
        graph,
        "geopolitical_news_radar",
        args={"window_ticks": 400, "limit": 8},
        simulation={
            "crawler_graph": {
                "status": {"queue_length": 3, "active_fetches": 1},
                "events": [
                    {
                        "id": "evt-1",
                        "kind": "web_fetch_completed",
                        "ts": 221.0,
                        "url_id": "url:ukmto",
                        "status": "ok",
                        "reason": "",
                    }
                ],
            }
        },
    )

    assert payload.get("query") == "geopolitical_news_radar"
    result = payload.get("result", {})
    assert int(result.get("count", 0) or 0) >= 2
    assert int(result.get("source_count", 0) or 0) >= 2
    assert all(
        not str(row.get("kind", "")).strip().lower().startswith("github:")
        for row in (result.get("threats", []) or [])
        if isinstance(row, dict)
    )
    assert all(
        "github" not in str(row.get("canonical_url", "")).strip().lower()
        for row in (result.get("threats", []) or [])
        if isinstance(row, dict)
    )
    discovery = result.get("discovery", {})
    frontier = (
        discovery.get("frontier_candidates", []) if isinstance(discovery, dict) else []
    )
    assert any(
        str(row.get("canonical_url", "")).startswith("https://www.bbc.com/")
        and int(row.get("inbound_links", 0) or 0) >= 2
        for row in frontier
        if isinstance(row, dict)
    )
    traversal = result.get("traversal", {})
    assert int(traversal.get("queue_length", 0) or 0) == 3


def test_run_named_graph_query_global_alias_maps_to_geopolitical_news_radar(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        graph_queries_module,
        "_load_world_watchlist_sources",
        lambda: [],
    )
    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "global",
        args={"window_ticks": 400, "limit": 8},
    )
    assert payload.get("query") == "geopolitical_news_radar"
    result = payload.get("result", {})
    assert result.get("count") == 0


def test_run_named_graph_query_geopolitical_news_radar_surfaces_watchlist_sources_when_empty(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        graph_queries_module,
        "_load_world_watchlist_sources",
        lambda: [
            {
                "url": "https://www.ukmto.org/advisory/003-26",
                "kind": "maritime:ukmto_advisory",
                "title": "UKMTO Advisory 003-26",
                "source_type": "website",
                "domain_id": "hormuz",
            }
        ],
    )
    monkeypatch.setattr(
        graph_queries_module,
        "_load_weaver_fetched_source_rows",
        lambda: [],
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "geopolitical_news_radar",
        args={"window_ticks": 400, "limit": 8},
        simulation={"crawler_graph": {"status": {}, "events": []}},
    )

    assert payload.get("query") == "geopolitical_news_radar"
    result = payload.get("result", {})
    assert int(result.get("count", 0) or 0) == 1
    assert int(result.get("low_count", 0) or 0) == 1
    assert int(result.get("source_count", 0) or 0) >= 1
    threats = result.get("threats", [])
    assert isinstance(threats, list)
    assert len(threats) == 1
    assert str(threats[0].get("risk_level", "") or "") == "low"
    assert bool(threats[0].get("provisional", False)) is True
    labels = threats[0].get("labels", [])
    assert isinstance(labels, list)
    assert "watchlist_seed" in [str(value) for value in labels]
    sources = result.get("sources", [])
    assert isinstance(sources, list)
    assert any(
        str(row.get("url", "") or "") == "https://www.ukmto.org/advisory/003-26"
        for row in sources
        if isinstance(row, dict)
    )


def test_run_named_graph_query_geopolitical_news_radar_can_omit_provisional_watchlist_rows(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        graph_queries_module,
        "_load_world_watchlist_sources",
        lambda: [
            {
                "url": "https://www.ukmto.org/advisory/003-26",
                "kind": "maritime:ukmto_advisory",
                "title": "UKMTO Advisory 003-26",
                "source_type": "website",
                "domain_id": "hormuz",
            }
        ],
    )
    monkeypatch.setattr(
        graph_queries_module,
        "_load_weaver_fetched_source_rows",
        lambda: [],
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "geopolitical_news_radar",
        args={"window_ticks": 400, "limit": 8, "include_provisional": "false"},
        simulation={"crawler_graph": {"status": {}, "events": []}},
    )

    assert payload.get("query") == "geopolitical_news_radar"
    result = payload.get("result", {})
    assert int(result.get("count", 0) or 0) == 0
    assert int(result.get("source_count", 0) or 0) >= 1
    threats = result.get("threats", [])
    assert isinstance(threats, list)
    assert len(threats) == 0


def test_run_named_graph_query_geopolitical_news_radar_honors_bool_false_include_provisional(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        graph_queries_module,
        "_load_world_watchlist_sources",
        lambda: [
            {
                "url": "https://www.ukmto.org/advisory/003-26",
                "kind": "maritime:ukmto_advisory",
                "title": "UKMTO Advisory 003-26",
                "source_type": "website",
                "domain_id": "hormuz",
            }
        ],
    )
    monkeypatch.setattr(
        graph_queries_module,
        "_load_weaver_fetched_source_rows",
        lambda: [],
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "geopolitical_news_radar",
        args={"window_ticks": 400, "limit": 8, "include_provisional": False},
        simulation={"crawler_graph": {"status": {}, "events": []}},
    )

    assert payload.get("query") == "geopolitical_news_radar"
    result = payload.get("result", {})
    assert int(result.get("count", 0) or 0) == 0
    assert int(result.get("provisional_count", 0) or 0) == 0
    quality = result.get("quality", {})
    assert isinstance(quality, dict)
    assert bool(quality.get("seed_only", False)) is False


def test_run_named_graph_query_geopolitical_news_radar_surfaces_non_provisional_weaver_source_evidence(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        graph_queries_module,
        "_load_world_watchlist_sources",
        lambda: [
            {
                "url": "https://www.ukmto.org/advisory/003-26",
                "kind": "maritime:ukmto_advisory",
                "title": "UKMTO Advisory 003-26",
                "source_type": "website",
                "domain_id": "hormuz",
            }
        ],
    )
    monkeypatch.setattr(
        graph_queries_module,
        "_load_weaver_fetched_source_rows",
        lambda: [
            {
                "url": "https://www.ukmto.org/advisory/003-26",
                "normalized_url": "https://www.ukmto.org/advisory/003-26",
                "fetched_ts": 1772648882.388,
                "title": "UKMTO advisory",
                "summary": "Fetched advisory payload with no elevated threat terms.",
                "text_excerpt": "Fetched advisory payload.",
            }
        ],
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "geopolitical_news_radar",
        args={
            "window_ticks": 400,
            "limit": 8,
            "include_provisional": False,
            "include_watchlist_source_evidence": True,
        },
        simulation={"crawler_graph": {"status": {}, "events": []}},
    )

    assert payload.get("query") == "geopolitical_news_radar"
    result = payload.get("result", {})
    assert int(result.get("count", 0) or 0) == 1
    assert int(result.get("provisional_count", 0) or 0) == 0
    assert int(result.get("non_provisional_count", 0) or 0) == 1
    threats = result.get("threats", [])
    assert isinstance(threats, list)
    assert len(threats) == 1
    row = threats[0]
    assert bool(row.get("provisional", False)) is False
    assert (
        str(row.get("canonical_url", "") or "")
        == "https://www.ukmto.org/advisory/003-26"
    )
    labels = row.get("labels", [])
    assert isinstance(labels, list)
    assert "crawl_evidence" in [str(value) for value in labels]
    sources = result.get("sources", [])
    assert isinstance(sources, list)
    assert any(
        str(source_row.get("url", "") or "") == "https://www.ukmto.org/advisory/003-26"
        and int(source_row.get("count", 0) or 0) >= 1
        for source_row in sources
        if isinstance(source_row, dict)
    )


def test_run_named_graph_query_geopolitical_news_radar_does_not_emit_watchlist_source_evidence_by_default(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        graph_queries_module,
        "_load_world_watchlist_sources",
        lambda: [
            {
                "url": "https://www.ukmto.org/advisory/003-26",
                "kind": "maritime:ukmto_advisory",
                "title": "UKMTO Advisory 003-26",
                "source_type": "website",
                "domain_id": "hormuz",
            }
        ],
    )
    monkeypatch.setattr(
        graph_queries_module,
        "_load_weaver_fetched_source_rows",
        lambda: [
            {
                "url": "https://www.ukmto.org/advisory/003-26",
                "normalized_url": "https://www.ukmto.org/advisory/003-26",
                "fetched_ts": 1772648882.388,
                "title": "UKMTO advisory",
                "summary": "Fetched advisory payload with no elevated threat terms.",
                "text_excerpt": "Fetched advisory payload.",
            }
        ],
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "geopolitical_news_radar",
        args={"window_ticks": 400, "limit": 8, "include_provisional": False},
        simulation={"crawler_graph": {"status": {}, "events": []}},
    )

    assert payload.get("query") == "geopolitical_news_radar"
    result = payload.get("result", {})
    assert int(result.get("count", 0) or 0) == 0
    threats = result.get("threats", [])
    assert isinstance(threats, list)
    assert len(threats) == 0


def test_run_named_graph_query_geopolitical_news_radar_filters_raw_feed_source_urls(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        graph_queries_module,
        "_load_world_watchlist_sources",
        lambda: [
            {
                "url": "https://hnrss.org/frontpage",
                "kind": "feed:rss",
                "title": "Hacker News Frontpage RSS",
                "source_type": "rss",
                "domain_id": "hacker_news",
            }
        ],
    )
    graph = {
        "nodes": [
            {
                "id": "res:feed",
                "role": "web:resource",
                "kind": "feed:rss",
                "canonical_url": "https://hnrss.org/frontpage",
                "title": "Hacker News feed",
                "summary": "military activity and shipping disruption",
                "fetched_ts": 220.0,
                "importance_score": 4,
                "atoms": [
                    {
                        "kind": "hazard",
                        "region": "hormuz",
                        "label": "military_activity",
                    }
                ],
            },
            {
                "id": "res:article",
                "role": "web:resource",
                "kind": "web:article",
                "canonical_url": "https://example.org/security/advisory-42",
                "title": "Shipping disruption advisory",
                "summary": "military activity and shipping disruption",
                "fetched_ts": 221.0,
                "importance_score": 4,
                "atoms": [
                    {
                        "kind": "hazard",
                        "region": "hormuz",
                        "label": "military_activity",
                    }
                ],
            },
            {
                "id": "url:feed",
                "role": "web:url",
                "canonical_url": "https://hnrss.org/frontpage",
                "last_status": "ok",
                "fail_count": 0,
            },
            {
                "id": "url:article",
                "role": "web:url",
                "canonical_url": "https://example.org/security/advisory-42",
                "last_status": "ok",
                "fail_count": 0,
            },
        ],
        "edges": [
            {"source": "res:feed", "target": "url:feed", "kind": "web:source_of"},
            {
                "source": "res:article",
                "target": "url:article",
                "kind": "web:source_of",
            },
        ],
    }

    payload = run_named_graph_query(
        graph,
        "geopolitical_news_radar",
        args={"window_ticks": 400, "limit": 8, "include_provisional": False},
        simulation={"crawler_graph": {"status": {}, "events": []}},
    )

    result = payload.get("result", {})
    threats = result.get("threats", [])
    assert isinstance(threats, list)
    assert any(
        str(row.get("canonical_url", "") or "")
        == "https://example.org/security/advisory-42"
        for row in threats
        if isinstance(row, dict)
    )
    assert not any(
        str(row.get("canonical_url", "") or "") == "https://hnrss.org/frontpage"
        for row in threats
        if isinstance(row, dict)
    )


def test_run_named_graph_query_github_threat_radar_dedupes_content_hash() -> None:
    graph = {
        "nodes": [
            {
                "id": "res:gh-dup-a",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 420,
                "canonical_url": "https://github.com/openai/codex/issues/420",
                "content_hash": "same-hash-1",
                "title": "CVE-2026-4242 token leak",
                "state": "open",
                "fetched_ts": 210.0,
                "atoms": [
                    {
                        "kind": "references_cve",
                        "repo": "openai/codex",
                        "cve_id": "CVE-2026-4242",
                    }
                ],
            },
            {
                "id": "res:gh-dup-b",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 421,
                "canonical_url": "https://github.com/openai/codex/issues/421",
                "content_hash": "same-hash-1",
                "title": "CVE-2026-4242 token leak mirror",
                "state": "open",
                "fetched_ts": 209.0,
                "atoms": [
                    {
                        "kind": "references_cve",
                        "repo": "openai/codex",
                        "cve_id": "CVE-2026-4242",
                    }
                ],
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
    assert result.get("count") == 1
    assert result.get("dedupe_count") == 1
    threats = result.get("threats", [])
    assert len(threats) == 1
    assert (
        threats[0].get("canonical_url") == "https://github.com/openai/codex/issues/420"
    )


def test_run_named_graph_query_github_threat_radar_applies_corroboration_and_source_fields() -> (
    None
):
    graph = {
        "nodes": [
            {
                "id": "res:gh-a",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 100,
                "canonical_url": "https://github.com/openai/codex/issues/100",
                "title": "CVE-2026-1000 token leak",
                "fetched_ts": 310.0,
                "atoms": [
                    {"kind": "references_cve", "cve_id": "CVE-2026-1000"},
                    {"kind": "has_label", "label": "security"},
                ],
            },
            {
                "id": "res:gh-b",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 101,
                "canonical_url": "https://github.com/openai/codex/issues/101",
                "title": "Follow-up for CVE-2026-1000",
                "fetched_ts": 309.0,
                "atoms": [
                    {"kind": "references_cve", "cve_id": "CVE-2026-1000"},
                ],
            },
            {
                "id": "res:gh-rel",
                "role": "web:resource",
                "kind": "github:release",
                "repo": "openai/codex",
                "number": 0,
                "canonical_url": "https://github.com/openai/codex/releases/tag/v1.2.3",
                "title": "Security release for CVE-2026-1000",
                "fetched_ts": 311.0,
                "atoms": [
                    {"kind": "references_cve", "cve_id": "CVE-2026-1000"},
                ],
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
    scoring = result.get("scoring", {})
    assert scoring.get("source_weighting") is True
    assert scoring.get("corroboration_enabled") is True

    threats = result.get("threats", [])
    assert threats
    assert any(int(row.get("corroboration_count", 0) or 0) >= 2 for row in threats)
    assert any(int(row.get("source_tier", 0) or 0) >= 3 for row in threats)
    for row in threats:
        weight = float(row.get("source_weight", 0.0) or 0.0)
        assert 0.0 <= weight <= 1.0


def test_run_named_graph_query_github_threat_radar_reports_classifier_fields(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(graph_queries_module, "_THREAT_RADAR_CLASSIFIER_ENABLED", True)

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
                ],
            }
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    result = payload.get("result", {})
    scoring = result.get("scoring", {})
    assert scoring.get("classifier_enabled") is True
    assert scoring.get("classifier_version") == str(
        graph_queries_module._THREAT_RADAR_CLASSIFIER_VERSION
    )
    first = (result.get("threats", []) or [{}])[0]
    assert first.get("classifier_score") == first.get("deterministic_score")
    assert first.get("deterministic_score_legacy", 0) > 0
    probability = float(first.get("classifier_probability", 0.0) or 0.0)
    assert 0.0 <= probability <= 1.0


def test_run_named_graph_query_github_threat_radar_reports_weak_label_registry() -> (
    None
):
    graph = {
        "nodes": [
            {
                "id": "res:gh-adv",
                "role": "web:resource",
                "kind": "github:advisory",
                "repo": "openai/codex",
                "number": 0,
                "canonical_url": "https://github.com/openai/codex/security/advisories/GHSA-abcd-efgh-ijkl",
                "title": "Critical advisory for auth token bypass CVE-2026-7777",
                "state": "open",
                "fetched_ts": 510.0,
                "importance_score": 9,
                "atoms": [
                    {
                        "kind": "references_cve",
                        "repo": "openai/codex",
                        "cve_id": "CVE-2026-7777",
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
            }
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    result = payload.get("result", {})
    scoring = result.get("scoring", {})
    assert scoring.get("weak_supervision_enabled") is True
    assert scoring.get("weak_supervision_version") == "github_lf_v1"
    weak_counts = scoring.get("weak_label_counts", {})
    assert isinstance(weak_counts, dict)

    first = (result.get("threats", []) or [{}])[0]
    assert str(first.get("weak_label", "")).strip() in {
        "security_likely",
        "security_possible",
        "uncertain",
        "low_security_relevance",
    }
    assert isinstance(first.get("weak_label_votes", []), list)
    assert len(first.get("weak_label_votes", [])) >= 1
    assert int(first.get("weak_label_score", 0) or 0) >= 1
    confidence = float(first.get("weak_label_confidence", 0.0) or 0.0)
    assert 0.0 <= confidence <= 1.0


def test_run_named_graph_query_github_threat_radar_can_filter_low_weak_label_rows() -> (
    None
):
    graph = {
        "nodes": [
            {
                "id": "res:gh-positive",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 321,
                "canonical_url": "https://github.com/openai/codex/issues/321",
                "title": "security token workflow concern",
                "state": "open",
                "fetched_ts": 601.0,
                "atoms": [{"kind": "has_label", "label": "security"}],
            },
            {
                "id": "res:gh-noise",
                "role": "web:resource",
                "kind": "github:pr",
                "repo": "openai/codex",
                "number": 322,
                "canonical_url": "https://github.com/openai/codex/pull/322",
                "title": "refactor docs for onboarding",
                "state": "open",
                "fetched_ts": 600.0,
                "atoms": [],
            },
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8, "min_weak_label_score": 1},
    )

    result = payload.get("result", {})
    threats = result.get("threats", [])
    assert isinstance(threats, list)
    assert len(threats) == 1
    first = threats[0]
    assert int(first.get("weak_label_score", 0) or 0) >= 1
    assert "security" in str(first.get("title", "") or "").lower()
    scoring = result.get("scoring", {})
    assert bool(scoring.get("weak_label_filter_enabled", False)) is True
    assert int(scoring.get("weak_label_min_score", 0) or 0) == 1
    assert int(scoring.get("weak_label_filtered_out", 0) or 0) >= 1


def test_run_named_graph_query_github_threat_radar_can_disable_llm_by_query(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(graph_queries_module, "_THREAT_RADAR_LLM_ENABLED", True)

    called = {"count": 0}

    def _fake_llm_metrics(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        called["count"] += 1
        return {
            "enabled": True,
            "applied": True,
            "model": "stub-llm",
            "error": "",
            "metrics": {},
        }

    monkeypatch.setattr(graph_queries_module, "_threat_llm_metrics", _fake_llm_metrics)

    graph = {
        "nodes": [
            {
                "id": "res:gh-llm-off",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 17,
                "canonical_url": "https://github.com/openai/codex/issues/17",
                "title": "token leak CVE-2026-3131",
                "state": "open",
                "fetched_ts": 500.0,
                "importance_score": 6,
                "atoms": [
                    {"kind": "references_cve", "cve_id": "CVE-2026-3131"},
                    {"kind": "has_label", "label": "security"},
                ],
            }
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8, "llm_enabled": "0"},
    )

    result = payload.get("result", {})
    scoring = result.get("scoring", {})
    assert scoring.get("llm_requested") is False
    assert scoring.get("llm_enabled") is False
    assert scoring.get("llm_applied") is False
    assert str(scoring.get("llm_error", "")) == "disabled_by_query"
    assert called["count"] == 0


def test_run_named_graph_query_github_threat_radar_binds_to_minimal_compute_budget(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(graph_queries_module, "_THREAT_RADAR_LLM_ENABLED", True)

    called = {"count": 0}

    def _fake_llm_metrics(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        called["count"] += 1
        return {
            "enabled": True,
            "applied": True,
            "model": "stub-llm",
            "error": "",
            "metrics": {},
        }

    monkeypatch.setattr(graph_queries_module, "_threat_llm_metrics", _fake_llm_metrics)

    graph = {
        "nodes": [
            {
                "id": "res:gh-budget-min",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 44,
                "canonical_url": "https://github.com/openai/codex/issues/44",
                "title": "token leak CVE-2026-4444",
                "state": "open",
                "fetched_ts": 700.0,
                "importance_score": 8,
                "atoms": [
                    {"kind": "references_cve", "cve_id": "CVE-2026-4444"},
                    {"kind": "has_label", "label": "security"},
                ],
            }
        ],
        "edges": [],
    }
    simulation = {
        "presence_dynamics": {
            "resource_consumption": {
                "queue_ratio": 0.12,
                "cpu_sentinel_burn_active": False,
                "control_budget": {
                    "mode": "minimal",
                    "ratio": 0.04,
                },
            },
            "compute_jobs_180s": 3,
        }
    }

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8},
        simulation=simulation,
    )

    result = payload.get("result", {})
    scoring = result.get("scoring", {})
    budget = scoring.get("compute_budget", {})
    assert scoring.get("classifier_enabled") is False
    assert scoring.get("llm_requested") is True
    assert scoring.get("llm_allowed") is False
    assert scoring.get("llm_enabled") is False
    assert scoring.get("llm_applied") is False
    assert str(scoring.get("llm_error", "")) == "disabled_by_compute_budget"
    assert bool(budget.get("bound", False)) is True
    assert str(budget.get("mode", "")) == "minimal"
    assert bool(budget.get("allow_classifier", True)) is False
    assert bool(budget.get("allow_llm", True)) is False
    assert int(budget.get("llm_item_cap", 0) or 0) == 0
    assert called["count"] == 0


def test_run_named_graph_query_github_threat_radar_applies_budget_llm_item_cap(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(graph_queries_module, "_THREAT_RADAR_LLM_ENABLED", True)

    captured = {"max_items": -1}

    def _fake_llm_metrics(
        *,
        domain: str,
        rows: list[dict[str, Any]],
        max_items: int | None = None,
    ) -> dict[str, Any]:
        captured["max_items"] = -1 if max_items is None else int(max_items)
        return {
            "enabled": True,
            "applied": False,
            "model": "stub-llm",
            "error": "llm_empty_metrics",
            "metrics": {},
        }

    monkeypatch.setattr(graph_queries_module, "_threat_llm_metrics", _fake_llm_metrics)

    graph = {
        "nodes": [
            {
                "id": "res:gh-budget-cap-1",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 101,
                "canonical_url": "https://github.com/openai/codex/issues/101",
                "title": "auth token leak report",
                "state": "open",
                "fetched_ts": 900.0,
                "importance_score": 5,
                "atoms": [{"kind": "has_label", "label": "security"}],
            },
            {
                "id": "res:gh-budget-cap-2",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 102,
                "canonical_url": "https://github.com/openai/codex/issues/102",
                "title": "supply chain concern",
                "state": "open",
                "fetched_ts": 899.0,
                "importance_score": 5,
                "atoms": [{"kind": "mentions", "term": "supply"}],
            },
            {
                "id": "res:gh-budget-cap-3",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 103,
                "canonical_url": "https://github.com/openai/codex/issues/103",
                "title": "credential and token warning",
                "state": "open",
                "fetched_ts": 898.0,
                "importance_score": 5,
                "atoms": [{"kind": "mentions", "term": "credential"}],
            },
        ],
        "edges": [],
    }
    simulation = {
        "presence_dynamics": {
            "resource_consumption": {
                "queue_ratio": 0.22,
                "cpu_sentinel_burn_active": False,
                "control_budget": {
                    "mode": "moderate",
                    "ratio": 0.25,
                },
            },
            "compute_jobs_180s": 6,
        }
    }

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 500, "limit": 8, "llm_enabled": True},
        simulation=simulation,
    )

    result = payload.get("result", {})
    scoring = result.get("scoring", {})
    budget = scoring.get("compute_budget", {})
    assert scoring.get("llm_requested") is True
    assert scoring.get("llm_allowed") is True
    assert captured["max_items"] == 2
    assert bool(budget.get("bound", False)) is True
    assert str(budget.get("mode", "")) == "moderate"
    assert int(budget.get("llm_item_cap", 0) or 0) == 2


def test_run_named_graph_query_github_threat_radar_applies_proximity_state_boost(
    monkeypatch: Any,
) -> None:
    graph = {
        "nodes": [
            {
                "id": "res:gh-prox",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 314,
                "canonical_url": "https://github.com/openai/codex/issues/314",
                "title": "token leak in auth flow",
                "summary": "security triage",
                "state": "open",
                "fetched_ts": 420.0,
                "importance_score": 2,
                "atoms": [
                    {"kind": "mentions", "term": "token"},
                    {"kind": "has_label", "label": "security"},
                ],
            }
        ],
        "edges": [],
    }

    def _fake_proximity(
        _nodes: list[dict[str, Any]],
        _args: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "window_ticks": 400,
            "state_bins": 6,
            "repo": "",
            "seed_set": ["token", "security"],
            "count": 1,
            "state_counts": {
                "background": 0,
                "emerging": 0,
                "active": 0,
                "critical": 1,
            },
            "active_or_higher_count": 1,
            "terms": [
                {
                    "term": "token",
                    "state": "critical",
                    "p_active": 0.81,
                    "p_critical": 0.76,
                    "score": 0.92,
                    "promotion_gate_passed": True,
                }
            ],
            "proximity_hits": [],
        }

    monkeypatch.setattr(graph_queries_module, "_query_proximity_radar", _fake_proximity)

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    result = payload.get("result", {})
    scoring = result.get("scoring", {})
    assert scoring.get("proximity_state_influence") is True
    first = (result.get("threats", []) or [{}])[0]
    assert int(first.get("proximity_boost", 0) or 0) >= 1
    assert float(first.get("proximity_p_critical_max", 0.0) or 0.0) >= 0.7
    assert any(
        str(token).strip().lower() == "proximity_critical_state"
        for token in (first.get("signals", []) or [])
    )


def test_run_named_graph_query_threat_radar_alias_aggregates_domains() -> None:
    graph = {
        "nodes": [
            {
                "id": "res:gh",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "octocat/hello-world",
                "number": 11,
                "canonical_url": "https://github.com/octocat/hello-world/issues/11",
                "title": "security token regression",
                "state": "open",
                "fetched_ts": 300.0,
                "atoms": [{"kind": "has_label", "label": "security"}],
            },
            {
                "id": "res:hormuz",
                "role": "web:resource",
                "kind": "maritime:ukmto_advisory",
                "canonical_url": "https://www.ukmto.org/advisory/003-26",
                "title": "UKMTO advisory",
                "fetched_ts": 301.0,
                "atoms": [
                    {"kind": "hazard", "region": "hormuz", "label": "military_activity"}
                ],
            },
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    assert payload.get("query") == "multi_threat_radar"
    result = payload.get("result", {})
    domains = result.get("domains", {}) if isinstance(result, dict) else {}
    github_domain = domains.get("github", {}) if isinstance(domains, dict) else {}
    hormuz_domain = domains.get("hormuz", {}) if isinstance(domains, dict) else {}
    assert github_domain.get("count") == 1
    assert hormuz_domain.get("count") == 1
    assert len(result.get("threats", [])) == 2


def test_run_named_graph_query_proximity_radar_scores_emerging_terms() -> None:
    graph = {
        "nodes": [
            {
                "id": "res:gh-1",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 10,
                "canonical_url": "https://github.com/openai/codex/issues/10",
                "title": "token leak suspected in auth parser path",
                "summary": "security reviewers mention CVE-2026-2222",
                "fetched_ts": 200.0,
                "atoms": [
                    {"kind": "mentions", "term": "token"},
                    {"kind": "mentions", "term": "auth"},
                    {"kind": "references_cve", "cve_id": "CVE-2026-2222"},
                ],
            },
            {
                "id": "res:gh-2",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 11,
                "canonical_url": "https://github.com/openai/codex/issues/11",
                "title": "oauth token bug in auth middleware",
                "summary": "open security issue",
                "fetched_ts": 190.0,
                "atoms": [
                    {"kind": "mentions", "term": "token"},
                    {"kind": "mentions", "term": "oauth"},
                ],
            },
            {
                "id": "res:gh-3",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 3,
                "canonical_url": "https://github.com/openai/codex/issues/3",
                "title": "legacy parser discussion",
                "summary": "general parser notes",
                "fetched_ts": 120.0,
                "atoms": [{"kind": "mentions", "term": "parser"}],
            },
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "proximity_radar",
        args={
            "window_ticks": 120,
            "limit": 6,
            "seed_set": "token security auth cve",
        },
    )

    result = payload.get("result", {})
    assert result.get("count", 0) >= 1
    assert int(result.get("state_bins", 0)) == 6
    assert int(result.get("promotion_gate_pass_count", 0) or 0) >= 1
    state_counts = result.get("state_counts", {})
    assert isinstance(state_counts, dict)
    assert "background" in state_counts
    assert "critical" in state_counts
    terms = result.get("terms", [])
    token_row = next(
        row for row in terms if str(row.get("term", "")).strip().lower() == "token"
    )
    assert token_row.get("burst_score", 0.0) > 0.0
    assert token_row.get("source_diversity", 0) >= 2
    assert token_row.get("promotion_gate_passed") is True
    assert int(token_row.get("agreement_count", 0) or 0) >= 2
    axes = token_row.get("agreement_axes", [])
    assert isinstance(axes, list)
    assert len(axes) >= 2
    assert str(token_row.get("state", "")) in {
        "background",
        "emerging",
        "active",
        "critical",
    }
    assert 0.0 <= float(token_row.get("score", 0.0) or 0.0) <= 1.0
    assert 0.0 <= float(token_row.get("score_raw", 0.0) or 0.0) <= 1.0
    assert 0.0 <= float(token_row.get("p_active", 0.0) or 0.0) <= 1.0
    assert 0.0 <= float(token_row.get("p_critical", 0.0) or 0.0) <= 1.0
    assert int(token_row.get("last_transition_bin", -1) or -1) >= 0
    assert float(token_row.get("last_transition_ts", 0.0) or 0.0) >= 0.0
    assert isinstance(token_row.get("state_path", []), list)
    assert str(token_row.get("embed_top1", "")).strip().lower() in {
        "token",
        "security",
        "auth",
        "cve",
    }

    hits = result.get("proximity_hits", [])
    token_hit_kinds = {
        str(row.get("kind", "")).strip().lower()
        for row in hits
        if str(row.get("term_id", "")).strip().lower() == "token"
    }
    assert token_hit_kinds == {"embed", "graph", "burst"}
    assert all(
        str(row.get("atom", "")).strip().lower() == "proximity_hit"
        for row in hits
        if str(row.get("term_id", "")).strip().lower() == "token"
    )


def test_run_named_graph_query_entity_risk_state_returns_compact_state_payload() -> (
    None
):
    graph = {
        "nodes": [
            {
                "id": "res:gh-a",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 77,
                "canonical_url": "https://github.com/openai/codex/issues/77",
                "title": "token leak and auth bypass",
                "summary": "security incident triage",
                "fetched_ts": 205.0,
                "atoms": [
                    {"kind": "mentions", "term": "token"},
                    {"kind": "mentions", "term": "auth"},
                ],
            }
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "entity_risk_state",
        args={"window_ticks": 120, "limit": 4},
    )

    assert payload.get("query") == "entity_risk_state"
    result = payload.get("result", {})
    assert result.get("count", 0) >= 1
    assert int(result.get("promotion_gate_pass_count", 0) or 0) >= 0
    entities = result.get("entities", [])
    assert entities
    first = entities[0]
    assert str(first.get("state", "")).strip() in {
        "background",
        "emerging",
        "active",
        "critical",
    }
    assert 0.0 <= float(first.get("p_active", 0.0) or 0.0) <= 1.0
    assert 0.0 <= float(first.get("p_critical", 0.0) or 0.0) <= 1.0
    assert isinstance(first.get("promotion_gate_passed", False), bool)
    assert int(first.get("agreement_count", 0) or 0) >= 0


def test_run_named_graph_query_cyber_regime_state_reports_posterior_and_policy() -> (
    None
):
    graph = {
        "nodes": [
            {
                "id": "res:gh-advisory",
                "role": "web:resource",
                "kind": "github:advisory",
                "repo": "openai/codex",
                "number": 501,
                "canonical_url": "https://github.com/openai/codex/security/advisories/GHSA-aaaa-bbbb-cccc",
                "title": "GHSA advisory for token leak",
                "summary": "CVE-2026-4111 auth bypass",
                "fetched_ts": 880.0,
                "state": "open",
                "atoms": [
                    {"kind": "references_cve", "cve_id": "CVE-2026-4111"},
                    {"kind": "mentions", "term": "token"},
                ],
            },
            {
                "id": "res:gh-issue",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 502,
                "canonical_url": "https://github.com/openai/codex/issues/502",
                "title": "security triage thread",
                "summary": "investigating auth token handling",
                "fetched_ts": 860.0,
                "state": "open",
                "atoms": [
                    {"kind": "has_label", "label": "security"},
                    {"kind": "mentions", "term": "auth"},
                ],
            },
        ],
        "edges": [],
    }

    payload = run_named_graph_query(
        graph,
        "cyber_regime_state",
        args={"window_ticks": 120, "state_bins": 6, "threat_limit": 64},
    )

    assert payload.get("query") == "cyber_regime_state"
    result = payload.get("result", {})
    assert str(result.get("state", "")).strip() in {
        "baseline",
        "elevated_chatter",
        "active_exploitation_wave",
        "supply_chain_campaign",
        "geopolitical_targeting_shift",
    }
    posterior = result.get("posterior", {})
    assert isinstance(posterior, dict)
    assert "baseline" in posterior
    policy = result.get("policy", {})
    assert isinstance(policy, dict)
    assert 4 <= int(policy.get("risk_score_threshold", 0) or 0) <= 10
    assert float(policy.get("crawl_budget_multiplier", 0.0) or 0.0) >= 1.0
    assert float(policy.get("query_expansion_multiplier", 0.0) or 0.0) >= 1.0
    bins = result.get("bins", [])
    assert isinstance(bins, list)
    assert len(bins) == 6


def test_run_named_graph_query_regime_radar_alias_applies_threshold(
    monkeypatch: Any,
) -> None:
    def _fake_regime(
        _nodes: list[dict[str, Any]],
        _args: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "state": "active_exploitation_wave",
            "posterior": {
                "baseline": 0.11,
                "elevated_chatter": 0.14,
                "active_exploitation_wave": 0.63,
                "supply_chain_campaign": 0.08,
                "geopolitical_targeting_shift": 0.04,
            },
            "policy": {
                "risk_score_threshold": 9,
                "crawl_budget_multiplier": 1.8,
                "query_expansion_multiplier": 1.8,
                "pressure": 0.71,
            },
            "observation_mean": 0.54,
            "observation_peak": 0.82,
        }

    def _fake_github_radar(
        _nodes: list[dict[str, Any]],
        _args: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "repo": "",
            "window_ticks": 300,
            "scoring": {"mode": "classifier"},
            "threats": [
                {
                    "repo": "openai/codex",
                    "kind": "github:advisory",
                    "title": "critical advisory",
                    "canonical_url": "https://github.com/openai/codex/security/advisories/GHSA-xy",
                    "risk_score": 11,
                    "fetched_ts": 990.0,
                },
                {
                    "repo": "openai/codex",
                    "kind": "github:issue",
                    "title": "low confidence thread",
                    "canonical_url": "https://github.com/openai/codex/issues/77",
                    "risk_score": 7,
                    "fetched_ts": 995.0,
                },
            ],
        }

    monkeypatch.setattr(graph_queries_module, "_query_cyber_regime_state", _fake_regime)
    monkeypatch.setattr(
        graph_queries_module, "_query_github_threat_radar", _fake_github_radar
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "regime_radar",
        args={"limit": 8},
    )

    assert payload.get("query") == "cyber_risk_radar"
    result = payload.get("result", {})
    assert result.get("apply_regime_threshold") is True
    assert int(result.get("count", 0) or 0) == 1
    first = (result.get("threats", []) or [{}])[0]
    assert first.get("regime_state") == "active_exploitation_wave"
    assert first.get("passes_regime_threshold") is True
    assert int(first.get("regime_risk_score_threshold", 0) or 0) == 9
    scoring = result.get("scoring", {})
    assert scoring.get("mode") == "cyber_regime_context"


def test_run_named_graph_query_cyber_risk_radar_applies_threshold_fallback_when_empty(
    monkeypatch: Any,
) -> None:
    def _fake_regime(
        _nodes: list[dict[str, Any]],
        _args: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "state": "baseline",
            "posterior": {
                "baseline": 0.84,
                "elevated_chatter": 0.09,
                "active_exploitation_wave": 0.03,
                "supply_chain_campaign": 0.02,
                "geopolitical_targeting_shift": 0.02,
            },
            "policy": {
                "risk_score_threshold": 8,
                "crawl_budget_multiplier": 1.0,
                "query_expansion_multiplier": 1.0,
                "pressure": 0.08,
            },
            "observation_mean": 0.11,
            "observation_peak": 0.21,
        }

    def _fake_github_radar(
        _nodes: list[dict[str, Any]],
        _args: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "repo": "",
            "window_ticks": 300,
            "scoring": {"mode": "classifier"},
            "threats": [
                {
                    "repo": "openai/codex",
                    "kind": "github:issue",
                    "title": "security auth workflow discussion",
                    "canonical_url": "https://github.com/openai/codex/issues/602",
                    "risk_score": 5,
                    "fetched_ts": 1200.0,
                    "weak_label": "security_possible",
                    "weak_label_score": 1,
                },
                {
                    "repo": "openai/codex",
                    "kind": "github:issue",
                    "title": "docs formatting cleanup",
                    "canonical_url": "https://github.com/openai/codex/issues/603",
                    "risk_score": 7,
                    "fetched_ts": 1198.0,
                    "weak_label": "low_security_relevance",
                    "weak_label_score": -2,
                },
            ],
        }

    monkeypatch.setattr(graph_queries_module, "_query_cyber_regime_state", _fake_regime)
    monkeypatch.setattr(
        graph_queries_module,
        "_query_github_threat_radar",
        _fake_github_radar,
    )

    payload = run_named_graph_query(
        {"nodes": [], "edges": []},
        "cyber_risk_radar",
        args={"limit": 8},
    )

    result = payload.get("result", {})
    assert int(result.get("count", 0) or 0) == 1
    first = (result.get("threats", []) or [{}])[0]
    assert int(first.get("risk_score", 0) or 0) == 5
    assert bool(first.get("passes_security_gate", False)) is True
    assert int(first.get("regime_risk_score_threshold", 0) or 0) == 5
    scoring = result.get("scoring", {})
    assert bool(scoring.get("risk_score_threshold_fallback_applied", False)) is True
    assert int(scoring.get("risk_score_threshold_base", 0) or 0) == 8
    assert int(scoring.get("risk_score_threshold", 0) or 0) == 5
    assert int(scoring.get("security_gate_filtered_count", 0) or 0) == 1


def test_run_named_graph_query_github_threat_radar_ignores_proximity_when_gate_fails(
    monkeypatch: Any,
) -> None:
    graph = {
        "nodes": [
            {
                "id": "res:gh-prox-fail",
                "role": "web:resource",
                "kind": "github:issue",
                "repo": "openai/codex",
                "number": 315,
                "canonical_url": "https://github.com/openai/codex/issues/315",
                "title": "token leak report",
                "summary": "security triage",
                "state": "open",
                "fetched_ts": 430.0,
                "importance_score": 2,
                "atoms": [{"kind": "mentions", "term": "token"}],
            }
        ],
        "edges": [],
    }

    def _fake_proximity(
        _nodes: list[dict[str, Any]],
        _args: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "window_ticks": 400,
            "state_bins": 6,
            "repo": "",
            "seed_set": ["token", "security"],
            "count": 1,
            "state_counts": {
                "background": 0,
                "emerging": 0,
                "active": 0,
                "critical": 1,
            },
            "active_or_higher_count": 1,
            "terms": [
                {
                    "term": "token",
                    "state": "critical",
                    "p_active": 0.91,
                    "p_critical": 0.89,
                    "score": 0.95,
                    "promotion_gate_passed": False,
                }
            ],
            "proximity_hits": [],
        }

    monkeypatch.setattr(graph_queries_module, "_query_proximity_radar", _fake_proximity)

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    first = (payload.get("result", {}).get("threats", []) or [{}])[0]
    assert int(first.get("proximity_boost", 0) or 0) == 0
    assert float(first.get("proximity_p_critical_max", 0.0) or 0.0) == 0.0
    assert all(
        str(token).strip().lower() != "proximity_critical_state"
        for token in (first.get("signals", []) or [])
    )


def test_run_named_graph_query_github_threat_radar_attaches_llm_metrics(
    monkeypatch: Any,
) -> None:
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
                    }
                ],
            }
        ],
        "edges": [],
    }

    def _fake_llm(
        *,
        domain: str,
        rows: list[dict[str, object]],
        max_items: int | None = None,
    ) -> dict[str, object]:
        assert domain == "github"
        assert max_items is None or max_items >= 1
        threat_id = str(rows[0].get("_threat_id", ""))
        return {
            "applied": True,
            "enabled": True,
            "model": "qwen3-vl:2b-instruct",
            "error": "",
            "metrics": {
                threat_id: {
                    "overall_score": 92,
                    "confidence": 81,
                    "severity": 88,
                    "immediacy": 74,
                    "impact": 84,
                    "exploitability": 77,
                    "credibility": 83,
                    "exposure": 65,
                    "novelty": 43,
                    "operational_risk": 79,
                    "rationale": "high confidence security issue",
                }
            },
            "cache": "miss",
        }

    monkeypatch.setattr(graph_queries_module, "_threat_llm_metrics", _fake_llm)

    payload = run_named_graph_query(
        graph,
        "github_threat_radar",
        args={"window_ticks": 400, "limit": 8},
    )

    result = payload.get("result", {})
    scoring = result.get("scoring", {})
    assert scoring.get("mode") == "llm_blend"
    assert scoring.get("llm_applied") is True
    first = (result.get("threats", []) or [{}])[0]
    assert first.get("llm_model") == "qwen3-vl:2b-instruct"
    assert first.get("llm_score", 0) > 0
    assert isinstance(first.get("threat_metrics", {}), dict)


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
