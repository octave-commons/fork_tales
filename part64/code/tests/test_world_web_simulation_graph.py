from __future__ import annotations

import json
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any

import pytest
import code.world_web as world_web_module
import code.world_web.simulation as simulation_module

from code.world_web import build_simulation_state, collect_catalog


@pytest.fixture(autouse=True)
def _reset_simulation_runtime_state() -> None:
    simulation_module.reset_simulation_bootstrap_state()


def _create_fixture_tree(root: Path) -> None:
    manifest = {
        "part": 64,
        "seed_label": "eta_mu_part_64",
        "files": [
            {"path": "artifacts/audio/test.wav", "role": "audio/canonical"},
            {"path": "world_state/constraints.md", "role": "world_state"},
        ],
    }
    (root / "artifacts" / "audio").mkdir(parents=True)
    (root / "world_state").mkdir(parents=True)
    sample = array("h", [0, 1200, -1200, 300, -300, 0])
    with wave.open(str(root / "artifacts" / "audio" / "test.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(sample.tobytes())
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "world_state" / "constraints.md").write_text(
        "# Constraints\n\n- C-64-world-snapshot: active\n",
        encoding="utf-8",
    )


def test_simulation_state_includes_file_graph_nodes() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 2,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 2,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:abc",
                    "name": "new_witness_note.md",
                    "kind": "text",
                    "x": 0.22,
                    "y": 0.31,
                    "hue": 212,
                    "importance": 0.7,
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 7,
                "file_count": 1,
                "edge_count": 1,
                "kind_counts": {"text": 1},
                "field_counts": {"f6": 1},
                "knowledge_entries": 1,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    assert simulation.get("file_graph", {}).get("record") == "ημ.file-graph.v1"
    assert simulation.get("total", 0) >= 1
    assert len(simulation.get("points", [])) == simulation.get("total", 0)


def test_simulation_state_applies_document_similarity_layout_to_points() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 2,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 2,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:left",
                    "name": "alpha_notes.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "x": 0.5,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
                {
                    "id": "file:right",
                    "name": "alpha_archive.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "x": 0.54,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
            ],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 2},
                "field_counts": {"f6": 2},
                "knowledge_entries": 2,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    sim_graph = simulation.get("file_graph", {})
    sim_nodes = sim_graph.get("file_nodes", [])
    assert isinstance(sim_nodes, list)
    assert len(sim_nodes) == 2

    left = sim_nodes[0]
    right = sim_nodes[1]
    assert float(left.get("x", 0.0)) > 0.5
    assert float(right.get("x", 1.0)) < 0.54

    points = simulation.get("points", [])
    assert len(points) >= 2
    left_point = points[0]
    right_point = points[1]
    expected_left_x = round((float(left.get("x", 0.5)) * 2.0) - 1.0, 5)
    expected_right_x = round((float(right.get("x", 0.5)) * 2.0) - 1.0, 5)
    assert abs(float(left_point.get("x", 0.0)) - expected_left_x) <= 1e-5
    assert abs(float(right_point.get("x", 0.0)) - expected_right_x) <= 1e-5


def test_simulation_state_document_similarity_layout_is_subtle() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 2,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 2,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:left",
                    "name": "alpha_notes.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "x": 0.4,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
                {
                    "id": "file:right",
                    "name": "gamma_image.png",
                    "kind": "image",
                    "summary": "solar flare telemetry",
                    "tags": ["solar", "flare"],
                    "dominant_field": "f1",
                    "x": 0.44,
                    "y": 0.5,
                    "hue": 320,
                    "importance": 1.0,
                },
            ],
            "edges": [],
            "stats": {
                "field_count": 2,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 1, "image": 1},
                "field_counts": {"f1": 1, "f6": 1},
                "knowledge_entries": 2,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    sim_nodes = simulation.get("file_graph", {}).get("file_nodes", [])
    assert isinstance(sim_nodes, list)
    assert len(sim_nodes) == 2

    left_x = float(sim_nodes[0].get("x", 0.4))
    right_x = float(sim_nodes[1].get("x", 0.44))
    assert left_x < 0.4
    assert right_x > 0.44

    assert abs(left_x - 0.4) < 0.03
    assert abs(right_x - 0.44) < 0.03


def test_simulation_state_embedded_nodes_repel_non_embedded_nodes() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 2,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 2,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:embedded",
                    "name": "alpha_embed.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "vecstore_collection": "eta_mu_nexus_v1",
                    "embed_layer_count": 1,
                    "x": 0.5,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
                {
                    "id": "file:plain",
                    "name": "alpha_plain.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "x": 0.52,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
            ],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 2},
                "field_counts": {"f6": 2},
                "knowledge_entries": 2,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    sim_nodes = simulation.get("file_graph", {}).get("file_nodes", [])
    assert isinstance(sim_nodes, list)
    assert len(sim_nodes) == 2

    embedded_x = float(sim_nodes[0].get("x", 0.5))
    plain_x = float(sim_nodes[1].get("x", 0.52))
    assert embedded_x < 0.5
    assert plain_x > 0.52
    assert abs(embedded_x - 0.5) < 0.03
    assert abs(plain_x - 0.52) < 0.03


def test_simulation_state_emits_embedding_particles_for_embedded_files(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module.time, "time", lambda: 1_700_001_234.0)
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 2,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 2,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:embed-left",
                    "name": "left.md",
                    "kind": "text",
                    "summary": "alpha witness archive stream",
                    "tags": ["alpha", "witness", "stream"],
                    "dominant_field": "f6",
                    "vecstore_collection": "eta_mu_nexus_v1",
                    "embed_layer_count": 1,
                    "x": 0.45,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 0.9,
                },
                {
                    "id": "file:embed-right",
                    "name": "right.md",
                    "kind": "text",
                    "summary": "alpha witness archive stream",
                    "tags": ["alpha", "witness", "archive"],
                    "dominant_field": "f6",
                    "vecstore_collection": "eta_mu_nexus_v1",
                    "embed_layer_count": 1,
                    "x": 0.55,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 0.9,
                },
            ],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 2},
                "field_counts": {"f6": 2},
                "knowledge_entries": 2,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    embedding_particles = simulation.get("embedding_particles", [])
    assert isinstance(embedding_particles, list)
    assert len(embedding_particles) >= 6
    assert any(float(row.get("size", 0.0)) > 2.0 for row in embedding_particles)

    for row in embedding_particles[:3]:
        assert -1.0 <= float(row.get("x", 0.0)) <= 1.0
        assert -1.0 <= float(row.get("y", 0.0)) <= 1.0

    graph_particles = simulation.get("file_graph", {}).get("embedding_particles", [])
    assert isinstance(graph_particles, list)
    assert len(graph_particles) == len(embedding_particles)


def test_embedding_particles_bias_toward_denser_nearby_documents(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module.time, "time", lambda: 1_700_009_999.0)

    dense_text = "alpha witness archive stream " * 80
    sparse_text = "alpha witness"

    def _catalog_for_density(left_summary: str, right_summary: str) -> dict[str, Any]:
        return {
            "items": [],
            "counts": {},
            "file_graph": {
                "record": "ημ.file-graph.v1",
                "generated_at": "2026-02-16T00:00:00+00:00",
                "inbox": {
                    "record": "ημ.inbox.v1",
                    "path": "/tmp/.ημ",
                    "pending_count": 0,
                    "processed_count": 2,
                    "failed_count": 0,
                    "is_empty": True,
                    "knowledge_entries": 2,
                    "last_ingested_at": "2026-02-16T00:00:00+00:00",
                    "errors": [],
                },
                "nodes": [],
                "field_nodes": [],
                "file_nodes": [
                    {
                        "id": "file:left",
                        "name": "alpha_note.md",
                        "kind": "text",
                        "summary": left_summary,
                        "text_excerpt": left_summary,
                        "tags": ["alpha", "witness", "archive"],
                        "dominant_field": "f6",
                        "vecstore_collection": "eta_mu_nexus_v1",
                        "embed_layer_count": 1,
                        "x": 0.44,
                        "y": 0.5,
                        "hue": 210,
                        "importance": 0.9,
                    },
                    {
                        "id": "file:right",
                        "name": "alpha_note.md",
                        "kind": "text",
                        "summary": right_summary,
                        "text_excerpt": right_summary,
                        "tags": ["alpha", "witness", "archive"],
                        "dominant_field": "f6",
                        "vecstore_collection": "eta_mu_nexus_v1",
                        "embed_layer_count": 1,
                        "x": 0.56,
                        "y": 0.5,
                        "hue": 210,
                        "importance": 0.9,
                    },
                ],
                "edges": [],
                "stats": {
                    "field_count": 1,
                    "file_count": 2,
                    "edge_count": 0,
                    "kind_counts": {"text": 2},
                    "field_counts": {"f6": 2},
                    "knowledge_entries": 2,
                },
            },
        }

    left_dense = build_simulation_state(_catalog_for_density(dense_text, sparse_text))
    right_dense = build_simulation_state(_catalog_for_density(sparse_text, dense_text))

    left_particles = left_dense.get("embedding_particles", [])
    right_particles = right_dense.get("embedding_particles", [])
    assert isinstance(left_particles, list)
    assert isinstance(right_particles, list)
    assert left_particles
    assert right_particles

    left_mean_x = sum(
        (float(row.get("x", 0.0)) + 1.0) * 0.5 for row in left_particles
    ) / len(left_particles)
    right_mean_x = sum(
        (float(row.get("x", 0.0)) + 1.0) * 0.5 for row in right_particles
    ) / len(right_particles)

    assert left_mean_x < right_mean_x
    assert abs(right_mean_x - left_mean_x) > 0.0004


def test_logical_graph_includes_world_log_event_nodes() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 0,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 1,
                "last_ingested_at": "",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:abc",
                    "node_id": "knowledge:abc",
                    "node_type": "file",
                    "label": "pending_note.md",
                    "source_rel_path": ".ημ/pending_note.md",
                    "x": 0.24,
                    "y": 0.31,
                    "hue": 210,
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 0,
                "file_count": 1,
                "edge_count": 0,
                "kind_counts": {},
                "field_counts": {},
                "knowledge_entries": 1,
            },
        },
        "world_log": {
            "ok": True,
            "record": "ημ.world-log.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "count": 2,
            "limit": 180,
            "pending_inbox": 1,
            "sources": {"eta_mu_inbox": 1, "receipt": 1},
            "kinds": {"eta_mu.pending": 1, ":decision": 1},
            "relation_count": 1,
            "events": [
                {
                    "id": "evt_a",
                    "source": "eta_mu_inbox",
                    "kind": "eta_mu.pending",
                    "status": "pending",
                    "title": "pending inbox file",
                    "detail": "awaiting ingest",
                    "refs": [".ημ/pending_note.md"],
                    "x": 0.4,
                    "y": 0.5,
                    "relations": [{"event_id": "evt_b", "score": 0.8}],
                },
                {
                    "id": "evt_b",
                    "source": "receipt",
                    "kind": ":decision",
                    "status": "recorded",
                    "title": "decision",
                    "detail": "recorded",
                    "refs": ["receipts.log"],
                    "x": 0.6,
                    "y": 0.55,
                    "relations": [{"event_id": "evt_a", "score": 0.8}],
                },
            ],
        },
    }

    simulation = build_simulation_state(catalog)
    logical_graph = simulation.get("logical_graph", {})
    nodes = logical_graph.get("nodes", [])
    edges = logical_graph.get("edges", [])

    assert any(
        isinstance(node, dict) and str(node.get("kind", "")) == "event"
        for node in nodes
    )
    assert any(
        isinstance(edge, dict) and str(edge.get("kind", "")) == "mentions"
        for edge in edges
    )
    assert any(
        isinstance(edge, dict) and str(edge.get("kind", "")) == "correlates"
        for edge in edges
    )


def test_catalog_includes_crawler_graph_nodes_and_edges(monkeypatch: Any) -> None:
    from code.world_web import simulation as simulation_module

    def fake_fetch(_part_root: Path) -> dict[str, Any]:
        return {
            "ok": True,
            "source": "http://127.0.0.1:8793/api/weaver/graph",
            "status": {"alive": 1, "queue_size": 2},
            "graph": {
                "nodes": [
                    {
                        "id": "url:https://example.org/guide",
                        "kind": "url",
                        "url": "https://example.org/guide",
                        "domain": "example.org",
                        "title": "Guide",
                        "status": "fetched",
                        "depth": 1,
                        "cooldown_until": 1_900_000_000_000,
                        "content_hash": "abc123",
                        "fetched_at": 1_800_000_000_000,
                        "compliance": "allowed",
                    },
                    {
                        "id": "url:https://example.org/song",
                        "kind": "url",
                        "url": "https://example.org/song",
                        "domain": "example.org",
                        "title": "Song",
                        "status": "discovered",
                        "depth": 2,
                        "compliance": "pending",
                    },
                    {
                        "id": "domain:example.org",
                        "kind": "domain",
                        "domain": "example.org",
                    },
                    {
                        "id": "content:https://example.org/audio.mp3",
                        "kind": "content",
                        "url": "https://example.org/audio.mp3",
                        "domain": "example.org",
                        "title": "Field Song",
                        "content_type": "audio/mpeg",
                        "status": "fetched",
                        "depth": 2,
                        "compliance": "allowed",
                    },
                ],
                "edges": [
                    {
                        "id": "edge:1",
                        "source": "url:https://example.org/guide",
                        "target": "domain:example.org",
                        "kind": "domain_membership",
                    },
                    {
                        "id": "edge:2",
                        "source": "url:https://example.org/guide",
                        "target": "url:https://example.org/song",
                        "kind": "hyperlink",
                    },
                ],
                "counts": {"nodes_total": 4, "edges_total": 2, "url_nodes_total": 2},
            },
        }

    monkeypatch.setattr(world_web_module, "_fetch_weaver_graph_payload", fake_fetch)
    monkeypatch.setattr(simulation_module, "_fetch_weaver_graph_payload", fake_fetch)

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        catalog = collect_catalog(
            part,
            vault,
            sync_inbox=False,
            include_world_log=False,
            include_pi_archive=False,
        )
        crawler_graph = catalog.get("crawler_graph", {})
        assert crawler_graph.get("record") == "ημ.crawler-graph.v1"
        assert crawler_graph.get("stats", {}).get("crawler_count", 0) >= 2
        assert crawler_graph.get("stats", {}).get("edge_count", 0) >= 2
        assert crawler_graph.get("status", {}).get("alive") == 1
        assert any(
            str(node.get("url", "")).startswith("https://")
            for node in crawler_graph.get("crawler_nodes", [])
        )
        audio_node = next(
            node
            for node in crawler_graph.get("crawler_nodes", [])
            if str(node.get("content_type", "")).startswith("audio/")
        )
        assert audio_node.get("resource_kind") == "audio"
        assert audio_node.get("modality") == "audio"
        assert (
            crawler_graph.get("stats", {})
            .get("resource_kind_counts", {})
            .get("audio", 0)
            >= 1
        )
        web_role_counts = crawler_graph.get("stats", {}).get("web_role_counts", {})
        assert web_role_counts.get("web:url", 0) >= 2
        assert web_role_counts.get("web:resource", 0) >= 1
        web_edges = [
            row for row in crawler_graph.get("edges", []) if isinstance(row, dict)
        ]
        assert any(
            str(row.get("kind", "")).strip().lower() == "web:source_of"
            for row in web_edges
        )
        assert any(
            str(row.get("kind", "")).strip().lower() == "web:links_to"
            for row in web_edges
        )
        url_node = next(
            row
            for row in crawler_graph.get("crawler_nodes", [])
            if str(row.get("web_node_role", "")) == "web:url"
            and str(row.get("canonical_url", "")) == "https://example.org/guide"
        )
        assert float(url_node.get("next_allowed_fetch_ts", 0.0)) > 0.0
        assert str(url_node.get("last_status", "")) == "ok"


def test_simulation_state_includes_crawler_graph_nodes() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "crawler_graph": {
            "record": "ημ.crawler-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "source": {
                "endpoint": "http://127.0.0.1:8793/api/weaver/graph",
                "service": "web-graph-weaver",
            },
            "status": {"alive": 1, "queue_size": 0},
            "nodes": [],
            "field_nodes": [],
            "crawler_nodes": [
                {
                    "id": "crawler:abc",
                    "node_id": "url:https://example.org",
                    "node_type": "crawler",
                    "crawler_kind": "url",
                    "label": "https://example.org",
                    "x": 0.68,
                    "y": 0.32,
                    "hue": 200,
                    "importance": 0.8,
                    "url": "https://example.org",
                    "dominant_field": "f2",
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 7,
                "crawler_count": 1,
                "edge_count": 0,
                "kind_counts": {"url": 1},
                "field_counts": {"f2": 1},
                "nodes_total": 1,
                "edges_total": 0,
                "url_nodes_total": 1,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    assert simulation.get("crawler_graph", {}).get("record") == "ημ.crawler-graph.v1"
    assert simulation.get("total", 0) >= 1
    assert len(simulation.get("points", [])) == simulation.get("total", 0)


def test_catalog_includes_truth_state_snapshot() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        catalog = collect_catalog(
            part,
            vault,
            sync_inbox=False,
            include_world_log=False,
            include_pi_archive=False,
        )
        truth_state = catalog.get("truth_state", {})
        assert truth_state.get("record") == "ημ.truth-state.v1"
        assert truth_state.get("claim", {}).get("id") == "claim.push_truth_gate_ready"
        assert isinstance(truth_state.get("claims"), list)
        assert isinstance(truth_state.get("proof", {}).get("entries"), list)


def test_simulation_state_includes_truth_state() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "truth_state": {
            "record": "ημ.truth-state.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "name_binding": {
                "id": "gates_of_truth",
                "symbol": "Gates_of_Truth",
                "glyph": "真",
                "ascii": "TRUTH",
                "law": "Truth requires world scope (ω) + proof refs + receipts.",
            },
            "world": {
                "id": "127.0.0.1:8787",
                "ctx/ω-world": "127.0.0.1:8787",
                "ctx_omega_world": "127.0.0.1:8787",
            },
            "claim": {
                "id": "claim.push_truth_gate_ready",
                "text": "push-truth gate is ready for apply",
                "status": "proved",
                "kappa": 0.86,
                "world": "127.0.0.1:8787",
                "proof_refs": ["runtime:/api/push-truth/dry-run"],
                "theta": 0.72,
            },
            "claims": [
                {
                    "id": "claim.push_truth_gate_ready",
                    "text": "push-truth gate is ready for apply",
                    "status": "proved",
                    "kappa": 0.86,
                    "world": "127.0.0.1:8787",
                    "proof_refs": ["runtime:/api/push-truth/dry-run"],
                    "theta": 0.72,
                }
            ],
            "guard": {"theta": 0.72, "passes": True},
            "gate": {"target": "push-truth", "blocked": False, "reasons": []},
            "invariants": {
                "world_scoped": True,
                "proof_required": True,
                "proof_kind_subset": True,
                "receipts_parse_ok": True,
                "sim_bead_mint_blocked": True,
                "truth_binding_registered": True,
            },
            "proof": {
                "required_kinds": [":logic/bridge"],
                "entries": [
                    {
                        "kind": ":logic/bridge",
                        "ref": "manifest.lith",
                        "present": True,
                        "detail": "manifest proof-schema source",
                    }
                ],
                "counts": {"total": 1, "present": 1, "by_kind": {":logic/bridge": 1}},
            },
            "artifacts": {
                "pi_zip_count": 1,
                "host_handle": "github:err",
                "host_has_github_gist": True,
                "truth_receipt_count": 1,
                "decision_receipt_count": 1,
            },
            "schema": {
                "source": "manifest.lith",
                "required_refs": ["receipts.log"],
                "required_hashes": ["sha256:manifest"],
                "host_handle": "github:err",
                "missing_refs": [],
                "missing_hashes": [],
            },
            "needs": [],
        },
    }

    simulation = build_simulation_state(catalog)
    assert simulation.get("truth_state", {}).get("record") == "ημ.truth-state.v1"
    assert simulation.get("truth_state", {}).get("claim", {}).get("status") == "proved"
    assert simulation.get("total", 0) >= 1
    assert len(simulation.get("points", [])) == simulation.get("total", 0)


def test_simulation_state_exposes_logical_graph_and_pain_field() -> None:
    source_path = "notes/proof.md"
    file_id = world_web_module._file_id_for_path(source_path)
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 1,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 1,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:proof",
                    "name": "proof.md",
                    "label": "proof.md",
                    "kind": "text",
                    "x": 0.31,
                    "y": 0.42,
                    "hue": 212,
                    "importance": 0.7,
                    "source_rel_path": source_path,
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 7,
                "file_count": 1,
                "edge_count": 0,
                "kind_counts": {"text": 1},
                "field_counts": {"f6": 1},
                "knowledge_entries": 1,
            },
        },
        "truth_state": {
            "record": "ημ.truth-state.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "claim": {
                "id": "claim.push_truth_gate_ready",
                "text": "push-truth gate is ready for apply",
                "status": "undecided",
                "kappa": 0.5,
                "world": "127.0.0.1:8787",
                "proof_refs": [source_path],
                "theta": 0.72,
            },
            "claims": [
                {
                    "id": "claim.push_truth_gate_ready",
                    "text": "push-truth gate is ready for apply",
                    "status": "undecided",
                    "kappa": 0.5,
                    "world": "127.0.0.1:8787",
                    "proof_refs": [source_path],
                    "theta": 0.72,
                }
            ],
            "proof": {
                "required_kinds": [":logic/bridge"],
                "entries": [
                    {
                        "kind": ":logic/bridge",
                        "ref": source_path,
                        "present": True,
                        "detail": "manifest proof-schema source",
                    }
                ],
            },
            "gate": {
                "target": "push-truth",
                "blocked": True,
                "reasons": ["missing-receipt"],
            },
        },
        "test_failures": [
            {
                "name": "test_push_truth_gate",
                "status": "failed",
                "message": "missing receipt",
                "covered_files": [source_path],
                "severity": 0.9,
            }
        ],
    }

    simulation = build_simulation_state(catalog)
    logical_graph = simulation.get("logical_graph", {})
    assert logical_graph.get("record") == "ημ.logical-graph.v1"
    assert (
        logical_graph.get("joins", {}).get("file_index", {}).get(source_path) == file_id
    )
    assert logical_graph.get("stats", {}).get("fact_nodes", 0) >= 1

    pain_field = simulation.get("pain_field", {})
    assert pain_field.get("record") == "ημ.pain-field.v1"
    assert pain_field.get("active") is True
    failing = pain_field.get("failing_tests", [])
    assert isinstance(failing, list)
    assert failing and file_id in failing[0].get("file_ids", [])
    assert any(row.get("heat", 0) > 0.5 for row in pain_field.get("node_heat", []))
    debug_target = pain_field.get("debug", {})
    assert debug_target.get("meaning") == "DEBUG"
    assert debug_target.get("grounded") is True
    assert debug_target.get("file_id") == file_id
    assert debug_target.get("reason") == "points-to-hottest-file"


def test_pain_field_ingests_test_covers_span_relations() -> None:
    source_path = "notes/proof.md"
    file_id = world_web_module._file_id_for_path(source_path)
    logical_graph = {
        "nodes": [
            {
                "id": "logical:file:proof",
                "kind": "file",
                "file_id": file_id,
                "path": source_path,
                "label": "proof.md",
                "x": 0.31,
                "y": 0.42,
            }
        ],
        "edges": [],
        "joins": {
            "file_index": {source_path: file_id},
        },
    }
    catalog = {
        "test_failures": [
            {
                "name": "test_push_truth_gate",
                "status": "failed",
                "message": "missing receipt",
                "severity": 1.0,
            }
        ],
        "test_coverage": {
            "by_test_spans": {
                "test_push_truth_gate": [
                    {
                        "file": source_path,
                        "start_line": 12,
                        "end_line": 20,
                        "weight": 0.75,
                    },
                    {
                        "file": source_path,
                        "start_line": 28,
                        "end_line": 34,
                        "weight": 0.25,
                    },
                ]
            }
        },
    }

    pain_field = world_web_module._build_pain_field(catalog, logical_graph)
    relations = pain_field.get("relations", {})
    test_covers = relations.get("覆/test-covers-span", [])
    span_maps = relations.get("覆/span-maps-to-region", [])

    assert len(test_covers) == 2
    assert len(span_maps) == 2
    assert all(float(row.get("w", 0.0)) > 0.0 for row in test_covers)

    failing = pain_field.get("failing_tests", [])
    assert len(failing) == 1
    assert len(failing[0].get("span_ids", [])) == 2
    assert len(failing[0].get("region_ids", [])) == 1

    spans = pain_field.get("spans", [])
    assert len(spans) == 2
    assert {int(row.get("start_line", 0)) for row in spans} == {12, 28}

    heat_regions = pain_field.get("heat_regions", [])
    assert heat_regions and float(heat_regions[0].get("heat", 0.0)) > 0.0
    assert any(
        row.get("node_id") == "logical:file:proof"
        for row in pain_field.get("node_heat", [])
    )

    heat_values = world_web_module._materialize_heat_values(
        {
            "logical_graph": logical_graph,
            "pain_field": pain_field,
        },
        pain_field,
    )
    assert heat_values.get("record") == "ημ.heat-values.v1"
    assert heat_values.get("active") is True
    facts = heat_values.get("facts", [])
    assert any(str(row.get("kind", "")) == "熱/value" for row in facts)
    assert any(float(row.get("value", 0.0)) > 0.0 for row in facts)
    locate_rows = heat_values.get("locate", [])
    assert isinstance(locate_rows, list)
    assert any(str(row.get("kind", "")) == "址" for row in locate_rows)


def test_load_test_signal_artifacts_prefers_lcov_and_failing_test_list() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part_root = root / "part"
        vault_root = root / "vault"
        (part_root / "world_state").mkdir(parents=True)
        (part_root / "coverage").mkdir(parents=True)
        (vault_root / ".opencode" / "runtime").mkdir(parents=True)

        (part_root / "world_state" / "failing_tests.txt").write_text(
            "test_receipt_gate\ntest_truth_flow | code/world_web.py code/lore.py\n",
            encoding="utf-8",
        )
        (part_root / "coverage" / "lcov.info").write_text(
            "TN:test_receipt_gate\n"
            "SF:code/world_web.py\n"
            "DA:10,0\n"
            "DA:11,0\n"
            "DA:12,1\n"
            "LF:3\n"
            "LH:1\n"
            "end_of_record\n"
            "TN:test_truth_flow\n"
            "SF:code/lore.py\n"
            "DA:1,1\n"
            "DA:2,1\n"
            "LF:2\n"
            "LH:2\n"
            "end_of_record\n",
            encoding="utf-8",
        )

        failures, coverage = world_web_module._load_test_signal_artifacts(
            part_root,
            vault_root,
        )

        assert [row.get("name") for row in failures] == [
            "test_receipt_gate",
            "test_truth_flow",
        ]
        assert failures[1].get("covered_files") == ["code/world_web.py", "code/lore.py"]

        assert coverage.get("source") == "lcov"
        by_test = coverage.get("by_test", {})
        assert isinstance(by_test, dict)
        assert "test_receipt_gate" in by_test
        assert "code/world_web.py" in by_test["test_receipt_gate"]
        by_test_spans = coverage.get("by_test_spans", {})
        assert isinstance(by_test_spans, dict)
        assert "test_receipt_gate" in by_test_spans
        receipt_spans = by_test_spans.get("test_receipt_gate", [])
        assert isinstance(receipt_spans, list)
        assert receipt_spans
        assert receipt_spans[0].get("file") == "code/world_web.py"
        assert int(receipt_spans[0].get("start_line", 0)) == 12
        assert int(receipt_spans[0].get("end_line", 0)) == 12
        assert float(receipt_spans[0].get("weight", 0.0)) > 0.0

        truth_spans = by_test_spans.get("test_truth_flow", [])
        assert isinstance(truth_spans, list)
        assert truth_spans
        assert truth_spans[0].get("file") == "code/lore.py"
        assert int(truth_spans[0].get("start_line", 0)) == 1
        assert int(truth_spans[0].get("end_line", 0)) == 2
        assert coverage.get("hottest_files", [""])[0] == "code/world_web.py"


def test_pain_field_uses_hottest_coverage_file_when_failure_has_no_mapping() -> None:
    hot_path = "code/world_web.py"
    cool_path = "code/lore.py"
    hot_id = world_web_module._file_id_for_path(hot_path)
    cool_id = world_web_module._file_id_for_path(cool_path)

    logical_graph = {
        "nodes": [
            {
                "id": "logical:file:hot",
                "kind": "file",
                "file_id": hot_id,
                "x": 0.22,
                "y": 0.33,
                "label": "world_web.py",
            },
            {
                "id": "logical:file:cool",
                "kind": "file",
                "file_id": cool_id,
                "x": 0.67,
                "y": 0.58,
                "label": "lore.py",
            },
        ],
        "edges": [],
        "joins": {
            "file_index": {
                hot_path: hot_id,
                cool_path: cool_id,
            }
        },
    }
    catalog = {
        "test_failures": [
            {
                "name": "test_receipt_gate",
                "status": "failed",
                "message": "assertion failed",
            }
        ],
        "test_coverage": {
            "files": {
                hot_path: {"line_rate": 0.12, "lines_found": 120},
                cool_path: {"line_rate": 0.96, "lines_found": 120},
            }
        },
    }

    pain_field = world_web_module._build_pain_field(catalog, logical_graph)
    failing_tests = pain_field.get("failing_tests", [])

    assert failing_tests
    assert hot_path in failing_tests[0].get("covered_files", [])
    assert hot_id in failing_tests[0].get("file_ids", [])
    debug_target = pain_field.get("debug", {})
    assert debug_target.get("meaning") == "DEBUG"
    assert debug_target.get("grounded") is True
    assert debug_target.get("path") == hot_path
    assert debug_target.get("file_id") == hot_id
    assert debug_target.get("source") in {
        "pain_field.max_heat",
        "coverage.hottest_files",
    }
    assert any(
        row.get("file_id") == hot_id and row.get("heat", 0) > 0.0
        for row in pain_field.get("node_heat", [])
    )


def test_simulation_state_includes_presence_dynamics_and_file_sentinel() -> None:
    simulation = build_simulation_state(
        {"items": [], "counts": {"audio": 3}},
        influence_snapshot={
            "clicks_45s": 2,
            "file_changes_120s": 4,
            "compute_jobs_180s": 2,
            "compute_summary": {
                "llm_jobs": 1,
                "embedding_jobs": 1,
                "ok_count": 1,
                "error_count": 1,
                "resource_counts": {"gpu": 2},
            },
            "compute_jobs": [
                {
                    "id": "compute:test-1",
                    "at": "2026-02-18T00:00:00+00:00",
                    "ts": 1_700_000_000.0,
                    "kind": "llm",
                    "op": "text_generate.ollama",
                    "backend": "ollama",
                    "resource": "gpu",
                    "emitter_presence_id": "health_sentinel_gpu1",
                    "target_presence_id": "witness_thread",
                    "model": "qwen3-vl:2b-instruct",
                    "status": "ok",
                    "latency_ms": 84.2,
                    "error": "",
                }
            ],
            "recent_click_targets": ["particle_field"],
            "recent_file_paths": ["receipts.log"],
            "fork_tax": {
                "law_en": "Pay the fork tax.",
                "law_ja": "フォーク税は法。",
                "debt": 5.0,
                "paid": 2.0,
                "balance": 3.0,
                "paid_ratio": 0.4,
            },
            "ghost": {
                "id": "file_sentinel",
                "en": "File Sentinel",
                "ja": "ファイルの哨戒者",
                "auto_commit_pulse": 0.5,
                "queue_pending": 1,
                "status_en": "staging receipts",
                "status_ja": "領収書を段取り中",
            },
        },
        queue_snapshot={"pending_count": 1, "event_count": 4},
    )
    dynamics = simulation.get("presence_dynamics", {})
    assert dynamics.get("fork_tax", {}).get("law_ja") == "フォーク税は法。"
    assert dynamics.get("ghost", {}).get("id") == "file_sentinel"
    witness = dynamics.get("witness_thread", {})
    assert witness.get("id") == "witness_thread"
    assert witness.get("ja") == "証人の糸"
    lineage = witness.get("lineage", [])
    assert any(item.get("ref") == "particle_field" for item in lineage)
    assert any(item.get("ref") == "receipts.log" for item in lineage)
    impacts = dynamics.get("presence_impacts", [])
    assert any(item.get("id") == "receipt_river" for item in impacts)
    assert any(item.get("id") == "file_sentinel" for item in impacts)
    assert dynamics.get("compute_jobs_180s") == 2
    assert dynamics.get("compute_summary", {}).get("llm_jobs") == 1
    compute_jobs = dynamics.get("compute_jobs", [])
    assert isinstance(compute_jobs, list)
    assert compute_jobs[0].get("id") == "compute:test-1"
    assert compute_jobs[0].get("emitter_presence_id") == "health_sentinel_gpu1"
    simulation_budget = dynamics.get("simulation_budget", {})
    assert int(simulation_budget.get("point_limit", 0)) <= int(
        simulation_budget.get("point_limit_max", 0)
    )
    slice_offload = simulation_budget.get("slice_offload", {})
    assert str(slice_offload.get("source", "")).strip()
    assert "fallback" in slice_offload
    field_particles = dynamics.get("field_particles", [])
    assert isinstance(field_particles, list)
    assert dynamics.get("field_particles_record") == "ημ.field-particles.v1"
    assert simulation.get("field_particles") == field_particles
    outcome_summary = dynamics.get("daimoi_outcome_summary", {})
    assert isinstance(outcome_summary, dict)
    assert "food" in outcome_summary
    assert "death" in outcome_summary
    assert "total" in outcome_summary
    trails = dynamics.get("daimoi_outcome_trails", [])
    assert isinstance(trails, list)
    resource_daimoi = dynamics.get("resource_daimoi", {})
    assert resource_daimoi.get("record") == "eta-mu.resource-daimoi-flow.v1"
    assert "delivered_packets" in resource_daimoi
    assert "total_transfer" in resource_daimoi
    resource_consumption = dynamics.get("resource_consumption", {})
    assert resource_consumption.get("record") == "eta-mu.resource-daimoi-consumption.v1"
    assert "action_packets" in resource_consumption
    assert "blocked_packets" in resource_consumption
    assert "consumed_total" in resource_consumption
    if field_particles:
        first_particle = field_particles[0]
        assert str(first_particle.get("presence_id", "")).strip()
        assert str(first_particle.get("presence_role", "")).strip()
        assert first_particle.get("particle_mode") in {"neutral", "role-bound"}
        assert 0.0 <= float(first_particle.get("r", 0.0)) <= 0.8
        assert 0.0 <= float(first_particle.get("g", 0.0)) <= 0.8
        assert 0.0 <= float(first_particle.get("b", 0.0)) <= 0.8
        if first_particle.get("resource_consume_amount") is not None:
            assert float(first_particle.get("resource_consume_amount", 0.0)) >= 0.0
        if first_particle.get("resource_action_blocked") is not None:
            assert isinstance(first_particle.get("resource_action_blocked"), bool)


def test_simulation_state_witness_thread_uses_idle_lineage_without_events() -> None:
    simulation = build_simulation_state(
        {"items": [], "counts": {}},
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_click_targets": [],
            "recent_file_paths": [],
            "fork_tax": {
                "law_en": "Pay the fork tax.",
                "law_ja": "フォーク税は法。",
                "debt": 0.0,
                "paid": 0.0,
                "balance": 0.0,
                "paid_ratio": 1.0,
            },
            "ghost": {
                "id": "file_sentinel",
                "en": "File Sentinel",
                "ja": "ファイルの哨戒者",
                "auto_commit_pulse": 0.0,
                "queue_pending": 0,
                "status_en": "gate idle",
                "status_ja": "門前で待機中",
            },
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )
    witness = simulation.get("presence_dynamics", {}).get("witness_thread", {})
    lineage = witness.get("lineage", [])
    assert isinstance(lineage, list)
    assert lineage[0]["kind"] == "idle"
    assert lineage[0]["ref"] == "awaiting-touch"


def test_backend_field_particles_shift_toward_embedded_similarity(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module.time, "time", lambda: 1_700_100_000.0)

    base_catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 2,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 2,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 2},
                "field_counts": {"f2": 2},
                "knowledge_entries": 2,
            },
        },
    }

    embed_catalog = json.loads(json.dumps(base_catalog))
    embed_catalog["file_graph"]["file_nodes"] = [
        {
            "id": "file:witness-embed",
            "name": "witness_trace.md",
            "kind": "text",
            "summary": "witness trace continuity lineage",
            "text_excerpt": "witness trace continuity lineage",
            "tags": ["witness", "trace", "lineage"],
            "dominant_field": "f2",
            "field_scores": {"f2": 0.92, "f7": 0.08},
            "vecstore_collection": "eta_mu_nexus_v1",
            "embed_layer_count": 1,
            "x": 0.7,
            "y": 0.3,
            "hue": 250,
            "importance": 0.95,
        },
        {
            "id": "file:witness-support",
            "name": "witness_context.md",
            "kind": "text",
            "summary": "witness field context",
            "tags": ["witness", "field"],
            "dominant_field": "f2",
            "field_scores": {"f2": 0.86},
            "x": 0.64,
            "y": 0.34,
            "hue": 242,
            "importance": 0.78,
        },
    ]

    plain_catalog = json.loads(json.dumps(base_catalog))
    plain_catalog["file_graph"]["file_nodes"] = [
        {
            "id": "file:witness-plain",
            "name": "witness_trace.md",
            "kind": "text",
            "summary": "witness trace continuity lineage",
            "text_excerpt": "witness trace continuity lineage",
            "tags": ["witness", "trace", "lineage"],
            "dominant_field": "f2",
            "field_scores": {"f2": 0.92, "f7": 0.08},
            "x": 0.7,
            "y": 0.3,
            "hue": 250,
            "importance": 0.95,
        },
        {
            "id": "file:witness-support",
            "name": "witness_context.md",
            "kind": "text",
            "summary": "witness field context",
            "tags": ["witness", "field"],
            "dominant_field": "f2",
            "field_scores": {"f2": 0.86},
            "x": 0.64,
            "y": 0.34,
            "hue": 242,
            "importance": 0.78,
        },
    ]

    cache = getattr(world_web_module, "_DAIMO_DYNAMICS_CACHE", {})
    if isinstance(cache, dict):
        cache["field_particles"] = {}
    embed_simulation = build_simulation_state(embed_catalog)

    if isinstance(cache, dict):
        cache["field_particles"] = {}
    plain_simulation = build_simulation_state(plain_catalog)

    def _witness_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        rows = payload.get("presence_dynamics", {}).get("field_particles", [])
        return [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("presence_id", "")).strip() == "witness_thread"
        ]

    witness_embed = _witness_rows(embed_simulation)
    witness_plain = _witness_rows(plain_simulation)

    assert witness_embed
    assert witness_plain

    target_x = 0.67
    target_y = 0.32

    def _mean_distance(rows: list[dict[str, Any]]) -> float:
        total = 0.0
        for row in rows:
            dx = float(row.get("x", 0.5)) - target_x
            dy = float(row.get("y", 0.5)) - target_y
            total += (dx * dx + dy * dy) ** 0.5
        return total / max(1, len(rows))

    assert embed_simulation.get("embedding_particles")
    assert plain_simulation.get("embedding_particles") == []
    assert witness_embed != witness_plain
    assert abs(_mean_distance(witness_embed) - _mean_distance(witness_plain)) > 0.0001


def test_backend_field_particles_scale_with_local_cluster_density(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module.time, "time", lambda: 1_700_200_000.0)

    def _catalog_with_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "items": [],
            "counts": {},
            "file_graph": {
                "record": "ημ.file-graph.v1",
                "generated_at": "2026-02-18T00:00:00+00:00",
                "inbox": {
                    "record": "ημ.inbox.v1",
                    "path": "/tmp/.ημ",
                    "pending_count": 0,
                    "processed_count": len(nodes),
                    "failed_count": 0,
                    "is_empty": True,
                    "knowledge_entries": len(nodes),
                    "last_ingested_at": "2026-02-18T00:00:00+00:00",
                    "errors": [],
                },
                "nodes": [],
                "field_nodes": [],
                "file_nodes": nodes,
                "edges": [],
                "stats": {
                    "field_count": 1,
                    "file_count": len(nodes),
                    "edge_count": 0,
                    "kind_counts": {"text": len(nodes)},
                    "field_counts": {"f2": len(nodes)},
                    "knowledge_entries": len(nodes),
                },
            },
        }

    dense_nodes: list[dict[str, Any]] = []
    for idx in range(5):
        dense_nodes.append(
            {
                "id": f"file:dense:{idx}",
                "name": f"witness_dense_{idx}.md",
                "kind": "text",
                "summary": "witness continuity cluster",
                "tags": ["witness", "trace", "cluster"],
                "dominant_field": "f2",
                "field_scores": {"f2": 0.9, "f7": 0.1},
                "x": 0.62 + (idx * 0.012),
                "y": 0.31 + (idx * 0.008),
                "importance": 0.72,
            }
        )

    sparse_nodes: list[dict[str, Any]] = []
    for idx in range(5):
        sparse_nodes.append(
            {
                "id": f"file:sparse:{idx}",
                "name": f"witness_sparse_{idx}.md",
                "kind": "text",
                "summary": "witness continuity distributed",
                "tags": ["witness", "trace", "cluster"],
                "dominant_field": "f2",
                "field_scores": {"f2": 0.9, "f7": 0.1},
                "x": 0.08 + (idx * 0.18),
                "y": 0.92 - (idx * 0.18),
                "importance": 0.72,
            }
        )

    cache = getattr(world_web_module, "_DAIMO_DYNAMICS_CACHE", {})
    if isinstance(cache, dict):
        cache["field_particles"] = {}
    dense_simulation = build_simulation_state(_catalog_with_nodes(dense_nodes))

    if isinstance(cache, dict):
        cache["field_particles"] = {}
    sparse_simulation = build_simulation_state(_catalog_with_nodes(sparse_nodes))

    def _witness_count(payload: dict[str, Any]) -> int:
        rows = payload.get("presence_dynamics", {}).get("field_particles", [])
        return len(
            [
                row
                for row in rows
                if isinstance(row, dict)
                and str(row.get("presence_id", "")).strip() == "witness_thread"
            ]
        )

    dense_count = _witness_count(dense_simulation)
    sparse_count = _witness_count(sparse_simulation)

    assert dense_count > sparse_count
    assert dense_count >= 6
