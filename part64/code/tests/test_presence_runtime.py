from __future__ import annotations

import time
from typing import Any

import pytest

from code.world_web import (
    InMemoryPresenceStorage,
    PresenceRuntimeManager,
    build_simulation_delta,
    build_simulation_state,
    reset_presence_runtime_state_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_runtime_state() -> Any:
    reset_presence_runtime_state_for_tests()
    yield
    reset_presence_runtime_state_for_tests()


def _particle(
    *,
    particle_id: str,
    presence_id: str,
    x: float,
    y: float,
    size: float = 1.2,
) -> dict[str, Any]:
    return {
        "id": particle_id,
        "presence_id": presence_id,
        "x": x,
        "y": y,
        "size": size,
        "r": 0.35,
        "g": 0.42,
        "b": 0.51,
    }


def _synthetic_file_graph(
    *, file_count: int, edges_per_file: int = 3
) -> dict[str, Any]:
    field_nodes = [
        {
            "id": "field:witness_thread",
            "node_id": "witness_thread",
            "node_type": "field",
            "field": "f2",
            "label": "Witness Thread",
            "x": 0.5,
            "y": 0.5,
            "hue": 200,
        }
    ]
    file_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    safe_count = max(1, int(file_count))
    for index in range(safe_count):
        node_id = f"file:test:{index:04d}"
        rel_path = f"docs/node_{index:04d}.md"
        importance = 0.18 + ((index % 7) * 0.08)
        file_nodes.append(
            {
                "id": node_id,
                "node_id": rel_path,
                "node_type": "file",
                "name": f"node_{index:04d}.md",
                "label": f"node_{index:04d}.md",
                "kind": "text",
                "x": round(((index % 19) + 1) / 20.0, 4),
                "y": round((((index // 19) % 19) + 1) / 20.0, 4),
                "hue": int((180 + (index * 7)) % 360),
                "importance": round(min(1.0, importance), 4),
                "source_rel_path": rel_path,
                "dominant_field": "f2" if index % 3 else "f6",
                "dominant_presence": "witness_thread"
                if index % 3
                else "anchor_registry",
                "field_scores": {"f2": 0.62, "f6": 0.38},
                "summary": "synthetic graph node",
                "embed_layer_count": 1 if index % 5 else 0,
                "vecstore_collection": "eta_mu_nexus_v1" if index % 4 else "",
            }
        )

    for index in range(safe_count):
        source_id = f"file:test:{index:04d}"
        for offset in range(1, max(1, int(edges_per_file)) + 1):
            target_index = (index + offset) % safe_count
            target_id = f"file:test:{target_index:04d}"
            edges.append(
                {
                    "id": f"edge:test:{index:04d}:{target_index:04d}",
                    "source": source_id,
                    "target": target_id,
                    "field": "f2",
                    "weight": 0.42,
                    "kind": "relates_tag",
                }
            )
        edges.append(
            {
                "id": f"edge:test:field:{index:04d}",
                "source": source_id,
                "target": "field:witness_thread",
                "field": "f2",
                "weight": 0.66,
                "kind": "categorizes",
            }
        )

    return {
        "record": "ημ.file-graph.v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "nodes": [*field_nodes, *file_nodes],
        "field_nodes": field_nodes,
        "tag_nodes": [],
        "file_nodes": file_nodes,
        "embed_layers": [],
        "edges": edges,
        "stats": {
            "field_count": len(field_nodes),
            "file_count": len(file_nodes),
            "edge_count": len(edges),
            "kind_counts": {"text": len(file_nodes)},
            "field_counts": {"f2": len(file_nodes)},
            "knowledge_entries": len(file_nodes),
        },
    }


def _retarget_graph_to_mage_hub(graph: dict[str, Any]) -> None:
    mage_field_id = "field:mage_of_receipts"
    mage_field = {
        "id": mage_field_id,
        "node_id": "mage_of_receipts",
        "node_type": "field",
        "field": "f6",
        "label": "Mage of Receipts",
        "x": 0.33,
        "y": 0.71,
        "hue": 286,
    }

    field_nodes = graph.get("field_nodes", [])
    if isinstance(field_nodes, list) and not any(
        str(row.get("id", "")) == mage_field_id
        for row in field_nodes
        if isinstance(row, dict)
    ):
        field_nodes.append(dict(mage_field))

    nodes = graph.get("nodes", [])
    if isinstance(nodes, list) and not any(
        str(row.get("id", "")) == mage_field_id
        for row in nodes
        if isinstance(row, dict)
    ):
        nodes.append(dict(mage_field))

    for node in graph.get("file_nodes", []):
        if not isinstance(node, dict):
            continue
        node["dominant_field"] = "f6"
        node["dominant_presence"] = "mage_of_receipts"
        node["field_scores"] = {"f6": 1.0}

    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        if str(edge.get("kind", "")) != "categorizes":
            continue
        edge["target"] = mage_field_id
        edge["field"] = "f6"
        edge["weight"] = 0.9


def test_presence_runtime_handoff_moves_ownership_and_emits_inbox_event() -> None:
    storage = InMemoryPresenceStorage()
    manager = PresenceRuntimeManager(
        storage=storage,
        enabled=True,
        backend_name="memory",
        instance_id="writer-a",
        lease_ttl_ms=5000,
        max_writes_per_presence=32,
    )

    first = manager.sync(
        field_particles=[
            _particle(
                particle_id="field:witness_thread:0",
                presence_id="witness_thread",
                x=0.22,
                y=0.33,
            )
        ],
        presence_impacts=[{"id": "witness_thread"}],
        queue_ratio=0.1,
        resource_ratio=0.2,
    )
    assert first["counts"]["daimoi_updates"] == 1
    assert storage.presence_daimoi_ids("witness_thread") == {"field:witness_thread:0"}

    second = manager.sync(
        field_particles=[
            _particle(
                particle_id="field:witness_thread:0",
                presence_id="anchor_registry",
                x=0.71,
                y=0.45,
            )
        ],
        presence_impacts=[{"id": "witness_thread"}, {"id": "anchor_registry"}],
        queue_ratio=0.12,
        resource_ratio=0.28,
    )

    assert second["counts"]["handoffs"] == 1
    assert storage.presence_daimoi_ids("witness_thread") == set()
    assert storage.presence_daimoi_ids("anchor_registry") == {"field:witness_thread:0"}

    owner, _ = storage.get_daimoi_owner_ver("field:witness_thread:0")
    assert owner == "anchor_registry"

    inbox_rows = storage.inbox_events("anchor_registry", limit=20)
    assert any(
        str(row.get("event", {}).get("kind", "")) == "daimoi.owner.handoff"
        for row in inbox_rows
    )


def test_presence_runtime_reports_pause_resume_dedupe_and_rate_limit() -> None:
    storage = InMemoryPresenceStorage()
    manager = PresenceRuntimeManager(
        storage=storage,
        enabled=True,
        backend_name="memory",
        instance_id="writer-a",
        lease_ttl_ms=5000,
        max_writes_per_presence=1,
    )

    now_ms = int(time.time() * 1000)
    storage.seed_lease("witness_thread", "writer-b", expires_ms=now_ms + 8000)

    paused_snapshot = manager.sync(
        field_particles=[
            _particle(
                particle_id="field:witness_thread:0",
                presence_id="witness_thread",
                x=0.4,
                y=0.4,
            )
        ],
        presence_impacts=[{"id": "witness_thread"}],
        queue_ratio=0.0,
        resource_ratio=0.0,
    )
    assert paused_snapshot["counts"]["paused"] == 1
    assert paused_snapshot["counts"]["active_writers"] == 0

    storage.seed_lease("witness_thread", "writer-a", expires_ms=now_ms + 8000)
    resumed_snapshot = manager.sync(
        field_particles=[
            _particle(
                particle_id="field:witness_thread:0",
                presence_id="witness_thread",
                x=0.45,
                y=0.42,
            )
        ],
        presence_impacts=[{"id": "witness_thread"}],
        queue_ratio=0.0,
        resource_ratio=0.0,
    )
    assert resumed_snapshot["counts"]["resumed"] == 1
    assert resumed_snapshot["counts"]["daimoi_updates"] == 1

    dedupe_snapshot = manager.sync(
        field_particles=[
            _particle(
                particle_id="field:witness_thread:0",
                presence_id="witness_thread",
                x=0.45,
                y=0.42,
            )
        ],
        presence_impacts=[{"id": "witness_thread"}],
        queue_ratio=0.0,
        resource_ratio=0.0,
    )
    assert dedupe_snapshot["counts"]["deduped"] >= 1

    rate_limited_snapshot = manager.sync(
        field_particles=[
            _particle(
                particle_id="field:witness_thread:1",
                presence_id="witness_thread",
                x=0.5,
                y=0.49,
            ),
            _particle(
                particle_id="field:witness_thread:2",
                presence_id="witness_thread",
                x=0.53,
                y=0.5,
            ),
        ],
        presence_impacts=[{"id": "witness_thread"}],
        queue_ratio=0.0,
        resource_ratio=0.0,
    )
    assert rate_limited_snapshot["counts"]["rate_limited"] >= 1

    kinds = [
        str(row.get("event", {}).get("kind", ""))
        for row in storage.list_bus_events(limit=120)
    ]
    assert "presence.writer.paused" in kinds
    assert "presence.writer.resumed" in kinds
    assert "daimoi.write.deduped" in kinds
    assert "presence.write.rate-limited" in kinds


def test_presence_runtime_compliance_blocked_for_invalid_particle_identity() -> None:
    storage = InMemoryPresenceStorage()
    manager = PresenceRuntimeManager(
        storage=storage,
        enabled=True,
        backend_name="memory",
        instance_id="writer-a",
        lease_ttl_ms=5000,
        max_writes_per_presence=8,
    )

    snapshot = manager.sync(
        field_particles=[{"presence_id": "witness_thread", "x": 0.2, "y": 0.3}],
        presence_impacts=[{"id": "witness_thread"}],
        queue_ratio=0.0,
        resource_ratio=0.0,
    )

    assert snapshot["counts"]["compliance_blocked"] >= 1
    events = storage.list_bus_events(limit=16)
    assert any(
        str(row.get("event", {}).get("kind", "")) == "presence.compliance.blocked"
        for row in events
    )


def test_build_simulation_delta_reports_changed_keys() -> None:
    previous = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "total": 1,
        "audio": 0,
        "image": 0,
        "video": 0,
        "points": [{"x": 0.0, "y": 0.0}],
        "presence_dynamics": {"click_events": 0},
        "daimoi": {"record": "eta-mu.daimoi.v1"},
    }
    current = {
        "timestamp": "2026-01-01T00:00:01+00:00",
        "total": 2,
        "audio": 1,
        "image": 0,
        "video": 0,
        "points": [{"x": 0.2, "y": 0.2}],
        "presence_dynamics": {"click_events": 2},
        "daimoi": {"record": "eta-mu.daimoi.v1"},
    }

    delta = build_simulation_delta(previous, current)

    assert delta["record"] == "eta-mu.simulation-delta.v1"
    assert delta["has_changes"] is True
    assert "points" in delta["changed_keys"]
    assert "presence_dynamics" in delta["changed_keys"]
    assert delta["patch"]["total"] == 2


def test_build_simulation_state_includes_distributed_runtime_snapshot(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("PRESENCE_RUNTIME_BACKEND", "memory")
    reset_presence_runtime_state_for_tests()

    simulation = build_simulation_state(
        {"items": [], "counts": {"audio": 2, "image": 1, "video": 0}},
        influence_snapshot={
            "clicks_45s": 1,
            "file_changes_120s": 2,
            "recent_click_targets": ["witness_thread"],
            "recent_file_paths": ["receipts.log"],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    runtime_snapshot = simulation.get("presence_dynamics", {}).get(
        "distributed_runtime", {}
    )
    assert runtime_snapshot.get("record") == "eta-mu.presence-runtime.snapshot.v1"
    assert runtime_snapshot.get("schema_version") == "presence.redis.v1"
    assert runtime_snapshot.get("backend") in {"memory", "redis"}
    counts = runtime_snapshot.get("counts", {})
    assert int(counts.get("presences", 0)) >= 1
    assert "daimoi_updates" in counts


def test_build_simulation_state_growth_guard_deploys_daimoi_consolidation() -> None:
    original_file_count = 300
    graph = _synthetic_file_graph(file_count=original_file_count, edges_per_file=3)

    simulation = build_simulation_state(
        {
            "items": [
                {
                    "rel_path": f"docs/item_{index:04d}.md",
                    "part": "part64",
                    "kind": "text",
                }
                for index in range(360)
            ],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
        },
        influence_snapshot={
            "clicks_45s": 4,
            "file_changes_120s": 12,
            "recent_file_paths": ["docs/node_0001.md", "docs/node_0002.md"],
        },
        queue_snapshot={"pending_count": 8, "event_count": 12},
    )

    dynamics = simulation.get("presence_dynamics", {})
    guard = dynamics.get("growth_guard", {})
    assert guard.get("record") == "eta-mu.simulation-growth-guard.v1"
    assert guard.get("schema_version") == "simulation.growth-guard.v1"
    assert guard.get("active") is True
    action = guard.get("action", {})
    assert action.get("kind") == "daimoi.consolidation.deployed"
    assert int(action.get("collapsed_file_nodes", 0)) > 0
    assert int(action.get("clusters", 0)) >= 1

    compacted_file_nodes = simulation.get("file_graph", {}).get("file_nodes", [])
    assert len(compacted_file_nodes) < original_file_count
    compacted_stats = simulation.get("file_graph", {}).get("stats", {})
    assert bool(compacted_stats.get("consolidation_applied", False)) is True

    guard_events = guard.get("events", [])
    assert any(
        str(row.get("kind", "")) == "daimoi.consolidation.deployed"
        for row in guard_events
        if isinstance(row, dict)
    )

    daimoi_state = simulation.get("daimoi", {})
    assert isinstance(daimoi_state.get("growth_guard", {}), dict)
    daimoi_rows = daimoi_state.get("daimoi", [])
    assert any(
        str(row.get("id", "")).startswith("daimo:consolidator")
        for row in daimoi_rows
        if isinstance(row, dict)
    )


def test_growth_guard_preserves_recent_hot_paths_when_consolidating() -> None:
    graph = _synthetic_file_graph(file_count=300, edges_per_file=3)
    hot_paths = ["docs/hot_alpha.md", "docs/hot_beta.md"]
    file_nodes = graph.get("file_nodes", [])
    assert isinstance(file_nodes, list)
    if len(file_nodes) >= 2:
        file_nodes[0]["source_rel_path"] = hot_paths[0]
        file_nodes[0]["name"] = "hot_alpha.md"
        file_nodes[1]["source_rel_path"] = hot_paths[1]
        file_nodes[1]["name"] = "hot_beta.md"

    simulation = build_simulation_state(
        {
            "items": [
                {
                    "rel_path": f"docs/item_{index:04d}.md",
                    "part": "part64",
                    "kind": "text",
                }
                for index in range(360)
            ],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
        },
        influence_snapshot={
            "clicks_45s": 2,
            "file_changes_120s": 10,
            "recent_file_paths": hot_paths,
        },
        queue_snapshot={"pending_count": 6, "event_count": 10},
    )

    compacted_nodes = simulation.get("file_graph", {}).get("file_nodes", [])
    compacted_paths = {
        str(row.get("source_rel_path", "")).strip()
        for row in compacted_nodes
        if isinstance(row, dict)
    }
    for hot_path in hot_paths:
        assert hot_path in compacted_paths


def test_simulation_projection_collapses_hub_edges_and_preserves_membership() -> None:
    graph = _synthetic_file_graph(file_count=180, edges_per_file=2)
    _retarget_graph_to_mage_hub(graph)
    original_edge_ids = {
        str(row.get("id", "")).strip()
        for row in graph.get("edges", [])
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }

    simulation = build_simulation_state(
        {
            "items": [
                {
                    "rel_path": f"docs/projection_item_{index:04d}.md",
                    "part": "part64",
                    "kind": "text",
                }
                for index in range(24)
            ],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
        },
        influence_snapshot={
            "clicks_45s": 1,
            "file_changes_120s": 3,
            "recent_file_paths": [
                "docs/node_0001.md",
                "docs/node_0002.md",
                "docs/node_0003.md",
            ],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    projected_graph = simulation.get("file_graph", {})
    projected_edges = projected_graph.get("edges", [])
    projection = projected_graph.get("projection", {})
    assert projection.get("record") == "ημ.file-graph-projection.v1"
    assert projection.get("schema_version") == "file-graph.projection.v1"
    assert bool(projection.get("active", False)) is True
    assert int(projection.get("collapsed_edges", 0)) > 0
    assert len(projected_edges) < len(original_edge_ids)

    stats = projected_graph.get("stats", {})
    assert int(stats.get("projection_collapsed_edge_count", 0)) > 0
    assert int(stats.get("projection_overflow_edge_count", 0)) >= 1
    overflow_nodes = [
        row
        for row in projected_graph.get("file_nodes", [])
        if isinstance(row, dict) and bool(row.get("projection_overflow", False))
    ]
    assert overflow_nodes
    assert all(str(row.get("graph_scope", "")) == "view" for row in overflow_nodes)
    assert all(
        str(row.get("simulation_semantic_role", "")) == "view_compaction_aggregate"
        for row in overflow_nodes
    )
    assert all(
        float(row.get("semantic_bundle_gravity", 0.0)) > 0.0 for row in overflow_nodes
    )

    recovered_edge_ids = {
        str(row.get("id", "")).strip()
        for row in projected_edges
        if isinstance(row, dict) and str(row.get("id", "")).strip() in original_edge_ids
    }
    groups = projection.get("groups", [])
    assert isinstance(groups, list)
    assert len(groups) >= 1
    assert all(
        bool(group.get("surface_visible", False))
        for group in groups
        if isinstance(group, dict)
    )
    for group in groups:
        if not isinstance(group, dict):
            continue
        for edge_id in group.get("member_edge_ids", []):
            clean = str(edge_id).strip()
            if clean:
                recovered_edge_ids.add(clean)
    assert recovered_edge_ids == original_edge_ids

    guard_events = (
        simulation.get("presence_dynamics", {})
        .get("growth_guard", {})
        .get("events", [])
    )
    assert any(
        str(row.get("kind", "")) == "simulation.file_graph.projection.applied"
        for row in guard_events
        if isinstance(row, dict)
    )

    field_particles = simulation.get("presence_dynamics", {}).get("field_particles", [])
    bundle_nexus_rows = [
        row
        for row in field_particles
        if isinstance(row, dict)
        and bool(row.get("is_nexus", False))
        and bool(row.get("is_view_compaction_bundle", False))
    ]
    assert bundle_nexus_rows
    assert all(
        str(row.get("simulation_semantic_role", "")) == "view_compaction_aggregate"
        for row in bundle_nexus_rows
    )
    assert max(float(row.get("mass", 0.0)) for row in bundle_nexus_rows) >= 4.0


def test_simulation_exposes_truth_and_view_graph_contracts() -> None:
    graph = _synthetic_file_graph(file_count=180, edges_per_file=2)
    _retarget_graph_to_mage_hub(graph)

    simulation = build_simulation_state(
        {
            "items": [
                {
                    "rel_path": f"docs/projection_item_{index:04d}.md",
                    "part": "part64",
                    "kind": "text",
                }
                for index in range(24)
            ],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
        },
        influence_snapshot={
            "clicks_45s": 1,
            "file_changes_120s": 3,
            "recent_file_paths": ["docs/node_0001.md", "docs/node_0002.md"],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    truth_graph = simulation.get("truth_graph", {})
    assert truth_graph.get("record") == "eta-mu.truth-graph.v1"
    assert truth_graph.get("schema_version") == "truth.graph.v1"
    assert int(truth_graph.get("node_count", 0)) > 0
    assert int(truth_graph.get("edge_count", 0)) > 0
    truth_semantics = truth_graph.get("semantics", {})
    assert truth_semantics.get("graph_domain") == "truth_graph"
    assert truth_semantics.get("includes_projection_bundles") is False

    view_graph = simulation.get("view_graph", {})
    assert view_graph.get("record") == "eta-mu.view-graph.v1"
    assert view_graph.get("schema_version") == "view.graph.v1"
    assert int(view_graph.get("node_count", 0)) > 0
    assert int(view_graph.get("edge_count", 0)) > 0
    view_semantics = view_graph.get("semantics", {})
    assert view_semantics.get("graph_domain") == "view_graph"

    projection = view_graph.get("projection", {})
    assert projection.get("mode") in {"hub-overflow", "none"}
    pi_contract = view_graph.get("projection_pi", {})
    assert pi_contract.get("kind") in {"identity", "edge-bundle"}
    if bool(projection.get("active", False)):
        assert int(projection.get("bundle_ledger_count", 0)) >= 1
        assert int(projection.get("reconstructable_bundle_count", 0)) >= 1
        ledgers = projection.get("bundle_ledgers", [])
        assert isinstance(ledgers, list)
        assert int(ledgers[0].get("member_edge_count", 0)) >= 1
        assert view_semantics.get("includes_projection_bundles") is True
        assert int(view_semantics.get("projection_bundle_node_count", 0)) >= 1


def test_simulation_projection_is_deterministic_for_same_input() -> None:
    graph = _synthetic_file_graph(file_count=180, edges_per_file=2)
    _retarget_graph_to_mage_hub(graph)
    payload = {
        "items": [
            {
                "rel_path": f"docs/projection_item_{index:04d}.md",
                "part": "part64",
                "kind": "text",
            }
            for index in range(24)
        ],
        "counts": {"audio": 0, "image": 0, "video": 0},
        "file_graph": graph,
    }
    influence = {
        "clicks_45s": 1,
        "file_changes_120s": 3,
        "recent_file_paths": ["docs/node_0001.md", "docs/node_0002.md"],
    }
    queue = {"pending_count": 0, "event_count": 0}

    first = build_simulation_state(
        payload,
        influence_snapshot=influence,
        queue_snapshot=queue,
    )
    second = build_simulation_state(
        payload,
        influence_snapshot=influence,
        queue_snapshot=queue,
    )

    first_edge_ids = [
        str(row.get("id", "")).strip()
        for row in first.get("file_graph", {}).get("edges", [])
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    ]
    second_edge_ids = [
        str(row.get("id", "")).strip()
        for row in second.get("file_graph", {}).get("edges", [])
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    ]
    assert first_edge_ids == second_edge_ids

    first_groups = [
        (
            str(group.get("id", "")),
            str(group.get("member_edge_digest", "")),
            int(group.get("member_edge_count", 0)),
            bool(group.get("surface_visible", False)),
        )
        for group in first.get("file_graph", {}).get("projection", {}).get("groups", [])
        if isinstance(group, dict)
    ]
    second_groups = [
        (
            str(group.get("id", "")),
            str(group.get("member_edge_digest", "")),
            int(group.get("member_edge_count", 0)),
            bool(group.get("surface_visible", False)),
        )
        for group in second.get("file_graph", {})
        .get("projection", {})
        .get("groups", [])
        if isinstance(group, dict)
    ]
    assert first_groups == second_groups


def test_simulation_projection_skips_small_graphs() -> None:
    graph = _synthetic_file_graph(file_count=32, edges_per_file=2)
    simulation = build_simulation_state(
        {
            "items": [{"rel_path": "docs/a.md", "part": "part64", "kind": "text"}],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
        },
        influence_snapshot={"clicks_45s": 0, "file_changes_120s": 0},
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    projected_graph = simulation.get("file_graph", {})
    projection = projected_graph.get("projection", {})
    assert projection.get("record") == "ημ.file-graph-projection.v1"
    assert bool(projection.get("active", True)) is False
    assert projection.get("reason") in {"below_threshold", "within_projection_limits"}
    assert len(projected_graph.get("edges", [])) == len(graph.get("edges", []))


def test_simulation_projection_can_activate_from_cpu_sentinel_pressure() -> None:
    graph = _synthetic_file_graph(file_count=90, edges_per_file=1)
    crawler_graph = {
        "record": "eta-mu.crawler-graph.v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "field_nodes": [],
        "crawler_nodes": [
            {"id": "crawler:a", "node_id": "crawler:a", "node_type": "crawler"},
            {"id": "crawler:b", "node_id": "crawler:b", "node_type": "crawler"},
        ],
        "edges": [
            {
                "id": f"crawler-edge:{index:04d}",
                "source": "crawler:a",
                "target": "crawler:b",
                "kind": "crawl_ref",
                "weight": 0.42,
            }
            for index in range(520)
        ],
        "stats": {"field_count": 0, "crawler_count": 2, "edge_count": 520},
    }

    simulation = build_simulation_state(
        {
            "items": [
                {"rel_path": "docs/a.md", "part": "part64", "kind": "text"},
                {"rel_path": "docs/b.md", "part": "part64", "kind": "text"},
            ],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
            "crawler_graph": crawler_graph,
        },
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_file_paths": ["docs/node_0001.md", "docs/node_0002.md"],
            "compute_jobs_180s": 140,
            "compute_summary": {"resource_counts": {"cpu": 48}},
            "resource_heartbeat": {
                "devices": {
                    "cpu": {"utilization": 98.0},
                }
            },
        },
        queue_snapshot={"pending_count": 28, "event_count": 144},
    )

    projection = simulation.get("file_graph", {}).get("projection", {})
    assert bool(projection.get("active", False)) is True
    assert projection.get("reason") == "cpu_sentinel_compaction_pressure"
    assert int(projection.get("collapsed_edges", 0)) > 0

    limits = projection.get("limits", {})
    assert int(limits.get("edge_threshold", 0)) < int(
        limits.get("edge_threshold_base", 0)
    )
    assert int(limits.get("edge_cap", 0)) < int(limits.get("edge_cap_base", 0))

    policy = projection.get("policy", {})
    assert policy.get("presence_id") == "health_sentinel_cpu"
    assert float(policy.get("compaction_drive", 0.0)) >= 0.65


def test_simulation_projection_can_activate_from_view_edge_pressure() -> None:
    graph = _synthetic_file_graph(file_count=120, edges_per_file=1)
    crawler_graph = {
        "record": "eta-mu.crawler-graph.v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "field_nodes": [],
        "crawler_nodes": [
            {"id": "crawler:a", "node_id": "crawler:a", "node_type": "crawler"},
            {"id": "crawler:b", "node_id": "crawler:b", "node_type": "crawler"},
        ],
        "edges": [
            {
                "id": f"crawler-edge:view-pressure:{index:04d}",
                "source": "crawler:a",
                "target": "crawler:b",
                "kind": "crawl_ref",
                "weight": 0.42,
            }
            for index in range(800)
        ],
        "stats": {"field_count": 0, "crawler_count": 2, "edge_count": 800},
    }

    simulation = build_simulation_state(
        {
            "items": [
                {"rel_path": "docs/a.md", "part": "part64", "kind": "text"},
                {"rel_path": "docs/b.md", "part": "part64", "kind": "text"},
            ],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
            "crawler_graph": crawler_graph,
        },
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_file_paths": ["docs/node_0001.md"],
            "compute_jobs_180s": 0,
            "compute_summary": {"resource_counts": {"cpu": 0}},
            "resource_heartbeat": {
                "devices": {
                    "cpu": {"utilization": 22.0},
                }
            },
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    projection = simulation.get("file_graph", {}).get("projection", {})
    assert bool(projection.get("active", False)) is True
    assert projection.get("reason") == "cpu_sentinel_compaction_pressure"

    policy = projection.get("policy", {})
    limits = projection.get("limits", {})
    assert int(policy.get("edge_count_file", 0)) < int(limits.get("edge_threshold", 0))
    assert int(policy.get("edge_count_effective", 0)) >= int(
        limits.get("edge_threshold", 0)
    )
    assert float(policy.get("compaction_drive", 0.0)) >= 0.2


def test_simulation_projection_can_activate_from_memory_sentinel_pressure() -> None:
    graph = _synthetic_file_graph(file_count=90, edges_per_file=1)
    crawler_graph = {
        "record": "eta-mu.crawler-graph.v1",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "field_nodes": [],
        "crawler_nodes": [
            {"id": "crawler:a", "node_id": "crawler:a", "node_type": "crawler"},
            {"id": "crawler:b", "node_id": "crawler:b", "node_type": "crawler"},
        ],
        "edges": [
            {
                "id": f"crawler-edge:memory-pressure:{index:04d}",
                "source": "crawler:a",
                "target": "crawler:b",
                "kind": "crawl_ref",
                "weight": 0.42,
            }
            for index in range(640)
        ],
        "stats": {"field_count": 0, "crawler_count": 2, "edge_count": 640},
    }

    simulation = build_simulation_state(
        {
            "items": [
                {"rel_path": "docs/a.md", "part": "part64", "kind": "text"},
                {"rel_path": "docs/b.md", "part": "part64", "kind": "text"},
            ],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
            "crawler_graph": crawler_graph,
        },
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_file_paths": ["docs/node_0001.md"],
            "compute_jobs_180s": 0,
            "compute_summary": {"resource_counts": {"cpu": 0}},
            "resource_heartbeat": {
                "devices": {
                    "cpu": {"utilization": 18.0, "memory_pressure": 0.96},
                },
                "resource_monitor": {
                    "memory_percent": 96.0,
                },
            },
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    projection = simulation.get("file_graph", {}).get("projection", {})
    assert bool(projection.get("active", False)) is True
    assert projection.get("reason") == "memory_sentinel_compaction_pressure"

    policy = projection.get("policy", {})
    limits = projection.get("limits", {})
    assert float(policy.get("memory_pressure", 0.0)) >= 0.7
    assert float(policy.get("cpu_pressure", 0.0)) <= 0.05
    assert policy.get("memory_source") in {"resource_monitor", "cpu.memory_pressure"}
    assert int(limits.get("edge_cap", 0)) < int(limits.get("edge_cap_base", 0))


def test_simulation_projection_can_enter_decompression_mode_with_headroom() -> None:
    graph = _synthetic_file_graph(file_count=160, edges_per_file=2)
    simulation = build_simulation_state(
        {
            "items": [
                {"rel_path": "docs/a.md", "part": "part64", "kind": "text"},
            ],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": graph,
        },
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_file_paths": ["docs/node_0001.md"],
            "compute_jobs_180s": 0,
            "compute_summary": {"resource_counts": {"cpu": 0}},
            "resource_heartbeat": {
                "devices": {
                    "cpu": {"utilization": 16.0},
                }
            },
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    projection = simulation.get("file_graph", {}).get("projection", {})
    assert bool(projection.get("active", False)) is True
    assert projection.get("reason") == "decompression_budget"

    policy = projection.get("policy", {})
    limits = projection.get("limits", {})
    assert policy.get("control_mode") == "decompression"
    assert bool(policy.get("decompression_enabled", False)) is True
    assert float(policy.get("decompression_drive", 0.0)) >= 0.2
    assert int(limits.get("edge_threshold", 0)) > int(
        limits.get("edge_threshold_base", 0)
    )
    assert int(limits.get("edge_cap", 0)) >= int(limits.get("edge_cap_base", 0))


def test_user_search_query_emits_query_daimoi_packet_components() -> None:
    simulation = build_simulation_state(
        {
            "items": [{"rel_path": "docs/a.md", "part": "part64", "kind": "text"}],
            "counts": {"audio": 0, "image": 0, "video": 0},
        },
        influence_snapshot={
            "recent_user_inputs": [
                {
                    "id": "user-search:001",
                    "kind": "search_query",
                    "target": "nexus mage_of_receipts",
                    "message": "receipt graph drift",
                    "embed_daimoi": True,
                    "meta": {
                        "query": "receipt graph drift",
                        "search_daimoi": {
                            "components": [
                                {
                                    "component_id": "query:base",
                                    "component_type": "query-term",
                                    "text": "receipt graph drift",
                                    "weight": 0.92,
                                    "variant_rank": 0,
                                },
                                {
                                    "component_id": "query:alt",
                                    "component_type": "query-term",
                                    "text": "receipt drift graph",
                                    "weight": 0.76,
                                    "variant_rank": 1,
                                },
                            ],
                            "target_presence_ids": ["mage_of_receipts"],
                        },
                    },
                }
            ]
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    rows = simulation.get("presence_dynamics", {}).get("field_particles", [])
    user_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("presence_id", "")) == "presence.user.operator"
        and str(row.get("top_job", "")) == "emit_query_daimoi_packet"
    ]
    assert user_rows
    packet_components = user_rows[0].get("packet_components", [])
    assert isinstance(packet_components, list)
    assert any(
        str(component.get("component_type", "")) == "query-term"
        and "receipt graph drift" in str(component.get("text", ""))
        for component in packet_components
        if isinstance(component, dict)
    )


def test_user_query_transient_edges_promote_after_repeated_hits() -> None:
    def _influence(event_id: str) -> dict[str, Any]:
        return {
            "recent_user_inputs": [
                {
                    "id": event_id,
                    "kind": "search_query",
                    "target": "nexus witness_thread",
                    "message": "where are recent witness receipts",
                    "embed_daimoi": True,
                    "meta": {
                        "query": "where are recent witness receipts",
                        "search_daimoi": {
                            "components": [
                                {
                                    "component_id": "query:witness",
                                    "component_type": "query-term",
                                    "text": "witness receipts",
                                    "weight": 0.9,
                                }
                            ],
                            "target_presence_ids": ["witness_thread"],
                        },
                    },
                }
            ]
        }

    simulation: dict[str, Any] = {}
    for event_id in ("query-hit:1", "query-hit:2", "query-hit:3"):
        simulation = build_simulation_state(
            {
                "items": [{"rel_path": "docs/a.md", "part": "part64", "kind": "text"}],
                "counts": {"audio": 0, "image": 0, "video": 0},
            },
            influence_snapshot=_influence(event_id),
            queue_snapshot={"pending_count": 0, "event_count": 0},
        )

    dynamics = simulation.get("presence_dynamics", {})
    transient_count = int(dynamics.get("user_query_transient_edge_count", 0))
    promoted_count = int(dynamics.get("user_query_promoted_edge_count", 0))
    assert transient_count >= 1
    assert promoted_count >= 1
