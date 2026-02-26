from __future__ import annotations

import math
from typing import Any

import pytest
import code.world_web.server as server_module
import code.world_web.simulation as simulation_module

from code.world_web import build_simulation_state
from code.world_web import daimoi_probabilistic as daimoi_module
from code.world_web.daimoi_probabilistic import (
    build_probabilistic_daimoi_particles,
    reset_probabilistic_daimoi_state_for_tests,
    run_probabilistic_collision_stress,
)


@pytest.fixture(autouse=True)
def _reset_probabilistic_cache() -> Any:
    reset_probabilistic_daimoi_state_for_tests()
    yield
    reset_probabilistic_daimoi_state_for_tests()


def test_probabilistic_collision_stress_has_no_nan_and_normalized_probabilities() -> (
    None
):
    summary = run_probabilistic_collision_stress(iterations=10_000, seed=29)

    assert summary["ok"] is True
    assert summary["nan_count"] == 0
    assert summary["negative_alpha_count"] == 0
    assert float(summary["probability_sum_error_max"]) <= 1e-6


def test_probabilistic_builder_emits_job_distribution_and_actions() -> None:
    particles, model_summary = build_probabilistic_daimoi_particles(
        file_graph={"file_nodes": []},
        presence_impacts=[
            {
                "id": "witness_thread",
                "affected_by": {"files": 0.4, "clicks": 0.6, "resource": 0.2},
                "affects": {"world": 0.7, "ledger": 0.5},
            },
            {
                "id": "anchor_registry",
                "affected_by": {"files": 0.4, "clicks": 0.5, "resource": 0.3},
                "affects": {"world": 0.6, "ledger": 0.6},
            },
        ],
        resource_heartbeat={
            "devices": {
                "cpu": {"utilization": 30.0},
                "gpu1": {"utilization": 22.0},
                "gpu2": {"utilization": 18.0},
                "npu0": {"utilization": 12.0},
            }
        },
        compute_jobs=[],
        queue_ratio=0.2,
        now=1_700_000_000.0,
    )

    assert particles
    assert model_summary["record"] == "ημ.daimoi-probabilistic.v1"

    row = particles[0]
    assert str(row.get("record", "")) == "ημ.daimoi-probabilistic.v1"
    assert 0.0 <= float(row.get("message_probability", 0.0)) <= 1.0
    jobs = row.get("job_probabilities", {})
    assert isinstance(jobs, dict)
    assert jobs
    assert math.isclose(
        sum(float(value) for value in jobs.values()), 1.0, rel_tol=0.0, abs_tol=5e-6
    )
    actions = row.get("action_probabilities", {})
    assert isinstance(actions, dict)
    assert math.isclose(
        float(actions.get("deflect", 0.0)) + float(actions.get("diffuse", 0.0)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert set(row.get("behavior_actions", [])) == {"deflect", "diffuse"}
    assert isinstance(row.get("vx"), float)
    assert isinstance(row.get("vy"), float)
    packet_components = row.get("packet_components", [])
    assert isinstance(packet_components, list)
    assert packet_components
    assert "p_i" in packet_components[0]
    assert "req" in packet_components[0]
    resource_signature = row.get("resource_signature", {})
    assert isinstance(resource_signature, dict)
    assert set(resource_signature.keys()) == set(daimoi_module.DAIMOI_RESOURCE_KEYS)
    absorb_sampler = row.get("absorb_sampler", {})
    assert isinstance(absorb_sampler, dict)
    assert absorb_sampler.get("method") == "gumbel-max"

    packet_contract = model_summary.get("packet_contract", {})
    assert packet_contract.get("schema_version") == "daimoi.packet-components.v1"
    absorb_contract = model_summary.get("absorb_sampler", {})
    assert absorb_contract.get("schema_version") == "daimoi.absorb-sampler.v1"
    assert absorb_contract.get("method") == "gumbel-max"
    assert 0.0 <= float(model_summary.get("resource_pressure", 0.0)) <= 1.0
    assert 0.0 <= float(model_summary.get("queue_pressure", 0.0)) <= 1.0
    assert 0.0 <= float(model_summary.get("compute_pressure", 0.0)) <= 1.0
    assert 0.0 <= float(model_summary.get("compute_availability", 0.0)) <= 1.0
    assert float(model_summary.get("availability_scale", 0.0)) >= 0.72
    assert model_summary.get("decompression_hint") is True


def test_probabilistic_builder_emits_anti_clump_summary() -> None:
    _, model_summary = build_probabilistic_daimoi_particles(
        file_graph={"file_nodes": []},
        presence_impacts=[
            {
                "id": "witness_thread",
                "affected_by": {"files": 0.5, "clicks": 0.3, "resource": 0.2},
                "affects": {"world": 0.7, "ledger": 0.6},
            },
            {
                "id": "anchor_registry",
                "affected_by": {"files": 0.4, "clicks": 0.4, "resource": 0.3},
                "affects": {"world": 0.6, "ledger": 0.5},
            },
        ],
        resource_heartbeat={
            "devices": {
                "cpu": {"utilization": 18.0},
                "gpu1": {"utilization": 12.0},
                "gpu2": {"utilization": 11.0},
                "npu0": {"utilization": 9.0},
            }
        },
        compute_jobs=[],
        queue_ratio=0.1,
        now=1_700_000_150.0,
    )

    anti_clump = model_summary.get("anti_clump", {})
    assert isinstance(anti_clump, dict)
    assert 0.0 <= float(model_summary.get("clump_score", 0.0)) <= 1.0
    assert -1.0 <= float(model_summary.get("anti_clump_drive", 0.0)) <= 1.0
    assert 0.0 <= float(anti_clump.get("target", 0.0)) <= 1.0
    assert 0.0 <= float(anti_clump.get("clump_score", 0.0)) <= 1.0
    assert -1.0 <= float(anti_clump.get("drive", 0.0)) <= 1.0

    metrics = anti_clump.get("metrics", {})
    assert isinstance(metrics, dict)
    assert 0.0 <= float(metrics.get("nn_term", 0.0)) <= 1.0
    assert 0.0 <= float(metrics.get("entropy_norm", 0.0)) <= 1.0
    assert 0.0 <= float(metrics.get("hotspot_term", 0.0)) <= 1.0
    assert 0.0 <= float(metrics.get("collision_term", 0.0)) <= 1.0

    scales = anti_clump.get("scales", {})
    assert isinstance(scales, dict)
    assert 0.34 <= float(scales.get("semantic", 0.0)) <= 1.21
    assert 0.39 <= float(scales.get("edge", 0.0)) <= 1.11
    assert 0.44 <= float(scales.get("anchor", 0.0)) <= 1.11
    assert 0.49 <= float(scales.get("spawn", 0.0)) <= 1.06
    assert 0.79 <= float(scales.get("tangent", 0.0)) <= 1.81


def test_anti_clump_metrics_score_cluster_higher_than_spread() -> None:
    clustered: list[tuple[float, float]] = []
    for index in range(64):
        angle = (float(index) / 64.0) * math.tau
        radius = 0.01 + ((index % 5) * 0.0015)
        clustered.append(
            (
                0.5 + (math.cos(angle) * radius),
                0.5 + (math.sin(angle) * radius),
            )
        )

    spread: list[tuple[float, float]] = []
    for row in range(8):
        for col in range(8):
            spread.append((0.08 + (col * 0.11), 0.08 + (row * 0.11)))

    clustered_metrics = daimoi_module._anti_clump_metrics(
        clustered,
        previous_collision_count=0,
    )
    spread_metrics = daimoi_module._anti_clump_metrics(
        spread,
        previous_collision_count=0,
    )

    assert float(clustered_metrics.get("clump_score", 0.0)) > float(
        spread_metrics.get("clump_score", 0.0)
    )


def test_absorb_sampler_emits_component_logits_and_gumbel_scores() -> None:
    components = daimoi_module._packet_components_from_job_probabilities(
        {
            "deliver_message": 0.58,
            "invoke_truth_gate": 0.22,
            "invoke_graph_crawl": 0.2,
        }
    )

    sample = daimoi_module._sample_absorb_component(
        components=components,
        lens_embedding=daimoi_module._embedding_from_text("truth witness gate"),
        need_by_resource={
            "cpu": 0.8,
            "gpu": 0.25,
            "npu": 0.18,
            "ram": 0.54,
            "disk": 0.44,
            "network": 0.73,
        },
        context={
            "pressure": 0.37,
            "congestion": 0.28,
            "wallet_pressure": 0.48,
            "message_entropy": 0.41,
            "queue": 0.22,
            "contact": 0.66,
        },
        seed="unit-absorb-sampler",
    )

    assert sample.get("record") == "eta-mu.daimoi-absorb-sampler.v1"
    assert sample.get("schema_version") == "daimoi.absorb-sampler.v1"
    assert sample.get("method") == "gumbel-max"
    assert 0.0 <= float(sample.get("beta", 0.0))
    assert float(sample.get("temperature", 0.0)) > 0.0

    component_rows = sample.get("components", [])
    assert isinstance(component_rows, list)
    assert component_rows
    prob_total = sum(float(row.get("probability", 0.0)) for row in component_rows)
    assert math.isclose(prob_total, 1.0, rel_tol=0.0, abs_tol=1e-6)
    selected_id = str(sample.get("selected_component_id", ""))
    assert selected_id in {str(row.get("component_id", "")) for row in component_rows}
    first = component_rows[0]
    assert "q_i" in first
    assert "s_i" in first
    assert "logit" in first
    assert "gumbel_score" in first


def test_world_edge_pressure_pushes_probabilistic_daimoi_inward() -> None:
    left_push = daimoi_module._world_edge_inward_pressure(
        0.01,
        edge_band=daimoi_module.DAIMOI_WORLD_EDGE_BAND,
        pressure=daimoi_module.DAIMOI_WORLD_EDGE_PRESSURE,
    )
    center_push = daimoi_module._world_edge_inward_pressure(
        0.5,
        edge_band=daimoi_module.DAIMOI_WORLD_EDGE_BAND,
        pressure=daimoi_module.DAIMOI_WORLD_EDGE_PRESSURE,
    )
    right_push = daimoi_module._world_edge_inward_pressure(
        0.99,
        edge_band=daimoi_module.DAIMOI_WORLD_EDGE_BAND,
        pressure=daimoi_module.DAIMOI_WORLD_EDGE_PRESSURE,
    )

    assert left_push > 0.0
    assert center_push == pytest.approx(0.0, abs=1e-12)
    assert right_push < 0.0


def test_world_edge_reflection_deflects_probabilistic_daimoi_from_walls() -> None:
    left_pos, left_v = daimoi_module._reflect_world_axis(
        -0.014,
        -0.02,
        bounce=daimoi_module.DAIMOI_WORLD_EDGE_BOUNCE,
    )
    right_pos, right_v = daimoi_module._reflect_world_axis(
        1.014,
        0.02,
        bounce=daimoi_module.DAIMOI_WORLD_EDGE_BOUNCE,
    )

    assert 0.0 <= left_pos <= 1.0
    assert 0.0 <= right_pos <= 1.0
    assert left_v > 0.0
    assert right_v < 0.0


def test_semantic_pair_force_changes_sign_with_similarity() -> None:
    target = [1.0, 0.0] + ([0.0] * (daimoi_module.DAIMOI_EMBED_DIMS - 2))
    aligned = [1.0, 0.0] + ([0.0] * (daimoi_module.DAIMOI_EMBED_DIMS - 2))
    opposite = [-1.0, 0.0] + ([0.0] * (daimoi_module.DAIMOI_EMBED_DIMS - 2))

    attract_fx, attract_fy = daimoi_module._semantic_pair_force(
        target_unit=target,
        dx=0.04,
        dy=0.0,
        distance_sq=0.04 * 0.04,
        source_vector=aligned,
        source_weight=1.0,
        strength_base=daimoi_module.DAIMOI_SEMANTIC_PARTICLE_STRENGTH,
    )
    repel_fx, repel_fy = daimoi_module._semantic_pair_force(
        target_unit=target,
        dx=0.04,
        dy=0.0,
        distance_sq=0.04 * 0.04,
        source_vector=opposite,
        source_weight=1.0,
        strength_base=daimoi_module.DAIMOI_SEMANTIC_PARTICLE_STRENGTH,
    )

    assert attract_fx > 0.0
    assert math.isclose(attract_fy, 0.0, abs_tol=1e-12)
    assert repel_fx < 0.0
    assert math.isclose(repel_fy, 0.0, abs_tol=1e-12)


def test_barnes_hut_semantic_force_matches_far_cluster_expectation() -> None:
    semantic_items = [
        {
            "id": "near",
            "x": 0.54,
            "y": 0.5,
            "weight": 1.0,
            "vector": [1.0] + ([0.0] * (daimoi_module.DAIMOI_EMBED_DIMS - 1)),
        },
        {
            "id": "far-a",
            "x": 0.86,
            "y": 0.78,
            "weight": 1.0,
            "vector": [1.0] + ([0.0] * (daimoi_module.DAIMOI_EMBED_DIMS - 1)),
        },
        {
            "id": "far-b",
            "x": 0.9,
            "y": 0.82,
            "weight": 1.0,
            "vector": [1.0] + ([0.0] * (daimoi_module.DAIMOI_EMBED_DIMS - 1)),
        },
    ]
    semantic_tree = daimoi_module._quadtree_build(
        semantic_items,
        bounds=(0.0, 0.0, 1.0, 1.0),
        max_items=1,
        max_depth=8,
    )
    daimoi_module._quadtree_semantic_aggregate(semantic_tree)

    target_unit = [1.0] + ([0.0] * (daimoi_module.DAIMOI_EMBED_DIMS - 1))
    bh_fx, bh_fy = daimoi_module._barnes_hut_semantic_force(
        node=semantic_tree,
        target_id="target",
        target_x=0.5,
        target_y=0.5,
        target_unit=target_unit,
        near_radius=0.08,
        theta=0.75,
        strength_base=daimoi_module.DAIMOI_SEMANTIC_PARTICLE_STRENGTH,
        exclude_ids={"near"},
    )

    expected_fx = 0.0
    expected_fy = 0.0
    for source in (semantic_items[1], semantic_items[2]):
        dx = float(source["x"]) - 0.5
        dy = float(source["y"]) - 0.5
        direct_fx, direct_fy = daimoi_module._semantic_pair_force(
            target_unit=target_unit,
            dx=dx,
            dy=dy,
            distance_sq=(dx * dx) + (dy * dy),
            source_vector=list(source["vector"]),
            source_weight=1.0,
            strength_base=daimoi_module.DAIMOI_SEMANTIC_PARTICLE_STRENGTH,
        )
        expected_fx += direct_fx
        expected_fy += direct_fy

    assert bh_fx > 0.0
    assert bh_fy > 0.0
    assert math.isclose(bh_fx, expected_fx, rel_tol=0.2, abs_tol=1e-9)
    assert math.isclose(bh_fy, expected_fy, rel_tol=0.2, abs_tol=1e-9)


def test_pain_field_daimoi_motion_deflects_off_world_edges() -> None:
    with simulation_module._DAIMO_DYNAMICS_LOCK:
        simulation_module._DAIMO_DYNAMICS_CACHE["entities"] = {}

    pain_field = {
        "node_heat": [
            {
                "node_id": "edge-daimo",
                "x": 0.002,
                "y": 0.004,
                "heat": 0.25,
            }
        ]
    }
    daimoi_state = {
        "relations": {
            "霊/push": [
                {
                    "entity_id": "edge-daimo",
                    "fx": -6.0,
                    "fy": -6.0,
                }
            ]
        },
        "physics": {
            "dt": 0.2,
            "damping": 0.88,
        },
    }

    updated = simulation_module._apply_daimoi_dynamics_to_pain_field(
        pain_field,
        daimoi_state,
    )
    rows = updated.get("node_heat", [])
    assert rows
    row = rows[0]
    assert float(row.get("x", 0.0)) > 0.0
    assert float(row.get("y", 0.0)) > 0.0
    assert float(row.get("vx", 0.0)) > 0.0
    assert float(row.get("vy", 0.0)) > 0.0
    motion = updated.get("motion", {})
    assert float(motion.get("edge_pressure", 0.0)) > 0.0


def test_simulation_payload_exposes_probabilistic_daimoi_summary() -> None:
    simulation = build_simulation_state(
        {
            "items": [],
            "counts": {"audio": 3, "image": 1, "video": 0},
            "file_graph": {
                "file_nodes": [
                    {
                        "id": "file:witness-1",
                        "name": "witness_trace.md",
                        "summary": "witness continuity lineage",
                        "dominant_field": "f2",
                        "x": 0.64,
                        "y": 0.31,
                        "importance": 0.9,
                        "embed_layer_count": 1,
                        "vecstore_collection": "eta_mu_nexus_v1",
                    }
                ]
            },
        },
        influence_snapshot={
            "clicks_45s": 2,
            "file_changes_120s": 4,
            "recent_click_targets": ["witness_thread"],
            "recent_file_paths": ["receipts.log"],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    dynamics = simulation.get("presence_dynamics", {})
    assert dynamics.get("daimoi_probabilistic_record") == "ημ.daimoi-probabilistic.v1"

    summary = dynamics.get("daimoi_probabilistic", {})
    assert isinstance(summary, dict)
    assert summary.get("schema_version") == "daimoi.probabilistic.v1"
    assert int(summary.get("active", 0)) >= 1
    assert "collisions" in summary
    assert "deflects" in summary
    assert "diffuses" in summary

    rows = dynamics.get("field_particles", [])
    assert isinstance(rows, list)
    assert rows
    first = rows[0]
    assert "job_probabilities" in first
    assert "packet_components" in first
    assert "resource_signature" in first
    assert "absorb_sampler" in first
    assert "action_probabilities" in first
    assert "vx" in first
    assert "vy" in first

    packet_contract = summary.get("packet_contract", {})
    assert packet_contract.get("record") == "eta-mu.daimoi-packet-components.v1"
    absorb_sampler = summary.get("absorb_sampler", {})
    assert absorb_sampler.get("record") == "eta-mu.daimoi-absorb-sampler.v1"


def test_probabilistic_builder_emits_passive_nexus_rows() -> None:
    particles, _ = build_probabilistic_daimoi_particles(
        file_graph={
            "file_nodes": [
                {
                    "id": "file:witness-1",
                    "name": "witness_trace.md",
                    "summary": "witness continuity lineage",
                    "dominant_field": "f2",
                    "dominant_presence": "witness_thread",
                    "x": 0.64,
                    "y": 0.31,
                    "importance": 0.9,
                }
            ]
        },
        presence_impacts=[
            {
                "id": "witness_thread",
                "affected_by": {"files": 0.4, "clicks": 0.6, "resource": 0.2},
                "affects": {"world": 0.7, "ledger": 0.5},
            },
            {
                "id": "anchor_registry",
                "affected_by": {"files": 0.2, "clicks": 0.3, "resource": 0.2},
                "affects": {"world": 0.4, "ledger": 0.6},
            },
        ],
        resource_heartbeat={"devices": {"cpu": {"utilization": 24.0}}},
        compute_jobs=[],
        queue_ratio=0.1,
        now=1_700_000_100.0,
    )

    nexus_rows = [row for row in particles if bool(row.get("is_nexus", False))]
    assert nexus_rows
    nexus = nexus_rows[0]
    assert str(nexus.get("particle_mode", "")) == "static-daimoi"
    assert str(nexus.get("presence_role", "")) == "nexus-passive"
    assert float(nexus.get("message_probability", 1.0)) == 0.0
    assert str(nexus.get("source_node_id", "")).strip()
    nexus_actions = nexus.get("action_probabilities", {})
    assert math.isclose(
        float(nexus_actions.get("deflect", 0.0)),
        0.92,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(nexus_actions.get("diffuse", 0.0)),
        0.08,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_build_nexus_adjacency_falls_back_to_nearest_neighbors() -> None:
    file_nodes = [
        {"id": "node:a", "x": 0.1, "y": 0.1},
        {"id": "node:b", "x": 0.2, "y": 0.1},
        {"id": "node:c", "x": 0.8, "y": 0.8},
    ]
    adjacency = daimoi_module._build_nexus_adjacency(
        file_nodes=file_nodes,
        edge_rows=[],
        fallback_neighbors=1,
    )

    assert adjacency["node:a"]
    assert adjacency["node:b"]
    assert adjacency["node:c"]
    assert "node:b" in adjacency["node:a"]
    assert "node:a" in adjacency["node:b"]


def test_select_downhill_adjacent_nexus_prefers_lower_balance_neighbor() -> None:
    selected, probability = daimoi_module._select_downhill_adjacent_nexus(
        current_node_id="node:root",
        adjacency_by_node={"node:root": ["node:left", "node:right"]},
        node_balance_by_node={
            "node:root": 1.2,
            "node:left": 0.9,
            "node:right": 0.35,
        },
        node_position_by_node={
            "node:root": (0.5, 0.5),
            "node:left": (0.4, 0.5),
            "node:right": (0.65, 0.5),
        },
        target_xy=(0.8, 0.5),
        semantic_vector=daimoi_module._embedding_from_text("database index lineage"),
        node_vector_by_id={
            "node:left": daimoi_module._embedding_from_text("archive file path"),
            "node:right": daimoi_module._embedding_from_text("database index lineage"),
        },
        previous_node_id="",
    )

    assert selected == "node:right"
    assert 0.0 < probability <= 1.0


def test_probabilistic_builder_emits_graph_route_fields_for_active_particles() -> None:
    particles, _ = build_probabilistic_daimoi_particles(
        file_graph={
            "file_nodes": [
                {
                    "id": "file:route-a",
                    "name": "route_a.md",
                    "summary": "semantic root a",
                    "dominant_field": "f2",
                    "dominant_presence": "witness_thread",
                    "x": 0.28,
                    "y": 0.32,
                    "importance": 0.8,
                },
                {
                    "id": "file:route-b",
                    "name": "route_b.md",
                    "summary": "semantic root b",
                    "dominant_field": "f2",
                    "dominant_presence": "anchor_registry",
                    "x": 0.68,
                    "y": 0.64,
                    "importance": 0.72,
                },
            ],
            "edges": [
                {
                    "id": "edge:route",
                    "source": "file:route-a",
                    "target": "file:route-b",
                    "kind": "relates",
                    "weight": 0.8,
                }
            ],
        },
        presence_impacts=[
            {
                "id": "witness_thread",
                "affected_by": {"files": 0.5, "clicks": 0.4, "resource": 0.2},
                "affects": {"world": 0.7, "ledger": 0.5},
            },
            {
                "id": "anchor_registry",
                "affected_by": {"files": 0.4, "clicks": 0.5, "resource": 0.3},
                "affects": {"world": 0.6, "ledger": 0.6},
            },
        ],
        resource_heartbeat={"devices": {"cpu": {"utilization": 21.0}}},
        compute_jobs=[],
        queue_ratio=0.1,
        now=1_700_001_000.0,
    )

    active = [row for row in particles if not bool(row.get("is_nexus", False))]
    assert active
    first = active[0]
    assert str(first.get("graph_node_id", "")).strip()
    assert "route_probability" in first


def test_simulation_resource_cores_emit_resource_daimoi_to_subsim_wallets(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIM_PARTICLE_BACKEND", "python")
    monkeypatch.setattr(
        simulation_module,
        "_resource_monitor_snapshot",
        lambda: {
            "devices": {
                "cpu": {"utilization": 12.0},
                "gpu1": {"utilization": 9.0},
                "gpu2": {"utilization": 11.0},
                "npu0": {"utilization": 8.0},
            },
            "resource_monitor": {
                "cpu_percent": 12.0,
                "memory_percent": 28.0,
                "disk_percent": 31.0,
                "network_percent": 22.0,
            },
        },
    )

    # Mock PresenceRuntimeManager to provide enough CPU resource for pressure > 0.15
    from code.world_web.presence_runtime import get_presence_runtime_manager

    manager = get_presence_runtime_manager()
    manager.reset()
    state = manager.get_state("presence.core.cpu")
    wallet = state.setdefault("resource_wallet", {})
    wallet["cpu"] = 32.0  # High enough pressure

    simulation = build_simulation_state(
        {
            "items": [],
            "counts": {"audio": 1, "image": 0, "video": 0},
            "file_graph": {"file_nodes": []},
        },
        influence_snapshot={
            "clicks_45s": 1,
            "file_changes_120s": 2,
            "recent_click_targets": ["witness_thread"],
            "recent_file_paths": ["receipts.log"],
        },
        queue_snapshot={"pending_count": 1, "event_count": 2},
        docker_snapshot={
            "simulations": [
                {
                    "id": "a" * 64,
                    "name": "sim-alpha",
                    "resources": {
                        "usage": {
                            "cpu_percent": 18.0,
                            "memory_usage_bytes": 128 * 1024 * 1024,
                        },
                        "limits": {
                            "nano_cpus": 900_000_000,
                            "memory_limit_bytes": 2 * 1024 * 1024 * 1024,
                        },
                    },
                }
            ]
        },
    )

    dynamics = simulation.get("presence_dynamics", {})
    resource_daimoi = dynamics.get("resource_daimoi", {})
    assert resource_daimoi.get("record") == "eta-mu.resource-daimoi-flow.v1"
    assert int(resource_daimoi.get("emitter_rows", 0)) >= 1
    assert int(resource_daimoi.get("delivered_packets", 0)) >= 1
    assert float(resource_daimoi.get("total_transfer", 0.0)) > 0.0

    field_particles = dynamics.get("field_particles", [])
    core_emitters = [
        row
        for row in field_particles
        if str(row.get("presence_id", "")).startswith("presence.core.")
        and bool(row.get("resource_daimoi", False))
    ]
    assert core_emitters
    assert all(
        str(row.get("top_job", "")) == "emit_resource_packet" for row in core_emitters
    )

    impacts = dynamics.get("presence_impacts", [])
    subsim = next(
        (
            row
            for row in impacts
            if str(row.get("id", "")).strip() == "presence.sim.sim-alpha"
        ),
        None,
    )
    assert isinstance(subsim, dict)
    wallet = subsim.get("resource_wallet", {}) if isinstance(subsim, dict) else {}
    assert isinstance(wallet, dict)
    recipient_credit = 0.0
    for row in resource_daimoi.get("recipients", []):
        if str(row.get("presence_id", "")).strip() != "presence.sim.sim-alpha":
            continue
        recipient_credit = float(row.get("credited", 0.0))
        break
    assert (
        any(float(value) > 0.0 for value in wallet.values()) or recipient_credit > 0.0
    )


def test_simulation_core_emitters_use_coupled_wallets(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIM_PARTICLE_BACKEND", "python")
    monkeypatch.setenv("SIMULATION_CORE_RESOURCES", "cpu,ram")
    monkeypatch.setenv("SIMULATION_RESET_DAIMOI_ON_BOOT", "1")
    monkeypatch.setattr(
        simulation_module,
        "_resource_monitor_snapshot",
        lambda: {
            "devices": {
                "cpu": {"utilization": 11.0},
                "gpu1": {"utilization": 8.0},
                "gpu2": {"utilization": 9.0},
                "npu0": {"utilization": 7.0},
            },
            "resource_monitor": {
                "cpu_percent": 11.0,
                "memory_percent": 24.0,
                "disk_percent": 21.0,
                "network_percent": 14.0,
            },
        },
    )

    # Mock PresenceRuntimeManager to provide enough CPU resource for pressure > 0.15
    from code.world_web.presence_runtime import get_presence_runtime_manager

    manager = get_presence_runtime_manager()
    manager.reset()
    state = manager.get_state("presence.core.cpu")
    wallet = state.setdefault("resource_wallet", {})
    wallet["cpu"] = 32.0  # High enough pressure

    simulation = build_simulation_state(
        {
            "items": [],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": {"file_nodes": []},
        },
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_click_targets": [],
            "recent_file_paths": [],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    particles = (
        simulation.get("presence_dynamics", {}).get("field_particles", [])
        if isinstance(simulation, dict)
        else []
    )
    assert isinstance(particles, list)

    cpu_emitters = [
        row
        for row in particles
        if str(row.get("presence_id", "")).strip() == "presence.core.cpu"
        and bool(row.get("resource_daimoi", False))
    ]
    assert cpu_emitters

    ram_rows = [
        row
        for row in particles
        if str(row.get("presence_id", "")).strip() == "presence.core.ram"
    ]
    assert ram_rows
    assert any(bool(row.get("resource_daimoi", False)) for row in ram_rows)
    assert any(str(row.get("resource_type", "")).strip() == "ram" for row in ram_rows)

    non_core_rows = [
        row
        for row in particles
        if not str(row.get("presence_id", "")).startswith("presence.core.")
    ]
    assert non_core_rows
    consume_rows = [row for row in non_core_rows if "resource_consume_type" in row]
    assert consume_rows
    assert all(
        str(row.get("resource_consume_type", "")).strip() == "cpu"
        for row in consume_rows
    )


def test_simulation_minimal_presence_profile_uses_concept_and_cpu_presences(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIM_PARTICLE_BACKEND", "python")
    monkeypatch.setenv("SIMULATION_PRESENCE_PROFILE", "concept_cpu")
    monkeypatch.setenv("SIMULATION_CORE_RESOURCES", "cpu")
    monkeypatch.setenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "75")
    monkeypatch.setattr(
        simulation_module,
        "_resource_monitor_snapshot",
        lambda: {
            "devices": {
                "cpu": {"utilization": 18.0},
                "gpu1": {"utilization": 21.0},
                "gpu2": {"utilization": 19.0},
                "npu0": {"utilization": 17.0},
            },
            "resource_monitor": {
                "cpu_percent": 18.0,
                "memory_percent": 30.0,
                "disk_percent": 24.0,
                "network_percent": 16.0,
            },
        },
    )

    simulation = build_simulation_state(
        {
            "items": [],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": {"file_nodes": []},
        },
        influence_snapshot={
            "clicks_45s": 1,
            "file_changes_120s": 1,
            "recent_click_targets": ["witness_thread"],
            "recent_file_paths": ["receipts.log"],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    dynamics = simulation.get("presence_dynamics", {})
    impacts = dynamics.get("presence_impacts", [])
    impact_ids = {
        str(row.get("id", "")).strip()
        for row in impacts
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }

    assert "receipt_river" in impact_ids
    assert "witness_thread" in impact_ids
    assert "anchor_registry" in impact_ids
    assert "gates_of_truth" in impact_ids
    assert "health_sentinel_cpu" in impact_ids
    assert "presence.core.cpu" in impact_ids
    assert "health_sentinel_gpu1" not in impact_ids
    assert "health_sentinel_npu0" not in impact_ids

    policy = dynamics.get("emission_policy", {})
    assert policy.get("presence_profile") == "concept_cpu"
    assert policy.get("cpu_core_emitter_enabled") is True
    assert policy.get("core_resource_emitters") == ["cpu"]


def test_simulation_cpu_daimoi_gate_disables_cpu_core_emitter_at_threshold(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIM_PARTICLE_BACKEND", "python")
    monkeypatch.setenv("SIMULATION_PRESENCE_PROFILE", "concept_cpu")
    monkeypatch.setenv("SIMULATION_CORE_RESOURCES", "cpu")
    monkeypatch.setenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "75")
    monkeypatch.setattr(
        simulation_module,
        "_resource_monitor_snapshot",
        lambda: {
            "devices": {
                "cpu": {"utilization": 88.0},
                "gpu1": {"utilization": 24.0},
                "gpu2": {"utilization": 18.0},
                "npu0": {"utilization": 22.0},
            },
            "resource_monitor": {
                "cpu_percent": 88.0,
                "memory_percent": 42.0,
                "disk_percent": 36.0,
                "network_percent": 20.0,
            },
        },
    )

    simulation = build_simulation_state(
        {
            "items": [],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": {"file_nodes": []},
        },
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_click_targets": [],
            "recent_file_paths": [],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    dynamics = simulation.get("presence_dynamics", {})
    impacts = dynamics.get("presence_impacts", [])
    impact_ids = {
        str(row.get("id", "")).strip()
        for row in impacts
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }
    assert "presence.core.cpu" not in impact_ids

    core_emitters = [
        row
        for row in dynamics.get("field_particles", [])
        if str(row.get("presence_id", "")) == "presence.core.cpu"
    ]
    assert core_emitters == []

    policy = dynamics.get("emission_policy", {})
    assert policy.get("cpu_core_emitter_enabled") is False
    assert policy.get("cpu_daimoi_stop_percent") == 75.0


def test_simulation_cpu_sentinel_idles_below_burn_threshold(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIM_PARTICLE_BACKEND", "python")
    monkeypatch.setenv("SIMULATION_PRESENCE_PROFILE", "concept_cpu")
    monkeypatch.setenv("SIMULATION_CORE_RESOURCES", "cpu")
    monkeypatch.setenv("SIMULATION_CPU_SENTINEL_BURN_START_PERCENT", "90")
    monkeypatch.setenv("SIMULATION_RESET_DAIMOI_ON_BOOT", "0")
    monkeypatch.setattr(
        simulation_module,
        "_resource_monitor_snapshot",
        lambda: {
            "devices": {
                "cpu": {"utilization": 82.0},
                "gpu1": {"utilization": 18.0},
                "gpu2": {"utilization": 16.0},
                "npu0": {"utilization": 20.0},
            },
            "resource_monitor": {
                "cpu_percent": 82.0,
                "memory_percent": 52.0,
                "disk_percent": 34.0,
                "network_percent": 27.0,
            },
        },
    )

    from code.world_web.presence_runtime import get_presence_runtime_manager

    manager = get_presence_runtime_manager()
    manager.reset()
    sentinel_state = manager.get_state("health_sentinel_cpu")
    sentinel_wallet = sentinel_state.setdefault("resource_wallet", {})
    sentinel_wallet["cpu"] = 16.0

    simulation = build_simulation_state(
        {
            "items": [],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": {"file_nodes": []},
        },
        influence_snapshot={
            "clicks_45s": 1,
            "file_changes_120s": 1,
            "recent_click_targets": ["witness_thread"],
            "recent_file_paths": ["receipts.log"],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    dynamics = simulation.get("presence_dynamics", {})
    field_particles = dynamics.get("field_particles", [])
    sentinel_rows = [
        row
        for row in field_particles
        if str(row.get("presence_id", "")).strip() == "health_sentinel_cpu"
    ]
    assert sentinel_rows
    assert all("resource_consume_type" not in row for row in sentinel_rows)
    assert any(bool(row.get("resource_sentinel_idle", False)) for row in sentinel_rows)

    resource_consumption = dynamics.get("resource_consumption", {})
    assert resource_consumption.get("cpu_sentinel_burn_active") is False
    active_presence_ids = {
        str(row.get("presence_id", "")).strip()
        for row in resource_consumption.get("active_presences", [])
        if isinstance(row, dict)
    }
    assert "health_sentinel_cpu" not in active_presence_ids


def test_simulation_cpu_sentinel_burns_wallet_above_threshold(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIM_PARTICLE_BACKEND", "python")
    monkeypatch.setenv("SIMULATION_PRESENCE_PROFILE", "concept_cpu")
    monkeypatch.setenv("SIMULATION_CORE_RESOURCES", "cpu")
    monkeypatch.setenv("SIMULATION_CPU_SENTINEL_BURN_START_PERCENT", "90")
    monkeypatch.setenv("SIMULATION_RESET_DAIMOI_ON_BOOT", "0")
    monkeypatch.setattr(
        simulation_module,
        "_resource_monitor_snapshot",
        lambda: {
            "devices": {
                "cpu": {"utilization": 96.0},
                "gpu1": {"utilization": 31.0},
                "gpu2": {"utilization": 24.0},
                "npu0": {"utilization": 29.0},
            },
            "resource_monitor": {
                "cpu_percent": 96.0,
                "memory_percent": 63.0,
                "disk_percent": 39.0,
                "network_percent": 31.0,
            },
        },
    )

    from code.world_web.presence_runtime import get_presence_runtime_manager

    manager = get_presence_runtime_manager()
    manager.reset()
    sentinel_state = manager.get_state("health_sentinel_cpu")
    sentinel_wallet = sentinel_state.setdefault("resource_wallet", {})
    sentinel_wallet["cpu"] = 24.0

    simulation = build_simulation_state(
        {
            "items": [],
            "counts": {"audio": 0, "image": 0, "video": 0},
            "file_graph": {"file_nodes": []},
        },
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_click_targets": [],
            "recent_file_paths": [],
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )

    dynamics = simulation.get("presence_dynamics", {})
    field_particles = dynamics.get("field_particles", [])
    sentinel_rows = [
        row
        for row in field_particles
        if str(row.get("presence_id", "")).strip() == "health_sentinel_cpu"
        and str(row.get("resource_consume_type", "")).strip() == "cpu"
    ]
    assert sentinel_rows
    assert any(
        float(row.get("resource_action_cost", 0.0) or 0.0) > 0.0
        for row in sentinel_rows
    )
    assert any(
        float(row.get("resource_sentinel_burn_intensity", 0.0) or 0.0) > 0.0
        for row in sentinel_rows
    )
    assert all(
        not bool(row.get("resource_sentinel_idle", False)) for row in sentinel_rows
    )
    assert any(
        str(row.get("top_job", "")).strip()
        in {"burn_resource_packet", "resource_starved"}
        for row in sentinel_rows
    )

    resource_consumption = dynamics.get("resource_consumption", {})
    assert resource_consumption.get("cpu_sentinel_burn_active") is True
    active_ids = {
        str(row.get("presence_id", "")).strip()
        for row in resource_consumption.get("active_presences", [])
        if isinstance(row, dict)
    }
    starved_ids = {
        str(row.get("presence_id", "")).strip()
        for row in resource_consumption.get("starved_presences", [])
        if isinstance(row, dict)
    }
    assert "health_sentinel_cpu" in (active_ids | starved_ids)


def test_simulation_cpu_sentinel_forces_cpu_resource_targets_when_hot(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "99")
    monkeypatch.setattr(
        simulation_module,
        "_RESOURCE_DAIMOI_CPU_SENTINEL_ATTRACTOR_START_PERCENT",
        90.0,
    )
    monkeypatch.setattr(
        simulation_module,
        "_RESOURCE_DAIMOI_CPU_SENTINEL_ATTRACTOR_ALL_DAIMOI",
        True,
    )
    monkeypatch.setitem(simulation_module._RESOURCE_DAIMOI_WALLET_CAP, "cpu", 1.0)

    field_particles = [
        {
            "id": "particle:core-cpu:01",
            "presence_id": "presence.core.cpu",
            "is_nexus": False,
            "x": 0.45,
            "y": 0.48,
            "influence_power": 0.8,
            "message_probability": 0.72,
            "route_probability": 0.6,
            "drift_score": 0.1,
            "gravity_potential": 0.2,
            "local_price": 1.0,
        }
    ]
    presence_impacts = [
        {
            "id": "presence.core.cpu",
            "presence_type": "core",
            "x": 0.45,
            "y": 0.48,
            "affected_by": {"resource": 0.6},
            "resource_wallet": {"cpu": 1.0},
        },
        {
            "id": "health_sentinel_cpu",
            "presence_type": "presence",
            "x": 0.52,
            "y": 0.52,
            "affected_by": {"resource": 0.2},
            "resource_wallet": {"cpu": 0.2},
        },
    ]

    summary = simulation_module._apply_resource_daimoi_emissions(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat={"devices": {"cpu": {"utilization": 96.0}}},
        queue_ratio=0.0,
    )

    assert summary.get("cpu_sentinel_attractor_active") is True
    assert int(float(summary.get("cpu_sentinel_forced_packets", 0) or 0.0)) > 0

    assert bool(field_particles[0].get("resource_daimoi", False)) is True
    assert str(field_particles[0].get("resource_type", "")).strip() == "cpu"
    assert (
        str(field_particles[0].get("resource_target_presence_id", "")).strip()
        == "health_sentinel_cpu"
    )
    assert bool(field_particles[0].get("cpu_sentinel_attractor_active", False)) is True
    assert (
        str(field_particles[0].get("resource_forced_target", "")).strip()
        == "cpu_sentinel_attractor"
    )


def test_simulation_global_cpu_cutoff_disables_all_core_emitters() -> None:
    field_particles = [
        {
            "id": "particle:core-cpu:cutoff",
            "presence_id": "presence.core.cpu",
            "is_nexus": False,
            "x": 0.41,
            "y": 0.47,
            "influence_power": 0.7,
            "message_probability": 0.66,
            "route_probability": 0.53,
            "drift_score": 0.08,
            "gravity_potential": 0.2,
        },
        {
            "id": "particle:core-ram:cutoff",
            "presence_id": "presence.core.ram",
            "is_nexus": False,
            "x": 0.58,
            "y": 0.52,
            "influence_power": 0.69,
            "message_probability": 0.61,
            "route_probability": 0.49,
            "drift_score": 0.07,
            "gravity_potential": 0.21,
        },
    ]
    presence_impacts = [
        {
            "id": "presence.core.cpu",
            "presence_type": "core",
            "resource_wallet": {"cpu": 2.0},
        },
        {
            "id": "presence.core.ram",
            "presence_type": "core",
            "resource_wallet": {"ram": 2.0, "cpu": 1.0},
        },
        {
            "id": "health_sentinel_cpu",
            "presence_type": "presence",
            "resource_wallet": {"cpu": 0.0},
        },
    ]

    summary = simulation_module._apply_resource_daimoi_emissions(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat={"devices": {"cpu": {"utilization": 96.0}}},
        queue_ratio=0.0,
    )

    assert summary.get("cpu_emitter_cutoff_active") is True
    assert int(float(summary.get("delivered_packets", 0) or 0.0)) == 0
    assert (
        str(field_particles[0].get("resource_emit_disabled_reason", "")).strip()
        == "global_cpu_cutoff"
    )
    assert (
        str(field_particles[1].get("resource_emit_disabled_reason", "")).strip()
        == "global_cpu_cutoff"
    )


def test_resource_emission_credits_wallet_denoms() -> None:
    field_particles = [
        {
            "id": "particle:core-cpu:wallet-credit",
            "presence_id": "presence.core.cpu",
            "is_nexus": False,
            "x": 0.42,
            "y": 0.46,
            "influence_power": 0.71,
            "message_probability": 0.63,
            "route_probability": 0.57,
            "drift_score": 0.09,
            "gravity_potential": 0.2,
        }
    ]
    presence_impacts = [
        {
            "id": "presence.core.cpu",
            "presence_type": "core",
            "x": 0.42,
            "y": 0.46,
            "resource_wallet": {"cpu": 3.0},
        },
        {
            "id": "health_sentinel_cpu",
            "presence_type": "presence",
            "x": 0.5,
            "y": 0.5,
            "resource_wallet": {"cpu": 0.0},
            "resource_wallet_denoms": [],
        },
    ]

    summary = simulation_module._apply_resource_daimoi_emissions(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat={"devices": {"cpu": {"utilization": 22.0}}},
        queue_ratio=0.0,
    )

    assert int(float(summary.get("delivered_packets", 0) or 0.0)) > 0
    target = next(
        (
            row
            for row in presence_impacts
            if str(row.get("id", "")).strip() == "health_sentinel_cpu"
        ),
        {},
    )
    wallet = target.get("resource_wallet", {})
    assert isinstance(wallet, dict)
    assert any(float(value or 0.0) > 0.0 for value in wallet.values())
    denoms = target.get("resource_wallet_denoms", [])
    assert isinstance(denoms, list)
    assert denoms
    assert bool(field_particles[0].get("resource_emit_wallet_credit", False)) is True


def test_resource_action_consumption_supports_coupled_wallet_affordability() -> None:
    field_particles = [
        {
            "id": "particle:witness:coupled",
            "presence_id": "witness_thread",
            "is_nexus": False,
            "message_probability": 0.62,
            "route_probability": 0.48,
            "influence_power": 0.57,
            "drift_score": 0.12,
        }
    ]
    presence_impacts = [
        {
            "id": "witness_thread",
            "presence_type": "presence",
            "resource_wallet": {
                "cpu": 0.0,
                "ram": 0.4,
            },
        }
    ]
    summary = simulation_module._apply_resource_daimoi_action_consumption(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat={"devices": {"cpu": {"utilization": 42.0}}},
        queue_ratio=0.0,
    )

    row = field_particles[0]
    assert bool(row.get("resource_action_blocked", True)) is False
    payment = row.get("resource_payment_vector", {})
    assert isinstance(payment, dict)
    assert float(payment.get("ram", 0.0) or 0.0) > 0.0
    by_resource = summary.get("by_resource", {})
    assert isinstance(by_resource, dict)
    assert float(by_resource.get("ram", 0.0) or 0.0) > 0.0


def test_resource_action_consumption_uses_wallet_denoms_greedy_subset() -> None:
    vector_small = {
        "cpu": 0.00002,
        "ram": 0.000012,
        "disk": 0.000008,
        "network": 0.000008,
        "gpu": 0.00001,
        "npu": 0.00001,
    }
    vector_large = {
        "cpu": 0.03,
        "ram": 0.02,
        "disk": 0.015,
        "network": 0.015,
        "gpu": 0.018,
        "npu": 0.018,
    }
    field_particles = [
        {
            "id": "particle:witness:denom-knapsack",
            "presence_id": "witness_thread",
            "is_nexus": False,
            "message_probability": 0.0,
            "route_probability": 0.0,
            "influence_power": 0.0,
            "drift_score": 0.0,
        }
    ]
    presence_impacts = [
        {
            "id": "witness_thread",
            "presence_type": "presence",
            "resource_wallet": {
                key: float(
                    vector_small.get(key, 0.0) * 2.0 + vector_large.get(key, 0.0)
                )
                for key in ("cpu", "ram", "disk", "network", "gpu", "npu")
            },
            "resource_wallet_denoms": [
                {"vector": dict(vector_small), "count": 2},
                {"vector": dict(vector_large), "count": 1},
            ],
        }
    ]

    summary = simulation_module._apply_resource_daimoi_action_consumption(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat={"devices": {"cpu": {"utilization": 28.0}}},
        queue_ratio=0.0,
    )

    row = field_particles[0]
    assert bool(row.get("resource_action_blocked", True)) is False
    assert str(row.get("resource_burn_strategy", "")).strip() == "denom_knapsack"
    payment = row.get("resource_payment_vector", {})
    assert isinstance(payment, dict)
    paid_total = sum(float(value or 0.0) for value in payment.values())
    assert paid_total > 0.0
    assert (
        paid_total < 0.001
    )  # should pick small bucket(s), not the large overpay packet

    denoms_after = presence_impacts[0].get("resource_wallet_denoms", [])
    assert isinstance(denoms_after, list)
    small_bucket_after = next(
        (
            bucket
            for bucket in denoms_after
            if isinstance(bucket, dict)
            and isinstance(bucket.get("vector"), dict)
            and float((bucket.get("vector", {}) or {}).get("cpu", 0.0) or 0.0) < 0.001
        ),
        {},
    )
    assert int(float((small_bucket_after or {}).get("count", 0) or 0.0)) == 1
    assert float(summary.get("consumed_total", 0.0) or 0.0) > 0.0


def test_resource_action_consumption_blocks_without_partial_spend() -> None:
    field_particles = [
        {
            "id": "particle:witness:starved",
            "presence_id": "witness_thread",
            "is_nexus": False,
            "message_probability": 0.95,
            "route_probability": 0.95,
            "influence_power": 0.95,
            "drift_score": 0.95,
        }
    ]
    presence_impacts = [
        {
            "id": "witness_thread",
            "presence_type": "presence",
            "resource_wallet": {
                "cpu": 0.000001,
                "ram": 0.0,
                "disk": 0.0,
                "network": 0.0,
                "gpu": 0.0,
                "npu": 0.0,
            },
        }
    ]

    summary = simulation_module._apply_resource_daimoi_action_consumption(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat={"devices": {"cpu": {"utilization": 32.0}}},
        queue_ratio=0.0,
    )

    row = field_particles[0]
    assert bool(row.get("resource_action_blocked", False)) is True
    assert float(row.get("resource_consume_amount", 1.0) or 0.0) == pytest.approx(0.0)
    payment = row.get("resource_payment_vector", {})
    assert isinstance(payment, dict)
    assert not payment
    wallet = presence_impacts[0].get("resource_wallet", {})
    assert isinstance(wallet, dict)
    assert float(wallet.get("cpu", 0.0) or 0.0) == pytest.approx(0.000001)
    assert float(summary.get("consumed_total", 1.0) or 0.0) == pytest.approx(0.0)


def test_resource_action_consumption_control_budget_degrades_complexity(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIMULATION_RESOURCE_CTL_BUDGET_RECHARGE", "0")
    monkeypatch.setenv("SIMULATION_RESOURCE_CTL_BUDGET_CAP_CPU", "0.01")
    monkeypatch.setenv("SIMULATION_RESOURCE_CTL_BUDGET_CAP_RAM", "0.01")
    monkeypatch.setenv("SIMULATION_RESOURCE_CTL_BUDGET_CAP_DISK", "0.01")
    monkeypatch.setenv("SIMULATION_RESOURCE_CTL_BUDGET_CAP_NETWORK", "0.01")
    monkeypatch.setenv("SIMULATION_RESOURCE_CTL_BUDGET_CAP_GPU", "0.01")
    monkeypatch.setenv("SIMULATION_RESOURCE_CTL_BUDGET_CAP_NPU", "0.01")
    simulation_module.reset_simulation_bootstrap_state(
        clear_layout_cache=False,
        rearm_boot_reset=False,
    )

    field_particles = [
        {
            "id": f"particle:witness:budget:{index:02d}",
            "presence_id": "witness_thread",
            "is_nexus": False,
            "message_probability": 0.2,
            "route_probability": 0.1,
            "influence_power": 0.2,
            "drift_score": 0.05,
        }
        for index in range(48)
    ]
    presence_impacts = [
        {
            "id": "witness_thread",
            "presence_type": "presence",
            "resource_wallet": {
                "cpu": 1.0,
                "ram": 1.0,
                "disk": 1.0,
                "network": 1.0,
                "gpu": 1.0,
                "npu": 1.0,
            },
            "resource_wallet_denoms": [
                {
                    "vector": {
                        "cpu": 0.00005,
                        "ram": 0.00003,
                        "disk": 0.00002,
                        "network": 0.00002,
                        "gpu": 0.00002,
                        "npu": 0.00002,
                    },
                    "count": 100,
                }
            ],
        }
    ]

    summary = simulation_module._apply_resource_daimoi_action_consumption(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat={"devices": {"cpu": {"utilization": 32.0}}},
        queue_ratio=1.0,
    )

    control = summary.get("control_budget", {})
    assert isinstance(control, dict)
    assert str(control.get("mode", "")).strip() in {"minimal", "reduced"}
    assert bool(control.get("allow_denom", True)) is False
    assert int(float(control.get("scheduled_rows", 0) or 0.0)) < int(
        float(control.get("candidate_rows", 0) or 0.0)
    )
    strategies = [
        str(row.get("resource_burn_strategy", "")).strip()
        for row in field_particles
        if "resource_burn_strategy" in row
    ]
    assert "aggregate_mix" in strategies


def test_resource_sentinel_burn_prefers_focus_resource_under_pressure() -> None:
    field_particles = [
        {
            "id": "particle:gpu-sentinel:burn",
            "presence_id": "health_sentinel_gpu1",
            "is_nexus": False,
            "message_probability": 0.4,
            "route_probability": 0.3,
            "influence_power": 0.5,
            "drift_score": 0.2,
        }
    ]
    presence_impacts = [
        {
            "id": "health_sentinel_gpu1",
            "presence_type": "presence",
            "resource_wallet": {
                "gpu": 2.0,
                "cpu": 2.0,
            },
        }
    ]

    summary = simulation_module._apply_resource_daimoi_action_consumption(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat={
            "devices": {
                "cpu": {"utilization": 31.0},
                "gpu1": {"utilization": 95.0},
                "gpu2": {"utilization": 96.0},
            }
        },
        queue_ratio=0.0,
    )

    row = field_particles[0]
    assert bool(row.get("resource_sentinel_idle", True)) is False
    assert str(row.get("resource_consume_type", "")).strip() == "gpu"
    payment = row.get("resource_payment_vector", {})
    assert isinstance(payment, dict)
    assert float(payment.get("gpu", 0.0) or 0.0) > 0.0
    sentinel_active = summary.get("sentinel_burn_active", {})
    assert isinstance(sentinel_active, dict)
    assert sentinel_active.get("health_sentinel_gpu1") is True


def test_config_payload_exposes_magic_number_constants_by_module() -> None:
    payload = server_module._config_payload()

    assert payload.get("ok") is True
    modules = payload.get("modules", {})
    assert isinstance(modules, dict)
    assert {"daimoi_probabilistic", "simulation", "server"}.issubset(
        set(modules.keys())
    )

    daimoi_constants = modules.get("daimoi_probabilistic", {}).get("constants", {})
    assert float(daimoi_constants.get("NEXUS_DAMPING", 0.0)) == pytest.approx(
        daimoi_module.NEXUS_DAMPING
    )
    assert float(daimoi_constants.get("DAIMOI_TRANSFER_LAMBDA", 0.0)) == pytest.approx(
        daimoi_module.DAIMOI_TRANSFER_LAMBDA
    )

    simulation_constants = modules.get("simulation", {}).get("constants", {})
    assert float(
        simulation_constants.get("SIMULATION_STREAM_DAIMOI_FRICTION", 0.0)
    ) == pytest.approx(simulation_module.SIMULATION_STREAM_DAIMOI_FRICTION)
    assert float(
        simulation_constants.get("SIMULATION_STREAM_NEXUS_FRICTION", 0.0)
    ) == pytest.approx(simulation_module.SIMULATION_STREAM_NEXUS_FRICTION)

    server_constants = modules.get("server", {}).get("constants", {})
    assert float(server_constants.get("_SIMULATION_HTTP_CACHE_SECONDS", 0.0)) > 0.0


def test_config_payload_supports_single_module_filter() -> None:
    payload = server_module._config_payload(module_filter="simulation")

    assert payload.get("ok") is True
    modules = payload.get("modules", {})
    assert set(modules.keys()) == {"simulation"}
    assert modules.get("simulation", {}).get("constant_count", 0) > 0


def test_config_payload_rejects_unknown_module_filter() -> None:
    payload = server_module._config_payload(module_filter="unknown-module")

    assert payload.get("ok") is False
    assert payload.get("error") == "unknown_module"
    assert "daimoi_probabilistic" in payload.get("available_modules", [])


def test_config_update_and_reset_scalar_constant() -> None:
    baseline = float(simulation_module.SIMULATION_STREAM_DAIMOI_FRICTION)
    requested = baseline * 0.75
    expected = max(0.0, min(2.0, requested))

    update = server_module._config_apply_update(
        module_name="simulation",
        key_name="SIMULATION_STREAM_DAIMOI_FRICTION",
        path_tokens=[],
        value=requested,
    )

    assert update.get("ok") is True
    assert float(simulation_module.SIMULATION_STREAM_DAIMOI_FRICTION) == pytest.approx(
        expected
    )

    reset = server_module._config_reset_updates(
        module_name="simulation",
        key_name="SIMULATION_STREAM_DAIMOI_FRICTION",
    )
    assert reset.get("ok") is True
    assert float(simulation_module.SIMULATION_STREAM_DAIMOI_FRICTION) == pytest.approx(
        baseline
    )


def test_config_update_clamps_stream_friction_bounds() -> None:
    baseline = float(simulation_module.SIMULATION_STREAM_NEXUS_FRICTION)

    high_update = server_module._config_apply_update(
        module_name="simulation",
        key_name="SIMULATION_STREAM_NEXUS_FRICTION",
        path_tokens=[],
        value=31.5552,
    )
    assert high_update.get("ok") is True
    assert float(simulation_module.SIMULATION_STREAM_NEXUS_FRICTION) == pytest.approx(
        2.0
    )

    low_update = server_module._config_apply_update(
        module_name="simulation",
        key_name="SIMULATION_STREAM_NEXUS_FRICTION",
        path_tokens=[],
        value=-4.0,
    )
    assert low_update.get("ok") is True
    assert float(simulation_module.SIMULATION_STREAM_NEXUS_FRICTION) == pytest.approx(
        0.0
    )

    reset = server_module._config_reset_updates(
        module_name="simulation",
        key_name="SIMULATION_STREAM_NEXUS_FRICTION",
    )
    assert reset.get("ok") is True
    assert float(simulation_module.SIMULATION_STREAM_NEXUS_FRICTION) == pytest.approx(
        baseline
    )


def test_config_update_and_reset_nested_leaf() -> None:
    baseline = float(
        daimoi_module._ROLE_PRIOR_WEIGHTS["crawl-routing"]["invoke_graph_crawl"]
    )

    update = server_module._config_apply_update(
        module_name="daimoi_probabilistic",
        key_name="_ROLE_PRIOR_WEIGHTS",
        path_tokens=["crawl-routing", "invoke_graph_crawl"],
        value=baseline + 0.45,
    )

    assert update.get("ok") is True
    assert float(
        daimoi_module._ROLE_PRIOR_WEIGHTS["crawl-routing"]["invoke_graph_crawl"]
    ) == pytest.approx(baseline + 0.45)

    reset = server_module._config_reset_updates(
        module_name="daimoi_probabilistic",
        key_name="_ROLE_PRIOR_WEIGHTS",
        path_tokens=["crawl-routing", "invoke_graph_crawl"],
    )
    assert reset.get("ok") is True
    assert float(
        daimoi_module._ROLE_PRIOR_WEIGHTS["crawl-routing"]["invoke_graph_crawl"]
    ) == pytest.approx(baseline)


def test_field_particle_friction_is_split_between_daimoi_and_nexus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIM_TICK_SECONDS", "0.08")
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_DAIMOI_FRICTION", 0.95)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_FRICTION", 0.8)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_VELOCITY_SCALE", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_FIELD_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_CENTER_GRAVITY", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_JITTER_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_MAX_SPEED", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NOOI_FLOW_GAIN", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NOOI_NEXUS_FLOW_GAIN", 0.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_STATIC_FRICTION", 0.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_STATIC_RELEASE_SPEED", 0.0
    )
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_STATIC_CREEP", 1.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_QUADRATIC_DRAG", 0.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_DRAG_SPEED_REF", 1.0
    )
    simulation_module._reset_nooi_field_state()

    simulation: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "daimoi",
                    "presence_id": "presence.alpha",
                    "x": 0.2,
                    "y": 0.2,
                    "vx": 0.2,
                    "vy": 0.0,
                    "is_nexus": False,
                },
                {
                    "id": "nexus",
                    "presence_id": "presence.beta",
                    "x": 0.8,
                    "y": 0.8,
                    "vx": 0.2,
                    "vy": 0.0,
                    "is_nexus": True,
                },
            ]
        }
    }

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=0.0,
    )

    rows = simulation.get("presence_dynamics", {}).get("field_particles", [])
    row_by_id = {str(row.get("id", "")): row for row in rows if isinstance(row, dict)}
    assert float(row_by_id["daimoi"]["vx"]) == pytest.approx(0.19, abs=1e-6)
    assert float(row_by_id["nexus"]["vx"]) == pytest.approx(0.16, abs=1e-6)


def test_field_particle_prefers_target_presence_over_origin_attractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIM_TICK_SECONDS", "0.08")
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_DAIMOI_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_VELOCITY_SCALE", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_FIELD_FORCE", 0.4)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_CENTER_GRAVITY", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_JITTER_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_MAX_SPEED", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NOOI_FLOW_GAIN", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NOOI_NEXUS_FLOW_GAIN", 0.0
    )
    simulation_module._reset_nooi_field_state()

    simulation: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "field:presence.alpha:000001",
                    "presence_id": "presence.alpha",
                    "owner_presence_id": "presence.alpha",
                    "origin_presence_id": "presence.alpha",
                    "target_presence_id": "presence.beta",
                    "x": 0.2,
                    "y": 0.5,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_nexus": False,
                },
                {
                    "id": "center:beta",
                    "presence_id": "presence.beta",
                    "x": 0.8,
                    "y": 0.5,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_nexus": True,
                },
            ]
        }
    }

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=0.0,
    )

    rows = simulation.get("presence_dynamics", {}).get("field_particles", [])
    row_by_id = {str(row.get("id", "")): row for row in rows if isinstance(row, dict)}
    mover = row_by_id["field:presence.alpha:000001"]
    assert float(mover.get("vx", 0.0)) > 0.0
    assert float(mover.get("x", 0.0)) > 0.2


def test_nexus_payload_weighting_increases_semantic_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIM_TICK_SECONDS", "0.08")
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_DAIMOI_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_VELOCITY_SCALE", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_FIELD_FORCE", 0.34)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_CENTER_GRAVITY", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_JITTER_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_MAX_SPEED", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NOOI_FLOW_GAIN", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NOOI_NEXUS_FLOW_GAIN", 0.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_SEMANTIC_WEIGHT", 0.82
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_SEMANTIC_WALLET_SCALE", 24.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_MAX_SPEED_SCALE", 1.0
    )
    simulation_module._reset_nooi_field_state()

    simulation: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "nexus-low",
                    "presence_id": "presence.low",
                    "x": 0.2,
                    "y": 0.45,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_nexus": True,
                    "route_node_id": "node.alpha",
                    "graph_node_id": "node.alpha",
                    "route_x": 0.8,
                    "route_y": 0.45,
                    "graph_x": 0.8,
                    "graph_y": 0.45,
                    "route_probability": 0.9,
                    "influence_power": 0.6,
                    "semantic_text_chars": 12.0,
                    "semantic_mass": 0.45,
                    "resource_wallet_total": 0.2,
                },
                {
                    "id": "nexus-high",
                    "presence_id": "presence.high",
                    "x": 0.2,
                    "y": 0.62,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_nexus": True,
                    "route_node_id": "node.beta",
                    "graph_node_id": "node.beta",
                    "route_x": 0.8,
                    "route_y": 0.62,
                    "graph_x": 0.8,
                    "graph_y": 0.62,
                    "route_probability": 0.9,
                    "influence_power": 0.6,
                    "semantic_text_chars": 1600.0,
                    "semantic_mass": 6.5,
                    "resource_wallet_total": 24.0,
                },
            ]
        }
    }

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=45.0,
    )

    rows = simulation.get("presence_dynamics", {}).get("field_particles", [])
    row_by_id = {str(row.get("id", "")): row for row in rows if isinstance(row, dict)}
    assert float(row_by_id["nexus-high"]["vx"]) > float(row_by_id["nexus-low"]["vx"])


def test_orbit_damping_reduces_daimoi_tangential_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIM_TICK_SECONDS", "0.08")
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_DAIMOI_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_VELOCITY_SCALE", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_FIELD_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_CENTER_GRAVITY", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_JITTER_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_MAX_SPEED", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NOOI_FLOW_GAIN", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NOOI_NEXUS_FLOW_GAIN", 0.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_DAIMOI_ORBIT_DAMPING", 1.4
    )
    simulation_module._reset_nooi_field_state()

    simulation: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "orbiting-daimoi",
                    "presence_id": "presence.alpha",
                    "x": 0.2,
                    "y": 0.5,
                    "vx": 0.0,
                    "vy": 0.4,
                    "is_nexus": False,
                    "route_node_id": "node.alpha",
                    "graph_node_id": "node.alpha",
                    "route_x": 0.8,
                    "route_y": 0.5,
                    "graph_x": 0.8,
                    "graph_y": 0.5,
                    "route_probability": 1.0,
                    "influence_power": 1.0,
                }
            ]
        }
    }

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=88.0,
    )

    rows = simulation.get("presence_dynamics", {}).get("field_particles", [])
    row_by_id = {str(row.get("id", "")): row for row in rows if isinstance(row, dict)}
    assert abs(float(row_by_id["orbiting-daimoi"]["vy"])) < 0.3


def test_nexus_static_friction_requires_breakaway_drive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIM_TICK_SECONDS", "0.08")
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_DAIMOI_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_VELOCITY_SCALE", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_FIELD_FORCE", 0.26)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_CENTER_GRAVITY", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_JITTER_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_MAX_SPEED", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NOOI_FLOW_GAIN", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NOOI_NEXUS_FLOW_GAIN", 0.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_STATIC_FRICTION", 0.42
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_STATIC_RELEASE_SPEED", 0.06
    )
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_STATIC_CREEP", 0.05)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_QUADRATIC_DRAG", 0.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_DRAG_SPEED_REF", 1.0
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_SEMANTIC_WEIGHT", 0.82
    )
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NEXUS_MAX_SPEED_SCALE", 1.0
    )
    simulation_module._reset_nooi_field_state()

    simulation: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "nexus-stiction",
                    "presence_id": "presence.test",
                    "x": 0.5,
                    "y": 0.5,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_nexus": True,
                    "route_node_id": "node.slow",
                    "graph_node_id": "node.slow",
                    "route_x": 0.5005,
                    "route_y": 0.5,
                    "graph_x": 0.5005,
                    "graph_y": 0.5,
                    "route_probability": 0.45,
                    "influence_power": 0.2,
                    "semantic_text_chars": 8.0,
                    "semantic_mass": 0.4,
                    "resource_wallet_total": 0.0,
                }
            ]
        }
    }

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=91.0,
    )
    first_row = simulation.get("presence_dynamics", {}).get("field_particles", [])[0]
    first_vx = abs(float(first_row.get("vx", 0.0)))

    first_row["route_x"] = 0.9
    first_row["graph_x"] = 0.9
    first_row["semantic_text_chars"] = 1900.0
    first_row["semantic_mass"] = 6.8
    first_row["resource_wallet_total"] = 28.0

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=92.0,
    )
    second_row = simulation.get("presence_dynamics", {}).get("field_particles", [])[0]
    second_vx = abs(float(second_row.get("vx", 0.0)))

    assert first_vx < 0.02
    assert second_vx > first_vx * 2.0


def test_backend_stream_motion_overlays_emit_graph_and_presence_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIM_TICK_SECONDS", "0.08")
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_DAIMOI_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_VELOCITY_SCALE", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_FIELD_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_CENTER_GRAVITY", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_JITTER_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_MAX_SPEED", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NOOI_FLOW_GAIN", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NOOI_NEXUS_FLOW_GAIN", 0.0
    )
    simulation_module._reset_nooi_field_state()

    cache = getattr(simulation_module, "_DAIMO_DYNAMICS_CACHE", {})
    if isinstance(cache, dict):
        cache["graph_nodes"] = {}
        cache["presence_anchors"] = {}

    simulation: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "p1",
                    "presence_id": "presence.alpha",
                    "x": 0.2,
                    "y": 0.2,
                    "vx": 0.0,
                    "vy": 0.0,
                    "graph_node_id": "node.alpha",
                    "route_node_id": "node.alpha",
                    "graph_x": 0.6,
                    "graph_y": 0.6,
                    "route_x": 0.6,
                    "route_y": 0.6,
                    "route_probability": 0.85,
                    "influence_power": 0.72,
                },
                {
                    "id": "p2",
                    "presence_id": "presence.alpha",
                    "x": 0.24,
                    "y": 0.18,
                    "vx": 0.0,
                    "vy": 0.0,
                    "graph_node_id": "node.alpha",
                    "route_node_id": "node.alpha",
                    "graph_x": 0.6,
                    "graph_y": 0.6,
                    "route_x": 0.6,
                    "route_y": 0.6,
                    "route_probability": 0.85,
                    "influence_power": 0.72,
                },
            ]
        }
    }

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=100.0,
    )

    dynamics = simulation.get("presence_dynamics", {})
    graph_positions = dynamics.get("graph_node_positions", {})
    presence_positions = dynamics.get("presence_anchor_positions", {})

    assert isinstance(graph_positions, dict)
    assert isinstance(presence_positions, dict)
    assert "node.alpha" in graph_positions
    assert "presence.alpha" in presence_positions

    first_graph_x = float(graph_positions["node.alpha"]["x"])
    first_presence_x = float(presence_positions["presence.alpha"]["x"])

    rows = dynamics.get("field_particles", [])
    assert isinstance(rows, list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        row["x"] = 0.82
        row["y"] = 0.78

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=101.0,
    )

    updated_dynamics = simulation.get("presence_dynamics", {})
    updated_graph_positions = updated_dynamics.get("graph_node_positions", {})
    updated_presence_positions = updated_dynamics.get("presence_anchor_positions", {})

    assert isinstance(updated_graph_positions, dict)
    assert isinstance(updated_presence_positions, dict)
    assert float(updated_graph_positions["node.alpha"]["x"]) > first_graph_x
    assert float(updated_presence_positions["presence.alpha"]["x"]) > first_presence_x


def test_advance_particles_flow_with_nooi_for_daimoi_and_nexus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module, "SIMULATION_DISABLE_DAIMOI", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_RANDOM_FIELD_VECTORS_ON_BOOT", 0.0
    )
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_FIELD_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_CENTER_GRAVITY", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_JITTER_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_VELOCITY_SCALE", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_MAX_SPEED", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_DAIMOI_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NOOI_FLOW_GAIN", 0.7)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NOOI_NEXUS_FLOW_GAIN", 0.7
    )

    simulation_module._reset_nooi_field_state()
    for _ in range(8):
        simulation_module._NOOI_FIELD.deposit(0.2, 0.2, 1.0, 0.0)
        simulation_module._NOOI_FIELD.deposit(0.8, 0.8, 1.0, 0.0)

    simulation: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "daimoi",
                    "presence_id": "presence.alpha",
                    "x": 0.2,
                    "y": 0.2,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_nexus": False,
                },
                {
                    "id": "nexus",
                    "presence_id": "presence.beta",
                    "x": 0.8,
                    "y": 0.8,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_nexus": True,
                },
            ]
        }
    }

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=200.0,
    )

    rows = simulation.get("presence_dynamics", {}).get("field_particles", [])
    row_by_id = {str(row.get("id", "")): row for row in rows if isinstance(row, dict)}
    assert float(row_by_id["daimoi"]["vx"]) > 0.0
    assert float(row_by_id["nexus"]["vx"]) > 0.0


def test_nooi_field_is_influenced_only_by_non_nexus_daimoi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simulation_module, "SIMULATION_DISABLE_DAIMOI", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_RANDOM_FIELD_VECTORS_ON_BOOT", 0.0
    )
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_FIELD_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_CENTER_GRAVITY", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_JITTER_FORCE", 0.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_VELOCITY_SCALE", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_MAX_SPEED", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_DAIMOI_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NEXUS_FRICTION", 1.0)
    monkeypatch.setattr(simulation_module, "SIMULATION_STREAM_NOOI_FLOW_GAIN", 0.0)
    monkeypatch.setattr(
        simulation_module, "SIMULATION_STREAM_NOOI_NEXUS_FLOW_GAIN", 0.0
    )

    simulation_module._reset_nooi_field_state()

    nexus_only: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "nexus-only",
                    "presence_id": "presence.beta",
                    "x": 0.5,
                    "y": 0.5,
                    "vx": 0.35,
                    "vy": 0.0,
                    "is_nexus": True,
                }
            ]
        }
    }
    simulation_module.advance_simulation_field_particles(
        nexus_only,
        dt_seconds=0.08,
        now_seconds=300.0,
    )
    nooi_nexus = (
        nexus_only.get("presence_dynamics", {}).get("nooi_field", {}).get("cells", [])
    )
    assert isinstance(nooi_nexus, list)
    assert len(nooi_nexus) == 0

    simulation_module._reset_nooi_field_state()
    daimoi_only: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "daimoi-only",
                    "presence_id": "presence.alpha",
                    "x": 0.5,
                    "y": 0.5,
                    "vx": 0.35,
                    "vy": 0.0,
                    "is_nexus": False,
                }
            ]
        }
    }
    simulation_module.advance_simulation_field_particles(
        daimoi_only,
        dt_seconds=0.08,
        now_seconds=301.0,
    )
    nooi_daimoi = (
        daimoi_only.get("presence_dynamics", {}).get("nooi_field", {}).get("cells", [])
    )
    assert isinstance(nooi_daimoi, list)
    assert len(nooi_daimoi) > 0


def test_advance_simulation_field_particles_applies_policy_tick_signals_and_cap() -> (
    None
):
    simulation_module._reset_nooi_field_state()

    simulation: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {
                    "id": "p-1",
                    "presence_id": "presence.alpha",
                    "x": 0.1,
                    "y": 0.1,
                    "vx": 0.0,
                    "vy": 0.0,
                    "route_node_id": "node.alpha",
                    "graph_node_id": "node.alpha",
                },
                {
                    "id": "p-2",
                    "presence_id": "presence.beta",
                    "x": 0.5,
                    "y": 0.5,
                    "vx": 0.0,
                    "vy": 0.0,
                    "route_node_id": "node.beta",
                    "graph_node_id": "node.beta",
                },
                {
                    "id": "p-3",
                    "presence_id": "presence.gamma",
                    "x": 0.9,
                    "y": 0.9,
                    "vx": 0.0,
                    "vy": 0.0,
                    "route_node_id": "node.gamma",
                    "graph_node_id": "node.gamma",
                },
            ]
        }
    }

    simulation_module.advance_simulation_field_particles(
        simulation,
        dt_seconds=0.08,
        now_seconds=400.0,
        policy={
            "slack_ms": 6.5,
            "tick_budget_ms": 8.0,
            "ingestion_pressure": 0.72,
            "ws_particle_max": 2,
            "guard_mode": "degraded",
        },
    )

    presence_dynamics = simulation.get("presence_dynamics", {})
    assert isinstance(presence_dynamics, dict)
    rows = presence_dynamics.get("field_particles", [])
    assert isinstance(rows, list)
    assert len(rows) == 2

    tick_signals = presence_dynamics.get("tick_signals", {})
    assert isinstance(tick_signals, dict)
    assert tick_signals.get("slack_ms") == 6.5
    assert tick_signals.get("tick_budget_ms") == 8.0
    assert tick_signals.get("ingestion_pressure") == 0.72
    assert tick_signals.get("ws_particle_max") == 2
    assert tick_signals.get("guard_mode") == "degraded"
