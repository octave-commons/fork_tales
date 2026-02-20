from __future__ import annotations

import math
from typing import Any

import pytest
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
    assert any(float(value) > 0.0 for value in wallet.values())
