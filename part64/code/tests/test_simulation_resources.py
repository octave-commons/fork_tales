from __future__ import annotations

from typing import Any

import pytest
import code.world_web.simulation as simulation_module

from code.world_web import build_simulation_state


@pytest.fixture(autouse=True)
def _reset_simulation_runtime_state() -> None:
    simulation_module.reset_simulation_bootstrap_state()


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
