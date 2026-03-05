from __future__ import annotations

from typing import Any

from code.world_web import (
    simulation_ws_tick_policy_controller as tick_policy_controller_module,
)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def test_tick_policy_sets_lite_mode_when_inbox_lock_active() -> None:
    state = tick_policy_controller_module.build_simulation_ws_tick_policy(
        catalog={},
        simulation_payload={},
        runtime_inbox_lock_active=True,
        safe_float=_safe_float,
        ws_send_pressure=0.2,
        ws_network_particle_cap=300,
        governor_particle_cap=300,
        stream_particle_max=300,
        particle_payload_key="full",
        slack_ms_before_sim=18.0,
        tick_budget_ms=120.0,
        guard_mode="normal",
        inbox_pending_soft=64.0,
    )

    assert state.ingestion_pressure == 1.0
    assert state.ws_particle_max == 96
    assert state.effective_particle_payload_key == "lite"
    assert state.tick_policy.get("guard_mode") == "normal"


def test_tick_policy_send_pressure_caps_particles_aggressively() -> None:
    state = tick_policy_controller_module.build_simulation_ws_tick_policy(
        catalog={},
        simulation_payload={},
        runtime_inbox_lock_active=False,
        safe_float=_safe_float,
        ws_send_pressure=0.86,
        ws_network_particle_cap=300,
        governor_particle_cap=300,
        stream_particle_max=300,
        particle_payload_key="full",
        slack_ms_before_sim=18.0,
        tick_budget_ms=120.0,
        guard_mode="normal",
        inbox_pending_soft=64.0,
    )

    assert state.ingestion_pressure == 0.0
    assert state.ws_particle_max == 48
    assert state.effective_particle_payload_key == "lite"


def test_tick_policy_uses_gpu_pressure_from_resource_heartbeat() -> None:
    state = tick_policy_controller_module.build_simulation_ws_tick_policy(
        catalog={},
        simulation_payload={
            "presence_dynamics": {
                "resource_heartbeat": {
                    "devices": {
                        "gpu1": {"utilization": 82.0},
                        "gpu2": {"utilization": 12.0},
                        "npu0": {"utilization": 5.0},
                    }
                }
            }
        },
        runtime_inbox_lock_active=False,
        safe_float=_safe_float,
        ws_send_pressure=0.2,
        ws_network_particle_cap=300,
        governor_particle_cap=300,
        stream_particle_max=300,
        particle_payload_key="full",
        slack_ms_before_sim=18.0,
        tick_budget_ms=120.0,
        guard_mode="normal",
        inbox_pending_soft=64.0,
    )

    assert state.ingestion_pressure == 0.82
    assert state.ws_particle_max == 96
    assert state.effective_particle_payload_key == "lite"


def test_tick_policy_keeps_full_payload_when_pressure_is_low() -> None:
    state = tick_policy_controller_module.build_simulation_ws_tick_policy(
        catalog={"eta_mu_inbox": {"pending_count": 0}},
        simulation_payload={},
        runtime_inbox_lock_active=False,
        safe_float=_safe_float,
        ws_send_pressure=0.2,
        ws_network_particle_cap=300,
        governor_particle_cap=300,
        stream_particle_max=300,
        particle_payload_key="full",
        slack_ms_before_sim=18.0,
        tick_budget_ms=120.0,
        guard_mode="normal",
        inbox_pending_soft=64.0,
    )

    assert state.ingestion_pressure == 0.0
    assert state.ws_particle_max == 300
    assert state.effective_particle_payload_key == "full"
