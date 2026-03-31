from __future__ import annotations

from typing import Any

from code.world_web.simulation_ws_tick_strategy import (
    build_simulation_ws_pressure_snapshot,
    resolve_effective_particle_payload_key,
    resolve_ws_particle_max,
)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def test_build_simulation_ws_pressure_snapshot_uses_runtime_and_device_pressure() -> (
    None
):
    snapshot = build_simulation_ws_pressure_snapshot(
        catalog={"eta_mu_inbox": {"pending_count": 16}},
        simulation_payload={
            "presence_dynamics": {
                "resource_heartbeat": {
                    "devices": {
                        "gpu1": {"utilization": 40.0},
                        "gpu2": {"utilization": 77.0},
                        "npu0": {"utilization": 12.0},
                    }
                }
            }
        },
        runtime_inbox_lock_active=False,
        safe_float=_safe_float,
        inbox_pending_soft=64.0,
    )
    assert snapshot.ingestion_pressure == 0.77


def test_resolve_ws_particle_max_applies_caps_in_priority_order() -> None:
    cap = resolve_ws_particle_max(
        ingestion_pressure=0.8,
        ws_send_pressure=0.86,
        ws_network_particle_cap=300,
        governor_particle_cap=300,
        stream_particle_max=300,
        slack_ms_before_sim=3.5,
    )
    assert cap == 48


def test_resolve_effective_particle_payload_key_forces_lite_under_pressure() -> None:
    assert (
        resolve_effective_particle_payload_key(
            particle_payload_key="full",
            ingestion_pressure=0.2,
            slack_ms_before_sim=18.0,
            ws_send_pressure=0.2,
        )
        == "full"
    )
    assert (
        resolve_effective_particle_payload_key(
            particle_payload_key="full",
            ingestion_pressure=0.2,
            slack_ms_before_sim=18.0,
            ws_send_pressure=0.6,
        )
        == "lite"
    )
