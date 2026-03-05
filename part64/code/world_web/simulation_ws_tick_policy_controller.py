from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SimulationWsTickPolicyState:
    ingestion_pressure: float
    ws_particle_max: int
    effective_particle_payload_key: str
    tick_policy: dict[str, Any]


def build_simulation_ws_tick_policy(
    *,
    catalog: dict[str, Any],
    simulation_payload: dict[str, Any],
    runtime_inbox_lock_active: bool,
    safe_float: Callable[[Any, float], float],
    ws_send_pressure: float,
    ws_network_particle_cap: int,
    governor_particle_cap: int,
    stream_particle_max: int,
    particle_payload_key: str,
    slack_ms_before_sim: float,
    tick_budget_ms: float,
    guard_mode: str,
    inbox_pending_soft: float,
) -> SimulationWsTickPolicyState:
    inbox_state = catalog.get("eta_mu_inbox", {}) if isinstance(catalog, dict) else {}
    if not isinstance(inbox_state, dict):
        inbox_state = {}
    inbox_pending = max(0, int(safe_float(inbox_state.get("pending_count", 0), 0.0)))
    inbox_pending_soft_normalized = max(1.0, float(inbox_pending_soft))
    inbox_pending_pressure = max(
        0.0,
        min(1.0, inbox_pending / inbox_pending_soft_normalized),
    )

    dynamics_snapshot = simulation_payload.get("presence_dynamics", {})
    if not isinstance(dynamics_snapshot, dict):
        dynamics_snapshot = {}
    resource_heartbeat_snapshot = dynamics_snapshot.get("resource_heartbeat", {})
    if not isinstance(resource_heartbeat_snapshot, dict):
        resource_heartbeat_snapshot = {}
    resource_devices_snapshot = resource_heartbeat_snapshot.get("devices", {})
    if not isinstance(resource_devices_snapshot, dict):
        resource_devices_snapshot = {}
    gpu1_state = resource_devices_snapshot.get("gpu1", {})
    if not isinstance(gpu1_state, dict):
        gpu1_state = {}
    gpu2_state = resource_devices_snapshot.get("gpu2", {})
    if not isinstance(gpu2_state, dict):
        gpu2_state = {}
    npu0_state = resource_devices_snapshot.get("npu0", {})
    if not isinstance(npu0_state, dict):
        npu0_state = {}

    gpu_utilization_pressure = (
        max(
            safe_float(gpu1_state.get("utilization", 0.0), 0.0),
            safe_float(gpu2_state.get("utilization", 0.0), 0.0),
        )
        / 100.0
    )
    npu_utilization_pressure = (
        safe_float(npu0_state.get("utilization", 0.0), 0.0) / 100.0
    )

    ingestion_pressure = max(
        0.0,
        min(
            1.0,
            max(
                1.0 if runtime_inbox_lock_active else 0.0,
                inbox_pending_pressure,
                gpu_utilization_pressure,
                npu_utilization_pressure,
            ),
        ),
    )

    ws_particle_max = max(
        24,
        min(
            int(stream_particle_max),
            int(governor_particle_cap),
            int(ws_network_particle_cap),
        ),
    )
    if ingestion_pressure >= 0.7:
        ws_particle_max = min(ws_particle_max, 96)
    if ws_send_pressure >= 0.85:
        ws_particle_max = min(ws_particle_max, 48)
    elif ws_send_pressure >= 0.65:
        ws_particle_max = min(ws_particle_max, 72)
    elif ws_send_pressure >= 0.5:
        ws_particle_max = min(ws_particle_max, 96)
    if slack_ms_before_sim <= 4.0:
        ws_particle_max = min(ws_particle_max, 64)
    elif slack_ms_before_sim <= 10.0:
        ws_particle_max = min(ws_particle_max, 96)

    effective_particle_payload_key = str(particle_payload_key or "lite")
    if (
        ingestion_pressure >= 0.7
        or slack_ms_before_sim <= 4.0
        or ws_send_pressure >= 0.55
    ):
        effective_particle_payload_key = "lite"

    tick_policy = {
        "tick_budget_ms": float(tick_budget_ms),
        "slack_ms": float(slack_ms_before_sim),
        "ingestion_pressure": float(ingestion_pressure),
        "ws_particle_max": int(ws_particle_max),
        "guard_mode": str(guard_mode),
    }
    return SimulationWsTickPolicyState(
        ingestion_pressure=float(ingestion_pressure),
        ws_particle_max=int(ws_particle_max),
        effective_particle_payload_key=effective_particle_payload_key,
        tick_policy=tick_policy,
    )
