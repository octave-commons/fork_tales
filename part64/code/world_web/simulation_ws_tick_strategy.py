from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SimulationWsPressureSnapshot:
    ingestion_pressure: float


_WS_SEND_PRESSURE_CAPS: tuple[tuple[float, int], ...] = (
    (0.85, 48),
    (0.65, 72),
    (0.50, 96),
)

_WS_SLACK_CAPS: tuple[tuple[float, int], ...] = (
    (4.0, 64),
    (10.0, 96),
)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def build_simulation_ws_pressure_snapshot(
    *,
    catalog: dict[str, Any],
    simulation_payload: dict[str, Any],
    runtime_inbox_lock_active: bool,
    safe_float: Callable[[Any, float], float],
    inbox_pending_soft: float,
) -> SimulationWsPressureSnapshot:
    inbox_state = catalog.get("eta_mu_inbox", {}) if isinstance(catalog, dict) else {}
    if not isinstance(inbox_state, dict):
        inbox_state = {}
    inbox_pending = max(0, int(safe_float(inbox_state.get("pending_count", 0), 0.0)))
    inbox_pending_soft_normalized = max(1.0, float(inbox_pending_soft))
    inbox_pending_pressure = _clamp01(inbox_pending / inbox_pending_soft_normalized)

    dynamics_snapshot = (
        simulation_payload.get("presence_dynamics", {})
        if isinstance(simulation_payload, dict)
        else {}
    )
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

    return SimulationWsPressureSnapshot(
        ingestion_pressure=_clamp01(
            max(
                1.0 if runtime_inbox_lock_active else 0.0,
                inbox_pending_pressure,
                gpu_utilization_pressure,
                npu_utilization_pressure,
            )
        )
    )


def resolve_ws_particle_max(
    *,
    ingestion_pressure: float,
    ws_send_pressure: float,
    ws_network_particle_cap: int,
    governor_particle_cap: int,
    stream_particle_max: int,
    slack_ms_before_sim: float,
) -> int:
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

    for threshold, cap in _WS_SEND_PRESSURE_CAPS:
        if ws_send_pressure >= threshold:
            ws_particle_max = min(ws_particle_max, cap)
            break

    for threshold, cap in _WS_SLACK_CAPS:
        if slack_ms_before_sim <= threshold:
            ws_particle_max = min(ws_particle_max, cap)
            break

    return int(ws_particle_max)


def resolve_effective_particle_payload_key(
    *,
    particle_payload_key: str,
    ingestion_pressure: float,
    slack_ms_before_sim: float,
    ws_send_pressure: float,
) -> str:
    if (
        ingestion_pressure >= 0.7
        or slack_ms_before_sim <= 4.0
        or ws_send_pressure >= 0.55
    ):
        return "lite"
    return str(particle_payload_key or "lite")
