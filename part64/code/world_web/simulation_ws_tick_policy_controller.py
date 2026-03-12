from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .simulation_ws_tick_strategy import (
    build_simulation_ws_pressure_snapshot,
    resolve_effective_particle_payload_key,
    resolve_ws_particle_max,
)


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
    pressure_snapshot = build_simulation_ws_pressure_snapshot(
        catalog=catalog,
        simulation_payload=simulation_payload,
        runtime_inbox_lock_active=runtime_inbox_lock_active,
        safe_float=safe_float,
        inbox_pending_soft=inbox_pending_soft,
    )
    ingestion_pressure = float(pressure_snapshot.ingestion_pressure)

    ws_particle_max = resolve_ws_particle_max(
        ingestion_pressure=ingestion_pressure,
        ws_send_pressure=ws_send_pressure,
        ws_network_particle_cap=ws_network_particle_cap,
        governor_particle_cap=governor_particle_cap,
        stream_particle_max=stream_particle_max,
        slack_ms_before_sim=slack_ms_before_sim,
    )

    effective_particle_payload_key = resolve_effective_particle_payload_key(
        particle_payload_key=particle_payload_key,
        ingestion_pressure=ingestion_pressure,
        slack_ms_before_sim=slack_ms_before_sim,
        ws_send_pressure=ws_send_pressure,
    )

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
