"""Context preparation helpers for simulation field-particle advance."""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from typing import Any, Callable


def apply_ws_particle_cap(
    presence_dynamics: dict[str, Any],
    rows: Any,
    *,
    policy: dict[str, Any] | None,
    safe_float: Callable[[Any, float], float],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    policy_obj = policy if isinstance(policy, dict) else {}
    ws_particle_max = 0
    if policy_obj:
        ws_particle_max = max(
            0,
            int(safe_float(policy_obj.get("ws_particle_max", 0.0), 0.0)),
        )
    clipped_rows = rows
    if ws_particle_max > 0 and len(clipped_rows) > ws_particle_max:
        clipped_rows = list(clipped_rows[:ws_particle_max])
        presence_dynamics["field_particles"] = clipped_rows
    return [row for row in clipped_rows if isinstance(row, dict)]


def prepare_disabled_or_empty_state(
    simulation: dict[str, Any],
    presence_dynamics: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    disable_daimoi: bool,
    dt_seconds: float,
    safe_float: Callable[[Any, float], float],
    reset_nooi_field_state: Callable[[], None],
    maybe_seed_random_nooi_field_vectors: Callable[[], None],
    nooi_field: Any,
) -> bool:
    if disable_daimoi:
        if rows:
            reset_nooi_field_state()
        maybe_seed_random_nooi_field_vectors()
        dt = max(0.001, safe_float(dt_seconds, 0.08))
        nooi_field.decay(dt)
        presence_dynamics["field_particles"] = []
        presence_dynamics["nooi_field"] = nooi_field.get_grid_snapshot([])
        presence_dynamics.pop("graph_node_positions", None)
        presence_dynamics.pop("presence_anchor_positions", None)
        simulation["presence_dynamics"] = presence_dynamics
        return True

    if not rows:
        maybe_seed_random_nooi_field_vectors()
        dt = max(0.001, safe_float(dt_seconds, 0.08))
        nooi_field.decay(dt)
        presence_dynamics["nooi_field"] = nooi_field.get_grid_snapshot([])
        presence_dynamics.pop("graph_node_positions", None)
        presence_dynamics.pop("presence_anchor_positions", None)
        simulation["presence_dynamics"] = presence_dynamics
        return True

    return False


def stream_friction_context(
    dt_seconds: float,
    *,
    now_seconds: float | None = None,
    safe_float: Callable[[Any, float], float],
    simulation_tick_seconds: Callable[[], float],
    stream_daimoi_friction: float,
    stream_nexus_friction: float,
    stream_daimoi_friction_default: float,
) -> dict[str, float]:
    dt = max(0.001, safe_float(dt_seconds, 0.08))
    base_dt = max(
        0.001,
        safe_float(
            os.getenv("SIM_TICK_SECONDS", str(simulation_tick_seconds()))
            or str(simulation_tick_seconds()),
            simulation_tick_seconds(),
        ),
    )
    now_value = safe_float(now_seconds, time.time())
    daimoi_friction_base = max(
        0.0,
        min(2.0, safe_float(stream_daimoi_friction, stream_daimoi_friction_default)),
    )
    nexus_friction_base = max(
        0.0,
        min(2.0, safe_float(stream_nexus_friction, daimoi_friction_base)),
    )
    daimoi_friction_tick = max(
        0.0,
        min(1.2, daimoi_friction_base ** (dt / base_dt)),
    )
    nexus_friction_tick = max(
        0.0,
        min(1.2, nexus_friction_base ** (dt / base_dt)),
    )
    return {
        "dt": dt,
        "base_dt": base_dt,
        "now_value": now_value,
        "daimoi_friction_tick": daimoi_friction_tick,
        "nexus_friction_tick": nexus_friction_tick,
    }


def gravity_max_from_rows(
    rows: list[dict[str, Any]],
    *,
    safe_float: Callable[[Any, float], float],
) -> float:
    gravity_max = 1e-6
    for row in rows:
        gravity_max = max(
            gravity_max,
            safe_float(row.get("gravity_potential", 0.0), 0.0),
        )
    return gravity_max


def aggregate_presence_centers(
    rows: list[dict[str, Any]],
    *,
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[float], float],
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    presence_centers: dict[str, tuple[float, float]] = {}
    presence_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        presence_id = str(row.get("presence_id", "") or "").strip()
        if not presence_id:
            continue
        x_value = clamp01(safe_float(row.get("x", 0.5), 0.5))
        y_value = clamp01(safe_float(row.get("y", 0.5), 0.5))
        current_x, current_y = presence_centers.get(presence_id, (0.0, 0.0))
        presence_centers[presence_id] = (current_x + x_value, current_y + y_value)
        presence_counts[presence_id] = int(presence_counts.get(presence_id, 0)) + 1

    for presence_id, count in list(presence_counts.items()):
        if count <= 0 or presence_id not in presence_centers:
            continue
        total_x, total_y = presence_centers[presence_id]
        presence_centers[presence_id] = (total_x / count, total_y / count)
    return presence_centers, dict(presence_counts)


def cpu_sentinel_context(
    presence_dynamics: dict[str, Any],
    presence_centers: dict[str, tuple[float, float]],
    *,
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[float], float],
    cpu_sentinel_id: str,
    cpu_sentinel_attractor_start_percent: float,
) -> dict[str, Any]:
    resource_consumption_state = presence_dynamics.get("resource_consumption", {})
    if not isinstance(resource_consumption_state, dict):
        resource_consumption_state = {}
    resource_heartbeat_state = presence_dynamics.get("resource_heartbeat", {})
    if not isinstance(resource_heartbeat_state, dict):
        resource_heartbeat_state = {}
    resource_devices_state = resource_heartbeat_state.get("devices", {})
    if not isinstance(resource_devices_state, dict):
        resource_devices_state = {}
    cpu_device_state = resource_devices_state.get("cpu", {})
    if not isinstance(cpu_device_state, dict):
        cpu_device_state = {}

    cpu_utilization_stream = max(
        0.0,
        min(100.0, safe_float(cpu_device_state.get("utilization", 0.0), 0.0)),
    )
    cpu_sentinel_attractor_active_stream = bool(
        resource_consumption_state.get("cpu_sentinel_burn_active", False)
    ) or (cpu_utilization_stream >= cpu_sentinel_attractor_start_percent)
    cpu_sentinel_pressure_stream = clamp01(
        (cpu_utilization_stream - cpu_sentinel_attractor_start_percent)
        / max(1.0, (100.0 - cpu_sentinel_attractor_start_percent))
    )

    cpu_sentinel_center = presence_centers.get(cpu_sentinel_id)
    if not (isinstance(cpu_sentinel_center, tuple) and len(cpu_sentinel_center) == 2):
        anchor_positions = presence_dynamics.get("presence_anchor_positions", {})
        if isinstance(anchor_positions, dict):
            anchor_state = anchor_positions.get(cpu_sentinel_id)
            if isinstance(anchor_state, dict):
                cpu_sentinel_center = (
                    clamp01(safe_float(anchor_state.get("x", 0.5), 0.5)),
                    clamp01(safe_float(anchor_state.get("y", 0.5), 0.5)),
                )
    if not (isinstance(cpu_sentinel_center, tuple) and len(cpu_sentinel_center) == 2):
        cpu_sentinel_center = None

    return {
        "cpu_utilization_stream": cpu_utilization_stream,
        "cpu_sentinel_attractor_active_stream": cpu_sentinel_attractor_active_stream,
        "cpu_sentinel_pressure_stream": cpu_sentinel_pressure_stream,
        "cpu_sentinel_center": cpu_sentinel_center,
    }


def aggregate_node_centers(
    rows: list[dict[str, Any]],
    *,
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[float], float],
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    node_centers: dict[str, tuple[float, float]] = {}
    node_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        x_value = clamp01(safe_float(row.get("x", 0.5), 0.5))
        y_value = clamp01(safe_float(row.get("y", 0.5), 0.5))
        tokens = {
            str(row.get("graph_node_id", "") or "").strip(),
            str(row.get("route_node_id", "") or "").strip(),
        }
        for token in tokens:
            if not token:
                continue
            total_x, total_y = node_centers.get(token, (0.0, 0.0))
            node_centers[token] = (total_x + x_value, total_y + y_value)
            node_counts[token] = int(node_counts.get(token, 0)) + 1

    for node_id, count in list(node_counts.items()):
        if count <= 0 or node_id not in node_centers:
            continue
        total_x, total_y = node_centers[node_id]
        node_centers[node_id] = (total_x / count, total_y / count)

    return node_centers, dict(node_counts)
