"""Websocket governor helper utilities for simulation stream control."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable


def simulation_ws_governor_estimate_work(
    simulation_payload: dict[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
) -> float:
    if not isinstance(simulation_payload, dict):
        return 1.0

    dynamics = simulation_payload.get("presence_dynamics", {})
    field_particles = (
        dynamics.get("field_particles", []) if isinstance(dynamics, dict) else []
    )
    particle_count = len(field_particles) if isinstance(field_particles, list) else 0

    points = simulation_payload.get("points", [])
    point_count = len(points) if isinstance(points, list) else 0
    total_count = max(0.0, safe_float(simulation_payload.get("total", 0.0), 0.0))

    estimated = (particle_count * 1.2) + (point_count * 0.08) + (total_count * 0.05)
    return max(1.0, estimated)


def simulation_ws_governor_ingestion_signal(
    catalog: dict[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
) -> tuple[int, int, int, int]:
    if not isinstance(catalog, dict):
        return (0, 0, 0, 0)

    inbox_state = catalog.get("eta_mu_inbox", {})
    if not isinstance(inbox_state, dict):
        inbox_state = {}

    pending_count = max(0, int(safe_float(inbox_state.get("pending_count", 0), 0.0)))
    deferred_count = max(0, int(safe_float(inbox_state.get("deferred_count", 0), 0.0)))

    file_graph_stats = (
        catalog.get("file_graph", {}).get("stats", {})
        if isinstance(catalog.get("file_graph", {}), dict)
        else {}
    )
    if not isinstance(file_graph_stats, dict):
        file_graph_stats = {}

    compressed_total = max(
        0.0,
        safe_float(file_graph_stats.get("compressed_bytes_total", 0.0), 0.0),
    )
    knowledge_entries = max(
        0.0, safe_float(inbox_state.get("knowledge_entries", 0.0), 0.0)
    )
    avg_entry_bytes = (
        compressed_total / knowledge_entries
        if knowledge_entries > 0.0
        else 64.0 * 1024.0
    )
    avg_entry_bytes = max(16.0 * 1024.0, min(6.0 * 1024.0 * 1024.0, avg_entry_bytes))
    bytes_pending = int(max(0.0, pending_count * avg_entry_bytes))

    embedding_backlog = pending_count + deferred_count
    disk_queue_depth = deferred_count
    return (pending_count, bytes_pending, embedding_backlog, disk_queue_depth)


def simulation_ws_governor_stock_pressure(
    part_root: Path,
    *,
    safe_float: Callable[[Any, float], float],
    resource_monitor_snapshot: Callable[[Path], dict[str, Any]],
) -> tuple[float, float]:
    mem_pressure = 0.0
    try:
        resource_snapshot = resource_monitor_snapshot(part_root)
    except Exception:
        resource_snapshot = {}

    if isinstance(resource_snapshot, dict):
        devices = resource_snapshot.get("devices", {})
        if isinstance(devices, dict):
            cpu = devices.get("cpu", {})
            if isinstance(cpu, dict):
                mem_pressure = safe_float(cpu.get("memory_pressure", 0.0), 0.0)

    disk_pressure = 0.0
    try:
        usage = shutil.disk_usage(part_root)
    except OSError:
        usage = None
    if usage is not None and usage.total > 0:
        disk_pressure = usage.used / float(usage.total)

    return (
        max(0.0, min(1.5, mem_pressure)),
        max(0.0, min(1.5, disk_pressure)),
    )


def simulation_ws_governor_particle_cap(
    base_cap: int,
    *,
    fidelity_signal: str,
    ingestion_pressure: float,
    min_particle_cap: int,
    safe_float: Callable[[Any, float], float],
) -> int:
    min_cap = max(1, min_particle_cap)
    max_cap = max(min_cap, int(base_cap))
    pressure = max(0.0, min(1.0, safe_float(ingestion_pressure, 0.0)))

    if fidelity_signal == "decrease":
        scaled = int(round(max_cap * (0.68 - (0.24 * pressure))))
        return max(min_cap, min(max_cap, scaled))
    if fidelity_signal == "increase":
        scaled = int(round(max_cap * (1.0 + (0.12 * (1.0 - pressure)))))
        return max(min_cap, min(max_cap, scaled))
    return max(min_cap, max_cap)


def simulation_ws_governor_graph_heartbeat_scale(
    fidelity_signal: str,
    *,
    degrade_scale: float,
    increase_scale: float,
) -> float:
    if fidelity_signal == "decrease":
        return degrade_scale
    if fidelity_signal == "increase":
        return increase_scale
    return 1.0
