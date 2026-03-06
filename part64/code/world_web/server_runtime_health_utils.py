from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def runtime_ws_client_snapshot(
    *, active_count: int, max_clients: int
) -> dict[str, int]:
    return {
        "active_clients": int(active_count),
        "max_clients": int(max_clients),
    }


def runtime_ws_try_acquire_client_slot(
    *, active_count: int, max_clients: int
) -> tuple[bool, int]:
    if int(active_count) >= int(max_clients):
        return False, int(active_count)
    return True, int(active_count) + 1


def runtime_ws_release_client_slot(*, active_count: int) -> int:
    return max(0, int(active_count) - 1)


def runtime_guard_state(
    resource_snapshot: dict[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
    cpu_utilization_critical: float,
    memory_pressure_critical: float,
    log_error_ratio_critical: float,
) -> dict[str, Any]:
    snapshot = resource_snapshot if isinstance(resource_snapshot, dict) else {}
    devices = (
        snapshot.get("devices", {}) if isinstance(snapshot.get("devices"), dict) else {}
    )
    cpu = devices.get("cpu", {}) if isinstance(devices.get("cpu"), dict) else {}
    log_watch = (
        snapshot.get("log_watch", {})
        if isinstance(snapshot.get("log_watch"), dict)
        else {}
    )

    cpu_utilization = safe_float(cpu.get("utilization", 0.0), 0.0)
    memory_pressure = safe_float(cpu.get("memory_pressure", 0.0), 0.0)
    error_ratio = safe_float(log_watch.get("error_ratio", 0.0), 0.0)
    hot_devices = [
        str(item).strip()
        for item in snapshot.get("hot_devices", [])
        if str(item).strip()
    ]

    reasons: list[str] = []
    mode = "normal"

    if cpu_utilization >= cpu_utilization_critical:
        mode = "critical"
        reasons.append("cpu_hot")
    if memory_pressure >= memory_pressure_critical:
        mode = "critical"
        reasons.append("memory_pressure_high")
    if error_ratio >= log_error_ratio_critical:
        mode = "critical"
        reasons.append("runtime_log_error_ratio_high")

    if mode == "normal":
        if hot_devices:
            mode = "degraded"
            reasons.append("hot_devices")
        if error_ratio >= (log_error_ratio_critical * 0.65):
            mode = "degraded"
            reasons.append("runtime_log_warning_ratio")
        if cpu_utilization >= (cpu_utilization_critical * 0.84):
            mode = "degraded"
            reasons.append("cpu_watch")
        if memory_pressure >= (memory_pressure_critical * 0.85):
            mode = "degraded"
            reasons.append("memory_pressure_watch")

    return {
        "mode": mode,
        "reasons": reasons,
        "cpu_utilization": round(cpu_utilization, 2),
        "memory_pressure": round(memory_pressure, 4),
        "log_error_ratio": round(error_ratio, 4),
        "hot_devices": hot_devices,
        "critical_thresholds": {
            "cpu_utilization": cpu_utilization_critical,
            "memory_pressure": memory_pressure_critical,
            "log_error_ratio": log_error_ratio_critical,
        },
    }


def runtime_health_payload(
    part_root: Path,
    *,
    resource_monitor_snapshot: Callable[[Path], dict[str, Any]],
    guard_state_builder: Callable[[dict[str, Any]], dict[str, Any]],
    ws_snapshot_builder: Callable[[], dict[str, int]],
) -> dict[str, Any]:
    resource_snapshot = resource_monitor_snapshot(part_root)
    guard = guard_state_builder(resource_snapshot)
    ws = ws_snapshot_builder()
    return {
        "ok": True,
        "record": "eta-mu.runtime-health.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guard": guard,
        "websocket": ws,
        "degraded": str(guard.get("mode", "normal")) != "normal",
    }
