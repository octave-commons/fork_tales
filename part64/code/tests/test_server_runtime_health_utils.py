from __future__ import annotations

from pathlib import Path

from code.world_web.server_runtime_health_utils import (
    runtime_guard_state,
    runtime_health_payload,
    runtime_ws_client_snapshot,
    runtime_ws_release_client_slot,
    runtime_ws_try_acquire_client_slot,
)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def test_runtime_ws_helpers_enforce_capacity() -> None:
    snapshot = runtime_ws_client_snapshot(active_count=1, max_clients=3)
    assert snapshot == {"active_clients": 1, "max_clients": 3}
    acquired, next_count = runtime_ws_try_acquire_client_slot(
        active_count=2, max_clients=2
    )
    assert acquired is False
    assert next_count == 2
    assert runtime_ws_release_client_slot(active_count=1) == 0


def test_runtime_guard_state_reports_critical_and_degraded_modes() -> None:
    critical = runtime_guard_state(
        {
            "devices": {"cpu": {"utilization": 95.0, "memory_pressure": 0.95}},
            "log_watch": {"error_ratio": 0.7},
            "hot_devices": ["cpu"],
        },
        safe_float=_safe_float,
        cpu_utilization_critical=92.0,
        memory_pressure_critical=0.9,
        log_error_ratio_critical=0.55,
    )
    degraded = runtime_guard_state(
        {
            "devices": {"cpu": {"utilization": 30.0, "memory_pressure": 0.2}},
            "log_watch": {"error_ratio": 0.2},
            "hot_devices": ["gpu0"],
        },
        safe_float=_safe_float,
        cpu_utilization_critical=92.0,
        memory_pressure_critical=0.9,
        log_error_ratio_critical=0.55,
    )
    assert critical["mode"] == "critical"
    assert "cpu_hot" in critical["reasons"]
    assert degraded["mode"] == "degraded"
    assert "hot_devices" in degraded["reasons"]


def test_runtime_health_payload_assembles_sections() -> None:
    payload = runtime_health_payload(
        Path("/tmp"),
        resource_monitor_snapshot=lambda _path: {
            "devices": {"cpu": {"utilization": 10.0}}
        },
        guard_state_builder=lambda snapshot: {"mode": "normal", "snapshot": snapshot},
        ws_snapshot_builder=lambda: {"active_clients": 0, "max_clients": 12},
    )
    assert payload["ok"] is True
    assert payload["record"] == "eta-mu.runtime-health.v1"
    assert payload["degraded"] is False
