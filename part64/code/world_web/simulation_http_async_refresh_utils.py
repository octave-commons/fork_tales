"""Helpers for simulation HTTP async refresh control state."""

from __future__ import annotations

import copy
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable


def full_async_refresh_snapshot(
    *,
    refresh_state: dict[str, Any],
    refresh_lock: Any,
) -> dict[str, Any]:
    with refresh_lock:
        return copy.deepcopy(refresh_state)


def full_async_refresh_cancel(
    *,
    refresh_state: dict[str, Any],
    refresh_lock: Any,
    reason: str,
    job_id: str = "",
) -> tuple[bool, dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    requested_job_id = str(job_id or "").strip()
    cancel_reason = str(reason or "operator_cancel").strip() or "operator_cancel"

    with refresh_lock:
        running = bool(refresh_state.get("running", False))
        active_job_id = str(refresh_state.get("job_id", "") or "").strip()
        if requested_job_id and active_job_id and requested_job_id != active_job_id:
            return (False, copy.deepcopy(refresh_state))
        if not running:
            return (False, copy.deepcopy(refresh_state))

        refresh_state["running"] = False
        refresh_state["status"] = "canceled"
        refresh_state["updated_at"] = now_iso
        refresh_state["completed_at"] = now_iso
        refresh_state["error"] = cancel_reason
        refresh_state["started_monotonic"] = 0.0
        snapshot = copy.deepcopy(refresh_state)

    return (True, snapshot)


def full_async_throttle_remaining_seconds(
    snapshot: dict[str, Any],
    *,
    perspective: str,
    min_interval_seconds: float,
    normalize_projection_perspective: Callable[[str], str],
    safe_float: Callable[[Any, float], float],
    now_monotonic: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> float:
    min_interval = max(0.0, safe_float(min_interval_seconds, 0.0))
    if min_interval <= 0.0:
        return 0.0

    perspective_key = normalize_projection_perspective(str(perspective or ""))
    if not perspective_key:
        return 0.0

    rows = snapshot.get("last_start_monotonic_by_perspective", {})
    if not isinstance(rows, dict):
        return 0.0

    last_start = safe_float(rows.get(perspective_key, 0.0), 0.0)
    if last_start <= 0.0:
        return 0.0

    now_mono = (
        max(0.0, safe_float(now_monotonic, 0.0))
        if now_monotonic is not None
        else monotonic_fn()
    )
    elapsed = max(0.0, now_mono - last_start)
    return max(0.0, min_interval - elapsed)


def full_async_refresh_headers(
    snapshot: dict[str, Any],
    *,
    scheduled: bool,
    safe_float: Callable[[Any, float], float],
) -> dict[str, str]:
    status_value = str(snapshot.get("status", "idle") or "idle").strip() or "idle"
    running = bool(snapshot.get("running", False))
    refresh_state = "scheduled" if scheduled else "running" if running else status_value
    headers = {
        "X-Eta-Mu-Simulation-Refresh": refresh_state,
        "X-Eta-Mu-Simulation-Refresh-Status": status_value,
    }
    job_id = str(snapshot.get("job_id", "") or "").strip()
    if job_id:
        headers["X-Eta-Mu-Simulation-Refresh-Job"] = job_id
    updated_at = str(snapshot.get("updated_at", "") or "").strip()
    if updated_at:
        headers["X-Eta-Mu-Simulation-Refresh-Updated-At"] = updated_at
    remaining_seconds = max(
        0.0, safe_float(snapshot.get("throttle_remaining_seconds", 0.0), 0.0)
    )
    if status_value == "throttled" and remaining_seconds > 0.0:
        headers["Retry-After"] = str(max(1, int(math.ceil(remaining_seconds))))
        headers["X-Eta-Mu-Simulation-Refresh-Retry-After-Seconds"] = str(
            round(remaining_seconds, 3)
        )
    return headers


def full_async_refresh_start(
    *,
    refresh_state: dict[str, Any],
    refresh_lock: Any,
    perspective: str,
    cache_perspective: str,
    cache_key: str,
    trigger: str,
    runner: Callable[[], None],
    allow_throttle_bypass: bool,
    default_perspective: str,
    max_running_seconds: float,
    min_interval_seconds: float,
    safe_float: Callable[[Any, float], float],
    normalize_projection_perspective: Callable[[str], str],
    thread_name: str,
    monotonic_fn: Callable[[], float] = time.monotonic,
    time_seconds_fn: Callable[[], float] = time.time,
) -> tuple[bool, dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    now_monotonic = monotonic_fn()
    job_id = f"sim-full:{int(time_seconds_fn() * 1000):x}"
    perspective_key = normalize_projection_perspective(str(perspective or ""))

    with refresh_lock:
        if bool(refresh_state.get("running", False)):
            started_monotonic = safe_float(
                refresh_state.get("started_monotonic", 0.0), 0.0
            )
            running_age = (
                max(0.0, now_monotonic - started_monotonic)
                if started_monotonic > 0.0
                else 0.0
            )
            if started_monotonic > 0.0 and running_age > max_running_seconds:
                refresh_state["running"] = False
                refresh_state["status"] = "timeout"
                refresh_state["updated_at"] = now_iso
                refresh_state["completed_at"] = now_iso
                refresh_state["error"] = (
                    f"refresh_timeout:{max(1, int(math.ceil(running_age)))}s"
                )
                refresh_state["started_monotonic"] = 0.0
            else:
                refresh_state["updated_at"] = now_iso
                refresh_state["trigger"] = (
                    str(trigger or "full-fallback").strip() or "full-fallback"
                )
                return (False, copy.deepcopy(refresh_state))

        throttle_rows = refresh_state.get("last_start_monotonic_by_perspective", {})
        if not isinstance(throttle_rows, dict):
            throttle_rows = {}
            refresh_state["last_start_monotonic_by_perspective"] = throttle_rows

        min_interval = max(0.0, safe_float(min_interval_seconds, 0.0))
        if (
            not allow_throttle_bypass
            and min_interval > 0.0
            and perspective_key
            and isinstance(throttle_rows, dict)
        ):
            last_start = safe_float(throttle_rows.get(perspective_key, 0.0), 0.0)
            if last_start > 0.0:
                remaining = max(0.0, min_interval - (now_monotonic - last_start))
                if remaining > 0.0:
                    snapshot = copy.deepcopy(refresh_state)
                    snapshot["status"] = "throttled"
                    snapshot["throttle_remaining_seconds"] = round(remaining, 3)
                    snapshot["throttle_perspective"] = perspective_key
                    snapshot["throttled"] = True
                    return (False, snapshot)

        refresh_state["running"] = True
        refresh_state["status"] = "running"
        refresh_state["job_id"] = job_id
        refresh_state["trigger"] = (
            str(trigger or "full-fallback").strip() or "full-fallback"
        )
        refresh_state["perspective"] = (
            str(perspective or "").strip() or default_perspective
        )
        refresh_state["cache_perspective"] = (
            str(cache_perspective or "").strip() or default_perspective
        )
        refresh_state["cache_key"] = str(cache_key or "").strip()
        if perspective_key and isinstance(throttle_rows, dict):
            throttle_rows[perspective_key] = now_monotonic
        refresh_state["started_at"] = now_iso
        refresh_state["started_monotonic"] = now_monotonic
        refresh_state["updated_at"] = now_iso
        refresh_state["completed_at"] = ""
        refresh_state["error"] = ""
        snapshot = copy.deepcopy(refresh_state)

    def _run() -> None:
        status = "ok"
        error_detail = ""
        try:
            runner()
        except Exception as exc:
            status = "error"
            error_detail = f"{exc.__class__.__name__}: {exc}"
        completed_at = datetime.now(timezone.utc).isoformat()
        with refresh_lock:
            active_job_id = str(refresh_state.get("job_id", "") or "").strip()
            if active_job_id != job_id:
                return
            refresh_state["running"] = False
            refresh_state["status"] = status
            refresh_state["updated_at"] = completed_at
            refresh_state["completed_at"] = completed_at
            refresh_state["error"] = error_detail
            refresh_state["started_monotonic"] = 0.0

    threading.Thread(target=_run, daemon=True, name=thread_name).start()
    return (True, snapshot)
