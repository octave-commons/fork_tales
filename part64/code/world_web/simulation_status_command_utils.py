"""Helpers for simulation status command payload shaping."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass(frozen=True)
class SimulationRefreshStatusContext:
    perspective: str
    cache_perspective: str


def simulation_bootstrap_status_payload(
    *,
    job_snapshot: dict[str, Any],
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "record": "eta-mu.simulation-bootstrap.status.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job": job_snapshot,
        "report": report,
    }


def simulation_refresh_status_context(
    params: dict[str, list[str]],
    *,
    default_perspective: str,
    normalize_projection_perspective: Callable[[str], str],
) -> SimulationRefreshStatusContext:
    perspective = normalize_projection_perspective(
        str(
            params.get(
                "perspective",
                [default_perspective],
            )[0]
            or default_perspective
        )
    )
    cache_perspective = f"{perspective}|profile:full"
    return SimulationRefreshStatusContext(
        perspective=perspective,
        cache_perspective=cache_perspective,
    )


def simulation_refresh_public_snapshot(
    refresh_snapshot: dict[str, Any],
    *,
    perspective: str,
    throttle_remaining_seconds: Callable[[dict[str, Any], str], float],
    safe_float: Callable[[Any, float], float],
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    started_monotonic = safe_float(
        refresh_snapshot.get("started_monotonic", 0.0),
        0.0,
    )
    running = bool(refresh_snapshot.get("running", False))
    refresh_public = dict(refresh_snapshot)
    refresh_public.pop("started_monotonic", None)
    refresh_public.pop("last_start_monotonic_by_perspective", None)
    if running and started_monotonic > 0.0:
        refresh_public["running_for_seconds"] = round(
            max(0.0, monotonic_fn() - started_monotonic),
            3,
        )

    throttle_remaining = throttle_remaining_seconds(refresh_snapshot, perspective)
    if throttle_remaining > 0.0:
        refresh_public["throttle_remaining_seconds"] = round(throttle_remaining, 3)
    return refresh_public


def simulation_refresh_status_availability(
    *,
    part_root: Any,
    cache_perspective: str,
    cache_seconds: float,
    stale_max_age_seconds: float,
    disk_cache_seconds: float,
    cached_body_reader: Callable[..., bytes | None],
    disk_cache_has_payload: Callable[..., bool],
) -> dict[str, bool]:
    return {
        "fresh_cache": cached_body_reader(
            perspective=cache_perspective,
            max_age_seconds=cache_seconds,
        )
        is not None,
        "stale_cache": cached_body_reader(
            perspective=cache_perspective,
            max_age_seconds=stale_max_age_seconds,
        )
        is not None,
        "disk_cache": disk_cache_has_payload(
            part_root,
            perspective=cache_perspective,
            max_age_seconds=max(stale_max_age_seconds, disk_cache_seconds),
        ),
    }


def simulation_refresh_status_payload(
    *,
    perspective: str,
    full_async_enabled: bool,
    cache_seconds: float,
    stale_max_age_seconds: float,
    lock_timeout_seconds: float,
    max_running_seconds: float,
    start_min_interval_seconds: float,
    availability: dict[str, bool],
    refresh_public: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "record": "eta-mu.simulation.refresh-status.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "perspective": perspective,
        "profile": "full",
        "full_async_enabled": bool(full_async_enabled),
        "timing": {
            "cache_seconds": round(cache_seconds, 3),
            "stale_max_age_seconds": round(stale_max_age_seconds, 3),
            "lock_timeout_seconds": round(lock_timeout_seconds, 3),
            "max_running_seconds": round(max_running_seconds, 3),
            "start_min_interval_seconds": round(start_min_interval_seconds, 3),
        },
        "availability": dict(availability),
        "refresh": dict(refresh_public),
    }
