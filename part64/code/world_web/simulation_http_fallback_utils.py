"""Fallback and error helpers for simulation HTTP responses."""

from __future__ import annotations

import math
from typing import Any, Callable


def simulation_http_fallback_headers(
    *,
    fallback: str,
    error: str = "",
    backoff_remaining_seconds: float | None = None,
    backoff_streak: int | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "X-Eta-Mu-Simulation-Fallback": str(fallback or "").strip() or "stale-cache"
    }
    error_value = str(error or "").strip()
    if error_value:
        headers["X-Eta-Mu-Simulation-Error"] = error_value
    if backoff_remaining_seconds is not None:
        headers["X-Eta-Mu-Simulation-Backoff-Seconds"] = str(
            max(1, int(math.ceil(max(0.0, float(backoff_remaining_seconds)))))
        )
    if backoff_streak is not None:
        headers["X-Eta-Mu-Simulation-Backoff-Streak"] = str(max(0, int(backoff_streak)))
    return headers


def simulation_http_stale_or_disk_body(
    *,
    part_root: Any,
    cache_perspective: str,
    stale_max_age_seconds: float,
    disk_max_age_seconds: float,
    cache_key: str,
    cached_body_reader: Callable[..., bytes | None],
    disk_cache_loader: Callable[..., bytes | None],
    cache_store: Callable[[str, bytes], None],
) -> tuple[bytes | None, str]:
    stale_body = cached_body_reader(
        perspective=cache_perspective,
        max_age_seconds=stale_max_age_seconds,
    )
    if stale_body is not None:
        return stale_body, "stale-cache"

    disk_body = disk_cache_loader(
        part_root,
        perspective=cache_perspective,
        max_age_seconds=max(stale_max_age_seconds, disk_max_age_seconds),
    )
    if disk_body is None:
        return None, ""

    if str(cache_key or "").strip():
        cache_store(cache_key, disk_body)
    return disk_body, "disk-cache"


def simulation_http_compact_stale_fallback_body(
    *,
    part_root: Any,
    cache_perspective: str,
    max_age_seconds: float,
    safe_float: Callable[[Any, float], float],
    cached_body_reader: Callable[..., bytes | None],
    disk_cache_loader: Callable[..., bytes | None],
    cache_store: Callable[[str, bytes], None],
) -> tuple[bytes | None, str]:
    stale_max_age = max(0.0, safe_float(max_age_seconds, 0.0))
    if stale_max_age <= 0.0:
        return None, ""

    perspective_key = str(cache_perspective or "").strip().lower()
    if not perspective_key:
        return None, ""

    stale_body = cached_body_reader(
        perspective=perspective_key,
        max_age_seconds=stale_max_age,
    )
    if stale_body is not None:
        return stale_body, "stale-cache"

    disk_body = disk_cache_loader(
        part_root,
        perspective=perspective_key,
        max_age_seconds=stale_max_age,
    )
    if disk_body is None:
        return None, ""

    cache_store(
        f"{perspective_key}|disk-compact-fallback|simulation",
        disk_body,
    )
    return disk_body, "disk-cache"


def simulation_http_error_payload(
    *,
    perspective: str,
    error: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(error or "simulation_unavailable").strip()
        or "simulation_unavailable",
        "record": "eta-mu.simulation.error.v1",
        "perspective": str(perspective or "").strip(),
        "detail": str(detail or "Exception").strip() or "Exception",
    }
