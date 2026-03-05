"""State/cache helpers for simulation HTTP paths."""

from __future__ import annotations

import time
from typing import Any, Callable


def cache_store(
    *,
    cache_state: dict[str, Any],
    cache_lock: Any,
    cache_key: str,
    body: bytes,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> None:
    if not cache_key:
        return
    if not isinstance(body, (bytes, bytearray)):
        return
    body_bytes = bytes(body)
    if not body_bytes:
        return
    with cache_lock:
        cache_state["key"] = cache_key
        cache_state["prepared_monotonic"] = monotonic_fn()
        cache_state["body"] = body_bytes


def runtime_catalog_http_cache_store(
    *,
    cache_state: dict[str, dict[str, Any]],
    cache_lock: Any,
    perspective: str,
    body: bytes,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> None:
    perspective_key = str(perspective or "").strip().lower()
    if not perspective_key:
        return
    if not isinstance(body, (bytes, bytearray)):
        return
    body_bytes = bytes(body)
    if not body_bytes:
        return
    with cache_lock:
        cache_state[perspective_key] = {
            "prepared_monotonic": monotonic_fn(),
            "body": body_bytes,
        }


def runtime_catalog_http_cached_body(
    *,
    cache_state: dict[str, dict[str, Any]],
    cache_lock: Any,
    perspective: str,
    max_age_seconds: float,
    safe_float: Callable[[Any, float], float],
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> bytes | None:
    if max_age_seconds <= 0.0:
        return None
    perspective_key = str(perspective or "").strip().lower()
    if not perspective_key:
        return None

    with cache_lock:
        cache_row = cache_state.get(perspective_key)
        row = dict(cache_row) if isinstance(cache_row, dict) else None

    if not isinstance(row, dict):
        return None
    cached_body = row.get("body", b"")
    if not isinstance(cached_body, (bytes, bytearray)) or not cached_body:
        return None
    cached_age = monotonic_fn() - safe_float(row.get("prepared_monotonic", 0.0), 0.0)
    if cached_age < 0.0 or cached_age > max_age_seconds:
        return None
    return bytes(cached_body)


def runtime_catalog_http_cache_invalidate(
    *,
    cache_state: dict[str, dict[str, Any]],
    cache_lock: Any,
) -> None:
    with cache_lock:
        cache_state.clear()


def cache_reset(*, cache_state: dict[str, Any], cache_lock: Any) -> None:
    with cache_lock:
        cache_state["key"] = ""
        cache_state["prepared_monotonic"] = 0.0
        cache_state["body"] = b""


def cache_cached_body(
    *,
    cache_state: dict[str, Any],
    cache_lock: Any,
    cache_key: str = "",
    perspective: str = "",
    max_age_seconds: float,
    require_exact_key: bool = False,
    match_requested_key_when_not_exact: bool = True,
    safe_float: Callable[[Any, float], float],
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> bytes | None:
    if max_age_seconds <= 0.0:
        return None

    requested_key = str(cache_key or "").strip()
    requested_perspective = str(perspective or "").strip()
    with cache_lock:
        cached_key = str(cache_state.get("key", "") or "").strip()
        cached_body = cache_state.get("body", b"")
        cached_age = monotonic_fn() - safe_float(
            cache_state.get("prepared_monotonic", 0.0), 0.0
        )

    if not cached_key:
        return None
    if cached_age < 0.0 or cached_age > max_age_seconds:
        return None
    if not isinstance(cached_body, (bytes, bytearray)) or not cached_body:
        return None

    if require_exact_key:
        if not requested_key or requested_key != cached_key:
            return None
    elif match_requested_key_when_not_exact and requested_key:
        if requested_key != cached_key:
            return None
    elif requested_perspective and not cached_key.startswith(
        f"{requested_perspective}|"
    ):
        return None

    return bytes(cached_body)


def wait_for_exact_cache(
    *,
    cache_key: str,
    perspective: str,
    max_wait_seconds: float,
    poll_seconds: float,
    max_cache_age_seconds: float,
    cached_body_reader: Callable[..., bytes | None],
    safe_float: Callable[[Any, float], float],
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bytes | None:
    wait_window = max(0.0, safe_float(max_wait_seconds, 0.0))
    if wait_window <= 0.0:
        return None

    poll_interval = max(0.01, safe_float(poll_seconds, 0.05))
    deadline = monotonic_fn() + wait_window

    while True:
        cached_body = cached_body_reader(
            cache_key=cache_key,
            perspective=perspective,
            max_age_seconds=max_cache_age_seconds,
            require_exact_key=True,
        )
        if cached_body is not None:
            return cached_body

        now_monotonic = monotonic_fn()
        if now_monotonic >= deadline:
            return None
        sleep_fn(min(poll_interval, max(0.0, deadline - now_monotonic)))


def is_cold_start(
    *,
    disk_cold_start_seconds: float,
    server_boot_monotonic: float,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> bool:
    if disk_cold_start_seconds <= 0.0:
        return False
    uptime = monotonic_fn() - server_boot_monotonic
    return 0.0 <= uptime <= disk_cold_start_seconds


def failure_backoff_snapshot(
    *,
    failure_state: dict[str, Any],
    failure_lock: Any,
    cooldown_seconds: float,
    safe_float: Callable[[Any, float], float],
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> tuple[float, str, int]:
    cooldown = max(0.0, safe_float(cooldown_seconds, 0.0))
    if cooldown <= 0.0:
        return (0.0, "", 0)

    with failure_lock:
        last_failure = safe_float(failure_state.get("last_failure_monotonic", 0.0), 0.0)
        streak = max(0, int(safe_float(failure_state.get("streak", 0), 0.0)))
        error_name = str(failure_state.get("last_error", "") or "").strip()

    if last_failure <= 0.0:
        return (0.0, "", 0)

    age = monotonic_fn() - last_failure
    if age < 0.0:
        age = 0.0
    remaining = max(0.0, cooldown - age)
    return (remaining, error_name, streak)


def failure_record(
    *,
    failure_state: dict[str, Any],
    failure_lock: Any,
    error_name: str,
    streak_reset_seconds: float,
    safe_float: Callable[[Any, float], float],
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> None:
    now_monotonic = monotonic_fn()
    reset_window = max(0.0, safe_float(streak_reset_seconds, 0.0))
    with failure_lock:
        previous_failure = safe_float(
            failure_state.get("last_failure_monotonic", 0.0), 0.0
        )
        streak = int(safe_float(failure_state.get("streak", 0), 0.0))
        if previous_failure > 0.0 and reset_window > 0.0:
            if (now_monotonic - previous_failure) > reset_window:
                streak = 0
        failure_state["streak"] = max(0, streak) + 1
        failure_state["last_failure_monotonic"] = now_monotonic
        failure_state["last_error"] = (
            str(error_name or "Exception").strip() or "Exception"
        )


def failure_clear(*, failure_state: dict[str, Any], failure_lock: Any) -> None:
    with failure_lock:
        failure_state["last_failure_monotonic"] = 0.0
        failure_state["last_error"] = ""
        failure_state["streak"] = 0
