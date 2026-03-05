"""Build-lock gating helpers for /api/simulation HTTP flow."""

from __future__ import annotations

from typing import Any, Callable


def simulation_http_acquire_build_lock_or_respond(
    *,
    build_lock: Any,
    compact_response_mode: bool,
    compact_build_wait_seconds: float,
    build_wait_seconds: float,
    cache_key: str,
    cache_perspective: str,
    stale_fallback_seconds: float,
    disk_cache_seconds: float,
    build_lock_acquire_timeout_seconds: float,
    part_root: Any,
    wait_for_exact_cache: Callable[..., bytes | None],
    cached_body_reader: Callable[..., bytes | None],
    disk_cache_loader: Callable[..., bytes | None],
    cache_store: Callable[[str, bytes], None],
    send_simulation_response: Callable[..., None],
    fallback_headers_builder: Callable[..., dict[str, str]],
    error_payload_builder: Callable[..., dict[str, Any]],
    send_json: Callable[..., None],
    perspective: str,
    service_unavailable_status: int,
) -> bool:
    lock_acquired = bool(build_lock.acquire(blocking=False))
    if lock_acquired:
        return True

    wait_seconds = (
        compact_build_wait_seconds if compact_response_mode else build_wait_seconds
    )
    inflight_body = wait_for_exact_cache(
        cache_key=cache_key,
        perspective=cache_perspective,
        max_wait_seconds=wait_seconds,
    )
    if inflight_body is not None:
        send_simulation_response(
            inflight_body,
            cache_key_for_compact=cache_key,
            extra_headers=fallback_headers_builder(fallback="inflight-cache"),
        )
        return False

    stale_body = cached_body_reader(
        perspective=cache_perspective,
        max_age_seconds=stale_fallback_seconds,
    )
    if stale_body is not None:
        send_simulation_response(
            stale_body,
            cache_key_for_compact=cache_key,
            extra_headers=fallback_headers_builder(
                fallback="stale-cache",
                error="build_inflight",
            ),
        )
        return False

    lock_acquired = bool(build_lock.acquire(timeout=build_lock_acquire_timeout_seconds))
    if lock_acquired:
        return True

    disk_stale_body = disk_cache_loader(
        part_root,
        perspective=cache_perspective,
        max_age_seconds=max(stale_fallback_seconds, disk_cache_seconds),
    )
    if disk_stale_body is not None:
        cache_store(cache_key, disk_stale_body)
        send_simulation_response(
            disk_stale_body,
            cache_key_for_compact=cache_key,
            extra_headers=fallback_headers_builder(
                fallback="disk-cache",
                error="build_lock_timeout",
            ),
        )
        return False

    send_json(
        error_payload_builder(
            perspective=perspective,
            error="simulation_build_busy",
            detail="build_lock_timeout",
        ),
        status=service_unavailable_status,
    )
    return False


def simulation_http_send_inflight_cached_response_if_any(
    *,
    cache_key: str,
    cache_perspective: str,
    cache_seconds: float,
    cached_body_reader: Callable[..., bytes | None],
    send_simulation_response: Callable[..., None],
    fallback_headers_builder: Callable[..., dict[str, str]],
) -> bool:
    cached_body = cached_body_reader(
        cache_key=cache_key,
        perspective=cache_perspective,
        max_age_seconds=cache_seconds,
        require_exact_key=True,
    )
    if cached_body is None:
        return False

    send_simulation_response(
        cached_body,
        cache_key_for_compact=cache_key,
        extra_headers=fallback_headers_builder(fallback="inflight-cache"),
    )
    return True
