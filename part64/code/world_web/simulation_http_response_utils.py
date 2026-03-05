"""Response flow helpers for /api/simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SimulationHttpRequestProfile:
    perspective: str
    compact_response_mode: bool
    full_async_preferred: bool
    cache_profile: str
    cache_perspective: str


def simulation_http_request_profile(
    params: dict[str, list[str]],
    *,
    default_perspective: str,
    full_async_rebuild_enabled: bool,
    normalize_projection_perspective: Callable[[str], str],
    normalize_payload_mode: Callable[[str], str],
    safe_bool_query: Callable[[str, bool], bool],
) -> SimulationHttpRequestProfile:
    perspective = normalize_projection_perspective(
        str(params.get("perspective", [default_perspective])[0] or default_perspective)
    )
    compact_response = safe_bool_query(
        str(params.get("compact", ["false"])[0] or "false"),
        False,
    )
    payload_mode = normalize_payload_mode(
        str(params.get("payload", ["full"])[0] or "full")
    )
    compact_response_mode = bool(compact_response or payload_mode == "trimmed")
    full_response_mode = not compact_response_mode
    full_async_preferred = (
        full_response_mode
        and full_async_rebuild_enabled
        and not safe_bool_query(
            str(params.get("wait", ["false"])[0] or "false"),
            default=False,
        )
    )
    cache_profile = "compact" if compact_response_mode else "full"
    cache_perspective = f"{perspective}|profile:{cache_profile}"
    return SimulationHttpRequestProfile(
        perspective=perspective,
        compact_response_mode=compact_response_mode,
        full_async_preferred=full_async_preferred,
        cache_profile=cache_profile,
        cache_perspective=cache_perspective,
    )


def simulation_http_schedule_full_async_refresh(
    *,
    full_async_preferred: bool,
    cache_key: str,
    cache_perspective: str,
    perspective: str,
    trigger: str,
    schedule_refresh: Callable[..., tuple[bool, dict[str, Any]]],
    refresh_snapshot: Callable[[], dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    if not full_async_preferred:
        return (False, refresh_snapshot())

    cache_key_hint = str(cache_key).strip() or f"{cache_perspective}|pending|full"
    return schedule_refresh(
        perspective=perspective,
        cache_perspective=cache_perspective,
        cache_key_hint=cache_key_hint,
        trigger=trigger,
    )


def simulation_http_send_response(
    body: bytes,
    *,
    compact_response_mode: bool,
    cache_key_for_compact: str,
    cache_seconds: float,
    compact_cached_body_reader: Callable[..., bytes | None],
    compact_cache_store: Callable[[str, bytes], None],
    compact_response_body_builder: Callable[[bytes], bytes],
    send_bytes: Callable[..., None],
    content_type: str,
    extra_headers: dict[str, str] | None = None,
) -> None:
    response_body = body
    if compact_response_mode:
        compact_cache_key = (
            f"{str(cache_key_for_compact).strip()}|compact"
            if str(cache_key_for_compact).strip()
            else ""
        )
        if compact_cache_key:
            cached_compact = compact_cached_body_reader(
                cache_key=compact_cache_key,
                max_age_seconds=cache_seconds,
            )
            if cached_compact is not None:
                response_body = cached_compact
            else:
                response_body = compact_response_body_builder(body)
                compact_cache_store(compact_cache_key, response_body)
        else:
            response_body = compact_response_body_builder(body)

    send_bytes(
        response_body,
        content_type,
        extra_headers=extra_headers,
    )


def simulation_http_refresh_retry_payload(
    *,
    refresh_snapshot: dict[str, Any],
    refresh_scheduled: bool,
    perspective: str,
    safe_float: Callable[[Any, float], float],
) -> tuple[int, dict[str, Any]]:
    refresh_status_value = str(refresh_snapshot.get("status", "") or "").strip().lower()
    response_status = 202
    response_state = "scheduled" if refresh_scheduled else "running"
    if not refresh_scheduled:
        if refresh_status_value == "throttled":
            response_status = 429
            response_state = "throttled"
        elif refresh_status_value:
            response_state = refresh_status_value

    response_payload: dict[str, Any] = {
        "ok": True,
        "record": "eta-mu.simulation.refresh.v1",
        "status": response_state,
        "perspective": perspective,
        "payload": "full",
        "refresh": refresh_snapshot,
    }
    if response_state == "throttled":
        remaining = max(
            0.0,
            safe_float(refresh_snapshot.get("throttle_remaining_seconds", 0.0), 0.0),
        )
        if remaining > 0.0:
            response_payload["retry_after_seconds"] = round(remaining, 3)

    return response_status, response_payload
