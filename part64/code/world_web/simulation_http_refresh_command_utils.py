"""Helpers for /api/simulation/refresh command handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SimulationHttpRefreshCommandContext:
    action: str
    perspective: str
    cache_perspective: str


def simulation_http_refresh_command_context(
    req: dict[str, Any],
    *,
    default_perspective: str,
    normalize_projection_perspective: Callable[[str], str],
) -> SimulationHttpRefreshCommandContext:
    action = str(req.get("action", "start") or "start").strip().lower()
    perspective = normalize_projection_perspective(
        str(req.get("perspective", default_perspective) or "")
    )
    cache_perspective = f"{perspective}|profile:full"
    return SimulationHttpRefreshCommandContext(
        action=action,
        perspective=perspective,
        cache_perspective=cache_perspective,
    )


def simulation_http_refresh_status_payload(
    *,
    perspective: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "record": "eta-mu.simulation.refresh.command.v1",
        "action": "status",
        "perspective": perspective,
        "refresh": snapshot,
    }


def simulation_http_refresh_cancel_payload(
    *,
    perspective: str,
    canceled: bool,
    snapshot: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    return (
        (200 if canceled else 409),
        {
            "ok": True,
            "record": "eta-mu.simulation.refresh.command.v1",
            "action": "cancel",
            "status": "canceled" if canceled else "no-op",
            "perspective": perspective,
            "refresh": snapshot,
        },
    )


def simulation_http_refresh_start_control(
    req: dict[str, Any],
    *,
    cache_perspective: str,
    safe_bool_query: Callable[[str, bool], bool],
) -> tuple[bool, str, str]:
    force = safe_bool_query(
        str(req.get("force", "false") or "false"),
        default=False,
    )
    trigger_label = str(req.get("trigger", "manual") or "manual").strip()
    trigger_value = f"manual:{trigger_label}" if trigger_label else "manual"
    cache_key_hint = (
        str(req.get("cache_key", "") or "").strip()
        or f"{cache_perspective}|manual|full"
    )
    return force, trigger_value, cache_key_hint


def simulation_http_refresh_start_payload(
    *,
    perspective: str,
    scheduled: bool,
    snapshot: dict[str, Any],
    safe_float: Callable[[Any, float], float],
) -> tuple[int, dict[str, Any]]:
    refresh_status_value = str(snapshot.get("status", "") or "").strip().lower()
    response_status = 202 if scheduled else 200
    response_state = "scheduled" if scheduled else "running"
    if not scheduled:
        if refresh_status_value == "throttled":
            response_status = 429
            response_state = "throttled"
        elif refresh_status_value:
            response_state = refresh_status_value

    payload: dict[str, Any] = {
        "ok": True,
        "record": "eta-mu.simulation.refresh.command.v1",
        "action": "start",
        "status": response_state,
        "perspective": perspective,
        "refresh": snapshot,
    }
    if response_state == "throttled":
        remaining = max(
            0.0,
            safe_float(snapshot.get("throttle_remaining_seconds", 0.0), 0.0),
        )
        if remaining > 0.0:
            payload["retry_after_seconds"] = round(remaining, 3)
    return response_status, payload


def simulation_http_refresh_disabled_payload() -> dict[str, Any]:
    return {
        "ok": False,
        "error": "full_async_refresh_disabled",
        "record": "eta-mu.simulation.refresh.command.v1",
    }


def simulation_http_refresh_unsupported_payload() -> dict[str, Any]:
    return {
        "ok": False,
        "error": "unsupported_refresh_action",
        "record": "eta-mu.simulation.refresh.command.v1",
        "allowed_actions": ["start", "cancel", "status"],
    }
