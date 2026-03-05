"""Helpers for /api/simulation/bootstrap command handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass(frozen=True)
class SimulationBootstrapCommandContext:
    perspective: str
    normalized_perspective: str
    sync_inbox: bool
    include_simulation_payload: bool
    wait: bool


def simulation_bootstrap_command_context(
    req: dict[str, Any],
    *,
    default_perspective: str,
    normalize_projection_perspective: Callable[[str], str],
    safe_bool_query: Callable[[str, bool], bool],
) -> SimulationBootstrapCommandContext:
    perspective = str(req.get("perspective", default_perspective) or "")
    normalized_perspective = normalize_projection_perspective(perspective)
    sync_inbox = safe_bool_query(
        str(req.get("sync_inbox", "false") or "false"),
        default=False,
    )
    include_simulation_payload = safe_bool_query(
        str(req.get("include_simulation", "false") or "false"),
        default=False,
    )
    wait = safe_bool_query(
        str(req.get("wait", "false") or "false"),
        default=False,
    )
    return SimulationBootstrapCommandContext(
        perspective=perspective,
        normalized_perspective=normalized_perspective,
        sync_inbox=sync_inbox,
        include_simulation_payload=include_simulation_payload,
        wait=wait,
    )


def simulation_bootstrap_request_payload(
    context: SimulationBootstrapCommandContext,
) -> dict[str, Any]:
    return {
        "perspective": context.normalized_perspective,
        "sync_inbox": bool(context.sync_inbox),
        "include_simulation": bool(context.include_simulation_payload),
    }


def simulation_bootstrap_queue_payload(job_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "record": "eta-mu.simulation-bootstrap.queue.v1",
        "status": "running",
        "job": job_snapshot,
    }


def simulation_bootstrap_failure_report(
    *,
    perspective: str,
    exc: Exception,
    normalize_projection_perspective: Callable[[str], str],
) -> dict[str, Any]:
    error = f"simulation_bootstrap_failed:{exc.__class__.__name__}"
    return {
        "ok": False,
        "record": "eta-mu.simulation-bootstrap.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "perspective": normalize_projection_perspective(perspective),
        "error": error,
        "detail": f"{exc.__class__.__name__}: {exc}",
    }


def simulation_bootstrap_thread_name(job_id: str) -> str:
    suffix = job_id.split(":")[-1][:8] if job_id else "job"
    return f"simulation-bootstrap-{suffix}"
