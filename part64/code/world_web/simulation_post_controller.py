from __future__ import annotations

import threading
from http import HTTPStatus
from typing import Any, Callable

from . import (
    simulation_bootstrap_command_utils as simulation_bootstrap_command_utils_module,
)
from . import (
    simulation_http_refresh_command_utils as simulation_http_refresh_command_utils_module,
)


def handle_simulation_refresh_post(
    *,
    req: dict[str, Any],
    default_perspective: str,
    normalize_projection_perspective: Callable[[str], str],
    safe_bool_query: Callable[[str, bool], bool],
    full_async_refresh_enabled: bool,
    full_async_refresh_snapshot: Callable[[], dict[str, Any]],
    full_async_refresh_cancel: Callable[..., tuple[bool, dict[str, Any]]],
    schedule_full_simulation_async_refresh: Callable[..., tuple[bool, dict[str, Any]]],
    safe_float: Callable[[Any, float], float],
    send_json: Callable[..., None],
) -> None:
    refresh_context = simulation_http_refresh_command_utils_module.simulation_http_refresh_command_context(
        req,
        default_perspective=default_perspective,
        normalize_projection_perspective=normalize_projection_perspective,
    )
    action = refresh_context.action
    perspective = refresh_context.perspective
    cache_perspective = refresh_context.cache_perspective

    if action in {"status", "snapshot"}:
        snapshot = full_async_refresh_snapshot()
        send_json(
            simulation_http_refresh_command_utils_module.simulation_http_refresh_status_payload(
                perspective=perspective,
                snapshot=snapshot,
            )
        )
        return

    if action == "cancel":
        canceled, snapshot = full_async_refresh_cancel(
            reason=str(req.get("reason", "operator_cancel") or "operator_cancel"),
            job_id=str(req.get("job_id", "") or ""),
        )
        response_status, response_payload = (
            simulation_http_refresh_command_utils_module.simulation_http_refresh_cancel_payload(
                perspective=perspective,
                canceled=canceled,
                snapshot=snapshot,
            )
        )
        send_json(
            response_payload,
            status=response_status,
        )
        return

    if action in {"start", "refresh", "trigger"}:
        if not full_async_refresh_enabled:
            send_json(
                simulation_http_refresh_command_utils_module.simulation_http_refresh_disabled_payload(),
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        force, trigger_value, cache_key_hint = (
            simulation_http_refresh_command_utils_module.simulation_http_refresh_start_control(
                req,
                cache_perspective=cache_perspective,
                safe_bool_query=safe_bool_query,
            )
        )
        if force:
            full_async_refresh_cancel(
                reason="force_restart",
                job_id=str(req.get("job_id", "") or ""),
            )
        scheduled, snapshot = schedule_full_simulation_async_refresh(
            perspective=perspective,
            cache_perspective=cache_perspective,
            cache_key_hint=cache_key_hint,
            trigger=trigger_value,
            allow_throttle_bypass=force,
        )
        response_status, response_payload = (
            simulation_http_refresh_command_utils_module.simulation_http_refresh_start_payload(
                perspective=perspective,
                scheduled=scheduled,
                snapshot=snapshot,
                safe_float=safe_float,
            )
        )
        send_json(
            response_payload,
            status=response_status,
        )
        return

    send_json(
        simulation_http_refresh_command_utils_module.simulation_http_refresh_unsupported_payload(),
        status=HTTPStatus.BAD_REQUEST,
    )


def handle_simulation_bootstrap_post(
    *,
    req: dict[str, Any],
    default_perspective: str,
    normalize_projection_perspective: Callable[[str], str],
    safe_bool_query: Callable[[str, bool], bool],
    run_simulation_bootstrap: Callable[..., tuple[dict[str, Any], HTTPStatus]],
    simulation_bootstrap_job_start: Callable[..., tuple[bool, dict[str, Any]]],
    simulation_bootstrap_job_mark_phase: Callable[..., None],
    simulation_bootstrap_job_complete: Callable[..., None],
    simulation_bootstrap_job_fail: Callable[..., None],
    send_json: Callable[..., None],
) -> None:
    bootstrap_context = (
        simulation_bootstrap_command_utils_module.simulation_bootstrap_command_context(
            req,
            default_perspective=default_perspective,
            normalize_projection_perspective=normalize_projection_perspective,
            safe_bool_query=safe_bool_query,
        )
    )
    perspective = bootstrap_context.perspective
    sync_inbox = bootstrap_context.sync_inbox
    include_simulation_payload = bootstrap_context.include_simulation_payload

    if bootstrap_context.wait:
        payload, status = run_simulation_bootstrap(
            perspective=perspective,
            sync_inbox=sync_inbox,
            include_simulation_payload=include_simulation_payload,
        )
        send_json(payload, status=status)
        return

    request_payload = (
        simulation_bootstrap_command_utils_module.simulation_bootstrap_request_payload(
            bootstrap_context
        )
    )
    started, job_snapshot = simulation_bootstrap_job_start(
        request_payload=request_payload
    )
    queue_payload = (
        simulation_bootstrap_command_utils_module.simulation_bootstrap_queue_payload(
            job_snapshot
        )
    )
    if not started:
        send_json(
            queue_payload,
            status=HTTPStatus.ACCEPTED,
        )
        return

    job_id = str(job_snapshot.get("job_id", "") or "")

    def _run_bootstrap_job() -> None:
        try:
            payload, status = run_simulation_bootstrap(
                perspective=perspective,
                sync_inbox=sync_inbox,
                include_simulation_payload=include_simulation_payload,
                phase_callback=lambda phase, detail: (
                    simulation_bootstrap_job_mark_phase(
                        job_id=job_id,
                        phase=phase,
                        detail=detail,
                    )
                ),
            )
            if status == HTTPStatus.OK and bool(payload.get("ok", False)):
                simulation_bootstrap_job_complete(
                    job_id=job_id,
                    report=payload,
                )
            else:
                simulation_bootstrap_job_fail(
                    job_id=job_id,
                    error=str(
                        payload.get("error", "simulation_bootstrap_failed:unknown")
                    ),
                    report=payload,
                )
        except Exception as exc:
            simulation_bootstrap_job_fail(
                job_id=job_id,
                error=f"simulation_bootstrap_failed:{exc.__class__.__name__}",
                report=simulation_bootstrap_command_utils_module.simulation_bootstrap_failure_report(
                    perspective=perspective,
                    exc=exc,
                    normalize_projection_perspective=normalize_projection_perspective,
                ),
            )

    thread_name = (
        simulation_bootstrap_command_utils_module.simulation_bootstrap_thread_name(
            job_id
        )
    )
    threading.Thread(
        target=_run_bootstrap_job,
        daemon=True,
        name=thread_name,
    ).start()
    send_json(
        queue_payload,
        status=HTTPStatus.ACCEPTED,
    )
