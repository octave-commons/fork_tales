"""State-machine helpers for simulation bootstrap jobs."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


def bootstrap_report_store(
    *, report_state: dict[str, Any], report_lock: Any, report: dict[str, Any]
) -> None:
    if not isinstance(report, dict):
        return
    with report_lock:
        report_state.clear()
        report_state.update(dict(report))


def bootstrap_report_snapshot(
    *, report_state: dict[str, Any], report_lock: Any
) -> dict[str, Any] | None:
    with report_lock:
        if not isinstance(report_state, dict):
            return None
        if not report_state:
            return None
        return dict(report_state)


def bootstrap_job_snapshot(
    *, job_state: dict[str, Any], job_lock: Any
) -> dict[str, Any]:
    with job_lock:
        snapshot = dict(job_state)
        request_payload = snapshot.get("request", {})
        if isinstance(request_payload, dict):
            snapshot["request"] = dict(request_payload)
        else:
            snapshot["request"] = {}
        if isinstance(snapshot.get("report"), dict):
            snapshot["report"] = dict(snapshot.get("report", {}))
        else:
            snapshot["report"] = None
        phase_detail = snapshot.get("phase_detail", {})
        snapshot["phase_detail"] = (
            dict(phase_detail) if isinstance(phase_detail, dict) else {}
        )
        return snapshot


def bootstrap_job_start(
    *,
    job_state: dict[str, Any],
    job_lock: Any,
    request_payload: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    request_row = dict(request_payload) if isinstance(request_payload, dict) else {}
    with job_lock:
        if str(job_state.get("status", "")).strip().lower() == "running":
            snapshot = dict(job_state)
            request_snapshot = snapshot.get("request", {})
            snapshot["request"] = (
                dict(request_snapshot) if isinstance(request_snapshot, dict) else {}
            )
            report_snapshot = snapshot.get("report")
            snapshot["report"] = (
                dict(report_snapshot) if isinstance(report_snapshot, dict) else None
            )
            phase_detail_snapshot = snapshot.get("phase_detail", {})
            snapshot["phase_detail"] = (
                dict(phase_detail_snapshot)
                if isinstance(phase_detail_snapshot, dict)
                else {}
            )
            return False, snapshot

        seed = (
            f"{now_iso}|{request_row.get('perspective', '')}|"
            f"{request_row.get('sync_inbox', False)}"
        )
        job_id = "bootstrap:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        job_state["status"] = "running"
        job_state["job_id"] = job_id
        job_state["started_at"] = now_iso
        job_state["updated_at"] = now_iso
        job_state["completed_at"] = ""
        job_state["phase"] = "queued"
        job_state["phase_started_at"] = now_iso
        job_state["phase_detail"] = {}
        job_state["error"] = ""
        job_state["request"] = request_row
        job_state["report"] = None

        snapshot = dict(job_state)
        snapshot["request"] = dict(request_row)
        snapshot["phase_detail"] = {}
        snapshot["report"] = None
        return True, snapshot


def bootstrap_job_mark_phase(
    *,
    job_state: dict[str, Any],
    job_lock: Any,
    job_id: str,
    phase: str,
    detail: dict[str, Any] | None = None,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with job_lock:
        if str(job_state.get("job_id", "")) != str(job_id):
            return
        next_phase = str(phase or "").strip().lower()
        if not next_phase:
            return
        current_phase = str(job_state.get("phase", "")).strip().lower()
        job_state["phase"] = next_phase
        if next_phase != current_phase:
            job_state["phase_started_at"] = now_iso
        job_state["updated_at"] = now_iso
        job_state["phase_detail"] = dict(detail) if isinstance(detail, dict) else {}


def bootstrap_job_complete(
    *,
    job_state: dict[str, Any],
    job_lock: Any,
    job_id: str,
    report: dict[str, Any],
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with job_lock:
        if str(job_state.get("job_id", "")) != str(job_id):
            return
        job_state["status"] = "completed"
        job_state["phase"] = "completed"
        job_state["phase_started_at"] = now_iso
        job_state["phase_detail"] = {}
        job_state["updated_at"] = now_iso
        job_state["completed_at"] = now_iso
        job_state["error"] = ""
        job_state["report"] = dict(report)


def bootstrap_job_fail(
    *,
    job_state: dict[str, Any],
    job_lock: Any,
    job_id: str,
    error: str,
    report: dict[str, Any] | None = None,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with job_lock:
        if str(job_state.get("job_id", "")) != str(job_id):
            return
        job_state["status"] = "failed"
        job_state["phase"] = "failed"
        job_state["phase_started_at"] = now_iso
        job_state["phase_detail"] = {}
        job_state["updated_at"] = now_iso
        job_state["completed_at"] = now_iso
        job_state["error"] = str(error or "simulation_bootstrap_failed")
        job_state["report"] = dict(report) if isinstance(report, dict) else None
