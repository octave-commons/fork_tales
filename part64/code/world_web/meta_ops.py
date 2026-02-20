from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


META_NOTE_EVENT_VERSION = "eta-mu.meta-note.v1"
META_RUN_EVENT_VERSION = "eta-mu.meta-run.v1"
META_OVERVIEW_RECORD = "eta-mu.meta-overview.v1"

META_NOTES_LOG_REL = ".opencode/runtime/meta_notes.v1.jsonl"
META_RUNS_LOG_REL = ".opencode/runtime/meta_runs.v1.jsonl"

_META_NOTES_LOCK = threading.Lock()
_META_RUNS_LOCK = threading.Lock()

_ALLOWED_NOTE_SEVERITIES = {"info", "watch", "warning", "critical"}
_ALLOWED_NOTE_CATEGORIES = {
    "observation",
    "failure",
    "hypothesis",
    "action",
    "dataset",
    "training",
    "evaluation",
}
_ALLOWED_RUN_TYPES = {
    "training",
    "evaluation",
    "dataset-curation",
    "benchmark",
    "analysis",
}
_ALLOWED_RUN_STATUSES = {
    "planned",
    "running",
    "completed",
    "failed",
    "cancelled",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_owner(value: Any) -> str:
    owner = str(value or "Err").strip()
    if not owner:
        return "Err"
    return owner[:80]


def _normalize_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _normalize_string_list(
    value: Any,
    *,
    max_items: int = 12,
    max_chars: int = 80,
) -> list[str]:
    if isinstance(value, str):
        source = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        source = [str(item or "").strip() for item in value]
    else:
        source = []

    out: list[str] = []
    seen: set[str] = set()
    for item in source:
        if not item:
            continue
        clean = item[:max_chars]
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= max_items:
            break
    return out


def _normalize_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, Any] = {}
    for key, raw in value.items():
        clean_key = str(key or "").strip()[:80]
        if not clean_key:
            continue
        if isinstance(raw, (bool, int, float, str)):
            metrics[clean_key] = raw
            continue
        try:
            metrics[clean_key] = json.dumps(raw, ensure_ascii=False)[:200]
        except (TypeError, ValueError):
            continue
    return metrics


def _append_jsonl(path: Path, row: dict[str, Any], *, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _load_jsonl(path: Path, *, lock: threading.Lock) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with lock:
        for raw in path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)
    rows.sort(key=lambda item: str(item.get("ts", "")), reverse=True)
    return rows


def meta_notes_log_path(vault_root: Path) -> Path:
    return (vault_root / META_NOTES_LOG_REL).resolve()


def meta_runs_log_path(vault_root: Path) -> Path:
    return (vault_root / META_RUNS_LOG_REL).resolve()


def create_meta_note(
    vault_root: Path,
    *,
    text: str,
    owner: str = "Err",
    title: str = "",
    tags: Any = None,
    targets: Any = None,
    severity: str = "info",
    category: str = "observation",
    context: Any = None,
) -> dict[str, Any]:
    body = _normalize_text(text, max_chars=2400)
    if not body:
        return {
            "ok": False,
            "error": "missing_text",
            "required": ["text"],
        }

    normalized_owner = _normalize_owner(owner)
    normalized_title = _normalize_text(title, max_chars=160)
    normalized_tags = _normalize_string_list(tags, max_items=16, max_chars=48)
    normalized_targets = _normalize_string_list(targets, max_items=16, max_chars=96)
    severity_value = str(severity or "info").strip().lower() or "info"
    if severity_value not in _ALLOWED_NOTE_SEVERITIES:
        severity_value = "info"
    category_value = str(category or "observation").strip().lower() or "observation"
    if category_value not in _ALLOWED_NOTE_CATEGORIES:
        category_value = "observation"
    context_value = context if isinstance(context, dict) else {}

    ts = _now_iso()
    seed = f"{normalized_owner}|{ts}|{body}|{time.time_ns()}"
    note_id = f"note-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"

    note = {
        "v": META_NOTE_EVENT_VERSION,
        "ts": ts,
        "id": note_id,
        "owner": normalized_owner,
        "title": normalized_title,
        "text": body,
        "severity": severity_value,
        "category": category_value,
        "tags": normalized_tags,
        "targets": normalized_targets,
        "context": context_value,
    }
    path = meta_notes_log_path(vault_root)
    _append_jsonl(path, note, lock=_META_NOTES_LOCK)
    return {
        "ok": True,
        "record": META_NOTE_EVENT_VERSION,
        "note": note,
        "path": str(path),
    }


def list_meta_notes(
    vault_root: Path,
    *,
    limit: int = 24,
    tag: str = "",
    target: str = "",
    category: str = "",
    severity: str = "",
) -> dict[str, Any]:
    safe_limit = max(1, min(256, int(limit or 24)))
    path = meta_notes_log_path(vault_root)
    rows = _load_jsonl(path, lock=_META_NOTES_LOCK)

    tag_filter = str(tag or "").strip().lower()
    target_filter = str(target or "").strip().lower()
    category_filter = str(category or "").strip().lower()
    severity_filter = str(severity or "").strip().lower()

    filtered: list[dict[str, Any]] = []
    for row in rows:
        tags = [str(item or "").lower() for item in row.get("tags", [])]
        targets = [str(item or "").lower() for item in row.get("targets", [])]
        row_category = str(row.get("category", "")).strip().lower()
        row_severity = str(row.get("severity", "")).strip().lower()

        if tag_filter and tag_filter not in tags:
            continue
        if target_filter and target_filter not in targets:
            continue
        if category_filter and category_filter != row_category:
            continue
        if severity_filter and severity_filter != row_severity:
            continue
        filtered.append(dict(row))
        if len(filtered) >= safe_limit:
            break

    return {
        "ok": True,
        "record": "eta-mu.meta-notes.v1",
        "generated_at": _now_iso(),
        "path": str(path),
        "count": len(filtered),
        "notes": filtered,
    }


def create_meta_run(
    vault_root: Path,
    *,
    run_type: str,
    title: str,
    owner: str = "Err",
    status: str = "planned",
    objective: str = "",
    model_ref: str = "",
    dataset_ref: str = "",
    notes: str = "",
    tags: Any = None,
    targets: Any = None,
    metrics: Any = None,
    links: Any = None,
) -> dict[str, Any]:
    run_type_value = str(run_type or "").strip().lower()
    if run_type_value not in _ALLOWED_RUN_TYPES:
        return {
            "ok": False,
            "error": "invalid_run_type",
            "allowed_run_types": sorted(_ALLOWED_RUN_TYPES),
        }

    status_value = str(status or "planned").strip().lower() or "planned"
    if status_value not in _ALLOWED_RUN_STATUSES:
        status_value = "planned"

    title_value = _normalize_text(title, max_chars=180)
    if not title_value:
        return {
            "ok": False,
            "error": "missing_title",
            "required": ["title"],
        }

    normalized_owner = _normalize_owner(owner)
    objective_value = _normalize_text(objective, max_chars=500)
    model_ref_value = _normalize_text(model_ref, max_chars=180)
    dataset_ref_value = _normalize_text(dataset_ref, max_chars=240)
    note_value = _normalize_text(notes, max_chars=2400)
    tag_values = _normalize_string_list(tags, max_items=20, max_chars=48)
    target_values = _normalize_string_list(targets, max_items=20, max_chars=96)
    link_values = _normalize_string_list(links, max_items=20, max_chars=240)
    metric_values = _normalize_metrics(metrics)

    ts = _now_iso()
    seed = (
        f"{run_type_value}|{title_value}|{normalized_owner}|{status_value}|"
        f"{model_ref_value}|{dataset_ref_value}|{ts}|{time.time_ns()}"
    )
    run_id = f"run-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"

    run = {
        "v": META_RUN_EVENT_VERSION,
        "ts": ts,
        "id": run_id,
        "owner": normalized_owner,
        "run_type": run_type_value,
        "status": status_value,
        "title": title_value,
        "objective": objective_value,
        "model_ref": model_ref_value,
        "dataset_ref": dataset_ref_value,
        "notes": note_value,
        "tags": tag_values,
        "targets": target_values,
        "links": link_values,
        "metrics": metric_values,
    }
    path = meta_runs_log_path(vault_root)
    _append_jsonl(path, run, lock=_META_RUNS_LOCK)
    return {
        "ok": True,
        "record": META_RUN_EVENT_VERSION,
        "run": run,
        "path": str(path),
    }


def list_meta_runs(
    vault_root: Path,
    *,
    limit: int = 24,
    run_type: str = "",
    status: str = "",
    target: str = "",
) -> dict[str, Any]:
    safe_limit = max(1, min(256, int(limit or 24)))
    path = meta_runs_log_path(vault_root)
    rows = _load_jsonl(path, lock=_META_RUNS_LOCK)

    type_filter = str(run_type or "").strip().lower()
    status_filter = str(status or "").strip().lower()
    target_filter = str(target or "").strip().lower()

    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_type = str(row.get("run_type", "")).strip().lower()
        row_status = str(row.get("status", "")).strip().lower()
        targets = [str(item or "").lower() for item in row.get("targets", [])]

        if type_filter and type_filter != row_type:
            continue
        if status_filter and status_filter != row_status:
            continue
        if target_filter and target_filter not in targets:
            continue
        filtered.append(dict(row))
        if len(filtered) >= safe_limit:
            break

    return {
        "ok": True,
        "record": "eta-mu.meta-runs.v1",
        "generated_at": _now_iso(),
        "path": str(path),
        "count": len(filtered),
        "runs": filtered,
    }


def build_meta_overview(
    vault_root: Path,
    *,
    docker_snapshot: dict[str, Any],
    queue_snapshot: dict[str, Any],
    notes_limit: int = 12,
    runs_limit: int = 12,
    simulation_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    notes_payload = list_meta_notes(vault_root, limit=notes_limit)
    runs_payload = list_meta_runs(vault_root, limit=runs_limit)

    economy_payload: dict[str, Any] = {"presences": []}
    if isinstance(simulation_state, dict):
        dynamics = simulation_state.get("presence_dynamics", {})
        if isinstance(dynamics, dict):
            impacts = dynamics.get("presence_impacts", [])
            if isinstance(impacts, list):
                economy_presences: list[dict[str, Any]] = []
                for p in impacts:
                    if not isinstance(p, dict):
                        continue
                    p_type = str(p.get("presence_type", "normal"))
                    if p_type in {"core", "sub-sim"}:
                        economy_presences.append(
                            {
                                "id": p.get("id"),
                                "label": p.get("label", p.get("en")),
                                "presence_type": p_type,
                                "resource_wallet": p.get("resource_wallet", {}),
                                "economy_last_update": p.get("economy_last_update", ""),
                            }
                        )
                economy_payload["presences"] = economy_presences

    simulations = (
        docker_snapshot.get("simulations", [])
        if isinstance(docker_snapshot.get("simulations"), list)
        else []
    )
    failures: list[dict[str, Any]] = []
    degraded: list[dict[str, Any]] = []
    for row in simulations:
        if not isinstance(row, dict):
            continue
        lifecycle = (
            row.get("lifecycle", {}) if isinstance(row.get("lifecycle"), dict) else {}
        )
        resources = (
            row.get("resources", {}) if isinstance(row.get("resources"), dict) else {}
        )
        pressure = (
            resources.get("pressure", {})
            if isinstance(resources.get("pressure"), dict)
            else {}
        )
        pressure_state = str(pressure.get("state", "")).strip().lower()
        stability = str(lifecycle.get("stability", "healthy")).strip().lower()

        item = {
            "id": str(row.get("id", "") or ""),
            "name": str(row.get("name", "") or ""),
            "service": str(row.get("service", "") or ""),
            "project": str(row.get("project", "") or ""),
            "state": str(row.get("state", "") or ""),
            "status": str(row.get("status", "") or ""),
            "stability": stability or "healthy",
            "health_status": str(lifecycle.get("health_status", "none") or "none"),
            "restart_count": _safe_int(lifecycle.get("restart_count"), 0),
            "oom_killed": bool(lifecycle.get("oom_killed", False)),
            "pressure_state": pressure_state,
            "signals": [
                str(signal)
                for signal in lifecycle.get("signals", [])
                if str(signal).strip()
            ],
        }

        if stability == "failing":
            failures.append(item)
        elif stability == "degraded":
            degraded.append(item)
        elif pressure_state == "critical":
            item["stability"] = "failing"
            failures.append(item)
        elif pressure_state == "warning":
            item["stability"] = "degraded"
            degraded.append(item)

    queue_pending = _safe_int(queue_snapshot.get("pending_count"), 0)
    run_rows = (
        runs_payload.get("runs", [])
        if isinstance(runs_payload.get("runs"), list)
        else []
    )
    active_runs = [
        row
        for row in run_rows
        if str(row.get("status", "")).strip().lower() in {"planned", "running"}
    ]

    suggestions: list[str] = []
    if failures:
        suggestions.append(
            "Capture a failure note and queue a stabilization objective for failing simulations."
        )
    if degraded:
        suggestions.append(
            "Queue targeted evaluation tasks for degraded simulations before promoting outputs."
        )
    if queue_pending == 0:
        suggestions.append(
            "Task queue is idle; enqueue one training/evaluation objective to keep progress observable."
        )
    if not active_runs:
        suggestions.append(
            "No active training/evaluation runs logged; register current experiment runs for traceability."
        )

    return {
        "ok": True,
        "record": META_OVERVIEW_RECORD,
        "generated_at": _now_iso(),
        "docker_summary": docker_snapshot.get("summary", {}),
        "queue": queue_snapshot,
        "failures": {
            "failing_count": len(failures),
            "degraded_count": len(degraded),
            "failing": failures,
            "degraded": degraded,
        },
        "notes": {
            "count": _safe_int(notes_payload.get("count"), 0),
            "items": notes_payload.get("notes", []),
        },
        "runs": {
            "count": _safe_int(runs_payload.get("count"), 0),
            "active_count": len(active_runs),
            "items": run_rows,
        },
        "economy": economy_payload,
        "suggestions": suggestions,
    }
