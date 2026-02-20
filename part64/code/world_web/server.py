from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

from .ai import (
    _apply_embedding_provider_options,
    _embedding_provider_options,
    _ollama_embed,
    build_chat_reply,
    build_image_commentary,
    build_presence_say_payload,
    build_voice_lines,
    transcribe_audio_bytes,
    utterances_to_ledger_rows,
)
from .catalog import (
    _read_library_archive_member,
    build_world_payload,
    collect_catalog,
    collect_zip_catalog,
    load_manifest,
    resolve_library_member,
    resolve_library_path,
    sync_eta_mu_inbox,
)
from .chamber import (
    CouncilChamber,
    TaskQueue,
    _load_study_snapshot_events,
    _study_snapshot_log_path,
    build_witness_lineage_payload,
    build_world_log_payload,
    build_pi_archive_payload,
    build_drift_scan_payload,
    build_push_truth_dry_run_payload,
    build_study_snapshot,
    export_study_snapshot,
    validate_pi_archive_portable,
)
from .constants import (
    AUDIO_SUFFIXES,
    CATALOG_BROADCAST_HEARTBEAT_SECONDS,
    CATALOG_REFRESH_SECONDS,
    COUNCIL_DECISION_LOG_REL,
    ENTITY_MANIFEST,
    IMAGE_SUFFIXES,
    PROJECTION_DEFAULT_PERSPECTIVE,
    SIM_TICK_SECONDS,
    TTS_BASE_URL,
    USER_PRESENCE_ID,
    USER_PRESENCE_MAX_EVENTS,
    _USER_PRESENCE_INPUT_CACHE,
    _USER_PRESENCE_INPUT_LOCK,
    VIDEO_SUFFIXES,
    WEAVER_AUTOSTART,
    WEAVER_HOST_ENV,
    WEAVER_PORT,
    WS_MAGIC,
)
from .db import (
    _embedding_db_delete,
    _embedding_db_list,
    _embedding_db_query,
    _embedding_db_status,
    _embedding_db_upsert,
    _create_image_comment,
    _get_chroma_collection,
    _list_image_comments,
    _list_presence_accounts,
    _load_life_interaction_builder,
    _load_life_tracker_class,
    _load_mycelial_echo_documents,
    _load_myth_tracker_class,
    _normalize_embedding_vector,
    _upsert_presence_account,
    _upsert_simulation_metadata,
    _list_simulation_metadata,
)
from .docker_runtime import (
    DOCKER_SIMULATION_WS_HEARTBEAT_SECONDS,
    DOCKER_SIMULATION_WS_REFRESH_SECONDS,
    collect_docker_simulation_snapshot,
    control_simulation_container,
)
from .metrics import _INFLUENCE_TRACKER, _safe_float, _resource_monitor_snapshot
from .meta_ops import (
    build_meta_overview,
    create_meta_note,
    create_meta_run,
    list_meta_notes,
    list_meta_runs,
)
from .muse_runtime import get_muse_runtime_manager
from .paths import _ensure_receipts_log_path
from .projection import (
    attach_ui_projection,
    build_ui_projection,
    normalize_projection_perspective,
    projection_perspective_options,
)
from .simulation import (
    build_simulation_delta,
    build_mix_stream,
    build_named_field_overlays,
    build_simulation_state,
)


_RUNTIME_CATALOG_CACHE_LOCK = threading.Lock()
_RUNTIME_CATALOG_REFRESH_LOCK = threading.Lock()
_RUNTIME_CATALOG_CACHE: dict[str, Any] = {
    "catalog": None,
    "refreshed_monotonic": 0.0,
    "last_error": "",
    "inbox_sync_monotonic": 0.0,
    "inbox_sync_snapshot": None,
    "inbox_sync_error": "",
}
_RUNTIME_CATALOG_CACHE_SECONDS = max(
    CATALOG_REFRESH_SECONDS,
    float(os.getenv("RUNTIME_CATALOG_CACHE_SECONDS", "10.0") or "10.0"),
)
_RUNTIME_ETA_MU_SYNC_SECONDS = max(
    0.5,
    float(os.getenv("RUNTIME_ETA_MU_SYNC_SECONDS", "6.0") or "6.0"),
)
_RUNTIME_INBOX_SYNC_LOCK = threading.Lock()
_RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS = max(
    8.0,
    float(os.getenv("RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS", "75.0") or "75.0"),
)
_RUNTIME_CATALOG_SUBPROCESS_ENABLED = str(
    os.getenv("RUNTIME_CATALOG_SUBPROCESS_ENABLED", "1") or "1"
).strip().lower() not in {"0", "false", "no", "off"}
_SIMULATION_HTTP_CACHE_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_CACHE_SECONDS", "5.0") or "5.0"),
)
_SIMULATION_HTTP_STALE_FALLBACK_SECONDS = max(
    _SIMULATION_HTTP_CACHE_SECONDS,
    float(os.getenv("SIMULATION_HTTP_STALE_FALLBACK_SECONDS", "12.0") or "12.0"),
)
_SIMULATION_HTTP_BUILD_WAIT_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_BUILD_WAIT_SECONDS", "12.0") or "12.0"),
)
_SIMULATION_HTTP_WARMUP_ENABLED = str(
    os.getenv("SIMULATION_HTTP_WARMUP_ENABLED", "0") or "0"
).strip().lower() not in {"0", "false", "no", "off"}
_SIMULATION_HTTP_WARMUP_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_WARMUP_DELAY_SECONDS", "2.0") or "2.0"),
)
_SIMULATION_HTTP_WARMUP_TIMEOUT_SECONDS = max(
    6.0,
    float(os.getenv("SIMULATION_HTTP_WARMUP_TIMEOUT_SECONDS", "90.0") or "90.0"),
)
_SIMULATION_HTTP_WARMUP_RETRY_SECONDS = max(
    1.0,
    float(os.getenv("SIMULATION_HTTP_WARMUP_RETRY_SECONDS", "15.0") or "15.0"),
)
_SIMULATION_HTTP_WARMUP_MAX_ATTEMPTS = max(
    1,
    int(float(os.getenv("SIMULATION_HTTP_WARMUP_MAX_ATTEMPTS", "2") or "2")),
)
_SIMULATION_HTTP_DISK_CACHE_ENABLED = str(
    os.getenv("SIMULATION_HTTP_DISK_CACHE_ENABLED", "1") or "1"
).strip().lower() not in {"0", "false", "no", "off"}
_SIMULATION_HTTP_DISK_CACHE_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_DISK_CACHE_SECONDS", "900.0") or "900.0"),
)
_SIMULATION_HTTP_DISK_COLD_START_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_DISK_COLD_START_SECONDS", "180.0") or "180.0"),
)
_SIMULATION_HTTP_FAILURE_COOLDOWN_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_FAILURE_COOLDOWN_SECONDS", "45.0") or "45.0"),
)
_SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS = max(
    _SIMULATION_HTTP_FAILURE_COOLDOWN_SECONDS,
    float(
        os.getenv("SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS", "300.0") or "300.0"
    ),
)
_SIMULATION_HTTP_TRIM_ENABLED = str(
    os.getenv("SIMULATION_HTTP_TRIM_ENABLED", "1") or "1"
).strip().lower() not in {"0", "false", "no", "off"}
_SIMULATION_HTTP_MAX_ITEMS = max(
    128,
    int(float(os.getenv("SIMULATION_HTTP_MAX_ITEMS", "640") or "640")),
)
_SIMULATION_HTTP_MAX_FILE_NODES = max(
    128,
    int(float(os.getenv("SIMULATION_HTTP_MAX_FILE_NODES", "360") or "360")),
)
_SIMULATION_HTTP_MAX_FILE_EDGES = max(
    256,
    int(float(os.getenv("SIMULATION_HTTP_MAX_FILE_EDGES", "900") or "900")),
)
_SIMULATION_HTTP_MAX_FIELD_NODES = max(
    64,
    int(float(os.getenv("SIMULATION_HTTP_MAX_FIELD_NODES", "120") or "120")),
)
_SIMULATION_HTTP_MAX_TAG_NODES = max(
    64,
    int(float(os.getenv("SIMULATION_HTTP_MAX_TAG_NODES", "180") or "180")),
)
_SIMULATION_HTTP_MAX_RENDER_NODES = max(
    256,
    int(float(os.getenv("SIMULATION_HTTP_MAX_RENDER_NODES", "640") or "640")),
)
_SIMULATION_HTTP_MAX_CRAWLER_NODES = max(
    96,
    int(float(os.getenv("SIMULATION_HTTP_MAX_CRAWLER_NODES", "280") or "280")),
)
_SIMULATION_HTTP_MAX_CRAWLER_EDGES = max(
    128,
    int(float(os.getenv("SIMULATION_HTTP_MAX_CRAWLER_EDGES", "700") or "700")),
)
_SIMULATION_HTTP_MAX_CRAWLER_FIELD_NODES = max(
    32,
    int(float(os.getenv("SIMULATION_HTTP_MAX_CRAWLER_FIELD_NODES", "96") or "96")),
)
_SIMULATION_HTTP_MAX_TEXT_EXCERPT_CHARS = max(
    160,
    int(float(os.getenv("SIMULATION_HTTP_MAX_TEXT_EXCERPT_CHARS", "640") or "640")),
)
_SIMULATION_HTTP_MAX_SUMMARY_CHARS = max(
    120,
    int(float(os.getenv("SIMULATION_HTTP_MAX_SUMMARY_CHARS", "420") or "420")),
)
_SIMULATION_HTTP_MAX_EMBED_LAYER_POINTS = max(
    0,
    int(float(os.getenv("SIMULATION_HTTP_MAX_EMBED_LAYER_POINTS", "8") or "8")),
)
_SIMULATION_HTTP_MAX_EMBED_IDS = max(
    0,
    int(float(os.getenv("SIMULATION_HTTP_MAX_EMBED_IDS", "16") or "16")),
)
_SIMULATION_HTTP_MAX_EMBEDDING_LINKS = max(
    0,
    int(float(os.getenv("SIMULATION_HTTP_MAX_EMBEDDING_LINKS", "28") or "28")),
)
_SIMULATION_HTTP_CACHE_LOCK = threading.Lock()
_SIMULATION_HTTP_BUILD_LOCK = threading.Lock()
_SIMULATION_HTTP_CACHE: dict[str, Any] = {
    "key": "",
    "prepared_monotonic": 0.0,
    "body": b"",
}
_SIMULATION_HTTP_FAILURE_LOCK = threading.Lock()
_SIMULATION_HTTP_FAILURE_STATE: dict[str, Any] = {
    "last_failure_monotonic": 0.0,
    "last_error": "",
    "streak": 0,
}
_SERVER_BOOT_MONOTONIC = time.monotonic()

_RUNTIME_WS_CLIENT_LOCK = threading.Lock()
_RUNTIME_WS_CLIENT_COUNT = 0
_RUNTIME_WS_MAX_CLIENTS = max(
    2,
    int(float(os.getenv("RUNTIME_WS_MAX_CLIENTS", "12") or "12")),
)
_RUNTIME_GUARD_CPU_UTILIZATION_CRITICAL = max(
    40.0,
    min(
        99.0,
        float(os.getenv("RUNTIME_GUARD_CPU_UTILIZATION_CRITICAL", "92") or "92"),
    ),
)
_RUNTIME_GUARD_MEMORY_PRESSURE_CRITICAL = max(
    0.45,
    min(
        0.99,
        float(os.getenv("RUNTIME_GUARD_MEMORY_PRESSURE_CRITICAL", "0.9") or "0.9"),
    ),
)
_RUNTIME_GUARD_LOG_ERROR_RATIO_CRITICAL = max(
    0.1,
    min(
        1.0,
        float(os.getenv("RUNTIME_GUARD_LOG_ERROR_RATIO_CRITICAL", "0.55") or "0.55"),
    ),
)
_RUNTIME_GUARD_INTERVAL_SCALE = max(
    1.0,
    float(os.getenv("RUNTIME_GUARD_INTERVAL_SCALE", "3.0") or "3.0"),
)
_RUNTIME_GUARD_SKIP_SIMULATION_ON_CRITICAL = str(
    os.getenv("RUNTIME_GUARD_SKIP_SIMULATION_ON_CRITICAL", "1") or "1"
).strip().lower() not in {"0", "false", "no", "off"}
_RUNTIME_GUARD_HEARTBEAT_SECONDS = max(
    1.0,
    float(os.getenv("RUNTIME_GUARD_HEARTBEAT_SECONDS", "4.5") or "4.5"),
)


def _runtime_ws_client_snapshot() -> dict[str, int]:
    with _RUNTIME_WS_CLIENT_LOCK:
        return {
            "active_clients": int(_RUNTIME_WS_CLIENT_COUNT),
            "max_clients": int(_RUNTIME_WS_MAX_CLIENTS),
        }


def _runtime_ws_try_acquire_client_slot() -> bool:
    global _RUNTIME_WS_CLIENT_COUNT
    with _RUNTIME_WS_CLIENT_LOCK:
        if _RUNTIME_WS_CLIENT_COUNT >= _RUNTIME_WS_MAX_CLIENTS:
            return False
        _RUNTIME_WS_CLIENT_COUNT += 1
    return True


def _runtime_ws_release_client_slot() -> None:
    global _RUNTIME_WS_CLIENT_COUNT
    with _RUNTIME_WS_CLIENT_LOCK:
        _RUNTIME_WS_CLIENT_COUNT = max(0, _RUNTIME_WS_CLIENT_COUNT - 1)


def _runtime_guard_state(resource_snapshot: dict[str, Any]) -> dict[str, Any]:
    snapshot = resource_snapshot if isinstance(resource_snapshot, dict) else {}
    devices = (
        snapshot.get("devices", {}) if isinstance(snapshot.get("devices"), dict) else {}
    )
    cpu = devices.get("cpu", {}) if isinstance(devices.get("cpu"), dict) else {}
    log_watch = (
        snapshot.get("log_watch", {})
        if isinstance(snapshot.get("log_watch"), dict)
        else {}
    )

    cpu_utilization = _safe_float(cpu.get("utilization", 0.0), 0.0)
    memory_pressure = _safe_float(cpu.get("memory_pressure", 0.0), 0.0)
    error_ratio = _safe_float(log_watch.get("error_ratio", 0.0), 0.0)
    hot_devices = [
        str(item).strip()
        for item in snapshot.get("hot_devices", [])
        if str(item).strip()
    ]

    reasons: list[str] = []
    mode = "normal"

    if cpu_utilization >= _RUNTIME_GUARD_CPU_UTILIZATION_CRITICAL:
        mode = "critical"
        reasons.append("cpu_hot")
    if memory_pressure >= _RUNTIME_GUARD_MEMORY_PRESSURE_CRITICAL:
        mode = "critical"
        reasons.append("memory_pressure_high")
    if error_ratio >= _RUNTIME_GUARD_LOG_ERROR_RATIO_CRITICAL:
        mode = "critical"
        reasons.append("runtime_log_error_ratio_high")

    if mode == "normal":
        if hot_devices:
            mode = "degraded"
            reasons.append("hot_devices")
        if error_ratio >= (_RUNTIME_GUARD_LOG_ERROR_RATIO_CRITICAL * 0.65):
            mode = "degraded"
            reasons.append("runtime_log_warning_ratio")
        if cpu_utilization >= (_RUNTIME_GUARD_CPU_UTILIZATION_CRITICAL * 0.84):
            mode = "degraded"
            reasons.append("cpu_watch")
        if memory_pressure >= (_RUNTIME_GUARD_MEMORY_PRESSURE_CRITICAL * 0.85):
            mode = "degraded"
            reasons.append("memory_pressure_watch")

    return {
        "mode": mode,
        "reasons": reasons,
        "cpu_utilization": round(cpu_utilization, 2),
        "memory_pressure": round(memory_pressure, 4),
        "log_error_ratio": round(error_ratio, 4),
        "hot_devices": hot_devices,
        "critical_thresholds": {
            "cpu_utilization": _RUNTIME_GUARD_CPU_UTILIZATION_CRITICAL,
            "memory_pressure": _RUNTIME_GUARD_MEMORY_PRESSURE_CRITICAL,
            "log_error_ratio": _RUNTIME_GUARD_LOG_ERROR_RATIO_CRITICAL,
        },
    }


def _runtime_health_payload(part_root: Path) -> dict[str, Any]:
    resource_snapshot = _resource_monitor_snapshot(part_root=part_root)
    guard = _runtime_guard_state(resource_snapshot)
    ws = _runtime_ws_client_snapshot()
    return {
        "ok": True,
        "record": "eta-mu.runtime-health.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "guard": guard,
        "websocket": ws,
        "degraded": str(guard.get("mode", "normal")) != "normal",
    }


def _json_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _simulation_http_cache_key(
    *,
    perspective: str,
    catalog: dict[str, Any],
    queue_snapshot: dict[str, Any],
    influence_snapshot: dict[str, Any],
) -> str:
    file_graph = catalog.get("file_graph", {}) if isinstance(catalog, dict) else {}
    crawler_graph = (
        catalog.get("crawler_graph", {}) if isinstance(catalog, dict) else {}
    )
    file_stats = file_graph.get("stats", {}) if isinstance(file_graph, dict) else {}
    crawler_stats = (
        crawler_graph.get("stats", {}) if isinstance(crawler_graph, dict) else {}
    )

    file_count = int(_safe_float(file_stats.get("file_count", 0), 0.0))
    file_edge_count = int(_safe_float(file_stats.get("edge_count", 0), 0.0))
    crawler_count = int(_safe_float(crawler_stats.get("crawler_count", 0), 0.0))
    crawler_edge_count = int(_safe_float(crawler_stats.get("edge_count", 0), 0.0))

    if file_count <= 0:
        file_nodes = (
            file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
        )
        if isinstance(file_nodes, list):
            file_count = len(file_nodes)
    if file_edge_count <= 0:
        file_edges = file_graph.get("edges", []) if isinstance(file_graph, dict) else []
        if isinstance(file_edges, list):
            file_edge_count = len(file_edges)
    if crawler_count <= 0:
        crawler_nodes = (
            crawler_graph.get("crawler_nodes", [])
            if isinstance(crawler_graph, dict)
            else []
        )
        if isinstance(crawler_nodes, list):
            crawler_count = len(crawler_nodes)
    if crawler_edge_count <= 0:
        crawler_edges = (
            crawler_graph.get("edges", []) if isinstance(crawler_graph, dict) else []
        )
        if isinstance(crawler_edges, list):
            crawler_edge_count = len(crawler_edges)

    fingerprint = (
        f"{max(0, file_count)}:{max(0, file_edge_count)}:"
        f"{max(0, crawler_count)}:{max(0, crawler_edge_count)}"
    )
    if fingerprint == "0:0:0:0":
        file_graph_generated_at = (
            str(file_graph.get("generated_at", "")).strip()
            if isinstance(file_graph, dict)
            else ""
        )
        crawler_generated_at = (
            str(crawler_graph.get("generated_at", "")).strip()
            if isinstance(crawler_graph, dict)
            else ""
        )
        fingerprint = f"ts:{file_graph_generated_at}:{crawler_generated_at}"

    queue_pending = int(_safe_float(queue_snapshot.get("pending_count", 0), 0.0))
    queue_events = int(_safe_float(queue_snapshot.get("event_count", 0), 0.0))
    clicks_recent = int(_safe_float(influence_snapshot.get("clicks_45s", 0), 0.0))
    user_inputs_recent = int(
        _safe_float(influence_snapshot.get("user_inputs_120s", 0), 0.0)
    )
    user_rows = (
        influence_snapshot.get("recent_user_inputs", [])
        if isinstance(influence_snapshot, dict)
        else []
    )
    if isinstance(user_rows, list) and user_rows:
        newest = user_rows[0] if isinstance(user_rows[0], dict) else {}
        user_signal = "|".join(
            [
                str(newest.get("kind", "")),
                str(newest.get("target", "")),
                str(newest.get("message", ""))[:48],
                str(newest.get("x_ratio", "")),
                str(newest.get("y_ratio", "")),
            ]
        )
    else:
        user_signal = ""
    user_signal_hash = hashlib.sha1(user_signal.encode("utf-8")).hexdigest()[:10]
    return (
        f"{perspective}|{fingerprint}|"
        f"q:{max(0, queue_pending)}:{max(0, queue_events)}|"
        f"i:{max(0, clicks_recent)}:{max(0, user_inputs_recent)}:{user_signal_hash}|"
        "simulation"
    )


def _simulation_http_cache_store(cache_key: str, body: bytes) -> None:
    if not cache_key:
        return
    if not isinstance(body, (bytes, bytearray)):
        return
    body_bytes = bytes(body)
    if not body_bytes:
        return
    with _SIMULATION_HTTP_CACHE_LOCK:
        _SIMULATION_HTTP_CACHE["key"] = cache_key
        _SIMULATION_HTTP_CACHE["prepared_monotonic"] = time.monotonic()
        _SIMULATION_HTTP_CACHE["body"] = body_bytes


def _simulation_http_cached_body(
    *,
    cache_key: str = "",
    perspective: str = "",
    max_age_seconds: float,
    require_exact_key: bool = False,
) -> bytes | None:
    if max_age_seconds <= 0.0:
        return None

    requested_key = str(cache_key or "").strip()
    requested_perspective = str(perspective or "").strip()
    with _SIMULATION_HTTP_CACHE_LOCK:
        cached_key = str(_SIMULATION_HTTP_CACHE.get("key", "") or "").strip()
        cached_body = _SIMULATION_HTTP_CACHE.get("body", b"")
        cached_age = time.monotonic() - _safe_float(
            _SIMULATION_HTTP_CACHE.get("prepared_monotonic", 0.0),
            0.0,
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
    elif requested_perspective and not cached_key.startswith(
        f"{requested_perspective}|"
    ):
        return None

    return bytes(cached_body)


def _simulation_http_wait_for_exact_cache(
    *,
    cache_key: str,
    perspective: str,
    max_wait_seconds: float,
    poll_seconds: float = 0.05,
) -> bytes | None:
    wait_window = max(0.0, _safe_float(max_wait_seconds, 0.0))
    if wait_window <= 0.0:
        return None

    poll_interval = max(0.01, _safe_float(poll_seconds, 0.05))
    deadline = time.monotonic() + wait_window
    max_cache_age = max(
        _SIMULATION_HTTP_CACHE_SECONDS,
        _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
        wait_window,
    )

    while True:
        cached_body = _simulation_http_cached_body(
            cache_key=cache_key,
            perspective=perspective,
            max_age_seconds=max_cache_age,
            require_exact_key=True,
        )
        if cached_body is not None:
            return cached_body

        now_monotonic = time.monotonic()
        if now_monotonic >= deadline:
            return None
        time.sleep(min(poll_interval, max(0.0, deadline - now_monotonic)))


def _simulation_http_is_cold_start() -> bool:
    if _SIMULATION_HTTP_DISK_COLD_START_SECONDS <= 0.0:
        return False
    uptime = time.monotonic() - _SERVER_BOOT_MONOTONIC
    return 0.0 <= uptime <= _SIMULATION_HTTP_DISK_COLD_START_SECONDS


def _simulation_http_failure_backoff_snapshot() -> tuple[float, str, int]:
    cooldown = max(0.0, _safe_float(_SIMULATION_HTTP_FAILURE_COOLDOWN_SECONDS, 0.0))
    if cooldown <= 0.0:
        return (0.0, "", 0)

    with _SIMULATION_HTTP_FAILURE_LOCK:
        last_failure = _safe_float(
            _SIMULATION_HTTP_FAILURE_STATE.get("last_failure_monotonic", 0.0),
            0.0,
        )
        streak = max(
            0, int(_safe_float(_SIMULATION_HTTP_FAILURE_STATE.get("streak", 0), 0.0))
        )
        error_name = str(
            _SIMULATION_HTTP_FAILURE_STATE.get("last_error", "") or ""
        ).strip()

    if last_failure <= 0.0:
        return (0.0, "", 0)

    age = time.monotonic() - last_failure
    if age < 0.0:
        age = 0.0
    remaining = max(0.0, cooldown - age)
    return (remaining, error_name, streak)


def _simulation_http_failure_record(error_name: str) -> None:
    now_monotonic = time.monotonic()
    reset_window = max(
        0.0,
        _safe_float(_SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS, 0.0),
    )
    with _SIMULATION_HTTP_FAILURE_LOCK:
        previous_failure = _safe_float(
            _SIMULATION_HTTP_FAILURE_STATE.get("last_failure_monotonic", 0.0),
            0.0,
        )
        streak = int(_safe_float(_SIMULATION_HTTP_FAILURE_STATE.get("streak", 0), 0.0))
        if previous_failure > 0.0 and reset_window > 0.0:
            if (now_monotonic - previous_failure) > reset_window:
                streak = 0
        _SIMULATION_HTTP_FAILURE_STATE["streak"] = max(0, streak) + 1
        _SIMULATION_HTTP_FAILURE_STATE["last_failure_monotonic"] = now_monotonic
        _SIMULATION_HTTP_FAILURE_STATE["last_error"] = (
            str(error_name or "Exception").strip() or "Exception"
        )


def _simulation_http_failure_clear() -> None:
    with _SIMULATION_HTTP_FAILURE_LOCK:
        _SIMULATION_HTTP_FAILURE_STATE["last_failure_monotonic"] = 0.0
        _SIMULATION_HTTP_FAILURE_STATE["last_error"] = ""
        _SIMULATION_HTTP_FAILURE_STATE["streak"] = 0


def _simulation_http_slice_rows(value: Any, *, max_rows: int) -> list[Any]:
    rows = value if isinstance(value, list) else []
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    return list(rows[:max_rows])


def _simulation_http_compact_embed_layer_point(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    compact = dict(value)
    embed_ids = compact.get("embed_ids")
    if isinstance(embed_ids, list):
        compact["embed_ids"] = [
            str(entry or "")
            for entry in embed_ids[:_SIMULATION_HTTP_MAX_EMBED_IDS]
            if str(entry or "").strip()
        ]
    return compact


def _simulation_http_compact_embedding_link(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    target = str(value.get("target", "") or "").strip()
    kind = str(value.get("kind", "") or "").strip()
    member_path = str(value.get("member_path", "") or "").strip()
    weight = _safe_float(value.get("weight", 0.0), 0.0)

    compact: dict[str, Any] = {}
    if target:
        compact["target"] = target
    if kind:
        compact["kind"] = kind
    if member_path:
        compact["member_path"] = member_path
    if weight > 0.0:
        compact["weight"] = round(weight, 6)
    return compact if compact else None


def _simulation_http_compact_file_node(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    compact = dict(value)

    excerpt = compact.get("text_excerpt")
    if (
        isinstance(excerpt, str)
        and len(excerpt) > _SIMULATION_HTTP_MAX_TEXT_EXCERPT_CHARS
    ):
        compact["text_excerpt"] = excerpt[:_SIMULATION_HTTP_MAX_TEXT_EXCERPT_CHARS]

    summary = compact.get("summary")
    if isinstance(summary, str) and len(summary) > _SIMULATION_HTTP_MAX_SUMMARY_CHARS:
        compact["summary"] = summary[:_SIMULATION_HTTP_MAX_SUMMARY_CHARS]

    layer_points = compact.get("embed_layer_points")
    if isinstance(layer_points, list):
        compact_layers: list[dict[str, Any]] = []
        for row in layer_points[:_SIMULATION_HTTP_MAX_EMBED_LAYER_POINTS]:
            compact_row = _simulation_http_compact_embed_layer_point(row)
            if compact_row is not None:
                compact_layers.append(compact_row)
        compact["embed_layer_points"] = compact_layers
        compact["embed_layer_count"] = len(compact_layers)

    embedding_links = compact.get("embedding_links")
    if isinstance(embedding_links, list):
        compact_links: list[dict[str, Any]] = []
        for row in embedding_links[:_SIMULATION_HTTP_MAX_EMBEDDING_LINKS]:
            compact_row = _simulation_http_compact_embedding_link(row)
            if compact_row is not None:
                compact_links.append(compact_row)
        compact["embedding_links"] = compact_links

    return compact


def _simulation_http_compact_file_graph_node(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    node_type = str(value.get("node_type", "") or "").strip().lower()
    kind = str(value.get("kind", "") or "").strip().lower()
    has_file_fields = any(
        key in value
        for key in (
            "embed_layer_points",
            "embedding_links",
            "source_rel_path",
            "archive_rel_path",
            "archived_rel_path",
            "text_excerpt",
        )
    )
    if node_type == "file" or kind == "file" or has_file_fields:
        return _simulation_http_compact_file_node(value)

    return dict(value)


def _simulation_http_trim_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    if not _SIMULATION_HTTP_TRIM_ENABLED:
        return catalog
    if not isinstance(catalog, dict):
        return {}

    trimmed = dict(catalog)

    items = catalog.get("items", [])
    if isinstance(items, list) and len(items) > _SIMULATION_HTTP_MAX_ITEMS:
        trimmed["items"] = list(items[:_SIMULATION_HTTP_MAX_ITEMS])

    file_graph = catalog.get("file_graph") if isinstance(catalog, dict) else None
    if isinstance(file_graph, dict):
        compact_file_graph = dict(file_graph)
        compact_file_nodes = _simulation_http_slice_rows(
            file_graph.get("file_nodes", []),
            max_rows=_SIMULATION_HTTP_MAX_FILE_NODES,
        )
        compact_file_graph["file_nodes"] = [
            compact_row
            for compact_row in (
                _simulation_http_compact_file_node(row) for row in compact_file_nodes
            )
            if compact_row is not None
        ]
        compact_file_graph["field_nodes"] = _simulation_http_slice_rows(
            file_graph.get("field_nodes", []),
            max_rows=_SIMULATION_HTTP_MAX_FIELD_NODES,
        )
        compact_file_graph["tag_nodes"] = _simulation_http_slice_rows(
            file_graph.get("tag_nodes", []),
            max_rows=_SIMULATION_HTTP_MAX_TAG_NODES,
        )
        compact_nodes = _simulation_http_slice_rows(
            file_graph.get("nodes", []),
            max_rows=_SIMULATION_HTTP_MAX_RENDER_NODES,
        )
        compact_file_graph["nodes"] = [
            compact_row
            for compact_row in (
                _simulation_http_compact_file_graph_node(row) for row in compact_nodes
            )
            if compact_row is not None
        ]
        compact_file_graph["edges"] = _simulation_http_slice_rows(
            file_graph.get("edges", []),
            max_rows=_SIMULATION_HTTP_MAX_FILE_EDGES,
        )

        compact_stats = (
            dict(file_graph.get("stats", {}))
            if isinstance(file_graph.get("stats", {}), dict)
            else {}
        )
        compact_stats["file_count"] = len(compact_file_graph.get("file_nodes", []))
        compact_stats["edge_count"] = len(compact_file_graph.get("edges", []))
        compact_file_graph["stats"] = compact_stats
        trimmed["file_graph"] = compact_file_graph

    crawler_graph = catalog.get("crawler_graph") if isinstance(catalog, dict) else None
    if isinstance(crawler_graph, dict):
        compact_crawler_graph = dict(crawler_graph)
        compact_crawler_graph["crawler_nodes"] = _simulation_http_slice_rows(
            crawler_graph.get("crawler_nodes", []),
            max_rows=_SIMULATION_HTTP_MAX_CRAWLER_NODES,
        )
        compact_crawler_graph["field_nodes"] = _simulation_http_slice_rows(
            crawler_graph.get("field_nodes", []),
            max_rows=_SIMULATION_HTTP_MAX_CRAWLER_FIELD_NODES,
        )
        compact_crawler_graph["nodes"] = _simulation_http_slice_rows(
            crawler_graph.get("nodes", []),
            max_rows=max(
                _SIMULATION_HTTP_MAX_CRAWLER_NODES,
                _SIMULATION_HTTP_MAX_CRAWLER_FIELD_NODES,
            ),
        )
        compact_crawler_graph["edges"] = _simulation_http_slice_rows(
            crawler_graph.get("edges", []),
            max_rows=_SIMULATION_HTTP_MAX_CRAWLER_EDGES,
        )
        compact_crawler_stats = (
            dict(crawler_graph.get("stats", {}))
            if isinstance(crawler_graph.get("stats", {}), dict)
            else {}
        )
        compact_crawler_stats["crawler_count"] = len(
            compact_crawler_graph.get("crawler_nodes", [])
        )
        compact_crawler_stats["edge_count"] = len(
            compact_crawler_graph.get("edges", [])
        )
        compact_crawler_graph["stats"] = compact_crawler_stats
        trimmed["crawler_graph"] = compact_crawler_graph

    return trimmed


def _simulation_ws_trim_simulation_payload(
    simulation: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(simulation, dict):
        return {}

    trimmed = dict(simulation)
    for key in (
        "field_particles",
        "nexus_graph",
        "logical_graph",
        "pain_field",
        "file_graph",
        "crawler_graph",
    ):
        trimmed.pop(key, None)
    return trimmed


def _simulation_http_compact_simulation_payload(
    simulation: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(simulation, dict):
        return {}

    compact = dict(simulation)
    for key in (
        "nexus_graph",
        "logical_graph",
        "pain_field",
        "file_graph",
        "crawler_graph",
    ):
        compact.pop(key, None)

    dynamics = compact.get("presence_dynamics")
    if isinstance(dynamics, dict) and isinstance(dynamics.get("field_particles"), list):
        compact.pop("field_particles", None)
    return compact


def _simulation_http_compact_response_body(body: bytes) -> bytes:
    if not body:
        return body
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body
    compact = _simulation_http_compact_simulation_payload(payload)
    return _json_compact(compact).encode("utf-8")


def _schedule_simulation_http_warmup(*, host: str, port: int) -> None:
    if not _SIMULATION_HTTP_WARMUP_ENABLED:
        return
    if int(port) <= 0:
        return

    warm_host = "127.0.0.1"
    if str(host).strip() in {"127.0.0.1", "localhost"}:
        warm_host = str(host).strip()
    warm_url = (
        f"http://{warm_host}:{int(port)}/api/simulation"
        f"?perspective={PROJECTION_DEFAULT_PERSPECTIVE}"
    )

    def _warm() -> None:
        delay = _SIMULATION_HTTP_WARMUP_DELAY_SECONDS
        if delay > 0.0:
            time.sleep(delay)
        for attempt in range(_SIMULATION_HTTP_WARMUP_MAX_ATTEMPTS):
            try:
                req = Request(warm_url, method="GET")
                with urlopen(
                    req, timeout=_SIMULATION_HTTP_WARMUP_TIMEOUT_SECONDS
                ) as resp:
                    resp.read(1)
                    if int(getattr(resp, "status", 0)) >= 200:
                        return
            except Exception:
                pass
            if attempt + 1 < _SIMULATION_HTTP_WARMUP_MAX_ATTEMPTS:
                time.sleep(_SIMULATION_HTTP_WARMUP_RETRY_SECONDS)

    threading.Thread(target=_warm, daemon=True, name="simulation-http-warmup").start()


def _simulation_http_disk_cache_path(part_root: Path, perspective: str) -> Path:
    key = str(perspective or PROJECTION_DEFAULT_PERSPECTIVE).strip().lower()
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in key)
    safe = safe.strip("_") or PROJECTION_DEFAULT_PERSPECTIVE
    return (part_root / "world_state" / f"simulation_http_cache_{safe}.json").resolve()


def _simulation_http_disk_cache_load(
    part_root: Path,
    *,
    perspective: str,
    max_age_seconds: float,
) -> bytes | None:
    if not _SIMULATION_HTTP_DISK_CACHE_ENABLED:
        return None
    cache_age_limit = max(0.0, _safe_float(max_age_seconds, 0.0))
    if cache_age_limit <= 0.0:
        return None

    cache_path = _simulation_http_disk_cache_path(part_root, perspective)
    try:
        if not cache_path.exists() or not cache_path.is_file():
            return None
        stat = cache_path.stat()
        age = time.time() - float(stat.st_mtime)
        if age < 0.0 or age > cache_age_limit:
            return None
        payload = cache_path.read_bytes()
        if not payload:
            return None
        return payload
    except Exception:
        return None


def _simulation_http_disk_cache_store(
    part_root: Path,
    *,
    perspective: str,
    body: bytes,
) -> None:
    if not _SIMULATION_HTTP_DISK_CACHE_ENABLED:
        return
    if not isinstance(body, (bytes, bytearray)):
        return
    payload = bytes(body)
    if not payload:
        return

    cache_path = _simulation_http_disk_cache_path(part_root, perspective)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(payload)
        tmp_path.replace(cache_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


_RUNTIME_CATALOG_SUBPROCESS_SCRIPT = (
    "import json,sys;"
    "from pathlib import Path;"
    "import code.world_web as ww;"
    "payload=ww.collect_catalog("
    "Path(sys.argv[1]),"
    "Path(sys.argv[2]),"
    "sync_inbox=False,"
    "include_pi_archive=False,"
    "include_world_log=False"
    ");"
    "sys.stdout.write(json.dumps(payload,ensure_ascii=False))"
)


def _effective_request_embed_model(model: str | None) -> str | None:
    force_nomic = str(
        os.getenv("OLLAMA_EMBED_FORCE_NOMIC", "0") or "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if force_nomic:
        return "nomic-embed-text"
    normalized = str(model or "").strip()
    return normalized or None


def _resolve_runtime_library_path(
    vault_root: Path,
    part_root: Path,
    request_path: str,
) -> Path | None:
    lib_path = resolve_library_path(vault_root, request_path)
    if lib_path is not None:
        return lib_path

    parsed = urlparse(request_path)
    raw_path = unquote(parsed.path)
    if not raw_path.startswith("/library/"):
        return None
    relative = raw_path.removeprefix("/library/")
    if not relative:
        return None

    candidate = (part_root.resolve() / relative).resolve()
    part_resolved = part_root.resolve()
    if candidate == part_resolved or part_resolved in candidate.parents:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _collect_runtime_catalog_isolated(
    part_root: Path,
    vault_root: Path,
) -> tuple[dict[str, Any] | None, str]:
    if not _RUNTIME_CATALOG_SUBPROCESS_ENABLED:
        return None, "catalog_subprocess_disabled"

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _RUNTIME_CATALOG_SUBPROCESS_SCRIPT,
                str(part_root),
                str(vault_root),
            ],
            capture_output=True,
            text=True,
            timeout=_RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except Exception as exc:
        return None, f"catalog_subprocess_failed:{exc.__class__.__name__}"

    if proc.returncode != 0:
        return None, f"catalog_subprocess_exit:{proc.returncode}"

    stdout = str(proc.stdout or "").strip()
    if not stdout:
        return None, "catalog_subprocess_empty_output"

    candidates = [stdout, *reversed(stdout.splitlines())]
    for candidate in candidates:
        line = str(candidate).strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload, ""

    return None, "catalog_subprocess_invalid_json"


def _fallback_kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type and mime_type.startswith("text/"):
        return "text"
    return "file"


def _fallback_rel_path(path: Path, vault_root: Path, part_root: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(vault_root.resolve())
        return str(rel_path).replace("\\", "/")
    except ValueError:
        try:
            rel_path = path.resolve().relative_to(part_root.resolve())
            return str(rel_path).replace("\\", "/")
        except ValueError:
            return path.name


def _runtime_catalog_fallback_items(
    part_root: Path,
    vault_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest = load_manifest(part_root)
    manifest_entries: list[dict[str, Any]] = []
    for key in ("files", "artifacts"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    manifest_entries.append(item)

    part_label = str(manifest.get("part") or manifest.get("name") or part_root.name)
    items: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    seen_paths: set[str] = set()

    for entry in manifest_entries:
        rel_source = str(entry.get("path", "")).strip()
        if not rel_source:
            continue
        candidate = (part_root / rel_source).resolve()
        if not candidate.exists() or not candidate.is_file():
            continue

        rel_path = _fallback_rel_path(candidate, vault_root, part_root)
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)

        kind = _fallback_kind_for_path(candidate)
        counts[kind] = int(counts.get(kind, 0)) + 1
        role = str(entry.get("role", "unknown")).strip() or "unknown"
        stat = candidate.stat()
        items.append(
            {
                "part": part_label,
                "name": candidate.name,
                "role": role,
                "display_name": {"en": candidate.name, "ja": candidate.name},
                "display_role": {"en": role, "ja": role},
                "kind": kind,
                "bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "rel_path": rel_path,
                "url": "/library/" + quote(rel_path),
            }
        )

    items.sort(
        key=lambda row: (
            str(row.get("part", "")),
            str(row.get("kind", "")),
            str(row.get("name", "")),
        ),
        reverse=True,
    )
    return items, counts


def _runtime_catalog_fallback(part_root: Path, vault_root: Path) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    fallback_items, fallback_counts = _runtime_catalog_fallback_items(
        part_root, vault_root
    )
    cover_fields = [
        {
            "id": str(item.get("rel_path", "")),
            "part": str(item.get("part", "")),
            "display_name": item.get("display_name", {}),
            "display_role": item.get("display_role", {}),
            "url": str(item.get("url", "")),
            "seed": hashlib.sha1(
                str(item.get("rel_path", "")).encode("utf-8")
            ).hexdigest(),
        }
        for item in fallback_items
        if str(item.get("role", "")) == "cover_art"
    ]
    inbox_stub = {
        "record": "ημ.inbox.v1",
        "path": str((part_root / ".ημ").resolve()),
        "pending_count": 0,
        "processed_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "rejected_count": 0,
        "deferred_count": 0,
        "is_empty": True,
        "knowledge_entries": 0,
        "registry_entries": 0,
        "last_ingested_at": "",
        "errors": [],
        "sync_status": "deferred",
    }
    file_graph_stub = {
        "record": "ημ.file-graph.v1",
        "generated_at": now_iso,
        "inbox": inbox_stub,
        "nodes": [],
        "field_nodes": [],
        "tag_nodes": [],
        "file_nodes": [],
        "edges": [],
        "stats": {
            "field_count": 0,
            "file_count": 0,
            "edge_count": 0,
            "kind_counts": {},
            "field_counts": {},
            "knowledge_entries": 0,
        },
    }
    crawler_graph_stub = {
        "record": "ημ.crawler-graph.v1",
        "generated_at": now_iso,
        "source": {"endpoint": "", "service": "weaver"},
        "status": {},
        "nodes": [],
        "field_nodes": [],
        "crawler_nodes": [],
        "edges": [],
        "stats": {
            "field_count": 0,
            "crawler_count": 0,
            "edge_count": 0,
            "kind_counts": {},
            "field_counts": {},
            "nodes_total": 0,
            "edges_total": 0,
            "url_nodes_total": 0,
        },
    }
    return {
        "generated_at": now_iso,
        "part_roots": [str(part_root.resolve())],
        "counts": fallback_counts,
        "canonical_terms": [],
        "entity_manifest": ENTITY_MANIFEST,
        "named_fields": build_named_field_overlays(ENTITY_MANIFEST),
        "ui_default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
        "ui_perspectives": projection_perspective_options(),
        "cover_fields": cover_fields,
        "eta_mu_inbox": inbox_stub,
        "file_graph": file_graph_stub,
        "crawler_graph": crawler_graph_stub,
        "truth_state": {},
        "test_failures": [],
        "test_coverage": {},
        "promptdb": {},
        "world_log": {
            "ok": True,
            "record": "ημ.world-log.v1",
            "generated_at": now_iso,
            "count": 0,
            "limit": 0,
            "pending_inbox": 0,
            "sources": {},
            "kinds": {},
            "relation_count": 0,
            "events": [],
        },
        "items": fallback_items,
        "pi_archive": {
            "record": "ημ.pi-archive.v1",
            "generated_at": "",
            "hash": {},
            "signature": {},
            "portable": {},
            "ledger_count": 0,
            "status": "deferred",
        },
        "runtime_state": "fallback",
    }


def _weaver_probe_host(bind_host: str) -> str:
    host = bind_host.strip().lower()
    if not host or host in {"0.0.0.0", "::", "localhost"}:
        return "127.0.0.1"
    return bind_host


def _weaver_health_check(host: str, port: int, timeout_s: float = 0.8) -> bool:
    target = f"http://{host}:{port}/healthz"
    req = Request(target, method="GET")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except Exception:
        return False


def _ensure_weaver_service(part_root: Path, world_host: str) -> None:
    del part_root
    if not WEAVER_AUTOSTART:
        return
    probe_host = _weaver_probe_host(WEAVER_HOST_ENV or world_host)
    if _weaver_health_check(probe_host, WEAVER_PORT):
        return

    script_path = (
        Path(__file__).resolve().parent.parent / "web_graph_weaver.js"
    ).resolve()
    if not script_path.exists() or not script_path.is_file():
        return
    node_binary = shutil.which("node")
    if not node_binary:
        return

    env = os.environ.copy()
    env.setdefault("WEAVER_HOST", WEAVER_HOST_ENV)
    env.setdefault("WEAVER_PORT", str(WEAVER_PORT))
    try:
        subprocess.Popen(
            [node_binary, str(script_path)],
            cwd=str(script_path.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return


def _parse_multipart_form(raw_body: bytes, content_type: str) -> dict[str, Any] | None:
    import re

    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type, re.I)
    if match is None:
        return None
    boundary_token = (match.group(1) or match.group(2) or "").strip()
    if not boundary_token:
        return None

    delimiter = b"--" + boundary_token.encode("utf-8", errors="ignore")
    data: dict[str, Any] = {}
    for part in raw_body.split(delimiter):
        chunk = part.strip()
        if not chunk or chunk == b"--":
            continue
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip()
        head, sep, body = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]

        disposition = ""
        part_content_type = ""
        for line in head.decode("utf-8", errors="ignore").split("\r\n"):
            low = line.lower()
            if low.startswith("content-disposition:"):
                disposition = line.split(":", 1)[1].strip()
            elif low.startswith("content-type:"):
                part_content_type = line.split(":", 1)[1].strip()

        name_match = re.search(r'name="([^"]+)"', disposition)
        if name_match is None:
            continue
        field_name = name_match.group(1)
        file_match = re.search(r'filename="([^"]*)"', disposition)
        if file_match is not None:
            data[field_name] = {
                "filename": file_match.group(1),
                "content_type": part_content_type,
                "value": body,
            }
        else:
            data[field_name] = body.decode("utf-8", errors="ignore")
    return data


def resolve_artifact_path(part_root: Path, request_path: str) -> Path | None:
    parsed = urlparse(request_path)
    raw_path = unquote(parsed.path)
    if not raw_path.startswith("/artifacts/"):
        return None

    relative = raw_path.removeprefix("/")
    if not relative:
        return None

    candidate = (part_root / relative).resolve()
    artifacts_root = (part_root / "artifacts").resolve()
    if artifacts_root == candidate or artifacts_root in candidate.parents:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def websocket_accept_value(client_key: str) -> str:
    accept_seed = client_key + WS_MAGIC
    digest = hashlib.sha1(accept_seed.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("utf-8")


_WS_CLIENT_FRAME_MAX_BYTES = 1_048_576


def websocket_frame(opcode: int, payload: bytes = b"") -> bytes:
    data = bytes(payload)
    length = len(data)
    header = bytearray([0x80 | (opcode & 0x0F)])
    if length <= 125:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + data


def websocket_frame_text(message: str) -> bytes:
    return websocket_frame(0x1, message.encode("utf-8"))


def _recv_ws_exact(connection: socket.socket, size: int) -> bytes | None:
    if size <= 0:
        return b""

    data = bytearray()
    while len(data) < size:
        try:
            chunk = connection.recv(size - len(data))
        except socket.timeout:
            if not data:
                raise
            continue
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _read_ws_client_frame(connection: socket.socket) -> tuple[int, bytes] | None:
    header = _recv_ws_exact(connection, 2)
    if header is None:
        return None

    first, second = header
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    payload_len = second & 0x7F

    if payload_len == 126:
        extended = _recv_ws_exact(connection, 2)
        if extended is None:
            return None
        payload_len = struct.unpack("!H", extended)[0]
    elif payload_len == 127:
        extended = _recv_ws_exact(connection, 8)
        if extended is None:
            return None
        payload_len = struct.unpack("!Q", extended)[0]

    if not masked or payload_len > _WS_CLIENT_FRAME_MAX_BYTES:
        return None

    mask_key = _recv_ws_exact(connection, 4)
    if mask_key is None:
        return None

    payload = _recv_ws_exact(connection, payload_len)
    if payload is None:
        return None

    if payload_len:
        payload = bytes(
            byte ^ mask_key[index % 4] for index, byte in enumerate(payload)
        )

    return opcode, payload


def _consume_ws_client_frame(connection: socket.socket) -> bool:
    frame = _read_ws_client_frame(connection)
    if frame is None:
        return False

    opcode, payload = frame
    if opcode == 0x8:
        try:
            connection.sendall(websocket_frame(0x8, payload[:125]))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        return False

    if opcode == 0x9:
        connection.sendall(websocket_frame(0xA, payload[:125]))
        return True

    if opcode in {0x0, 0x1, 0x2, 0xA}:
        return True

    return False


def render_index(payload: dict[str, Any], catalog: dict[str, Any]) -> str:
    del payload, catalog
    return ""


def _safe_bool_query(value: str, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _docker_simulation_identifier_set(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    direct_keys = ("id", "short_id", "name", "service")
    for key in direct_keys:
        clean = str(row.get(key, "") or "").strip().lower()
        if clean:
            values.add(clean)

    route = row.get("route", {}) if isinstance(row.get("route"), dict) else {}
    route_id = str(route.get("id", "") or "").strip().lower()
    if route_id:
        values.add(route_id)

    return values


def _find_docker_simulation_row(
    snapshot: dict[str, Any],
    identifier: str,
) -> dict[str, Any] | None:
    lookup = str(identifier or "").strip().lower()
    if not lookup:
        return None
    rows = snapshot.get("simulations", []) if isinstance(snapshot, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if lookup in _docker_simulation_identifier_set(row):
            return row
    return None


def _project_vector(embedding: list[float] | None) -> list[float]:
    if not isinstance(embedding, list) or not embedding:
        return [0.0, 0.0, 0.0]
    head = [float(value) for value in embedding[:3]]
    while len(head) < 3:
        head.append(0.0)
    max_mag = max(abs(head[0]), abs(head[1]), abs(head[2]), 1.0)
    return [
        round(head[0] / max_mag, 6),
        round(head[1] / max_mag, 6),
        round(head[2] / max_mag, 6),
    ]


def _normalize_audio_upload_name(file_name: str, mime: str) -> str:
    source_name = Path(str(file_name or "upload")).name
    source_name = source_name.replace("\x00", "").strip() or "upload"
    base = "".join(
        ch if (ch.isalnum() or ch in {"-", "_"}) else "-"
        for ch in Path(source_name).stem
    ).strip("-")
    if not base:
        base = "upload"

    ext = Path(source_name).suffix.lower()
    if not ext or len(ext) > 12:
        guessed_ext = mimetypes.guess_extension(mime or "") or ""
        ext = guessed_ext.lower() if guessed_ext else ".mp3"

    digest = hashlib.sha1(
        f"{source_name}|{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{base[:48]}-{digest}{ext}"


class WorldHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    part_root: Path = Path(".")
    vault_root: Path = Path("..")
    host_label: str = "127.0.0.1:8787"
    task_queue: TaskQueue
    council_chamber: CouncilChamber
    myth_tracker: Any
    life_tracker: Any
    life_interaction_builder: Any

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if isinstance(extra_headers, dict):
            for key, value in extra_headers.items():
                self.send_header(str(key), str(value))
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                OSError,
            ):
                pass

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        self._send_bytes(
            _json_compact(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_ws_event(self, payload: dict[str, Any]) -> None:
        frame = websocket_frame_text(_json_compact(payload))
        timeout_before = self.connection.gettimeout()
        timeout_overridden = False
        if timeout_before is not None:
            self.connection.settimeout(None)
            timeout_overridden = True
        try:
            self.connection.sendall(frame)
        finally:
            if timeout_overridden:
                self.connection.settimeout(timeout_before)

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return None
        raw = self.rfile.read(length)
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, dict) else None

    def _read_raw_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _muse_manager(self) -> Any:
        return get_muse_runtime_manager()

    def _muse_tool_callback(self, *, tool_name: str) -> dict[str, Any]:
        clean_tool = str(tool_name or "").strip().lower()
        if clean_tool == "study_snapshot":
            payload = build_study_snapshot(
                self.part_root,
                self.vault_root,
                queue_snapshot=self.task_queue.snapshot(include_pending=True),
                council_snapshot=self.council_chamber.snapshot(
                    include_decisions=True,
                    limit=16,
                ),
                drift_payload=build_drift_scan_payload(self.part_root, self.vault_root),
                truth_gate_blocked=None,
                resource_snapshot=_resource_monitor_snapshot(part_root=self.part_root),
            )
            return {
                "ok": True,
                "summary": "study snapshot generated",
                "record": str(payload.get("record", "")),
            }
        if clean_tool == "drift_scan":
            payload = build_drift_scan_payload(self.part_root, self.vault_root)
            return {
                "ok": True,
                "summary": "drift scan generated",
                "blocked_gates": len(payload.get("blocked_gates", [])),
            }
        if clean_tool == "push_truth_dry_run":
            payload = build_push_truth_dry_run_payload(self.part_root, self.vault_root)
            gate = payload.get("gate", {}) if isinstance(payload, dict) else {}
            return {
                "ok": True,
                "summary": "push-truth dry run generated",
                "blocked": bool(gate.get("blocked", False))
                if isinstance(gate, dict)
                else False,
            }
        return {"ok": False, "error": "unsupported_tool"}

    def _muse_reply_builder(
        self,
        *,
        messages: list[dict[str, Any]],
        context_block: str,
        mode: str,
        muse_id: str = "",
        turn_id: str = "",
    ) -> dict[str, Any]:
        del turn_id
        model_mode = (
            "canonical" if str(mode).strip().lower() == "deterministic" else "ollama"
        )
        clean_muse_id = str(muse_id or "").strip() or "witness_thread"
        response = build_chat_reply(
            messages=[
                {"role": "system", "text": context_block},
                *messages,
            ],
            mode=model_mode,
            context=build_world_payload(self.part_root),
            multi_entity=True,
            presence_ids=[clean_muse_id],
        )
        if not isinstance(response, dict):
            return {"reply": "", "mode": "canonical", "model": None}
        return {
            "reply": str(response.get("reply", "") or "").strip(),
            "mode": str(response.get("mode", model_mode) or model_mode),
            "model": response.get("model"),
        }

    def _collect_catalog_fast(self) -> dict[str, Any]:
        return collect_catalog(
            self.part_root,
            self.vault_root,
            sync_inbox=False,
            include_pi_archive=False,
            include_world_log=False,
        )

    def _schedule_runtime_inbox_sync(self) -> None:
        if not _RUNTIME_INBOX_SYNC_LOCK.acquire(blocking=False):
            return

        def _sync() -> None:
            try:
                snapshot = sync_eta_mu_inbox(self.vault_root)
                now_monotonic = time.monotonic()
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = now_monotonic
                    _RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = dict(snapshot)
                    _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""
                    cached_catalog = _RUNTIME_CATALOG_CACHE.get("catalog")
                    if isinstance(cached_catalog, dict):
                        next_catalog = dict(cached_catalog)
                        next_catalog["eta_mu_inbox"] = dict(snapshot)
                        _RUNTIME_CATALOG_CACHE["catalog"] = next_catalog
            except Exception as sync_exc:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = (
                        f"inbox_sync_failed:{sync_exc.__class__.__name__}"
                    )
            finally:
                _RUNTIME_INBOX_SYNC_LOCK.release()

        threading.Thread(target=_sync, daemon=True).start()

    def _schedule_runtime_catalog_refresh(self) -> None:
        if not _RUNTIME_CATALOG_REFRESH_LOCK.acquire(blocking=False):
            return

        def _refresh() -> None:
            try:
                now_monotonic = time.monotonic()
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    last_sync = float(
                        _RUNTIME_CATALOG_CACHE.get("inbox_sync_monotonic", 0.0)
                    )
                    previous_sync_snapshot = _RUNTIME_CATALOG_CACHE.get(
                        "inbox_sync_snapshot"
                    )
                should_sync = (
                    now_monotonic - last_sync >= _RUNTIME_ETA_MU_SYNC_SECONDS
                    or previous_sync_snapshot is None
                )

                fresh_catalog = self._collect_catalog_fast()
                if isinstance(previous_sync_snapshot, dict):
                    fresh_catalog["eta_mu_inbox"] = dict(previous_sync_snapshot)
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["catalog"] = fresh_catalog
                    _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = time.monotonic()
                    _RUNTIME_CATALOG_CACHE["last_error"] = ""

                if should_sync:
                    self._schedule_runtime_inbox_sync()
            except Exception as exc:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["last_error"] = (
                        f"catalog_refresh_failed:{exc.__class__.__name__}"
                    )
            finally:
                _RUNTIME_CATALOG_REFRESH_LOCK.release()

        threading.Thread(target=_refresh, daemon=True).start()

    def _runtime_catalog_base(self) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        with _RUNTIME_CATALOG_CACHE_LOCK:
            cached_catalog = _RUNTIME_CATALOG_CACHE.get("catalog")
            refreshed_monotonic = float(
                _RUNTIME_CATALOG_CACHE.get("refreshed_monotonic", 0.0)
            )
            inbox_sync_snapshot = _RUNTIME_CATALOG_CACHE.get("inbox_sync_snapshot")

        if not isinstance(cached_catalog, dict):
            fresh_catalog, isolated_error = _collect_runtime_catalog_isolated(
                self.part_root,
                self.vault_root,
            )
            try:
                if fresh_catalog is None:
                    fresh_catalog = self._collect_catalog_fast()
                cache_error = isolated_error
                if cache_error == "catalog_subprocess_disabled":
                    cache_error = ""
                if isinstance(inbox_sync_snapshot, dict):
                    fresh_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["catalog"] = fresh_catalog
                    _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = time.monotonic()
                    _RUNTIME_CATALOG_CACHE["last_error"] = cache_error
                self._schedule_runtime_inbox_sync()
                return dict(fresh_catalog)
            except Exception as exc:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["last_error"] = (
                        f"catalog_inline_failed:{exc.__class__.__name__}"
                    )

        cache_age = now_monotonic - refreshed_monotonic
        if cache_age >= _RUNTIME_CATALOG_CACHE_SECONDS:
            self._schedule_runtime_catalog_refresh()

        if isinstance(cached_catalog, dict):
            return dict(cached_catalog)
        fallback_catalog = _runtime_catalog_fallback(self.part_root, self.vault_root)
        if isinstance(inbox_sync_snapshot, dict):
            fallback_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
        return fallback_catalog

    def _runtime_catalog(
        self,
        *,
        perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
        include_projection: bool = True,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        catalog = self._runtime_catalog_base()
        queue_snapshot = self.task_queue.snapshot(include_pending=False)
        council_snapshot = self.council_chamber.snapshot(include_decisions=False)
        catalog["task_queue"] = queue_snapshot
        catalog["council"] = council_snapshot

        resource_snapshot = _resource_monitor_snapshot(part_root=self.part_root)
        _INFLUENCE_TRACKER.record_resource_heartbeat(
            resource_snapshot,
            source="runtime.catalog",
        )
        influence_snapshot = _INFLUENCE_TRACKER.snapshot(
            queue_snapshot=queue_snapshot,
            part_root=self.part_root,
        )
        catalog["presence_runtime"] = influence_snapshot
        muse_runtime = self._muse_manager().snapshot()
        catalog["muse_runtime"] = muse_runtime

        if include_projection:
            attach_ui_projection(
                catalog,
                perspective=perspective,
                queue_snapshot=queue_snapshot,
                influence_snapshot=influence_snapshot,
            )
        return (
            catalog,
            queue_snapshot,
            council_snapshot,
            influence_snapshot,
            resource_snapshot,
        )

    def _runtime_simulation(
        self,
        catalog: dict[str, Any],
        queue_snapshot: dict[str, Any],
        influence_snapshot: dict[str, Any],
        *,
        perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        simulation_catalog = _simulation_http_trim_catalog(catalog)
        try:
            myth_summary = self.myth_tracker.snapshot(simulation_catalog)
        except Exception:
            myth_summary = {}
        try:
            world_summary = self.life_tracker.snapshot(
                simulation_catalog,
                myth_summary,
                ENTITY_MANIFEST,
            )
        except Exception:
            world_summary = {}

        docker_snapshot = collect_docker_simulation_snapshot()
        simulation = build_simulation_state(
            simulation_catalog,
            myth_summary,
            world_summary,
            influence_snapshot=influence_snapshot,
            queue_snapshot=queue_snapshot,
            docker_snapshot=docker_snapshot,
        )
        projection = build_ui_projection(
            simulation_catalog,
            simulation,
            perspective=perspective,
            queue_snapshot=queue_snapshot,
            influence_snapshot=influence_snapshot,
        )
        simulation["projection"] = projection
        simulation["perspective"] = perspective
        return simulation, projection

    def _handle_docker_websocket(self) -> None:
        ws_key = str(self.headers.get("Sec-WebSocket-Key", "")).strip()
        if not ws_key:
            self._send_bytes(
                b"missing websocket key",
                "text/plain; charset=utf-8",
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if not _runtime_ws_try_acquire_client_slot():
            self._send_json(
                {
                    "ok": False,
                    "error": "websocket_capacity_reached",
                    "record": "eta-mu.runtime-health.v1",
                    "websocket": _runtime_ws_client_snapshot(),
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        try:
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", websocket_accept_value(ws_key))
            self.end_headers()
            self.close_connection = True
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ):
            _runtime_ws_release_client_slot()
            return

        self.connection.settimeout(1.0)
        last_docker_refresh = time.monotonic()
        last_docker_broadcast = last_docker_refresh
        last_docker_fingerprint = ""

        try:
            try:
                docker_snapshot = collect_docker_simulation_snapshot(force_refresh=True)
                last_docker_fingerprint = str(
                    docker_snapshot.get("fingerprint", "") or ""
                )
                self._send_ws_event(
                    {
                        "type": "docker_simulations",
                        "docker": docker_snapshot,
                    }
                )
            except Exception as exc:
                self._send_ws_event(
                    {
                        "type": "docker_simulations_error",
                        "error": exc.__class__.__name__,
                    }
                )

            while True:
                now_monotonic = time.monotonic()
                if (
                    now_monotonic - last_docker_refresh
                    >= DOCKER_SIMULATION_WS_REFRESH_SECONDS
                ):
                    docker_snapshot = collect_docker_simulation_snapshot()
                    docker_fingerprint = str(
                        docker_snapshot.get("fingerprint", "") or ""
                    )
                    docker_changed = docker_fingerprint != last_docker_fingerprint
                    docker_heartbeat_due = (
                        now_monotonic - last_docker_broadcast
                        >= DOCKER_SIMULATION_WS_HEARTBEAT_SECONDS
                    )
                    if docker_changed or docker_heartbeat_due:
                        self._send_ws_event(
                            {
                                "type": "docker_simulations",
                                "docker": docker_snapshot,
                            }
                        )
                        last_docker_broadcast = now_monotonic
                        last_docker_fingerprint = docker_fingerprint
                    last_docker_refresh = now_monotonic

                try:
                    if not _consume_ws_client_frame(self.connection):
                        break
                except socket.timeout:
                    continue
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ):
            pass
        finally:
            _runtime_ws_release_client_slot()

    def _handle_websocket(self, *, perspective: str) -> None:
        ws_key = str(self.headers.get("Sec-WebSocket-Key", "")).strip()
        if not ws_key:
            self._send_bytes(
                b"missing websocket key",
                "text/plain; charset=utf-8",
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if not _runtime_ws_try_acquire_client_slot():
            self._send_json(
                {
                    "ok": False,
                    "error": "websocket_capacity_reached",
                    "record": "eta-mu.runtime-health.v1",
                    "websocket": _runtime_ws_client_snapshot(),
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        try:
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", websocket_accept_value(ws_key))
            self.end_headers()
            self.close_connection = True
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ):
            _runtime_ws_release_client_slot()
            return

        perspective_key = normalize_projection_perspective(perspective)
        self.connection.settimeout(1.0)
        catalog: dict[str, Any] = {}
        queue_snapshot: dict[str, Any] = {}
        influence_snapshot: dict[str, Any] = {}
        last_simulation_for_delta: dict[str, Any] | None = None
        muse_event_seq = 0
        last_docker_fingerprint = ""

        try:
            catalog, queue_snapshot, _, influence_snapshot, _ = self._runtime_catalog(
                perspective=perspective_key,
                include_projection=False,
            )
            catalog_payload = _simulation_http_trim_catalog(catalog)
            _, mix_meta = build_mix_stream(catalog, self.vault_root)
            self._send_ws_event(
                {
                    "type": "catalog",
                    "catalog": catalog_payload,
                    "mix": mix_meta,
                }
            )
            muse_bootstrap_events = self._muse_manager().list_events(
                since_seq=0,
                limit=96,
            )
            if muse_bootstrap_events:
                muse_event_seq = max(
                    int(row.get("seq", 0))
                    for row in muse_bootstrap_events
                    if isinstance(row, dict)
                )
                self._send_ws_event(
                    {
                        "type": "muse_events",
                        "events": muse_bootstrap_events,
                        "since_seq": 0,
                        "next_seq": muse_event_seq,
                    }
                )

            simulation, projection = self._runtime_simulation(
                catalog,
                queue_snapshot,
                influence_snapshot,
                perspective=perspective_key,
            )
            simulation_payload = _simulation_ws_trim_simulation_payload(simulation)
            self._send_ws_event(
                {
                    "type": "simulation",
                    "simulation": simulation_payload,
                    "projection": projection,
                }
            )
            last_simulation_for_delta = simulation_payload

            docker_snapshot = collect_docker_simulation_snapshot()
            self._send_ws_event(
                {
                    "type": "docker_simulations",
                    "docker": docker_snapshot,
                }
            )
            last_docker_fingerprint = str(docker_snapshot.get("fingerprint", "") or "")
        except Exception:
            try:
                self._send_ws_event(
                    {
                        "type": "error",
                        "error": "initial websocket payload failed",
                    }
                )
            except Exception:
                _runtime_ws_release_client_slot()
                return
            _runtime_ws_release_client_slot()
            return

        last_catalog_refresh = time.monotonic()
        last_catalog_broadcast = last_catalog_refresh
        last_sim_tick = last_catalog_refresh
        last_docker_refresh = last_catalog_refresh
        last_docker_broadcast = last_catalog_refresh
        last_runtime_guard_broadcast = last_catalog_refresh
        runtime_guard: dict[str, Any] = {
            "mode": "normal",
            "reasons": [],
        }

        try:
            while True:
                now_monotonic = time.monotonic()
                if now_monotonic - last_catalog_refresh >= CATALOG_REFRESH_SECONDS:
                    catalog, queue_snapshot, _, influence_snapshot, _ = (
                        self._runtime_catalog(
                            perspective=perspective_key,
                            include_projection=False,
                        )
                    )
                    last_catalog_refresh = now_monotonic

                if (
                    now_monotonic - last_runtime_guard_broadcast
                    >= _RUNTIME_GUARD_HEARTBEAT_SECONDS
                ):
                    runtime_health = _runtime_health_payload(self.part_root)
                    runtime_guard = runtime_health.get("guard", {})
                    self._send_ws_event(
                        {
                            "type": "runtime_health",
                            "runtime": runtime_health,
                        }
                    )
                    last_runtime_guard_broadcast = now_monotonic

                guard_mode = str(runtime_guard.get("mode", "normal") or "normal")
                load_scale = 1.0
                if guard_mode == "critical":
                    load_scale = _RUNTIME_GUARD_INTERVAL_SCALE
                elif guard_mode == "degraded":
                    load_scale = max(1.0, _RUNTIME_GUARD_INTERVAL_SCALE * 0.6)

                catalog_broadcast_interval = (
                    CATALOG_BROADCAST_HEARTBEAT_SECONDS * load_scale
                )
                sim_tick_interval = SIM_TICK_SECONDS * load_scale
                docker_refresh_interval = (
                    DOCKER_SIMULATION_WS_REFRESH_SECONDS * load_scale
                )
                simulation_guard_skip = (
                    _RUNTIME_GUARD_SKIP_SIMULATION_ON_CRITICAL
                    and guard_mode == "critical"
                )

                if now_monotonic - last_catalog_broadcast >= catalog_broadcast_interval:
                    catalog_payload = _simulation_http_trim_catalog(catalog)
                    _, mix_meta = build_mix_stream(catalog, self.vault_root)
                    self._send_ws_event(
                        {
                            "type": "catalog",
                            "catalog": catalog_payload,
                            "mix": mix_meta,
                        }
                    )
                    last_catalog_broadcast = now_monotonic

                if now_monotonic - last_sim_tick >= sim_tick_interval:
                    if simulation_guard_skip:
                        self._send_ws_event(
                            {
                                "type": "simulation_guard",
                                "record": "eta-mu.simulation-guard.v1",
                                "mode": guard_mode,
                                "reasons": runtime_guard.get("reasons", []),
                                "skipped": True,
                                "interval_scale": load_scale,
                            }
                        )
                    else:
                        simulation, projection = self._runtime_simulation(
                            catalog,
                            queue_snapshot,
                            influence_snapshot,
                            perspective=perspective_key,
                        )
                        simulation_payload = _simulation_ws_trim_simulation_payload(
                            simulation
                        )
                        self._send_ws_event(
                            {
                                "type": "simulation",
                                "simulation": simulation_payload,
                                "projection": projection,
                            }
                        )
                        delta = build_simulation_delta(
                            last_simulation_for_delta,
                            simulation_payload,
                        )
                        if bool(delta.get("has_changes", False)):
                            self._send_ws_event(
                                {
                                    "type": "simulation_delta",
                                    "delta": delta,
                                }
                            )
                        last_simulation_for_delta = simulation_payload

                    previous_muse_seq = muse_event_seq
                    muse_events = self._muse_manager().list_events(
                        since_seq=previous_muse_seq,
                        limit=96,
                    )
                    if muse_events:
                        muse_event_seq = max(
                            previous_muse_seq,
                            max(
                                int(row.get("seq", 0))
                                for row in muse_events
                                if isinstance(row, dict)
                            ),
                        )
                        self._send_ws_event(
                            {
                                "type": "muse_events",
                                "events": muse_events,
                                "since_seq": previous_muse_seq,
                                "next_seq": muse_event_seq,
                            }
                        )
                    last_sim_tick = now_monotonic

                if now_monotonic - last_docker_refresh >= docker_refresh_interval:
                    docker_snapshot = collect_docker_simulation_snapshot()
                    docker_fingerprint = str(
                        docker_snapshot.get("fingerprint", "") or ""
                    )
                    docker_changed = docker_fingerprint != last_docker_fingerprint
                    docker_heartbeat_due = (
                        now_monotonic - last_docker_broadcast
                        >= DOCKER_SIMULATION_WS_HEARTBEAT_SECONDS
                    )
                    if docker_changed or docker_heartbeat_due:
                        self._send_ws_event(
                            {
                                "type": "docker_simulations",
                                "docker": docker_snapshot,
                            }
                        )
                        last_docker_broadcast = now_monotonic
                        last_docker_fingerprint = docker_fingerprint
                    last_docker_refresh = now_monotonic

                try:
                    if not _consume_ws_client_frame(self.connection):
                        break
                except socket.timeout:
                    continue
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ):
            pass
        finally:
            _runtime_ws_release_client_slot()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors_headers()
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/ws":
            stream = str(params.get("stream", [""])[0] or "").strip().lower()
            if stream in {"docker", "meta", "simulations"}:
                self._handle_docker_websocket()
                return

            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            self._handle_websocket(perspective=perspective)
            return

        if parsed.path == "/api/voice-lines":
            mode = str(params.get("mode", ["canonical"])[0] or "canonical")
            payload_voice = build_voice_lines(
                "ollama" if mode.strip().lower() == "ollama" else "canonical"
            )
            self._send_json(payload_voice)
            return

        if parsed.path == "/healthz":
            payload = build_world_payload(self.part_root)
            catalog = self._runtime_catalog_base()
            file_count = len(catalog.get("items", []))
            entropy = int(time.time() * 1000) % 100
            self._send_json(
                {
                    "ok": True,
                    "status": "alive",
                    "organism": {
                        "spore_count": file_count,
                        "mycelial_density": 1.0,
                        "pulse_rate": "78bpm",
                        "substrate_entropy": f"{entropy}%",
                        "growth_phase": "fruiting",
                    },
                    "part": payload.get("part"),
                    "items": file_count,
                }
            )
            return

        if parsed.path == "/api/mix":
            catalog, _, _, _, _ = self._runtime_catalog(
                perspective=PROJECTION_DEFAULT_PERSPECTIVE,
                include_projection=False,
            )
            _, mix_meta = build_mix_stream(catalog, self.vault_root)
            self._send_json(mix_meta)
            return

        if parsed.path == "/api/tts":
            text = str(params.get("text", [""])[0] or "").strip()
            speed = str(params.get("speed", ["1.0"])[0] or "1.0").strip()

            if not text:
                self._send_json(
                    {"ok": False, "error": "empty text"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            tts_url = f"{TTS_BASE_URL}/tts?text={quote(text)}&speed={speed}"
            try:
                with urlopen(Request(tts_url, method="GET"), timeout=30) as resp:
                    self._send_bytes(resp.read(), "audio/wav")
                return
            except Exception as sidecar_error:
                fallback_error = ""
                speed_ratio = max(0.6, min(1.6, _safe_float(speed, 1.0)))
                words_per_minute = int(round(max(90, min(320, 170 * speed_ratio))))
                safe_text = text[:600]

                with tempfile.NamedTemporaryFile(
                    prefix="eta_mu_tts_",
                    suffix=".wav",
                    delete=False,
                ) as tmp_file:
                    fallback_path = Path(tmp_file.name)

                try:
                    command_candidates = (
                        [
                            "espeak-ng",
                            "-s",
                            str(words_per_minute),
                            "-w",
                            str(fallback_path),
                            safe_text,
                        ],
                        [
                            "espeak",
                            "-s",
                            str(words_per_minute),
                            "-w",
                            str(fallback_path),
                            safe_text,
                        ],
                    )
                    rendered = False
                    for command in command_candidates:
                        try:
                            result = subprocess.run(
                                command,
                                check=False,
                                capture_output=True,
                                text=True,
                                timeout=18,
                            )
                        except FileNotFoundError:
                            continue

                        if result.returncode != 0:
                            fallback_error = (
                                result.stderr or result.stdout or ""
                            ).strip()
                            continue

                        if (
                            not fallback_path.exists()
                            or fallback_path.stat().st_size <= 44
                        ):
                            fallback_error = "fallback wav missing or empty"
                            continue

                        self._send_bytes(fallback_path.read_bytes(), "audio/wav")
                        rendered = True
                        break

                    if rendered:
                        return

                    if not fallback_error:
                        fallback_error = "espeak fallback unavailable"
                except Exception as fallback_exc:
                    fallback_error = str(fallback_exc)
                finally:
                    try:
                        fallback_path.unlink(missing_ok=True)
                    except OSError:
                        pass

                self._send_json(
                    {
                        "ok": False,
                        "error": str(sidecar_error),
                        "fallback_error": fallback_error,
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/embeddings/db/status":
            self._send_json(_embedding_db_status(self.vault_root))
            return

        if parsed.path == "/api/embeddings/provider/options":
            self._send_json(_embedding_provider_options())
            return

        if parsed.path == "/api/embeddings/db/list":
            limit = max(
                1,
                min(
                    500,
                    int(_safe_float(str(params.get("limit", ["50"])[0] or "50"), 50.0)),
                ),
            )
            include_vectors = _safe_bool_query(
                str(params.get("include_vectors", ["false"])[0] or "false"),
                default=False,
            )
            self._send_json(
                _embedding_db_list(
                    self.vault_root,
                    limit=limit,
                    include_vectors=include_vectors,
                )
            )
            return

        if parsed.path == "/api/presence/accounts":
            limit = max(
                1,
                min(
                    500,
                    int(_safe_float(str(params.get("limit", ["64"])[0] or "64"), 64.0)),
                ),
            )
            self._send_json(_list_presence_accounts(self.vault_root, limit=limit))
            return

        if parsed.path == "/api/image/comments":
            image_ref = str(params.get("image_ref", [""])[0] or "").strip()
            limit = max(
                1,
                min(
                    1000,
                    int(
                        _safe_float(
                            str(params.get("limit", ["120"])[0] or "120"),
                            120.0,
                        )
                    ),
                ),
            )
            self._send_json(
                _list_image_comments(
                    self.vault_root,
                    image_ref=image_ref,
                    limit=limit,
                )
            )
            return

        if parsed.path == "/api/world/events":
            limit = max(
                12,
                min(
                    800,
                    int(
                        _safe_float(
                            str(params.get("limit", ["180"])[0] or "180"),
                            180.0,
                        )
                    ),
                ),
            )
            self._send_json(
                build_world_log_payload(
                    self.part_root,
                    self.vault_root,
                    limit=limit,
                )
            )
            return

        if parsed.path == "/api/simulation/presets":
            presets_path = self.part_root / "world_state" / "sim_presets.json"
            if presets_path.exists():
                self._send_bytes(presets_path.read_bytes(), "application/json")
            else:
                self._send_json({"presets": []})
            return

        if parsed.path == "/api/simulation/instances":
            try:
                res = subprocess.check_output(
                    [sys.executable, "scripts/sim_manager.py", "list-active"],
                    cwd=self.part_root,
                    text=True,
                )
                self._send_bytes(res.encode("utf-8"), "application/json")
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/catalog":
            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            catalog, _, _, _, _ = self._runtime_catalog(perspective=perspective)
            self._send_json(_simulation_http_trim_catalog(catalog))
            return

        if parsed.path == "/api/docker/simulations":
            refresh = _safe_bool_query(
                str(params.get("refresh", ["false"])[0] or "false"),
                default=False,
            )
            self._send_json(collect_docker_simulation_snapshot(force_refresh=refresh))
            return

        if parsed.path == "/api/runtime/health":
            self._send_json(_runtime_health_payload(self.part_root))
            return

        if parsed.path == "/api/meta/notes":
            limit = max(
                1,
                min(
                    256,
                    int(_safe_float(str(params.get("limit", ["24"])[0] or "24"), 24.0)),
                ),
            )
            tag = str(params.get("tag", [""])[0] or "").strip()
            target = str(params.get("target", [""])[0] or "").strip()
            category = str(params.get("category", [""])[0] or "").strip()
            severity = str(params.get("severity", [""])[0] or "").strip()
            self._send_json(
                list_meta_notes(
                    self.vault_root,
                    limit=limit,
                    tag=tag,
                    target=target,
                    category=category,
                    severity=severity,
                )
            )
            return

        if parsed.path == "/api/meta/runs":
            limit = max(
                1,
                min(
                    256,
                    int(_safe_float(str(params.get("limit", ["24"])[0] or "24"), 24.0)),
                ),
            )
            run_type = str(params.get("run_type", [""])[0] or "").strip()
            status = str(params.get("status", [""])[0] or "").strip()
            target = str(params.get("target", [""])[0] or "").strip()
            self._send_json(
                list_meta_runs(
                    self.vault_root,
                    limit=limit,
                    run_type=run_type,
                    status=status,
                    target=target,
                )
            )
            return

        if parsed.path == "/api/simulation/metadata":
            if self.command == "POST":
                req = self._read_json_body() or {}
                result = _upsert_simulation_metadata(
                    self.vault_root,
                    presence_id=str(req.get("presence_id", "") or "").strip(),
                    label=str(req.get("label", "") or "").strip(),
                    description=str(req.get("description", "") or "").strip(),
                    tags=[
                        str(item).strip()
                        for item in req.get("tags", [])
                        if str(item).strip()
                    ],
                    process_info=req.get("process_info"),
                    benchmark_results=req.get("benchmark_results"),
                )
                status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
                self._send_json(result, status=status)
                return

            # GET
            limit_raw = params.get("limit", ["64"])[0]
            try:
                limit = int(limit_raw)
            except (ValueError, TypeError):
                limit = 64
            self._send_json(_list_simulation_metadata(self.vault_root, limit=limit))
            return

        if parsed.path == "/api/meta/overview":
            notes_limit = max(
                1,
                min(
                    128,
                    int(
                        _safe_float(
                            str(params.get("notes_limit", ["12"])[0] or "12"),
                            12.0,
                        )
                    ),
                ),
            )
            runs_limit = max(
                1,
                min(
                    128,
                    int(
                        _safe_float(
                            str(params.get("runs_limit", ["12"])[0] or "12"),
                            12.0,
                        )
                    ),
                ),
            )
            refresh = _safe_bool_query(
                str(params.get("refresh", ["false"])[0] or "false"),
                default=False,
            )
            docker_snapshot = collect_docker_simulation_snapshot(force_refresh=refresh)
            queue_snapshot = self.task_queue.snapshot(include_pending=True)
            self._send_json(
                build_meta_overview(
                    self.vault_root,
                    docker_snapshot=docker_snapshot,
                    queue_snapshot=queue_snapshot,
                    notes_limit=notes_limit,
                    runs_limit=runs_limit,
                )
            )
            return

        if parsed.path == "/api/muse/runtime":
            manager = self._muse_manager()
            self._send_json({"ok": True, "runtime": manager.snapshot()})
            return

        if parsed.path == "/api/muse/events":
            manager = self._muse_manager()
            muse_id = str(params.get("muse_id", [""])[0] or "").strip()
            since_seq = max(
                0,
                int(
                    _safe_float(
                        str(params.get("since_seq", ["0"])[0] or "0"),
                        0.0,
                    )
                ),
            )
            limit = max(
                1,
                min(
                    512,
                    int(
                        _safe_float(
                            str(params.get("limit", ["96"])[0] or "96"),
                            96.0,
                        )
                    ),
                ),
            )
            events = manager.list_events(
                muse_id=muse_id,
                since_seq=since_seq,
                limit=limit,
            )
            next_seq = since_seq
            if events:
                next_seq = max(
                    int(row.get("seq", since_seq))
                    for row in events
                    if isinstance(row, dict)
                )
            self._send_json(
                {
                    "ok": True,
                    "record": "eta-mu.muse-event-page.v1",
                    "muse_id": muse_id,
                    "since_seq": since_seq,
                    "next_seq": next_seq,
                    "events": events,
                }
            )
            return

        if parsed.path == "/api/muse/context":
            manager = self._muse_manager()
            muse_id = str(params.get("muse_id", [""])[0] or "").strip()
            turn_id = str(params.get("turn_id", [""])[0] or "").strip()
            if not muse_id or not turn_id:
                self._send_json(
                    {"ok": False, "error": "muse_id_and_turn_id_required"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            manifest = manager.get_context_manifest(muse_id, turn_id)
            if not isinstance(manifest, dict):
                self._send_json(
                    {"ok": False, "error": "context_manifest_not_found"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_json({"ok": True, "manifest": manifest})
            return

        if parsed.path == "/api/zips":
            member_limit = int(
                _safe_float(
                    str(params.get("member_limit", ["220"])[0] or "220"),
                    220.0,
                )
            )
            self._send_json(
                collect_zip_catalog(
                    self.part_root,
                    self.vault_root,
                    member_limit=member_limit,
                )
            )
            return

        if parsed.path == "/api/pi/archive":
            catalog = self._collect_catalog_fast()
            queue_snapshot = self.task_queue.snapshot(include_pending=False)
            archive = build_pi_archive_payload(
                self.part_root,
                self.vault_root,
                catalog=catalog,
                queue_snapshot=queue_snapshot,
            )
            self._send_json({"ok": True, "archive": archive})
            return

        if parsed.path == "/api/ui/projection":
            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            catalog, queue_snapshot, _, influence_snapshot, _ = self._runtime_catalog(
                perspective=perspective,
                include_projection=False,
            )
            simulation, projection = self._runtime_simulation(
                catalog,
                queue_snapshot,
                influence_snapshot,
                perspective=perspective,
            )
            self._send_json(
                {
                    "ok": True,
                    "projection": projection,
                    "simulation": simulation,
                    "default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
                    "perspectives": projection_perspective_options(),
                }
            )
            return

        if parsed.path == "/api/task/queue":
            self._send_json(
                {
                    "ok": True,
                    "queue": self.task_queue.snapshot(include_pending=True),
                }
            )
            return

        if parsed.path == "/api/council":
            limit = max(
                1,
                min(
                    128,
                    int(_safe_float(str(params.get("limit", ["16"])[0] or "16"), 16.0)),
                ),
            )
            self._send_json(
                {
                    "ok": True,
                    "council": self.council_chamber.snapshot(
                        include_decisions=True,
                        limit=limit,
                    ),
                }
            )
            return

        if parsed.path == "/api/study":
            limit = max(
                1,
                min(
                    128,
                    int(_safe_float(str(params.get("limit", ["16"])[0] or "16"), 16.0)),
                ),
            )
            include_truth_state = _safe_bool_query(
                str(params.get("include_truth", ["false"])[0] or "false"),
                default=False,
            )
            queue_snapshot = self.task_queue.snapshot(include_pending=True)
            council_snapshot = self.council_chamber.snapshot(
                include_decisions=True,
                limit=limit,
            )
            drift_payload = build_drift_scan_payload(self.part_root, self.vault_root)
            resource_snapshot = _resource_monitor_snapshot(part_root=self.part_root)
            _INFLUENCE_TRACKER.record_resource_heartbeat(
                resource_snapshot,
                source="api.study",
            )

            truth_gate_blocked: bool | None = None
            if include_truth_state:
                try:
                    truth_state = self._collect_catalog_fast().get("truth_state", {})
                    gate = (
                        truth_state.get("gate", {})
                        if isinstance(truth_state, dict)
                        else {}
                    )
                    if isinstance(gate, dict):
                        truth_gate_blocked = bool(gate.get("blocked", False))
                except Exception:
                    truth_gate_blocked = None

            self._send_json(
                build_study_snapshot(
                    self.part_root,
                    self.vault_root,
                    queue_snapshot=queue_snapshot,
                    council_snapshot=council_snapshot,
                    drift_payload=drift_payload,
                    truth_gate_blocked=truth_gate_blocked,
                    resource_snapshot=resource_snapshot,
                )
            )
            return

        if parsed.path == "/api/study/history":
            limit = max(
                1,
                min(
                    256,
                    int(_safe_float(str(params.get("limit", ["16"])[0] or "16"), 16.0)),
                ),
            )
            events = _load_study_snapshot_events(self.vault_root, limit=limit)
            self._send_json(
                {
                    "ok": True,
                    "record": "ημ.study-history.v1",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "path": str(_study_snapshot_log_path(self.vault_root)),
                    "count": len(events),
                    "events": events,
                }
            )
            return

        if parsed.path == "/api/witness/lineage":
            self._send_json(build_witness_lineage_payload(self.part_root))
            return

        if parsed.path == "/api/resource/heartbeat":
            heartbeat = _resource_monitor_snapshot(part_root=self.part_root)
            _INFLUENCE_TRACKER.record_resource_heartbeat(
                heartbeat,
                source="api.resource.heartbeat",
            )
            queue_snapshot = self.task_queue.snapshot(include_pending=False)
            runtime_snapshot = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )
            self._send_json(
                {
                    "ok": True,
                    "record": "ημ.resource-heartbeat.response.v1",
                    "heartbeat": heartbeat,
                    "runtime": runtime_snapshot,
                }
            )
            return

        if parsed.path == "/api/named-fields":
            self._send_json(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "mode": "gradients",
                    "named_fields": build_named_field_overlays(ENTITY_MANIFEST),
                }
            )
            return

        if parsed.path == "/api/memories":
            docs = _load_mycelial_echo_documents(limit=10)
            now_iso = datetime.now(timezone.utc).isoformat()
            memories = [
                {
                    "id": f"mem:{idx}",
                    "text": str(doc),
                    "metadata": {"timestamp": now_iso},
                }
                for idx, doc in enumerate(docs)
                if str(doc).strip()
            ]
            self._send_json({"ok": True, "memories": memories})
            return

        if parsed.path == "/api/myth":
            catalog = self._collect_catalog_fast()
            try:
                myth_summary = self.myth_tracker.snapshot(catalog)
            except Exception:
                myth_summary = {}
            self._send_json(myth_summary)
            return

        if parsed.path == "/api/world":
            catalog = self._collect_catalog_fast()
            try:
                myth_summary = self.myth_tracker.snapshot(catalog)
            except Exception:
                myth_summary = {}
            try:
                world_summary = self.life_tracker.snapshot(
                    catalog,
                    myth_summary,
                    ENTITY_MANIFEST,
                )
            except Exception:
                world_summary = {"generated_at": datetime.now(timezone.utc).isoformat()}
            self._send_json(world_summary)
            return

        if parsed.path == "/api/simulation":
            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            compact_response = _safe_bool_query(
                str(params.get("compact", ["false"])[0] or "false"),
                False,
            )

            def _send_simulation_response(
                body: bytes,
                *,
                extra_headers: dict[str, str] | None = None,
            ) -> None:
                response_body = (
                    _simulation_http_compact_response_body(body)
                    if compact_response
                    else body
                )
                self._send_bytes(
                    response_body,
                    "application/json; charset=utf-8",
                    extra_headers=extra_headers,
                )

            cache_key = ""
            try:
                if _simulation_http_is_cold_start():
                    cold_disk_body = _simulation_http_disk_cache_load(
                        self.part_root,
                        perspective=perspective,
                        max_age_seconds=max(
                            _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                            _SIMULATION_HTTP_DISK_CACHE_SECONDS,
                        ),
                    )
                    if cold_disk_body is not None:
                        _simulation_http_cache_store(
                            f"{perspective}|disk-cold-start|simulation",
                            cold_disk_body,
                        )
                        _send_simulation_response(
                            cold_disk_body,
                            extra_headers={
                                "X-Eta-Mu-Simulation-Fallback": "disk-cache-cold-start",
                            },
                        )
                        return

                catalog, queue_snapshot, _, influence_snapshot, _ = (
                    self._runtime_catalog(
                        perspective=perspective,
                        include_projection=False,
                    )
                )
                user_inputs_recent = int(
                    _safe_float(influence_snapshot.get("user_inputs_120s", 0), 0.0)
                )
                cache_key = _simulation_http_cache_key(
                    perspective=perspective,
                    catalog=catalog,
                    queue_snapshot=queue_snapshot,
                    influence_snapshot=influence_snapshot,
                )
                cached_body = _simulation_http_cached_body(
                    cache_key=cache_key,
                    perspective=perspective,
                    max_age_seconds=_SIMULATION_HTTP_CACHE_SECONDS,
                    require_exact_key=True,
                )
                if cached_body is not None:
                    _send_simulation_response(
                        cached_body,
                    )
                    return

                if user_inputs_recent <= 0:
                    disk_cached_body = _simulation_http_disk_cache_load(
                        self.part_root,
                        perspective=perspective,
                        max_age_seconds=max(
                            _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                            _SIMULATION_HTTP_DISK_CACHE_SECONDS,
                        ),
                    )
                    if disk_cached_body is not None:
                        _simulation_http_cache_store(cache_key, disk_cached_body)
                        _send_simulation_response(
                            disk_cached_body,
                            extra_headers={
                                "X-Eta-Mu-Simulation-Fallback": "disk-cache",
                            },
                        )
                        return

                backoff_remaining, backoff_error, backoff_streak = (
                    _simulation_http_failure_backoff_snapshot()
                )
                if backoff_remaining > 0.0:
                    stale_body = _simulation_http_cached_body(
                        perspective=perspective,
                        max_age_seconds=_SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                    )
                    if stale_body is None:
                        stale_body = _simulation_http_disk_cache_load(
                            self.part_root,
                            perspective=perspective,
                            max_age_seconds=max(
                                _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                                _SIMULATION_HTTP_DISK_CACHE_SECONDS,
                            ),
                        )
                        if stale_body is not None:
                            _simulation_http_cache_store(cache_key, stale_body)
                    if stale_body is not None:
                        _send_simulation_response(
                            stale_body,
                            extra_headers={
                                "X-Eta-Mu-Simulation-Fallback": "failure-backoff",
                                "X-Eta-Mu-Simulation-Error": backoff_error
                                or "build_failure",
                                "X-Eta-Mu-Simulation-Backoff-Seconds": str(
                                    max(1, int(math.ceil(backoff_remaining)))
                                ),
                                "X-Eta-Mu-Simulation-Backoff-Streak": str(
                                    max(0, int(backoff_streak))
                                ),
                            },
                        )
                        return

                lock_acquired = _SIMULATION_HTTP_BUILD_LOCK.acquire(blocking=False)
                if not lock_acquired:
                    inflight_body = _simulation_http_wait_for_exact_cache(
                        cache_key=cache_key,
                        perspective=perspective,
                        max_wait_seconds=_SIMULATION_HTTP_BUILD_WAIT_SECONDS,
                    )
                    if inflight_body is not None:
                        _send_simulation_response(
                            inflight_body,
                            extra_headers={
                                "X-Eta-Mu-Simulation-Fallback": "inflight-cache",
                            },
                        )
                        return

                    stale_body = _simulation_http_cached_body(
                        perspective=perspective,
                        max_age_seconds=_SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                    )
                    if stale_body is not None:
                        _send_simulation_response(
                            stale_body,
                            extra_headers={
                                "X-Eta-Mu-Simulation-Fallback": "stale-cache",
                                "X-Eta-Mu-Simulation-Error": "build_inflight",
                            },
                        )
                        return

                    _SIMULATION_HTTP_BUILD_LOCK.acquire()
                    lock_acquired = True

                try:
                    cached_body = _simulation_http_cached_body(
                        cache_key=cache_key,
                        perspective=perspective,
                        max_age_seconds=_SIMULATION_HTTP_CACHE_SECONDS,
                        require_exact_key=True,
                    )
                    if cached_body is not None:
                        _send_simulation_response(
                            cached_body,
                            extra_headers={
                                "X-Eta-Mu-Simulation-Fallback": "inflight-cache",
                            },
                        )
                        return

                    simulation, projection = self._runtime_simulation(
                        catalog,
                        queue_snapshot,
                        influence_snapshot,
                        perspective=perspective,
                    )
                    simulation["projection"] = projection
                    response_body = _json_compact(simulation).encode("utf-8")
                    _simulation_http_cache_store(cache_key, response_body)
                    _simulation_http_disk_cache_store(
                        self.part_root,
                        perspective=perspective,
                        body=response_body,
                    )
                    _simulation_http_failure_clear()
                    _send_simulation_response(
                        response_body,
                    )
                finally:
                    if lock_acquired:
                        _SIMULATION_HTTP_BUILD_LOCK.release()
            except Exception as exc:
                _simulation_http_failure_record(exc.__class__.__name__)
                stale_body = _simulation_http_cached_body(
                    perspective=perspective,
                    max_age_seconds=_SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                )
                if stale_body is not None:
                    _send_simulation_response(
                        stale_body,
                        extra_headers={
                            "X-Eta-Mu-Simulation-Fallback": "stale-cache",
                            "X-Eta-Mu-Simulation-Error": exc.__class__.__name__,
                        },
                    )
                    return

                disk_stale_body = _simulation_http_disk_cache_load(
                    self.part_root,
                    perspective=perspective,
                    max_age_seconds=max(
                        _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                        _SIMULATION_HTTP_DISK_CACHE_SECONDS,
                    ),
                )
                if disk_stale_body is not None:
                    _simulation_http_cache_store(cache_key, disk_stale_body)
                    _send_simulation_response(
                        disk_stale_body,
                        extra_headers={
                            "X-Eta-Mu-Simulation-Fallback": "disk-cache",
                            "X-Eta-Mu-Simulation-Error": exc.__class__.__name__,
                        },
                    )
                    return

                self._send_json(
                    {
                        "ok": False,
                        "error": "simulation_unavailable",
                        "record": "eta-mu.simulation.error.v1",
                        "perspective": perspective,
                        "detail": exc.__class__.__name__,
                    },
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            return

        if parsed.path == "/stream/mix.wav":
            catalog, _, _, _, _ = self._runtime_catalog(
                perspective=PROJECTION_DEFAULT_PERSPECTIVE,
                include_projection=False,
            )
            wav_bytes, _ = build_mix_stream(catalog, self.vault_root)
            if wav_bytes:
                self._send_bytes(wav_bytes, "audio/wav")
            else:
                self._send_bytes(
                    b"no wav sources available for mix",
                    "text/plain; charset=utf-8",
                    status=HTTPStatus.NOT_FOUND,
                )
            return

        if parsed.path.startswith("/library/"):
            member = resolve_library_member(self.path)
            lib_path = _resolve_runtime_library_path(
                self.vault_root,
                self.part_root,
                self.path,
            )
            if lib_path is None:
                self._send_json(
                    {"ok": False, "error": "library_not_found"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            if member:
                payload = _read_library_archive_member(lib_path, member)
                if payload is None:
                    self._send_json(
                        {"ok": False, "error": "library_member_not_found"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                payload_bytes, payload_type = payload
                self._send_bytes(payload_bytes, payload_type)
                return

            mime_type = (
                mimetypes.guess_type(lib_path.name)[0] or "application/octet-stream"
            )
            self._send_bytes(lib_path.read_bytes(), mime_type)
            return

        if parsed.path.startswith("/artifacts/"):
            artifact = resolve_artifact_path(self.part_root, self.path)
            if artifact is None:
                self._send_json(
                    {"ok": False, "error": "artifact_not_found"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            mime_type = (
                mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
            )
            self._send_bytes(artifact.read_bytes(), mime_type)
            return

        if parsed.path == "/":
            index_path = (self.part_root / "frontend" / "dist" / "index.html").resolve()
            if index_path.exists() and index_path.is_file():
                self._send_bytes(index_path.read_bytes(), "text/html; charset=utf-8")
                return

            world_payload = build_world_payload(self.part_root)
            catalog, _, _, _, _ = self._runtime_catalog(
                perspective=PROJECTION_DEFAULT_PERSPECTIVE,
            )
            html_doc = render_index(world_payload, catalog)
            body = html_doc.encode("utf-8") if html_doc else b""
            self._send_bytes(body, "text/html; charset=utf-8")
            return

        if parsed.path in {
            "/dashboard/docker",
            "/dashboard/docker-simulations",
            "/dashboard/bench",
            "/dashboard/profile",
        }:
            dashboard_filename = (
                "simulation_bench_dashboard.html"
                if parsed.path == "/dashboard/bench"
                else "simulation_profile.html"
                if parsed.path == "/dashboard/profile"
                else "docker_simulations_dashboard.html"
            )
            dashboard_path = (
                self.part_root / "code" / "static" / dashboard_filename
            ).resolve()
            if dashboard_path.exists() and dashboard_path.is_file():
                self._send_bytes(
                    dashboard_path.read_bytes(),
                    "text/html; charset=utf-8",
                )
            else:
                self._send_json(
                    {
                        "ok": False,
                        "error": "dashboard_not_found",
                    },
                    status=HTTPStatus.NOT_FOUND,
                )
            return

        # Optional static frontend assets when served from part64 runtime directly.
        dist_root = (self.part_root / "frontend" / "dist").resolve()
        if parsed.path != "/":
            requested = parsed.path.lstrip("/")
            candidate = (dist_root / requested).resolve()
            if (
                dist_root in candidate.parents
                and candidate.exists()
                and candidate.is_file()
            ):
                mime_type = (
                    mimetypes.guess_type(candidate.name)[0]
                    or "application/octet-stream"
                )
                self._send_bytes(candidate.read_bytes(), mime_type)
                return

        self._send_json(
            {"ok": False, "error": "not found"},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/simulation/instances/"):
            instance_id = parsed.path.split("/")[-1]
            try:
                res = subprocess.check_output(
                    [sys.executable, "scripts/sim_manager.py", "stop", instance_id],
                    cwd=self.part_root,
                    text=True,
                )
                self._send_bytes(res.encode("utf-8"), "application/json")
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)})
            return

        self._send_json(
            {"ok": False, "error": "method not allowed"},
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/api/docker/simulations/control":
            req = self._read_json_body() or {}
            target_id = str(
                req.get("id", "") or req.get("identifier", "") or ""
            ).strip()
            action = str(req.get("action", "") or "").strip().lower()
            stop_timeout_seconds = max(
                2.0,
                min(
                    45.0,
                    _safe_float(req.get("stop_timeout_seconds", 12.0), 12.0),
                ),
            )

            if not target_id:
                self._send_json(
                    {
                        "ok": False,
                        "error": "simulation_id_required",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            if action not in {"start", "stop"}:
                self._send_json(
                    {
                        "ok": False,
                        "error": "unsupported_action",
                        "allowed_actions": ["start", "stop"],
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            snapshot = collect_docker_simulation_snapshot(force_refresh=True)
            simulation = _find_docker_simulation_row(snapshot, target_id)
            if simulation is None:
                self._send_json(
                    {
                        "ok": False,
                        "error": "simulation_not_found",
                        "id": target_id,
                    },
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            control = (
                simulation.get("control", {})
                if isinstance(simulation.get("control"), dict)
                else {}
            )
            can_start_stop = bool(control.get("can_start_stop", False))
            if not can_start_stop:
                self._send_json(
                    {
                        "ok": False,
                        "error": "simulation_control_blocked",
                        "reason": str(
                            control.get("reason", "control_protected")
                            or "control_protected"
                        ),
                        "simulation": simulation,
                    },
                    status=HTTPStatus.FORBIDDEN,
                )
                return

            simulation_id = str(simulation.get("id", "") or "").strip()
            if not simulation_id:
                self._send_json(
                    {
                        "ok": False,
                        "error": "simulation_missing_id",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            before_state = str(simulation.get("state", "") or "").strip().lower()
            ok, control_error = control_simulation_container(
                simulation_id,
                action=action,
                stop_timeout_seconds=stop_timeout_seconds,
            )
            if not ok:
                self._send_json(
                    {
                        "ok": False,
                        "error": "simulation_control_failed",
                        "detail": control_error,
                        "action": action,
                        "id": simulation_id,
                        "before_state": before_state,
                    },
                    status=HTTPStatus.BAD_GATEWAY,
                )
                return

            refreshed = collect_docker_simulation_snapshot(force_refresh=True)
            refreshed_sim = _find_docker_simulation_row(refreshed, simulation_id)
            after_state = (
                str(refreshed_sim.get("state", "") or "").strip().lower()
                if isinstance(refreshed_sim, dict)
                else ""
            )
            self._send_json(
                {
                    "ok": True,
                    "action": action,
                    "id": simulation_id,
                    "before_state": before_state,
                    "after_state": after_state,
                    "detail": control_error,
                    "simulation": refreshed_sim
                    if isinstance(refreshed_sim, dict)
                    else simulation,
                }
            )
            return

        if parsed.path == "/api/simulation/instances/spawn":
            req = self._read_json_body() or {}
            preset_id = str(req.get("preset_id", "")).strip()
            # Dynamic port allocation (simple range for bench)
            # Check active ones first to avoid collisions
            port = 18900
            try:
                active_res = subprocess.check_output(
                    [sys.executable, "scripts/sim_manager.py", "list-active"],
                    cwd=self.part_root,
                    text=True,
                )
                active = json.loads(active_res)
                used_ports = {i.get("port") for i in active if i.get("port")}
                while port in used_ports and port < 18950:
                    port += 1

                res = subprocess.check_output(
                    [
                        sys.executable,
                        "scripts/sim_manager.py",
                        "spawn",
                        "--preset",
                        preset_id,
                        "--port",
                        str(port),
                    ],
                    cwd=self.part_root,
                    text=True,
                )
                self._send_bytes(res.encode("utf-8"), "application/json")
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/simulation/benchmark":
            req = self._read_json_body() or {}

            def _rewrite_benchmark_url(raw_url: str) -> str:
                url_value = str(raw_url or "").strip()
                if not url_value:
                    return ""
                parsed_url = urlparse(url_value)
                host = str(parsed_url.hostname or "").strip().lower()
                port = parsed_url.port
                if host not in {"127.0.0.1", "localhost"}:
                    return url_value
                service_map = {
                    18877: ("eta-mu-local", 8787),
                    18880: ("eta-mu-cdb", 8787),
                    18878: ("eta-mu-redis", 8787),
                    18879: ("eta-mu-uds", 8787),
                }
                service_target = service_map.get(int(port or 0))
                if service_target is None:
                    return url_value
                service_host, service_port = service_target
                scheme = str(parsed_url.scheme or "http")
                path = str(parsed_url.path or "/")
                query = f"?{parsed_url.query}" if parsed_url.query else ""
                return f"{scheme}://{service_host}:{service_port}{path}{query}"

            baseline_url = _rewrite_benchmark_url(
                str(req.get("baseline_url", "")).strip()
            )
            offload_url = _rewrite_benchmark_url(
                str(req.get("offload_url", "")).strip()
            )
            requests = int(req.get("requests", 10))
            warmup = int(req.get("warmup", 2))
            try:
                timeout_seconds = float(req.get("timeout_seconds", 60.0))
            except Exception:
                timeout_seconds = 60.0
            timeout_seconds = max(10.0, min(180.0, timeout_seconds))

            try:
                # Run bench_sim_compare.py
                cmd = [
                    sys.executable,
                    "scripts/bench_sim_compare.py",
                    "--baseline-url",
                    baseline_url,
                    "--offload-url",
                    offload_url,
                    "--requests",
                    str(requests),
                    "--warmup",
                    str(warmup),
                    "--timeout",
                    str(timeout_seconds),
                    "--retries",
                    "2",
                    "--json",
                ]
                res = subprocess.run(
                    cmd,
                    cwd=self.part_root,
                    capture_output=True,
                    text=True,
                    timeout=int(max(120.0, timeout_seconds * 3.0)),
                )
                if res.returncode == 0:
                    self._send_bytes(res.stdout.encode("utf-8"), "application/json")
                else:
                    self._send_json({"ok": False, "error": res.stderr or res.stdout})

            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)})
            return

        if parsed.path == "/api/eta-mu-ledger":
            req = self._read_json_body() or {}
            rows_input: list[str] = []
            utterances_raw = req.get("utterances")
            if isinstance(utterances_raw, list):
                rows_input = [str(item) for item in utterances_raw]
            else:
                text_raw = req.get("text")
                if isinstance(text_raw, str):
                    rows_input = text_raw.splitlines()

            rows = utterances_to_ledger_rows(rows_input)
            jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            self._send_json(
                {
                    "ok": True,
                    "rows": rows,
                    "jsonl": f"{jsonl}\n" if jsonl else "",
                }
            )
            return

        if parsed.path == "/api/eta-mu/sync":
            req = self._read_json_body() or {}
            wait = _safe_bool_query(
                str(req.get("wait", "false") or "false"), default=False
            )
            force = _safe_bool_query(
                str(req.get("force", "true") or "true"), default=True
            )

            if force:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = 0.0

            if wait:
                try:
                    snapshot = sync_eta_mu_inbox(self.vault_root)
                    with _RUNTIME_CATALOG_CACHE_LOCK:
                        _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = (
                            time.monotonic()
                        )
                        _RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = dict(snapshot)
                        _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""
                        _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = 0.0
                    self._schedule_runtime_catalog_refresh()
                    self._send_json(
                        {
                            "ok": True,
                            "record": "ημ.inbox.sync.v1",
                            "status": "completed",
                            "snapshot": snapshot,
                        }
                    )
                except Exception as exc:
                    with _RUNTIME_CATALOG_CACHE_LOCK:
                        _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = (
                            f"inbox_sync_failed:{exc.__class__.__name__}"
                        )
                    self._send_json(
                        {
                            "ok": False,
                            "record": "ημ.inbox.sync.v1",
                            "status": "failed",
                            "error": f"{exc.__class__.__name__}: {exc}",
                        },
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return

            self._schedule_runtime_catalog_refresh()
            with _RUNTIME_CATALOG_CACHE_LOCK:
                snapshot = _RUNTIME_CATALOG_CACHE.get("inbox_sync_snapshot")
                sync_error = str(_RUNTIME_CATALOG_CACHE.get("inbox_sync_error", ""))
            self._send_json(
                {
                    "ok": True,
                    "record": "ημ.inbox.sync.v1",
                    "status": "scheduled",
                    "snapshot": dict(snapshot) if isinstance(snapshot, dict) else None,
                    "error": sync_error,
                }
            )
            return

        if parsed.path == "/api/presence/accounts/upsert":
            req = self._read_json_body() or {}
            presence_id = str(req.get("presence_id", "") or "").strip()
            display_name = str(req.get("display_name", "") or "").strip()
            handle = str(req.get("handle", "") or "").strip()
            avatar = str(req.get("avatar", "") or "").strip()
            bio = str(req.get("bio", "") or "").strip()
            tags_raw = req.get("tags", [])
            tags = (
                [str(item).strip() for item in tags_raw if str(item).strip()]
                if isinstance(tags_raw, list)
                else []
            )
            result = _upsert_presence_account(
                self.vault_root,
                presence_id=presence_id,
                display_name=display_name,
                handle=handle,
                avatar=avatar,
                bio=bio,
                tags=tags,
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/image/commentary":
            req = self._read_json_body() or {}
            image_b64 = str(req.get("image_base64", "") or "").strip()
            image_ref = str(req.get("image_ref", "") or "").strip()
            mime = str(req.get("mime", "image/png") or "image/png").strip()
            presence_id = str(
                req.get("presence_id", "witness_thread") or "witness_thread"
            ).strip()
            prompt = str(req.get("prompt", "") or "").strip()
            persist = _safe_bool_query(
                str(req.get("persist", "true") or "true"),
                default=True,
            )

            if not image_b64:
                self._send_json(
                    {"ok": False, "error": "missing image_base64"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                image_bytes = base64.b64decode(image_b64, validate=False)
            except (ValueError, OSError):
                self._send_json(
                    {"ok": False, "error": "invalid image_base64 payload"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            commentary_payload = build_image_commentary(
                image_bytes=image_bytes,
                mime=mime,
                image_ref=image_ref,
                presence_id=presence_id,
                prompt=prompt,
                model=str(req.get("model", "") or "").strip() or None,
            )
            if not bool(commentary_payload.get("ok", False)):
                self._send_json(commentary_payload, status=HTTPStatus.BAD_REQUEST)
                return

            _upsert_presence_account(
                self.vault_root,
                presence_id=presence_id,
                display_name=str(req.get("display_name", "") or presence_id),
                handle=str(req.get("handle", "") or presence_id),
                avatar=str(req.get("avatar", "") or ""),
                bio=str(req.get("bio", "") or ""),
                tags=["image-commentary"],
            )

            entry_payload: dict[str, Any] | None = None
            if persist:
                created = _create_image_comment(
                    self.vault_root,
                    image_ref=image_ref
                    or str(
                        commentary_payload.get("analysis", {}).get("image_sha256", "")
                    )[:16],
                    presence_id=presence_id,
                    comment=str(commentary_payload.get("commentary", "")),
                    metadata={
                        "mime": mime,
                        "model": commentary_payload.get("model", ""),
                        "backend": commentary_payload.get("backend", ""),
                        "analysis": commentary_payload.get("analysis", {}),
                    },
                )
                if bool(created.get("ok", False)):
                    entry_payload = dict(created.get("entry", {}))

            self._send_json(
                {
                    "ok": True,
                    "record": "ημ.image-commentary.v1",
                    "presence_id": presence_id,
                    "image_ref": image_ref,
                    "commentary": commentary_payload.get("commentary", ""),
                    "model": commentary_payload.get("model", ""),
                    "backend": commentary_payload.get("backend", ""),
                    "analysis": commentary_payload.get("analysis", {}),
                    "persisted": bool(entry_payload),
                    "entry": entry_payload,
                }
            )
            return

        if parsed.path == "/api/image/comments":
            req = self._read_json_body() or {}
            image_ref = str(req.get("image_ref", "") or "").strip()
            presence_id = str(req.get("presence_id", "") or "").strip()
            comment = str(req.get("comment", "") or "").strip()
            metadata_raw = req.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            result = _create_image_comment(
                self.vault_root,
                image_ref=image_ref,
                presence_id=presence_id,
                comment=comment,
                metadata=metadata,
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/embeddings/db/upsert":
            req = self._read_json_body() or {}
            entry_id = str(req.get("id", "") or "").strip()
            text = str(req.get("text", "") or "")
            metadata_raw = req.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            model = _effective_request_embed_model(req.get("model"))

            embedding = _normalize_embedding_vector(req.get("embedding"))
            if embedding is None and text.strip():
                embedding = _normalize_embedding_vector(
                    _ollama_embed(text, model=model)
                )

            if embedding is None:
                self._send_json(
                    {
                        "ok": False,
                        "error": "missing or invalid embedding; provide embedding or text with reachable embeddings backend",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            result = _embedding_db_upsert(
                self.vault_root,
                entry_id=entry_id,
                text=text,
                embedding=embedding,
                metadata=metadata,
                model=model,
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/embeddings/provider/options":
            req = self._read_json_body() or {}
            result = _apply_embedding_provider_options(req)
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/embeddings/db/query":
            req = self._read_json_body() or {}
            query_text = str(req.get("query", req.get("text", "")) or "").strip()
            model = _effective_request_embed_model(req.get("model"))
            query_embedding = _normalize_embedding_vector(req.get("embedding"))
            if query_embedding is None and query_text:
                query_embedding = _normalize_embedding_vector(
                    _ollama_embed(query_text, model=model)
                )

            if query_embedding is None:
                self._send_json(
                    {
                        "ok": False,
                        "error": "missing or invalid query embedding; provide embedding or query text with reachable embeddings backend",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            top_k = max(
                1,
                min(100, int(_safe_float(str(req.get("top_k", 5) or 5), 5.0))),
            )
            min_score = _safe_float(str(req.get("min_score", -1.0) or -1.0), -1.0)
            include_vectors = _safe_bool_query(
                str(req.get("include_vectors", "false") or "false"),
                default=False,
            )

            result = _embedding_db_query(
                self.vault_root,
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                include_vectors=include_vectors,
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/embeddings/db/delete":
            req = self._read_json_body() or {}
            result = _embedding_db_delete(
                self.vault_root,
                entry_id=str(req.get("id", "") or "").strip(),
            )
            status = (
                HTTPStatus.OK if bool(result.get("ok", False)) else HTTPStatus.NOT_FOUND
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/presence/user/input":
            req_raw = self._read_json_body()
            req = req_raw if isinstance(req_raw, dict) else {}
            default_meta = (
                req.get("meta") if isinstance(req.get("meta"), dict) else None
            )

            events_raw = req.get("events") if isinstance(req, dict) else None
            candidate_rows: list[dict[str, Any]] = []
            if isinstance(events_raw, list):
                candidate_rows = [row for row in events_raw if isinstance(row, dict)]
            elif isinstance(req_raw, list):
                candidate_rows = [row for row in req_raw if isinstance(row, dict)]
            elif isinstance(req, dict):
                candidate_rows = [req]

            if not candidate_rows:
                self._send_json(
                    {
                        "ok": False,
                        "error": "invalid_user_input_payload",
                        "record": "eta-mu.user-input.error.v1",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            rows_for_ingest = candidate_rows[-USER_PRESENCE_MAX_EVENTS:]
            batch_now_unix = time.time()
            batch_now_mono = time.monotonic()
            processed_events: list[dict[str, Any]] = []
            tracker_events: list[dict[str, Any]] = []

            for index, row in enumerate(rows_for_ingest):
                kind = (
                    str(row.get("kind", row.get("type", "input")) or "input")
                    .strip()
                    .lower()
                    or "input"
                )
                target = (
                    str(row.get("target", "simulation") or "simulation").strip()
                    or "simulation"
                )
                x_raw = row.get("x_ratio", row.get("x"))
                y_raw = row.get("y_ratio", row.get("y"))
                has_pointer = x_raw is not None and y_raw is not None
                x_ratio = max(0.0, min(1.0, _safe_float(x_raw, 0.5)))
                y_ratio = max(0.0, min(1.0, _safe_float(y_raw, 0.5)))
                embed_daimoi = bool(
                    row.get(
                        "embed_daimoi",
                        row.get(
                            "embedDaimoi",
                            kind
                            in {
                                "hover",
                                "click",
                                "keydown",
                                "key",
                                "input",
                                "focus",
                            },
                        ),
                    )
                )
                message = str(row.get("message", "") or "").strip()
                if not message:
                    if kind == "hover":
                        message = f"mouse hover over {target}"
                    elif kind == "mouse_move":
                        message = "mouse move in simulation"
                    elif kind in {"click", "tap"}:
                        message = f"click {target}"
                    elif kind in {"keydown", "key"}:
                        message = f"keyboard input on {target}"
                    else:
                        message = f"{kind} on {target}"

                event_unix = batch_now_unix + (index * 0.0001)
                event_mono = batch_now_mono + (index * 0.0001)
                event_iso = datetime.fromtimestamp(event_unix, timezone.utc).isoformat()
                event_id = hashlib.sha1(
                    (
                        f"{event_iso}|{kind}|{target}|{message}|"
                        f"{x_ratio:.5f}|{y_ratio:.5f}|{int(embed_daimoi)}|{index}"
                    ).encode("utf-8")
                ).hexdigest()[:14]
                event_row: dict[str, Any] = {
                    "id": f"user-input:{event_id}",
                    "record": "ημ.user-input.v1",
                    "presence_id": USER_PRESENCE_ID,
                    "kind": kind,
                    "target": target[:240],
                    "message": message[:320],
                    "embed_daimoi": bool(embed_daimoi),
                    "ts": event_iso,
                    "ts_monotonic": round(event_mono, 6),
                }
                if has_pointer:
                    event_row["x_ratio"] = round(x_ratio, 6)
                    event_row["y_ratio"] = round(y_ratio, 6)

                meta = row.get("meta") if isinstance(row.get("meta"), dict) else None
                if not isinstance(meta, dict):
                    meta = default_meta if isinstance(default_meta, dict) else None

                processed_events.append(event_row)
                tracker_events.append(
                    {
                        "kind": kind,
                        "target": target,
                        "message": message,
                        "has_pointer": has_pointer,
                        "x_ratio": x_ratio,
                        "y_ratio": y_ratio,
                        "embed_daimoi": embed_daimoi,
                        "meta": meta,
                        "event_unix": event_unix,
                        "event_mono": event_mono,
                    }
                )

            event_count = 0
            anchor_target = {"x": 0.5, "y": 0.72}
            with _USER_PRESENCE_INPUT_LOCK:
                cache = _USER_PRESENCE_INPUT_CACHE
                rows = cache.get("events", [])
                if not isinstance(rows, list):
                    rows = []

                seq = int(cache.get("seq", 0))
                for event_row, tracker_row in zip(processed_events, tracker_events):
                    has_pointer = bool(tracker_row.get("has_pointer", False))
                    if has_pointer:
                        cache["target_x"] = _safe_float(tracker_row.get("x_ratio"), 0.5)
                        cache["target_y"] = _safe_float(tracker_row.get("y_ratio"), 0.5)
                        cache["last_pointer_unix"] = _safe_float(
                            tracker_row.get("event_unix"), batch_now_unix
                        )
                        cache["last_pointer_monotonic"] = _safe_float(
                            tracker_row.get("event_mono"), batch_now_mono
                        )

                    cache["last_input_unix"] = _safe_float(
                        tracker_row.get("event_unix"), batch_now_unix
                    )
                    cache["last_input_monotonic"] = _safe_float(
                        tracker_row.get("event_mono"), batch_now_mono
                    )
                    cache["latest_message"] = str(event_row.get("message", "") or "")[
                        :320
                    ]
                    cache["latest_target"] = str(event_row.get("target", "") or "")[
                        :240
                    ]
                    seq += 1
                    rows.append(event_row)

                if len(rows) > USER_PRESENCE_MAX_EVENTS:
                    rows = rows[-USER_PRESENCE_MAX_EVENTS:]
                cache["events"] = rows
                cache["seq"] = seq
                event_count = len(rows)
                anchor_target = {
                    "x": max(0.0, min(1.0, _safe_float(cache.get("target_x"), 0.5))),
                    "y": max(0.0, min(1.0, _safe_float(cache.get("target_y"), 0.72))),
                }

            for tracker_row in tracker_events:
                kind = str(tracker_row.get("kind", "input") or "input")
                target = str(tracker_row.get("target", "simulation") or "simulation")
                message = str(tracker_row.get("message", "") or "")
                has_pointer = bool(tracker_row.get("has_pointer", False))
                _INFLUENCE_TRACKER.record_user_input(
                    kind=kind,
                    target=target,
                    message=message,
                    x_ratio=(
                        _safe_float(tracker_row.get("x_ratio"), 0.5)
                        if has_pointer
                        else None
                    ),
                    y_ratio=(
                        _safe_float(tracker_row.get("y_ratio"), 0.5)
                        if has_pointer
                        else None
                    ),
                    embed_daimoi=bool(tracker_row.get("embed_daimoi", False)),
                    meta=(
                        tracker_row.get("meta")
                        if isinstance(tracker_row.get("meta"), dict)
                        else None
                    ),
                )

                if kind != "mouse_move":
                    _INFLUENCE_TRACKER.record_witness(
                        event_type=f"user_{kind}",
                        target=target,
                    )

            response: dict[str, Any] = {
                "ok": True,
                "presence_id": USER_PRESENCE_ID,
                "events": processed_events,
                "processed": len(processed_events),
                "event_count": event_count,
                "anchor_target": anchor_target,
            }
            if processed_events:
                response["event"] = processed_events[-1]
            self._send_json(
                response,
                status=HTTPStatus.OK,
            )
            return

        if parsed.path == "/api/input-stream":
            req = self._read_json_body() or {}
            stream_type = str(req.get("type", "unknown") or "unknown")
            data = req.get("data", {})

            if isinstance(data, dict) and stream_type in {
                "runtime_log",
                "log",
                "stderr",
                "stdout",
            }:
                _INFLUENCE_TRACKER.record_runtime_log(
                    level=str(data.get("level", "info") or "info"),
                    message=str(data.get("message", "") or ""),
                    source=str(data.get("source", "stream") or "stream"),
                )

            if isinstance(data, dict) and stream_type in {
                "resource_heartbeat",
                "heartbeat",
                "health",
            }:
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    data,
                    source="input-stream",
                )

            input_str = (
                f"{stream_type}: {json.dumps(data, ensure_ascii=False, default=str)}"
            )
            embedding = _normalize_embedding_vector(_ollama_embed(input_str))
            force_vector = _project_vector(embedding)

            collection = _get_chroma_collection()
            if collection and embedding:
                try:
                    memory_id = f"mem_{int(time.time() * 1000)}"
                    collection.add(
                        ids=[memory_id],
                        embeddings=[embedding],
                        metadatas=[
                            {
                                "type": stream_type,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        ],
                        documents=[input_str],
                    )
                except Exception:
                    pass

            queue_snapshot = self.task_queue.snapshot(include_pending=False)
            influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )

            council_catalog: dict[str, Any] = {}
            if stream_type in {"file_changed", "file_added", "file_removed"}:
                try:
                    council_catalog = self._collect_catalog_fast()
                except Exception:
                    council_catalog = {}

            council_result = self.council_chamber.consider_event(
                event_type=stream_type,
                data=data if isinstance(data, dict) else {"value": data},
                catalog=council_catalog,
                influence_snapshot=influence_snapshot,
            )

            self._send_json(
                {
                    "ok": True,
                    "force": force_vector,
                    "embedding_dim": len(embedding) if embedding else 0,
                    "resource": influence_snapshot.get("resource_heartbeat", {}),
                    "council": council_result,
                }
            )
            return

        if parsed.path == "/api/upload":
            content_type = str(self.headers.get("Content-Type", "") or "")
            file_name = "upload.mp3"
            mime = "audio/mpeg"
            language = "ja"
            file_bytes = b""

            if content_type.lower().startswith("multipart/form-data"):
                form_data = (
                    _parse_multipart_form(self._read_raw_body(), content_type) or {}
                )
                file_field = form_data.get("file")
                if not isinstance(file_field, dict):
                    self._send_json(
                        {"ok": False, "error": "missing multipart file field 'file'"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                file_name = str(
                    file_field.get("filename", "upload.mp3") or "upload.mp3"
                )
                mime = str(
                    file_field.get("content_type")
                    or form_data.get("mime")
                    or mimetypes.guess_type(file_name)[0]
                    or "audio/mpeg"
                )
                language = str(form_data.get("language", "ja") or "ja")
                raw_value = file_field.get("value", b"")
                if isinstance(raw_value, (bytes, bytearray)):
                    file_bytes = bytes(raw_value)
            else:
                req = self._read_json_body() or {}
                file_name = str(req.get("name", "upload.mp3") or "upload.mp3")
                mime = str(req.get("mime", "audio/mpeg") or "audio/mpeg")
                language = str(req.get("language", "ja") or "ja")
                file_b64 = str(req.get("base64", "") or "").strip()
                if file_b64:
                    try:
                        file_bytes = base64.b64decode(file_b64, validate=False)
                    except (ValueError, OSError):
                        file_bytes = b""

            if not file_bytes:
                self._send_json(
                    {"ok": False, "error": "missing or invalid audio payload"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                safe_name = _normalize_audio_upload_name(file_name, mime)
                save_path = self.part_root / "artifacts" / "audio" / safe_name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(file_bytes)

                result = transcribe_audio_bytes(
                    file_bytes,
                    mime=mime,
                    language=language,
                )
                transcribed_text = str(result.get("text", "") or "").strip()

                collection = _get_chroma_collection()
                if collection and transcribed_text:
                    memory_text = (
                        f"The Weaver learned a new frequency: {transcribed_text}"
                    )
                    payload: dict[str, Any] = {
                        "ids": [f"learn_{int(time.time() * 1000)}"],
                        "metadatas": [
                            {
                                "type": "learned_echo",
                                "source": safe_name,
                                "mime": mime,
                                "language": language,
                                "engine": result.get("engine", "none"),
                            }
                        ],
                        "documents": [memory_text],
                    }
                    embed = _normalize_embedding_vector(_ollama_embed(memory_text))
                    if embed:
                        payload["embeddings"] = [embed]
                    try:
                        collection.add(**payload)
                    except Exception:
                        pass

                self._send_json(
                    {
                        "ok": True,
                        "status": "learned" if transcribed_text else "stored",
                        "engine": result.get("engine", "none"),
                        "text": transcribed_text,
                        "transcription_error": result.get("error"),
                        "url": f"/artifacts/audio/{safe_name}",
                    }
                )
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/handoff":
            handoff_report: list[str] = []
            collection = _get_chroma_collection()
            if collection:
                try:
                    results = collection.get(limit=50)
                    docs = (
                        results.get("documents", [])
                        if isinstance(results, dict)
                        else []
                    )
                    handoff_report.append("# MISSION HANDOFF / 引き継ぎ")
                    handoff_report.append(
                        f"Generated at: {datetime.now(timezone.utc).isoformat()}"
                    )
                    handoff_report.append("\n## RECENT MEMORY ECHOES")
                    for item in list(docs)[-10:]:
                        handoff_report.append(f"- {str(item)}")
                except Exception:
                    pass

            try:
                constraints_path = self.part_root / "world_state" / "constraints.md"
                handoff_report.append("\n## ACTIVE CONSTRAINTS")
                handoff_report.append(constraints_path.read_text("utf-8"))
            except Exception:
                pass

            self._send_bytes(
                "\n".join(handoff_report).encode("utf-8"),
                "text/markdown; charset=utf-8",
            )
            return

        if parsed.path == "/api/fork-tax/pay":
            req = self._read_json_body() or {}
            amount_raw = req.get("amount", 1.0)
            try:
                amount = float(amount_raw)
            except (TypeError, ValueError):
                self._send_json(
                    {"ok": False, "error": "amount must be numeric"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            source = str(req.get("source", "command-center") or "command-center")
            target = str(req.get("target", "fork_tax_canticle") or "fork_tax_canticle")
            payment = _INFLUENCE_TRACKER.pay_fork_tax(
                amount=amount,
                source=source,
                target=target,
            )

            timestamp = datetime.now(timezone.utc).isoformat()
            entry = {
                "timestamp": timestamp,
                "event": "fork_tax_payment",
                "target": target,
                "source": source,
                "amount": payment["applied"],
                "witness_id": hashlib.sha1(
                    str(time.time_ns()).encode("utf-8")
                ).hexdigest()[:8],
            }
            ledger_path = (
                self.part_root / "world_state" / "decision_ledger.jsonl"
            ).resolve()
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

            queue_snapshot = self.task_queue.snapshot(include_pending=False)
            runtime_snapshot = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )
            self._send_json(
                {
                    "ok": True,
                    "status": "recorded",
                    "payment": payment,
                    "runtime": runtime_snapshot,
                }
            )
            return

        if parsed.path == "/api/muse/create":
            req = self._read_json_body() or {}
            manager = self._muse_manager()
            payload = manager.create_muse(
                muse_id=str(req.get("muse_id", "") or "").strip(),
                label=str(req.get("label", "") or "").strip(),
                anchor=req.get("anchor")
                if isinstance(req.get("anchor"), dict)
                else None,
                user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
            )
            status = (
                HTTPStatus.OK
                if bool(payload.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(payload, status=status)
            return

        if parsed.path == "/api/muse/pause":
            req = self._read_json_body() or {}
            manager = self._muse_manager()
            muse_id = str(req.get("muse_id", "") or "").strip()
            paused = _safe_bool_query(
                str(req.get("paused", "true") or "true"), default=True
            )
            payload = manager.set_pause(
                muse_id,
                paused=paused,
                reason=str(req.get("reason", "") or "").strip(),
                user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
            )
            if not bool(payload.get("ok", False)):
                error = str(payload.get("error", ""))
                status = (
                    HTTPStatus.NOT_FOUND
                    if error == "muse_not_found"
                    else HTTPStatus.BAD_REQUEST
                )
                self._send_json(payload, status=status)
                return
            self._send_json(payload)
            return

        if parsed.path == "/api/muse/pin":
            req = self._read_json_body() or {}
            manager = self._muse_manager()
            payload = manager.pin_node(
                str(req.get("muse_id", "") or "").strip(),
                node_id=str(req.get("node_id", "") or "").strip(),
                user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
                reason=str(req.get("reason", "") or "").strip(),
            )
            if not bool(payload.get("ok", False)):
                error = str(payload.get("error", ""))
                status = (
                    HTTPStatus.NOT_FOUND
                    if error == "muse_not_found"
                    else HTTPStatus.BAD_REQUEST
                )
                self._send_json(payload, status=status)
                return
            self._send_json(payload)
            return

        if parsed.path == "/api/muse/unpin":
            req = self._read_json_body() or {}
            manager = self._muse_manager()
            payload = manager.unpin_node(
                str(req.get("muse_id", "") or "").strip(),
                node_id=str(req.get("node_id", "") or "").strip(),
                user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
            )
            if not bool(payload.get("ok", False)):
                error = str(payload.get("error", ""))
                status = (
                    HTTPStatus.NOT_FOUND
                    if error == "muse_not_found"
                    else HTTPStatus.BAD_REQUEST
                )
                self._send_json(payload, status=status)
                return
            self._send_json(payload)
            return

        if parsed.path == "/api/muse/bind-nexus":
            req = self._read_json_body() or {}
            manager = self._muse_manager()
            payload = manager.bind_nexus(
                str(req.get("muse_id", "") or "").strip(),
                nexus_id=str(req.get("nexus_id", "") or "").strip(),
                reason=str(req.get("reason", "") or "").strip(),
                user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
            )
            if not bool(payload.get("ok", False)):
                error = str(payload.get("error", ""))
                status = (
                    HTTPStatus.NOT_FOUND
                    if error == "muse_not_found"
                    else HTTPStatus.BAD_REQUEST
                )
                self._send_json(payload, status=status)
                return
            self._send_json(payload)
            return

        if parsed.path == "/api/muse/sync-pins":
            req = self._read_json_body() or {}
            manager = self._muse_manager()
            pinned_node_ids = req.get("pinned_node_ids", [])
            payload = manager.sync_workspace_pins(
                str(req.get("muse_id", "") or "").strip(),
                pinned_node_ids=pinned_node_ids
                if isinstance(pinned_node_ids, list)
                else [],
                reason=str(req.get("reason", "") or "").strip(),
                user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
            )
            if not bool(payload.get("ok", False)):
                error = str(payload.get("error", ""))
                status = (
                    HTTPStatus.NOT_FOUND
                    if error == "muse_not_found"
                    else HTTPStatus.BAD_REQUEST
                )
                self._send_json(payload, status=status)
                return
            self._send_json(payload)
            return

        if parsed.path == "/api/muse/message":
            req = self._read_json_body() or {}
            manager = self._muse_manager()
            catalog = self._runtime_catalog_base()
            graph_revision = str(
                req.get("graph_revision", catalog.get("generated_at", ""))
                or catalog.get("generated_at", "")
            ).strip()
            idempotency_key = str(
                req.get("idempotency_key", self.headers.get("Idempotency-Key", ""))
                or ""
            ).strip()
            surrounding_nodes = req.get("surrounding_nodes", [])
            payload = manager.send_message(
                muse_id=str(req.get("muse_id", "") or "").strip(),
                text=str(req.get("text", "") or "").strip(),
                mode=str(req.get("mode", "stochastic") or "stochastic").strip(),
                token_budget=max(
                    320,
                    min(
                        8192,
                        int(
                            _safe_float(
                                str(req.get("token_budget", 2048) or 2048), 2048.0
                            )
                        ),
                    ),
                ),
                idempotency_key=idempotency_key,
                graph_revision=graph_revision,
                surrounding_nodes=surrounding_nodes
                if isinstance(surrounding_nodes, list)
                else [],
                tool_callback=self._muse_tool_callback,
                reply_builder=self._muse_reply_builder,
                seed=str(req.get("seed", "") or "").strip(),
            )
            if not bool(payload.get("ok", False)):
                status_code = int(payload.get("status_code", HTTPStatus.BAD_REQUEST))
                self._send_json(payload, status=status_code)
                return
            self._send_json(payload)
            return

        if parsed.path == "/api/chat":
            req = self._read_json_body() or {}
            messages_raw = req.get("messages", [])
            mode = str(req.get("mode", "ollama") or "ollama")
            multi_entity = bool(req.get("multi_entity", False))
            raw_presence_ids = req.get("presence_ids", [])

            presence_ids: list[str] = []
            if isinstance(raw_presence_ids, list):
                for item in raw_presence_ids:
                    value = str(item).strip()
                    if value:
                        presence_ids.append(value)

            if isinstance(messages_raw, list):
                messages = [item for item in messages_raw if isinstance(item, dict)]
            else:
                messages = []

            context = build_world_payload(self.part_root)
            resource_heartbeat = _resource_monitor_snapshot(part_root=self.part_root)
            _INFLUENCE_TRACKER.record_resource_heartbeat(
                resource_heartbeat,
                source="api.chat",
            )
            context["resource_heartbeat"] = resource_heartbeat
            self._send_json(
                build_chat_reply(
                    messages,
                    mode=mode,
                    context=context,
                    multi_entity=multi_entity,
                    presence_ids=presence_ids,
                )
            )
            return

        if parsed.path == "/api/world/interact":
            req = self._read_json_body() or {}
            person_id = str(req.get("person_id", "") or "").strip()
            action = str(req.get("action", "speak") or "speak").strip() or "speak"

            catalog = self._collect_catalog_fast()
            try:
                myth_summary = self.myth_tracker.snapshot(catalog)
            except Exception:
                myth_summary = {}
            try:
                world_summary = self.life_tracker.snapshot(
                    catalog,
                    myth_summary,
                    ENTITY_MANIFEST,
                )
            except Exception:
                world_summary = {}

            try:
                result = self.life_interaction_builder(world_summary, person_id, action)
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": f"interaction_runtime_error:{exc.__class__.__name__}",
                    "line_en": "Interaction failed.",
                    "line_ja": "対話に失敗しました。",
                }
            if not isinstance(result, dict):
                result = {
                    "ok": False,
                    "error": "invalid_interaction_payload",
                    "line_en": "Interaction failed.",
                    "line_ja": "対話に失敗しました。",
                }

            if bool(result.get("ok", False)):
                timestamp = datetime.now(timezone.utc).isoformat()
                entry = {
                    "timestamp": timestamp,
                    "event": "world_interact",
                    "target": str((result.get("presence") or {}).get("id", "unknown")),
                    "person_id": person_id,
                    "action": action,
                    "witness_id": hashlib.sha1(
                        str(time.time_ns()).encode("utf-8")
                    ).hexdigest()[:8],
                }
                ledger_path = (
                    self.part_root / "world_state" / "decision_ledger.jsonl"
                ).resolve()
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with ledger_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                _INFLUENCE_TRACKER.record_witness(
                    event_type="world_interact",
                    target=str(entry.get("target", "unknown")),
                )

            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/presence/say":
            req = self._read_json_body() or {}
            text = str(req.get("text", "") or "").strip()
            presence_id = str(req.get("presence_id", "") or "").strip()
            catalog, queue_snapshot, _, _, _ = self._runtime_catalog(
                perspective=PROJECTION_DEFAULT_PERSPECTIVE,
                include_projection=False,
            )
            catalog["presence_runtime"] = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )
            self._send_json(build_presence_say_payload(catalog, text, presence_id))
            return

        if parsed.path == "/api/drift/scan":
            self._send_json(build_drift_scan_payload(self.part_root, self.vault_root))
            return

        if parsed.path == "/api/push-truth/dry-run":
            self._send_json(
                build_push_truth_dry_run_payload(self.part_root, self.vault_root)
            )
            return

        if parsed.path == "/api/pi/archive/portable":
            req = self._read_json_body() or {}
            archive_raw = req.get("archive")
            archive = archive_raw if isinstance(archive_raw, dict) else {}
            payload = validate_pi_archive_portable(archive)
            status = (
                HTTPStatus.OK
                if bool(payload.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(payload, status=status)
            return

        if parsed.path == "/api/study/export":
            req = self._read_json_body() or {}
            label = str(req.get("label", "") or "").strip()
            owner = str(req.get("owner", "Err") or "Err").strip() or "Err"
            include_truth_state = bool(req.get("include_truth", False))
            refs_raw = req.get("refs", [])
            refs = (
                [str(item).strip() for item in refs_raw if str(item).strip()]
                if isinstance(refs_raw, list)
                else []
            )

            queue_snapshot = self.task_queue.snapshot(include_pending=True)
            council_snapshot = self.council_chamber.snapshot(
                include_decisions=True,
                limit=128,
            )
            drift_payload = build_drift_scan_payload(self.part_root, self.vault_root)
            resource_snapshot = _resource_monitor_snapshot(part_root=self.part_root)
            _INFLUENCE_TRACKER.record_resource_heartbeat(
                resource_snapshot,
                source="api.study.export",
            )

            truth_gate_blocked: bool | None = None
            if include_truth_state:
                try:
                    truth_state = self._collect_catalog_fast().get("truth_state", {})
                    gate = (
                        truth_state.get("gate", {})
                        if isinstance(truth_state, dict)
                        else {}
                    )
                    if isinstance(gate, dict):
                        truth_gate_blocked = bool(gate.get("blocked", False))
                except Exception:
                    truth_gate_blocked = None

            self._send_json(
                export_study_snapshot(
                    self.part_root,
                    self.vault_root,
                    queue_snapshot=queue_snapshot,
                    council_snapshot=council_snapshot,
                    drift_payload=drift_payload,
                    truth_gate_blocked=truth_gate_blocked,
                    resource_snapshot=resource_snapshot,
                    label=label,
                    owner=owner,
                    refs=refs,
                    host=self.host_label,
                    manifest="manifest.lith",
                )
            )
            return

        if parsed.path == "/api/transcribe":
            req = self._read_json_body() or {}
            audio_b64 = str(req.get("audio_base64", "") or "").strip()
            mime = str(req.get("mime", "audio/webm") or "audio/webm")
            language = str(req.get("language", "ja") or "ja")
            if not audio_b64:
                self._send_json(
                    {
                        "ok": False,
                        "engine": "none",
                        "text": "",
                        "error": "missing audio_base64",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                audio_bytes = base64.b64decode(
                    audio_b64.encode("utf-8"), validate=False
                )
            except (ValueError, OSError):
                self._send_json(
                    {
                        "ok": False,
                        "engine": "none",
                        "text": "",
                        "error": "invalid base64 audio",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            self._send_json(
                transcribe_audio_bytes(audio_bytes, mime=mime, language=language)
            )
            return

        if parsed.path == "/api/ux/critique":
            req = self._read_json_body() or {}
            projection_raw = req.get("projection")
            projection: dict[str, Any]
            if isinstance(projection_raw, dict):
                projection = dict(projection_raw)
            else:
                projection = {}
            try:
                try:
                    from ux_critic import critique_ux
                except ImportError:
                    from code.ux_critic import critique_ux

                critique = critique_ux(projection)
                self._send_json({"ok": True, "critique": critique})
            except ImportError as exc:
                self._send_json(
                    {"ok": False, "error": f"ux_critic module missing: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/witness":
            req = self._read_json_body() or {}
            event_type = str(req.get("type", "touch") or "touch")
            target = str(req.get("target", "unknown") or "unknown")

            timestamp = datetime.now(timezone.utc).isoformat()
            entry = {
                "timestamp": timestamp,
                "event": event_type,
                "target": target,
                "witness_id": hashlib.sha1(
                    str(time.time_ns()).encode("utf-8")
                ).hexdigest()[:8],
            }
            ledger_path = (
                self.part_root / "world_state" / "decision_ledger.jsonl"
            ).resolve()
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _INFLUENCE_TRACKER.record_witness(event_type=event_type, target=target)
            self._send_json(
                {
                    "ok": True,
                    "status": "recorded",
                    "collapse_id": entry["witness_id"],
                }
            )
            return

        if parsed.path == "/api/meta/notes":
            req = self._read_json_body() or {}
            result = create_meta_note(
                self.vault_root,
                text=str(req.get("text", "") or ""),
                owner=str(req.get("owner", "Err") or "Err"),
                title=str(req.get("title", "") or ""),
                tags=req.get("tags", []),
                targets=req.get("targets", []),
                severity=str(req.get("severity", "info") or "info"),
                category=str(req.get("category", "observation") or "observation"),
                context=req.get("context", {}),
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/meta/runs":
            req = self._read_json_body() or {}
            result = create_meta_run(
                self.vault_root,
                run_type=str(req.get("run_type", "") or ""),
                title=str(req.get("title", "") or ""),
                owner=str(req.get("owner", "Err") or "Err"),
                status=str(req.get("status", "planned") or "planned"),
                objective=str(req.get("objective", "") or ""),
                model_ref=str(req.get("model_ref", "") or ""),
                dataset_ref=str(req.get("dataset_ref", "") or ""),
                notes=str(req.get("notes", "") or ""),
                tags=req.get("tags", []),
                targets=req.get("targets", []),
                metrics=req.get("metrics", {}),
                links=req.get("links", []),
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/meta/objective/enqueue":
            req = self._read_json_body() or {}
            objective_text = str(req.get("objective", "") or "").strip()
            if not objective_text:
                self._send_json(
                    {
                        "ok": False,
                        "error": "missing_objective",
                        "required": ["objective"],
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            objective_type = str(
                req.get("objective_type", "evaluation") or "evaluation"
            )
            objective_type = objective_type.strip().lower() or "evaluation"
            target = str(req.get("target", "") or "").strip()
            priority = str(req.get("priority", "medium") or "medium").strip().lower()
            owner = str(req.get("owner", "Err") or "Err").strip() or "Err"
            refs_raw = req.get("refs", [])
            refs = (
                [str(item).strip() for item in refs_raw if str(item).strip()]
                if isinstance(refs_raw, list)
                else []
            )
            metadata_raw = req.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

            payload = {
                "objective": objective_text,
                "objective_type": objective_type,
                "target": target,
                "priority": priority,
                "metadata": metadata,
                "created_via": "api/meta/objective/enqueue",
            }
            dedupe_key = str(req.get("dedupe_key", "") or "").strip()
            if not dedupe_key:
                dedupe_key = (
                    f"meta-objective:{objective_type}:{target}:{objective_text}"
                )

            self._send_json(
                self.task_queue.enqueue(
                    kind="meta-objective",
                    payload=payload,
                    dedupe_key=dedupe_key,
                    owner=owner,
                    dod="meta objective queued with persisted log and receipt",
                    refs=[
                        "api:meta/objective/enqueue",
                        "dashboard:meta-operations",
                        *refs,
                    ],
                )
            )
            return

        if parsed.path == "/api/task/enqueue":
            req = self._read_json_body() or {}
            kind = str(req.get("kind", "runtime-task") or "runtime-task").strip()
            payload_raw = req.get("payload", {})
            payload = (
                payload_raw if isinstance(payload_raw, dict) else {"value": payload_raw}
            )
            dedupe_key = str(req.get("dedupe_key", "") or "").strip()
            owner = str(req.get("owner", "Err") or "Err").strip()
            refs_raw = req.get("refs", [])
            refs = (
                [str(item).strip() for item in refs_raw if str(item).strip()]
                if isinstance(refs_raw, list)
                else []
            )
            self._send_json(
                self.task_queue.enqueue(
                    kind=kind,
                    payload=payload,
                    dedupe_key=dedupe_key,
                    owner=owner,
                    refs=refs,
                )
            )
            return

        if parsed.path == "/api/task/dequeue":
            req = self._read_json_body() or {}
            owner = str(req.get("owner", "Err") or "Err").strip()
            refs_raw = req.get("refs", [])
            refs = (
                [str(item).strip() for item in refs_raw if str(item).strip()]
                if isinstance(refs_raw, list)
                else []
            )
            result = self.task_queue.dequeue(owner=owner, refs=refs)
            status = (
                HTTPStatus.OK if bool(result.get("ok", False)) else HTTPStatus.CONFLICT
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/council/vote":
            req = self._read_json_body() or {}
            decision_id = str(req.get("decision_id", "") or "").strip()
            member_id = str(req.get("member_id", "") or "").strip()
            vote_value = str(req.get("vote", "") or "").strip().lower()
            reason = str(req.get("reason", "") or "").strip()
            actor = str(req.get("actor", "Err") or "Err").strip() or "Err"
            if (
                not decision_id
                or not member_id
                or vote_value not in {"yes", "no", "abstain"}
            ):
                self._send_json(
                    {
                        "ok": False,
                        "error": "invalid_request",
                        "required": ["decision_id", "member_id", "vote"],
                        "allowed_votes": ["yes", "no", "abstain"],
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            result = self.council_chamber.vote(
                decision_id=decision_id,
                member_id=member_id,
                vote=vote_value,
                reason=reason,
                actor=actor,
            )
            status = HTTPStatus.OK
            if not bool(result.get("ok", False)):
                error = str(result.get("error", "")).strip()
                if error == "decision_not_found":
                    status = HTTPStatus.NOT_FOUND
                elif error in {"member_not_in_council", "invalid_vote"}:
                    status = HTTPStatus.BAD_REQUEST
                else:
                    status = HTTPStatus.CONFLICT
            self._send_json(result, status=status)
            return

        self._send_json(
            {"ok": False, "error": "not found"},
            status=HTTPStatus.NOT_FOUND,
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print(f"[world-web] {self.address_string()} - {format % args}")


def make_handler(
    part_root: Path,
    vault_root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
):
    receipts_path = _ensure_receipts_log_path(vault_root, part_root)
    queue_log_path = (
        vault_root / ".opencode" / "runtime" / "task_queue.v1.jsonl"
    ).resolve()
    council_log_path = (vault_root / COUNCIL_DECISION_LOG_REL).resolve()

    task_queue = TaskQueue(
        queue_log_path,
        receipts_path,
        owner="Err",
        host=f"{host}:{port}",
    )
    council_chamber = CouncilChamber(
        council_log_path,
        receipts_path,
        owner="Err",
        host=f"{host}:{port}",
        part_root=part_root,
        vault_root=vault_root,
    )

    myth_tracker_class = _load_myth_tracker_class()
    life_tracker_class = _load_life_tracker_class()
    life_interaction_builder = _load_life_interaction_builder()

    try:
        myth_tracker = myth_tracker_class()
    except Exception:

        class _NullMythTracker:
            def snapshot(self, _catalog: dict[str, Any]) -> dict[str, Any]:
                return {}

        myth_tracker = _NullMythTracker()

    try:
        life_tracker = life_tracker_class()
    except Exception:

        class _NullLifeTracker:
            def snapshot(
                self,
                _catalog: dict[str, Any],
                _myth_summary: dict[str, Any],
                _entity_manifest: list[dict[str, Any]],
            ) -> dict[str, Any]:
                return {}

        life_tracker = _NullLifeTracker()

    class BoundWorldHandler(WorldHandler):
        pass

    BoundWorldHandler.part_root = part_root.resolve()
    BoundWorldHandler.vault_root = vault_root.resolve()
    BoundWorldHandler.host_label = f"{host}:{port}"
    BoundWorldHandler.task_queue = task_queue
    BoundWorldHandler.council_chamber = council_chamber
    BoundWorldHandler.myth_tracker = myth_tracker
    BoundWorldHandler.life_tracker = life_tracker
    BoundWorldHandler.life_interaction_builder = life_interaction_builder
    return BoundWorldHandler


def serve(
    part_root: Path,
    vault_root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
):
    _ensure_weaver_service(part_root, host)
    handler_class = make_handler(part_root, vault_root, host, port)
    server = ThreadingHTTPServer((host, port), handler_class)
    print(f"Starting server on {host}:{port}")
    _schedule_simulation_http_warmup(host=host, port=port)
    server.serve_forever()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--part-root", type=Path, default=Path("."))
    parser.add_argument("--vault-root", type=Path, default=Path(".."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    serve(args.part_root, args.vault_root, args.host, args.port)
    return 0
