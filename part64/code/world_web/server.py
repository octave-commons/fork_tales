# SPDX-License-Identifier: GPL-3.0-or-later
# This file is part of Fork Tales.
# Copyright (C) 2024-2025 Fork Tales Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import argparse
import copy
import base64
import hashlib
import json
import math
import mimetypes
import os
import queue
import shutil
import socket
import subprocess
import select
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, cast
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

from . import daimoi_probabilistic as daimoi_probabilistic_module
from . import simulation as simulation_module
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
from .governor import LaneType, Packet, get_governor
from .graph_queries import build_facts_snapshot, run_named_graph_query
from .paths import _ensure_receipts_log_path
from .projection import (
    attach_ui_projection,
    build_ui_projection,
    normalize_projection_perspective,
    projection_perspective_options,
)
from .simulation import (
    advance_simulation_field_particles,
    build_simulation_delta,
    build_mix_stream,
    build_named_field_overlays,
    build_simulation_state,
)
from .ws import (
    WS_CLIENT_FRAME_MAX_BYTES,
    consume_ws_client_frame as _consume_ws_client_frame,
    websocket_accept_value,
    websocket_frame_text,
)


_RUNTIME_CATALOG_CACHE_LOCK = threading.Lock()
_RUNTIME_CATALOG_REFRESH_LOCK = threading.Lock()
_RUNTIME_CATALOG_COLLECT_LOCK = threading.Lock()
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
_RUNTIME_CATALOG_HTTP_CACHE_SECONDS = max(
    0.0,
    float(os.getenv("RUNTIME_CATALOG_HTTP_CACHE_SECONDS", "0.75") or "0.75"),
)
_RUNTIME_CATALOG_HTTP_CACHE_LOCK = threading.Lock()
_RUNTIME_CATALOG_HTTP_CACHE: dict[str, dict[str, Any]] = {}
_RUNTIME_ETA_MU_SYNC_SECONDS = max(
    0.5,
    float(os.getenv("RUNTIME_ETA_MU_SYNC_SECONDS", "6.0") or "6.0"),
)
_RUNTIME_ETA_MU_SYNC_ENABLED = str(
    os.getenv("RUNTIME_ETA_MU_SYNC_ENABLED", "1") or "1"
).strip().lower() not in {"0", "false", "no", "off"}
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
_SIMULATION_HTTP_COMPACT_STALE_FALLBACK_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_COMPACT_STALE_FALLBACK_SECONDS", "4.0") or "4.0"),
)
_SIMULATION_HTTP_BUILD_WAIT_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_BUILD_WAIT_SECONDS", "12.0") or "12.0"),
)
_SIMULATION_HTTP_COMPACT_BUILD_WAIT_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_COMPACT_BUILD_WAIT_SECONDS", "1.5") or "1.5"),
)
_SIMULATION_HTTP_WARMUP_ENABLED = str(
    os.getenv("SIMULATION_HTTP_WARMUP_ENABLED", "0") or "0"
).strip().lower() not in {"0", "false", "no", "off"}
_SIMULATION_HTTP_CACHE_IGNORE_INFLUENCE = str(
    os.getenv("SIMULATION_HTTP_CACHE_IGNORE_INFLUENCE", "0") or "0"
).strip().lower() in {"1", "true", "yes", "on"}
_SIMULATION_HTTP_CACHE_IGNORE_QUEUE = str(
    os.getenv("SIMULATION_HTTP_CACHE_IGNORE_QUEUE", "0") or "0"
).strip().lower() in {"1", "true", "yes", "on"}
_SIMULATION_HTTP_WARMUP_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_HTTP_WARMUP_DELAY_SECONDS", "2.0") or "2.0"),
)


def _normalize_query_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    return " ".join(raw.split())[:220]


def _query_variant_terms(query_text: str) -> list[str]:
    base = _normalize_query_text(query_text)
    if not base:
        return []

    lowered = base.lower()
    token_rows = [
        token
        for token in "".join(
            ch if (ch.isalnum() or ch.isspace()) else " " for ch in lowered
        ).split()
        if token
    ]

    variants: list[str] = []
    for candidate in (
        base,
        lowered,
        " ".join(token_rows),
        " ".join(token_rows[:4]),
        " ".join(token_rows[-4:]),
    ):
        clean = _normalize_query_text(candidate)
        if clean and clean not in variants:
            variants.append(clean)
        if len(variants) >= 6:
            break
    return variants


def _build_search_daimoi_meta(
    query_text: str,
    *,
    target: str,
    model: str | None,
) -> dict[str, Any]:
    variants = _query_variant_terms(query_text)
    if not variants:
        return {}

    target_text = str(target or "").strip().lower()
    target_presence_ids: list[str] = []
    for row in ENTITY_MANIFEST:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("id", "") or "").strip()
        if not presence_id:
            continue
        if (
            presence_id.lower() in target_text
            and presence_id not in target_presence_ids
        ):
            target_presence_ids.append(presence_id)

    component_rows: list[dict[str, Any]] = []
    for index, term in enumerate(variants[:6]):
        component_id = hashlib.sha1(f"{term}|{index}".encode("utf-8")).hexdigest()[:12]
        embedding = _normalize_embedding_vector(_ollama_embed(term, model=model))
        component: dict[str, Any] = {
            "component_id": f"query:{component_id}",
            "component_type": "query-term",
            "kind": "search",
            "text": term,
            "weight": round(max(0.2, 1.0 - (index * 0.12)), 6),
            "variant_rank": index,
            "embedding_dim": 0,
        }
        if embedding:
            component["embedding_dim"] = len(embedding)
            component["embedding_preview"] = [
                round(float(value), 6) for value in embedding[:8]
            ]
        component_rows.append(component)

    return {
        "record": "ημ.user-search-daimoi.v1",
        "schema_version": "user.search.daimoi.v1",
        "query": variants[0],
        "variant_count": len(variants),
        "embed_model": str(model or "").strip(),
        "target_presence_ids": target_presence_ids,
        "components": component_rows,
    }


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
_SIMULATION_HTTP_DISK_FALLBACK_MAX_AGE_SECONDS = max(
    0.0,
    float(
        os.getenv("SIMULATION_HTTP_DISK_FALLBACK_MAX_AGE_SECONDS", "180.0") or "180.0"
    ),
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
_SIMULATION_HTTP_COMPACT_MAX_POINTS = max(
    256,
    int(float(os.getenv("SIMULATION_HTTP_COMPACT_MAX_POINTS", "2400") or "2400")),
)
_SIMULATION_HTTP_COMPACT_MAX_FIELD_PARTICLES = max(
    64,
    int(
        float(os.getenv("SIMULATION_HTTP_COMPACT_MAX_FIELD_PARTICLES", "900") or "900")
    ),
)
_SIMULATION_HTTP_CACHE_LOCK = threading.Lock()
_SIMULATION_HTTP_BUILD_LOCK = threading.Lock()
_SIMULATION_HTTP_CACHE: dict[str, Any] = {
    "key": "",
    "prepared_monotonic": 0.0,
    "body": b"",
}
_SIMULATION_HTTP_COMPACT_CACHE_LOCK = threading.Lock()
_SIMULATION_HTTP_COMPACT_CACHE: dict[str, Any] = {
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
_SIMULATION_BOOTSTRAP_REPORT_LOCK = threading.Lock()
_SIMULATION_BOOTSTRAP_LAST_REPORT: dict[str, Any] = {}
_SIMULATION_BOOTSTRAP_JOB_LOCK = threading.Lock()
_SIMULATION_BOOTSTRAP_JOB: dict[str, Any] = {
    "status": "idle",
    "job_id": "",
    "started_at": "",
    "updated_at": "",
    "completed_at": "",
    "phase": "",
    "phase_started_at": "",
    "phase_detail": {},
    "error": "",
    "request": {},
    "report": None,
}
_SIMULATION_BOOTSTRAP_MAX_SECONDS = max(
    30.0,
    float(os.getenv("SIMULATION_BOOTSTRAP_MAX_SECONDS", "240.0") or "240.0"),
)
_SIMULATION_BOOTSTRAP_HEARTBEAT_SECONDS = max(
    0.5,
    float(os.getenv("SIMULATION_BOOTSTRAP_HEARTBEAT_SECONDS", "3.0") or "3.0"),
)
_SIMULATION_BOOTSTRAP_MAX_EXCLUDED_FILES = max(
    24,
    int(float(os.getenv("SIMULATION_BOOTSTRAP_MAX_EXCLUDED_FILES", "320") or "320")),
)

_RUNTIME_WS_CLIENT_LOCK = threading.Lock()
_RUNTIME_WS_CLIENT_COUNT = 0
_RUNTIME_HTTP_MAX_THREADS = max(
    16,
    int(float(os.getenv("RUNTIME_HTTP_MAX_THREADS", "192") or "192")),
)
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
_SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS", "12.0") or "12.0"),
)
_SIMULATION_WS_PROJECTION_HEARTBEAT_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_WS_PROJECTION_HEARTBEAT_SECONDS", "2.5") or "2.5"),
)
_SIMULATION_WS_GRAPH_POSITION_HEARTBEAT_SECONDS = max(
    0.0,
    float(os.getenv("SIMULATION_WS_GRAPH_POSITION_HEARTBEAT_SECONDS", "2.0") or "2.0"),
)
_SIMULATION_WS_DELTA_STREAM_MODE = (
    str(os.getenv("SIMULATION_WS_DELTA_STREAM_MODE", "world") or "world")
    .strip()
    .lower()
)
_SIMULATION_WS_USE_CACHED_SNAPSHOTS = str(
    os.getenv("SIMULATION_WS_USE_CACHED_SNAPSHOTS", "0") or "0"
).strip().lower() in {"1", "true", "yes", "on"}
_SIMULATION_WS_CACHE_REFRESH_SECONDS = max(
    SIM_TICK_SECONDS,
    float(os.getenv("SIMULATION_WS_CACHE_REFRESH_SECONDS", "1.0") or "1.0"),
)
_SIMULATION_WS_CACHE_MAX_AGE_SECONDS = max(
    _SIMULATION_HTTP_CACHE_SECONDS,
    float(os.getenv("SIMULATION_WS_CACHE_MAX_AGE_SECONDS", "300.0") or "300.0"),
)
_SIMULATION_WS_CACHE_PARTICLE_CONTINUITY_BLEND = max(
    0.0,
    min(
        1.0,
        float(
            os.getenv("SIMULATION_WS_CACHE_PARTICLE_CONTINUITY_BLEND", "0.96") or "0.96"
        ),
    ),
)
_SIMULATION_WS_SKIP_CATALOG_BOOTSTRAP = str(
    os.getenv("SIMULATION_WS_SKIP_CATALOG_BOOTSTRAP", "0") or "0"
).strip().lower() in {"1", "true", "yes", "on"}
_SIMULATION_WS_BOOTSTRAP_REQUIRE_LIVE_REBUILD = str(
    os.getenv("SIMULATION_WS_BOOTSTRAP_REQUIRE_LIVE_REBUILD", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
_SIMULATION_WS_MUSE_POLL_SECONDS = max(
    0.05,
    float(os.getenv("SIMULATION_WS_MUSE_POLL_SECONDS", "0.5") or "0.5"),
)
_SIMULATION_WS_STREAM_PARTICLE_MAX = max(
    48,
    int(float(os.getenv("SIMULATION_WS_STREAM_PARTICLE_MAX", "180") or "180")),
)
_SIMULATION_WS_TICK_GOVERNOR_ENABLED = str(
    os.getenv("SIMULATION_WS_TICK_GOVERNOR_ENABLED", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
_SIMULATION_WS_TICK_GOVERNOR_RESOURCE_REFRESH_SECONDS = max(
    0.2,
    float(
        os.getenv(
            "SIMULATION_WS_TICK_GOVERNOR_RESOURCE_REFRESH_SECONDS",
            "1.0",
        )
        or "1.0"
    ),
)
_SIMULATION_WS_GOVERNOR_MIN_PARTICLE_CAP = max(
    24,
    int(float(os.getenv("SIMULATION_WS_GOVERNOR_MIN_PARTICLE_CAP", "72") or "72")),
)
_SIMULATION_WS_GOVERNOR_DEGRADE_GRAPH_HEARTBEAT_SCALE = max(
    1.0,
    float(
        os.getenv(
            "SIMULATION_WS_GOVERNOR_DEGRADE_GRAPH_HEARTBEAT_SCALE",
            "1.8",
        )
        or "1.8"
    ),
)
_SIMULATION_WS_GOVERNOR_INCREASE_GRAPH_HEARTBEAT_SCALE = max(
    0.1,
    min(
        1.0,
        float(
            os.getenv(
                "SIMULATION_WS_GOVERNOR_INCREASE_GRAPH_HEARTBEAT_SCALE",
                "0.9",
            )
            or "0.9"
        ),
    ),
)
_SIMULATION_WS_CHUNK_ENABLED = str(
    os.getenv("SIMULATION_WS_CHUNK_ENABLED", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
_SIMULATION_WS_CHUNK_CHARS = max(
    4096,
    int(float(os.getenv("SIMULATION_WS_CHUNK_CHARS", "48000") or "48000")),
)
_SIMULATION_WS_CHUNK_DELTA_MIN_CHARS = max(
    _SIMULATION_WS_CHUNK_CHARS,
    int(float(os.getenv("SIMULATION_WS_CHUNK_DELTA_MIN_CHARS", "96000") or "96000")),
)
_SIMULATION_WS_CHUNK_MAX_CHUNKS = max(
    8,
    int(float(os.getenv("SIMULATION_WS_CHUNK_MAX_CHUNKS", "256") or "256")),
)
_CATALOG_STREAM_CHUNK_ROWS = max(
    16,
    int(float(os.getenv("CATALOG_STREAM_CHUNK_ROWS", "192") or "192")),
)
_SIMULATION_WS_CHUNK_MESSAGE_TYPES = {
    str(item).strip().lower()
    for item in str(
        os.getenv(
            "SIMULATION_WS_CHUNK_MESSAGE_TYPES",
            "catalog,simulation,simulation_delta",
        )
        or "catalog,simulation,simulation_delta"
    ).split(",")
    if str(item).strip()
}
_SIMULATION_WS_PARTICLE_PAYLOAD_MODE_DEFAULT = (
    str(os.getenv("SIMULATION_WS_PARTICLE_PAYLOAD_MODE", "lite") or "lite")
    .strip()
    .lower()
)
_WS_WIRE_ARRAY_SCHEMA = "eta-mu.ws.arr.v1"
_WS_WIRE_MODE_DEFAULT = str(os.getenv("WS_WIRE_MODE", "json") or "json").strip().lower()
_SIMULATION_WS_CACHE_FORCE_JSON_WIRE = str(
    os.getenv("SIMULATION_WS_CACHE_FORCE_JSON_WIRE", "0") or "0"
).strip().lower() in {"1", "true", "yes", "on"}
_WS_PACK_TAG_OBJECT = -1
_WS_PACK_TAG_ARRAY = -2
_WS_PACK_TAG_STRING = -3
_WS_PACK_TAG_BOOL = -4
_WS_PACK_TAG_NULL = -5
_CATALOG_STREAM_SECTION_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("items", ("items",)),
    ("file_nodes", ("file_graph", "file_nodes")),
    ("file_edges", ("file_graph", "edges")),
    ("file_embed_layers", ("file_graph", "embed_layers")),
    ("crawler_nodes", ("crawler_graph", "crawler_nodes")),
    ("crawler_edges", ("crawler_graph", "edges")),
)
_SIMULATION_WS_PARTICLE_LITE_KEYS: tuple[str, ...] = (
    "id",
    "presence_id",
    "owner_presence_id",
    "target_presence_id",
    "presence_role",
    "particle_mode",
    "is_nexus",
    "x",
    "y",
    "size",
    "r",
    "g",
    "b",
    "vx",
    "vy",
    "resource_daimoi",
    "resource_type",
    "resource_consume_type",
    "top_job",
    "route_node_id",
    "graph_node_id",
    "route_probability",
    "influence_power",
    "route_resource_focus",
)
_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_LOCK = threading.Lock()
_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_STATE: dict[str, Any] = {
    "positions": {},
    "score": 0.0,
    "raw_score": 0.0,
    "peak_score": 0.0,
    "mean_displacement": 0.0,
    "p90_displacement": 0.0,
    "active_share": 0.0,
    "shared_nodes": 0,
    "sampled_nodes": 0,
}
_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NODE_LIMIT = max(
    64,
    int(
        float(
            os.getenv("SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NODE_LIMIT", "2048")
            or "2048"
        )
    ),
)
_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_DISTANCE_REF = max(
    0.001,
    float(
        os.getenv("SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_DISTANCE_REF", "0.02")
        or "0.02"
    ),
)
_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_EMA_ALPHA = max(
    0.01,
    min(
        1.0,
        float(
            os.getenv("SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_EMA_ALPHA", "0.2")
            or "0.2"
        ),
    ),
)
_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NOISE_GAIN = max(
    0.0,
    min(
        2.0,
        float(
            os.getenv("SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NOISE_GAIN", "0.8")
            or "0.8"
        ),
    ),
)
_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_ROUTE_DAMP = max(
    0.0,
    min(
        0.8,
        float(
            os.getenv("SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_ROUTE_DAMP", "0.3")
            or "0.3"
        ),
    ),
)
_SIMULATION_WS_DAIMOI_LIVE_METRICS_MIN_INTERVAL_SECONDS = max(
    0.0,
    _safe_float(
        os.getenv("SIMULATION_WS_DAIMOI_LIVE_METRICS_MIN_INTERVAL_SECONDS", "0.45")
        or "0.45",
        0.45,
    ),
)
_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_MIN_INTERVAL_SECONDS = max(
    0.0,
    _safe_float(
        os.getenv(
            "SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_MIN_INTERVAL_SECONDS",
            "0.55",
        )
        or "0.55",
        0.55,
    ),
)
_SIMULATION_WS_DAIMOI_METRICS_MIN_SLACK_MS = max(
    0.0,
    _safe_float(
        os.getenv("SIMULATION_WS_DAIMOI_METRICS_MIN_SLACK_MS", "3.5") or "3.5", 3.5
    ),
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


_CONFIG_MODULE_SPECS: dict[str, dict[str, Any]] = {
    "daimoi_probabilistic": {
        "module": daimoi_probabilistic_module,
        "prefixes": (
            "DAIMOI_",
            "NEXUS_",
            "_DAIMOI_",
            "_ABSORB_",
            "_ROLE_PRIOR_WEIGHTS",
        ),
        "exact_names": (),
    },
    "simulation": {
        "module": simulation_module,
        "prefixes": (
            "SIMULATION_",
            "_RESOURCE_DAIMOI_",
        ),
        "exact_names": (),
    },
    "server": {
        "module": sys.modules[__name__],
        "prefixes": (
            "_SIMULATION_HTTP_",
            "_SIMULATION_WS_",
            "_RUNTIME_GUARD_",
            "_WS_PACK_TAG_",
        ),
        "exact_names": (
            "_RUNTIME_ETA_MU_SYNC_SECONDS",
            "_RUNTIME_CATALOG_CACHE_SECONDS",
            "_RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS",
            "_RUNTIME_WS_MAX_CLIENTS",
            "_WS_CLIENT_FRAME_MAX_BYTES",
        ),
    },
}

_CONFIG_RUNTIME_EDIT_LOCK = threading.Lock()
_CONFIG_RUNTIME_BASELINE: dict[str, dict[str, Any]] = {}
_CONFIG_RUNTIME_VERSION_LOCK = threading.Lock()
_CONFIG_RUNTIME_VERSION = 0


def _config_runtime_version_snapshot() -> int:
    with _CONFIG_RUNTIME_VERSION_LOCK:
        return int(_CONFIG_RUNTIME_VERSION)


def _config_runtime_version_bump() -> int:
    global _CONFIG_RUNTIME_VERSION
    with _CONFIG_RUNTIME_VERSION_LOCK:
        _CONFIG_RUNTIME_VERSION = int(_CONFIG_RUNTIME_VERSION) + 1
        return int(_CONFIG_RUNTIME_VERSION)


def _config_numeric_scalar(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    return None


def _config_numeric_only(value: Any) -> Any:
    scalar = _config_numeric_scalar(value)
    if scalar is not None:
        return scalar

    if isinstance(value, dict):
        nested: dict[str, Any] = {}
        for key in sorted(value.keys(), key=lambda item: str(item)):
            numeric_value = _config_numeric_only(value[key])
            if numeric_value is None:
                continue
            nested[str(key)] = numeric_value
        return nested or None

    if isinstance(value, (list, tuple, set)):
        nested_list: list[Any] = []
        sequence = value
        if isinstance(value, set):
            sequence = sorted(value, key=lambda item: str(item))
        for item in sequence:
            numeric_value = _config_numeric_only(item)
            if numeric_value is None:
                continue
            nested_list.append(numeric_value)
        return nested_list or None

    return None


def _config_numeric_leaf_count(value: Any) -> int:
    scalar = _config_numeric_scalar(value)
    if scalar is not None:
        return 1
    if isinstance(value, dict):
        return sum(_config_numeric_leaf_count(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_config_numeric_leaf_count(item) for item in value)
    return 0


def _config_collect_module_constants(
    module: Any,
    *,
    prefixes: tuple[str, ...],
    exact_names: tuple[str, ...],
) -> tuple[dict[str, Any], int]:
    names = sorted(str(name) for name in vars(module).keys())
    selected: dict[str, Any] = {}
    numeric_leaf_count = 0
    for name in names:
        include = name in exact_names
        if not include:
            include = any(name.startswith(prefix) for prefix in prefixes)
        if not include:
            continue
        numeric_value = _config_numeric_only(getattr(module, name, None))
        if numeric_value is None:
            continue
        selected[name] = numeric_value
        numeric_leaf_count += _config_numeric_leaf_count(numeric_value)
    return selected, numeric_leaf_count


def _config_collect_selected_names(
    module: Any,
    *,
    prefixes: tuple[str, ...],
    exact_names: tuple[str, ...],
) -> list[str]:
    names = sorted(str(name) for name in vars(module).keys())
    selected: list[str] = []
    for name in names:
        include = name in exact_names
        if not include:
            include = any(name.startswith(prefix) for prefix in prefixes)
        if not include:
            continue
        numeric_value = _config_numeric_only(getattr(module, name, None))
        if numeric_value is None:
            continue
        selected.append(name)
    return selected


def _config_capture_runtime_baseline() -> dict[str, dict[str, Any]]:
    baseline: dict[str, dict[str, Any]] = {}
    for module_name, spec in _CONFIG_MODULE_SPECS.items():
        module = spec.get("module")
        if module is None:
            continue
        selected_names = _config_collect_selected_names(
            module,
            prefixes=tuple(spec.get("prefixes", ())),
            exact_names=tuple(spec.get("exact_names", ())),
        )
        module_baseline: dict[str, Any] = {}
        for name in selected_names:
            module_baseline[name] = copy.deepcopy(getattr(module, name, None))
        baseline[module_name] = module_baseline
    return baseline


def _config_normalize_path_tokens(path_raw: Any) -> list[str]:
    if path_raw is None:
        return []
    if isinstance(path_raw, list):
        return [str(item).strip() for item in path_raw if str(item).strip()]
    if isinstance(path_raw, str):
        clean = path_raw.strip()
        if not clean:
            return []
        normalized = clean.replace("[", ".").replace("]", "")
        return [token for token in normalized.split(".") if token]
    return []


def _config_resolve_dict_key(container: dict[Any, Any], token: str) -> Any:
    if token in container:
        return token
    for key in container.keys():
        if str(key) == token:
            return key
    raise KeyError(token)


def _config_parse_list_index(token: str, length: int) -> int:
    try:
        index = int(token)
    except Exception as exc:
        raise ValueError(f"invalid_index:{token}") from exc
    if index < 0:
        index = length + index
    if index < 0 or index >= length:
        raise IndexError(f"index_out_of_range:{token}")
    return index


def _config_coerce_numeric_like(reference: Any, value: float | int) -> float | int:
    if isinstance(reference, bool):
        return int(round(float(value)))
    if isinstance(reference, int):
        return int(round(float(value)))
    return float(value)


_CONFIG_SCALAR_LIMITS: dict[tuple[str, str], tuple[float, float]] = {
    ("simulation", "SIMULATION_STREAM_DAIMOI_FRICTION"): (0.0, 2.0),
    ("simulation", "SIMULATION_STREAM_NEXUS_FRICTION"): (0.0, 2.0),
    ("simulation", "SIMULATION_STREAM_FRICTION"): (0.0, 2.0),
}


def _config_clamp_scalar_update(
    *,
    module_name: str,
    key_name: str,
    value: float | int,
) -> float | int:
    limits = _CONFIG_SCALAR_LIMITS.get((module_name, key_name))
    if limits is None:
        return value
    lower, upper = limits
    clamped = max(lower, min(upper, float(value)))
    return clamped


def _config_get_at_path(root: Any, path_tokens: list[str]) -> Any:
    value = root
    for token in path_tokens:
        if isinstance(value, dict):
            value = value[_config_resolve_dict_key(value, token)]
            continue
        if isinstance(value, (list, tuple)):
            index = _config_parse_list_index(token, len(value))
            value = value[index]
            continue
        raise TypeError(f"non_container_at:{token}")
    return value


def _config_set_at_path(
    root: Any, path_tokens: list[str], new_scalar: float | int
) -> Any:
    if not path_tokens:
        scalar = _config_numeric_scalar(root)
        if scalar is None:
            raise TypeError("target_not_numeric")
        return _config_coerce_numeric_like(root, new_scalar)

    token = path_tokens[0]
    tail = path_tokens[1:]
    if isinstance(root, dict):
        key = _config_resolve_dict_key(root, token)
        updated = dict(root)
        updated[key] = _config_set_at_path(root[key], tail, new_scalar)
        return updated

    if isinstance(root, list):
        index = _config_parse_list_index(token, len(root))
        updated_list = list(root)
        updated_list[index] = _config_set_at_path(root[index], tail, new_scalar)
        return updated_list

    if isinstance(root, tuple):
        index = _config_parse_list_index(token, len(root))
        updated_list = list(root)
        updated_list[index] = _config_set_at_path(root[index], tail, new_scalar)
        return tuple(updated_list)

    raise TypeError(f"non_container_at:{token}")


def _config_apply_update(
    *,
    module_name: str,
    key_name: str,
    path_tokens: list[str],
    value: Any,
) -> dict[str, Any]:
    requested_module = str(module_name or "").strip().lower()
    requested_key = str(key_name or "").strip()
    if not requested_module:
        return {"ok": False, "error": "module_required"}
    if not requested_key:
        return {"ok": False, "error": "key_required"}
    if requested_module not in _CONFIG_MODULE_SPECS:
        return {
            "ok": False,
            "error": "unknown_module",
            "module": requested_module,
            "available_modules": sorted(_CONFIG_MODULE_SPECS.keys()),
        }

    next_scalar = _config_numeric_scalar(value)
    if next_scalar is None and isinstance(value, str):
        text = value.strip()
        if text:
            try:
                next_scalar = float(text)
            except Exception:
                next_scalar = None
    if next_scalar is None:
        return {
            "ok": False,
            "error": "numeric_value_required",
            "module": requested_module,
            "key": requested_key,
        }
    next_scalar_value: float | int = next_scalar
    next_scalar_value = _config_clamp_scalar_update(
        module_name=requested_module,
        key_name=requested_key,
        value=next_scalar_value,
    )

    spec = _CONFIG_MODULE_SPECS[requested_module]
    module = spec.get("module")
    if module is None:
        return {
            "ok": False,
            "error": "module_unavailable",
            "module": requested_module,
        }

    selected_names = set(
        _config_collect_selected_names(
            module,
            prefixes=tuple(spec.get("prefixes", ())),
            exact_names=tuple(spec.get("exact_names", ())),
        )
    )
    if requested_key not in selected_names:
        return {
            "ok": False,
            "error": "unknown_constant",
            "module": requested_module,
            "key": requested_key,
        }

    try:
        with _CONFIG_RUNTIME_EDIT_LOCK:
            current_value = copy.deepcopy(getattr(module, requested_key, None))
            previous_leaf = _config_get_at_path(current_value, path_tokens)
            updated_value = _config_set_at_path(
                current_value,
                path_tokens,
                next_scalar_value,
            )
            setattr(module, requested_key, updated_value)
            current_after = copy.deepcopy(getattr(module, requested_key, None))
            current_leaf = _config_get_at_path(current_after, path_tokens)
    except Exception as exc:
        return {
            "ok": False,
            "error": "config_update_failed",
            "detail": f"{exc.__class__.__name__}: {exc}",
            "module": requested_module,
            "key": requested_key,
            "path": path_tokens,
        }

    return {
        "ok": True,
        "record": "eta-mu.runtime-config.update.v1",
        "schema_version": "runtime.config.update.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module": requested_module,
        "key": requested_key,
        "path": path_tokens,
        "previous": _config_numeric_only(previous_leaf),
        "current": _config_numeric_only(current_leaf),
    }


def _config_reset_updates(
    *,
    module_name: str = "",
    key_name: str = "",
    path_tokens: list[str] | None = None,
) -> dict[str, Any]:
    requested_module = str(module_name or "").strip().lower()
    requested_key = str(key_name or "").strip()
    normalized_path = path_tokens or []

    if normalized_path and not requested_key:
        return {
            "ok": False,
            "error": "key_required_for_path_reset",
        }

    available_modules = sorted(_CONFIG_MODULE_SPECS.keys())
    if requested_module and requested_module not in _CONFIG_MODULE_SPECS:
        return {
            "ok": False,
            "error": "unknown_module",
            "module": requested_module,
            "available_modules": available_modules,
        }

    module_names = [requested_module] if requested_module else available_modules
    applied: list[dict[str, Any]] = []

    with _CONFIG_RUNTIME_EDIT_LOCK:
        for module_item in module_names:
            spec = _CONFIG_MODULE_SPECS[module_item]
            module = spec.get("module")
            if module is None:
                continue
            baseline_module = _CONFIG_RUNTIME_BASELINE.get(module_item, {})
            if not baseline_module:
                continue
            selected_names = set(
                _config_collect_selected_names(
                    module,
                    prefixes=tuple(spec.get("prefixes", ())),
                    exact_names=tuple(spec.get("exact_names", ())),
                )
            )
            key_names = (
                [requested_key] if requested_key else sorted(baseline_module.keys())
            )
            for key_item in key_names:
                if key_item not in selected_names:
                    continue
                if key_item not in baseline_module:
                    continue

                baseline_value = copy.deepcopy(baseline_module[key_item])
                if normalized_path:
                    try:
                        baseline_leaf = _config_get_at_path(
                            baseline_value, normalized_path
                        )
                        baseline_scalar = _config_numeric_scalar(baseline_leaf)
                        if baseline_scalar is None:
                            continue
                        current_value = copy.deepcopy(getattr(module, key_item, None))
                        updated_value = _config_set_at_path(
                            current_value,
                            normalized_path,
                            baseline_scalar,
                        )
                    except Exception:
                        continue
                else:
                    updated_value = baseline_value

                setattr(module, key_item, updated_value)
                applied.append(
                    {
                        "module": module_item,
                        "key": key_item,
                        "path": list(normalized_path),
                    }
                )

    if requested_key and not applied:
        return {
            "ok": False,
            "error": "unknown_constant",
            "module": requested_module,
            "key": requested_key,
        }

    return {
        "ok": True,
        "record": "eta-mu.runtime-config.reset.v1",
        "schema_version": "runtime.config.reset.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module": requested_module,
        "key": requested_key,
        "path": normalized_path,
        "reset_count": len(applied),
        "resets": applied[:128],
    }


_CONFIG_RUNTIME_BASELINE = _config_capture_runtime_baseline()


def _config_payload(*, module_filter: str = "") -> dict[str, Any]:
    requested = str(module_filter or "").strip().lower()
    available_modules = sorted(_CONFIG_MODULE_SPECS.keys())
    if requested and requested not in _CONFIG_MODULE_SPECS:
        return {
            "ok": False,
            "error": "unknown_module",
            "requested_module": requested,
            "available_modules": available_modules,
        }

    module_names = [requested] if requested else available_modules

    modules_payload: dict[str, Any] = {}
    total_constants = 0
    total_numeric_leaf_count = 0
    for module_name in module_names:
        spec = _CONFIG_MODULE_SPECS[module_name]
        constants, leaf_count = _config_collect_module_constants(
            spec["module"],
            prefixes=tuple(spec["prefixes"]),
            exact_names=tuple(spec["exact_names"]),
        )
        modules_payload[module_name] = {
            "constants": copy.deepcopy(constants),
            "constant_count": len(constants),
            "numeric_leaf_count": leaf_count,
        }
        total_constants += len(constants)
        total_numeric_leaf_count += leaf_count

    return {
        "ok": True,
        "record": "eta-mu.runtime-config.v1",
        "schema_version": "runtime.config.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_config_version": _config_runtime_version_snapshot(),
        "available_modules": available_modules,
        "requested_module": requested,
        "module_count": len(modules_payload),
        "constant_count": total_constants,
        "numeric_leaf_count": total_numeric_leaf_count,
        "modules": modules_payload,
    }


def _json_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _normalize_ws_wire_mode(mode: str) -> str:
    clean = str(mode or "").strip().lower()
    if clean in {"arr", "array", "arrays", "packed", "compact"}:
        return "arr"
    if clean in {"json", "object", "objects", "legacy"}:
        return "json"
    default_mode = str(_WS_WIRE_MODE_DEFAULT or "json").strip().lower()
    if default_mode in {"arr", "array", "arrays", "packed", "compact"}:
        return "arr"
    return "json"


def _ws_pack_value(
    value: Any,
    *,
    key_table: list[str],
    key_index: dict[str, int],
) -> Any:
    if value is None:
        return [_WS_PACK_TAG_NULL]
    if isinstance(value, bool):
        return [_WS_PACK_TAG_BOOL, 1 if value else 0]
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return [_WS_PACK_TAG_STRING, "0"]
        return value
    if isinstance(value, str):
        return [_WS_PACK_TAG_STRING, value]
    if isinstance(value, list):
        encoded_items = [
            _ws_pack_value(item, key_table=key_table, key_index=key_index)
            for item in value
        ]
        return [_WS_PACK_TAG_ARRAY, *encoded_items]
    if isinstance(value, dict):
        encoded_pairs: list[Any] = [_WS_PACK_TAG_OBJECT]
        for key, nested in value.items():
            key_name = str(key)
            key_slot = key_index.get(key_name)
            if key_slot is None:
                key_slot = len(key_table)
                key_index[key_name] = key_slot
                key_table.append(key_name)
            encoded_pairs.append(key_slot)
            encoded_pairs.append(
                _ws_pack_value(nested, key_table=key_table, key_index=key_index)
            )
        return encoded_pairs
    return [_WS_PACK_TAG_STRING, str(value)]


def _ws_pack_message(payload: dict[str, Any]) -> list[Any]:
    key_table: list[str] = []
    key_index: dict[str, int] = {}
    encoded = _ws_pack_value(payload, key_table=key_table, key_index=key_index)
    return [_WS_WIRE_ARRAY_SCHEMA, key_table, encoded]


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

    queue_pending = 0
    queue_events = 0
    if not _SIMULATION_HTTP_CACHE_IGNORE_QUEUE:
        queue_pending = int(_safe_float(queue_snapshot.get("pending_count", 0), 0.0))
        queue_events = int(_safe_float(queue_snapshot.get("event_count", 0), 0.0))
    clicks_recent = 0
    user_inputs_recent = 0
    user_signal = ""
    if not _SIMULATION_HTTP_CACHE_IGNORE_INFLUENCE:
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
    user_signal_hash = hashlib.sha1(user_signal.encode("utf-8")).hexdigest()[:10]
    config_version = max(0, _config_runtime_version_snapshot())
    return (
        f"{perspective}|{fingerprint}|"
        f"q:{max(0, queue_pending)}:{max(0, queue_events)}|"
        f"i:{max(0, clicks_recent)}:{max(0, user_inputs_recent)}:{user_signal_hash}|"
        f"cfg:{config_version}|simulation"
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


def _runtime_catalog_http_cache_store(*, perspective: str, body: bytes) -> None:
    perspective_key = str(perspective or "").strip().lower()
    if not perspective_key:
        return
    if not isinstance(body, (bytes, bytearray)):
        return
    body_bytes = bytes(body)
    if not body_bytes:
        return
    with _RUNTIME_CATALOG_HTTP_CACHE_LOCK:
        _RUNTIME_CATALOG_HTTP_CACHE[perspective_key] = {
            "prepared_monotonic": time.monotonic(),
            "body": body_bytes,
        }


def _runtime_catalog_http_cached_body(
    *,
    perspective: str,
    max_age_seconds: float,
) -> bytes | None:
    if max_age_seconds <= 0.0:
        return None
    perspective_key = str(perspective or "").strip().lower()
    if not perspective_key:
        return None

    with _RUNTIME_CATALOG_HTTP_CACHE_LOCK:
        cache_row = _RUNTIME_CATALOG_HTTP_CACHE.get(perspective_key)
        row = dict(cache_row) if isinstance(cache_row, dict) else None

    if not isinstance(row, dict):
        return None
    cached_body = row.get("body", b"")
    if not isinstance(cached_body, (bytes, bytearray)) or not cached_body:
        return None
    cached_age = time.monotonic() - _safe_float(row.get("prepared_monotonic", 0.0), 0.0)
    if cached_age < 0.0 or cached_age > max_age_seconds:
        return None
    return bytes(cached_body)


def _runtime_catalog_http_cache_invalidate() -> None:
    with _RUNTIME_CATALOG_HTTP_CACHE_LOCK:
        _RUNTIME_CATALOG_HTTP_CACHE.clear()


def _simulation_http_compact_stale_fallback_body(
    *,
    part_root: Path,
    perspective: str,
    max_age_seconds: float,
) -> tuple[bytes | None, str]:
    stale_max_age = max(0.0, _safe_float(max_age_seconds, 0.0))
    if stale_max_age <= 0.0:
        return None, ""

    perspective_key = str(perspective or "").strip().lower()
    stale_body = _simulation_http_cached_body(
        perspective=perspective_key,
        max_age_seconds=stale_max_age,
    )
    if stale_body is not None:
        return stale_body, "stale-cache"

    disk_body = _simulation_http_disk_cache_load(
        part_root,
        perspective=perspective_key,
        max_age_seconds=stale_max_age,
    )
    if disk_body is None:
        return None, ""

    _simulation_http_cache_store(
        f"{perspective_key}|disk-compact-fallback|simulation",
        disk_body,
    )
    return disk_body, "disk-cache"


def _simulation_http_compact_cache_store(cache_key: str, body: bytes) -> None:
    if not cache_key:
        return
    if not isinstance(body, (bytes, bytearray)):
        return
    body_bytes = bytes(body)
    if not body_bytes:
        return
    with _SIMULATION_HTTP_COMPACT_CACHE_LOCK:
        _SIMULATION_HTTP_COMPACT_CACHE["key"] = cache_key
        _SIMULATION_HTTP_COMPACT_CACHE["prepared_monotonic"] = time.monotonic()
        _SIMULATION_HTTP_COMPACT_CACHE["body"] = body_bytes


def _simulation_http_compact_cached_body(
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
    with _SIMULATION_HTTP_COMPACT_CACHE_LOCK:
        cached_key = str(_SIMULATION_HTTP_COMPACT_CACHE.get("key", "") or "").strip()
        cached_body = _SIMULATION_HTTP_COMPACT_CACHE.get("body", b"")
        cached_age = time.monotonic() - _safe_float(
            _SIMULATION_HTTP_COMPACT_CACHE.get("prepared_monotonic", 0.0),
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
    elif requested_key:
        if requested_key != cached_key:
            return None
    elif requested_perspective and not cached_key.startswith(
        f"{requested_perspective}|"
    ):
        return None

    return bytes(cached_body)


def _simulation_http_cache_invalidate(*, part_root: Path | None = None) -> None:
    _runtime_catalog_http_cache_invalidate()
    with _SIMULATION_HTTP_CACHE_LOCK:
        _SIMULATION_HTTP_CACHE["key"] = ""
        _SIMULATION_HTTP_CACHE["prepared_monotonic"] = 0.0
        _SIMULATION_HTTP_CACHE["body"] = b""
    with _SIMULATION_HTTP_COMPACT_CACHE_LOCK:
        _SIMULATION_HTTP_COMPACT_CACHE["key"] = ""
        _SIMULATION_HTTP_COMPACT_CACHE["prepared_monotonic"] = 0.0
        _SIMULATION_HTTP_COMPACT_CACHE["body"] = b""

    if part_root is None or not _SIMULATION_HTTP_DISK_CACHE_ENABLED:
        return

    perspectives: set[str] = set()
    default_perspective = str(PROJECTION_DEFAULT_PERSPECTIVE or "").strip().lower()
    if default_perspective:
        perspectives.add(default_perspective)
    for option in projection_perspective_options():
        if isinstance(option, dict):
            option_id = str(option.get("id", "") or "").strip().lower()
            if option_id:
                perspectives.add(option_id)
    for perspective in perspectives:
        cache_path = _simulation_http_disk_cache_path(part_root, perspective)
        try:
            cache_path.unlink(missing_ok=True)
        except Exception:
            continue


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
        "heat_values",
        "file_graph",
        "crawler_graph",
        "field_registry",
    ):
        trimmed.pop(key, None)
    return trimmed


def _simulation_ws_compact_graph_payload(
    simulation: dict[str, Any],
    *,
    assume_trimmed: bool = False,
) -> dict[str, Any]:
    if not isinstance(simulation, dict):
        return {}

    if assume_trimmed:
        graph_payload: dict[str, Any] = {}
        file_graph = simulation.get("file_graph")
        crawler_graph = simulation.get("crawler_graph")
        if isinstance(file_graph, dict):
            graph_payload["file_graph"] = file_graph
        if isinstance(crawler_graph, dict):
            graph_payload["crawler_graph"] = crawler_graph
        return graph_payload

    catalog_like: dict[str, Any] = {}
    file_graph = simulation.get("file_graph")
    crawler_graph = simulation.get("crawler_graph")
    if isinstance(file_graph, dict):
        catalog_like["file_graph"] = file_graph
    if isinstance(crawler_graph, dict):
        catalog_like["crawler_graph"] = crawler_graph
    if not catalog_like:
        return {}

    compact_catalog = _simulation_http_trim_catalog(catalog_like)
    if not isinstance(compact_catalog, dict):
        return {}

    graph_payload: dict[str, Any] = {}
    compact_file_graph = compact_catalog.get("file_graph")
    if isinstance(compact_file_graph, dict):
        graph_payload["file_graph"] = compact_file_graph
    compact_crawler_graph = compact_catalog.get("crawler_graph")
    if isinstance(compact_crawler_graph, dict):
        graph_payload["crawler_graph"] = compact_crawler_graph
    return graph_payload


def _simulation_ws_placeholder_payload(
    *,
    perspective: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    simulation: dict[str, Any] = {
        "ok": True,
        "generated_at": now_iso,
        "timestamp": now_iso,
        "total": 0,
        "audio": 0,
        "image": 0,
        "video": 0,
        "points": [],
        "presence_dynamics": {
            "generated_at": now_iso,
            "simulation_budget": {
                "point_limit": 0,
                "point_limit_max": 0,
                "cpu_utilization": 0.0,
                "slice_offload": {},
            },
            "emission_policy": {},
            "presence_impacts": [],
            "field_particles": [],
            "daimoi_probabilistic": {},
        },
        "perspective": perspective,
    }
    return simulation, {}


def _simulation_ws_compact_field_particles(rows: Any) -> list[dict[str, Any]]:
    return _simulation_ws_compact_field_particles_with_nodes(
        rows,
        node_positions={},
        node_text_chars={},
    )


def _simulation_ws_collect_node_positions(
    simulation_payload: dict[str, Any],
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    if not isinstance(simulation_payload, dict):
        return {}, {}

    node_positions: dict[str, tuple[float, float]] = {}
    node_text_chars: dict[str, float] = {}

    def _node_text_weight(row: dict[str, Any]) -> float:
        text_keys = (
            "summary",
            "text_excerpt",
            "excerpt",
            "title",
            "label",
            "name",
            "url",
            "dominant_field",
            "dominant_presence",
        )
        total = 0
        for key in text_keys:
            value = row.get(key)
            if isinstance(value, str):
                total += len(value)
            elif isinstance(value, list):
                total += sum(len(str(item)) for item in value)
        return float(total)

    for graph_key in ("file_graph", "crawler_graph", "nexus_graph"):
        graph_payload = simulation_payload.get(graph_key, {})
        if not isinstance(graph_payload, dict):
            continue
        for section_key in (
            "nodes",
            "file_nodes",
            "crawler_nodes",
            "field_nodes",
            "tag_nodes",
        ):
            rows = graph_payload.get(section_key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                node_id = str(row.get("id", "") or "").strip()
                if not node_id or node_id in node_positions:
                    continue
                x_value = max(0.0, min(1.0, _safe_float(row.get("x", 0.5), 0.5)))
                y_value = max(0.0, min(1.0, _safe_float(row.get("y", 0.5), 0.5)))
                node_positions[node_id] = (x_value, y_value)
                node_text_chars[node_id] = _node_text_weight(row)

    return node_positions, node_text_chars


def _simulation_ws_compact_field_particles_with_nodes(
    rows: Any,
    *,
    node_positions: dict[str, tuple[float, float]],
    node_text_chars: dict[str, float],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []

    compact_rows: list[dict[str, Any]] = []
    limit = max(1, _SIMULATION_WS_STREAM_PARTICLE_MAX)
    for index, row in enumerate(rows):
        if index >= limit:
            break
        if not isinstance(row, dict):
            continue
        particle_id = str(row.get("id", "") or "").strip() or f"ws:{index}"
        x_value = max(0.0, min(1.0, _safe_float(row.get("x", 0.5), 0.5)))
        y_value = max(0.0, min(1.0, _safe_float(row.get("y", 0.5), 0.5)))
        vx_value = _safe_float(row.get("vx", 0.0), 0.0)
        vy_value = _safe_float(row.get("vy", 0.0), 0.0)

        if abs(vx_value) + abs(vy_value) < 1e-6:
            seed = int(hashlib.sha1(particle_id.encode("utf-8")).hexdigest()[:8], 16)
            angle = (float(seed % 6283) / 1000.0) * math.pi
            vx_value = math.cos(angle) * 0.01
            vy_value = math.sin(angle) * 0.01

        graph_node_id = str(row.get("graph_node_id", "") or "")
        route_node_id = str(row.get("route_node_id", "") or "")
        graph_anchor = node_positions.get(graph_node_id)
        route_anchor = node_positions.get(route_node_id)
        graph_x = (
            _safe_float(graph_anchor[0], x_value)
            if isinstance(graph_anchor, tuple)
            else _safe_float(row.get("graph_x", x_value), x_value)
        )
        graph_y = (
            _safe_float(graph_anchor[1], y_value)
            if isinstance(graph_anchor, tuple)
            else _safe_float(row.get("graph_y", y_value), y_value)
        )
        route_x = (
            _safe_float(route_anchor[0], graph_x)
            if isinstance(route_anchor, tuple)
            else _safe_float(row.get("route_x", graph_x), graph_x)
        )
        route_y = (
            _safe_float(route_anchor[1], graph_y)
            if isinstance(route_anchor, tuple)
            else _safe_float(row.get("route_y", graph_y), graph_y)
        )
        semantic_text_chars = max(
            0.0,
            _safe_float(row.get("semantic_text_chars", 0.0), 0.0),
            _safe_float(node_text_chars.get(route_node_id, 0.0), 0.0),
            _safe_float(node_text_chars.get(graph_node_id, 0.0), 0.0),
        )
        message_probability = max(
            0.0,
            _safe_float(row.get("message_probability", 0.0), 0.0),
        )
        package_entropy = max(
            0.0,
            _safe_float(row.get("package_entropy", 0.0), 0.0),
        )
        daimoi_energy = max(
            0.0,
            _safe_float(row.get("daimoi_energy", 0.0), 0.0),
            message_probability + (package_entropy * 0.35),
        )
        semantic_mass = max(
            0.05,
            _safe_float(row.get("semantic_mass", 0.0), 0.0),
            _safe_float(row.get("mass", 0.0), 0.0),
        )

        compact_rows.append(
            {
                "id": particle_id,
                "presence_id": str(row.get("presence_id", "") or ""),
                "presence_role": str(row.get("presence_role", "") or ""),
                "particle_mode": str(row.get("particle_mode", "") or ""),
                "is_nexus": bool(row.get("is_nexus", False)),
                "graph_node_id": graph_node_id,
                "route_node_id": route_node_id,
                "route_resource_focus": str(row.get("route_resource_focus", "") or ""),
                "graph_x": round(max(0.0, min(1.0, graph_x)), 5),
                "graph_y": round(max(0.0, min(1.0, graph_y)), 5),
                "route_x": round(max(0.0, min(1.0, route_x)), 5),
                "route_y": round(max(0.0, min(1.0, route_y)), 5),
                "semantic_text_chars": round(semantic_text_chars, 3),
                "semantic_mass": round(semantic_mass, 6),
                "daimoi_energy": round(daimoi_energy, 6),
                "message_probability": round(message_probability, 6),
                "package_entropy": round(package_entropy, 6),
                "collision_count": int(
                    max(0, _safe_float(row.get("collision_count", 0.0), 0.0))
                ),
                "x": round(x_value, 5),
                "y": round(y_value, 5),
                "size": round(
                    max(0.2, min(6.0, _safe_float(row.get("size", 1.2), 1.2))),
                    4,
                ),
                "r": round(max(0.0, min(1.0, _safe_float(row.get("r", 0.4), 0.4))), 4),
                "g": round(max(0.0, min(1.0, _safe_float(row.get("g", 0.5), 0.5))), 4),
                "b": round(max(0.0, min(1.0, _safe_float(row.get("b", 0.7), 0.7))), 4),
                "drift_cost_semantic_term": round(
                    _safe_float(row.get("drift_cost_semantic_term", 0.0), 0.0), 6
                ),
                "drift_gravity_term": round(
                    _safe_float(row.get("drift_gravity_term", 0.0), 0.0), 6
                ),
                "valve_gravity_term": round(
                    _safe_float(row.get("valve_gravity_term", 0.0), 0.0), 6
                ),
                "drift_cost_term": round(
                    _safe_float(row.get("drift_cost_term", 0.0), 0.0), 6
                ),
                "route_probability": round(
                    max(
                        0.0,
                        min(1.0, _safe_float(row.get("route_probability", 0.0), 0.0)),
                    ),
                    6,
                ),
                "influence_power": round(
                    max(
                        0.0, min(1.0, _safe_float(row.get("influence_power", 0.0), 0.0))
                    ),
                    6,
                ),
                "node_saturation": round(
                    max(
                        0.0, min(1.0, _safe_float(row.get("node_saturation", 0.0), 0.0))
                    ),
                    6,
                ),
                "gravity_potential": round(
                    max(0.0, _safe_float(row.get("gravity_potential", 0.0), 0.0)), 6
                ),
                "route_resource_focus_contribution": round(
                    _safe_float(row.get("route_resource_focus_contribution", 0.0), 0.0),
                    6,
                ),
                "vx": round(vx_value, 6),
                "vy": round(vy_value, 6),
            }
        )
    return compact_rows


def _simulation_ws_extract_stream_particles(
    simulation_payload: dict[str, Any],
    *,
    node_positions: dict[str, tuple[float, float]] | None = None,
    node_text_chars: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    dynamics = (
        simulation_payload.get("presence_dynamics", {})
        if isinstance(simulation_payload, dict)
        else {}
    )
    if not isinstance(dynamics, dict):
        return []

    compact_rows = _simulation_ws_compact_field_particles_with_nodes(
        dynamics.get("field_particles", []),
        node_positions=node_positions or {},
        node_text_chars=node_text_chars or {},
    )
    dynamics["field_particles"] = compact_rows
    _simulation_ws_ensure_daimoi_summary(simulation_payload)
    simulation_payload["presence_dynamics"] = dynamics
    return compact_rows


def _simulation_ws_capture_particle_motion_state(
    simulation_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(simulation_payload, dict):
        return {}
    dynamics = simulation_payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return {}
    rows = dynamics.get("field_particles", [])
    if not isinstance(rows, list) or not rows:
        return {}

    captured: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        particle_id = str(row.get("id", "") or "").strip() or f"ws:{index}"
        captured[particle_id] = {
            "x": max(0.0, min(1.0, _safe_float(row.get("x", 0.5), 0.5))),
            "y": max(0.0, min(1.0, _safe_float(row.get("y", 0.5), 0.5))),
            "vx": _safe_float(row.get("vx", 0.0), 0.0),
            "vy": _safe_float(row.get("vy", 0.0), 0.0),
            "presence_id": str(row.get("presence_id", "") or "").strip(),
            "is_nexus": bool(row.get("is_nexus", False)),
        }
    return captured


def _simulation_ws_restore_particle_motion_state(
    simulation_payload: dict[str, Any],
    motion_state: dict[str, dict[str, Any]],
) -> int:
    if not isinstance(simulation_payload, dict):
        return 0
    if not isinstance(motion_state, dict) or not motion_state:
        return 0

    dynamics = simulation_payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return 0
    rows = dynamics.get("field_particles", [])
    if not isinstance(rows, list) or not rows:
        return 0

    blend = _SIMULATION_WS_CACHE_PARTICLE_CONTINUITY_BLEND
    restored = 0

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        particle_id = str(row.get("id", "") or "").strip() or f"ws:{index}"
        previous = motion_state.get(particle_id)
        if not isinstance(previous, dict):
            continue

        current_presence_id = str(row.get("presence_id", "") or "").strip()
        previous_presence_id = str(previous.get("presence_id", "") or "").strip()
        if (
            previous_presence_id
            and current_presence_id
            and previous_presence_id != current_presence_id
        ):
            continue
        if bool(previous.get("is_nexus", False)) != bool(row.get("is_nexus", False)):
            continue

        previous_x = max(0.0, min(1.0, _safe_float(previous.get("x", 0.5), 0.5)))
        previous_y = max(0.0, min(1.0, _safe_float(previous.get("y", 0.5), 0.5)))
        current_x = max(
            0.0, min(1.0, _safe_float(row.get("x", previous_x), previous_x))
        )
        current_y = max(
            0.0, min(1.0, _safe_float(row.get("y", previous_y), previous_y))
        )
        previous_vx = _safe_float(previous.get("vx", 0.0), 0.0)
        previous_vy = _safe_float(previous.get("vy", 0.0), 0.0)
        current_vx = _safe_float(row.get("vx", previous_vx), previous_vx)
        current_vy = _safe_float(row.get("vy", previous_vy), previous_vy)

        row["x"] = round((previous_x * blend) + (current_x * (1.0 - blend)), 5)
        row["y"] = round((previous_y * blend) + (current_y * (1.0 - blend)), 5)
        row["vx"] = round((previous_vx * blend) + (current_vx * (1.0 - blend)), 6)
        row["vy"] = round((previous_vy * blend) + (current_vy * (1.0 - blend)), 6)
        restored += 1

    if restored > 0:
        dynamics["field_particles"] = rows
        simulation_payload["presence_dynamics"] = dynamics
    return restored


def _simulation_ws_decode_cached_payload(cached_body: Any) -> dict[str, Any] | None:
    if not isinstance(cached_body, (bytes, bytearray)):
        return None
    body = bytes(cached_body)
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _simulation_ws_payload_is_sparse(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return True
    total_count = max(0, int(_safe_float(payload.get("total", 0), 0.0)))
    point_rows = payload.get("points", [])
    point_count = len(point_rows) if isinstance(point_rows, list) else 0

    particle_count = 0
    dynamics = payload.get("presence_dynamics", {})
    if isinstance(dynamics, dict):
        rows = dynamics.get("field_particles", [])
        if isinstance(rows, list):
            particle_count = len(rows)

    return total_count <= 0 and point_count <= 0 and particle_count <= 0


def _simulation_ws_payload_missing_graph_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return True

    if isinstance(payload.get("file_graph"), dict) or isinstance(
        payload.get("crawler_graph"), dict
    ):
        return False

    total_count = max(0, int(_safe_float(payload.get("total", 0), 0.0)))
    view_graph = payload.get("view_graph", {})
    truth_graph = payload.get("truth_graph", {})
    view_node_count = (
        max(0, int(_safe_float(view_graph.get("node_count", 0), 0.0)))
        if isinstance(view_graph, dict)
        else 0
    )
    truth_node_count = (
        max(0, int(_safe_float(truth_graph.get("node_count", 0), 0.0)))
        if isinstance(truth_graph, dict)
        else 0
    )

    return total_count > 0 or view_node_count > 0 or truth_node_count > 0


def _ws_clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _simulation_ws_graph_node_position_map(
    node_positions: Any,
) -> dict[str, tuple[float, float]]:
    if not isinstance(node_positions, dict):
        return {}

    mapped: dict[str, tuple[float, float]] = {}
    limit = max(1, int(_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NODE_LIMIT))
    for node_id, row in node_positions.items():
        if len(mapped) >= limit:
            break
        node_key = str(node_id or "").strip()
        if not node_key or not isinstance(row, dict):
            continue
        mapped[node_key] = (
            _ws_clamp01(_safe_float(row.get("x", 0.5), 0.5)),
            _ws_clamp01(_safe_float(row.get("y", 0.5), 0.5)),
        )
    return mapped


def _simulation_ws_graph_variability_update(node_positions: Any) -> dict[str, Any]:
    current_positions = _simulation_ws_graph_node_position_map(node_positions)

    with _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_LOCK:
        previous_state = dict(_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_STATE)
        previous_positions_raw = previous_state.get("positions", {})
        previous_positions = (
            dict(previous_positions_raw)
            if isinstance(previous_positions_raw, dict)
            else {}
        )

        displacements: list[float] = []
        moved_count = 0
        moved_threshold = max(
            0.0005,
            _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_DISTANCE_REF * 0.35,
        )
        for node_id, (cx, cy) in current_positions.items():
            prior = previous_positions.get(node_id)
            if not isinstance(prior, tuple) or len(prior) < 2:
                continue
            px = _ws_clamp01(_safe_float(prior[0], cx))
            py = _ws_clamp01(_safe_float(prior[1], cy))
            displacement = math.hypot(cx - px, cy - py)
            displacements.append(displacement)
            if displacement >= moved_threshold:
                moved_count += 1

        shared_nodes = len(displacements)
        mean_displacement = (
            (sum(displacements) / float(shared_nodes)) if shared_nodes > 0 else 0.0
        )
        if len(displacements) >= 2:
            ordered = sorted(displacements)
            p90_index = max(
                0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.9)))
            )
            p90_displacement = ordered[p90_index]
        else:
            p90_displacement = mean_displacement
        active_share = (
            (float(moved_count) / float(shared_nodes)) if shared_nodes > 0 else 0.0
        )

        mean_term = _ws_clamp01(
            mean_displacement
            / max(1e-6, _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_DISTANCE_REF)
        )
        p90_term = _ws_clamp01(
            p90_displacement
            / max(1e-6, _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_DISTANCE_REF * 1.8)
        )
        active_term = _ws_clamp01(active_share / 0.45)
        raw_score = _ws_clamp01(
            (mean_term * 0.55) + (p90_term * 0.25) + (active_term * 0.2)
        )

        alpha = _ws_clamp01(
            _safe_float(_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_EMA_ALPHA, 0.2)
        )
        previous_score = _ws_clamp01(
            _safe_float(previous_state.get("score", raw_score), raw_score)
        )
        score = _ws_clamp01((previous_score * (1.0 - alpha)) + (raw_score * alpha))
        peak_score = max(
            score,
            _ws_clamp01(
                _safe_float(previous_state.get("peak_score", score), score) * 0.92
            ),
        )

        _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_STATE.clear()
        _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_STATE.update(
            {
                "positions": dict(current_positions),
                "score": score,
                "raw_score": raw_score,
                "peak_score": peak_score,
                "mean_displacement": mean_displacement,
                "p90_displacement": p90_displacement,
                "active_share": active_share,
                "shared_nodes": shared_nodes,
                "sampled_nodes": len(current_positions),
            }
        )

        return {
            "score": score,
            "raw_score": raw_score,
            "peak_score": peak_score,
            "mean_displacement": mean_displacement,
            "p90_displacement": p90_displacement,
            "active_share": active_share,
            "shared_nodes": shared_nodes,
            "sampled_nodes": len(current_positions),
        }


def _simulation_ws_daimoi_live_metrics(
    rows: Any,
    *,
    default_target: float,
) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        return {}

    positions_fn = getattr(
        daimoi_probabilistic_module,
        "_anti_clump_positions_from_particles",
        None,
    )
    metrics_fn = getattr(daimoi_probabilistic_module, "_anti_clump_metrics", None)
    if not callable(positions_fn) or not callable(metrics_fn):
        return {}

    try:
        positions = positions_fn(rows)
        try:
            metrics = metrics_fn(
                positions,
                previous_collision_count=0,
                particles=rows,
            )
        except TypeError:
            metrics = metrics_fn(positions, previous_collision_count=0)
    except Exception:
        return {}

    if not isinstance(metrics, dict):
        return {}

    clump_score = _ws_clamp01(_safe_float(metrics.get("clump_score", 0.0), 0.0))
    target = _ws_clamp01(_safe_float(default_target, 0.38))
    snr_valid = _safe_float(metrics.get("snr_valid", 0.0), 0.0) > 0.5
    snr_low_gap = max(0.0, _safe_float(metrics.get("snr_low_gap", 0.0), 0.0))
    snr_high_gap = max(0.0, _safe_float(metrics.get("snr_high_gap", 0.0), 0.0))
    if snr_valid:
        drive_estimate = max(-1.0, min(1.0, snr_low_gap + (snr_high_gap * 0.35)))
    else:
        drive_estimate = max(-1.0, min(1.0, (clump_score - target) * 2.2))

    return {
        "clump_score": clump_score,
        "drive_estimate": drive_estimate,
        "metrics": {
            "nn_term": _ws_clamp01(_safe_float(metrics.get("nn_term", 0.0), 0.0)),
            "entropy_norm": _ws_clamp01(
                _safe_float(metrics.get("entropy_norm", 1.0), 1.0)
            ),
            "hotspot_term": _ws_clamp01(
                _safe_float(metrics.get("hotspot_term", 0.0), 0.0)
            ),
            "collision_term": _ws_clamp01(
                _safe_float(metrics.get("collision_term", 0.0), 0.0)
            ),
            "collision_rate": max(
                0.0,
                _safe_float(metrics.get("collision_rate", 0.0), 0.0),
            ),
            "median_distance": max(
                0.0,
                _safe_float(metrics.get("median_distance", 0.0), 0.0),
            ),
            "target_distance": max(
                0.0,
                _safe_float(metrics.get("target_distance", 0.0), 0.0),
            ),
            "top_share": _ws_clamp01(_safe_float(metrics.get("top_share", 0.0), 0.0)),
            "mean_spacing": max(
                0.0,
                _safe_float(metrics.get("mean_spacing", 0.0), 0.0),
            ),
            "fano_factor": max(
                0.0,
                _safe_float(metrics.get("fano_factor", 0.0), 0.0),
            ),
            "fano_excess": max(
                0.0,
                _safe_float(metrics.get("fano_excess", 0.0), 0.0),
            ),
            "spatial_noise": max(
                0.0,
                _safe_float(metrics.get("spatial_noise", 0.0), 0.0),
            ),
            "motion_signal": max(
                0.0,
                _safe_float(metrics.get("motion_signal", 0.0), 0.0),
            ),
            "motion_noise": max(
                0.0,
                _safe_float(metrics.get("motion_noise", 0.0), 0.0),
            ),
            "motion_samples": max(
                0,
                int(_safe_float(metrics.get("motion_samples", 0), 0.0)),
            ),
            "semantic_noise": max(
                0.0,
                _safe_float(metrics.get("semantic_noise", 0.0), 0.0),
            ),
            "snr_signal": max(
                0.0,
                _safe_float(metrics.get("snr_signal", 0.0), 0.0),
            ),
            "snr_noise": max(
                0.0,
                _safe_float(metrics.get("snr_noise", 0.0), 0.0),
            ),
            "snr": max(0.0, _safe_float(metrics.get("snr", 0.0), 0.0)),
            "snr_valid": bool(snr_valid),
            "snr_low_gap": snr_low_gap,
            "snr_high_gap": snr_high_gap,
            "snr_min": max(
                0.05,
                _safe_float(metrics.get("snr_min", 0.85), 0.85),
            ),
            "snr_max": max(
                0.1,
                _safe_float(metrics.get("snr_max", 1.65), 1.65),
            ),
            "snr_in_band": bool(
                _safe_float(metrics.get("snr_in_band", 0.0), 0.0) > 0.5
            ),
        },
    }


def _simulation_ws_ensure_daimoi_summary(
    payload: dict[str, Any],
    *,
    include_live_metrics: bool = True,
    include_graph_variability: bool = True,
) -> None:
    if not isinstance(payload, dict):
        return
    dynamics = payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return

    summary_raw = dynamics.get("daimoi_probabilistic", {})
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}

    anti_raw = summary.get("anti_clump", {})
    anti = dict(anti_raw) if isinstance(anti_raw, dict) else {}

    metrics_raw = anti.get("metrics", {})
    metrics = dict(metrics_raw) if isinstance(metrics_raw, dict) else {}
    scales_raw = anti.get("scales", {})
    scales = dict(scales_raw) if isinstance(scales_raw, dict) else {}

    default_target = _safe_float(
        getattr(daimoi_probabilistic_module, "DAIMOI_ANTI_CLUMP_TARGET", 0.33),
        0.33,
    )
    anti_target = _ws_clamp01(
        _safe_float(anti.get("target", default_target), default_target)
    )

    rows = dynamics.get("field_particles", [])
    live_metrics: dict[str, Any] = {}
    if include_live_metrics:
        live_metrics = _simulation_ws_daimoi_live_metrics(
            rows,
            default_target=anti_target,
        )

    clump_score = _ws_clamp01(_safe_float(summary.get("clump_score", 0.0), 0.0))
    if isinstance(live_metrics, dict) and live_metrics:
        clump_score = _ws_clamp01(
            _safe_float(live_metrics.get("clump_score", clump_score), clump_score)
        )

    drive_default = max(-1.0, min(1.0, (clump_score - anti_target) * 2.2))
    anti_drive = drive_default
    if isinstance(live_metrics, dict) and live_metrics:
        anti_drive = max(
            -1.0,
            min(
                1.0,
                _safe_float(
                    live_metrics.get("drive_estimate", drive_default), drive_default
                ),
            ),
        )
    else:
        anti_drive = max(
            -1.0,
            min(
                1.0,
                _safe_float(
                    summary.get("anti_clump_drive", anti.get("drive", drive_default)),
                    drive_default,
                ),
            ),
        )

    graph_variability_raw = anti.get("graph_variability", {})
    graph_variability: dict[str, Any] = (
        dict(graph_variability_raw) if isinstance(graph_variability_raw, dict) else {}
    )
    if include_graph_variability:
        graph_variability = _simulation_ws_graph_variability_update(
            dynamics.get("graph_node_positions", {})
        )
    graph_score = _ws_clamp01(
        _safe_float(
            graph_variability.get("score", 0.0)
            if isinstance(graph_variability, dict)
            else 0.0,
            0.0,
        )
    )
    noise_gain = max(
        1.0,
        min(
            2.2,
            1.0 + (graph_score * _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NOISE_GAIN),
        ),
    )
    route_damp = max(
        0.55,
        min(
            1.0,
            1.0 - (graph_score * _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_ROUTE_DAMP),
        ),
    )

    anti["target"] = round(anti_target, 6)
    anti["drive"] = round(anti_drive, 6)
    anti["clump_score"] = round(clump_score, 6)
    live_metrics_map = (
        dict(live_metrics.get("metrics", {}))
        if isinstance(live_metrics, dict)
        and isinstance(live_metrics.get("metrics", {}), dict)
        else {}
    )
    anti["metrics"] = {
        "nn_term": max(
            0.0,
            _safe_float(
                live_metrics_map.get("nn_term", metrics.get("nn_term", 0.0)),
                0.0,
            ),
        ),
        "entropy_norm": max(
            0.0,
            _safe_float(
                live_metrics_map.get(
                    "entropy_norm",
                    metrics.get("entropy_norm", 1.0),
                ),
                1.0,
            ),
        ),
        "hotspot_term": max(
            0.0,
            _safe_float(
                live_metrics_map.get("hotspot_term", metrics.get("hotspot_term", 0.0)),
                0.0,
            ),
        ),
        "collision_term": max(
            0.0,
            _safe_float(
                live_metrics_map.get(
                    "collision_term",
                    metrics.get("collision_term", 0.0),
                ),
                0.0,
            ),
        ),
        "collision_rate": max(
            0.0,
            _safe_float(
                live_metrics_map.get(
                    "collision_rate",
                    metrics.get("collision_rate", 0.0),
                ),
                0.0,
            ),
        ),
        "median_distance": max(
            0.0,
            _safe_float(
                live_metrics_map.get(
                    "median_distance",
                    metrics.get("median_distance", 0.0),
                ),
                0.0,
            ),
        ),
        "target_distance": max(
            0.0,
            _safe_float(
                live_metrics_map.get(
                    "target_distance",
                    metrics.get("target_distance", 0.0),
                ),
                0.0,
            ),
        ),
        "top_share": _ws_clamp01(
            _safe_float(
                live_metrics_map.get("top_share", metrics.get("top_share", 0.0)),
                0.0,
            )
        ),
        "mean_spacing": max(
            0.0,
            _safe_float(
                live_metrics_map.get("mean_spacing", metrics.get("mean_spacing", 0.0)),
                0.0,
            ),
        ),
        "fano_factor": max(
            0.0,
            _safe_float(
                live_metrics_map.get("fano_factor", metrics.get("fano_factor", 0.0)),
                0.0,
            ),
        ),
        "fano_excess": max(
            0.0,
            _safe_float(
                live_metrics_map.get("fano_excess", metrics.get("fano_excess", 0.0)),
                0.0,
            ),
        ),
        "spatial_noise": max(
            0.0,
            _safe_float(
                live_metrics_map.get(
                    "spatial_noise",
                    metrics.get("spatial_noise", 0.0),
                ),
                0.0,
            ),
        ),
        "motion_signal": max(
            0.0,
            _safe_float(
                live_metrics_map.get(
                    "motion_signal",
                    metrics.get("motion_signal", 0.0),
                ),
                0.0,
            ),
        ),
        "motion_noise": max(
            0.0,
            _safe_float(
                live_metrics_map.get("motion_noise", metrics.get("motion_noise", 0.0)),
                0.0,
            ),
        ),
        "motion_samples": max(
            0,
            int(
                _safe_float(
                    live_metrics_map.get(
                        "motion_samples",
                        metrics.get("motion_samples", 0),
                    ),
                    0.0,
                )
            ),
        ),
        "semantic_noise": max(
            0.0,
            _safe_float(
                live_metrics_map.get(
                    "semantic_noise",
                    metrics.get("semantic_noise", 0.0),
                ),
                0.0,
            ),
        ),
        "snr_signal": max(
            0.0,
            _safe_float(
                live_metrics_map.get("snr_signal", metrics.get("snr_signal", 0.0)),
                0.0,
            ),
        ),
        "snr_noise": max(
            0.0,
            _safe_float(
                live_metrics_map.get("snr_noise", metrics.get("snr_noise", 0.0)),
                0.0,
            ),
        ),
        "snr": max(
            0.0,
            _safe_float(live_metrics_map.get("snr", metrics.get("snr", 0.0)), 0.0),
        ),
        "snr_valid": bool(
            _safe_float(
                live_metrics_map.get("snr_valid", metrics.get("snr_valid", 0.0)),
                0.0,
            )
            > 0.5
        ),
        "snr_low_gap": max(
            0.0,
            _safe_float(
                live_metrics_map.get("snr_low_gap", metrics.get("snr_low_gap", 0.0)),
                0.0,
            ),
        ),
        "snr_high_gap": max(
            0.0,
            _safe_float(
                live_metrics_map.get("snr_high_gap", metrics.get("snr_high_gap", 0.0)),
                0.0,
            ),
        ),
        "snr_min": max(
            0.05,
            _safe_float(
                live_metrics_map.get("snr_min", metrics.get("snr_min", 0.85)),
                0.85,
            ),
        ),
        "snr_max": max(
            0.1,
            _safe_float(
                live_metrics_map.get("snr_max", metrics.get("snr_max", 1.65)),
                1.65,
            ),
        ),
        "snr_in_band": bool(
            _safe_float(
                live_metrics_map.get("snr_in_band", metrics.get("snr_in_band", 0.0)),
                0.0,
            )
            > 0.5
        ),
    }
    anti["snr"] = round(
        max(0.0, _safe_float(anti["metrics"].get("snr", 0.0), 0.0)),
        6,
    )
    anti["snr_valid"] = bool(anti["metrics"].get("snr_valid", False))
    snr_band_min = max(0.05, _safe_float(anti["metrics"].get("snr_min", 0.85), 0.85))
    snr_band_max = max(
        snr_band_min + 0.05,
        _safe_float(anti["metrics"].get("snr_max", 1.65), 1.65),
    )
    anti["snr_band"] = {
        "min": round(snr_band_min, 6),
        "max": round(snr_band_max, 6),
        "low_gap": round(
            max(0.0, _safe_float(anti["metrics"].get("snr_low_gap", 0.0), 0.0)),
            6,
        ),
        "high_gap": round(
            max(0.0, _safe_float(anti["metrics"].get("snr_high_gap", 0.0), 0.0)),
            6,
        ),
        "in_band": bool(anti["metrics"].get("snr_in_band", False)),
    }
    anti["scales"] = {
        "spawn": max(0.0, _safe_float(scales.get("spawn", 1.0), 1.0)),
        "anchor": max(0.0, _safe_float(scales.get("anchor", 1.0), 1.0)),
        "semantic": max(0.0, _safe_float(scales.get("semantic", 1.0), 1.0)),
        "edge": max(0.0, _safe_float(scales.get("edge", 1.0), 1.0)),
        "tangent": max(0.0, _safe_float(scales.get("tangent", 1.0), 1.0)),
        "noise_gain": round(noise_gain, 6),
        "route_damp": round(route_damp, 6),
    }
    if isinstance(graph_variability, dict):
        anti["graph_variability"] = {
            "score": round(
                _ws_clamp01(_safe_float(graph_variability.get("score", 0.0), 0.0)), 6
            ),
            "raw_score": round(
                _ws_clamp01(_safe_float(graph_variability.get("raw_score", 0.0), 0.0)),
                6,
            ),
            "peak_score": round(
                _ws_clamp01(_safe_float(graph_variability.get("peak_score", 0.0), 0.0)),
                6,
            ),
            "mean_displacement": round(
                max(
                    0.0,
                    _safe_float(graph_variability.get("mean_displacement", 0.0), 0.0),
                ),
                6,
            ),
            "p90_displacement": round(
                max(
                    0.0,
                    _safe_float(graph_variability.get("p90_displacement", 0.0), 0.0),
                ),
                6,
            ),
            "active_share": round(
                _ws_clamp01(
                    _safe_float(graph_variability.get("active_share", 0.0), 0.0)
                ),
                6,
            ),
            "shared_nodes": int(
                max(0, _safe_float(graph_variability.get("shared_nodes", 0), 0.0))
            ),
            "sampled_nodes": int(
                max(0, _safe_float(graph_variability.get("sampled_nodes", 0), 0.0))
            ),
        }

    summary["clump_score"] = round(clump_score, 6)
    summary["anti_clump_drive"] = round(anti_drive, 6)
    summary["snr"] = round(max(0.0, _safe_float(anti.get("snr", 0.0), 0.0)), 6)
    summary["anti_clump"] = anti
    dynamics["daimoi_probabilistic"] = summary
    payload["presence_dynamics"] = dynamics


def _simulation_ws_payload_missing_daimoi_summary(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return True
    if "presence_dynamics" not in payload:
        return True
    dynamics = payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict) or not dynamics:
        return True
    _simulation_ws_ensure_daimoi_summary(payload)
    dynamics = payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return True
    summary = dynamics.get("daimoi_probabilistic", {})
    if not isinstance(summary, dict) or not summary:
        return True
    if "clump_score" not in summary and "anti_clump" not in summary:
        return True
    return False


def _simulation_bootstrap_store_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        return
    with _SIMULATION_BOOTSTRAP_REPORT_LOCK:
        _SIMULATION_BOOTSTRAP_LAST_REPORT.clear()
        _SIMULATION_BOOTSTRAP_LAST_REPORT.update(dict(report))


def _simulation_bootstrap_snapshot_report() -> dict[str, Any] | None:
    with _SIMULATION_BOOTSTRAP_REPORT_LOCK:
        if not isinstance(_SIMULATION_BOOTSTRAP_LAST_REPORT, dict):
            return None
        if not _SIMULATION_BOOTSTRAP_LAST_REPORT:
            return None
        return dict(_SIMULATION_BOOTSTRAP_LAST_REPORT)


def _simulation_bootstrap_job_snapshot() -> dict[str, Any]:
    with _SIMULATION_BOOTSTRAP_JOB_LOCK:
        snapshot = dict(_SIMULATION_BOOTSTRAP_JOB)
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


def _simulation_bootstrap_job_start(
    *,
    request_payload: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    request_row = dict(request_payload) if isinstance(request_payload, dict) else {}
    with _SIMULATION_BOOTSTRAP_JOB_LOCK:
        if (
            str(_SIMULATION_BOOTSTRAP_JOB.get("status", "")).strip().lower()
            == "running"
        ):
            snapshot = dict(_SIMULATION_BOOTSTRAP_JOB)
            request_snapshot = snapshot.get("request", {})
            snapshot["request"] = (
                dict(request_snapshot) if isinstance(request_snapshot, dict) else {}
            )
            report_snapshot = snapshot.get("report")
            snapshot["report"] = (
                dict(report_snapshot) if isinstance(report_snapshot, dict) else None
            )
            return False, snapshot

        seed = (
            f"{now_iso}|{request_row.get('perspective', '')}|"
            f"{request_row.get('sync_inbox', False)}"
        )
        job_id = "bootstrap:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
        _SIMULATION_BOOTSTRAP_JOB["status"] = "running"
        _SIMULATION_BOOTSTRAP_JOB["job_id"] = job_id
        _SIMULATION_BOOTSTRAP_JOB["started_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["updated_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["completed_at"] = ""
        _SIMULATION_BOOTSTRAP_JOB["phase"] = "queued"
        _SIMULATION_BOOTSTRAP_JOB["phase_started_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["phase_detail"] = {}
        _SIMULATION_BOOTSTRAP_JOB["error"] = ""
        _SIMULATION_BOOTSTRAP_JOB["request"] = request_row
        _SIMULATION_BOOTSTRAP_JOB["report"] = None
        snapshot = dict(_SIMULATION_BOOTSTRAP_JOB)
        snapshot["request"] = dict(request_row)
        snapshot["phase_detail"] = {}
        snapshot["report"] = None
        return True, snapshot


def _simulation_bootstrap_job_mark_phase(
    *,
    job_id: str,
    phase: str,
    detail: dict[str, Any] | None = None,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with _SIMULATION_BOOTSTRAP_JOB_LOCK:
        if str(_SIMULATION_BOOTSTRAP_JOB.get("job_id", "")) != str(job_id):
            return
        next_phase = str(phase or "").strip().lower()
        if not next_phase:
            return
        current_phase = str(_SIMULATION_BOOTSTRAP_JOB.get("phase", "")).strip().lower()
        _SIMULATION_BOOTSTRAP_JOB["phase"] = next_phase
        if next_phase != current_phase:
            _SIMULATION_BOOTSTRAP_JOB["phase_started_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["updated_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["phase_detail"] = (
            dict(detail) if isinstance(detail, dict) else {}
        )


def _simulation_bootstrap_job_complete(
    *,
    job_id: str,
    report: dict[str, Any],
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with _SIMULATION_BOOTSTRAP_JOB_LOCK:
        if str(_SIMULATION_BOOTSTRAP_JOB.get("job_id", "")) != str(job_id):
            return
        _SIMULATION_BOOTSTRAP_JOB["status"] = "completed"
        _SIMULATION_BOOTSTRAP_JOB["phase"] = "completed"
        _SIMULATION_BOOTSTRAP_JOB["phase_started_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["phase_detail"] = {}
        _SIMULATION_BOOTSTRAP_JOB["updated_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["completed_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["error"] = ""
        _SIMULATION_BOOTSTRAP_JOB["report"] = dict(report)


def _simulation_bootstrap_job_fail(
    *,
    job_id: str,
    error: str,
    report: dict[str, Any] | None = None,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    with _SIMULATION_BOOTSTRAP_JOB_LOCK:
        if str(_SIMULATION_BOOTSTRAP_JOB.get("job_id", "")) != str(job_id):
            return
        _SIMULATION_BOOTSTRAP_JOB["status"] = "failed"
        _SIMULATION_BOOTSTRAP_JOB["phase"] = "failed"
        _SIMULATION_BOOTSTRAP_JOB["phase_started_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["phase_detail"] = {}
        _SIMULATION_BOOTSTRAP_JOB["updated_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["completed_at"] = now_iso
        _SIMULATION_BOOTSTRAP_JOB["error"] = str(error or "simulation_bootstrap_failed")
        _SIMULATION_BOOTSTRAP_JOB["report"] = (
            dict(report) if isinstance(report, dict) else None
        )


def _simulation_bootstrap_embed_layer_row(layer: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(layer.get("id", "") or "").strip(),
        "label": str(layer.get("label", "") or "").strip(),
        "collection": str(layer.get("collection", "") or "").strip(),
        "space_id": str(layer.get("space_id", "") or "").strip(),
        "model_name": str(layer.get("model_name", "") or "").strip(),
        "file_count": max(0, int(_safe_float(layer.get("file_count", 0), 0.0))),
        "reference_count": max(
            0, int(_safe_float(layer.get("reference_count", 0), 0.0))
        ),
        "active": bool(layer.get("active", False)),
    }


def _simulation_bootstrap_normalize_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _simulation_bootstrap_file_path(row: dict[str, Any]) -> str:
    for key in (
        "source_rel_path",
        "archive_rel_path",
        "archived_rel_path",
        "archive_member_path",
        "name",
        "label",
        "node_id",
        "id",
    ):
        text = _simulation_bootstrap_normalize_path(row.get(key, ""))
        if text:
            return text
    return ""


def _simulation_bootstrap_file_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id", "") or "").strip(),
        "node_id": str(row.get("node_id", "") or "").strip(),
        "name": str(row.get("name", "") or "").strip(),
        "kind": str(row.get("kind", "") or "").strip(),
        "path": _simulation_bootstrap_file_path(row),
        "source_rel_path": _simulation_bootstrap_normalize_path(
            row.get("source_rel_path", "")
        ),
        "archive_rel_path": _simulation_bootstrap_normalize_path(
            row.get("archive_rel_path", "")
        ),
        "archived_rel_path": _simulation_bootstrap_normalize_path(
            row.get("archived_rel_path", "")
        ),
        "projection_overflow": bool(row.get("projection_overflow", False)),
        "consolidated": bool(row.get("consolidated", False)),
        "consolidated_count": max(
            0, int(_safe_float(row.get("consolidated_count", 0), 0.0))
        ),
        "projection_group_id": str(row.get("projection_group_id", "") or "").strip(),
    }


def _simulation_bootstrap_graph_diff(
    *,
    catalog: dict[str, Any],
    simulation: dict[str, Any],
) -> dict[str, Any]:
    catalog_file_graph = (
        catalog.get("file_graph", {})
        if isinstance(catalog, dict) and isinstance(catalog.get("file_graph", {}), dict)
        else {}
    )
    simulation_file_graph = (
        simulation.get("file_graph", {})
        if isinstance(simulation, dict)
        and isinstance(simulation.get("file_graph", {}), dict)
        else {}
    )

    true_file_nodes = [
        row
        for row in (
            catalog_file_graph.get("file_nodes", [])
            if isinstance(catalog_file_graph.get("file_nodes", []), list)
            else []
        )
        if isinstance(row, dict)
    ]
    view_file_nodes = [
        row
        for row in (
            simulation_file_graph.get("file_nodes", [])
            if isinstance(simulation_file_graph.get("file_nodes", []), list)
            else []
        )
        if isinstance(row, dict)
    ]

    true_by_id: dict[str, dict[str, Any]] = {}
    view_ids: set[str] = set()
    for row in true_file_nodes:
        row_id = str(row.get("id", "") or "").strip()
        if row_id and row_id not in true_by_id:
            true_by_id[row_id] = row
    for row in view_file_nodes:
        row_id = str(row.get("id", "") or "").strip()
        if row_id:
            view_ids.add(row_id)

    projection_payload = (
        simulation_file_graph.get("projection", {})
        if isinstance(simulation_file_graph.get("projection", {}), dict)
        else {}
    )
    groups = [
        row
        for row in (
            projection_payload.get("groups", [])
            if isinstance(projection_payload.get("groups", []), list)
            else []
        )
        if isinstance(row, dict)
    ]

    grouped_sources: dict[str, list[dict[str, Any]]] = {}
    surface_visible_group_count = 0
    for group in groups:
        group_id = str(group.get("id", "") or "").strip()
        if not group_id:
            continue
        surface_visible = bool(group.get("surface_visible", False))
        if surface_visible:
            surface_visible_group_count += 1
        reasons_payload = (
            group.get("reasons", {})
            if isinstance(group.get("reasons", {}), dict)
            else {}
        )
        reason_rows = {
            str(key): int(_safe_float(value, 0.0))
            for key, value in reasons_payload.items()
            if str(key).strip()
        }
        refs = {
            "group_id": group_id,
            "kind": str(group.get("kind", "") or "").strip(),
            "target": str(group.get("target", "") or "").strip(),
            "surface_visible": surface_visible,
            "reasons": reason_rows,
        }
        member_source_ids = (
            group.get("member_source_ids", [])
            if isinstance(group.get("member_source_ids", []), list)
            else []
        )
        for source_id in member_source_ids:
            clean_source_id = str(source_id or "").strip()
            if not clean_source_id:
                continue
            rows = grouped_sources.setdefault(clean_source_id, [])
            if len(rows) < 4:
                rows.append(dict(refs))

    overflow_rows = [
        _simulation_bootstrap_file_row(row)
        for row in view_file_nodes
        if bool(row.get("projection_overflow", False))
        or bool(row.get("consolidated", False))
        or str(row.get("kind", "") or "").strip().lower() == "projection_overflow"
    ]
    overflow_rows.sort(
        key=lambda row: (
            -max(0, int(_safe_float(row.get("consolidated_count", 0), 0.0))),
            str(row.get("name", "")),
            str(row.get("id", "")),
        )
    )

    missing_ids = sorted({*true_by_id.keys()} - view_ids)
    missing_rows: list[dict[str, Any]] = []
    for node_id in missing_ids:
        source_row = true_by_id.get(node_id, {})
        group_refs = grouped_sources.get(node_id, [])
        reason = "trimmed_before_projection"
        if group_refs:
            if any(bool(row.get("surface_visible", False)) for row in group_refs):
                reason = "grouped_in_projection_bundle"
            else:
                reason = "grouped_in_hidden_projection_bundle"
        row_payload = _simulation_bootstrap_file_row(source_row)
        row_payload["reason"] = reason
        row_payload["projection_group_refs"] = [
            {
                "group_id": str(row.get("group_id", "") or ""),
                "kind": str(row.get("kind", "") or ""),
                "target": str(row.get("target", "") or ""),
                "surface_visible": bool(row.get("surface_visible", False)),
                "reasons": (
                    dict(row.get("reasons", {}))
                    if isinstance(row.get("reasons", {}), dict)
                    else {}
                ),
            }
            for row in group_refs
        ]
        missing_rows.append(row_payload)

    missing_rows.sort(
        key=lambda row: (
            str(row.get("path", "")),
            str(row.get("name", "")),
            str(row.get("id", "")),
        )
    )

    item_rows: dict[str, dict[str, Any]] = {}
    catalog_items = (
        catalog.get("items", [])
        if isinstance(catalog, dict) and isinstance(catalog.get("items", []), list)
        else []
    )
    for item in catalog_items:
        if not isinstance(item, dict):
            continue
        rel_path = _simulation_bootstrap_normalize_path(item.get("rel_path", ""))
        if not rel_path:
            continue
        if rel_path not in item_rows:
            item_rows[rel_path] = {
                "path": rel_path,
                "kind": str(item.get("kind", "") or "").strip(),
                "name": str(item.get("name", "") or "").strip(),
                "role": str(item.get("role", "") or "").strip(),
            }

    true_paths = {
        _simulation_bootstrap_file_path(row)
        for row in true_file_nodes
        if _simulation_bootstrap_file_path(row)
    }
    missing_item_rows = [
        {
            **item_rows[path],
            "reason": "ingested_item_not_present_in_true_graph",
        }
        for path in sorted(item_rows)
        if path not in true_paths
    ]

    max_rows = _SIMULATION_BOOTSTRAP_MAX_EXCLUDED_FILES
    collapsed_edges = max(
        0, int(_safe_float(projection_payload.get("collapsed_edges", 0), 0.0))
    )
    compaction_mode = "identity_or_within_limits"
    if collapsed_edges > 0 and overflow_rows:
        compaction_mode = "compacted_with_projection_overflow"
    elif collapsed_edges > 0:
        compaction_mode = "pruned_without_overflow_nodes"
    elif missing_rows:
        compaction_mode = "trimmed_before_projection"

    return {
        "truth_file_node_count": len(true_file_nodes),
        "view_file_node_count": len(view_file_nodes),
        "truth_file_nodes_missing_from_view_count": len(missing_rows),
        "truth_file_nodes_missing_from_view": missing_rows[:max_rows],
        "truth_file_nodes_missing_from_view_truncated": len(missing_rows) > max_rows,
        "view_projection_overflow_node_count": len(overflow_rows),
        "view_projection_overflow_nodes": overflow_rows[:max_rows],
        "view_projection_overflow_nodes_truncated": len(overflow_rows) > max_rows,
        "projection_group_count": len(groups),
        "projection_surface_visible_group_count": surface_visible_group_count,
        "projection_hidden_group_count": max(
            0,
            len(groups) - surface_visible_group_count,
        ),
        "projection_group_member_source_count": len(grouped_sources),
        "ingested_item_count": len(item_rows),
        "ingested_items_missing_from_truth_graph_count": len(missing_item_rows),
        "ingested_items_missing_from_truth_graph": missing_item_rows[:max_rows],
        "ingested_items_missing_from_truth_graph_truncated": len(missing_item_rows)
        > max_rows,
        "compaction_mode": compaction_mode,
        "view_graph_reconstructable_from_truth_graph": True,
        "notes": [
            "truth graph remains canonical; view graph is derived by projection rules",
            "projection bundles preserve edge-member lineage for reconstructability",
        ],
    }


def _simulation_bootstrap_graph_report(
    *,
    perspective: str,
    catalog: dict[str, Any],
    simulation: dict[str, Any],
    projection: dict[str, Any],
    phase_ms: dict[str, float] | None = None,
    reset_summary: dict[str, Any] | None = None,
    inbox_sync: dict[str, Any] | None = None,
    cache_key: str = "",
) -> dict[str, Any]:
    catalog_file_graph = (
        catalog.get("file_graph", {})
        if isinstance(catalog, dict) and isinstance(catalog.get("file_graph", {}), dict)
        else {}
    )
    simulation_file_graph = (
        simulation.get("file_graph", {})
        if isinstance(simulation, dict)
        and isinstance(simulation.get("file_graph", {}), dict)
        else {}
    )

    embed_layers_raw = simulation_file_graph.get("embed_layers", [])
    if not isinstance(embed_layers_raw, list) or not embed_layers_raw:
        embed_layers_raw = catalog_file_graph.get("embed_layers", [])
    embed_layers = [row for row in embed_layers_raw if isinstance(row, dict)]
    active_layers = [row for row in embed_layers if bool(row.get("active", False))]
    selected_layers = active_layers if active_layers else embed_layers

    projection_payload = simulation_file_graph.get("projection", {})
    if not isinstance(projection_payload, dict):
        projection_payload = {}

    projection_before = (
        projection_payload.get("before", {})
        if isinstance(projection_payload.get("before", {}), dict)
        else {}
    )
    projection_after = (
        projection_payload.get("after", {})
        if isinstance(projection_payload.get("after", {}), dict)
        else {}
    )
    projection_limits = (
        projection_payload.get("limits", {})
        if isinstance(projection_payload.get("limits", {}), dict)
        else {}
    )

    before_edges = max(
        0,
        int(
            _safe_float(
                projection_before.get(
                    "edges",
                    len(catalog_file_graph.get("edges", []))
                    if isinstance(catalog_file_graph.get("edges", []), list)
                    else 0,
                ),
                0.0,
            )
        ),
    )
    after_edges = max(
        0,
        int(
            _safe_float(
                projection_after.get(
                    "edges",
                    len(simulation_file_graph.get("edges", []))
                    if isinstance(simulation_file_graph.get("edges", []), list)
                    else before_edges,
                ),
                0.0,
            )
        ),
    )
    before_file_nodes = max(
        0,
        int(
            _safe_float(
                projection_before.get(
                    "file_nodes",
                    len(catalog_file_graph.get("file_nodes", []))
                    if isinstance(catalog_file_graph.get("file_nodes", []), list)
                    else 0,
                ),
                0.0,
            )
        ),
    )
    after_file_nodes = max(
        0,
        int(
            _safe_float(
                projection_after.get(
                    "file_nodes",
                    len(simulation_file_graph.get("file_nodes", []))
                    if isinstance(simulation_file_graph.get("file_nodes", []), list)
                    else before_file_nodes,
                ),
                0.0,
            )
        ),
    )

    collapsed_edges = max(
        0,
        int(
            _safe_float(
                projection_payload.get(
                    "collapsed_edges",
                    max(0, before_edges - after_edges),
                ),
                0.0,
            )
        ),
    )
    overflow_nodes = max(
        0, int(_safe_float(projection_payload.get("overflow_nodes", 0), 0.0))
    )
    overflow_edges = max(
        0, int(_safe_float(projection_payload.get("overflow_edges", 0), 0.0))
    )
    group_count = max(
        0, int(_safe_float(projection_payload.get("group_count", 0), 0.0))
    )
    edge_cap = max(0, int(_safe_float(projection_limits.get("edge_cap", 0), 0.0)))

    edge_reduction_ratio = 0.0
    if before_edges > 0:
        edge_reduction_ratio = max(
            0.0,
            min(1.0, collapsed_edges / float(max(1, before_edges))),
        )

    edge_cap_utilization = 0.0
    if edge_cap > 0:
        edge_cap_utilization = max(
            0.0,
            min(2.0, after_edges / float(max(1, edge_cap))),
        )

    presence_dynamics = (
        simulation.get("presence_dynamics", {})
        if isinstance(simulation, dict)
        and isinstance(simulation.get("presence_dynamics", {}), dict)
        else {}
    )
    field_particles = presence_dynamics.get("field_particles", [])

    report: dict[str, Any] = {
        "ok": True,
        "record": "eta-mu.simulation-bootstrap.v1",
        "schema_version": "simulation.bootstrap.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "perspective": normalize_projection_perspective(perspective),
        "runtime_config_version": _config_runtime_version_snapshot(),
        "cache_key": str(cache_key or "").strip(),
        "selection": {
            "graph_surface": (
                "projected-hub-overflow"
                if bool(projection_payload.get("active", False))
                else "full-file-graph"
            ),
            "projection_mode": str(projection_payload.get("mode", "hub-overflow")),
            "projection_reason": str(projection_payload.get("reason", "")),
            "embed_layer_count": len(embed_layers),
            "active_embed_layer_count": len(active_layers),
            "selected_embed_layers": [
                _simulation_bootstrap_embed_layer_row(row)
                for row in selected_layers[:16]
            ],
        },
        "compression": {
            "before_edges": before_edges,
            "after_edges": after_edges,
            "collapsed_edges": collapsed_edges,
            "edge_reduction_ratio": round(edge_reduction_ratio, 6),
            "edge_cap": edge_cap,
            "edge_cap_utilization": round(edge_cap_utilization, 6),
            "before_file_nodes": before_file_nodes,
            "after_file_nodes": after_file_nodes,
            "overflow_nodes": overflow_nodes,
            "overflow_edges": overflow_edges,
            "group_count": group_count,
            "active": bool(projection_payload.get("active", False)),
            "limits": {
                str(key): value
                for key, value in projection_limits.items()
                if str(key).strip()
            },
        },
        "graph_counts": {
            "catalog": {
                "file_nodes": len(catalog_file_graph.get("file_nodes", []))
                if isinstance(catalog_file_graph.get("file_nodes", []), list)
                else 0,
                "edges": len(catalog_file_graph.get("edges", []))
                if isinstance(catalog_file_graph.get("edges", []), list)
                else 0,
            },
            "simulation": {
                "file_nodes": len(simulation_file_graph.get("file_nodes", []))
                if isinstance(simulation_file_graph.get("file_nodes", []), list)
                else 0,
                "edges": len(simulation_file_graph.get("edges", []))
                if isinstance(simulation_file_graph.get("edges", []), list)
                else 0,
            },
        },
        "graph_diff": _simulation_bootstrap_graph_diff(
            catalog=catalog,
            simulation=simulation,
        ),
        "simulation_counts": {
            "total_points": max(0, int(_safe_float(simulation.get("total", 0), 0.0))),
            "point_rows": len(simulation.get("points", []))
            if isinstance(simulation.get("points", []), list)
            else 0,
            "embedding_particles": len(simulation.get("embedding_particles", []))
            if isinstance(simulation.get("embedding_particles", []), list)
            else 0,
            "field_particles": len(field_particles)
            if isinstance(field_particles, list)
            else 0,
        },
    }

    if isinstance(phase_ms, dict) and phase_ms:
        report["phase_ms"] = {
            str(key): round(max(0.0, _safe_float(value, 0.0)), 3)
            for key, value in phase_ms.items()
            if str(key).strip()
        }
    if isinstance(reset_summary, dict) and reset_summary:
        report["reset"] = dict(reset_summary)
    if isinstance(inbox_sync, dict) and inbox_sync:
        report["inbox_sync"] = dict(inbox_sync)
    if isinstance(projection, dict) and projection:
        report["projection"] = {
            "record": str(projection.get("record", "") or "").strip(),
            "perspective": str(projection.get("perspective", "") or "").strip(),
            "ts": int(_safe_float(projection.get("ts", 0), 0.0)),
        }
    return report


def _simulation_ws_load_cached_payload(
    *,
    part_root: Path,
    perspective: str,
    payload_mode: str = "trimmed",
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    payload_mode_key = _simulation_ws_normalize_payload_mode(payload_mode)
    loaded_from_compact_cache = False

    def _load_stale_disk_payload() -> dict[str, Any] | None:
        stale_cache_path = _simulation_http_disk_cache_path(part_root, perspective)
        try:
            if stale_cache_path.exists() and stale_cache_path.is_file():
                stale_body = stale_cache_path.read_bytes()
                return _simulation_ws_decode_cached_payload(stale_body)
        except Exception:
            return None
        return None

    payload: dict[str, Any] | None = None
    if payload_mode_key != "full":
        compact_cached_body = _simulation_http_compact_cached_body(
            perspective=perspective,
            max_age_seconds=_SIMULATION_WS_CACHE_MAX_AGE_SECONDS,
        )
        payload = _simulation_ws_decode_cached_payload(compact_cached_body)
        loaded_from_compact_cache = isinstance(payload, dict)

    if payload is None:
        cached_body = _simulation_http_cached_body(
            perspective=perspective,
            max_age_seconds=_SIMULATION_WS_CACHE_MAX_AGE_SECONDS,
        )
        if cached_body is None:
            cached_body = _simulation_http_disk_cache_load(
                part_root,
                perspective=perspective,
                max_age_seconds=_SIMULATION_WS_CACHE_MAX_AGE_SECONDS,
            )

        payload = _simulation_ws_decode_cached_payload(cached_body)
    if payload_mode_key == "full":
        stale_payload = _load_stale_disk_payload()
        if payload is None:
            payload = stale_payload
        elif (
            _simulation_ws_payload_is_sparse(payload)
            and isinstance(stale_payload, dict)
            and not _simulation_ws_payload_is_sparse(stale_payload)
        ):
            payload = stale_payload

    if payload is None:
        return None

    projection = payload.get("projection", {})
    if not isinstance(projection, dict):
        projection = {}
    if payload_mode_key == "full":
        simulation_payload = dict(payload)
        simulation_payload.pop("projection", None)
        return simulation_payload, projection

    node_positions, node_text_chars = _simulation_ws_collect_node_positions(payload)
    simulation_payload = _simulation_ws_trim_simulation_payload(payload)
    simulation_payload.pop("projection", None)
    simulation_payload.update(
        _simulation_ws_compact_graph_payload(
            payload,
            assume_trimmed=loaded_from_compact_cache,
        )
    )
    _simulation_ws_extract_stream_particles(
        simulation_payload,
        node_positions=node_positions,
        node_text_chars=node_text_chars,
    )
    dynamics = simulation_payload.get("presence_dynamics", {})
    if isinstance(dynamics, dict) and node_positions:
        graph_node_positions: dict[str, dict[str, float]] = {}
        for node_id, coords in node_positions.items():
            if len(graph_node_positions) >= 2200:
                break
            if not (isinstance(coords, tuple) and len(coords) >= 2):
                continue
            graph_node_positions[node_id] = {
                "x": round(max(0.0, min(1.0, _safe_float(coords[0], 0.5))), 5),
                "y": round(max(0.0, min(1.0, _safe_float(coords[1], 0.5))), 5),
            }
        dynamics["graph_node_positions"] = graph_node_positions
        simulation_payload["presence_dynamics"] = dynamics
    return simulation_payload, projection


def _simulation_ws_normalize_delta_stream_mode(mode: str) -> str:
    clean = str(mode or "").strip().lower()
    if clean in {"worker", "workers", "thread", "threads", "subsystem", "subsystems"}:
        return "workers"
    if clean in {"world", "single", "legacy", "combined"}:
        return "world"
    if _SIMULATION_WS_DELTA_STREAM_MODE in {
        "worker",
        "workers",
        "thread",
        "threads",
        "subsystem",
        "subsystems",
    }:
        return "workers"
    return "world"


def _simulation_ws_normalize_payload_mode(mode: str) -> str:
    clean = str(mode or "").strip().lower()
    if clean in {"full", "complete", "debug", "debug-full"}:
        return "full"
    return "trimmed"


def _simulation_ws_normalize_particle_payload_mode(mode: str) -> str:
    clean = str(mode or "").strip().lower()
    if clean in {"full", "rich", "complete", "debug"}:
        return "full"
    return "lite"


def _simulation_ws_lite_field_particles(
    rows: Any,
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    row_limit = len(rows) if max_rows is None else max(0, int(max_rows))
    if row_limit <= 0:
        return []
    compact_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index >= row_limit:
            break
        if not isinstance(row, dict):
            continue
        compact: dict[str, Any] = {}
        for key in _SIMULATION_WS_PARTICLE_LITE_KEYS:
            if key in row:
                compact[key] = row.get(key)
        if not str(compact.get("id", "") or "").strip():
            compact["id"] = str(row.get("id", "") or f"ws:{index}")
        compact_rows.append(compact)
    return compact_rows


def _simulation_ws_governor_estimate_work(simulation_payload: dict[str, Any]) -> float:
    if not isinstance(simulation_payload, dict):
        return 1.0

    dynamics = simulation_payload.get("presence_dynamics", {})
    field_particles = (
        dynamics.get("field_particles", []) if isinstance(dynamics, dict) else []
    )
    particle_count = len(field_particles) if isinstance(field_particles, list) else 0

    points = simulation_payload.get("points", [])
    point_count = len(points) if isinstance(points, list) else 0
    total_count = max(0.0, _safe_float(simulation_payload.get("total", 0.0), 0.0))

    estimated = (particle_count * 1.2) + (point_count * 0.08) + (total_count * 0.05)
    return max(1.0, estimated)


def _simulation_ws_governor_ingestion_signal(
    catalog: dict[str, Any],
) -> tuple[int, int, int, int]:
    if not isinstance(catalog, dict):
        return (0, 0, 0, 0)

    inbox_state = catalog.get("eta_mu_inbox", {})
    if not isinstance(inbox_state, dict):
        inbox_state = {}

    pending_count = max(0, int(_safe_float(inbox_state.get("pending_count", 0), 0.0)))
    deferred_count = max(0, int(_safe_float(inbox_state.get("deferred_count", 0), 0.0)))

    file_graph_stats = (
        catalog.get("file_graph", {}).get("stats", {})
        if isinstance(catalog.get("file_graph", {}), dict)
        else {}
    )
    if not isinstance(file_graph_stats, dict):
        file_graph_stats = {}

    compressed_total = max(
        0.0,
        _safe_float(file_graph_stats.get("compressed_bytes_total", 0.0), 0.0),
    )
    knowledge_entries = max(
        0.0,
        _safe_float(inbox_state.get("knowledge_entries", 0.0), 0.0),
    )
    avg_entry_bytes = (
        compressed_total / knowledge_entries
        if knowledge_entries > 0.0
        else 64.0 * 1024.0
    )
    avg_entry_bytes = max(16.0 * 1024.0, min(6.0 * 1024.0 * 1024.0, avg_entry_bytes))
    bytes_pending = int(max(0.0, pending_count * avg_entry_bytes))

    embedding_backlog = pending_count + deferred_count
    disk_queue_depth = deferred_count
    return (pending_count, bytes_pending, embedding_backlog, disk_queue_depth)


def _simulation_ws_governor_stock_pressure(part_root: Path) -> tuple[float, float]:
    mem_pressure = 0.0
    try:
        resource_snapshot = _resource_monitor_snapshot(part_root)
    except Exception:
        resource_snapshot = {}

    if isinstance(resource_snapshot, dict):
        devices = resource_snapshot.get("devices", {})
        if isinstance(devices, dict):
            cpu = devices.get("cpu", {})
            if isinstance(cpu, dict):
                mem_pressure = _safe_float(cpu.get("memory_pressure", 0.0), 0.0)

    disk_pressure = 0.0
    try:
        usage = shutil.disk_usage(part_root)
    except OSError:
        usage = None
    if usage is not None and usage.total > 0:
        disk_pressure = usage.used / float(usage.total)

    return (
        max(0.0, min(1.5, mem_pressure)),
        max(0.0, min(1.5, disk_pressure)),
    )


def _simulation_ws_governor_particle_cap(
    base_cap: int,
    *,
    fidelity_signal: str,
    ingestion_pressure: float,
) -> int:
    min_cap = max(1, _SIMULATION_WS_GOVERNOR_MIN_PARTICLE_CAP)
    max_cap = max(min_cap, int(base_cap))
    pressure = max(0.0, min(1.0, _safe_float(ingestion_pressure, 0.0)))

    if fidelity_signal == "decrease":
        scaled = int(round(max_cap * (0.68 - (0.24 * pressure))))
        return max(min_cap, min(max_cap, scaled))
    if fidelity_signal == "increase":
        scaled = int(round(max_cap * (1.0 + (0.12 * (1.0 - pressure)))))
        return max(min_cap, min(max_cap, scaled))
    return max(min_cap, max_cap)


def _simulation_ws_governor_graph_heartbeat_scale(fidelity_signal: str) -> float:
    if fidelity_signal == "decrease":
        return _SIMULATION_WS_GOVERNOR_DEGRADE_GRAPH_HEARTBEAT_SCALE
    if fidelity_signal == "increase":
        return _SIMULATION_WS_GOVERNOR_INCREASE_GRAPH_HEARTBEAT_SCALE
    return 1.0


def _catalog_stream_chunk_rows(value: Any) -> int:
    return max(
        1,
        min(
            2048,
            int(_safe_float(value, float(_CATALOG_STREAM_CHUNK_ROWS))),
        ),
    )


def _catalog_stream_get_path_value(
    payload: dict[str, Any], path: tuple[str, ...]
) -> Any:
    cursor: Any = payload
    for part in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor


def _catalog_stream_set_path_value(
    payload: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    if not path:
        return
    cursor: dict[str, Any] = payload
    for part in path[:-1]:
        nested = cursor.get(part)
        if not isinstance(nested, dict):
            nested = {}
            cursor[part] = nested
        cursor = nested
    cursor[path[-1]] = value


def _catalog_stream_meta(catalog: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(catalog, dict):
        return {}
    meta = copy.deepcopy(catalog)
    for section_name, path in _CATALOG_STREAM_SECTION_PATHS:
        rows = _catalog_stream_get_path_value(catalog, path)
        if isinstance(rows, list):
            _catalog_stream_set_path_value(
                meta,
                path,
                {
                    "streamed": True,
                    "section": section_name,
                    "count": len(rows),
                },
            )
    return meta


def _catalog_stream_iter_rows(
    catalog: dict[str, Any],
    *,
    chunk_rows: int,
) -> Iterator[dict[str, Any]]:
    chunk_size = _catalog_stream_chunk_rows(chunk_rows)
    catalog_payload = catalog if isinstance(catalog, dict) else {}
    section_stats: dict[str, dict[str, int]] = {}
    yield {
        "type": "meta",
        "record": "eta-mu.catalog.stream.meta.v1",
        "schema_version": "catalog.stream.meta.v1",
        "catalog": _catalog_stream_meta(catalog_payload),
    }

    for section_name, path in _CATALOG_STREAM_SECTION_PATHS:
        rows = _catalog_stream_get_path_value(catalog_payload, path)
        if not isinstance(rows, list):
            continue
        total = len(rows)
        chunk_count = 0
        for offset in range(0, total, chunk_size):
            chunk = rows[offset : offset + chunk_size]
            yield {
                "type": "rows",
                "record": "eta-mu.catalog.stream.rows.v1",
                "schema_version": "catalog.stream.rows.v1",
                "section": section_name,
                "offset": offset,
                "rows": chunk,
            }
            chunk_count += 1
        section_stats[section_name] = {
            "total": total,
            "chunks": chunk_count,
        }

    yield {
        "type": "done",
        "ok": True,
        "record": "eta-mu.catalog.stream.done.v1",
        "schema_version": "catalog.stream.done.v1",
        "chunk_rows": chunk_size,
        "sections": section_stats,
    }


def _simulation_ws_chunk_plan(
    payload: dict[str, Any],
    *,
    chunk_chars: int,
    message_seq: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(payload, dict):
        return [], None
    payload_type = str(payload.get("type", "") or "").strip().lower()
    if not payload_type or payload_type == "ws_chunk":
        return [], None
    if payload_type not in _SIMULATION_WS_CHUNK_MESSAGE_TYPES:
        return [], None

    chunk_size = max(4096, int(_safe_float(chunk_chars, _SIMULATION_WS_CHUNK_CHARS)))
    payload_text = _json_compact(payload)
    payload_chars = len(payload_text)
    if payload_type == "simulation_delta":
        delta_min_chars = max(chunk_size, int(_SIMULATION_WS_CHUNK_DELTA_MIN_CHARS))
        if payload_chars <= delta_min_chars:
            return [], payload_text
    if payload_chars <= chunk_size:
        return [], payload_text

    chunk_total = int(math.ceil(payload_chars / float(chunk_size)))
    if chunk_total <= 1:
        return [], payload_text
    if chunk_total > _SIMULATION_WS_CHUNK_MAX_CHUNKS:
        chunk_total = _SIMULATION_WS_CHUNK_MAX_CHUNKS
        chunk_size = int(math.ceil(payload_chars / float(chunk_total)))

    digest = hashlib.sha1(payload_text.encode("utf-8")).hexdigest()[:16]
    chunk_id = f"ws:{payload_type}:{int(message_seq)}:{digest}"

    rows: list[dict[str, Any]] = []
    for index in range(chunk_total):
        start = index * chunk_size
        end = min(payload_chars, (index + 1) * chunk_size)
        chunk_payload = payload_text[start:end]
        rows.append(
            {
                "type": "ws_chunk",
                "record": "eta-mu.ws.chunk.v1",
                "schema_version": "ws.chunk.v1",
                "chunk_id": chunk_id,
                "chunk_payload_type": payload_type,
                "chunk_index": index,
                "chunk_total": chunk_total,
                "payload_chars": payload_chars,
                "payload": chunk_payload,
            }
        )
    return rows, None


def _simulation_ws_chunk_messages(
    payload: dict[str, Any],
    *,
    chunk_chars: int,
    message_seq: int,
) -> list[dict[str, Any]]:
    rows, _ = _simulation_ws_chunk_plan(
        payload,
        chunk_chars=chunk_chars,
        message_seq=message_seq,
    )
    return rows


def _simulation_ws_worker_for_top_level_key(key: str) -> str:
    clean = str(key or "").strip().lower()
    if clean in {
        "timestamp",
        "total",
        "audio",
        "image",
        "video",
        "points",
        "truth_state",
        "perspective",
        "myth",
        "world",
        "tick_elapsed_ms",
        "slack_ms",
        "ingestion_pressure",
        "ws_particle_max",
        "particle_payload_mode",
        "graph_node_positions_truncated",
        "graph_node_positions_total",
        "presence_anchor_positions_truncated",
        "presence_anchor_positions_total",
    }:
        return "sim-core"
    if clean in {"projection"}:
        return "sim-projection"
    if clean in {"field_particles", "pain_field"}:
        return "sim-particles"
    if clean in {"daimoi", "entities", "echoes"}:
        return "sim-daimoi"
    if clean in {"presence_dynamics"}:
        return "sim-presence"
    return "sim-misc"


def _simulation_ws_worker_for_presence_key(key: str) -> str:
    clean = str(key or "").strip().lower()
    if clean in {
        "field_particles",
        "nooi_field",
        "graph_node_positions",
        "presence_anchor_positions",
    }:
        return "sim-particles"
    if clean in {
        "simulation_budget",
        "resource_heartbeat",
        "compute_jobs_180s",
        "compute_summary",
        "compute_jobs",
        "resource_daimoi",
        "resource_consumption",
        "river_flow",
    }:
        return "sim-resource"
    if clean in {
        "click_events",
        "file_events",
        "recent_click_targets",
        "recent_file_paths",
        "user_presence",
        "user_input_messages",
        "user_embedded_daimoi_count",
    }:
        return "sim-interaction"
    if clean in {
        "daimoi_probabilistic",
        "daimoi_probabilistic_record",
        "daimoi_behavior_defaults",
        "ghost",
        "fork_tax",
    }:
        return "sim-daimoi"
    return "sim-presence"


def _simulation_ws_split_delta_by_worker(
    delta: dict[str, Any],
) -> list[dict[str, Any]]:
    patch = delta.get("patch") if isinstance(delta, dict) else None
    if not isinstance(patch, dict) or not patch:
        return []

    worker_patch: dict[str, dict[str, Any]] = {}
    worker_changed_keys: dict[str, list[str]] = {}

    def _record_worker_patch(
        *,
        worker_id: str,
        key: str,
        value: Any,
        changed_key: str,
    ) -> None:
        bucket = worker_patch.get(worker_id)
        if not isinstance(bucket, dict):
            bucket = {}
            worker_patch[worker_id] = bucket
        bucket[key] = value

        changed = worker_changed_keys.get(worker_id)
        if not isinstance(changed, list):
            changed = []
            worker_changed_keys[worker_id] = changed
        if changed_key not in changed:
            changed.append(changed_key)

    for key, value in patch.items():
        if key == "presence_dynamics":
            if isinstance(value, dict):
                grouped_dynamics: dict[str, dict[str, Any]] = {}
                for dynamics_key, dynamics_value in value.items():
                    worker_id = _simulation_ws_worker_for_presence_key(dynamics_key)
                    grouped = grouped_dynamics.get(worker_id)
                    if not isinstance(grouped, dict):
                        grouped = {}
                        grouped_dynamics[worker_id] = grouped
                    grouped[dynamics_key] = dynamics_value
                for worker_id, dynamics_patch in grouped_dynamics.items():
                    bucket = worker_patch.get(worker_id)
                    if not isinstance(bucket, dict):
                        bucket = {}
                        worker_patch[worker_id] = bucket
                    existing_dynamics = bucket.get("presence_dynamics")
                    if not isinstance(existing_dynamics, dict):
                        existing_dynamics = {}
                        bucket["presence_dynamics"] = existing_dynamics
                    existing_dynamics.update(dynamics_patch)

                    changed = worker_changed_keys.get(worker_id)
                    if not isinstance(changed, list):
                        changed = []
                        worker_changed_keys[worker_id] = changed
                    for dynamics_key in dynamics_patch:
                        changed_key = f"presence_dynamics.{dynamics_key}"
                        if changed_key not in changed:
                            changed.append(changed_key)
            else:
                _record_worker_patch(
                    worker_id="sim-presence",
                    key="presence_dynamics",
                    value=value,
                    changed_key="presence_dynamics",
                )
            continue

        _record_worker_patch(
            worker_id=_simulation_ws_worker_for_top_level_key(key),
            key=key,
            value=value,
            changed_key=key,
        )

    timestamp_value = patch.get("timestamp")
    if timestamp_value is not None:
        for row in worker_patch.values():
            if isinstance(row, dict) and "timestamp" not in row:
                row["timestamp"] = timestamp_value

    worker_priority = {
        "sim-core": 0,
        "sim-particles": 1,
        "sim-resource": 2,
        "sim-interaction": 3,
        "sim-presence": 4,
        "sim-daimoi": 5,
        "sim-projection": 6,
        "sim-misc": 7,
    }
    rows: list[dict[str, Any]] = []
    for worker_id, worker_row_patch in worker_patch.items():
        changed = worker_changed_keys.get(worker_id, [])
        rows.append(
            {
                "worker_id": worker_id,
                "patch": worker_row_patch,
                "changed_keys": changed,
            }
        )
    rows.sort(
        key=lambda row: (
            worker_priority.get(str(row.get("worker_id", "")), 99),
            str(row.get("worker_id", "")),
        )
    )
    return rows


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
        "heat_values",
        "file_graph",
        "crawler_graph",
        "field_registry",
    ):
        compact.pop(key, None)

    compact.update(
        _simulation_ws_compact_graph_payload(simulation, assume_trimmed=False)
    )

    point_rows = compact.get("points")
    if isinstance(point_rows, list):
        point_total = len(point_rows)
        if point_total > _SIMULATION_HTTP_COMPACT_MAX_POINTS:
            compact["points"] = _simulation_http_slice_rows(
                point_rows,
                max_rows=_SIMULATION_HTTP_COMPACT_MAX_POINTS,
            )
            compact["points_total"] = point_total
            compact["points_compacted"] = True

    dynamics = compact.get("presence_dynamics")
    if isinstance(dynamics, dict):
        compact_dynamics = dict(dynamics)
        particle_rows = compact_dynamics.get("field_particles")
        if isinstance(particle_rows, list):
            particle_total = len(particle_rows)
            if particle_total > _SIMULATION_HTTP_COMPACT_MAX_FIELD_PARTICLES:
                compact_dynamics["field_particles"] = _simulation_http_slice_rows(
                    particle_rows,
                    max_rows=_SIMULATION_HTTP_COMPACT_MAX_FIELD_PARTICLES,
                )
                compact_dynamics["field_particles_total"] = particle_total
                compact_dynamics["field_particles_compacted"] = True
            compact["presence_dynamics"] = compact_dynamics
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


def _simulation_http_runtime_reference_mtime(part_root: Path) -> float:
    candidate_paths = [
        part_root / "code" / "world_web" / "server.py",
        part_root / "code" / "world_web" / "simulation.py",
        part_root / "code" / "world_web" / "c_double_buffer_backend.py",
        part_root / "code" / "world_web" / "daimoi_probabilistic.py",
        part_root / "code" / "world_web" / "native" / "libc_double_buffer_sim.so",
    ]
    newest = 0.0
    for path in candidate_paths:
        try:
            if path.exists() and path.is_file():
                newest = max(newest, float(path.stat().st_mtime))
        except Exception:
            continue
    return newest


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
        runtime_reference_mtime = _simulation_http_runtime_reference_mtime(part_root)
        if (
            runtime_reference_mtime > 0.0
            and float(stat.st_mtime) < runtime_reference_mtime
        ):
            return None
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
        os.getenv(
            "EMBED_FORCE_NOMIC",
            os.getenv("OLLAMA_EMBED_FORCE_NOMIC", "0"),
        )
        or "0"
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


_WS_CLIENT_FRAME_MAX_BYTES = WS_CLIENT_FRAME_MAX_BYTES


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

    def _send_ndjson_row(self, payload: dict[str, Any]) -> bool:
        line = (_json_compact(payload) + "\n").encode("utf-8")
        try:
            self.wfile.write(line)
            self.wfile.flush()
            return True
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ):
            return False

    def _send_catalog_stream(
        self,
        *,
        perspective: str,
        chunk_rows: int,
        trim_catalog: bool,
    ) -> None:
        stream_chunk_rows = _catalog_stream_chunk_rows(chunk_rows)
        self.send_response(HTTPStatus.OK)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        if not self._send_ndjson_row(
            {
                "type": "start",
                "ok": True,
                "record": "eta-mu.catalog.stream.v1",
                "schema_version": "catalog.stream.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "perspective": perspective,
                "chunk_rows": stream_chunk_rows,
                "trimmed": bool(trim_catalog),
            }
        ):
            return

        stream_started = time.monotonic()
        progress_rows: queue.Queue[dict[str, Any]] = queue.Queue()
        done = threading.Event()
        result: dict[str, Any] = {}

        def emit_progress(stage: str, detail: dict[str, Any] | None = None) -> None:
            progress_rows.put(
                {
                    "stage": str(stage or "").strip().lower(),
                    "detail": dict(detail) if isinstance(detail, dict) else {},
                }
            )

        def collect_worker() -> None:
            try:
                emit_progress("collect_inline_start")
                catalog = collect_catalog(
                    self.part_root,
                    self.vault_root,
                    sync_inbox=False,
                    include_pi_archive=False,
                    include_world_log=False,
                    progress_callback=emit_progress,
                )

                emit_progress("runtime_enrichment_start")
                queue_snapshot = self.task_queue.snapshot(include_pending=False)
                council_snapshot = self.council_chamber.snapshot(
                    include_decisions=False
                )
                catalog["task_queue"] = queue_snapshot
                catalog["council"] = council_snapshot
                resource_snapshot = _resource_monitor_snapshot(part_root=self.part_root)
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    resource_snapshot,
                    source="runtime.catalog.stream",
                )
                influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot,
                    part_root=self.part_root,
                )
                catalog["presence_runtime"] = influence_snapshot
                catalog["muse_runtime"] = self._muse_manager().snapshot()
                attach_ui_projection(
                    catalog,
                    perspective=perspective,
                    queue_snapshot=queue_snapshot,
                    influence_snapshot=influence_snapshot,
                )
                emit_progress(
                    "runtime_enrichment_done",
                    {
                        "item_count": len(catalog.get("items", []))
                        if isinstance(catalog.get("items", []), list)
                        else 0,
                    },
                )
                result["catalog"] = catalog
            except Exception as exc:
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(
            target=collect_worker,
            daemon=True,
            name="catalog-stream-collector",
        ).start()

        heartbeat_count = 0
        while not done.is_set() or not progress_rows.empty():
            try:
                progress_row = progress_rows.get(timeout=1.0)
            except queue.Empty:
                heartbeat_count += 1
                if not self._send_ndjson_row(
                    {
                        "type": "heartbeat",
                        "record": "eta-mu.catalog.stream.progress.v1",
                        "schema_version": "catalog.stream.progress.v1",
                        "heartbeat_count": heartbeat_count,
                        "elapsed_ms": round(
                            (time.monotonic() - stream_started) * 1000.0,
                            3,
                        ),
                    }
                ):
                    return
                continue

            if not self._send_ndjson_row(
                {
                    "type": "progress",
                    "record": "eta-mu.catalog.stream.progress.v1",
                    "schema_version": "catalog.stream.progress.v1",
                    "stage": str(progress_row.get("stage", "") or ""),
                    "detail": (
                        dict(progress_row.get("detail", {}))
                        if isinstance(progress_row.get("detail", {}), dict)
                        else {}
                    ),
                    "elapsed_ms": round(
                        (time.monotonic() - stream_started) * 1000.0,
                        3,
                    ),
                }
            ):
                return

        if "error" in result:
            exc = cast(Exception, result.get("error"))
            self._send_ndjson_row(
                {
                    "type": "error",
                    "ok": False,
                    "record": "eta-mu.catalog.stream.error.v1",
                    "schema_version": "catalog.stream.error.v1",
                    "error": f"catalog_stream_failed:{exc.__class__.__name__}",
                    "detail": str(exc),
                }
            )
            self._send_ndjson_row(
                {
                    "type": "done",
                    "ok": False,
                    "record": "eta-mu.catalog.stream.done.v1",
                    "schema_version": "catalog.stream.done.v1",
                    "chunk_rows": stream_chunk_rows,
                }
            )
            return

        stream_catalog = (
            _simulation_http_trim_catalog(result.get("catalog", {}))
            if trim_catalog
            else dict(result.get("catalog", {}))
            if isinstance(result.get("catalog", {}), dict)
            else {}
        )
        for row in _catalog_stream_iter_rows(
            stream_catalog,
            chunk_rows=stream_chunk_rows,
        ):
            if not self._send_ndjson_row(row):
                return

    def _send_ws_frame(self, frame: bytes) -> None:
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

    def _send_ws_text(self, payload_text: str) -> None:
        self._send_ws_frame(websocket_frame_text(payload_text))

    def _send_ws_event(
        self, payload: dict[str, Any], *, wire_mode: str = "json"
    ) -> None:
        wire_mode_key = (
            wire_mode
            if wire_mode in {"arr", "json"}
            else _normalize_ws_wire_mode(wire_mode)
        )
        if wire_mode_key == "arr":
            ws_payload: Any = _ws_pack_message(payload)
        else:
            ws_payload = payload
        self._send_ws_text(_json_compact(ws_payload))

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
        if clean_tool == "facts_snapshot":
            simulation = build_simulation_state(self._collect_catalog_fast())
            payload = build_facts_snapshot(simulation, part_root=self.part_root)
            counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
            node_count = max(
                0,
                int(
                    _safe_float(
                        sum(
                            int(_safe_float(value, 0.0))
                            for value in (
                                counts.get("nodes_by_role", {}).values()
                                if isinstance(counts.get("nodes_by_role", {}), dict)
                                else []
                            )
                        ),
                        0.0,
                    )
                ),
            )
            edge_count = max(
                0,
                int(
                    _safe_float(
                        sum(
                            int(_safe_float(value, 0.0))
                            for value in (
                                counts.get("edges_by_kind", {}).values()
                                if isinstance(counts.get("edges_by_kind", {}), dict)
                                else []
                            )
                        ),
                        0.0,
                    )
                ),
            )
            return {
                "ok": True,
                "summary": "facts snapshot generated",
                "record": str(payload.get("record", "")),
                "snapshot_hash": str(payload.get("snapshot_hash", "")),
                "snapshot_path": str(payload.get("snapshot_path", "")),
                "node_count": node_count,
                "edge_count": edge_count,
            }
        if clean_tool.startswith("graph:"):
            graph_tail = str(
                clean_tool.split(":", 1)[1] if ":" in clean_tool else ""
            ).strip()
            tail_parts = [piece for piece in graph_tail.split(" ") if piece]
            query_name = str(tail_parts[0] if tail_parts else "").strip().lower()
            query_arg = " ".join(tail_parts[1:]).strip()
            if not query_name:
                query_name = "overview"
            query_args: dict[str, Any] = {}
            if query_name in {"neighbors", "node_neighbors"} and query_arg:
                query_args["node_id"] = query_arg
            elif query_name == "search" and query_arg:
                query_args["q"] = query_arg
            elif query_name in {"url_status", "resource_for_url"} and query_arg:
                query_args["target"] = query_arg
            elif query_name == "recently_updated" and query_arg:
                query_args["limit"] = max(1, int(_safe_float(query_arg, 24.0)))
            elif query_name == "role_slice" and query_arg:
                query_args["role"] = query_arg
            simulation = build_simulation_state(self._collect_catalog_fast())
            nexus_graph = (
                simulation.get("nexus_graph", {})
                if isinstance(simulation.get("nexus_graph", {}), dict)
                else {}
            )
            payload = run_named_graph_query(nexus_graph, query_name, args=query_args)
            result = payload.get("result", {})
            if isinstance(result, dict) and result.get("error"):
                return {
                    "ok": False,
                    "error": str(result.get("error", "unknown_query")),
                    "query": query_name,
                }
            result_count = 0
            if isinstance(result, dict):
                for count_key in (
                    "count",
                    "neighbor_count",
                    "node_count",
                    "edge_count",
                ):
                    if count_key in result:
                        result_count = max(
                            result_count,
                            int(_safe_float(result.get(count_key, 0), 0.0)),
                        )
            return {
                "ok": True,
                "summary": f"graph query {query_name} generated",
                "query": query_name,
                "snapshot_hash": str(payload.get("snapshot_hash", "")),
                "result_count": int(result_count),
                "result": result if isinstance(result, dict) else {},
            }
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
            "canonical" if str(mode).strip().lower() == "deterministic" else "llm"
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
        if not _RUNTIME_ETA_MU_SYNC_ENABLED:
            return
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
                now_monotonic = time.monotonic()
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = now_monotonic
                    snapshot_value = _RUNTIME_CATALOG_CACHE.get("inbox_sync_snapshot")
                    if not isinstance(snapshot_value, dict):
                        _RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = {}
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
                should_sync = _RUNTIME_ETA_MU_SYNC_ENABLED and (
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

    def _runtime_catalog_base(
        self,
        *,
        allow_inline_collect: bool = True,
        strict_collect: bool = False,
    ) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        with _RUNTIME_CATALOG_CACHE_LOCK:
            cached_catalog = _RUNTIME_CATALOG_CACHE.get("catalog")
            refreshed_monotonic = float(
                _RUNTIME_CATALOG_CACHE.get("refreshed_monotonic", 0.0)
            )
            inbox_sync_snapshot = _RUNTIME_CATALOG_CACHE.get("inbox_sync_snapshot")

        if not isinstance(cached_catalog, dict):
            with _RUNTIME_CATALOG_COLLECT_LOCK:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    recached_catalog = _RUNTIME_CATALOG_CACHE.get("catalog")
                    if isinstance(recached_catalog, dict):
                        return dict(recached_catalog)
                    inbox_sync_snapshot = _RUNTIME_CATALOG_CACHE.get(
                        "inbox_sync_snapshot"
                    )

                fresh_catalog, isolated_error = _collect_runtime_catalog_isolated(
                    self.part_root,
                    self.vault_root,
                )
                try:
                    if fresh_catalog is None and allow_inline_collect:
                        fresh_catalog = self._collect_catalog_fast()
                    if fresh_catalog is None and strict_collect:
                        raise RuntimeError(
                            isolated_error or "catalog_collect_unavailable"
                        )
                    if fresh_catalog is None:
                        raise RuntimeError(
                            isolated_error or "catalog_collect_unavailable"
                        )
                    cache_error = isolated_error
                    if cache_error == "catalog_subprocess_disabled":
                        cache_error = ""
                    if isinstance(inbox_sync_snapshot, dict):
                        fresh_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
                    with _RUNTIME_CATALOG_CACHE_LOCK:
                        _RUNTIME_CATALOG_CACHE["catalog"] = fresh_catalog
                        _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = time.monotonic()
                        _RUNTIME_CATALOG_CACHE["last_error"] = cache_error
                    if _RUNTIME_ETA_MU_SYNC_ENABLED:
                        self._schedule_runtime_inbox_sync()
                    return dict(fresh_catalog)
                except Exception as exc:
                    with _RUNTIME_CATALOG_CACHE_LOCK:
                        _RUNTIME_CATALOG_CACHE["last_error"] = (
                            f"catalog_inline_failed:{exc.__class__.__name__}"
                        )
                    if strict_collect:
                        raise

        cache_age = now_monotonic - refreshed_monotonic
        if cache_age >= _RUNTIME_CATALOG_CACHE_SECONDS:
            self._schedule_runtime_catalog_refresh()

        if isinstance(cached_catalog, dict):
            return dict(cached_catalog)
        fallback_catalog = _runtime_catalog_fallback(self.part_root, self.vault_root)
        if isinstance(inbox_sync_snapshot, dict):
            fallback_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
        with _RUNTIME_CATALOG_CACHE_LOCK:
            _RUNTIME_CATALOG_CACHE["catalog"] = dict(fallback_catalog)
            _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = time.monotonic()
        return fallback_catalog

    def _runtime_catalog(
        self,
        *,
        perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
        include_projection: bool = True,
        allow_inline_collect: bool = True,
        strict_collect: bool = False,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        catalog = self._runtime_catalog_base(
            allow_inline_collect=allow_inline_collect,
            strict_collect=strict_collect,
        )
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
        include_unified_graph: bool = True,
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
            include_unified_graph=include_unified_graph,
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

    def _run_simulation_bootstrap(
        self,
        *,
        perspective: str,
        sync_inbox: bool,
        include_simulation_payload: bool,
        phase_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
    ) -> tuple[dict[str, Any], HTTPStatus]:
        normalized_perspective = normalize_projection_perspective(perspective)
        phase_started = time.perf_counter()
        phase_ms: dict[str, float] = {}
        current_phase = "queued"

        def _mark_phase(phase: str, detail: dict[str, Any] | None = None) -> None:
            nonlocal current_phase
            current_phase = str(phase or "").strip().lower() or current_phase
            if callable(phase_callback):
                try:
                    phase_callback(current_phase, detail)
                except Exception:
                    pass

        def _check_timeout() -> None:
            elapsed = time.perf_counter() - phase_started
            if elapsed > _SIMULATION_BOOTSTRAP_MAX_SECONDS:
                raise TimeoutError(
                    f"bootstrap_timeout:{round(elapsed, 3)}s>"
                    f"{round(_SIMULATION_BOOTSTRAP_MAX_SECONDS, 3)}s"
                )

        def _run_with_phase(
            *,
            phase: str,
            operation: Callable[[], Any],
            detail: dict[str, Any] | None = None,
            heartbeat: bool = False,
        ) -> Any:
            phase_detail = dict(detail) if isinstance(detail, dict) else {}
            _mark_phase(phase, phase_detail if phase_detail else None)
            if not heartbeat or not callable(phase_callback):
                return operation()

            stop_heartbeat = threading.Event()
            phase_started_local = time.perf_counter()

            def _heartbeat_loop() -> None:
                heartbeat_count = 0
                while not stop_heartbeat.wait(_SIMULATION_BOOTSTRAP_HEARTBEAT_SECONDS):
                    heartbeat_count += 1
                    heartbeat_detail = dict(phase_detail)
                    heartbeat_detail["heartbeat_count"] = heartbeat_count
                    heartbeat_detail["phase_elapsed_ms"] = round(
                        (time.perf_counter() - phase_started_local) * 1000.0,
                        3,
                    )
                    _mark_phase(phase, heartbeat_detail)

            threading.Thread(
                target=_heartbeat_loop,
                daemon=True,
                name=f"simulation-bootstrap-heartbeat-{phase}",
            ).start()
            try:
                return operation()
            finally:
                stop_heartbeat.set()
                final_detail = dict(phase_detail)
                final_detail["phase_elapsed_ms"] = round(
                    (time.perf_counter() - phase_started_local) * 1000.0,
                    3,
                )
                _mark_phase(phase, final_detail if final_detail else None)

        reset_summary: dict[str, Any] = {}
        _mark_phase(
            "reset",
            {
                "perspective": normalized_perspective,
                "sync_inbox": bool(sync_inbox),
            },
        )
        try:
            reset_summary = simulation_module.reset_simulation_bootstrap_state(
                clear_layout_cache=True,
                rearm_boot_reset=True,
            )
        except Exception as exc:
            reset_summary = {
                "ok": False,
                "error": f"bootstrap_reset_failed:{exc.__class__.__name__}",
            }
        phase_ms["reset"] = (time.perf_counter() - phase_started) * 1000.0

        _mark_phase("cache_invalidate")
        with _RUNTIME_CATALOG_CACHE_LOCK:
            _RUNTIME_CATALOG_CACHE["catalog"] = None
            _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = 0.0
            _RUNTIME_CATALOG_CACHE["last_error"] = ""
            if sync_inbox:
                _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = 0.0
                _RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = None
                _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""

        _simulation_http_cache_invalidate(part_root=self.part_root)
        phase_ms["cache_invalidate"] = (time.perf_counter() - phase_started) * 1000.0

        inbox_sync: dict[str, Any] = {"status": "skipped"}
        _mark_phase("inbox_sync", {"enabled": bool(sync_inbox)})
        if sync_inbox:
            try:
                inbox_snapshot = sync_eta_mu_inbox(self.vault_root)
                inbox_sync = {
                    "status": "completed",
                    "pending_count": max(
                        0,
                        int(_safe_float(inbox_snapshot.get("pending_count", 0), 0.0)),
                    ),
                    "processed_count": max(
                        0,
                        int(_safe_float(inbox_snapshot.get("processed_count", 0), 0.0)),
                    ),
                    "failed_count": max(
                        0,
                        int(_safe_float(inbox_snapshot.get("failed_count", 0), 0.0)),
                    ),
                }
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = time.monotonic()
                    _RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = dict(inbox_snapshot)
                    _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""
            except Exception as exc:
                inbox_sync = {
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = (
                        f"inbox_sync_failed:{exc.__class__.__name__}"
                    )
        phase_ms["inbox_sync"] = (time.perf_counter() - phase_started) * 1000.0

        try:
            catalog_phase_detail = {
                "strict_collect": True,
                "allow_inline_collect": False,
            }
            try:
                catalog, queue_snapshot, _, influence_snapshot, _ = _run_with_phase(
                    phase="catalog",
                    detail=catalog_phase_detail,
                    heartbeat=True,
                    operation=lambda: self._runtime_catalog(
                        perspective=normalized_perspective,
                        include_projection=False,
                        allow_inline_collect=False,
                        strict_collect=True,
                    ),
                )
            except RuntimeError as catalog_exc:
                fallback_reason = str(catalog_exc or "").strip()
                catalog, queue_snapshot, _, influence_snapshot, _ = _run_with_phase(
                    phase="catalog_fallback_inline",
                    detail={
                        "strict_collect": False,
                        "allow_inline_collect": True,
                        "fallback_from": "catalog",
                        "fallback_reason": fallback_reason,
                    },
                    heartbeat=True,
                    operation=lambda: self._runtime_catalog(
                        perspective=normalized_perspective,
                        include_projection=False,
                        allow_inline_collect=True,
                        strict_collect=False,
                    ),
                )
                phase_ms["catalog_fallback_inline"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0
            phase_ms["catalog"] = (time.perf_counter() - phase_started) * 1000.0
            _check_timeout()

            simulation, projection = _run_with_phase(
                phase="simulation",
                heartbeat=True,
                operation=lambda: self._runtime_simulation(
                    catalog,
                    queue_snapshot,
                    influence_snapshot,
                    perspective=normalized_perspective,
                ),
            )
            phase_ms["simulation"] = (time.perf_counter() - phase_started) * 1000.0
            _check_timeout()

            _mark_phase("cache_store")
            simulation_response_payload = dict(simulation)
            simulation_response_payload["projection"] = projection
            response_body = _json_compact(simulation_response_payload).encode("utf-8")
            cache_key = _simulation_http_cache_key(
                perspective=normalized_perspective,
                catalog=catalog,
                queue_snapshot=queue_snapshot,
                influence_snapshot=influence_snapshot,
            )
            _simulation_http_cache_store(cache_key, response_body)
            _simulation_http_disk_cache_store(
                self.part_root,
                perspective=normalized_perspective,
                body=response_body,
            )
            phase_ms["cache_store"] = (time.perf_counter() - phase_started) * 1000.0
            _check_timeout()

            _mark_phase("report")
            payload = _simulation_bootstrap_graph_report(
                perspective=normalized_perspective,
                catalog=catalog,
                simulation=simulation,
                projection=projection,
                phase_ms=phase_ms,
                reset_summary=reset_summary,
                inbox_sync=inbox_sync,
                cache_key=cache_key,
            )
            if include_simulation_payload:
                payload["simulation"] = simulation
                payload["projection_payload"] = projection
            _simulation_bootstrap_store_report(payload)
            return payload, HTTPStatus.OK
        except Exception as exc:
            failed_phase = current_phase
            _mark_phase(
                "failed",
                {
                    "failed_phase": failed_phase,
                    "error": f"{exc.__class__.__name__}: {exc}",
                },
            )
            payload = {
                "ok": False,
                "record": "eta-mu.simulation-bootstrap.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "perspective": normalized_perspective,
                "error": f"simulation_bootstrap_failed:{exc.__class__.__name__}",
                "detail": f"{exc.__class__.__name__}: {exc}",
                "failed_phase": failed_phase,
                "phase_ms": {
                    str(key): round(max(0.0, _safe_float(value, 0.0)), 3)
                    for key, value in phase_ms.items()
                },
                "reset": reset_summary,
                "inbox_sync": inbox_sync,
            }
            _simulation_bootstrap_store_report(payload)
            return payload, HTTPStatus.INTERNAL_SERVER_ERROR

    def _handle_docker_websocket(self, *, wire_mode: str) -> None:
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

        # Use poll for non-blocking socket reads
        poll = select.poll()
        poll.register(self.connection, select.POLLIN)
        ws_wire_mode = _normalize_ws_wire_mode(wire_mode)

        def send_ws(payload: dict[str, Any]) -> None:
            self._send_ws_event(payload, wire_mode=ws_wire_mode)

        last_docker_refresh = time.monotonic()
        last_docker_broadcast = last_docker_refresh
        last_docker_fingerprint = ""

        try:
            try:
                docker_snapshot = collect_docker_simulation_snapshot(force_refresh=True)
                last_docker_fingerprint = str(
                    docker_snapshot.get("fingerprint", "") or ""
                )
                send_ws(
                    {
                        "type": "docker_simulations",
                        "docker": docker_snapshot,
                    }
                )
            except Exception as exc:
                send_ws(
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
                        send_ws(
                            {
                                "type": "docker_simulations",
                                "docker": docker_snapshot,
                            }
                        )
                        last_docker_broadcast = now_monotonic
                        last_docker_fingerprint = docker_fingerprint
                    last_docker_refresh = now_monotonic

                # Non-blocking check for incoming client data
                ready = poll.poll(10)
                if ready:
                    if not _consume_ws_client_frame(self.connection):
                        break
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError,
        ):
            pass
        finally:
            _runtime_ws_release_client_slot()

    def _handle_websocket(
        self,
        *,
        perspective: str,
        delta_stream_mode: str,
        wire_mode: str,
        payload_mode: str,
        particle_payload_mode: str,
        chunk_stream_enabled: bool,
        catalog_events_enabled: bool,
        skip_catalog_bootstrap: bool,
    ) -> None:
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
        stream_mode = _simulation_ws_normalize_delta_stream_mode(delta_stream_mode)
        payload_mode_key = _simulation_ws_normalize_payload_mode(payload_mode)
        particle_payload_key = _simulation_ws_normalize_particle_payload_mode(
            particle_payload_mode
        )
        use_cached_snapshots = _SIMULATION_WS_USE_CACHED_SNAPSHOTS
        effective_skip_catalog_bootstrap = bool(
            skip_catalog_bootstrap and use_cached_snapshots
        )
        ws_wire_mode = _normalize_ws_wire_mode(wire_mode)
        if (
            use_cached_snapshots
            and ws_wire_mode == "arr"
            and _SIMULATION_WS_CACHE_FORCE_JSON_WIRE
        ):
            ws_wire_mode = "json"
        ws_chunk_enabled = bool(chunk_stream_enabled)
        ws_chunk_chars = _SIMULATION_WS_CHUNK_CHARS
        ws_chunk_message_seq = 0
        ws_full_snapshot_min_slack_ms = max(
            0.0,
            _safe_float(
                os.getenv("SIM_WS_FULL_SNAPSHOT_MIN_SLACK_MS", "12") or "12",
                12.0,
            ),
        )
        ws_graph_pos_max_default = max(
            64,
            int(
                _safe_float(
                    os.getenv("SIM_WS_GRAPH_POS_MAX", "4096") or "4096",
                    4096.0,
                )
            ),
        )

        def send_ws(payload: dict[str, Any]) -> None:
            nonlocal ws_chunk_message_seq

            if ws_chunk_enabled:
                ws_chunk_message_seq += 1
                if ws_wire_mode == "json":
                    chunk_rows, payload_text = _simulation_ws_chunk_plan(
                        payload,
                        chunk_chars=ws_chunk_chars,
                        message_seq=ws_chunk_message_seq,
                    )
                    if chunk_rows:
                        for chunk_row in chunk_rows:
                            self._send_ws_event(chunk_row, wire_mode=ws_wire_mode)
                        return
                    if isinstance(payload_text, str):
                        self._send_ws_text(payload_text)
                        return
                else:
                    chunk_rows = _simulation_ws_chunk_messages(
                        payload,
                        chunk_chars=ws_chunk_chars,
                        message_seq=ws_chunk_message_seq,
                    )
                    if chunk_rows:
                        for chunk_row in chunk_rows:
                            self._send_ws_event(chunk_row, wire_mode=ws_wire_mode)
                        return

            self._send_ws_event(payload, wire_mode=ws_wire_mode)

        # Use poll for non-blocking socket reads
        poll = select.poll()
        poll.register(self.connection, select.POLLIN)
        catalog: dict[str, Any] = {}
        queue_snapshot: dict[str, Any] = {}
        influence_snapshot: dict[str, Any] = {}
        last_simulation_for_delta: dict[str, Any] | None = None
        muse_event_seq = 0
        last_docker_fingerprint = ""
        simulation_worker_seq = 0
        last_projection_delta_broadcast = 0.0
        last_graph_position_broadcast = 0.0
        last_cached_simulation_timestamp = ""
        last_daimoi_live_metrics_refresh = 0.0
        last_daimoi_graph_variability_refresh = 0.0
        last_ws_stream_cache_store = 0.0
        stream_particles: list[dict[str, Any]] = []
        last_muse_poll = 0.0
        last_config_runtime_version = _config_runtime_version_snapshot()
        tick_governor = get_governor() if _SIMULATION_WS_TICK_GOVERNOR_ENABLED else None
        last_governor_resource_refresh = 0.0
        governor_particle_cap = max(
            _SIMULATION_WS_GOVERNOR_MIN_PARTICLE_CAP,
            _SIMULATION_WS_STREAM_PARTICLE_MAX,
        )
        governor_graph_heartbeat_scale = 1.0

        def _build_ws_snapshot_payload(simulation: dict[str, Any]) -> dict[str, Any]:
            if payload_mode_key == "full":
                snapshot_payload = dict(simulation)
                snapshot_payload.pop("projection", None)
                return snapshot_payload

            snapshot_payload = _simulation_ws_trim_simulation_payload(simulation)
            snapshot_payload.pop("projection", None)
            snapshot_payload.update(
                _simulation_ws_compact_graph_payload(
                    simulation,
                    assume_trimmed=True,
                )
            )
            _simulation_ws_extract_stream_particles(snapshot_payload)
            return snapshot_payload

        def _build_ws_delta_payload(simulation: dict[str, Any]) -> dict[str, Any]:
            delta_payload = _simulation_ws_trim_simulation_payload(simulation)
            delta_payload.pop("projection", None)
            _simulation_ws_extract_stream_particles(delta_payload)
            return delta_payload

        def rebuild_live_simulation_payload():
            nonlocal catalog
            nonlocal queue_snapshot
            nonlocal influence_snapshot

            catalog, queue_snapshot, _, influence_snapshot, _ = self._runtime_catalog(
                perspective=perspective_key,
                include_projection=False,
            )
            simulation, projection = self._runtime_simulation(
                catalog,
                queue_snapshot,
                influence_snapshot,
                perspective=perspective_key,
                include_unified_graph=payload_mode_key == "full",
            )
            snapshot_payload = _build_ws_snapshot_payload(simulation)
            delta_payload = _build_ws_delta_payload(simulation)
            return snapshot_payload, delta_payload, projection

        def maybe_send_muse_events(now_monotonic: float) -> None:
            nonlocal muse_event_seq
            nonlocal last_muse_poll

            if now_monotonic - last_muse_poll < _SIMULATION_WS_MUSE_POLL_SECONDS:
                return

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
                send_ws(
                    {
                        "type": "muse_events",
                        "events": muse_events,
                        "since_seq": previous_muse_seq,
                        "next_seq": muse_event_seq,
                    }
                )
            last_muse_poll = now_monotonic

        try:
            if effective_skip_catalog_bootstrap:
                catalog = {}
                queue_snapshot = {}
                influence_snapshot = {}
            else:
                catalog, queue_snapshot, _, influence_snapshot, _ = (
                    self._runtime_catalog(
                        perspective=perspective_key,
                        include_projection=False,
                    )
                )

            if catalog_events_enabled and not effective_skip_catalog_bootstrap:
                catalog_payload = _simulation_http_trim_catalog(catalog)
                _, mix_meta = build_mix_stream(catalog, self.vault_root)
                send_ws(
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
                send_ws(
                    {
                        "type": "muse_events",
                        "events": muse_bootstrap_events,
                        "since_seq": 0,
                        "next_seq": muse_event_seq,
                    }
                )

            if use_cached_snapshots:
                cached_payload = _simulation_ws_load_cached_payload(
                    part_root=self.part_root,
                    perspective=perspective_key,
                    payload_mode=payload_mode_key,
                )
                if cached_payload is None:
                    simulation_payload, projection = _simulation_ws_placeholder_payload(
                        perspective=perspective_key,
                    )
                else:
                    simulation_payload, projection = cached_payload

                if payload_mode_key == "full":
                    simulation_delta_payload = _build_ws_delta_payload(
                        simulation_payload
                    )
                else:
                    simulation_delta_payload = simulation_payload
                stream_particles = _simulation_ws_extract_stream_particles(
                    simulation_delta_payload
                )
                needs_live_bootstrap = False
                if _SIMULATION_WS_BOOTSTRAP_REQUIRE_LIVE_REBUILD:
                    needs_live_bootstrap = bool(
                        cached_payload is None
                        or (not stream_particles)
                        or _simulation_ws_payload_missing_daimoi_summary(
                            simulation_delta_payload
                        )
                        or _simulation_ws_payload_missing_graph_payload(
                            simulation_payload
                        )
                    )
                if payload_mode_key != "full" and needs_live_bootstrap:
                    simulation_payload, simulation_delta_payload, projection = (
                        rebuild_live_simulation_payload()
                    )
                    stream_particles = _simulation_ws_extract_stream_particles(
                        simulation_delta_payload
                    )
            else:
                simulation, projection = self._runtime_simulation(
                    catalog,
                    queue_snapshot,
                    influence_snapshot,
                    perspective=perspective_key,
                    include_unified_graph=payload_mode_key == "full",
                )
                simulation_payload = _build_ws_snapshot_payload(simulation)
                simulation_delta_payload = _build_ws_delta_payload(simulation)
            send_ws(
                {
                    "type": "simulation",
                    "simulation": simulation_payload,
                    "projection": projection,
                }
            )
            last_simulation_for_delta = simulation_delta_payload
            stream_particles = _simulation_ws_extract_stream_particles(
                simulation_delta_payload
            )
            last_cached_simulation_timestamp = str(
                simulation_payload.get("timestamp", "") or ""
            )
            boot_tick = time.monotonic()
            last_simulation_full_broadcast = boot_tick
            last_projection_delta_broadcast = boot_tick
            last_graph_position_broadcast = boot_tick
            last_simulation_cache_refresh = boot_tick

            if not use_cached_snapshots:
                docker_snapshot = collect_docker_simulation_snapshot()
                send_ws(
                    {
                        "type": "docker_simulations",
                        "docker": docker_snapshot,
                    }
                )
                last_docker_fingerprint = str(
                    docker_snapshot.get("fingerprint", "") or ""
                )
        except Exception:
            try:
                send_ws(
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

        clock_now = time.monotonic()
        last_catalog_refresh = clock_now
        last_catalog_broadcast = clock_now
        last_sim_tick = clock_now
        last_docker_refresh = clock_now
        last_docker_broadcast = clock_now
        last_runtime_guard_broadcast = clock_now
        last_muse_poll = clock_now
        runtime_guard: dict[str, Any] = {
            "mode": "normal",
            "reasons": [],
        }

        try:
            while True:
                now_monotonic = time.monotonic()
                current_config_runtime_version = _config_runtime_version_snapshot()
                if current_config_runtime_version != last_config_runtime_version:
                    last_config_runtime_version = current_config_runtime_version
                    refresh_motion_state = _simulation_ws_capture_particle_motion_state(
                        simulation_payload
                    )
                    try:
                        (
                            simulation_payload,
                            simulation_delta_payload,
                            projection,
                        ) = rebuild_live_simulation_payload()
                    except Exception:
                        pass
                    else:
                        if refresh_motion_state:
                            _simulation_ws_restore_particle_motion_state(
                                simulation_payload,
                                refresh_motion_state,
                            )
                            if simulation_delta_payload is not simulation_payload:
                                _simulation_ws_restore_particle_motion_state(
                                    simulation_delta_payload,
                                    refresh_motion_state,
                                )
                        stream_particles = _simulation_ws_extract_stream_particles(
                            simulation_delta_payload
                        )
                        last_simulation_for_delta = simulation_delta_payload
                        last_cached_simulation_timestamp = str(
                            simulation_payload.get("timestamp", "") or ""
                        )
                        send_ws(
                            {
                                "type": "simulation",
                                "simulation": simulation_payload,
                                "projection": projection,
                            }
                        )
                        cache_source_payload = (
                            simulation_delta_payload
                            if payload_mode_key == "full"
                            else simulation_payload
                        )
                        ws_cache_payload = dict(cache_source_payload)
                        ws_cache_payload["projection"] = projection
                        _simulation_http_cache_store(
                            f"{perspective_key}|ws-stream|simulation",
                            _json_compact(ws_cache_payload).encode("utf-8"),
                        )
                        _simulation_http_disk_cache_store(
                            self.part_root,
                            perspective=perspective_key,
                            body=_json_compact(ws_cache_payload).encode("utf-8"),
                        )
                        last_ws_stream_cache_store = now_monotonic
                        last_simulation_full_broadcast = now_monotonic
                        last_projection_delta_broadcast = now_monotonic
                        last_graph_position_broadcast = now_monotonic
                        last_simulation_cache_refresh = now_monotonic
                        last_sim_tick = now_monotonic

                if (
                    not use_cached_snapshots
                ) and now_monotonic - last_catalog_refresh >= CATALOG_REFRESH_SECONDS:
                    catalog, queue_snapshot, _, influence_snapshot, _ = (
                        self._runtime_catalog(
                            perspective=perspective_key,
                            include_projection=False,
                        )
                    )
                    last_catalog_refresh = now_monotonic

                if (
                    (not use_cached_snapshots)
                    and now_monotonic - last_runtime_guard_broadcast
                    >= _RUNTIME_GUARD_HEARTBEAT_SECONDS
                ):
                    runtime_health = _runtime_health_payload(self.part_root)
                    runtime_guard = runtime_health.get("guard", {})
                    send_ws(
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
                sim_tick_interval = (
                    SIM_TICK_SECONDS
                    if use_cached_snapshots
                    else (SIM_TICK_SECONDS * load_scale)
                )
                docker_refresh_interval = (
                    DOCKER_SIMULATION_WS_REFRESH_SECONDS * load_scale
                )
                simulation_guard_skip = (
                    _RUNTIME_GUARD_SKIP_SIMULATION_ON_CRITICAL
                    and guard_mode == "critical"
                )

                if (
                    catalog_events_enabled
                    and not effective_skip_catalog_bootstrap
                    and now_monotonic - last_catalog_broadcast
                    >= catalog_broadcast_interval
                ):
                    catalog_payload = _simulation_http_trim_catalog(catalog)
                    _, mix_meta = build_mix_stream(catalog, self.vault_root)
                    send_ws(
                        {
                            "type": "catalog",
                            "catalog": catalog_payload,
                            "mix": mix_meta,
                        }
                    )
                    last_catalog_broadcast = now_monotonic

                if now_monotonic - last_sim_tick >= sim_tick_interval:
                    sim_tick_cycle_started = time.perf_counter()
                    tick_frame_start = time.monotonic()
                    tick_budget_ms = float(sim_tick_interval) * 1000.0

                    def tick_elapsed_ms() -> float:
                        return (time.monotonic() - tick_frame_start) * 1000.0

                    def tick_slack_ms() -> float:
                        return tick_budget_ms - tick_elapsed_ms()

                    governor_graph_heartbeat_scale = 1.0
                    ingestion_pressure = 0.0
                    ws_particle_max = max(
                        24,
                        min(_SIMULATION_WS_STREAM_PARTICLE_MAX, governor_particle_cap),
                    )
                    effective_particle_payload_key = particle_payload_key
                    if use_cached_snapshots:
                        cache_refresh_due = (
                            _SIMULATION_WS_CACHE_REFRESH_SECONDS <= 0.0
                            or (
                                now_monotonic - last_simulation_cache_refresh
                                >= (_SIMULATION_WS_CACHE_REFRESH_SECONDS * load_scale)
                            )
                        )
                        if cache_refresh_due:
                            refresh_motion_state = (
                                _simulation_ws_capture_particle_motion_state(
                                    simulation_payload
                                )
                            )
                            cached_payload = _simulation_ws_load_cached_payload(
                                part_root=self.part_root,
                                perspective=perspective_key,
                                payload_mode=payload_mode_key,
                            )
                            if cached_payload is not None:
                                simulation_payload, projection = cached_payload
                                if refresh_motion_state:
                                    _simulation_ws_restore_particle_motion_state(
                                        simulation_payload,
                                        refresh_motion_state,
                                    )
                                if payload_mode_key == "full":
                                    simulation_delta_payload = _build_ws_delta_payload(
                                        simulation_payload
                                    )
                                else:
                                    simulation_delta_payload = simulation_payload
                                stream_particles = (
                                    _simulation_ws_extract_stream_particles(
                                        simulation_delta_payload
                                    )
                                )
                            if (
                                payload_mode_key != "full"
                                and _SIMULATION_WS_BOOTSTRAP_REQUIRE_LIVE_REBUILD
                                and (
                                    cached_payload is None
                                    or (not stream_particles)
                                    or _simulation_ws_payload_missing_daimoi_summary(
                                        simulation_delta_payload
                                    )
                                    or _simulation_ws_payload_missing_graph_payload(
                                        simulation_payload
                                    )
                                )
                            ):
                                (
                                    simulation_payload,
                                    simulation_delta_payload,
                                    projection,
                                ) = rebuild_live_simulation_payload()
                                if refresh_motion_state:
                                    _simulation_ws_restore_particle_motion_state(
                                        simulation_payload,
                                        refresh_motion_state,
                                    )
                                    if (
                                        simulation_delta_payload
                                        is not simulation_payload
                                    ):
                                        _simulation_ws_restore_particle_motion_state(
                                            simulation_delta_payload,
                                            refresh_motion_state,
                                        )
                                stream_particles = (
                                    _simulation_ws_extract_stream_particles(
                                        simulation_delta_payload
                                    )
                                )
                            cached_timestamp = str(
                                simulation_payload.get("timestamp", "") or ""
                            )
                            full_snapshot_due = (
                                _SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS <= 0.0
                                or (
                                    now_monotonic - last_simulation_full_broadcast
                                    >= _SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS
                                )
                            )
                            if (
                                full_snapshot_due
                                and cached_timestamp
                                and cached_timestamp != last_cached_simulation_timestamp
                            ):
                                full_snapshot_budget_ok = (
                                    tick_slack_ms() >= ws_full_snapshot_min_slack_ms
                                )
                                inbox_active = bool(_RUNTIME_INBOX_SYNC_LOCK.locked())
                                if full_snapshot_budget_ok and not inbox_active:
                                    dynamics_state = simulation_payload.get(
                                        "presence_dynamics",
                                        {},
                                    )
                                    if isinstance(dynamics_state, dict):
                                        rows_state = dynamics_state.get(
                                            "field_particles",
                                            [],
                                        )
                                        if isinstance(rows_state, list):
                                            dynamics_state["field_particles"] = list(
                                                rows_state[
                                                    :_SIMULATION_WS_STREAM_PARTICLE_MAX
                                                ]
                                            )

                                        graph_state = dynamics_state.get(
                                            "graph_node_positions"
                                        )
                                        if (
                                            isinstance(graph_state, dict)
                                            and len(graph_state)
                                            > ws_graph_pos_max_default
                                        ):
                                            capped_graph: dict[str, Any] = {}
                                            for key, value in graph_state.items():
                                                capped_graph[key] = value
                                                if (
                                                    len(capped_graph)
                                                    >= ws_graph_pos_max_default
                                                ):
                                                    break
                                            dynamics_state["graph_node_positions"] = (
                                                capped_graph
                                            )
                                            simulation_payload[
                                                "graph_node_positions_truncated"
                                            ] = True
                                            simulation_payload[
                                                "graph_node_positions_total"
                                            ] = len(graph_state)

                                        if tick_slack_ms() < (
                                            ws_full_snapshot_min_slack_ms + 4.0
                                        ):
                                            dynamics_state.pop("nooi_field", None)

                                        simulation_payload["presence_dynamics"] = (
                                            dynamics_state
                                        )

                                    send_ws(
                                        {
                                            "type": "simulation",
                                            "simulation": simulation_payload,
                                            "projection": projection,
                                        }
                                    )
                                    last_simulation_full_broadcast = now_monotonic
                                    last_projection_delta_broadcast = now_monotonic
                                    last_simulation_for_delta = simulation_delta_payload
                                    last_cached_simulation_timestamp = cached_timestamp
                            last_simulation_cache_refresh = now_monotonic

                        inbox_active = bool(_RUNTIME_INBOX_SYNC_LOCK.locked())
                        inbox_state = (
                            catalog.get("eta_mu_inbox", {})
                            if isinstance(catalog, dict)
                            else {}
                        )
                        if not isinstance(inbox_state, dict):
                            inbox_state = {}
                        inbox_pending = max(
                            0,
                            int(_safe_float(inbox_state.get("pending_count", 0), 0.0)),
                        )
                        inbox_pending_soft = max(
                            1.0,
                            _safe_float(
                                os.getenv("ETA_MU_INBOX_PENDING_SOFT", "64") or "64",
                                64.0,
                            ),
                        )
                        inbox_pending_pressure = max(
                            0.0,
                            min(1.0, inbox_pending / inbox_pending_soft),
                        )

                        dynamics_snapshot = simulation_payload.get(
                            "presence_dynamics", {}
                        )
                        if not isinstance(dynamics_snapshot, dict):
                            dynamics_snapshot = {}
                        resource_heartbeat_snapshot = dynamics_snapshot.get(
                            "resource_heartbeat",
                            {},
                        )
                        if not isinstance(resource_heartbeat_snapshot, dict):
                            resource_heartbeat_snapshot = {}
                        resource_devices_snapshot = resource_heartbeat_snapshot.get(
                            "devices",
                            {},
                        )
                        if not isinstance(resource_devices_snapshot, dict):
                            resource_devices_snapshot = {}
                        gpu1_state = resource_devices_snapshot.get("gpu1", {})
                        if not isinstance(gpu1_state, dict):
                            gpu1_state = {}
                        gpu2_state = resource_devices_snapshot.get("gpu2", {})
                        if not isinstance(gpu2_state, dict):
                            gpu2_state = {}
                        npu0_state = resource_devices_snapshot.get("npu0", {})
                        if not isinstance(npu0_state, dict):
                            npu0_state = {}

                        gpu_utilization_pressure = (
                            max(
                                _safe_float(gpu1_state.get("utilization", 0.0), 0.0),
                                _safe_float(gpu2_state.get("utilization", 0.0), 0.0),
                            )
                            / 100.0
                        )
                        npu_utilization_pressure = (
                            _safe_float(npu0_state.get("utilization", 0.0), 0.0) / 100.0
                        )

                        ingestion_pressure = max(
                            0.0,
                            min(
                                1.0,
                                max(
                                    1.0 if inbox_active else 0.0,
                                    inbox_pending_pressure,
                                    gpu_utilization_pressure,
                                    npu_utilization_pressure,
                                ),
                            ),
                        )

                        slack_ms_before_sim = tick_slack_ms()
                        ws_particle_max = max(
                            24,
                            min(
                                _SIMULATION_WS_STREAM_PARTICLE_MAX,
                                governor_particle_cap,
                            ),
                        )
                        if ingestion_pressure >= 0.7:
                            ws_particle_max = min(ws_particle_max, 96)
                        if slack_ms_before_sim <= 4.0:
                            ws_particle_max = min(ws_particle_max, 64)
                        elif slack_ms_before_sim <= 10.0:
                            ws_particle_max = min(ws_particle_max, 96)

                        effective_particle_payload_key = particle_payload_key
                        if ingestion_pressure >= 0.7 or slack_ms_before_sim <= 4.0:
                            effective_particle_payload_key = "lite"

                        tick_policy = {
                            "tick_budget_ms": tick_budget_ms,
                            "slack_ms": slack_ms_before_sim,
                            "ingestion_pressure": ingestion_pressure,
                            "ws_particle_max": ws_particle_max,
                            "guard_mode": guard_mode,
                        }

                        tick_timestamp = datetime.now(timezone.utc).isoformat()
                        if tick_governor is not None:
                            (
                                pending_count,
                                bytes_pending,
                                embedding_backlog,
                                disk_queue_depth,
                            ) = _simulation_ws_governor_ingestion_signal(catalog)
                            tick_governor.update_ingestion_status(
                                pending_count,
                                bytes_pending,
                                embedding_backlog=embedding_backlog,
                                disk_queue_depth=disk_queue_depth,
                            )

                            if (
                                now_monotonic - last_governor_resource_refresh
                                >= _SIMULATION_WS_TICK_GOVERNOR_RESOURCE_REFRESH_SECONDS
                            ):
                                mem_pressure, disk_pressure = (
                                    _simulation_ws_governor_stock_pressure(
                                        self.part_root
                                    )
                                )
                                tick_governor.update_stock_pressure(
                                    mem_pressure=mem_pressure,
                                    disk_pressure=disk_pressure,
                                )
                                last_governor_resource_refresh = now_monotonic

                            def _run_sim_tick() -> None:
                                advance_simulation_field_particles(
                                    simulation_payload,
                                    dt_seconds=sim_tick_interval,
                                    now_seconds=now_monotonic,
                                    policy=tick_policy,
                                )

                            def _run_sim_tick_degraded() -> None:
                                dynamics = simulation_payload.get(
                                    "presence_dynamics", {}
                                )
                                rows = (
                                    dynamics.get("field_particles", [])
                                    if isinstance(dynamics, dict)
                                    else []
                                )
                                if (
                                    not isinstance(dynamics, dict)
                                    or not isinstance(rows, list)
                                    or len(rows) < 2
                                ):
                                    _run_sim_tick()
                                    return

                                total_rows = len(rows)
                                pressure = max(
                                    0.0,
                                    min(
                                        1.0,
                                        _safe_float(
                                            tick_governor.ingestion.get(
                                                "pressure",
                                                0.0,
                                            ),
                                            0.0,
                                        ),
                                    ),
                                )
                                update_ratio = max(
                                    0.34,
                                    min(0.8, 0.62 - (0.25 * pressure)),
                                )
                                target_updates = min(
                                    total_rows,
                                    max(32, int(round(total_rows * update_ratio))),
                                )
                                if target_updates >= total_rows:
                                    _run_sim_tick()
                                    return

                                stride = max(
                                    1,
                                    int(
                                        math.ceil(
                                            total_rows / float(max(1, target_updates))
                                        )
                                    ),
                                )
                                phase = int(now_monotonic * 1000.0) % stride
                                selected_indexes = list(
                                    range(phase, total_rows, stride)
                                )
                                selected_index_set = set(selected_indexes)
                                if len(selected_indexes) < target_updates:
                                    for index in range(total_rows):
                                        if len(selected_indexes) >= target_updates:
                                            break
                                        if index in selected_index_set:
                                            continue
                                        selected_indexes.append(index)
                                        selected_index_set.add(index)
                                selected_indexes = sorted(
                                    selected_indexes[:target_updates]
                                )

                                subset_rows = [
                                    dict(rows[index])
                                    if isinstance(rows[index], dict)
                                    else {}
                                    for index in selected_indexes
                                ]
                                degraded_simulation = dict(simulation_payload)
                                degraded_dynamics = dict(dynamics)
                                degraded_dynamics["field_particles"] = subset_rows
                                degraded_simulation["presence_dynamics"] = (
                                    degraded_dynamics
                                )

                                advance_simulation_field_particles(
                                    degraded_simulation,
                                    dt_seconds=sim_tick_interval,
                                    now_seconds=now_monotonic,
                                    policy=tick_policy,
                                )

                                updated_subset = degraded_dynamics.get(
                                    "field_particles", []
                                )
                                if isinstance(updated_subset, list):
                                    for offset, index in enumerate(selected_indexes):
                                        if offset >= len(updated_subset):
                                            break
                                        updated_row = updated_subset[offset]
                                        if isinstance(updated_row, dict):
                                            rows[index] = updated_row

                                dynamics["field_particles"] = rows
                                dynamics["governor_tick_mode"] = "degraded"
                                dynamics["governor_tick_update_ratio"] = round(
                                    target_updates / float(max(1, total_rows)),
                                    4,
                                )
                                simulation_payload["presence_dynamics"] = dynamics

                            def _run_tick_filler() -> None:
                                dynamics = simulation_payload.get(
                                    "presence_dynamics", {}
                                )
                                rows = (
                                    dynamics.get("field_particles", [])
                                    if isinstance(dynamics, dict)
                                    else []
                                )
                                if isinstance(rows, list) and rows:
                                    _simulation_ws_lite_field_particles(
                                        rows,
                                        max_rows=max(
                                            24,
                                            min(governor_particle_cap, len(rows)),
                                        ),
                                    )

                            dynamics_for_cost = simulation_payload.get(
                                "presence_dynamics",
                                {},
                            )
                            field_rows_for_cost = (
                                dynamics_for_cost.get("field_particles", [])
                                if isinstance(dynamics_for_cost, dict)
                                else []
                            )
                            field_count_for_cost = (
                                len(field_rows_for_cost)
                                if isinstance(field_rows_for_cost, list)
                                else 0
                            )
                            mem_cost = max(
                                0.02,
                                min(0.72, 0.03 + (field_count_for_cost / 1900.0)),
                            )
                            overhead_ms = (
                                time.perf_counter() - sim_tick_cycle_started
                            ) * 1000.0

                            sim_work_estimate = _simulation_ws_governor_estimate_work(
                                simulation_payload
                            )

                            sim_packet = Packet(
                                id=f"sim.tick:{int(now_monotonic * 1000.0)}",
                                work=sim_work_estimate,
                                value=1000.0,
                                deadline="tick",
                                executable=_run_sim_tick,
                                lane_efficiency={
                                    LaneType.CPU: 1.0,
                                    LaneType.RTX: 0.22,
                                    LaneType.ARC: 0.2,
                                    LaneType.NPU: 0.12,
                                },
                                cost_vector={
                                    "mem": mem_cost,
                                    "disk": 0.03,
                                },
                                tags=["simulation", "tick"],
                                family="simulation",
                                allow_degrade=True,
                                degraded_executable=_run_sim_tick_degraded,
                                degrade_work_factor=0.55,
                                degrade_value_factor=0.9,
                            )

                            filler_packet = Packet(
                                id=f"sim.filler.prefetch:{int(now_monotonic * 1000.0)}",
                                work=max(1.0, sim_work_estimate * 0.28),
                                value=4.5,
                                deadline="best-effort",
                                executable=_run_tick_filler,
                                lane_efficiency={
                                    LaneType.CPU: 1.0,
                                    LaneType.RTX: 0.12,
                                    LaneType.ARC: 0.1,
                                    LaneType.NPU: 0.06,
                                },
                                cost_vector={
                                    "mem": min(0.3, mem_cost * 0.45),
                                    "disk": 0.01,
                                },
                                tags=["simulation", "filler", "cache-warm"],
                                family="filler",
                            )

                            governor_result = tick_governor.tick(
                                [sim_packet, filler_packet],
                                dt_ms=(sim_tick_interval * 1000.0),
                                overhead_ms=overhead_ms,
                            )

                            packet_ran = any(
                                str(row.get("packet_id", "")) == sim_packet.id
                                and str(row.get("status", "")) == "ok"
                                for row in governor_result.receipts
                                if isinstance(row, dict)
                            )
                            if not packet_ran:
                                dynamics_state = simulation_payload.get(
                                    "presence_dynamics",
                                    {},
                                )
                                if isinstance(dynamics_state, dict):
                                    dynamics_state["governor_tick_mode"] = "deferred"
                                    dynamics_state["governor_tick_update_ratio"] = 0.0
                                    simulation_payload["presence_dynamics"] = (
                                        dynamics_state
                                    )
                            elif governor_result.required_downgraded <= 0:
                                dynamics_state = simulation_payload.get(
                                    "presence_dynamics",
                                    {},
                                )
                                if isinstance(dynamics_state, dict):
                                    dynamics_state["governor_tick_mode"] = "full"
                                    dynamics_state["governor_tick_update_ratio"] = 1.0
                                    simulation_payload["presence_dynamics"] = (
                                        dynamics_state
                                    )

                            governor_particle_cap = _simulation_ws_governor_particle_cap(
                                _SIMULATION_WS_STREAM_PARTICLE_MAX,
                                fidelity_signal=governor_result.fidelity_signal,
                                ingestion_pressure=governor_result.ingestion_pressure,
                            )
                            governor_graph_heartbeat_scale = (
                                _simulation_ws_governor_graph_heartbeat_scale(
                                    governor_result.fidelity_signal
                                )
                            )
                        else:
                            advance_simulation_field_particles(
                                simulation_payload,
                                dt_seconds=sim_tick_interval,
                                now_seconds=now_monotonic,
                                policy=tick_policy,
                            )
                        simulation_payload["timestamp"] = tick_timestamp
                        simulation_payload["generated_at"] = tick_timestamp
                        dynamics_state = simulation_payload.get("presence_dynamics", {})
                        summary_state = (
                            dynamics_state.get("daimoi_probabilistic", {})
                            if isinstance(dynamics_state, dict)
                            else {}
                        )
                        summary_ready = isinstance(summary_state, dict) and bool(
                            summary_state
                        )
                        daimoi_slack_ms = tick_slack_ms()
                        allow_daimoi_refresh = (
                            daimoi_slack_ms
                            >= _SIMULATION_WS_DAIMOI_METRICS_MIN_SLACK_MS
                            and ingestion_pressure < 0.85
                        )
                        live_metrics_due = (
                            _SIMULATION_WS_DAIMOI_LIVE_METRICS_MIN_INTERVAL_SECONDS
                            <= 0.0
                            or (
                                now_monotonic - last_daimoi_live_metrics_refresh
                                >= _SIMULATION_WS_DAIMOI_LIVE_METRICS_MIN_INTERVAL_SECONDS
                            )
                        )
                        graph_variability_due = (
                            _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_MIN_INTERVAL_SECONDS
                            <= 0.0
                            or (
                                now_monotonic - last_daimoi_graph_variability_refresh
                                >= _SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_MIN_INTERVAL_SECONDS
                            )
                        )
                        include_live_metrics = (not summary_ready) or (
                            allow_daimoi_refresh and live_metrics_due
                        )
                        include_graph_variability = (not summary_ready) or (
                            allow_daimoi_refresh and graph_variability_due
                        )
                        _simulation_ws_ensure_daimoi_summary(
                            simulation_payload,
                            include_live_metrics=include_live_metrics,
                            include_graph_variability=include_graph_variability,
                        )
                        if include_live_metrics:
                            last_daimoi_live_metrics_refresh = now_monotonic
                        if include_graph_variability:
                            last_daimoi_graph_variability_refresh = now_monotonic
                        dynamics = simulation_payload.get("presence_dynamics", {})
                        if isinstance(dynamics, dict):
                            dynamics["generated_at"] = tick_timestamp
                            simulation_payload["presence_dynamics"] = dynamics

                        cache_source_payload = (
                            simulation_delta_payload
                            if payload_mode_key == "full"
                            else simulation_payload
                        )
                        ws_cache_payload = dict(cache_source_payload)
                        ws_cache_payload["projection"] = projection
                        cache_dynamics = simulation_payload.get("presence_dynamics", {})
                        cache_stream_particles = (
                            cache_dynamics.get("field_particles", [])
                            if isinstance(cache_dynamics, dict)
                            and isinstance(
                                cache_dynamics.get("field_particles", []), list
                            )
                            else []
                        )
                        cache_total = max(
                            0,
                            int(_safe_float(simulation_payload.get("total", 0), 0.0)),
                        )
                        cache_store_due = (
                            _SIMULATION_WS_CACHE_REFRESH_SECONDS <= 0.0
                            or (
                                now_monotonic - last_ws_stream_cache_store
                                >= max(0.25, _SIMULATION_WS_CACHE_REFRESH_SECONDS)
                            )
                        )
                        if (
                            cache_stream_particles or cache_total > 0
                        ) and cache_store_due:
                            _simulation_http_cache_store(
                                f"{perspective_key}|ws-stream|simulation",
                                _json_compact(ws_cache_payload).encode("utf-8"),
                            )
                            last_ws_stream_cache_store = now_monotonic

                        dynamics = simulation_payload.get("presence_dynamics", {})
                        stream_particles = []
                        if isinstance(dynamics, dict) and isinstance(
                            dynamics.get("field_particles"), list
                        ):
                            stream_particles = dynamics.get("field_particles", [])
                        graph_node_positions = (
                            dynamics.get("graph_node_positions", {})
                            if isinstance(dynamics, dict)
                            else {}
                        )
                        presence_anchor_positions = (
                            dynamics.get("presence_anchor_positions", {})
                            if isinstance(dynamics, dict)
                            else {}
                        )
                        tick_elapsed_ms_value = tick_elapsed_ms()
                        slack_ms_value = tick_slack_ms()
                        tick_patch: dict[str, Any] = {
                            "timestamp": tick_timestamp,
                            "tick_elapsed_ms": round(tick_elapsed_ms_value, 4),
                            "slack_ms": round(slack_ms_value, 4),
                            "ingestion_pressure": round(ingestion_pressure, 4),
                            "ws_particle_max": int(ws_particle_max),
                            "particle_payload_mode": effective_particle_payload_key,
                        }
                        tick_changed_keys = [
                            "timestamp",
                            "tick_elapsed_ms",
                            "slack_ms",
                            "ingestion_pressure",
                            "ws_particle_max",
                            "particle_payload_mode",
                        ]
                        dynamics_patch: dict[str, Any] = {}
                        if stream_particles:
                            tick_particles = (
                                _simulation_ws_lite_field_particles(
                                    stream_particles,
                                    max_rows=ws_particle_max,
                                )
                                if effective_particle_payload_key == "lite"
                                else []
                            )
                            if effective_particle_payload_key != "lite":
                                for row in stream_particles:
                                    if len(tick_particles) >= ws_particle_max:
                                        break
                                    if isinstance(row, dict):
                                        tick_particles.append(dict(row))
                            dynamics_patch["field_particles"] = tick_particles
                            tick_changed_keys.append(
                                "presence_dynamics.field_particles"
                            )
                        daimoi_summary = (
                            dynamics.get("daimoi_probabilistic", {})
                            if isinstance(dynamics, dict)
                            else {}
                        )
                        if isinstance(daimoi_summary, dict) and daimoi_summary:
                            anti_payload = (
                                daimoi_summary.get("anti_clump", {})
                                if isinstance(
                                    daimoi_summary.get("anti_clump", {}), dict
                                )
                                else {}
                            )
                            dynamics_patch["daimoi_probabilistic"] = {
                                "clump_score": _ws_clamp01(
                                    _safe_float(
                                        daimoi_summary.get("clump_score", 0.0), 0.0
                                    )
                                ),
                                "anti_clump_drive": max(
                                    -1.0,
                                    min(
                                        1.0,
                                        _safe_float(
                                            daimoi_summary.get("anti_clump_drive", 0.0),
                                            0.0,
                                        ),
                                    ),
                                ),
                                "anti_clump": anti_payload,
                            }
                            tick_changed_keys.append(
                                "presence_dynamics.daimoi_probabilistic"
                            )
                        graph_position_heartbeat_due = (
                            _SIMULATION_WS_GRAPH_POSITION_HEARTBEAT_SECONDS <= 0.0
                            or (
                                now_monotonic - last_graph_position_broadcast
                                >= (
                                    _SIMULATION_WS_GRAPH_POSITION_HEARTBEAT_SECONDS
                                    * max(0.1, governor_graph_heartbeat_scale)
                                )
                            )
                        ) and tick_slack_ms() > 2.0
                        graph_pos_cap = int(ws_graph_pos_max_default)
                        if ingestion_pressure >= 0.7:
                            graph_pos_cap = min(graph_pos_cap, 512)
                        if slack_ms_value <= 4.0:
                            graph_pos_cap = min(graph_pos_cap, 512)
                        elif slack_ms_value <= 10.0:
                            graph_pos_cap = min(graph_pos_cap, 1024)
                        if (
                            isinstance(graph_node_positions, dict)
                            and graph_node_positions
                            and graph_position_heartbeat_due
                        ):
                            if len(graph_node_positions) > graph_pos_cap:
                                capped_graph_positions: dict[str, Any] = {}
                                for key, value in graph_node_positions.items():
                                    capped_graph_positions[key] = value
                                    if len(capped_graph_positions) >= graph_pos_cap:
                                        break
                                dynamics_patch["graph_node_positions"] = (
                                    capped_graph_positions
                                )
                                tick_patch["graph_node_positions_truncated"] = True
                                tick_patch["graph_node_positions_total"] = len(
                                    graph_node_positions
                                )
                                tick_changed_keys.append(
                                    "graph_node_positions_truncated"
                                )
                                tick_changed_keys.append("graph_node_positions_total")
                            else:
                                dynamics_patch["graph_node_positions"] = (
                                    graph_node_positions
                                )
                            tick_changed_keys.append(
                                "presence_dynamics.graph_node_positions"
                            )
                        if (
                            isinstance(presence_anchor_positions, dict)
                            and presence_anchor_positions
                            and graph_position_heartbeat_due
                        ):
                            anchor_pos_cap = max(64, min(graph_pos_cap, 2048))
                            if len(presence_anchor_positions) > anchor_pos_cap:
                                capped_anchor_positions: dict[str, Any] = {}
                                for key, value in presence_anchor_positions.items():
                                    capped_anchor_positions[key] = value
                                    if len(capped_anchor_positions) >= anchor_pos_cap:
                                        break
                                dynamics_patch["presence_anchor_positions"] = (
                                    capped_anchor_positions
                                )
                                tick_patch["presence_anchor_positions_truncated"] = True
                                tick_patch["presence_anchor_positions_total"] = len(
                                    presence_anchor_positions
                                )
                                tick_changed_keys.append(
                                    "presence_anchor_positions_truncated"
                                )
                                tick_changed_keys.append(
                                    "presence_anchor_positions_total"
                                )
                            else:
                                dynamics_patch["presence_anchor_positions"] = (
                                    presence_anchor_positions
                                )
                            tick_changed_keys.append(
                                "presence_dynamics.presence_anchor_positions"
                            )
                        if graph_position_heartbeat_due and (
                            (
                                isinstance(graph_node_positions, dict)
                                and bool(graph_node_positions)
                            )
                            or (
                                isinstance(presence_anchor_positions, dict)
                                and bool(presence_anchor_positions)
                            )
                        ):
                            last_graph_position_broadcast = now_monotonic
                        if dynamics_patch:
                            tick_patch["presence_dynamics"] = dynamics_patch

                        if stream_mode == "workers":
                            simulation_worker_seq += 1
                            send_ws(
                                {
                                    "type": "simulation_delta",
                                    "stream": "workers",
                                    "worker_id": "sim-core",
                                    "worker_seq": simulation_worker_seq,
                                    "delta": {
                                        "record": "eta-mu.simulation-worker-delta.v1",
                                        "schema_version": "simulation.worker-delta.v1",
                                        "generated_at": tick_timestamp,
                                        "worker_id": "sim-core",
                                        "worker_seq": simulation_worker_seq,
                                        "previous_fingerprint": "",
                                        "current_fingerprint": "",
                                        "changed_keys": tick_changed_keys,
                                        "has_changes": True,
                                        "patch": tick_patch,
                                    },
                                }
                            )
                        else:
                            send_ws(
                                {
                                    "type": "simulation_delta",
                                    "delta": {
                                        "record": "eta-mu.simulation-delta.v1",
                                        "schema_version": "simulation.delta.v1",
                                        "generated_at": tick_timestamp,
                                        "previous_fingerprint": "",
                                        "current_fingerprint": "",
                                        "changed_keys": tick_changed_keys,
                                        "has_changes": True,
                                        "patch": tick_patch,
                                    },
                                }
                            )

                        if tick_slack_ms() > 1.0:
                            maybe_send_muse_events(now_monotonic)
                        last_sim_tick = now_monotonic
                        continue

                    if simulation_guard_skip:
                        send_ws(
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
                            include_unified_graph=payload_mode_key == "full",
                        )
                        simulation_snapshot_payload = _build_ws_snapshot_payload(
                            simulation
                        )
                        simulation_delta_payload = _build_ws_delta_payload(simulation)
                        delta = build_simulation_delta(
                            last_simulation_for_delta,
                            simulation_delta_payload,
                        )
                        delta_patch = (
                            delta.get("patch") if isinstance(delta, dict) else None
                        )
                        if isinstance(delta_patch, dict):
                            delta_patch.pop("projection", None)
                        changed_keys = delta.get("changed_keys", [])
                        if isinstance(changed_keys, list):
                            changed_keys = [
                                key for key in changed_keys if key != "projection"
                            ]
                        else:
                            changed_keys = []
                        delta["changed_keys"] = changed_keys
                        delta["has_changes"] = bool(changed_keys)
                        delta_has_changes = bool(delta.get("has_changes", False))
                        full_snapshot_due = (
                            _SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS <= 0.0
                            or (
                                now_monotonic - last_simulation_full_broadcast
                                >= _SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS
                            )
                        )
                        full_snapshot_budget_ok = (
                            tick_slack_ms() >= ws_full_snapshot_min_slack_ms
                        )
                        if full_snapshot_due and full_snapshot_budget_ok:
                            send_ws(
                                {
                                    "type": "simulation",
                                    "simulation": simulation_snapshot_payload,
                                    "projection": projection,
                                }
                            )
                            last_simulation_full_broadcast = now_monotonic
                            last_projection_delta_broadcast = now_monotonic
                        elif delta_has_changes:
                            if stream_mode == "workers":
                                worker_delta_rows = (
                                    _simulation_ws_split_delta_by_worker(delta)
                                )
                                for worker_row in worker_delta_rows:
                                    simulation_worker_seq += 1
                                    send_ws(
                                        {
                                            "type": "simulation_delta",
                                            "stream": "workers",
                                            "worker_id": worker_row.get(
                                                "worker_id", "sim-misc"
                                            ),
                                            "worker_seq": simulation_worker_seq,
                                            "delta": {
                                                "record": "eta-mu.simulation-worker-delta.v1",
                                                "schema_version": "simulation.worker-delta.v1",
                                                "generated_at": datetime.now(
                                                    timezone.utc
                                                ).isoformat(),
                                                "worker_id": worker_row.get(
                                                    "worker_id", "sim-misc"
                                                ),
                                                "worker_seq": simulation_worker_seq,
                                                "previous_fingerprint": delta.get(
                                                    "previous_fingerprint", ""
                                                ),
                                                "current_fingerprint": delta.get(
                                                    "current_fingerprint", ""
                                                ),
                                                "changed_keys": worker_row.get(
                                                    "changed_keys", []
                                                ),
                                                "has_changes": bool(
                                                    worker_row.get("changed_keys", [])
                                                ),
                                                "patch": worker_row.get("patch", {}),
                                            },
                                        }
                                    )
                            else:
                                send_ws(
                                    {
                                        "type": "simulation_delta",
                                        "delta": delta,
                                    }
                                )

                        projection_heartbeat_due = (
                            _SIMULATION_WS_PROJECTION_HEARTBEAT_SECONDS > 0.0
                            and (
                                now_monotonic - last_projection_delta_broadcast
                                >= _SIMULATION_WS_PROJECTION_HEARTBEAT_SECONDS
                            )
                        )
                        if (not full_snapshot_due) and projection_heartbeat_due:
                            projection_patch: dict[str, Any] = {
                                "projection": projection,
                            }
                            timestamp_value = simulation_delta_payload.get("timestamp")
                            if timestamp_value is not None:
                                projection_patch["timestamp"] = timestamp_value
                            send_ws(
                                {
                                    "type": "simulation_delta",
                                    "stream": "projection",
                                    "worker_id": "sim-projection",
                                    "delta": {
                                        "record": "eta-mu.simulation-delta.v1",
                                        "schema_version": "simulation.delta.v1",
                                        "generated_at": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "previous_fingerprint": delta.get(
                                            "previous_fingerprint", ""
                                        ),
                                        "current_fingerprint": delta.get(
                                            "current_fingerprint", ""
                                        ),
                                        "changed_keys": ["projection"],
                                        "has_changes": True,
                                        "patch": projection_patch,
                                    },
                                }
                            )
                            last_projection_delta_broadcast = now_monotonic
                        last_simulation_for_delta = simulation_delta_payload

                    if tick_slack_ms() > 1.0:
                        maybe_send_muse_events(now_monotonic)
                    last_sim_tick = now_monotonic

                if (
                    not use_cached_snapshots
                ) and now_monotonic - last_docker_refresh >= docker_refresh_interval:
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
                        send_ws(
                            {
                                "type": "docker_simulations",
                                "docker": docker_snapshot,
                            }
                        )
                        last_docker_broadcast = now_monotonic
                        last_docker_fingerprint = docker_fingerprint
                    last_docker_refresh = now_monotonic

                # Non-blocking check for incoming client data
                ready = poll.poll(10)
                if ready:
                    if not _consume_ws_client_frame(self.connection):
                        break
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
            wire_mode = str(
                params.get("wire", [params.get("ws_wire", [""])[0]])[0] or ""
            )
            if stream in {"docker", "meta", "simulations"}:
                self._handle_docker_websocket(wire_mode=wire_mode)
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
            delta_stream = str(
                params.get("delta_stream", [params.get("sim_stream", [""])[0]])[0] or ""
            )
            payload_mode = str(
                params.get(
                    "simulation_payload",
                    [params.get("sim_payload", [""])[0]],
                )[0]
                or ""
            )
            particle_payload_mode = str(
                params.get(
                    "particle_payload",
                    [params.get("particle_mode", [""])[0]],
                )[0]
                or _SIMULATION_WS_PARTICLE_PAYLOAD_MODE_DEFAULT
            )
            ws_chunk = _safe_bool_query(
                str(
                    params.get(
                        "ws_chunk",
                        [params.get("chunk", [""])[0]],
                    )[0]
                    or ""
                ),
                default=_SIMULATION_WS_CHUNK_ENABLED,
            )
            catalog_events = _safe_bool_query(
                str(
                    params.get(
                        "catalog_events",
                        [params.get("catalog_ws", [""])[0]],
                    )[0]
                    or ""
                ),
                default=not _SIMULATION_WS_SKIP_CATALOG_BOOTSTRAP,
            )
            skip_catalog_bootstrap = _safe_bool_query(
                str(
                    params.get(
                        "skip_catalog_bootstrap",
                        [params.get("skip_catalog", [""])[0]],
                    )[0]
                    or ""
                ),
                default=_SIMULATION_WS_SKIP_CATALOG_BOOTSTRAP,
            )
            self._handle_websocket(
                perspective=perspective,
                delta_stream_mode=delta_stream,
                wire_mode=wire_mode,
                payload_mode=payload_mode,
                particle_payload_mode=particle_payload_mode,
                chunk_stream_enabled=ws_chunk,
                catalog_events_enabled=catalog_events,
                skip_catalog_bootstrap=skip_catalog_bootstrap,
            )
            return

        if parsed.path == "/api/voice-lines":
            mode = str(params.get("mode", ["canonical"])[0] or "canonical")
            payload_voice = build_voice_lines(
                "llm" if mode.strip().lower() in {"ollama", "llm"} else "canonical"
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

        if parsed.path == "/api/simulation/bootstrap":
            report = _simulation_bootstrap_snapshot_report()
            self._send_json(
                {
                    "ok": True,
                    "record": "eta-mu.simulation-bootstrap.status.v1",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "job": _simulation_bootstrap_job_snapshot(),
                    "report": report,
                }
            )
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

        if parsed.path == "/api/config":
            module_filter = str(params.get("module", [""])[0] or "").strip().lower()
            payload = _config_payload(module_filter=module_filter)
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.BAD_REQUEST
            self._send_json(payload, status=status)
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
            cached_body = _runtime_catalog_http_cached_body(
                perspective=perspective,
                max_age_seconds=_RUNTIME_CATALOG_HTTP_CACHE_SECONDS,
            )
            if cached_body is not None:
                self._send_bytes(
                    cached_body,
                    "application/json; charset=utf-8",
                    extra_headers={"X-Eta-Mu-Catalog-Fallback": "http-cache"},
                )
                return

            catalog, _, _, _, _ = self._runtime_catalog(perspective=perspective)
            body = _json_compact(_simulation_http_trim_catalog(catalog)).encode("utf-8")
            _runtime_catalog_http_cache_store(
                perspective=perspective,
                body=body,
            )
            self._send_bytes(body, "application/json; charset=utf-8")
            return

        if parsed.path == "/api/catalog/stream":
            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            chunk_rows = _catalog_stream_chunk_rows(
                str(
                    params.get("chunk_rows", [str(_CATALOG_STREAM_CHUNK_ROWS)])[0]
                    or str(_CATALOG_STREAM_CHUNK_ROWS)
                )
            )
            trim_catalog = _safe_bool_query(
                str(
                    params.get(
                        "trim",
                        [params.get("trimmed", ["0"])[0]],
                    )[0]
                    or "0"
                ),
                default=False,
            )
            self._send_catalog_stream(
                perspective=perspective,
                chunk_rows=chunk_rows,
                trim_catalog=trim_catalog,
            )
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
            payload_mode = _simulation_ws_normalize_payload_mode(
                str(params.get("payload", ["full"])[0] or "full")
            )
            compact_response_mode = bool(compact_response or payload_mode == "trimmed")

            def _send_simulation_response(
                body: bytes,
                *,
                cache_key_for_compact: str = "",
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
                        cached_compact = _simulation_http_compact_cached_body(
                            cache_key=compact_cache_key,
                            max_age_seconds=_SIMULATION_HTTP_CACHE_SECONDS,
                        )
                        if cached_compact is not None:
                            response_body = cached_compact
                        else:
                            response_body = _simulation_http_compact_response_body(body)
                            _simulation_http_compact_cache_store(
                                compact_cache_key,
                                response_body,
                            )
                    else:
                        response_body = _simulation_http_compact_response_body(body)
                self._send_bytes(
                    response_body,
                    "application/json; charset=utf-8",
                    extra_headers=extra_headers,
                )

            if compact_response_mode:
                cached_compact_by_perspective = _simulation_http_compact_cached_body(
                    perspective=perspective,
                    max_age_seconds=_SIMULATION_HTTP_CACHE_SECONDS,
                )
                if cached_compact_by_perspective is not None:
                    self._send_bytes(
                        cached_compact_by_perspective,
                        "application/json; charset=utf-8",
                        extra_headers={
                            "X-Eta-Mu-Simulation-Fallback": "compact-cache",
                        },
                    )
                    return
                stale_compact_body, stale_compact_source = (
                    _simulation_http_compact_stale_fallback_body(
                        part_root=self.part_root,
                        perspective=perspective,
                        max_age_seconds=_SIMULATION_HTTP_COMPACT_STALE_FALLBACK_SECONDS,
                    )
                )
                if stale_compact_body is not None:
                    fallback_source = (
                        str(stale_compact_source or "stale-cache").strip().lower()
                        or "stale-cache"
                    )
                    _send_simulation_response(
                        stale_compact_body,
                        cache_key_for_compact=(
                            f"{perspective}|{fallback_source}|compact-fallback|simulation"
                        ),
                        extra_headers={
                            "X-Eta-Mu-Simulation-Fallback": f"{fallback_source}-compact",
                        },
                    )
                    return

            cache_key = ""
            try:
                if _simulation_http_is_cold_start():
                    cold_disk_body = _simulation_http_disk_cache_load(
                        self.part_root,
                        perspective=perspective,
                        max_age_seconds=_SIMULATION_HTTP_DISK_COLD_START_SECONDS,
                    )
                    if cold_disk_body is not None:
                        _simulation_http_cache_store(
                            f"{perspective}|disk-cold-start|simulation",
                            cold_disk_body,
                        )
                        _send_simulation_response(
                            cold_disk_body,
                            cache_key_for_compact=f"{perspective}|disk-cold-start|simulation",
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
                        cache_key_for_compact=cache_key,
                    )
                    return

                if user_inputs_recent <= 0:
                    disk_cached_body = _simulation_http_disk_cache_load(
                        self.part_root,
                        perspective=perspective,
                        max_age_seconds=max(
                            _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                            _SIMULATION_HTTP_DISK_FALLBACK_MAX_AGE_SECONDS,
                        ),
                    )
                    if disk_cached_body is not None:
                        _simulation_http_cache_store(cache_key, disk_cached_body)
                        _send_simulation_response(
                            disk_cached_body,
                            cache_key_for_compact=cache_key,
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
                            cache_key_for_compact=cache_key,
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
                    wait_seconds = (
                        _SIMULATION_HTTP_COMPACT_BUILD_WAIT_SECONDS
                        if compact_response_mode
                        else _SIMULATION_HTTP_BUILD_WAIT_SECONDS
                    )
                    inflight_body = _simulation_http_wait_for_exact_cache(
                        cache_key=cache_key,
                        perspective=perspective,
                        max_wait_seconds=wait_seconds,
                    )
                    if inflight_body is not None:
                        _send_simulation_response(
                            inflight_body,
                            cache_key_for_compact=cache_key,
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
                            cache_key_for_compact=cache_key,
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
                            cache_key_for_compact=cache_key,
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
                        include_unified_graph=not compact_response_mode,
                    )
                    simulation["projection"] = projection
                    response_body = _json_compact(simulation).encode("utf-8")
                    _simulation_http_cache_store(cache_key, response_body)
                    if compact_response_mode:
                        compact_cache_key = f"{cache_key}|compact"
                        compact_payload = _simulation_http_compact_simulation_payload(
                            simulation
                        )
                        _simulation_http_compact_cache_store(
                            compact_cache_key,
                            _json_compact(compact_payload).encode("utf-8"),
                        )
                    _simulation_http_disk_cache_store(
                        self.part_root,
                        perspective=perspective,
                        body=response_body,
                    )
                    _simulation_http_failure_clear()
                    _send_simulation_response(
                        response_body,
                        cache_key_for_compact=cache_key,
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
                        cache_key_for_compact=cache_key,
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
                        cache_key_for_compact=cache_key,
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

        if parsed.path == "/api/simulation/bootstrap":
            req = self._read_json_body() or {}
            perspective = str(
                req.get("perspective", PROJECTION_DEFAULT_PERSPECTIVE) or ""
            )
            sync_inbox = _safe_bool_query(
                str(req.get("sync_inbox", "false") or "false"),
                default=False,
            )
            include_simulation_payload = _safe_bool_query(
                str(req.get("include_simulation", "false") or "false"),
                default=False,
            )
            wait = _safe_bool_query(
                str(req.get("wait", "false") or "false"),
                default=False,
            )

            if wait:
                payload, status = self._run_simulation_bootstrap(
                    perspective=perspective,
                    sync_inbox=sync_inbox,
                    include_simulation_payload=include_simulation_payload,
                )
                self._send_json(payload, status=status)
                return

            request_payload = {
                "perspective": normalize_projection_perspective(perspective),
                "sync_inbox": bool(sync_inbox),
                "include_simulation": bool(include_simulation_payload),
            }
            started, job_snapshot = _simulation_bootstrap_job_start(
                request_payload=request_payload
            )
            if not started:
                self._send_json(
                    {
                        "ok": True,
                        "record": "eta-mu.simulation-bootstrap.queue.v1",
                        "status": "running",
                        "job": job_snapshot,
                    },
                    status=HTTPStatus.ACCEPTED,
                )
                return

            job_id = str(job_snapshot.get("job_id", "") or "")

            def _run_bootstrap_job() -> None:
                try:
                    payload, status = self._run_simulation_bootstrap(
                        perspective=perspective,
                        sync_inbox=sync_inbox,
                        include_simulation_payload=include_simulation_payload,
                        phase_callback=lambda phase, detail: (
                            _simulation_bootstrap_job_mark_phase(
                                job_id=job_id,
                                phase=phase,
                                detail=detail,
                            )
                        ),
                    )
                    if status == HTTPStatus.OK and bool(payload.get("ok", False)):
                        _simulation_bootstrap_job_complete(
                            job_id=job_id,
                            report=payload,
                        )
                    else:
                        _simulation_bootstrap_job_fail(
                            job_id=job_id,
                            error=str(
                                payload.get(
                                    "error", "simulation_bootstrap_failed:unknown"
                                )
                            ),
                            report=payload,
                        )
                except Exception as exc:
                    _simulation_bootstrap_job_fail(
                        job_id=job_id,
                        error=f"simulation_bootstrap_failed:{exc.__class__.__name__}",
                        report={
                            "ok": False,
                            "record": "eta-mu.simulation-bootstrap.v1",
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                            "perspective": normalize_projection_perspective(
                                perspective
                            ),
                            "error": f"simulation_bootstrap_failed:{exc.__class__.__name__}",
                            "detail": f"{exc.__class__.__name__}: {exc}",
                        },
                    )

            thread_name = "simulation-bootstrap-" + (
                job_id.split(":")[-1][:8] if job_id else "job"
            )
            threading.Thread(
                target=_run_bootstrap_job,
                daemon=True,
                name=thread_name,
            ).start()
            self._send_json(
                {
                    "ok": True,
                    "record": "eta-mu.simulation-bootstrap.queue.v1",
                    "status": "running",
                    "job": job_snapshot,
                },
                status=HTTPStatus.ACCEPTED,
            )
            return

        if parsed.path == "/api/config/update":
            req = self._read_json_body() or {}
            result = _config_apply_update(
                module_name=str(req.get("module", "") or ""),
                key_name=str(req.get("key", "") or ""),
                path_tokens=_config_normalize_path_tokens(req.get("path")),
                value=req.get("value"),
            )
            if bool(result.get("ok", False)):
                _config_runtime_version_bump()
                _simulation_http_cache_invalidate(part_root=self.part_root)
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/config/reset":
            req = self._read_json_body() or {}
            result = _config_reset_updates(
                module_name=str(req.get("module", "") or ""),
                key_name=str(req.get("key", "") or ""),
                path_tokens=_config_normalize_path_tokens(req.get("path")),
            )
            if bool(result.get("ok", False)):
                _config_runtime_version_bump()
                _simulation_http_cache_invalidate(part_root=self.part_root)
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

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

                meta = row.get("meta") if isinstance(row.get("meta"), dict) else None
                if not isinstance(meta, dict):
                    meta = default_meta if isinstance(default_meta, dict) else None
                meta_copy = dict(meta) if isinstance(meta, dict) else {}

                if kind in {"search", "search_query", "query", "semantic_search"}:
                    meta_query = _normalize_query_text(
                        str(meta_copy.get("query", "") or "")
                    )
                    query_text = meta_query or _normalize_query_text(message)
                    if query_text:
                        search_meta = _build_search_daimoi_meta(
                            query_text,
                            target=target,
                            model=_effective_request_embed_model(
                                meta_copy.get("model")
                            ),
                        )
                        if search_meta:
                            meta_copy["query"] = query_text
                            meta_copy["search_daimoi"] = search_meta
                            message = query_text

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
                if meta_copy:
                    event_row["meta"] = {
                        str(key): value for key, value in list(meta_copy.items())[:16]
                    }

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
                        "meta": meta_copy,
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
            mode = str(req.get("mode", "llm") or "llm")
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


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class,
        *,
        max_threads: int,
    ):
        self._max_threads = max(1, int(max_threads))
        self._thread_slots = threading.BoundedSemaphore(self._max_threads)
        super().__init__(server_address, request_handler_class)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._thread_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\n"
                    b"Content-Length: 0\r\n\r\n"
                )
            except Exception:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._thread_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._thread_slots.release()


def serve(
    part_root: Path,
    vault_root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
):
    _ensure_weaver_service(part_root, host)
    handler_class = make_handler(part_root, vault_root, host, port)
    server = BoundedThreadingHTTPServer(
        (host, port),
        handler_class,
        max_threads=_RUNTIME_HTTP_MAX_THREADS,
    )
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
