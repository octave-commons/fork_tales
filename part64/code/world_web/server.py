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
import urllib.error
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
from .ws_stream_utils import (
    normalize_ws_wire_mode as _normalize_ws_wire_mode_impl,
    simulation_ws_chunk_messages as _simulation_ws_chunk_messages_impl,
    simulation_ws_chunk_plan as _simulation_ws_chunk_plan_impl,
    simulation_ws_split_delta_by_worker as _simulation_ws_split_delta_by_worker,
    ws_pack_message as _ws_pack_message_impl,
)
from .ws_payload_utils import (
    simulation_ws_capture_particle_motion_state as _simulation_ws_capture_particle_motion_state_impl,
    simulation_ws_compact_graph_payload as _simulation_ws_compact_graph_payload_impl,
    simulation_ws_decode_cached_payload as _simulation_ws_decode_cached_payload_impl,
    simulation_ws_payload_has_disabled_particle_dynamics as _simulation_ws_payload_has_disabled_particle_dynamics_impl,
    simulation_ws_payload_is_bootstrap_only as _simulation_ws_payload_is_bootstrap_only_impl,
    simulation_ws_payload_is_sparse as _simulation_ws_payload_is_sparse_impl,
    simulation_ws_payload_missing_graph_payload as _simulation_ws_payload_missing_graph_payload_impl,
    simulation_ws_restore_particle_motion_state as _simulation_ws_restore_particle_motion_state_impl,
    simulation_ws_sample_particle_page as _simulation_ws_sample_particle_page_impl,
    simulation_ws_trim_simulation_payload as _simulation_ws_trim_simulation_payload_impl,
    ws_clamp01 as _ws_clamp01_impl,
)
from . import (
    simulation_ws_daimoi_summary_utils as simulation_ws_daimoi_summary_utils_module,
)
from . import simulation_ws_cache_utils as simulation_ws_cache_utils_module
from . import simulation_ws_governor_utils as simulation_ws_governor_utils_module
from . import catalog_stream_utils as catalog_stream_utils_module
from . import runtime_catalog_fallback_utils as runtime_catalog_fallback_utils_module
from . import runtime_io_utils as runtime_io_utils_module
from . import github_conversation_utils as github_conversation_utils_module
from . import server_query_daimoi_utils as server_query_daimoi_utils_module
from . import server_runtime_health_utils as server_runtime_health_utils_module
from . import server_runtime_config_utils as server_runtime_config_utils_module
from . import server_misc_utils as server_misc_utils_module
from . import simulation_ws_particles_utils as simulation_ws_particles_utils_module
from . import simulation_http_trim_utils as simulation_http_trim_utils_module
from . import simulation_http_cache_key_utils as simulation_http_cache_key_utils_module
from . import (
    simulation_http_cache_state_utils as simulation_http_cache_state_utils_module,
)
from . import (
    simulation_http_async_refresh_utils as simulation_http_async_refresh_utils_module,
)
from . import (
    simulation_http_disk_cache_utils as simulation_http_disk_cache_utils_module,
)
from . import simulation_http_response_utils as simulation_http_response_utils_module
from . import simulation_http_fallback_utils as simulation_http_fallback_utils_module
from . import (
    simulation_http_build_gate_utils as simulation_http_build_gate_utils_module,
)
from . import simulation_docker_ws_controller as simulation_docker_ws_controller_module
from . import simulation_get_controller as simulation_get_controller_module
from . import (
    simulation_management_controller as simulation_management_controller_module,
)
from . import simulation_post_controller as simulation_post_controller_module
from . import simulation_status_command_utils as simulation_status_command_utils_module
from . import (
    simulation_ws_delta_patch_controller as simulation_ws_delta_patch_controller_module,
)
from . import simulation_ws_send_controller as simulation_ws_send_controller_module
from . import (
    simulation_ws_shared_payload_utils as simulation_ws_shared_payload_utils_module,
)
from . import (
    simulation_ws_tick_policy_controller as simulation_ws_tick_policy_controller_module,
)
from . import simulation_ws_shared_controller as simulation_ws_shared_controller_module
from . import ws_upgrade_controller as ws_upgrade_controller_module
from . import (
    simulation_bootstrap_state_utils as simulation_bootstrap_state_utils_module,
)
from . import (
    simulation_bootstrap_graph_utils as simulation_bootstrap_graph_utils_module,
)
from . import (
    simulation_bootstrap_report_utils as simulation_bootstrap_report_utils_module,
)
from . import world_runtime_controller as world_runtime_controller_module
from . import muse_mvc_controller as muse_mvc_controller_module
from . import muse_threat_radar_utils as muse_threat_radar_utils_module
from . import muse_runtime_backend_utils as muse_runtime_backend_utils_module
from . import muse_ws_stream_utils as muse_ws_stream_utils_module
from .study_snapshot_response_utils import get_or_build_study_snapshot_response


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
_STUDY_RESPONSE_CACHE_SECONDS = max(
    0.25,
    float(os.getenv("STUDY_RESPONSE_CACHE_SECONDS", "2.5") or "2.5"),
)
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
_SIMULATION_HTTP_BUILD_LOCK_ACQUIRE_TIMEOUT_SECONDS = max(
    0.5,
    float(
        os.getenv("SIMULATION_HTTP_BUILD_LOCK_ACQUIRE_TIMEOUT_SECONDS", "18.0")
        or "18.0"
    ),
)
_SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED = str(
    os.getenv("SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
_SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS = max(
    _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
    float(
        os.getenv("SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS", "180.0")
        or "180.0"
    ),
)
_SIMULATION_HTTP_FULL_ASYNC_LOCK_TIMEOUT_SECONDS = max(
    0.5,
    float(
        os.getenv("SIMULATION_HTTP_FULL_ASYNC_LOCK_TIMEOUT_SECONDS", "18.0") or "18.0"
    ),
)
_SIMULATION_HTTP_FULL_ASYNC_MAX_RUNNING_SECONDS = max(
    5.0,
    float(
        os.getenv("SIMULATION_HTTP_FULL_ASYNC_MAX_RUNNING_SECONDS", "480.0") or "480.0"
    ),
)
_SIMULATION_HTTP_FULL_ASYNC_START_MIN_INTERVAL_SECONDS = max(
    0.0,
    float(
        os.getenv("SIMULATION_HTTP_FULL_ASYNC_START_MIN_INTERVAL_SECONDS", "5.0")
        or "5.0"
    ),
)
_SIMULATION_HTTP_WARMUP_ENABLED = str(
    os.getenv("SIMULATION_HTTP_WARMUP_ENABLED", "0") or "0"
).strip().lower() not in {"0", "false", "no", "off"}
_SIMULATION_HTTP_COLD_PLACEHOLDER_ENABLED = str(
    os.getenv("SIMULATION_HTTP_COLD_PLACEHOLDER_ENABLED", "1") or "1"
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
    return server_query_daimoi_utils_module.normalize_query_text(text)


def _query_variant_terms(query_text: str) -> list[str]:
    return server_query_daimoi_utils_module.query_variant_terms(query_text)


def _build_search_daimoi_meta(
    query_text: str,
    *,
    target: str,
    model: str | None,
) -> dict[str, Any]:
    return server_query_daimoi_utils_module.build_search_daimoi_meta(
        query_text,
        target=target,
        model=model,
        entity_manifest=ENTITY_MANIFEST,
        normalize_embedding_vector=_normalize_embedding_vector,
        ollama_embed=_ollama_embed,
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
_SIMULATION_HTTP_FULL_ASYNC_REFRESH_LOCK = threading.Lock()
_SIMULATION_HTTP_FULL_ASYNC_REFRESH_STATE: dict[str, Any] = {
    "running": False,
    "status": "idle",
    "job_id": "",
    "trigger": "",
    "perspective": "",
    "cache_perspective": "",
    "cache_key": "",
    "started_at": "",
    "started_monotonic": 0.0,
    "last_start_monotonic_by_perspective": {},
    "updated_at": "",
    "completed_at": "",
    "error": "",
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
    float(os.getenv("SIMULATION_BOOTSTRAP_MAX_SECONDS", "480.0") or "480.0"),
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
_MUSE_THREAT_RADAR_ENABLED = str(
    os.getenv("MUSE_THREAT_RADAR_ENABLED", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
_MUSE_THREAT_RADAR_MUSE_ID = (
    str(os.getenv("MUSE_THREAT_RADAR_MUSE_ID", "github_security_review") or "")
    .strip()
    .lower()
)
if not _MUSE_THREAT_RADAR_MUSE_ID:
    _MUSE_THREAT_RADAR_MUSE_ID = "github_security_review"
_MUSE_THREAT_RADAR_LABEL = (
    str(os.getenv("MUSE_THREAT_RADAR_LABEL", "Threat Radar") or "Threat Radar").strip()
    or "Threat Radar"
)
_MUSE_THREAT_RADAR_INTERVAL_SECONDS = max(
    8.0,
    float(os.getenv("MUSE_THREAT_RADAR_INTERVAL_SECONDS", "45") or "45"),
)
_MUSE_THREAT_RADAR_TOKEN_BUDGET = max(
    320,
    min(
        4096,
        int(float(os.getenv("MUSE_THREAT_RADAR_TOKEN_BUDGET", "1400") or "1400")),
    ),
)
_MUSE_THREAT_RADAR_PROMPT = (
    str(
        os.getenv(
            "MUSE_THREAT_RADAR_PROMPT",
            "/facts /graph multi_threat_radar 1440 24",
        )
        or "/facts /graph multi_threat_radar 1440 24"
    ).strip()
    or "/facts /graph multi_threat_radar 1440 24"
)
_MUSE_THREAT_RADAR_LOCK = threading.Lock()
_MUSE_THREAT_RADAR_STATE: dict[str, Any] = {
    "next_run_monotonic": 0.0,
    "last_run_monotonic": 0.0,
    "last_run_at": "",
    "last_status": "idle",
    "last_turn_id": "",
    "last_error": "",
    "last_reason": "",
    "last_skipped_reason": "",
    "global_seed_only_streak": 0,
    "global_seed_only_alert": False,
    "last_non_provisional_global_at": "",
}
_MUSE_THREAT_RADAR_REPORT_CACHE_LOCK = threading.Lock()
_MUSE_THREAT_RADAR_REPORT_CACHE: dict[str, dict[str, Any]] = {}
_MUSE_THREAT_RADAR_REPORT_CACHE_MAX = max(
    16,
    int(float(os.getenv("MUSE_THREAT_RADAR_REPORT_CACHE_MAX", "128") or "128")),
)

# Active threat context for muses - stores top threats per radar type
# These are made available as surrounding nodes when the muse generates context.
_MUSE_ACTIVE_THREATS_LOCK = threading.Lock()
_MUSE_ACTIVE_THREATS: dict[str, dict[str, Any]] = {
    "local": {"threats": [], "hot_repos": [], "updated_at": ""},
    "global": {"threats": [], "hot_domains": [], "watch_sources": [], "updated_at": ""},
}
_MUSE_ACTIVE_THREATS_MAX_PER_RADAR = 8
_MUSE_THREAT_RADAR_GLOBAL_SEED_ONLY_ALERT_STREAK = max(
    1,
    min(
        64,
        int(
            float(
                os.getenv(
                    "MUSE_THREAT_RADAR_GLOBAL_SEED_ONLY_ALERT_STREAK",
                    "3",
                )
                or "3"
            )
        ),
    ),
)
_SIMULATION_WS_STREAM_PARTICLE_MAX = max(
    48,
    int(float(os.getenv("SIMULATION_WS_STREAM_PARTICLE_MAX", "180") or "180")),
)
_SIMULATION_WS_HTTP_CACHE_BRIDGE_MIN_FIELD_PARTICLES = max(
    1,
    int(
        float(
            os.getenv(
                "SIMULATION_WS_HTTP_CACHE_BRIDGE_MIN_FIELD_PARTICLES",
                "64",
            )
            or "64"
        )
    ),
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
        return server_runtime_health_utils_module.runtime_ws_client_snapshot(
            active_count=_RUNTIME_WS_CLIENT_COUNT,
            max_clients=_RUNTIME_WS_MAX_CLIENTS,
        )


def _runtime_ws_try_acquire_client_slot() -> bool:
    global _RUNTIME_WS_CLIENT_COUNT
    with _RUNTIME_WS_CLIENT_LOCK:
        acquired, next_count = (
            server_runtime_health_utils_module.runtime_ws_try_acquire_client_slot(
                active_count=_RUNTIME_WS_CLIENT_COUNT,
                max_clients=_RUNTIME_WS_MAX_CLIENTS,
            )
        )
        if not acquired:
            return False
        _RUNTIME_WS_CLIENT_COUNT = int(next_count)
    return True


def _runtime_ws_release_client_slot() -> None:
    global _RUNTIME_WS_CLIENT_COUNT
    with _RUNTIME_WS_CLIENT_LOCK:
        _RUNTIME_WS_CLIENT_COUNT = (
            server_runtime_health_utils_module.runtime_ws_release_client_slot(
                active_count=_RUNTIME_WS_CLIENT_COUNT,
            )
        )


def _runtime_guard_state(resource_snapshot: dict[str, Any]) -> dict[str, Any]:
    return server_runtime_health_utils_module.runtime_guard_state(
        resource_snapshot,
        safe_float=_safe_float,
        cpu_utilization_critical=_RUNTIME_GUARD_CPU_UTILIZATION_CRITICAL,
        memory_pressure_critical=_RUNTIME_GUARD_MEMORY_PRESSURE_CRITICAL,
        log_error_ratio_critical=_RUNTIME_GUARD_LOG_ERROR_RATIO_CRITICAL,
    )


def _runtime_health_payload(part_root: Path) -> dict[str, Any]:
    return server_runtime_health_utils_module.runtime_health_payload(
        part_root,
        resource_monitor_snapshot=lambda path: _resource_monitor_snapshot(
            part_root=path
        ),
        guard_state_builder=_runtime_guard_state,
        ws_snapshot_builder=_runtime_ws_client_snapshot,
    )


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


_CONFIG_SCALAR_LIMITS: dict[tuple[str, str], tuple[float, float]] = {
    ("simulation", "SIMULATION_STREAM_DAIMOI_FRICTION"): (0.0, 2.0),
    ("simulation", "SIMULATION_STREAM_NEXUS_FRICTION"): (0.0, 2.0),
    ("simulation", "SIMULATION_STREAM_FRICTION"): (0.0, 2.0),
}


def _config_apply_update(
    *,
    module_name: str,
    key_name: str,
    path_tokens: list[str],
    value: Any,
) -> dict[str, Any]:
    return server_runtime_config_utils_module.config_apply_update(
        module_name=module_name,
        key_name=key_name,
        path_tokens=path_tokens,
        value=value,
        config_module_specs=_CONFIG_MODULE_SPECS,
        config_runtime_edit_lock=_CONFIG_RUNTIME_EDIT_LOCK,
        scalar_limits=_CONFIG_SCALAR_LIMITS,
    )


def _config_reset_updates(
    *,
    module_name: str = "",
    key_name: str = "",
    path_tokens: list[str] | None = None,
) -> dict[str, Any]:
    return server_runtime_config_utils_module.config_reset_updates(
        module_name=module_name,
        key_name=key_name,
        path_tokens=path_tokens,
        config_module_specs=_CONFIG_MODULE_SPECS,
        config_runtime_baseline=_CONFIG_RUNTIME_BASELINE,
        config_runtime_edit_lock=_CONFIG_RUNTIME_EDIT_LOCK,
    )


def _config_capture_runtime_baseline() -> dict[str, dict[str, Any]]:
    return server_runtime_config_utils_module.config_capture_runtime_baseline(
        _CONFIG_MODULE_SPECS,
    )


def _config_normalize_path_tokens(path_raw: Any) -> list[str]:
    return server_runtime_config_utils_module.config_normalize_path_tokens(path_raw)


_CONFIG_RUNTIME_BASELINE = _config_capture_runtime_baseline()


def _simulation_http_trim_config() -> (
    simulation_http_trim_utils_module.SimulationHttpTrimConfig
):
    return simulation_http_trim_utils_module.SimulationHttpTrimConfig(
        trim_enabled=_SIMULATION_HTTP_TRIM_ENABLED,
        max_items=_SIMULATION_HTTP_MAX_ITEMS,
        max_file_nodes=_SIMULATION_HTTP_MAX_FILE_NODES,
        max_file_edges=_SIMULATION_HTTP_MAX_FILE_EDGES,
        max_field_nodes=_SIMULATION_HTTP_MAX_FIELD_NODES,
        max_tag_nodes=_SIMULATION_HTTP_MAX_TAG_NODES,
        max_render_nodes=_SIMULATION_HTTP_MAX_RENDER_NODES,
        max_crawler_nodes=_SIMULATION_HTTP_MAX_CRAWLER_NODES,
        max_crawler_edges=_SIMULATION_HTTP_MAX_CRAWLER_EDGES,
        max_crawler_field_nodes=_SIMULATION_HTTP_MAX_CRAWLER_FIELD_NODES,
        max_text_excerpt_chars=_SIMULATION_HTTP_MAX_TEXT_EXCERPT_CHARS,
        max_summary_chars=_SIMULATION_HTTP_MAX_SUMMARY_CHARS,
        max_embed_layer_points=_SIMULATION_HTTP_MAX_EMBED_LAYER_POINTS,
        max_embed_ids=_SIMULATION_HTTP_MAX_EMBED_IDS,
        max_embedding_links=_SIMULATION_HTTP_MAX_EMBEDDING_LINKS,
    )


def _config_payload(*, module_filter: str = "") -> dict[str, Any]:
    return server_runtime_config_utils_module.config_payload(
        module_filter=module_filter,
        config_module_specs=_CONFIG_MODULE_SPECS,
        runtime_version_snapshot=_config_runtime_version_snapshot,
    )


def _json_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(_safe_float(value, float(fallback)))
    except (TypeError, ValueError):
        return int(fallback)


def _normalize_ws_wire_mode(mode: str) -> str:
    return _normalize_ws_wire_mode_impl(mode, default_mode=_WS_WIRE_MODE_DEFAULT)


def _ws_pack_message(payload: dict[str, Any]) -> list[Any]:
    return _ws_pack_message_impl(payload, schema=_WS_WIRE_ARRAY_SCHEMA)


def _simulation_http_cache_key(
    *,
    perspective: str,
    catalog: dict[str, Any],
    queue_snapshot: dict[str, Any],
    influence_snapshot: dict[str, Any],
) -> str:
    return simulation_http_cache_key_utils_module.simulation_http_cache_key(
        perspective=perspective,
        catalog=catalog,
        queue_snapshot=queue_snapshot,
        influence_snapshot=influence_snapshot,
        cache_ignore_queue=_SIMULATION_HTTP_CACHE_IGNORE_QUEUE,
        cache_ignore_influence=_SIMULATION_HTTP_CACHE_IGNORE_INFLUENCE,
        config_version=_config_runtime_version_snapshot(),
    )


def _simulation_http_cache_store(cache_key: str, body: bytes) -> None:
    simulation_http_cache_state_utils_module.cache_store(
        cache_state=_SIMULATION_HTTP_CACHE,
        cache_lock=_SIMULATION_HTTP_CACHE_LOCK,
        cache_key=cache_key,
        body=body,
    )


def _runtime_catalog_http_cache_store(*, perspective: str, body: bytes) -> None:
    simulation_http_cache_state_utils_module.runtime_catalog_http_cache_store(
        cache_state=_RUNTIME_CATALOG_HTTP_CACHE,
        cache_lock=_RUNTIME_CATALOG_HTTP_CACHE_LOCK,
        perspective=perspective,
        body=body,
    )


def _runtime_catalog_http_cached_body(
    *,
    perspective: str,
    max_age_seconds: float,
) -> bytes | None:
    return simulation_http_cache_state_utils_module.runtime_catalog_http_cached_body(
        cache_state=_RUNTIME_CATALOG_HTTP_CACHE,
        cache_lock=_RUNTIME_CATALOG_HTTP_CACHE_LOCK,
        perspective=perspective,
        max_age_seconds=max_age_seconds,
        safe_float=_safe_float,
    )


def _runtime_catalog_http_cache_invalidate() -> None:
    simulation_http_cache_state_utils_module.runtime_catalog_http_cache_invalidate(
        cache_state=_RUNTIME_CATALOG_HTTP_CACHE,
        cache_lock=_RUNTIME_CATALOG_HTTP_CACHE_LOCK,
    )


def _simulation_http_compact_stale_fallback_body(
    *,
    part_root: Path,
    perspective: str,
    max_age_seconds: float,
) -> tuple[bytes | None, str]:
    return simulation_http_fallback_utils_module.simulation_http_compact_stale_fallback_body(
        part_root=part_root,
        cache_perspective=perspective,
        max_age_seconds=max_age_seconds,
        safe_float=_safe_float,
        cached_body_reader=_simulation_http_cached_body,
        disk_cache_loader=_simulation_http_disk_cache_load,
        cache_store=_simulation_http_cache_store,
    )


def _simulation_http_compact_cache_store(cache_key: str, body: bytes) -> None:
    simulation_http_cache_state_utils_module.cache_store(
        cache_state=_SIMULATION_HTTP_COMPACT_CACHE,
        cache_lock=_SIMULATION_HTTP_COMPACT_CACHE_LOCK,
        cache_key=cache_key,
        body=body,
    )


def _simulation_http_compact_cached_body(
    *,
    cache_key: str = "",
    perspective: str = "",
    max_age_seconds: float,
    require_exact_key: bool = False,
) -> bytes | None:
    return simulation_http_cache_state_utils_module.cache_cached_body(
        cache_state=_SIMULATION_HTTP_COMPACT_CACHE,
        cache_lock=_SIMULATION_HTTP_COMPACT_CACHE_LOCK,
        cache_key=cache_key,
        perspective=perspective,
        max_age_seconds=max_age_seconds,
        require_exact_key=require_exact_key,
        safe_float=_safe_float,
        match_requested_key_when_not_exact=True,
    )


def _simulation_http_cache_invalidate(*, part_root: Path | None = None) -> None:
    _runtime_catalog_http_cache_invalidate()
    simulation_http_cache_state_utils_module.cache_reset(
        cache_state=_SIMULATION_HTTP_CACHE,
        cache_lock=_SIMULATION_HTTP_CACHE_LOCK,
    )
    simulation_http_cache_state_utils_module.cache_reset(
        cache_state=_SIMULATION_HTTP_COMPACT_CACHE,
        cache_lock=_SIMULATION_HTTP_COMPACT_CACHE_LOCK,
    )
    simulation_http_disk_cache_utils_module.simulation_http_disk_cache_invalidate(
        part_root=part_root,
        disk_cache_enabled=_SIMULATION_HTTP_DISK_CACHE_ENABLED,
        default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
        perspective_options=projection_perspective_options(),
    )


def _simulation_http_cached_body(
    *,
    cache_key: str = "",
    perspective: str = "",
    max_age_seconds: float,
    require_exact_key: bool = False,
) -> bytes | None:
    return simulation_http_cache_state_utils_module.cache_cached_body(
        cache_state=_SIMULATION_HTTP_CACHE,
        cache_lock=_SIMULATION_HTTP_CACHE_LOCK,
        cache_key=cache_key,
        perspective=perspective,
        max_age_seconds=max_age_seconds,
        require_exact_key=require_exact_key,
        safe_float=_safe_float,
        match_requested_key_when_not_exact=False,
    )


def _simulation_http_wait_for_exact_cache(
    *,
    cache_key: str,
    perspective: str,
    max_wait_seconds: float,
    poll_seconds: float = 0.05,
) -> bytes | None:
    max_cache_age = max(
        _SIMULATION_HTTP_CACHE_SECONDS,
        _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
        max(0.0, _safe_float(max_wait_seconds, 0.0)),
    )
    return simulation_http_cache_state_utils_module.wait_for_exact_cache(
        cache_key=cache_key,
        perspective=perspective,
        max_wait_seconds=max_wait_seconds,
        poll_seconds=poll_seconds,
        max_cache_age_seconds=max_cache_age,
        cached_body_reader=_simulation_http_cached_body,
        safe_float=_safe_float,
    )


def _simulation_http_is_cold_start() -> bool:
    return simulation_http_cache_state_utils_module.is_cold_start(
        disk_cold_start_seconds=_SIMULATION_HTTP_DISK_COLD_START_SECONDS,
        server_boot_monotonic=_SERVER_BOOT_MONOTONIC,
    )


def _simulation_http_failure_backoff_snapshot() -> tuple[float, str, int]:
    return simulation_http_cache_state_utils_module.failure_backoff_snapshot(
        failure_state=_SIMULATION_HTTP_FAILURE_STATE,
        failure_lock=_SIMULATION_HTTP_FAILURE_LOCK,
        cooldown_seconds=_SIMULATION_HTTP_FAILURE_COOLDOWN_SECONDS,
        safe_float=_safe_float,
    )


def _simulation_http_failure_record(error_name: str) -> None:
    simulation_http_cache_state_utils_module.failure_record(
        failure_state=_SIMULATION_HTTP_FAILURE_STATE,
        failure_lock=_SIMULATION_HTTP_FAILURE_LOCK,
        error_name=error_name,
        streak_reset_seconds=_SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS,
        safe_float=_safe_float,
    )


def _simulation_http_failure_clear() -> None:
    simulation_http_cache_state_utils_module.failure_clear(
        failure_state=_SIMULATION_HTTP_FAILURE_STATE,
        failure_lock=_SIMULATION_HTTP_FAILURE_LOCK,
    )


def _simulation_http_full_async_refresh_snapshot() -> dict[str, Any]:
    return simulation_http_async_refresh_utils_module.full_async_refresh_snapshot(
        refresh_state=_SIMULATION_HTTP_FULL_ASYNC_REFRESH_STATE,
        refresh_lock=_SIMULATION_HTTP_FULL_ASYNC_REFRESH_LOCK,
    )


def _simulation_http_full_async_refresh_cancel(
    *,
    reason: str,
    job_id: str = "",
) -> tuple[bool, dict[str, Any]]:
    return simulation_http_async_refresh_utils_module.full_async_refresh_cancel(
        refresh_state=_SIMULATION_HTTP_FULL_ASYNC_REFRESH_STATE,
        refresh_lock=_SIMULATION_HTTP_FULL_ASYNC_REFRESH_LOCK,
        reason=reason,
        job_id=job_id,
    )


def _simulation_http_full_async_throttle_remaining_seconds(
    snapshot: dict[str, Any],
    *,
    perspective: str,
    now_monotonic: float | None = None,
) -> float:
    return simulation_http_async_refresh_utils_module.full_async_throttle_remaining_seconds(
        snapshot,
        perspective=perspective,
        min_interval_seconds=_SIMULATION_HTTP_FULL_ASYNC_START_MIN_INTERVAL_SECONDS,
        normalize_projection_perspective=normalize_projection_perspective,
        safe_float=_safe_float,
        now_monotonic=now_monotonic,
    )


def _simulation_http_full_async_refresh_headers(
    snapshot: dict[str, Any],
    *,
    scheduled: bool,
) -> dict[str, str]:
    return simulation_http_async_refresh_utils_module.full_async_refresh_headers(
        snapshot,
        scheduled=scheduled,
        safe_float=_safe_float,
    )


def _simulation_http_full_async_refresh_start(
    *,
    perspective: str,
    cache_perspective: str,
    cache_key: str,
    trigger: str,
    runner: Callable[[], None],
    allow_throttle_bypass: bool = False,
) -> tuple[bool, dict[str, Any]]:
    return simulation_http_async_refresh_utils_module.full_async_refresh_start(
        refresh_state=_SIMULATION_HTTP_FULL_ASYNC_REFRESH_STATE,
        refresh_lock=_SIMULATION_HTTP_FULL_ASYNC_REFRESH_LOCK,
        perspective=perspective,
        cache_perspective=cache_perspective,
        cache_key=cache_key,
        trigger=trigger,
        runner=runner,
        allow_throttle_bypass=allow_throttle_bypass,
        default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
        max_running_seconds=_SIMULATION_HTTP_FULL_ASYNC_MAX_RUNNING_SECONDS,
        min_interval_seconds=_SIMULATION_HTTP_FULL_ASYNC_START_MIN_INTERVAL_SECONDS,
        safe_float=_safe_float,
        normalize_projection_perspective=normalize_projection_perspective,
        thread_name="simulation-http-full-async-refresh",
    )


def _simulation_http_slice_rows(value: Any, *, max_rows: int) -> list[Any]:
    return simulation_http_trim_utils_module.simulation_http_slice_rows(
        value,
        max_rows=max_rows,
    )


def _simulation_http_compact_embed_layer_point(value: Any) -> dict[str, Any] | None:
    return simulation_http_trim_utils_module.simulation_http_compact_embed_layer_point(
        value,
        config=_simulation_http_trim_config(),
    )


def _simulation_http_compact_embedding_link(value: Any) -> dict[str, Any] | None:
    return simulation_http_trim_utils_module.simulation_http_compact_embedding_link(
        value
    )


def _simulation_http_compact_file_node(value: Any) -> dict[str, Any] | None:
    return simulation_http_trim_utils_module.simulation_http_compact_file_node(
        value,
        config=_simulation_http_trim_config(),
    )


def _simulation_http_compact_file_graph_node(value: Any) -> dict[str, Any] | None:
    return simulation_http_trim_utils_module.simulation_http_compact_file_graph_node(
        value,
        config=_simulation_http_trim_config(),
    )


def _simulation_http_trim_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    return simulation_http_trim_utils_module.simulation_http_trim_catalog(
        catalog,
        config=_simulation_http_trim_config(),
    )


def _simulation_ws_trim_simulation_payload(
    simulation: dict[str, Any],
) -> dict[str, Any]:
    return _simulation_ws_trim_simulation_payload_impl(simulation)


def _simulation_ws_compact_graph_payload(
    simulation: dict[str, Any],
    *,
    assume_trimmed: bool = False,
) -> dict[str, Any]:
    return _simulation_ws_compact_graph_payload_impl(
        simulation,
        trim_catalog=_simulation_http_trim_catalog,
        assume_trimmed=assume_trimmed,
    )


def _simulation_ws_compact_field_particles(rows: Any) -> list[dict[str, Any]]:
    return _simulation_ws_compact_field_particles_with_nodes(
        rows,
        node_positions={},
        node_text_chars={},
    )


def _simulation_ws_collect_node_positions(
    simulation_payload: dict[str, Any],
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    return simulation_ws_particles_utils_module.simulation_ws_collect_node_positions(
        simulation_payload,
        safe_float=_safe_float,
    )


def _simulation_ws_compact_field_particles_with_nodes(
    rows: Any,
    *,
    node_positions: dict[str, tuple[float, float]],
    node_text_chars: dict[str, float],
) -> list[dict[str, Any]]:
    return simulation_ws_particles_utils_module.simulation_ws_compact_field_particles_with_nodes(
        rows,
        node_positions=node_positions,
        node_text_chars=node_text_chars,
        stream_particle_max=_SIMULATION_WS_STREAM_PARTICLE_MAX,
        safe_float=_safe_float,
    )


def _simulation_ws_extract_stream_particles(
    simulation_payload: dict[str, Any],
    *,
    node_positions: dict[str, tuple[float, float]] | None = None,
    node_text_chars: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    return simulation_ws_particles_utils_module.simulation_ws_extract_stream_particles(
        simulation_payload,
        node_positions=node_positions,
        node_text_chars=node_text_chars,
        ensure_daimoi_summary=_simulation_ws_ensure_daimoi_summary,
        compact_field_particles_with_nodes=_simulation_ws_compact_field_particles_with_nodes,
    )


def _simulation_ws_capture_particle_motion_state(
    simulation_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return _simulation_ws_capture_particle_motion_state_impl(simulation_payload)


def _simulation_ws_restore_particle_motion_state(
    simulation_payload: dict[str, Any],
    motion_state: dict[str, dict[str, Any]],
) -> int:
    return _simulation_ws_restore_particle_motion_state_impl(
        simulation_payload,
        motion_state,
        blend=_SIMULATION_WS_CACHE_PARTICLE_CONTINUITY_BLEND,
    )


def _simulation_ws_decode_cached_payload(cached_body: Any) -> dict[str, Any] | None:
    return _simulation_ws_decode_cached_payload_impl(cached_body)


def _simulation_ws_payload_is_sparse(payload: dict[str, Any]) -> bool:
    return _simulation_ws_payload_is_sparse_impl(payload)


def _simulation_ws_payload_has_disabled_particle_dynamics(
    payload: dict[str, Any],
) -> bool:
    return _simulation_ws_payload_has_disabled_particle_dynamics_impl(payload)


def _simulation_ws_payload_is_bootstrap_only(payload: dict[str, Any]) -> bool:
    return _simulation_ws_payload_is_bootstrap_only_impl(payload)


def _simulation_ws_payload_missing_graph_payload(payload: dict[str, Any]) -> bool:
    return _simulation_ws_payload_missing_graph_payload_impl(payload)


def _ws_clamp01(value: float) -> float:
    return _ws_clamp01_impl(value)


def _simulation_ws_sample_particle_page(
    rows: list[Any],
    *,
    max_rows: int,
    page_cursor: int,
    jitter_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    return _simulation_ws_sample_particle_page_impl(
        rows,
        max_rows=max_rows,
        page_cursor=page_cursor,
        jitter_seed=jitter_seed,
    )


def _simulation_ws_graph_node_position_map(
    node_positions: Any,
) -> dict[str, tuple[float, float]]:
    return (
        simulation_ws_daimoi_summary_utils_module.simulation_ws_graph_node_position_map(
            node_positions,
            node_limit=_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NODE_LIMIT,
            clamp01=_ws_clamp01,
            safe_float=_safe_float,
        )
    )


def _simulation_ws_graph_variability_update(node_positions: Any) -> dict[str, Any]:
    return simulation_ws_daimoi_summary_utils_module.simulation_ws_graph_variability_update(
        node_positions,
        graph_variability_lock=_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_LOCK,
        graph_variability_state=_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_STATE,
        node_limit=_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NODE_LIMIT,
        distance_ref=_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_DISTANCE_REF,
        ema_alpha=_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_EMA_ALPHA,
        clamp01=_ws_clamp01,
        safe_float=_safe_float,
    )


def _simulation_ws_daimoi_live_metrics(
    rows: Any,
    *,
    default_target: float,
) -> dict[str, Any]:
    return simulation_ws_daimoi_summary_utils_module.simulation_ws_daimoi_live_metrics(
        rows,
        default_target=default_target,
        daimoi_probabilistic_module=daimoi_probabilistic_module,
        clamp01=_ws_clamp01,
        safe_float=_safe_float,
    )


def _simulation_ws_ensure_daimoi_summary(
    payload: dict[str, Any],
    *,
    include_live_metrics: bool = True,
    include_graph_variability: bool = True,
) -> None:
    simulation_ws_daimoi_summary_utils_module.simulation_ws_ensure_daimoi_summary(
        payload,
        include_live_metrics=include_live_metrics,
        include_graph_variability=include_graph_variability,
        daimoi_probabilistic_module=daimoi_probabilistic_module,
        graph_variability_noise_gain=_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_NOISE_GAIN,
        graph_variability_route_damp=_SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_ROUTE_DAMP,
        live_metrics_builder=_simulation_ws_daimoi_live_metrics,
        graph_variability_builder=_simulation_ws_graph_variability_update,
        clamp01=_ws_clamp01,
        safe_float=_safe_float,
    )


def _simulation_ws_payload_missing_daimoi_summary(payload: dict[str, Any]) -> bool:
    return simulation_ws_daimoi_summary_utils_module.simulation_ws_payload_missing_daimoi_summary(
        payload,
        ensure_summary=_simulation_ws_ensure_daimoi_summary,
    )


def _simulation_bootstrap_store_report(report: dict[str, Any]) -> None:
    simulation_bootstrap_state_utils_module.bootstrap_report_store(
        report_state=_SIMULATION_BOOTSTRAP_LAST_REPORT,
        report_lock=_SIMULATION_BOOTSTRAP_REPORT_LOCK,
        report=report,
    )


def _simulation_bootstrap_snapshot_report() -> dict[str, Any] | None:
    return simulation_bootstrap_state_utils_module.bootstrap_report_snapshot(
        report_state=_SIMULATION_BOOTSTRAP_LAST_REPORT,
        report_lock=_SIMULATION_BOOTSTRAP_REPORT_LOCK,
    )


def _simulation_bootstrap_job_snapshot() -> dict[str, Any]:
    return simulation_bootstrap_state_utils_module.bootstrap_job_snapshot(
        job_state=_SIMULATION_BOOTSTRAP_JOB,
        job_lock=_SIMULATION_BOOTSTRAP_JOB_LOCK,
    )


def _simulation_bootstrap_job_start(
    *,
    request_payload: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    return simulation_bootstrap_state_utils_module.bootstrap_job_start(
        job_state=_SIMULATION_BOOTSTRAP_JOB,
        job_lock=_SIMULATION_BOOTSTRAP_JOB_LOCK,
        request_payload=request_payload,
    )


def _simulation_bootstrap_job_mark_phase(
    *,
    job_id: str,
    phase: str,
    detail: dict[str, Any] | None = None,
) -> None:
    simulation_bootstrap_state_utils_module.bootstrap_job_mark_phase(
        job_state=_SIMULATION_BOOTSTRAP_JOB,
        job_lock=_SIMULATION_BOOTSTRAP_JOB_LOCK,
        job_id=job_id,
        phase=phase,
        detail=detail,
    )


def _simulation_bootstrap_job_complete(
    *,
    job_id: str,
    report: dict[str, Any],
) -> None:
    simulation_bootstrap_state_utils_module.bootstrap_job_complete(
        job_state=_SIMULATION_BOOTSTRAP_JOB,
        job_lock=_SIMULATION_BOOTSTRAP_JOB_LOCK,
        job_id=job_id,
        report=report,
    )


def _simulation_bootstrap_job_fail(
    *,
    job_id: str,
    error: str,
    report: dict[str, Any] | None = None,
) -> None:
    simulation_bootstrap_state_utils_module.bootstrap_job_fail(
        job_state=_SIMULATION_BOOTSTRAP_JOB,
        job_lock=_SIMULATION_BOOTSTRAP_JOB_LOCK,
        job_id=job_id,
        error=error,
        report=report,
    )


def _simulation_bootstrap_embed_layer_row(layer: dict[str, Any]) -> dict[str, Any]:
    return simulation_bootstrap_graph_utils_module.bootstrap_embed_layer_row(layer)


def _simulation_bootstrap_normalize_path(value: Any) -> str:
    return simulation_bootstrap_graph_utils_module.bootstrap_normalize_path(value)


def _simulation_bootstrap_file_path(row: dict[str, Any]) -> str:
    return simulation_bootstrap_graph_utils_module.bootstrap_file_path(row)


def _simulation_bootstrap_file_row(row: dict[str, Any]) -> dict[str, Any]:
    return simulation_bootstrap_graph_utils_module.bootstrap_file_row(row)


def _simulation_bootstrap_graph_diff(
    *,
    catalog: dict[str, Any],
    simulation: dict[str, Any],
) -> dict[str, Any]:
    return simulation_bootstrap_graph_utils_module.bootstrap_graph_diff(
        catalog=catalog,
        simulation=simulation,
        max_excluded_files=_SIMULATION_BOOTSTRAP_MAX_EXCLUDED_FILES,
    )


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
    return simulation_bootstrap_report_utils_module.bootstrap_graph_report(
        perspective=perspective,
        catalog=catalog,
        simulation=simulation,
        projection=projection,
        phase_ms=phase_ms,
        reset_summary=reset_summary,
        inbox_sync=inbox_sync,
        cache_key=cache_key,
        max_excluded_files=_SIMULATION_BOOTSTRAP_MAX_EXCLUDED_FILES,
        normalize_projection_perspective=normalize_projection_perspective,
        runtime_config_version_snapshot=_config_runtime_version_snapshot,
    )


def _simulation_ws_load_cached_payload(
    *,
    part_root: Path,
    perspective: str,
    payload_mode: str = "trimmed",
    allow_disabled_particle_dynamics: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    return simulation_ws_cache_utils_module.simulation_ws_load_cached_payload(
        part_root=part_root,
        perspective=perspective,
        payload_mode=payload_mode,
        allow_disabled_particle_dynamics=allow_disabled_particle_dynamics,
        normalize_payload_mode=_simulation_ws_normalize_payload_mode,
        http_compact_cached_body_reader=_simulation_http_compact_cached_body,
        http_cached_body_reader=_simulation_http_cached_body,
        http_disk_cache_load=_simulation_http_disk_cache_load,
        http_disk_cache_path=_simulation_http_disk_cache_path,
        ws_decode_cached_payload=_simulation_ws_decode_cached_payload,
        ws_cache_max_age_seconds=_SIMULATION_WS_CACHE_MAX_AGE_SECONDS,
        ws_payload_is_sparse=_simulation_ws_payload_is_sparse,
        ws_payload_has_disabled_particle_dynamics=_simulation_ws_payload_has_disabled_particle_dynamics,
        ws_payload_is_bootstrap_only=_simulation_ws_payload_is_bootstrap_only,
        ws_collect_node_positions=_simulation_ws_collect_node_positions,
        ws_trim_simulation_payload=_simulation_ws_trim_simulation_payload,
        ws_compact_graph_payload=_simulation_ws_compact_graph_payload,
        ws_extract_stream_particles=_simulation_ws_extract_stream_particles,
        safe_float=_safe_float,
        max_graph_node_positions=2200,
    )


def _simulation_ws_normalize_delta_stream_mode(mode: str) -> str:
    return simulation_ws_cache_utils_module.simulation_ws_normalize_delta_stream_mode(
        mode,
        default_delta_stream_mode=_SIMULATION_WS_DELTA_STREAM_MODE,
    )


def _simulation_ws_normalize_payload_mode(mode: str) -> str:
    return simulation_ws_cache_utils_module.simulation_ws_normalize_payload_mode(mode)


def _simulation_ws_normalize_particle_payload_mode(mode: str) -> str:
    return (
        simulation_ws_cache_utils_module.simulation_ws_normalize_particle_payload_mode(
            mode
        )
    )


def _simulation_ws_lite_field_particles(
    rows: Any,
    *,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    return simulation_ws_cache_utils_module.simulation_ws_lite_field_particles(
        rows,
        max_rows=max_rows,
        particle_lite_keys=_SIMULATION_WS_PARTICLE_LITE_KEYS,
    )


def _simulation_ws_governor_estimate_work(simulation_payload: dict[str, Any]) -> float:
    return simulation_ws_governor_utils_module.simulation_ws_governor_estimate_work(
        simulation_payload,
        safe_float=_safe_float,
    )


def _simulation_ws_governor_ingestion_signal(
    catalog: dict[str, Any],
) -> tuple[int, int, int, int]:
    return simulation_ws_governor_utils_module.simulation_ws_governor_ingestion_signal(
        catalog,
        safe_float=_safe_float,
    )


def _simulation_ws_governor_stock_pressure(part_root: Path) -> tuple[float, float]:
    return simulation_ws_governor_utils_module.simulation_ws_governor_stock_pressure(
        part_root,
        safe_float=_safe_float,
        resource_monitor_snapshot=_resource_monitor_snapshot,
    )


def _simulation_ws_governor_particle_cap(
    base_cap: int,
    *,
    fidelity_signal: str,
    ingestion_pressure: float,
) -> int:
    return simulation_ws_governor_utils_module.simulation_ws_governor_particle_cap(
        base_cap,
        fidelity_signal=fidelity_signal,
        ingestion_pressure=ingestion_pressure,
        min_particle_cap=_SIMULATION_WS_GOVERNOR_MIN_PARTICLE_CAP,
        safe_float=_safe_float,
    )


def _simulation_ws_governor_graph_heartbeat_scale(fidelity_signal: str) -> float:
    return simulation_ws_governor_utils_module.simulation_ws_governor_graph_heartbeat_scale(
        fidelity_signal,
        degrade_scale=_SIMULATION_WS_GOVERNOR_DEGRADE_GRAPH_HEARTBEAT_SCALE,
        increase_scale=_SIMULATION_WS_GOVERNOR_INCREASE_GRAPH_HEARTBEAT_SCALE,
    )


def _catalog_stream_chunk_rows(value: Any) -> int:
    return catalog_stream_utils_module.catalog_stream_chunk_rows(
        value,
        default_chunk_rows=_CATALOG_STREAM_CHUNK_ROWS,
        safe_float=_safe_float,
    )


def _catalog_stream_get_path_value(
    payload: dict[str, Any], path: tuple[str, ...]
) -> Any:
    return catalog_stream_utils_module.catalog_stream_get_path_value(payload, path)


def _catalog_stream_set_path_value(
    payload: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    catalog_stream_utils_module.catalog_stream_set_path_value(payload, path, value)


def _catalog_stream_meta(catalog: dict[str, Any]) -> dict[str, Any]:
    return catalog_stream_utils_module.catalog_stream_meta(
        catalog,
        section_paths=_CATALOG_STREAM_SECTION_PATHS,
    )


def _catalog_stream_iter_rows(
    catalog: dict[str, Any],
    *,
    chunk_rows: int,
) -> Iterator[dict[str, Any]]:
    return catalog_stream_utils_module.catalog_stream_iter_rows(
        catalog,
        chunk_rows=chunk_rows,
        section_paths=_CATALOG_STREAM_SECTION_PATHS,
        default_chunk_rows=_CATALOG_STREAM_CHUNK_ROWS,
        safe_float=_safe_float,
    )


def _simulation_ws_chunk_plan(
    payload: dict[str, Any],
    *,
    chunk_chars: int,
    message_seq: int,
) -> tuple[list[dict[str, Any]], str | None]:
    return _simulation_ws_chunk_plan_impl(
        payload,
        chunk_chars=chunk_chars,
        message_seq=message_seq,
        allowed_types=_SIMULATION_WS_CHUNK_MESSAGE_TYPES,
        default_chunk_chars=_SIMULATION_WS_CHUNK_CHARS,
        delta_min_chars=_SIMULATION_WS_CHUNK_DELTA_MIN_CHARS,
        max_chunks=_SIMULATION_WS_CHUNK_MAX_CHUNKS,
    )


def _simulation_ws_chunk_messages(
    payload: dict[str, Any],
    *,
    chunk_chars: int,
    message_seq: int,
) -> list[dict[str, Any]]:
    return _simulation_ws_chunk_messages_impl(
        payload,
        chunk_chars=chunk_chars,
        message_seq=message_seq,
        allowed_types=_SIMULATION_WS_CHUNK_MESSAGE_TYPES,
        default_chunk_chars=_SIMULATION_WS_CHUNK_CHARS,
        delta_min_chars=_SIMULATION_WS_CHUNK_DELTA_MIN_CHARS,
        max_chunks=_SIMULATION_WS_CHUNK_MAX_CHUNKS,
    )


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
    return simulation_http_disk_cache_utils_module.simulation_http_disk_cache_path(
        part_root,
        perspective,
        default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
    )


def _simulation_http_runtime_reference_mtime(part_root: Path) -> float:
    candidate_paths = [
        part_root / "code" / "world_web" / "server.py",
        part_root / "code" / "world_web" / "simulation.py",
        part_root / "code" / "world_web" / "simulation_http_trim_utils.py",
        part_root / "code" / "world_web" / "simulation_http_cache_key_utils.py",
        part_root / "code" / "world_web" / "simulation_http_cache_state_utils.py",
        part_root / "code" / "world_web" / "simulation_http_async_refresh_utils.py",
        part_root / "code" / "world_web" / "simulation_http_disk_cache_utils.py",
        part_root / "code" / "world_web" / "c_double_buffer_backend.py",
        part_root / "code" / "world_web" / "daimoi_probabilistic.py",
        part_root / "code" / "world_web" / "native" / "libc_double_buffer_sim.so",
    ]
    return (
        simulation_http_disk_cache_utils_module.simulation_http_runtime_reference_mtime(
            candidate_paths
        )
    )


def _simulation_http_disk_cache_load(
    part_root: Path,
    *,
    perspective: str,
    max_age_seconds: float,
) -> bytes | None:
    return simulation_http_disk_cache_utils_module.simulation_http_disk_cache_load(
        part_root,
        perspective=perspective,
        max_age_seconds=max_age_seconds,
        disk_cache_enabled=_SIMULATION_HTTP_DISK_CACHE_ENABLED,
        default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
        runtime_reference_mtime=_simulation_http_runtime_reference_mtime(part_root),
        safe_float=_safe_float,
    )


def _simulation_http_disk_cache_has_payload(
    part_root: Path,
    *,
    perspective: str,
    max_age_seconds: float,
) -> bool:
    return (
        simulation_http_disk_cache_utils_module.simulation_http_disk_cache_has_payload(
            part_root,
            perspective=perspective,
            max_age_seconds=max_age_seconds,
            disk_cache_enabled=_SIMULATION_HTTP_DISK_CACHE_ENABLED,
            default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
            runtime_reference_mtime=_simulation_http_runtime_reference_mtime(part_root),
            safe_float=_safe_float,
        )
    )


def _simulation_http_disk_cache_store(
    part_root: Path,
    *,
    perspective: str,
    body: bytes,
) -> None:
    simulation_http_disk_cache_utils_module.simulation_http_disk_cache_store(
        part_root,
        perspective=perspective,
        body=body,
        disk_cache_enabled=_SIMULATION_HTTP_DISK_CACHE_ENABLED,
        default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
    )


_RUNTIME_CATALOG_SUBPROCESS_SCRIPT = (
    "import json,sys;"
    "from pathlib import Path;"
    "import code.world_web as ww;"
    "payload=ww.collect_catalog("
    "Path(sys.argv[1]),"
    "Path(sys.argv[2]),"
    "sync_inbox=False,"
    "include_pi_archive=False,"
    "include_world_log=False,"
    "include_embedding_state=False"
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
    return runtime_catalog_fallback_utils_module.collect_runtime_catalog_isolated(
        part_root,
        vault_root,
        runtime_catalog_subprocess_enabled=_RUNTIME_CATALOG_SUBPROCESS_ENABLED,
        runtime_catalog_subprocess_script=_RUNTIME_CATALOG_SUBPROCESS_SCRIPT,
        runtime_catalog_subprocess_timeout_seconds=_RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS,
    )


def _fallback_kind_for_path(path: Path) -> str:
    return runtime_catalog_fallback_utils_module.fallback_kind_for_path(
        path,
        audio_suffixes=AUDIO_SUFFIXES,
        image_suffixes=IMAGE_SUFFIXES,
        video_suffixes=VIDEO_SUFFIXES,
    )


def _fallback_rel_path(path: Path, vault_root: Path, part_root: Path) -> str:
    return runtime_catalog_fallback_utils_module.fallback_rel_path(
        path,
        vault_root,
        part_root,
    )


def _runtime_catalog_fallback_items(
    part_root: Path,
    vault_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    return runtime_catalog_fallback_utils_module.runtime_catalog_fallback_items(
        part_root,
        vault_root,
        load_manifest=load_manifest,
        fallback_rel_path_fn=_fallback_rel_path,
        fallback_kind_for_path_fn=_fallback_kind_for_path,
    )


def _runtime_catalog_fallback(part_root: Path, vault_root: Path) -> dict[str, Any]:
    return runtime_catalog_fallback_utils_module.runtime_catalog_fallback(
        part_root,
        vault_root,
        runtime_catalog_fallback_items_fn=_runtime_catalog_fallback_items,
        entity_manifest=ENTITY_MANIFEST,
        build_named_field_overlays=build_named_field_overlays,
        projection_default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
        projection_perspective_options=projection_perspective_options,
    )


def _weaver_probe_host(bind_host: str) -> str:
    return runtime_io_utils_module.weaver_probe_host(bind_host)


def _weaver_health_check(host: str, port: int, timeout_s: float = 0.8) -> bool:
    return runtime_io_utils_module.weaver_health_check(host, port, timeout_s=timeout_s)


def _ensure_weaver_service(part_root: Path, world_host: str) -> None:
    runtime_io_utils_module.ensure_weaver_service(
        part_root,
        world_host,
        weaver_autostart=WEAVER_AUTOSTART,
        weaver_host_env=WEAVER_HOST_ENV,
        weaver_port=WEAVER_PORT,
        weaver_probe_host_fn=_weaver_probe_host,
        weaver_health_check_fn=_weaver_health_check,
    )


def _parse_multipart_form(raw_body: bytes, content_type: str) -> dict[str, Any] | None:
    return runtime_io_utils_module.parse_multipart_form(raw_body, content_type)


def resolve_artifact_path(part_root: Path, request_path: str) -> Path | None:
    return runtime_io_utils_module.resolve_artifact_path(part_root, request_path)


_WS_CLIENT_FRAME_MAX_BYTES = WS_CLIENT_FRAME_MAX_BYTES


def render_index(payload: dict[str, Any], catalog: dict[str, Any]) -> str:
    del payload, catalog
    return ""


def _safe_bool_query(value: str, default: bool = False) -> bool:
    return server_misc_utils_module.safe_bool_query(value, default=default)


def _github_conversation_headers() -> dict[str, str]:
    return github_conversation_utils_module.github_conversation_headers()


def _github_conversation_fetch_json(
    url: str,
    *,
    timeout_s: float,
) -> tuple[bool, Any, int, str]:
    return github_conversation_utils_module.github_conversation_fetch_json(
        url,
        timeout_s=timeout_s,
        safe_float=_safe_float,
        headers_builder=_github_conversation_headers,
    )


def _github_conversation_comment_rows(
    payload: Any,
    *,
    channel: str,
    max_comments: int,
    max_body_chars: int,
) -> list[dict[str, Any]]:
    return github_conversation_utils_module.github_conversation_comment_rows(
        payload,
        channel=channel,
        max_comments=max_comments,
        max_body_chars=max_body_chars,
    )


def _github_conversation_markdown(
    *,
    repo: str,
    number: int,
    kind: str,
    title: str,
    state: str,
    html_url: str,
    root_body: str,
    comments: list[dict[str, Any]],
    max_markdown_chars: int,
) -> str:
    return github_conversation_utils_module.github_conversation_markdown(
        repo=repo,
        number=number,
        kind=kind,
        title=title,
        state=state,
        html_url=html_url,
        root_body=root_body,
        comments=comments,
        max_markdown_chars=max_markdown_chars,
    )


def _github_conversation_payload(
    *,
    repo: str,
    number: int,
    kind: str,
    max_comments: int,
    max_root_body_chars: int,
    max_comment_body_chars: int,
    max_markdown_chars: int,
    include_review_comments: bool,
    timeout_s: float,
) -> dict[str, Any]:
    def _fetch(url: str) -> tuple[bool, Any, int, str]:
        return _github_conversation_fetch_json(url, timeout_s=timeout_s)

    return github_conversation_utils_module.github_conversation_payload(
        repo=repo,
        number=number,
        kind=kind,
        max_comments=max_comments,
        max_root_body_chars=max_root_body_chars,
        max_comment_body_chars=max_comment_body_chars,
        max_markdown_chars=max_markdown_chars,
        include_review_comments=include_review_comments,
        timeout_s=timeout_s,
        fetch_json=_fetch,
        comment_rows=_github_conversation_comment_rows,
        markdown_builder=_github_conversation_markdown,
    )


def _docker_simulation_identifier_set(row: dict[str, Any]) -> set[str]:
    return server_misc_utils_module.docker_simulation_identifier_set(row)


def _find_docker_simulation_row(
    snapshot: dict[str, Any],
    identifier: str,
) -> dict[str, Any] | None:
    return server_misc_utils_module.find_docker_simulation_row(snapshot, identifier)


def _project_vector(embedding: list[float] | None) -> list[float]:
    return server_misc_utils_module.project_vector(embedding)


def _normalize_audio_upload_name(file_name: str, mime: str) -> str:
    return server_misc_utils_module.normalize_audio_upload_name(file_name, mime)


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
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, PATCH, DELETE, OPTIONS",
        )

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

    def _send_ws_upgrade_response(self, ws_key: str) -> None:
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", websocket_accept_value(ws_key))
        self.end_headers()
        self.close_connection = True

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

    def _schedule_full_simulation_async_refresh(
        self,
        *,
        perspective: str,
        cache_perspective: str,
        cache_key_hint: str,
        trigger: str,
        allow_throttle_bypass: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        resolved_perspective = normalize_projection_perspective(
            str(perspective or PROJECTION_DEFAULT_PERSPECTIVE)
        )
        resolved_cache_perspective = (
            str(cache_perspective or "").strip()
            or f"{resolved_perspective}|profile:full"
        )
        resolved_cache_key = (
            str(cache_key_hint or "").strip()
            or f"{resolved_cache_perspective}|manual|full"
        )
        resolved_trigger = str(trigger or "manual").strip() or "manual"

        def _runner() -> None:
            lock_acquired = _SIMULATION_HTTP_BUILD_LOCK.acquire(
                timeout=_SIMULATION_HTTP_FULL_ASYNC_LOCK_TIMEOUT_SECONDS
            )
            if not lock_acquired:
                raise TimeoutError("build_lock_timeout")

            try:
                (
                    refresh_catalog,
                    refresh_queue_snapshot,
                    _,
                    refresh_influence_snapshot,
                    _,
                ) = self._runtime_catalog(
                    perspective=resolved_perspective,
                    include_projection=False,
                    include_runtime_fields=False,
                    allow_inline_collect=False,
                )
                refresh_cache_key = _simulation_http_cache_key(
                    perspective=resolved_perspective,
                    catalog=refresh_catalog,
                    queue_snapshot=refresh_queue_snapshot,
                    influence_snapshot=refresh_influence_snapshot,
                )
                refresh_cache_key = f"{refresh_cache_key}|profile:full"
                simulation, projection = self._runtime_simulation(
                    refresh_catalog,
                    refresh_queue_snapshot,
                    refresh_influence_snapshot,
                    perspective=resolved_perspective,
                    include_unified_graph=True,
                    include_particle_dynamics=True,
                )
                simulation["projection"] = projection
                response_body = _json_compact(simulation).encode("utf-8")
                _simulation_http_cache_store(refresh_cache_key, response_body)
                _simulation_http_disk_cache_store(
                    self.part_root,
                    perspective=resolved_cache_perspective,
                    body=response_body,
                )
                _simulation_http_failure_clear()
            finally:
                _SIMULATION_HTTP_BUILD_LOCK.release()

        return _simulation_http_full_async_refresh_start(
            perspective=resolved_perspective,
            cache_perspective=resolved_cache_perspective,
            cache_key=resolved_cache_key,
            trigger=resolved_trigger,
            runner=_runner,
            allow_throttle_bypass=bool(allow_throttle_bypass),
        )

    def _proxy_weaver_request(self, *, parsed: Any, method: str) -> bool:
        path = str(getattr(parsed, "path", "") or "")
        if not path.startswith("/api/weaver/"):
            return False

        method_key = str(method or "GET").strip().upper() or "GET"
        upstream_host = _weaver_probe_host(WEAVER_HOST_ENV or "127.0.0.1")
        upstream_url = f"http://{upstream_host}:{WEAVER_PORT}{path}"
        query = str(getattr(parsed, "query", "") or "").strip()
        if query:
            upstream_url = f"{upstream_url}?{query}"

        request_headers: dict[str, str] = {}
        accept = str(self.headers.get("Accept", "") or "").strip()
        content_type = str(self.headers.get("Content-Type", "") or "").strip()
        if accept:
            request_headers["Accept"] = accept
        if content_type:
            request_headers["Content-Type"] = content_type

        request_body = b""
        if method_key in {"POST", "PUT", "PATCH", "DELETE"}:
            request_body = self._read_raw_body()

        timeout_seconds = 60.0 if method_key == "GET" else 90.0
        req = Request(
            upstream_url,
            data=request_body if request_body else None,
            headers=request_headers,
            method=method_key,
        )

        try:
            with urlopen(req, timeout=timeout_seconds) as resp:
                payload = resp.read()
                status_raw = int(getattr(resp, "status", HTTPStatus.OK))
                try:
                    status = HTTPStatus(status_raw)
                except ValueError:
                    status = HTTPStatus.OK
                response_type = str(
                    resp.headers.get("Content-Type", "application/json; charset=utf-8")
                    or "application/json; charset=utf-8"
                )
                self._send_bytes(payload, response_type, status=status)
                return True
        except urllib.error.HTTPError as exc:
            status_raw = int(getattr(exc, "code", HTTPStatus.BAD_GATEWAY))
            try:
                status = HTTPStatus(status_raw)
            except ValueError:
                status = HTTPStatus.BAD_GATEWAY
            payload = exc.read()
            response_type = "application/json; charset=utf-8"
            if getattr(exc, "headers", None) is not None:
                response_type = str(
                    exc.headers.get("Content-Type", response_type) or response_type
                )
            if payload:
                self._send_bytes(payload, response_type, status=status)
            else:
                self._send_json(
                    {
                        "ok": False,
                        "error": "weaver_upstream_error",
                        "status": status_raw,
                    },
                    status=status,
                )
            return True
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": "weaver_upstream_unreachable",
                    "detail": f"{exc.__class__.__name__}: {exc}",
                },
                status=HTTPStatus.BAD_GATEWAY,
            )
            return True

    def _muse_manager(self) -> Any:
        return get_muse_runtime_manager()

    def _muse_threat_radar_status(self) -> dict[str, Any]:
        return muse_threat_radar_utils_module.muse_threat_radar_status(
            server_module=sys.modules[__name__]
        )

    def _update_active_threats(
        self,
        radar: str,
        result: dict[str, Any],
    ) -> None:
        muse_threat_radar_utils_module.update_active_threats(
            server_module=sys.modules[__name__],
            radar=radar,
            result=result,
        )

    def _get_active_threat_nodes(self, radar: str) -> list[dict[str, Any]]:
        return muse_threat_radar_utils_module.get_active_threat_nodes(
            server_module=sys.modules[__name__],
            radar=radar,
        )

    def _muse_threat_radar_tick(
        self,
        *,
        now_monotonic: float | None = None,
        force: bool = False,
        reason: str = "manual",
    ) -> dict[str, Any]:
        return muse_threat_radar_utils_module.muse_threat_radar_tick(
            handler=self,
            server_module=sys.modules[__name__],
            now_monotonic=now_monotonic,
            force=force,
            reason=reason,
        )

    def _muse_tool_callback(self, *, tool_name: str) -> dict[str, Any]:
        return muse_runtime_backend_utils_module.muse_tool_callback(
            handler=self,
            tool_name=tool_name,
            server_module=sys.modules[__name__],
        )

    def _muse_reply_builder(
        self,
        *,
        messages: list[dict[str, Any]],
        context_block: str,
        mode: str,
        muse_id: str = "",
        turn_id: str = "",
    ) -> dict[str, Any]:
        return muse_runtime_backend_utils_module.muse_reply_builder(
            handler=self,
            messages=messages,
            context_block=context_block,
            mode=mode,
            muse_id=muse_id,
            turn_id=turn_id,
            server_module=sys.modules[__name__],
        )

    def _collect_catalog_fast(self) -> dict[str, Any]:
        return collect_catalog(
            self.part_root,
            self.vault_root,
            sync_inbox=False,
            include_pi_archive=False,
            include_world_log=False,
            include_embedding_state=False,
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

    def _runtime_controller(
        self,
    ) -> world_runtime_controller_module.WorldRuntimeController:
        return world_runtime_controller_module.WorldRuntimeController(
            part_root=self.part_root,
            vault_root=self.vault_root,
            task_queue=self.task_queue,
            council_chamber=self.council_chamber,
            myth_tracker=self.myth_tracker,
            life_tracker=self.life_tracker,
            collect_catalog_fast=self._collect_catalog_fast,
            schedule_runtime_catalog_refresh=self._schedule_runtime_catalog_refresh,
            schedule_runtime_inbox_sync=self._schedule_runtime_inbox_sync,
            collect_runtime_catalog_isolated=_collect_runtime_catalog_isolated,
            runtime_catalog_fallback=_runtime_catalog_fallback,
            runtime_catalog_cache_lock=_RUNTIME_CATALOG_CACHE_LOCK,
            runtime_catalog_collect_lock=_RUNTIME_CATALOG_COLLECT_LOCK,
            runtime_catalog_cache=_RUNTIME_CATALOG_CACHE,
            runtime_catalog_cache_seconds=_RUNTIME_CATALOG_CACHE_SECONDS,
            runtime_eta_mu_sync_enabled=_RUNTIME_ETA_MU_SYNC_ENABLED,
            resource_monitor_snapshot=_resource_monitor_snapshot,
            influence_tracker=_INFLUENCE_TRACKER,
            muse_runtime_snapshot=lambda: self._muse_manager().snapshot(),
            attach_ui_projection=attach_ui_projection,
            simulation_http_trim_catalog=_simulation_http_trim_catalog,
            collect_docker_simulation_snapshot=collect_docker_simulation_snapshot,
            build_simulation_state=build_simulation_state,
            build_ui_projection=build_ui_projection,
            entity_manifest=list(ENTITY_MANIFEST),
        )

    def _simulation_get_dependencies(
        self,
    ) -> simulation_get_controller_module.SimulationGetDependencies:
        return simulation_get_controller_module.SimulationGetDependencies(
            default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
            full_async_rebuild_enabled=_SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED,
            cache_seconds=_SIMULATION_HTTP_CACHE_SECONDS,
            compact_stale_fallback_seconds=_SIMULATION_HTTP_COMPACT_STALE_FALLBACK_SECONDS,
            disk_cold_start_seconds=_SIMULATION_HTTP_DISK_COLD_START_SECONDS,
            full_async_stale_max_age_seconds=_SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS,
            disk_cache_seconds=_SIMULATION_HTTP_DISK_CACHE_SECONDS,
            stale_fallback_seconds=_SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
            disk_fallback_max_age_seconds=_SIMULATION_HTTP_DISK_FALLBACK_MAX_AGE_SECONDS,
            compact_build_wait_seconds=_SIMULATION_HTTP_COMPACT_BUILD_WAIT_SECONDS,
            build_wait_seconds=_SIMULATION_HTTP_BUILD_WAIT_SECONDS,
            build_lock_acquire_timeout_seconds=_SIMULATION_HTTP_BUILD_LOCK_ACQUIRE_TIMEOUT_SECONDS,
            simulation_build_lock=_SIMULATION_HTTP_BUILD_LOCK,
            normalize_projection_perspective=normalize_projection_perspective,
            normalize_payload_mode=_simulation_ws_normalize_payload_mode,
            safe_bool_query=_safe_bool_query,
            safe_float=_safe_float,
            json_compact=_json_compact,
            simulation_http_request_profile=simulation_http_response_utils_module.simulation_http_request_profile,
            simulation_http_schedule_full_async_refresh=simulation_http_response_utils_module.simulation_http_schedule_full_async_refresh,
            simulation_http_send_response=simulation_http_response_utils_module.simulation_http_send_response,
            simulation_http_refresh_retry_payload=simulation_http_response_utils_module.simulation_http_refresh_retry_payload,
            simulation_http_stale_or_disk_body=simulation_http_fallback_utils_module.simulation_http_stale_or_disk_body,
            simulation_http_fallback_headers=simulation_http_fallback_utils_module.simulation_http_fallback_headers,
            simulation_http_error_payload=simulation_http_fallback_utils_module.simulation_http_error_payload,
            simulation_http_acquire_build_lock_or_respond=simulation_http_build_gate_utils_module.simulation_http_acquire_build_lock_or_respond,
            simulation_http_send_inflight_cached_response_if_any=simulation_http_build_gate_utils_module.simulation_http_send_inflight_cached_response_if_any,
            simulation_http_full_async_refresh_headers=_simulation_http_full_async_refresh_headers,
            simulation_http_compact_stale_fallback_body=_simulation_http_compact_stale_fallback_body,
            simulation_http_is_cold_start=_simulation_http_is_cold_start,
            simulation_http_cache_key=_simulation_http_cache_key,
            simulation_http_cached_body=_simulation_http_cached_body,
            simulation_http_disk_cache_load=_simulation_http_disk_cache_load,
            simulation_http_compact_cached_body=_simulation_http_compact_cached_body,
            simulation_http_cache_store=_simulation_http_cache_store,
            simulation_http_compact_cache_store=_simulation_http_compact_cache_store,
            simulation_http_compact_response_body=_simulation_http_compact_response_body,
            simulation_http_compact_simulation_payload=_simulation_http_compact_simulation_payload,
            simulation_http_disk_cache_store=_simulation_http_disk_cache_store,
            simulation_http_failure_backoff_snapshot=_simulation_http_failure_backoff_snapshot,
            simulation_http_failure_clear=_simulation_http_failure_clear,
            simulation_http_failure_record=_simulation_http_failure_record,
            simulation_http_wait_for_exact_cache=_simulation_http_wait_for_exact_cache,
            simulation_http_full_async_refresh_snapshot=_simulation_http_full_async_refresh_snapshot,
            simulation_ws_decode_cached_payload=_simulation_ws_decode_cached_payload,
            simulation_ws_payload_missing_graph_payload=_simulation_ws_payload_missing_graph_payload,
        )

    def _runtime_catalog_base(
        self,
        *,
        allow_inline_collect: bool = True,
        strict_collect: bool = False,
    ) -> dict[str, Any]:
        controller = self._runtime_controller()
        return controller.runtime_catalog_base(
            allow_inline_collect=allow_inline_collect,
            strict_collect=strict_collect,
        )

    def _runtime_catalog(
        self,
        *,
        perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
        include_projection: bool = True,
        include_runtime_fields: bool = True,
        allow_inline_collect: bool = True,
        strict_collect: bool = False,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        controller = self._runtime_controller()
        return controller.runtime_catalog(
            perspective=perspective,
            include_projection=include_projection,
            include_runtime_fields=include_runtime_fields,
            allow_inline_collect=allow_inline_collect,
            strict_collect=strict_collect,
        )

    def _runtime_simulation(
        self,
        catalog: dict[str, Any],
        queue_snapshot: dict[str, Any],
        influence_snapshot: dict[str, Any],
        *,
        perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
        include_unified_graph: bool = True,
        include_particle_dynamics: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        controller = self._runtime_controller()
        return controller.runtime_simulation(
            catalog,
            queue_snapshot,
            influence_snapshot,
            perspective=perspective,
            include_unified_graph=include_unified_graph,
            include_particle_dynamics=include_particle_dynamics,
        )

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
        if not ws_upgrade_controller_module.upgrade_websocket_or_respond(
            ws_key=ws_key,
            try_acquire_client_slot=_runtime_ws_try_acquire_client_slot,
            runtime_ws_client_snapshot=_runtime_ws_client_snapshot,
            send_json=self._send_json,
            send_bytes=self._send_bytes,
            send_upgrade_response=self._send_ws_upgrade_response,
            release_client_slot=_runtime_ws_release_client_slot,
        ):
            return

        ws_wire_mode = _normalize_ws_wire_mode(wire_mode)
        simulation_docker_ws_controller_module.handle_docker_websocket_stream(
            connection=self.connection,
            send_ws=lambda payload: self._send_ws_event(
                payload, wire_mode=ws_wire_mode
            ),
            collect_docker_simulation_snapshot=collect_docker_simulation_snapshot,
            consume_ws_client_frame=_consume_ws_client_frame,
            release_client_slot=_runtime_ws_release_client_slot,
            docker_refresh_seconds=DOCKER_SIMULATION_WS_REFRESH_SECONDS,
            docker_heartbeat_seconds=DOCKER_SIMULATION_WS_HEARTBEAT_SECONDS,
        )

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
        if not ws_upgrade_controller_module.upgrade_websocket_or_respond(
            ws_key=ws_key,
            try_acquire_client_slot=_runtime_ws_try_acquire_client_slot,
            runtime_ws_client_snapshot=_runtime_ws_client_snapshot,
            send_json=self._send_json,
            send_bytes=self._send_bytes,
            send_upgrade_response=self._send_ws_upgrade_response,
            release_client_slot=_runtime_ws_release_client_slot,
        ):
            return

        perspective_key = normalize_projection_perspective(perspective)
        stream_mode = _simulation_ws_normalize_delta_stream_mode(delta_stream_mode)
        payload_mode_key = _simulation_ws_normalize_payload_mode(payload_mode)
        particle_payload_key = _simulation_ws_normalize_particle_payload_mode(
            particle_payload_mode
        )
        use_cached_snapshots = _SIMULATION_WS_USE_CACHED_SNAPSHOTS
        effective_skip_catalog_bootstrap = bool(skip_catalog_bootstrap)
        stream_cached_snapshots = bool(
            use_cached_snapshots or effective_skip_catalog_bootstrap
        )
        ws_wire_mode = _normalize_ws_wire_mode(wire_mode)
        if (
            stream_cached_snapshots
            and ws_wire_mode == "arr"
            and _SIMULATION_WS_CACHE_FORCE_JSON_WIRE
        ):
            ws_wire_mode = "json"
        ws_send_controller = (
            simulation_ws_send_controller_module.SimulationWsSendController(
                chunk_enabled=bool(chunk_stream_enabled),
                wire_mode=ws_wire_mode,
                chunk_chars=_SIMULATION_WS_CHUNK_CHARS,
                stream_particle_max=_SIMULATION_WS_STREAM_PARTICLE_MAX,
                sim_tick_seconds=SIM_TICK_SECONDS,
                json_compact=_json_compact,
                simulation_ws_chunk_plan=_simulation_ws_chunk_plan,
                simulation_ws_chunk_messages=_simulation_ws_chunk_messages,
                send_ws_event=(
                    lambda payload, mode: self._send_ws_event(payload, wire_mode=mode)
                ),
                send_ws_text=self._send_ws_text,
                ws_clamp01=_ws_clamp01,
            )
        )
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
            ws_send_controller.send(payload)

        use_shared_ws_stream = bool(
            stream_mode == "workers"
            and stream_cached_snapshots
            and effective_skip_catalog_bootstrap
            and not catalog_events_enabled
        )
        if use_shared_ws_stream:

            def _collect_shared_stream_frame() -> tuple[dict[str, Any], str]:
                return simulation_ws_shared_payload_utils_module.collect_shared_stream_frame(
                    part_root=self.part_root,
                    perspective=perspective_key,
                    payload_mode=payload_mode_key,
                    load_cached_payload=_simulation_ws_load_cached_payload,
                    json_compact=_json_compact,
                )

            shared_stream_key = (
                f"sim-ws:{perspective_key}:{payload_mode_key}:{particle_payload_key}"
            )
            simulation_ws_shared_controller_module.handle_shared_simulation_websocket_stream(
                connection=self.connection,
                send_ws=send_ws,
                consume_ws_client_frame=_consume_ws_client_frame,
                release_client_slot=_runtime_ws_release_client_slot,
                stream_key=shared_stream_key,
                collect_frame=_collect_shared_stream_frame,
                refresh_seconds=max(0.25, _SIMULATION_WS_CACHE_REFRESH_SECONDS),
                heartbeat_seconds=max(
                    SIM_TICK_SECONDS,
                    _SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS,
                ),
            )
            return

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
        particle_page_cursor = 0
        last_muse_poll = 0.0
        last_config_runtime_version = _config_runtime_version_snapshot()
        tick_governor = get_governor() if _SIMULATION_WS_TICK_GOVERNOR_ENABLED else None
        last_governor_resource_refresh = 0.0
        governor_particle_cap = max(
            _SIMULATION_WS_GOVERNOR_MIN_PARTICLE_CAP,
            _SIMULATION_WS_STREAM_PARTICLE_MAX,
        )
        governor_graph_heartbeat_scale = 1.0

        ws_fast_bootstrap_particle_rows = max(
            24,
            min(96, int(_SIMULATION_WS_STREAM_PARTICLE_MAX)),
        )
        ws_fast_bootstrap_particles: list[dict[str, Any]] = []
        for index in range(ws_fast_bootstrap_particle_rows):
            angle = (float(index) / float(ws_fast_bootstrap_particle_rows)) * (
                2.0 * math.pi
            )
            band = float((index % 6) + 1) / 6.0
            radius = 0.14 + (0.22 * band)
            x_value = _ws_clamp01(0.5 + (math.cos(angle) * radius))
            y_value = _ws_clamp01(0.5 + (math.sin(angle) * radius))
            tangent = angle + (math.pi / 2.0)
            speed = 0.004 + (0.002 * (1.0 - band))
            ws_fast_bootstrap_particles.append(
                {
                    "id": f"ws-bootstrap:{index}",
                    "presence_id": "ws_bootstrap",
                    "owner_presence_id": "ws_bootstrap",
                    "target_presence_id": "ws_bootstrap",
                    "x": round(x_value, 5),
                    "y": round(y_value, 5),
                    "vx": round(math.cos(tangent) * speed, 6),
                    "vy": round(math.sin(tangent) * speed, 6),
                    "gravity_potential": round(0.2 + (0.1 * band), 5),
                    "energy": round(0.55 + (0.35 * band), 5),
                    "size": round(0.55 + (0.35 * band), 5),
                    "hue": int((index * 37) % 360),
                }
            )

        ws_fast_bootstrap_timestamp = datetime.now(timezone.utc).isoformat()
        ws_fast_bootstrap_simulation: dict[str, Any] = {
            "ok": True,
            "record": "eta-mu.ws.simulation-fast-bootstrap.v1",
            "generated_at": ws_fast_bootstrap_timestamp,
            "timestamp": ws_fast_bootstrap_timestamp,
            "perspective": perspective_key,
            "total": 0,
            "points": [],
            "presence_dynamics": {
                "generated_at": ws_fast_bootstrap_timestamp,
                "field_particles": ws_fast_bootstrap_particles,
                "daimoi_probabilistic": {
                    "record": "ημ.daimoi-probabilistic.v1",
                    "schema_version": "daimoi.probabilistic.v1",
                    "active": 0,
                    "spawned": 0,
                    "collisions": 0,
                    "deflects": 0,
                    "diffuses": 0,
                    "handoffs": 0,
                    "deliveries": 0,
                    "job_triggers": {},
                    "disabled": True,
                    "disabled_reason": "ws_fast_bootstrap",
                },
            },
        }
        ws_fast_bootstrap_projection: dict[str, Any] = {
            "record": "家_映.v1",
            "perspective": perspective_key,
            "ts": int(time.time() * 1000),
            "name": perspective_key,
            "summary": {
                "view": perspective_key,
                "active_entities": 0,
                "total_items": 0,
                "active_queue": 0,
            },
            "highlights": [],
            "narrative": {
                "tone": "bootstrap",
                "line_en": "stream bootstrap",
                "line_ja": "stream bootstrap",
            },
        }

        self._send_ws_event(
            {
                "type": "simulation",
                "simulation": ws_fast_bootstrap_simulation,
                "projection": ws_fast_bootstrap_projection,
            },
            wire_mode=ws_wire_mode,
        )

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

        def _build_ws_bootstrap_placeholder_payload() -> tuple[
            dict[str, Any], dict[str, Any]
        ]:
            timestamp_value = datetime.now(timezone.utc).isoformat()
            simulation_payload: dict[str, Any] = {
                "ok": True,
                "generated_at": timestamp_value,
                "timestamp": timestamp_value,
                "perspective": perspective_key,
                "total": 0,
                "audio": 0,
                "image": 0,
                "video": 0,
                "points": [],
                "presence_dynamics": {
                    "generated_at": timestamp_value,
                    "field_particles": [],
                    "daimoi_probabilistic": {
                        "record": "ημ.daimoi-probabilistic.v1",
                        "schema_version": "daimoi.probabilistic.v1",
                        "active": 0,
                        "spawned": 0,
                        "collisions": 0,
                        "deflects": 0,
                        "diffuses": 0,
                        "handoffs": 0,
                        "deliveries": 0,
                        "job_triggers": {},
                        "disabled": True,
                        "disabled_reason": "ws_bootstrap_cache_miss",
                    },
                },
            }
            projection_payload: dict[str, Any] = {
                "record": "家_映.v1",
                "perspective": perspective_key,
                "ts": int(time.time() * 1000),
                "name": perspective_key,
                "summary": {
                    "view": perspective_key,
                    "active_entities": 0,
                    "total_items": 0,
                    "active_queue": 0,
                },
                "highlights": [],
                "narrative": {
                    "tone": "bootstrap",
                    "line_en": "stream bootstrap",
                    "line_ja": "stream bootstrap",
                },
            }
            return simulation_payload, projection_payload

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
                include_particle_dynamics=True,
            )
            snapshot_payload = _build_ws_snapshot_payload(simulation)
            delta_payload = _build_ws_delta_payload(simulation)
            return snapshot_payload, delta_payload, projection

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
            if not effective_skip_catalog_bootstrap:
                muse_event_seq = (
                    muse_ws_stream_utils_module.stream_muse_bootstrap_events(
                        handler=self,
                        send_ws=send_ws,
                        muse_event_seq=muse_event_seq,
                        enabled=True,
                        event_limit=96,
                    )
                )

            if stream_cached_snapshots:
                if effective_skip_catalog_bootstrap:
                    cached_payload = _simulation_ws_load_cached_payload(
                        part_root=self.part_root,
                        perspective=perspective_key,
                        payload_mode=payload_mode_key,
                        allow_disabled_particle_dynamics=True,
                    )
                    if cached_payload is None:
                        simulation_payload = dict(ws_fast_bootstrap_simulation)
                        projection = dict(ws_fast_bootstrap_projection)
                        if payload_mode_key == "full":
                            simulation_delta_payload = _build_ws_delta_payload(
                                simulation_payload
                            )
                        else:
                            simulation_delta_payload = simulation_payload
                    else:
                        simulation_payload, projection = cached_payload
                        if payload_mode_key == "full":
                            simulation_delta_payload = _build_ws_delta_payload(
                                simulation_payload
                            )
                        else:
                            simulation_delta_payload = simulation_payload
                else:
                    cached_payload = _simulation_ws_load_cached_payload(
                        part_root=self.part_root,
                        perspective=perspective_key,
                        payload_mode=payload_mode_key,
                    )
                    if cached_payload is None:
                        (
                            simulation_payload,
                            projection,
                        ) = _build_ws_bootstrap_placeholder_payload()
                        if payload_mode_key == "full":
                            simulation_delta_payload = _build_ws_delta_payload(
                                simulation_payload
                            )
                        else:
                            simulation_delta_payload = simulation_payload
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
                if (
                    _SIMULATION_WS_BOOTSTRAP_REQUIRE_LIVE_REBUILD
                    and not effective_skip_catalog_bootstrap
                ):
                    needs_live_bootstrap = bool(
                        (not stream_particles)
                        or _simulation_ws_payload_missing_daimoi_summary(
                            simulation_delta_payload
                        )
                        or _simulation_ws_payload_missing_graph_payload(
                            simulation_payload
                        )
                    )
                if (
                    payload_mode_key != "full"
                    and effective_skip_catalog_bootstrap
                    and (
                        cached_payload is None
                        or _simulation_ws_payload_is_bootstrap_only(simulation_payload)
                        or _simulation_ws_payload_missing_graph_payload(
                            simulation_payload
                        )
                    )
                ):
                    needs_live_bootstrap = True
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
                    include_particle_dynamics=True,
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

            if not stream_cached_snapshots:
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
                        if simulation_ws_cache_utils_module.simulation_ws_should_bridge_to_http_cache(
                            ws_cache_payload,
                            min_field_particles=_SIMULATION_WS_HTTP_CACHE_BRIDGE_MIN_FIELD_PARTICLES,
                            payload_has_disabled_particle_dynamics=_simulation_ws_payload_has_disabled_particle_dynamics,
                        ):
                            ws_cache_body = _json_compact(ws_cache_payload).encode(
                                "utf-8"
                            )
                            _simulation_http_cache_store(
                                f"{perspective_key}|ws-stream|simulation",
                                ws_cache_body,
                            )
                            _simulation_http_disk_cache_store(
                                self.part_root,
                                perspective=perspective_key,
                                body=ws_cache_body,
                            )
                            last_ws_stream_cache_store = now_monotonic
                        last_simulation_full_broadcast = now_monotonic
                        last_projection_delta_broadcast = now_monotonic
                        last_graph_position_broadcast = now_monotonic
                        last_simulation_cache_refresh = now_monotonic
                        last_sim_tick = now_monotonic

                if (
                    not stream_cached_snapshots
                ) and now_monotonic - last_catalog_refresh >= CATALOG_REFRESH_SECONDS:
                    catalog, queue_snapshot, _, influence_snapshot, _ = (
                        self._runtime_catalog(
                            perspective=perspective_key,
                            include_projection=False,
                        )
                    )
                    last_catalog_refresh = now_monotonic

                if (
                    (not stream_cached_snapshots)
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
                    if stream_cached_snapshots
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
                        min(
                            _SIMULATION_WS_STREAM_PARTICLE_MAX,
                            governor_particle_cap,
                            ws_send_controller.network_particle_cap,
                        ),
                    )
                    effective_particle_payload_key = particle_payload_key
                    if stream_cached_snapshots:
                        cache_refresh_seconds = _SIMULATION_WS_CACHE_REFRESH_SECONDS
                        if effective_skip_catalog_bootstrap:
                            cache_refresh_seconds = max(1.0, cache_refresh_seconds)
                        cache_refresh_due = cache_refresh_seconds <= 0.0 or (
                            now_monotonic - last_simulation_cache_refresh
                            >= (
                                cache_refresh_seconds
                                * (
                                    1.0
                                    if effective_skip_catalog_bootstrap
                                    else load_scale
                                )
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
                                allow_disabled_particle_dynamics=effective_skip_catalog_bootstrap,
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
                            live_rebuild_needed = False
                            if (
                                payload_mode_key != "full"
                                and _SIMULATION_WS_BOOTSTRAP_REQUIRE_LIVE_REBUILD
                                and not effective_skip_catalog_bootstrap
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
                                live_rebuild_needed = True
                            if (
                                payload_mode_key != "full"
                                and effective_skip_catalog_bootstrap
                                and (
                                    cached_payload is None
                                    or _simulation_ws_payload_is_bootstrap_only(
                                        simulation_payload
                                    )
                                    or _simulation_ws_payload_missing_graph_payload(
                                        simulation_payload
                                    )
                                )
                            ):
                                live_rebuild_needed = True
                            if live_rebuild_needed:
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

                        tick_policy_state = simulation_ws_tick_policy_controller_module.build_simulation_ws_tick_policy(
                            catalog=catalog,
                            simulation_payload=simulation_payload,
                            runtime_inbox_lock_active=bool(
                                _RUNTIME_INBOX_SYNC_LOCK.locked()
                            ),
                            safe_float=_safe_float,
                            ws_send_pressure=ws_send_controller.send_pressure,
                            ws_network_particle_cap=ws_send_controller.network_particle_cap,
                            governor_particle_cap=governor_particle_cap,
                            stream_particle_max=_SIMULATION_WS_STREAM_PARTICLE_MAX,
                            particle_payload_key=particle_payload_key,
                            slack_ms_before_sim=tick_slack_ms(),
                            tick_budget_ms=tick_budget_ms,
                            guard_mode=guard_mode,
                            inbox_pending_soft=max(
                                1.0,
                                _safe_float(
                                    os.getenv("ETA_MU_INBOX_PENDING_SOFT", "64")
                                    or "64",
                                    64.0,
                                ),
                            ),
                        )
                        ingestion_pressure = tick_policy_state.ingestion_pressure
                        ws_particle_max = tick_policy_state.ws_particle_max
                        effective_particle_payload_key = (
                            tick_policy_state.effective_particle_payload_key
                        )
                        tick_policy = dict(tick_policy_state.tick_policy)

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
                        if effective_skip_catalog_bootstrap:
                            include_live_metrics = False
                            include_graph_variability = False
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
                        cache_payload_is_bootstrap = (
                            _simulation_ws_payload_is_bootstrap_only(simulation_payload)
                        )
                        if (
                            not cache_payload_is_bootstrap
                            and (cache_stream_particles or cache_total > 0)
                        ) and cache_store_due:
                            if simulation_ws_cache_utils_module.simulation_ws_should_bridge_to_http_cache(
                                ws_cache_payload,
                                min_field_particles=_SIMULATION_WS_HTTP_CACHE_BRIDGE_MIN_FIELD_PARTICLES,
                                payload_has_disabled_particle_dynamics=_simulation_ws_payload_has_disabled_particle_dynamics,
                            ):
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
                        if (
                            not stream_particles
                            and effective_skip_catalog_bootstrap
                            and ws_fast_bootstrap_particles
                            and isinstance(dynamics, dict)
                        ):
                            dynamics["field_particles"] = ws_fast_bootstrap_particles
                            simulation_payload["presence_dynamics"] = dynamics
                            stream_particles = ws_fast_bootstrap_particles
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
                            "network_send_pressure": round(
                                ws_send_controller.send_pressure,
                                4,
                            ),
                            "network_send_ema_ms": round(
                                ws_send_controller.send_ema_ms,
                                4,
                            ),
                            "network_particle_cap": int(
                                ws_send_controller.network_particle_cap
                            ),
                        }
                        tick_changed_keys = [
                            "timestamp",
                            "tick_elapsed_ms",
                            "slack_ms",
                            "ingestion_pressure",
                            "ws_particle_max",
                            "particle_payload_mode",
                            "network_send_pressure",
                            "network_send_ema_ms",
                            "network_particle_cap",
                        ]
                        dynamics_patch: dict[str, Any] = {}
                        if stream_particles:
                            (
                                sampled_particles,
                                sampled_meta,
                                particle_page_cursor,
                            ) = _simulation_ws_sample_particle_page(
                                stream_particles,
                                max_rows=ws_particle_max,
                                page_cursor=particle_page_cursor,
                                jitter_seed=(
                                    int(now_monotonic * 1000.0)
                                    + (simulation_worker_seq * 17)
                                ),
                            )
                            tick_particles = (
                                _simulation_ws_lite_field_particles(
                                    sampled_particles,
                                    max_rows=ws_particle_max,
                                )
                                if effective_particle_payload_key == "lite"
                                else []
                            )
                            if effective_particle_payload_key != "lite":
                                for row in sampled_particles:
                                    if len(tick_particles) >= ws_particle_max:
                                        break
                                    if isinstance(row, dict):
                                        tick_particles.append(dict(row))
                            dynamics_patch["field_particles"] = tick_particles
                            dynamics_patch["field_particles_page"] = {
                                "record": "eta-mu.ws.field-particles-page.v1",
                                "page_index": int(sampled_meta.get("page_index", 0)),
                                "page_total": int(sampled_meta.get("page_total", 1)),
                                "sample_size": int(
                                    sampled_meta.get("sample_size", len(tick_particles))
                                ),
                                "total": int(sampled_meta.get("total", 0)),
                                "start_index": int(sampled_meta.get("start_index", 0)),
                                "network_particle_cap": int(
                                    ws_send_controller.network_particle_cap
                                ),
                                "network_send_pressure": round(
                                    ws_send_controller.send_pressure,
                                    4,
                                ),
                            }
                            tick_changed_keys.append(
                                "presence_dynamics.field_particles"
                            )
                            tick_changed_keys.append(
                                "presence_dynamics.field_particles_page"
                            )
                        last_graph_position_broadcast = simulation_ws_delta_patch_controller_module.apply_simulation_ws_dynamics_patch(
                            dynamics=(dynamics if isinstance(dynamics, dict) else {}),
                            dynamics_patch=dynamics_patch,
                            tick_patch=tick_patch,
                            tick_changed_keys=tick_changed_keys,
                            graph_node_positions=(
                                graph_node_positions
                                if isinstance(graph_node_positions, dict)
                                else {}
                            ),
                            presence_anchor_positions=(
                                presence_anchor_positions
                                if isinstance(presence_anchor_positions, dict)
                                else {}
                            ),
                            now_monotonic=now_monotonic,
                            last_graph_position_broadcast=last_graph_position_broadcast,
                            graph_position_heartbeat_seconds=_SIMULATION_WS_GRAPH_POSITION_HEARTBEAT_SECONDS,
                            governor_graph_heartbeat_scale=governor_graph_heartbeat_scale,
                            tick_slack_ms=tick_slack_ms(),
                            ingestion_pressure=ingestion_pressure,
                            slack_ms_value=slack_ms_value,
                            ws_graph_pos_max_default=ws_graph_pos_max_default,
                            safe_int=_safe_int,
                            safe_float=_safe_float,
                            ws_clamp01=_ws_clamp01,
                        )

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

                        if (
                            tick_slack_ms() > 1.0
                            and not effective_skip_catalog_bootstrap
                        ):
                            (
                                muse_event_seq,
                                last_muse_poll,
                            ) = muse_ws_stream_utils_module.maybe_send_muse_events(
                                handler=self,
                                send_ws=send_ws,
                                now_monotonic=now_monotonic,
                                muse_event_seq=muse_event_seq,
                                last_muse_poll=last_muse_poll,
                                server_module=sys.modules[__name__],
                                event_limit=96,
                            )
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
                            include_particle_dynamics=True,
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

                    if tick_slack_ms() > 1.0 and not effective_skip_catalog_bootstrap:
                        (
                            muse_event_seq,
                            last_muse_poll,
                        ) = muse_ws_stream_utils_module.maybe_send_muse_events(
                            handler=self,
                            send_ws=send_ws,
                            now_monotonic=now_monotonic,
                            muse_event_seq=muse_event_seq,
                            last_muse_poll=last_muse_poll,
                            server_module=sys.modules[__name__],
                            event_limit=96,
                        )
                    last_sim_tick = now_monotonic

                if (
                    not stream_cached_snapshots
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

        if self._proxy_weaver_request(parsed=parsed, method="GET"):
            return

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
            payload, status = (
                simulation_management_controller_module.simulation_presets_get_response(
                    part_root=self.part_root
                )
            )
            if isinstance(payload, dict):
                self._send_json(payload, status=status)
            else:
                self._send_bytes(
                    _json_compact(payload).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status=status,
                )
            return

        if parsed.path == "/api/simulation/bootstrap":
            report = _simulation_bootstrap_snapshot_report()
            self._send_json(
                simulation_status_command_utils_module.simulation_bootstrap_status_payload(
                    job_snapshot=_simulation_bootstrap_job_snapshot(),
                    report=report,
                )
            )
            return

        if parsed.path == "/api/simulation/refresh-status":
            refresh_context = simulation_status_command_utils_module.simulation_refresh_status_context(
                params,
                default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
                normalize_projection_perspective=normalize_projection_perspective,
            )
            perspective = refresh_context.perspective
            cache_perspective = refresh_context.cache_perspective
            refresh_snapshot = _simulation_http_full_async_refresh_snapshot()
            refresh_public = simulation_status_command_utils_module.simulation_refresh_public_snapshot(
                refresh_snapshot,
                perspective=perspective,
                throttle_remaining_seconds=lambda snapshot, perspective_key: (
                    _simulation_http_full_async_throttle_remaining_seconds(
                        snapshot,
                        perspective=perspective_key,
                    )
                ),
                safe_float=_safe_float,
            )
            availability = simulation_status_command_utils_module.simulation_refresh_status_availability(
                part_root=self.part_root,
                cache_perspective=cache_perspective,
                cache_seconds=_SIMULATION_HTTP_CACHE_SECONDS,
                stale_max_age_seconds=_SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS,
                disk_cache_seconds=_SIMULATION_HTTP_DISK_CACHE_SECONDS,
                cached_body_reader=_simulation_http_cached_body,
                disk_cache_has_payload=_simulation_http_disk_cache_has_payload,
            )

            self._send_json(
                simulation_status_command_utils_module.simulation_refresh_status_payload(
                    perspective=perspective,
                    full_async_enabled=_SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED,
                    cache_seconds=_SIMULATION_HTTP_CACHE_SECONDS,
                    stale_max_age_seconds=_SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS,
                    lock_timeout_seconds=_SIMULATION_HTTP_FULL_ASYNC_LOCK_TIMEOUT_SECONDS,
                    max_running_seconds=_SIMULATION_HTTP_FULL_ASYNC_MAX_RUNNING_SECONDS,
                    start_min_interval_seconds=_SIMULATION_HTTP_FULL_ASYNC_START_MIN_INTERVAL_SECONDS,
                    availability=availability,
                    refresh_public=refresh_public,
                )
            )
            return

        if parsed.path == "/api/simulation/instances":
            payload, status = (
                simulation_management_controller_module.simulation_instances_list_response(
                    part_root=self.part_root
                )
            )
            if isinstance(payload, dict):
                self._send_json(payload, status=status)
            else:
                self._send_bytes(
                    _json_compact(payload).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status=status,
                )
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

            catalog, _, _, _, _ = self._runtime_catalog(
                perspective=perspective,
                allow_inline_collect=False,
                include_projection=False,
                include_runtime_fields=False,
            )
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

        if muse_mvc_controller_module.handle_muse_get_route(
            handler=self,
            path=parsed.path,
            params=params,
            send_json=self._send_json,
            server_module=sys.modules[__name__],
        ):
            return

        if muse_mvc_controller_module.handle_muse_threat_report_get_route(
            handler=self,
            path=parsed.path,
            params=params,
            send_json=self._send_json,
            server_module=sys.modules[__name__],
        ):
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

            cached_simulation_body = _simulation_http_cached_body(
                perspective=perspective,
                max_age_seconds=_SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
            )
            if cached_simulation_body is None:
                cached_simulation_body = _simulation_http_disk_cache_load(
                    self.part_root,
                    perspective=perspective,
                    max_age_seconds=max(
                        _SIMULATION_HTTP_STALE_FALLBACK_SECONDS,
                        _SIMULATION_HTTP_DISK_CACHE_SECONDS,
                    ),
                )
            if cached_simulation_body is not None:
                try:
                    cached_simulation_payload = json.loads(
                        cached_simulation_body.decode("utf-8")
                    )
                except Exception:
                    cached_simulation_payload = None
                if isinstance(cached_simulation_payload, dict):
                    cached_projection = cached_simulation_payload.get("projection")
                    if isinstance(cached_projection, dict):
                        self._send_json(
                            {
                                "ok": True,
                                "projection": cached_projection,
                                "simulation": cached_simulation_payload,
                                "default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
                                "perspectives": projection_perspective_options(),
                            }
                        )
                        return

            catalog, queue_snapshot, _, influence_snapshot, _ = self._runtime_catalog(
                perspective=perspective,
                include_projection=False,
                include_runtime_fields=False,
                allow_inline_collect=False,
            )
            simulation, projection = self._runtime_simulation(
                catalog,
                queue_snapshot,
                influence_snapshot,
                perspective=perspective,
                include_unified_graph=False,
                include_particle_dynamics=False,
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
            study_cache_key = f"limit={limit}|include_truth={int(include_truth_state)}"

            def _build_study_payload() -> dict[str, Any]:
                queue_snapshot = self.task_queue.snapshot(include_pending=True)
                council_snapshot = self.council_chamber.snapshot(
                    include_decisions=True,
                    limit=limit,
                )
                drift_payload = build_drift_scan_payload(
                    self.part_root, self.vault_root
                )
                resource_snapshot = _resource_monitor_snapshot(part_root=self.part_root)
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    resource_snapshot,
                    source="api.study",
                )

                truth_gate_blocked: bool | None = None
                if include_truth_state:
                    try:
                        truth_state = self._collect_catalog_fast().get(
                            "truth_state", {}
                        )
                        gate = (
                            truth_state.get("gate", {})
                            if isinstance(truth_state, dict)
                            else {}
                        )
                        if isinstance(gate, dict):
                            truth_gate_blocked = bool(gate.get("blocked", False))
                    except Exception:
                        truth_gate_blocked = None

                return build_study_snapshot(
                    self.part_root,
                    self.vault_root,
                    queue_snapshot=queue_snapshot,
                    council_snapshot=council_snapshot,
                    drift_payload=drift_payload,
                    truth_gate_blocked=truth_gate_blocked,
                    resource_snapshot=resource_snapshot,
                )

            self._send_json(
                get_or_build_study_snapshot_response(
                    cache_key=study_cache_key,
                    max_age_seconds=_STUDY_RESPONSE_CACHE_SECONDS,
                    builder=_build_study_payload,
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

        if parsed.path == "/api/github/conversation":
            repo_raw = str(params.get("repo", [""])[0] or "").strip()
            repo_parts = [
                segment.strip() for segment in repo_raw.split("/") if segment.strip()
            ]
            if len(repo_parts) != 2:
                self._send_json(
                    {"ok": False, "error": "repo_query_param_must_be_owner_slash_repo"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            repo = f"{repo_parts[0]}/{repo_parts[1]}"

            number = int(
                _safe_float(
                    str(params.get("number", ["0"])[0] or "0"),
                    0.0,
                )
            )
            if number <= 0:
                self._send_json(
                    {"ok": False, "error": "number_query_param_must_be_positive"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            kind = (
                str(params.get("kind", ["github:issue"])[0] or "github:issue")
                .strip()
                .lower()
            )
            max_comments = max(
                1,
                min(
                    240,
                    int(
                        _safe_float(
                            str(params.get("max_comments", ["120"])[0] or "120"),
                            120.0,
                        )
                    ),
                ),
            )
            max_root_body_chars = max(
                120,
                min(
                    48000,
                    int(
                        _safe_float(
                            str(
                                params.get("max_root_body_chars", ["18000"])[0]
                                or "18000"
                            ),
                            18000.0,
                        )
                    ),
                ),
            )
            max_comment_body_chars = max(
                120,
                min(
                    24000,
                    int(
                        _safe_float(
                            str(
                                params.get("max_comment_body_chars", ["8000"])[0]
                                or "8000"
                            ),
                            8000.0,
                        )
                    ),
                ),
            )
            max_markdown_chars = max(
                4000,
                min(
                    240000,
                    int(
                        _safe_float(
                            str(
                                params.get("max_markdown_chars", ["160000"])[0]
                                or "160000"
                            ),
                            160000.0,
                        )
                    ),
                ),
            )
            include_review_comments = _safe_bool_query(
                str(params.get("include_review_comments", ["true"])[0] or "true"),
                default=True,
            )
            timeout_s = max(
                2.0,
                min(
                    30.0,
                    _safe_float(
                        str(params.get("timeout_s", ["12"])[0] or "12"),
                        12.0,
                    ),
                ),
            )

            payload = _github_conversation_payload(
                repo=repo,
                number=number,
                kind=kind,
                max_comments=max_comments,
                max_root_body_chars=max_root_body_chars,
                max_comment_body_chars=max_comment_body_chars,
                max_markdown_chars=max_markdown_chars,
                include_review_comments=include_review_comments,
                timeout_s=timeout_s,
            )
            if bool(payload.get("ok", False)):
                self._send_json(payload)
                return

            status_code = int(_safe_float(payload.get("status", 0), 0.0))
            if status_code == 404:
                status = HTTPStatus.NOT_FOUND
            elif status_code == 400:
                status = HTTPStatus.BAD_REQUEST
            else:
                status = HTTPStatus.BAD_GATEWAY
            self._send_json(payload, status=status)
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
            simulation_get_controller_module.handle_simulation_get(
                params=params,
                part_root=self.part_root,
                runtime_catalog=self._runtime_catalog,
                runtime_simulation=self._runtime_simulation,
                schedule_full_simulation_async_refresh=self._schedule_full_simulation_async_refresh,
                send_json=self._send_json,
                send_bytes=self._send_bytes,
                dependencies=self._simulation_get_dependencies(),
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
        if self._proxy_weaver_request(parsed=parsed, method="DELETE"):
            return

        if parsed.path.startswith("/api/simulation/instances/"):
            instance_id = parsed.path.split("/")[-1]
            payload, status = (
                simulation_management_controller_module.simulation_instance_delete_response(
                    part_root=self.part_root,
                    instance_id=instance_id,
                )
            )
            if isinstance(payload, dict):
                self._send_json(payload, status=status)
            else:
                self._send_bytes(
                    _json_compact(payload).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status=status,
                )
            return

        self._send_json(
            {"ok": False, "error": "method not allowed"},
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if self._proxy_weaver_request(parsed=parsed, method="POST"):
            return

        if parsed.path == "/api/simulation/refresh":
            req = self._read_json_body() or {}
            simulation_post_controller_module.handle_simulation_refresh_post(
                req=req,
                default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
                normalize_projection_perspective=normalize_projection_perspective,
                safe_bool_query=_safe_bool_query,
                full_async_refresh_enabled=_SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED,
                full_async_refresh_snapshot=_simulation_http_full_async_refresh_snapshot,
                full_async_refresh_cancel=_simulation_http_full_async_refresh_cancel,
                schedule_full_simulation_async_refresh=self._schedule_full_simulation_async_refresh,
                safe_float=_safe_float,
                send_json=self._send_json,
            )
            return

        if parsed.path == "/api/simulation/bootstrap":
            req = self._read_json_body() or {}
            simulation_post_controller_module.handle_simulation_bootstrap_post(
                req=req,
                default_perspective=PROJECTION_DEFAULT_PERSPECTIVE,
                normalize_projection_perspective=normalize_projection_perspective,
                safe_bool_query=_safe_bool_query,
                run_simulation_bootstrap=self._run_simulation_bootstrap,
                simulation_bootstrap_job_start=_simulation_bootstrap_job_start,
                simulation_bootstrap_job_mark_phase=_simulation_bootstrap_job_mark_phase,
                simulation_bootstrap_job_complete=_simulation_bootstrap_job_complete,
                simulation_bootstrap_job_fail=_simulation_bootstrap_job_fail,
                send_json=self._send_json,
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
            payload, status = (
                simulation_management_controller_module.simulation_instances_spawn_response(
                    part_root=self.part_root,
                    req_payload=req,
                )
            )
            if isinstance(payload, dict):
                self._send_json(payload, status=status)
            else:
                self._send_bytes(
                    _json_compact(payload).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status=status,
                )
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

        if muse_mvc_controller_module.handle_muse_post_route(
            handler=self,
            path=parsed.path,
            read_json_body=self._read_json_body,
            send_json=self._send_json,
            headers=self.headers,
            server_module=sys.modules[__name__],
        ):
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
    *,
    host_label: str | None = None,
):
    runtime_host = f"{host}:{port}"
    resolved_host_label = str(host_label or runtime_host).strip() or runtime_host
    receipts_path = _ensure_receipts_log_path(vault_root, part_root)
    queue_log_path = (
        vault_root / ".opencode" / "runtime" / "task_queue.v1.jsonl"
    ).resolve()
    council_log_path = (vault_root / COUNCIL_DECISION_LOG_REL).resolve()

    task_queue = TaskQueue(
        queue_log_path,
        receipts_path,
        owner="Err",
        host=runtime_host,
    )
    council_chamber = CouncilChamber(
        council_log_path,
        receipts_path,
        owner="Err",
        host=runtime_host,
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
    BoundWorldHandler.host_label = resolved_host_label
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


def create_http_server(
    part_root: Path,
    vault_root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    host_label: str | None = None,
) -> BoundedThreadingHTTPServer:
    _ensure_weaver_service(part_root, host)
    handler_class = make_handler(
        part_root,
        vault_root,
        host,
        port,
        host_label=host_label,
    )
    return BoundedThreadingHTTPServer(
        (host, port),
        handler_class,
        max_threads=_RUNTIME_HTTP_MAX_THREADS,
    )


def serve(
    part_root: Path,
    vault_root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
):
    server = create_http_server(
        part_root,
        vault_root,
        host,
        port,
        host_label=f"{host}:{port}",
    )
    print(f"Starting server on {host}:{port}")
    _schedule_simulation_http_warmup(host=host, port=port)
    server.serve_forever()


def serve_asgi(
    part_root: Path,
    vault_root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
    *,
    legacy_port: int = 0,
) -> None:
    from .asgi_transport import run_asgi_transport

    run_asgi_transport(
        part_root=part_root,
        vault_root=vault_root,
        host=host,
        port=port,
        legacy_port=max(0, int(legacy_port)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_transport = (
        str(os.getenv("WORLD_WEB_TRANSPORT", "asgi") or "asgi").strip().lower()
    )
    if default_transport not in {"legacy", "asgi"}:
        default_transport = "legacy"
    default_legacy_port = max(
        0,
        int(
            _safe_float(
                str(os.getenv("WORLD_WEB_LEGACY_PORT", "0") or "0"),
                0.0,
            )
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--transport",
        choices=["legacy", "asgi"],
        default=default_transport,
    )
    parser.add_argument("--legacy-port", type=int, default=default_legacy_port)
    parser.add_argument("--part-root", type=Path, default=Path("."))
    parser.add_argument("--vault-root", type=Path, default=Path(".."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if str(args.transport or "legacy").strip().lower() == "asgi":
        serve_asgi(
            args.part_root,
            args.vault_root,
            args.host,
            int(args.port),
            legacy_port=max(0, int(args.legacy_port)),
        )
    else:
        serve(args.part_root, args.vault_root, args.host, int(args.port))
    return 0
