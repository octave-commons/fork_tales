from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import math
import os
import re
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from .ai import build_presence_say_payload, _eta_mu_deterministic_vector
from .constants import (
    COUNCIL_DECISION_HISTORY_LIMIT,
    COUNCIL_EVENT_VERSION,
    COUNCIL_MIN_MEMBER_WORLD_INFLUENCE,
    COUNCIL_MIN_OVERLAP_MEMBERS,
    DOCKER_AUTORESTART_COOLDOWN_SECONDS,
    DOCKER_AUTORESTART_ENABLED,
    DOCKER_AUTORESTART_EXCLUDE_GLOBS,
    DOCKER_AUTORESTART_INCLUDE_GLOBS,
    DOCKER_AUTORESTART_REQUIRE_COUNCIL,
    DOCKER_AUTORESTART_SERVICES,
    DOCKER_AUTORESTART_TIMEOUT_SECONDS,
    ETA_MU_TRUTH_STATE_RECORD,
    FIELD_TO_PRESENCE,
    FILE_ORGANIZER_PROFILE,
    FILE_SENTINEL_PROFILE,
    PROMPTDB_CONTRACT_GLOBS,
    PROMPTDB_OPEN_QUESTIONS_PACKET,
    PROMPTDB_PACKET_GLOBS,
    PROMPTDB_REFRESH_DEBOUNCE_SECONDS,
    STUDY_EVENT_VERSION,
    STUDY_SNAPSHOT_HISTORY_LIMIT,
    STUDY_SNAPSHOT_LOG_REL,
    TASK_QUEUE_EVENT_VERSION,
    THE_COUNCIL_PROFILE,
    TRUTH_ALLOWED_PROOF_KINDS,
    TRUTH_BINDING_CACHE_SECONDS,
    TRUTH_BINDING_GUARD_THETA,
    ETA_MU_INGEST_TEXT_DIMS,
    WORLD_LOG_EVENT_LIMIT,
    WORLD_LOG_RELATION_LIMIT,
    WS_MAGIC,
    WIKIMEDIA_EVENTSTREAMS_BASE_URL,
    WIKIMEDIA_EVENTSTREAMS_DEDUPE_TTL_SECONDS,
    WIKIMEDIA_EVENTSTREAMS_DETAIL_CHAR_LIMIT,
    WIKIMEDIA_EVENTSTREAMS_ENABLED,
    WIKIMEDIA_EVENTSTREAMS_FETCH_TIMEOUT_SECONDS,
    WIKIMEDIA_EVENTSTREAMS_MAX_BYTES,
    WIKIMEDIA_EVENTSTREAMS_MAX_EVENTS_PER_POLL,
    WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS,
    WIKIMEDIA_EVENTSTREAMS_RATE_LIMIT_PER_POLL,
    WIKIMEDIA_EVENTSTREAMS_STREAMS,
    NWS_ALERTS_ENABLED,
    NWS_ALERTS_ENDPOINT,
    NWS_ALERTS_POLL_INTERVAL_SECONDS,
    NWS_ALERTS_FETCH_TIMEOUT_SECONDS,
    NWS_ALERTS_MAX_BYTES,
    NWS_ALERTS_MAX_ALERTS_PER_POLL,
    NWS_ALERTS_RATE_LIMIT_PER_POLL,
    NWS_ALERTS_DEDUPE_TTL_SECONDS,
    NWS_ALERTS_DETAIL_CHAR_LIMIT,
    SWPC_ALERTS_ENABLED,
    SWPC_ALERTS_ENDPOINT,
    SWPC_ALERTS_POLL_INTERVAL_SECONDS,
    SWPC_ALERTS_FETCH_TIMEOUT_SECONDS,
    SWPC_ALERTS_MAX_BYTES,
    SWPC_ALERTS_MAX_ALERTS_PER_POLL,
    SWPC_ALERTS_RATE_LIMIT_PER_POLL,
    SWPC_ALERTS_DEDUPE_TTL_SECONDS,
    SWPC_ALERTS_DETAIL_CHAR_LIMIT,
    GIBS_LAYERS_ENABLED,
    GIBS_LAYERS_CAPABILITIES_ENDPOINT,
    GIBS_LAYERS_TARGETS,
    GIBS_LAYERS_POLL_INTERVAL_SECONDS,
    GIBS_LAYERS_FETCH_TIMEOUT_SECONDS,
    GIBS_LAYERS_MAX_BYTES,
    GIBS_LAYERS_MAX_LAYERS_PER_POLL,
    GIBS_LAYERS_RATE_LIMIT_PER_POLL,
    GIBS_LAYERS_DEDUPE_TTL_SECONDS,
    GIBS_LAYERS_DETAIL_CHAR_LIMIT,
    EONET_EVENTS_ENABLED,
    EONET_EVENTS_ENDPOINT,
    EONET_EVENTS_POLL_INTERVAL_SECONDS,
    EONET_EVENTS_FETCH_TIMEOUT_SECONDS,
    EONET_EVENTS_MAX_BYTES,
    EONET_EVENTS_MAX_EVENTS_PER_POLL,
    EONET_EVENTS_RATE_LIMIT_PER_POLL,
    EONET_EVENTS_DEDUPE_TTL_SECONDS,
    EONET_EVENTS_DETAIL_CHAR_LIMIT,
    GIBS_LAYERS_TILE_MATRIX_SET,
    GIBS_LAYERS_TILE_MATRIX,
    GIBS_LAYERS_TILE_ROW,
    GIBS_LAYERS_TILE_COL,
    EMSC_STREAM_ENABLED,
    EMSC_STREAM_URL,
    EMSC_STREAM_POLL_INTERVAL_SECONDS,
    EMSC_STREAM_FETCH_TIMEOUT_SECONDS,
    EMSC_STREAM_MAX_BYTES,
    EMSC_STREAM_MAX_EVENTS_PER_POLL,
    EMSC_STREAM_RATE_LIMIT_PER_POLL,
    EMSC_STREAM_DEDUPE_TTL_SECONDS,
    EMSC_STREAM_DETAIL_CHAR_LIMIT,
    _PROMPTDB_CACHE,
    _PROMPTDB_CACHE_LOCK,
    _STUDY_SNAPSHOT_CACHE,
    _STUDY_SNAPSHOT_LOCK,
    _TRUTH_BINDING_CACHE,
    _TRUTH_BINDING_CACHE_LOCK,
)
from .metrics import (
    _clamp01,
    _dominant_eta_mu_field,
    _infer_eta_mu_field_scores,
    _json_deep_clone,
    _resource_monitor_snapshot,
    _safe_float,
    _stable_ratio,
)
from .paths import (
    _append_receipt_line,
    _ensure_receipts_log_path,
    _eta_mu_inbox_rel_path,
    _eta_mu_inbox_root,
    _eta_mu_scan_candidates,
    _eta_mu_substrate_root,
    _locate_receipts_log,
    _parse_receipt_line,
    _safe_rel_path,
    _split_receipt_refs,
    _nws_alerts_log_path,
    _swpc_alerts_log_path,
    _gibs_layers_log_path,
    _eonet_events_log_path,
    _emsc_stream_log_path,
    _wikimedia_stream_log_path,
)
from .projection import _projection_presence_impacts, _semantic_xy_from_embedding
from .simulation import _normalize_path_for_file_id
from .db import (
    _cosine_similarity,
    _embedding_db_upsert_append_only,
    _load_eta_mu_registry_entries,
    _load_image_comment_entries,
    _load_presence_account_entries,
)


_WORLD_LOG_EMBEDDING_IDS: set[str] = set()
_WORLD_LOG_EMBEDDING_IDS_LOCK = threading.Lock()

WIKIMEDIA_EVENT_RECORD = "eta-mu.wikimedia-event.v1"
WIKIMEDIA_STREAM_EVENT_RECORD = "eta-mu.wikimedia-stream.v1"
NWS_ALERT_RECORD = "eta-mu.nws-alert.v1"
NWS_ALERT_STREAM_EVENT_RECORD = "eta-mu.nws-alert-stream.v1"
SWPC_ALERT_RECORD = "eta-mu.swpc-alert.v1"
SWPC_ALERT_STREAM_EVENT_RECORD = "eta-mu.swpc-alert-stream.v1"
GIBS_LAYER_RECORD = "eta-mu.gibs-layer.v1"
GIBS_LAYER_STREAM_EVENT_RECORD = "eta-mu.gibs-layer-stream.v1"
EONET_EVENT_RECORD = "eta-mu.eonet-event.v1"
EONET_STREAM_EVENT_RECORD = "eta-mu.eonet-stream.v1"
EMSC_EVENT_RECORD = "eta-mu.emsc-event.v1"
EMSC_STREAM_EVENT_RECORD = "eta-mu.emsc-stream.v1"

_WIKIMEDIA_STREAM_LOCK = threading.Lock()
_WIKIMEDIA_STREAM_CACHE: dict[str, Any] = {
    "last_poll_monotonic": 0.0,
    "connected": False,
    "paused": False,
    "seen_ids": {},
}

_NWS_ALERTS_LOCK = threading.Lock()
_NWS_ALERTS_CACHE: dict[str, Any] = {
    "last_poll_monotonic": 0.0,
    "connected": False,
    "paused": False,
    "seen_ids": {},
}

_SWPC_ALERTS_LOCK = threading.Lock()
_SWPC_ALERTS_CACHE: dict[str, Any] = {
    "last_poll_monotonic": 0.0,
    "connected": False,
    "paused": False,
    "seen_ids": {},
}

_GIBS_LAYERS_LOCK = threading.Lock()
_GIBS_LAYERS_CACHE: dict[str, Any] = {
    "last_poll_monotonic": 0.0,
    "connected": False,
    "paused": False,
    "seen_ids": {},
}

_EONET_EVENTS_LOCK = threading.Lock()
_EONET_EVENTS_CACHE: dict[str, Any] = {
    "last_poll_monotonic": 0.0,
    "connected": False,
    "paused": False,
    "seen_ids": {},
}

_EMSC_STREAM_LOCK = threading.Lock()
_EMSC_STREAM_CACHE: dict[str, Any] = {
    "last_poll_monotonic": 0.0,
    "connected": False,
    "paused": False,
    "seen_ids": {},
}


def _world_web_symbol(name: str, default: Any) -> Any:
    module = sys.modules.get("code.world_web")
    if module is None:
        return default
    return getattr(module, name, default)


def _cfg_bool(name: str, default: bool) -> bool:
    return bool(_world_web_symbol(name, default))


def _cfg_int(name: str, default: int) -> int:
    return int(_safe_float(_world_web_symbol(name, default), float(default)))


def _cfg_float(name: str, default: float) -> float:
    return _safe_float(_world_web_symbol(name, default), default)


def _cfg_str(name: str, default: str) -> str:
    return str(_world_web_symbol(name, default) or default)


def _split_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _extract_lisp_string(source: str, key: str) -> str | None:
    match = re.search(rf"\({re.escape(key)}\s+\"([^\"]+)\"\)", source)
    if match:
        return match.group(1)
    return None


def _extract_lisp_keyword(source: str, key: str) -> str | None:
    match = re.search(rf"\({re.escape(key)}\s+(:[-\w]+)\)", source)
    if match:
        return match.group(1)
    return None


def _extract_contract_name(source: str) -> str | None:
    match = re.search(r"\(contract\s+\"([^\"]+)\"", source)
    if match:
        return match.group(1)
    match = re.search(r"\(契\b[\s\S]*?\(id\s+\"([^\"]+)\"", source)
    if match:
        return match.group(1)
    return None


def parse_promptdb_packet(packet_path: Path, promptdb_root: Path) -> dict[str, Any]:
    source = packet_path.read_text("utf-8")
    packet_id = _extract_lisp_string(source, "id")
    title = _extract_lisp_string(source, "title")
    version = _extract_lisp_string(source, "v")
    kind = _extract_lisp_keyword(source, "kind")

    tags_match = re.search(r"\(tags\s+\[([^\]]*)\]\)", source, re.DOTALL)
    tags = re.findall(r":[-\w]+", tags_match.group(1)) if tags_match else []

    routing = {
        "target": _extract_lisp_keyword(source, "target"),
        "handler": _extract_lisp_keyword(source, "handler"),
        "mode": _extract_lisp_keyword(source, "mode"),
    }

    rel_path = str(packet_path.relative_to(promptdb_root.parent)).replace("\\", "/")
    node_key = packet_id or rel_path

    return {
        "node_key": node_key,
        "path": rel_path,
        "id": packet_id,
        "v": version,
        "kind": kind,
        "title": title,
        "tags": tags,
        "routing": routing,
    }


def parse_promptdb_contract(contract_path: Path, promptdb_root: Path) -> dict[str, Any]:
    source = contract_path.read_text("utf-8")
    contract_name = _extract_contract_name(source)
    rel_path = str(contract_path.relative_to(promptdb_root.parent)).replace("\\", "/")
    node_key = contract_name or rel_path
    return {
        "node_key": node_key,
        "path": rel_path,
        "id": contract_name,
        "v": "",
        "kind": ":contract",
        "title": contract_name or contract_path.stem,
        "tags": [":contract"],
        "routing": {"target": None, "handler": None, "mode": None},
    }


def _iter_promptdb_files(promptdb_root: Path) -> list[Path]:
    rows: list[Path] = []
    for pattern in (*PROMPTDB_PACKET_GLOBS, *PROMPTDB_CONTRACT_GLOBS):
        rows.extend(path for path in promptdb_root.rglob(pattern) if path.is_file())
    deduped = {path.resolve(): path for path in rows}
    return [deduped[key] for key in sorted(deduped)]


def _promptdb_signature(paths: list[Path], promptdb_root: Path) -> str:
    rows: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(promptdb_root)).replace("\\", "/")
        rows.append(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}")
    rows.sort()
    return hashlib.sha1("\n".join(rows).encode("utf-8")).hexdigest()


def _clone_promptdb_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": str(snapshot.get("root", "")),
        "packet_count": int(snapshot.get("packet_count", 0)),
        "contract_count": int(snapshot.get("contract_count", 0)),
        "file_count": int(snapshot.get("file_count", 0)),
        "packets": [dict(item) for item in snapshot.get("packets", [])],
        "contracts": [dict(item) for item in snapshot.get("contracts", [])],
        "errors": [dict(item) for item in snapshot.get("errors", [])],
    }


def _build_promptdb_snapshot(promptdb_root: Path, paths: list[Path]) -> dict[str, Any]:
    packets: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    for packet_path in paths:
        is_contract = packet_path.name.endswith(".contract.lisp")
        try:
            parsed = (
                parse_promptdb_contract(packet_path, promptdb_root)
                if is_contract
                else parse_promptdb_packet(packet_path, promptdb_root)
            )
            node_key = str(parsed.get("node_key") or "")
            if not node_key or node_key in seen_keys:
                continue
            seen_keys.add(node_key)
            if is_contract:
                contracts.append(parsed)
            else:
                packets.append(parsed)
        except Exception as exc:
            rel = str(packet_path).replace("\\", "/")
            errors.append({"path": rel, "error": str(exc)})

    return {
        "root": str(promptdb_root),
        "packet_count": len(packets),
        "contract_count": len(contracts),
        "file_count": len(paths),
        "packets": packets,
        "contracts": contracts,
        "errors": errors,
    }


def locate_promptdb_root(vault_root: Path) -> Path | None:
    candidates: list[Path] = []
    candidates.append(vault_root.resolve())
    candidates.extend(vault_root.resolve().parents)
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        candidate = base / ".opencode" / "promptdb"
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def collect_promptdb_packets(vault_root: Path) -> dict[str, Any]:
    promptdb_root = locate_promptdb_root(vault_root)
    checked_at = datetime.now(timezone.utc).isoformat()
    if promptdb_root is None:
        return {
            "root": "",
            "packet_count": 0,
            "contract_count": 0,
            "file_count": 0,
            "packets": [],
            "contracts": [],
            "errors": [],
            "refresh": {
                "strategy": "polling+debounce",
                "debounce_ms": int(PROMPTDB_REFRESH_DEBOUNCE_SECONDS * 1000),
                "checks": 0,
                "refreshes": 0,
                "cache_hits": 0,
                "last_checked_at": checked_at,
                "last_refreshed_at": "",
                "last_decision": "promptdb-root-missing",
            },
        }

    with _PROMPTDB_CACHE_LOCK:
        root_str = str(promptdb_root)
        now_monotonic = time.monotonic()
        _PROMPTDB_CACHE["checks"] = int(_PROMPTDB_CACHE.get("checks", 0)) + 1
        _PROMPTDB_CACHE["last_checked_at"] = checked_at

        if _PROMPTDB_CACHE.get("root") != root_str:
            _PROMPTDB_CACHE["root"] = root_str
            _PROMPTDB_CACHE["signature"] = ""
            _PROMPTDB_CACHE["snapshot"] = None
            _PROMPTDB_CACHE["last_decision"] = "root-changed"
            _PROMPTDB_CACHE["last_check_monotonic"] = 0.0

        elapsed = now_monotonic - float(
            _PROMPTDB_CACHE.get("last_check_monotonic", 0.0)
        )
        snapshot = _PROMPTDB_CACHE.get("snapshot")
        if snapshot is not None and elapsed < PROMPTDB_REFRESH_DEBOUNCE_SECONDS:
            _PROMPTDB_CACHE["cache_hits"] = (
                int(_PROMPTDB_CACHE.get("cache_hits", 0)) + 1
            )
            _PROMPTDB_CACHE["last_decision"] = "debounced-cache"
        else:
            paths = _iter_promptdb_files(promptdb_root)
            signature = _promptdb_signature(paths, promptdb_root)
            if snapshot is None or signature != _PROMPTDB_CACHE.get("signature"):
                snapshot = _build_promptdb_snapshot(promptdb_root, paths)
                _PROMPTDB_CACHE["snapshot"] = snapshot
                _PROMPTDB_CACHE["signature"] = signature
                _PROMPTDB_CACHE["refreshes"] = (
                    int(_PROMPTDB_CACHE.get("refreshes", 0)) + 1
                )
                _PROMPTDB_CACHE["last_refreshed_at"] = checked_at
                _PROMPTDB_CACHE["last_decision"] = "refreshed"
            else:
                _PROMPTDB_CACHE["cache_hits"] = (
                    int(_PROMPTDB_CACHE.get("cache_hits", 0)) + 1
                )
                _PROMPTDB_CACHE["last_decision"] = "cache-hit"
        _PROMPTDB_CACHE["last_check_monotonic"] = now_monotonic

        stable_snapshot = _clone_promptdb_snapshot(
            _PROMPTDB_CACHE.get("snapshot")
            or {
                "root": root_str,
                "packet_count": 0,
                "contract_count": 0,
                "file_count": 0,
                "packets": [],
                "contracts": [],
                "errors": [],
            }
        )

        stable_snapshot["refresh"] = {
            "strategy": "polling+debounce",
            "debounce_ms": int(PROMPTDB_REFRESH_DEBOUNCE_SECONDS * 1000),
            "checks": int(_PROMPTDB_CACHE.get("checks", 0)),
            "refreshes": int(_PROMPTDB_CACHE.get("refreshes", 0)),
            "cache_hits": int(_PROMPTDB_CACHE.get("cache_hits", 0)),
            "last_checked_at": str(_PROMPTDB_CACHE.get("last_checked_at", "")),
            "last_refreshed_at": str(_PROMPTDB_CACHE.get("last_refreshed_at", "")),
            "last_decision": str(_PROMPTDB_CACHE.get("last_decision", "unknown")),
        }
        return stable_snapshot


def _docker_compose_service_names(compose_path: Path) -> list[str]:
    if not compose_path.exists() or not compose_path.is_file():
        return []
    try:
        lines = compose_path.read_text("utf-8").splitlines()
    except OSError:
        return []

    services: list[str] = []
    in_services = False
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if not in_services:
            if stripped == "services:":
                in_services = True
            continue
        if indent == 0:
            break
        if indent == 2:
            match = re.match(r"([A-Za-z0-9_.-]+):\s*$", stripped)
            if match:
                services.append(match.group(1))
    return services


def _path_glob_allowed(
    path_value: str, includes: list[str], excludes: list[str]
) -> bool:
    normalized = _normalize_path_for_file_id(path_value)
    if not normalized:
        return False
    include_match = True
    if includes:
        include_match = any(
            fnmatch.fnmatch(normalized, pattern) for pattern in includes
        )
    if not include_match:
        return False
    if excludes and any(fnmatch.fnmatch(normalized, pattern) for pattern in excludes):
        return False
    return True


def _event_source_rel_path(path_value: str, vault_root: Path, part_root: Path) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/library/"):
        normalized = normalized[len("/library/") :]
    candidate = Path(normalized)
    if candidate.is_absolute():
        for base in [vault_root.resolve(), part_root.resolve()]:
            try:
                rel = candidate.resolve().relative_to(base)
                return _normalize_path_for_file_id(str(rel))
            except (OSError, ValueError):
                continue
    return _normalize_path_for_file_id(normalized)


def _pairwise_overlap_boundaries(members: list[str]) -> list[dict[str, str]]:
    clean_members = [item for item in members if str(item).strip()]
    boundaries: list[dict[str, str]] = []
    for left_idx in range(len(clean_members)):
        for right_idx in range(left_idx + 1, len(clean_members)):
            left = clean_members[left_idx]
            right = clean_members[right_idx]
            boundary_id = hashlib.sha1(f"{left}|{right}".encode("utf-8")).hexdigest()[
                :12
            ]
            boundaries.append(
                {"id": f"boundary:{boundary_id}", "left": left, "right": right}
            )
    return boundaries


def _influence_world_map(influence_snapshot: dict[str, Any]) -> dict[str, float]:
    rows = _projection_presence_impacts(None, influence_snapshot)
    world_map: dict[str, float] = {}
    for row in rows:
        presence_id = str(row.get("id", "")).strip()
        if not presence_id:
            continue
        affects = row.get("affects", {}) if isinstance(row.get("affects"), dict) else {}
        world_map[presence_id] = _clamp01(_safe_float(affects.get("world", 0.0), 0.0))

    file_ratio = _clamp01(
        _safe_float(influence_snapshot.get("file_changes_120s", 0.0), 0.0) / 24.0
    )
    queue_pending_ratio = _clamp01(
        _safe_float(
            (influence_snapshot.get("task_queue") or {}).get("pending_count", 0.0), 0.0
        )
        / 8.0
    )
    world_map.setdefault(
        FILE_SENTINEL_PROFILE["id"], _clamp01((file_ratio * 0.72) + 0.22)
    )
    world_map.setdefault(
        FILE_ORGANIZER_PROFILE["id"], _clamp01((file_ratio * 0.58) + 0.28)
    )
    world_map.setdefault(
        THE_COUNCIL_PROFILE["id"], _clamp01((queue_pending_ratio * 0.6) + 0.34)
    )
    return world_map


def _file_graph_overlap_context(
    file_graph: dict[str, Any], source_rel_path: str
) -> dict[str, Any]:
    normalized_source = _normalize_path_for_file_id(source_rel_path)
    file_nodes = (
        file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
    )
    if not isinstance(file_nodes, list):
        file_nodes = []

    best_node: dict[str, Any] | None = None
    best_score = -1
    for node in file_nodes:
        if not isinstance(node, dict):
            continue
        candidates = [
            _normalize_path_for_file_id(str(node.get("source_rel_path", ""))),
            _normalize_path_for_file_id(str(node.get("archived_rel_path", ""))),
            _normalize_path_for_file_id(str(node.get("archive_rel_path", ""))),
            _normalize_path_for_file_id(str(node.get("name", ""))),
        ]
        candidates = [value for value in candidates if value]
        for candidate in candidates:
            score = 0
            if candidate == normalized_source:
                score = 100 + len(candidate)
            elif normalized_source and (
                candidate.endswith(normalized_source)
                or normalized_source.endswith(candidate)
            ):
                score = min(len(candidate), len(normalized_source))
            if score > best_score:
                best_score = score
                best_node = node

    members: list[str] = []
    field = "f3"
    node_id = ""
    if isinstance(best_node, dict):
        node_id = str(best_node.get("id", "")).strip()
        field = str(best_node.get("dominant_field", "f3")).strip() or "f3"
        dominant_presence = str(best_node.get("dominant_presence", "")).strip()
        if dominant_presence:
            members.append(dominant_presence)
        field_scores = (
            best_node.get("field_scores", {})
            if isinstance(best_node.get("field_scores"), dict)
            else {}
        )
        ranked_fields = sorted(
            [
                (str(key), _safe_float(value, 0.0))
                for key, value in field_scores.items()
                if _safe_float(value, 0.0) > 0.0
            ],
            key=lambda row: row[1],
            reverse=True,
        )
        for field_id, weight in ranked_fields[:3]:
            if weight <= 0.12:
                continue
            member = FIELD_TO_PRESENCE.get(field_id, "")
            if member:
                members.append(member)
        concept_presence_id = str(best_node.get("concept_presence_id", "")).strip()
        if concept_presence_id:
            members.append(concept_presence_id)
    else:
        from .catalog import classify_kind

        inferred_kind = classify_kind(Path(normalized_source or "unknown.txt"))
        scores = _infer_eta_mu_field_scores(
            rel_path=normalized_source,
            kind=inferred_kind,
            text_excerpt="",
        )
        ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)
        if ranked:
            field = str(ranked[0][0])
        for field_id, weight in ranked[:3]:
            if _safe_float(weight, 0.0) <= 0.1:
                continue
            mapped = FIELD_TO_PRESENCE.get(str(field_id), "")
            if mapped:
                members.append(mapped)

    members.append(FILE_SENTINEL_PROFILE["id"])
    members = [str(item).strip() for item in members if str(item).strip()]
    members = list(dict.fromkeys(members))
    boundaries = _pairwise_overlap_boundaries(members)
    return {
        "source_rel_path": normalized_source,
        "node_id": node_id,
        "field": field,
        "members": members,
        "boundary_pairs": boundaries,
    }


def _council_auto_vote(
    member_id: str,
    *,
    event_type: str,
    gate_blocked: bool,
    influence_world: dict[str, float],
    overlap_count: int,
) -> tuple[str, str]:
    min_overlap = _cfg_int("COUNCIL_MIN_OVERLAP_MEMBERS", COUNCIL_MIN_OVERLAP_MEMBERS)
    member = str(member_id).strip()
    if member == "gates_of_truth":
        if gate_blocked:
            return "no", "gate blocked by unresolved drift signals"
        return "yes", "gate is clear"
    if member == FILE_SENTINEL_PROFILE["id"]:
        if event_type in {"file_changed", "file_added", "file_removed"}:
            return "yes", "file sentinel witnessed actionable file delta"
        return "abstain", "no file delta observed"
    if member == THE_COUNCIL_PROFILE["id"]:
        if overlap_count >= min_overlap and not gate_blocked:
            return "yes", "overlap boundary quorum satisfied"
        return "no", "overlap quorum or gate condition failed"
    if member.startswith("presence:concept:"):
        if overlap_count >= min_overlap:
            return "yes", "concept boundary overlaps impacted resource"
        return "abstain", "concept not strongly coupled to current overlap"

    influence = _clamp01(_safe_float(influence_world.get(member, 0.5), 0.5))
    if influence >= COUNCIL_MIN_MEMBER_WORLD_INFLUENCE:
        return "yes", f"world influence {influence:.2f} >= threshold"
    return "abstain", f"world influence {influence:.2f} below threshold"


class CouncilChamber:
    def __init__(
        self,
        decision_log_path: Path,
        receipts_path: Path,
        *,
        owner: str,
        host: str,
        part_root: Path,
        vault_root: Path,
        manifest: str = "manifest.lith",
    ) -> None:
        self._decision_log_path = decision_log_path.resolve()
        self._receipts_path = receipts_path.resolve()
        self._owner = owner
        self._host = host
        self._part_root = part_root.resolve()
        self._vault_root = vault_root.resolve()
        self._manifest = manifest
        self._lock = threading.Lock()
        self._event_count = 0
        self._decisions: dict[str, dict[str, Any]] = {}
        self._last_restart_ts = 0.0
        self._load_from_log()

    def _load_from_log(self) -> None:
        if not self._decision_log_path.exists():
            return
        for raw in self._decision_log_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision = event.get("decision")
            if isinstance(decision, dict):
                decision_id = str(decision.get("id", "")).strip()
                if decision_id:
                    self._decisions[decision_id] = decision
                    action = decision.get("action", {})
                    if isinstance(action, dict) and bool(action.get("ok", False)):
                        self._last_restart_ts = max(
                            self._last_restart_ts,
                            _safe_float(action.get("unix_ts", 0.0), 0.0),
                        )
            self._event_count += 1

    def _append_event(self, *, op: str, decision: dict[str, Any]) -> None:
        self._decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "v": COUNCIL_EVENT_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": op,
            "decision": decision,
        }
        with self._decision_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._event_count += 1

    def _required_yes(self, member_count: int) -> int:
        return max(2, int(math.ceil(max(1, member_count) * 0.6)))

    def _tally_votes(
        self, votes: list[dict[str, Any]], member_count: int
    ) -> dict[str, Any]:
        yes = 0
        no = 0
        abstain = 0
        for row in votes:
            vote = str(row.get("vote", "abstain")).strip().lower()
            if vote == "yes":
                yes += 1
            elif vote == "no":
                no += 1
            else:
                abstain += 1
        required_yes = self._required_yes(member_count)
        approved = yes >= required_yes and no == 0
        return {
            "yes": yes,
            "no": no,
            "abstain": abstain,
            "required_yes": required_yes,
            "approved": approved,
        }

    def _restart_action(self, decision: dict[str, Any]) -> dict[str, Any]:
        del decision
        now = time.time()
        cooldown = _cfg_float(
            "DOCKER_AUTORESTART_COOLDOWN_SECONDS",
            DOCKER_AUTORESTART_COOLDOWN_SECONDS,
        )
        if now - self._last_restart_ts < cooldown:
            return {
                "attempted": False,
                "ok": False,
                "result": "cooldown",
                "reason": "restart cooldown active",
                "unix_ts": now,
            }

        compose_rel = (
            str(
                os.getenv("DOCKER_AUTORESTART_COMPOSE_FILE", "docker-compose.yml")
                or "docker-compose.yml"
            ).strip()
            or "docker-compose.yml"
        )
        compose_path = (self._part_root / compose_rel).resolve()
        services_available = _docker_compose_service_names(compose_path)
        if not services_available:
            return {
                "attempted": False,
                "ok": False,
                "result": "compose-missing",
                "reason": "docker compose file/services not found",
                "compose_file": str(compose_path),
                "unix_ts": now,
            }

        service_targets = _split_csv_items(
            _cfg_str("DOCKER_AUTORESTART_SERVICES", DOCKER_AUTORESTART_SERVICES)
        )
        if not service_targets:
            service_targets = list(services_available)
        unknown = [item for item in service_targets if item not in services_available]
        if unknown:
            return {
                "attempted": False,
                "ok": False,
                "result": "invalid-services",
                "reason": "unknown compose services",
                "unknown": unknown,
                "available": services_available,
                "compose_file": str(compose_path),
                "unix_ts": now,
            }

        command = [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "restart",
            *service_targets,
        ]
        timeout_seconds = _cfg_float(
            "DOCKER_AUTORESTART_TIMEOUT_SECONDS",
            DOCKER_AUTORESTART_TIMEOUT_SECONDS,
        )
        try:
            completed = subprocess.run(
                command,
                check=True,
                timeout=timeout_seconds,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return {
                "attempted": True,
                "ok": False,
                "result": "docker-missing",
                "reason": "docker binary not found",
                "command": command,
                "unix_ts": now,
            }
        except subprocess.TimeoutExpired:
            return {
                "attempted": True,
                "ok": False,
                "result": "timeout",
                "reason": "docker restart timed out",
                "command": command,
                "unix_ts": now,
            }
        except subprocess.CalledProcessError as exc:
            return {
                "attempted": True,
                "ok": False,
                "result": "failed",
                "reason": "docker compose restart failed",
                "command": command,
                "stdout": str(exc.stdout or "")[:1200],
                "stderr": str(exc.stderr or "")[:1200],
                "unix_ts": now,
            }

        self._last_restart_ts = now
        return {
            "attempted": True,
            "ok": True,
            "result": "executed",
            "command": command,
            "stdout": str(completed.stdout or "")[:1200],
            "stderr": str(completed.stderr or "")[:1200],
            "services": service_targets,
            "compose_file": str(compose_path),
            "unix_ts": now,
        }

    def _sorted_decisions(
        self, *, limit: int = COUNCIL_DECISION_HISTORY_LIMIT
    ) -> list[dict[str, Any]]:
        decisions = [
            dict(item) for item in self._decisions.values() if isinstance(item, dict)
        ]
        decisions.sort(
            key=lambda row: _safe_float(row.get("created_unix", 0.0), 0.0),
            reverse=True,
        )
        return decisions[: max(1, limit)]

    def snapshot(
        self, *, include_decisions: bool = False, limit: int = 16
    ) -> dict[str, Any]:
        decisions = self._sorted_decisions(limit=COUNCIL_DECISION_HISTORY_LIMIT)
        pending = [
            row
            for row in decisions
            if str(row.get("status", "")).strip() in {"pending", "awaiting-votes"}
        ]
        approved = [
            row
            for row in decisions
            if str(row.get("status", "")).strip() in {"approved", "executed"}
        ]
        payload = {
            "decision_log": str(self._decision_log_path),
            "event_count": self._event_count,
            "decision_count": len(decisions),
            "pending_count": len(pending),
            "approved_count": len(approved),
            "auto_restart_enabled": _cfg_bool(
                "DOCKER_AUTORESTART_ENABLED", DOCKER_AUTORESTART_ENABLED
            ),
            "require_council": _cfg_bool(
                "DOCKER_AUTORESTART_REQUIRE_COUNCIL", DOCKER_AUTORESTART_REQUIRE_COUNCIL
            ),
            "cooldown_seconds": _cfg_float(
                "DOCKER_AUTORESTART_COOLDOWN_SECONDS",
                DOCKER_AUTORESTART_COOLDOWN_SECONDS,
            ),
        }
        if include_decisions:
            payload["decisions"] = decisions[: max(1, limit)]
        return payload

    def consider_event(
        self,
        *,
        event_type: str,
        data: dict[str, Any],
        catalog: dict[str, Any],
        influence_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        event_kind = str(event_type or "").strip()
        if event_kind not in {"file_changed", "file_added", "file_removed"}:
            return {"ok": False, "status": "ignored", "reason": "event-not-actionable"}
        if not _cfg_bool("DOCKER_AUTORESTART_ENABLED", DOCKER_AUTORESTART_ENABLED):
            return {
                "ok": False,
                "status": "disabled",
                "reason": "docker auto-restart disabled",
            }

        source_rel_path = _event_source_rel_path(
            str(data.get("path", "")), self._vault_root, self._part_root
        )
        includes = _split_csv_items(
            _cfg_str(
                "DOCKER_AUTORESTART_INCLUDE_GLOBS", DOCKER_AUTORESTART_INCLUDE_GLOBS
            )
        )
        excludes = _split_csv_items(
            _cfg_str(
                "DOCKER_AUTORESTART_EXCLUDE_GLOBS", DOCKER_AUTORESTART_EXCLUDE_GLOBS
            )
        )
        if not _path_glob_allowed(source_rel_path, includes, excludes):
            return {
                "ok": False,
                "status": "filtered",
                "source_rel_path": source_rel_path,
                "reason": "path filtered by include/exclude policy",
            }

        file_graph = catalog.get("file_graph", {}) if isinstance(catalog, dict) else {}
        overlap = _file_graph_overlap_context(
            file_graph if isinstance(file_graph, dict) else {}, source_rel_path
        )
        members = [
            str(item) for item in overlap.get("members", []) if str(item).strip()
        ]
        if THE_COUNCIL_PROFILE["id"] not in members:
            members.append(THE_COUNCIL_PROFILE["id"])
        members = list(dict.fromkeys(members))

        min_overlap = _cfg_int(
            "COUNCIL_MIN_OVERLAP_MEMBERS", COUNCIL_MIN_OVERLAP_MEMBERS
        )
        if len(members) < min_overlap:
            return {
                "ok": False,
                "status": "no-council",
                "source_rel_path": source_rel_path,
                "overlap": overlap,
                "reason": "insufficient presence overlap",
            }

        drift_fn = _world_web_symbol(
            "build_drift_scan_payload", build_drift_scan_payload
        )
        drift = drift_fn(self._part_root, self._vault_root)
        blocked_gates = (
            drift.get("blocked_gates", [])
            if isinstance(drift.get("blocked_gates"), list)
            else []
        )
        gate_reasons = [
            str(item.get("reason", "unknown"))
            for item in blocked_gates
            if isinstance(item, dict)
        ]
        gate_blocked = bool(gate_reasons)

        created_at = datetime.now(timezone.utc).isoformat()
        created_unix = time.time()
        decision_seed = (
            f"{event_kind}|{source_rel_path}|{','.join(sorted(members))}|{created_at}"
        )
        decision_id = (
            "decision:council:"
            + hashlib.sha1(decision_seed.encode("utf-8")).hexdigest()[:14]
        )
        council_id = (
            "council:"
            + hashlib.sha1(
                f"{source_rel_path}|{','.join(sorted(members))}".encode("utf-8")
            ).hexdigest()[:12]
        )

        influence_world = _influence_world_map(influence_snapshot)
        auto_vote_fn = _world_web_symbol("_council_auto_vote", _council_auto_vote)
        votes: list[dict[str, Any]] = []
        for member_id in members:
            vote, reason = auto_vote_fn(
                member_id,
                event_type=event_kind,
                gate_blocked=gate_blocked,
                influence_world=influence_world,
                overlap_count=len(members),
            )
            votes.append(
                {
                    "member_id": member_id,
                    "vote": vote,
                    "reason": reason,
                    "mode": "auto",
                    "ts": created_at,
                }
            )

        tally = self._tally_votes(votes, len(members))
        approved = bool(tally.get("approved", False)) and not gate_blocked
        status = "approved" if approved else "awaiting-votes"
        if gate_blocked:
            status = "blocked"

        decision = {
            "id": decision_id,
            "kind": "docker.restart.on-change",
            "status": status,
            "created_at": created_at,
            "created_unix": created_unix,
            "source_event": {"type": event_kind, "data": dict(data)},
            "resource": {
                "source_rel_path": source_rel_path,
                "field": str(overlap.get("field", "f3")),
                "node_id": str(overlap.get("node_id", "")),
            },
            "space": {
                "field": str(overlap.get("field", "f3")),
                "members": members,
                "boundary_pairs": overlap.get("boundary_pairs", []),
                "overlap_count": len(members),
            },
            "council": {
                "id": council_id,
                "members": members,
                "required_yes": int(tally.get("required_yes", 2)),
                "votes": votes,
                "tally": tally,
            },
            "gate": {"blocked": gate_blocked, "reasons": gate_reasons},
            "action": {
                "attempted": False,
                "ok": False,
                "result": "pending-approval",
                "unix_ts": created_unix,
            },
        }

        require_council = _cfg_bool(
            "DOCKER_AUTORESTART_REQUIRE_COUNCIL",
            DOCKER_AUTORESTART_REQUIRE_COUNCIL,
        )
        if approved and (not require_council or approved):
            decision["action"] = self._restart_action(decision)
            action = decision.get("action", {})
            if isinstance(action, dict):
                if bool(action.get("ok", False)):
                    decision["status"] = "executed"
                elif str(action.get("result", "")) in {
                    "cooldown",
                    "compose-missing",
                    "invalid-services",
                }:
                    decision["status"] = "blocked"
                else:
                    decision["status"] = "error"

        with self._lock:
            self._decisions[decision_id] = decision
            self._append_event(op="decision", decision=decision)

        _append_receipt_line(
            self._receipts_path,
            kind=":decision",
            origin="council",
            owner=self._owner,
            dod="Council evaluated docker restart decision and applied boundary vote policy",
            pi="part64-runtime-system",
            host=self._host,
            manifest=self._manifest,
            refs=[
                f"decision:{decision_id}",
                f"council:{council_id}",
                source_rel_path,
                "council:docker-autorestart",
                ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
            ],
            note=f"status={decision.get('status', '')}; members={len(members)}; gate_blocked={gate_blocked}",
        )
        return {
            "ok": True,
            "decision": decision,
            "council": self.snapshot(include_decisions=False),
        }

    def vote(
        self,
        *,
        decision_id: str,
        member_id: str,
        vote: str,
        reason: str,
        actor: str,
    ) -> dict[str, Any]:
        with self._lock:
            decision = self._decisions.get(decision_id)
            if not isinstance(decision, dict):
                return {"ok": False, "error": "decision_not_found"}

            council = decision.get("council", {})
            members = [
                str(item).strip()
                for item in council.get("members", [])
                if str(item).strip()
            ]
            member = str(member_id).strip()
            if member not in members:
                return {
                    "ok": False,
                    "error": "member_not_in_council",
                    "members": members,
                }

            vote_value = str(vote).strip().lower()
            if vote_value not in {"yes", "no", "abstain"}:
                return {
                    "ok": False,
                    "error": "invalid_vote",
                    "allowed": ["yes", "no", "abstain"],
                }

            votes = council.get("votes", [])
            if not isinstance(votes, list):
                votes = []
            ts = datetime.now(timezone.utc).isoformat()
            replaced = False
            for row in votes:
                if str(row.get("member_id", "")) == member:
                    row["vote"] = vote_value
                    row["reason"] = str(reason).strip()
                    row["mode"] = "manual"
                    row["actor"] = actor
                    row["ts"] = ts
                    replaced = True
                    break
            if not replaced:
                votes.append(
                    {
                        "member_id": member,
                        "vote": vote_value,
                        "reason": str(reason).strip(),
                        "mode": "manual",
                        "actor": actor,
                        "ts": ts,
                    }
                )

            tally = self._tally_votes(votes, len(members))
            gate = (
                decision.get("gate", {})
                if isinstance(decision.get("gate"), dict)
                else {}
            )
            gate_blocked = bool(gate.get("blocked", False))
            approved = bool(tally.get("approved", False)) and not gate_blocked
            decision_status = "approved" if approved else "awaiting-votes"
            if gate_blocked:
                decision_status = "blocked"

            council["votes"] = votes
            council["tally"] = tally
            decision["council"] = council
            decision["status"] = decision_status

            if approved and not bool(
                (decision.get("action") or {}).get("attempted", False)
            ):
                decision["action"] = self._restart_action(decision)
                action = (
                    decision.get("action", {})
                    if isinstance(decision.get("action"), dict)
                    else {}
                )
                if bool(action.get("ok", False)):
                    decision["status"] = "executed"
                elif str(action.get("result", "")) in {
                    "cooldown",
                    "compose-missing",
                    "invalid-services",
                }:
                    decision["status"] = "blocked"
                else:
                    decision["status"] = "error"

            self._decisions[decision_id] = decision
            self._append_event(op="vote", decision=decision)

        return {
            "ok": True,
            "decision": decision,
            "council": self.snapshot(include_decisions=False),
        }


class TaskQueue:
    def __init__(
        self,
        queue_log_path: Path,
        receipts_path: Path,
        *,
        owner: str,
        host: str,
        manifest: str = "manifest.lith",
    ) -> None:
        self._queue_log_path = queue_log_path.resolve()
        self._receipts_path = receipts_path.resolve()
        self._owner = owner
        self._host = host
        self._manifest = manifest
        self._lock = threading.Lock()
        self._pending: list[dict[str, Any]] = []
        self._dedupe_index: dict[str, str] = {}
        self._event_count = 0
        self._load_from_log()

    def _load_from_log(self) -> None:
        if not self._queue_log_path.exists():
            return
        for raw in self._queue_log_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            op = str(event.get("op", "")).strip()
            if op == "enqueue":
                task = event.get("task")
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("id", "")).strip()
                if not task_id or any(
                    str(item.get("id", "")) == task_id for item in self._pending
                ):
                    continue
                self._pending.append(task)
                dedupe_key = str(task.get("dedupe_key", "")).strip()
                if dedupe_key:
                    self._dedupe_index[dedupe_key] = task_id
            elif op == "dequeue":
                task_id = str(event.get("task_id", "")).strip()
                self._remove_pending_task(task_id)
            self._event_count += 1

    def _remove_pending_task(self, task_id: str) -> dict[str, Any] | None:
        if not task_id:
            return None
        for idx, task in enumerate(self._pending):
            if str(task.get("id", "")).strip() != task_id:
                continue
            removed = self._pending.pop(idx)
            dedupe_key = str(removed.get("dedupe_key", "")).strip()
            if dedupe_key:
                self._dedupe_index.pop(dedupe_key, None)
            return removed
        return None

    def _append_event(self, event: dict[str, Any]) -> None:
        self._queue_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._queue_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._event_count += 1

    def enqueue(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        dedupe_key: str,
        owner: str | None = None,
        dod: str = "task queued with persisted log and receipt",
        refs: list[str] | None = None,
    ) -> dict[str, Any]:
        refs = refs or []
        normalized_dedupe = (
            dedupe_key.strip()
            or f"{kind}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
        )
        owner_value = (owner or self._owner).strip() or self._owner
        with self._lock:
            existing_id = self._dedupe_index.get(normalized_dedupe)
            if existing_id:
                existing_task = next(
                    (
                        item
                        for item in self._pending
                        if str(item.get("id", "")) == existing_id
                    ),
                    None,
                )
                if existing_task is not None:
                    return {
                        "ok": True,
                        "deduped": True,
                        "task": existing_task,
                        "queue": self.snapshot(include_pending=False),
                    }

            created_at = datetime.now(timezone.utc).isoformat()
            task_id_seed = f"{normalized_dedupe}|{created_at}|{time.time_ns()}"
            task_id = (
                f"task-{hashlib.sha1(task_id_seed.encode('utf-8')).hexdigest()[:12]}"
            )
            task = {
                "id": task_id,
                "kind": kind,
                "payload": payload,
                "dedupe_key": normalized_dedupe,
                "owner": owner_value,
                "status": "pending",
                "created_at": created_at,
            }
            event = {
                "v": TASK_QUEUE_EVENT_VERSION,
                "ts": created_at,
                "op": "enqueue",
                "task": task,
            }
            self._pending.append(task)
            self._dedupe_index[normalized_dedupe] = task_id
            self._append_event(event)
            _append_receipt_line(
                self._receipts_path,
                kind=":decision",
                origin="task-queue",
                owner=owner_value,
                dod=dod,
                pi="part64-runtime-system",
                host=self._host,
                manifest=self._manifest,
                refs=[
                    "task-queue:enqueue",
                    f"task:{task_id}",
                    ".opencode/promptdb/diagrams/part64_runtime_system.packet.lisp",
                    ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
                    *refs,
                ],
            )
            return {
                "ok": True,
                "deduped": False,
                "task": task,
                "queue": self.snapshot(include_pending=False),
            }

    def dequeue(
        self,
        *,
        owner: str | None = None,
        dod: str = "task dequeued with persisted log and receipt",
        refs: list[str] | None = None,
    ) -> dict[str, Any]:
        refs = refs or []
        owner_value = (owner or self._owner).strip() or self._owner
        with self._lock:
            if not self._pending:
                return {
                    "ok": False,
                    "error": "empty_queue",
                    "queue": self.snapshot(include_pending=False),
                }

            task = self._pending.pop(0)
            dedupe_key = str(task.get("dedupe_key", "")).strip()
            if dedupe_key:
                self._dedupe_index.pop(dedupe_key, None)

            ts = datetime.now(timezone.utc).isoformat()
            event = {
                "v": TASK_QUEUE_EVENT_VERSION,
                "ts": ts,
                "op": "dequeue",
                "task_id": task.get("id"),
            }
            self._append_event(event)
            _append_receipt_line(
                self._receipts_path,
                kind=":decision",
                origin="task-queue",
                owner=owner_value,
                dod=dod,
                pi="part64-runtime-system",
                host=self._host,
                manifest=self._manifest,
                refs=[
                    "task-queue:dequeue",
                    f"task:{task.get('id', '')}",
                    ".opencode/promptdb/diagrams/part64_runtime_system.packet.lisp",
                    ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
                    *refs,
                ],
            )
            return {
                "ok": True,
                "task": task,
                "queue": self.snapshot(include_pending=False),
            }

    def snapshot(self, *, include_pending: bool = False) -> dict[str, Any]:
        data = {
            "queue_log": str(self._queue_log_path),
            "pending_count": len(self._pending),
            "dedupe_keys": len(self._dedupe_index),
            "event_count": self._event_count,
        }
        if include_pending:
            data["pending"] = [dict(item) for item in self._pending]
        return data


def _extract_lisp_vector_strings(source: str, key: str) -> list[str]:
    match = re.search(rf"\({re.escape(key)}\s+\[([^\]]*)\]\)", source, re.DOTALL)
    if not match:
        return []
    return [item.strip() for item in re.findall(r'"([^\"]+)"', match.group(1))]


def _collect_open_questions(vault_root: Path) -> list[dict[str, str]]:
    promptdb_root = locate_promptdb_root(vault_root)
    if promptdb_root is None:
        return []
    packet_path = promptdb_root / PROMPTDB_OPEN_QUESTIONS_PACKET
    if not packet_path.exists() or not packet_path.is_file():
        return []

    source = packet_path.read_text("utf-8")
    matches = re.findall(
        r"\(q\s+\(id\s+([^)\s]+)\)\s+\(text\s+\"([^\"]+)\"\)\)", source
    )
    return [{"id": qid.strip(), "text": qtext.strip()} for qid, qtext in matches]


def _partition_open_questions(
    open_questions: list[dict[str, str]],
    receipt_refs: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    resolved: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    refs_text = "\n".join(receipt_refs)
    for item in open_questions:
        question_id = str(item.get("id", "")).strip()
        if question_id and question_id in refs_text:
            resolved.append(item)
        else:
            unresolved.append(item)
    return resolved, unresolved


def _build_keeper_of_contracts_signal(
    unresolved_questions: list[dict[str, str]],
    blocked_gates: list[dict[str, Any]],
    promptdb_packet_count: int,
) -> dict[str, Any] | None:
    if not unresolved_questions or not blocked_gates:
        return None

    top_questions = unresolved_questions[:3]
    text = " ".join(f"{q['id']}: {q['text']}" for q in top_questions)
    payload = build_presence_say_payload(
        {
            "items": [],
            "promptdb": {"packet_count": promptdb_packet_count},
        },
        text=f"gate unresolved {text}",
        requested_presence_id="keeper_of_contracts",
    )

    asks = [f"Resolve {item['id']}" for item in top_questions]
    payload["say_intent"]["asks"] = asks
    payload["say_intent"]["facts"].append(f"blocked_gates={len(blocked_gates)}")
    payload["say_intent"]["facts"].append(
        f"unresolved_questions={len(unresolved_questions)}"
    )
    payload["say_intent"]["repairs"].append(
        "Append receipt refs for resolved question ids (q.*) and gate decisions."
    )
    return payload


def _extract_manifest_proof_schema(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.exists():
        return {
            "source": "",
            "required_refs": [],
            "required_hashes": [],
            "host_handle": "",
        }

    source = manifest_path.read_text("utf-8")
    return {
        "source": str(manifest_path),
        "required_refs": _extract_lisp_vector_strings(source, "required-refs"),
        "required_hashes": _extract_lisp_vector_strings(source, "required-hashes"),
        "host_handle": _extract_lisp_string(source, "host-handle") or "",
    }


def _proof_ref_exists(ref: str, vault_root: Path, part_root: Path) -> bool:
    candidate_ref = ref.strip()
    if not candidate_ref:
        return False
    if "://" in candidate_ref:
        return True
    if candidate_ref.startswith("runtime:"):
        return True
    if candidate_ref.startswith("artifact:"):
        return True

    path_candidate = Path(candidate_ref)
    if path_candidate.is_absolute():
        return path_candidate.exists()

    for base in (vault_root.resolve(), part_root.resolve(), Path.cwd().resolve()):
        resolved = (base / candidate_ref).resolve()
        if resolved.exists():
            return True
    return False


def _sha256_for_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pi_zip_name_hash_token(path: Path) -> str:
    stem = path.stem.strip()
    for prefix in ("Π.", "Pi.", "pi."):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem.strip().lower()


def _pi_zip_name_check(path: Path, sha256_hex: str) -> dict[str, Any]:
    expected_sha12 = sha256_hex[:12].lower()
    name_sha12 = _pi_zip_name_hash_token(path)
    return {
        "path": str(path),
        "name": path.name,
        "expected_sha12": expected_sha12,
        "name_sha12": name_sha12,
        "matches_sha12": name_sha12 == expected_sha12,
    }


def build_drift_scan_payload(part_root: Path, vault_root: Path) -> dict[str, Any]:
    required_keys = [
        "ts",
        "kind",
        "origin",
        "owner",
        "dod",
        "pi",
        "host",
        "manifest",
        "refs",
    ]
    receipts_path = _locate_receipts_log(vault_root, part_root)
    active_drifts: list[dict[str, Any]] = []
    blocked_gates: list[dict[str, Any]] = []
    parse_ok = True
    row_count = 0
    has_intent_ref = False
    parsed_refs: list[str] = []

    promptdb_index = collect_promptdb_packets(vault_root)
    promptdb_packet_count = int(promptdb_index.get("packet_count", 0))
    open_questions = _collect_open_questions(vault_root)

    if receipts_path is None:
        parse_ok = False
        active_drifts.append(
            {
                "id": "missing_receipts_log",
                "severity": "high",
                "detail": "receipts.log not found",
            }
        )
    else:
        for raw_line in receipts_path.read_text("utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            row_count += 1
            row = _parse_receipt_line(line)
            if not all(key in row for key in required_keys):
                parse_ok = False
                active_drifts.append(
                    {
                        "id": "receipt_line_missing_keys",
                        "severity": "medium",
                        "detail": f"line {row_count} missing required keys",
                    }
                )
                continue
            refs = _split_receipt_refs(row.get("refs", ""))
            parsed_refs.extend(refs)
            if any("00_wire_world.intent.lisp" in ref for ref in refs):
                has_intent_ref = True

    if not parse_ok:
        blocked_gates.append(
            {"target": "push-truth", "reason": "receipts-parse-failed"}
        )
    if not has_intent_ref:
        active_drifts.append(
            {
                "id": "missing_intent_receipt_ref",
                "severity": "high",
                "detail": "push-truth receipt ref to 00_wire_world.intent.lisp not found",
            }
        )
        blocked_gates.append({"target": "push-truth", "reason": "missing-intent-ref"})

    resolved_questions, unresolved_questions = _partition_open_questions(
        open_questions, parsed_refs
    )
    if unresolved_questions:
        active_drifts.append(
            {
                "id": "open_questions_unresolved",
                "severity": "medium",
                "detail": f"{len(unresolved_questions)} open gate questions unresolved",
                "question_ids": [item["id"] for item in unresolved_questions],
            }
        )
        blocked_gates.append(
            {
                "target": "push-truth",
                "reason": "open-questions-unresolved",
                "question_ids": [item["id"] for item in unresolved_questions],
            }
        )

    keeper_signal = _build_keeper_of_contracts_signal(
        unresolved_questions=unresolved_questions,
        blocked_gates=blocked_gates,
        promptdb_packet_count=promptdb_packet_count,
    )

    receipts_parse = {
        "path": str(receipts_path) if receipts_path else "",
        "ok": parse_ok,
        "rows": row_count,
        "has_intent_ref": has_intent_ref,
    }

    return {
        "ok": True,
        "receipts": {
            "path": receipts_parse["path"],
            "parse_ok": receipts_parse["ok"],
            "rows": receipts_parse["rows"],
            "has_intent_ref": receipts_parse["has_intent_ref"],
        },
        "receipts_parse": receipts_parse,
        "drifts": active_drifts,
        "active_drifts": active_drifts,
        "blocked_gates": blocked_gates,
        "open_questions": {
            "total": len(open_questions),
            "resolved_count": len(resolved_questions),
            "unresolved_count": len(unresolved_questions),
            "resolved": resolved_questions,
            "unresolved": unresolved_questions,
        },
        "keeper_of_contracts": keeper_signal,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _run_git_command(
    cwd: Path,
    args: list[str],
    *,
    timeout_s: float = 2.5,
) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.5, float(timeout_s)),
        )
    except (OSError, subprocess.SubprocessError):
        return False, ""

    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "").strip()
    return True, (proc.stdout or "").strip()


def build_witness_lineage_payload(part_root: Path) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()

    repo_root = ""
    branch = ""
    upstream = ""
    remote = ""
    remote_url = ""
    ahead = 0
    behind = 0
    push_obligation_unknown = False
    latest_commit = ""
    staged = 0
    unstaged = 0
    untracked = 0

    ok_repo, inside = _run_git_command(
        part_root, ["rev-parse", "--is-inside-work-tree"]
    )
    if not ok_repo or inside.lower() != "true":
        return {
            "ok": True,
            "record": "ημ.witness-lineage.v1",
            "generated_at": generated_at,
            "repo": {
                "available": False,
                "root": "",
                "branch": "",
                "upstream": "",
                "remote": "",
                "remote_url": "",
            },
            "checkpoint": {
                "branch": "",
                "upstream": "",
                "ahead": 0,
                "behind": 0,
            },
            "working_tree": {
                "dirty": False,
                "staged": 0,
                "unstaged": 0,
                "untracked": 0,
            },
            "latest_commit": "",
            "push_obligation": False,
            "push_obligation_unknown": True,
            "continuity_drift": {
                "active": True,
                "code": "git_repo_unavailable",
                "message": "runtime path is not inside a git repository",
            },
        }

    ok_root, root_text = _run_git_command(part_root, ["rev-parse", "--show-toplevel"])
    repo_root = root_text if ok_root else str(part_root.resolve())

    ok_branch, branch_text = _run_git_command(
        part_root, ["rev-parse", "--abbrev-ref", "HEAD"]
    )
    branch = branch_text if ok_branch else ""

    ok_upstream, upstream_text = _run_git_command(
        part_root,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
    )
    if ok_upstream:
        upstream = upstream_text
        remote = upstream.split("/", 1)[0] if "/" in upstream else ""

    if upstream:
        ok_ahead_behind, counts = _run_git_command(
            part_root,
            ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
        )
        if ok_ahead_behind:
            pieces = [piece for piece in counts.replace("\t", " ").split(" ") if piece]
            if len(pieces) >= 2:
                ahead = max(0, int(_safe_float(pieces[0], 0.0)))
                behind = max(0, int(_safe_float(pieces[1], 0.0)))
            else:
                push_obligation_unknown = True
        else:
            push_obligation_unknown = True
    else:
        push_obligation_unknown = True

    ok_status, status_output = _run_git_command(part_root, ["status", "--porcelain"])
    if ok_status:
        for raw in status_output.splitlines():
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("??"):
                untracked += 1
                continue
            staged_flag = line[0:1]
            unstaged_flag = line[1:2]
            if staged_flag and staged_flag not in {" ", "?"}:
                staged += 1
            if unstaged_flag and unstaged_flag not in {" ", "?"}:
                unstaged += 1

    ok_commit, commit_text = _run_git_command(
        part_root,
        ["log", "-1", "--pretty=%h %s"],
    )
    if ok_commit:
        latest_commit = commit_text

    if remote:
        ok_remote_url, remote_url_text = _run_git_command(
            part_root,
            ["remote", "get-url", remote],
        )
        if ok_remote_url:
            remote_url = remote_url_text

    continuity_drift = {
        "active": False,
        "code": "ok",
        "message": "witness continuity aligned with upstream lineage",
    }
    if not upstream:
        continuity_drift = {
            "active": True,
            "code": "missing_upstream",
            "message": "upstream tracking branch missing; push obligations cannot be verified",
        }
    elif behind > 0:
        continuity_drift = {
            "active": True,
            "code": "behind_upstream",
            "message": f"branch is behind upstream by {behind} commits",
        }

    dirty = (staged + unstaged + untracked) > 0
    push_obligation = ahead > 0 and bool(upstream)

    return {
        "ok": True,
        "record": "ημ.witness-lineage.v1",
        "generated_at": generated_at,
        "repo": {
            "available": True,
            "root": repo_root,
            "branch": branch,
            "upstream": upstream,
            "remote": remote,
            "remote_url": remote_url,
        },
        "checkpoint": {
            "branch": branch,
            "upstream": upstream,
            "ahead": ahead,
            "behind": behind,
        },
        "working_tree": {
            "dirty": dirty,
            "staged": staged,
            "unstaged": unstaged,
            "untracked": untracked,
        },
        "latest_commit": latest_commit,
        "push_obligation": push_obligation,
        "push_obligation_unknown": push_obligation_unknown,
        "continuity_drift": continuity_drift,
    }


def _world_log_timestamp_value(raw: str) -> float:
    text = str(raw or "").strip()
    if not text:
        return 0.0
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0.0


def _world_log_event_id(
    source: str,
    kind: str,
    ts: str,
    title: str,
    detail: str,
) -> str:
    seed = "|".join([source, kind, ts, title, detail])
    return "evt_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:18]


def _world_log_event_text(event: dict[str, Any]) -> str:
    refs = [str(item) for item in event.get("refs", []) if str(item).strip()]
    tags = [str(item) for item in event.get("tags", []) if str(item).strip()]
    bits = [
        str(event.get("title", "")).strip(),
        str(event.get("detail", "")).strip(),
        f"source={str(event.get('source', '')).strip()}",
        f"kind={str(event.get('kind', '')).strip()}",
    ]
    if refs:
        bits.append("refs=" + ", ".join(refs[:4]))
    if tags:
        bits.append("tags=" + ", ".join(tags[:4]))
    return " | ".join(bit for bit in bits if bit)


def _world_log_vector_dims(vault_root: Path) -> int:
    _ = vault_root
    return max(8, int(ETA_MU_INGEST_TEXT_DIMS))


def _compact_world_text(text: str, limit: int) -> str:
    value = str(text or "").strip().replace("\n", " ")
    if len(value) <= limit:
        return value
    safe_limit = max(4, int(limit))
    return value[: safe_limit - 3].rstrip() + "..."


def _world_log_kind_slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower())
    slug = slug.strip("-")
    return slug or fallback


def _world_event_from_stream_row(
    row: dict[str, Any],
    log_path: str,
    *,
    default_source: str,
    default_kind: str,
) -> dict[str, Any] | None:
    event_id = str(row.get("id", "")).strip()
    if not event_id:
        return None
    refs = [str(item) for item in row.get("refs", []) if str(item).strip()]
    tags = [str(item) for item in row.get("tags", []) if str(item).strip()]
    return {
        "id": event_id,
        "ts": str(row.get("ts", "")).strip(),
        "source": str(row.get("source", default_source)).strip() or default_source,
        "kind": str(row.get("kind", default_kind)).strip() or default_kind,
        "status": str(row.get("status", "recorded")).strip() or "recorded",
        "title": str(row.get("title", default_source)).strip() or default_source,
        "detail": str(row.get("detail", "")).strip(),
        "refs": refs,
        "tags": tags,
        "path": log_path,
    }


def _world_stream_error_event(
    *,
    source: str,
    kind: str,
    detail: str,
    path: str,
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()
    event_id = _world_log_event_id(source, kind, ts, kind, detail)
    refs = [path] if path else []
    return {
        "id": event_id,
        "ts": ts,
        "source": source,
        "kind": kind,
        "status": "error",
        "title": kind,
        "detail": _compact_world_text(detail, 240),
        "refs": refs,
        "tags": ["stream", "error"],
        "path": path,
    }


def _nws_alert_id(raw: dict[str, Any], payload_text: str) -> str:
    properties = (
        raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
    )
    if not isinstance(properties, dict):
        properties = {}
    base_id = str(raw.get("id", "")).strip()
    if not base_id:
        for key in ("@id", "id", "uri"):
            value = str(properties.get(key, "")).strip()
            if value:
                base_id = value
                break
    seed = "id:" + base_id if base_id else "raw:" + payload_text
    return "nws:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]


def _nws_alert_row(raw: dict[str, Any], *, received_at: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    detail_limit = _cfg_int(
        "NWS_ALERTS_DETAIL_CHAR_LIMIT", NWS_ALERTS_DETAIL_CHAR_LIMIT
    )
    properties = (
        raw.get("properties") if isinstance(raw.get("properties"), dict) else {}
    )
    if not isinstance(properties, dict):
        properties = {}

    event_name = str(properties.get("event", "alert")).strip() or "alert"
    event_slug = _world_log_kind_slug(event_name, "alert")
    headline = str(properties.get("headline", "")).strip()
    area_desc = _compact_world_text(str(properties.get("areaDesc", "")).strip(), 160)
    severity = str(properties.get("severity", "")).strip().lower()
    urgency = str(properties.get("urgency", "")).strip().lower()
    certainty = str(properties.get("certainty", "")).strip().lower()
    message_type = str(properties.get("messageType", "")).strip().lower()
    status = str(properties.get("status", "actual")).strip().lower() or "actual"
    sent_ts = str(properties.get("sent", "")).strip()
    effective_ts = str(properties.get("effective", "")).strip()
    expires_ts = str(properties.get("expires", "")).strip()
    ts = sent_ts or effective_ts or received_at

    description = _compact_world_text(
        str(properties.get("description", "")).strip(), detail_limit
    )
    detail_parts: list[str] = []
    if area_desc:
        detail_parts.append("area=" + area_desc)
    if severity:
        detail_parts.append("severity=" + severity)
    if urgency:
        detail_parts.append("urgency=" + urgency)
    if certainty:
        detail_parts.append("certainty=" + certainty)
    if expires_ts:
        detail_parts.append("expires=" + expires_ts)
    if description:
        detail_parts.append(description)
    detail = _compact_world_text(" | ".join(detail_parts), detail_limit)

    refs: list[str] = []
    for candidate in (
        str(raw.get("id", "")).strip(),
        str(properties.get("@id", "")).strip(),
        str(properties.get("uri", "")).strip(),
        str(properties.get("web", "")).strip(),
    ):
        if candidate and candidate not in refs:
            refs.append(candidate)

    payload_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    alert_id = _nws_alert_id(raw, payload_text)
    tags = [
        token
        for token in [
            "nws",
            event_slug,
            severity,
            urgency,
            certainty,
            message_type,
            status,
            str(properties.get("senderName", "")).strip().lower(),
        ]
        if token
    ]

    title = headline
    if not title:
        title = event_name
        if area_desc:
            title += " - " + area_desc

    return {
        "record": NWS_ALERT_RECORD,
        "id": alert_id,
        "ts": ts,
        "source": "nws_alerts",
        "kind": "nws.alert." + event_slug,
        "status": status,
        "title": title,
        "detail": detail,
        "refs": refs,
        "tags": tags,
        "meta": {
            "event": event_name,
            "message_type": message_type,
            "sent": sent_ts,
            "effective": effective_ts,
            "expires": expires_ts,
        },
    }


def _nws_stream_event_row(
    kind: str,
    detail: str,
    *,
    status: str = "recorded",
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    event_id = _world_log_event_id("nws_alerts", kind, now_iso, kind, detail)
    return {
        "record": NWS_ALERT_STREAM_EVENT_RECORD,
        "id": event_id,
        "ts": now_iso,
        "source": "nws_alerts",
        "kind": kind,
        "status": status,
        "title": kind,
        "detail": _compact_world_text(detail, 240),
        "refs": [],
        "tags": ["nws", "alerts"],
        "meta": {
            "endpoint": _cfg_str("NWS_ALERTS_ENDPOINT", NWS_ALERTS_ENDPOINT),
        },
    }


def _append_nws_alert_rows(vault_root: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    log_path = _nws_alerts_log_path(vault_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return log_path


def _load_nws_alert_rows(
    vault_root: Path,
    *,
    limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    log_path = _nws_alerts_log_path(vault_root)
    if not log_path.exists() or not log_path.is_file():
        return str(log_path), []
    bounded = max(1, int(limit or 1))
    selected: list[dict[str, Any]] = []
    for raw_line in log_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            selected.append(row)
    if len(selected) > bounded:
        selected = selected[-bounded:]
    return str(log_path), selected


def _nws_fetch_active_alerts(
    endpoint: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_alerts: int,
) -> dict[str, Any]:
    req = Request(
        endpoint,
        headers={
            "Accept": "application/geo+json,application/ld+json,application/json",
            "Cache-Control": "no-cache",
            "User-Agent": "eta-mu-world-web/1.0",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=max(0.1, timeout_seconds)) as response:
            payload_bytes = response.read(max(1, max_bytes))
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}:{exc}",
            "alerts": [],
            "parse_errors": 0,
            "bytes_read": 0,
        }

    bytes_read = len(payload_bytes)
    parse_errors = 0
    try:
        payload = json.loads(payload_bytes.decode("utf-8", errors="replace"))
    except (ValueError, json.JSONDecodeError):
        return {
            "ok": False,
            "error": "JSONDecodeError:invalid_payload",
            "alerts": [],
            "parse_errors": 1,
            "bytes_read": bytes_read,
        }

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "payload_not_object",
            "alerts": [],
            "parse_errors": 1,
            "bytes_read": bytes_read,
        }

    features = payload.get("features", [])
    if not isinstance(features, list):
        features = []
        parse_errors = 1

    alerts = [item for item in features if isinstance(item, dict)]
    if len(alerts) > max_alerts:
        alerts = alerts[:max_alerts]
    return {
        "ok": True,
        "error": "",
        "alerts": alerts,
        "parse_errors": parse_errors,
        "bytes_read": bytes_read,
    }


def _collect_nws_alert_rows(vault_root: Path) -> None:
    enabled = _cfg_bool("NWS_ALERTS_ENABLED", NWS_ALERTS_ENABLED)
    poll_interval = _cfg_float(
        "NWS_ALERTS_POLL_INTERVAL_SECONDS",
        NWS_ALERTS_POLL_INTERVAL_SECONDS,
    )
    now_monotonic = time.monotonic()

    with _NWS_ALERTS_LOCK:
        was_paused = bool(_NWS_ALERTS_CACHE.get("paused", False))
        if not enabled:
            _NWS_ALERTS_CACHE["paused"] = True
            if not was_paused:
                _append_nws_alert_rows(
                    vault_root,
                    [
                        _nws_stream_event_row(
                            "nws.alerts.paused",
                            "NWS polling disabled by configuration",
                            status="paused",
                        )
                    ],
                )
            return

        if was_paused:
            _append_nws_alert_rows(
                vault_root,
                [
                    _nws_stream_event_row(
                        "nws.alerts.resumed",
                        "NWS polling resumed by configuration",
                    )
                ],
            )
            _NWS_ALERTS_CACHE["paused"] = False

        last_poll = _safe_float(_NWS_ALERTS_CACHE.get("last_poll_monotonic", 0.0), 0.0)
        if now_monotonic - last_poll < max(1.0, poll_interval):
            return
        _NWS_ALERTS_CACHE["last_poll_monotonic"] = now_monotonic

    endpoint = _cfg_str("NWS_ALERTS_ENDPOINT", NWS_ALERTS_ENDPOINT).strip()
    if not endpoint:
        _append_nws_alert_rows(
            vault_root,
            [
                _nws_stream_event_row(
                    "nws.alerts.error",
                    "NWS endpoint is empty",
                    status="error",
                )
            ],
        )
        return

    fetch_payload = _nws_fetch_active_alerts(
        endpoint,
        timeout_seconds=_cfg_float(
            "NWS_ALERTS_FETCH_TIMEOUT_SECONDS",
            NWS_ALERTS_FETCH_TIMEOUT_SECONDS,
        ),
        max_bytes=_cfg_int("NWS_ALERTS_MAX_BYTES", NWS_ALERTS_MAX_BYTES),
        max_alerts=_cfg_int(
            "NWS_ALERTS_MAX_ALERTS_PER_POLL",
            NWS_ALERTS_MAX_ALERTS_PER_POLL,
        ),
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    rows_to_append: list[dict[str, Any]] = []
    with _NWS_ALERTS_LOCK:
        if fetch_payload.get("ok") is not True:
            rows_to_append.append(
                _nws_stream_event_row(
                    "nws.alerts.error",
                    str(fetch_payload.get("error", "fetch_failed")),
                    status="error",
                )
            )
            _NWS_ALERTS_CACHE["connected"] = False
            _append_nws_alert_rows(vault_root, rows_to_append)
            return

        if not bool(_NWS_ALERTS_CACHE.get("connected", False)):
            rows_to_append.append(
                _nws_stream_event_row(
                    "nws.alerts.connected",
                    "NWS active alerts fetch succeeded",
                )
            )
        _NWS_ALERTS_CACHE["connected"] = True

        seen_ids = _NWS_ALERTS_CACHE.get("seen_ids")
        if not isinstance(seen_ids, dict):
            seen_ids = {}
            _NWS_ALERTS_CACHE["seen_ids"] = seen_ids

        dedupe_ttl = _cfg_float(
            "NWS_ALERTS_DEDUPE_TTL_SECONDS",
            NWS_ALERTS_DEDUPE_TTL_SECONDS,
        )
        stale_keys = [
            key
            for key, seen_at in seen_ids.items()
            if now_monotonic - _safe_float(seen_at, 0.0) > max(15.0, dedupe_ttl)
        ]
        for key in stale_keys:
            seen_ids.pop(key, None)

        dedupe_dropped = 0
        rate_dropped = 0
        accepted = 0
        allowed = _cfg_int(
            "NWS_ALERTS_RATE_LIMIT_PER_POLL",
            NWS_ALERTS_RATE_LIMIT_PER_POLL,
        )

        for raw_alert in fetch_payload.get("alerts", []):
            event_row = _nws_alert_row(raw_alert, received_at=now_iso)
            if event_row is None:
                continue
            dedupe_key = str(event_row.get("id", "")).strip()
            if dedupe_key in seen_ids:
                dedupe_dropped += 1
                continue
            seen_ids[dedupe_key] = now_monotonic
            if accepted >= max(1, allowed):
                rate_dropped += 1
                continue
            rows_to_append.append(event_row)
            accepted += 1

        parse_errors = int(_safe_float(fetch_payload.get("parse_errors", 0), 0.0))
        status = "recorded"
        if parse_errors > 0 or dedupe_dropped > 0 or rate_dropped > 0:
            status = "degraded"
        rows_to_append.append(
            _nws_stream_event_row(
                "nws.alerts.poll",
                "accepted="
                + str(accepted)
                + " parse_errors="
                + str(parse_errors)
                + " dedupe_dropped="
                + str(dedupe_dropped)
                + " rate_dropped="
                + str(rate_dropped)
                + " bytes="
                + str(int(_safe_float(fetch_payload.get("bytes_read", 0), 0.0))),
                status=status,
            )
        )
        if parse_errors > 0:
            rows_to_append.append(
                _nws_stream_event_row(
                    "nws.alerts.parse-error",
                    f"parse_errors={parse_errors}",
                    status="degraded",
                )
            )
        if dedupe_dropped > 0:
            rows_to_append.append(
                _nws_stream_event_row(
                    "nws.alerts.dedupe",
                    f"dropped={dedupe_dropped}",
                    status="degraded",
                )
            )
        if rate_dropped > 0:
            rows_to_append.append(
                _nws_stream_event_row(
                    "nws.alerts.rate-limit",
                    f"dropped={rate_dropped}",
                    status="degraded",
                )
            )

    _append_nws_alert_rows(vault_root, rows_to_append)


def _world_event_from_nws_row(
    row: dict[str, Any], log_path: str
) -> dict[str, Any] | None:
    return _world_event_from_stream_row(
        row,
        log_path,
        default_source="nws_alerts",
        default_kind="nws.alert",
    )


def _swpc_alert_id(raw: dict[str, Any], payload_text: str) -> str:
    seed_parts = [
        str(raw.get("message_code", "")).strip(),
        str(raw.get("serial_number", "")).strip(),
        str(raw.get("issue_datetime", "")).strip(),
        str(raw.get("message_type", "")).strip(),
    ]
    seed = "|".join(seed_parts)
    if not seed.strip("|"):
        seed = payload_text
    return "swpc:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]


def _swpc_alert_row(raw: dict[str, Any], *, received_at: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    detail_limit = _cfg_int(
        "SWPC_ALERTS_DETAIL_CHAR_LIMIT", SWPC_ALERTS_DETAIL_CHAR_LIMIT
    )
    message_type = str(raw.get("message_type", "")).strip().lower()
    message_code = str(raw.get("message_code", "")).strip().lower()
    serial_number = str(raw.get("serial_number", "")).strip()
    issue_datetime = str(raw.get("issue_datetime", "")).strip()
    begin_datetime = str(raw.get("begin_datetime", "")).strip()
    end_datetime = str(raw.get("end_datetime", "")).strip()
    message_text = _compact_world_text(
        str(raw.get("message", "")).strip(), detail_limit
    )
    product = str(raw.get("product", "")).strip().lower()
    title = " ".join(
        [
            token
            for token in [
                message_type.upper() if message_type else "SWPC",
                message_code.upper() if message_code else "",
                ("#" + serial_number) if serial_number else "",
            ]
            if token
        ]
    )
    if not title:
        title = "SWPC alert"
    detail_parts: list[str] = []
    if begin_datetime:
        detail_parts.append("begin=" + begin_datetime)
    if end_datetime:
        detail_parts.append("end=" + end_datetime)
    if message_text:
        detail_parts.append(message_text)
    detail = _compact_world_text(" | ".join(detail_parts), detail_limit)
    ts = issue_datetime or begin_datetime or received_at

    refs: list[str] = []
    for candidate in [
        str(raw.get("url", "")).strip(),
        str(raw.get("link", "")).strip(),
    ]:
        if candidate and candidate not in refs:
            refs.append(candidate)

    payload_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    alert_id = _swpc_alert_id(raw, payload_text)
    kind_slug = _world_log_kind_slug(message_type or message_code or product, "alert")
    tags = [
        token
        for token in [
            "swpc",
            "space-weather",
            message_type,
            message_code,
            product,
        ]
        if token
    ]
    return {
        "record": SWPC_ALERT_RECORD,
        "id": alert_id,
        "ts": ts,
        "source": "swpc_alerts",
        "kind": "swpc.alert." + kind_slug,
        "status": "actual",
        "title": title,
        "detail": detail,
        "refs": refs,
        "tags": tags,
        "meta": {
            "message_type": message_type,
            "message_code": message_code,
            "serial_number": serial_number,
            "issue_datetime": issue_datetime,
            "begin_datetime": begin_datetime,
            "end_datetime": end_datetime,
        },
    }


def _swpc_stream_event_row(
    kind: str,
    detail: str,
    *,
    status: str = "recorded",
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    event_id = _world_log_event_id("swpc_alerts", kind, now_iso, kind, detail)
    return {
        "record": SWPC_ALERT_STREAM_EVENT_RECORD,
        "id": event_id,
        "ts": now_iso,
        "source": "swpc_alerts",
        "kind": kind,
        "status": status,
        "title": kind,
        "detail": _compact_world_text(detail, 240),
        "refs": [],
        "tags": ["swpc", "space-weather"],
        "meta": {
            "endpoint": _cfg_str("SWPC_ALERTS_ENDPOINT", SWPC_ALERTS_ENDPOINT),
        },
    }


def _append_swpc_alert_rows(
    vault_root: Path, rows: list[dict[str, Any]]
) -> Path | None:
    if not rows:
        return None
    log_path = _swpc_alerts_log_path(vault_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return log_path


def _load_swpc_alert_rows(
    vault_root: Path,
    *,
    limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    log_path = _swpc_alerts_log_path(vault_root)
    if not log_path.exists() or not log_path.is_file():
        return str(log_path), []
    bounded = max(1, int(limit or 1))
    selected: list[dict[str, Any]] = []
    for raw_line in log_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            selected.append(row)
    if len(selected) > bounded:
        selected = selected[-bounded:]
    return str(log_path), selected


def _swpc_json_rows(
    payload: Any, *, max_alerts: int
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    parse_errors = 0

    if isinstance(payload, list):
        headers: list[str] = []
        start_idx = 0
        if payload and isinstance(payload[0], list):
            raw_headers = [
                _world_log_kind_slug(str(token), "field")
                for token in payload[0]
                if str(token).strip()
            ]
            if raw_headers:
                headers = raw_headers
                start_idx = 1
        for item in payload[start_idx:]:
            if isinstance(item, dict):
                rows.append(dict(item))
                continue
            if isinstance(item, list) and headers:
                row: dict[str, Any] = {}
                for idx, key in enumerate(headers):
                    if idx >= len(item):
                        break
                    row[key] = item[idx]
                if row:
                    rows.append(row)
                else:
                    parse_errors += 1
            else:
                parse_errors += 1
    elif isinstance(payload, dict):
        candidates = payload.get(
            "alerts", payload.get("products", payload.get("events", []))
        )
        if isinstance(candidates, list):
            for item in candidates:
                if isinstance(item, dict):
                    rows.append(dict(item))
                else:
                    parse_errors += 1
        else:
            parse_errors += 1
    else:
        parse_errors += 1

    if len(rows) > max_alerts:
        rows = rows[:max_alerts]
    return rows, parse_errors


def _swpc_fetch_alert_rows(
    endpoint: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_alerts: int,
) -> dict[str, Any]:
    req = Request(
        endpoint,
        headers={
            "Accept": "application/json,text/json;q=0.9,*/*;q=0.8",
            "Cache-Control": "no-cache",
            "User-Agent": "eta-mu-world-web/1.0",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=max(0.1, timeout_seconds)) as response:
            payload_bytes = response.read(max(1, max_bytes))
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}:{exc}",
            "alerts": [],
            "parse_errors": 0,
            "bytes_read": 0,
        }

    bytes_read = len(payload_bytes)
    try:
        payload = json.loads(payload_bytes.decode("utf-8", errors="replace"))
    except (ValueError, json.JSONDecodeError):
        return {
            "ok": False,
            "error": "JSONDecodeError:invalid_payload",
            "alerts": [],
            "parse_errors": 1,
            "bytes_read": bytes_read,
        }

    alerts, parse_errors = _swpc_json_rows(payload, max_alerts=max_alerts)
    return {
        "ok": True,
        "error": "",
        "alerts": alerts,
        "parse_errors": parse_errors,
        "bytes_read": bytes_read,
    }


def _collect_swpc_alert_rows(vault_root: Path) -> None:
    enabled = _cfg_bool("SWPC_ALERTS_ENABLED", SWPC_ALERTS_ENABLED)
    poll_interval = _cfg_float(
        "SWPC_ALERTS_POLL_INTERVAL_SECONDS",
        SWPC_ALERTS_POLL_INTERVAL_SECONDS,
    )
    now_monotonic = time.monotonic()

    with _SWPC_ALERTS_LOCK:
        was_paused = bool(_SWPC_ALERTS_CACHE.get("paused", False))
        if not enabled:
            _SWPC_ALERTS_CACHE["paused"] = True
            if not was_paused:
                _append_swpc_alert_rows(
                    vault_root,
                    [
                        _swpc_stream_event_row(
                            "swpc.alerts.paused",
                            "SWPC polling disabled by configuration",
                            status="paused",
                        )
                    ],
                )
            return

        if was_paused:
            _append_swpc_alert_rows(
                vault_root,
                [
                    _swpc_stream_event_row(
                        "swpc.alerts.resumed",
                        "SWPC polling resumed by configuration",
                    )
                ],
            )
            _SWPC_ALERTS_CACHE["paused"] = False

        last_poll = _safe_float(_SWPC_ALERTS_CACHE.get("last_poll_monotonic", 0.0), 0.0)
        if now_monotonic - last_poll < max(1.0, poll_interval):
            return
        _SWPC_ALERTS_CACHE["last_poll_monotonic"] = now_monotonic

    endpoint = _cfg_str("SWPC_ALERTS_ENDPOINT", SWPC_ALERTS_ENDPOINT).strip()
    if not endpoint:
        _append_swpc_alert_rows(
            vault_root,
            [
                _swpc_stream_event_row(
                    "swpc.alerts.error",
                    "SWPC endpoint is empty",
                    status="error",
                )
            ],
        )
        return

    fetch_payload = _swpc_fetch_alert_rows(
        endpoint,
        timeout_seconds=_cfg_float(
            "SWPC_ALERTS_FETCH_TIMEOUT_SECONDS",
            SWPC_ALERTS_FETCH_TIMEOUT_SECONDS,
        ),
        max_bytes=_cfg_int("SWPC_ALERTS_MAX_BYTES", SWPC_ALERTS_MAX_BYTES),
        max_alerts=_cfg_int(
            "SWPC_ALERTS_MAX_ALERTS_PER_POLL",
            SWPC_ALERTS_MAX_ALERTS_PER_POLL,
        ),
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    rows_to_append: list[dict[str, Any]] = []
    with _SWPC_ALERTS_LOCK:
        if fetch_payload.get("ok") is not True:
            rows_to_append.append(
                _swpc_stream_event_row(
                    "swpc.alerts.error",
                    str(fetch_payload.get("error", "fetch_failed")),
                    status="error",
                )
            )
            _SWPC_ALERTS_CACHE["connected"] = False
            _append_swpc_alert_rows(vault_root, rows_to_append)
            return

        if not bool(_SWPC_ALERTS_CACHE.get("connected", False)):
            rows_to_append.append(
                _swpc_stream_event_row(
                    "swpc.alerts.connected",
                    "SWPC alerts fetch succeeded",
                )
            )
        _SWPC_ALERTS_CACHE["connected"] = True

        seen_ids = _SWPC_ALERTS_CACHE.get("seen_ids")
        if not isinstance(seen_ids, dict):
            seen_ids = {}
            _SWPC_ALERTS_CACHE["seen_ids"] = seen_ids

        dedupe_ttl = _cfg_float(
            "SWPC_ALERTS_DEDUPE_TTL_SECONDS",
            SWPC_ALERTS_DEDUPE_TTL_SECONDS,
        )
        stale_keys = [
            key
            for key, seen_at in seen_ids.items()
            if now_monotonic - _safe_float(seen_at, 0.0) > max(15.0, dedupe_ttl)
        ]
        for key in stale_keys:
            seen_ids.pop(key, None)

        dedupe_dropped = 0
        rate_dropped = 0
        accepted = 0
        allowed = _cfg_int(
            "SWPC_ALERTS_RATE_LIMIT_PER_POLL",
            SWPC_ALERTS_RATE_LIMIT_PER_POLL,
        )

        for raw_alert in fetch_payload.get("alerts", []):
            event_row = _swpc_alert_row(raw_alert, received_at=now_iso)
            if event_row is None:
                continue
            dedupe_key = str(event_row.get("id", "")).strip()
            if dedupe_key in seen_ids:
                dedupe_dropped += 1
                continue
            seen_ids[dedupe_key] = now_monotonic
            if accepted >= max(1, allowed):
                rate_dropped += 1
                continue
            rows_to_append.append(event_row)
            accepted += 1

        parse_errors = int(_safe_float(fetch_payload.get("parse_errors", 0), 0.0))
        status = "recorded"
        if parse_errors > 0 or dedupe_dropped > 0 or rate_dropped > 0:
            status = "degraded"
        rows_to_append.append(
            _swpc_stream_event_row(
                "swpc.alerts.poll",
                "accepted="
                + str(accepted)
                + " parse_errors="
                + str(parse_errors)
                + " dedupe_dropped="
                + str(dedupe_dropped)
                + " rate_dropped="
                + str(rate_dropped)
                + " bytes="
                + str(int(_safe_float(fetch_payload.get("bytes_read", 0), 0.0))),
                status=status,
            )
        )
        if parse_errors > 0:
            rows_to_append.append(
                _swpc_stream_event_row(
                    "swpc.alerts.parse-error",
                    f"parse_errors={parse_errors}",
                    status="degraded",
                )
            )
        if dedupe_dropped > 0:
            rows_to_append.append(
                _swpc_stream_event_row(
                    "swpc.alerts.dedupe",
                    f"dropped={dedupe_dropped}",
                    status="degraded",
                )
            )
        if rate_dropped > 0:
            rows_to_append.append(
                _swpc_stream_event_row(
                    "swpc.alerts.rate-limit",
                    f"dropped={rate_dropped}",
                    status="degraded",
                )
            )

    _append_swpc_alert_rows(vault_root, rows_to_append)


def _world_event_from_swpc_row(
    row: dict[str, Any], log_path: str
) -> dict[str, Any] | None:
    return _world_event_from_stream_row(
        row,
        log_path,
        default_source="swpc_alerts",
        default_kind="swpc.alert",
    )


def _eonet_event_time(raw: dict[str, Any], fallback_ts: str) -> str:
    geometry_value = raw.get("geometry")
    geometries: list[Any] = []
    if isinstance(geometry_value, list):
        geometries = geometry_value
    dates = [
        str(item.get("date", "")).strip()
        for item in geometries
        if isinstance(item, dict) and str(item.get("date", "")).strip()
    ]
    if dates:
        return max(dates)
    return str(raw.get("closed", "")).strip() or fallback_ts


def _eonet_event_id(raw: dict[str, Any], payload_text: str) -> str:
    base_id = str(raw.get("id", "")).strip()
    seed = "id:" + base_id if base_id else "raw:" + payload_text
    return "eonet:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]


def _eonet_categories(raw: dict[str, Any]) -> list[str]:
    categories_value = raw.get("categories")
    if not isinstance(categories_value, list):
        return []

    categories = [
        str(item.get("title", "")).strip().lower()
        for item in categories_value
        if isinstance(item, dict) and str(item.get("title", "")).strip()
    ]
    return list(dict.fromkeys(categories))


def _eonet_latest_geometry_type(geometries: list[Any]) -> str:
    for item in reversed(geometries):
        if not isinstance(item, dict):
            continue
        latest_geometry_type = str(item.get("type", "")).strip().lower()
        if latest_geometry_type:
            return latest_geometry_type
    return ""


def _eonet_event_refs(raw: dict[str, Any], sources: list[Any]) -> list[str]:
    refs: list[str] = []
    for candidate in [
        str(raw.get("link", "")).strip(),
        *[
            str(source.get("url", "")).strip()
            for source in sources
            if isinstance(source, dict)
        ],
    ]:
        if candidate and candidate not in refs:
            refs.append(candidate)
    return refs


def _eonet_event_row(raw: dict[str, Any], *, received_at: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    detail_limit = _cfg_int(
        "EONET_EVENTS_DETAIL_CHAR_LIMIT", EONET_EVENTS_DETAIL_CHAR_LIMIT
    )
    title = str(raw.get("title", "")).strip() or "EONET event"
    categories = _eonet_categories(raw)
    primary_category = categories[0] if categories else "event"
    kind_slug = _world_log_kind_slug(primary_category, "event")
    closed_at = str(raw.get("closed", "")).strip()
    status = "closed" if closed_at else "open"

    geometry_value = raw.get("geometry")
    geometries: list[Any] = []
    if isinstance(geometry_value, list):
        geometries = geometry_value
    geometry_count = len([item for item in geometries if isinstance(item, dict)])
    latest_geometry_type = _eonet_latest_geometry_type(geometries)

    detail_parts = [
        "status=" + status,
        "categories=" + ",".join(categories[:3]) if categories else "",
        "geometry=" + str(geometry_count),
        "latest_type=" + latest_geometry_type if latest_geometry_type else "",
    ]
    detail = _compact_world_text(
        " | ".join([token for token in detail_parts if token]), detail_limit
    )
    ts = _eonet_event_time(raw, received_at)

    sources_value = raw.get("sources")
    sources = sources_value if isinstance(sources_value, list) else []
    refs = _eonet_event_refs(raw, sources)

    payload_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    event_id = _eonet_event_id(raw, payload_text)
    tags = ["eonet", "nasa", status, *categories[:4]]
    tags = [token for token in tags if token]
    return {
        "record": EONET_EVENT_RECORD,
        "id": event_id,
        "ts": ts,
        "source": "nasa_eonet",
        "kind": "eonet.event." + kind_slug,
        "status": status,
        "title": title,
        "detail": detail,
        "refs": refs,
        "tags": tags,
        "meta": {
            "categories": categories,
            "closed": closed_at,
            "geometry_count": geometry_count,
            "latest_geometry_type": latest_geometry_type,
        },
    }


def _eonet_stream_event_row(
    kind: str,
    detail: str,
    *,
    status: str = "recorded",
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    event_id = _world_log_event_id("nasa_eonet", kind, now_iso, kind, detail)
    return {
        "record": EONET_STREAM_EVENT_RECORD,
        "id": event_id,
        "ts": now_iso,
        "source": "nasa_eonet",
        "kind": kind,
        "status": status,
        "title": kind,
        "detail": _compact_world_text(detail, 260),
        "refs": [],
        "tags": ["eonet", "nasa", "hazards"],
        "meta": {
            "endpoint": _cfg_str("EONET_EVENTS_ENDPOINT", EONET_EVENTS_ENDPOINT),
        },
    }


def _append_eonet_event_rows(
    vault_root: Path, rows: list[dict[str, Any]]
) -> Path | None:
    if not rows:
        return None
    log_path = _eonet_events_log_path(vault_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return log_path


def _load_eonet_event_rows(
    vault_root: Path,
    *,
    limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    log_path = _eonet_events_log_path(vault_root)
    if not log_path.exists() or not log_path.is_file():
        return str(log_path), []
    bounded = max(1, int(limit or 1))
    selected: list[dict[str, Any]] = []
    for raw_line in log_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            selected.append(row)
    if len(selected) > bounded:
        selected = selected[-bounded:]
    return str(log_path), selected


def _eonet_fetch_events(
    endpoint: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_events: int,
) -> dict[str, Any]:
    req = Request(
        endpoint,
        headers={
            "Accept": "application/json,text/json;q=0.9,*/*;q=0.8",
            "Cache-Control": "no-cache",
            "User-Agent": "eta-mu-world-web/1.0",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=max(0.1, timeout_seconds)) as response:
            payload_bytes = response.read(max(1, max_bytes))
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}:{exc}",
            "events": [],
            "parse_errors": 0,
            "bytes_read": 0,
        }

    bytes_read = len(payload_bytes)
    try:
        payload = json.loads(payload_bytes.decode("utf-8", errors="replace"))
    except (ValueError, json.JSONDecodeError):
        return {
            "ok": False,
            "error": "JSONDecodeError:invalid_payload",
            "events": [],
            "parse_errors": 1,
            "bytes_read": bytes_read,
        }

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "payload_not_object",
            "events": [],
            "parse_errors": 1,
            "bytes_read": bytes_read,
        }

    parse_errors = 0
    events_raw = payload.get("events", [])
    if not isinstance(events_raw, list):
        parse_errors = 1
        events_raw = []
    events = [item for item in events_raw if isinstance(item, dict)]
    parse_errors += max(0, len(events_raw) - len(events))
    if len(events) > max_events:
        events = events[:max_events]
    return {
        "ok": True,
        "error": "",
        "events": events,
        "parse_errors": parse_errors,
        "bytes_read": bytes_read,
    }


def _collect_eonet_event_rows(vault_root: Path) -> None:
    enabled = _cfg_bool("EONET_EVENTS_ENABLED", EONET_EVENTS_ENABLED)
    poll_interval = _cfg_float(
        "EONET_EVENTS_POLL_INTERVAL_SECONDS",
        EONET_EVENTS_POLL_INTERVAL_SECONDS,
    )
    now_monotonic = time.monotonic()

    with _EONET_EVENTS_LOCK:
        was_paused = bool(_EONET_EVENTS_CACHE.get("paused", False))
        if not enabled:
            _EONET_EVENTS_CACHE["paused"] = True
            if not was_paused:
                _append_eonet_event_rows(
                    vault_root,
                    [
                        _eonet_stream_event_row(
                            "eonet.events.paused",
                            "EONET polling disabled by configuration",
                            status="paused",
                        )
                    ],
                )
            return

        if was_paused:
            _append_eonet_event_rows(
                vault_root,
                [
                    _eonet_stream_event_row(
                        "eonet.events.resumed",
                        "EONET polling resumed by configuration",
                    )
                ],
            )
            _EONET_EVENTS_CACHE["paused"] = False

        last_poll = _safe_float(
            _EONET_EVENTS_CACHE.get("last_poll_monotonic", 0.0), 0.0
        )
        if now_monotonic - last_poll < max(1.0, poll_interval):
            return
        _EONET_EVENTS_CACHE["last_poll_monotonic"] = now_monotonic

    endpoint = _cfg_str("EONET_EVENTS_ENDPOINT", EONET_EVENTS_ENDPOINT).strip()
    if not endpoint:
        _append_eonet_event_rows(
            vault_root,
            [
                _eonet_stream_event_row(
                    "eonet.events.error",
                    "EONET endpoint is empty",
                    status="error",
                )
            ],
        )
        return

    fetch_payload = _eonet_fetch_events(
        endpoint,
        timeout_seconds=_cfg_float(
            "EONET_EVENTS_FETCH_TIMEOUT_SECONDS",
            EONET_EVENTS_FETCH_TIMEOUT_SECONDS,
        ),
        max_bytes=_cfg_int("EONET_EVENTS_MAX_BYTES", EONET_EVENTS_MAX_BYTES),
        max_events=_cfg_int(
            "EONET_EVENTS_MAX_EVENTS_PER_POLL",
            EONET_EVENTS_MAX_EVENTS_PER_POLL,
        ),
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    rows_to_append: list[dict[str, Any]] = []
    with _EONET_EVENTS_LOCK:
        if fetch_payload.get("ok") is not True:
            rows_to_append.append(
                _eonet_stream_event_row(
                    "eonet.events.error",
                    str(fetch_payload.get("error", "fetch_failed")),
                    status="error",
                )
            )
            _EONET_EVENTS_CACHE["connected"] = False
            _append_eonet_event_rows(vault_root, rows_to_append)
            return

        if not bool(_EONET_EVENTS_CACHE.get("connected", False)):
            rows_to_append.append(
                _eonet_stream_event_row(
                    "eonet.events.connected",
                    "EONET event fetch succeeded",
                )
            )
        _EONET_EVENTS_CACHE["connected"] = True

        seen_ids = _EONET_EVENTS_CACHE.get("seen_ids")
        if not isinstance(seen_ids, dict):
            seen_ids = {}
            _EONET_EVENTS_CACHE["seen_ids"] = seen_ids

        dedupe_ttl = _cfg_float(
            "EONET_EVENTS_DEDUPE_TTL_SECONDS",
            EONET_EVENTS_DEDUPE_TTL_SECONDS,
        )
        stale_keys = [
            key
            for key, seen_at in seen_ids.items()
            if now_monotonic - _safe_float(seen_at, 0.0) > max(15.0, dedupe_ttl)
        ]
        for key in stale_keys:
            seen_ids.pop(key, None)

        dedupe_dropped = 0
        rate_dropped = 0
        accepted = 0
        allowed = _cfg_int(
            "EONET_EVENTS_RATE_LIMIT_PER_POLL",
            EONET_EVENTS_RATE_LIMIT_PER_POLL,
        )

        for raw_event in fetch_payload.get("events", []):
            event_row = _eonet_event_row(raw_event, received_at=now_iso)
            if event_row is None:
                continue
            dedupe_key = str(event_row.get("id", "")).strip()
            if dedupe_key in seen_ids:
                dedupe_dropped += 1
                continue
            seen_ids[dedupe_key] = now_monotonic
            if accepted >= max(1, allowed):
                rate_dropped += 1
                continue
            rows_to_append.append(event_row)
            accepted += 1

        parse_errors = int(_safe_float(fetch_payload.get("parse_errors", 0), 0.0))
        status = "recorded"
        if parse_errors > 0 or dedupe_dropped > 0 or rate_dropped > 0:
            status = "degraded"
        rows_to_append.append(
            _eonet_stream_event_row(
                "eonet.events.poll",
                "accepted="
                + str(accepted)
                + " parse_errors="
                + str(parse_errors)
                + " dedupe_dropped="
                + str(dedupe_dropped)
                + " rate_dropped="
                + str(rate_dropped)
                + " bytes="
                + str(int(_safe_float(fetch_payload.get("bytes_read", 0), 0.0))),
                status=status,
            )
        )
        if parse_errors > 0:
            rows_to_append.append(
                _eonet_stream_event_row(
                    "eonet.events.parse-error",
                    f"parse_errors={parse_errors}",
                    status="degraded",
                )
            )
        if dedupe_dropped > 0:
            rows_to_append.append(
                _eonet_stream_event_row(
                    "eonet.events.dedupe",
                    f"dropped={dedupe_dropped}",
                    status="degraded",
                )
            )
        if rate_dropped > 0:
            rows_to_append.append(
                _eonet_stream_event_row(
                    "eonet.events.rate-limit",
                    f"dropped={rate_dropped}",
                    status="degraded",
                )
            )

    _append_eonet_event_rows(vault_root, rows_to_append)


def _world_event_from_eonet_row(
    row: dict[str, Any],
    log_path: str,
) -> dict[str, Any] | None:
    return _world_event_from_stream_row(
        row,
        log_path,
        default_source="nasa_eonet",
        default_kind="eonet.event",
    )


def _gibs_target_layers() -> list[str]:
    raw = _cfg_str("GIBS_LAYERS_TARGETS", GIBS_LAYERS_TARGETS)
    names = [
        token
        for token in _split_csv_items(raw)
        if re.fullmatch(r"[A-Za-z0-9._-]+", token)
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _xml_child_text(node: ET.Element, local_name: str) -> str:
    for child in list(node):
        if str(child.tag).split("}")[-1] != local_name:
            continue
        return str(child.text or "").strip()
    return ""


def _gibs_time_token(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if "/" not in token:
        return token
    parts = [part.strip() for part in token.split("/") if part.strip()]
    if len(parts) >= 2:
        return parts[1]
    if parts:
        return parts[0]
    return ""


def _gibs_dimension_times(layer: ET.Element) -> tuple[str, str]:
    default_time = ""
    latest_time = ""
    for dimension in layer.findall("{*}Dimension"):
        identifier = _xml_child_text(dimension, "Identifier").lower()
        if identifier not in {"time", "date"}:
            continue
        default_time = _xml_child_text(dimension, "Default")
        for value_node in dimension.findall("{*}Value"):
            value = str(value_node.text or "").strip()
            if not value:
                continue
            for piece in value.split(","):
                token = _gibs_time_token(piece)
                if token:
                    latest_time = token
        break
    return default_time, latest_time


def _gibs_layer_matrix_set(layer: ET.Element) -> str:
    for link in layer.findall("{*}TileMatrixSetLink"):
        matrix_set = _xml_child_text(link, "TileMatrixSet")
        if matrix_set:
            return matrix_set
    return _cfg_str("GIBS_LAYERS_TILE_MATRIX_SET", GIBS_LAYERS_TILE_MATRIX_SET).strip()


def _gibs_layers_from_capabilities_xml(
    caps_xml: str,
    *,
    target_layers: list[str],
    max_layers: int,
) -> tuple[list[dict[str, Any]], int]:
    try:
        root = ET.fromstring(caps_xml)
    except ET.ParseError:
        return [], 1

    target_set = set(target_layers)
    bounded = max(1, int(max_layers or 1))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for layer in root.findall(".//{*}Layer"):
        layer_id = _xml_child_text(layer, "Identifier")
        if not layer_id or layer_id in seen:
            continue
        if target_set and layer_id not in target_set:
            continue
        seen.add(layer_id)
        title = _xml_child_text(layer, "Title") or layer_id
        default_time, latest_time = _gibs_dimension_times(layer)
        formats: list[str] = []
        for format_node in layer.findall("{*}Format"):
            value = str(format_node.text or "").strip()
            if value and value not in formats:
                formats.append(value)
        selected.append(
            {
                "layer_id": layer_id,
                "title": title,
                "default_time": default_time,
                "latest_time": latest_time,
                "tile_matrix_set": _gibs_layer_matrix_set(layer),
                "formats": formats,
            }
        )
        if len(selected) >= bounded:
            break

    parse_errors = 0
    if target_set and not selected:
        parse_errors = 1
    return selected, parse_errors


def _gibs_tile_extension(formats: list[str]) -> str:
    for raw_format in formats:
        value = str(raw_format or "").strip().lower()
        if "png" in value:
            return "png"
        if "jpeg" in value or "jpg" in value:
            return "jpg"
        if "webp" in value:
            return "webp"
    return "jpg"


def _gibs_tile_date_token(default_time: str, latest_time: str) -> str:
    candidate = str(default_time or latest_time or "").strip()
    if candidate:
        candidate = _gibs_time_token(candidate.split(",", 1)[0])
        if "T" in candidate:
            candidate = candidate.split("T", 1)[0]
        if candidate:
            return candidate
    return datetime.now(timezone.utc).date().isoformat()


def _gibs_tile_base_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    scheme = str(parsed.scheme or "").strip().lower()
    netloc = str(parsed.netloc or "").strip()
    if not scheme or not netloc:
        return ""
    path = str(parsed.path or "").strip()
    if path.endswith("/wmts.cgi"):
        path = path[: -len("/wmts.cgi")]
    elif path.endswith("wmts.cgi"):
        path = path[: -len("wmts.cgi")]
    path = path.rstrip("/")
    if not path:
        path = "/wmts/epsg4326/best"
    return f"{scheme}://{netloc}{path}"


def _gibs_tile_url(
    endpoint: str,
    *,
    layer_id: str,
    date_token: str,
    matrix_set: str,
    extension: str,
) -> str:
    base = _gibs_tile_base_url(endpoint)
    if not base:
        return ""
    tile_matrix = str(_cfg_int("GIBS_LAYERS_TILE_MATRIX", GIBS_LAYERS_TILE_MATRIX))
    tile_row = str(_cfg_int("GIBS_LAYERS_TILE_ROW", GIBS_LAYERS_TILE_ROW))
    tile_col = str(_cfg_int("GIBS_LAYERS_TILE_COL", GIBS_LAYERS_TILE_COL))
    return (
        base
        + "/"
        + quote(layer_id, safe="")
        + "/default/"
        + quote(date_token, safe="-")
        + "/"
        + quote(matrix_set, safe="")
        + "/"
        + tile_matrix
        + "/"
        + tile_row
        + "/"
        + tile_col
        + "."
        + extension
    )


def _gibs_layer_row(
    raw: dict[str, Any],
    *,
    endpoint: str,
    received_at: str,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    layer_id = str(raw.get("layer_id", "")).strip()
    if not layer_id:
        return None

    detail_limit = _cfg_int(
        "GIBS_LAYERS_DETAIL_CHAR_LIMIT", GIBS_LAYERS_DETAIL_CHAR_LIMIT
    )
    title = str(raw.get("title", "")).strip() or layer_id
    default_time = str(raw.get("default_time", "")).strip()
    latest_time = str(raw.get("latest_time", "")).strip()
    matrix_set = (
        str(raw.get("tile_matrix_set", "")).strip()
        or _cfg_str("GIBS_LAYERS_TILE_MATRIX_SET", GIBS_LAYERS_TILE_MATRIX_SET).strip()
        or "250m"
    )
    formats = raw.get("formats", [])
    if not isinstance(formats, list):
        formats = []

    date_token = _gibs_tile_date_token(default_time, latest_time)
    extension = _gibs_tile_extension(
        [str(item) for item in formats if str(item).strip()]
    )
    tile_url = _gibs_tile_url(
        endpoint,
        layer_id=layer_id,
        date_token=date_token,
        matrix_set=matrix_set,
        extension=extension,
    )
    event_seed = "|".join([layer_id, date_token, matrix_set, tile_url])
    event_id = "gibs:" + hashlib.sha1(event_seed.encode("utf-8")).hexdigest()[:20]

    slug = _world_log_kind_slug(layer_id, "layer")
    detail_parts = [
        "layer=" + layer_id,
        "date=" + date_token,
        "matrix_set=" + matrix_set,
    ]
    if default_time:
        detail_parts.append("default=" + default_time)
    if latest_time:
        detail_parts.append("latest=" + latest_time)
    detail = _compact_world_text(" | ".join(detail_parts), detail_limit)

    refs = [endpoint]
    if tile_url:
        refs.append(tile_url)
    tags = ["gibs", "nasa", "satellite", slug]
    lower_layer = layer_id.lower()
    if "truecolor" in lower_layer or "correctedreflectance" in lower_layer:
        tags.append("true-color")

    return {
        "record": GIBS_LAYER_RECORD,
        "id": event_id,
        "ts": received_at,
        "source": "nasa_gibs",
        "kind": "gibs.layer." + slug,
        "status": "recorded",
        "title": title,
        "detail": detail,
        "refs": refs,
        "tags": tags,
        "meta": {
            "layer_id": layer_id,
            "default_time": default_time,
            "latest_time": latest_time,
            "tile_date": date_token,
            "tile_matrix_set": matrix_set,
            "tile_url": tile_url,
        },
    }


def _gibs_stream_event_row(
    kind: str,
    detail: str,
    *,
    status: str = "recorded",
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    event_id = _world_log_event_id("nasa_gibs", kind, now_iso, kind, detail)
    return {
        "record": GIBS_LAYER_STREAM_EVENT_RECORD,
        "id": event_id,
        "ts": now_iso,
        "source": "nasa_gibs",
        "kind": kind,
        "status": status,
        "title": kind,
        "detail": _compact_world_text(detail, 260),
        "refs": [],
        "tags": ["gibs", "nasa", "satellite"],
        "meta": {
            "endpoint": _cfg_str(
                "GIBS_LAYERS_CAPABILITIES_ENDPOINT",
                GIBS_LAYERS_CAPABILITIES_ENDPOINT,
            ),
            "targets": _gibs_target_layers(),
        },
    }


def _append_gibs_layer_rows(
    vault_root: Path, rows: list[dict[str, Any]]
) -> Path | None:
    if not rows:
        return None
    log_path = _gibs_layers_log_path(vault_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return log_path


def _load_gibs_layer_rows(
    vault_root: Path,
    *,
    limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    log_path = _gibs_layers_log_path(vault_root)
    if not log_path.exists() or not log_path.is_file():
        return str(log_path), []
    bounded = max(1, int(limit or 1))
    selected: list[dict[str, Any]] = []
    for raw_line in log_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            selected.append(row)
    if len(selected) > bounded:
        selected = selected[-bounded:]
    return str(log_path), selected


def _gibs_endpoint_compliance_error(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    scheme = str(parsed.scheme or "").strip().lower()
    if scheme != "https":
        return "endpoint_must_use_https"
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return "endpoint_missing_host"
    if host != "gibs.earthdata.nasa.gov" and not host.endswith(".earthdata.nasa.gov"):
        return "endpoint_host_not_earthdata_nasa_gov"
    return ""


def _gibs_fetch_capabilities_layers(
    endpoint: str,
    *,
    target_layers: list[str],
    timeout_seconds: float,
    max_bytes: int,
    max_layers: int,
) -> dict[str, Any]:
    req = Request(
        endpoint,
        headers={
            "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
            "Cache-Control": "no-cache",
            "User-Agent": "eta-mu-world-web/1.0 (+https://github.com/err/fork_tales)",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=max(0.1, timeout_seconds)) as response:
            payload_bytes = response.read(max(1, max_bytes))
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}:{exc}",
            "layers": [],
            "parse_errors": 0,
            "bytes_read": 0,
        }

    bytes_read = len(payload_bytes)
    payload_text = payload_bytes.decode("utf-8", errors="replace")
    layers, parse_errors = _gibs_layers_from_capabilities_xml(
        payload_text,
        target_layers=target_layers,
        max_layers=max_layers,
    )
    return {
        "ok": True,
        "error": "",
        "layers": layers,
        "parse_errors": parse_errors,
        "bytes_read": bytes_read,
    }


def _collect_gibs_layer_rows(vault_root: Path) -> None:
    enabled = _cfg_bool("GIBS_LAYERS_ENABLED", GIBS_LAYERS_ENABLED)
    poll_interval = _cfg_float(
        "GIBS_LAYERS_POLL_INTERVAL_SECONDS",
        GIBS_LAYERS_POLL_INTERVAL_SECONDS,
    )
    now_monotonic = time.monotonic()

    with _GIBS_LAYERS_LOCK:
        was_paused = bool(_GIBS_LAYERS_CACHE.get("paused", False))
        if not enabled:
            _GIBS_LAYERS_CACHE["paused"] = True
            if not was_paused:
                _append_gibs_layer_rows(
                    vault_root,
                    [
                        _gibs_stream_event_row(
                            "gibs.layers.paused",
                            "GIBS polling disabled by configuration",
                            status="paused",
                        )
                    ],
                )
            return

        if was_paused:
            _append_gibs_layer_rows(
                vault_root,
                [
                    _gibs_stream_event_row(
                        "gibs.layers.resumed",
                        "GIBS polling resumed by configuration",
                    )
                ],
            )
            _GIBS_LAYERS_CACHE["paused"] = False

        last_poll = _safe_float(_GIBS_LAYERS_CACHE.get("last_poll_monotonic", 0.0), 0.0)
        if now_monotonic - last_poll < max(1.0, poll_interval):
            return
        _GIBS_LAYERS_CACHE["last_poll_monotonic"] = now_monotonic

    endpoint = _cfg_str(
        "GIBS_LAYERS_CAPABILITIES_ENDPOINT",
        GIBS_LAYERS_CAPABILITIES_ENDPOINT,
    ).strip()
    if not endpoint:
        _append_gibs_layer_rows(
            vault_root,
            [
                _gibs_stream_event_row(
                    "gibs.layers.error",
                    "GIBS capabilities endpoint is empty",
                    status="error",
                )
            ],
        )
        return

    compliance_error = _gibs_endpoint_compliance_error(endpoint)
    if compliance_error:
        with _GIBS_LAYERS_LOCK:
            _GIBS_LAYERS_CACHE["connected"] = False
        _append_gibs_layer_rows(
            vault_root,
            [
                _gibs_stream_event_row(
                    "gibs.layers.compliance",
                    compliance_error,
                    status="blocked",
                )
            ],
        )
        return

    fetch_payload = _gibs_fetch_capabilities_layers(
        endpoint,
        target_layers=_gibs_target_layers(),
        timeout_seconds=_cfg_float(
            "GIBS_LAYERS_FETCH_TIMEOUT_SECONDS",
            GIBS_LAYERS_FETCH_TIMEOUT_SECONDS,
        ),
        max_bytes=_cfg_int("GIBS_LAYERS_MAX_BYTES", GIBS_LAYERS_MAX_BYTES),
        max_layers=_cfg_int(
            "GIBS_LAYERS_MAX_LAYERS_PER_POLL",
            GIBS_LAYERS_MAX_LAYERS_PER_POLL,
        ),
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    rows_to_append: list[dict[str, Any]] = []
    with _GIBS_LAYERS_LOCK:
        if fetch_payload.get("ok") is not True:
            rows_to_append.append(
                _gibs_stream_event_row(
                    "gibs.layers.error",
                    str(fetch_payload.get("error", "fetch_failed")),
                    status="error",
                )
            )
            _GIBS_LAYERS_CACHE["connected"] = False
            _append_gibs_layer_rows(vault_root, rows_to_append)
            return

        if not bool(_GIBS_LAYERS_CACHE.get("connected", False)):
            rows_to_append.append(
                _gibs_stream_event_row(
                    "gibs.layers.connected",
                    "GIBS capabilities fetch succeeded",
                )
            )
        _GIBS_LAYERS_CACHE["connected"] = True

        seen_ids = _GIBS_LAYERS_CACHE.get("seen_ids")
        if not isinstance(seen_ids, dict):
            seen_ids = {}
            _GIBS_LAYERS_CACHE["seen_ids"] = seen_ids

        dedupe_ttl = _cfg_float(
            "GIBS_LAYERS_DEDUPE_TTL_SECONDS",
            GIBS_LAYERS_DEDUPE_TTL_SECONDS,
        )
        stale_keys = [
            key
            for key, seen_at in seen_ids.items()
            if now_monotonic - _safe_float(seen_at, 0.0) > max(15.0, dedupe_ttl)
        ]
        for key in stale_keys:
            seen_ids.pop(key, None)

        dedupe_dropped = 0
        rate_dropped = 0
        accepted = 0
        allowed = _cfg_int(
            "GIBS_LAYERS_RATE_LIMIT_PER_POLL",
            GIBS_LAYERS_RATE_LIMIT_PER_POLL,
        )

        for raw_layer in fetch_payload.get("layers", []):
            layer_row = _gibs_layer_row(
                raw_layer, endpoint=endpoint, received_at=now_iso
            )
            if layer_row is None:
                continue
            dedupe_key = str(layer_row.get("id", "")).strip()
            if dedupe_key in seen_ids:
                dedupe_dropped += 1
                continue
            seen_ids[dedupe_key] = now_monotonic
            if accepted >= max(1, allowed):
                rate_dropped += 1
                continue
            rows_to_append.append(layer_row)
            accepted += 1

        parse_errors = int(_safe_float(fetch_payload.get("parse_errors", 0), 0.0))
        status = "recorded"
        if parse_errors > 0 or dedupe_dropped > 0 or rate_dropped > 0:
            status = "degraded"
        rows_to_append.append(
            _gibs_stream_event_row(
                "gibs.layers.poll",
                "accepted="
                + str(accepted)
                + " parse_errors="
                + str(parse_errors)
                + " dedupe_dropped="
                + str(dedupe_dropped)
                + " rate_dropped="
                + str(rate_dropped)
                + " bytes="
                + str(int(_safe_float(fetch_payload.get("bytes_read", 0), 0.0))),
                status=status,
            )
        )
        if parse_errors > 0:
            rows_to_append.append(
                _gibs_stream_event_row(
                    "gibs.layers.parse-error",
                    f"parse_errors={parse_errors}",
                    status="degraded",
                )
            )
        if dedupe_dropped > 0:
            rows_to_append.append(
                _gibs_stream_event_row(
                    "gibs.layers.dedupe",
                    f"dropped={dedupe_dropped}",
                    status="degraded",
                )
            )
        if rate_dropped > 0:
            rows_to_append.append(
                _gibs_stream_event_row(
                    "gibs.layers.rate-limit",
                    f"dropped={rate_dropped}",
                    status="degraded",
                )
            )

    _append_gibs_layer_rows(vault_root, rows_to_append)


def _world_event_from_gibs_row(
    row: dict[str, Any],
    log_path: str,
) -> dict[str, Any] | None:
    return _world_event_from_stream_row(
        row,
        log_path,
        default_source="nasa_gibs",
        default_kind="gibs.layer",
    )


def _ws_recv_exact(sock: socket.socket, size: int) -> bytes:
    if size <= 0:
        return b""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise OSError("socket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _socket_read_until(sock: socket.socket, marker: bytes, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        return b""
    data = bytearray()
    while len(data) < max_bytes:
        chunk = sock.recv(1)
        if not chunk:
            break
        data.extend(chunk)
        if marker in data:
            break
    return bytes(data)


def _ws_read_frame(
    sock: socket.socket,
    *,
    max_payload: int,
) -> tuple[int, bytes, int] | None:
    try:
        head = _ws_recv_exact(sock, 2)
    except (socket.timeout, TimeoutError, OSError):
        return None

    first, second = head[0], head[1]
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    payload_len = second & 0x7F
    header_len = 2

    if payload_len == 126:
        ext = _ws_recv_exact(sock, 2)
        payload_len = struct.unpack("!H", ext)[0]
        header_len += 2
    elif payload_len == 127:
        ext = _ws_recv_exact(sock, 8)
        payload_len = struct.unpack("!Q", ext)[0]
        header_len += 8

    mask_key = b""
    if masked:
        mask_key = _ws_recv_exact(sock, 4)
        header_len += 4

    if payload_len < 0:
        return None
    if payload_len > max(0, max_payload):
        return None

    payload = _ws_recv_exact(sock, int(payload_len)) if payload_len > 0 else b""
    if masked and payload:
        payload = bytes(value ^ mask_key[idx % 4] for idx, value in enumerate(payload))
    return opcode, payload, header_len + int(payload_len)


def _json_objects_from_text_payload(
    payload_text: str,
) -> tuple[list[dict[str, Any]], int]:
    text = str(payload_text or "").strip()
    if not text:
        return [], 0

    rows: list[dict[str, Any]] = []
    parse_errors = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            parse_errors += 1
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            parse_errors += 1

    if rows:
        return rows, parse_errors

    try:
        parsed = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return [], max(1, parse_errors)
    if isinstance(parsed, dict):
        return [parsed], parse_errors
    return [], max(1, parse_errors)


def _emsc_fetch_ws_events(
    stream_url: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_events: int,
) -> dict[str, Any]:
    parsed = urlparse(stream_url)
    scheme = parsed.scheme.strip().lower()
    if scheme not in {"ws", "wss"}:
        return {
            "ok": False,
            "error": f"invalid_scheme:{scheme or '(empty)'}",
            "events": [],
            "parse_errors": 0,
            "bytes_read": 0,
        }

    host = str(parsed.hostname or "").strip()
    if not host:
        return {
            "ok": False,
            "error": "missing_host",
            "events": [],
            "parse_errors": 0,
            "bytes_read": 0,
        }

    port = int(parsed.port or (443 if scheme == "wss" else 80))
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    sock: socket.socket | None = None
    parse_errors = 0
    bytes_read = 0
    events: list[dict[str, Any]] = []
    try:
        raw_sock = socket.create_connection(
            (host, port),
            timeout=max(0.1, timeout_seconds),
        )
        if scheme == "wss":
            context = ssl.create_default_context()
            sock = context.wrap_socket(raw_sock, server_hostname=host)
        else:
            sock = raw_sock
        sock.settimeout(max(0.1, timeout_seconds))

        ws_key = base64.b64encode(os.urandom(16)).decode("ascii")
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {ws_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "User-Agent: eta-mu-world-web/1.0\r\n"
            "\r\n"
        )
        sock.sendall(handshake.encode("utf-8"))

        header_bytes = _socket_read_until(
            sock,
            marker=b"\r\n\r\n",
            max_bytes=min(16384, max(1024, max_bytes)),
        )
        bytes_read += len(header_bytes)
        header_text = header_bytes.decode("utf-8", errors="replace")
        header_head = header_text.split("\r\n\r\n", 1)[0]
        header_lines = [line for line in header_head.split("\r\n") if line.strip()]
        if not header_lines:
            return {
                "ok": False,
                "error": "empty_handshake",
                "events": [],
                "parse_errors": parse_errors,
                "bytes_read": bytes_read,
            }
        status_line = header_lines[0]
        if " 101 " not in f" {status_line} ":
            return {
                "ok": False,
                "error": "handshake_failed:" + status_line,
                "events": [],
                "parse_errors": parse_errors,
                "bytes_read": bytes_read,
            }

        headers: dict[str, str] = {}
        for line in header_lines[1:]:
            key, sep, value = line.partition(":")
            if not sep:
                continue
            headers[key.strip().lower()] = value.strip()
        expected_accept = base64.b64encode(
            hashlib.sha1((ws_key + WS_MAGIC).encode("utf-8")).digest()
        ).decode("ascii")
        if str(headers.get("sec-websocket-accept", "")).strip() != expected_accept:
            return {
                "ok": False,
                "error": "handshake_accept_mismatch",
                "events": [],
                "parse_errors": parse_errors,
                "bytes_read": bytes_read,
            }

        deadline = time.monotonic() + max(0.2, timeout_seconds)
        while (
            len(events) < max_events
            and bytes_read < max_bytes
            and time.monotonic() < deadline
        ):
            frame = _ws_read_frame(
                sock,
                max_payload=max(1, min(131072, max_bytes - bytes_read)),
            )
            if frame is None:
                break
            opcode, payload, frame_bytes = frame
            bytes_read += frame_bytes
            if opcode == 0x8:
                break
            if opcode != 0x1:
                continue
            payload_text = payload.decode("utf-8", errors="replace")
            rows, row_errors = _json_objects_from_text_payload(payload_text)
            parse_errors += row_errors
            for row in rows:
                events.append(row)
                if len(events) >= max_events:
                    break
    except (TimeoutError, OSError, ssl.SSLError) as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}:{exc}",
            "events": [],
            "parse_errors": parse_errors,
            "bytes_read": bytes_read,
        }
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    return {
        "ok": True,
        "error": "",
        "events": events,
        "parse_errors": parse_errors,
        "bytes_read": bytes_read,
    }


def _emsc_event_row(raw: dict[str, Any], *, received_at: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    detail_limit = _cfg_int(
        "EMSC_STREAM_DETAIL_CHAR_LIMIT", EMSC_STREAM_DETAIL_CHAR_LIMIT
    )
    payload: dict[str, Any] = raw
    data_value = raw.get("data")
    if isinstance(data_value, dict):
        payload = data_value
    elif isinstance(data_value, str):
        try:
            decoded = json.loads(data_value)
        except (ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, dict):
            payload = decoded

    properties = (
        payload.get("properties") if isinstance(payload.get("properties"), dict) else {}
    )
    if not isinstance(properties, dict):
        properties = {}
    geometry = (
        payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}
    )
    coordinates = geometry.get("coordinates", []) if isinstance(geometry, dict) else []
    if not isinstance(coordinates, list):
        coordinates = []

    lon = _safe_float(coordinates[0], 0.0) if len(coordinates) >= 1 else 0.0
    lat = _safe_float(coordinates[1], 0.0) if len(coordinates) >= 2 else 0.0
    depth_km = _safe_float(coordinates[2], 0.0) if len(coordinates) >= 3 else 0.0
    if depth_km <= 0.0:
        depth_km = _safe_float(properties.get("depth", 0.0), 0.0)

    magnitude = _safe_float(
        properties.get("mag", payload.get("mag", payload.get("magnitude", 0.0))),
        0.0,
    )
    region = str(
        properties.get("flynn_region")
        or properties.get("region")
        or payload.get("region")
        or payload.get("place")
        or ""
    ).strip()
    action = str(raw.get("action", "")).strip().lower()
    event_type = _world_log_kind_slug(
        str(raw.get("type") or payload.get("type") or "earthquake"),
        "earthquake",
    )
    ts = (
        str(
            properties.get("time")
            or properties.get("lastupdate")
            or payload.get("time")
            or raw.get("time")
            or received_at
        ).strip()
        or received_at
    )

    unid = str(
        properties.get("unid")
        or payload.get("unid")
        or payload.get("id")
        or raw.get("id")
        or ""
    ).strip()
    payload_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    seed = "id:" + unid if unid else "raw:" + payload_text
    event_id = "emsc:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]

    title = "earthquake event"
    if magnitude > 0.0:
        title = f"M {magnitude:.1f} earthquake"
    if region:
        title += " - " + region

    detail_parts = []
    if action:
        detail_parts.append("action=" + action)
    if magnitude > 0.0:
        detail_parts.append(f"mag={magnitude:.1f}")
    if region:
        detail_parts.append("region=" + region)
    if lat != 0.0 or lon != 0.0:
        detail_parts.append(f"lat={lat:.3f} lon={lon:.3f}")
    if depth_km > 0.0:
        detail_parts.append(f"depth_km={depth_km:.1f}")
    detail = _compact_world_text(" | ".join(detail_parts), detail_limit)

    refs = []
    if unid:
        refs.append(
            "https://www.seismicportal.eu/eventdetails.html?unid="
            + quote(unid, safe="")
        )

    tags = [
        token
        for token in [
            "emsc",
            "earthquake",
            event_type,
            action,
            "mag5plus" if magnitude >= 5.0 else "",
        ]
        if token
    ]

    return {
        "record": EMSC_EVENT_RECORD,
        "id": event_id,
        "ts": ts,
        "source": "emsc_stream",
        "kind": "emsc." + event_type,
        "status": "recorded",
        "title": title,
        "detail": detail,
        "refs": refs,
        "tags": tags,
        "meta": {
            "action": action,
            "unid": unid,
            "region": region,
            "magnitude": round(magnitude, 3),
        },
    }


def _emsc_stream_event_row(
    kind: str,
    detail: str,
    *,
    status: str = "recorded",
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    event_id = _world_log_event_id("emsc_stream", kind, now_iso, kind, detail)
    return {
        "record": EMSC_STREAM_EVENT_RECORD,
        "id": event_id,
        "ts": now_iso,
        "source": "emsc_stream",
        "kind": kind,
        "status": status,
        "title": kind,
        "detail": _compact_world_text(detail, 240),
        "refs": [],
        "tags": ["emsc", "earthquake"],
        "meta": {
            "stream_url": _cfg_str("EMSC_STREAM_URL", EMSC_STREAM_URL),
        },
    }


def _append_emsc_stream_rows(
    vault_root: Path, rows: list[dict[str, Any]]
) -> Path | None:
    if not rows:
        return None
    log_path = _emsc_stream_log_path(vault_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return log_path


def _load_emsc_stream_rows(
    vault_root: Path,
    *,
    limit: int,
) -> tuple[str, list[dict[str, Any]]]:
    log_path = _emsc_stream_log_path(vault_root)
    if not log_path.exists() or not log_path.is_file():
        return str(log_path), []
    bounded = max(1, int(limit or 1))
    selected: list[dict[str, Any]] = []
    for raw_line in log_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            selected.append(row)
    if len(selected) > bounded:
        selected = selected[-bounded:]
    return str(log_path), selected


def _collect_emsc_stream_rows(vault_root: Path) -> None:
    enabled = _cfg_bool("EMSC_STREAM_ENABLED", EMSC_STREAM_ENABLED)
    poll_interval = _cfg_float(
        "EMSC_STREAM_POLL_INTERVAL_SECONDS",
        EMSC_STREAM_POLL_INTERVAL_SECONDS,
    )
    now_monotonic = time.monotonic()

    with _EMSC_STREAM_LOCK:
        was_paused = bool(_EMSC_STREAM_CACHE.get("paused", False))
        if not enabled:
            _EMSC_STREAM_CACHE["paused"] = True
            if not was_paused:
                _append_emsc_stream_rows(
                    vault_root,
                    [
                        _emsc_stream_event_row(
                            "emsc.stream.paused",
                            "EMSC polling disabled by configuration",
                            status="paused",
                        )
                    ],
                )
            return

        if was_paused:
            _append_emsc_stream_rows(
                vault_root,
                [
                    _emsc_stream_event_row(
                        "emsc.stream.resumed",
                        "EMSC polling resumed by configuration",
                    )
                ],
            )
            _EMSC_STREAM_CACHE["paused"] = False

        last_poll = _safe_float(_EMSC_STREAM_CACHE.get("last_poll_monotonic", 0.0), 0.0)
        if now_monotonic - last_poll < max(1.0, poll_interval):
            return
        _EMSC_STREAM_CACHE["last_poll_monotonic"] = now_monotonic

    stream_url = _cfg_str("EMSC_STREAM_URL", EMSC_STREAM_URL).strip()
    if not stream_url:
        _append_emsc_stream_rows(
            vault_root,
            [
                _emsc_stream_event_row(
                    "emsc.stream.error",
                    "EMSC stream URL is empty",
                    status="error",
                )
            ],
        )
        return

    fetch_payload = _emsc_fetch_ws_events(
        stream_url,
        timeout_seconds=_cfg_float(
            "EMSC_STREAM_FETCH_TIMEOUT_SECONDS",
            EMSC_STREAM_FETCH_TIMEOUT_SECONDS,
        ),
        max_bytes=_cfg_int("EMSC_STREAM_MAX_BYTES", EMSC_STREAM_MAX_BYTES),
        max_events=_cfg_int(
            "EMSC_STREAM_MAX_EVENTS_PER_POLL",
            EMSC_STREAM_MAX_EVENTS_PER_POLL,
        ),
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    rows_to_append: list[dict[str, Any]] = []
    with _EMSC_STREAM_LOCK:
        if fetch_payload.get("ok") is not True:
            rows_to_append.append(
                _emsc_stream_event_row(
                    "emsc.stream.error",
                    str(fetch_payload.get("error", "fetch_failed")),
                    status="error",
                )
            )
            _EMSC_STREAM_CACHE["connected"] = False
            _append_emsc_stream_rows(vault_root, rows_to_append)
            return

        if not bool(_EMSC_STREAM_CACHE.get("connected", False)):
            rows_to_append.append(
                _emsc_stream_event_row(
                    "emsc.stream.connected",
                    "EMSC websocket fetch succeeded",
                )
            )
        _EMSC_STREAM_CACHE["connected"] = True

        seen_ids = _EMSC_STREAM_CACHE.get("seen_ids")
        if not isinstance(seen_ids, dict):
            seen_ids = {}
            _EMSC_STREAM_CACHE["seen_ids"] = seen_ids

        dedupe_ttl = _cfg_float(
            "EMSC_STREAM_DEDUPE_TTL_SECONDS",
            EMSC_STREAM_DEDUPE_TTL_SECONDS,
        )
        stale_keys = [
            key
            for key, seen_at in seen_ids.items()
            if now_monotonic - _safe_float(seen_at, 0.0) > max(15.0, dedupe_ttl)
        ]
        for key in stale_keys:
            seen_ids.pop(key, None)

        dedupe_dropped = 0
        rate_dropped = 0
        accepted = 0
        allowed = _cfg_int(
            "EMSC_STREAM_RATE_LIMIT_PER_POLL",
            EMSC_STREAM_RATE_LIMIT_PER_POLL,
        )

        for raw_event in fetch_payload.get("events", []):
            event_row = _emsc_event_row(raw_event, received_at=now_iso)
            if event_row is None:
                continue
            dedupe_key = str(event_row.get("id", "")).strip()
            if dedupe_key in seen_ids:
                dedupe_dropped += 1
                continue
            seen_ids[dedupe_key] = now_monotonic
            if accepted >= max(1, allowed):
                rate_dropped += 1
                continue
            rows_to_append.append(event_row)
            accepted += 1

        parse_errors = int(_safe_float(fetch_payload.get("parse_errors", 0), 0.0))
        status = "recorded"
        if parse_errors > 0 or dedupe_dropped > 0 or rate_dropped > 0:
            status = "degraded"
        rows_to_append.append(
            _emsc_stream_event_row(
                "emsc.stream.poll",
                "accepted="
                + str(accepted)
                + " parse_errors="
                + str(parse_errors)
                + " dedupe_dropped="
                + str(dedupe_dropped)
                + " rate_dropped="
                + str(rate_dropped)
                + " bytes="
                + str(int(_safe_float(fetch_payload.get("bytes_read", 0), 0.0))),
                status=status,
            )
        )
        if parse_errors > 0:
            rows_to_append.append(
                _emsc_stream_event_row(
                    "emsc.stream.parse-error",
                    f"parse_errors={parse_errors}",
                    status="degraded",
                )
            )
        if dedupe_dropped > 0:
            rows_to_append.append(
                _emsc_stream_event_row(
                    "emsc.stream.dedupe",
                    f"dropped={dedupe_dropped}",
                    status="degraded",
                )
            )
        if rate_dropped > 0:
            rows_to_append.append(
                _emsc_stream_event_row(
                    "emsc.stream.rate-limit",
                    f"dropped={rate_dropped}",
                    status="degraded",
                )
            )

    _append_emsc_stream_rows(vault_root, rows_to_append)


def _world_event_from_emsc_row(
    row: dict[str, Any],
    log_path: str,
) -> dict[str, Any] | None:
    return _world_event_from_stream_row(
        row,
        log_path,
        default_source="emsc_stream",
        default_kind="emsc.earthquake",
    )


def _wikimedia_stream_names() -> list[str]:
    raw = _cfg_str("WIKIMEDIA_EVENTSTREAMS_STREAMS", WIKIMEDIA_EVENTSTREAMS_STREAMS)
    names = [
        token
        for token in _split_csv_items(raw)
        if re.fullmatch(r"[A-Za-z0-9._-]+", token)
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _wikimedia_stream_url() -> str:
    names = _wikimedia_stream_names()
    if not names:
        return ""
    base = _cfg_str("WIKIMEDIA_EVENTSTREAMS_BASE_URL", WIKIMEDIA_EVENTSTREAMS_BASE_URL)
    return base.rstrip("/") + "/" + ",".join(names)


def _wikimedia_event_time(raw: dict[str, Any], fallback_ts: str) -> str:
    meta_value = raw.get("meta")
    meta: dict[str, Any] = dict(meta_value) if isinstance(meta_value, dict) else {}
    for key in ["dt", "timestamp", "time"]:
        value = str(meta.get(key, "")).strip()
        if value:
            return value
    for key in ["dt", "timestamp", "time"]:
        value = str(raw.get(key, "")).strip()
        if value:
            return value
    return fallback_ts


def _wikimedia_event_id(raw: dict[str, Any], payload_text: str) -> str:
    meta_value = raw.get("meta")
    meta: dict[str, Any] = dict(meta_value) if isinstance(meta_value, dict) else {}
    base_id = str(meta.get("id", "")).strip()
    if base_id:
        seed = "meta:" + base_id
    else:
        title = str(raw.get("title", "")).strip()
        wiki = str(raw.get("wiki", "")).strip()
        event_type = str(raw.get("type", "")).strip()
        seed = "raw:" + "|".join([wiki, event_type, title, payload_text])
    return "wikimedia:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]


def _wikimedia_public_url(raw: dict[str, Any]) -> str:
    server_url = str(raw.get("server_url", "")).strip().rstrip("/")
    title = str(raw.get("title", "")).strip().replace(" ", "_")
    if server_url and title:
        return server_url + "/wiki/" + title
    return ""


def _wikimedia_event_row(
    raw: dict[str, Any], *, received_at: str
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    detail_limit = _cfg_int(
        "WIKIMEDIA_EVENTSTREAMS_DETAIL_CHAR_LIMIT",
        WIKIMEDIA_EVENTSTREAMS_DETAIL_CHAR_LIMIT,
    )
    event_type = str(raw.get("type", "event")).strip().lower() or "event"
    stream_name = ""
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    if isinstance(meta, dict):
        stream_name = str(meta.get("stream", "")).strip()

    title = str(raw.get("title", "")).strip() or "(untitled)"
    comment = _compact_world_text(str(raw.get("comment", "")).strip(), detail_limit)
    wiki = str(raw.get("wiki", "")).strip()
    detail_parts = [token for token in [comment, wiki, stream_name] if token]
    detail = " | ".join(detail_parts)
    ts = _wikimedia_event_time(raw, received_at)
    payload_text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    event_id = _wikimedia_event_id(raw, payload_text)
    tags = [
        token
        for token in [
            wiki,
            event_type,
            stream_name,
            "bot" if bool(raw.get("bot")) else "",
            "minor" if bool(raw.get("minor")) else "",
        ]
        if token
    ]
    refs = [token for token in [_wikimedia_public_url(raw)] if token]
    return {
        "record": WIKIMEDIA_EVENT_RECORD,
        "id": event_id,
        "ts": ts,
        "source": "wikimedia_eventstreams",
        "kind": "wikimedia." + event_type,
        "status": "recorded",
        "title": title,
        "detail": detail,
        "refs": refs,
        "tags": tags,
        "meta": {
            "stream": stream_name,
            "wiki": wiki,
            "server_name": str(raw.get("server_name", "")).strip(),
            "meta_id": str(meta.get("id", "")).strip()
            if isinstance(meta, dict)
            else "",
        },
    }


def _wikimedia_stream_event_row(
    kind: str, detail: str, *, status: str = "recorded"
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    event_id = _world_log_event_id("wikimedia_stream", kind, now_iso, kind, detail)
    return {
        "record": WIKIMEDIA_STREAM_EVENT_RECORD,
        "id": event_id,
        "ts": now_iso,
        "source": "wikimedia_stream",
        "kind": kind,
        "status": status,
        "title": kind,
        "detail": _compact_world_text(detail, 240),
        "refs": [],
        "tags": ["wikimedia", "eventstreams"],
        "meta": {
            "stream_url": _wikimedia_stream_url(),
        },
    }


def _append_wikimedia_stream_rows(
    vault_root: Path, rows: list[dict[str, Any]]
) -> Path | None:
    if not rows:
        return None
    log_path = _wikimedia_stream_log_path(vault_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return log_path


def _load_wikimedia_stream_rows(
    vault_root: Path, *, limit: int
) -> tuple[str, list[dict[str, Any]]]:
    log_path = _wikimedia_stream_log_path(vault_root)
    if not log_path.exists() or not log_path.is_file():
        return str(log_path), []
    bounded = max(1, int(limit or 1))
    selected: list[dict[str, Any]] = []
    for raw_line in log_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            selected.append(row)
    if len(selected) > bounded:
        selected = selected[-bounded:]
    return str(log_path), selected


def _wikimedia_fetch_sse_events(
    stream_url: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_events: int,
) -> dict[str, Any]:
    req = Request(
        stream_url,
        headers={
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "User-Agent": "eta-mu-world-web/1.0",
        },
        method="GET",
    )
    events: list[dict[str, Any]] = []
    parse_errors = 0
    bytes_read = 0
    data_lines: list[str] = []
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    try:
        with urlopen(req, timeout=max(0.1, timeout_seconds)) as response:
            while len(events) < max_events and bytes_read < max_bytes:
                if time.monotonic() >= deadline:
                    break
                raw_line = response.readline(max(1, max_bytes - bytes_read))
                if not raw_line:
                    break
                bytes_read += len(raw_line)
                text_line = raw_line.decode("utf-8", errors="replace")
                stripped = text_line.strip()
                if not stripped:
                    if not data_lines:
                        continue
                    payload = "\n".join(data_lines)
                    data_lines = []
                    if not payload or payload.startswith(":"):
                        continue
                    try:
                        parsed = json.loads(payload)
                    except (ValueError, json.JSONDecodeError):
                        parse_errors += 1
                        continue
                    if isinstance(parsed, dict):
                        events.append(parsed)
                    else:
                        parse_errors += 1
                    continue
                if stripped.startswith(":"):
                    continue
                if stripped.startswith("data:"):
                    data_lines.append(stripped[5:].strip())
    except (URLError, TimeoutError, OSError) as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}:{exc}",
            "events": [],
            "parse_errors": parse_errors,
            "bytes_read": bytes_read,
        }
    return {
        "ok": True,
        "error": "",
        "events": events,
        "parse_errors": parse_errors,
        "bytes_read": bytes_read,
    }


def _collect_wikimedia_stream_rows(vault_root: Path) -> None:
    enabled = _cfg_bool(
        "WIKIMEDIA_EVENTSTREAMS_ENABLED", WIKIMEDIA_EVENTSTREAMS_ENABLED
    )
    poll_interval = _cfg_float(
        "WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS",
        WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS,
    )
    now_monotonic = time.monotonic()

    with _WIKIMEDIA_STREAM_LOCK:
        was_paused = bool(_WIKIMEDIA_STREAM_CACHE.get("paused", False))
        if not enabled:
            _WIKIMEDIA_STREAM_CACHE["paused"] = True
            if not was_paused:
                _append_wikimedia_stream_rows(
                    vault_root,
                    [
                        _wikimedia_stream_event_row(
                            "wikimedia.stream.paused",
                            "stream polling disabled by configuration",
                            status="paused",
                        )
                    ],
                )
            return

        if was_paused:
            _append_wikimedia_stream_rows(
                vault_root,
                [
                    _wikimedia_stream_event_row(
                        "wikimedia.stream.resumed",
                        "stream polling resumed by configuration",
                    )
                ],
            )
            _WIKIMEDIA_STREAM_CACHE["paused"] = False

        last_poll = _safe_float(
            _WIKIMEDIA_STREAM_CACHE.get("last_poll_monotonic", 0.0), 0.0
        )
        if now_monotonic - last_poll < max(1.0, poll_interval):
            return
        _WIKIMEDIA_STREAM_CACHE["last_poll_monotonic"] = now_monotonic

    stream_url = _wikimedia_stream_url()
    if not stream_url:
        _append_wikimedia_stream_rows(
            vault_root,
            [
                _wikimedia_stream_event_row(
                    "wikimedia.stream.error",
                    "no stream names configured",
                    status="error",
                )
            ],
        )
        return

    fetch_payload = _wikimedia_fetch_sse_events(
        stream_url,
        timeout_seconds=_cfg_float(
            "WIKIMEDIA_EVENTSTREAMS_FETCH_TIMEOUT_SECONDS",
            WIKIMEDIA_EVENTSTREAMS_FETCH_TIMEOUT_SECONDS,
        ),
        max_bytes=_cfg_int(
            "WIKIMEDIA_EVENTSTREAMS_MAX_BYTES", WIKIMEDIA_EVENTSTREAMS_MAX_BYTES
        ),
        max_events=_cfg_int(
            "WIKIMEDIA_EVENTSTREAMS_MAX_EVENTS_PER_POLL",
            WIKIMEDIA_EVENTSTREAMS_MAX_EVENTS_PER_POLL,
        ),
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    rows_to_append: list[dict[str, Any]] = []
    with _WIKIMEDIA_STREAM_LOCK:
        if fetch_payload.get("ok") is not True:
            rows_to_append.append(
                _wikimedia_stream_event_row(
                    "wikimedia.stream.error",
                    str(fetch_payload.get("error", "fetch_failed")),
                    status="error",
                )
            )
            _WIKIMEDIA_STREAM_CACHE["connected"] = False
            _append_wikimedia_stream_rows(vault_root, rows_to_append)
            return

        if not bool(_WIKIMEDIA_STREAM_CACHE.get("connected", False)):
            rows_to_append.append(
                _wikimedia_stream_event_row(
                    "wikimedia.stream.connected",
                    "eventstream fetch succeeded",
                )
            )
        _WIKIMEDIA_STREAM_CACHE["connected"] = True

        seen_ids = _WIKIMEDIA_STREAM_CACHE.get("seen_ids")
        if not isinstance(seen_ids, dict):
            seen_ids = {}
            _WIKIMEDIA_STREAM_CACHE["seen_ids"] = seen_ids
        dedupe_ttl = _cfg_float(
            "WIKIMEDIA_EVENTSTREAMS_DEDUPE_TTL_SECONDS",
            WIKIMEDIA_EVENTSTREAMS_DEDUPE_TTL_SECONDS,
        )
        stale_keys = [
            key
            for key, seen_at in seen_ids.items()
            if now_monotonic - _safe_float(seen_at, 0.0) > max(15.0, dedupe_ttl)
        ]
        for key in stale_keys:
            seen_ids.pop(key, None)

        dedupe_dropped = 0
        rate_dropped = 0
        accepted = 0
        allowed = _cfg_int(
            "WIKIMEDIA_EVENTSTREAMS_RATE_LIMIT_PER_POLL",
            WIKIMEDIA_EVENTSTREAMS_RATE_LIMIT_PER_POLL,
        )

        for raw_event in fetch_payload.get("events", []):
            event_row = _wikimedia_event_row(raw_event, received_at=now_iso)
            if event_row is None:
                continue
            dedupe_key = str(event_row.get("id", "")).strip()
            if dedupe_key in seen_ids:
                dedupe_dropped += 1
                continue
            seen_ids[dedupe_key] = now_monotonic
            if accepted >= max(1, allowed):
                rate_dropped += 1
                continue
            rows_to_append.append(event_row)
            accepted += 1

        parse_errors = int(_safe_float(fetch_payload.get("parse_errors", 0), 0.0))
        status = "recorded"
        if parse_errors > 0 or dedupe_dropped > 0 or rate_dropped > 0:
            status = "degraded"
        rows_to_append.append(
            _wikimedia_stream_event_row(
                "wikimedia.stream.poll",
                "accepted="
                + str(accepted)
                + " parse_errors="
                + str(parse_errors)
                + " dedupe_dropped="
                + str(dedupe_dropped)
                + " rate_dropped="
                + str(rate_dropped)
                + " bytes="
                + str(int(_safe_float(fetch_payload.get("bytes_read", 0), 0.0))),
                status=status,
            )
        )
        if parse_errors > 0:
            rows_to_append.append(
                _wikimedia_stream_event_row(
                    "wikimedia.stream.parse-error",
                    f"parse_errors={parse_errors}",
                    status="degraded",
                )
            )
        if dedupe_dropped > 0:
            rows_to_append.append(
                _wikimedia_stream_event_row(
                    "wikimedia.stream.dedupe",
                    f"dropped={dedupe_dropped}",
                    status="degraded",
                )
            )
        if rate_dropped > 0:
            rows_to_append.append(
                _wikimedia_stream_event_row(
                    "wikimedia.stream.rate-limit",
                    f"dropped={rate_dropped}",
                    status="degraded",
                )
            )

    _append_wikimedia_stream_rows(vault_root, rows_to_append)


def _world_event_from_wikimedia_row(
    row: dict[str, Any], log_path: str
) -> dict[str, Any] | None:
    event_id = str(row.get("id", "")).strip()
    if not event_id:
        return None
    return {
        "id": event_id,
        "ts": str(row.get("ts", "")).strip(),
        "source": str(row.get("source", "wikimedia_eventstreams")).strip()
        or "wikimedia_eventstreams",
        "kind": str(row.get("kind", "wikimedia.event")).strip() or "wikimedia.event",
        "status": str(row.get("status", "recorded")).strip() or "recorded",
        "title": str(row.get("title", "wikimedia")).strip() or "wikimedia",
        "detail": str(row.get("detail", "")).strip(),
        "refs": [str(item) for item in row.get("refs", []) if str(item).strip()],
        "tags": [str(item) for item in row.get("tags", []) if str(item).strip()],
        "path": log_path,
    }


def build_world_log_payload(
    part_root: Path,
    vault_root: Path,
    *,
    limit: int = WORLD_LOG_EVENT_LIMIT,
) -> dict[str, Any]:
    bounded_limit = max(12, min(800, int(limit or WORLD_LOG_EVENT_LIMIT)))
    relation_limit = max(1, min(8, int(WORLD_LOG_RELATION_LIMIT)))
    events: list[dict[str, Any]] = []

    def _append_stream_events(
        *,
        source: str,
        error_kind: str,
        fallback_log_path: Path,
        collect_fn: Any,
        load_fn: Any,
        mapper_fn: Any,
    ) -> None:
        try:
            collect_fn(vault_root)
            log_path, rows = load_fn(
                vault_root,
                limit=max(6, bounded_limit * 2),
            )
        except Exception as exc:
            events.append(
                _world_stream_error_event(
                    source=source,
                    kind=error_kind,
                    detail=f"{exc.__class__.__name__}:{exc}",
                    path=str(fallback_log_path),
                )
            )
            return

        for row in rows:
            if not isinstance(row, dict):
                continue
            event = mapper_fn(row, log_path)
            if event is not None:
                events.append(event)

    _append_stream_events(
        source="emsc_stream",
        error_kind="emsc.stream.error",
        fallback_log_path=_emsc_stream_log_path(vault_root),
        collect_fn=_collect_emsc_stream_rows,
        load_fn=_load_emsc_stream_rows,
        mapper_fn=_world_event_from_emsc_row,
    )

    _append_stream_events(
        source="nws_alerts",
        error_kind="nws.alerts.error",
        fallback_log_path=_nws_alerts_log_path(vault_root),
        collect_fn=_collect_nws_alert_rows,
        load_fn=_load_nws_alert_rows,
        mapper_fn=_world_event_from_nws_row,
    )

    _append_stream_events(
        source="swpc_alerts",
        error_kind="swpc.alerts.error",
        fallback_log_path=_swpc_alerts_log_path(vault_root),
        collect_fn=_collect_swpc_alert_rows,
        load_fn=_load_swpc_alert_rows,
        mapper_fn=_world_event_from_swpc_row,
    )

    _append_stream_events(
        source="nasa_gibs",
        error_kind="gibs.layers.error",
        fallback_log_path=_gibs_layers_log_path(vault_root),
        collect_fn=_collect_gibs_layer_rows,
        load_fn=_load_gibs_layer_rows,
        mapper_fn=_world_event_from_gibs_row,
    )

    _append_stream_events(
        source="nasa_eonet",
        error_kind="eonet.events.error",
        fallback_log_path=_eonet_events_log_path(vault_root),
        collect_fn=_collect_eonet_event_rows,
        load_fn=_load_eonet_event_rows,
        mapper_fn=_world_event_from_eonet_row,
    )

    _append_stream_events(
        source="wikimedia_stream",
        error_kind="wikimedia.stream.error",
        fallback_log_path=_wikimedia_stream_log_path(vault_root),
        collect_fn=_collect_wikimedia_stream_rows,
        load_fn=_load_wikimedia_stream_rows,
        mapper_fn=_world_event_from_wikimedia_row,
    )

    receipts_path = _locate_receipts_log(vault_root, part_root)
    if receipts_path and receipts_path.exists() and receipts_path.is_file():
        for raw_line in receipts_path.read_text("utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            row = _parse_receipt_line(line)
            ts = str(row.get("ts", "")).strip()
            source = "receipt"
            kind = str(row.get("kind", "decision")).strip().lower() or "decision"
            origin = str(row.get("origin", "runtime")).strip()
            owner = str(row.get("owner", "")).strip()
            title = f"{kind} / {origin}".strip()
            detail = str(row.get("dod", "")).strip() or str(row.get("note", "")).strip()
            refs = _split_receipt_refs(str(row.get("refs", "")))
            event_id = _world_log_event_id(source, kind, ts, title, detail)
            events.append(
                {
                    "id": event_id,
                    "ts": ts,
                    "source": source,
                    "kind": kind,
                    "status": "recorded",
                    "title": title,
                    "detail": detail,
                    "refs": refs,
                    "tags": [
                        token
                        for token in [origin, owner, str(row.get("pi", "")).strip()]
                        if token
                    ],
                    "path": str(receipts_path),
                }
            )

    registry_entries = _load_eta_mu_registry_entries(vault_root)
    for row in registry_entries[-bounded_limit:]:
        if not isinstance(row, dict):
            continue
        ts = str(row.get("time", "")).strip()
        source = "eta_mu_registry"
        event = str(row.get("event", "ingested")).strip().lower() or "ingested"
        status = str(row.get("status", "ok")).strip().lower() or "ok"
        rel_path = str(row.get("source_rel_path", "")).strip()
        kind = f"eta_mu.{event}"
        title = f"eta-mu {event}: {Path(rel_path).name if rel_path else '(unknown)'}"
        detail = str(row.get("reason", "")).strip() or rel_path
        refs = [
            token
            for token in [rel_path, str(row.get("knowledge_id", "")).strip()]
            if token
        ]
        event_id = _world_log_event_id(source, kind, ts, title, detail)
        events.append(
            {
                "id": event_id,
                "ts": ts,
                "source": source,
                "kind": kind,
                "status": status,
                "title": title,
                "detail": detail,
                "refs": refs,
                "tags": [
                    token
                    for token in [
                        str(row.get("source_kind", "")).strip(),
                        str(row.get("source_mime", "")).strip(),
                    ]
                    if token
                ],
                "path": str(row.get("source_rel_path", "")).strip(),
            }
        )

    inbox_root = _eta_mu_inbox_root(vault_root)
    if inbox_root.exists() and inbox_root.is_dir():
        substrate_root = _eta_mu_substrate_root(vault_root)
        for pending_path in _eta_mu_scan_candidates(inbox_root)[: bounded_limit // 3]:
            rel_path = _eta_mu_inbox_rel_path(pending_path, inbox_root)
            try:
                stat = pending_path.stat()
            except OSError:
                continue
            ts = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            kind = "eta_mu.pending"
            title = f"pending inbox file: {pending_path.name}"
            detail = f"awaiting ingest from .ημ ({rel_path})"
            refs = [_safe_rel_path((inbox_root / rel_path).resolve(), substrate_root)]
            event_id = _world_log_event_id("eta_mu_inbox", kind, ts, title, detail)
            events.append(
                {
                    "id": event_id,
                    "ts": ts,
                    "source": "eta_mu_inbox",
                    "kind": kind,
                    "status": "pending",
                    "title": title,
                    "detail": detail,
                    "refs": refs,
                    "tags": [pending_path.suffix.lower().lstrip(".") or "file"],
                    "path": rel_path,
                }
            )

    for row in _load_image_comment_entries(vault_root)[-bounded_limit // 2 :]:
        if not isinstance(row, dict):
            continue
        ts = str(row.get("created_at", row.get("time", ""))).strip()
        presence_id = str(row.get("presence_id", "")).strip()
        image_ref = str(row.get("image_ref", "")).strip()
        comment_text = str(row.get("comment", "")).strip()
        kind = "image.comment"
        title = f"image comment by {presence_id or 'presence'}"
        detail = comment_text
        refs = [token for token in [image_ref] if token]
        event_id = _world_log_event_id("image_comments", kind, ts, title, detail)
        events.append(
            {
                "id": event_id,
                "ts": ts,
                "source": "image_comments",
                "kind": kind,
                "status": "recorded",
                "title": title,
                "detail": detail,
                "refs": refs,
                "tags": [token for token in [presence_id] if token],
                "path": image_ref,
            }
        )

    for row in _load_presence_account_entries(vault_root)[-bounded_limit // 2 :]:
        if not isinstance(row, dict):
            continue
        ts = str(row.get("updated_at", row.get("time", ""))).strip()
        presence_id = str(row.get("presence_id", "")).strip()
        if not presence_id:
            continue
        kind = "presence.account"
        title = f"presence account: {presence_id}"
        detail = (
            str(row.get("bio", "")).strip() or str(row.get("display_name", "")).strip()
        )
        event_id = _world_log_event_id("presence_accounts", kind, ts, title, detail)
        events.append(
            {
                "id": event_id,
                "ts": ts,
                "source": "presence_accounts",
                "kind": kind,
                "status": "active",
                "title": title,
                "detail": detail,
                "refs": [],
                "tags": [
                    token
                    for token in [presence_id, str(row.get("handle", "")).strip()]
                    if token
                ],
                "path": "",
            }
        )

    for row in _load_study_snapshot_events(
        vault_root, limit=max(6, bounded_limit // 3)
    ):
        if not isinstance(row, dict):
            continue
        ts = str(row.get("ts", row.get("time", ""))).strip()
        label = str(row.get("label", "")).strip() or "study snapshot"
        event_id = _world_log_event_id("study", "study.snapshot", ts, label, "")
        refs = [str(item) for item in row.get("refs", []) if str(item).strip()]
        events.append(
            {
                "id": event_id,
                "ts": ts,
                "source": "study",
                "kind": "study.snapshot",
                "status": "recorded",
                "title": label,
                "detail": str(row.get("owner", "")).strip(),
                "refs": refs,
                "tags": [str(row.get("owner", "")).strip()],
                "path": str(row.get("path", "")).strip(),
            }
        )

    events.sort(
        key=lambda row: (
            _world_log_timestamp_value(str(row.get("ts", ""))),
            str(row.get("id", "")),
        ),
        reverse=True,
    )
    if len(events) > bounded_limit:
        events = events[:bounded_limit]

    dims = _world_log_vector_dims(vault_root)
    embedding_state: dict[str, dict[str, Any]] = {}
    vectors_by_event_id: dict[str, list[float]] = {}
    event_id_to_node_id: dict[str, str] = {}
    source_counts: dict[str, int] = defaultdict(int)
    kind_counts: dict[str, int] = defaultdict(int)

    for index, event in enumerate(events):
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue

        source = str(event.get("source", "runtime")).strip() or "runtime"
        kind = str(event.get("kind", "event")).strip() or "event"
        source_counts[source] += 1
        kind_counts[kind] += 1

        embedding_id = (
            "world-event:" + hashlib.sha1(event_id.encode("utf-8")).hexdigest()[:18]
        )
        event_text = _world_log_event_text(event)
        existing_row = embedding_state.get(embedding_id, {})
        vector = (
            existing_row.get("embedding", []) if isinstance(existing_row, dict) else []
        )
        if not isinstance(vector, list) or not vector:
            vector = _eta_mu_deterministic_vector(embedding_id + "|" + event_text, dims)
            should_persist = False
            with _WORLD_LOG_EMBEDDING_IDS_LOCK:
                if embedding_id not in _WORLD_LOG_EMBEDDING_IDS:
                    _WORLD_LOG_EMBEDDING_IDS.add(embedding_id)
                    should_persist = True
            if should_persist:
                try:
                    _embedding_db_upsert_append_only(
                        vault_root,
                        entry_id=embedding_id,
                        text=event_text,
                        embedding=vector,
                        metadata={
                            "entity_kind": "world_event",
                            "event_id": event_id,
                            "event_source": source,
                            "event_kind": kind,
                            "refs": list(event.get("refs", [])),
                        },
                        model="world-log:deterministic-v1",
                    )
                except Exception as exc:
                    status = str(event.get("status", "recorded")).strip() or "recorded"
                    if status == "recorded":
                        event["status"] = "degraded"
                    detail = str(event.get("detail", "")).strip()
                    error_detail = f"embedding_error={exc.__class__.__name__}:{exc}"
                    if error_detail not in detail:
                        merged = (
                            detail + " | " + error_detail if detail else error_detail
                        )
                        event["detail"] = _compact_world_text(merged, 240)
            embedding_state[embedding_id] = {
                "embedding": list(vector),
            }

        vectors_by_event_id[event_id] = list(vector)
        semantic_xy = _semantic_xy_from_embedding(list(vector))
        if semantic_xy is None:
            semantic_xy = (
                round(_stable_ratio(event_id, index * 7 + 1), 4),
                round(_stable_ratio(event_id, index * 7 + 3), 4),
            )

        refs = [str(item) for item in event.get("refs", []) if str(item).strip()]
        seed_rel = refs[0] if refs else str(event.get("path", "")).strip()
        field_scores = _infer_eta_mu_field_scores(
            rel_path=seed_rel,
            kind="text",
            text_excerpt=event_text,
        )
        dominant_field, dominant_weight = _dominant_eta_mu_field(field_scores)
        dominant_presence = FIELD_TO_PRESENCE.get(dominant_field, "anchor_registry")

        node_id = (
            "world-event-node:"
            + hashlib.sha1(event_id.encode("utf-8")).hexdigest()[:18]
        )
        event_id_to_node_id[event_id] = node_id
        event["embedding_id"] = embedding_id
        event["node_id"] = node_id
        event["x"] = round(_clamp01(_safe_float(semantic_xy[0], 0.5)), 4)
        event["y"] = round(_clamp01(_safe_float(semantic_xy[1], 0.5)), 4)
        event["dominant_field"] = dominant_field
        event["dominant_presence"] = dominant_presence
        event["dominant_weight"] = round(_clamp01(dominant_weight), 4)
        event["relations"] = []

    for event in events:
        source_id = str(event.get("id", "")).strip()
        source_vector = vectors_by_event_id.get(source_id)
        if source_vector is None:
            continue
        related: list[tuple[str, float]] = []
        for candidate in events:
            candidate_id = str(candidate.get("id", "")).strip()
            if not candidate_id or candidate_id == source_id:
                continue
            candidate_vector = vectors_by_event_id.get(candidate_id)
            if candidate_vector is None:
                continue
            similarity = _cosine_similarity(source_vector, candidate_vector)
            if similarity is None:
                continue
            if similarity < 0.2:
                continue
            related.append((candidate_id, similarity))
        related.sort(key=lambda row: row[1], reverse=True)
        event["relations"] = [
            {
                "event_id": candidate_id,
                "node_id": event_id_to_node_id.get(candidate_id, ""),
                "score": round(score, 4),
                "kind": "semantic-neighbor",
            }
            for candidate_id, score in related[:relation_limit]
        ]

    pending_count = sum(
        1 for event in events if str(event.get("kind", "")).strip() == "eta_mu.pending"
    )
    relation_count = sum(
        len(event.get("relations", []))
        for event in events
        if isinstance(event.get("relations", []), list)
    )

    return {
        "ok": True,
        "record": "ημ.world-log.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(events),
        "limit": bounded_limit,
        "pending_inbox": pending_count,
        "sources": dict(sorted(source_counts.items())),
        "kinds": dict(sorted(kind_counts.items())),
        "relation_count": relation_count,
        "events": events,
    }


def _find_truth_artifacts(vault_root: Path) -> list[str]:
    artifacts: list[str] = []
    for path in vault_root.rglob("Π*.zip"):
        if path.is_file():
            artifacts.append(str(path))
    artifacts.sort()
    return artifacts


def build_push_truth_dry_run_payload(
    part_root: Path, vault_root: Path
) -> dict[str, Any]:
    drift = build_drift_scan_payload(part_root, vault_root)
    artifact_paths = [Path(path) for path in _find_truth_artifacts(vault_root)]
    artifact_hashes: list[dict[str, str]] = []
    pi_zip_name_checks: list[dict[str, Any]] = []
    pi_zip_name_mismatches: list[dict[str, Any]] = []
    for path in artifact_paths:
        if not path.exists() or not path.is_file():
            continue
        sha256_hex = _sha256_for_path(path)
        artifact_hashes.append({"path": str(path), "sha256": sha256_hex})
        name_check = _pi_zip_name_check(path, sha256_hex)
        pi_zip_name_checks.append(name_check)
        if not bool(name_check.get("matches_sha12", False)):
            pi_zip_name_mismatches.append(name_check)

    manifest_candidates = [
        vault_root / "manifest.lith",
        part_root / "manifest.lith",
        Path.cwd() / "manifest.lith",
    ]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    manifest_sha = _sha256_for_path(manifest_path) if manifest_path else ""

    proof_schema = _extract_manifest_proof_schema(manifest_path)
    required_refs = [str(item) for item in proof_schema.get("required_refs", [])]
    required_hashes = [str(item) for item in proof_schema.get("required_hashes", [])]
    host_handle = str(proof_schema.get("host_handle", "")).strip()

    missing_required_refs = [
        ref
        for ref in required_refs
        if not _proof_ref_exists(ref, vault_root, part_root)
    ]
    available_hashes = {
        "sha256:pi_zip": [
            item.get("sha256") for item in artifact_hashes if item.get("sha256")
        ],
        "sha256:manifest": [manifest_sha] if manifest_sha else [],
    }
    missing_required_hashes = [
        key for key in required_hashes if not available_hashes.get(key)
    ]
    host_has_github_gist = bool(host_handle) and (
        "github:" in host_handle
        or "gist:" in host_handle
        or "gist.github.com" in host_handle
    )

    needs: list[str] = []
    advisories: list[str] = []
    if not artifact_paths:
        needs.append("truth artifact zip (artifacts/truth/Π*.zip)")
    if manifest_path is None:
        needs.append("manifest.lith")
    if drift.get("blocked_gates"):
        needs.append("resolve blocked push-truth gates")
    if not required_refs:
        needs.append("manifest proof-schema required_refs")
    if missing_required_refs:
        needs.append(
            "proof refs missing: " + ", ".join(sorted(set(missing_required_refs)))
        )
    if not required_hashes:
        needs.append("manifest proof-schema required_hashes")
    if missing_required_hashes:
        needs.append(
            "proof hashes missing: " + ", ".join(sorted(set(missing_required_hashes)))
        )
    if pi_zip_name_mismatches:
        mismatch_preview = ", ".join(
            f"{Path(str(item.get('path', ''))).name}->{str(item.get('expected_sha12', ''))}"
            for item in pi_zip_name_mismatches[:4]
        )
        advisories.append("Pi zip filename/hash mismatch: " + mismatch_preview)
    if not host_handle:
        needs.append("manifest proof-schema host-handle")

    blocked = bool(needs)
    reasons = [item.get("reason", "unknown") for item in drift.get("blocked_gates", [])]
    if missing_required_refs:
        reasons.append("missing-proof-refs")
    if missing_required_hashes:
        reasons.append("missing-proof-hashes")
    if not host_handle:
        reasons.append("missing-host-handle")

    return {
        "ok": True,
        "gate": {"target": "push-truth", "blocked": blocked, "reasons": reasons},
        "needs": needs,
        "advisories": advisories,
        "predicted_drifts": drift.get("active_drifts", []),
        "proof_schema": {
            "source": str(proof_schema.get("source", "")),
            "required_refs": required_refs,
            "required_hashes": required_hashes,
            "host_handle": host_handle,
            "missing_refs": missing_required_refs,
            "missing_hashes": missing_required_hashes,
        },
        "artifacts": {
            "pi_zip": [str(path) for path in artifact_paths],
            "pi_zip_hashes": artifact_hashes,
            "pi_zip_name_checks": pi_zip_name_checks,
            "pi_zip_name_mismatch_count": len(pi_zip_name_mismatches),
            "host_has_github_gist": host_has_github_gist,
            "host_handle": host_handle,
            "manifest": {
                "path": str(manifest_path) if manifest_path else "",
                "sha256": manifest_sha,
            },
        },
        "plan": [
            "Run /api/drift/scan and resolve blocked gates",
            "Generate or locate Π zip artifact",
            "Satisfy manifest proof schema refs/hashes/host-handle",
            "Bind manifest and append push-truth receipt",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _study_stability_label(score: float) -> str:
    if score >= 0.8:
        return "stable"
    if score >= 0.56:
        return "watch"
    return "unstable"


def build_study_snapshot(
    part_root: Path,
    vault_root: Path,
    *,
    queue_snapshot: dict[str, Any] | None = None,
    council_snapshot: dict[str, Any] | None = None,
    drift_payload: dict[str, Any] | None = None,
    truth_gate_blocked: bool | None = None,
    resource_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    queue = queue_snapshot if isinstance(queue_snapshot, dict) else {}
    council = council_snapshot if isinstance(council_snapshot, dict) else {}
    drift = drift_payload if isinstance(drift_payload, dict) else {}
    resource = (
        resource_snapshot
        if isinstance(resource_snapshot, dict)
        else _resource_monitor_snapshot(part_root=part_root)
    )
    if not isinstance(resource, dict):
        resource = {}

    blocked_gates = [
        item for item in drift.get("blocked_gates", []) if isinstance(item, dict)
    ]
    active_drifts = [
        item for item in drift.get("active_drifts", []) if isinstance(item, dict)
    ]
    open_questions = (
        drift.get("open_questions", {})
        if isinstance(drift.get("open_questions"), dict)
        else {}
    )
    receipts_parse = (
        drift.get("receipts_parse", {})
        if isinstance(drift.get("receipts_parse"), dict)
        else {}
    )

    blocked_gate_count = len(blocked_gates)
    active_drift_count = len(active_drifts)
    queue_pending = int(_safe_float(queue.get("pending_count", 0), 0.0))
    queue_events = int(_safe_float(queue.get("event_count", 0), 0.0))
    council_pending = int(_safe_float(council.get("pending_count", 0), 0.0))
    council_approved = int(_safe_float(council.get("approved_count", 0), 0.0))
    unresolved_questions = int(
        _safe_float(open_questions.get("unresolved_count", 0), 0.0)
    )
    resource_hot_count = len(
        [item for item in resource.get("hot_devices", []) if str(item).strip()]
    )
    resource_log_watch = (
        resource.get("log_watch", {})
        if isinstance(resource.get("log_watch"), dict)
        else {}
    )
    resource_log_error_ratio = _safe_float(
        resource_log_watch.get("error_ratio", 0.0), 0.0
    )

    truth_blocked = bool(truth_gate_blocked)
    if truth_gate_blocked is None:
        truth_blocked = blocked_gate_count > 0

    blocked_penalty = _clamp01(blocked_gate_count / 4.0) * 0.34
    drift_penalty = _clamp01(active_drift_count / 8.0) * 0.18
    queue_penalty = _clamp01(queue_pending / 8.0) * 0.2
    council_penalty = _clamp01(council_pending / 5.0) * 0.16
    truth_penalty = 0.12 if truth_blocked else 0.0
    resource_penalty = _clamp01(resource_hot_count / 3.0) * 0.08
    resource_log_penalty = _clamp01(resource_log_error_ratio) * 0.06
    score = _clamp01(
        1.0
        - blocked_penalty
        - drift_penalty
        - queue_penalty
        - council_penalty
        - truth_penalty
        - resource_penalty
        - resource_log_penalty
    )

    warnings: list[dict[str, Any]] = []
    for gate in blocked_gates[:8]:
        warnings.append(
            {
                "code": "gate.blocked",
                "severity": "high",
                "message": f"{gate.get('target', 'gate')}: {gate.get('reason', 'blocked')}",
            }
        )
    if unresolved_questions > 0:
        warnings.append(
            {
                "code": "drift.open_questions",
                "severity": "medium",
                "message": f"{unresolved_questions} open questions unresolved",
            }
        )
    if queue_pending > 0:
        warnings.append(
            {
                "code": "queue.pending",
                "severity": "medium",
                "message": f"{queue_pending} tasks pending in queue",
            }
        )
    if council_pending > 0:
        warnings.append(
            {
                "code": "council.pending",
                "severity": "medium",
                "message": f"{council_pending} council decisions awaiting closure",
            }
        )
    if resource_hot_count > 0:
        warnings.append(
            {
                "code": "runtime.resource_hot",
                "severity": "medium",
                "message": "hot resources: "
                + ", ".join(
                    [
                        str(item)
                        for item in resource.get("hot_devices", [])
                        if str(item).strip()
                    ]
                ),
            }
        )
    if resource_log_error_ratio >= 0.45:
        warnings.append(
            {
                "code": "runtime.log_error_ratio",
                "severity": "medium",
                "message": "runtime log error ratio elevated: "
                + str(round(resource_log_error_ratio, 3)),
            }
        )

    receipts_path = str(receipts_parse.get("path", "")).strip()
    receipts_within_vault = False
    if receipts_path:
        try:
            resolved = Path(receipts_path).resolve()
            vault_resolved = vault_root.resolve()
            receipts_within_vault = (
                resolved == vault_resolved or vault_resolved in resolved.parents
            )
        except OSError:
            receipts_within_vault = False

    if receipts_path and not receipts_within_vault:
        warnings.append(
            {
                "code": "runtime.receipts_path_outside_vault",
                "severity": "high",
                "message": f"receipts path not under vault root: {receipts_path}",
            }
        )

    decisions = [
        item for item in council.get("decisions", []) if isinstance(item, dict)
    ]
    decision_status_counts: dict[str, int] = {}
    for row in decisions:
        status = str(row.get("status", "unknown")).strip() or "unknown"
        decision_status_counts[status] = decision_status_counts.get(status, 0) + 1

    return {
        "ok": True,
        "record": "ημ.study-snapshot.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stability": {
            "score": round(score, 4),
            "label": _study_stability_label(score),
            "components": {
                "blocked_gate_penalty": round(blocked_penalty, 4),
                "drift_penalty": round(drift_penalty, 4),
                "queue_penalty": round(queue_penalty, 4),
                "council_penalty": round(council_penalty, 4),
                "truth_penalty": round(truth_penalty, 4),
                "resource_penalty": round(resource_penalty, 4),
                "resource_log_penalty": round(resource_log_penalty, 4),
            },
        },
        "signals": {
            "blocked_gate_count": blocked_gate_count,
            "active_drift_count": active_drift_count,
            "queue_pending_count": queue_pending,
            "queue_event_count": queue_events,
            "council_pending_count": council_pending,
            "council_approved_count": council_approved,
            "council_decision_count": int(
                _safe_float(council.get("decision_count", 0), 0.0)
            ),
            "decision_status_counts": decision_status_counts,
            "truth_gate_blocked": truth_blocked,
            "open_questions_unresolved": unresolved_questions,
            "resource_hot_count": resource_hot_count,
            "resource_log_error_ratio": round(resource_log_error_ratio, 4),
        },
        "runtime": {
            "part_root": str(part_root.resolve()),
            "vault_root": str(vault_root.resolve()),
            "receipts_path": receipts_path,
            "receipts_parse_ok": bool(receipts_parse.get("ok", False)),
            "receipts_rows": int(_safe_float(receipts_parse.get("rows", 0), 0.0)),
            "receipts_has_intent_ref": bool(
                receipts_parse.get("has_intent_ref", False)
            ),
            "receipts_path_within_vault": receipts_within_vault,
            "resource": resource,
        },
        "warnings": warnings,
        "drift": drift,
        "queue": queue,
        "council": council,
    }


def _study_snapshot_log_path(vault_root: Path) -> Path:
    return (vault_root / STUDY_SNAPSHOT_LOG_REL).resolve()


def _load_study_snapshot_events(
    vault_root: Path, *, limit: int = 16
) -> list[dict[str, Any]]:
    safe_limit = max(1, min(STUDY_SNAPSHOT_HISTORY_LIMIT, int(limit or 16)))
    log_path = _study_snapshot_log_path(vault_root)
    if not log_path.exists() or not log_path.is_file():
        return []

    try:
        stat = log_path.stat()
    except OSError:
        return []

    with _STUDY_SNAPSHOT_LOCK:
        events: list[dict[str, Any]] = []
        for raw in log_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                events.append(row)

        events.sort(key=lambda row: str(row.get("ts", "")), reverse=True)
        _STUDY_SNAPSHOT_CACHE["path"] = str(log_path)
        _STUDY_SNAPSHOT_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _STUDY_SNAPSHOT_CACHE["events"] = [dict(item) for item in events]
        return [dict(item) for item in events[:safe_limit]]


def _append_study_snapshot_event(vault_root: Path, event: dict[str, Any]) -> Path:
    log_path = _study_snapshot_log_path(vault_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with _STUDY_SNAPSHOT_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        _STUDY_SNAPSHOT_CACHE["path"] = ""
        _STUDY_SNAPSHOT_CACHE["mtime_ns"] = 0
        _STUDY_SNAPSHOT_CACHE["events"] = []
    return log_path


def export_study_snapshot(
    part_root: Path,
    vault_root: Path,
    *,
    queue_snapshot: dict[str, Any],
    council_snapshot: dict[str, Any],
    drift_payload: dict[str, Any],
    truth_gate_blocked: bool | None = None,
    resource_snapshot: dict[str, Any] | None = None,
    label: str = "",
    owner: str = "Err",
    refs: list[str] | None = None,
    host: str = "127.0.0.1:8787",
    manifest: str = "manifest.lith",
) -> dict[str, Any]:
    snapshot = build_study_snapshot(
        part_root,
        vault_root,
        queue_snapshot=queue_snapshot,
        council_snapshot=council_snapshot,
        drift_payload=drift_payload,
        truth_gate_blocked=truth_gate_blocked,
        resource_snapshot=resource_snapshot,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    digest_seed = "|".join(
        [
            now_iso,
            str(label).strip(),
            str((snapshot.get("stability", {}) or {}).get("score", "0")),
            str((snapshot.get("signals", {}) or {}).get("blocked_gate_count", "0")),
            str((snapshot.get("signals", {}) or {}).get("active_drift_count", "0")),
            str((snapshot.get("signals", {}) or {}).get("queue_pending_count", "0")),
            str((snapshot.get("signals", {}) or {}).get("council_pending_count", "0")),
        ]
    )
    event_id = f"study:{hashlib.sha1(digest_seed.encode('utf-8')).hexdigest()[:16]}"
    event = {
        "v": STUDY_EVENT_VERSION,
        "ts": now_iso,
        "op": "export",
        "id": event_id,
        "label": str(label).strip(),
        "owner": str(owner).strip() or "Err",
        "snapshot": snapshot,
    }
    log_path = _append_study_snapshot_event(vault_root, event)
    refs_rows = [
        str(log_path),
        "study:export",
        f"study:{event_id}",
        ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
        *([str(item).strip() for item in refs or [] if str(item).strip()]),
    ]
    ensure_receipts = _world_web_symbol(
        "_ensure_receipts_log_path", _ensure_receipts_log_path
    )
    receipts_path = ensure_receipts(vault_root, part_root)
    _append_receipt_line(
        receipts_path,
        kind=":decision",
        origin="study",
        owner=str(owner).strip() or "Err",
        dod="exported study snapshot evidence",
        pi="part64-runtime-system",
        host=host,
        manifest=manifest,
        refs=refs_rows,
        note=("" if not str(label).strip() else f"label={str(label).strip()}"),
    )
    latest = _load_study_snapshot_events(vault_root, limit=8)
    history_count = len(
        _load_study_snapshot_events(vault_root, limit=STUDY_SNAPSHOT_HISTORY_LIMIT)
    )
    return {
        "ok": True,
        "event": event,
        "history": {
            "record": "ημ.study-history.v1",
            "path": str(log_path),
            "count": history_count,
            "latest": latest,
        },
    }


def build_pi_archive_payload(
    part_root: Path,
    vault_root: Path,
    *,
    catalog: dict[str, Any] | None = None,
    queue_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    queue = queue_snapshot if isinstance(queue_snapshot, dict) else {}
    truth_state = (
        ((catalog or {}).get("truth_state") or {}) if isinstance(catalog, dict) else {}
    )
    snapshot = {
        "record": "ημ.study-snapshot.v1",
        "signals": {
            "queue_pending_count": int(_safe_float(queue.get("pending_count", 0), 0.0)),
            "queue_event_count": int(_safe_float(queue.get("event_count", 0), 0.0)),
            "truth_gate_blocked": bool(
                (
                    truth_state.get("gate", {}) if isinstance(truth_state, dict) else {}
                ).get("blocked", True)
            ),
        },
    }
    ledger = {
        "count": int(
            _safe_float(
                ((catalog or {}).get("promptdb") or {}).get("packet_count", 0), 0.0
            )
        ),
        "receipts": str((_locate_receipts_log(vault_root, part_root) or "")),
    }
    payload_catalog = {
        "counts": dict((catalog or {}).get("counts", {})),
        "item_count": len((catalog or {}).get("items", [])),
        "truth_state": dict((catalog or {}).get("truth_state", {})),
    }
    canonical = {
        "record": "ημ.pi-archive.v1",
        "snapshot": snapshot,
        "ledger": ledger,
        "catalog": payload_catalog,
    }
    canonical_json = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    canonical_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    signature_value = hashlib.sha256(
        f"pi-archive|{canonical_sha256}".encode("utf-8")
    ).hexdigest()
    archive = {
        "ok": True,
        "record": "ημ.pi-archive.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": snapshot,
        "ledger": ledger,
        "catalog": payload_catalog,
        "hash": {"canonical_sha256": canonical_sha256},
        "signature": {"value": signature_value},
    }
    portable = validate_pi_archive_portable(archive)
    archive["portable"] = portable
    archive["value"] = canonical_json
    return archive


def validate_pi_archive_portable(archive: dict[str, Any]) -> dict[str, Any]:
    required_sections = ["snapshot", "ledger", "catalog", "hash", "signature"]
    errors = [
        f"missing:{key}"
        for key in required_sections
        if not isinstance(archive.get(key), dict)
    ]
    portable = len(errors) == 0
    return {
        "ok": portable,
        "portable": portable,
        "valid": portable,
        "errors": errors,
        "required_sections": required_sections,
    }


def _truth_world_id() -> str:
    text = str(os.getenv("TRUTH_BINDING_WORLD_ID", "") or "").strip()
    if text:
        return text
    return "127.0.0.1:8787"


def _resolve_truth_ref(ref: str, vault_root: Path, part_root: Path) -> str:
    token = str(ref or "").strip()
    if not token:
        return ""
    if "://" in token or token.startswith(("runtime:", "artifact:", "gate:")):
        return token

    path_candidate = Path(token)
    if path_candidate.is_absolute():
        return str(path_candidate)

    vault_resolved = vault_root.resolve()
    for base in (vault_resolved, part_root.resolve(), Path.cwd().resolve()):
        resolved = (base / token).resolve()
        if not resolved.exists():
            continue
        try:
            return str(resolved.relative_to(vault_resolved)).replace("\\", "/")
        except ValueError:
            return str(resolved)
    return token


def _receipt_kind_counts(receipts_path: Path | None) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if (
        receipts_path is None
        or not receipts_path.exists()
        or not receipts_path.is_file()
    ):
        return {}
    for raw_line in receipts_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = _parse_receipt_line(line)
        kind = str(row.get("kind", "")).strip()
        if kind:
            counts[kind] += 1
    return dict(counts)


def _default_truth_state() -> dict[str, Any]:
    world_id = _truth_world_id()
    claim = {
        "id": "claim.push_truth_gate_ready",
        "text": "push-truth gate is ready for apply",
        "status": "undecided",
        "kappa": 0.0,
        "world": world_id,
        "proof_refs": [],
        "theta": TRUTH_BINDING_GUARD_THETA,
    }
    return {
        "record": ETA_MU_TRUTH_STATE_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name_binding": {
            "id": "gates_of_truth",
            "symbol": "Gates_of_Truth",
            "glyph": "真",
            "ascii": "TRUTH",
            "law": "Truth requires world scope (ω) + proof refs + receipts.",
        },
        "world": {
            "id": world_id,
            "ctx/ω-world": world_id,
            "ctx_omega_world": world_id,
        },
        "claim": claim,
        "claims": [claim],
        "guard": {
            "theta": TRUTH_BINDING_GUARD_THETA,
            "passes": False,
        },
        "gate": {
            "target": "push-truth",
            "blocked": True,
            "reasons": ["truth-state-unavailable"],
        },
        "invariants": {
            "world_scoped": bool(world_id),
            "proof_required": False,
            "proof_kind_subset": True,
            "receipts_parse_ok": False,
            "sim_bead_mint_blocked": True,
            "truth_binding_registered": False,
        },
        "proof": {
            "required_kinds": list(TRUTH_ALLOWED_PROOF_KINDS),
            "entries": [],
            "counts": {
                "total": 0,
                "present": 0,
                "by_kind": {kind: 0 for kind in TRUTH_ALLOWED_PROOF_KINDS},
            },
        },
        "artifacts": {
            "pi_zip_count": 0,
            "host_handle": "",
            "host_has_github_gist": False,
            "truth_receipt_count": 0,
        },
        "schema": {
            "source": "",
            "required_refs": [],
            "required_hashes": [],
            "host_handle": "",
            "missing_refs": [],
            "missing_hashes": [],
        },
        "needs": ["truth-state-unavailable"],
    }


def _build_truth_binding_state_uncached(
    part_root: Path,
    vault_root: Path,
    *,
    promptdb_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        dry_run = build_push_truth_dry_run_payload(part_root, vault_root)
    except Exception as exc:
        fallback = _default_truth_state()
        fallback["gate"] = {
            "target": "push-truth",
            "blocked": True,
            "reasons": [f"dry-run-error:{exc.__class__.__name__}"],
        }
        fallback["needs"] = ["push-truth dry-run payload unavailable"]
        return fallback

    promptdb = promptdb_index if isinstance(promptdb_index, dict) else {}
    if not promptdb:
        promptdb = collect_promptdb_packets(vault_root)

    proof_schema = dry_run.get("proof_schema", {})
    if not isinstance(proof_schema, dict):
        proof_schema = {}
    artifacts = dry_run.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    gate = dry_run.get("gate", {})
    if not isinstance(gate, dict):
        gate = {}

    world_id = _truth_world_id()
    required_refs = [
        str(item).strip()
        for item in proof_schema.get("required_refs", [])
        if str(item).strip()
    ]
    required_hashes = [
        str(item).strip()
        for item in proof_schema.get("required_hashes", [])
        if str(item).strip()
    ]
    missing_refs = [
        str(item).strip()
        for item in proof_schema.get("missing_refs", [])
        if str(item).strip()
    ]
    missing_hashes = [
        str(item).strip()
        for item in proof_schema.get("missing_hashes", [])
        if str(item).strip()
    ]
    pi_zip_paths = [
        str(item).strip() for item in artifacts.get("pi_zip", []) if str(item).strip()
    ]
    host_handle = str(
        artifacts.get("host_handle") or proof_schema.get("host_handle", "")
    ).strip()
    host_has_github_gist = bool(artifacts.get("host_has_github_gist", False))
    gate_blocked = bool(gate.get("blocked", True))
    gate_reasons = [
        str(item).strip() for item in gate.get("reasons", []) if str(item).strip()
    ]

    truth_refs = [
        ".opencode/protocol/truth.v1.lisp",
        ".opencode/promptdb/contracts/truth-layer.contract.lisp",
        ".opencode/promptdb/03_bind_truth.intent.lisp",
    ]
    truth_ref_presence = {
        ref: _proof_ref_exists(ref, vault_root, part_root) for ref in truth_refs
    }
    truth_binding_registered = all(truth_ref_presence.values())

    packet_paths = [
        str(item.get("path", ""))
        for item in promptdb.get("packets", [])
        if isinstance(item, dict)
    ]
    contract_paths = [
        str(item.get("path", ""))
        for item in promptdb.get("contracts", [])
        if isinstance(item, dict)
    ]
    truth_packet_indexed = any(
        path.endswith("03_bind_truth.intent.lisp") for path in packet_paths
    )
    truth_contract_indexed = any(
        path.endswith("truth-layer.contract.lisp") for path in contract_paths
    )

    receipts_path = _locate_receipts_log(vault_root, part_root)
    receipt_kinds = _receipt_kind_counts(receipts_path)

    proof_entries: list[dict[str, Any]] = []
    schema_source = str(proof_schema.get("source", "")).strip()
    if schema_source:
        proof_entries.append(
            {
                "kind": ":logic/bridge",
                "ref": _resolve_truth_ref(schema_source, vault_root, part_root),
                "present": True,
                "detail": "manifest proof-schema source",
            }
        )

    for ref in required_refs:
        present = _proof_ref_exists(ref, vault_root, part_root)
        proof_entries.append(
            {
                "kind": ":trace/record",
                "ref": _resolve_truth_ref(ref, vault_root, part_root),
                "present": bool(present),
                "detail": "required-proof-ref",
            }
        )

    for path in pi_zip_paths[:8]:
        proof_entries.append(
            {
                "kind": ":evidence/record",
                "ref": _resolve_truth_ref(path, vault_root, part_root),
                "present": True,
                "detail": "pi-zip-artifact",
            }
        )

    if host_handle:
        proof_entries.append(
            {
                "kind": ":trace/record",
                "ref": host_handle,
                "present": host_has_github_gist,
                "detail": "host-handle",
            }
        )

    proof_entries.append(
        {
            "kind": ":score/run",
            "ref": "runtime:/api/push-truth/dry-run",
            "present": True,
            "detail": "gate dry-run",
        }
    )
    proof_entries.append(
        {
            "kind": ":gov/adjudication",
            "ref": "gate:push-truth",
            "present": not gate_blocked,
            "detail": "gate-pass"
            if not gate_blocked
            else ",".join(gate_reasons) or "blocked",
        }
    )

    for ref in truth_refs:
        proof_entries.append(
            {
                "kind": ":trace/record",
                "ref": ref,
                "present": bool(truth_ref_presence.get(ref)),
                "detail": "truth-binding-artifact",
            }
        )

    if receipts_path is not None:
        proof_entries.append(
            {
                "kind": ":trace/record",
                "ref": _resolve_truth_ref(str(receipts_path), vault_root, part_root),
                "present": True,
                "detail": "receipts-log",
            }
        )

    proof_kind_counts: dict[str, int] = defaultdict(int)
    proof_present_count = 0
    for entry in proof_entries:
        kind = str(entry.get("kind", "")).strip()
        if kind:
            proof_kind_counts[kind] += 1
        if bool(entry.get("present")):
            proof_present_count += 1
    for kind in TRUTH_ALLOWED_PROOF_KINDS:
        proof_kind_counts.setdefault(kind, 0)

    proof_kind_subset = all(
        kind in TRUTH_ALLOWED_PROOF_KINDS for kind in proof_kind_counts.keys()
    )
    receipts_parse_ok = "receipts-parse-failed" not in gate_reasons

    score = 0.0
    score += 0.3 if not gate_blocked else 0.0
    score += 0.16 if receipts_parse_ok else 0.0
    score += 0.11 if schema_source else 0.0
    score += 0.1 if required_refs and not missing_refs else 0.0
    score += 0.08 if required_hashes and not missing_hashes else 0.0
    score += 0.08 if pi_zip_paths else 0.0
    score += 0.07 if host_has_github_gist else 0.0
    score += 0.1 if truth_binding_registered else 0.0
    score += 0.06 if truth_packet_indexed else 0.0
    score += 0.04 if truth_contract_indexed else 0.0
    if gate_blocked:
        score = min(score, 0.49)
    elif score < 0.55:
        score = min(1.0, score + 0.12)
    kappa = round(_clamp01(score), 4)

    if gate_blocked:
        claim_status = "refuted"
    elif proof_present_count <= 0 or not proof_kind_subset or not world_id:
        claim_status = "undecided"
    else:
        claim_status = "proved"

    artifact_claim_status = "proved" if truth_binding_registered else "refuted"
    artifact_claim_kappa = 0.91 if truth_binding_registered else 0.33

    primary_proof_refs = [
        str(entry.get("ref", ""))
        for entry in proof_entries
        if bool(entry.get("present")) and str(entry.get("ref", "")).strip()
    ][:8]
    claims = [
        {
            "id": "claim.push_truth_gate_ready",
            "text": "push-truth gate is ready for apply",
            "status": claim_status,
            "kappa": kappa,
            "world": world_id,
            "proof_refs": primary_proof_refs,
            "theta": TRUTH_BINDING_GUARD_THETA,
        },
        {
            "id": "claim.truth_binding_registered",
            "text": "truth protocol, contract, and intent are registered",
            "status": artifact_claim_status,
            "kappa": artifact_claim_kappa,
            "world": world_id,
            "proof_refs": truth_refs,
            "theta": TRUTH_BINDING_GUARD_THETA,
        },
    ]

    return {
        "record": ETA_MU_TRUTH_STATE_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name_binding": {
            "id": "gates_of_truth",
            "symbol": "Gates_of_Truth",
            "glyph": "真",
            "ascii": "TRUTH",
            "law": "Truth requires world scope (ω) + proof refs + receipts.",
        },
        "world": {
            "id": world_id,
            "ctx/ω-world": world_id,
            "ctx_omega_world": world_id,
        },
        "claim": claims[0],
        "claims": claims,
        "guard": {
            "theta": TRUTH_BINDING_GUARD_THETA,
            "passes": claims[0]["status"] == "proved"
            and _safe_float(claims[0].get("kappa", 0.0), 0.0)
            >= TRUTH_BINDING_GUARD_THETA,
        },
        "gate": {
            "target": str(gate.get("target", "push-truth") or "push-truth"),
            "blocked": gate_blocked,
            "reasons": gate_reasons,
        },
        "invariants": {
            "world_scoped": bool(world_id),
            "proof_required": proof_present_count > 0,
            "proof_kind_subset": proof_kind_subset,
            "receipts_parse_ok": receipts_parse_ok,
            "sim_bead_mint_blocked": True,
            "truth_binding_registered": truth_binding_registered,
            "promptdb_truth_packet_indexed": truth_packet_indexed,
            "promptdb_truth_contract_indexed": truth_contract_indexed,
        },
        "proof": {
            "required_kinds": list(TRUTH_ALLOWED_PROOF_KINDS),
            "entries": proof_entries,
            "counts": {
                "total": len(proof_entries),
                "present": proof_present_count,
                "by_kind": dict(proof_kind_counts),
            },
        },
        "artifacts": {
            "pi_zip_count": len(pi_zip_paths),
            "host_handle": host_handle,
            "host_has_github_gist": host_has_github_gist,
            "truth_receipt_count": int(
                _safe_float(receipt_kinds.get(":truth", 0), 0.0)
                + _safe_float(receipt_kinds.get(":refutation", 0), 0.0)
                + _safe_float(receipt_kinds.get(":adjudication", 0), 0.0)
            ),
            "decision_receipt_count": int(
                _safe_float(receipt_kinds.get(":decision", 0), 0.0)
            ),
        },
        "schema": {
            "source": schema_source,
            "required_refs": required_refs,
            "required_hashes": required_hashes,
            "host_handle": host_handle,
            "missing_refs": missing_refs,
            "missing_hashes": missing_hashes,
        },
        "needs": [str(item) for item in dry_run.get("needs", []) if str(item).strip()],
    }


def build_truth_binding_state(
    part_root: Path,
    vault_root: Path,
    *,
    promptdb_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    promptdb_packet_count = int(
        _safe_float(
            (promptdb_index or {}).get("packet_count", 0)
            if isinstance(promptdb_index, dict)
            else 0,
            0.0,
        )
    )
    promptdb_contract_count = int(
        _safe_float(
            (promptdb_index or {}).get("contract_count", 0)
            if isinstance(promptdb_index, dict)
            else 0,
            0.0,
        )
    )
    cache_key = (
        f"{part_root.resolve()}|{vault_root.resolve()}"
        f"|p{promptdb_packet_count}|c{promptdb_contract_count}"
    )
    now_monotonic = time.monotonic()

    with _TRUTH_BINDING_CACHE_LOCK:
        cached_key = str(_TRUTH_BINDING_CACHE.get("key", ""))
        cached_snapshot = _TRUTH_BINDING_CACHE.get("snapshot")
        elapsed = now_monotonic - float(
            _TRUTH_BINDING_CACHE.get("checked_monotonic", 0.0)
        )
        if (
            cached_snapshot is not None
            and cached_key == cache_key
            and elapsed < TRUTH_BINDING_CACHE_SECONDS
        ):
            return _json_deep_clone(cached_snapshot)

    snapshot = _build_truth_binding_state_uncached(
        part_root,
        vault_root,
        promptdb_index=promptdb_index,
    )
    with _TRUTH_BINDING_CACHE_LOCK:
        _TRUTH_BINDING_CACHE["key"] = cache_key
        _TRUTH_BINDING_CACHE["snapshot"] = _json_deep_clone(snapshot)
        _TRUTH_BINDING_CACHE["checked_monotonic"] = now_monotonic
    return snapshot
