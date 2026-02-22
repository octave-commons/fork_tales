from __future__ import annotations
import os
import time
import math
import hashlib
import threading
import colorsys
import base64
import struct
import json
import re
import socket
import sys
import io
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import defaultdict
from array import array
from hashlib import sha1
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen
from urllib.error import URLError

from .constants import (
    DAIMO_PROFILE_DEFS,
    DAIMO_FORCE_KAPPA,
    DAIMO_DAMPING,
    DAIMO_DT_SECONDS,
    DAIMO_MAX_TRACKED_ENTITIES,
    ENTITY_MANIFEST,
    USER_PRESENCE_ID,
    USER_PRESENCE_LABEL_EN,
    USER_PRESENCE_LABEL_JA,
    USER_PRESENCE_DEFAULT_X,
    USER_PRESENCE_DEFAULT_Y,
    USER_PRESENCE_DRIFT_ALPHA,
    USER_PRESENCE_EVENT_TTL_SECONDS,
    USER_PRESENCE_MAX_EVENTS,
    _DAIMO_DYNAMICS_LOCK,
    _DAIMO_DYNAMICS_CACHE,
    _USER_PRESENCE_INPUT_LOCK,
    _USER_PRESENCE_INPUT_CACHE,
    _MIX_CACHE_LOCK,
    _MIX_CACHE,
    CANONICAL_NAMED_FIELD_IDS,
    FIELD_TO_PRESENCE,
    MAX_SIM_POINTS,
    WS_MAGIC,
    WEAVER_HOST_ENV,
    WEAVER_PORT,
    WEAVER_GRAPH_HEALTH_TIMEOUT_SECONDS,
    WEAVER_GRAPH_NODE_LIMIT,
    WEAVER_GRAPH_EDGE_LIMIT,
    WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS,
    WEAVER_GRAPH_CACHE_SECONDS,
    ETA_MU_FIELD_KEYWORDS,
    ETA_MU_FILE_GRAPH_RECORD,
    ETA_MU_CRAWLER_GRAPH_RECORD,
    FILE_SENTINEL_PROFILE,
    FILE_ORGANIZER_PROFILE,
    HEALTH_SENTINEL_CPU_PROFILE,
    HEALTH_SENTINEL_GPU1_PROFILE,
    HEALTH_SENTINEL_GPU2_PROFILE,
    HEALTH_SENTINEL_NPU0_PROFILE,
    RESOURCE_CORE_PROFILE,  # New profile
    _WEAVER_GRAPH_CACHE_LOCK,
    _WEAVER_GRAPH_CACHE,
)
from .metrics import (
    _safe_float,
    _safe_int,
    _clamp01,
    _stable_ratio,
    _normalize_field_scores,
    _resource_monitor_snapshot,
    _INFLUENCE_TRACKER,
)
from .paths import _safe_rel_path, _eta_mu_substrate_root
from .db import (
    _normalize_embedding_vector,
    _load_embeddings_db_state,
    _get_chroma_collection,
    _cosine_similarity,
    _load_eta_mu_knowledge_entries,
)
from .nooi import NooiField
from .daimoi_probabilistic import (
    build_probabilistic_daimoi_particles,
    DAIMOI_JOB_KEYS,
    _simplex_noise_2d,
)
from .presence_runtime import (
    simulation_fingerprint,
    sync_presence_runtime_state,
    get_presence_runtime_manager,
)
from .sim_slice_bridge import resolve_sim_point_budget_slice
from .resource_economy import (
    sync_sub_sim_presences,
    process_resource_cycle,
)


SIMULATION_GROWTH_GUARD_RECORD = "eta-mu.simulation-growth-guard.v1"
SIMULATION_GROWTH_GUARD_SCHEMA_VERSION = "simulation.growth-guard.v1"
SIMULATION_GROWTH_EVENT_RECORD = "eta-mu.simulation-event.v1"
SIMULATION_GROWTH_EVENT_SCHEMA_VERSION = "simulation.events.v1"
SIMULATION_GROWTH_WATCH_THRESHOLD = 0.62
SIMULATION_GROWTH_CRITICAL_THRESHOLD = 0.82
SIMULATION_GROWTH_MAX_CLUSTER_NODES = 18
SIMULATION_FILE_GRAPH_PROJECTION_RECORD = "ημ.file-graph-projection.v1"
SIMULATION_FILE_GRAPH_PROJECTION_SCHEMA_VERSION = "file-graph.projection.v1"
SIMULATION_TRUTH_GRAPH_RECORD = "eta-mu.truth-graph.v1"
SIMULATION_TRUTH_GRAPH_SCHEMA_VERSION = "truth.graph.v1"
SIMULATION_VIEW_GRAPH_RECORD = "eta-mu.view-graph.v1"
SIMULATION_VIEW_GRAPH_SCHEMA_VERSION = "view.graph.v1"
SIMULATION_FILE_GRAPH_PROJECTION_EDGE_THRESHOLD = max(
    120,
    int(os.getenv("SIMULATION_FILE_GRAPH_PROJECTION_EDGE_THRESHOLD", "340") or "340"),
)
SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_MIN = max(
    120,
    int(os.getenv("SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_MIN", "220") or "220"),
)
SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_MAX = max(
    SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_MIN,
    int(os.getenv("SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_MAX", "860") or "860"),
)
SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_FACTOR = max(
    0.6,
    float(
        os.getenv("SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_FACTOR", "1.55") or "1.55"
    ),
)
SIMULATION_LAYOUT_CACHE_TTL_SECONDS = max(
    1.0,
    _safe_float(
        os.getenv("SIMULATION_LAYOUT_CACHE_TTL_SECONDS", "24.0") or "24.0",
        24.0,
    ),
)
SIMULATION_FILE_GRAPH_SUMMARY_CHARS = max(
    160,
    _safe_int(os.getenv("SIMULATION_FILE_GRAPH_SUMMARY_CHARS", "320") or "320", 320),
)
SIMULATION_FILE_GRAPH_EXCERPT_CHARS = max(
    120,
    _safe_int(os.getenv("SIMULATION_FILE_GRAPH_EXCERPT_CHARS", "280") or "280", 280),
)
SIMULATION_FILE_GRAPH_EMBED_LAYER_POINT_CAP = max(
    2,
    _safe_int(
        os.getenv("SIMULATION_FILE_GRAPH_EMBED_LAYER_POINT_CAP", "6") or "6",
        6,
    ),
)
SIMULATION_FILE_GRAPH_EMBED_IDS_CAP = max(
    1,
    _safe_int(os.getenv("SIMULATION_FILE_GRAPH_EMBED_IDS_CAP", "4") or "4", 4),
)
SIMULATION_FILE_GRAPH_EMBED_LINK_CAP = max(
    2,
    _safe_int(os.getenv("SIMULATION_FILE_GRAPH_EMBED_LINK_CAP", "8") or "8", 8),
)
SIMULATION_FILE_GRAPH_EDGE_RESPONSE_CAP = max(
    512,
    _safe_int(
        os.getenv("SIMULATION_FILE_GRAPH_EDGE_RESPONSE_CAP", "4096") or "4096",
        4096,
    ),
)
SIMULATION_FILE_GRAPH_EDGE_RESPONSE_FACTOR = max(
    1.0,
    _safe_float(
        os.getenv("SIMULATION_FILE_GRAPH_EDGE_RESPONSE_FACTOR", "2.0") or "2.0",
        2.0,
    ),
)


_SIMULATION_MINIMAL_PRESENCE_IDS: tuple[str, ...] = (
    "receipt_river",
    "witness_thread",
    "anchor_registry",
    "gates_of_truth",
    "health_sentinel_cpu",
)
_SIMULATION_RESOURCE_ALIASES: dict[str, str] = {
    "cpu": "cpu",
    "ram": "ram",
    "memory": "ram",
    "disk": "disk",
    "network": "network",
    "net": "network",
    "gpu": "gpu",
    "gpu1": "gpu",
    "gpu2": "gpu",
    "npu": "npu",
    "npu0": "npu",
}

SIMULATION_STREAM_FIELD_FORCE = max(
    0.0,
    _safe_float(os.getenv("SIMULATION_WS_STREAM_FIELD_FORCE", "0.22") or "0.22", 0.22),
)
SIMULATION_STREAM_VELOCITY_SCALE = max(
    0.0,
    _safe_float(
        os.getenv("SIMULATION_WS_STREAM_VELOCITY_SCALE", "0.75") or "0.75",
        0.75,
    ),
)
SIMULATION_STREAM_CENTER_GRAVITY = max(
    0.0,
    _safe_float(
        os.getenv("SIMULATION_WS_STREAM_CENTER_GRAVITY", "0.09") or "0.09",
        0.09,
    ),
)
SIMULATION_STREAM_JITTER_FORCE = max(
    0.0,
    _safe_float(
        os.getenv("SIMULATION_WS_STREAM_JITTER_FORCE", "0.085") or "0.085",
        0.085,
    ),
)
SIMULATION_STREAM_SIMPLEX_SCALE = max(
    0.0,
    _safe_float(
        os.getenv("SIMULATION_WS_STREAM_SIMPLEX_SCALE", "2.4") or "2.4",
        2.4,
    ),
)
SIMULATION_STREAM_FRICTION = max(
    0.8,
    min(
        0.9999,
        _safe_float(
            os.getenv("SIMULATION_WS_STREAM_FRICTION", "0.997") or "0.997",
            0.997,
        ),
    ),
)
SIMULATION_STREAM_MAX_SPEED = max(
    0.005,
    _safe_float(os.getenv("SIMULATION_WS_STREAM_MAX_SPEED", "0.095") or "0.095", 0.095),
)


def _csv_env_values(raw: str | None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for item in str(raw or "").split(","):
        token = str(item or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        values.append(token)
    return values


def _simulation_presence_profile() -> str:
    profile = str(os.getenv("SIMULATION_PRESENCE_PROFILE", "full") or "full")
    return profile.strip().lower() or "full"


def _simulation_presence_impact_order() -> list[str]:
    explicit = _csv_env_values(os.getenv("SIMULATION_PRESENCE_IDS", ""))
    if explicit:
        return explicit

    profile = _simulation_presence_profile()
    if profile in {"minimal", "concept_cpu", "concept-cpu", "light"}:
        return list(_SIMULATION_MINIMAL_PRESENCE_IDS)

    ordered: list[str] = [*CANONICAL_NAMED_FIELD_IDS]
    ordered.extend(
        [
            FILE_SENTINEL_PROFILE["id"],
            FILE_ORGANIZER_PROFILE["id"],
            HEALTH_SENTINEL_CPU_PROFILE["id"],
            HEALTH_SENTINEL_GPU1_PROFILE["id"],
            HEALTH_SENTINEL_GPU2_PROFILE["id"],
            HEALTH_SENTINEL_NPU0_PROFILE["id"],
        ]
    )

    deduped: list[str] = []
    seen: set[str] = set()
    for presence_id in ordered:
        token = str(presence_id or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _simulation_core_resource_emitters(
    *,
    cpu_utilization: float,
) -> tuple[list[str], bool, float]:
    explicit = _csv_env_values(os.getenv("SIMULATION_CORE_RESOURCES", ""))
    profile = _simulation_presence_profile()

    if explicit:
        requested = explicit
    elif profile in {"minimal", "concept_cpu", "concept-cpu", "light"}:
        requested = ["cpu"]
    else:
        requested = ["cpu", "ram", "disk", "network", "gpu", "npu"]

    resources: list[str] = []
    seen: set[str] = set()
    for token in requested:
        canonical = _SIMULATION_RESOURCE_ALIASES.get(str(token or "").strip().lower())
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        resources.append(canonical)

    cpu_daimoi_stop_percent = max(
        0.0,
        min(
            100.0,
            _safe_float(
                os.getenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "75") or "75",
                75.0,
            ),
        ),
    )
    cpu_core_emitter_enabled = (
        _safe_float(cpu_utilization, 0.0) < cpu_daimoi_stop_percent
    )
    if not cpu_core_emitter_enabled:
        resources = [resource for resource in resources if resource != "cpu"]

    return resources, cpu_core_emitter_enabled, cpu_daimoi_stop_percent


SIMULATION_FILE_GRAPH_NODE_FIELDS: tuple[str, ...] = (
    "id",
    "node_id",
    "node_type",
    "field",
    "tag",
    "label",
    "label_ja",
    "presence_kind",
    "name",
    "kind",
    "x",
    "y",
    "hue",
    "importance",
    "source_rel_path",
    "archived_rel_path",
    "archive_rel_path",
    "url",
    "dominant_field",
    "dominant_presence",
    "field_scores",
    "text_excerpt",
    "summary",
    "tags",
    "labels",
    "member_count",
    "embed_layer_points",
    "embed_layer_count",
    "vecstore_collection",
    "concept_presence_id",
    "concept_presence_label",
    "organized_by",
    "embedding_links",
)

SIMULATION_FILE_GRAPH_RENDER_NODE_FIELDS: tuple[str, ...] = (
    "id",
    "node_id",
    "node_type",
    "field",
    "tag",
    "label",
    "label_ja",
    "presence_kind",
    "name",
    "kind",
    "x",
    "y",
    "hue",
    "importance",
    "source_rel_path",
    "dominant_field",
    "dominant_presence",
    "embed_layer_count",
    "vecstore_collection",
    "concept_presence_id",
    "concept_presence_label",
    "organized_by",
    "resource_wallet",  # Exposed for debugging/visualization
)

_SIMULATION_LAYOUT_CACHE_LOCK = threading.Lock()
_SIMULATION_LAYOUT_CACHE: dict[str, Any] = {
    "key": "",
    "prepared_monotonic": 0.0,
    "prepared_graph": None,
    "embedding_points": [],
}
_NOOI_FIELD = NooiField()


def _world_web_symbol(name: str, default: Any) -> Any:
    module = sys.modules.get("code.world_web")
    if module is None:
        return default
    return getattr(module, name, default)


def _normalize_path_for_file_id(path_like: str) -> str:
    raw = str(path_like or "").strip().replace("\\", "/")
    if not raw:
        return ""
    parts: list[str] = []
    for token in raw.split("/"):
        piece = token.strip()
        if not piece or piece == ".":
            continue
        if piece == "..":
            if parts:
                parts.pop()
            continue
        parts.append(piece)
    return "/".join(parts)


def _file_id_for_path(path_like: str) -> str:
    norm = _normalize_path_for_file_id(path_like)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest() if norm else ""


def _file_node_usage_path(node: dict[str, Any]) -> str:
    return _normalize_path_for_file_id(
        str(
            node.get("source_rel_path")
            or node.get("archived_rel_path")
            or node.get("archive_rel_path")
            or node.get("name")
            or node.get("label")
            or ""
        )
    )


def _file_node_usage_score(
    node: dict[str, Any],
    *,
    recent_paths: set[str],
) -> tuple[float, bool, str]:
    usage_path = _file_node_usage_path(node)
    recent_hit = bool(usage_path and usage_path in recent_paths)
    importance = _clamp01(_safe_float(node.get("importance", 0.25), 0.25))
    layer_ratio = _clamp01(_safe_int(node.get("embed_layer_count", 0), 0) / 4.0)
    collection_bonus = 0.08 if str(node.get("vecstore_collection", "")).strip() else 0.0
    recent_bonus = 0.34 if recent_hit else 0.0
    score = _clamp01(
        (importance * 0.56) + (layer_ratio * 0.2) + collection_bonus + recent_bonus
    )
    return score, recent_hit, usage_path


def _growth_guard_pressure_native(
    *,
    file_count: int,
    edge_count: int,
    crawler_count: int,
    item_count: int,
    sim_point_budget: int,
    queue_pending_count: int,
    queue_event_count: int,
    cpu_utilization: float,
) -> dict[str, Any] | None:
    try:
        from .c_double_buffer_backend import compute_growth_guard_pressure_native

        return compute_growth_guard_pressure_native(
            file_count=file_count,
            edge_count=edge_count,
            crawler_count=crawler_count,
            item_count=item_count,
            sim_point_budget=sim_point_budget,
            queue_pending_count=queue_pending_count,
            queue_event_count=queue_event_count,
            cpu_utilization=cpu_utilization,
            weaver_graph_node_limit=_safe_float(WEAVER_GRAPH_NODE_LIMIT, 1.0),
            watch_threshold=SIMULATION_GROWTH_WATCH_THRESHOLD,
            critical_threshold=SIMULATION_GROWTH_CRITICAL_THRESHOLD,
        )
    except Exception:
        return None


def _growth_guard_scores_native(
    *,
    importance: list[float],
    layer_counts: list[int],
    has_collection: list[bool],
    recent_hit: list[bool],
) -> list[float] | None:
    try:
        from .c_double_buffer_backend import compute_growth_guard_scores_native

        return compute_growth_guard_scores_native(
            importance=importance,
            layer_counts=layer_counts,
            has_collection=has_collection,
            recent_hit=recent_hit,
        )
    except Exception:
        return None


def _graph_rows(
    file_graph: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = file_graph if isinstance(file_graph, dict) else {}
    node_rows = [
        row
        for row in (
            graph.get("nodes", []) if isinstance(graph.get("nodes", []), list) else []
        )
        if isinstance(row, dict)
    ]
    if not node_rows:
        node_rows = [
            *[
                row
                for row in (
                    graph.get("field_nodes", [])
                    if isinstance(graph.get("field_nodes", []), list)
                    else []
                )
                if isinstance(row, dict)
            ],
            *[
                row
                for row in (
                    graph.get("tag_nodes", [])
                    if isinstance(graph.get("tag_nodes", []), list)
                    else []
                )
                if isinstance(row, dict)
            ],
            *[
                row
                for row in (
                    graph.get("file_nodes", [])
                    if isinstance(graph.get("file_nodes", []), list)
                    else []
                )
                if isinstance(row, dict)
            ],
            *[
                row
                for row in (
                    graph.get("crawler_nodes", [])
                    if isinstance(graph.get("crawler_nodes", []), list)
                    else []
                )
                if isinstance(row, dict)
            ],
        ]
    edge_rows = [
        row
        for row in (
            graph.get("edges", []) if isinstance(graph.get("edges", []), list) else []
        )
        if isinstance(row, dict)
    ]
    return node_rows, edge_rows


def _graph_node_type_counts(node_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {
        "field": 0,
        "tag": 0,
        "file": 0,
        "crawler": 0,
        "other": 0,
    }
    for row in node_rows:
        node_type = str(row.get("node_type", "")).strip().lower()
        if node_type == "field":
            counts["field"] += 1
        elif node_type == "tag":
            counts["tag"] += 1
        elif node_type == "file":
            if str(row.get("url", "")).strip():
                counts["crawler"] += 1
            else:
                counts["file"] += 1
        elif node_type == "crawler":
            counts["crawler"] += 1
        else:
            counts["other"] += 1
    return counts


def _build_truth_graph_contract(file_graph: dict[str, Any] | None) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    node_rows, edge_rows = _graph_rows(file_graph)
    node_type_counts = _graph_node_type_counts(node_rows)
    node_ids = sorted(
        {
            str(row.get("id", "")).strip()
            for row in node_rows
            if str(row.get("id", "")).strip()
        }
    )
    edge_ids = sorted(
        {
            str(row.get("id", "")).strip()
            for row in edge_rows
            if str(row.get("id", "")).strip()
        }
    )
    node_digest_input = "\n".join(node_ids)
    edge_digest_input = "\n".join(edge_ids)

    return {
        "record": SIMULATION_TRUTH_GRAPH_RECORD,
        "schema_version": SIMULATION_TRUTH_GRAPH_SCHEMA_VERSION,
        "generated_at": now_iso,
        "node_count": int(len(node_rows)),
        "edge_count": int(len(edge_rows)),
        "node_type_counts": node_type_counts,
        "node_id_digest": sha1(node_digest_input.encode("utf-8")).hexdigest()
        if node_digest_input
        else "",
        "edge_id_digest": sha1(edge_digest_input.encode("utf-8")).hexdigest()
        if edge_digest_input
        else "",
        "provenance": {
            "source": "catalog.file_graph",
            "lossless": True,
        },
    }


def _build_view_graph_contract(file_graph: dict[str, Any] | None) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    node_rows, edge_rows = _graph_rows(file_graph)
    node_type_counts = _graph_node_type_counts(node_rows)
    projection = (
        file_graph.get("projection", {})
        if isinstance(file_graph, dict)
        and isinstance(file_graph.get("projection", {}), dict)
        else {}
    )
    groups = [
        row
        for row in (
            projection.get("groups", [])
            if isinstance(projection.get("groups", []), list)
            else []
        )
        if isinstance(row, dict)
    ]
    bundle_ledgers: list[dict[str, Any]] = []
    bundle_member_edges_total = 0
    reconstructable_bundle_count = 0
    surface_visible_count = 0
    for group in groups:
        member_edge_count = max(0, _safe_int(group.get("member_edge_count", 0), 0))
        member_edge_ids = group.get("member_edge_ids", [])
        has_member_ids = isinstance(member_edge_ids, list) and bool(member_edge_ids)
        if has_member_ids:
            reconstructable_bundle_count += 1
        if bool(group.get("surface_visible", False)):
            surface_visible_count += 1
        bundle_member_edges_total += member_edge_count
        bundle_ledgers.append(
            {
                "bundle_id": str(group.get("id", "")),
                "kind": str(group.get("kind", "")),
                "field": str(group.get("field", "")),
                "target": str(group.get("target", "")),
                "member_edge_count": int(member_edge_count),
                "member_source_count": max(
                    0, _safe_int(group.get("member_source_count", 0), 0)
                ),
                "member_target_count": max(
                    0, _safe_int(group.get("member_target_count", 0), 0)
                ),
                "member_edge_digest": str(group.get("member_edge_digest", "")),
                "surface_visible": bool(group.get("surface_visible", False)),
            }
        )

    return {
        "record": SIMULATION_VIEW_GRAPH_RECORD,
        "schema_version": SIMULATION_VIEW_GRAPH_SCHEMA_VERSION,
        "generated_at": now_iso,
        "node_count": int(len(node_rows)),
        "edge_count": int(len(edge_rows)),
        "node_type_counts": node_type_counts,
        "projection": {
            "mode": str(projection.get("mode", "none") or "none"),
            "active": bool(projection.get("active", False)),
            "reason": str(projection.get("reason", "") or ""),
            "bundle_ledger_count": int(len(bundle_ledgers)),
            "bundle_member_edge_count_total": int(bundle_member_edges_total),
            "reconstructable_bundle_count": int(reconstructable_bundle_count),
            "surface_visible_bundle_count": int(surface_visible_count),
            "bundle_ledgers": bundle_ledgers,
            "ledger_ref": "file_graph.projection.groups",
        },
        "projection_pi": {
            "kind": "edge-bundle" if bundle_ledgers else "identity",
            "bundle_count": int(len(bundle_ledgers)),
            "bundle_member_edge_count_total": int(bundle_member_edges_total),
            "reconstructable_bundle_count": int(reconstructable_bundle_count),
        },
    }


def _default_growth_guard(
    *,
    generated_at: str,
    sim_point_budget: int,
) -> dict[str, Any]:
    return {
        "record": SIMULATION_GROWTH_GUARD_RECORD,
        "schema_version": SIMULATION_GROWTH_GUARD_SCHEMA_VERSION,
        "generated_at": generated_at,
        "active": False,
        "mode": "normal",
        "thresholds": {
            "watch": round(SIMULATION_GROWTH_WATCH_THRESHOLD, 4),
            "critical": round(SIMULATION_GROWTH_CRITICAL_THRESHOLD, 4),
        },
        "pressure": {
            "blend": 0.0,
            "points": 0.0,
            "files": 0.0,
            "edges": 0.0,
            "crawler": 0.0,
            "queue": 0.0,
            "resource": 0.0,
        },
        "capacity": {
            "sim_point_budget": int(sim_point_budget),
            "target_file_nodes": 0,
            "target_edges": 0,
        },
        "demand": {
            "items": 0,
            "file_nodes": 0,
            "edges": 0,
            "crawler_nodes": 0,
        },
        "action": {
            "kind": "noop",
            "reason": "within_capacity",
            "collapsed_file_nodes": 0,
            "collapsed_edges": 0,
            "clusters": 0,
        },
        "daimoi": [],
        "events": [],
    }


def _apply_daimoi_growth_guard_to_file_graph(
    *,
    file_graph: dict[str, Any] | None,
    crawler_graph: dict[str, Any] | None,
    item_count: int,
    sim_point_budget: int,
    queue_snapshot: dict[str, Any],
    influence_snapshot: dict[str, Any],
    cpu_utilization: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    guard = _default_growth_guard(
        generated_at=now_iso, sim_point_budget=sim_point_budget
    )

    def _event(
        kind: str, status: str, reason: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "record": SIMULATION_GROWTH_EVENT_RECORD,
            "schema_version": SIMULATION_GROWTH_EVENT_SCHEMA_VERSION,
            "kind": kind,
            "status": status,
            "reason": reason,
            "ts": now_iso,
            "payload": payload,
        }

    if not isinstance(file_graph, dict):
        return file_graph, guard

    try:
        file_nodes_raw = file_graph.get("file_nodes", [])
        file_nodes = [row for row in file_nodes_raw if isinstance(row, dict)]
        edge_rows = [
            row for row in file_graph.get("edges", []) if isinstance(row, dict)
        ]
        crawler_nodes = (
            [
                row
                for row in crawler_graph.get("crawler_nodes", [])
                if isinstance(row, dict)
            ]
            if isinstance(crawler_graph, dict)
            else []
        )

        file_count = len(file_nodes)
        edge_count = len(edge_rows)
        crawler_count = len(crawler_nodes)
        queue_pending_count = int(queue_snapshot.get("pending_count", 0))
        queue_event_count = int(queue_snapshot.get("event_count", 0))
        native_pressure = _growth_guard_pressure_native(
            file_count=file_count,
            edge_count=edge_count,
            crawler_count=crawler_count,
            item_count=item_count,
            sim_point_budget=sim_point_budget,
            queue_pending_count=queue_pending_count,
            queue_event_count=queue_event_count,
            cpu_utilization=cpu_utilization,
        )

        if isinstance(native_pressure, dict):
            target_file_nodes = max(
                24,
                _safe_int(native_pressure.get("target_file_nodes", 96), 96),
            )
            target_edge_count = max(
                120,
                _safe_int(native_pressure.get("target_edge_count", 240), 240),
            )
            point_ratio = _clamp01(
                _safe_float(native_pressure.get("point_ratio", 0.0), 0.0)
            )
            file_ratio = _clamp01(
                _safe_float(native_pressure.get("file_ratio", 0.0), 0.0)
            )
            edge_ratio = _clamp01(
                _safe_float(native_pressure.get("edge_ratio", 0.0), 0.0)
            )
            crawler_ratio = _clamp01(
                _safe_float(native_pressure.get("crawler_ratio", 0.0), 0.0)
            )
            queue_ratio = _clamp01(
                _safe_float(native_pressure.get("queue_ratio", 0.0), 0.0)
            )
            resource_ratio = _clamp01(
                _safe_float(native_pressure.get("resource_ratio", 0.0), 0.0)
            )
            blend = _clamp01(_safe_float(native_pressure.get("blend", 0.0), 0.0))
            mode = str(native_pressure.get("mode", "normal") or "normal")
        else:
            target_file_nodes = max(
                96,
                min(
                    256,
                    int(max(96.0, _safe_float(sim_point_budget, 0.0) * 0.36)),
                ),
            )
            if cpu_utilization >= 88.0:
                target_file_nodes = max(72, int(target_file_nodes * 0.72))
            elif cpu_utilization >= 78.0:
                target_file_nodes = max(84, int(target_file_nodes * 0.84))
            target_edge_count = max(240, int(target_file_nodes * 3.0))

            queue_ratio = _clamp01(
                (queue_pending_count + (queue_event_count * 0.25)) / 16.0
            )
            resource_ratio = _clamp01(_safe_float(cpu_utilization, 0.0) / 100.0)
            point_ratio = _clamp01(
                max(0.0, _safe_float(item_count, 0.0))
                / max(1.0, _safe_float(sim_point_budget, 1.0))
            )
            file_ratio = _clamp01(
                max(0.0, _safe_float(file_count, 0.0))
                / max(1.0, _safe_float(target_file_nodes, 1.0))
            )
            edge_ratio = _clamp01(
                max(0.0, _safe_float(edge_count, 0.0))
                / max(1.0, _safe_float(target_edge_count, 1.0))
            )
            crawler_ratio = _clamp01(
                max(0.0, _safe_float(crawler_count, 0.0))
                / max(1.0, _safe_float(WEAVER_GRAPH_NODE_LIMIT, 1.0))
            )
            blend = _clamp01(
                (file_ratio * 0.48)
                + (edge_ratio * 0.24)
                + (point_ratio * 0.14)
                + (queue_ratio * 0.08)
                + (resource_ratio * 0.06)
            )

            mode = "normal"
            if blend >= SIMULATION_GROWTH_CRITICAL_THRESHOLD:
                mode = "critical"
            elif blend >= SIMULATION_GROWTH_WATCH_THRESHOLD:
                mode = "watch"

        guard["mode"] = mode
        guard["pressure"] = {
            "blend": round(blend, 4),
            "points": round(point_ratio, 4),
            "files": round(file_ratio, 4),
            "edges": round(edge_ratio, 4),
            "crawler": round(crawler_ratio, 4),
            "queue": round(queue_ratio, 4),
            "resource": round(resource_ratio, 4),
        }
        guard["capacity"] = {
            "sim_point_budget": int(sim_point_budget),
            "target_file_nodes": int(target_file_nodes),
            "target_edges": int(target_edge_count),
        }
        guard["demand"] = {
            "items": int(max(0, item_count)),
            "file_nodes": int(file_count),
            "edges": int(edge_count),
            "crawler_nodes": int(crawler_count),
        }

        should_attempt = (
            mode != "normal"
            and (file_count > target_file_nodes or edge_count > target_edge_count)
            and file_count > 0
        )
        if not should_attempt:
            if mode != "normal":
                guard["events"] = [
                    _event(
                        "daimoi.consolidation.skipped",
                        "ok",
                        "within_target_limits",
                        {
                            "mode": mode,
                            "file_nodes": file_count,
                            "edges": edge_count,
                            "target_file_nodes": target_file_nodes,
                            "target_edges": target_edge_count,
                        },
                    )
                ]
            return file_graph, guard

        recent_paths = {
            _normalize_path_for_file_id(str(path))
            for path in (
                influence_snapshot.get("recent_file_paths", [])
                if isinstance(influence_snapshot, dict)
                else []
            )
            if _normalize_path_for_file_id(str(path))
        }

        scored_node_ids: list[str] = []
        scored_nodes: list[dict[str, Any]] = []
        usage_paths: list[str] = []
        recent_hits: list[bool] = []
        importance_values: list[float] = []
        layer_counts: list[int] = []
        has_collection_flags: list[bool] = []
        for node in file_nodes:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            node_copy = dict(node)
            usage_path = _file_node_usage_path(node_copy)
            recent_hit = bool(usage_path and usage_path in recent_paths)
            scored_node_ids.append(node_id)
            scored_nodes.append(node_copy)
            usage_paths.append(usage_path)
            recent_hits.append(recent_hit)
            importance_values.append(
                _clamp01(_safe_float(node_copy.get("importance", 0.25), 0.25))
            )
            layer_counts.append(
                max(0, _safe_int(node_copy.get("embed_layer_count", 0), 0))
            )
            has_collection_flags.append(
                bool(str(node_copy.get("vecstore_collection", "")).strip())
            )

        native_scores = _growth_guard_scores_native(
            importance=importance_values,
            layer_counts=layer_counts,
            has_collection=has_collection_flags,
            recent_hit=recent_hits,
        )
        scored_entries: list[dict[str, Any]] = []
        for index, node_id in enumerate(scored_node_ids):
            node = scored_nodes[index]
            if isinstance(native_scores, list) and index < len(native_scores):
                usage_score = _clamp01(_safe_float(native_scores[index], 0.0))
                recent_hit = recent_hits[index]
                usage_path = usage_paths[index]
            else:
                usage_score, recent_hit, usage_path = _file_node_usage_score(
                    node,
                    recent_paths=recent_paths,
                )
            scored_entries.append(
                {
                    "id": node_id,
                    "node": node,
                    "usage_score": usage_score,
                    "recent_hit": recent_hit,
                    "usage_path": usage_path,
                }
            )
        if not scored_entries:
            guard["events"] = [
                _event(
                    "daimoi.consolidation.skipped",
                    "ok",
                    "no_file_nodes",
                    {
                        "mode": mode,
                    },
                )
            ]
            return file_graph, guard

        scored_entries.sort(
            key=lambda row: (
                -_safe_float(row.get("usage_score", 0.0), 0.0),
                str(row.get("id", "")),
            )
        )

        protected_ids = {
            str(row.get("id", "")).strip()
            for row in scored_entries
            if bool(row.get("recent_hit", False))
            or _safe_float(row.get("usage_score", 0.0), 0.0) >= 0.74
        }
        max_clusters = max(6, SIMULATION_GROWTH_MAX_CLUSTER_NODES)
        if mode == "critical":
            keep_target = max(48, int(target_file_nodes * 0.72) - max_clusters)
        else:
            keep_target = max(64, int(target_file_nodes * 0.86) - max_clusters)
        keep_target = min(len(scored_entries), max(24, keep_target))

        keep_ids = set(protected_ids)
        for row in scored_entries:
            candidate_id = str(row.get("id", "")).strip()
            if not candidate_id:
                continue
            if len(keep_ids) >= keep_target and candidate_id not in protected_ids:
                break
            keep_ids.add(candidate_id)

        low_entries = [
            row
            for row in scored_entries
            if str(row.get("id", "")).strip()
            and str(row.get("id", "")).strip() not in keep_ids
        ]
        if not low_entries:
            guard["events"] = [
                _event(
                    "daimoi.consolidation.skipped",
                    "ok",
                    "no_low_use_candidates",
                    {
                        "mode": mode,
                        "protected": len(protected_ids),
                        "keep_target": keep_target,
                    },
                )
            ]
            return file_graph, guard

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in low_entries:
            node = row.get("node", {}) if isinstance(row, dict) else {}
            field_id = (
                str(
                    (node if isinstance(node, dict) else {}).get("dominant_field", "f3")
                ).strip()
                or "f3"
            )
            node_kind = (
                str((node if isinstance(node, dict) else {}).get("kind", "file"))
                .strip()
                .lower()
                or "file"
            )
            grouped[f"{field_id}|{node_kind}"].append(row)

        grouped_rows = sorted(
            grouped.items(),
            key=lambda pair: (-len(pair[1]), pair[0]),
        )
        if len(grouped_rows) > max_clusters:
            overflow_rows: list[dict[str, Any]] = []
            for _, rows in grouped_rows[max_clusters - 1 :]:
                overflow_rows.extend(rows)
            grouped_rows = grouped_rows[: max_clusters - 1]
            if overflow_rows:
                grouped_rows.append(("f3|overflow", overflow_rows))

        cluster_nodes: list[dict[str, Any]] = []
        low_id_to_cluster: dict[str, str] = {}
        for cluster_index, (group_key, rows) in enumerate(grouped_rows):
            if not rows:
                continue
            field_id, _, node_kind = group_key.partition("|")
            member_ids = sorted(
                {
                    str(row.get("id", "")).strip()
                    for row in rows
                    if str(row.get("id", "")).strip()
                }
            )
            if not member_ids:
                continue
            seed = f"{group_key}|{len(member_ids)}|{'|'.join(member_ids[:18])}"
            cluster_id = f"file:cluster:{sha1(seed.encode('utf-8')).hexdigest()[:14]}"

            x_weighted = 0.0
            y_weighted = 0.0
            hue_weighted = 0.0
            weight_total = 0.0
            importance_sum = 0.0
            for row in rows:
                node = row.get("node", {}) if isinstance(row, dict) else {}
                node_usage = _safe_float(row.get("usage_score", 0.0), 0.0)
                node_importance = _clamp01(
                    _safe_float(
                        (node if isinstance(node, dict) else {}).get(
                            "importance", 0.25
                        ),
                        0.25,
                    )
                )
                node_weight = max(0.08, (node_usage * 0.4) + (node_importance * 0.6))
                x_weighted += (
                    _clamp01(
                        _safe_float(
                            (node if isinstance(node, dict) else {}).get("x", 0.5), 0.5
                        )
                    )
                    * node_weight
                )
                y_weighted += (
                    _clamp01(
                        _safe_float(
                            (node if isinstance(node, dict) else {}).get("y", 0.5), 0.5
                        )
                    )
                    * node_weight
                )
                hue_weighted += (
                    _safe_float(
                        (node if isinstance(node, dict) else {}).get("hue", 200),
                        200.0,
                    )
                    * node_weight
                )
                weight_total += node_weight
                importance_sum += node_importance
                low_id_to_cluster[str(row.get("id", "")).strip()] = cluster_id

            if weight_total <= 1e-8:
                weight_total = 1.0
            centroid_x = _clamp01(x_weighted / weight_total)
            centroid_y = _clamp01(y_weighted / weight_total)
            mean_importance = _clamp01(importance_sum / max(1, len(rows)))
            cluster_importance = _clamp01(0.18 + (mean_importance * 0.46))
            dominant_presence = (
                str(FIELD_TO_PRESENCE.get(field_id, "anchor_registry")).strip()
                or "anchor_registry"
            )
            cluster_label = f"Consolidated {field_id} {node_kind} ({len(member_ids)})"

            cluster_nodes.append(
                {
                    "id": cluster_id,
                    "node_id": cluster_id,
                    "node_type": "file",
                    "name": cluster_label,
                    "label": cluster_label,
                    "kind": "cluster",
                    "x": round(centroid_x, 4),
                    "y": round(centroid_y, 4),
                    "hue": int(round(hue_weighted / weight_total)) % 360,
                    "importance": round(cluster_importance, 4),
                    "source_rel_path": f"_consolidated/{field_id}/{node_kind}-{cluster_index + 1}",
                    "archive_kind": "cluster",
                    "dominant_field": field_id,
                    "dominant_presence": dominant_presence,
                    "field_scores": {field_id: 1.0},
                    "summary": (
                        "Daimoi consolidated low-use files to keep simulation load stable."
                    ),
                    "consolidated": True,
                    "consolidated_count": len(member_ids),
                    "consolidated_node_ids": member_ids[:32],
                }
            )

        keep_nodes = [
            dict(row.get("node", {}))
            for row in scored_entries
            if str(row.get("id", "")).strip() in keep_ids
            and isinstance(row.get("node", {}), dict)
        ]
        next_file_nodes = keep_nodes + cluster_nodes

        cluster_ids = {
            str(row.get("id", "")).strip()
            for row in cluster_nodes
            if str(row.get("id", "")).strip()
        }
        edge_buckets: dict[tuple[str, str, str, str], dict[str, float]] = {}
        for edge in edge_rows:
            source_id = str(edge.get("source", "")).strip()
            target_id = str(edge.get("target", "")).strip()
            if source_id in low_id_to_cluster:
                source_id = low_id_to_cluster[source_id]
            if target_id in low_id_to_cluster:
                target_id = low_id_to_cluster[target_id]
            if not source_id or not target_id or source_id == target_id:
                continue
            edge_kind = str(edge.get("kind", "relates")).strip().lower() or "relates"
            edge_field = str(edge.get("field", "")).strip()
            edge_weight = _clamp01(_safe_float(edge.get("weight", 0.42), 0.42))
            key = (source_id, target_id, edge_kind, edge_field)
            bucket = edge_buckets.setdefault(key, {"weight_sum": 0.0, "count": 0.0})
            bucket["weight_sum"] += edge_weight
            bucket["count"] += 1.0

        next_edges: list[dict[str, Any]] = []
        for (source_id, target_id, edge_kind, edge_field), bucket in sorted(
            edge_buckets.items(),
            key=lambda item: (
                -(
                    _safe_float(item[1].get("weight_sum", 0.0), 0.0)
                    / max(1.0, _safe_float(item[1].get("count", 1.0), 1.0))
                ),
                item[0][0],
                item[0][1],
                item[0][2],
            ),
        ):
            avg_weight = _clamp01(
                _safe_float(bucket.get("weight_sum", 0.0), 0.0)
                / max(1.0, _safe_float(bucket.get("count", 1.0), 1.0))
            )
            edge_seed = f"{source_id}|{target_id}|{edge_kind}|{edge_field}"
            next_edges.append(
                {
                    "id": "edge:" + sha1(edge_seed.encode("utf-8")).hexdigest()[:16],
                    "source": source_id,
                    "target": target_id,
                    "field": edge_field,
                    "weight": round(avg_weight, 4),
                    "kind": edge_kind,
                }
            )

        cluster_source_ids = {
            str(edge.get("source", "")).strip()
            for edge in next_edges
            if isinstance(edge, dict)
        }
        for node in cluster_nodes:
            cluster_id = str(node.get("id", "")).strip()
            if not cluster_id or cluster_id in cluster_source_ids:
                continue
            field_id = str(node.get("dominant_field", "f3")).strip() or "f3"
            target_presence = (
                str(
                    node.get(
                        "dominant_presence",
                        FIELD_TO_PRESENCE.get(field_id, "anchor_registry"),
                    )
                ).strip()
                or "anchor_registry"
            )
            target_node = f"field:{target_presence}"
            seed = f"{cluster_id}|{target_node}|categorizes"
            next_edges.append(
                {
                    "id": "edge:" + sha1(seed.encode("utf-8")).hexdigest()[:16],
                    "source": cluster_id,
                    "target": target_node,
                    "field": field_id,
                    "weight": 0.64,
                    "kind": "categorizes",
                }
            )

        edge_cap = max(target_edge_count, len(next_file_nodes) * 3)
        if len(next_edges) > edge_cap:
            next_edges = next_edges[:edge_cap]

        collapsed_file_nodes = max(0, len(file_nodes) - len(next_file_nodes))
        collapsed_edges = max(0, len(edge_rows) - len(next_edges))
        if collapsed_file_nodes <= 0 and collapsed_edges <= 0:
            guard["events"] = [
                _event(
                    "daimoi.consolidation.skipped",
                    "ok",
                    "no_reduction_needed",
                    {
                        "mode": mode,
                        "file_nodes": file_count,
                        "edges": edge_count,
                    },
                )
            ]
            return file_graph, guard

        graph_nodes_raw = file_graph.get("nodes", [])
        non_file_nodes = [
            dict(node)
            for node in (graph_nodes_raw if isinstance(graph_nodes_raw, list) else [])
            if isinstance(node, dict)
            and str(node.get("node_type", "")).strip().lower() != "file"
        ]

        updated_graph = dict(file_graph)
        updated_graph["file_nodes"] = next_file_nodes
        updated_graph["nodes"] = non_file_nodes + next_file_nodes
        updated_graph["edges"] = next_edges
        graph_stats = (
            dict(file_graph.get("stats", {}))
            if isinstance(file_graph.get("stats", {}), dict)
            else {}
        )
        graph_stats["file_count"] = len(next_file_nodes)
        graph_stats["edge_count"] = len(next_edges)
        graph_stats["consolidated_file_count"] = collapsed_file_nodes
        graph_stats["consolidated_cluster_count"] = len(cluster_nodes)
        graph_stats["consolidation_applied"] = True
        graph_stats["consolidation_mode"] = mode
        updated_graph["stats"] = graph_stats

        consolidation_info = {
            "record": SIMULATION_GROWTH_GUARD_RECORD,
            "schema_version": SIMULATION_GROWTH_GUARD_SCHEMA_VERSION,
            "generated_at": now_iso,
            "mode": mode,
            "collapsed_file_nodes": collapsed_file_nodes,
            "collapsed_edges": collapsed_edges,
            "clusters": len(cluster_nodes),
            "protected_recent_paths": len(recent_paths),
        }
        updated_graph["consolidation"] = consolidation_info

        deployment_row = {
            "id": "daimo:consolidator",
            "name": "Consolidator Daimoi",
            "state": "deployed",
            "mode": mode,
            "collapsed_file_nodes": collapsed_file_nodes,
            "collapsed_edges": collapsed_edges,
            "clusters": len(cluster_nodes),
            "at_iso": now_iso,
        }
        guard["active"] = True
        guard["action"] = {
            "kind": "daimoi.consolidation.deployed",
            "reason": "growth_pressure_high",
            "collapsed_file_nodes": collapsed_file_nodes,
            "collapsed_edges": collapsed_edges,
            "clusters": len(cluster_nodes),
        }
        guard["daimoi"] = [deployment_row]
        guard["events"] = [
            _event(
                "simulation.growth.pressure",
                "ok",
                "growth_pressure_high",
                {
                    "mode": mode,
                    "blend": round(blend, 4),
                    "file_ratio": round(file_ratio, 4),
                    "edge_ratio": round(edge_ratio, 4),
                },
            ),
            _event(
                "daimoi.consolidation.deployed",
                "ok",
                "growth_pressure_high",
                {
                    "mode": mode,
                    "collapsed_file_nodes": collapsed_file_nodes,
                    "collapsed_edges": collapsed_edges,
                    "clusters": len(cluster_nodes),
                    "protected_recent_paths": len(recent_paths),
                },
            ),
        ]
        return updated_graph, guard
    except Exception as exc:
        guard["mode"] = "watch"
        guard["events"] = [
            _event(
                "daimoi.consolidation.failed",
                "blocked",
                "fail_safe_noop",
                {
                    "error": exc.__class__.__name__,
                },
            )
        ]
        return file_graph, guard


def _project_file_graph_for_simulation(
    *,
    file_graph: dict[str, Any] | None,
    influence_snapshot: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(file_graph, dict):
        return file_graph, None

    now_iso = datetime.now(timezone.utc).isoformat()

    def _event(
        kind: str, status: str, reason: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "record": SIMULATION_GROWTH_EVENT_RECORD,
            "schema_version": SIMULATION_GROWTH_EVENT_SCHEMA_VERSION,
            "kind": kind,
            "status": status,
            "reason": reason,
            "ts": now_iso,
            "payload": payload,
        }

    try:
        file_nodes = [
            dict(row)
            for row in file_graph.get("file_nodes", [])
            if isinstance(row, dict)
        ]
        field_nodes = [
            dict(row)
            for row in file_graph.get("field_nodes", [])
            if isinstance(row, dict)
        ]
        tag_nodes = [
            dict(row)
            for row in file_graph.get("tag_nodes", [])
            if isinstance(row, dict)
        ]
        graph_nodes = [
            dict(row) for row in file_graph.get("nodes", []) if isinstance(row, dict)
        ]
        if not graph_nodes:
            graph_nodes = [*field_nodes, *tag_nodes, *file_nodes]

        node_by_id: dict[str, dict[str, Any]] = {}
        for node in graph_nodes:
            node_id = str(node.get("id", "")).strip()
            if node_id and node_id not in node_by_id:
                node_by_id[node_id] = node

        raw_edges = file_graph.get("edges", [])
        if not isinstance(raw_edges, list):
            raw_edges = []

        edge_rows: list[dict[str, Any]] = []
        for index, edge in enumerate(raw_edges):
            if not isinstance(edge, dict):
                continue
            source_id = str(edge.get("source", "")).strip()
            target_id = str(edge.get("target", "")).strip()
            if not source_id or not target_id or source_id == target_id:
                continue
            if source_id not in node_by_id or target_id not in node_by_id:
                continue
            edge_rows.append(
                {
                    "id": str(
                        edge.get(
                            "id",
                            "edge:"
                            + sha1(
                                f"{source_id}|{target_id}|{index}".encode("utf-8")
                            ).hexdigest()[:16],
                        )
                    ),
                    "source": source_id,
                    "target": target_id,
                    "field": str(edge.get("field", "")).strip(),
                    "kind": str(edge.get("kind", "relates")).strip().lower()
                    or "relates",
                    "weight": round(
                        _clamp01(_safe_float(edge.get("weight", 0.22), 0.22)), 4
                    ),
                }
            )

        edge_count_before = len(edge_rows)
        file_count_before = len(file_nodes)
        if edge_count_before <= 0:
            projection_payload = {
                "record": SIMULATION_FILE_GRAPH_PROJECTION_RECORD,
                "schema_version": SIMULATION_FILE_GRAPH_PROJECTION_SCHEMA_VERSION,
                "generated_at": now_iso,
                "mode": "hub-overflow",
                "active": False,
                "reason": "no_edges",
                "limits": {
                    "edge_threshold": int(
                        SIMULATION_FILE_GRAPH_PROJECTION_EDGE_THRESHOLD
                    ),
                },
                "before": {
                    "file_nodes": int(file_count_before),
                    "edges": int(edge_count_before),
                },
                "after": {
                    "file_nodes": int(file_count_before),
                    "edges": int(edge_count_before),
                },
                "collapsed_edges": 0,
                "overflow_nodes": 0,
                "overflow_edges": 0,
                "group_count": 0,
                "groups": [],
            }
            updated_graph = dict(file_graph)
            updated_graph["projection"] = projection_payload
            return updated_graph, None

        if edge_count_before < SIMULATION_FILE_GRAPH_PROJECTION_EDGE_THRESHOLD:
            projection_payload = {
                "record": SIMULATION_FILE_GRAPH_PROJECTION_RECORD,
                "schema_version": SIMULATION_FILE_GRAPH_PROJECTION_SCHEMA_VERSION,
                "generated_at": now_iso,
                "mode": "hub-overflow",
                "active": False,
                "reason": "below_threshold",
                "limits": {
                    "edge_threshold": int(
                        SIMULATION_FILE_GRAPH_PROJECTION_EDGE_THRESHOLD
                    ),
                },
                "before": {
                    "file_nodes": int(file_count_before),
                    "edges": int(edge_count_before),
                },
                "after": {
                    "file_nodes": int(file_count_before),
                    "edges": int(edge_count_before),
                },
                "collapsed_edges": 0,
                "overflow_nodes": 0,
                "overflow_edges": 0,
                "group_count": 0,
                "groups": [],
            }
            updated_graph = dict(file_graph)
            updated_graph["projection"] = projection_payload
            return updated_graph, None

        recent_paths = {
            _normalize_path_for_file_id(str(path))
            for path in (
                influence_snapshot.get("recent_file_paths", [])
                if isinstance(influence_snapshot, dict)
                else []
            )
            if _normalize_path_for_file_id(str(path))
        }

        kind_bonus = {
            "spawns_presence": 1.24,
            "organized_by_presence": 1.08,
            "categorizes": 0.94,
            "labeled_as": 0.66,
            "relates_tag": 0.42,
        }

        target_degree: dict[str, int] = defaultdict(int)
        for row in edge_rows:
            target_degree[str(row.get("target", ""))] += 1

        scored_edges: list[dict[str, Any]] = []
        for row in edge_rows:
            source_id = str(row.get("source", "")).strip()
            target_id = str(row.get("target", "")).strip()
            source_node = node_by_id.get(source_id, {})
            target_node = node_by_id.get(target_id, {})
            source_score, source_recent_hit, _ = _file_node_usage_score(
                source_node,
                recent_paths=recent_paths,
            )
            target_importance = _clamp01(
                _safe_float(target_node.get("importance", 0.24), 0.24)
            )
            weight = _clamp01(_safe_float(row.get("weight", 0.22), 0.22))
            kind = str(row.get("kind", "relates")).strip().lower() or "relates"
            hub_penalty = 1.0 / (
                1.0
                + (
                    max(0, target_degree.get(target_id, 1) - 1)
                    * (0.018 if kind == "categorizes" else 0.006)
                )
            )
            score = (
                (weight * 1.46)
                + float(kind_bonus.get(kind, 0.58))
                + (source_score * 0.64)
                + (target_importance * 0.28)
                + (0.22 if source_recent_hit else 0.0)
            ) * hub_penalty
            scored_edges.append(
                {
                    "row": row,
                    "score": round(score, 8),
                    "source_node": source_node,
                }
            )

        scored_edges.sort(
            key=lambda item: (
                -_safe_float(item.get("score", 0.0), 0.0),
                str(item.get("row", {}).get("kind", "")),
                str(item.get("row", {}).get("source", "")),
                str(item.get("row", {}).get("target", "")),
                str(item.get("row", {}).get("id", "")),
            )
        )

        file_count = max(1, file_count_before)
        global_edge_cap = max(
            SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_MIN,
            min(
                SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_MAX,
                int(
                    max(64.0, _safe_float(file_count, 64.0))
                    * SIMULATION_FILE_GRAPH_PROJECTION_EDGE_CAP_FACTOR
                ),
            ),
        )
        per_source_cap = 4
        per_source_categorizes_cap = 1
        per_field_hub_cap = max(26, min(92, int(file_count * 0.13)))
        per_concept_hub_cap = max(24, min(120, int(file_count * 0.16)))
        tag_member_cap = max(46, min(240, int(file_count * 0.5)))
        tag_pair_cap = max(18, min(110, int(file_count * 0.24)))
        overflow_group_limit = max(6, min(48, int(file_count * 0.14)))

        kept_edges: list[dict[str, Any]] = []
        source_counts: dict[str, int] = defaultdict(int)
        source_categorize_counts: dict[str, int] = defaultdict(int)
        field_target_counts: dict[str, int] = defaultdict(int)
        concept_target_counts: dict[str, int] = defaultdict(int)
        picked_tag_member_edges = 0
        picked_tag_pair_edges = 0

        grouped_dropped: dict[str, dict[str, Any]] = {}

        for scored in scored_edges:
            row = scored.get("row", {}) if isinstance(scored, dict) else {}
            if not isinstance(row, dict):
                continue
            source_id = str(row.get("source", "")).strip()
            target_id = str(row.get("target", "")).strip()
            kind = str(row.get("kind", "")).strip().lower() or "relates"
            field_id = str(row.get("field", "")).strip()
            source_node = (
                scored.get("source_node", {})
                if isinstance(scored.get("source_node", {}), dict)
                else {}
            )

            drop_reason = ""
            if len(kept_edges) >= global_edge_cap:
                drop_reason = "global_cap"
            elif source_counts[source_id] >= per_source_cap:
                drop_reason = "per_source_cap"
            elif kind == "categorizes":
                if source_categorize_counts[source_id] >= per_source_categorizes_cap:
                    drop_reason = "per_source_categorizes_cap"
                elif (
                    target_id.startswith("field:")
                    and field_target_counts[target_id] >= per_field_hub_cap
                ):
                    drop_reason = "field_hub_cap"
            elif (
                kind == "organized_by_presence"
                and target_id.startswith("presence:concept:")
                and concept_target_counts[target_id] >= per_concept_hub_cap
            ):
                drop_reason = "concept_hub_cap"
            elif kind == "labeled_as" and picked_tag_member_edges >= tag_member_cap:
                drop_reason = "tag_member_cap"
            elif kind == "relates_tag" and picked_tag_pair_edges >= tag_pair_cap:
                drop_reason = "tag_pair_cap"

            if not drop_reason:
                kept_edges.append(row)
                source_counts[source_id] += 1
                if kind == "categorizes":
                    source_categorize_counts[source_id] += 1
                    if target_id.startswith("field:"):
                        field_target_counts[target_id] += 1
                if kind == "organized_by_presence" and target_id.startswith(
                    "presence:concept:"
                ):
                    concept_target_counts[target_id] += 1
                if kind == "labeled_as":
                    picked_tag_member_edges += 1
                if kind == "relates_tag":
                    picked_tag_pair_edges += 1
                continue

            source_field = str(source_node.get("dominant_field", "f3")).strip() or "f3"
            if kind in {"categorizes", "organized_by_presence", "spawns_presence"}:
                bucket_key = f"{kind}|{target_id}|{field_id or source_field}"
            elif kind in {"labeled_as", "relates_tag"}:
                bucket_key = f"{kind}|{source_field}|{field_id or source_field}"
            else:
                bucket_key = f"{kind}|{target_id}|{field_id or source_field}"

            group = grouped_dropped.setdefault(
                bucket_key,
                {
                    "id": "projection-group:"
                    + sha1(bucket_key.encode("utf-8")).hexdigest()[:14],
                    "kind": kind,
                    "target": (
                        target_id
                        if kind
                        in {"categorizes", "organized_by_presence", "spawns_presence"}
                        else ""
                    ),
                    "field": field_id or source_field,
                    "member_edge_ids": [],
                    "member_source_ids": set(),
                    "member_target_ids": set(),
                    "weight_sum": 0.0,
                    "weight_count": 0,
                    "x_weighted": 0.0,
                    "y_weighted": 0.0,
                    "hue_weighted": 0.0,
                    "weight_total": 0.0,
                    "reason_counts": defaultdict(int),
                },
            )
            group["member_edge_ids"].append(str(row.get("id", "")))
            group["member_source_ids"].add(source_id)
            group["member_target_ids"].add(target_id)
            group["weight_sum"] += _safe_float(row.get("weight", 0.0), 0.0)
            group["weight_count"] += 1
            group["reason_counts"][drop_reason] += 1

            source_weight = max(
                0.08,
                _clamp01(_safe_float(source_node.get("importance", 0.24), 0.24)),
            )
            source_x = _clamp01(_safe_float(source_node.get("x", 0.5), 0.5))
            source_y = _clamp01(_safe_float(source_node.get("y", 0.5), 0.5))
            source_hue = _safe_float(source_node.get("hue", 200), 200.0)
            group["x_weighted"] += source_x * source_weight
            group["y_weighted"] += source_y * source_weight
            group["hue_weighted"] += source_hue * source_weight
            group["weight_total"] += source_weight

        collapsed_edges = max(0, edge_count_before - len(kept_edges))
        if collapsed_edges <= 0:
            projection_payload = {
                "record": SIMULATION_FILE_GRAPH_PROJECTION_RECORD,
                "schema_version": SIMULATION_FILE_GRAPH_PROJECTION_SCHEMA_VERSION,
                "generated_at": now_iso,
                "mode": "hub-overflow",
                "active": False,
                "reason": "within_projection_limits",
                "limits": {
                    "edge_threshold": int(
                        SIMULATION_FILE_GRAPH_PROJECTION_EDGE_THRESHOLD
                    ),
                    "edge_cap": int(global_edge_cap),
                },
                "before": {
                    "file_nodes": int(file_count_before),
                    "edges": int(edge_count_before),
                },
                "after": {
                    "file_nodes": int(file_count_before),
                    "edges": int(edge_count_before),
                },
                "collapsed_edges": 0,
                "overflow_nodes": 0,
                "overflow_edges": 0,
                "group_count": 0,
                "groups": [],
            }
            updated_graph = dict(file_graph)
            updated_graph["projection"] = projection_payload
            return updated_graph, None

        finalized_groups: list[dict[str, Any]] = []
        grouped_rows = sorted(
            grouped_dropped.values(),
            key=lambda item: (
                -len(item.get("member_edge_ids", [])),
                str(item.get("id", "")),
            ),
        )
        for group in grouped_rows:
            member_edge_ids = sorted(
                {
                    str(edge_id).strip()
                    for edge_id in group.get("member_edge_ids", [])
                    if str(edge_id).strip()
                }
            )
            member_source_ids = sorted(
                {
                    str(node_id).strip()
                    for node_id in group.get("member_source_ids", set())
                    if str(node_id).strip()
                }
            )
            member_target_ids = sorted(
                {
                    str(node_id).strip()
                    for node_id in group.get("member_target_ids", set())
                    if str(node_id).strip()
                }
            )
            if not member_edge_ids:
                continue
            digest_input = "\n".join(member_edge_ids)
            reason_counts = group.get("reason_counts", {})
            reason_rows = {
                str(key): int(value)
                for key, value in sorted(
                    (
                        (str(k), int(v))
                        for k, v in reason_counts.items()
                        if str(k).strip()
                    ),
                    key=lambda row: row[0],
                )
            }
            finalized_groups.append(
                {
                    "id": str(group.get("id", "")),
                    "kind": str(group.get("kind", "relates")),
                    "target": str(group.get("target", "")),
                    "field": str(group.get("field", "")),
                    "member_edge_count": len(member_edge_ids),
                    "member_source_count": len(member_source_ids),
                    "member_target_count": len(member_target_ids),
                    "member_edge_ids": member_edge_ids,
                    "member_source_ids": member_source_ids,
                    "member_target_ids": member_target_ids,
                    "member_edge_digest": sha1(
                        digest_input.encode("utf-8")
                    ).hexdigest(),
                    "reasons": reason_rows,
                    "weight_sum": _safe_float(group.get("weight_sum", 0.0), 0.0),
                    "weight_count": max(1, _safe_int(group.get("weight_count", 1), 1)),
                    "x_weighted": _safe_float(group.get("x_weighted", 0.0), 0.0),
                    "y_weighted": _safe_float(group.get("y_weighted", 0.0), 0.0),
                    "hue_weighted": _safe_float(group.get("hue_weighted", 0.0), 0.0),
                    "weight_total": max(
                        1e-6,
                        _safe_float(group.get("weight_total", 0.0), 0.0),
                    ),
                }
            )

        visual_groups = [
            group
            for group in finalized_groups
            if (
                str(group.get("kind", "")) == "categorizes"
                and str(group.get("target", "")).startswith("field:")
            )
            or (
                str(group.get("kind", "")) == "organized_by_presence"
                and str(group.get("target", "")).startswith("presence:concept:")
            )
        ]
        visual_groups.sort(
            key=lambda item: (
                -int(item.get("member_edge_count", 0)),
                str(item.get("id", "")),
            )
        )
        visual_groups = visual_groups[:overflow_group_limit]
        visual_group_ids = {
            str(group.get("id", "")).strip()
            for group in visual_groups
            if str(group.get("id", "")).strip()
        }

        overflow_nodes: list[dict[str, Any]] = []
        overflow_edges: list[dict[str, Any]] = []
        for index, group in enumerate(visual_groups):
            group_id = str(group.get("id", "")).strip()
            target_id = str(group.get("target", "")).strip()
            if not group_id or not target_id:
                continue
            target_node = node_by_id.get(target_id)
            if not isinstance(target_node, dict):
                continue

            group_weight_total = max(
                1e-6, _safe_float(group.get("weight_total", 0.0), 0.0)
            )
            centroid_x = _clamp01(
                _safe_float(group.get("x_weighted", 0.0), 0.0) / group_weight_total
            )
            centroid_y = _clamp01(
                _safe_float(group.get("y_weighted", 0.0), 0.0) / group_weight_total
            )
            hue_fallback = _safe_float(target_node.get("hue", 200), 200.0)
            hue_weighted = _safe_float(
                group.get("hue_weighted", hue_fallback), hue_fallback
            )
            hue = (
                int(
                    round(
                        hue_weighted / group_weight_total
                        if group_weight_total > 1e-6
                        else hue_fallback
                    )
                )
                % 360
            )

            field_id = str(group.get("field", "")).strip()
            if not field_id:
                field_id = str(target_node.get("field", "")).strip() or "f3"

            dominant_presence = ""
            if target_id.startswith("field:"):
                dominant_presence = target_id.split("field:", 1)[1]
            elif target_id.startswith("presence:"):
                dominant_presence = target_id.split("presence:", 1)[1]
            if not dominant_presence:
                dominant_presence = (
                    str(target_node.get("dominant_presence", "")).strip()
                    or str(target_node.get("node_id", "")).strip()
                    or "anchor_registry"
                )

            target_label = (
                str(target_node.get("label", "")).strip()
                or str(target_node.get("name", "")).strip()
                or target_id
            )
            kind = str(group.get("kind", "categorizes")).strip() or "categorizes"
            overflow_label = f"Overflow {kind} -> {target_label}"
            overflow_node_id = (
                "file:projection:"
                + sha1(f"{group_id}|node|{index}".encode("utf-8")).hexdigest()[:14]
            )
            member_sources = group.get("member_source_ids", [])
            member_source_count = (
                len(member_sources) if isinstance(member_sources, list) else 0
            )
            overflow_importance = _clamp01(
                0.2
                + min(
                    0.62,
                    math.log1p(max(1, member_source_count)) / 5.8,
                )
            )

            overflow_nodes.append(
                {
                    "id": overflow_node_id,
                    "node_id": overflow_node_id,
                    "node_type": "file",
                    "name": overflow_label,
                    "label": overflow_label,
                    "kind": "projection_overflow",
                    "x": round(centroid_x, 4),
                    "y": round(centroid_y, 4),
                    "hue": int(hue),
                    "importance": round(overflow_importance, 4),
                    "source_rel_path": (
                        "_projection/" + sha1(group_id.encode("utf-8")).hexdigest()[:18]
                    ),
                    "archive_kind": "projection",
                    "dominant_field": field_id,
                    "dominant_presence": dominant_presence,
                    "field_scores": {field_id: 1.0},
                    "summary": "Simulation projection bucket preserving grouped edge lineage.",
                    "consolidated": True,
                    "consolidated_count": member_source_count,
                    "projection_overflow": True,
                    "projection_group_id": group_id,
                }
            )

            overflow_edges.append(
                {
                    "id": "edge:projection:"
                    + sha1(
                        f"{overflow_node_id}|{target_id}|{kind}".encode("utf-8")
                    ).hexdigest()[:16],
                    "source": overflow_node_id,
                    "target": target_id,
                    "field": field_id,
                    "weight": round(
                        _clamp01(
                            _safe_float(group.get("weight_sum", 0.0), 0.0)
                            / max(1.0, _safe_float(group.get("weight_count", 1), 1.0))
                        ),
                        4,
                    ),
                    "kind": kind,
                    "projection_overflow": True,
                    "projection_group_id": group_id,
                    "projection_member_edge_count": int(
                        group.get("member_edge_count", 0)
                    ),
                    "projection_member_edge_digest": str(
                        group.get("member_edge_digest", "")
                    ),
                }
            )

        next_file_nodes = file_nodes + overflow_nodes
        non_file_nodes = [
            dict(node)
            for node in graph_nodes
            if str(node.get("node_type", "")).strip().lower() != "file"
        ]
        next_edges = kept_edges + overflow_edges

        projection_groups = []
        for group in finalized_groups:
            group_row = dict(group)
            group_row["surface_visible"] = (
                str(group.get("id", "")).strip() in visual_group_ids
            )
            projection_groups.append(group_row)

        projection_payload = {
            "record": SIMULATION_FILE_GRAPH_PROJECTION_RECORD,
            "schema_version": SIMULATION_FILE_GRAPH_PROJECTION_SCHEMA_VERSION,
            "generated_at": now_iso,
            "mode": "hub-overflow",
            "active": True,
            "reason": "edge_budget",
            "limits": {
                "edge_threshold": int(SIMULATION_FILE_GRAPH_PROJECTION_EDGE_THRESHOLD),
                "edge_cap": int(global_edge_cap),
                "per_source_cap": int(per_source_cap),
                "per_source_categorizes_cap": int(per_source_categorizes_cap),
                "field_hub_cap": int(per_field_hub_cap),
                "concept_hub_cap": int(per_concept_hub_cap),
                "tag_member_cap": int(tag_member_cap),
                "tag_pair_cap": int(tag_pair_cap),
                "overflow_group_cap": int(overflow_group_limit),
            },
            "before": {
                "file_nodes": int(file_count_before),
                "edges": int(edge_count_before),
            },
            "after": {
                "file_nodes": int(len(next_file_nodes)),
                "edges": int(len(next_edges)),
            },
            "collapsed_edges": int(collapsed_edges),
            "overflow_nodes": int(len(overflow_nodes)),
            "overflow_edges": int(len(overflow_edges)),
            "group_count": int(len(projection_groups)),
            "groups": projection_groups,
        }

        updated_graph = dict(file_graph)
        updated_graph["file_nodes"] = next_file_nodes
        updated_graph["nodes"] = non_file_nodes + next_file_nodes
        updated_graph["edges"] = next_edges
        updated_graph["projection"] = projection_payload

        graph_stats = (
            dict(file_graph.get("stats", {}))
            if isinstance(file_graph.get("stats", {}), dict)
            else {}
        )
        graph_stats["file_count"] = len(next_file_nodes)
        graph_stats["edge_count"] = len(next_edges)
        graph_stats["projection_active"] = True
        graph_stats["projection_collapsed_edge_count"] = int(collapsed_edges)
        graph_stats["projection_overflow_node_count"] = len(overflow_nodes)
        graph_stats["projection_overflow_edge_count"] = len(overflow_edges)
        graph_stats["projection_group_count"] = len(projection_groups)
        updated_graph["stats"] = graph_stats

        return (
            updated_graph,
            _event(
                "simulation.file_graph.projection.applied",
                "ok",
                "edge_budget",
                {
                    "edge_count_before": int(edge_count_before),
                    "edge_count_after": int(len(next_edges)),
                    "collapsed_edges": int(collapsed_edges),
                    "overflow_nodes": int(len(overflow_nodes)),
                    "overflow_edges": int(len(overflow_edges)),
                    "group_count": int(len(projection_groups)),
                    "edge_cap": int(global_edge_cap),
                },
            ),
        )
    except Exception as exc:
        failed_payload = {
            "record": SIMULATION_FILE_GRAPH_PROJECTION_RECORD,
            "schema_version": SIMULATION_FILE_GRAPH_PROJECTION_SCHEMA_VERSION,
            "generated_at": now_iso,
            "mode": "hub-overflow",
            "active": False,
            "reason": "fail_safe_noop",
            "error": exc.__class__.__name__,
            "limits": {
                "edge_threshold": int(SIMULATION_FILE_GRAPH_PROJECTION_EDGE_THRESHOLD),
            },
            "before": {
                "file_nodes": len(
                    [
                        row
                        for row in file_graph.get("file_nodes", [])
                        if isinstance(row, dict)
                    ]
                ),
                "edges": len(
                    [
                        row
                        for row in file_graph.get("edges", [])
                        if isinstance(row, dict)
                    ]
                ),
            },
            "after": {
                "file_nodes": len(
                    [
                        row
                        for row in file_graph.get("file_nodes", [])
                        if isinstance(row, dict)
                    ]
                ),
                "edges": len(
                    [
                        row
                        for row in file_graph.get("edges", [])
                        if isinstance(row, dict)
                    ]
                ),
            },
            "collapsed_edges": 0,
            "overflow_nodes": 0,
            "overflow_edges": 0,
            "group_count": 0,
            "groups": [],
        }
        updated_graph = dict(file_graph)
        updated_graph["projection"] = failed_payload
        return (
            updated_graph,
            _event(
                "simulation.file_graph.projection.failed",
                "blocked",
                "fail_safe_noop",
                {
                    "error": exc.__class__.__name__,
                },
            ),
        )


def _stable_entity_id(prefix: str, seed: str, width: int = 20) -> str:
    token = hashlib.sha256(seed.encode("utf-8")).hexdigest()[: max(8, width)]
    return f"{prefix}:{token}"


def _field_scores_from_position(
    x: float,
    y: float,
    field_anchors: dict[str, tuple[float, float]],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for field_id in FIELD_TO_PRESENCE:
        anchor_x, anchor_y = field_anchors.get(field_id, (0.5, 0.5))
        dx, dy = x - anchor_x, y - anchor_y
        distance = math.sqrt((dx * dx) + (dy * dy))
        raw[field_id] = 1.0 / (0.04 + (distance * distance * 6.0))
    return _normalize_field_scores(raw)


def _daimoi_softmax_weights(
    rows: list[tuple[str, float]], *, temperature: float
) -> dict[str, float]:
    if not rows:
        return {}
    temp = max(0.05, _safe_float(temperature, 0.42))
    max_score = max(_safe_float(score, 0.0) for _, score in rows)
    expo = [
        (eid, math.exp((_safe_float(s, 0.0) - max_score) / temp)) for eid, s in rows
    ]
    total = sum(v for _, v in expo)
    return (
        {eid: v / total for eid, v in expo}
        if total > 0.0
        else {eid: 1.0 / len(expo) for eid, _ in expo}
    )


def _build_daimoi_state(
    heat_values: dict[str, Any],
    pain_field: dict[str, Any],
    *,
    queue_ratio: float = 0.0,
    resource_ratio: float = 0.0,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    node_heat_rows = (
        pain_field.get("node_heat", []) if isinstance(pain_field, dict) else []
    )
    relations: dict[str, list[dict[str, Any]]] = {
        "霊/attend": [],
        "霊/push": [],
        "霊/link": [],
        "霊/bind": [],
    }

    if not node_heat_rows:
        return {
            "record": "ημ.daimoi.v1",
            "generated_at": generated_at,
            "glyph": "霊",
            "active": False,
            "pressure": {
                "queue_ratio": round(_clamp01(_safe_float(queue_ratio)), 4),
                "resource_ratio": round(_clamp01(_safe_float(resource_ratio)), 4),
            },
            "daimoi": [],
            "relations": relations,
            "entities": [],
            "physics": {
                "kappa": round(DAIMO_FORCE_KAPPA, 6),
                "damping": round(DAIMO_DAMPING, 6),
                "dt": round(DAIMO_DT_SECONDS, 6),
            },
        }

    region_heat, region_centers = {}, {}
    for row in heat_values.get("regions", []):
        rid = str(row.get("region_id", "")).strip()
        if rid:
            region_heat[rid] = _clamp01(
                _safe_float(row.get("value", row.get("heat", 0.0)))
            )
            region_centers[rid] = (
                _clamp01(_safe_float(row.get("x", 0.5))),
                _clamp01(_safe_float(row.get("y", 0.5))),
            )
    for row in heat_values.get("facts", []):
        rid = str(row.get("region_id", "")).strip()
        if rid:
            region_heat[rid] = max(
                region_heat.get(rid, 0.0), _clamp01(_safe_float(row.get("value")))
            )
            region_centers.setdefault(rid, (0.5, 0.5))

    entity_manifest_by_id = {
        str(row.get("id")): row
        for row in ENTITY_MANIFEST
        if str(row.get("id", "")).strip()
    }
    field_anchors = {}
    for fid, pid in FIELD_TO_PRESENCE.items():
        c = region_centers.get(fid)
        if c:
            field_anchors[fid] = c
        else:
            e = entity_manifest_by_id.get(pid, {})
            field_anchors[fid] = (
                _clamp01(_safe_float(e.get("x", 0.5))),
                _clamp01(_safe_float(e.get("y", 0.5))),
            )

    locate_by_entity = defaultdict(dict)
    for row in heat_values.get("locate", []):
        eid, rid = str(row.get("entity_id")), str(row.get("region_id"))
        if eid and rid:
            locate_by_entity[eid][rid] = max(
                locate_by_entity[eid].get(rid, 0.0),
                _clamp01(_safe_float(row.get("weight"))),
            )

    entities = []
    for row in node_heat_rows[:DAIMO_MAX_TRACKED_ENTITIES]:
        eid = str(row.get("node_id"))
        if not eid:
            continue
        x, y, h = (
            _clamp01(_safe_float(row.get("x", 0.5))),
            _clamp01(_safe_float(row.get("y", 0.5))),
            _clamp01(_safe_float(row.get("heat", 0.0))),
        )
        locate = dict(locate_by_entity.get(eid, {}))
        if not locate:
            locate = _field_scores_from_position(x, y, field_anchors)
        score = sum(
            _clamp01(_safe_float(w)) * _clamp01(_safe_float(region_heat.get(rid, 0.0)))
            for rid, w in locate.items()
        ) or (h * 0.1)
        entities.append(
            {
                "id": eid,
                "x": x,
                "y": y,
                "heat": h,
                "score": score,
                "mass": max(0.35, 0.8 + ((1.0 - h) * 2.2)),
                "locate": locate,
            }
        )

    entities.sort(key=lambda r: (-_safe_float(r.get("score")), str(r.get("id"))))
    entity_by_id = {str(r["id"]): r for r in entities}
    pressure = _clamp01(
        (_clamp01(_safe_float(queue_ratio)) * 0.58)
        + (_clamp01(_safe_float(resource_ratio)) * 0.42)
    )
    budget_scale = max(0.4, 1.0 - (pressure * 0.5))

    daimo_rows, force_by_entity = [], defaultdict(lambda: [0.0, 0.0])
    for idx, profile in enumerate(DAIMO_PROFILE_DEFS):
        did = str(profile.get("id", f"daimo:{idx}"))
        ctx, dw = (
            str(profile.get("ctx", "世")),
            _clamp01(_safe_float(profile.get("w", 0.88))),
        )
        budget = max(
            1, int(round(_safe_float(profile.get("base_budget", 6.0)) * budget_scale))
        )
        temp, top_k = (
            max(0.05, _safe_float(profile.get("temperature", 0.42))),
            max(1, min(6, budget // 2)),
        )

        scored = []
        for e_idx, e in enumerate(entities[:64]):
            eid, eh = e["id"], e["heat"]
            gain = {
                "主": 1.08 + eh * 0.08,
                "己": 0.95 + (1 - eh) * 0.08,
                "汝": 0.98 + abs(e["x"] - 0.5) * 0.06,
                "彼": 0.98 + abs(e["y"] - 0.5) * 0.06,
            }.get(ctx, 1.02 + pressure * 0.08)
            s = max(
                0.0,
                (e["score"] * gain)
                + (_stable_ratio(f"{did}|{eid}", e_idx) - 0.5) * 0.12,
            )
            if s > 0.0:
                scored.append((eid, s))

        scored.sort(key=lambda r: (-r[1], r[0]))
        top = scored[:top_k]
        attn = _daimoi_softmax_weights(top, temperature=temp)
        counts = {"attend": 0, "push": 0, "bind": 0, "link": 0}

        for eid, escore in top:
            aw = _clamp01(_safe_float(attn.get(eid)))
            if aw <= 0.0:
                continue
            counts["attend"] += 1
            relations["霊/attend"].append(
                {
                    "id": _stable_entity_id("edge", f"{did}|{eid}|霊/attend"),
                    "rel": "霊/attend",
                    "daimo_id": did,
                    "entity_id": eid,
                    "w": round(aw, 6),
                    "score": round(escore, 6),
                }
            )
            e = entity_by_id[eid]
            vx = vy = 0.0
            best_r, best_s = "", 0.0
            for rid, lw in e["locate"].items():
                sig = _clamp01(_safe_float(lw)) * region_heat.get(str(rid), 0.0)
                if sig <= 0.0:
                    continue
                bx, by = field_anchors.get(str(rid), (0.5, 0.5))
                dx, dy = bx - e["x"], by - e["y"]
                mag = math.sqrt(dx**2 + dy**2)
                if mag > 1e-8:
                    vx += sig * (dx / mag)
                    vy += sig * (dy / mag)
                if sig > best_s:
                    best_s, best_r = sig, str(rid)
            v_mag = math.sqrt(vx**2 + vy**2)
            dx, dy = (vx / v_mag, vy / v_mag) if v_mag > 1e-8 else (0.0, 0.0)
            fx, fy = DAIMO_FORCE_KAPPA * dw * aw * dx, DAIMO_FORCE_KAPPA * dw * aw * dy
            if abs(fx) + abs(fy) > 1e-10:
                force_by_entity[eid][0] += fx
                force_by_entity[eid][1] += fy
            counts["push"] += 1
            relations["霊/push"].append(
                {
                    "id": _stable_entity_id("edge", f"{did}|{eid}|霊/push"),
                    "rel": "霊/push",
                    "daimo_id": did,
                    "entity_id": eid,
                    "region_id": best_r,
                    "fx": round(fx, 8),
                    "fy": round(fy, 8),
                    "w": round(aw, 6),
                }
            )

        if len(top) >= 2:
            counts["link"] += 1
            relations["霊/link"].append(
                {
                    "id": _stable_entity_id(
                        "edge", f"{did}|{top[0][0]}|{top[1][0]}|霊/link"
                    ),
                    "rel": "霊/link",
                    "daimo_id": did,
                    "entity_a": top[0][0],
                    "entity_b": top[1][0],
                    "w": round(math.sqrt(attn[top[0][0]] * attn[top[1][0]]), 6),
                }
            )

        daimo_rows.append(
            {
                "id": did,
                "name": str(profile.get("name", did)),
                "ctx": ctx,
                "state": "idle"
                if not top
                else ("move" if counts["push"] > 0 else "seek"),
                "budget": float(budget),
                "w": round(dw, 4),
                "at_iso": generated_at,
                "emitted": {**counts, "total": sum(counts.values())},
            }
        )

    e_rows = [
        {
            "id": eid,
            "x": round(e["x"], 4),
            "y": round(e["y"], 4),
            "heat": round(e["heat"], 4),
            "score": round(e["score"], 6),
            "mass": round(e["mass"], 6),
            "force": {
                "fx": round(force_by_entity[eid][0], 8),
                "fy": round(force_by_entity[eid][1], 8),
                "magnitude": round(
                    math.sqrt(
                        force_by_entity[eid][0] ** 2 + force_by_entity[eid][1] ** 2
                    ),
                    8,
                ),
            },
        }
        for eid, e in entity_by_id.items()
    ]
    return {
        "record": "ημ.daimoi.v1",
        "generated_at": generated_at,
        "glyph": "霊",
        "active": bool(relations["霊/attend"]),
        "pressure": {
            "queue_ratio": round(_clamp01(_safe_float(queue_ratio)), 4),
            "resource_ratio": round(_clamp01(_safe_float(resource_ratio)), 4),
            "blend": round(pressure, 4),
        },
        "daimoi": daimo_rows,
        "relations": relations,
        "entities": e_rows,
        "physics": {
            "kappa": round(DAIMO_FORCE_KAPPA, 6),
            "damping": round(DAIMO_DAMPING, 6),
            "dt": round(DAIMO_DT_SECONDS, 6),
        },
    }


def _apply_daimoi_dynamics_to_pain_field(
    pain_field: dict[str, Any], daimoi_state: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(pain_field, dict):
        return {}
    node_heat_rows = pain_field.get("node_heat", [])
    relations = (
        daimoi_state.get("relations", {}) if isinstance(daimoi_state, dict) else {}
    )
    push_rows = relations.get("霊/push", []) if isinstance(relations, dict) else []
    force_by_entity = defaultdict(lambda: [0.0, 0.0])
    for row in push_rows:
        eid = str(row.get("entity_id")).strip()
        if eid:
            force_by_entity[eid][0] += _safe_float(row.get("fx"))
            force_by_entity[eid][1] += _safe_float(row.get("fy"))

    physics = daimoi_state.get("physics", {}) if isinstance(daimoi_state, dict) else {}
    dt, damping = (
        max(0.02, min(0.4, _safe_float(physics.get("dt", DAIMO_DT_SECONDS)))),
        max(0.0, min(0.99, _safe_float(physics.get("damping", DAIMO_DAMPING)))),
    )
    edge_band = 0.12
    edge_pressure = 0.08
    edge_bounce = 0.74
    updated_rows, active_ids, now_mono = [], set(), time.monotonic()

    with _DAIMO_DYNAMICS_LOCK:
        cache = _DAIMO_DYNAMICS_CACHE.get("entities", {})
        for row in node_heat_rows:
            eid = str(row.get("node_id")).strip()
            if not eid:
                updated_rows.append(dict(row))
                continue
            active_ids.add(eid)
            bx, by, h = (
                _clamp01(_safe_float(row.get("x", 0.5))),
                _clamp01(_safe_float(row.get("y", 0.5))),
                _clamp01(_safe_float(row.get("heat", 0.0))),
            )
            mass, c = max(0.35, 0.7 + ((1.0 - h) * 2.0)), cache.get(eid, {})
            px, py, pvx, pvy = (
                _clamp01(_safe_float(c.get("x", bx))),
                _clamp01(_safe_float(c.get("y", by))),
                _safe_float(c.get("vx")),
                _safe_float(c.get("vy")),
            )
            fx, fy = (
                force_by_entity[eid][0] + (bx - px) * 0.18,
                force_by_entity[eid][1] + (by - py) * 0.18,
            )

            if px < edge_band:
                fx += ((edge_band - px) / edge_band) * edge_pressure
            elif px > (1.0 - edge_band):
                fx -= ((px - (1.0 - edge_band)) / edge_band) * edge_pressure
            if py < edge_band:
                fy += ((edge_band - py) / edge_band) * edge_pressure
            elif py > (1.0 - edge_band):
                fy -= ((py - (1.0 - edge_band)) / edge_band) * edge_pressure

            nvx, nvy = (
                (pvx * damping) + ((dt / mass) * fx),
                (pvy * damping) + ((dt / mass) * fy),
            )
            nx, ny = px + (dt * nvx), py + (dt * nvy)
            if nx < 0.0:
                nx = -nx
                nvx = abs(nvx) * edge_bounce
            elif nx > 1.0:
                nx = 2.0 - nx
                nvx = -abs(nvx) * edge_bounce
            if ny < 0.0:
                ny = -ny
                nvy = abs(nvy) * edge_bounce
            elif ny > 1.0:
                ny = 2.0 - ny
                nvy = -abs(nvy) * edge_bounce
            nx, ny = _clamp01(nx), _clamp01(ny)
            cache[eid] = {"x": nx, "y": ny, "vx": nvx, "vy": nvy, "ts": now_mono}
            updated_rows.append(
                {
                    **row,
                    "x": round(nx, 4),
                    "y": round(ny, 4),
                    "vx": round(nvx, 6),
                    "vy": round(nvy, 6),
                    "speed": round(math.sqrt(nvx**2 + nvy**2), 6),
                }
            )
        stale = now_mono - 120.0
        for k in list(cache.keys()):
            if k not in active_ids and _safe_float(cache[k].get("ts")) < stale:
                cache.pop(k, None)
        _DAIMO_DYNAMICS_CACHE["entities"] = cache

    return {
        **pain_field,
        "node_heat": updated_rows,
        "motion": {
            "record": "ημ.daimoi-motion.v1",
            "glyph": "霊",
            "active": bool(force_by_entity),
            "dt": round(dt, 6),
            "damping": round(damping, 6),
            "edge_band": round(edge_band, 4),
            "edge_pressure": round(edge_pressure, 6),
            "entity_count": len(updated_rows),
            "forced_entities": len(force_by_entity),
        },
    }


def _load_test_failures_from_path(candidate: Path) -> list[dict[str, Any]]:
    try:
        text = candidate.read_text("utf-8")
    except:
        return []
    if candidate.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
        if candidate.suffix.lower() == ".json":
            try:
                return _coerce_test_failure_rows(json.loads(text))
            except:
                return []
        rows = []
        for line in text.splitlines():
            try:
                rows.extend(_coerce_test_failure_rows(json.loads(line)))
            except:
                continue
        return rows
    return _parse_test_failures_text(text)


def _coerce_test_failure_rows(payload: Any) -> list[dict[str, Any]]:
    rows = []

    def _add(item):
        if isinstance(item, dict):
            rows.append(dict(item))
        elif str(item).strip():
            rows.append({"name": str(item).strip(), "status": "failed"})

    if isinstance(payload, list):
        for i in payload:
            _add(i)
    elif isinstance(payload, dict):
        for k in ("failures", "failed_tests", "failing_tests", "tests"):
            if isinstance(payload.get(k), list):
                for i in payload[k]:
                    if k == "tests" and isinstance(i, dict):
                        s = str(i.get("status") or i.get("outcome") or "").lower()
                        if s in {"failed", "error", "failing", "xfailed"}:
                            r = dict(i)
                            r.setdefault("status", s or "failed")
                            r.setdefault(
                                "name",
                                str(
                                    i.get("nodeid")
                                    or i.get("test")
                                    or i.get("id")
                                    or ""
                                ),
                            )
                            rows.append(r)
                    else:
                        _add(i)
                if rows:
                    return rows
        n = str(
            payload.get("name") or payload.get("test") or payload.get("nodeid") or ""
        ).strip()
        if n:
            r = dict(payload)
            r.setdefault("status", "failed")
            rows.append(r)
    return rows


def _parse_test_failures_text(raw_text: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw_text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        n, sep, c = line.partition("|")
        if n.strip():
            row: dict[str, Any] = {"name": n.strip(), "status": "failed"}
            if sep:
                row["covered_files"] = [
                    t.strip() for t in re.split(r"[,\s]+", c.strip()) if t.strip()
                ]
            rows.append(row)
    return rows


def _load_test_coverage_from_path(
    candidate: Path, part_root: Path, vault_root: Path
) -> dict[str, Any]:
    try:
        text = candidate.read_text("utf-8")
    except:
        return {}
    if candidate.suffix.lower() == ".json":
        try:
            p = json.loads(text)
            return p if isinstance(p, dict) else {}
        except:
            return {}
    if candidate.name.lower().endswith(".info") and "lcov" in candidate.name.lower():
        return _parse_lcov_payload(text, part_root, vault_root)
    return {}


def _parse_lcov_payload(text: str, part_root: Path, vault_root: Path) -> dict[str, Any]:
    files, by_test_sets, by_test_spans = {}, defaultdict(set), defaultdict(list)
    cur_t = cur_s = ""
    da_f = da_h = lf = lh = 0
    cur_hits = []

    def _flush():
        nonlocal cur_s, da_f, da_h, lf, lh, cur_hits
        if not cur_s:
            return
        n = _normalize_coverage_source_path(cur_s, part_root, vault_root)
        if n:
            e = files.setdefault(
                n,
                {
                    "file_id": _file_id_for_path(n),
                    "lines_found": 0,
                    "lines_hit": 0,
                    "tests": [],
                },
            )
            e["lines_found"] += lf or da_f
            e["lines_hit"] += lh or da_h
            if cur_t.strip():
                by_test_sets[cur_t.strip()].add(n)
                if cur_t.strip() not in e["tests"]:
                    e["tests"].append(cur_t.strip())
                for sp in _line_hits_to_spans(cur_hits):
                    by_test_spans[cur_t.strip()].append(
                        {
                            "file": n,
                            "start_line": sp["start_line"],
                            "end_line": sp["end_line"],
                            "hits": sp["hits"],
                            "weight": 1.0,
                        }
                    )
        cur_s = ""
        da_f = da_h = lf = lh = 0
        cur_hits = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("TN:"):
            cur_t = line[3:].strip()
        elif line.startswith("SF:"):
            _flush()
            cur_s = line[3:].strip()
        elif line == "end_of_record":
            _flush()
        elif cur_s:
            if line.startswith("DA:"):
                p = line[3:].split(",")
                ln, h = int(_safe_float(p[0])), int(_safe_float(p[1]))
                cur_hits.append((ln, h))
                da_f += 1
                da_h += 1 if h > 0 else 0
            elif line.startswith("LF:"):
                lf = int(_safe_float(line[3:]))
            elif line.startswith("LH:"):
                lh = int(_safe_float(line[3:]))
    _flush()
    f_pay = {
        k: {
            **v,
            "line_rate": round(_clamp01(v["lines_hit"] / v["lines_found"]), 6)
            if v["lines_found"] > 0
            else 0.0,
        }
        for k, v in files.items()
    }
    return {
        "record": "ημ.test-coverage.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "lcov",
        "files": f_pay,
        "by_test": {k: sorted(list(v)) for k, v in by_test_sets.items()},
        "by_test_spans": dict(by_test_spans),
        "hottest_files": sorted(
            f_pay.keys(), key=lambda k: (f_pay[k].get("line_rate", 1.0), k)
        ),
    }


def _normalize_coverage_source_path(raw: str, part_root: Path, vault_root: Path) -> str:
    s = str(raw or "").strip()
    if s.startswith("file://"):
        s = unquote(urlparse(s).path)
    p = Path(s.strip())
    if p.is_absolute():
        try:
            r = p.resolve(strict=False)
        except:
            r = p
        for root in (part_root, vault_root):
            try:
                return _normalize_path_for_file_id(str(r.relative_to(root.resolve())))
            except:
                continue
    return _normalize_path_for_file_id(s)


def _line_hits_to_spans(hits: list[tuple[int, int]]) -> list[dict[str, Any]]:
    sorted_hits = sorted([(ln, h) for ln, h in hits if h > 0], key=lambda r: r[0])
    if not sorted_hits:
        return []
    spans, sl, pl, t = [], sorted_hits[0][0], sorted_hits[0][0], sorted_hits[0][1]
    for ln, h in sorted_hits[1:]:
        if ln <= pl + 1:
            pl, t = ln, t + h
        else:
            spans.append({"start_line": sl, "end_line": pl, "hits": t})
            sl, pl, t = ln, ln, h
    spans.append({"start_line": sl, "end_line": pl, "hits": t})
    return spans


def _extract_coverage_spans(raw: Any) -> list[dict[str, Any]]:
    spans = []

    def _walk(item, fp, fw):
        if isinstance(item, str):
            n = _normalize_path_for_file_id(item)
            if n:
                spans.append(
                    {
                        "path": n,
                        "start_line": 1,
                        "end_line": 1,
                        "symbol": "",
                        "weight": fw,
                    }
                )
        elif isinstance(item, list):
            for s in item:
                _walk(s, fp, fw)
        elif isinstance(item, dict):
            p = next(
                (
                    item.get(k)
                    for k in ("file", "path", "source")
                    if isinstance(item.get(k), str)
                ),
                fp,
            )
            w = _safe_float(
                next(
                    (item.get(k) for k in ("w", "weight") if item.get(k) is not None),
                    fw,
                )
            )
            for k in ("spans", "files", "coverage"):
                if item.get(k):
                    _walk(item[k], p, w)
            if p and not any(item.get(k) for k in ("spans", "files", "coverage")):
                spans.append(
                    {
                        "path": _normalize_path_for_file_id(p),
                        "start_line": int(_safe_int(item.get("start_line", 1))),
                        "end_line": int(_safe_int(item.get("end_line", 1))),
                        "symbol": str(item.get("symbol", "")),
                        "weight": w,
                    }
                )

    _walk(raw, "", 1.0)
    return spans


def _load_test_signal_artifacts(
    part_root: Path, vault_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    f: list[dict[str, Any]] = []
    for c in [
        part_root / "world_state" / "failing_tests.txt",
        part_root / "world_state" / "failing_tests.json",
        part_root / ".opencode" / "runtime" / "failing_tests.json",
        vault_root / ".opencode" / "runtime" / "failing_tests.json",
    ]:
        if c.exists():
            rows = _load_test_failures_from_path(c)
            if rows:
                f = rows
                break
    cov: dict[str, Any] = {}
    for c in [
        part_root / "coverage" / "lcov.info",
        part_root / "world_state" / "test_coverage.json",
    ]:
        if c.exists():
            p = _load_test_coverage_from_path(c, part_root, vault_root)
            if p:
                cov = p
                break
    return f, cov


def _build_logical_graph(catalog: dict[str, Any]) -> dict[str, Any]:
    file_graph = catalog.get("file_graph") if isinstance(catalog, dict) else {}
    truth_state = catalog.get("truth_state") if isinstance(catalog, dict) else {}
    if not isinstance(file_graph, dict):
        file_graph = {}
    if not isinstance(truth_state, dict):
        truth_state = {}

    file_nodes_raw = file_graph.get("file_nodes", [])
    if not isinstance(file_nodes_raw, list):
        file_nodes_raw = []
    file_edges_raw = file_graph.get("edges", [])
    if not isinstance(file_edges_raw, list):
        file_edges_raw = []
    tag_nodes_raw = file_graph.get("tag_nodes", [])
    if not isinstance(tag_nodes_raw, list):
        fallback_graph_nodes = file_graph.get("nodes", [])
        if isinstance(fallback_graph_nodes, list):
            tag_nodes_raw = [
                row
                for row in fallback_graph_nodes
                if isinstance(row, dict)
                and str(row.get("node_type", "")).strip().lower() == "tag"
            ]
        else:
            tag_nodes_raw = []

    claims_raw = truth_state.get("claims", [])
    if not isinstance(claims_raw, list) or not claims_raw:
        claim_single = truth_state.get("claim", {})
        if isinstance(claim_single, dict) and claim_single:
            claims_raw = [claim_single]
        else:
            claims_raw = []

    proof = truth_state.get("proof", {})
    if not isinstance(proof, dict):
        proof = {}
    proof_entries = proof.get("entries", [])
    if not isinstance(proof_entries, list):
        proof_entries = []
    required_kinds = proof.get("required_kinds", [])
    if not isinstance(required_kinds, list):
        required_kinds = []

    gate = truth_state.get("gate", {})
    if not isinstance(gate, dict):
        gate = {}

    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    joins_source_to_file: dict[str, str] = {}
    file_path_to_node: dict[str, str] = {}
    file_id_to_node: dict[str, str] = {}
    file_graph_node_to_logical: dict[str, str] = {}
    tag_graph_node_to_logical: dict[str, str] = {}
    tag_token_to_logical: dict[str, str] = {}

    test_failures = (
        catalog.get("test_failures", []) if isinstance(catalog, dict) else []
    )
    if not isinstance(test_failures, list):
        test_failures = []

    for file_node in file_nodes_raw:
        if not isinstance(file_node, dict):
            continue
        source_rel_path = str(
            file_node.get("source_rel_path")
            or file_node.get("archived_rel_path")
            or file_node.get("archive_rel_path")
            or file_node.get("name")
            or ""
        )
        normalized_path = _normalize_path_for_file_id(source_rel_path)
        if not normalized_path:
            continue
        file_id = _file_id_for_path(normalized_path)
        if not file_id:
            continue
        node_id = f"logical:file:{file_id[:24]}"
        source_uri = f"library:/{normalized_path}"
        file_path_to_node[normalized_path] = node_id
        file_id_to_node[file_id] = node_id
        joins_source_to_file[source_uri] = file_id
        file_graph_node_id = str(file_node.get("id", "")).strip()
        if file_graph_node_id:
            file_graph_node_to_logical[file_graph_node_id] = node_id
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "file",
                "label": str(
                    file_node.get("label") or file_node.get("name") or normalized_path
                ),
                "file_id": file_id,
                "source_uri": source_uri,
                "path": normalized_path,
                "x": round(_clamp01(_safe_float(file_node.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(file_node.get("y", 0.5), 0.5)), 4),
                "confidence": 1.0,
                "provenance": {
                    "source_uri": source_uri,
                    "file_id": file_id,
                },
            }
        )

    for idx, tag_node in enumerate(tag_nodes_raw):
        if not isinstance(tag_node, dict):
            continue
        raw_tag = str(
            tag_node.get("tag")
            or tag_node.get("node_id")
            or tag_node.get("label")
            or ""
        ).strip()
        normalized_tag = re.sub(r"\s+", "_", raw_tag.lower())
        normalized_tag = re.sub(r"[^a-z0-9_]+", "", normalized_tag)
        normalized_tag = normalized_tag.strip("_")
        if not normalized_tag:
            continue
        node_id = (
            "logical:tag:"
            + hashlib.sha256(normalized_tag.encode("utf-8")).hexdigest()[:22]
        )
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "tag",
                "label": str(tag_node.get("label") or raw_tag or normalized_tag),
                "status": "active",
                "confidence": round(
                    _clamp01(
                        min(
                            1.0,
                            _safe_float(tag_node.get("member_count", 1), 1.0) / 8.0,
                        )
                    ),
                    4,
                ),
                "x": round(_clamp01(_safe_float(tag_node.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(tag_node.get("y", 0.5), 0.5)), 4),
                "provenance": {
                    "tag": normalized_tag,
                    "member_count": int(
                        _safe_float(tag_node.get("member_count", 0), 0.0)
                    ),
                },
            }
        )
        graph_tag_id = str(tag_node.get("id", "")).strip()
        if graph_tag_id:
            tag_graph_node_to_logical[graph_tag_id] = node_id
        tag_token_to_logical[normalized_tag] = node_id

    tag_edge_seen: set[tuple[str, str, str]] = set()
    for edge in file_edges_raw:
        if not isinstance(edge, dict):
            continue
        kind = str(edge.get("kind", "")).strip().lower()
        if kind not in {"labeled_as", "relates_tag"}:
            continue
        source_key = str(edge.get("source", "")).strip()
        target_key = str(edge.get("target", "")).strip()
        source_id = file_graph_node_to_logical.get(
            source_key
        ) or tag_graph_node_to_logical.get(source_key)
        target_id = file_graph_node_to_logical.get(
            target_key
        ) or tag_graph_node_to_logical.get(target_key)
        if not source_id or not target_id or source_id == target_id:
            continue
        edge_key = (source_id, target_id, kind)
        if edge_key in tag_edge_seen:
            continue
        tag_edge_seen.add(edge_key)
        graph_edges.append(
            {
                "id": "logical:edge:tag:"
                + hashlib.sha256(
                    f"{source_id}|{target_id}|{kind}".encode("utf-8")
                ).hexdigest()[:20],
                "source": source_id,
                "target": target_id,
                "kind": kind,
                "weight": round(
                    _clamp01(_safe_float(edge.get("weight", 0.55), 0.55)), 4
                ),
            }
        )

    for file_node in file_nodes_raw:
        if not isinstance(file_node, dict):
            continue
        source_logical_id = file_graph_node_to_logical.get(str(file_node.get("id", "")))
        if not source_logical_id:
            continue
        tags_raw = file_node.get("tags", [])
        if not isinstance(tags_raw, list):
            continue
        for tag_raw in tags_raw:
            normalized_tag = re.sub(r"\s+", "_", str(tag_raw or "").strip().lower())
            normalized_tag = re.sub(r"[^a-z0-9_]+", "", normalized_tag).strip("_")
            if not normalized_tag:
                continue
            target_logical_id = tag_token_to_logical.get(normalized_tag)
            if not target_logical_id:
                continue
            edge_key = (source_logical_id, target_logical_id, "labeled_as")
            if edge_key in tag_edge_seen:
                continue
            tag_edge_seen.add(edge_key)
            graph_edges.append(
                {
                    "id": "logical:edge:tag:"
                    + hashlib.sha256(
                        f"{source_logical_id}|{target_logical_id}|fallback".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:20],
                    "source": source_logical_id,
                    "target": target_logical_id,
                    "kind": "labeled_as",
                    "weight": 0.58,
                }
            )

    for idx, row in enumerate(test_failures):
        if not isinstance(row, dict):
            continue
        test_name = str(row.get("name") or row.get("test") or "").strip()
        if not test_name:
            continue
        test_id_seed = f"test:{test_name}|{idx}"
        test_node_id = f"logical:test:{hashlib.sha256(test_id_seed.encode('utf-8')).hexdigest()[:24]}"
        graph_nodes.append(
            {
                "id": test_node_id,
                "kind": "test",
                "label": test_name,
                "glyph": "試",
                "status": str(row.get("status", "failed")),
                "x": 0.5,
                "y": 0.5,
                "confidence": 1.0,
            }
        )

        covered_files = row.get("covered_files", [])
        if isinstance(covered_files, list):
            for path_item in covered_files:
                normalized_path = _normalize_path_for_file_id(str(path_item))
                target_node_id = file_path_to_node.get(normalized_path)
                if not target_node_id:
                    file_id = _file_id_for_path(normalized_path)
                    target_node_id = file_id_to_node.get(file_id)
                if target_node_id:
                    graph_edges.append(
                        {
                            "source": test_node_id,
                            "target": target_node_id,
                            "kind": "covers",
                            "weight": 0.8,
                        }
                    )

    world_log = catalog.get("world_log") if isinstance(catalog, dict) else {}
    world_log_events = (
        world_log.get("events", []) if isinstance(world_log, dict) else []
    )
    if not isinstance(world_log_events, list):
        world_log_events = []

    event_node_by_event_id: dict[str, str] = {}
    event_relation_pairs: set[tuple[str, str]] = set()
    event_link_count = 0
    event_relation_count = 0

    for idx, event in enumerate(world_log_events[:120]):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue

        node_id = (
            "logical:event:" + hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:22]
        )
        event_node_by_event_id[event_id] = node_id

        x = _clamp01(
            _safe_float(
                event.get("x", _stable_ratio(event_id, idx * 11 + 3)),
                _stable_ratio(event_id, idx * 11 + 3),
            )
        )
        y = _clamp01(
            _safe_float(
                event.get("y", _stable_ratio(event_id, idx * 11 + 7)),
                _stable_ratio(event_id, idx * 11 + 7),
            )
        )
        importance = _clamp01(_safe_float(event.get("dominant_weight", 0.62), 0.62))

        graph_nodes.append(
            {
                "id": node_id,
                "kind": "event",
                "label": str(event.get("title") or event.get("kind") or event_id),
                "status": str(event.get("status", "recorded") or "recorded"),
                "confidence": round(importance, 4),
                "x": round(x, 4),
                "y": round(y, 4),
                "provenance": {
                    "event_id": event_id,
                    "source": str(event.get("source", "")),
                    "event_kind": str(event.get("kind", "")),
                    "ts": str(event.get("ts", "")),
                    "embedding_id": str(event.get("embedding_id", "")),
                    "refs": [
                        str(item) for item in event.get("refs", []) if str(item).strip()
                    ],
                },
            }
        )

        refs = [str(item) for item in event.get("refs", []) if str(item).strip()]
        for ref in refs[:6]:
            normalized_ref = _normalize_path_for_file_id(ref)
            if not normalized_ref:
                continue
            file_node_id = file_path_to_node.get(normalized_ref)
            if not file_node_id:
                file_id = _file_id_for_path(normalized_ref)
                file_node_id = file_id_to_node.get(file_id)
            if not file_node_id:
                continue
            graph_edges.append(
                {
                    "id": "logical:edge:mentions:"
                    + hashlib.sha256(
                        f"{node_id}|{file_node_id}|{normalized_ref}".encode("utf-8")
                    ).hexdigest()[:20],
                    "source": node_id,
                    "target": file_node_id,
                    "kind": "mentions",
                    "weight": 0.66,
                }
            )
            event_link_count += 1

    for event in world_log_events[:120]:
        if not isinstance(event, dict):
            continue
        source_event_id = str(event.get("id", "")).strip()
        source_node_id = event_node_by_event_id.get(source_event_id)
        if not source_node_id:
            continue
        relations_raw = event.get("relations", [])
        if not isinstance(relations_raw, list):
            continue
        for relation in relations_raw:
            if not isinstance(relation, dict):
                continue
            target_event_id = str(relation.get("event_id", "")).strip()
            target_node_id = event_node_by_event_id.get(target_event_id)
            if not target_node_id or target_node_id == source_node_id:
                continue
            pair = (
                source_event_id
                if source_event_id < target_event_id
                else target_event_id,
                target_event_id
                if source_event_id < target_event_id
                else source_event_id,
            )
            if pair in event_relation_pairs:
                continue
            event_relation_pairs.add(pair)
            graph_edges.append(
                {
                    "id": "logical:edge:correlates:"
                    + hashlib.sha256(
                        f"{source_node_id}|{target_node_id}".encode("utf-8")
                    ).hexdigest()[:20],
                    "source": source_node_id,
                    "target": target_node_id,
                    "kind": "correlates",
                    "weight": round(
                        _clamp01(_safe_float(relation.get("score", 0.4), 0.4)),
                        4,
                    ),
                }
            )
            event_relation_count += 1

    rule_nodes_by_kind: dict[str, str] = {}
    for idx, kind in enumerate(required_kinds):
        kind_text = str(kind).strip()
        if not kind_text:
            continue
        node_id = (
            f"logical:rule:{hashlib.sha256(kind_text.encode('utf-8')).hexdigest()[:20]}"
        )
        x = 0.18 + (_stable_ratio(kind_text, idx) * 0.2)
        y = 0.2 + (_stable_ratio(kind_text, idx + 19) * 0.35)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "rule",
                "label": kind_text,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "confidence": 1.0,
                "provenance": {"required_kind": kind_text},
            }
        )
        rule_nodes_by_kind[kind_text] = node_id

    fact_nodes: list[str] = []
    for idx, claim in enumerate(claims_raw):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or f"claim:{idx}")
        claim_text = str(claim.get("text") or claim_id)
        status = str(claim.get("status", "undecided")).strip() or "undecided"
        kappa = round(_clamp01(_safe_float(claim.get("kappa", 0.0), 0.0)), 4)
        node_id = (
            f"logical:fact:{hashlib.sha256(claim_id.encode('utf-8')).hexdigest()[:22]}"
        )
        radius = 0.14 + (_stable_ratio(claim_id, idx) * 0.09)
        angle = _stable_ratio(claim_id, idx + 7) * math.tau
        x = 0.72 + math.cos(angle) * radius
        y = 0.5 + math.sin(angle) * radius
        proof_refs = [
            str(item).strip()
            for item in claim.get("proof_refs", [])
            if str(item).strip()
        ]
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "fact",
                "label": claim_text,
                "status": status,
                "confidence": kappa,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {
                    "claim_id": claim_id,
                    "proof_refs": proof_refs,
                },
            }
        )
        fact_nodes.append(node_id)

        for ref in proof_refs:
            normalized_ref = _normalize_path_for_file_id(ref)
            file_node_id = file_path_to_node.get(normalized_ref)
            if not file_node_id:
                continue
            graph_edges.append(
                {
                    "id": f"logical:edge:prove:{hashlib.sha256((file_node_id + node_id + ref).encode('utf-8')).hexdigest()[:20]}",
                    "source": file_node_id,
                    "target": node_id,
                    "kind": "proves",
                    "weight": 1.0,
                }
            )

    derivation_nodes: list[str] = []
    for idx, entry in enumerate(proof_entries):
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("ref", "")).strip()
        kind = str(entry.get("kind", "")).strip()
        present = bool(entry.get("present", False))
        detail = str(entry.get("detail", "")).strip()
        base = f"{kind}|{ref}|{idx}"
        node_id = f"logical:derivation:{hashlib.sha256(base.encode('utf-8')).hexdigest()[:20]}"
        x = 0.42 + (_stable_ratio(base, idx) * 0.22)
        y = 0.42 + (_stable_ratio(base, idx + 27) * 0.3)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "derivation",
                "label": detail or ref or kind or f"derivation-{idx + 1}",
                "status": "present" if present else "missing",
                "confidence": 1.0 if present else 0.0,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {
                    "kind": kind,
                    "ref": ref,
                    "present": present,
                },
            }
        )
        derivation_nodes.append(node_id)

        rule_node = rule_nodes_by_kind.get(kind)
        if rule_node:
            graph_edges.append(
                {
                    "id": f"logical:edge:rule:{hashlib.sha256((rule_node + node_id).encode('utf-8')).hexdigest()[:20]}",
                    "source": rule_node,
                    "target": node_id,
                    "kind": "requires",
                    "weight": 0.9,
                }
            )

        if fact_nodes:
            target_fact = fact_nodes[idx % len(fact_nodes)]
            graph_edges.append(
                {
                    "id": f"logical:edge:derive:{hashlib.sha256((node_id + target_fact).encode('utf-8')).hexdigest()[:20]}",
                    "source": node_id,
                    "target": target_fact,
                    "kind": "derives",
                    "weight": 0.82 if present else 0.36,
                }
            )

        normalized_ref = _normalize_path_for_file_id(ref)
        file_node_id = file_path_to_node.get(normalized_ref)
        if file_node_id:
            graph_edges.append(
                {
                    "id": f"logical:edge:source:{hashlib.sha256((file_node_id + node_id + normalized_ref).encode('utf-8')).hexdigest()[:20]}",
                    "source": file_node_id,
                    "target": node_id,
                    "kind": "source",
                    "weight": 0.92,
                }
            )

    gate_target = str(gate.get("target") or "push-truth")
    gate_node_id = (
        f"logical:gate:{hashlib.sha256(gate_target.encode('utf-8')).hexdigest()[:20]}"
    )
    graph_nodes.append(
        {
            "id": gate_node_id,
            "kind": "gate",
            "label": gate_target,
            "status": "blocked" if bool(gate.get("blocked", True)) else "ready",
            "confidence": 1.0,
            "x": 0.76,
            "y": 0.54,
            "provenance": {"target": gate_target},
        }
    )

    for fact_id in fact_nodes:
        graph_edges.append(
            {
                "id": f"logical:edge:gate:{hashlib.sha256((fact_id + gate_node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": fact_id,
                "target": gate_node_id,
                "kind": "feeds",
                "weight": 0.74,
            }
        )

    contradiction_nodes = 0
    gate_reasons = [
        str(item).strip() for item in gate.get("reasons", []) if str(item).strip()
    ]
    for idx, reason in enumerate(gate_reasons[:6]):
        node_id = f"logical:contradiction:{hashlib.sha256(reason.encode('utf-8')).hexdigest()[:20]}"
        x = 0.86 + (_stable_ratio(reason, idx) * 0.1)
        y = 0.42 + (_stable_ratio(reason, idx + 33) * 0.24)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "contradiction",
                "label": reason,
                "status": "active",
                "confidence": 1.0,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {"reason": reason},
            }
        )
        graph_edges.append(
            {
                "id": f"logical:edge:block:{hashlib.sha256((node_id + gate_node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": node_id,
                "target": gate_node_id,
                "kind": "blocks",
                "weight": 1.0,
            }
        )
        contradiction_nodes += 1

    for node in graph_nodes:
        if node.get("kind") != "fact" or str(node.get("status")) != "refuted":
            continue
        reason = str(node.get("label", "refuted-fact"))
        node_id = f"logical:contradiction:{hashlib.sha256((reason + ':fact').encode('utf-8')).hexdigest()[:20]}"
        x = _clamp01(_safe_float(node.get("x", 0.5), 0.5) + 0.08)
        y = _clamp01(_safe_float(node.get("y", 0.5), 0.5) + 0.04)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "contradiction",
                "label": reason,
                "status": "refuted",
                "confidence": 1.0,
                "x": round(x, 4),
                "y": round(y, 4),
                "provenance": {"from_fact": str(node.get("id", ""))},
            }
        )
        graph_edges.append(
            {
                "id": f"logical:edge:contradict:{hashlib.sha256((str(node.get('id')) + node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": str(node.get("id", "")),
                "target": node_id,
                "kind": "contradicts",
                "weight": 1.0,
            }
        )
        contradiction_nodes += 1

    return {
        "record": "ημ.logical-graph.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": graph_nodes,
        "edges": graph_edges,
        "joins": {
            "file_ids": sorted(file_id_to_node.keys()),
            "file_index": {
                path: _file_id_for_path(path)
                for path in sorted(file_path_to_node.keys())
            },
            "source_to_file": dict(sorted(joins_source_to_file.items())),
        },
        "stats": {
            "file_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "file"]
            ),
            "tag_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "tag"]
            ),
            "fact_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "fact"]
            ),
            "event_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "event"]
            ),
            "rule_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "rule"]
            ),
            "derivation_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "derivation"]
            ),
            "contradiction_nodes": contradiction_nodes,
            "gate_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "gate"]
            ),
            "tag_edges": len(
                [
                    edge
                    for edge in graph_edges
                    if str(edge.get("kind", "")).strip().lower()
                    in {"labeled_as", "relates_tag"}
                ]
            ),
            "event_links": event_link_count,
            "event_relations": event_relation_count,
            "edge_count": len(graph_edges),
        },
    }


def _build_pain_field(
    catalog: dict[str, Any], logical_graph: dict[str, Any]
) -> dict[str, Any]:
    failures_raw = catalog.get("test_failures", []) if isinstance(catalog, dict) else []
    coverage_raw = catalog.get("test_coverage", {}) if isinstance(catalog, dict) else {}
    if not isinstance(failures_raw, list):
        failures_raw = []
    if not isinstance(coverage_raw, dict):
        coverage_raw = {}

    nodes = logical_graph.get("nodes", []) if isinstance(logical_graph, dict) else []
    edges = logical_graph.get("edges", []) if isinstance(logical_graph, dict) else []
    joins = logical_graph.get("joins", {}) if isinstance(logical_graph, dict) else {}
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    if not isinstance(joins, dict):
        joins = {}

    node_by_id = {
        str(node.get("id", "")): node
        for node in nodes
        if isinstance(node, dict) and str(node.get("id", "")).strip()
    }
    file_index = joins.get("file_index", {})
    if not isinstance(file_index, dict):
        file_index = {}
    file_id_to_path: dict[str, str] = {}
    for path_key, file_id_value in file_index.items():
        normalized_path = _normalize_path_for_file_id(str(path_key))
        file_id_key = str(file_id_value).strip()
        if normalized_path and file_id_key:
            file_id_to_path[file_id_key] = normalized_path
    file_id_to_node = {
        str(node.get("file_id", "")): str(node.get("id", ""))
        for node in nodes
        if isinstance(node, dict)
        and str(node.get("kind", "")) == "file"
        and str(node.get("file_id", "")).strip()
    }

    region_rows: list[dict[str, Any]] = []
    region_by_id: dict[str, dict[str, Any]] = {}
    region_by_file_id: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("kind", "")).strip() != "file":
            continue
        file_id = str(node.get("file_id", "")).strip()
        node_id = str(node.get("id", "")).strip()
        if not file_id or not node_id:
            continue
        region_key = str(node.get("path") or node.get("label") or file_id)
        region_seed = f"world-web|node|{node_id}|{region_key}"
        region_id = _stable_entity_id("region", region_seed)
        region_row: dict[str, Any] = {
            "region_id": region_id,
            "region_kind": "node",
            "region_key": region_key,
            "node_id": node_id,
            "file_id": file_id,
            "x": round(_clamp01(_safe_float(node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(node.get("y", 0.5), 0.5)), 4),
            "label": str(node.get("label", "")),
        }
        region_rows.append(region_row)
        region_by_id[region_id] = region_row
        region_by_file_id[file_id] = region_id

    region_rows.sort(key=lambda row: str(row.get("region_id", "")))

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        weight = _clamp01(_safe_float(edge.get("weight", 0.4), 0.4))
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight * 0.92))

    coverage_by_test = coverage_raw.get("by_test", {})
    if not isinstance(coverage_by_test, dict):
        coverage_by_test = {}

    coverage_by_test_spans = coverage_raw.get("by_test_spans", {})
    if not isinstance(coverage_by_test_spans, dict):
        coverage_by_test_spans = {}

    coverage_by_test_lower: dict[str, Any] = {}
    for key, value in coverage_by_test.items():
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            continue
        coverage_by_test_lower[normalized_key] = value

    coverage_by_test_spans_lower: dict[str, Any] = {}
    for key, value in coverage_by_test_spans.items():
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            continue
        coverage_by_test_spans_lower[normalized_key] = value

    hottest_files_raw = coverage_raw.get("hottest_files", [])
    hottest_files: list[str] = []
    if isinstance(hottest_files_raw, list):
        hottest_files = [str(path) for path in hottest_files_raw if str(path).strip()]
    if not hottest_files:
        files_metrics = coverage_raw.get("files", {})
        if isinstance(files_metrics, dict):
            scored_paths: list[tuple[str, float, float]] = []
            for path_key, metrics in files_metrics.items():
                path_text = str(path_key).strip()
                if not path_text:
                    continue
                line_rate = _clamp01(
                    _safe_float(
                        metrics.get("line_rate", 0.0)
                        if isinstance(metrics, dict)
                        else 0.0,
                        0.0,
                    )
                )
                lines_found = _safe_float(
                    metrics.get("lines_found", 0.0)
                    if isinstance(metrics, dict)
                    else 0.0,
                    0.0,
                )
                uncovered = max(0.0, 1.0 - line_rate)
                scored_paths.append((path_text, uncovered, lines_found))
            hottest_files = [
                path
                for path, _, _ in sorted(
                    scored_paths,
                    key=lambda row: (-row[1], -row[2], row[0]),
                )
            ]

    hottest_file_rank: dict[str, int] = {}
    for index, path_key in enumerate(hottest_files):
        normalized = _normalize_path_for_file_id(path_key)
        if normalized and normalized not in hottest_file_rank:
            hottest_file_rank[normalized] = index

    failing_tests: list[dict[str, Any]] = []
    test_span_weights: dict[tuple[str, str], float] = {}
    span_region_weights: dict[str, dict[str, float]] = defaultdict(dict)
    span_rows_by_id: dict[str, dict[str, Any]] = {}
    region_heat_raw: dict[str, float] = defaultdict(float)
    seeded_node_heat: dict[str, float] = defaultdict(float)

    for idx, row in enumerate(failures_raw):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "failed")).strip().lower()
        if status not in {"failed", "error", "xfailed", "failing"}:
            continue
        test_name = str(
            row.get("name") or row.get("test") or row.get("nodeid") or f"test-{idx + 1}"
        ).strip()
        if not test_name:
            continue
        message = str(row.get("message") or row.get("error") or "")

        coverage_sources: list[Any] = []
        for key in (
            "covered_spans",
            "spans",
            "coverage_spans",
            "covered_files",
            "files",
            "coverage",
        ):
            value = row.get(key)
            if value is not None:
                coverage_sources.append(value)

        from_coverage_spans = coverage_by_test_spans.get(test_name)
        if from_coverage_spans is None:
            from_coverage_spans = coverage_by_test_spans_lower.get(test_name.lower())
        if from_coverage_spans is not None:
            coverage_sources.append(from_coverage_spans)

        from_coverage = coverage_by_test.get(test_name)
        if from_coverage is None:
            from_coverage = coverage_by_test_lower.get(test_name.lower())
        if from_coverage is not None:
            coverage_sources.append(from_coverage)

        if not coverage_sources and hottest_files:
            coverage_sources.append(hottest_files[:3])

        normalized_spans: list[dict[str, Any]] = []
        for source in coverage_sources:
            normalized_spans.extend(_extract_coverage_spans(source))
        if not normalized_spans:
            continue

        severity = max(0.0, _safe_float(row.get("severity", 1.0), 1.0))
        signal_w = _clamp01(
            _safe_float(
                row.get(
                    "signal_w",
                    row.get("signal_weight", row.get("signal/w", 1.0)),
                ),
                1.0,
            )
        )
        suite_name = str(
            row.get("suite") or row.get("module") or row.get("file") or ""
        ).strip()
        runner_name = str(
            row.get("runner") or row.get("framework") or row.get("tool") or ""
        ).strip()
        test_id = _stable_entity_id(
            "test",
            f"{test_name}|{suite_name}|{runner_name}",
        )

        span_weights_for_test: dict[str, float] = defaultdict(float)
        covered_paths: set[str] = set()
        covered_file_ids: set[str] = set()

        for span in normalized_spans:
            path_value = _normalize_path_for_file_id(str(span.get("path") or ""))
            if not path_value:
                continue
            file_id = _file_id_for_path(path_value)
            if not file_id:
                continue

            start_line = max(1, _safe_int(span.get("start_line", 1), 1))
            end_line = max(
                start_line,
                _safe_int(span.get("end_line", start_line), start_line),
            )
            symbol = str(span.get("symbol", "")).strip()
            weight_raw = max(0.0, _safe_float(span.get("weight", 1.0), 1.0))
            if weight_raw <= 0.0:
                weight_raw = 1.0

            span_id = _stable_entity_id(
                "span",
                f"{file_id}|{start_line}|{end_line}|{symbol}",
            )
            span_weights_for_test[span_id] += weight_raw
            covered_paths.add(path_value)
            covered_file_ids.add(file_id)

            span_row = span_rows_by_id.get(span_id)
            if span_row is None:
                span_row = {
                    "id": span_id,
                    "file_id": file_id,
                    "path": path_value,
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol": symbol,
                }
                span_rows_by_id[span_id] = span_row

            region_id = region_by_file_id.get(file_id, "")
            if region_id:
                span_region_weights.setdefault(span_id, {})[region_id] = max(
                    span_region_weights.get(span_id, {}).get(region_id, 0.0),
                    1.0,
                )

        if not span_weights_for_test:
            continue

        total_span_weight = sum(span_weights_for_test.values())
        if total_span_weight <= 0.0:
            total_span_weight = float(len(span_weights_for_test))

        region_ids_for_test: set[str] = set()
        for span_id, raw_weight in sorted(span_weights_for_test.items()):
            edge_weight = (
                raw_weight / total_span_weight if total_span_weight > 0 else 0.0
            )
            if edge_weight <= 0.0:
                continue
            test_span_key = (test_id, span_id)
            test_span_weights[test_span_key] = max(
                test_span_weights.get(test_span_key, 0.0),
                edge_weight,
            )

            for region_id, region_weight in sorted(
                span_region_weights.get(span_id, {}).items()
            ):
                region_ids_for_test.add(region_id)
                contrib = severity * signal_w * edge_weight * max(0.0, region_weight)
                if contrib <= 0.0:
                    continue
                region_heat_raw[region_id] += contrib
                region_info = region_by_id.get(region_id, {})
                node_id = str(region_info.get("node_id", "")).strip()
                if node_id:
                    seeded_node_heat[node_id] += contrib

        span_ids_sorted = sorted(span_weights_for_test.keys())
        normalized_files = sorted(covered_paths)
        file_ids = sorted(covered_file_ids)
        if not normalized_files and hottest_files:
            normalized_files = [
                _normalize_path_for_file_id(path)
                for path in hottest_files[:3]
                if _normalize_path_for_file_id(path)
            ]

        failing_tests.append(
            {
                "id": test_id,
                "name": test_name,
                "status": status,
                "message": message,
                "severity": round(severity, 4),
                "signal_w": round(signal_w, 4),
                "failure_glyph": "破",
                "covered_files": normalized_files,
                "file_ids": file_ids,
                "span_ids": span_ids_sorted,
                "region_ids": sorted(region_ids_for_test),
            }
        )

    node_heat: dict[str, float] = {
        node_id: _clamp01(_safe_float(heat, 0.0))
        for node_id, heat in seeded_node_heat.items()
        if _safe_float(heat, 0.0) > 0.0
    }

    hop_decay = 0.58
    max_hops = 4
    current_frontier = sorted(node_heat.items(), key=lambda row: row[0])
    for _hop in range(max_hops):
        next_frontier: list[tuple[str, float]] = []
        for node_id, heat in current_frontier:
            if heat <= 0.02:
                continue
            for neighbor_id, edge_weight in adjacency.get(node_id, []):
                next_heat = _clamp01(heat * hop_decay * max(0.1, edge_weight))
                if next_heat <= 0.01:
                    continue
                if next_heat <= node_heat.get(neighbor_id, 0.0) + 0.004:
                    continue
                node_heat[neighbor_id] = next_heat
                next_frontier.append((neighbor_id, next_heat))
        current_frontier = next_frontier
        if not current_frontier:
            break

    def _heat_sort_key(item: tuple[str, float]) -> tuple[float, int, str]:
        node_id, heat_value = item
        node = node_by_id.get(node_id, {})
        if not isinstance(node, dict):
            node = {}
        file_id = str(node.get("file_id", "")).strip()
        path_value = file_id_to_path.get(file_id, "")
        if not path_value:
            path_value = _normalize_path_for_file_id(str(node.get("path", "")))
        rank = hottest_file_rank.get(path_value, 1_000_000)
        return (-_clamp01(_safe_float(heat_value, 0.0)), rank, node_id)

    heat_nodes: list[dict[str, Any]] = []
    for node_id, heat in sorted(node_heat.items(), key=_heat_sort_key):
        node = node_by_id.get(node_id, {})
        if not isinstance(node, dict):
            node = {}
        node_file_id = str(node.get("file_id", "")).strip()
        node_path = _normalize_path_for_file_id(str(node.get("path", "")))
        if node_file_id and node_path and node_file_id not in file_id_to_path:
            file_id_to_path[node_file_id] = node_path
        heat_nodes.append(
            {
                "node_id": node_id,
                "kind": str(node.get("kind", "unknown")),
                "heat": round(_clamp01(heat), 4),
                "x": round(_clamp01(_safe_float(node.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(node.get("y", 0.5), 0.5)), 4),
                "file_id": str(node.get("file_id", "")),
                "label": str(node.get("label", "")),
            }
        )

    debug_target: dict[str, Any] = {
        "meaning": "DEBUG",
        "glyph": "診",
        "grounded": False,
        "source": "none",
        "node_id": "",
        "file_id": "",
        "region_id": "",
        "path": "",
        "label": "",
        "heat": 0.0,
        "x": 0.5,
        "y": 0.5,
        "reason": "no-active-failure-signal",
    }

    hottest_node = next(
        (
            row
            for row in heat_nodes
            if str(row.get("node_id", "")).strip()
            or str(row.get("file_id", "")).strip()
        ),
        None,
    )
    if isinstance(hottest_node, dict):
        node_id = str(hottest_node.get("node_id", "")).strip()
        file_id = str(hottest_node.get("file_id", "")).strip()
        node = node_by_id.get(node_id, {}) if node_id else {}
        if not isinstance(node, dict):
            node = {}

        path_value = file_id_to_path.get(file_id, "")
        if not path_value:
            path_value = _normalize_path_for_file_id(str(node.get("path", "")))
        if file_id and path_value and file_id not in file_id_to_path:
            file_id_to_path[file_id] = path_value

        label_value = str(hottest_node.get("label", "")).strip()
        if not label_value:
            label_value = str(node.get("label", "")).strip()
        if not label_value and path_value:
            label_value = Path(path_value).name

        debug_target = {
            "meaning": "DEBUG",
            "glyph": "診",
            "grounded": True,
            "source": "pain_field.max_heat",
            "node_id": node_id,
            "file_id": file_id,
            "region_id": region_by_file_id.get(file_id, ""),
            "path": path_value,
            "label": label_value,
            "heat": round(_clamp01(_safe_float(hottest_node.get("heat", 0.0), 0.0)), 4),
            "x": round(_clamp01(_safe_float(hottest_node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(hottest_node.get("y", 0.5), 0.5)), 4),
            "reason": "points-to-hottest-file",
        }
    elif hottest_files:
        fallback_path = _normalize_path_for_file_id(str(hottest_files[0]))
        fallback_file_id = _file_id_for_path(fallback_path) if fallback_path else ""
        fallback_node_id = file_id_to_node.get(fallback_file_id, "")
        fallback_node = node_by_id.get(fallback_node_id, {}) if fallback_node_id else {}
        if not isinstance(fallback_node, dict):
            fallback_node = {}
        if (
            fallback_file_id
            and fallback_path
            and fallback_file_id not in file_id_to_path
        ):
            file_id_to_path[fallback_file_id] = fallback_path

        label_value = str(fallback_node.get("label", "")).strip()
        if not label_value and fallback_path:
            label_value = Path(fallback_path).name

        debug_target = {
            "meaning": "DEBUG",
            "glyph": "診",
            "grounded": bool(fallback_path),
            "source": "coverage.hottest_files",
            "node_id": fallback_node_id,
            "file_id": fallback_file_id,
            "region_id": region_by_file_id.get(fallback_file_id, ""),
            "path": fallback_path,
            "label": label_value,
            "heat": 0.0,
            "x": round(_clamp01(_safe_float(fallback_node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(fallback_node.get("y", 0.5), 0.5)), 4),
            "reason": "fallback-to-coverage-hottest-file",
        }

    heat_regions: list[dict[str, Any]] = []
    for region_id, raw_heat in sorted(
        region_heat_raw.items(), key=lambda row: (-row[1], row[0])
    ):
        if raw_heat <= 0.0:
            continue
        region = region_by_id.get(region_id, {})
        heat_regions.append(
            {
                "region_id": region_id,
                "node_id": str(region.get("node_id", "")),
                "file_id": str(region.get("file_id", "")),
                "heat": round(_clamp01(_safe_float(raw_heat, 0.0)), 4),
                "heat_raw": round(max(0.0, _safe_float(raw_heat, 0.0)), 6),
                "glyph": "熱",
            }
        )

    span_rows = sorted(
        span_rows_by_id.values(),
        key=lambda row: (
            str(row.get("path", "")),
            int(row.get("start_line", 0)),
            int(row.get("end_line", 0)),
            str(row.get("id", "")),
        ),
    )

    test_covers_span_rows: list[dict[str, Any]] = []
    for (test_id, span_id), weight in sorted(
        test_span_weights.items(), key=lambda row: (row[0][0], row[0][1])
    ):
        test_covers_span_rows.append(
            {
                "id": _stable_entity_id(
                    "edge", f"{test_id}|{span_id}|覆/test-covers-span"
                ),
                "rel": "覆/test-covers-span",
                "test_id": test_id,
                "span_id": span_id,
                "w": round(_clamp01(_safe_float(weight, 0.0)), 6),
            }
        )

    span_maps_region_rows: list[dict[str, Any]] = []
    for span_id, region_weights in sorted(span_region_weights.items()):
        for region_id, weight in sorted(region_weights.items()):
            span_maps_region_rows.append(
                {
                    "id": _stable_entity_id(
                        "edge", f"{span_id}|{region_id}|覆/span-maps-to-region"
                    ),
                    "rel": "覆/span-maps-to-region",
                    "span_id": span_id,
                    "region_id": region_id,
                    "w": round(_clamp01(_safe_float(weight, 0.0)), 6),
                }
            )

    max_heat = max((row.get("heat", 0.0) for row in heat_nodes), default=0.0)
    return {
        "record": "ημ.pain-field.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": bool(failing_tests),
        "decay": hop_decay,
        "hops": max_hops,
        "failing_tests": failing_tests,
        "spans": span_rows,
        "regions": region_rows,
        "relations": {
            "覆/test-covers-span": test_covers_span_rows,
            "覆/span-maps-to-region": span_maps_region_rows,
        },
        "heat_regions": heat_regions,
        "glyphs": {
            "locus": "址",
            "heat": "熱",
            "coverage": "覆",
            "failure": "破",
            "debug": "診",
        },
        "debug": debug_target,
        "grounded_meanings": {"DEBUG": debug_target},
        "node_heat": heat_nodes,
        "max_heat": round(_clamp01(_safe_float(max_heat, 0.0)), 4),
        "join_key": "file_id=sha256(normalized_path)",
        "region_join_key": "region_id=sha256(world|region_kind|region_key)",
    }


def _materialize_heat_values(
    catalog: dict[str, Any], pain_field: dict[str, Any]
) -> dict[str, Any]:
    named_fields = catalog.get("named_fields", []) if isinstance(catalog, dict) else []
    if not isinstance(named_fields, list):
        named_fields = []

    by_presence: dict[str, dict[str, Any]] = {}
    for row in named_fields:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("id", "")).strip()
        if presence_id:
            by_presence[presence_id] = row

    for entity in ENTITY_MANIFEST:
        if not isinstance(entity, dict):
            continue
        presence_id = str(entity.get("id", "")).strip()
        if presence_id and presence_id not in by_presence:
            by_presence[presence_id] = entity

    field_anchors: dict[str, tuple[float, float]] = {}
    region_meta: dict[str, dict[str, Any]] = {}
    for field_id, presence_id in FIELD_TO_PRESENCE.items():
        item = by_presence.get(presence_id, {})
        x = _clamp01(_safe_float(item.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(item.get("y", 0.5), 0.5))
        field_anchors[field_id] = (x, y)
        region_meta[field_id] = {
            "region_id": field_id,
            "presence_id": presence_id,
            "en": str(item.get("en", presence_id)),
            "ja": str(item.get("ja", "")),
            "x": round(x, 4),
            "y": round(y, 4),
        }

    node_heat_rows = (
        pain_field.get("node_heat", []) if isinstance(pain_field, dict) else []
    )
    if not isinstance(node_heat_rows, list):
        node_heat_rows = []

    region_heat_raw: dict[str, float] = {field_id: 0.0 for field_id in field_anchors}
    locate_rows: list[dict[str, Any]] = []
    for row in node_heat_rows[:240]:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("node_id", "")).strip()
        if not entity_id:
            continue
        heat = _clamp01(_safe_float(row.get("heat", 0.0), 0.0))
        if heat <= 0.0:
            continue

        x = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        locate_scores = _field_scores_from_position(x, y, field_anchors)
        ranked_scores = sorted(
            locate_scores.items(),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )

        for field_id, locate_weight in ranked_scores:
            region_heat_raw[field_id] += heat * _clamp01(
                _safe_float(locate_weight, 0.0)
            )

        for field_id, locate_weight in ranked_scores[:4]:
            locate_rows.append(
                {
                    "kind": "址",
                    "entity_id": entity_id,
                    "region_id": field_id,
                    "weight": round(_clamp01(_safe_float(locate_weight, 0.0)), 4),
                }
            )

    max_raw_heat = max(region_heat_raw.values(), default=0.0)
    regions: list[dict[str, Any]] = []
    for rank, (field_id, raw_heat) in enumerate(
        sorted(region_heat_raw.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        value = 0.0
        if max_raw_heat > 0.0:
            value = _clamp01(raw_heat / max_raw_heat)
        meta = region_meta.get(
            field_id,
            {
                "region_id": field_id,
                "presence_id": FIELD_TO_PRESENCE.get(field_id, ""),
                "en": field_id,
                "ja": "",
                "x": 0.5,
                "y": 0.5,
            },
        )
        regions.append(
            {
                **meta,
                "rank": rank,
                "raw": round(max(0.0, raw_heat), 6),
                "value": round(value, 4),
            }
        )

    facts = [
        {
            "kind": "熱/value",
            "region_id": row.get("region_id", ""),
            "value": row.get("value", 0.0),
            "raw": row.get("raw", 0.0),
        }
        for row in regions
    ]
    return {
        "record": "ημ.heat-values.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": max_raw_heat > 0.0,
        "source": "pain_field.node_heat",
        "regions": regions,
        "facts": facts,
        "locate": locate_rows,
        "max_raw": round(max(0.0, max_raw_heat), 6),
    }


def build_named_field_overlays(
    entity_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for entity in entity_manifest:
        key = str(entity.get("id", "")).strip()
        if key:
            by_id[key] = entity

    overlays: list[dict[str, Any]] = []
    canonical_presence_ids = [
        "receipt_river",
        "witness_thread",
        "fork_tax_canticle",
        "mage_of_receipts",
        "keeper_of_receipts",
        "anchor_registry",
        "gates_of_truth",
    ]
    for idx, field_id in enumerate(canonical_presence_ids):
        item = by_id.get(field_id)
        if item is None:
            continue

        hue = int(item.get("hue", 200))
        overlays.append(
            {
                "id": field_id,
                "en": str(item.get("en", field_id.replace("_", " ").title())),
                "ja": str(item.get("ja", "")),
                "type": str(item.get("type", "flow")),
                "x": float(item.get("x", 0.5)),
                "y": float(item.get("y", 0.5)),
                "freq": float(item.get("freq", 220.0)),
                "hue": hue,
                "gradient": {
                    "mode": "radial",
                    "radius": round(0.2 + (idx % 3) * 0.035, 3),
                    "stops": [
                        {
                            "offset": 0.0,
                            "color": f"hsla({hue}, 88%, 74%, 0.36)",
                        },
                        {
                            "offset": 0.52,
                            "color": f"hsla({hue}, 76%, 58%, 0.2)",
                        },
                        {
                            "offset": 1.0,
                            "color": f"hsla({(hue + 28) % 360}, 72%, 44%, 0.0)",
                        },
                    ],
                },
                "motion": {
                    "drift_hz": round(0.07 + idx * 0.013, 3),
                    "wobble_px": 5 + (idx % 4) * 3,
                },
            }
        )

    return overlays


def _mix_fingerprint(catalog: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in catalog.get("items", []):
        rel_path = str(item.get("rel_path", ""))
        if rel_path.lower().endswith(".wav"):
            rows.append(
                "|".join(
                    [
                        rel_path,
                        str(item.get("bytes", 0)),
                        str(item.get("mtime_utc", "")),
                    ]
                )
            )
    rows.sort()
    return sha1("\n".join(rows).encode("utf-8")).hexdigest()


def _collect_mix_sources(catalog: dict[str, Any], vault_root: Path) -> list[Path]:
    paths: list[Path] = []
    for item in catalog.get("items", []):
        rel_path = str(item.get("rel_path", ""))
        if not rel_path.lower().endswith(".wav"):
            continue
        candidate = (vault_root / rel_path).resolve()
        if candidate.exists() and candidate.is_file():
            paths.append(candidate)
    return paths


def _mix_wav_sources(sources: list[Path]) -> tuple[bytes, dict[str, Any]]:
    if not sources:
        return b"", {"sources": 0, "sample_rate": 0, "duration_seconds": 0.0}

    sample_rate = 44100
    clips: list[tuple[array, int]] = []
    max_frames = 0

    for src in sources:
        with wave.open(str(src), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            if sampwidth != 2:
                continue
            if channels not in (1, 2):
                continue

            frames_raw = wf.readframes(wf.getnframes())
            pcm = array("h")
            pcm.frombytes(frames_raw)
            frames = len(pcm) // channels
            if frames == 0:
                continue

            sample_rate = framerate
            clips.append((pcm, channels))
            if frames > max_frames:
                max_frames = frames

    if not clips or max_frames == 0:
        return b"", {"sources": 0, "sample_rate": 0, "duration_seconds": 0.0}

    gain = 1.0 / max(1, len(clips))
    mix = [0] * (max_frames * 2)

    for pcm, channels in clips:
        if channels == 1:
            frame_count = len(pcm)
            for i in range(frame_count):
                value = int(pcm[i] * gain)
                idx = i * 2
                mix[idx] += value
                mix[idx + 1] += value
            continue

        frame_count = len(pcm) // 2
        for i in range(frame_count):
            src_idx = i * 2
            dst_idx = i * 2
            mix[dst_idx] += int(pcm[src_idx] * gain)
            mix[dst_idx + 1] += int(pcm[src_idx + 1] * gain)

    out = array("h")
    for value in mix:
        if value > 32767:
            out.append(32767)
        elif value < -32768:
            out.append(-32768)
        else:
            out.append(value)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf_out:
        wf_out.setnchannels(2)
        wf_out.setsampwidth(2)
        wf_out.setframerate(sample_rate)
        wf_out.writeframes(out.tobytes())

    meta = {
        "sources": len(clips),
        "sample_rate": sample_rate,
        "duration_seconds": round(max_frames / sample_rate, 3),
    }
    return buffer.getvalue(), meta


def build_mix_stream(
    catalog: dict[str, Any], vault_root: Path
) -> tuple[bytes, dict[str, Any]]:
    fingerprint = _mix_fingerprint(catalog)
    with _MIX_CACHE_LOCK:
        cached_fingerprint = str(_MIX_CACHE.get("fingerprint", ""))
        cached_wav = _MIX_CACHE.get("wav", b"")
        cached_meta = _MIX_CACHE.get("meta", {})
        if (
            cached_fingerprint == fingerprint
            and isinstance(cached_wav, (bytes, bytearray))
            and bool(cached_wav)
        ):
            return bytes(cached_wav), (
                dict(cached_meta) if isinstance(cached_meta, dict) else {}
            )

    sources = _collect_mix_sources(catalog, vault_root)
    wav, meta = _mix_wav_sources(sources)
    meta["fingerprint"] = fingerprint

    with _MIX_CACHE_LOCK:
        _MIX_CACHE["fingerprint"] = fingerprint
        _MIX_CACHE["wav"] = wav
        _MIX_CACHE["meta"] = dict(meta)
    return wav, meta


def websocket_accept_value(client_key: str) -> str:
    digest = sha1((client_key + WS_MAGIC).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def websocket_frame_text(message: str) -> bytes:
    payload = message.encode("utf-8")
    length = len(payload)
    header = bytearray([0x81])
    if length <= 125:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def _weaver_probe_host(bind_host: str) -> str:
    host = str(bind_host or "127.0.0.1").strip()
    return "127.0.0.1" if host == "0.0.0.0" else host


def _weaver_health_check(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
            return True
    except:
        return False


def _weaver_service_base_url() -> str:
    from .constants import WEAVER_HOST_ENV, WEAVER_PORT

    return f"http://{_weaver_probe_host(WEAVER_HOST_ENV or '127.0.0.1')}:{WEAVER_PORT}"


def _read_weaver_snapshot_file(part_root: Path) -> dict[str, Any] | None:
    p = (part_root / "world_state" / "web_graph_weaver.snapshot.json").resolve()
    if not p.exists() or not p.is_file():
        return None
    try:
        pay = json.loads(p.read_text("utf-8"))
        if isinstance(pay, dict):
            return {
                "ok": True,
                "graph": pay.get("graph", {}),
                "status": pay.get("status", {}),
                "source": str(p),
            }
    except:
        pass
    return None


def _fetch_weaver_graph_payload(part_root: Path) -> dict[str, Any]:
    def _graph_node_count(graph_payload: dict[str, Any]) -> int:
        counts = graph_payload.get("counts", {})
        if isinstance(counts, dict):
            from_counts = int(_safe_float(counts.get("nodes_total", 0), 0.0))
            if from_counts > 0:
                return from_counts
        nodes = graph_payload.get("nodes", [])
        return len(nodes) if isinstance(nodes, list) else 0

    base = _weaver_service_base_url()
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    if _weaver_health_check(
        host,
        WEAVER_PORT,
        timeout_s=WEAVER_GRAPH_HEALTH_TIMEOUT_SECONDS,
    ):
        try:
            with urlopen(
                Request(
                    f"{base}/api/weaver/graph?node_limit={WEAVER_GRAPH_NODE_LIMIT}&edge_limit={WEAVER_GRAPH_EDGE_LIMIT}",
                    method="GET",
                ),
                timeout=WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS,
            ) as response:
                graph_payload = json.loads(
                    response.read().decode("utf-8", errors="ignore")
                )
            with urlopen(
                Request(f"{base}/api/weaver/status", method="GET"),
                timeout=WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS,
            ) as response:
                status_payload = json.loads(
                    response.read().decode("utf-8", errors="ignore")
                )

            graph = (
                graph_payload.get("graph", {})
                if isinstance(graph_payload, dict)
                else {}
            )
            status = status_payload if isinstance(status_payload, dict) else {}
            if isinstance(graph, dict):
                live_nodes = _graph_node_count(graph)
                if live_nodes <= 0:
                    fallback = _read_weaver_snapshot_file(part_root)
                    if fallback is not None:
                        fallback_graph = fallback.get("graph", {})
                        if isinstance(fallback_graph, dict):
                            fallback_nodes = _graph_node_count(fallback_graph)
                            if fallback_nodes > 0:
                                fallback_status = fallback.get("status", {})
                                merged_status: dict[str, Any] = {}
                                if isinstance(fallback_status, dict):
                                    merged_status.update(fallback_status)
                                if isinstance(status, dict):
                                    merged_status.update(status)
                                return {
                                    "ok": True,
                                    "graph": fallback_graph,
                                    "status": merged_status,
                                    "source": str(fallback.get("source", "")),
                                }
                return {
                    "ok": True,
                    "graph": graph,
                    "status": status,
                    "source": f"{base}/api/weaver/graph",
                }
        except Exception:
            pass

    fallback = _read_weaver_snapshot_file(part_root)
    if fallback is not None:
        return fallback

    return {
        "ok": False,
        "graph": {"nodes": [], "edges": [], "counts": {}},
        "status": {},
        "source": "",
    }


def _json_deep_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _bounded_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]


def _compact_embed_layer_points(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact_rows: list[dict[str, Any]] = []
    for row in value[:SIMULATION_FILE_GRAPH_EMBED_LAYER_POINT_CAP]:
        if not isinstance(row, dict):
            continue
        embed_ids_raw = row.get("embed_ids", [])
        embed_ids = (
            [
                str(embed_id).strip()
                for embed_id in embed_ids_raw[:SIMULATION_FILE_GRAPH_EMBED_IDS_CAP]
                if str(embed_id).strip()
            ]
            if isinstance(embed_ids_raw, list)
            else []
        )
        compact_rows.append(
            {
                "id": str(row.get("id", "")).strip(),
                "key": str(row.get("key", "")).strip(),
                "x": round(_clamp01(_safe_float(row.get("x", 0.5), 0.5)), 5),
                "y": round(_clamp01(_safe_float(row.get("y", 0.5), 0.5)), 5),
                "hue": round(_safe_float(row.get("hue", 210.0), 210.0), 3),
                "active": bool(row.get("active", True)),
                "embed_ids": embed_ids,
            }
        )
    return compact_rows


def _compact_file_graph_node(node: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        key: node[key] for key in SIMULATION_FILE_GRAPH_NODE_FIELDS if key in node
    }
    compact["x"] = round(_clamp01(_safe_float(compact.get("x", 0.5), 0.5)), 6)
    compact["y"] = round(_clamp01(_safe_float(compact.get("y", 0.5), 0.5)), 6)
    compact["hue"] = int(round(_safe_float(compact.get("hue", 200.0), 200.0))) % 360
    compact["importance"] = round(
        _clamp01(_safe_float(compact.get("importance", 0.24), 0.24)),
        6,
    )

    compact["summary"] = _bounded_text(
        compact.get("summary", ""),
        limit=SIMULATION_FILE_GRAPH_SUMMARY_CHARS,
    )
    compact["text_excerpt"] = _bounded_text(
        compact.get("text_excerpt", ""),
        limit=SIMULATION_FILE_GRAPH_EXCERPT_CHARS,
    )

    tags_raw = compact.get("tags", [])
    compact["tags"] = (
        [str(tag).strip() for tag in tags_raw[:16] if str(tag).strip()]
        if isinstance(tags_raw, list)
        else []
    )
    labels_raw = compact.get("labels", [])
    compact["labels"] = (
        [str(label).strip() for label in labels_raw[:16] if str(label).strip()]
        if isinstance(labels_raw, list)
        else []
    )

    field_scores_raw = compact.get("field_scores", {})
    if isinstance(field_scores_raw, dict):
        compact["field_scores"] = {
            str(key).strip(): round(_clamp01(_safe_float(value, 0.0)), 6)
            for key, value in list(field_scores_raw.items())[:24]
            if str(key).strip()
        }
    else:
        compact["field_scores"] = {}

    embedding_links_raw = compact.get("embedding_links", [])
    compact["embedding_links"] = (
        [
            str(link).strip()
            for link in embedding_links_raw[:SIMULATION_FILE_GRAPH_EMBED_LINK_CAP]
            if str(link).strip()
        ]
        if isinstance(embedding_links_raw, list)
        else []
    )

    compact["embed_layer_points"] = _compact_embed_layer_points(
        compact.get("embed_layer_points", [])
    )
    compact["embed_layer_count"] = int(
        _safe_int(compact.get("embed_layer_count", 0), 0)
    )
    return compact


def _compact_file_graph_nodes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_compact_file_graph_node(node) for node in value if isinstance(node, dict)]


def _compact_file_graph_render_node(node: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: node[key]
        for key in SIMULATION_FILE_GRAPH_RENDER_NODE_FIELDS
        if key in node
    }
    compact["x"] = round(_clamp01(_safe_float(compact.get("x", 0.5), 0.5)), 6)
    compact["y"] = round(_clamp01(_safe_float(compact.get("y", 0.5), 0.5)), 6)
    compact["hue"] = int(round(_safe_float(compact.get("hue", 200.0), 200.0))) % 360
    compact["importance"] = round(
        _clamp01(_safe_float(compact.get("importance", 0.24), 0.24)),
        6,
    )
    compact["embed_layer_count"] = int(
        _safe_int(compact.get("embed_layer_count", 0), 0)
    )
    return compact


def _compact_file_graph_for_simulation(file_graph: dict[str, Any]) -> dict[str, Any]:
    compact_file_nodes = _compact_file_graph_nodes(file_graph.get("file_nodes", []))
    compact_field_nodes = _compact_file_graph_nodes(file_graph.get("field_nodes", []))
    compact_tag_nodes = _compact_file_graph_nodes(file_graph.get("tag_nodes", []))
    file_node_ids = {
        str(node.get("id", "")).strip()
        for node in compact_file_nodes
        if isinstance(node, dict) and str(node.get("id", "")).strip()
    }

    raw_nodes = file_graph.get("nodes", [])
    compact_non_file_nodes: list[dict[str, Any]] = []
    non_file_seen_ids: set[str] = set()
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("node_type", "")).strip().lower()
        node_id = str(node.get("id", "")).strip()
        if node_type == "file":
            continue
        if not node_type and node_id and node_id in file_node_ids:
            continue
        compact_node = _compact_file_graph_render_node(node)
        compact_node_id = str(compact_node.get("id", "")).strip()
        if compact_node_id and compact_node_id in non_file_seen_ids:
            continue
        if compact_node_id:
            non_file_seen_ids.add(compact_node_id)
        compact_non_file_nodes.append(compact_node)

    if not compact_non_file_nodes:
        for node in [*compact_field_nodes, *compact_tag_nodes]:
            compact_node = _compact_file_graph_render_node(node)
            compact_node_id = str(compact_node.get("id", "")).strip()
            if compact_node_id and compact_node_id in non_file_seen_ids:
                continue
            if compact_node_id:
                non_file_seen_ids.add(compact_node_id)
            compact_non_file_nodes.append(compact_node)

    compact_file_nodes_for_render = [
        _compact_file_graph_render_node(node) for node in compact_file_nodes
    ]
    compact_nodes = [*compact_non_file_nodes, *compact_file_nodes_for_render]

    edges_raw = file_graph.get("edges", [])
    compact_edges = [
        {
            "id": str(edge.get("id", "")).strip(),
            "source": str(edge.get("source", "")).strip(),
            "target": str(edge.get("target", "")).strip(),
            "field": str(edge.get("field", "")).strip(),
            "weight": round(_clamp01(_safe_float(edge.get("weight", 0.42), 0.42)), 6),
            "kind": str(edge.get("kind", "relates")).strip().lower() or "relates",
        }
        for edge in edges_raw
        if isinstance(edge, dict)
    ]
    dynamic_edge_cap = max(
        384,
        min(
            SIMULATION_FILE_GRAPH_EDGE_RESPONSE_CAP,
            max(
                384,
                int(
                    round(
                        max(1, len(compact_file_nodes))
                        * SIMULATION_FILE_GRAPH_EDGE_RESPONSE_FACTOR
                    )
                ),
            ),
        ),
    )
    if len(compact_edges) > dynamic_edge_cap:
        compact_edges = compact_edges[:dynamic_edge_cap]

    compact_stats_raw = file_graph.get("stats", {})
    compact_stats = (
        dict(compact_stats_raw) if isinstance(compact_stats_raw, dict) else {}
    )
    compact_stats["file_count"] = int(len(compact_file_nodes))
    compact_stats["edge_count"] = int(len(compact_edges))

    return {
        "record": str(file_graph.get("record", ETA_MU_FILE_GRAPH_RECORD)),
        "generated_at": str(
            file_graph.get("generated_at", datetime.now(timezone.utc).isoformat())
        ),
        "inbox": (
            dict(file_graph.get("inbox", {}))
            if isinstance(file_graph.get("inbox", {}), dict)
            else {}
        ),
        "embed_layers": [
            dict(row)
            for row in file_graph.get("embed_layers", [])
            if isinstance(row, dict)
        ],
        "organizer_presence": (
            dict(file_graph.get("organizer_presence", {}))
            if isinstance(file_graph.get("organizer_presence", {}), dict)
            else {}
        ),
        "concept_presences": [
            dict(row)
            for row in file_graph.get("concept_presences", [])
            if isinstance(row, dict)
        ],
        "field_nodes": compact_field_nodes,
        "tag_nodes": compact_tag_nodes,
        "file_nodes": compact_file_nodes,
        "nodes": compact_nodes,
        "edges": compact_edges,
        "stats": compact_stats,
    }


def _file_graph_layout_cache_key(file_graph: dict[str, Any]) -> str:
    file_nodes = file_graph.get("file_nodes", [])
    edges = file_graph.get("edges", [])
    file_count = len(file_nodes) if isinstance(file_nodes, list) else 0
    edge_count = len(edges) if isinstance(edges, list) else 0

    digest = hashlib.sha1()
    if isinstance(file_nodes, list):
        for node in file_nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            layer_count = _safe_int(node.get("embed_layer_count", 0), 0)
            has_collection = (
                "1" if str(node.get("vecstore_collection", "")).strip() else "0"
            )
            embedding_links = node.get("embedding_links", [])
            link_count = (
                len(embedding_links) if isinstance(embedding_links, list) else 0
            )
            importance = round(
                _clamp01(_safe_float(node.get("importance", 0.0), 0.0)), 4
            )
            usage_path = _file_node_usage_path(node)
            dominant_field = str(node.get("dominant_field", "")).strip()
            kind = str(node.get("kind", "")).strip().lower()
            summary_text = _bounded_text(
                node.get("summary", ""),
                limit=SIMULATION_FILE_GRAPH_SUMMARY_CHARS,
            )
            excerpt_text = _bounded_text(
                node.get("text_excerpt", ""),
                limit=SIMULATION_FILE_GRAPH_EXCERPT_CHARS,
            )
            text_signature = hashlib.sha1(
                f"{summary_text}|{excerpt_text}".encode("utf-8")
            ).hexdigest()[:12]
            digest.update(
                f"{node_id}|{usage_path}|{dominant_field}|{kind}|{layer_count}|{has_collection}|{link_count}|{importance}|{text_signature}".encode(
                    "utf-8"
                )
            )
    if isinstance(edges, list):
        for edge in edges[:256]:
            if not isinstance(edge, dict):
                continue
            source_id = str(edge.get("source", "")).strip()
            target_id = str(edge.get("target", "")).strip()
            kind = str(edge.get("kind", "")).strip().lower()
            weight = round(_clamp01(_safe_float(edge.get("weight", 0.0), 0.0)), 4)
            digest.update(f"{source_id}|{target_id}|{kind}|{weight}".encode("utf-8"))
    return f"{file_count}|{edge_count}|{digest.hexdigest()[:24]}"


def _clone_prepared_file_graph(prepared_graph: dict[str, Any]) -> dict[str, Any]:
    clone = dict(prepared_graph)
    clone["inbox"] = (
        dict(prepared_graph.get("inbox", {}))
        if isinstance(prepared_graph.get("inbox", {}), dict)
        else {}
    )
    clone["stats"] = (
        dict(prepared_graph.get("stats", {}))
        if isinstance(prepared_graph.get("stats", {}), dict)
        else {}
    )
    clone["organizer_presence"] = (
        dict(prepared_graph.get("organizer_presence", {}))
        if isinstance(prepared_graph.get("organizer_presence", {}), dict)
        else {}
    )
    for key in (
        "embed_layers",
        "concept_presences",
        "field_nodes",
        "tag_nodes",
        "file_nodes",
        "nodes",
        "edges",
        "embedding_particles",
    ):
        value = prepared_graph.get(key, [])
        clone[key] = list(value) if isinstance(value, list) else []
    return clone


def _prepare_file_graph_for_simulation(
    file_graph: dict[str, Any], *, now: float
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    cache_key = _file_graph_layout_cache_key(file_graph)
    now_monotonic = time.monotonic()
    with _SIMULATION_LAYOUT_CACHE_LOCK:
        cached_key = str(_SIMULATION_LAYOUT_CACHE.get("key", ""))
        cache_age = now_monotonic - _safe_float(
            _SIMULATION_LAYOUT_CACHE.get("prepared_monotonic", 0.0),
            0.0,
        )
        cached_graph_raw = _SIMULATION_LAYOUT_CACHE.get("prepared_graph")
        cached_points_raw = _SIMULATION_LAYOUT_CACHE.get("embedding_points", [])
        if (
            cache_key
            and cache_key == cached_key
            and cache_age <= SIMULATION_LAYOUT_CACHE_TTL_SECONDS
            and isinstance(cached_graph_raw, dict)
            and isinstance(cached_points_raw, list)
        ):
            return _clone_prepared_file_graph(cached_graph_raw), [
                dict(row) for row in cached_points_raw if isinstance(row, dict)
            ]

    compact_graph = _compact_file_graph_for_simulation(file_graph)

    embedding_points = _apply_file_graph_document_similarity_layout(
        compact_graph, now=now
    )
    with _SIMULATION_LAYOUT_CACHE_LOCK:
        _SIMULATION_LAYOUT_CACHE["key"] = cache_key
        _SIMULATION_LAYOUT_CACHE["prepared_monotonic"] = now_monotonic
        _SIMULATION_LAYOUT_CACHE["prepared_graph"] = compact_graph
        _SIMULATION_LAYOUT_CACHE["embedding_points"] = [
            dict(row) for row in embedding_points if isinstance(row, dict)
        ]
    return _clone_prepared_file_graph(compact_graph), [
        dict(row) for row in embedding_points if isinstance(row, dict)
    ]


def _build_unified_nexus_graph(
    file_graph: dict[str, Any] | None,
    crawler_graph: dict[str, Any] | None,
    *,
    include_crawler_in_file_nodes: bool,
) -> dict[str, Any] | None:
    if not isinstance(file_graph, dict):
        return file_graph if isinstance(file_graph, dict) else None

    unified = dict(file_graph)
    field_nodes = [
        dict(row) for row in file_graph.get("field_nodes", []) if isinstance(row, dict)
    ]
    tag_nodes = [
        dict(row) for row in file_graph.get("tag_nodes", []) if isinstance(row, dict)
    ]
    file_nodes = [
        dict(row) for row in file_graph.get("file_nodes", []) if isinstance(row, dict)
    ]

    nodes_raw = file_graph.get("nodes", [])
    if isinstance(nodes_raw, list) and nodes_raw:
        nodes = [dict(row) for row in nodes_raw if isinstance(row, dict)]
    else:
        nodes = [*field_nodes, *tag_nodes, *file_nodes]

    edges = [dict(row) for row in file_graph.get("edges", []) if isinstance(row, dict)]
    stats = (
        dict(file_graph.get("stats", {}))
        if isinstance(file_graph.get("stats", {}), dict)
        else {}
    )

    node_id_set: set[str] = {
        str(row.get("id", "")).strip()
        for row in nodes
        if str(row.get("id", "")).strip()
    }
    file_node_id_set: set[str] = {
        str(row.get("id", "")).strip()
        for row in file_nodes
        if str(row.get("id", "")).strip()
    }

    field_target_aliases: dict[str, str] = {}
    for field_node in field_nodes:
        field_id = str(field_node.get("id", "")).strip()
        node_id = str(field_node.get("node_id", "")).strip()
        source_tokens = [field_id, node_id]
        for token in source_tokens:
            if not token:
                continue
            presence_id = token
            if token.startswith("field:"):
                presence_id = token.split("field:", 1)[1].strip()
            if presence_id:
                canonical_field_id = field_id or f"field:{presence_id}"
                field_target_aliases[f"crawler-field:{presence_id}"] = (
                    canonical_field_id
                )

    crawler_rows = (
        crawler_graph.get("crawler_nodes", [])
        if isinstance(crawler_graph, dict)
        and isinstance(crawler_graph.get("crawler_nodes", []), list)
        else []
    )
    merged_crawler_nodes: list[dict[str, Any]] = [
        dict(row) for row in unified.get("crawler_nodes", []) if isinstance(row, dict)
    ]
    merged_crawler_id_set: set[str] = {
        str(row.get("id", "")).strip()
        for row in merged_crawler_nodes
        if str(row.get("id", "")).strip()
    }

    for row in crawler_rows:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        if not node_id:
            continue
        normalized = dict(row)
        normalized["id"] = node_id
        normalized["node_type"] = "crawler"
        crawler_kind = str(
            normalized.get("crawler_kind", normalized.get("kind", "url"))
        ).strip()
        normalized["crawler_kind"] = crawler_kind or "url"
        if not str(normalized.get("kind", "")).strip():
            normalized["kind"] = normalized["crawler_kind"]
        normalized["x"] = round(_clamp01(_safe_float(normalized.get("x", 0.5), 0.5)), 4)
        normalized["y"] = round(_clamp01(_safe_float(normalized.get("y", 0.5), 0.5)), 4)
        normalized["importance"] = round(
            _clamp01(_safe_float(normalized.get("importance", 0.28), 0.28)), 4
        )
        normalized["hue"] = int(_safe_float(normalized.get("hue", 198), 198.0))

        if node_id not in node_id_set:
            nodes.append(normalized)
            node_id_set.add(node_id)
        if include_crawler_in_file_nodes and node_id not in file_node_id_set:
            file_nodes.append(normalized)
            file_node_id_set.add(node_id)
        if node_id not in merged_crawler_id_set:
            merged_crawler_nodes.append(normalized)
            merged_crawler_id_set.add(node_id)

    seen_edges: set[tuple[str, str, str]] = set()
    for row in edges:
        source_id = str(row.get("source", "")).strip()
        target_id = str(row.get("target", "")).strip()
        kind = str(row.get("kind", "")).strip().lower()
        if source_id and target_id:
            seen_edges.add((source_id, target_id, kind))

    crawler_edges = (
        crawler_graph.get("edges", [])
        if isinstance(crawler_graph, dict)
        and isinstance(crawler_graph.get("edges", []), list)
        else []
    )
    for row in crawler_edges:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source", "")).strip()
        target_id = str(row.get("target", "")).strip()
        if not source_id or not target_id:
            continue
        source_id = field_target_aliases.get(source_id, source_id)
        target_id = field_target_aliases.get(target_id, target_id)
        if source_id.startswith("crawler-field:"):
            source_id = source_id.replace("crawler-field:", "field:", 1)
        if target_id.startswith("crawler-field:"):
            target_id = target_id.replace("crawler-field:", "field:", 1)
        if source_id == target_id:
            continue
        if source_id not in node_id_set or target_id not in node_id_set:
            continue
        kind = str(row.get("kind", "hyperlink")).strip().lower() or "hyperlink"
        edge_key = (source_id, target_id, kind)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_id = str(row.get("id", "")).strip()
        if not edge_id:
            edge_id = (
                "nexus-crawler-edge:"
                + sha1(f"{source_id}|{target_id}|{kind}".encode("utf-8")).hexdigest()[
                    :18
                ]
            )
        edges.append(
            {
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "field": str(row.get("field", "")).strip(),
                "kind": kind,
                "weight": round(
                    _clamp01(_safe_float(row.get("weight", 0.28), 0.28)), 4
                ),
            }
        )

    stats["crawler_nexus_count"] = len(merged_crawler_nodes)
    stats["nexus_node_count"] = len(nodes)
    stats["nexus_edge_count"] = len(edges)
    if include_crawler_in_file_nodes:
        stats["file_count"] = len(file_nodes)

    unified["field_nodes"] = field_nodes
    unified["tag_nodes"] = tag_nodes
    unified["file_nodes"] = file_nodes
    unified["nodes"] = nodes
    unified["edges"] = edges
    unified["crawler_nodes"] = merged_crawler_nodes
    unified["stats"] = stats
    return unified


# ============================================================================
# CANONICAL UNIFIED MODEL BUILDERS (v2)
# ============================================================================
#
# These builders produce the canonical unified types:
# - NexusNode / NexusEdge / NexusGraph
# - Field / FieldRegistry
# - Presence (unified)
# - Daimon (unified)
#
# See specs/drafts/part64-deep-research-09-unified-nexus-graph.md
# See specs/drafts/part64-deep-research-10-shared-fields-daimoi-dynamics.md
# ============================================================================


# Role mapping from legacy node_type to canonical NexusRole
_NEXUS_ROLE_MAP: dict[str, str] = {
    "field": "field",
    "file": "file",
    "tag": "tag",
    "crawler": "crawler",
    "presence": "presence",
    "concept": "concept",
    "organizer": "presence",
    "resource": "resource",
    "anchor": "anchor",
    "logical": "logical",
    "fact": "logical",
    "rule": "logical",
    "derivation": "logical",
    "contradiction": "logical",
    "gate": "logical",
    "event": "event",
    "test": "test_failure",
    "test_failure": "test_failure",
}


def _build_canonical_nexus_node(
    legacy_node: dict[str, Any],
    *,
    default_role: str = "file",
    origin_graph: str = "unknown",
) -> dict[str, Any]:
    """
    Convert a legacy graph node to canonical NexusNode format.

    Canonical NexusNode schema:
    - id: string
    - role: NexusRole (file, field, tag, crawler, resource, concept, anchor, logical, presence, event, etc.)
    - label: string
    - label_ja?: string
    - embedding?: { vector?: number[], centroid?: {x,y,z} }
    - x, y, z?: number
    - hue: number
    - capacity?: { cap, load, pressure }
    - demand?: { types: Record<string,number>, intensity }
    - provenance?: { source_uri, file_id, path, origin_graph, created_at, hash }
    - extension?: Record<string, unknown>
    - confidence?: number
    - status?: string
    - importance?: number
    """
    if not isinstance(legacy_node, dict):
        return {}

    node_id = str(legacy_node.get("id") or legacy_node.get("node_id") or "").strip()
    if not node_id:
        return {}

    # Map legacy node_type to canonical role
    legacy_type = (
        str(legacy_node.get("node_type", "") or legacy_node.get("kind", "") or "")
        .strip()
        .lower()
    )
    role = _NEXUS_ROLE_MAP.get(legacy_type, default_role)

    # Special handling for presence kinds
    presence_kind = str(legacy_node.get("presence_kind", "") or "").strip().lower()
    if presence_kind == "concept":
        role = "concept"
    elif presence_kind == "organizer":
        role = "presence"

    # Build provenance
    provenance: dict[str, Any] = {
        "origin_graph": origin_graph,
    }
    source_rel_path = str(
        legacy_node.get("source_rel_path") or legacy_node.get("archived_rel_path") or ""
    ).strip()
    if source_rel_path:
        provenance["path"] = source_rel_path
        provenance["source_uri"] = f"library:/{source_rel_path}"
    if legacy_node.get("file_id"):
        provenance["file_id"] = str(legacy_node.get("file_id"))
    if legacy_node.get("url"):
        provenance["source_uri"] = str(legacy_node.get("url"))

    # Build canonical node
    canonical: dict[str, Any] = {
        "id": node_id,
        "role": role,
        "label": str(legacy_node.get("label") or legacy_node.get("name") or node_id),
        "x": round(_clamp01(_safe_float(legacy_node.get("x", 0.5), 0.5)), 4),
        "y": round(_clamp01(_safe_float(legacy_node.get("y", 0.5), 0.5)), 4),
        "hue": int(_safe_float(legacy_node.get("hue", 198), 198.0)),
        "provenance": provenance,
    }

    # Optional fields
    if legacy_node.get("label_ja"):
        canonical["label_ja"] = str(legacy_node.get("label_ja"))

    if legacy_node.get("importance") is not None:
        canonical["importance"] = round(
            _clamp01(_safe_float(legacy_node.get("importance", 0.5), 0.5)), 4
        )

    if legacy_node.get("confidence") is not None:
        canonical["confidence"] = round(
            _clamp01(_safe_float(legacy_node.get("confidence", 1.0), 1.0)), 4
        )

    if legacy_node.get("status"):
        canonical["status"] = str(legacy_node.get("status"))

    # Copy relevant extension fields based on role
    extension: dict[str, Any] = {}
    if role == "file":
        for key in (
            "source_rel_path",
            "archived_rel_path",
            "archive_rel_path",
            "tags",
            "summary",
            "text_excerpt",
        ):
            if legacy_node.get(key):
                extension[key] = legacy_node[key]
    elif role == "crawler":
        for key in (
            "url",
            "domain",
            "title",
            "content_type",
            "crawler_kind",
            "compliance",
            "dominant_field",
        ):
            if legacy_node.get(key):
                extension[key] = legacy_node[key]
    elif role == "field":
        for key in ("field", "dominant_presence"):
            if legacy_node.get(key):
                extension[key] = legacy_node[key]
    elif role == "tag":
        for key in ("tag", "member_count"):
            if legacy_node.get(key):
                extension[key] = legacy_node[key]

    if extension:
        canonical["extension"] = extension

    return canonical


def _build_canonical_nexus_edge(
    legacy_edge: dict[str, Any],
    *,
    node_id_set: set[str],
) -> dict[str, Any] | None:
    """
    Convert a legacy graph edge to canonical NexusEdge format.

    Canonical NexusEdge schema:
    - id: string
    - source: string (node id)
    - target: string (node id)
    - kind: NexusEdgeKind
    - weight: number
    - cost?: number
    - affinity?: number
    - saturation?: number
    - health?: number
    - field?: string
    """
    if not isinstance(legacy_edge, dict):
        return None

    source_id = str(legacy_edge.get("source", "")).strip()
    target_id = str(legacy_edge.get("target", "")).strip()
    if not source_id or not target_id:
        return None
    if source_id not in node_id_set or target_id not in node_id_set:
        return None

    edge_id = str(legacy_edge.get("id", "")).strip()
    if not edge_id:
        edge_id = f"nexus-edge:{hashlib.sha256(f'{source_id}|{target_id}'.encode('utf-8')).hexdigest()[:16]}"

    kind = str(legacy_edge.get("kind", "relates")).strip().lower() or "relates"
    weight = round(_clamp01(_safe_float(legacy_edge.get("weight", 0.5), 0.5)), 4)

    canonical: dict[str, Any] = {
        "id": edge_id,
        "source": source_id,
        "target": target_id,
        "kind": kind,
        "weight": weight,
    }

    if legacy_edge.get("field"):
        canonical["field"] = str(legacy_edge.get("field"))

    # Edge dynamics (if available)
    if legacy_edge.get("cost") is not None:
        canonical["cost"] = _safe_float(legacy_edge.get("cost"), 0.5)
    if legacy_edge.get("affinity") is not None:
        canonical["affinity"] = round(
            _clamp01(_safe_float(legacy_edge.get("affinity", 0.5), 0.5)), 4
        )
    if legacy_edge.get("saturation") is not None:
        canonical["saturation"] = round(
            _clamp01(_safe_float(legacy_edge.get("saturation", 0.0), 0.0)), 4
        )
    if legacy_edge.get("health") is not None:
        canonical["health"] = round(
            _clamp01(_safe_float(legacy_edge.get("health", 1.0), 1.0)), 4
        )

    return canonical


def _build_canonical_nexus_graph(
    file_graph: dict[str, Any] | None,
    crawler_graph: dict[str, Any] | None,
    logical_graph: dict[str, Any] | None,
    *,
    include_crawler: bool = True,
    include_logical: bool = True,
) -> dict[str, Any]:
    """
    Build the canonical unified NexusGraph from all legacy graph sources.

    This is the single source of truth for graph data. All other graph
    payloads (file_graph, crawler_graph, logical_graph) become projections
    of this canonical graph.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_id_set: set[str] = set()
    edge_key_set: set[tuple[str, str, str]] = set()

    # Joins indices
    by_role: dict[str, list[str]] = {}
    by_path: dict[str, str] = {}
    by_source_uri: dict[str, str] = {}
    by_file_id: dict[str, str] = {}

    def _add_node(node: dict[str, Any], origin_graph: str) -> None:
        if not isinstance(node, dict):
            return
        canonical = _build_canonical_nexus_node(node, origin_graph=origin_graph)
        node_id = canonical.get("id", "")
        if not node_id or node_id in node_id_set:
            return
        nodes.append(canonical)
        node_id_set.add(node_id)

        # Update indices
        role = canonical.get("role", "unknown")
        if role not in by_role:
            by_role[role] = []
        by_role[role].append(node_id)

        prov = canonical.get("provenance", {})
        if prov.get("path"):
            by_path[prov["path"]] = node_id
        if prov.get("source_uri"):
            by_source_uri[prov["source_uri"]] = node_id
        if prov.get("file_id"):
            by_file_id[prov["file_id"]] = node_id

    def _add_edge(edge: dict[str, Any]) -> None:
        canonical = _build_canonical_nexus_edge(edge, node_id_set=node_id_set)
        if not canonical:
            return
        source_id = canonical["source"]
        target_id = canonical["target"]
        kind = canonical["kind"]
        edge_key = (source_id, target_id, kind)
        if edge_key in edge_key_set:
            return
        edges.append(canonical)
        edge_key_set.add(edge_key)

    # Process file_graph
    if isinstance(file_graph, dict):
        for node in file_graph.get("field_nodes", []):
            _add_node(node, "file_graph")
        for node in file_graph.get("tag_nodes", []):
            _add_node(node, "file_graph")
        for node in file_graph.get("file_nodes", []):
            _add_node(node, "file_graph")
        for node in file_graph.get("nodes", []):
            _add_node(node, "file_graph")
        for edge in file_graph.get("edges", []):
            _add_edge(edge)

    # Process crawler_graph
    if include_crawler and isinstance(crawler_graph, dict):
        for node in crawler_graph.get("field_nodes", []):
            _add_node(node, "crawler_graph")
        for node in crawler_graph.get("crawler_nodes", []):
            _add_node(node, "crawler_graph")
        for edge in crawler_graph.get("edges", []):
            _add_edge(edge)

    # Process logical_graph (Logos projection)
    if include_logical and isinstance(logical_graph, dict):
        for node in logical_graph.get("nodes", []):
            _add_node(node, "logical_graph")
        for edge in logical_graph.get("edges", []):
            _add_edge(edge)

    # Build stats
    role_counts: dict[str, int] = {}
    for role, ids in by_role.items():
        role_counts[role] = len(ids)

    edge_kind_counts: dict[str, int] = {}
    for edge in edges:
        kind = edge.get("kind", "unknown")
        edge_kind_counts[kind] = edge_kind_counts.get(kind, 0) + 1

    # Mean connectivity
    connectivity_sum = sum(
        sum(1 for e in edges if e["source"] == node_id or e["target"] == node_id)
        for node_id in node_id_set
    )
    mean_connectivity = (connectivity_sum / len(node_id_set)) if node_id_set else 0.0

    return {
        "record": "ημ.nexus-graph.v1",
        "schema_version": "nexus.graph.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "edges": edges,
        "joins": {
            "by_role": by_role,
            "by_path": by_path,
            "by_source_uri": by_source_uri,
            "by_file_id": by_file_id,
        },
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "role_counts": role_counts,
            "edge_kind_counts": edge_kind_counts,
            "mean_connectivity": round(mean_connectivity, 4),
        },
    }


def _build_field_registry(
    catalog: dict[str, Any],
    graph_runtime: dict[str, Any] | None,
    *,
    kernel_width: float = 0.3,
    decay_rate: float = 0.1,
    resolution: int = 32,
) -> dict[str, Any]:
    """
    Build the shared field registry from catalog and graph runtime data.

    The field registry contains a bounded set of shared fields:
    - demand: Where in semantic space is there active demand
    - flow: Aggregate movement patterns
    - entropy: Where things are uncertain/unresolved
    - graph: The compiled graph's influence on particle motion

    All presences contribute to these shared fields, not to individual fields.
    """
    from .constants import FIELD_KINDS, MAX_FIELD_COUNT

    fields: dict[str, dict[str, Any]] = {}

    # Extract gravity data for demand field
    gravity = (
        graph_runtime.get("gravity", []) if isinstance(graph_runtime, dict) else []
    )
    node_count = len(gravity) if isinstance(gravity, list) else 0

    # Build demand field samples from gravity
    demand_samples: list[dict[str, Any]] = []
    demand_max = 0.0
    demand_sum = 0.0
    demand_peak_loc: dict[str, float] | None = None

    if gravity and isinstance(gravity, list):
        # Sample at grid resolution
        for i in range(min(resolution, node_count)):
            g_val = _safe_float(gravity[i], 0.0) if i < len(gravity) else 0.0
            x = (i % resolution) / resolution
            y = (i // resolution) / resolution
            if g_val > 0.001:
                demand_samples.append(
                    {"x": round(x, 4), "y": round(y, 4), "value": round(g_val, 6)}
                )
                demand_sum += g_val
                if g_val > demand_max:
                    demand_max = g_val
                    demand_peak_loc = {"x": x, "y": y}

    fields["demand"] = {
        "kind": "demand",
        "record": "ημ.shared-field.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": demand_samples[:256],  # Cap samples
        "stats": {
            "mean": round(demand_sum / max(1, len(demand_samples)), 6),
            "max": round(demand_max, 6),
            "min": 0.0,
            "integral": round(demand_sum, 6),
            "peak_location": demand_peak_loc,
        },
        "top_contributors": [],  # TODO: Add attribution
        "params": {
            "kernel_width": kernel_width,
            "decay_rate": decay_rate,
            "resolution": resolution,
        },
    }

    # Build flow field (placeholder - would track daimon movement)
    fields["flow"] = {
        "kind": "flow",
        "record": "ημ.shared-field.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": [],
        "stats": {"mean": 0.0, "max": 0.0, "min": 0.0, "integral": 0.0},
        "top_contributors": [],
        "params": {
            "kernel_width": kernel_width,
            "decay_rate": decay_rate,
            "resolution": resolution,
        },
    }

    # Build entropy field (placeholder - would track type distribution entropy)
    fields["entropy"] = {
        "kind": "entropy",
        "record": "ημ.shared-field.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": [],
        "stats": {"mean": 0.0, "max": 0.0, "min": 0.0, "integral": 0.0},
        "top_contributors": [],
        "params": {
            "kernel_width": kernel_width,
            "decay_rate": decay_rate,
            "resolution": resolution,
        },
    }

    # Build graph field from graph runtime node prices
    graph_samples: list[dict[str, Any]] = []
    node_prices = (
        graph_runtime.get("node_prices", []) if isinstance(graph_runtime, dict) else []
    )
    graph_sum = 0.0
    graph_max = 0.0

    if node_prices and isinstance(node_prices, list):
        for i, price in enumerate(node_prices[: resolution * resolution]):
            p_val = _safe_float(price, 0.0)
            if p_val > 0.001:
                x = (i % resolution) / resolution
                y = (i // resolution) / resolution
                graph_samples.append(
                    {"x": round(x, 4), "y": round(y, 4), "value": round(p_val, 6)}
                )
                graph_sum += p_val
                graph_max = max(graph_max, p_val)

    fields["graph"] = {
        "kind": "graph",
        "record": "ημ.shared-field.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples": graph_samples[:256],
        "stats": {
            "mean": round(graph_sum / max(1, len(graph_samples)), 6),
            "max": round(graph_max, 6),
            "min": 0.0,
            "integral": round(graph_sum, 6),
        },
        "top_contributors": [],
        "params": {
            "kernel_width": kernel_width,
            "decay_rate": decay_rate,
            "resolution": resolution,
        },
    }

    return {
        "record": "ημ.field-registry.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fields": {k: fields.get(k, {}) for k in FIELD_KINDS},
        "weights": {
            "demand": 0.4,
            "flow": 0.2,
            "entropy": 0.15,
            "graph": 0.25,
        },
        "field_count": len(FIELD_KINDS),
        "bounded": len(FIELD_KINDS) <= MAX_FIELD_COUNT,
    }


def _project_legacy_file_graph_from_nexus(
    nexus_graph: dict[str, Any],
) -> dict[str, Any]:
    """
    Project the legacy file_graph payload from the canonical nexus_graph.
    This provides backward compatibility during migration.
    """
    if not isinstance(nexus_graph, dict):
        return {
            "record": "ημ.file-graph.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "nodes": [],
            "field_nodes": [],
            "tag_nodes": [],
            "file_nodes": [],
            "edges": [],
            "stats": {},
        }

    nodes = nexus_graph.get("nodes", [])
    edges = nexus_graph.get("edges", [])

    # Partition nodes by role
    field_nodes = [n for n in nodes if n.get("role") == "field"]
    tag_nodes = [n for n in nodes if n.get("role") == "tag"]
    file_nodes = [n for n in nodes if n.get("role") in ("file", "resource")]

    return {
        "record": "ημ.file-graph.v1",
        "generated_at": nexus_graph.get(
            "generated_at", datetime.now(timezone.utc).isoformat()
        ),
        "nodes": nodes,
        "field_nodes": field_nodes,
        "tag_nodes": tag_nodes,
        "file_nodes": file_nodes,
        "edges": edges,
        "stats": nexus_graph.get("stats", {}),
    }


def _project_legacy_logical_graph_from_nexus(
    nexus_graph: dict[str, Any],
) -> dict[str, Any]:
    """
    Project the legacy logical_graph payload from the canonical nexus_graph.
    The logical graph is just the nodes with role in logical roles.
    """
    if not isinstance(nexus_graph, dict):
        return {
            "record": "ημ.logical-graph.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "nodes": [],
            "edges": [],
            "joins": {},
            "stats": {},
        }

    logical_roles = {
        "logical",
        "fact",
        "rule",
        "derivation",
        "contradiction",
        "gate",
        "event",
        "tag",
        "file",
    }
    nodes = nexus_graph.get("nodes", [])
    edges = nexus_graph.get("edges", [])

    logical_nodes = [n for n in nodes if n.get("role") in logical_roles]
    logical_node_ids = {n.get("id") for n in logical_nodes}

    # Filter edges to only those between logical nodes
    logical_edges = [
        e
        for e in edges
        if e.get("source") in logical_node_ids and e.get("target") in logical_node_ids
    ]

    return {
        "record": "ημ.logical-graph.v1",
        "generated_at": nexus_graph.get(
            "generated_at", datetime.now(timezone.utc).isoformat()
        ),
        "nodes": logical_nodes,
        "edges": logical_edges,
        "joins": nexus_graph.get("joins", {}),
        "stats": {
            "file_nodes": len([n for n in logical_nodes if n.get("role") == "file"]),
            "tag_nodes": len([n for n in logical_nodes if n.get("role") == "tag"]),
            "fact_nodes": len(
                [n for n in logical_nodes if n.get("role") in ("fact", "logical")]
            ),
            "event_nodes": len([n for n in logical_nodes if n.get("role") == "event"]),
            "edge_count": len(logical_edges),
        },
    }


def _clean_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_-]+", text.lower()) if token]


def _document_layout_range_from_importance(importance: float) -> float:
    normalized = _clamp01(_safe_float(importance, 0.2))
    return 0.018 + (normalized * 0.055)


def _document_layout_tokens(node: dict[str, Any]) -> list[str]:
    values: list[str] = []
    tags = node.get("tags", [])
    labels = node.get("labels", [])
    if isinstance(tags, list):
        values.extend(str(tag) for tag in tags)
    if isinstance(labels, list):
        values.extend(str(label) for label in labels)
    values.extend(
        [
            _bounded_text(
                node.get("summary", ""),
                limit=SIMULATION_FILE_GRAPH_SUMMARY_CHARS,
            ),
            _bounded_text(
                node.get("text_excerpt", ""),
                limit=SIMULATION_FILE_GRAPH_EXCERPT_CHARS,
            ),
            _bounded_text(node.get("source_rel_path", ""), limit=160),
            _bounded_text(node.get("archived_rel_path", ""), limit=160),
            _bounded_text(node.get("archive_rel_path", ""), limit=160),
            _bounded_text(node.get("name", ""), limit=160),
            _bounded_text(node.get("kind", ""), limit=64),
            _bounded_text(node.get("dominant_field", ""), limit=32),
            _bounded_text(node.get("vecstore_collection", ""), limit=96),
        ]
    )

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _clean_tokens(value):
            if len(token) < 3:
                continue
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= 80:
                return deduped
    return deduped


def _document_layout_text_density(node: dict[str, Any], tokens: list[str]) -> float:
    token_density = min(1.0, len(tokens) / 42.0)
    summary_len = len(str(node.get("summary", "")).strip())
    excerpt_len = len(str(node.get("text_excerpt", "")).strip())
    label_len = len(str(node.get("name", "")).strip()) + len(
        str(node.get("label", "")).strip()
    )
    char_density = min(1.0, (summary_len + excerpt_len + label_len) / 760.0)

    tags = node.get("tags", [])
    labels = node.get("labels", [])
    embedding_links = node.get("embedding_links", [])
    tag_count = len(tags) if isinstance(tags, list) else 0
    label_count = len(labels) if isinstance(labels, list) else 0
    link_count = len(embedding_links) if isinstance(embedding_links, list) else 0
    layer_count = _safe_int(node.get("embed_layer_count", 0), 0)
    structural = min(
        1.0,
        (tag_count * 0.08)
        + (label_count * 0.05)
        + (layer_count * 0.22)
        + (link_count * 0.04),
    )

    density = 0.2 + (token_density * 0.42) + (char_density * 0.26) + (structural * 0.36)
    return max(0.12, min(1.9, density))


def _document_layout_semantic_vector(
    node: dict[str, Any],
    tokens: list[str],
    *,
    dimensions: int = 8,
) -> list[float]:
    if dimensions <= 0:
        return []

    raw_tokens = list(tokens)
    if not raw_tokens:
        raw_tokens = _clean_tokens(
            " ".join(
                [
                    str(node.get("dominant_field", "")),
                    str(node.get("kind", "")),
                    str(node.get("vecstore_collection", "")),
                    str(node.get("name", "")),
                    str(node.get("label", "")),
                ]
            )
        )
    if not raw_tokens:
        raw_tokens = ["eta", "mu", "field"]

    accum = [0.0 for _ in range(dimensions)]
    for token_index, token in enumerate(raw_tokens[:96]):
        weight = 0.8 + min(1.6, len(token) / 6.5)
        digest = hashlib.sha1(
            f"{token}|{token_index}|{dimensions}".encode("utf-8")
        ).digest()
        for axis in range(dimensions):
            byte = digest[axis % len(digest)]
            signed = (float(byte) / 127.5) - 1.0
            accum[axis] += signed * weight

    field_token = str(node.get("dominant_field", "")).strip()
    kind_token = str(node.get("kind", "")).strip().lower()
    for marker, gain in ((field_token, 0.36), (kind_token, 0.22)):
        if not marker:
            continue
        digest = hashlib.sha1(f"marker:{marker}".encode("utf-8")).digest()
        for axis in range(dimensions):
            byte = digest[(axis * 3) % len(digest)]
            signed = (float(byte) / 127.5) - 1.0
            accum[axis] += signed * gain

    magnitude = math.sqrt(sum(value * value for value in accum))
    if magnitude <= 1e-8:
        fallback = [0.0 for _ in range(dimensions)]
        fallback[0] = 1.0
        return fallback
    return [value / magnitude for value in accum]


def _semantic_vector_blend(
    base: list[float], target: list[float], blend: float
) -> list[float]:
    if not base and not target:
        return []
    if not base:
        return list(target)
    if not target:
        return list(base)

    mix = max(0.0, min(1.0, _safe_float(blend, 0.5)))
    size = min(len(base), len(target))
    if size <= 0:
        return list(base)

    merged = [(base[i] * (1.0 - mix)) + (target[i] * mix) for i in range(size)]
    magnitude = math.sqrt(sum(value * value for value in merged))
    if magnitude <= 1e-8:
        return [0.0 for _ in range(size)]
    return [value / magnitude for value in merged]


def _semantic_vector_cosine(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(size))
    left_mag = sum(left[i] * left[i] for i in range(size))
    right_mag = sum(right[i] * right[i] for i in range(size))
    if left_mag <= 1e-12 or right_mag <= 1e-12:
        return 0.0
    cosine = dot / math.sqrt(left_mag * right_mag)
    return max(-1.0, min(1.0, cosine))


def _semantic_vector_hue(vector: list[float]) -> float:
    if not vector:
        return 210.0
    vx = _safe_float(vector[0], 0.0)
    vy = _safe_float(vector[1], 0.0) if len(vector) > 1 else 0.0
    if abs(vx) <= 1e-8 and abs(vy) <= 1e-8:
        return 210.0
    return (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0


def _document_layout_similarity(
    left_node: dict[str, Any],
    right_node: dict[str, Any],
    left_tokens: list[str],
    right_tokens: list[str],
) -> float:
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set.intersection(right_set))
    union = max(1, len(left_set) + len(right_set) - overlap)
    token_jaccard = overlap / float(union)

    left_field = str(left_node.get("dominant_field", "")).strip()
    right_field = str(right_node.get("dominant_field", "")).strip()

    # Strong attraction for same field
    same_field = 1.0 if left_field and left_field == right_field else 0.0

    # REPULSION: Strong penalty for unrelated fields
    field_repulsion = 0.0
    if left_field and right_field and left_field != right_field:
        # Base repulsion for any different fields
        field_repulsion = -0.35

        # Enhanced repulsion for "opposite" philosophical concepts
        opposite_pairs = [
            ("f9", "f10"),  # good vs evil
            ("f10", "f9"),  # evil vs good
            ("f11", "f12"),  # right vs wrong
            ("f12", "f11"),  # wrong vs right
            ("f13", "f14"),  # dead vs living
            ("f14", "f13"),  # living vs dead
        ]
        if (left_field, right_field) in opposite_pairs:
            field_repulsion = -0.62  # Strong repulsion for opposites

    same_kind = (
        1.0
        if str(left_node.get("kind", "")).strip().lower()
        and str(left_node.get("kind", "")).strip().lower()
        == str(right_node.get("kind", "")).strip().lower()
        else 0.0
    )
    left_collection = str(left_node.get("vecstore_collection", "")).strip()
    right_collection = str(right_node.get("vecstore_collection", "")).strip()
    same_collection = (
        1.0 if left_collection and left_collection == right_collection else 0.0
    )

    score = (
        (token_jaccard * 0.78)
        + (same_field * 0.12)
        + (same_kind * 0.06)
        + (same_collection * 0.04)
        + field_repulsion  # Add repulsion penalty
    )

    # Increased penalty threshold for unrelated nodes
    if token_jaccard < 0.05 and same_field <= 0.0 and same_kind <= 0.0:
        score *= 0.25  # Reduced from 0.45 for stronger separation

    return _clamp01(score)


def _document_layout_is_embedded(node: dict[str, Any]) -> bool:
    if _safe_int(node.get("embed_layer_count", 0), 0) > 0:
        return True

    layer_points = node.get("embed_layer_points", [])
    if isinstance(layer_points, list):
        for row in layer_points:
            if not isinstance(row, dict):
                continue
            if bool(row.get("active", True)):
                return True

    if str(node.get("vecstore_collection", "")).strip():
        return True

    embedding_links = node.get("embedding_links", [])
    if isinstance(embedding_links, list) and len(embedding_links) > 0:
        return True

    return False


def _apply_file_graph_document_similarity_layout(
    file_graph: dict[str, Any], *, now: float | None = None
) -> list[dict[str, float]]:
    file_nodes_raw = file_graph.get("file_nodes", [])
    if not isinstance(file_nodes_raw, list) or len(file_nodes_raw) <= 0:
        file_graph["embedding_particles"] = []
        return []

    entries: list[dict[str, Any]] = []
    for index, node in enumerate(file_nodes_raw):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip() or f"file:{index}"
        x = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        importance = _clamp01(_safe_float(node.get("importance", 0.2), 0.2))
        local_range = _document_layout_range_from_importance(importance)
        tokens = _document_layout_tokens(node)
        semantic_vector = _document_layout_semantic_vector(node, tokens)
        text_density = _document_layout_text_density(node, tokens)
        entries.append(
            {
                "id": node_id,
                "index": len(entries),
                "node": node,
                "x": x,
                "y": y,
                "importance": importance,
                "range": local_range,
                "embedded": _document_layout_is_embedded(node),
                "tokens": tokens,
                "vector": semantic_vector,
                "text_density": text_density,
            }
        )

    if not entries:
        file_graph["embedding_particles"] = []
        return []

    cell_size = 0.08
    grid: dict[tuple[int, int], list[int]] = {}
    for index, entry in enumerate(entries):
        gx = int(entry["x"] / cell_size)
        gy = int(entry["y"] / cell_size)
        grid.setdefault((gx, gy), []).append(index)

    offsets: list[list[float]] = [[0.0, 0.0] for _ in entries]
    if len(entries) > 1:
        for index, left in enumerate(entries):
            gx = int(left["x"] / cell_size)
            gy = int(left["y"] / cell_size)
            radius_cells = max(1, int(math.ceil(left["range"] / cell_size)))

            for oy in range(-radius_cells, radius_cells + 1):
                for ox in range(-radius_cells, radius_cells + 1):
                    bucket = grid.get((gx + ox, gy + oy), [])
                    for other_index in bucket:
                        if other_index <= index:
                            continue
                        right = entries[other_index]
                        pair_range = max(left["range"], right["range"])
                        dx = right["x"] - left["x"]
                        dy = right["y"] - left["y"]
                        distance = math.sqrt((dx * dx) + (dy * dy))
                        if distance <= 1e-8 or distance > pair_range:
                            continue

                        similarity = _document_layout_similarity(
                            left["node"],
                            right["node"],
                            left.get("tokens", []),
                            right.get("tokens", []),
                        )
                        semantic_signed = max(
                            -1.0, min(1.0, (similarity - 0.52) / 0.48)
                        )
                        mixed_embedding = bool(left["embedded"]) != bool(
                            right["embedded"]
                        )
                        signed_similarity = (
                            -max(0.46, abs(semantic_signed) * 0.72)
                            if mixed_embedding
                            else semantic_signed
                        )
                        if abs(signed_similarity) < 0.22:
                            continue

                        falloff = _clamp01(1.0 - (distance / max(pair_range, 1e-6)))
                        importance_mix = (
                            left["importance"] + right["importance"]
                        ) * 0.5
                        density_mix = (
                            _safe_float(left.get("text_density"), 0.45)
                            + _safe_float(right.get("text_density"), 0.45)
                        ) * 0.5
                        strength = (
                            falloff
                            * abs(signed_similarity)
                            * (1.2 if mixed_embedding else 1.0)
                            * (0.00145 + (importance_mix * 0.0022))
                            * (0.66 + (density_mix * 0.3))
                        )
                        if strength <= 0.0:
                            continue

                        ux = dx / distance
                        uy = dy / distance
                        direction = 1.0 if signed_similarity >= 0.0 else -1.0
                        fx = ux * strength * direction
                        fy = uy * strength * direction

                        offsets[index][0] += fx
                        offsets[index][1] += fy
                        offsets[other_index][0] -= fx
                        offsets[other_index][1] -= fy

    embedding_particle_points: list[dict[str, float]] = []
    embedding_particle_nodes: list[dict[str, float | str]] = []
    embedded_entries = [entry for entry in entries if bool(entry.get("embedded"))]
    if embedded_entries:
        now_seconds = _safe_float(now, time.time()) if now is not None else time.time()
        particle_count = max(6, min(42, int(round(len(embedded_entries) * 1.8))))
        particles: list[dict[str, Any]] = []
        source_weights = [
            max(0.08, _safe_float(entry.get("text_density", 0.45), 0.45))
            for entry in embedded_entries
        ]
        source_weight_total = sum(source_weights)

        for index in range(particle_count):
            source = embedded_entries[index % len(embedded_entries)]
            if source_weight_total > 1e-8 and len(embedded_entries) > 1:
                ratio_slot = (float(index) + 0.5) / float(max(1, particle_count))
                cumulative = 0.0
                for entry, weight in zip(embedded_entries, source_weights):
                    cumulative += weight / source_weight_total
                    if ratio_slot <= cumulative:
                        source = entry
                        break
            seed = f"{source['id']}|particle|{index}"
            scatter = 0.006 + (
                _stable_ratio(seed, 31)
                * max(0.018, _safe_float(source["range"], 0.03) * 0.64)
            )
            seed_x = (_stable_ratio(seed, 47) * 2.0) - 1.0
            seed_y = (_stable_ratio(seed, 53) * 2.0) - 1.0
            x = _clamp01(_safe_float(source["x"], 0.5) + (seed_x * scatter))
            y = _clamp01(_safe_float(source["y"], 0.5) + (seed_y * scatter * 0.82))
            particles.append(
                {
                    "id": f"embed-particle:{index}",
                    "x": x,
                    "y": y,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vector": list(source.get("vector", [])),
                    "text_density": _safe_float(source.get("text_density"), 0.45),
                    "focus_x": x,
                    "focus_y": y,
                    "cohesion": 0.0,
                    "drift": (_stable_ratio(seed, 41) * 2.0) - 1.0,
                }
            )

        for _ in range(4):
            particle_forces: list[list[float]] = [[0.0, 0.0] for _ in particles]

            for particle_index, particle in enumerate(particles):
                influence_total = 0.0
                avg_x = 0.0
                avg_y = 0.0
                avg_vector = [0.0 for _ in particle.get("vector", [])]
                doc_radius = 0.22

                for entry in embedded_entries:
                    dx = _safe_float(entry["x"], 0.5) - _safe_float(particle["x"], 0.5)
                    dy = _safe_float(entry["y"], 0.5) - _safe_float(particle["y"], 0.5)
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance > doc_radius:
                        continue
                    if distance <= 1e-8:
                        jitter = (
                            _stable_ratio(
                                f"{particle['id']}|{entry['id']}|jitter",
                                particle_index + 1,
                            )
                            - 0.5
                        ) * 0.0012
                        dx += jitter
                        dy -= jitter
                        distance = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))

                    similarity = _semantic_vector_cosine(
                        particle.get("vector", []),
                        entry.get("vector", []),
                    )
                    distance_weight = _clamp01(1.0 - (distance / doc_radius))
                    density_weight = 0.24 + (
                        _safe_float(entry.get("text_density"), 0.45) * 0.92
                    )
                    similarity_weight = 0.28 + ((similarity + 1.0) * 0.36)
                    influence_weight = (
                        distance_weight
                        * distance_weight
                        * density_weight
                        * similarity_weight
                    )
                    if influence_weight <= 0.0:
                        continue

                    influence_total += influence_weight
                    avg_x += _safe_float(entry["x"], 0.5) * influence_weight
                    avg_y += _safe_float(entry["y"], 0.5) * influence_weight
                    entry_vector = entry.get("vector", [])
                    for axis in range(min(len(avg_vector), len(entry_vector))):
                        avg_vector[axis] += (
                            _safe_float(entry_vector[axis], 0.0) * influence_weight
                        )

                    direction = 1.0 if similarity >= 0.0 else -1.0
                    force_strength = (
                        (0.00072 + (abs(similarity) * 0.0024))
                        * distance_weight
                        * density_weight
                    )
                    particle_forces[particle_index][0] += (
                        (dx / distance) * force_strength * direction
                    )
                    particle_forces[particle_index][1] += (
                        (dy / distance) * force_strength * direction
                    )

                if influence_total > 0.0:
                    target_x = avg_x / influence_total
                    target_y = avg_y / influence_total
                    particle["focus_x"] = target_x
                    particle["focus_y"] = target_y
                    particle["cohesion"] = _clamp01(
                        (_safe_float(particle.get("cohesion", 0.0), 0.0) * 0.55)
                        + min(1.0, influence_total * 0.48)
                    )
                    pull_strength = min(0.0052, 0.0012 + (influence_total * 0.0019))
                    particle_forces[particle_index][0] += (
                        target_x - _safe_float(particle["x"], 0.5)
                    ) * pull_strength
                    particle_forces[particle_index][1] += (
                        target_y - _safe_float(particle["y"], 0.5)
                    ) * pull_strength

                    if avg_vector:
                        avg_magnitude = math.sqrt(
                            sum(value * value for value in avg_vector)
                        )
                        if avg_magnitude > 1e-8:
                            normalized_avg = [
                                value / avg_magnitude for value in avg_vector
                            ]
                            particle["vector"] = _semantic_vector_blend(
                                list(particle.get("vector", [])),
                                normalized_avg,
                                0.26,
                            )
                else:
                    particle["cohesion"] = _clamp01(
                        _safe_float(particle.get("cohesion", 0.0), 0.0) * 0.86
                    )

            for left_index in range(len(particles)):
                left = particles[left_index]
                for right_index in range(left_index + 1, len(particles)):
                    right = particles[right_index]
                    dx = _safe_float(right["x"], 0.5) - _safe_float(left["x"], 0.5)
                    dy = _safe_float(right["y"], 0.5) - _safe_float(left["y"], 0.5)
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance > 0.2:
                        continue
                    if distance <= 1e-8:
                        jitter = (
                            _stable_ratio(
                                f"{left['id']}|{right['id']}|pair", left_index + 3
                            )
                            - 0.5
                        ) * 0.001
                        dx += jitter
                        dy -= jitter
                        distance = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))

                    similarity = _semantic_vector_cosine(
                        left.get("vector", []),
                        right.get("vector", []),
                    )
                    falloff = _clamp01(1.0 - (distance / 0.2))
                    pair_strength = (0.00044 + (abs(similarity) * 0.00186)) * falloff
                    direction = 1.0 if similarity >= 0.0 else -1.0
                    fx = (dx / distance) * pair_strength * direction
                    fy = (dy / distance) * pair_strength * direction

                    particle_forces[left_index][0] += fx
                    particle_forces[left_index][1] += fy
                    particle_forces[right_index][0] -= fx
                    particle_forces[right_index][1] -= fy

            for particle_index, particle in enumerate(particles):
                drift_phase = (
                    now_seconds
                    * (0.62 + abs(_safe_float(particle.get("drift", 0.0), 0.0)) * 0.42)
                ) + (particle_index * 0.41)
                particle_forces[particle_index][0] += math.cos(drift_phase) * 0.00021
                particle_forces[particle_index][1] += math.sin(drift_phase) * 0.00017

                particle_x = _safe_float(particle.get("x", 0.5), 0.5)
                particle_y = _safe_float(particle.get("y", 0.5), 0.5)
                simplex_amp = 0.00011 + (
                    abs(_safe_float(particle.get("drift", 0.0), 0.0)) * 0.00017
                )
                simplex_phase = now_seconds * 0.31
                simplex_x = _simplex_noise_2d(
                    (particle_x * 4.6) + (particle_index * 0.19) + simplex_phase,
                    (particle_y * 4.6) + (simplex_phase * 0.71),
                    seed=particle_index + 17,
                )
                simplex_y = _simplex_noise_2d(
                    (particle_x * 4.6) + 17.0 + (simplex_phase * 0.59),
                    (particle_y * 4.6) + 11.0 + simplex_phase,
                    seed=particle_index + 29,
                )
                particle_forces[particle_index][0] += simplex_x * simplex_amp
                particle_forces[particle_index][1] += simplex_y * simplex_amp

                vx = (
                    _safe_float(particle.get("vx", 0.0), 0.0)
                    + particle_forces[particle_index][0]
                ) * 0.84
                vy = (
                    _safe_float(particle.get("vy", 0.0), 0.0)
                    + particle_forces[particle_index][1]
                ) * 0.84
                speed = math.sqrt((vx * vx) + (vy * vy))
                speed_limit = 0.0062 + (
                    _safe_float(particle.get("text_density", 0.45), 0.45) * 0.0024
                )
                if speed > speed_limit and speed > 1e-8:
                    scale = speed_limit / speed
                    vx *= scale
                    vy *= scale

                particle["vx"] = vx
                particle["vy"] = vy
                particle["x"] = _clamp01(_safe_float(particle.get("x", 0.5), 0.5) + vx)
                particle["y"] = _clamp01(_safe_float(particle.get("y", 0.5), 0.5) + vy)

        if len(embedded_entries) > 1:
            for entry in embedded_entries:
                entry_index = int(entry.get("index", 0))
                if entry_index < 0 or entry_index >= len(offsets):
                    continue
                influence_x = 0.0
                influence_y = 0.0
                influence_radius = max(
                    0.08,
                    min(
                        0.26,
                        (_safe_float(entry.get("range", 0.03), 0.03) * 2.4) + 0.05,
                    ),
                )

                for particle in particles:
                    dx = _safe_float(particle.get("x", 0.5), 0.5) - _safe_float(
                        entry.get("x", 0.5), 0.5
                    )
                    dy = _safe_float(particle.get("y", 0.5), 0.5) - _safe_float(
                        entry.get("y", 0.5), 0.5
                    )
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance > influence_radius:
                        continue
                    if distance <= 1e-8:
                        continue

                    similarity = _semantic_vector_cosine(
                        entry.get("vector", []),
                        particle.get("vector", []),
                    )
                    falloff = _clamp01(1.0 - (distance / influence_radius))
                    density_mix = 0.58 + (
                        (
                            _safe_float(entry.get("text_density"), 0.45)
                            + _safe_float(particle.get("text_density"), 0.45)
                        )
                        * 0.24
                    )
                    strength = (
                        (0.00016 + (abs(similarity) * 0.00052)) * falloff * density_mix
                    )
                    direction = 1.0 if similarity >= 0.0 else -1.0
                    influence_x += (dx / distance) * strength * direction
                    influence_y += (dy / distance) * strength * direction

                max_influence = 0.0032 + (
                    _safe_float(entry.get("importance", 0.2), 0.2) * 0.0048
                )
                offsets[entry_index][0] += max(
                    -max_influence, min(max_influence, influence_x)
                )
                offsets[entry_index][1] += max(
                    -max_influence, min(max_influence, influence_y)
                )

        density_center_weight_total = sum(source_weights)
        if density_center_weight_total > 1e-8:
            density_center_x = (
                sum(
                    _safe_float(entry.get("x", 0.5), 0.5) * weight
                    for entry, weight in zip(embedded_entries, source_weights)
                )
                / density_center_weight_total
            )
            density_center_y = (
                sum(
                    _safe_float(entry.get("y", 0.5), 0.5) * weight
                    for entry, weight in zip(embedded_entries, source_weights)
                )
                / density_center_weight_total
            )
            density_spread = max(source_weights) - min(source_weights)
            center_pull = min(0.18, 0.06 + (density_spread * 0.09))
            for particle in particles:
                particle_x = _safe_float(particle.get("x", 0.5), 0.5)
                particle_y = _safe_float(particle.get("y", 0.5), 0.5)
                particle["x"] = _clamp01(
                    particle_x + ((density_center_x - particle_x) * center_pull)
                )
                particle["y"] = _clamp01(
                    particle_y + ((density_center_y - particle_y) * center_pull)
                )

        for particle in particles[:48]:
            hue = _semantic_vector_hue(list(particle.get("vector", [])))
            cohesion = _clamp01(_safe_float(particle.get("cohesion", 0.0), 0.0))
            saturation = max(0.52, min(0.92, 0.64 + (cohesion * 0.2)))
            value = max(0.72, min(0.98, 0.84 + (cohesion * 0.14)))
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (hue % 360.0) / 360.0,
                saturation,
                value,
            )
            size = (
                1.8
                + (_safe_float(particle.get("text_density", 0.45), 0.45) * 1.1)
                + (cohesion * 1.8)
            )
            x_norm = _clamp01(_safe_float(particle.get("x", 0.5), 0.5))
            y_norm = _clamp01(_safe_float(particle.get("y", 0.5), 0.5))
            embedding_particle_points.append(
                {
                    "x": round((x_norm * 2.0) - 1.0, 5),
                    "y": round(1.0 - (y_norm * 2.0), 5),
                    "size": round(size, 5),
                    "r": round(r_raw, 5),
                    "g": round(g_raw, 5),
                    "b": round(b_raw, 5),
                }
            )
            embedding_particle_nodes.append(
                {
                    "id": str(particle.get("id", "")),
                    "x": round(x_norm, 5),
                    "y": round(y_norm, 5),
                    "hue": round(hue, 4),
                    "cohesion": round(cohesion, 5),
                    "text_density": round(
                        _safe_float(particle.get("text_density", 0.45), 0.45), 5
                    ),
                }
            )

    file_graph["embedding_particles"] = embedding_particle_nodes

    position_by_id: dict[str, tuple[float, float]] = {}
    for index, entry in enumerate(entries):
        max_offset = 0.008 + (entry["importance"] * 0.014)
        offset_x = max(-max_offset, min(max_offset, offsets[index][0]))
        offset_y = max(-max_offset, min(max_offset, offsets[index][1]))
        x = round(_clamp01(entry["x"] + offset_x), 6)
        y = round(_clamp01(entry["y"] + offset_y), 6)
        entry["node"]["x"] = x
        entry["node"]["y"] = y
        position_by_id[entry["id"]] = (x, y)

    graph_nodes = file_graph.get("nodes", [])
    if isinstance(graph_nodes, list):
        for node in graph_nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("node_type", "")).strip().lower() != "file":
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            position = position_by_id.get(node_id)
            if position is None:
                continue
            node["x"] = position[0]
            node["y"] = position[1]

    return embedding_particle_points


def _build_backend_field_particles(
    *,
    file_graph: dict[str, Any] | None,
    presence_impacts: list[dict[str, Any]],
    resource_heartbeat: dict[str, Any],
    compute_jobs: list[dict[str, Any]],
    now: float,
) -> list[dict[str, float | str]]:
    if not presence_impacts:
        return []

    file_nodes_raw = (
        file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
    )
    file_nodes = [row for row in file_nodes_raw if isinstance(row, dict)]
    embedding_nodes_raw = (
        file_graph.get("embedding_particles", [])
        if isinstance(file_graph, dict)
        else []
    )
    embedding_nodes = [row for row in embedding_nodes_raw if isinstance(row, dict)]

    manifest_by_id = {
        str(row.get("id", "")).strip(): row
        for row in ENTITY_MANIFEST
        if str(row.get("id", "")).strip()
    }
    presence_to_field: dict[str, str] = {}
    for field_id, presence_id in FIELD_TO_PRESENCE.items():
        pid = str(presence_id).strip()
        if pid and pid not in presence_to_field:
            presence_to_field[pid] = str(field_id).strip()

    devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    if not isinstance(devices, dict):
        devices = {}
    resource_pressure = 0.0
    for device_key in ("cpu", "gpu1", "gpu2", "npu0"):
        row = devices.get(device_key, {})
        util = _safe_float(
            (row if isinstance(row, dict) else {}).get("utilization", 0.0), 0.0
        )
        resource_pressure = max(resource_pressure, _clamp01(util / 100.0))

    compute_pressure = _clamp01(len(compute_jobs) / 24.0)

    field_particles: list[dict[str, float | str]] = []
    now_mono = time.monotonic()
    live_ids: set[str] = set()

    def _node_field_similarity(
        node: dict[str, Any], target_field_id: str, target_presence_id: str
    ) -> float:
        if not target_field_id:
            return 0.0
        score = 0.0
        field_scores = node.get("field_scores", {})
        if isinstance(field_scores, dict):
            score = _clamp01(_safe_float(field_scores.get(target_field_id, 0.0), 0.0))
        dominant_field = str(node.get("dominant_field", "")).strip()
        if dominant_field and dominant_field == target_field_id:
            score = max(score, 0.85)
        dominant_presence = str(node.get("dominant_presence", "")).strip()
        if dominant_presence and dominant_presence == target_presence_id:
            score = max(score, 1.0)
        return _clamp01(score)

    with _DAIMO_DYNAMICS_LOCK:
        runtime = _DAIMO_DYNAMICS_CACHE.get("field_particles", {})
        if not isinstance(runtime, dict):
            runtime = {}
        # Handle nested runtime structure: {'particles': {...}, 'surfaces': {...}}
        particle_cache = runtime.get("particles", {})
        if not isinstance(particle_cache, dict):
            particle_cache = {}

        for impact in presence_impacts:
            presence_id = str(impact.get("id", "")).strip()
            if not presence_id:
                continue

            presence_meta = manifest_by_id.get(presence_id, {})
            anchor_x = _clamp01(
                _safe_float(
                    presence_meta.get("x", _stable_ratio(f"{presence_id}|anchor", 3)),
                    _stable_ratio(f"{presence_id}|anchor", 3),
                )
            )
            anchor_y = _clamp01(
                _safe_float(
                    presence_meta.get("y", _stable_ratio(f"{presence_id}|anchor", 9)),
                    _stable_ratio(f"{presence_id}|anchor", 9),
                )
            )
            base_hue = _safe_float(presence_meta.get("hue", 200.0), 200.0)
            target_field_id = presence_to_field.get(presence_id, "")
            presence_role, particle_mode = _particle_role_and_mode_for_presence(
                presence_id
            )

            affected_by = (
                impact.get("affected_by", {}) if isinstance(impact, dict) else {}
            )
            affects = impact.get("affects", {}) if isinstance(impact, dict) else {}
            file_influence = _clamp01(
                _safe_float(
                    (affected_by if isinstance(affected_by, dict) else {}).get(
                        "files", 0.0
                    ),
                    0.0,
                )
            )
            world_influence = _clamp01(
                _safe_float(
                    (affects if isinstance(affects, dict) else {}).get("world", 0.0),
                    0.0,
                )
            )
            ledger_influence = _clamp01(
                _safe_float(
                    (affects if isinstance(affects, dict) else {}).get("ledger", 0.0),
                    0.0,
                )
            )

            node_signals: list[dict[str, float]] = []
            cluster_map: dict[tuple[int, int], dict[str, float]] = {}
            cluster_bucket_size = 0.18
            local_density_score = 0.0
            for node in file_nodes:
                nx = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
                ny = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
                field_similarity = _node_field_similarity(
                    node, target_field_id, presence_id
                )
                embed_signal = _clamp01(
                    (_safe_float(node.get("embed_layer_count", 0.0), 0.0) / 3.0)
                    + (
                        0.35
                        if str(node.get("vecstore_collection", "")).strip()
                        else 0.0
                    )
                )
                signed_similarity = max(
                    -1.0,
                    min(
                        1.0,
                        (field_similarity * 0.72) + (embed_signal * 0.34) - 0.43,
                    ),
                )
                node_importance = _clamp01(
                    _safe_float(node.get("importance", 0.25), 0.25)
                )
                distance_to_anchor = math.sqrt(
                    ((nx - anchor_x) * (nx - anchor_x))
                    + ((ny - anchor_y) * (ny - anchor_y))
                )
                anchor_proximity = _clamp01(1.0 - (distance_to_anchor / 0.55))
                relevance = (
                    (abs(signed_similarity) * 0.62)
                    + (node_importance * 0.24)
                    + (anchor_proximity * 0.14)
                )
                if relevance < 0.12 and anchor_proximity <= 0.04:
                    continue

                if distance_to_anchor <= 0.24:
                    local_density_score += _clamp01(
                        1.0 - (distance_to_anchor / 0.24)
                    ) * (0.35 + (node_importance * 0.65))

                node_signals.append(
                    {
                        "x": nx,
                        "y": ny,
                        "signed": signed_similarity,
                        "importance": node_importance,
                        "relevance": relevance,
                    }
                )

                cluster_key = (
                    int(nx / cluster_bucket_size),
                    int(ny / cluster_bucket_size),
                )
                cluster_weight = (
                    0.24 + (node_importance * 0.64) + (abs(signed_similarity) * 0.82)
                )
                cluster_row = cluster_map.setdefault(
                    cluster_key,
                    {
                        "xw": 0.0,
                        "yw": 0.0,
                        "signed": 0.0,
                        "weight_raw": 0.0,
                    },
                )
                cluster_row["xw"] += nx * cluster_weight
                cluster_row["yw"] += ny * cluster_weight
                cluster_row["signed"] += signed_similarity * cluster_weight
                cluster_row["weight_raw"] += cluster_weight

            if len(node_signals) > 140:
                node_signals.sort(
                    key=lambda row: _safe_float(row.get("relevance", 0.0), 0.0),
                    reverse=True,
                )
                node_signals = node_signals[:140]

            clusters: list[dict[str, float]] = []
            for cluster_row in cluster_map.values():
                weight_raw = _safe_float(cluster_row.get("weight_raw", 0.0), 0.0)
                if weight_raw <= 1e-8:
                    continue
                clusters.append(
                    {
                        "x": _clamp01(
                            _safe_float(cluster_row.get("xw", 0.0), 0.0) / weight_raw
                        ),
                        "y": _clamp01(
                            _safe_float(cluster_row.get("yw", 0.0), 0.0) / weight_raw
                        ),
                        "signed": max(
                            -1.0,
                            min(
                                1.0,
                                _safe_float(cluster_row.get("signed", 0.0), 0.0)
                                / weight_raw,
                            ),
                        ),
                        "weight_raw": weight_raw,
                        "weight": 0.0,
                    }
                )
            clusters.sort(
                key=lambda row: _safe_float(row.get("weight_raw", 0.0), 0.0),
                reverse=True,
            )
            if len(clusters) > 8:
                clusters = clusters[:8]

            cluster_weight_total = 0.0
            for row in clusters:
                cluster_weight_total += _safe_float(row.get("weight_raw", 0.0), 0.0)
            if cluster_weight_total > 1e-8:
                for row in clusters:
                    row["weight"] = _clamp01(
                        _safe_float(row.get("weight_raw", 0.0), 0.0)
                        / cluster_weight_total
                    )

            local_density_ratio = _clamp01(local_density_score / 3.0)
            cluster_ratio = _clamp01(len(clusters) / 6.0)

            field_center_x = anchor_x
            field_center_y = anchor_y
            if clusters:
                primary_cluster = clusters[0]
                cluster_pull = _clamp01(
                    0.22
                    + (local_density_ratio * 0.42)
                    + (file_influence * 0.28)
                    + (cluster_ratio * 0.2)
                )
                field_center_x = _clamp01(
                    (anchor_x * (1.0 - cluster_pull))
                    + (
                        _safe_float(primary_cluster.get("x", anchor_x), anchor_x)
                        * cluster_pull
                    )
                )
                field_center_y = _clamp01(
                    (anchor_y * (1.0 - cluster_pull))
                    + (
                        _safe_float(primary_cluster.get("y", anchor_y), anchor_y)
                        * cluster_pull
                    )
                )

            raw_count = (
                4.0
                + (world_influence * 4.0)
                + (file_influence * 4.2)
                + (local_density_ratio * 8.6)
                + (cluster_ratio * 2.2)
                - (resource_pressure * 1.2)
            )
            particle_count = max(4, min(22, int(round(raw_count))))

            short_range_radius = 0.16 + (local_density_ratio * 0.04)
            interaction_radius = 0.36
            long_range_radius = 0.92

            for local_index in range(particle_count):
                particle_id = f"field:{presence_id}:{local_index}"
                live_ids.add(particle_id)
                cache_row = particle_cache.get(particle_id, {})
                if not isinstance(cache_row, dict):
                    cache_row = {}

                seed_ratio = _stable_ratio(f"{particle_id}|seed", local_index + 11)
                spread = max(0.018, 0.085 - (local_density_ratio * 0.045))
                home_dx = (
                    (_stable_ratio(f"{particle_id}|home-x", local_index + 19) * 2.0)
                    - 1.0
                ) * spread
                home_dy = (
                    (
                        (_stable_ratio(f"{particle_id}|home-y", local_index + 29) * 2.0)
                        - 1.0
                    )
                    * spread
                    * 0.82
                )
                home_x = _clamp01(field_center_x + home_dx)
                home_y = _clamp01(field_center_y + home_dy)

                px = _clamp01(_safe_float(cache_row.get("x", home_x), home_x))
                py = _clamp01(_safe_float(cache_row.get("y", home_y), home_y))
                pvx = _safe_float(cache_row.get("vx", 0.0), 0.0)
                pvy = _safe_float(cache_row.get("vy", 0.0), 0.0)

                fx = (home_x - px) * (0.18 + (ledger_influence * 0.18))
                fy = (home_y - py) * (0.18 + (ledger_influence * 0.18))

                for node in node_signals:
                    dx = _safe_float(node.get("x", 0.5), 0.5) - px
                    dy = _safe_float(node.get("y", 0.5), 0.5) - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= 1e-8 or distance > interaction_radius:
                        continue

                    signed_similarity = max(
                        -1.0,
                        min(1.0, _safe_float(node.get("signed", 0.0), 0.0)),
                    )
                    if abs(signed_similarity) <= 0.03:
                        continue
                    node_importance = _clamp01(
                        _safe_float(node.get("importance", 0.25), 0.25)
                    )

                    if distance <= short_range_radius:
                        falloff = _clamp01(1.0 - (distance / short_range_radius))
                        strength = (
                            (0.00125 + (node_importance * 0.00245))
                            * (falloff * falloff)
                            * (0.78 + (abs(signed_similarity) * 0.94))
                            * (0.72 + (file_influence * 0.58))
                        )
                    else:
                        transition = max(1e-8, interaction_radius - short_range_radius)
                        band = _clamp01((interaction_radius - distance) / transition)
                        strength = (
                            (0.00024 + (node_importance * 0.00082))
                            * band
                            * (0.46 + (abs(signed_similarity) * 0.54))
                        )

                    direction = 1.0 if signed_similarity >= 0.0 else -1.0
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                for cluster in clusters:
                    dx = _safe_float(cluster.get("x", 0.5), 0.5) - px
                    dy = _safe_float(cluster.get("y", 0.5), 0.5) - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= short_range_radius or distance > long_range_radius:
                        continue

                    cluster_signed = max(
                        -1.0,
                        min(1.0, _safe_float(cluster.get("signed", 0.0), 0.0)),
                    )
                    if abs(cluster_signed) <= 0.04:
                        continue
                    cluster_weight = _clamp01(
                        _safe_float(cluster.get("weight", 0.0), 0.0)
                    )
                    range_span = max(1e-8, long_range_radius - short_range_radius)
                    falloff = _clamp01((long_range_radius - distance) / range_span)
                    strength = (
                        (0.00012 + (cluster_weight * 0.00044))
                        * falloff
                        * (0.54 + (abs(cluster_signed) * 0.56))
                        * (0.6 + (cluster_ratio * 0.5))
                    )
                    direction = 1.0 if cluster_signed >= 0.0 else -1.0
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                for embed in embedding_nodes:
                    ex = _clamp01(_safe_float(embed.get("x", 0.5), 0.5))
                    ey = _clamp01(_safe_float(embed.get("y", 0.5), 0.5))
                    dx = ex - px
                    dy = ey - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= 1e-8 or distance > 0.23:
                        continue
                    falloff = _clamp01(1.0 - (distance / 0.23))
                    if falloff <= 0.0:
                        continue
                    cohesion = _clamp01(_safe_float(embed.get("cohesion", 0.0), 0.0))
                    density = _clamp01(
                        _safe_float(embed.get("text_density", 0.45), 0.45)
                    )
                    signed = (
                        (file_influence * 0.74)
                        + (cohesion * 0.52)
                        + (density * 0.26)
                        - 0.58
                    )
                    direction = 1.0 if signed >= 0.0 else -1.0
                    strength = (0.00042 + (abs(signed) * 0.00108)) * (falloff * falloff)
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                jitter_angle = (now * (0.34 + (compute_pressure * 0.4))) + (
                    local_index * 0.93
                )
                jitter_power = (
                    0.00006
                    + ((1.0 - resource_pressure) * 0.0001)
                    + (local_density_ratio * 0.00005)
                )
                fx += math.cos(jitter_angle) * jitter_power
                fy += math.sin(jitter_angle) * jitter_power

                simplex_phase = now * (0.28 + (compute_pressure * 0.24))
                simplex_amp = (
                    0.00005
                    + ((1.0 - resource_pressure) * 0.00007)
                    + (local_density_ratio * 0.00004)
                )
                simplex_seed = (local_index + 1) * 73 + len(presence_id)
                simplex_x = _simplex_noise_2d(
                    (px * 4.8) + simplex_phase + (local_index * 0.23),
                    (py * 4.8) + (simplex_phase * 0.69),
                    seed=simplex_seed,
                )
                simplex_y = _simplex_noise_2d(
                    (px * 4.8) + 13.0 + (simplex_phase * 0.57),
                    (py * 4.8) + 29.0 + simplex_phase,
                    seed=simplex_seed + 41,
                )
                fx += simplex_x * simplex_amp
                fy += simplex_y * simplex_amp

                damping = max(0.74, 0.91 - (resource_pressure * 0.13))
                vx = (pvx * damping) + fx
                vy = (pvy * damping) + fy
                speed = math.sqrt((vx * vx) + (vy * vy))
                speed_limit = (
                    0.0042
                    + ((1.0 - resource_pressure) * 0.0021)
                    + (local_density_ratio * 0.0018)
                )
                if speed > speed_limit and speed > 1e-8:
                    scale = speed_limit / speed
                    vx *= scale
                    vy *= scale

                nx = _clamp01(px + vx)
                ny = _clamp01(py + vy)
                particle_cache[particle_id] = {
                    "x": nx,
                    "y": ny,
                    "vx": vx,
                    "vy": vy,
                    "ts": now_mono,
                }

                saturation = max(
                    0.32,
                    min(
                        0.58,
                        0.4 + (world_influence * 0.16) + (local_density_ratio * 0.06),
                    ),
                )
                value = max(
                    0.38,
                    min(
                        0.68,
                        0.48
                        + (ledger_influence * 0.12)
                        + (local_density_ratio * 0.06)
                        - (resource_pressure * 0.12),
                    ),
                )
                r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                    (base_hue % 360.0) / 360.0,
                    saturation,
                    value,
                )
                particle_size = (
                    0.9
                    + (world_influence * 1.0)
                    + (file_influence * 0.8)
                    + (local_density_ratio * 0.9)
                )

                field_particles.append(
                    {
                        "id": particle_id,
                        "presence_id": presence_id,
                        "presence_role": presence_role,
                        "particle_mode": particle_mode,
                        "x": round(nx, 5),
                        "y": round(ny, 5),
                        "size": round(particle_size, 5),
                        "r": round(_clamp01(r_raw), 5),
                        "g": round(_clamp01(g_raw), 5),
                        "b": round(_clamp01(b_raw), 5),
                    }
                )

        # RENDER CHAOS BUTTERFLIES - convert chaos particles to field particles
        chaos_hue = 300.0  # Purple for chaos
        for pid, particle_state in particle_cache.items():
            if not isinstance(particle_state, dict):
                continue
            if not bool(particle_state.get("is_chaos_butterfly", False)):
                continue

            # Add to live_ids so they don't get cleaned up
            live_ids.add(pid)

            nx = _clamp01(_safe_float(particle_state.get("x", 0.5), 0.5))
            ny = _clamp01(_safe_float(particle_state.get("y", 0.5), 0.5))
            particle_size = _safe_float(particle_state.get("size", 0.5), 0.5)

            # Chaos butterflies have distinct purple color with high saturation
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (chaos_hue % 360.0) / 360.0,
                0.85,  # High saturation
                0.92,  # High brightness
            )

            field_particles.append(
                {
                    "id": pid,
                    "presence_id": "chaos_butterfly",
                    "presence_role": "chaos-agent",
                    "particle_mode": "noise-spreader",
                    "x": round(nx, 5),
                    "y": round(ny, 5),
                    "size": round(particle_size, 5),
                    "r": round(_clamp01(r_raw), 5),
                    "g": round(_clamp01(g_raw), 5),
                    "b": round(_clamp01(b_raw), 5),
                }
            )

        stale_before = now_mono - 180.0
        for pid in list(particle_cache.keys()):
            if pid in live_ids:
                continue
            row = particle_cache.get(pid, {})
            ts_value = _safe_float(
                (row if isinstance(row, dict) else {}).get("ts", 0.0), 0.0
            )
            if ts_value < stale_before:
                particle_cache.pop(pid, None)

        # Preserve nested runtime structure when saving back
        runtime["particles"] = particle_cache
        _DAIMO_DYNAMICS_CACHE["field_particles"] = runtime

    field_particles.sort(
        key=lambda row: (
            str(row.get("presence_id", "")),
            str(row.get("id", "")),
        )
    )
    return field_particles


_PARTICLE_ROLE_BY_PRESENCE: dict[str, str] = {
    "witness_thread": "crawl-routing",
    "keeper_of_receipts": "file-analysis",
    "mage_of_receipts": "image-captioning",
    "anchor_registry": "council-orchestration",
    "gates_of_truth": "compliance-gating",
}


def _particle_role_and_mode_for_presence(presence_id: str) -> tuple[str, str]:
    clean_presence_id = str(presence_id).strip()
    if not clean_presence_id:
        return "neutral", "neutral"
    role = str(_PARTICLE_ROLE_BY_PRESENCE.get(clean_presence_id, "")).strip()
    if not role:
        return "neutral", "neutral"
    return role, "role-bound"


def _dominant_eta_mu_field(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "f6", 1.0
    dominant_field = max(
        scores.keys(),
        key=lambda key: _safe_float(scores.get(key, 0.0), 0.0),
    )
    return dominant_field, _safe_float(scores.get(dominant_field, 0.0), 0.0)


def _infer_weaver_field_scores(node: dict[str, Any]) -> dict[str, float]:
    scores = {field_id: 0.0 for field_id in FIELD_TO_PRESENCE}
    kind = str(node.get("kind", "")).strip().lower()
    url = str(node.get("url", "") or node.get("label", "")).strip().lower()
    domain = str(node.get("domain", "")).strip().lower()
    title = str(node.get("title", "")).strip().lower()
    content_type = str(node.get("content_type", "")).strip().lower()

    if kind == "url":
        scores["f2"] += 0.24
        scores["f6"] += 0.24
        scores["f3"] += 0.12
    elif kind == "domain":
        scores["f2"] += 0.32
        scores["f8"] += 0.22
        scores["f3"] += 0.1
    elif kind == "content":
        if (
            content_type.startswith("image/")
            or content_type.startswith("audio/")
            or content_type.startswith("video/")
        ):
            scores["f1"] += 0.5
        else:
            scores["f6"] += 0.42
            scores["f3"] += 0.18

    combined = " ".join(filter(None, [url, domain, title, content_type]))
    tokens = _clean_tokens(combined)
    for token in tokens:
        for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
            if token in keywords:
                scores[field_id] += 0.06

    for needle in ("policy", "privacy", "terms", "robots", "compliance", "license"):
        if needle in combined:
            scores["f7"] += 0.15
    for needle in ("blog", "news", "article", "docs", "wiki", "readme"):
        if needle in combined:
            scores["f6"] += 0.12
            scores["f3"] += 0.08
    for needle in ("status", "dashboard", "metrics", "api", "admin"):
        if needle in combined:
            scores["f8"] += 0.11

    total = sum(max(0.0, value) for value in scores.values())
    if total <= 0.0:
        fallback = "f2" if kind in {"domain", "url"} else "f6"
        scores[fallback] = 1.0
        return scores

    normalized: dict[str, float] = {}
    for field_id, value in scores.items():
        normalized[field_id] = round(max(0.0, value) / total, 4)
    return normalized


def _crawler_node_importance(node: dict[str, Any], dominant_weight: float) -> float:
    kind = str(node.get("kind", "")).strip().lower()
    if kind == "domain":
        return _clamp01(0.35 + (dominant_weight * 0.55))
    if kind == "content":
        return _clamp01(0.28 + (dominant_weight * 0.5))

    depth = _safe_float(node.get("depth", 0), 0.0)
    status = str(node.get("status", "")).strip().lower()
    compliance = str(node.get("compliance", "")).strip().lower()
    score = 0.22 + (dominant_weight * 0.5)
    score += _clamp01(1.0 - (depth / 8.0)) * 0.18
    if status in {"fetched", "duplicate"}:
        score += 0.08
    if compliance in {"allowed", "pending"}:
        score += 0.05
    return _clamp01(score)


def _build_weaver_field_graph_uncached(
    part_root: Path,
    vault_root: Path,
    *,
    fetcher: Any | None = None,
) -> dict[str, Any]:
    del vault_root
    source_fetcher = fetcher or _world_web_symbol(
        "_fetch_weaver_graph_payload", _fetch_weaver_graph_payload
    )
    source_payload = source_fetcher(part_root)
    graph_payload = (
        source_payload.get("graph", {}) if isinstance(source_payload, dict) else {}
    )
    status_payload = (
        source_payload.get("status", {}) if isinstance(source_payload, dict) else {}
    )
    if not isinstance(graph_payload, dict):
        graph_payload = {}
    if not isinstance(status_payload, dict):
        status_payload = {}

    raw_nodes = graph_payload.get("nodes", [])
    raw_edges = graph_payload.get("edges", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_edges, list):
        raw_edges = []

    entity_lookup = {
        str(entity.get("id", "")): entity
        for entity in ENTITY_MANIFEST
        if str(entity.get("id", "")).strip()
    }

    field_nodes: list[dict[str, Any]] = []
    for field_id in CANONICAL_NAMED_FIELD_IDS:
        entity = entity_lookup.get(field_id)
        if entity is None:
            continue
        mapped_field = next(
            (
                key
                for key, presence_id in FIELD_TO_PRESENCE.items()
                if presence_id == field_id
            ),
            "f3",
        )
        field_nodes.append(
            {
                "id": f"crawler-field:{field_id}",
                "node_id": field_id,
                "node_type": "field",
                "field": mapped_field,
                "label": str(entity.get("en", field_id)),
                "label_ja": str(entity.get("ja", "")),
                "x": round(_safe_float(entity.get("x", 0.5), 0.5), 4),
                "y": round(_safe_float(entity.get("y", 0.5), 0.5), 4),
                "hue": int(_safe_float(entity.get("hue", 200), 200.0)),
            }
        )

    crawler_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_id_map: dict[str, str] = {}
    kind_counts: dict[str, int] = defaultdict(int)
    field_counts: dict[str, int] = defaultdict(int)

    for index, node in enumerate(raw_nodes[:WEAVER_GRAPH_NODE_LIMIT]):
        if not isinstance(node, dict):
            continue
        original_id = str(node.get("id", "")).strip()
        if not original_id:
            continue
        scores = _infer_weaver_field_scores(node)
        dominant_field, dominant_weight = _dominant_eta_mu_field(scores)
        dominant_presence = FIELD_TO_PRESENCE.get(dominant_field, "anchor_registry")
        anchor = entity_lookup.get(dominant_presence, {"x": 0.5, "y": 0.5, "hue": 200})
        seed = sha1(f"crawler|{original_id}|{index}".encode("utf-8")).digest()
        angle = (int.from_bytes(seed[0:2], "big") / 65535.0) * math.tau
        radius = 0.05 + (int.from_bytes(seed[2:4], "big") / 65535.0) * 0.2
        jitter_x = ((seed[4] / 255.0) - 0.5) * 0.042
        jitter_y = ((seed[5] / 255.0) - 0.5) * 0.042
        x = _clamp01(
            _safe_float(anchor.get("x", 0.5), 0.5) + math.cos(angle) * radius + jitter_x
        )
        y = _clamp01(
            _safe_float(anchor.get("y", 0.5), 0.5) + math.sin(angle) * radius + jitter_y
        )
        kind = str(node.get("kind", "url")).strip().lower() or "url"
        if kind == "domain":
            hue = 176
        elif kind == "content":
            hue = 22
        else:
            hue = int(_safe_float(anchor.get("hue", 200), 200.0))

        graph_node_id = f"crawler:{sha1(original_id.encode('utf-8')).hexdigest()[:16]}"
        node_id_map[original_id] = graph_node_id
        kind_counts[kind] += 1
        field_counts[dominant_field] += 1
        importance = _crawler_node_importance(node, dominant_weight)
        label = str(
            node.get("title", "")
            or node.get("domain", "")
            or node.get("label", "")
            or original_id
        )
        crawler_nodes.append(
            {
                "id": graph_node_id,
                "node_id": original_id,
                "node_type": "crawler",
                "crawler_kind": kind,
                "label": label,
                "x": round(x, 4),
                "y": round(y, 4),
                "hue": int(hue),
                "importance": round(importance, 4),
                "url": str(node.get("url", "") or ""),
                "domain": str(node.get("domain", "") or ""),
                "title": str(node.get("title", "") or ""),
                "status": str(node.get("status", "") or ""),
                "content_type": str(node.get("content_type", "") or ""),
                "compliance": str(node.get("compliance", "") or ""),
                "dominant_field": dominant_field,
                "dominant_presence": dominant_presence,
                "field_scores": {
                    key: round(_safe_float(value, 0.0), 4)
                    for key, value in scores.items()
                },
            }
        )

        ranked = sorted(
            [
                (str(field), _safe_float(weight, 0.0))
                for field, weight in scores.items()
            ],
            key=lambda row: row[1],
            reverse=True,
        )
        for edge_index, (field_id, weight) in enumerate(ranked[:2]):
            if weight <= 0:
                continue
            target_presence = FIELD_TO_PRESENCE.get(field_id, dominant_presence)
            if target_presence not in entity_lookup:
                continue
            edges.append(
                {
                    "id": f"crawler-edge:{graph_node_id}:{field_id}:{edge_index}",
                    "source": graph_node_id,
                    "target": f"crawler-field:{target_presence}",
                    "field": field_id,
                    "weight": round(_clamp01(weight), 4),
                    "kind": "categorizes",
                }
            )

    for edge in raw_edges[:WEAVER_GRAPH_EDGE_LIMIT]:
        if not isinstance(edge, dict):
            continue
        source_id = node_id_map.get(str(edge.get("source", "")).strip())
        target_id = node_id_map.get(str(edge.get("target", "")).strip())
        if not source_id or not target_id:
            continue
        kind = str(edge.get("kind", "hyperlink") or "hyperlink")
        if kind == "domain_membership":
            weight = 0.25
        elif kind == "content_membership":
            weight = 0.22
        elif kind == "canonical_redirect":
            weight = 0.34
        else:
            weight = 0.28
        edges.append(
            {
                "id": f"crawl-link:{str(edge.get('id', '')) or sha1((source_id + target_id + kind).encode('utf-8')).hexdigest()[:14]}",
                "source": source_id,
                "target": target_id,
                "field": "",
                "weight": round(weight, 4),
                "kind": kind,
            }
        )

    graph_counts = graph_payload.get("counts", {})
    if not isinstance(graph_counts, dict):
        graph_counts = {}
    nodes = [*field_nodes, *crawler_nodes]
    return {
        "record": ETA_MU_CRAWLER_GRAPH_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "endpoint": str(source_payload.get("source", "")),
            "service": "web-graph-weaver",
        },
        "status": status_payload,
        "nodes": nodes,
        "field_nodes": field_nodes,
        "crawler_nodes": crawler_nodes,
        "edges": edges,
        "stats": {
            "field_count": len(field_nodes),
            "crawler_count": len(crawler_nodes),
            "edge_count": len(edges),
            "kind_counts": dict(kind_counts),
            "field_counts": dict(field_counts),
            "nodes_total": int(
                _safe_float(
                    graph_counts.get("nodes_total", len(crawler_nodes)),
                    float(len(crawler_nodes)),
                )
            ),
            "edges_total": int(
                _safe_float(
                    graph_counts.get("edges_total", len(edges)), float(len(edges))
                )
            ),
            "url_nodes_total": int(
                _safe_float(
                    graph_counts.get("url_nodes_total", kind_counts.get("url", 0)),
                    float(kind_counts.get("url", 0)),
                )
            ),
        },
    }


def build_weaver_field_graph(part_root: Path, vault_root: Path) -> dict[str, Any]:
    fetcher = _world_web_symbol(
        "_fetch_weaver_graph_payload", _fetch_weaver_graph_payload
    )
    if fetcher is not _fetch_weaver_graph_payload:
        return _build_weaver_field_graph_uncached(
            part_root,
            vault_root,
            fetcher=fetcher,
        )

    substrate_root = _eta_mu_substrate_root(vault_root)
    cache_key = f"{part_root.resolve()}|{substrate_root}|{_weaver_service_base_url()}"
    now_monotonic = time.monotonic()
    with _WEAVER_GRAPH_CACHE_LOCK:
        cached_key = str(_WEAVER_GRAPH_CACHE.get("key", ""))
        cached_snapshot = _WEAVER_GRAPH_CACHE.get("snapshot")
        elapsed = now_monotonic - float(
            _WEAVER_GRAPH_CACHE.get("checked_monotonic", 0.0)
        )
        if (
            cached_snapshot is not None
            and cached_key == cache_key
            and elapsed < WEAVER_GRAPH_CACHE_SECONDS
        ):
            return _json_deep_clone(cached_snapshot)

    snapshot = _build_weaver_field_graph_uncached(part_root, vault_root)
    with _WEAVER_GRAPH_CACHE_LOCK:
        _WEAVER_GRAPH_CACHE["key"] = cache_key
        _WEAVER_GRAPH_CACHE["snapshot"] = _json_deep_clone(snapshot)
        _WEAVER_GRAPH_CACHE["checked_monotonic"] = now_monotonic
    return snapshot


def _build_nooi_field_cells(
    field_particles: list[dict[str, Any]],
    *,
    grid_cols: int = 18,
    grid_rows: int = 12,
) -> dict[str, Any]:
    cols = max(6, min(36, int(grid_cols)))
    rows = max(4, min(24, int(grid_rows)))
    cell_w = 1.0 / float(cols)
    cell_h = 1.0 / float(rows)

    cell_map: dict[str, dict[str, Any]] = {}
    for particle in field_particles if isinstance(field_particles, list) else []:
        if not isinstance(particle, dict):
            continue
        x = _clamp01(_safe_float(particle.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(particle.get("y", 0.5), 0.5))
        col = max(0, min(cols - 1, int(x * cols)))
        row = max(0, min(rows - 1, int(y * rows)))
        key = f"{col}:{row}"
        cell = cell_map.get(key)
        if not isinstance(cell, dict):
            cell = {
                "col": col,
                "row": row,
                "occupancy": 0,
                "sum_vx": 0.0,
                "sum_vy": 0.0,
                "sum_influence": 0.0,
                "sum_message": 0.0,
                "sum_route": 0.0,
                "presence_counts": {},
            }
            cell_map[key] = cell

        vx = _safe_float(particle.get("vx", 0.0), 0.0)
        vy = _safe_float(particle.get("vy", 0.0), 0.0)
        influence = _safe_float(particle.get("influence_power", -1.0), -1.0)
        if influence < 0.0:
            influence = _clamp01(
                (_safe_float(particle.get("message_probability", 0.0), 0.0) * 0.56)
                + (
                    _clamp01(abs(_safe_float(particle.get("drift_score", 0.0), 0.0)))
                    * 0.24
                )
                + (_safe_float(particle.get("route_probability", 0.0), 0.0) * 0.2)
            )

        presence_id = str(particle.get("presence_id", "")).strip()
        presence_counts = cell.get("presence_counts", {})
        if not isinstance(presence_counts, dict):
            presence_counts = {}
        if presence_id:
            presence_counts[presence_id] = (
                _safe_int(presence_counts.get(presence_id, 0), 0) + 1
            )
        cell["presence_counts"] = presence_counts

        cell["occupancy"] = _safe_int(cell.get("occupancy", 0), 0) + 1
        cell["sum_vx"] = _safe_float(cell.get("sum_vx", 0.0), 0.0) + vx
        cell["sum_vy"] = _safe_float(cell.get("sum_vy", 0.0), 0.0) + vy
        cell["sum_influence"] = _safe_float(
            cell.get("sum_influence", 0.0), 0.0
        ) + _clamp01(influence)
        cell["sum_message"] = _safe_float(cell.get("sum_message", 0.0), 0.0) + _clamp01(
            _safe_float(particle.get("message_probability", 0.0), 0.0)
        )
        cell["sum_route"] = _safe_float(cell.get("sum_route", 0.0), 0.0) + _clamp01(
            _safe_float(particle.get("route_probability", 0.0), 0.0)
        )

    cells: list[dict[str, Any]] = []
    max_influence = 0.0
    vector_peak = 0.0
    influence_total = 0.0

    for cell in cell_map.values():
        if not isinstance(cell, dict):
            continue
        occupancy = max(0, _safe_int(cell.get("occupancy", 0), 0))
        if occupancy <= 0:
            continue
        avg_vx = _safe_float(cell.get("sum_vx", 0.0), 0.0) / float(occupancy)
        avg_vy = _safe_float(cell.get("sum_vy", 0.0), 0.0) / float(occupancy)
        vector_mag = math.sqrt((avg_vx * avg_vx) + (avg_vy * avg_vy))
        avg_influence = _clamp01(
            _safe_float(cell.get("sum_influence", 0.0), 0.0) / float(occupancy)
        )
        avg_message = _clamp01(
            _safe_float(cell.get("sum_message", 0.0), 0.0) / float(occupancy)
        )
        avg_route = _clamp01(
            _safe_float(cell.get("sum_route", 0.0), 0.0) / float(occupancy)
        )
        occupancy_ratio = _clamp01(occupancy / 12.0)
        intensity = _clamp01(
            (avg_influence * 0.58) + (occupancy_ratio * 0.24) + (avg_message * 0.18)
        )

        presence_counts = (
            cell.get("presence_counts", {})
            if isinstance(cell.get("presence_counts"), dict)
            else {}
        )
        dominant_presence_id = ""
        if presence_counts:
            dominant_presence_id = max(
                sorted(presence_counts.keys()),
                key=lambda key: _safe_int(presence_counts.get(key, 0), 0),
            )

        col = max(0, min(cols - 1, _safe_int(cell.get("col", 0), 0)))
        row = max(0, min(rows - 1, _safe_int(cell.get("row", 0), 0)))

        cells.append(
            {
                "id": f"{col}:{row}",
                "col": col,
                "row": row,
                "x": round((float(col) + 0.5) / float(cols), 6),
                "y": round((float(row) + 0.5) / float(rows), 6),
                "occupancy": occupancy,
                "occupancy_ratio": round(occupancy_ratio, 6),
                "influence": round(avg_influence, 6),
                "intensity": round(intensity, 6),
                "message": round(avg_message, 6),
                "route": round(avg_route, 6),
                "vx": round(avg_vx, 6),
                "vy": round(avg_vy, 6),
                "vector_magnitude": round(vector_mag, 6),
                "dominant_presence_id": dominant_presence_id,
            }
        )
        max_influence = max(max_influence, avg_influence)
        vector_peak = max(vector_peak, vector_mag)
        influence_total += avg_influence

    cells.sort(
        key=lambda row: (
            -_safe_float(row.get("intensity", 0.0), 0.0),
            -_safe_int(row.get("occupancy", 0), 0),
            str(row.get("id", "")),
        )
    )

    active_cell_count = len(cells)
    return {
        "record": "eta-mu.nooi-field.v1",
        "schema_version": "nooi.field.v1",
        "grid_cols": cols,
        "grid_rows": rows,
        "cell_width": round(cell_w, 6),
        "cell_height": round(cell_h, 6),
        "active_cells": active_cell_count,
        "max_influence": round(max_influence, 6),
        "mean_influence": round(
            (influence_total / float(active_cell_count))
            if active_cell_count > 0
            else 0.0,
            6,
        ),
        "vector_peak": round(vector_peak, 6),
        "cells": cells,
    }


_RESOURCE_DAIMOI_TYPES: tuple[str, ...] = (
    "cpu",
    "ram",
    "disk",
    "network",
    "gpu",
    "npu",
)
_RESOURCE_DAIMOI_TYPE_ALIASES: dict[str, str] = {
    "gpu1": "gpu",
    "gpu2": "gpu",
    "gpu_intel": "gpu",
    "intel": "gpu",
    "npu0": "npu",
    "net": "network",
    "netup": "network",
    "netdown": "network",
}
_RESOURCE_DAIMOI_WALLET_FLOOR: dict[str, float] = {
    "cpu": 6.0,
    "ram": 6.0,
    "disk": 4.0,
    "network": 4.0,
    "gpu": 5.0,
    "npu": 5.0,
}
_RESOURCE_DAIMOI_WALLET_CAP: dict[str, float] = {
    "cpu": 48.0,
    "ram": 48.0,
    "disk": 32.0,
    "network": 32.0,
    "gpu": 40.0,
    "npu": 40.0,
}
_RESOURCE_DAIMOI_ACTION_BASE_COST = 0.00001
_RESOURCE_DAIMOI_ACTION_COST_MAX = 0.0028
_RESOURCE_DAIMOI_ACTION_SATISFIED_RATIO = 0.85

_SIMULATION_BOOT_RESET_LOCK = threading.Lock()
_SIMULATION_BOOT_RESET_APPLIED = False


def _maybe_reset_simulation_runtime_state() -> None:
    global _SIMULATION_BOOT_RESET_APPLIED
    reset_on_boot = str(
        os.getenv("SIMULATION_RESET_DAIMOI_ON_BOOT", "1") or "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not reset_on_boot:
        return
    with _SIMULATION_BOOT_RESET_LOCK:
        if _SIMULATION_BOOT_RESET_APPLIED:
            return
        get_presence_runtime_manager().reset()
        with _DAIMO_DYNAMICS_LOCK:
            _DAIMO_DYNAMICS_CACHE.clear()
        _SIMULATION_BOOT_RESET_APPLIED = True


def _canonical_resource_type(resource_type: str) -> str:
    key = str(resource_type or "").strip().lower()
    if not key:
        return ""
    if key in _RESOURCE_DAIMOI_TYPES:
        return key
    return str(_RESOURCE_DAIMOI_TYPE_ALIASES.get(key, "")).strip().lower()


def _core_resource_type_from_presence_id(presence_id: str) -> str:
    pid = str(presence_id or "").strip().lower()
    prefix = "presence.core."
    if not pid.startswith(prefix):
        return ""
    return _canonical_resource_type(pid[len(prefix) :])


def _normalize_resource_wallet(
    impact: dict[str, Any],
) -> dict[str, float]:
    wallet_raw = impact.get("resource_wallet", {})
    wallet: dict[str, float] = {}
    if isinstance(wallet_raw, dict):
        for key, value in wallet_raw.items():
            name_raw = str(key or "").strip().lower()
            if not name_raw:
                continue
            amount = max(0.0, _safe_float(value, 0.0))
            wallet[name_raw] = amount
            canonical = _canonical_resource_type(name_raw)
            if canonical:
                wallet[canonical] = max(amount, wallet.get(canonical, 0.0))
    impact["resource_wallet"] = wallet
    return wallet


def _presence_anchor_position(
    presence_id: str,
    impact: dict[str, Any],
    *,
    manifest_by_id: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    x_value = impact.get("x")
    y_value = impact.get("y")
    if x_value is not None and y_value is not None:
        return (
            _clamp01(_safe_float(x_value, 0.5)),
            _clamp01(_safe_float(y_value, 0.5)),
        )

    meta = manifest_by_id.get(presence_id, {})
    if isinstance(meta, dict):
        return (
            _clamp01(
                _safe_float(
                    meta.get("x", _stable_ratio(f"{presence_id}|anchor", 3)),
                    _stable_ratio(f"{presence_id}|anchor", 3),
                )
            ),
            _clamp01(
                _safe_float(
                    meta.get("y", _stable_ratio(f"{presence_id}|anchor", 11)),
                    _stable_ratio(f"{presence_id}|anchor", 11),
                )
            ),
        )

    return (
        _clamp01(_stable_ratio(f"{presence_id}|anchor", 3)),
        _clamp01(_stable_ratio(f"{presence_id}|anchor", 11)),
    )


def _resource_availability_ratio(
    resource_type: str,
    resource_heartbeat: dict[str, Any],
) -> float:
    kind = _canonical_resource_type(resource_type)
    if not kind:
        return 0.5

    devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    if not isinstance(devices, dict):
        devices = {}
    monitor = (
        resource_heartbeat.get("resource_monitor", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    if not isinstance(monitor, dict):
        monitor = {}

    usage_percent = 100.0
    if kind == "cpu":
        usage_percent = _safe_float(
            (
                devices.get("cpu", {}) if isinstance(devices.get("cpu"), dict) else {}
            ).get("utilization", monitor.get("cpu_percent", 100.0)),
            _safe_float(monitor.get("cpu_percent", 100.0), 100.0),
        )
    elif kind == "ram":
        usage_percent = _safe_float(monitor.get("memory_percent", 100.0), 100.0)
    elif kind == "disk":
        usage_percent = _safe_float(monitor.get("disk_percent", 100.0), 100.0)
    elif kind == "network":
        usage_percent = _safe_float(monitor.get("network_percent", 100.0), 100.0)
    elif kind == "gpu":
        gpu1 = _safe_float(
            (
                devices.get("gpu1", {}) if isinstance(devices.get("gpu1"), dict) else {}
            ).get("utilization", 100.0),
            100.0,
        )
        gpu2 = _safe_float(
            (
                devices.get("gpu2", {}) if isinstance(devices.get("gpu2"), dict) else {}
            ).get("utilization", 100.0),
            100.0,
        )
        usage_percent = min(gpu1, gpu2)
    elif kind == "npu":
        usage_percent = _safe_float(
            (
                devices.get("npu0", {}) if isinstance(devices.get("npu0"), dict) else {}
            ).get("utilization", 100.0),
            100.0,
        )

    usage_clamped = max(0.0, min(100.0, usage_percent))
    return _clamp01((100.0 - usage_clamped) / 100.0)


def _resource_need_ratio(
    impact: dict[str, Any],
    resource_type: str,
    *,
    queue_ratio: float,
) -> float:
    kind = _canonical_resource_type(resource_type)
    if not kind:
        return 0.0
    affected_by = impact.get("affected_by", {})
    if not isinstance(affected_by, dict):
        affected_by = {}

    wallet = _normalize_resource_wallet(impact)
    balance = max(0.0, _safe_float(wallet.get(kind, 0.0), 0.0))
    floor = max(0.1, _safe_float(_RESOURCE_DAIMOI_WALLET_FLOOR.get(kind, 4.0), 4.0))
    deficit_ratio = _clamp01((floor - balance) / floor)
    base_need = _clamp01(_safe_float(affected_by.get("resource", 0.0), 0.0))
    queue_push = _clamp01(_safe_float(queue_ratio, 0.0))
    is_sub_sim = bool(
        str(impact.get("presence_type", "")).strip() == "sub-sim"
        or str(impact.get("id", "")).strip().startswith("presence.sim.")
    )
    sub_sim_boost = 0.22 if is_sub_sim else 0.0
    return _clamp01(
        (base_need * 0.34)
        + (deficit_ratio * 0.46)
        + (queue_push * 0.18)
        + sub_sim_boost
    )


def _apply_resource_daimoi_emissions(
    *,
    field_particles: list[dict[str, Any]],
    presence_impacts: list[dict[str, Any]],
    resource_heartbeat: dict[str, Any],
    queue_ratio: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record": "eta-mu.resource-daimoi-flow.v1",
        "schema_version": "resource.daimoi.flow.v1",
        "emitter_rows": 0,
        "delivered_packets": 0,
        "total_transfer": 0.0,
        "by_resource": {},
        "recipients": [],
        "queue_ratio": round(_clamp01(_safe_float(queue_ratio, 0.0)), 6),
    }
    if not isinstance(field_particles, list) or not isinstance(presence_impacts, list):
        return summary

    manifest_by_id = {
        str(row.get("id", "")).strip(): row
        for row in ENTITY_MANIFEST
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }

    recipient_impacts: list[dict[str, Any]] = []
    fallback_recipients: list[dict[str, Any]] = []
    anchor_by_presence: dict[str, tuple[float, float]] = {}
    impact_by_id: dict[str, dict[str, Any]] = {}
    cpu_core_impact: dict[str, Any] | None = None
    for impact in presence_impacts:
        if not isinstance(impact, dict):
            continue
        presence_id = str(impact.get("id", "")).strip()
        if not presence_id:
            continue
        _normalize_resource_wallet(impact)
        impact_by_id[presence_id] = impact
        if presence_id == "presence.core.cpu":
            cpu_core_impact = impact
        if _core_resource_type_from_presence_id(presence_id):
            continue

        anchor_by_presence[presence_id] = _presence_anchor_position(
            presence_id,
            impact,
            manifest_by_id=manifest_by_id,
        )
        fallback_recipients.append(impact)
        # All presences with anchors are valid recipients
        recipient_impacts.append(impact)

    if not recipient_impacts:
        recipient_impacts = fallback_recipients
    if not recipient_impacts:
        return summary

    # Ambient fill for CPU Core
    if cpu_core_impact:
        wallet = cpu_core_impact.get("resource_wallet", {})
        if isinstance(wallet, dict):
            ambient_fill = 0.15  # Very high fill
            current = _safe_float(wallet.get("cpu", 0.0), 0.0)
            cap = _safe_float(_RESOURCE_DAIMOI_WALLET_CAP.get("cpu", 48.0), 48.0)
            wallet["cpu"] = min(cap, current + ambient_fill)

    resource_totals: dict[str, float] = {key: 0.0 for key in _RESOURCE_DAIMOI_TYPES}
    recipient_totals: dict[str, float] = {}
    packet_count = 0
    emitter_rows = 0

    for row in field_particles:
        if not isinstance(row, dict):
            continue
        if bool(row.get("is_nexus", False)):
            continue
        presence_id = str(row.get("presence_id", "")).strip()
        resource_type = _core_resource_type_from_presence_id(presence_id)
        if not resource_type:
            # Allow non-core presences to emit CPU if they have pressure
            resource_type = "cpu"

        emitter_cpu_cost = 0.0
        emitter_impact = impact_by_id.get(presence_id)
        if not isinstance(emitter_impact, dict):
            continue
        emitter_wallet = _normalize_resource_wallet(emitter_impact)

        # Pressure-based leak check
        resource_cap = max(
            0.1,
            _safe_float(_RESOURCE_DAIMOI_WALLET_CAP.get(resource_type, 32.0), 32.0),
        )
        resource_balance = max(
            0.0, _safe_float(emitter_wallet.get(resource_type, 0.0), 0.0)
        )
        resource_pressure = _clamp01(resource_balance / resource_cap)
        if resource_pressure < 0.15:
            # Not enough saturation to leak
            continue

        if resource_type != "cpu":
            emitter_cpu_cost = _RESOURCE_DAIMOI_ACTION_BASE_COST
            emitter_cpu_balance = max(
                0.0,
                _safe_float(emitter_wallet.get("cpu", 0.0), 0.0),
            )
            if emitter_cpu_balance + 1e-9 < emitter_cpu_cost:
                row["resource_action_blocked"] = True
                row["resource_block_reason"] = "cpu_wallet_required_for_emit"
                row["top_job"] = "resource_starved"
                continue

        emitter_rows += 1

        availability = _resource_availability_ratio(resource_type, resource_heartbeat)
        influence_power = _clamp01(
            _safe_float(
                row.get(
                    "influence_power",
                    row.get("message_probability", 0.0),
                ),
                0.0,
            )
        )
        route_probability = _clamp01(
            _safe_float(row.get("route_probability", 0.5), 0.5)
        )
        drift_score = _clamp01(abs(_safe_float(row.get("drift_score", 0.0), 0.0)))
        gravity_potential = max(
            0.0, _safe_float(row.get("gravity_potential", 0.0), 0.0)
        )
        gravity_signal = _clamp01(gravity_potential / (gravity_potential + 1.0))
        local_price = max(0.35, _safe_float(row.get("local_price", 1.0), 1.0))

        emit_amount = (
            0.25
            + (influence_power * 0.12)
            + (route_probability * 0.005)
            + (drift_score * 0.003)
            + (gravity_signal * 0.002)
        )
        # Modulate by pressure (higher pressure -> larger packets)
        emit_amount *= 0.3 + (availability * 0.7) + (resource_pressure * 0.5)
        emit_amount /= local_price
        emit_amount = max(0.0, emit_amount)
        if emit_amount <= 1e-7:
            continue

        px = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        py = _clamp01(_safe_float(row.get("y", 0.5), 0.5))

        best_target: dict[str, Any] | None = None
        best_target_id = ""
        best_score = -1.0
        for impact in recipient_impacts:
            target_id = str(impact.get("id", "")).strip()
            if not target_id:
                continue
            need_ratio = _resource_need_ratio(
                impact,
                resource_type,
                queue_ratio=queue_ratio,
            )
            ax, ay = anchor_by_presence.get(target_id, (0.5, 0.5))
            distance = math.sqrt(((ax - px) * (ax - px)) + ((ay - py) * (ay - py)))
            proximity = _clamp01(1.0 - min(1.0, distance / 1.15))
            score = (need_ratio * 0.72) + (proximity * 0.28)
            if score > best_score:
                best_score = score
                best_target = impact
                best_target_id = target_id

        if best_target is None or best_score <= 1e-8:
            continue

        target_wallet = _normalize_resource_wallet(best_target)
        wallet_cap = max(
            0.1,
            _safe_float(_RESOURCE_DAIMOI_WALLET_CAP.get(resource_type, 32.0), 32.0),
        )
        prior_value = max(0.0, _safe_float(target_wallet.get(resource_type, 0.0), 0.0))
        next_value = min(wallet_cap, prior_value + emit_amount)
        credited = max(0.0, next_value - prior_value)
        if credited <= 1e-8:
            continue

        # DELAYED CREDIT: Do not credit target immediately.
        # Resources are carried by the particle and delivered on absorption.
        # target_wallet[resource_type] = round(next_value, 6)
        # best_target["resource_wallet"] = target_wallet

        packet_count += 1
        resource_totals[resource_type] = (
            resource_totals.get(resource_type, 0.0) + credited
        )
        recipient_totals[best_target_id] = (
            recipient_totals.get(best_target_id, 0.0) + credited
        )

        row["resource_daimoi"] = True
        row["resource_type"] = resource_type
        row["resource_emit_amount"] = round(credited, 6)
        row["resource_target_presence_id"] = best_target_id
        row["resource_availability"] = round(availability, 6)
        row["resource_action_blocked"] = False
        row["top_job"] = "emit_resource_packet"
        row["job_probabilities"] = {
            "emit_resource_packet": round(0.74, 6),
            "invoke_resource_probe": round(0.16, 6),
            "deliver_message": round(0.10, 6),
        }
        # Decrement payload from source
        source_balance = max(
            0.0, _safe_float(emitter_wallet.get(resource_type, 0.0), 0.0)
        )
        source_after = max(0.0, source_balance - emit_amount)
        emitter_wallet[resource_type] = round(source_after, 6)

        if emitter_cpu_cost > 0.0 and isinstance(emitter_impact, dict):
            emitter_cpu_balance = max(
                0.0,
                _safe_float(emitter_wallet.get("cpu", 0.0), 0.0),
            )
            emitter_cpu_after = max(0.0, emitter_cpu_balance - emitter_cpu_cost)
            emitter_wallet["cpu"] = round(emitter_cpu_after, 6)
            row["resource_emit_cpu_cost"] = round(emitter_cpu_cost, 6)
            row["resource_emit_cpu_balance_after"] = round(emitter_cpu_after, 6)

        emitter_impact["resource_wallet"] = emitter_wallet

    summary["emitter_rows"] = int(emitter_rows)
    summary["delivered_packets"] = int(packet_count)
    summary["total_transfer"] = round(sum(resource_totals.values()), 6)
    summary["by_resource"] = {
        key: round(value, 6)
        for key, value in sorted(resource_totals.items())
        if value > 1e-8
    }
    summary["recipients"] = [
        {
            "presence_id": key,
            "credited": round(value, 6),
        }
        for key, value in sorted(
            recipient_totals.items(),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )[:16]
    ]
    return summary


def _apply_resource_daimoi_action_consumption(
    *,
    field_particles: list[dict[str, Any]],
    presence_impacts: list[dict[str, Any]],
    queue_ratio: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record": "eta-mu.resource-daimoi-consumption.v1",
        "schema_version": "resource.daimoi.consumption.v1",
        "action_packets": 0,
        "blocked_packets": 0,
        "consumed_total": 0.0,
        "by_resource": {},
        "starved_presences": [],
        "active_presences": [],
        "queue_ratio": round(_clamp01(_safe_float(queue_ratio, 0.0)), 6),
    }
    if not isinstance(field_particles, list) or not isinstance(presence_impacts, list):
        return summary

    impact_by_id: dict[str, dict[str, Any]] = {}
    for impact in presence_impacts:
        if not isinstance(impact, dict):
            continue
        presence_id = str(impact.get("id", "")).strip()
        if not presence_id:
            continue
        _normalize_resource_wallet(impact)
        impact_by_id[presence_id] = impact

    if not impact_by_id:
        return summary

    queue_push = _clamp01(_safe_float(queue_ratio, 0.0))
    consumed_by_resource: dict[str, float] = {
        key: 0.0 for key in _RESOURCE_DAIMOI_TYPES
    }
    consumed_by_presence: dict[str, float] = {}
    blocked_by_presence: dict[str, int] = {}
    blocked_packets = 0
    action_packets = 0

    for row in field_particles:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("presence_id", "")).strip()
        if not presence_id:
            continue
        if _core_resource_type_from_presence_id(presence_id):
            continue

        impact = impact_by_id.get(presence_id)
        if not isinstance(impact, dict):
            continue

        wallet = _normalize_resource_wallet(impact)
        focus_resource = "cpu"

        influence_power = _clamp01(
            _safe_float(
                row.get("influence_power", row.get("message_probability", 0.0)), 0.0
            )
        )
        message_probability = _clamp01(
            _safe_float(row.get("message_probability", 0.0), 0.0)
        )
        route_probability = _clamp01(
            _safe_float(row.get("route_probability", 0.0), 0.0)
        )
        drift_signal = _clamp01(abs(_safe_float(row.get("drift_score", 0.0), 0.0)))
        desired_cost = (
            _RESOURCE_DAIMOI_ACTION_BASE_COST
            + (influence_power * 0.00086)
            + (message_probability * 0.00054)
            + (route_probability * 0.00032)
            + (drift_signal * 0.00024)
            + (queue_push * 0.00028)
        )
        desired_cost = min(
            _RESOURCE_DAIMOI_ACTION_COST_MAX,
            max(_RESOURCE_DAIMOI_ACTION_BASE_COST, desired_cost),
        )

        available = max(0.0, _safe_float(wallet.get(focus_resource, 0.0), 0.0))
        consumed = min(available, desired_cost)
        remaining = max(0.0, available - consumed)
        wallet[focus_resource] = round(remaining, 6)
        impact["resource_wallet"] = wallet

        row["resource_consume_type"] = focus_resource
        row["resource_consume_amount"] = round(consumed, 6)
        row["resource_action_cost"] = round(desired_cost, 6)
        row["resource_balance_after"] = round(remaining, 6)

        action_packets += 1
        consumed_by_resource[focus_resource] = (
            consumed_by_resource.get(focus_resource, 0.0) + consumed
        )
        consumed_by_presence[presence_id] = (
            consumed_by_presence.get(presence_id, 0.0) + consumed
        )

        satisfied = desired_cost <= 1e-9 or consumed >= (
            desired_cost * _RESOURCE_DAIMOI_ACTION_SATISFIED_RATIO
        )
        if not satisfied:
            blocked_packets += 1
            blocked_by_presence[presence_id] = (
                blocked_by_presence.get(presence_id, 0) + 1
            )
            row["resource_action_blocked"] = True
            row["top_job"] = "resource_starved"
            row["message_probability"] = round(
                _clamp01(_safe_float(row.get("message_probability", 0.0), 0.0) * 0.22),
                6,
            )
            row["route_probability"] = round(
                _clamp01(_safe_float(row.get("route_probability", 0.0), 0.0) * 0.32),
                6,
            )
            row["influence_power"] = round(
                _clamp01(_safe_float(row.get("influence_power", 0.0), 0.0) * 0.28),
                6,
            )
            row["vx"] = round(_safe_float(row.get("vx", 0.0), 0.0) * 0.4, 6)
            row["vy"] = round(_safe_float(row.get("vy", 0.0), 0.0) * 0.4, 6)
            row["r"] = round(
                _clamp01((_safe_float(row.get("r", 0.4), 0.4) * 0.78) + 0.16),
                5,
            )
            row["g"] = round(
                _clamp01(_safe_float(row.get("g", 0.4), 0.4) * 0.42),
                5,
            )
            row["b"] = round(
                _clamp01(_safe_float(row.get("b", 0.4), 0.4) * 0.42),
                5,
            )
        else:
            row["resource_action_blocked"] = False
            top_job = str(row.get("top_job", "")).strip()
            if top_job in {"", "observe"}:
                row["top_job"] = "consume_resource_packet"

    summary["action_packets"] = int(action_packets)
    summary["blocked_packets"] = int(blocked_packets)
    summary["consumed_total"] = round(sum(consumed_by_resource.values()), 6)
    summary["by_resource"] = {
        resource: round(amount, 6)
        for resource, amount in sorted(consumed_by_resource.items())
        if amount > 1e-8
    }
    summary["starved_presences"] = [
        {
            "presence_id": presence_id,
            "blocked_packets": blocked,
        }
        for presence_id, blocked in sorted(
            blocked_by_presence.items(),
            key=lambda item: (-item[1], item[0]),
        )[:16]
    ]
    summary["active_presences"] = [
        {
            "presence_id": presence_id,
            "consumed": round(amount, 6),
        }
        for presence_id, amount in sorted(
            consumed_by_presence.items(),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )[:16]
        if amount > 1e-8
    ]
    return summary


def _snapshot_user_presence_runtime_state(
    now_monotonic: float,
    influence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now_unix = time.time()
    ttl_seconds = max(2.0, _safe_float(USER_PRESENCE_EVENT_TTL_SECONDS, 18.0), 18.0)
    influence_rows_raw = (
        influence.get("recent_user_inputs", []) if isinstance(influence, dict) else []
    )
    normalized_influence_rows: list[dict[str, Any]] = []
    if isinstance(influence_rows_raw, list):
        for index, row in enumerate(influence_rows_raw[:USER_PRESENCE_MAX_EVENTS]):
            if not isinstance(row, dict):
                continue
            fallback_age = min(ttl_seconds * 3.0, float(index) * 0.28)
            age_seconds = max(
                0.0,
                _safe_float(row.get("age_seconds", fallback_age), fallback_age),
            )
            ts_monotonic = now_monotonic - age_seconds
            event_id = str(row.get("id", f"influence:{index:02d}"))
            normalized_influence_rows.append(
                {
                    "id": event_id,
                    "kind": str(row.get("kind", "input") or "input"),
                    "target": str(row.get("target", "simulation") or "simulation"),
                    "message": str(row.get("message", "") or ""),
                    "embed_daimoi": bool(row.get("embed_daimoi", False)),
                    "x_ratio": row.get("x_ratio"),
                    "y_ratio": row.get("y_ratio"),
                    "ts_monotonic": ts_monotonic,
                    "age_seconds": round(age_seconds, 6),
                }
            )

    with _USER_PRESENCE_INPUT_LOCK:
        cache = _USER_PRESENCE_INPUT_CACHE
        target_x = _clamp01(
            _safe_float(
                cache.get("target_x", USER_PRESENCE_DEFAULT_X), USER_PRESENCE_DEFAULT_X
            )
        )
        target_y = _clamp01(
            _safe_float(
                cache.get("target_y", USER_PRESENCE_DEFAULT_Y), USER_PRESENCE_DEFAULT_Y
            )
        )
        anchor_x = _clamp01(
            _safe_float(
                cache.get("anchor_x", USER_PRESENCE_DEFAULT_X), USER_PRESENCE_DEFAULT_X
            )
        )
        anchor_y = _clamp01(
            _safe_float(
                cache.get("anchor_y", USER_PRESENCE_DEFAULT_Y), USER_PRESENCE_DEFAULT_Y
            )
        )
        last_pointer_monotonic = _safe_float(
            cache.get("last_pointer_monotonic", 0.0), 0.0
        )
        last_pointer_unix = _safe_float(cache.get("last_pointer_unix", 0.0), 0.0)
        last_input_monotonic = _safe_float(cache.get("last_input_monotonic", 0.0), 0.0)
        last_input_unix = _safe_float(cache.get("last_input_unix", 0.0), 0.0)
        latest_message = str(cache.get("latest_message", "") or "").strip()
        latest_target = str(cache.get("latest_target", "") or "").strip()
        seq = int(_safe_float(cache.get("seq", 0), 0.0))

        pointer_age_seconds = max(0.0, now_monotonic - last_pointer_monotonic)
        if last_pointer_monotonic <= 0.0 and last_pointer_unix > 0.0:
            pointer_age_seconds = max(0.0, now_unix - last_pointer_unix)
        if pointer_age_seconds > ttl_seconds:
            fallback_row = next(
                (
                    row
                    for row in normalized_influence_rows
                    if row.get("x_ratio") is not None and row.get("y_ratio") is not None
                ),
                None,
            )
            if isinstance(fallback_row, dict):
                target_x = _clamp01(
                    _safe_float(
                        fallback_row.get("x_ratio", USER_PRESENCE_DEFAULT_X),
                        USER_PRESENCE_DEFAULT_X,
                    )
                )
                target_y = _clamp01(
                    _safe_float(
                        fallback_row.get("y_ratio", USER_PRESENCE_DEFAULT_Y),
                        USER_PRESENCE_DEFAULT_Y,
                    )
                )
            else:
                target_x = USER_PRESENCE_DEFAULT_X
                target_y = USER_PRESENCE_DEFAULT_Y

        drift_alpha = _clamp01(_safe_float(USER_PRESENCE_DRIFT_ALPHA, 0.06))
        anchor_x = _clamp01(anchor_x + ((target_x - anchor_x) * drift_alpha))
        anchor_y = _clamp01(anchor_y + ((target_y - anchor_y) * drift_alpha))
        cache["anchor_x"] = anchor_x
        cache["anchor_y"] = anchor_y

        events_raw = cache.get("events", [])
        if not isinstance(events_raw, list):
            events_raw = []

        bounded_events: list[dict[str, Any]] = []
        for row in events_raw[-USER_PRESENCE_MAX_EVENTS:]:
            if not isinstance(row, dict):
                continue
            event_ts_mono = _safe_float(
                row.get("ts_monotonic", now_monotonic), now_monotonic
            )
            age_seconds = max(0.0, now_monotonic - event_ts_mono)
            if age_seconds > (ttl_seconds * 4.0):
                continue
            bounded_events.append(
                {
                    **row,
                    "age_seconds": round(age_seconds, 6),
                }
            )

        if not bounded_events and normalized_influence_rows:
            bounded_events = list(normalized_influence_rows)
        elif normalized_influence_rows:
            seen_ids = {
                str(row.get("id", "")).strip()
                for row in bounded_events
                if isinstance(row, dict)
            }
            for row in normalized_influence_rows:
                row_id = str(row.get("id", "")).strip()
                if row_id and row_id in seen_ids:
                    continue
                bounded_events.append(dict(row))
                if row_id:
                    seen_ids.add(row_id)
            bounded_events = bounded_events[-USER_PRESENCE_MAX_EVENTS:]

        if not latest_message and bounded_events:
            latest_message = str(bounded_events[-1].get("message", "") or "").strip()
        if not latest_target and bounded_events:
            latest_target = str(bounded_events[-1].get("target", "") or "").strip()

        if last_input_monotonic <= 0.0 and bounded_events:
            newest_age = min(
                (
                    _safe_float(row.get("age_seconds", ttl_seconds), ttl_seconds)
                    for row in bounded_events
                ),
                default=ttl_seconds,
            )
            last_input_monotonic = now_monotonic - max(0.0, newest_age)
        elif last_input_monotonic <= 0.0 and last_input_unix > 0.0:
            last_input_monotonic = now_monotonic - max(0.0, now_unix - last_input_unix)

        cache["events"] = bounded_events[-USER_PRESENCE_MAX_EVENTS:]

    return {
        "id": USER_PRESENCE_ID,
        "label": USER_PRESENCE_LABEL_EN,
        "label_ja": USER_PRESENCE_LABEL_JA,
        "target_x": round(target_x, 6),
        "target_y": round(target_y, 6),
        "anchor_x": round(anchor_x, 6),
        "anchor_y": round(anchor_y, 6),
        "latest_message": latest_message,
        "latest_target": latest_target,
        "sequence": seq,
        "events": bounded_events[-24:],
        "pointer_age_seconds": round(pointer_age_seconds, 6),
        "input_age_seconds": round(max(0.0, now_monotonic - last_input_monotonic), 6),
        "active": bool((now_monotonic - last_input_monotonic) <= (ttl_seconds * 2.0)),
    }


def _build_user_presence_embedded_daimoi_rows(
    user_presence_state: dict[str, Any],
    *,
    now: float,
) -> list[dict[str, Any]]:
    if not isinstance(user_presence_state, dict):
        return []
    events = user_presence_state.get("events", [])
    if not isinstance(events, list) or not events:
        return []

    anchor_x = _clamp01(_safe_float(user_presence_state.get("anchor_x", 0.5), 0.5))
    anchor_y = _clamp01(_safe_float(user_presence_state.get("anchor_y", 0.72), 0.72))
    ttl_seconds = max(2.0, _safe_float(USER_PRESENCE_EVENT_TTL_SECONDS, 18.0), 18.0)
    rows: list[dict[str, Any]] = []

    for index, event in enumerate(reversed(events[-24:])):
        if not isinstance(event, dict):
            continue
        if not bool(event.get("embed_daimoi", False)):
            continue
        age_seconds = max(0.0, _safe_float(event.get("age_seconds", 0.0), 0.0))
        life = _clamp01(1.0 - (age_seconds / ttl_seconds))
        if life <= 0.0:
            continue

        base_x = _clamp01(_safe_float(event.get("x_ratio", anchor_x), anchor_x))
        base_y = _clamp01(_safe_float(event.get("y_ratio", anchor_y), anchor_y))
        event_id = str(
            event.get("id", f"user-input:{index:02d}") or f"user-input:{index:02d}"
        ).strip()
        event_seed = int(hashlib.sha1(event_id.encode("utf-8")).hexdigest()[:8], 16)
        noise_scale = 0.007 + ((1.0 - life) * 0.006)
        noise_time = now * (0.44 + (life * 0.22))
        noise_x = _simplex_noise_2d(
            (base_x * 5.4) + (index * 0.27),
            noise_time,
            seed=(event_seed % 251) + 13,
        )
        noise_y = _simplex_noise_2d(
            (base_y * 5.4) + 19.0 + (index * 0.23),
            noise_time * 1.09,
            seed=(event_seed % 251) + 37,
        )
        x = _clamp01(((base_x * 0.62) + (anchor_x * 0.38)) + (noise_x * noise_scale))
        y = _clamp01(((base_y * 0.62) + (anchor_y * 0.38)) + (noise_y * noise_scale))
        vx = noise_x * noise_scale * 0.9
        vy = noise_y * noise_scale * 0.9

        message = str(event.get("message", "") or "").strip()
        kind = str(event.get("kind", "input") or "input").strip().lower() or "input"
        target = (
            str(event.get("target", "simulation") or "simulation").strip()
            or "simulation"
        )
        influence_power = _clamp01(0.34 + (life * 0.58))
        route_probability = _clamp01(0.26 + (life * 0.54))

        rows.append(
            {
                "id": f"{event_id}:daimoi",
                "record": "ημ.user-input-daimoi.v1",
                "schema_version": "user.input.daimoi.v1",
                "packet_record": "ημ.user-input-packet.v1",
                "packet_schema_version": "user.input.packet.v1",
                "presence_id": USER_PRESENCE_ID,
                "owner_presence_id": USER_PRESENCE_ID,
                "presence_role": "user-presence",
                "particle_mode": "role-bound",
                "x": round(x, 6),
                "y": round(y, 6),
                "vx": round(vx, 6),
                "vy": round(vy, 6),
                "size": round(1.0 + (life * 1.1), 6),
                "mass": round(0.6 + (life * 1.2), 6),
                "radius": round(0.26 + (life * 0.28), 6),
                "r": round(_clamp01(0.86 - (life * 0.08)), 5),
                "g": round(_clamp01(0.58 + (life * 0.22)), 5),
                "b": round(_clamp01(0.28 + (life * 0.12)), 5),
                "message_probability": round(_clamp01(life), 6),
                "route_probability": round(route_probability, 6),
                "influence_power": round(influence_power, 6),
                "top_job": "emit_user_input_message",
                "resource_daimoi": True,
                "resource_emit_amount": round(0.08 + (life * 0.22), 6),
                "resource_emit_type": "attention",
                "resource_emit_reason": kind,
                "resource_action_blocked": False,
                "packet_components": [
                    {
                        "component_id": f"{event_id}:message",
                        "component_type": "user-input",
                        "kind": kind,
                        "target": target,
                        "text": message[:180],
                        "weight": round(_clamp01(0.42 + (life * 0.5)), 6),
                    }
                ],
                "action_probabilities": {
                    "emit_user_input_message": round(_clamp01(0.64 + (life * 0.24)), 6),
                    "broadcast_ui_attention": round(_clamp01(0.36 + (life * 0.22)), 6),
                },
                "resource_signature": {
                    "attention": round(_clamp01(0.56 + (life * 0.34)), 6),
                    "memory": round(_clamp01(0.24 + (life * 0.16)), 6),
                    "compute": round(_clamp01(0.14 + (life * 0.12)), 6),
                },
            }
        )

    return rows[:24]


def build_simulation_state(
    catalog: dict[str, Any],
    myth_summary: dict[str, Any] | None = None,
    world_summary: dict[str, Any] | None = None,
    *,
    influence_snapshot: dict[str, Any] | None = None,
    queue_snapshot: dict[str, Any] | None = None,
    docker_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _prof_start = time.perf_counter()
    _maybe_reset_simulation_runtime_state()
    now = time.time()
    resource_budget_snapshot = _resource_monitor_snapshot()
    budget_devices = (
        resource_budget_snapshot.get("devices", {})
        if isinstance(resource_budget_snapshot, dict)
        else {}
    )
    budget_cpu = _safe_float(
        (budget_devices.get("cpu", {}) if isinstance(budget_devices, dict) else {}).get(
            "utilization", 0.0
        ),
        0.0,
    )
    sim_point_budget, sim_budget_slice = resolve_sim_point_budget_slice(
        cpu_utilization=budget_cpu,
        max_sim_points=MAX_SIM_POINTS,
    )

    queue_snapshot = queue_snapshot or {}
    influence = influence_snapshot or _INFLUENCE_TRACKER.snapshot(
        queue_snapshot=queue_snapshot
    )

    points: list[dict[str, float]] = []
    embedding_particle_points_raw: list[dict[str, float]] = []
    emitted_embedding_particles: list[dict[str, float]] = []
    field_particle_points_raw: list[dict[str, float | str]] = []
    emitted_field_particles: list[dict[str, float | str]] = []
    truth_graph_contract = _build_truth_graph_contract(None)
    view_graph_contract = _build_view_graph_contract(None)
    items = catalog.get("items", [])
    file_graph = catalog.get("file_graph") if isinstance(catalog, dict) else None
    if isinstance(file_graph, dict):
        file_graph, embedding_particle_points_raw = _prepare_file_graph_for_simulation(
            file_graph,
            now=now,
        )
    truth_graph_contract = _build_truth_graph_contract(
        file_graph if isinstance(file_graph, dict) else None
    )
    crawler_graph = catalog.get("crawler_graph") if isinstance(catalog, dict) else None
    file_graph, growth_guard = _apply_daimoi_growth_guard_to_file_graph(
        file_graph=file_graph if isinstance(file_graph, dict) else None,
        crawler_graph=crawler_graph if isinstance(crawler_graph, dict) else None,
        item_count=len(items) if isinstance(items, list) else 0,
        sim_point_budget=sim_point_budget,
        queue_snapshot=queue_snapshot,
        influence_snapshot=influence if isinstance(influence, dict) else {},
        cpu_utilization=budget_cpu,
    )
    file_graph, projection_event = _project_file_graph_for_simulation(
        file_graph=file_graph if isinstance(file_graph, dict) else None,
        influence_snapshot=influence if isinstance(influence, dict) else {},
    )
    if isinstance(projection_event, dict):
        guard_events = growth_guard.get("events", [])
        if isinstance(guard_events, list):
            guard_events = [dict(row) for row in guard_events if isinstance(row, dict)]
        else:
            guard_events = []
        guard_events.append(dict(projection_event))
        growth_guard["events"] = guard_events
    unified_nexus_runtime_graph = _build_unified_nexus_graph(
        file_graph=file_graph if isinstance(file_graph, dict) else None,
        crawler_graph=crawler_graph if isinstance(crawler_graph, dict) else None,
        include_crawler_in_file_nodes=True,
    )
    unified_nexus_output_graph = _build_unified_nexus_graph(
        file_graph=file_graph if isinstance(file_graph, dict) else None,
        crawler_graph=crawler_graph if isinstance(crawler_graph, dict) else None,
        include_crawler_in_file_nodes=False,
    )
    runtime_file_graph = (
        unified_nexus_runtime_graph
        if isinstance(unified_nexus_runtime_graph, dict)
        else (file_graph if isinstance(file_graph, dict) else None)
    )
    output_file_graph = (
        unified_nexus_output_graph
        if isinstance(unified_nexus_output_graph, dict)
        else (file_graph if isinstance(file_graph, dict) else None)
    )
    view_graph_contract = _build_view_graph_contract(
        output_file_graph if isinstance(output_file_graph, dict) else None
    )
    truth_state = catalog.get("truth_state") if isinstance(catalog, dict) else None
    logical_graph = catalog.get("logical_graph") if isinstance(catalog, dict) else None
    if not isinstance(logical_graph, dict):
        logical_graph = _build_logical_graph(
            catalog if isinstance(catalog, dict) else {}
        )
    pain_field = catalog.get("pain_field") if isinstance(catalog, dict) else None
    if not isinstance(pain_field, dict):
        pain_field = _build_pain_field(
            catalog if isinstance(catalog, dict) else {}, logical_graph
        )
    heat_values = _materialize_heat_values(
        catalog if isinstance(catalog, dict) else {}, pain_field
    )
    graph_file_nodes = (
        output_file_graph.get("file_nodes", [])
        if isinstance(output_file_graph, dict)
        else []
    )
    graph_crawler_nodes = (
        output_file_graph.get("crawler_nodes", [])
        if isinstance(output_file_graph, dict)
        else []
    )

    for idx, item in enumerate(items[:sim_point_budget]):
        key = f"{item.get('rel_path', '')}|{item.get('part', '')}|{item.get('kind', '')}|{idx}".encode(
            "utf-8"
        )
        digest = sha1(key).digest()

        x = (int.from_bytes(digest[0:2], "big") / 65535.0) * 2.0 - 1.0
        base_y = (int.from_bytes(digest[2:4], "big") / 65535.0) * 2.0 - 1.0
        phase = (digest[4] / 255.0) * math.tau
        speed = 0.4 + (digest[5] / 255.0) * 0.9
        wobble = math.sin(now * speed + phase) * 0.11
        y = max(-1.0, min(1.0, base_y + wobble))

        size = 2.8 + (digest[6] / 255.0) * 9.0
        r = 0.2 + (digest[7] / 255.0) * 0.75
        g = 0.2 + (digest[8] / 255.0) * 0.75
        b = 0.2 + (digest[9] / 255.0) * 0.75

        kind = str(item.get("kind", ""))
        if kind == "audio":
            size += 2.2
            r = min(1.0, r + 0.18)
            g = min(1.0, g + 0.16)
        elif kind == "video":
            b = min(1.0, b + 0.2)
        elif kind == "image":
            g = min(1.0, g + 0.1)

        points.append(
            {
                "x": round(x, 5),
                "y": round(y, 5),
                "size": round(size, 5),
                "r": round(r, 5),
                "g": round(g, 5),
                "b": round(b, 5),
            }
        )

    remaining_capacity = max(0, sim_point_budget - len(points))
    for node in list(graph_file_nodes)[:remaining_capacity]:
        if not isinstance(node, dict):
            continue
        x_norm = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y_norm = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        hue = _safe_float(node.get("hue", 200), 200.0)
        importance = _clamp01(_safe_float(node.get("importance", 0.4), 0.4))
        r_raw, g_raw, b_raw = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 0.58, 0.95)
        points.append(
            {
                "x": round((x_norm * 2.0) - 1.0, 5),
                "y": round(1.0 - (y_norm * 2.0), 5),
                "size": round(2.6 + (importance * 6.2), 5),
                "r": round(r_raw, 5),
                "g": round(g_raw, 5),
                "b": round(b_raw, 5),
            }
        )

    remaining_capacity = max(0, sim_point_budget - len(points))
    for particle in embedding_particle_points_raw[:remaining_capacity]:
        if not isinstance(particle, dict):
            continue
        particle_row = {
            "x": round(_safe_float(particle.get("x", 0.0), 0.0), 5),
            "y": round(_safe_float(particle.get("y", 0.0), 0.0), 5),
            "size": round(max(0.4, _safe_float(particle.get("size", 1.0), 1.0)), 5),
            "r": round(_clamp01(_safe_float(particle.get("r", 0.5), 0.5)), 5),
            "g": round(_clamp01(_safe_float(particle.get("g", 0.5), 0.5)), 5),
            "b": round(_clamp01(_safe_float(particle.get("b", 0.5), 0.5)), 5),
        }
        points.append(particle_row)
        emitted_embedding_particles.append(dict(particle_row))

    remaining_capacity = max(0, sim_point_budget - len(points))
    for node in list(graph_crawler_nodes)[:remaining_capacity]:
        if not isinstance(node, dict):
            continue
        x_norm = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y_norm = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        hue = _safe_float(node.get("hue", 180), 180.0)
        importance = _clamp01(_safe_float(node.get("importance", 0.3), 0.3))
        crawler_kind = str(node.get("crawler_kind", "url")).strip().lower()
        saturation = 0.66 if crawler_kind == "url" else 0.52
        value = 0.96 if crawler_kind == "url" else 0.9
        r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
            (hue % 360.0) / 360.0, saturation, value
        )
        points.append(
            {
                "x": round((x_norm * 2.0) - 1.0, 5),
                "y": round(1.0 - (y_norm * 2.0), 5),
                "size": round(2.2 + (importance * 5.0), 5),
                "r": round(r_raw, 5),
                "g": round(g_raw, 5),
                "b": round(b_raw, 5),
            }
        )

    truth_claims = (
        truth_state.get("claims", []) if isinstance(truth_state, dict) else []
    )
    if not isinstance(truth_claims, list):
        truth_claims = []
    truth_guard = truth_state.get("guard", {}) if isinstance(truth_state, dict) else {}
    if not isinstance(truth_guard, dict):
        truth_guard = {}
    truth_gate = truth_state.get("gate", {}) if isinstance(truth_state, dict) else {}
    if not isinstance(truth_gate, dict):
        truth_gate = {}
    truth_gate_blocked = bool(truth_gate.get("blocked", True))
    truth_guard_pass = bool(truth_guard.get("passes", False))

    remaining_capacity = max(0, sim_point_budget - len(points))
    if remaining_capacity > 0 and truth_claims:
        claim_x = 0.76
        claim_y = 0.54
        for claim_index, claim in enumerate(truth_claims[: min(3, remaining_capacity)]):
            if not isinstance(claim, dict):
                continue
            kappa = _clamp01(_safe_float(claim.get("kappa", 0.0), 0.0))
            status = str(claim.get("status", "undecided")).strip().lower()
            if status == "proved":
                hue = 136.0
            elif status == "refuted":
                hue = 12.0
            else:
                hue = 52.0
            if truth_guard_pass:
                hue = 150.0
            elif truth_gate_blocked:
                hue = max(0.0, hue - 12.0)

            spread = 0.012 + (claim_index * 0.014)
            offset_x = (
                (_stable_ratio(f"truth-claim:{claim_index}:x", claim_index + 3) * 2.0)
                - 1.0
            ) * spread
            offset_y = (
                (_stable_ratio(f"truth-claim:{claim_index}:y", claim_index + 7) * 2.0)
                - 1.0
            ) * spread
            x_norm = _clamp01(claim_x + offset_x)
            y_norm = _clamp01(claim_y + (offset_y * 0.86))
            saturation = 0.72 if status == "proved" else 0.78
            value = 0.96 if status == "proved" else 0.88
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (hue % 360.0) / 360.0, saturation, value
            )
            points.append(
                {
                    "x": round((x_norm * 2.0) - 1.0, 5),
                    "y": round(1.0 - (y_norm * 2.0), 5),
                    "size": round(3.2 + (kappa * 5.8), 5),
                    "r": round(r_raw, 5),
                    "g": round(g_raw, 5),
                    "b": round(b_raw, 5),
                }
            )

    counts = catalog.get("counts", {})

    entity_states = []
    for e in ENTITY_MANIFEST:
        base_seed = int(sha1(e["id"].encode("utf-8")).hexdigest()[:8], 16)
        t = now + (base_seed % 1000)
        bpm = 60 + (math.sin(t * 0.1) * 20) + ((base_seed % 20) - 10)

        vitals = {}
        for k, unit in e.get("flavor_vitals", {}).items():
            val_seed = (base_seed + hash(k)) % 1000
            val = abs(
                math.sin(t * (0.05 + (val_seed % 10) / 100)) * (100 + (val_seed % 50))
            )
            if unit == "%":
                val = val % 100
            vitals[k] = f"{val:.1f}{unit}"

        entity_states.append(
            {
                "id": e["id"],
                "bpm": round(bpm, 1),
                "stability": round(90 + math.sin(t * 0.02) * 9, 1),
                "resonance": round(e["freq"] + math.sin(t) * 2, 1),
                "vitals": vitals,
            }
        )

    echo_particles = []
    collection = _get_chroma_collection()
    if collection:
        try:
            results = collection.get(limit=12)
            docs = results.get("documents", [])
            for i, doc in enumerate(docs):
                seed = int(sha1(doc.encode("utf-8")).hexdigest()[:8], 16)
                t_off = now + (seed % 500)
                echo_particles.append(
                    {
                        "id": f"echo_{i}",
                        "text": doc[:24] + "...",
                        "x": 0.5 + math.sin(t_off * 0.15) * 0.35,
                        "y": 0.5 + math.cos(t_off * 0.12) * 0.35,
                        "hue": (200 + (seed % 100)) % 360,
                        "life": 0.5 + math.sin(t_off * 0.5) * 0.5,
                    }
                )
        except Exception:
            pass

    clicks_recent = int(influence.get("clicks_45s", 0))
    file_changes_recent = int(influence.get("file_changes_120s", 0))
    queue_pending_count = int(queue_snapshot.get("pending_count", 0))
    queue_event_count = int(queue_snapshot.get("event_count", 0))

    audio_count = int(counts.get("audio", 0))
    audio_ratio = _clamp01(audio_count / 12.0)
    click_ratio = _clamp01(clicks_recent / 18.0)
    file_ratio = _clamp01(file_changes_recent / 24.0)
    queue_ratio = _clamp01((queue_pending_count + queue_event_count * 0.25) / 16.0)
    resource_heartbeat = (
        influence.get("resource_heartbeat", {}) if isinstance(influence, dict) else {}
    )
    if not isinstance(resource_heartbeat, dict) or not resource_heartbeat:
        resource_heartbeat = resource_budget_snapshot
    resource_devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    resource_cpu_util = _safe_float(
        (
            resource_devices.get("cpu", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_gpu_util = _safe_float(
        (
            resource_devices.get("gpu1", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_npu_util = _safe_float(
        (
            resource_devices.get("npu0", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_ratio = _clamp01(
        max(resource_cpu_util, resource_gpu_util, resource_npu_util) / 100.0
    )

    river_flow_rate = round(
        1.2 + (audio_ratio * 4.4) + (file_ratio * 7.2) + (click_ratio * 2.6), 3
    )
    river_turbulence = round(_clamp01((file_ratio * 0.72) + (click_ratio * 0.4)), 4)

    manifest_lookup = {
        str(item.get("id", "")): item for item in ENTITY_MANIFEST if item.get("id")
    }
    presence_profile = _simulation_presence_profile()
    impact_order = _simulation_presence_impact_order()
    base_file = {
        "receipt_river": 0.94,
        "witness_thread": 0.38,
        "fork_tax_canticle": 0.84,
        "mage_of_receipts": 0.88,
        "keeper_of_receipts": 0.9,
        "anchor_registry": 0.64,
        "gates_of_truth": 0.73,
        "file_sentinel": 1.0,
        "file_organizer": 0.86,
        "health_sentinel_cpu": 0.58,
        "health_sentinel_gpu1": 0.54,
        "health_sentinel_gpu2": 0.5,
        "health_sentinel_npu0": 0.52,
    }
    base_click = {
        "receipt_river": 0.52,
        "witness_thread": 0.94,
        "fork_tax_canticle": 0.66,
        "mage_of_receipts": 0.57,
        "keeper_of_receipts": 0.61,
        "anchor_registry": 0.83,
        "gates_of_truth": 0.8,
        "file_sentinel": 0.55,
        "file_organizer": 0.62,
        "health_sentinel_cpu": 0.44,
        "health_sentinel_gpu1": 0.36,
        "health_sentinel_gpu2": 0.34,
        "health_sentinel_npu0": 0.32,
    }
    base_emit = {
        "receipt_river": 0.95,
        "witness_thread": 0.71,
        "fork_tax_canticle": 0.79,
        "mage_of_receipts": 0.73,
        "keeper_of_receipts": 0.81,
        "anchor_registry": 0.68,
        "gates_of_truth": 0.75,
        "file_sentinel": 0.82,
        "file_organizer": 0.78,
        "health_sentinel_cpu": 0.68,
        "health_sentinel_gpu1": 0.74,
        "health_sentinel_gpu2": 0.7,
        "health_sentinel_npu0": 0.77,
    }
    base_resource = {
        "receipt_river": 0.2,
        "witness_thread": 0.15,
        "fork_tax_canticle": 0.31,
        "mage_of_receipts": 0.28,
        "keeper_of_receipts": 0.26,
        "anchor_registry": 0.24,
        "gates_of_truth": 0.33,
        "file_sentinel": 0.44,
        "file_organizer": 0.36,
        "health_sentinel_cpu": 0.92,
        "health_sentinel_gpu1": 0.96,
        "health_sentinel_gpu2": 0.9,
        "health_sentinel_npu0": 0.95,
    }

    presence_impacts: list[dict[str, Any]] = []
    for presence_id in impact_order:
        if presence_id == FILE_SENTINEL_PROFILE["id"]:
            meta = FILE_SENTINEL_PROFILE
        elif presence_id == FILE_ORGANIZER_PROFILE["id"]:
            meta = FILE_ORGANIZER_PROFILE
        else:
            meta = manifest_lookup.get(
                presence_id,
                {
                    "id": presence_id,
                    "en": presence_id.replace("_", " ").title(),
                    "ja": "",
                },
            )

        file_influence = _clamp01(
            (file_ratio * float(base_file.get(presence_id, 0.5))) + (queue_ratio * 0.22)
        )
        click_influence = _clamp01(
            click_ratio * float(base_click.get(presence_id, 0.5))
        )
        resource_influence = _clamp01(
            resource_ratio * float(base_resource.get(presence_id, 0.22))
        )
        total_influence = _clamp01(
            (file_influence * 0.52)
            + (click_influence * 0.28)
            + (resource_influence * 0.2)
        )
        emits_flow = _clamp01(
            (total_influence * 0.72)
            + (audio_ratio * float(base_emit.get(presence_id, 0.5)) * 0.35)
        )

        if presence_id == "receipt_river":
            notes_en = (
                "River flow accelerates when files move and witnesses touch the field."
            )
            notes_ja = "ファイル変化と触れた証人で、川の流れは加速する。"
        elif presence_id == "file_sentinel":
            notes_en = "Auto-committing ghost stages proof paths before the gate asks."
            notes_ja = "自動コミットの幽霊は、門に問われる前に証明経路を段取る。"
        elif presence_id == "file_organizer":
            notes_en = "Organizer presence groups files into concept clusters from embedding space."
            notes_ja = "分類師プレゼンスは埋め込み空間から概念クラスタを編成する。"
        elif presence_id == "fork_tax_canticle":
            notes_en = "Fork tax pressure rises with unresolved file drift."
            notes_ja = "未解決のファイルドリフトでフォーク税圧は上がる。"
        elif presence_id == "witness_thread":
            notes_en = "Mouse touches tighten witness linkage across presences."
            notes_ja = "マウスの接触はプレゼンス間の証人連結を強める。"
        elif presence_id == "health_sentinel_cpu":
            notes_en = (
                "CPU sentinel throttles particle budgets when host pressure rises."
            )
            notes_ja = "CPU哨戒はホスト圧上昇時に粒子予算を絞る。"
        elif presence_id == "health_sentinel_gpu1":
            notes_en = (
                "GPU1 sentinel maps throughput and thermals into backend selection."
            )
            notes_ja = "GPU1哨戒は処理量と熱をバックエンド選択へ写像する。"
        elif presence_id == "health_sentinel_gpu2":
            notes_en = "GPU2 sentinel absorbs burst load to keep field vectors stable."
            notes_ja = "GPU2哨戒は突発負荷を吸収し、場のベクトルを安定化する。"
        elif presence_id == "health_sentinel_npu0":
            notes_en = (
                "NPU sentinel tracks efficient inferencing for embedding pathways."
            )
            notes_ja = "NPU哨戒は埋め込み経路の効率推論を監視する。"
        else:
            notes_en = "Presence responds to blended file and witness pressure."
            notes_ja = "このプレゼンスはファイル圧と証人圧の混合に応答する。"

        presence_impacts.append(
            {
                "id": presence_id,
                "en": str(meta.get("en", "Presence")),
                "ja": str(meta.get("ja", "プレゼンス")),
                "affected_by": {
                    "files": round(file_influence, 4),
                    "clicks": round(click_influence, 4),
                    "resource": round(resource_influence, 4),
                },
                "affects": {
                    "world": round(emits_flow, 4),
                    "ledger": round(_clamp01(total_influence * 0.86), 4),
                },
                "notes_en": notes_en,
                "notes_ja": notes_ja,
            }
        )

    witness_meta = manifest_lookup.get(
        "witness_thread",
        {
            "id": "witness_thread",
            "en": "Witness Thread",
            "ja": "証人の糸",
        },
    )
    witness_impact = next(
        (item for item in presence_impacts if item.get("id") == "witness_thread"), None
    )
    lineage: list[dict[str, str]] = []
    seen_lineage_refs: set[str] = set()

    for target in list(influence.get("recent_click_targets", []))[:6]:
        ref = str(target).strip()
        if not ref or ref in seen_lineage_refs:
            continue
        seen_lineage_refs.add(ref)
        lineage.append(
            {
                "kind": "touch",
                "ref": ref,
                "why_en": "Witness touch linked this target into continuity.",
                "why_ja": "証人の接触がこの対象を連続線へ接続した。",
            }
        )

    for path in list(influence.get("recent_file_paths", []))[:8]:
        ref = str(path).strip()
        if not ref or ref in seen_lineage_refs:
            continue
        seen_lineage_refs.add(ref)
        lineage.append(
            {
                "kind": "file",
                "ref": ref,
                "why_en": "File drift supplied provenance for witness continuity.",
                "why_ja": "ファイルドリフトが証人連続性の来歴を供給した。",
            }
        )

    if not lineage:
        lineage.append(
            {
                "kind": "idle",
                "ref": "awaiting-touch",
                "why_en": "No recent witness touch; continuity waits for the next trace.",
                "why_ja": "直近の証人接触なし。次の痕跡を待機中。",
            }
        )

    linked_presence_ids = [
        str(item.get("id", ""))
        for item in sorted(
            [row for row in presence_impacts if row.get("id") != "witness_thread"],
            key=lambda row: float(row.get("affected_by", {}).get("clicks", 0.0)),
            reverse=True,
        )
        if str(item.get("id", "")).strip()
    ][:4]

    # Create core resource presences
    core_resources, cpu_core_emitter_enabled, cpu_daimoi_stop_percent = (
        _simulation_core_resource_emitters(cpu_utilization=resource_cpu_util)
    )
    manager = get_presence_runtime_manager()
    for resource in core_resources:
        presence_id = f"presence.core.{resource}"
        if presence_id not in {p["id"] for p in presence_impacts}:
            meta = manifest_lookup.get(presence_id, {})
            anchor_x = _clamp01(_safe_float(meta.get("x", 0.5), 0.5))
            anchor_y = _clamp01(_safe_float(meta.get("y", 0.5), 0.5))
            presence_impacts.append(
                {
                    "id": presence_id,
                    "label": meta.get("en", f"Silent Core - {resource.upper()}"),
                    "label_ja": meta.get("ja", ""),
                    "presence_type": "core",
                    "x": anchor_x,
                    "y": anchor_y,
                    "hue": _safe_float(meta.get("hue", 0), 0.0),
                    "resource_wallet": {
                        resource: 1000.0
                    },  # Infinite source effectively
                    "active_nexus_id": "",
                    "pinned_node_ids": [],
                }
            )

    user_presence_state = _snapshot_user_presence_runtime_state(
        time.monotonic(),
        influence,
    )
    user_recent_events = user_presence_state.get("events", [])
    if not isinstance(user_recent_events, list):
        user_recent_events = []
    user_recent_events = [row for row in user_recent_events if isinstance(row, dict)][
        -8:
    ]

    presence_impacts = [
        row
        for row in presence_impacts
        if isinstance(row, dict) and str(row.get("id", "")).strip() != USER_PRESENCE_ID
    ]
    user_wallet = manager.get_state(USER_PRESENCE_ID).get("resource_wallet", {})
    if not isinstance(user_wallet, dict):
        user_wallet = {}

    user_presence_impact: dict[str, Any] = {
        "id": USER_PRESENCE_ID,
        "en": USER_PRESENCE_LABEL_EN,
        "ja": USER_PRESENCE_LABEL_JA,
        "presence_type": "operator",
        "x": _clamp01(
            _safe_float(
                user_presence_state.get("anchor_x", USER_PRESENCE_DEFAULT_X),
                USER_PRESENCE_DEFAULT_X,
            )
        ),
        "y": _clamp01(
            _safe_float(
                user_presence_state.get("anchor_y", USER_PRESENCE_DEFAULT_Y),
                USER_PRESENCE_DEFAULT_Y,
            )
        ),
        "hue": 28.0,
        "affected_by": {
            "files": round(_clamp01(file_ratio * 0.22), 4),
            "clicks": round(_clamp01(click_ratio * 0.96), 4),
            "resource": round(_clamp01(resource_ratio * 0.2), 4),
        },
        "affects": {
            "world": round(_clamp01(0.24 + (click_ratio * 0.56)), 4),
            "ledger": round(_clamp01(0.18 + (queue_ratio * 0.42)), 4),
        },
        "notes_en": "Operator input emits user daimoi packets and steers the user nexus anchor.",
        "notes_ja": "操作者入力はユーザーダイモイを放出し、ユーザーネクサス錨点をゆっくり誘導する。",
        "active_nexus_id": "nexus.user.cursor",
        "pinned_node_ids": [
            str(user_presence_state.get("latest_target", "simulation") or "simulation")[
                :120
            ]
        ],
        "user_message": str(user_presence_state.get("latest_message", "") or "")[:240],
        "recent_inputs": [
            {
                "id": str(row.get("id", ""))[:80],
                "kind": str(row.get("kind", "input"))[:40],
                "target": str(row.get("target", ""))[:180],
                "message": str(row.get("message", ""))[:240],
                "age_seconds": round(_safe_float(row.get("age_seconds", 0.0), 0.0), 4),
                "embed_daimoi": bool(row.get("embed_daimoi", False)),
            }
            for row in user_recent_events
        ],
    }
    if user_wallet:
        user_presence_impact["resource_wallet"] = user_wallet
    presence_impacts.append(user_presence_impact)

    witness_thread_state = {
        "id": str(witness_meta.get("id", "witness_thread")),
        "en": str(witness_meta.get("en", "Witness Thread")),
        "ja": str(witness_meta.get("ja", "証人の糸")),
        "continuity_index": round(
            _clamp01((click_ratio * 0.54) + (file_ratio * 0.3) + (queue_ratio * 0.16)),
            4,
        ),
        "click_pressure": round(click_ratio, 4),
        "file_pressure": round(file_ratio, 4),
        "linked_presences": linked_presence_ids,
        "lineage": lineage[:6],
        "notes_en": "",
        "notes_ja": "",
    }

    fork_tax = dict(influence.get("fork_tax", {}))
    if not fork_tax:
        fork_tax = {
            "law_en": "Pay the fork tax; annotate every drift with proof.",
            "law_ja": "フォーク税は法。",
            "debt": 0.0,
            "paid": 0.0,
            "balance": 0.0,
            "paid_ratio": 1.0,
        }
    if not str(fork_tax.get("law_ja", "")).strip():
        fork_tax["law_ja"] = "フォーク税は法。"

    ghost = dict(influence.get("ghost", {}))
    ghost.setdefault("id", FILE_SENTINEL_PROFILE["id"])
    ghost.setdefault("en", FILE_SENTINEL_PROFILE["en"])
    ghost.setdefault("ja", FILE_SENTINEL_PROFILE["ja"])
    ghost["auto_commit_pulse"] = round(
        _clamp01(
            float(ghost.get("auto_commit_pulse", 0.0))
            + (file_ratio * 0.12)
            + (queue_ratio * 0.08)
        ),
        4,
    )
    ghost["actions_60s"] = int((file_changes_recent * 0.5) + (queue_event_count * 0.8))
    ghost["status_en"] = str(ghost.get("status_en", "gate idle"))
    ghost["status_ja"] = str(ghost.get("status_ja", "門前で待機中"))

    ds = _build_daimoi_state(
        heat_values,
        pain_field,
        queue_ratio=queue_ratio,
        resource_ratio=resource_ratio,
    )
    if isinstance(ds, dict):
        ds["growth_guard"] = growth_guard
        deployment_rows = growth_guard.get("daimoi", [])
        if isinstance(deployment_rows, list) and deployment_rows:
            daimo_rows = ds.get("daimoi", [])
            if isinstance(daimo_rows, list):
                daimo_rows.extend(
                    [dict(row) for row in deployment_rows if isinstance(row, dict)]
                )
                ds["daimoi"] = daimo_rows
            ds["active"] = bool(
                ds.get("active", False) or growth_guard.get("active", False)
            )

    compute_jobs_raw = influence.get("compute_jobs", [])
    compute_jobs = compute_jobs_raw if isinstance(compute_jobs_raw, list) else []
    compute_summary_raw = influence.get("compute_summary", {})
    compute_summary = (
        compute_summary_raw if isinstance(compute_summary_raw, dict) else {}
    )
    compute_jobs_count = int(
        _safe_float(influence.get("compute_jobs_180s", len(compute_jobs)), 0.0)
    )

    for presence in presence_impacts:
        wallet = manager.get_state(presence["id"]).get("resource_wallet", {})
        if wallet:
            presence["resource_wallet"] = wallet

    # Process resource economy cycle
    if docker_snapshot:
        sync_sub_sim_presences(presence_impacts, docker_snapshot)
        process_resource_cycle(presence_impacts, now=now)

    # Calculate daimoi particles (using presence_impacts which now has updated wallets)
    _prof_pre_particles = time.perf_counter()
    particle_backend_mode = (
        str(os.getenv("SIM_PARTICLE_BACKEND", "python") or "python").strip().lower()
    )
    if os.getenv("SIM_PROFILE_INTERNAL") == "1":
        print(f"DEBUG: particle_backend_mode={particle_backend_mode}", flush=True)
    if particle_backend_mode in {"c", "cdb", "native", "double-buffer-c"}:
        try:
            from .c_double_buffer_backend import build_double_buffer_field_particles

            field_particle_points_raw, daimoi_probabilistic = (
                build_double_buffer_field_particles(
                    file_graph=runtime_file_graph,
                    presence_impacts=presence_impacts,
                    resource_heartbeat=resource_heartbeat,
                    compute_jobs=compute_jobs,
                    queue_ratio=queue_ratio,
                    now=now,
                    entity_manifest=list(ENTITY_MANIFEST),
                )
            )
        except Exception as exc:
            field_particle_points_raw, daimoi_probabilistic = (
                build_probabilistic_daimoi_particles(
                    file_graph=runtime_file_graph,
                    presence_impacts=presence_impacts,
                    resource_heartbeat=resource_heartbeat,
                    compute_jobs=compute_jobs,
                    queue_ratio=queue_ratio,
                    now=now,
                )
            )
            if isinstance(daimoi_probabilistic, dict):
                daimoi_probabilistic["backend"] = "python-fallback"
                daimoi_probabilistic["backend_error"] = (
                    f"{exc.__class__.__name__}:{exc}"
                )
    else:
        field_particle_points_raw, daimoi_probabilistic = (
            build_probabilistic_daimoi_particles(
                file_graph=runtime_file_graph,
                presence_impacts=presence_impacts,
                resource_heartbeat=resource_heartbeat,
                compute_jobs=compute_jobs,
                queue_ratio=queue_ratio,
                now=now,
            )
        )

    normalized_field_particles: list[dict[str, float | str]] = []
    for particle in field_particle_points_raw:
        if not isinstance(particle, dict):
            continue
        x_norm = _clamp01(_safe_float(particle.get("x", 0.5), 0.5))
        y_norm = _clamp01(_safe_float(particle.get("y", 0.5), 0.5))
        normalized_row: dict[str, Any] = {
            "id": str(particle.get("id", "")),
            "presence_id": str(particle.get("presence_id", "")),
            "presence_role": str(particle.get("presence_role", "neutral")),
            "particle_mode": str(particle.get("particle_mode", "neutral")),
            "x": round(x_norm, 5),
            "y": round(y_norm, 5),
            "size": round(max(0.6, _safe_float(particle.get("size", 1.0), 1.0)), 5),
            "r": round(_clamp01(_safe_float(particle.get("r", 0.4), 0.4)), 5),
            "g": round(_clamp01(_safe_float(particle.get("g", 0.4), 0.4)), 5),
            "b": round(_clamp01(_safe_float(particle.get("b", 0.4), 0.4)), 5),
        }
        for key in (
            "record",
            "schema_version",
            "packet_record",
            "packet_schema_version",
            "is_nexus",
            "owner_presence_id",
            "target_presence_id",
            "top_job",
            "package_entropy",
            "message_probability",
            "mass",
            "radius",
            "collision_count",
            "source_node_id",
            "graph_node_id",
            "graph_distance_cost",
            "gravity_potential",
            "local_price",
            "node_saturation",
            "route_node_id",
            "drift_score",
            "drift_gravity_term",
            "drift_cost_term",
            "drift_gravity_delta",
            "drift_gravity_delta_scalar",
            "drift_cost_latency_term",
            "drift_cost_congestion_term",
            "drift_cost_semantic_term",
            "drift_cost_upkeep_term",
            "route_gravity_mode",
            "route_resource_focus",
            "route_resource_focus_weight",
            "route_resource_focus_delta",
            "route_resource_focus_contribution",
            "route_probability",
            "selected_edge_cost",
            "selected_edge_health",
            "selected_edge_affinity",
            "selected_edge_saturation",
            "selected_edge_upkeep_penalty",
            "valve_pressure_term",
            "valve_gravity_term",
            "valve_affinity_term",
            "valve_saturation_term",
            "valve_health_term",
            "valve_score_proxy",
            "influence_power",
            "vx",
            "vy",
        ):
            if key in particle:
                normalized_row[key] = particle.get(key)
        for key in (
            "job_probabilities",
            "packet_components",
            "resource_signature",
            "absorb_sampler",
            "action_probabilities",
            "behavior_actions",
            "embedding_seed_preview",
            "embedding_curr_preview",
            "last_collision_matrix",
        ):
            value = particle.get(key)
            if isinstance(value, (dict, list)):
                normalized_row[key] = value
        normalized_field_particles.append(normalized_row)

    user_embedded_daimoi = _build_user_presence_embedded_daimoi_rows(
        user_presence_state,
        now=now,
    )
    if user_embedded_daimoi:
        normalized_field_particles.extend(user_embedded_daimoi)

    resource_daimoi = _apply_resource_daimoi_emissions(
        field_particles=normalized_field_particles,
        presence_impacts=presence_impacts,
        resource_heartbeat=resource_heartbeat,
        queue_ratio=queue_ratio,
    )
    resource_consumption = _apply_resource_daimoi_action_consumption(
        field_particles=normalized_field_particles,
        presence_impacts=presence_impacts,
        queue_ratio=queue_ratio,
    )
    if isinstance(daimoi_probabilistic, dict):
        daimoi_probabilistic["resource_daimoi"] = dict(resource_daimoi)
        daimoi_probabilistic["resource_consumption"] = dict(resource_consumption)

    remaining_capacity = max(0, sim_point_budget - len(points))
    for particle in normalized_field_particles[:remaining_capacity]:
        points.append(
            {
                "x": round((_safe_float(particle.get("x", 0.5), 0.5) * 2.0) - 1.0, 5),
                "y": round(1.0 - (_safe_float(particle.get("y", 0.5), 0.5) * 2.0), 5),
                "size": round(max(0.6, _safe_float(particle.get("size", 1.0), 1.0)), 5),
                "r": round(_clamp01(_safe_float(particle.get("r", 0.4), 0.4)), 5),
                "g": round(_clamp01(_safe_float(particle.get("g", 0.4), 0.4)), 5),
                "b": round(_clamp01(_safe_float(particle.get("b", 0.4), 0.4)), 5),
            }
        )

    emitted_field_particles = normalized_field_particles

    _NOOI_FIELD.decay(DAIMO_DT_SECONDS)
    for particle in emitted_field_particles:
        if not isinstance(particle, dict):
            continue
        _NOOI_FIELD.deposit(
            _safe_float(particle.get("x", 0.5), 0.5),
            _safe_float(particle.get("y", 0.5), 0.5),
            _safe_float(particle.get("vx", 0.0), 0.0),
            _safe_float(particle.get("vy", 0.0), 0.0),
        )
    nooi_field = _NOOI_FIELD.get_grid_snapshot()

    distributed_runtime = sync_presence_runtime_state(
        field_particles=emitted_field_particles,
        presence_impacts=presence_impacts,
        queue_ratio=queue_ratio,
        resource_ratio=resource_ratio,
    )

    presence_dynamics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulation_budget": {
            "point_limit": int(sim_point_budget),
            "point_limit_max": int(MAX_SIM_POINTS),
            "cpu_utilization": round(resource_cpu_util, 2),
            "slice_offload": sim_budget_slice,
        },
        "emission_policy": {
            "presence_profile": presence_profile,
            "core_resource_emitters": list(core_resources),
            "cpu_core_emitter_enabled": bool(cpu_core_emitter_enabled),
            "cpu_daimoi_stop_percent": round(cpu_daimoi_stop_percent, 2),
        },
        "click_events": clicks_recent,
        "file_events": file_changes_recent,
        "recent_click_targets": list(influence.get("recent_click_targets", []))[:6],
        "recent_file_paths": list(influence.get("recent_file_paths", []))[:8],
        "resource_heartbeat": resource_heartbeat,
        "compute_jobs_180s": max(0, compute_jobs_count),
        "compute_summary": compute_summary,
        "compute_jobs": compute_jobs[:32],
        "field_particles_record": "ημ.field-particles.v1",
        "field_particles": emitted_field_particles,
        "resource_daimoi": resource_daimoi,
        "resource_consumption": resource_consumption,
        "user_presence": user_presence_state,
        "user_embedded_daimoi_count": len(user_embedded_daimoi),
        "user_input_messages": [
            {
                "id": str(row.get("id", ""))[:80],
                "kind": str(row.get("kind", "input"))[:40],
                "target": str(row.get("target", ""))[:180],
                "message": str(row.get("message", ""))[:260],
                "age_seconds": round(_safe_float(row.get("age_seconds", 0.0), 0.0), 4),
            }
            for row in (
                user_presence_state.get("events", [])
                if isinstance(user_presence_state.get("events", []), list)
                else []
            )[-12:]
        ],
        "nooi_field": nooi_field,
        "river_flow": {
            "unit": "m3/s",
            "rate": river_flow_rate,
            "turbulence": river_turbulence,
        },
        "ghost": ghost,
        "fork_tax": fork_tax,
        "witness_thread": witness_thread_state,
        "presence_impacts": presence_impacts,
        "growth_guard": growth_guard,
        "daimoi_probabilistic_record": str(
            daimoi_probabilistic.get("record", "")
            if isinstance(daimoi_probabilistic, dict)
            else ""
        ),
        "daimoi_probabilistic": (
            daimoi_probabilistic
            if isinstance(daimoi_probabilistic, dict)
            else {
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
                "mean_package_entropy": 0.0,
                "mean_message_probability": 0.0,
                "matrix_mean": {"ss": 0.0, "sc": 0.0, "cs": 0.0, "cc": 0.0},
                "behavior_defaults": ["deflect", "diffuse"],
            }
        ),
        "daimoi_behavior_defaults": ["deflect", "diffuse"],
        "distributed_runtime": distributed_runtime,
    }

    default_truth_state = {
        "record": "ημ.truth-state.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": {"status": "undecided"},
        "claims": [],
        "proof": {"entries": []},
        "guard": {},
        "gate": {},
    }

    simulation = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(points),
        "audio": int(counts.get("audio", 0)),
        "image": int(counts.get("image", 0)),
        "video": int(counts.get("video", 0)),
        "points": points,
        "embedding_particles": emitted_embedding_particles,
        "field_particles": emitted_field_particles,
        "file_graph": output_file_graph
        if isinstance(output_file_graph, dict)
        else {
            "record": ETA_MU_FILE_GRAPH_RECORD,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "",
                "pending_count": 0,
                "processed_count": 0,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 0,
                "last_ingested_at": "",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [],
            "embedding_particles": [],
            "edges": [],
            "stats": {
                "field_count": 0,
                "file_count": 0,
                "edge_count": 0,
                "kind_counts": {},
                "field_counts": {},
                "knowledge_entries": 0,
            },
        },
        "truth_graph": truth_graph_contract,
        "view_graph": view_graph_contract,
        "crawler_graph": crawler_graph
        if isinstance(crawler_graph, dict)
        else {
            "record": ETA_MU_CRAWLER_GRAPH_RECORD,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {"endpoint": "", "service": "web-graph-weaver"},
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
        },
        "truth_state": truth_state
        if isinstance(truth_state, dict)
        else default_truth_state,
        "logical_graph": logical_graph,
        "pain_field": pain_field,
        "heat_values": heat_values,
        "daimoi": ds,
        "entities": entity_states,
        "echoes": echo_particles,
        "fork_tax": fork_tax,
        "ghost": ghost,
        "presence_dynamics": presence_dynamics,
        "myth": myth_summary or {},
        "world": world_summary or {},
        # =====================================================================
        # CANONICAL UNIFIED MODEL (v2) - single source of truth
        # =====================================================================
        # The nexus_graph is the unified graph - all other graph payloads are
        # projections of this. The field_registry contains the bounded shared
        # fields that all presences contribute to.
        # See specs/drafts/part64-deep-research-09-unified-nexus-graph.md
        # See specs/drafts/part64-deep-research-10-shared-fields-daimoi-dynamics.md
        # =====================================================================
        "nexus_graph": _build_canonical_nexus_graph(
            file_graph=output_file_graph,
            crawler_graph=crawler_graph,
            logical_graph=logical_graph,
            include_crawler=True,
            include_logical=True,
        ),
        "field_registry": _build_field_registry(
            catalog=catalog,
            graph_runtime=(
                daimoi_probabilistic.get("graph_runtime")
                if isinstance(daimoi_probabilistic, dict)
                else None
            ),
            kernel_width=0.3,
            decay_rate=0.1,
            resolution=32,
        ),
    }
    if os.getenv("SIM_PROFILE_INTERNAL") == "1":
        print(
            f"DEBUG PROFILE: pre_particles={(_prof_pre_particles - _prof_start) * 1000:.2f}ms, total={(time.perf_counter() - _prof_start) * 1000:.2f}ms",
            flush=True,
        )
    return simulation


def _stream_particle_effective_mass(row: dict[str, Any]) -> float:
    semantic_text_chars = max(
        0.0, _safe_float(row.get("semantic_text_chars", 0.0), 0.0)
    )
    semantic_mass = max(0.0, _safe_float(row.get("semantic_mass", 0.0), 0.0))
    daimoi_energy = max(0.0, _safe_float(row.get("daimoi_energy", 0.0), 0.0))
    message_probability = max(
        0.0,
        _safe_float(row.get("message_probability", 0.0), 0.0),
    )
    package_entropy = max(0.0, _safe_float(row.get("package_entropy", 0.0), 0.0))

    text_term = math.log1p(semantic_text_chars) * 0.32
    energy_term = math.log1p((daimoi_energy * 2.8) + (message_probability * 3.5)) * 0.42
    entropy_term = package_entropy * 0.08
    mass_term = semantic_mass * 0.15
    return max(0.35, min(8.5, 0.5 + text_term + energy_term + entropy_term + mass_term))


def _stream_particle_collision_radius(row: dict[str, Any], mass_value: float) -> float:
    size_value = max(0.35, _safe_float(row.get("size", 1.0), 1.0))
    return max(
        0.004, min(0.035, (size_value * 0.0044) + (math.sqrt(mass_value) * 0.0014))
    )


def _resolve_semantic_particle_collisions(rows: list[dict[str, Any]]) -> None:
    if not isinstance(rows, list):
        return
    particle_rows = [row for row in rows if isinstance(row, dict)]
    if len(particle_rows) < 2:
        return

    mass_by_id: dict[str, float] = {}
    radius_by_id: dict[str, float] = {}
    for row in particle_rows:
        particle_id = str(row.get("id", "") or id(row))
        mass_value = _stream_particle_effective_mass(row)
        mass_by_id[particle_id] = mass_value
        radius_by_id[particle_id] = _stream_particle_collision_radius(row, mass_value)

    cell_size = 0.04
    grid: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in particle_rows:
        x_value = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y_value = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        gx = int(x_value / cell_size)
        gy = int(y_value / cell_size)
        grid[(gx, gy)].append(row)

    restitution = 0.88
    separation_percent = 0.72
    collision_count_updates: dict[str, int] = defaultdict(int)

    visited_pairs: set[tuple[str, str]] = set()
    for (gx, gy), bucket in list(grid.items()):
        neighbors: list[dict[str, Any]] = []
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                neighbors.extend(grid.get((nx, ny), []))

        for row_a in bucket:
            id_a = str(row_a.get("id", "") or id(row_a))
            x_a = _clamp01(_safe_float(row_a.get("x", 0.5), 0.5))
            y_a = _clamp01(_safe_float(row_a.get("y", 0.5), 0.5))
            vx_a = _safe_float(row_a.get("vx", 0.0), 0.0)
            vy_a = _safe_float(row_a.get("vy", 0.0), 0.0)
            mass_a = max(0.2, _safe_float(mass_by_id.get(id_a, 1.0), 1.0))
            inv_mass_a = 1.0 / mass_a
            radius_a = _safe_float(radius_by_id.get(id_a, 0.01), 0.01)

            for row_b in neighbors:
                if row_a is row_b:
                    continue
                id_b = str(row_b.get("id", "") or id(row_b))
                pair = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                if pair in visited_pairs:
                    continue
                visited_pairs.add(pair)

                x_b = _clamp01(_safe_float(row_b.get("x", 0.5), 0.5))
                y_b = _clamp01(_safe_float(row_b.get("y", 0.5), 0.5))
                dx = x_b - x_a
                dy = y_b - y_a
                distance = math.hypot(dx, dy)

                mass_b = max(0.2, _safe_float(mass_by_id.get(id_b, 1.0), 1.0))
                inv_mass_b = 1.0 / mass_b
                radius_b = _safe_float(radius_by_id.get(id_b, 0.01), 0.01)
                min_distance = radius_a + radius_b
                if distance >= min_distance:
                    continue

                if distance < 1e-6:
                    seed = int(
                        hashlib.sha1(f"{id_a}|{id_b}".encode("utf-8")).hexdigest()[:8],
                        16,
                    )
                    theta = float(seed % 6283) / 1000.0
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    distance = 1e-6
                else:
                    nx = dx / distance
                    ny = dy / distance

                vx_b = _safe_float(row_b.get("vx", 0.0), 0.0)
                vy_b = _safe_float(row_b.get("vy", 0.0), 0.0)
                rel_vx = vx_a - vx_b
                rel_vy = vy_a - vy_b
                vel_normal = (rel_vx * nx) + (rel_vy * ny)

                if vel_normal < 0.0:
                    impulse = (-(1.0 + restitution) * vel_normal) / max(
                        1e-6, inv_mass_a + inv_mass_b
                    )
                    impulse_x = impulse * nx
                    impulse_y = impulse * ny
                    vx_a += impulse_x * inv_mass_a
                    vy_a += impulse_y * inv_mass_a
                    vx_b -= impulse_x * inv_mass_b
                    vy_b -= impulse_y * inv_mass_b

                    tangent_x = rel_vx - (vel_normal * nx)
                    tangent_y = rel_vy - (vel_normal * ny)
                    tangent_norm = math.hypot(tangent_x, tangent_y)
                    if tangent_norm > 1e-6:
                        tangent_x /= tangent_norm
                        tangent_y /= tangent_norm
                        tangent_impulse = min(
                            abs(impulse) * 0.18,
                            abs((rel_vx * tangent_x) + (rel_vy * tangent_y)),
                        )
                        vx_a -= tangent_impulse * tangent_x * inv_mass_a
                        vy_a -= tangent_impulse * tangent_y * inv_mass_a
                        vx_b += tangent_impulse * tangent_x * inv_mass_b
                        vy_b += tangent_impulse * tangent_y * inv_mass_b

                penetration = min_distance - distance
                correction = (
                    max(0.0, penetration) / max(1e-6, inv_mass_a + inv_mass_b)
                ) * separation_percent
                correction_x = correction * nx
                correction_y = correction * ny

                x_a -= correction_x * inv_mass_a
                y_a -= correction_y * inv_mass_a
                x_b += correction_x * inv_mass_b
                y_b += correction_y * inv_mass_b

                row_b["x"] = round(_clamp01(x_b), 5)
                row_b["y"] = round(_clamp01(y_b), 5)
                row_b["vx"] = round(vx_b, 6)
                row_b["vy"] = round(vy_b, 6)
                collision_count_updates[id_b] += 1

                collision_count_updates[id_a] += 1

            row_a["x"] = round(_clamp01(x_a), 5)
            row_a["y"] = round(_clamp01(y_a), 5)
            row_a["vx"] = round(vx_a, 6)
            row_a["vy"] = round(vy_a, 6)

    for row in particle_rows:
        particle_id = str(row.get("id", "") or id(row))
        collisions = int(collision_count_updates.get(particle_id, 0))
        row["collision_count"] = collisions


def advance_simulation_field_particles(
    simulation: dict[str, Any],
    *,
    dt_seconds: float,
    now_seconds: float | None = None,
) -> None:
    if not isinstance(simulation, dict):
        return
    presence_dynamics = simulation.get("presence_dynamics", {})
    if not isinstance(presence_dynamics, dict):
        return
    rows = presence_dynamics.get("field_particles", [])
    if not isinstance(rows, list) or not rows:
        return

    dt = max(0.001, _safe_float(dt_seconds, 0.08))
    base_dt = max(
        0.001, _safe_float(os.getenv("SIM_TICK_SECONDS", "0.08") or "0.08", 0.08)
    )
    now_value = _safe_float(now_seconds, time.time())
    friction_tick = max(
        0.8,
        min(0.99995, SIMULATION_STREAM_FRICTION ** (dt / base_dt)),
    )

    gravity_max = 1e-6
    for row in rows:
        if not isinstance(row, dict):
            continue
        gravity_max = max(
            gravity_max,
            _safe_float(row.get("gravity_potential", 0.0), 0.0),
        )

    presence_centers: dict[str, tuple[float, float]] = {}
    presence_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("presence_id", "") or "").strip()
        if not presence_id:
            continue
        x_value = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y_value = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        current_x, current_y = presence_centers.get(presence_id, (0.0, 0.0))
        presence_centers[presence_id] = (current_x + x_value, current_y + y_value)
        presence_counts[presence_id] = int(presence_counts.get(presence_id, 0)) + 1

    for presence_id, count in list(presence_counts.items()):
        if count <= 0 or presence_id not in presence_centers:
            continue
        total_x, total_y = presence_centers[presence_id]
        presence_centers[presence_id] = (total_x / count, total_y / count)

    node_centers: dict[str, tuple[float, float]] = {}
    node_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict):
            continue
        x_value = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y_value = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        tokens = {
            str(row.get("graph_node_id", "") or "").strip(),
            str(row.get("route_node_id", "") or "").strip(),
        }
        for token in tokens:
            if not token:
                continue
            total_x, total_y = node_centers.get(token, (0.0, 0.0))
            node_centers[token] = (total_x + x_value, total_y + y_value)
            node_counts[token] = int(node_counts.get(token, 0)) + 1

    for node_id, count in list(node_counts.items()):
        if count <= 0 or node_id not in node_centers:
            continue
        total_x, total_y = node_centers[node_id]
        node_centers[node_id] = (total_x / count, total_y / count)

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        particle_id = str(row.get("id", "") or f"field:{index}")
        x_value = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y_value = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        vx_value = (
            _safe_float(row.get("vx", 0.0), 0.0) * SIMULATION_STREAM_VELOCITY_SCALE
        )
        vy_value = (
            _safe_float(row.get("vy", 0.0), 0.0) * SIMULATION_STREAM_VELOCITY_SCALE
        )

        presence_id = str(row.get("presence_id", "") or "").strip()
        presence_center_x, presence_center_y = presence_centers.get(
            presence_id, (0.5, 0.5)
        )
        to_presence_x = presence_center_x - x_value
        to_presence_y = presence_center_y - y_value
        to_presence_mag = math.hypot(to_presence_x, to_presence_y)
        if to_presence_mag > 1e-6:
            to_presence_x /= to_presence_mag
            to_presence_y /= to_presence_mag

        graph_node_id = str(row.get("graph_node_id", "") or "").strip()
        route_node_id = str(row.get("route_node_id", "") or "").strip()
        route_anchor_x_raw = _safe_float(row.get("route_x", float("nan")), float("nan"))
        route_anchor_y_raw = _safe_float(row.get("route_y", float("nan")), float("nan"))
        graph_anchor_x_raw = _safe_float(row.get("graph_x", float("nan")), float("nan"))
        graph_anchor_y_raw = _safe_float(row.get("graph_y", float("nan")), float("nan"))
        route_anchor_valid = math.isfinite(route_anchor_x_raw) and math.isfinite(
            route_anchor_y_raw
        )
        graph_anchor_valid = math.isfinite(graph_anchor_x_raw) and math.isfinite(
            graph_anchor_y_raw
        )

        semantic_node_id = route_node_id or graph_node_id
        semantic_anchor: tuple[float, float] | None = None
        if route_anchor_valid:
            semantic_anchor = (
                _clamp01(route_anchor_x_raw),
                _clamp01(route_anchor_y_raw),
            )
        elif graph_anchor_valid:
            semantic_anchor = (
                _clamp01(graph_anchor_x_raw),
                _clamp01(graph_anchor_y_raw),
            )
        elif semantic_node_id:
            semantic_count = int(node_counts.get(semantic_node_id, 0))
            center_candidate = node_centers.get(semantic_node_id)
            if semantic_count > 1 and isinstance(center_candidate, tuple):
                semantic_anchor = center_candidate

        if semantic_anchor is None:
            semantic_anchor = presence_centers.get(presence_id, (0.5, 0.5))
        semantic_anchor_x, semantic_anchor_y = semantic_anchor
        semantic_dx = semantic_anchor_x - x_value
        semantic_dy = semantic_anchor_y - y_value
        semantic_dist = max(1e-6, math.hypot(semantic_dx, semantic_dy))
        semantic_nx = semantic_dx / semantic_dist
        semantic_ny = semantic_dy / semantic_dist

        center_dx = 0.5 - x_value
        center_dy = 0.5 - y_value
        center_dist = max(1e-6, math.hypot(center_dx, center_dy))
        center_nx = center_dx / center_dist
        center_ny = center_dy / center_dist
        edge_distance = max(abs(x_value - 0.5), abs(y_value - 0.5))
        edge_signal = _clamp01((edge_distance - 0.36) / 0.22)
        lateral_nx = -semantic_ny
        lateral_ny = semantic_nx

        drift_gravity_term = _safe_float(row.get("drift_gravity_term", 0.0), 0.0)
        valve_gravity_term = _safe_float(row.get("valve_gravity_term", 0.0), 0.0)
        gravity_potential = _safe_float(row.get("gravity_potential", 0.0), 0.0)
        influence_power = _clamp01(_safe_float(row.get("influence_power", 0.0), 0.0))
        route_probability = _clamp01(
            _safe_float(row.get("route_probability", 0.5), 0.5)
        )
        node_saturation = _clamp01(_safe_float(row.get("node_saturation", 0.0), 0.0))
        drift_cost_term = _safe_float(row.get("drift_cost_term", 0.0), 0.0)
        drift_cost_semantic_term = _safe_float(
            row.get("drift_cost_semantic_term", 0.0), 0.0
        )
        focus_contribution = _safe_float(
            row.get("route_resource_focus_contribution", 0.0), 0.0
        )
        semantic_text_chars = max(
            0.0,
            _safe_float(row.get("semantic_text_chars", 0.0), 0.0),
        )
        semantic_mass = max(0.0, _safe_float(row.get("semantic_mass", 0.0), 0.0))
        daimoi_energy = max(0.0, _safe_float(row.get("daimoi_energy", 0.0), 0.0))
        message_probability = max(
            0.0,
            _safe_float(row.get("message_probability", 0.0), 0.0),
        )
        package_entropy = max(0.0, _safe_float(row.get("package_entropy", 0.0), 0.0))

        gravity_signal = _clamp01(gravity_potential / max(1e-6, gravity_max))
        semantic_text_signal = _clamp01(math.log1p(semantic_text_chars) / 8.0)
        semantic_mass_signal = _clamp01(semantic_mass / 6.0)
        energy_signal = _clamp01((daimoi_energy * 0.35) + (message_probability * 0.45))
        entropy_signal = _clamp01(package_entropy / 3.0)
        semantic_signal = _clamp01(
            abs(drift_cost_semantic_term)
            + (semantic_text_signal * 0.6)
            + (semantic_mass_signal * 0.5)
            + (energy_signal * 0.45)
            + (entropy_signal * 0.25)
        )
        drift_cost_signal = _clamp01(abs(drift_cost_term))
        force_signal = _clamp01(
            (abs(drift_gravity_term) * 0.08)
            + (abs(valve_gravity_term) * 0.05)
            + (gravity_signal * 0.72)
            + (influence_power * 0.44)
            + (semantic_signal * 0.58)
        )

        semantic_gain = SIMULATION_STREAM_FIELD_FORCE * (
            0.18
            + (force_signal * 0.36)
            + (semantic_signal * 1.18)
            + (route_probability * 0.34)
            + min(0.24, abs(focus_contribution) * 0.08)
        )
        presence_gain = SIMULATION_STREAM_FIELD_FORCE * (
            0.03 + ((1.0 - route_probability) * 0.08) + ((1.0 - semantic_signal) * 0.06)
        )
        center_gain = SIMULATION_STREAM_CENTER_GRAVITY * (
            edge_signal
            * (
                0.22
                + (gravity_signal * 0.38)
                + (influence_power * 0.16)
                + ((1.0 - node_saturation) * 0.12)
            )
        )
        jitter_gain = SIMULATION_STREAM_JITTER_FORCE * (
            0.72
            + (semantic_signal * 0.84)
            + (influence_power * 0.36)
            + (edge_signal * 0.24)
            + min(0.62, abs(focus_contribution) * 0.22)
        )

        seed = int(hashlib.sha1(particle_id.encode("utf-8")).hexdigest()[:8], 16)
        noise_frequency = 2.8 + (semantic_signal * 4.2) + (route_probability * 1.1)
        noise_time_scale = 0.44 + (route_probability * 0.48) + (semantic_signal * 0.22)
        jitter_x_primary = _simplex_noise_2d(
            (x_value * noise_frequency) + (index * 0.019),
            now_value * noise_time_scale,
            seed=(seed % 251) + 7,
        )
        jitter_y_primary = _simplex_noise_2d(
            (y_value * noise_frequency) + 100.0 + (index * 0.017),
            now_value * (noise_time_scale * 1.07),
            seed=(seed % 251) + 19,
        )
        jitter_x_detail = _simplex_noise_2d(
            (x_value * (noise_frequency * 2.1)) + 37.0 + (index * 0.013),
            now_value * (noise_time_scale * 1.63),
            seed=(seed % 251) + 43,
        )
        jitter_y_detail = _simplex_noise_2d(
            (y_value * (noise_frequency * 2.05)) + 173.0 + (index * 0.011),
            now_value * (noise_time_scale * 1.71),
            seed=(seed % 251) + 71,
        )
        jitter_x = (
            (jitter_x_primary + (jitter_x_detail * 0.58))
            * jitter_gain
            * SIMULATION_STREAM_SIMPLEX_SCALE
        )
        jitter_y = (
            (jitter_y_primary + (jitter_y_detail * 0.58))
            * jitter_gain
            * SIMULATION_STREAM_SIMPLEX_SCALE
        )

        ax = (
            (semantic_nx * semantic_gain)
            + (to_presence_x * presence_gain)
            + (center_nx * center_gain)
            + jitter_x
        )
        ay = (
            (semantic_ny * semantic_gain)
            + (to_presence_y * presence_gain)
            + (center_ny * center_gain)
            + jitter_y
        )

        vx_value += ax * dt
        vy_value += ay * dt
        lateral_velocity = (vx_value * lateral_nx) + (vy_value * lateral_ny)
        lateral_damping = min(
            0.92,
            (0.24 + (semantic_signal * 0.34) + (route_probability * 0.2)) * dt,
        )
        vx_value -= lateral_velocity * lateral_nx * lateral_damping
        vy_value -= lateral_velocity * lateral_ny * lateral_damping

        base_drag = 1.0 - min(
            0.08, (node_saturation * 0.03) + (drift_cost_signal * 0.02)
        )
        vx_value *= friction_tick * base_drag
        vy_value *= friction_tick * base_drag

        dynamic_max_speed = SIMULATION_STREAM_MAX_SPEED * (
            0.68
            + (influence_power * 0.22)
            + (gravity_signal * 0.12)
            + (semantic_signal * 0.22)
            + (energy_signal * 0.08)
        )
        speed = math.hypot(vx_value, vy_value)
        if speed > dynamic_max_speed and speed > 0.0:
            speed_scale = dynamic_max_speed / speed
            vx_value *= speed_scale
            vy_value *= speed_scale

        next_x = x_value + (vx_value * dt)
        next_y = y_value + (vy_value * dt)

        if next_x < 0.0:
            next_x = 0.0
            vx_value = abs(vx_value) * 0.82
        elif next_x > 1.0:
            next_x = 1.0
            vx_value = -abs(vx_value) * 0.82

        if next_y < 0.0:
            next_y = 0.0
            vy_value = abs(vy_value) * 0.82
        elif next_y > 1.0:
            next_y = 1.0
            vy_value = -abs(vy_value) * 0.82

        row["x"] = round(_clamp01(next_x), 5)
        row["y"] = round(_clamp01(next_y), 5)
        row["vx"] = round(vx_value, 6)
        row["vy"] = round(vy_value, 6)

        # Absorption (Suck up)
        if bool(row.get("resource_daimoi", False)):
            target_id = str(row.get("resource_target_presence_id", "")).strip()
            if target_id:
                tx, ty = presence_centers.get(target_id, (0.5, 0.5))
                dist_sq = ((tx - next_x) ** 2) + ((ty - next_y) ** 2)
                if dist_sq < 0.0036:  # Radius approx 0.06
                    manager = get_presence_runtime_manager()
                    state = manager.get_state(target_id)
                    wallet = state.setdefault("resource_wallet", {})
                    if not isinstance(wallet, dict):
                        wallet = {}
                        state["resource_wallet"] = wallet

                    amount = _safe_float(row.get("resource_emit_amount", 0.0), 0.0)
                    res_type = str(row.get("resource_type", "cpu"))

                    # Check pressure for absorption probability
                    current_bal = _safe_float(wallet.get(res_type, 0.0), 0.0)
                    # Use fixed cap for now, similar to _apply_resource_daimoi_emissions default
                    # In a real implementation we'd import _RESOURCE_DAIMOI_WALLET_CAP
                    cap = 32.0
                    if target_id == "presence.core.cpu":
                        cap = 48.0

                    pressure = _clamp01(current_bal / cap)

                    # Probability of absorption inversely related to pressure
                    # Low pressure = High absorption chance (Suck up)
                    # High pressure = Low absorption chance (Deflect)
                    absorb_prob = 1.0 - (pressure * 0.85)  # Always at least 15% chance

                    seed_val = int(
                        hashlib.sha1(
                            f"{row.get('id')}|{now_value}".encode("utf-8")
                        ).hexdigest()[:8],
                        16,
                    )
                    rng_val = (seed_val % 1000) / 1000.0

                    if rng_val < absorb_prob:
                        wallet[res_type] = round(current_bal + amount, 6)
                        row["_absorbed"] = True
                    else:
                        # Deflect
                        row["_deflected"] = True
                        row["vx"] = -vx_value * 0.6
                        row["vy"] = -vy_value * 0.6
                        # Push away slightly
                        dx = next_x - tx
                        dy = next_y - ty
                        mag = math.hypot(dx, dy)
                        if mag > 1e-6:
                            next_x = _clamp01(next_x + (dx / mag * 0.03))
                            next_y = _clamp01(next_y + (dy / mag * 0.03))

    _resolve_semantic_particle_collisions(rows)

    # Remove absorbed
    presence_dynamics["field_particles"] = [r for r in rows if not r.get("_absorbed")]
    if len(presence_dynamics["field_particles"]) != len(rows):
        simulation["presence_dynamics"] = presence_dynamics


def build_simulation_delta(
    previous_simulation: dict[str, Any] | None,
    current_simulation: dict[str, Any] | None,
) -> dict[str, Any]:
    previous = previous_simulation if isinstance(previous_simulation, dict) else {}
    current = current_simulation if isinstance(current_simulation, dict) else {}
    changed_keys: list[str] = []
    patch: dict[str, Any] = {}

    for key in (
        "timestamp",
        "total",
        "audio",
        "image",
        "video",
        "points",
        "field_particles",
        "presence_dynamics",
        "daimoi",
        "pain_field",
        "truth_state",
        "projection",
        "perspective",
        "entities",
        "echoes",
        "myth",
        "world",
    ):
        if previous.get(key) != current.get(key):
            patch[key] = current.get(key)
            changed_keys.append(key)

    previous_fingerprint = simulation_fingerprint(previous)
    current_fingerprint = simulation_fingerprint(current)
    if not changed_keys and previous_fingerprint != current_fingerprint:
        patch = {
            "timestamp": current.get("timestamp"),
            "presence_dynamics": current.get("presence_dynamics", {}),
        }
        changed_keys = ["timestamp", "presence_dynamics"]

    return {
        "record": "eta-mu.simulation-delta.v1",
        "schema_version": "simulation.delta.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "previous_fingerprint": previous_fingerprint,
        "current_fingerprint": current_fingerprint,
        "changed_keys": changed_keys,
        "has_changes": bool(changed_keys),
        "patch": patch,
    }
