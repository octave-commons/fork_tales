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

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from hashlib import sha1
from urllib.parse import urlparse, unquote

from .metrics import _safe_float, _clamp01

_GRAPH_ARCHIVE_SUFFIXES: set[str] = {
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".zst",
}
_GRAPH_IMAGE_SUFFIXES: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
    ".avif",
    ".heic",
}
_GRAPH_AUDIO_SUFFIXES: set[str] = {
    ".mp3",
    ".wav",
    ".ogg",
    ".m4a",
    ".flac",
    ".aac",
    ".opus",
}
_GRAPH_VIDEO_SUFFIXES: set[str] = {
    ".mp4",
    ".m4v",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
}


def _graph_suffix_from_path_like(path_like: Any) -> str:
    raw = str(path_like or "").strip()
    if not raw:
        return ""
    parsed_path = urlparse(raw).path if "://" in raw else raw
    normalized = unquote(str(parsed_path or "")).strip()
    if not normalized:
        return ""
    return Path(normalized).suffix.lower()


def _graph_resource_kind_from_crawler_node(node: dict[str, Any]) -> str:
    crawler_kind = (
        str(node.get("crawler_kind", node.get("kind", "url"))).strip().lower()
    )
    content_type = str(node.get("content_type", "")).strip().lower()
    suffix = _graph_suffix_from_path_like(node.get("url", ""))

    if content_type.startswith("image/") or suffix in _GRAPH_IMAGE_SUFFIXES:
        return "image"
    if content_type.startswith("audio/") or suffix in _GRAPH_AUDIO_SUFFIXES:
        return "audio"
    if content_type == "application/pdf" or suffix == ".pdf":
        return "pdf"
    if content_type.startswith("video/") or suffix in _GRAPH_VIDEO_SUFFIXES:
        return "video"
    if "zip" in content_type or suffix in _GRAPH_ARCHIVE_SUFFIXES:
        return "archive"
    if content_type.startswith("text/"):
        return "text"
    if crawler_kind in {"domain", "content"}:
        return "website"
    if crawler_kind == "url":
        return "link"
    return "unknown"


def _graph_modality_from_resource_kind(resource_kind: str) -> str:
    normalized = str(resource_kind or "").strip().lower()
    if normalized == "pdf":
        return "text"
    if normalized in {"text", "image", "audio", "video"}:
        return normalized
    if normalized in {"website", "link"}:
        return "web"
    if normalized == "archive":
        return "archive"
    if normalized == "blob":
        return "binary"
    return "unknown"


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
        resource_kind = str(normalized.get("resource_kind", "")).strip().lower()
        if not resource_kind:
            resource_kind = _graph_resource_kind_from_crawler_node(normalized)
        normalized["resource_kind"] = resource_kind
        if not str(normalized.get("modality", "")).strip():
            normalized["modality"] = _graph_modality_from_resource_kind(resource_kind)
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
    "image": "image",
    "audio": "audio",
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
    "web:url": "web:url",
    "web:resource": "web:resource",
    "obs:event": "obs:event",
    "daimoi": "daimoi",
    "nexus": "nexus",
    "fact": "fact",
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

    web_node_role = str(legacy_node.get("web_node_role", "") or "").strip().lower()
    if web_node_role in {"web:url", "web:resource", "obs:event"}:
        role = web_node_role

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
            "resource_kind",
            "modality",
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
            "resource_kind",
            "modality",
            "compliance",
            "dominant_field",
        ):
            if legacy_node.get(key):
                extension[key] = legacy_node[key]
    elif role == "web:url":
        for key in (
            "url",
            "canonical_url",
            "domain",
            "title",
            "status",
            "content_type",
            "compliance",
            "next_allowed_fetch_ts",
            "fail_count",
            "last_fetch_ts",
            "last_status",
            "source_hint",
        ):
            if legacy_node.get(key) is not None:
                extension[key] = legacy_node[key]
    elif role == "web:resource":
        for key in (
            "canonical_url",
            "fetched_ts",
            "content_hash",
            "text_excerpt_hash",
            "text_excerpt",
            "summary",
            "title",
            "source_url_id",
            "resource_kind",
            "modality",
            "kind",
            "repo",
            "number",
            "labels",
            "authors",
            "updated_at",
            "importance_score",
            "atoms",
            "filenames_touched",
            "diff_keyword_hits",
            "conversation_markdown",
            "conversation_comment_count",
            "conversation_rows",
            "commit_count",
            "commit_rows",
            "api_endpoint",
            "state",
            "merged_at",
        ):
            if legacy_node.get(key) is not None:
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
    web_kind_map = {
        "hyperlink": "web:links_to",
        "citation": "web:links_to",
        "cross_reference": "web:links_to",
        "wiki_reference": "web:links_to",
        "paper_pdf": "web:links_to",
        "canonical_redirect": "web:canonical_of",
    }
    kind = web_kind_map.get(kind, kind)
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
