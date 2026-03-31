from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Callable


SIMULATION_FILE_GRAPH_PROJECTION_RECORD = "ημ.file-graph-projection.v1"
SIMULATION_FILE_GRAPH_PROJECTION_SCHEMA_VERSION = "file-graph.projection.v1"
SIMULATION_TRUTH_GRAPH_RECORD = "eta-mu.truth-graph.v1"
SIMULATION_TRUTH_GRAPH_SCHEMA_VERSION = "truth.graph.v1"
SIMULATION_VIEW_GRAPH_RECORD = "eta-mu.view-graph.v1"
SIMULATION_VIEW_GRAPH_SCHEMA_VERSION = "view.graph.v1"

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
    "resource_kind",
    "modality",
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
    "projection_overflow",
    "consolidated",
    "consolidated_count",
    "projection_group_id",
    "graph_scope",
    "truth_scope",
    "simulation_semantic_role",
    "semantic_bundle",
    "semantic_bundle_mass",
    "semantic_bundle_charge",
    "semantic_bundle_gravity",
    "semantic_bundle_member_edge_count",
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
    "resource_kind",
    "modality",
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
    "resource_wallet",
    "projection_overflow",
    "consolidated",
    "consolidated_count",
    "projection_group_id",
    "graph_scope",
    "truth_scope",
    "simulation_semantic_role",
    "semantic_bundle",
    "semantic_bundle_mass",
    "semantic_bundle_charge",
    "semantic_bundle_gravity",
    "semantic_bundle_member_edge_count",
)


def normalize_path_for_file_id(path_like: str) -> str:
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


def file_id_for_path(path_like: str) -> str:
    norm = normalize_path_for_file_id(path_like)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest() if norm else ""


def file_node_usage_path(node: dict[str, Any]) -> str:
    return normalize_path_for_file_id(
        str(
            node.get("source_rel_path")
            or node.get("archived_rel_path")
            or node.get("archive_rel_path")
            or node.get("name")
            or node.get("label")
            or ""
        )
    )


def file_node_usage_score(
    node: dict[str, Any],
    *,
    recent_paths: set[str],
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    clamp01: Callable[[Any], float],
) -> tuple[float, bool, str]:
    usage_path = file_node_usage_path(node)
    recent_hit = bool(usage_path and usage_path in recent_paths)
    importance = clamp01(safe_float(node.get("importance", 0.25), 0.25))
    layer_ratio = clamp01(safe_int(node.get("embed_layer_count", 0), 0) / 4.0)
    collection_bonus = 0.08 if str(node.get("vecstore_collection", "")).strip() else 0.0
    recent_bonus = 0.34 if recent_hit else 0.0
    score = clamp01(
        (importance * 0.56) + (layer_ratio * 0.2) + collection_bonus + recent_bonus
    )
    return score, recent_hit, usage_path


def graph_rows(
    file_graph: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = file_graph if isinstance(file_graph, dict) else {}
    node_rows: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()

    def _append_rows(raw_rows: Any) -> None:
        if not isinstance(raw_rows, list):
            return
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            node_id = str(row.get("id", "")).strip()
            if node_id and node_id in seen_node_ids:
                continue
            node_rows.append(row)
            if node_id:
                seen_node_ids.add(node_id)

    _append_rows(graph.get("nodes", []))
    _append_rows(graph.get("field_nodes", []))
    _append_rows(graph.get("tag_nodes", []))
    _append_rows(graph.get("file_nodes", []))
    _append_rows(graph.get("crawler_nodes", []))
    edge_rows = [
        row
        for row in (
            graph.get("edges", []) if isinstance(graph.get("edges", []), list) else []
        )
        if isinstance(row, dict)
    ]
    return node_rows, edge_rows


def graph_node_type_counts(node_rows: list[dict[str, Any]]) -> dict[str, int]:
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


def build_truth_graph_contract(file_graph: dict[str, Any] | None) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    node_rows, edge_rows = graph_rows(file_graph)
    node_type_counts = graph_node_type_counts(node_rows)
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
    projection_bundle_node_count = sum(
        1
        for row in node_rows
        if isinstance(row, dict)
        and (
            bool(row.get("projection_overflow", False))
            or bool(row.get("semantic_bundle", False))
            or str(row.get("kind", "")).strip().lower() == "projection_overflow"
        )
    )
    projection_bundle_edge_count = sum(
        1
        for row in edge_rows
        if isinstance(row, dict) and bool(row.get("projection_overflow", False))
    )

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
        "semantics": {
            "graph_domain": "truth_graph",
            "graph_scope": "truth",
            "includes_projection_bundles": False,
            "projection_bundle_node_count": int(projection_bundle_node_count),
            "projection_bundle_edge_count": int(projection_bundle_edge_count),
        },
    }


def build_view_graph_contract(
    file_graph: dict[str, Any] | None,
    *,
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    clamp01: Callable[[Any], float],
    projection_edge_threshold: int,
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    node_rows, edge_rows = graph_rows(file_graph)
    node_type_counts = graph_node_type_counts(node_rows)
    projection = (
        file_graph.get("projection", {})
        if isinstance(file_graph, dict)
        and isinstance(file_graph.get("projection", {}), dict)
        else {}
    )
    projection_policy = (
        projection.get("policy", {})
        if isinstance(projection, dict)
        and isinstance(projection.get("policy", {}), dict)
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
        member_edge_count = max(0, safe_int(group.get("member_edge_count", 0), 0))
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
                    0, safe_int(group.get("member_source_count", 0), 0)
                ),
                "member_target_count": max(
                    0, safe_int(group.get("member_target_count", 0), 0)
                ),
                "member_edge_digest": str(group.get("member_edge_digest", "")),
                "surface_visible": bool(group.get("surface_visible", False)),
            }
        )

    projection_bundle_node_count = sum(
        1
        for row in node_rows
        if isinstance(row, dict)
        and (
            bool(row.get("projection_overflow", False))
            or bool(row.get("semantic_bundle", False))
            or str(row.get("kind", "")).strip().lower() == "projection_overflow"
        )
    )
    projection_bundle_edge_count = sum(
        1
        for row in edge_rows
        if isinstance(row, dict) and bool(row.get("projection_overflow", False))
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
            "compaction_drive": round(
                clamp01(
                    safe_float(projection_policy.get("compaction_drive", 0.0), 0.0)
                ),
                6,
            ),
            "cpu_pressure": round(
                clamp01(safe_float(projection_policy.get("cpu_pressure", 0.0), 0.0)), 6
            ),
            "view_edge_pressure": round(
                clamp01(
                    safe_float(projection_policy.get("view_edge_pressure", 0.0), 0.0)
                ),
                6,
            ),
            "cpu_utilization": round(
                max(
                    0.0,
                    min(
                        100.0,
                        safe_float(projection_policy.get("cpu_utilization", 0.0), 0.0),
                    ),
                ),
                3,
            ),
            "cpu_sentinel_id": str(projection_policy.get("presence_id", "") or ""),
            "edge_threshold_base": int(
                safe_int(
                    projection_policy.get(
                        "edge_threshold_base", projection_edge_threshold
                    ),
                    projection_edge_threshold,
                )
            ),
            "edge_threshold_effective": int(
                safe_int(projection_policy.get("edge_threshold_effective", 0), 0)
            ),
            "edge_cap_base": int(
                safe_int(projection_policy.get("edge_cap_base", 0), 0)
            ),
            "edge_cap_effective": int(
                safe_int(projection_policy.get("edge_cap_effective", 0), 0)
            ),
            "bundle_ledger_count": int(len(bundle_ledgers)),
            "bundle_member_edge_count_total": int(bundle_member_edges_total),
            "reconstructable_bundle_count": int(reconstructable_bundle_count),
            "surface_visible_bundle_count": int(surface_visible_count),
            "bundle_ledgers": bundle_ledgers,
            "ledger_ref": "file_graph.projection.groups",
            "policy": projection_policy,
        },
        "projection_pi": {
            "kind": "edge-bundle" if bundle_ledgers else "identity",
            "bundle_count": int(len(bundle_ledgers)),
            "bundle_member_edge_count_total": int(bundle_member_edges_total),
            "reconstructable_bundle_count": int(reconstructable_bundle_count),
        },
        "semantics": {
            "graph_domain": "view_graph",
            "graph_scope": "view",
            "includes_projection_bundles": bool(bundle_ledgers),
            "projection_bundle_node_count": int(projection_bundle_node_count),
            "projection_bundle_edge_count": int(projection_bundle_edge_count),
            "bundle_semantic_role": "view_compaction_aggregate",
        },
    }
