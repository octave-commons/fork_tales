"""Utilities for compacting and preparing file graph simulation payloads."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from .metrics import _clamp01, _safe_float, _safe_int


def json_deep_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def bounded_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]


def compact_embed_layer_points(
    value: Any,
    *,
    embed_layer_point_cap: int,
    embed_ids_cap: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact_rows: list[dict[str, Any]] = []
    for row in value[:embed_layer_point_cap]:
        if not isinstance(row, dict):
            continue
        embed_ids_raw = row.get("embed_ids", [])
        embed_ids = (
            [
                str(embed_id).strip()
                for embed_id in embed_ids_raw[:embed_ids_cap]
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


def compact_file_graph_node(
    node: dict[str, Any],
    *,
    node_fields: Sequence[str],
    summary_chars: int,
    excerpt_chars: int,
    embed_link_cap: int,
    embed_layer_point_cap: int,
    embed_ids_cap: int,
) -> dict[str, Any]:
    compact: dict[str, Any] = {key: node[key] for key in node_fields if key in node}
    compact["x"] = round(_clamp01(_safe_float(compact.get("x", 0.5), 0.5)), 6)
    compact["y"] = round(_clamp01(_safe_float(compact.get("y", 0.5), 0.5)), 6)
    compact["hue"] = int(round(_safe_float(compact.get("hue", 200.0), 200.0))) % 360
    compact["importance"] = round(
        _clamp01(_safe_float(compact.get("importance", 0.24), 0.24)),
        6,
    )

    compact["summary"] = bounded_text(compact.get("summary", ""), limit=summary_chars)
    compact["text_excerpt"] = bounded_text(
        compact.get("text_excerpt", ""),
        limit=excerpt_chars,
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
            for link in embedding_links_raw[:embed_link_cap]
            if str(link).strip()
        ]
        if isinstance(embedding_links_raw, list)
        else []
    )

    compact["embed_layer_points"] = compact_embed_layer_points(
        compact.get("embed_layer_points", []),
        embed_layer_point_cap=embed_layer_point_cap,
        embed_ids_cap=embed_ids_cap,
    )
    compact["embed_layer_count"] = int(
        _safe_int(compact.get("embed_layer_count", 0), 0)
    )
    return compact


def compact_file_graph_nodes(
    value: Any,
    *,
    node_fields: Sequence[str],
    summary_chars: int,
    excerpt_chars: int,
    embed_link_cap: int,
    embed_layer_point_cap: int,
    embed_ids_cap: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        compact_file_graph_node(
            node,
            node_fields=node_fields,
            summary_chars=summary_chars,
            excerpt_chars=excerpt_chars,
            embed_link_cap=embed_link_cap,
            embed_layer_point_cap=embed_layer_point_cap,
            embed_ids_cap=embed_ids_cap,
        )
        for node in value
        if isinstance(node, dict)
    ]


def compact_file_graph_render_node(
    node: dict[str, Any], *, render_node_fields: Sequence[str]
) -> dict[str, Any]:
    compact = {key: node[key] for key in render_node_fields if key in node}
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


def compact_file_graph_for_simulation(
    file_graph: dict[str, Any],
    *,
    node_fields: Sequence[str],
    render_node_fields: Sequence[str],
    summary_chars: int,
    excerpt_chars: int,
    embed_link_cap: int,
    embed_layer_point_cap: int,
    embed_ids_cap: int,
    edge_response_cap: int,
    edge_response_factor: float,
    file_graph_record: str,
) -> dict[str, Any]:
    compact_file_nodes = compact_file_graph_nodes(
        file_graph.get("file_nodes", []),
        node_fields=node_fields,
        summary_chars=summary_chars,
        excerpt_chars=excerpt_chars,
        embed_link_cap=embed_link_cap,
        embed_layer_point_cap=embed_layer_point_cap,
        embed_ids_cap=embed_ids_cap,
    )
    compact_field_nodes = compact_file_graph_nodes(
        file_graph.get("field_nodes", []),
        node_fields=node_fields,
        summary_chars=summary_chars,
        excerpt_chars=excerpt_chars,
        embed_link_cap=embed_link_cap,
        embed_layer_point_cap=embed_layer_point_cap,
        embed_ids_cap=embed_ids_cap,
    )
    compact_tag_nodes = compact_file_graph_nodes(
        file_graph.get("tag_nodes", []),
        node_fields=node_fields,
        summary_chars=summary_chars,
        excerpt_chars=excerpt_chars,
        embed_link_cap=embed_link_cap,
        embed_layer_point_cap=embed_layer_point_cap,
        embed_ids_cap=embed_ids_cap,
    )
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
        compact_node = compact_file_graph_render_node(
            node,
            render_node_fields=render_node_fields,
        )
        compact_node_id = str(compact_node.get("id", "")).strip()
        if compact_node_id and compact_node_id in non_file_seen_ids:
            continue
        if compact_node_id:
            non_file_seen_ids.add(compact_node_id)
        compact_non_file_nodes.append(compact_node)

    if not compact_non_file_nodes:
        for node in [*compact_field_nodes, *compact_tag_nodes]:
            compact_node = compact_file_graph_render_node(
                node,
                render_node_fields=render_node_fields,
            )
            compact_node_id = str(compact_node.get("id", "")).strip()
            if compact_node_id and compact_node_id in non_file_seen_ids:
                continue
            if compact_node_id:
                non_file_seen_ids.add(compact_node_id)
            compact_non_file_nodes.append(compact_node)

    compact_file_nodes_for_render = [
        compact_file_graph_render_node(node, render_node_fields=render_node_fields)
        for node in compact_file_nodes
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
            edge_response_cap,
            max(
                384,
                int(round(max(1, len(compact_file_nodes)) * edge_response_factor)),
            ),
        ),
    )
    edge_count_before_projection = len(compact_edges)

    compact_stats_raw = file_graph.get("stats", {})
    compact_stats = (
        dict(compact_stats_raw) if isinstance(compact_stats_raw, dict) else {}
    )
    compact_stats["file_count"] = int(len(compact_file_nodes))
    compact_stats["edge_count"] = int(len(compact_edges))
    compact_stats["edge_count_before_projection"] = int(edge_count_before_projection)
    compact_stats["edge_response_cap"] = int(dynamic_edge_cap)

    return {
        "record": str(file_graph.get("record", file_graph_record)),
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


def file_graph_layout_cache_key(
    file_graph: dict[str, Any],
    *,
    file_node_usage_path: Callable[[dict[str, Any]], str],
    summary_chars: int,
    excerpt_chars: int,
) -> str:
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
            usage_path = file_node_usage_path(node)
            dominant_field = str(node.get("dominant_field", "")).strip()
            kind = str(node.get("kind", "")).strip().lower()
            summary_text = bounded_text(node.get("summary", ""), limit=summary_chars)
            excerpt_text = bounded_text(
                node.get("text_excerpt", ""), limit=excerpt_chars
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


def clone_prepared_file_graph(prepared_graph: dict[str, Any]) -> dict[str, Any]:
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
