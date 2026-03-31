"""Helpers for compacting simulation HTTP catalog payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import _safe_float


@dataclass(frozen=True)
class SimulationHttpTrimConfig:
    trim_enabled: bool
    max_items: int
    max_file_nodes: int
    max_file_edges: int
    max_field_nodes: int
    max_tag_nodes: int
    max_render_nodes: int
    max_crawler_nodes: int
    max_crawler_edges: int
    max_crawler_field_nodes: int
    max_text_excerpt_chars: int
    max_summary_chars: int
    max_embed_layer_points: int
    max_embed_ids: int
    max_embedding_links: int


def simulation_http_slice_rows(value: Any, *, max_rows: int) -> list[Any]:
    rows = value if isinstance(value, list) else []
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    return list(rows[:max_rows])


def simulation_http_compact_embed_layer_point(
    value: Any,
    *,
    config: SimulationHttpTrimConfig,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    compact = dict(value)
    embed_ids = compact.get("embed_ids")
    if isinstance(embed_ids, list):
        compact["embed_ids"] = [
            str(entry or "")
            for entry in embed_ids[: config.max_embed_ids]
            if str(entry or "").strip()
        ]
    return compact


def simulation_http_compact_embedding_link(
    value: Any,
) -> dict[str, Any] | None:
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


def simulation_http_compact_file_node(
    value: Any,
    *,
    config: SimulationHttpTrimConfig,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    compact = dict(value)

    excerpt = compact.get("text_excerpt")
    if isinstance(excerpt, str) and len(excerpt) > config.max_text_excerpt_chars:
        compact["text_excerpt"] = excerpt[: config.max_text_excerpt_chars]

    summary = compact.get("summary")
    if isinstance(summary, str) and len(summary) > config.max_summary_chars:
        compact["summary"] = summary[: config.max_summary_chars]

    layer_points = compact.get("embed_layer_points")
    if isinstance(layer_points, list):
        compact_layers: list[dict[str, Any]] = []
        for row in layer_points[: config.max_embed_layer_points]:
            compact_row = simulation_http_compact_embed_layer_point(row, config=config)
            if compact_row is not None:
                compact_layers.append(compact_row)
        compact["embed_layer_points"] = compact_layers
        compact["embed_layer_count"] = len(compact_layers)

    embedding_links = compact.get("embedding_links")
    if isinstance(embedding_links, list):
        compact_links: list[dict[str, Any]] = []
        for row in embedding_links[: config.max_embedding_links]:
            compact_row = simulation_http_compact_embedding_link(row)
            if compact_row is not None:
                compact_links.append(compact_row)
        compact["embedding_links"] = compact_links

    return compact


def simulation_http_compact_file_graph_node(
    value: Any,
    *,
    config: SimulationHttpTrimConfig,
) -> dict[str, Any] | None:
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
        return simulation_http_compact_file_node(value, config=config)

    return dict(value)


def simulation_http_trim_catalog(
    catalog: dict[str, Any],
    *,
    config: SimulationHttpTrimConfig,
) -> dict[str, Any]:
    if not config.trim_enabled:
        return catalog
    if not isinstance(catalog, dict):
        return {}

    trimmed = dict(catalog)

    items = catalog.get("items", [])
    if isinstance(items, list) and len(items) > config.max_items:
        trimmed["items"] = list(items[: config.max_items])

    file_graph = catalog.get("file_graph") if isinstance(catalog, dict) else None
    if isinstance(file_graph, dict):
        compact_file_graph = dict(file_graph)
        compact_file_nodes = simulation_http_slice_rows(
            file_graph.get("file_nodes", []),
            max_rows=config.max_file_nodes,
        )
        compact_file_graph["file_nodes"] = [
            compact_row
            for compact_row in (
                simulation_http_compact_file_node(row, config=config)
                for row in compact_file_nodes
            )
            if compact_row is not None
        ]
        compact_file_graph["field_nodes"] = simulation_http_slice_rows(
            file_graph.get("field_nodes", []),
            max_rows=config.max_field_nodes,
        )
        compact_file_graph["tag_nodes"] = simulation_http_slice_rows(
            file_graph.get("tag_nodes", []),
            max_rows=config.max_tag_nodes,
        )
        compact_nodes = simulation_http_slice_rows(
            file_graph.get("nodes", []),
            max_rows=config.max_render_nodes,
        )
        compact_file_graph["nodes"] = [
            compact_row
            for compact_row in (
                simulation_http_compact_file_graph_node(row, config=config)
                for row in compact_nodes
            )
            if compact_row is not None
        ]
        compact_file_graph["edges"] = simulation_http_slice_rows(
            file_graph.get("edges", []),
            max_rows=config.max_file_edges,
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
        compact_crawler_graph["crawler_nodes"] = simulation_http_slice_rows(
            crawler_graph.get("crawler_nodes", []),
            max_rows=config.max_crawler_nodes,
        )
        compact_crawler_graph["field_nodes"] = simulation_http_slice_rows(
            crawler_graph.get("field_nodes", []),
            max_rows=config.max_crawler_field_nodes,
        )
        compact_crawler_graph["nodes"] = simulation_http_slice_rows(
            crawler_graph.get("nodes", []),
            max_rows=max(config.max_crawler_nodes, config.max_crawler_field_nodes),
        )
        compact_crawler_graph["edges"] = simulation_http_slice_rows(
            crawler_graph.get("edges", []),
            max_rows=config.max_crawler_edges,
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

    nexus_graph = catalog.get("nexus_graph") if isinstance(catalog, dict) else None
    if isinstance(nexus_graph, dict):
        compact_nexus_graph = dict(nexus_graph)
        compact_nexus_graph["nodes"] = simulation_http_slice_rows(
            nexus_graph.get("nodes", []),
            max_rows=config.max_render_nodes,
        )
        compact_nexus_graph["edges"] = simulation_http_slice_rows(
            nexus_graph.get("edges", []),
            max_rows=config.max_file_edges,
        )
        compact_nexus_stats = (
            dict(nexus_graph.get("stats", {}))
            if isinstance(nexus_graph.get("stats", {}), dict)
            else {}
        )
        compact_nexus_stats["node_count"] = len(compact_nexus_graph.get("nodes", []))
        compact_nexus_stats["edge_count"] = len(compact_nexus_graph.get("edges", []))
        compact_nexus_graph["stats"] = compact_nexus_stats
        trimmed["nexus_graph"] = compact_nexus_graph

    return trimmed
