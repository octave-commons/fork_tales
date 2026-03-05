"""Graph normalization/diff helpers for simulation bootstrap reports."""

from __future__ import annotations

from typing import Any

from .metrics import _safe_float


def bootstrap_embed_layer_row(layer: dict[str, Any]) -> dict[str, Any]:
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


def bootstrap_normalize_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def bootstrap_file_path(row: dict[str, Any]) -> str:
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
        text = bootstrap_normalize_path(row.get(key, ""))
        if text:
            return text
    return ""


def bootstrap_file_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id", "") or "").strip(),
        "node_id": str(row.get("node_id", "") or "").strip(),
        "name": str(row.get("name", "") or "").strip(),
        "kind": str(row.get("kind", "") or "").strip(),
        "path": bootstrap_file_path(row),
        "source_rel_path": bootstrap_normalize_path(row.get("source_rel_path", "")),
        "archive_rel_path": bootstrap_normalize_path(row.get("archive_rel_path", "")),
        "archived_rel_path": bootstrap_normalize_path(row.get("archived_rel_path", "")),
        "projection_overflow": bool(row.get("projection_overflow", False)),
        "consolidated": bool(row.get("consolidated", False)),
        "consolidated_count": max(
            0, int(_safe_float(row.get("consolidated_count", 0), 0.0))
        ),
        "projection_group_id": str(row.get("projection_group_id", "") or "").strip(),
    }


def bootstrap_graph_diff(
    *,
    catalog: dict[str, Any],
    simulation: dict[str, Any],
    max_excluded_files: int,
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
        bootstrap_file_row(row)
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
        row_payload = bootstrap_file_row(source_row)
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
        rel_path = bootstrap_normalize_path(item.get("rel_path", ""))
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
        bootstrap_file_path(row) for row in true_file_nodes if bootstrap_file_path(row)
    }
    missing_item_rows = [
        {
            **item_rows[path],
            "reason": "ingested_item_not_present_in_true_graph",
        }
        for path in sorted(item_rows)
        if path not in true_paths
    ]

    max_rows = max_excluded_files
    collapsed_edges = max(
        0,
        int(_safe_float(projection_payload.get("collapsed_edges", 0), 0.0)),
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
