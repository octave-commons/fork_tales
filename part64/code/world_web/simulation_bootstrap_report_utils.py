"""Bootstrap report assembly helpers for simulation graph bring-up."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from .metrics import _safe_float
from . import (
    simulation_bootstrap_graph_utils as simulation_bootstrap_graph_utils_module,
)


def bootstrap_graph_report(
    *,
    perspective: str,
    catalog: dict[str, Any],
    simulation: dict[str, Any],
    projection: dict[str, Any],
    phase_ms: dict[str, float] | None = None,
    reset_summary: dict[str, Any] | None = None,
    inbox_sync: dict[str, Any] | None = None,
    cache_key: str = "",
    max_excluded_files: int,
    normalize_projection_perspective: Callable[[str], str],
    runtime_config_version_snapshot: Callable[[], int],
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
        "runtime_config_version": runtime_config_version_snapshot(),
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
                simulation_bootstrap_graph_utils_module.bootstrap_embed_layer_row(row)
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
        "graph_diff": simulation_bootstrap_graph_utils_module.bootstrap_graph_diff(
            catalog=catalog,
            simulation=simulation,
            max_excluded_files=max_excluded_files,
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
