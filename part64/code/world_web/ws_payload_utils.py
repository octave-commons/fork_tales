# SPDX-License-Identifier: GPL-3.0-or-later
# This file is part of Fork Tales.
# Copyright (C) 2024-2025 Fork Tales Contributors

from __future__ import annotations

import json
import math
from typing import Any, Callable


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def ws_clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def simulation_ws_trim_simulation_payload(simulation: dict[str, Any]) -> dict[str, Any]:
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


def simulation_ws_compact_graph_payload(
    simulation: dict[str, Any],
    *,
    trim_catalog: Callable[[dict[str, Any]], dict[str, Any]],
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

    compact_catalog = trim_catalog(catalog_like)
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


def simulation_ws_capture_particle_motion_state(
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


def simulation_ws_restore_particle_motion_state(
    simulation_payload: dict[str, Any],
    motion_state: dict[str, dict[str, Any]],
    *,
    blend: float,
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

    blend_value = max(0.0, min(1.0, _safe_float(blend, 0.96)))
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

        row["x"] = round(
            (previous_x * blend_value) + (current_x * (1.0 - blend_value)), 5
        )
        row["y"] = round(
            (previous_y * blend_value) + (current_y * (1.0 - blend_value)), 5
        )
        row["vx"] = round(
            (previous_vx * blend_value) + (current_vx * (1.0 - blend_value)), 6
        )
        row["vy"] = round(
            (previous_vy * blend_value) + (current_vy * (1.0 - blend_value)), 6
        )
        restored += 1

    if restored > 0:
        dynamics["field_particles"] = rows
        simulation_payload["presence_dynamics"] = dynamics
    return restored


def simulation_ws_decode_cached_payload(cached_body: Any) -> dict[str, Any] | None:
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


def simulation_ws_payload_is_sparse(payload: dict[str, Any]) -> bool:
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


def simulation_ws_payload_has_disabled_particle_dynamics(
    payload: dict[str, Any],
) -> bool:
    if not isinstance(payload, dict):
        return False
    dynamics = payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return False
    summary = dynamics.get("daimoi_probabilistic", {})
    if not isinstance(summary, dict):
        return False
    disabled = bool(summary.get("disabled", False))
    reason = str(summary.get("disabled_reason", "") or "").strip().lower()
    if not (disabled and reason == "include_particle_dynamics=false"):
        return False
    field_particles = dynamics.get("field_particles", [])
    if isinstance(field_particles, list) and field_particles:
        return False
    return True


def simulation_ws_payload_is_bootstrap_only(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False

    record = str(payload.get("record", "") or "").strip().lower()
    if record == "eta-mu.ws.simulation-fast-bootstrap.v1":
        return True

    dynamics = payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return False

    summary = dynamics.get("daimoi_probabilistic", {})
    if not isinstance(summary, dict):
        return False

    disabled_reason = str(summary.get("disabled_reason", "") or "").strip().lower()
    return disabled_reason in {"ws_bootstrap_cache_miss", "ws_fast_bootstrap"}


def simulation_ws_payload_missing_graph_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return True

    def _graph_has_nodes(graph_payload: Any, keys: tuple[str, ...]) -> bool:
        if not isinstance(graph_payload, dict):
            return False
        for key in keys:
            rows = graph_payload.get(key, [])
            if isinstance(rows, list) and rows:
                return True
        return False

    file_graph_has_nodes = _graph_has_nodes(
        payload.get("file_graph"),
        ("file_nodes", "nodes", "field_nodes", "tag_nodes"),
    )
    crawler_graph_has_nodes = _graph_has_nodes(
        payload.get("crawler_graph"),
        ("crawler_nodes", "nodes", "field_nodes"),
    )
    if file_graph_has_nodes or crawler_graph_has_nodes:
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


def simulation_ws_sample_particle_page(
    rows: list[Any],
    *,
    max_rows: int,
    page_cursor: int,
    jitter_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    total_rows = len(rows) if isinstance(rows, list) else 0
    if total_rows <= 0:
        return (
            [],
            {
                "total": 0,
                "sample_size": 0,
                "page_index": 0,
                "page_total": 0,
                "start_index": 0,
            },
            0,
        )

    sample_size = max(1, min(total_rows, int(max_rows)))
    page_total = max(1, int(math.ceil(total_rows / float(sample_size))))
    page_index = max(0, int(page_cursor)) % page_total
    jitter = 0
    if total_rows > sample_size and sample_size > 1:
        jitter = abs(int(jitter_seed)) % sample_size
    start_index = ((page_index * sample_size) + jitter) % total_rows

    sampled_rows: list[dict[str, Any]] = []
    for offset in range(sample_size):
        row = rows[(start_index + offset) % total_rows]
        if isinstance(row, dict):
            sampled_rows.append(dict(row))

    next_cursor = (page_index + 1) % page_total
    meta = {
        "total": total_rows,
        "sample_size": len(sampled_rows),
        "page_index": page_index,
        "page_total": page_total,
        "start_index": start_index,
    }
    return sampled_rows, meta, next_cursor
