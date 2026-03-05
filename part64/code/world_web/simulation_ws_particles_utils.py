"""Websocket particle compaction helpers for simulation payloads."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable


def simulation_ws_collect_node_positions(
    simulation_payload: dict[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    if not isinstance(simulation_payload, dict):
        return {}, {}

    node_positions: dict[str, tuple[float, float]] = {}
    node_text_chars: dict[str, float] = {}

    def _node_text_weight(row: dict[str, Any]) -> float:
        text_keys = (
            "summary",
            "text_excerpt",
            "excerpt",
            "title",
            "label",
            "name",
            "url",
            "dominant_field",
            "dominant_presence",
        )
        total = 0
        for key in text_keys:
            value = row.get(key)
            if isinstance(value, str):
                total += len(value)
            elif isinstance(value, list):
                total += sum(len(str(item)) for item in value)
        return float(total)

    for graph_key in ("file_graph", "crawler_graph", "nexus_graph"):
        graph_payload = simulation_payload.get(graph_key, {})
        if not isinstance(graph_payload, dict):
            continue
        for section_key in (
            "nodes",
            "file_nodes",
            "crawler_nodes",
            "field_nodes",
            "tag_nodes",
        ):
            rows = graph_payload.get(section_key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                node_id = str(row.get("id", "") or "").strip()
                if not node_id or node_id in node_positions:
                    continue
                x_value = max(0.0, min(1.0, safe_float(row.get("x", 0.5), 0.5)))
                y_value = max(0.0, min(1.0, safe_float(row.get("y", 0.5), 0.5)))
                node_positions[node_id] = (x_value, y_value)
                node_text_chars[node_id] = _node_text_weight(row)

    return node_positions, node_text_chars


def simulation_ws_compact_field_particles_with_nodes(
    rows: Any,
    *,
    node_positions: dict[str, tuple[float, float]],
    node_text_chars: dict[str, float],
    stream_particle_max: int,
    safe_float: Callable[[Any, float], float],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []

    compact_rows: list[dict[str, Any]] = []
    limit = max(1, int(stream_particle_max))
    for index, row in enumerate(rows):
        if index >= limit:
            break
        if not isinstance(row, dict):
            continue
        particle_id = str(row.get("id", "") or "").strip() or f"ws:{index}"
        x_value = max(0.0, min(1.0, safe_float(row.get("x", 0.5), 0.5)))
        y_value = max(0.0, min(1.0, safe_float(row.get("y", 0.5), 0.5)))
        vx_value = safe_float(row.get("vx", 0.0), 0.0)
        vy_value = safe_float(row.get("vy", 0.0), 0.0)

        if abs(vx_value) + abs(vy_value) < 1e-6:
            seed = int(hashlib.sha1(particle_id.encode("utf-8")).hexdigest()[:8], 16)
            angle = (float(seed % 6283) / 1000.0) * math.pi
            vx_value = math.cos(angle) * 0.01
            vy_value = math.sin(angle) * 0.01

        graph_node_id = str(row.get("graph_node_id", "") or "")
        route_node_id = str(row.get("route_node_id", "") or "")
        graph_anchor = node_positions.get(graph_node_id)
        route_anchor = node_positions.get(route_node_id)
        graph_x = (
            safe_float(graph_anchor[0], x_value)
            if isinstance(graph_anchor, tuple)
            else safe_float(row.get("graph_x", x_value), x_value)
        )
        graph_y = (
            safe_float(graph_anchor[1], y_value)
            if isinstance(graph_anchor, tuple)
            else safe_float(row.get("graph_y", y_value), y_value)
        )
        route_x = (
            safe_float(route_anchor[0], graph_x)
            if isinstance(route_anchor, tuple)
            else safe_float(row.get("route_x", graph_x), graph_x)
        )
        route_y = (
            safe_float(route_anchor[1], graph_y)
            if isinstance(route_anchor, tuple)
            else safe_float(row.get("route_y", graph_y), graph_y)
        )
        semantic_text_chars = max(
            0.0,
            safe_float(row.get("semantic_text_chars", 0.0), 0.0),
            safe_float(node_text_chars.get(route_node_id, 0.0), 0.0),
            safe_float(node_text_chars.get(graph_node_id, 0.0), 0.0),
        )
        message_probability = max(
            0.0, safe_float(row.get("message_probability", 0.0), 0.0)
        )
        package_entropy = max(0.0, safe_float(row.get("package_entropy", 0.0), 0.0))
        daimoi_energy = max(
            0.0,
            safe_float(row.get("daimoi_energy", 0.0), 0.0),
            message_probability + (package_entropy * 0.35),
        )
        semantic_mass = max(
            0.05,
            safe_float(row.get("semantic_mass", 0.0), 0.0),
            safe_float(row.get("mass", 0.0), 0.0),
        )

        compact_rows.append(
            {
                "id": particle_id,
                "presence_id": str(row.get("presence_id", "") or ""),
                "owner_presence_id": str(row.get("owner_presence_id", "") or ""),
                "target_presence_id": str(row.get("target_presence_id", "") or ""),
                "presence_role": str(row.get("presence_role", "") or ""),
                "particle_mode": str(row.get("particle_mode", "") or ""),
                "is_nexus": bool(row.get("is_nexus", False)),
                "resource_daimoi": bool(row.get("resource_daimoi", False)),
                "resource_type": str(row.get("resource_type", "") or ""),
                "resource_consume_type": str(
                    row.get("resource_consume_type", "") or ""
                ),
                "top_job": str(row.get("top_job", "") or ""),
                "graph_node_id": graph_node_id,
                "route_node_id": route_node_id,
                "route_resource_focus": str(row.get("route_resource_focus", "") or ""),
                "graph_x": round(max(0.0, min(1.0, graph_x)), 5),
                "graph_y": round(max(0.0, min(1.0, graph_y)), 5),
                "route_x": round(max(0.0, min(1.0, route_x)), 5),
                "route_y": round(max(0.0, min(1.0, route_y)), 5),
                "semantic_text_chars": round(semantic_text_chars, 3),
                "semantic_mass": round(semantic_mass, 6),
                "daimoi_energy": round(daimoi_energy, 6),
                "message_probability": round(message_probability, 6),
                "package_entropy": round(package_entropy, 6),
                "collision_count": int(
                    max(0, safe_float(row.get("collision_count", 0.0), 0.0))
                ),
                "x": round(x_value, 5),
                "y": round(y_value, 5),
                "size": round(
                    max(0.2, min(6.0, safe_float(row.get("size", 1.2), 1.2))), 4
                ),
                "r": round(max(0.0, min(1.0, safe_float(row.get("r", 0.4), 0.4))), 4),
                "g": round(max(0.0, min(1.0, safe_float(row.get("g", 0.5), 0.5))), 4),
                "b": round(max(0.0, min(1.0, safe_float(row.get("b", 0.7), 0.7))), 4),
                "drift_cost_semantic_term": round(
                    safe_float(row.get("drift_cost_semantic_term", 0.0), 0.0),
                    6,
                ),
                "drift_gravity_term": round(
                    safe_float(row.get("drift_gravity_term", 0.0), 0.0),
                    6,
                ),
                "valve_gravity_term": round(
                    safe_float(row.get("valve_gravity_term", 0.0), 0.0),
                    6,
                ),
                "drift_cost_term": round(
                    safe_float(row.get("drift_cost_term", 0.0), 0.0),
                    6,
                ),
                "route_probability": round(
                    max(
                        0.0,
                        min(1.0, safe_float(row.get("route_probability", 0.0), 0.0)),
                    ),
                    6,
                ),
                "influence_power": round(
                    max(
                        0.0, min(1.0, safe_float(row.get("influence_power", 0.0), 0.0))
                    ),
                    6,
                ),
                "node_saturation": round(
                    max(
                        0.0, min(1.0, safe_float(row.get("node_saturation", 0.0), 0.0))
                    ),
                    6,
                ),
                "gravity_potential": round(
                    max(0.0, safe_float(row.get("gravity_potential", 0.0), 0.0)),
                    6,
                ),
                "route_resource_focus_contribution": round(
                    safe_float(row.get("route_resource_focus_contribution", 0.0), 0.0),
                    6,
                ),
                "vx": round(vx_value, 6),
                "vy": round(vy_value, 6),
            }
        )
    return compact_rows


def simulation_ws_extract_stream_particles(
    simulation_payload: dict[str, Any],
    *,
    node_positions: dict[str, tuple[float, float]] | None,
    node_text_chars: dict[str, float] | None,
    ensure_daimoi_summary: Callable[[dict[str, Any]], None],
    compact_field_particles_with_nodes: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    dynamics = (
        simulation_payload.get("presence_dynamics", {})
        if isinstance(simulation_payload, dict)
        else {}
    )
    if not isinstance(dynamics, dict):
        return []

    compact_rows = compact_field_particles_with_nodes(
        dynamics.get("field_particles", []),
        node_positions=node_positions or {},
        node_text_chars=node_text_chars or {},
    )
    dynamics["field_particles"] = compact_rows
    ensure_daimoi_summary(simulation_payload)
    simulation_payload["presence_dynamics"] = dynamics
    return compact_rows
