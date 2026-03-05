from __future__ import annotations

from typing import Any, Callable


def _capped_mapping(values: dict[str, Any], max_items: int) -> dict[str, Any]:
    capped: dict[str, Any] = {}
    for key, value in values.items():
        capped[key] = value
        if len(capped) >= max_items:
            break
    return capped


def apply_simulation_ws_dynamics_patch(
    *,
    dynamics: dict[str, Any],
    dynamics_patch: dict[str, Any],
    tick_patch: dict[str, Any],
    tick_changed_keys: list[str],
    graph_node_positions: dict[str, Any],
    presence_anchor_positions: dict[str, Any],
    now_monotonic: float,
    last_graph_position_broadcast: float,
    graph_position_heartbeat_seconds: float,
    governor_graph_heartbeat_scale: float,
    tick_slack_ms: float,
    ingestion_pressure: float,
    slack_ms_value: float,
    ws_graph_pos_max_default: int,
    safe_int: Callable[[Any, int], int],
    safe_float: Callable[[Any, float], float],
    ws_clamp01: Callable[[float], float],
) -> float:
    daimoi_collision_events = dynamics.get("daimoi_collision_events", [])
    if isinstance(daimoi_collision_events, list):
        event_tail = [
            dict(row) for row in daimoi_collision_events[-96:] if isinstance(row, dict)
        ]
        if event_tail:
            dynamics_patch["daimoi_collision_events"] = event_tail
            tick_changed_keys.append("presence_dynamics.daimoi_collision_events")

    daimoi_collision_record = str(
        dynamics.get("daimoi_collision_events_record", "") or ""
    ).strip()
    if daimoi_collision_record:
        dynamics_patch["daimoi_collision_events_record"] = daimoi_collision_record
        tick_changed_keys.append("presence_dynamics.daimoi_collision_events_record")

    daimoi_collision_seq = max(
        0,
        safe_int(dynamics.get("daimoi_collision_event_seq", 0), 0),
    )
    dynamics_patch["daimoi_collision_event_seq"] = daimoi_collision_seq
    tick_changed_keys.append("presence_dynamics.daimoi_collision_event_seq")

    daimoi_summary = dynamics.get("daimoi_probabilistic", {})
    if isinstance(daimoi_summary, dict) and daimoi_summary:
        anti_payload = (
            daimoi_summary.get("anti_clump", {})
            if isinstance(daimoi_summary.get("anti_clump", {}), dict)
            else {}
        )
        dynamics_patch["daimoi_probabilistic"] = {
            "clump_score": ws_clamp01(
                safe_float(daimoi_summary.get("clump_score", 0.0), 0.0)
            ),
            "anti_clump_drive": max(
                -1.0,
                min(1.0, safe_float(daimoi_summary.get("anti_clump_drive", 0.0), 0.0)),
            ),
            "anti_clump": anti_payload,
        }
        tick_changed_keys.append("presence_dynamics.daimoi_probabilistic")

    graph_position_heartbeat_due = (
        graph_position_heartbeat_seconds <= 0.0
        or (
            now_monotonic - last_graph_position_broadcast
            >= (
                graph_position_heartbeat_seconds
                * max(0.1, governor_graph_heartbeat_scale)
            )
        )
    ) and tick_slack_ms > 2.0

    graph_pos_cap = int(ws_graph_pos_max_default)
    if ingestion_pressure >= 0.7:
        graph_pos_cap = min(graph_pos_cap, 512)
    if slack_ms_value <= 4.0:
        graph_pos_cap = min(graph_pos_cap, 512)
    elif slack_ms_value <= 10.0:
        graph_pos_cap = min(graph_pos_cap, 1024)

    if graph_node_positions and graph_position_heartbeat_due:
        if len(graph_node_positions) > graph_pos_cap:
            dynamics_patch["graph_node_positions"] = _capped_mapping(
                graph_node_positions,
                graph_pos_cap,
            )
            tick_patch["graph_node_positions_truncated"] = True
            tick_patch["graph_node_positions_total"] = len(graph_node_positions)
            tick_changed_keys.append("graph_node_positions_truncated")
            tick_changed_keys.append("graph_node_positions_total")
        else:
            dynamics_patch["graph_node_positions"] = graph_node_positions
        tick_changed_keys.append("presence_dynamics.graph_node_positions")

    if presence_anchor_positions and graph_position_heartbeat_due:
        anchor_pos_cap = max(64, min(graph_pos_cap, 2048))
        if len(presence_anchor_positions) > anchor_pos_cap:
            dynamics_patch["presence_anchor_positions"] = _capped_mapping(
                presence_anchor_positions,
                anchor_pos_cap,
            )
            tick_patch["presence_anchor_positions_truncated"] = True
            tick_patch["presence_anchor_positions_total"] = len(
                presence_anchor_positions
            )
            tick_changed_keys.append("presence_anchor_positions_truncated")
            tick_changed_keys.append("presence_anchor_positions_total")
        else:
            dynamics_patch["presence_anchor_positions"] = presence_anchor_positions
        tick_changed_keys.append("presence_dynamics.presence_anchor_positions")

    if graph_position_heartbeat_due and (
        graph_node_positions or presence_anchor_positions
    ):
        last_graph_position_broadcast = now_monotonic
    if dynamics_patch:
        tick_patch["presence_dynamics"] = dynamics_patch

    return float(last_graph_position_broadcast)
