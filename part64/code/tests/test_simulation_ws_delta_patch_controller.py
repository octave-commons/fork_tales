from __future__ import annotations

from typing import Any

from code.world_web import (
    simulation_ws_delta_patch_controller as delta_patch_controller_module,
)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def test_dynamics_patch_adds_daimoi_fields_and_summary() -> None:
    dynamics_patch: dict[str, Any] = {}
    tick_patch: dict[str, Any] = {}
    tick_changed_keys: list[str] = []

    last_graph = delta_patch_controller_module.apply_simulation_ws_dynamics_patch(
        dynamics={
            "daimoi_collision_events": [{"id": 1}, {"id": 2}],
            "daimoi_collision_events_record": "eta-mu.ws.events.v1",
            "daimoi_collision_event_seq": 7,
            "daimoi_probabilistic": {
                "clump_score": 1.7,
                "anti_clump_drive": 1.3,
                "anti_clump": {"hardened": True},
            },
        },
        dynamics_patch=dynamics_patch,
        tick_patch=tick_patch,
        tick_changed_keys=tick_changed_keys,
        graph_node_positions={},
        presence_anchor_positions={},
        now_monotonic=12.0,
        last_graph_position_broadcast=5.0,
        graph_position_heartbeat_seconds=10.0,
        governor_graph_heartbeat_scale=1.0,
        tick_slack_ms=4.0,
        ingestion_pressure=0.1,
        slack_ms_value=12.0,
        ws_graph_pos_max_default=2048,
        safe_int=_safe_int,
        safe_float=_safe_float,
        ws_clamp01=_clamp01,
    )

    assert last_graph == 5.0
    assert dynamics_patch.get("daimoi_collision_event_seq") == 7
    assert dynamics_patch.get("daimoi_collision_events_record") == "eta-mu.ws.events.v1"
    assert isinstance(dynamics_patch.get("daimoi_collision_events"), list)
    summary = dynamics_patch.get("daimoi_probabilistic", {})
    assert summary.get("clump_score") == 1.0
    assert summary.get("anti_clump_drive") == 1.0
    assert "presence_dynamics.daimoi_collision_event_seq" in tick_changed_keys
    assert "presence_dynamics.daimoi_probabilistic" in tick_changed_keys
    assert isinstance(tick_patch.get("presence_dynamics"), dict)


def test_dynamics_patch_truncates_graph_and_anchor_positions_when_due() -> None:
    graph = {f"g:{idx}": idx for idx in range(700)}
    anchors = {f"a:{idx}": idx for idx in range(900)}
    dynamics_patch: dict[str, Any] = {}
    tick_patch: dict[str, Any] = {}
    tick_changed_keys: list[str] = []

    last_graph = delta_patch_controller_module.apply_simulation_ws_dynamics_patch(
        dynamics={},
        dynamics_patch=dynamics_patch,
        tick_patch=tick_patch,
        tick_changed_keys=tick_changed_keys,
        graph_node_positions=graph,
        presence_anchor_positions=anchors,
        now_monotonic=100.0,
        last_graph_position_broadcast=10.0,
        graph_position_heartbeat_seconds=5.0,
        governor_graph_heartbeat_scale=1.0,
        tick_slack_ms=8.0,
        ingestion_pressure=0.8,
        slack_ms_value=3.0,
        ws_graph_pos_max_default=4096,
        safe_int=_safe_int,
        safe_float=_safe_float,
        ws_clamp01=_clamp01,
    )

    assert last_graph == 100.0
    assert len(dynamics_patch.get("graph_node_positions", {})) == 512
    assert len(dynamics_patch.get("presence_anchor_positions", {})) == 512
    assert tick_patch.get("graph_node_positions_truncated") is True
    assert tick_patch.get("graph_node_positions_total") == 700
    assert tick_patch.get("presence_anchor_positions_truncated") is True
    assert tick_patch.get("presence_anchor_positions_total") == 900


def test_dynamics_patch_skips_graph_when_heartbeat_not_due() -> None:
    graph = {"g:1": 1, "g:2": 2}
    anchors = {"a:1": 1}
    dynamics_patch: dict[str, Any] = {}
    tick_patch: dict[str, Any] = {}
    tick_changed_keys: list[str] = []

    last_graph = delta_patch_controller_module.apply_simulation_ws_dynamics_patch(
        dynamics={},
        dynamics_patch=dynamics_patch,
        tick_patch=tick_patch,
        tick_changed_keys=tick_changed_keys,
        graph_node_positions=graph,
        presence_anchor_positions=anchors,
        now_monotonic=2.0,
        last_graph_position_broadcast=0.0,
        graph_position_heartbeat_seconds=10.0,
        governor_graph_heartbeat_scale=1.0,
        tick_slack_ms=8.0,
        ingestion_pressure=0.1,
        slack_ms_value=12.0,
        ws_graph_pos_max_default=4096,
        safe_int=_safe_int,
        safe_float=_safe_float,
        ws_clamp01=_clamp01,
    )

    assert last_graph == 0.0
    assert "graph_node_positions" not in dynamics_patch
    assert "presence_anchor_positions" not in dynamics_patch
