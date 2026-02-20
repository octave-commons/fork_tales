from __future__ import annotations

from typing import Any

from code.world_web import c_double_buffer_backend


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def snapshot(
        self,
    ) -> tuple[
        int,
        int,
        int,
        int,
        int,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
    ]:
        return (
            3,
            42,
            420,
            421,
            422,
            [0.1, 0.5, 0.9],
            [0.2, 0.6, 0.8],
            [0.01, 0.0, -0.01],
            [0.0, 0.02, -0.02],
            [0.92, 0.45, 0.66],
            [0.0, 0.35, 0.27],
            [0.2, 0.7, 0.5],
            [0, 1, 2],
            [
                c_double_buffer_backend.CDB_FLAG_NEXUS,
                c_double_buffer_backend.CDB_FLAG_CHAOS,
                0,
            ],
        )


def test_c_double_buffer_builder_maps_numeric_ids_to_rows(monkeypatch: Any) -> None:
    fake_engine = _FakeEngine()

    def fake_get_engine(*, count: int, seed: int) -> _FakeEngine:
        fake_engine.calls.append((count, seed))
        return fake_engine

    monkeypatch.setattr(c_double_buffer_backend, "_get_engine", fake_get_engine)

    rows, summary = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={"file_nodes": [{"id": "file:1"}]},
        presence_impacts=[
            {"id": "witness_thread"},
            {"id": "anchor_registry"},
        ],
        resource_heartbeat={"devices": {"cpu": {"utilization": 43.0}}},
        compute_jobs=[],
        queue_ratio=0.2,
        now=1_700_900_100.0,
    )

    assert len(rows) == 3
    assert rows[0]["id"] == "cdb:0"
    assert rows[0]["is_nexus"] is True
    assert rows[0]["particle_mode"] == "static-daimoi"
    assert rows[1]["particle_mode"] == "chaos-butterfly"
    assert rows[2]["particle_mode"] == "neutral"
    assert rows[0]["presence_id"] == "witness_thread"
    assert rows[1]["presence_id"] == "anchor_registry"
    assert 0.45 <= float(rows[0]["x"]) <= 0.66
    assert 0.2 <= float(rows[0]["y"]) <= 0.36
    assert 0.42 <= float(rows[1]["x"]) <= 0.56
    assert 0.48 <= float(rows[1]["y"]) <= 0.62
    assert summary["backend"] == "c-double-buffer"
    assert summary["double_buffer"] is True
    assert summary["frame_id"] == 42
    assert summary["semantic_frame"] == 422
    assert summary["systems"] == ["force", "chaos", "semantic", "integrate"]
    assert summary["deflects"] == 2
    assert summary["diffuses"] == 1
    assert summary["mean_package_entropy"] == 0.466667
    assert summary["mean_message_probability"] == 0.206667
    assert rows[0]["action_probabilities"]["deflect"] == 0.92
    assert rows[2]["message_probability"] == 0.27
    assert fake_engine.calls


def test_c_double_buffer_builder_uses_default_presence_ids(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        c_double_buffer_backend,
        "_get_engine",
        lambda **_: _FakeEngine(),
    )

    rows, _ = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={"file_nodes": []},
        presence_impacts=[],
        resource_heartbeat={},
        compute_jobs=[],
        queue_ratio=0.0,
        now=1_700_900_200.0,
    )

    assert rows
    assert str(rows[0].get("presence_id", "")).strip()


def test_c_double_buffer_builder_honors_manifest_anchor_layout(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        c_double_buffer_backend,
        "_get_engine",
        lambda **_: _FakeEngine(),
    )

    rows, _ = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={"file_nodes": []},
        presence_impacts=[{"id": "alpha"}],
        resource_heartbeat={},
        compute_jobs=[],
        queue_ratio=0.0,
        now=1_700_900_300.0,
        entity_manifest=[{"id": "alpha", "x": 0.1, "y": 0.9, "hue": 15}],
    )

    assert rows
    assert rows[0]["presence_id"] == "alpha"
    assert 0.0 <= float(rows[0]["x"]) <= 0.2
    assert 0.74 <= float(rows[0]["y"]) <= 1.0


def test_c_double_buffer_builder_applies_graph_runtime_metrics(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        c_double_buffer_backend,
        "_get_engine",
        lambda **_: _FakeEngine(),
    )
    monkeypatch.setattr(
        c_double_buffer_backend,
        "compute_graph_runtime_maps_native",
        lambda **_: {
            "record": "eta-mu.graph-runtime.cdb.v1",
            "schema_version": "graph.runtime.cdb.v1",
            "node_ids": ["node:a", "node:b"],
            "source_node_index_by_presence": {
                "witness_thread": 0,
                "anchor_registry": 1,
            },
            "min_distance": [0.12, 0.84],
            "gravity": [1.4, 0.3],
            "node_price": [2.8, 1.2],
            "node_saturation": [0.76, 0.24],
            "edge_cost": [1.1, 1.5],
            "edge_health": [0.88, 0.73],
            "edge_affinity": [0.81, 0.44],
            "edge_saturation": [0.62, 0.31],
            "edge_latency_component": [1.0, 1.0],
            "edge_congestion_component": [1.24, 0.62],
            "edge_semantic_component": [0.19, 0.56],
            "edge_upkeep_penalty": [0.12, 0.18],
            "node_count": 2,
            "edge_count": 2,
            "source_count": 2,
            "global_saturation": 0.53,
            "valve_weights": {
                "pressure": 0.44,
                "gravity": 1.0,
                "affinity": 0.36,
                "saturation": 0.52,
                "health": 0.34,
            },
            "radius_cost": 6.0,
            "cost_weights": {"latency": 1.0, "congestion": 2.0, "semantic": 1.0},
            "edge_cost_mean": 1.3,
            "edge_cost_max": 1.5,
            "edge_health_mean": 0.805,
            "edge_health_max": 0.88,
            "edge_health_min": 0.73,
            "edge_saturation_mean": 0.465,
            "edge_saturation_max": 0.62,
            "edge_affinity_mean": 0.625,
            "edge_upkeep_penalty_mean": 0.15,
            "gravity_mean": 0.85,
            "gravity_max": 1.4,
            "price_mean": 2.0,
            "price_max": 2.8,
            "resource_types": ["cpu", "ram"],
            "active_resource_types": ["cpu", "ram"],
            "resource_gravity_peak_max": 1.2,
            "resource_gravity_peaks": {"cpu": 1.2, "ram": 0.8},
            "top_nodes": [{"node_id": "node:a", "gravity": 1.4, "local_price": 2.8}],
        },
    )

    rows, summary = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={
            "nodes": [
                {"id": "node:a", "x": 0.3, "y": 0.3},
                {"id": "node:b", "x": 0.7, "y": 0.7},
            ],
            "edges": [{"source": "node:a", "target": "node:b", "weight": 0.5}],
            "file_nodes": [{"id": "file:1"}],
        },
        presence_impacts=[
            {"id": "witness_thread"},
            {"id": "anchor_registry"},
        ],
        resource_heartbeat={"devices": {"cpu": {"utilization": 30.0}}},
        compute_jobs=[],
        queue_ratio=0.2,
        now=1_700_900_400.0,
    )

    assert rows
    assert rows[0]["graph_node_id"] == "node:a"
    assert rows[1]["graph_node_id"] == "node:b"
    assert rows[0]["route_node_id"] == "node:a"
    assert rows[1]["route_node_id"] == "node:b"
    assert rows[0]["local_price"] == 2.8
    assert rows[1]["node_saturation"] == 0.24
    assert "drift_gravity_term" in rows[0]
    assert "drift_cost_term" in rows[0]
    assert "selected_edge_health" in rows[0]
    assert "drift_cost_upkeep_term" in rows[0]
    assert "selected_edge_affinity" in rows[0]
    assert "selected_edge_saturation" in rows[0]
    assert "valve_score_proxy" in rows[0]
    assert "route_gravity_mode" in rows[0]
    assert "route_resource_focus" in rows[0]
    assert "route_resource_focus_weight" in rows[0]
    assert float(rows[2]["action_probabilities"]["deflect"]) < 0.66
    assert float(rows[2]["message_probability"]) > 0.27
    assert "mean_drift_score" in summary
    assert "mean_route_probability" in summary
    assert "mean_drift_gravity_term" in summary
    assert "mean_drift_cost_term" in summary
    assert "mean_drift_cost_upkeep_term" in summary
    assert "mean_selected_edge_health" in summary
    assert "mean_selected_edge_affinity" in summary
    assert "mean_selected_edge_saturation" in summary
    assert "mean_valve_score_proxy" in summary
    assert "mean_route_resource_focus_weight" in summary
    assert "mean_route_resource_focus_contribution" in summary
    assert summary.get("resource_routing_mode") in {
        "scalar-gravity",
        "resource-signature",
    }
    assert "resource_route_ratio" in summary

    graph_runtime = summary.get("graph_runtime", {})
    assert graph_runtime.get("record") == "eta-mu.graph-runtime.cdb.v1"
    assert graph_runtime.get("node_count") == 2
    assert graph_runtime.get("edge_health_mean") == 0.805
    assert graph_runtime.get("edge_saturation_mean") == 0.465
    assert graph_runtime.get("edge_affinity_mean") == 0.625
    assert graph_runtime.get("global_saturation") == 0.53
    assert graph_runtime.get("resource_types") == ["cpu", "ram"]
    assert graph_runtime.get("resource_gravity_peaks", {}).get("cpu") == 1.2
    assert summary.get("graph_systems") == [
        "edge-cost",
        "edge-upkeep-health",
        "bounded-gravity",
        "local-price",
        "valve-score-diagnostics",
    ]


def test_graph_route_step_native_returns_routes() -> None:
    route = c_double_buffer_backend.compute_graph_route_step_native(
        graph_runtime={
            "node_count": 3,
            "edge_count": 2,
            "edge_src_index": [0, 1],
            "edge_dst_index": [1, 2],
            "edge_cost": [1.0, 1.1],
            "edge_health": [0.9, 0.8],
            "edge_affinity": [0.8, 0.5],
            "edge_saturation": [0.4, 0.45],
            "edge_latency_component": [1.0, 1.0],
            "edge_congestion_component": [0.8, 0.9],
            "edge_semantic_component": [0.2, 0.5],
            "edge_upkeep_penalty": [0.05, 0.12],
            "gravity": [0.2, 0.8, 0.4],
            "node_price": [1.5, 1.2, 1.0],
        },
        particle_source_nodes=[0, 1, 2],
    )

    assert isinstance(route, dict)
    next_nodes = route.get("next_node_index", [])
    probabilities = route.get("route_probability", [])
    drifts = route.get("drift_score", [])
    gravity_terms = route.get("drift_gravity_term", [])
    cost_terms = route.get("drift_cost_term", [])
    edge_health = route.get("selected_edge_health", [])
    cost_upkeep_terms = route.get("drift_cost_upkeep_term", [])
    selected_affinity = route.get("selected_edge_affinity", [])
    valve_proxy = route.get("valve_score_proxy", [])
    route_mode = route.get("resource_routing_mode")
    route_focus = route.get("route_resource_focus", [])
    assert len(next_nodes) == 3
    assert len(probabilities) == 3
    assert len(drifts) == 3
    assert len(gravity_terms) == 3
    assert len(cost_terms) == 3
    assert len(edge_health) == 3
    assert len(cost_upkeep_terms) == 3
    assert len(selected_affinity) == 3
    assert len(valve_proxy) == 3
    assert len(route_focus) == 3
    assert route_mode == "scalar-gravity"
    assert next_nodes[0] == 1
    assert next_nodes[1] == 2
    assert next_nodes[2] == 2


def test_graph_route_step_native_uses_resource_signature_gravity() -> None:
    route = c_double_buffer_backend.compute_graph_route_step_native(
        graph_runtime={
            "node_count": 3,
            "edge_count": 4,
            "edge_src_index": [0, 0, 1, 2],
            "edge_dst_index": [1, 2, 0, 0],
            "edge_cost": [1.0, 1.0, 1.0, 1.0],
            "edge_health": [1.0, 1.0, 1.0, 1.0],
            "edge_affinity": [0.5, 0.5, 0.5, 0.5],
            "edge_saturation": [0.2, 0.2, 0.2, 0.2],
            "edge_latency_component": [1.0, 1.0, 1.0, 1.0],
            "edge_congestion_component": [0.4, 0.4, 0.4, 0.4],
            "edge_semantic_component": [0.5, 0.5, 0.5, 0.5],
            "edge_upkeep_penalty": [0.0, 0.0, 0.0, 0.0],
            "gravity": [0.0, 0.0, 0.0],
            "node_price": [1.0, 1.0, 1.0],
            "resource_gravity_maps": {
                "cpu": [0.0, 2.0, 0.1],
                "ram": [0.0, 0.1, 2.0],
                "gpu": [0.0, 0.0, 0.0],
                "npu": [0.0, 0.0, 0.0],
                "disk": [0.0, 0.0, 0.0],
                "network": [0.0, 0.0, 0.0],
            },
        },
        particle_source_nodes=[0, 0],
        particle_resource_signature=[
            {"cpu": 1.0},
            {"ram": 1.0},
        ],
        step_seed=0,
    )

    assert isinstance(route, dict)
    assert route.get("resource_routing_mode") == "resource-signature"
    next_nodes = route.get("next_node_index", [])
    focus = route.get("route_resource_focus", [])
    assert len(next_nodes) == 2
    assert len(focus) == 2
    assert next_nodes[0] == 1
    assert next_nodes[1] == 2
    assert focus[0] == "cpu"
    assert focus[1] == "ram"
