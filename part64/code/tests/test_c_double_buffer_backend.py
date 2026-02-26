from __future__ import annotations

from typing import Any

from code.world_web import c_double_buffer_backend


def _array_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return -1


class _FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.nooi_updates: list[list[float]] = []
        self.embedding_updates: list[list[float]] = []

    def update_nooi(self, data: list[float]) -> None:
        self.nooi_updates.append(data)

    def update_embeddings(self, data: list[float]) -> None:
        self.embedding_updates.append(data)

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
    assert float(rows[0].get("semantic_text_chars", 0.0)) >= 0.0
    assert float(rows[0].get("semantic_mass", 0.0)) > 0.0
    assert float(rows[0].get("resource_wallet_total", 0.0)) >= 0.0
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


def test_c_double_buffer_builder_emits_anti_clump_summary(monkeypatch: Any) -> None:
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
        now=1_700_900_125.0,
    )

    assert rows
    assert 0.0 <= float(summary.get("clump_score", 0.0)) <= 1.0
    assert -1.0 <= float(summary.get("anti_clump_drive", 0.0)) <= 1.0
    anti_clump = summary.get("anti_clump", {})
    assert isinstance(anti_clump, dict)
    assert 0.0 <= float(anti_clump.get("target", 0.0)) <= 1.0
    assert 0.0 <= float(anti_clump.get("clump_score", 0.0)) <= 1.0
    assert -1.0 <= float(anti_clump.get("drive", 0.0)) <= 1.0
    assert float(anti_clump.get("snr", 0.0)) >= 0.0
    metrics = anti_clump.get("metrics", {})
    scales = anti_clump.get("scales", {})
    assert isinstance(metrics, dict)
    assert isinstance(scales, dict)
    assert "nn_term" in metrics
    assert "fano_factor" in metrics
    assert "snr" in metrics
    assert "spawn" in scales


def test_c_double_buffer_builder_tracks_graph_node_variability(
    monkeypatch: Any,
) -> None:
    fake_engine = _FakeEngine()

    def fake_get_engine(*, count: int, seed: int) -> _FakeEngine:
        fake_engine.calls.append((count, seed))
        return fake_engine

    graph_call = {"count": 0}

    def fake_graph_runtime(**_: Any) -> dict[str, Any]:
        graph_call["count"] += 1
        shift = 0.0 if graph_call["count"] == 1 else 0.18
        return {
            "record": "eta-mu.graph-runtime.cdb.v1",
            "schema_version": "graph.runtime.cdb.v1",
            "node_ids": ["node:a", "node:b"],
            "node_positions": [
                (0.2 + shift, 0.2),
                (0.8 - shift, 0.8),
            ],
            "source_node_index_by_presence": {
                "witness_thread": 0,
                "anchor_registry": 1,
            },
            "min_distance": [0.1, 0.2],
            "gravity": [1.0, 0.5],
            "node_price": [2.0, 1.5],
            "node_saturation": [0.3, 0.4],
            "edge_src_index": [0, 1],
            "edge_dst_index": [1, 0],
            "edge_cost": [1.0, 1.0],
            "edge_health": [0.9, 0.9],
            "edge_affinity": [0.5, 0.5],
            "edge_saturation": [0.2, 0.2],
            "edge_latency_component": [1.0, 1.0],
            "edge_congestion_component": [0.4, 0.4],
            "edge_semantic_component": [0.5, 0.5],
            "edge_upkeep_penalty": [0.0, 0.0],
            "resource_types": ["cpu"],
            "resource_gravity_maps": {"cpu": [0.2, 0.1]},
            "resource_gravity_peaks": {"cpu": 0.2},
            "active_resource_types": ["cpu"],
            "node_count": 2,
            "edge_count": 2,
            "source_count": 2,
            "source_profiles": [],
            "presence_source_count": 0,
            "presence_model": {"mask": "nearest-k.v1"},
            "global_saturation": 0.2,
            "valve_weights": {"pressure": 0.44},
            "radius_cost": 6.0,
            "cost_weights": {"latency": 1.0, "congestion": 2.0, "semantic": 1.0},
            "edge_cost_mean": 1.0,
            "edge_cost_max": 1.0,
            "edge_health_mean": 0.9,
            "edge_health_max": 0.9,
            "edge_health_min": 0.9,
            "edge_saturation_mean": 0.2,
            "edge_saturation_max": 0.2,
            "edge_affinity_mean": 0.5,
            "edge_upkeep_penalty_mean": 0.0,
            "gravity_mean": 0.75,
            "gravity_max": 1.0,
            "price_mean": 1.75,
            "price_max": 2.0,
            "resource_gravity_peak_max": 0.2,
            "top_nodes": [{"node_id": "node:a", "gravity": 1.0, "local_price": 2.0}],
        }

    def fake_graph_route_step(**kwargs: Any) -> dict[str, Any]:
        source_nodes = kwargs.get("particle_source_nodes", [])
        count = len(source_nodes) if isinstance(source_nodes, list) else 0
        return {
            "next_node_index": [0 for _ in range(count)],
            "drift_score": [0.0 for _ in range(count)],
            "route_probability": [0.5 for _ in range(count)],
            "drift_gravity_term": [0.0 for _ in range(count)],
            "drift_cost_term": [0.0 for _ in range(count)],
            "drift_gravity_delta": [0.0 for _ in range(count)],
            "drift_gravity_delta_scalar": [0.0 for _ in range(count)],
            "selected_edge_cost": [0.0 for _ in range(count)],
            "selected_edge_health": [1.0 for _ in range(count)],
            "drift_cost_latency_term": [0.0 for _ in range(count)],
            "drift_cost_congestion_term": [0.0 for _ in range(count)],
            "drift_cost_semantic_term": [0.0 for _ in range(count)],
            "drift_cost_upkeep_term": [0.0 for _ in range(count)],
            "selected_edge_affinity": [0.5 for _ in range(count)],
            "selected_edge_saturation": [0.0 for _ in range(count)],
            "selected_edge_upkeep_penalty": [0.0 for _ in range(count)],
            "valve_pressure_term": [0.0 for _ in range(count)],
            "valve_gravity_term": [0.0 for _ in range(count)],
            "valve_affinity_term": [0.0 for _ in range(count)],
            "valve_saturation_term": [0.0 for _ in range(count)],
            "valve_health_term": [0.0 for _ in range(count)],
            "valve_score_proxy": [0.0 for _ in range(count)],
            "route_resource_focus": ["" for _ in range(count)],
            "route_resource_focus_weight": [0.0 for _ in range(count)],
            "route_resource_focus_delta": [0.0 for _ in range(count)],
            "route_resource_focus_contribution": [0.0 for _ in range(count)],
            "route_gravity_mode": ["scalar-gravity" for _ in range(count)],
            "resource_routing_mode": "scalar-gravity",
        }

    monkeypatch.setattr(c_double_buffer_backend, "_get_engine", fake_get_engine)
    monkeypatch.setattr(
        c_double_buffer_backend,
        "compute_graph_runtime_maps_native",
        fake_graph_runtime,
    )
    monkeypatch.setattr(
        c_double_buffer_backend,
        "compute_graph_route_step_native",
        fake_graph_route_step,
    )

    _, first_summary = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={"file_nodes": [{"id": "file:1"}]},
        presence_impacts=[
            {"id": "witness_thread"},
            {"id": "anchor_registry"},
        ],
        resource_heartbeat={"devices": {"cpu": {"utilization": 43.0}}},
        compute_jobs=[],
        queue_ratio=0.2,
        now=1_700_900_126.0,
    )
    _, second_summary = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={"file_nodes": [{"id": "file:1"}]},
        presence_impacts=[
            {"id": "witness_thread"},
            {"id": "anchor_registry"},
        ],
        resource_heartbeat={"devices": {"cpu": {"utilization": 43.0}}},
        compute_jobs=[],
        queue_ratio=0.2,
        now=1_700_900_127.0,
    )

    first_variability = (first_summary.get("anti_clump", {}) or {}).get(
        "graph_variability", {}
    ) or {}
    second_anti = second_summary.get("anti_clump", {}) or {}
    second_variability = second_anti.get("graph_variability", {}) or {}
    second_scales = second_anti.get("scales", {}) or {}

    assert float(first_variability.get("score", 0.0)) <= float(
        second_variability.get("score", 0.0)
    )
    assert float(second_variability.get("score", 0.0)) > 0.0
    assert float(second_variability.get("raw_score", 0.0)) > 0.0
    assert int(second_variability.get("shared_nodes", 0)) >= 2
    assert float(second_scales.get("noise_gain", 1.0)) > 1.0
    assert float(second_scales.get("route_damp", 1.0)) < 1.0
    assert float(second_scales.get("tangent_effective", 1.0)) >= float(
        second_scales.get("tangent_base", 1.0)
    )


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


def test_c_double_buffer_builder_reports_clamped_force_runtime_config(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        c_double_buffer_backend,
        "_get_engine",
        lambda **_: _FakeEngine(),
    )
    monkeypatch.setenv("CDB_FORCE_WORKERS", "999")
    monkeypatch.setenv("CDB_COLLISION_WORKERS", "999")
    monkeypatch.setenv("CDB_BH_THETA", "10")
    monkeypatch.setenv("CDB_BH_LEAF_CAP", "1")
    monkeypatch.setenv("CDB_BH_MAX_DEPTH", "99")
    monkeypatch.setenv("CDB_COLLISION_SPRING", "-5")
    monkeypatch.setenv("CDB_CLUSTER_THETA", "9")
    monkeypatch.setenv("CDB_CLUSTER_REST_LENGTH", "-2")
    monkeypatch.setenv("CDB_CLUSTER_STIFFNESS", "0")

    _, summary = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={"file_nodes": []},
        presence_impacts=[],
        resource_heartbeat={},
        compute_jobs=[],
        queue_ratio=0.0,
        now=1_700_900_225.0,
    )

    force_runtime = summary.get("force_runtime", {})
    assert force_runtime.get("barnes_hut_theta") == 1.4
    assert force_runtime.get("barnes_hut_leaf_capacity") == 1
    assert force_runtime.get("barnes_hut_max_depth") == 24
    assert force_runtime.get("collision_spring") == 1.0
    assert force_runtime.get("cluster_theta") == 1.6
    assert force_runtime.get("cluster_rest_length") == 0.01
    assert force_runtime.get("cluster_stiffness") == 0.1
    assert force_runtime.get("force_workers_requested") == 32
    assert force_runtime.get("force_workers_effective") == 3
    assert force_runtime.get("collision_workers_requested") == 32
    assert force_runtime.get("collision_workers_effective") == 3
    assert force_runtime.get("quadtree_capacity_estimate") == 256


def test_c_double_buffer_builder_disables_cpu_core_emitter_when_cpu_hot(
    monkeypatch: Any,
) -> None:
    fake_engine = _FakeEngine()

    def fake_get_engine(*, count: int, seed: int) -> _FakeEngine:
        fake_engine.calls.append((count, seed))
        return fake_engine

    monkeypatch.setattr(c_double_buffer_backend, "_get_engine", fake_get_engine)
    monkeypatch.setenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "75")

    rows, summary = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={"file_nodes": []},
        presence_impacts=[
            {"id": "presence.core.cpu"},
            {"id": "witness_thread"},
        ],
        resource_heartbeat={"devices": {"cpu": {"utilization": 88.0}}},
        compute_jobs=[],
        queue_ratio=0.0,
        now=1_700_900_250.0,
    )

    assert rows
    assert all(str(row.get("presence_id", "")) != "presence.core.cpu" for row in rows)
    assert summary.get("cpu_core_emitter_enabled") is False
    assert summary.get("cpu_daimoi_stop_percent") == 75.0


def test_c_double_buffer_builder_prioritizes_core_emitters_when_cpu_cool(
    monkeypatch: Any,
) -> None:
    fake_engine = _FakeEngine()

    def fake_get_engine(*, count: int, seed: int) -> _FakeEngine:
        fake_engine.calls.append((count, seed))
        return fake_engine

    monkeypatch.setattr(c_double_buffer_backend, "_get_engine", fake_get_engine)
    monkeypatch.setenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "75")

    rows, summary = c_double_buffer_backend.build_double_buffer_field_particles(
        file_graph={"file_nodes": []},
        presence_impacts=[
            {"id": "witness_thread"},
            {"id": "presence.core.cpu"},
            {"id": "health_sentinel_cpu"},
        ],
        resource_heartbeat={"devices": {"cpu": {"utilization": 18.0}}},
        compute_jobs=[],
        queue_ratio=0.0,
        now=1_700_900_275.0,
    )

    assert rows
    assert summary.get("cpu_core_emitter_enabled") is True
    assert any(str(row.get("presence_id", "")) == "presence.core.cpu" for row in rows)


def test_resolve_semantic_collisions_native_updates_overlapping_pairs() -> None:
    resolved = c_double_buffer_backend.resolve_semantic_collisions_native(
        x=[0.5, 0.506],
        y=[0.5, 0.5],
        vx=[0.02, -0.02],
        vy=[0.0, 0.0],
        radius=[0.02, 0.02],
        mass=[1.0, 1.0],
        worker_count=2,
    )

    assert resolved is not None
    x_next, y_next, vx_next, vy_next, collisions = resolved
    assert len(x_next) == 2
    assert len(y_next) == 2
    assert len(vx_next) == 2
    assert len(vy_next) == 2
    assert collisions[0] > 0
    assert collisions[1] > 0
    assert 0.0 <= x_next[0] <= 1.0
    assert 0.0 <= x_next[1] <= 1.0
    assert 0.0 <= y_next[0] <= 1.0
    assert 0.0 <= y_next[1] <= 1.0
    assert abs(x_next[1] - x_next[0]) >= 0.0001


def test_resolve_semantic_collisions_native_rejects_mismatched_lengths() -> None:
    resolved = c_double_buffer_backend.resolve_semantic_collisions_native(
        x=[0.5, 0.6],
        y=[0.5],
        vx=[0.0, 0.0],
        vy=[0.0, 0.0],
        radius=[0.01, 0.01],
        mass=[1.0, 1.0],
    )
    assert resolved is None


def test_resolve_semantic_collisions_native_inplace_updates_inputs() -> None:
    x = [0.5, 0.506]
    y = [0.5, 0.5]
    vx = [0.02, -0.02]
    vy = [0.0, 0.0]
    radius = [0.02, 0.02]
    mass = [1.0, 1.0]
    collisions: list[int] = []

    ok = c_double_buffer_backend.resolve_semantic_collisions_native_inplace(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        radius=radius,
        mass=mass,
        collisions_out=collisions,
        worker_count=2,
    )

    assert ok is True
    assert len(collisions) == 2
    assert collisions[0] > 0
    assert collisions[1] > 0
    assert abs(x[1] - x[0]) >= 0.0001
    assert 0.0 <= x[0] <= 1.0
    assert 0.0 <= x[1] <= 1.0


def test_resolve_semantic_collisions_native_inplace_rejects_mismatched_lengths() -> (
    None
):
    ok = c_double_buffer_backend.resolve_semantic_collisions_native_inplace(
        x=[0.5, 0.6],
        y=[0.5],
        vx=[0.0, 0.0],
        vy=[0.0, 0.0],
        radius=[0.01, 0.01],
        mass=[1.0, 1.0],
    )
    assert ok is False


def test_collision_ctype_scratch_shrinks_after_sustained_low_watermark(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(c_double_buffer_backend, "_COLLISION_SCRATCH_SHRINK_FACTOR", 4)
    monkeypatch.setattr(c_double_buffer_backend, "_COLLISION_SCRATCH_SHRINK_RUNS", 3)
    monkeypatch.setattr(
        c_double_buffer_backend, "_COLLISION_SCRATCH_SHRINK_MIN_CAPACITY", 16
    )
    releases: list[tuple[int, int]] = []

    def fake_release(
        *, prior_capacity: int, next_capacity: int, state: dict[str, Any]
    ) -> None:
        releases.append((prior_capacity, next_capacity))
        state["last_native_release_at"] = 123.0

    monkeypatch.setattr(
        c_double_buffer_backend,
        "_collision_release_native_thread_scratch_if_needed",
        fake_release,
    )

    seeded = c_double_buffer_backend._collision_ctype_state(128)
    seeded["shrink_candidate_runs"] = 2
    c_double_buffer_backend._COLLISION_CTYPE_SCRATCH.state = seeded

    shrunk = c_double_buffer_backend._collision_ctype_scratch(16)

    assert int(shrunk.get("capacity", 0)) == 16
    assert int(shrunk.get("shrink_candidate_runs", -1)) == 0
    assert _array_len(shrunk.get("x")) == 16
    assert _array_len(shrunk.get("collision")) == 16
    assert releases == [(128, 16)]


def test_collision_ctype_scratch_low_water_counter_resets_on_rebound(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(c_double_buffer_backend, "_COLLISION_SCRATCH_SHRINK_FACTOR", 4)
    monkeypatch.setattr(c_double_buffer_backend, "_COLLISION_SCRATCH_SHRINK_RUNS", 3)
    monkeypatch.setattr(
        c_double_buffer_backend, "_COLLISION_SCRATCH_SHRINK_MIN_CAPACITY", 16
    )

    seeded = c_double_buffer_backend._collision_ctype_state(128)
    c_double_buffer_backend._COLLISION_CTYPE_SCRATCH.state = seeded

    c_double_buffer_backend._collision_ctype_scratch(16)
    c_double_buffer_backend._collision_ctype_scratch(16)
    state_mid = c_double_buffer_backend._COLLISION_CTYPE_SCRATCH.state
    assert int(state_mid.get("capacity", 0)) == 128
    assert int(state_mid.get("shrink_candidate_runs", 0)) == 2

    c_double_buffer_backend._collision_ctype_scratch(96)
    state_rebound = c_double_buffer_backend._COLLISION_CTYPE_SCRATCH.state
    assert int(state_rebound.get("capacity", 0)) == 128
    assert int(state_rebound.get("shrink_candidate_runs", -1)) == 0


def test_mask_nodes_for_anchor_prefers_nearest_nodes() -> None:
    mask = c_double_buffer_backend._mask_nodes_for_anchor(
        node_ids=["node:a", "node:b", "node:c"],
        node_positions=[(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)],
        anchor_x=0.08,
        anchor_y=0.12,
        k=2,
    )

    assert len(mask) == 2
    assert mask[0]["node_id"] == "node:a"
    assert float(mask[0]["weight"]) > float(mask[1]["weight"])
    weight_sum = sum(float(row.get("weight", 0.0)) for row in mask)
    assert abs(weight_sum - 1.0) <= 1e-6


def test_presence_resource_need_model_uses_ema_and_logistic_thresholds() -> None:
    impact = {
        "id": "health_sentinel_cpu",
        "affected_by": {"resource": 0.9, "files": 0.4, "clicks": 0.2},
        "affects": {"world": 0.7},
        "resource_wallet": {"cpu": 2.0, "ram": 4.0, "disk": 3.0, "network": 1.5},
    }

    first = c_double_buffer_backend._presence_resource_need_model(
        presence_id="health_sentinel_cpu",
        impact=impact,
        queue_ratio=0.35,
        base_need=0.6,
    )

    assert first.get("alpha") == c_double_buffer_backend._RESOURCE_NEED_EMA_ALPHA
    assert first.get("priority", 0.0) > 0.0
    assert isinstance(first.get("needs"), dict)
    assert isinstance(first.get("util_ema"), dict)
    assert isinstance(first.get("util_raw"), dict)
    assert isinstance(first.get("thresholds"), dict)
    assert impact.get("_resource_util_ema") == first.get("util_ema")

    impact["resource_wallet"] = {
        "cpu": 20.0,
        "ram": 18.0,
        "disk": 15.0,
        "network": 14.0,
    }
    second = c_double_buffer_backend._presence_resource_need_model(
        presence_id="health_sentinel_cpu",
        impact=impact,
        queue_ratio=0.35,
        base_need=0.6,
    )

    first_cpu_ema = float((first.get("util_ema", {}) or {}).get("cpu", 0.0))
    second_cpu_ema = float((second.get("util_ema", {}) or {}).get("cpu", 0.0))
    second_cpu_raw = float((second.get("util_raw", {}) or {}).get("cpu", 0.0))
    assert second_cpu_raw < first_cpu_ema
    assert second_cpu_raw <= second_cpu_ema <= first_cpu_ema


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
            "source_profiles": [
                {
                    "presence_id": "witness_thread",
                    "source_node_id": "node:a",
                    "mask": {
                        "mode": "nearest-k",
                        "k": 2,
                        "nodes": [
                            {"node_id": "node:a", "weight": 0.7, "distance": 0.0},
                            {"node_id": "node:b", "weight": 0.3, "distance": 0.4},
                        ],
                    },
                    "influence": {
                        "mode": "anchor-mask",
                        "strength": 0.66,
                        "anchor": {"x": 0.3, "y": 0.3},
                    },
                    "need_scalar": 0.52,
                    "need_by_resource": {"cpu": 0.7, "ram": 0.3},
                    "mass": 1.4,
                }
            ],
            "presence_source_count": 1,
            "presence_model": {
                "mask": "nearest-k.v1",
                "need": "heuristic-resource-need.v1",
                "mass": "signal-wallet.v1",
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
    assert float(rows[2]["action_probabilities"]["deflect"]) <= 0.66
    assert float(rows[2]["message_probability"]) >= 0.27
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
    assert graph_runtime.get("presence_source_count") == 1
    assert graph_runtime.get("presence_model", {}).get("mask") == "nearest-k.v1"
    source_profiles = graph_runtime.get("source_profiles", [])
    assert isinstance(source_profiles, list)
    assert source_profiles[0].get("presence_id") == "witness_thread"
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


def test_graph_route_step_resource_signature_avoids_per_edge_term_dict_allocations(
    monkeypatch: Any,
) -> None:
    call_count = 0
    original = c_double_buffer_backend._route_terms_for_edge

    def _counted_route_terms_for_edge(**kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return original(**kwargs)

    monkeypatch.setattr(
        c_double_buffer_backend,
        "_route_terms_for_edge",
        _counted_route_terms_for_edge,
    )

    sources = [0, 0, 0]
    route = c_double_buffer_backend.compute_graph_route_step_native(
        graph_runtime={
            "node_count": 3,
            "edge_count": 2,
            "edge_src_index": [0, 0],
            "edge_dst_index": [1, 2],
            "edge_cost": [1.0, 1.0],
            "edge_health": [1.0, 1.0],
            "edge_affinity": [0.5, 0.5],
            "edge_saturation": [0.2, 0.2],
            "edge_latency_component": [1.0, 1.0],
            "edge_congestion_component": [0.4, 0.4],
            "edge_semantic_component": [0.5, 0.5],
            "edge_upkeep_penalty": [0.0, 0.0],
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
        particle_source_nodes=sources,
        particle_resource_signature=[{"cpu": 1.0} for _ in sources],
        step_seed=0,
    )

    assert isinstance(route, dict)
    assert route.get("resource_routing_mode") == "resource-signature"
    assert len(route.get("next_node_index", [])) == len(sources)
    assert call_count == len(sources)


def test_embed_seed_vector_returns_zeros_when_c_runtime_required(
    monkeypatch: Any,
) -> None:
    c_double_buffer_backend._clear_embed_seed_cache()
    monkeypatch.setattr(c_double_buffer_backend, "_get_c_embed_runtime", lambda: None)
    monkeypatch.setattr(
        c_double_buffer_backend,
        "_c_embed_runtime_required",
        lambda: True,
    )

    vec = c_double_buffer_backend._embed_seed_vector_24("strict-c-runtime")

    assert len(vec) == 24
    assert all(float(value) == 0.0 for value in vec)


def test_cpu_fallback_signal_detector_matches_openvino_warnings() -> None:
    assert (
        c_double_buffer_backend._is_cpu_fallback_signal(
            "OpenVINO NPU compile failed, falling back to OV CPU"
        )
        is True
    )
    assert (
        c_double_buffer_backend._is_cpu_fallback_signal(
            "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE while routing to CPU"
        )
        is True
    )
    assert (
        c_double_buffer_backend._is_cpu_fallback_signal(
            "OpenVINO NPU pipeline active with no fallback"
        )
        is False
    )


def test_embed_seed_vector_marks_cpu_fallback_and_fails_closed_when_required(
    monkeypatch: Any,
) -> None:
    class _FallbackRuntime:
        def embed_24(self, _: str) -> list[float]:
            return [0.1 for _ in range(24)]

        def last_error(self) -> str:
            return ""

        def cpu_fallback_detected(self) -> bool:
            return True

        def cpu_fallback_detail(self) -> str:
            return "OpenVINO NPU fallback to CPU detected"

    c_double_buffer_backend._clear_embed_seed_cache()
    monkeypatch.setattr(
        c_double_buffer_backend,
        "_get_c_embed_runtime",
        lambda: _FallbackRuntime(),
    )
    monkeypatch.setattr(
        c_double_buffer_backend,
        "_c_embed_runtime_required",
        lambda: True,
    )

    vec = c_double_buffer_backend._embed_seed_vector_24("strict-no-cpu-fallback")

    assert len(vec) == 24
    assert all(float(value) == 0.0 for value in vec)
    assert c_double_buffer_backend._EMBED_RUNTIME_CPU_FALLBACK is True
    assert "fallback" in c_double_buffer_backend._EMBED_RUNTIME_ERROR.lower()


def test_embed_seed_vector_fails_closed_when_c_not_required(
    monkeypatch: Any,
) -> None:
    c_double_buffer_backend._clear_embed_seed_cache()
    monkeypatch.setattr(c_double_buffer_backend, "_get_c_embed_runtime", lambda: None)
    monkeypatch.setattr(
        c_double_buffer_backend,
        "_c_embed_runtime_required",
        lambda: False,
    )

    vec = c_double_buffer_backend._embed_seed_vector_24("allow-python-fallback")

    assert len(vec) == 24
    assert all(float(value) == 0.0 for value in vec)


def test_embed_device_candidates_normalize_npu_gpu_auto() -> None:
    assert c_double_buffer_backend._embed_device_candidates("NPU") == ["NPU"]
    assert c_double_buffer_backend._embed_device_candidates("intel_npu") == ["NPU"]
    assert c_double_buffer_backend._embed_device_candidates("gpu") == ["GPU"]
    assert c_double_buffer_backend._embed_device_candidates("CUDA") == ["GPU"]
    assert c_double_buffer_backend._embed_device_candidates("") == ["NPU", "GPU"]
