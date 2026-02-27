from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from code.world_web import build_simulation_state


def test_catalog_stream_iter_rows_chunks_lists_and_reports_done() -> None:
    from code.world_web import server as server_module

    catalog = {
        "record": "eta-mu.catalog.v1",
        "items": [{"id": "item:1"}, {"id": "item:2"}, {"id": "item:3"}],
        "file_graph": {
            "file_nodes": [{"id": "file:1"}, {"id": "file:2"}],
            "edges": [{"id": "edge:1"}, {"id": "edge:2"}, {"id": "edge:3"}],
            "embed_layers": [{"id": "layer:1"}],
        },
        "crawler_graph": {
            "crawler_nodes": [{"id": "crawler:1"}],
            "edges": [],
        },
    }

    rows = list(server_module._catalog_stream_iter_rows(catalog, chunk_rows=2))
    assert rows
    meta_row = rows[0]
    assert meta_row.get("type") == "meta"
    meta_catalog = meta_row.get("catalog", {})
    assert isinstance(meta_catalog, dict)
    assert meta_catalog.get("items", {}).get("streamed") is True
    assert meta_catalog.get("items", {}).get("count") == 3
    assert (
        meta_catalog.get("file_graph", {}).get("file_nodes", {}).get("streamed") is True
    )

    item_rows = [
        row
        for row in rows
        if str(row.get("type", "")) == "rows" and str(row.get("section", "")) == "items"
    ]
    assert len(item_rows) == 2
    assert item_rows[0].get("offset") == 0
    assert len(item_rows[0].get("rows", [])) == 2
    assert item_rows[1].get("offset") == 2
    assert len(item_rows[1].get("rows", [])) == 1

    done_row = rows[-1]
    assert done_row.get("type") == "done"
    assert done_row.get("ok") is True
    sections = done_row.get("sections", {})
    assert isinstance(sections, dict)
    assert sections.get("items", {}).get("total") == 3
    assert sections.get("items", {}).get("chunks") == 2
    assert sections.get("file_edges", {}).get("total") == 3

def test_catalog_stream_chunk_rows_clamps_bounds() -> None:
    from code.world_web import server as server_module

    assert server_module._catalog_stream_chunk_rows("0") == 1
    assert server_module._catalog_stream_chunk_rows("17") == 17
    assert server_module._catalog_stream_chunk_rows("5000") == 2048

def test_reset_simulation_bootstrap_state_clears_layout_cache_and_rearms_boot() -> None:
    from code.world_web import simulation as simulation_module

    with simulation_module._SIMULATION_LAYOUT_CACHE_LOCK:
        previous_layout_cache = dict(simulation_module._SIMULATION_LAYOUT_CACHE)
        simulation_module._SIMULATION_LAYOUT_CACHE["key"] = "test-layout-key"
        simulation_module._SIMULATION_LAYOUT_CACHE["prepared_monotonic"] = 123.0
        simulation_module._SIMULATION_LAYOUT_CACHE["prepared_graph"] = {"ok": True}
        simulation_module._SIMULATION_LAYOUT_CACHE["embedding_points"] = [
            {"x": 0.1, "y": 0.2}
        ]

    previous_boot_applied = bool(simulation_module._SIMULATION_BOOT_RESET_APPLIED)
    with simulation_module._SIMULATION_BOOT_RESET_LOCK:
        simulation_module._SIMULATION_BOOT_RESET_APPLIED = True

    try:
        result = simulation_module.reset_simulation_bootstrap_state(
            clear_layout_cache=True,
            rearm_boot_reset=True,
        )
        assert result.get("ok") is True
        assert result.get("cleared_layout_cache") is True
        assert result.get("previous_layout_key") == "test-layout-key"
        assert int(result.get("previous_embedding_points", 0)) == 1
        assert result.get("rearmed_boot_reset") is True

        with simulation_module._SIMULATION_LAYOUT_CACHE_LOCK:
            assert simulation_module._SIMULATION_LAYOUT_CACHE["key"] == ""
            assert (
                simulation_module._SIMULATION_LAYOUT_CACHE["prepared_monotonic"] == 0.0
            )
            assert simulation_module._SIMULATION_LAYOUT_CACHE["prepared_graph"] is None
            assert simulation_module._SIMULATION_LAYOUT_CACHE["embedding_points"] == []
        with simulation_module._SIMULATION_BOOT_RESET_LOCK:
            assert simulation_module._SIMULATION_BOOT_RESET_APPLIED is False
    finally:
        with simulation_module._SIMULATION_LAYOUT_CACHE_LOCK:
            simulation_module._SIMULATION_LAYOUT_CACHE.clear()
            simulation_module._SIMULATION_LAYOUT_CACHE.update(previous_layout_cache)
        with simulation_module._SIMULATION_BOOT_RESET_LOCK:
            simulation_module._SIMULATION_BOOT_RESET_APPLIED = previous_boot_applied

def test_simulation_bootstrap_graph_report_summarizes_projection_and_layers() -> None:
    from code.world_web import server as server_module

    catalog = {
        "items": [
            {"rel_path": "src/a.py", "kind": "text", "name": "a.py"},
            {"rel_path": "src/b.py", "kind": "text", "name": "b.py"},
            {"rel_path": "docs/extra.md", "kind": "text", "name": "extra.md"},
        ],
        "file_graph": {
            "file_nodes": [
                {
                    "id": "file:a",
                    "node_id": "n:a",
                    "name": "a.py",
                    "kind": "text",
                    "source_rel_path": "src/a.py",
                },
                {
                    "id": "file:b",
                    "node_id": "n:b",
                    "name": "b.py",
                    "kind": "text",
                    "source_rel_path": "src/b.py",
                },
                {
                    "id": "file:c",
                    "node_id": "n:c",
                    "name": "c.py",
                    "kind": "text",
                    "source_rel_path": "src/c.py",
                },
                {
                    "id": "file:d",
                    "node_id": "n:d",
                    "name": "d.py",
                    "kind": "text",
                    "source_rel_path": "src/d.py",
                },
            ],
            "edges": [{"id": f"edge:{index}"} for index in range(20)],
            "embed_layers": [
                {
                    "id": "layer:active",
                    "label": "active",
                    "collection": "eta_mu",
                    "space_id": "space-a",
                    "model_name": "nomic",
                    "file_count": 6,
                    "reference_count": 11,
                    "active": True,
                },
                {
                    "id": "layer:inactive",
                    "label": "inactive",
                    "collection": "eta_mu",
                    "space_id": "space-b",
                    "model_name": "nomic",
                    "file_count": 2,
                    "reference_count": 3,
                    "active": False,
                },
            ],
        },
    }
    simulation = {
        "total": 16,
        "points": [{"x": 0.1, "y": 0.2} for _ in range(16)],
        "embedding_particles": [{"x": 0.0, "y": 0.0}],
        "presence_dynamics": {
            "field_particles": [{"id": "dm-1"}, {"id": "dm-2"}],
        },
        "file_graph": {
            "file_nodes": [
                {
                    "id": "file:a",
                    "node_id": "n:a",
                    "name": "a.py",
                    "kind": "text",
                    "source_rel_path": "src/a.py",
                },
                {
                    "id": "file:c",
                    "node_id": "n:c",
                    "name": "c.py",
                    "kind": "text",
                    "source_rel_path": "src/c.py",
                },
                {
                    "id": "file:projection:overflow1",
                    "node_id": "file:projection:overflow1",
                    "name": "Overflow categorizes -> Field",
                    "kind": "projection_overflow",
                    "projection_overflow": True,
                    "consolidated": True,
                    "consolidated_count": 2,
                    "projection_group_id": "projection-group:1",
                },
            ],
            "edges": [{"id": f"edge:{index}"} for index in range(12)],
            "embed_layers": catalog["file_graph"]["embed_layers"],
            "projection": {
                "active": True,
                "mode": "hub-overflow",
                "reason": "edge_budget",
                "before": {"file_nodes": 6, "edges": 20},
                "after": {"file_nodes": 8, "edges": 12},
                "collapsed_edges": 8,
                "overflow_nodes": 2,
                "overflow_edges": 2,
                "group_count": 3,
                "limits": {"edge_cap": 12},
                "groups": [
                    {
                        "id": "projection-group:1",
                        "kind": "categorizes",
                        "target": "field:anchor_registry",
                        "field": "f3",
                        "member_edge_count": 6,
                        "member_source_count": 2,
                        "member_target_count": 1,
                        "member_source_ids": ["file:b", "file:d"],
                        "surface_visible": False,
                        "reasons": {
                            "global_cap": 4,
                            "per_source_cap": 2,
                        },
                    }
                ],
            },
        },
    }

    report = server_module._simulation_bootstrap_graph_report(
        perspective="hybrid",
        catalog=catalog,
        simulation=simulation,
        projection={"record": "projection.v1", "perspective": "hybrid", "ts": 7},
        phase_ms={"catalog": 12.4, "simulation": 27.9},
        reset_summary={"ok": True},
        inbox_sync={"status": "completed"},
        cache_key="hybrid|demo|simulation",
    )

    assert report.get("ok") is True
    assert report.get("selection", {}).get("graph_surface") == "projected-hub-overflow"
    assert report.get("selection", {}).get("active_embed_layer_count") == 1
    selected_layers = report.get("selection", {}).get("selected_embed_layers", [])
    assert isinstance(selected_layers, list)
    assert selected_layers[0].get("id") == "layer:active"

    compression = report.get("compression", {})
    assert compression.get("before_edges") == 20
    assert compression.get("after_edges") == 12
    assert compression.get("collapsed_edges") == 8
    assert compression.get("edge_cap") == 12
    assert compression.get("group_count") == 3

    sim_counts = report.get("simulation_counts", {})
    assert sim_counts.get("total_points") == 16
    assert sim_counts.get("embedding_particles") == 1
    assert sim_counts.get("field_particles") == 2

    graph_diff = report.get("graph_diff", {})
    assert graph_diff.get("truth_file_node_count") == 4
    assert graph_diff.get("view_file_node_count") == 3
    assert graph_diff.get("truth_file_nodes_missing_from_view_count") == 2
    missing_rows = graph_diff.get("truth_file_nodes_missing_from_view", [])
    assert isinstance(missing_rows, list)
    assert missing_rows[0].get("reason") == "grouped_in_hidden_projection_bundle"
    assert graph_diff.get("view_projection_overflow_node_count") == 1
    assert graph_diff.get("ingested_items_missing_from_truth_graph_count") == 1
    assert graph_diff.get("compaction_mode") == "compacted_with_projection_overflow"

def test_simulation_bootstrap_graph_diff_flags_visible_bundle_without_overflow() -> (
    None
):
    from code.world_web import server as server_module

    catalog = {
        "items": [{"rel_path": "src/b.py", "kind": "text", "name": "b.py"}],
        "file_graph": {
            "file_nodes": [
                {
                    "id": "file:a",
                    "name": "a.py",
                    "kind": "text",
                    "source_rel_path": "src/a.py",
                },
                {
                    "id": "file:b",
                    "name": "b.py",
                    "kind": "text",
                    "source_rel_path": "src/b.py",
                },
            ]
        },
    }
    simulation = {
        "file_graph": {
            "file_nodes": [
                {
                    "id": "file:a",
                    "name": "a.py",
                    "kind": "text",
                    "source_rel_path": "src/a.py",
                }
            ],
            "projection": {
                "collapsed_edges": 3,
                "groups": [
                    {
                        "id": "projection-group:visible",
                        "kind": "categorizes",
                        "target": "field:anchor_registry",
                        "member_source_ids": ["file:b"],
                        "surface_visible": True,
                        "reasons": {"global_cap": 2},
                    }
                ],
            },
        }
    }

    graph_diff = server_module._simulation_bootstrap_graph_diff(
        catalog=catalog,
        simulation=simulation,
    )

    assert graph_diff.get("truth_file_nodes_missing_from_view_count") == 1
    missing = graph_diff.get("truth_file_nodes_missing_from_view", [])
    assert isinstance(missing, list)
    assert missing[0].get("reason") == "grouped_in_projection_bundle"
    assert graph_diff.get("view_projection_overflow_node_count") == 0
    assert graph_diff.get("projection_surface_visible_group_count") == 1
    assert graph_diff.get("projection_hidden_group_count") == 0
    assert graph_diff.get("compaction_mode") == "pruned_without_overflow_nodes"

def test_simulation_bootstrap_graph_diff_classifies_trimmed_and_identity_modes() -> (
    None
):
    from code.world_web import server as server_module

    catalog = {
        "file_graph": {
            "file_nodes": [
                {
                    "id": "file:a",
                    "name": "a.py",
                    "kind": "text",
                    "source_rel_path": "src/a.py",
                }
            ]
        }
    }

    trimmed_diff = server_module._simulation_bootstrap_graph_diff(
        catalog=catalog,
        simulation={
            "file_graph": {"file_nodes": [], "projection": {"collapsed_edges": 0}}
        },
    )
    assert trimmed_diff.get("compaction_mode") == "trimmed_before_projection"
    assert trimmed_diff.get("truth_file_nodes_missing_from_view_count") == 1

    identity_diff = server_module._simulation_bootstrap_graph_diff(
        catalog=catalog,
        simulation={
            "file_graph": {
                "file_nodes": [
                    {
                        "id": "file:a",
                        "name": "a.py",
                        "kind": "text",
                        "source_rel_path": "src/a.py",
                    }
                ],
                "projection": {"collapsed_edges": 0},
            }
        },
    )
    assert identity_diff.get("compaction_mode") == "identity_or_within_limits"
    assert identity_diff.get("truth_file_nodes_missing_from_view_count") == 0

def test_run_simulation_bootstrap_reports_phase_progress(monkeypatch: Any) -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:
        catalog_kwargs: dict[str, Any] = {}

        class _FakeHandler:
            def __init__(self, root: Path) -> None:
                self.part_root = root
                self.vault_root = root

            def _runtime_catalog(self, **kwargs: Any) -> tuple[Any, ...]:
                catalog_kwargs.update(kwargs)
                return (
                    {"file_graph": {"file_nodes": [], "edges": [], "embed_layers": []}},
                    {},
                    {},
                    {},
                    {},
                )

            def _runtime_simulation(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
                return (
                    {
                        "total": 1,
                        "points": [{"id": "p1"}],
                        "embedding_particles": [],
                        "presence_dynamics": {"field_particles": []},
                        "file_graph": {
                            "file_nodes": [],
                            "edges": [],
                            "embed_layers": [],
                            "projection": {
                                "active": False,
                                "before": {"file_nodes": 0, "edges": 0},
                                "after": {"file_nodes": 0, "edges": 0},
                                "limits": {"edge_cap": 10},
                            },
                        },
                    },
                    {"record": "projection.v1", "perspective": "hybrid", "ts": 1},
                )

        monkeypatch.setattr(
            server_module.simulation_module,
            "reset_simulation_bootstrap_state",
            lambda **kwargs: {"ok": True},
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_invalidate",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_key",
            lambda **kwargs: "bootstrap|cache|key",
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_store",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_disk_cache_store",
            lambda *args, **kwargs: None,
        )

        phases: list[tuple[str, dict[str, Any] | None]] = []
        handler = _FakeHandler(Path(td))
        payload, status = server_module.WorldHandler._run_simulation_bootstrap(
            cast(Any, handler),
            perspective="hybrid",
            sync_inbox=False,
            include_simulation_payload=False,
            phase_callback=lambda phase, detail: phases.append((phase, detail)),
        )

        assert status == server_module.HTTPStatus.OK
        assert payload.get("ok") is True
        phase_order: list[str] = []
        for row in phases:
            phase_name = str(row[0])
            if not phase_order or phase_order[-1] != phase_name:
                phase_order.append(phase_name)
        assert phase_order == [
            "reset",
            "cache_invalidate",
            "inbox_sync",
            "catalog",
            "simulation",
            "cache_store",
            "report",
        ]
        catalog_phase = [row for row in phases if row[0] == "catalog"]
        assert catalog_phase
        assert catalog_phase[0][1] == {
            "strict_collect": True,
            "allow_inline_collect": False,
        }
        assert catalog_kwargs.get("allow_inline_collect") is False
        assert catalog_kwargs.get("strict_collect") is True

def test_run_simulation_bootstrap_timeout_sets_failed_phase(monkeypatch: Any) -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:

        class _FakeHandler:
            def __init__(self, root: Path) -> None:
                self.part_root = root
                self.vault_root = root

            def _runtime_catalog(self, **kwargs: Any) -> tuple[Any, ...]:
                time.sleep(0.01)
                return ({"file_graph": {}}, {}, {}, {}, {})

            def _runtime_simulation(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
                return ({"total": 0, "points": []}, {})

        monkeypatch.setattr(server_module, "_SIMULATION_BOOTSTRAP_MAX_SECONDS", 0.001)
        monkeypatch.setattr(
            server_module.simulation_module,
            "reset_simulation_bootstrap_state",
            lambda **kwargs: {"ok": True},
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_invalidate",
            lambda **kwargs: None,
        )

        phases: list[tuple[str, dict[str, Any] | None]] = []
        handler = _FakeHandler(Path(td))
        payload, status = server_module.WorldHandler._run_simulation_bootstrap(
            cast(Any, handler),
            perspective="hybrid",
            sync_inbox=False,
            include_simulation_payload=False,
            phase_callback=lambda phase, detail: phases.append((phase, detail)),
        )

        assert status == server_module.HTTPStatus.INTERNAL_SERVER_ERROR
        assert payload.get("ok") is False
        assert payload.get("error") == "simulation_bootstrap_failed:TimeoutError"
        assert payload.get("failed_phase") == "catalog"
        assert phases[-1][0] == "failed"

def test_run_simulation_bootstrap_falls_back_to_inline_catalog(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:
        catalog_calls: list[dict[str, Any]] = []

        class _FakeHandler:
            def __init__(self, root: Path) -> None:
                self.part_root = root
                self.vault_root = root

            def _runtime_catalog(self, **kwargs: Any) -> tuple[Any, ...]:
                catalog_calls.append(dict(kwargs))
                if bool(kwargs.get("strict_collect", False)):
                    raise RuntimeError("catalog_subprocess_exit:1")
                return (
                    {"file_graph": {"file_nodes": [], "edges": [], "embed_layers": []}},
                    {},
                    {},
                    {},
                    {},
                )

            def _runtime_simulation(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
                return (
                    {
                        "total": 1,
                        "points": [{"id": "p1"}],
                        "embedding_particles": [],
                        "presence_dynamics": {"field_particles": []},
                        "file_graph": {
                            "file_nodes": [],
                            "edges": [],
                            "embed_layers": [],
                            "projection": {
                                "active": False,
                                "before": {"file_nodes": 0, "edges": 0},
                                "after": {"file_nodes": 0, "edges": 0},
                                "limits": {"edge_cap": 10},
                            },
                        },
                    },
                    {"record": "projection.v1", "perspective": "hybrid", "ts": 1},
                )

        monkeypatch.setattr(
            server_module.simulation_module,
            "reset_simulation_bootstrap_state",
            lambda **kwargs: {"ok": True},
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_invalidate",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_key",
            lambda **kwargs: "bootstrap|cache|key",
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_store",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_disk_cache_store",
            lambda *args, **kwargs: None,
        )

        phases: list[tuple[str, dict[str, Any] | None]] = []
        handler = _FakeHandler(Path(td))
        payload, status = server_module.WorldHandler._run_simulation_bootstrap(
            cast(Any, handler),
            perspective="hybrid",
            sync_inbox=False,
            include_simulation_payload=False,
            phase_callback=lambda phase, detail: phases.append((phase, detail)),
        )

        assert status == server_module.HTTPStatus.OK
        assert payload.get("ok") is True
        assert len(catalog_calls) == 2
        assert catalog_calls[0].get("allow_inline_collect") is False
        assert catalog_calls[0].get("strict_collect") is True
        assert catalog_calls[1].get("allow_inline_collect") is True
        assert catalog_calls[1].get("strict_collect") is False

        phase_order: list[str] = []
        for row in phases:
            phase_name = str(row[0])
            if not phase_order or phase_order[-1] != phase_name:
                phase_order.append(phase_name)
        assert "catalog_fallback_inline" in phase_order

        fallback_rows = [
            row[1]
            for row in phases
            if row[0] == "catalog_fallback_inline" and isinstance(row[1], dict)
        ]
        assert fallback_rows
        first_fallback = fallback_rows[0]
        assert first_fallback.get("fallback_from") == "catalog"
        assert str(first_fallback.get("fallback_reason", "")).startswith(
            "catalog_subprocess_exit:1"
        )

def test_simulation_bootstrap_job_mark_phase_preserves_phase_started_at() -> None:
    from code.world_web import server as server_module

    previous_snapshot = server_module._simulation_bootstrap_job_snapshot()
    started, snapshot = server_module._simulation_bootstrap_job_start(
        request_payload={"perspective": "hybrid", "sync_inbox": False}
    )
    assert started is True
    job_id = str(snapshot.get("job_id", "") or "")
    assert job_id

    try:
        server_module._simulation_bootstrap_job_mark_phase(
            job_id=job_id,
            phase="catalog",
            detail={"heartbeat_count": 1},
        )
        first_snapshot = server_module._simulation_bootstrap_job_snapshot()
        phase_started_at = str(first_snapshot.get("phase_started_at", "") or "")
        first_updated_at = str(first_snapshot.get("updated_at", "") or "")
        assert phase_started_at
        time.sleep(0.01)

        server_module._simulation_bootstrap_job_mark_phase(
            job_id=job_id,
            phase="catalog",
            detail={"heartbeat_count": 2},
        )
        second_snapshot = server_module._simulation_bootstrap_job_snapshot()
        assert second_snapshot.get("phase") == "catalog"
        assert second_snapshot.get("phase_started_at") == phase_started_at
        assert str(second_snapshot.get("updated_at", "") or "") >= first_updated_at
        assert second_snapshot.get("phase_detail") == {"heartbeat_count": 2}
    finally:
        with server_module._SIMULATION_BOOTSTRAP_JOB_LOCK:
            server_module._SIMULATION_BOOTSTRAP_JOB.clear()
            server_module._SIMULATION_BOOTSTRAP_JOB.update(previous_snapshot)

def test_run_simulation_bootstrap_emits_catalog_heartbeats(monkeypatch: Any) -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:

        class _FakeHandler:
            def __init__(self, root: Path) -> None:
                self.part_root = root
                self.vault_root = root

            def _runtime_catalog(self, **kwargs: Any) -> tuple[Any, ...]:
                time.sleep(0.03)
                return (
                    {"file_graph": {"file_nodes": [], "edges": [], "embed_layers": []}},
                    {},
                    {},
                    {},
                    {},
                )

            def _runtime_simulation(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
                return (
                    {
                        "total": 1,
                        "points": [{"id": "p1"}],
                        "embedding_particles": [],
                        "presence_dynamics": {"field_particles": []},
                        "file_graph": {
                            "file_nodes": [],
                            "edges": [],
                            "embed_layers": [],
                            "projection": {
                                "active": False,
                                "before": {"file_nodes": 0, "edges": 0},
                                "after": {"file_nodes": 0, "edges": 0},
                                "limits": {"edge_cap": 10},
                            },
                        },
                    },
                    {"record": "projection.v1", "perspective": "hybrid", "ts": 1},
                )

        monkeypatch.setattr(
            server_module.simulation_module,
            "reset_simulation_bootstrap_state",
            lambda **kwargs: {"ok": True},
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_invalidate",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_key",
            lambda **kwargs: "bootstrap|cache|key",
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_cache_store",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_disk_cache_store",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            server_module,
            "_SIMULATION_BOOTSTRAP_HEARTBEAT_SECONDS",
            0.005,
        )

        phases: list[tuple[str, dict[str, Any] | None]] = []
        handler = _FakeHandler(Path(td))
        payload, status = server_module.WorldHandler._run_simulation_bootstrap(
            cast(Any, handler),
            perspective="hybrid",
            sync_inbox=False,
            include_simulation_payload=False,
            phase_callback=lambda phase, detail: phases.append((phase, detail)),
        )

        assert status == server_module.HTTPStatus.OK
        assert payload.get("ok") is True
        catalog_details = [
            row[1] for row in phases if row[0] == "catalog" and isinstance(row[1], dict)
        ]
        assert catalog_details
        assert any(
            max(0, int(float(str(detail.get("heartbeat_count", 0) or "0")))) >= 1
            for detail in catalog_details
        )
        assert any("phase_elapsed_ms" in detail for detail in catalog_details)

def test_compact_simulation_payload_keeps_presence_particles_and_drops_heavy_graphs() -> (
    None
):
    from code.world_web import server as server_module

    payload = {
        "timestamp": "2026-02-20T22:20:00Z",
        "field_particles": [{"id": "root-dm-1"}],
        "presence_dynamics": {
            "field_particles": [{"id": "dyn-dm-1", "presence_id": "witness_thread"}],
        },
        "nexus_graph": {"nodes": [1]},
        "logical_graph": {"nodes": [1]},
        "pain_field": {"nodes": [1]},
        "heat_values": {"regions": [1]},
        "file_graph": {"file_nodes": [1]},
        "crawler_graph": {"crawler_nodes": [1]},
        "field_registry": {"fields": {"demand": {"samples": [1]}}},
    }

    compact = server_module._simulation_http_compact_simulation_payload(payload)

    assert "nexus_graph" not in compact
    assert "logical_graph" not in compact
    assert "pain_field" not in compact
    assert "heat_values" not in compact
    assert "file_graph" not in compact
    assert "crawler_graph" not in compact
    assert "field_registry" not in compact
    assert "field_particles" not in compact
    assert compact.get("presence_dynamics", {}).get("field_particles", []) == [
        {"id": "dyn-dm-1", "presence_id": "witness_thread"}
    ]

def test_compact_simulation_payload_caps_points_and_field_particles(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    monkeypatch.setattr(server_module, "_SIMULATION_HTTP_COMPACT_MAX_POINTS", 5)
    monkeypatch.setattr(
        server_module,
        "_SIMULATION_HTTP_COMPACT_MAX_FIELD_PARTICLES",
        3,
    )
    payload = {
        "points": [{"id": f"pt:{index}"} for index in range(12)],
        "presence_dynamics": {
            "field_particles": [{"id": f"dm:{index}"} for index in range(8)],
        },
        "field_particles": [{"id": "legacy:root"}],
    }

    compact = server_module._simulation_http_compact_simulation_payload(payload)

    assert len(compact.get("points", [])) == 5
    assert int(compact.get("points_total", 0)) == 12
    assert bool(compact.get("points_compacted", False)) is True

    dynamics = compact.get("presence_dynamics", {})
    assert len(dynamics.get("field_particles", [])) == 3
    assert int(dynamics.get("field_particles_total", 0)) == 8
    assert bool(dynamics.get("field_particles_compacted", False)) is True
    assert "field_particles" not in compact

def test_simulation_http_compact_cache_round_trip_and_invalidate() -> None:
    from code.world_web import server as server_module

    cache_key = "hybrid|demo-cache-key|compact"
    payload = b'{"ok":true,"record":"eta-mu.simulation.v1"}'

    server_module._simulation_http_compact_cache_store(cache_key, payload)
    cached = server_module._simulation_http_compact_cached_body(
        cache_key=cache_key,
        max_age_seconds=30.0,
    )
    assert cached == payload

    server_module._simulation_http_cache_invalidate()
    cached_after_invalidate = server_module._simulation_http_compact_cached_body(
        cache_key=cache_key,
        max_age_seconds=30.0,
    )
    assert cached_after_invalidate is None

def test_simulation_http_compact_cache_lookup_by_perspective() -> None:
    from code.world_web import server as server_module

    cache_key = "hybrid|demo-cache-key|compact"
    payload = b'{"ok":true,"record":"eta-mu.simulation.v1"}'

    server_module._simulation_http_compact_cache_store(cache_key, payload)
    by_perspective = server_module._simulation_http_compact_cached_body(
        perspective="hybrid",
        max_age_seconds=30.0,
    )
    assert by_perspective == payload

    wrong_perspective = server_module._simulation_http_compact_cached_body(
        perspective="narrative",
        max_age_seconds=30.0,
    )
    assert wrong_perspective is None

    server_module._simulation_http_cache_invalidate()

def test_runtime_catalog_http_cache_round_trip_and_invalidate() -> None:
    from code.world_web import server as server_module

    payload = b'{"ok":true,"record":"eta-mu.catalog.v1"}'
    server_module._runtime_catalog_http_cache_store(
        perspective="hybrid",
        body=payload,
    )

    cached = server_module._runtime_catalog_http_cached_body(
        perspective="hybrid",
        max_age_seconds=10.0,
    )
    assert cached == payload

    server_module._simulation_http_cache_invalidate()
    cached_after_invalidate = server_module._runtime_catalog_http_cached_body(
        perspective="hybrid",
        max_age_seconds=10.0,
    )
    assert cached_after_invalidate is None

def test_runtime_catalog_http_cache_is_perspective_scoped() -> None:
    from code.world_web import server as server_module

    payload_hybrid = b'{"ok":true,"record":"eta-mu.catalog.hybrid.v1"}'
    payload_file = b'{"ok":true,"record":"eta-mu.catalog.file.v1"}'
    server_module._runtime_catalog_http_cache_store(
        perspective="hybrid",
        body=payload_hybrid,
    )
    server_module._runtime_catalog_http_cache_store(
        perspective="file_focus",
        body=payload_file,
    )

    assert (
        server_module._runtime_catalog_http_cached_body(
            perspective="hybrid",
            max_age_seconds=10.0,
        )
        == payload_hybrid
    )
    assert (
        server_module._runtime_catalog_http_cached_body(
            perspective="file_focus",
            max_age_seconds=10.0,
        )
        == payload_file
    )

    server_module._runtime_catalog_http_cache_invalidate()

def test_simulation_http_compact_stale_fallback_prefers_memory_cache(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    calls: dict[str, int] = {"disk": 0}

    monkeypatch.setattr(
        server_module,
        "_simulation_http_cached_body",
        lambda **kwargs: b'{"ok":true,"source":"memory"}',
    )

    def _disk_loader(*args: Any, **kwargs: Any) -> None:
        calls["disk"] += 1
        return None

    monkeypatch.setattr(server_module, "_simulation_http_disk_cache_load", _disk_loader)

    body, source = server_module._simulation_http_compact_stale_fallback_body(
        part_root=Path("."),
        perspective="hybrid",
        max_age_seconds=5.0,
    )

    assert body == b'{"ok":true,"source":"memory"}'
    assert source == "stale-cache"
    assert calls["disk"] == 0

def test_simulation_http_compact_stale_fallback_uses_disk_when_memory_miss(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    stored_rows: list[tuple[str, bytes]] = []

    monkeypatch.setattr(
        server_module, "_simulation_http_cached_body", lambda **kwargs: None
    )
    monkeypatch.setattr(
        server_module,
        "_simulation_http_disk_cache_load",
        lambda *args, **kwargs: b'{"ok":true,"source":"disk"}',
    )
    monkeypatch.setattr(
        server_module,
        "_simulation_http_cache_store",
        lambda cache_key, body: stored_rows.append((str(cache_key), bytes(body))),
    )

    body, source = server_module._simulation_http_compact_stale_fallback_body(
        part_root=Path("."),
        perspective="hybrid",
        max_age_seconds=5.0,
    )

    assert body == b'{"ok":true,"source":"disk"}'
    assert source == "disk-cache"
    assert stored_rows == [
        ("hybrid|disk-compact-fallback|simulation", b'{"ok":true,"source":"disk"}')
    ]

def test_simulation_state_includes_world_summary_slot() -> None:
    simulation = build_simulation_state({"items": [], "counts": {}})
    assert isinstance(simulation.get("world"), dict)

def test_simulation_state_can_skip_unified_graph_payload_for_compact_http_mode() -> (
    None
):
    simulation = build_simulation_state(
        {"items": [], "counts": {}},
        include_unified_graph=False,
    )
    assert simulation.get("nexus_graph") == {}
    assert simulation.get("field_registry") == {}
