from __future__ import annotations

from code.world_web.simulation_graph_contract_utils import (
    build_truth_graph_contract,
    build_view_graph_contract,
    file_id_for_path,
    normalize_path_for_file_id,
)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _clamp01(value: object) -> float:
    numeric = _safe_float(value, 0.0)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def test_file_path_normalization_and_hashing_are_stable() -> None:
    normalized = normalize_path_for_file_id(" ./docs/../docs/guide.md ")
    assert normalized == "docs/guide.md"
    assert file_id_for_path("docs/guide.md") == file_id_for_path("./docs/guide.md")


def test_graph_contract_builders_report_projection_metadata() -> None:
    file_graph = {
        "nodes": [
            {"id": "node:file", "node_type": "file"},
            {
                "id": "node:bundle",
                "node_type": "file",
                "projection_overflow": True,
                "kind": "projection_overflow",
            },
        ],
        "edges": [
            {"id": "edge:1", "projection_overflow": True},
        ],
        "projection": {
            "mode": "compact",
            "active": True,
            "reason": "unit-test",
            "policy": {
                "compaction_drive": 0.8,
                "cpu_pressure": 0.6,
                "view_edge_pressure": 0.7,
                "cpu_utilization": 72.0,
                "presence_id": "health_sentinel_cpu",
                "edge_threshold_base": 340,
                "edge_threshold_effective": 280,
                "edge_cap_base": 640,
                "edge_cap_effective": 512,
            },
            "groups": [
                {
                    "id": "group:1",
                    "kind": "bundle",
                    "field": "receipt_river",
                    "target": "node:file",
                    "member_edge_count": 3,
                    "member_source_count": 2,
                    "member_target_count": 1,
                    "member_edge_digest": "abc",
                    "surface_visible": True,
                    "member_edge_ids": ["edge:a", "edge:b"],
                }
            ],
        },
    }

    truth = build_truth_graph_contract(file_graph)
    view = build_view_graph_contract(
        file_graph,
        safe_float=_safe_float,
        safe_int=_safe_int,
        clamp01=_clamp01,
        projection_edge_threshold=340,
    )

    assert truth["record"] == "eta-mu.truth-graph.v1"
    assert truth["semantics"]["projection_bundle_node_count"] == 1
    assert view["record"] == "eta-mu.view-graph.v1"
    assert view["projection"]["bundle_ledger_count"] == 1
    assert view["projection_pi"]["kind"] == "edge-bundle"
