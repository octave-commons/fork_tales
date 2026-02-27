from __future__ import annotations

import code.world_web as world_web_module

from code.world_web import build_simulation_state


def test_simulation_state_unifies_crawler_nodes_into_nexus_graph() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {},
            "nodes": [
                {
                    "id": "field:gates_of_truth",
                    "node_type": "field",
                    "node_id": "field:gates_of_truth",
                    "x": 0.2,
                    "y": 0.2,
                    "hue": 52,
                    "field": "f2",
                    "label": "gates_of_truth",
                },
                {
                    "id": "file:1",
                    "node_type": "file",
                    "x": 0.4,
                    "y": 0.4,
                    "hue": 200,
                    "importance": 0.5,
                    "source_rel_path": "notes/a.md",
                    "dominant_field": "f2",
                },
            ],
            "field_nodes": [
                {
                    "id": "field:gates_of_truth",
                    "node_type": "field",
                    "node_id": "field:gates_of_truth",
                    "x": 0.2,
                    "y": 0.2,
                    "hue": 52,
                    "field": "f2",
                    "label": "gates_of_truth",
                }
            ],
            "file_nodes": [
                {
                    "id": "file:1",
                    "node_type": "file",
                    "x": 0.4,
                    "y": 0.4,
                    "hue": 200,
                    "importance": 0.5,
                    "source_rel_path": "notes/a.md",
                    "dominant_field": "f2",
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 1,
                "edge_count": 0,
                "kind_counts": {},
                "field_counts": {"f2": 1},
                "knowledge_entries": 0,
            },
        },
        "crawler_graph": {
            "record": "ημ.crawler-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "source": {
                "endpoint": "http://127.0.0.1:8793/api/weaver/graph",
                "service": "web-graph-weaver",
            },
            "status": {"alive": 1, "queue_size": 0},
            "nodes": [],
            "field_nodes": [],
            "crawler_nodes": [
                {
                    "id": "crawler:abc",
                    "node_id": "url:https://example.org",
                    "node_type": "crawler",
                    "crawler_kind": "url",
                    "label": "https://example.org",
                    "x": 0.68,
                    "y": 0.32,
                    "hue": 200,
                    "importance": 0.8,
                    "url": "https://example.org",
                    "dominant_field": "f2",
                }
            ],
            "edges": [
                {
                    "id": "edge:field-to-crawler",
                    "source": "crawler-field:gates_of_truth",
                    "target": "crawler:abc",
                    "kind": "hyperlink",
                    "weight": 0.4,
                }
            ],
            "stats": {
                "field_count": 1,
                "crawler_count": 1,
                "edge_count": 1,
                "kind_counts": {"url": 1},
                "field_counts": {"f2": 1},
                "nodes_total": 1,
                "edges_total": 1,
                "url_nodes_total": 1,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    file_graph = simulation.get("file_graph", {})
    graph_nodes = file_graph.get("nodes", []) if isinstance(file_graph, dict) else []
    graph_edges = file_graph.get("edges", []) if isinstance(file_graph, dict) else []
    crawler_rows = (
        file_graph.get("crawler_nodes", []) if isinstance(file_graph, dict) else []
    )

    assert any(str(node.get("id", "")) == "crawler:abc" for node in graph_nodes)
    assert any(str(node.get("id", "")) == "crawler:abc" for node in crawler_rows)
    crawler_node = next(
        node for node in crawler_rows if str(node.get("id", "")) == "crawler:abc"
    )
    assert crawler_node.get("resource_kind") == "link"
    assert crawler_node.get("modality") == "web"
    assert any(
        str(edge.get("source", "")) == "field:gates_of_truth"
        and str(edge.get("target", "")) == "crawler:abc"
        for edge in graph_edges
    )

def test_simulation_state_includes_canonical_nexus_graph_and_field_registry() -> None:
    """Test that simulation state includes the unified canonical model types.

    See specs/drafts/part64-deep-research-09-unified-nexus-graph.md
    See specs/drafts/part64-deep-research-10-shared-fields-daimoi-dynamics.md
    """
    source_path = "docs/canonical_test.md"
    file_id = world_web_module._file_id_for_path(source_path)
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-20T00:00:00+00:00",
            "nodes": [],
            "field_nodes": [
                {
                    "id": "field:logos",
                    "node_id": "field:logos",
                    "node_type": "field",
                    "field": "logos",
                    "label": "Logos",
                    "x": 0.5,
                    "y": 0.5,
                    "hue": 200,
                }
            ],
            "tag_nodes": [
                {
                    "id": "tag:test",
                    "node_id": "tag:test",
                    "node_type": "tag",
                    "tag": "test",
                    "label": "Test",
                    "x": 0.4,
                    "y": 0.6,
                    "hue": 180,
                }
            ],
            "file_nodes": [
                {
                    "id": "file:canonical_test",
                    "node_id": "file:canonical_test",
                    "node_type": "file",
                    "name": "canonical_test.md",
                    "label": "Canonical Test",
                    "x": 0.31,
                    "y": 0.42,
                    "hue": 212,
                    "importance": 0.7,
                    "source_rel_path": source_path,
                }
            ],
            "edges": [
                {
                    "id": "edge:test:file:field",
                    "source": "file:canonical_test",
                    "target": "field:logos",
                    "kind": "belongs_to",
                    "weight": 0.8,
                }
            ],
            "stats": {
                "field_count": 1,
                "file_count": 1,
                "edge_count": 1,
            },
        },
        "crawler_graph": {
            "record": "ημ.crawler-graph.v1",
            "generated_at": "2026-02-20T00:00:00+00:00",
            "crawler_nodes": [
                {
                    "id": "crawler:example",
                    "node_type": "crawler",
                    "crawler_kind": "url",
                    "label": "Example",
                    "url": "https://example.com",
                    "x": 0.7,
                    "y": 0.3,
                    "hue": 150,
                }
            ],
            "edges": [],
            "stats": {},
        },
        "truth_state": {
            "record": "ημ.truth-state.v1",
            "generated_at": "2026-02-20T00:00:00+00:00",
            "claim": {
                "id": "claim.canonical",
                "text": "Canonical model is unified",
                "status": "proved",
                "kappa": 0.9,
            },
            "claims": [],
        },
    }

    simulation = build_simulation_state(catalog)

    # Check canonical nexus_graph
    nexus_graph = simulation.get("nexus_graph", {})
    assert nexus_graph.get("record") == "ημ.nexus-graph.v1"
    assert nexus_graph.get("schema_version") == "nexus.graph.v1"

    # Check nodes
    nodes = nexus_graph.get("nodes", [])
    assert isinstance(nodes, list)
    assert len(nodes) >= 3  # At least field, tag, file

    # Check node roles
    roles = {n.get("role") for n in nodes}
    assert "field" in roles
    assert "file" in roles
    assert "tag" in roles

    # Check node structure
    file_node = next((n for n in nodes if n.get("role") == "file"), None)
    assert file_node is not None
    assert file_node.get("id") == "file:canonical_test"
    assert file_node.get("label") == "Canonical Test"
    assert file_node.get("provenance", {}).get("path") == source_path

    # Check edges
    edges = nexus_graph.get("edges", [])
    assert isinstance(edges, list)
    assert len(edges) >= 1

    # Check joins
    joins = nexus_graph.get("joins", {})
    assert isinstance(joins.get("by_role", {}), dict)
    assert isinstance(joins.get("by_path", {}), dict)
    assert source_path in joins.get("by_path", {})

    # Check stats
    stats = nexus_graph.get("stats", {})
    assert stats.get("node_count") >= 3
    assert stats.get("edge_count") >= 1
    assert isinstance(stats.get("role_counts", {}), dict)
    assert stats.get("role_counts", {}).get("file", 0) >= 1

    # Check canonical field_registry
    field_registry = simulation.get("field_registry", {})
    assert field_registry.get("record") == "ημ.field-registry.v1"
    assert field_registry.get("bounded") is True
    assert field_registry.get("field_count") == 4  # demand, flow, entropy, graph

    # Check fields exist
    fields = field_registry.get("fields", {})
    assert "demand" in fields
    assert "flow" in fields
    assert "entropy" in fields
    assert "graph" in fields

    # Check field structure
    for field_name, field in fields.items():
        assert field.get("kind") == field_name
        assert field.get("record") == "ημ.shared-field.v1"
        assert isinstance(field.get("samples", []), list)
        assert isinstance(field.get("stats", {}), dict)

    # Check weights
    weights = field_registry.get("weights", {})
    assert weights.get("demand", 0) > 0
    assert weights.get("flow", 0) >= 0
    assert weights.get("entropy", 0) >= 0
    assert weights.get("graph", 0) > 0

    # Verify backward compatibility: legacy graph payloads still exist
    assert "file_graph" in simulation
    assert "crawler_graph" in simulation
    assert "logical_graph" in simulation
    assert simulation["file_graph"].get("record") == "ημ.file-graph.v1"

def test_canonical_nexus_node_builder_maps_legacy_types() -> None:
    """Test that _build_canonical_nexus_node correctly maps legacy node types."""
    # Test file node
    file_legacy = {
        "id": "test-file",
        "node_type": "file",
        "label": "Test File",
        "x": 0.5,
        "y": 0.3,
        "hue": 200,
        "source_rel_path": "test.md",
    }
    file_canonical = world_web_module._build_canonical_nexus_node(
        file_legacy, origin_graph="test"
    )
    assert file_canonical.get("role") == "file"
    assert file_canonical.get("label") == "Test File"
    assert file_canonical.get("provenance", {}).get("path") == "test.md"

    # Test crawler node
    crawler_legacy = {
        "id": "test-crawler",
        "node_type": "crawler",
        "crawler_kind": "url",
        "resource_kind": "audio",
        "modality": "audio",
        "label": "Test URL",
        "x": 0.7,
        "y": 0.4,
        "hue": 150,
        "url": "https://example.com",
    }
    crawler_canonical = world_web_module._build_canonical_nexus_node(
        crawler_legacy, origin_graph="crawler_graph"
    )
    assert crawler_canonical.get("role") == "crawler"
    assert crawler_canonical.get("extension", {}).get("url") == "https://example.com"
    assert crawler_canonical.get("extension", {}).get("resource_kind") == "audio"
    assert crawler_canonical.get("extension", {}).get("modality") == "audio"

    # Test field node
    field_legacy = {
        "id": "field:test",
        "node_type": "field",
        "field": "test",
        "label": "Test Field",
        "x": 0.2,
        "y": 0.8,
        "hue": 180,
    }
    field_canonical = world_web_module._build_canonical_nexus_node(
        field_legacy, origin_graph="file_graph"
    )
    assert field_canonical.get("role") == "field"

def test_canonical_field_registry_is_bounded() -> None:
    """Test that field registry has bounded field count (no per-presence fields)."""
    from code.world_web.constants import FIELD_KINDS, MAX_FIELD_COUNT

    # Field count must be bounded
    assert len(FIELD_KINDS) == 4
    assert MAX_FIELD_COUNT == 4
    assert set(FIELD_KINDS) == {"demand", "flow", "entropy", "graph"}

    # Build field registry and verify bounded
    field_registry = world_web_module._build_field_registry({}, None)

    assert field_registry.get("bounded") is True
    assert field_registry.get("field_count") == 4
    assert len(field_registry.get("fields", {})) == 4
