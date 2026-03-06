from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .lith_nexus_index import collect_lith_nexus_index
from .simulation_nexus import _build_canonical_nexus_graph


def build_lith_nexus_snapshot(
    repo_root: Path,
    *,
    include_text: bool = False,
    file_graph: dict[str, Any] | None = None,
    crawler_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a unified nexus snapshot from all graph sources.

    The nexus_graph is the single source of truth. All other graphs
    (file_graph, crawler_graph, logical_graph) are projections/views
    of this unified graph.

    Args:
        repo_root: Repository root path
        include_text: Whether to include full text content in index
        file_graph: Optional pre-built file_graph (if None, uses empty)
        crawler_graph: Optional pre-built crawler_graph (if None, uses empty)

    Returns:
        Snapshot with unified nexus_graph plus logical_graph projection

    NOTE: For the full unified graph with file_graph and crawler_graph nodes,
    use the /api/simulation endpoint from the running world_web server.
    This offline snapshot only includes lith_nexus (.lith file) nodes.

    The MCP backend prefers the runtime endpoint when available.
    """
    index = collect_lith_nexus_index(repo_root, include_text=include_text)

    # Build logical_graph from lith_nexus index (forms, contracts, protocols)
    logical_graph = _build_logical_graph_from_index(index)

    # The unified nexus_graph combines ALL sources:
    # - file_graph: files from the repo
    # - crawler_graph: web/URL nodes from crawling
    # - logical_graph: lith forms, contracts, protocols
    nexus_graph = _build_canonical_nexus_graph(
        file_graph=file_graph or {},
        crawler_graph=crawler_graph or {},
        logical_graph=logical_graph,
        include_crawler=True,
        include_logical=True,
    )

    return {
        "record": "eta-mu.lith-nexus.snapshot.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root.resolve()),
        "index": index,
        "logical_graph": logical_graph,
        "nexus_graph": nexus_graph,
    }


def _build_logical_graph_from_index(
    lith_nexus_index: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a minimal logical_graph from lith_nexus index nodes.

    This creates a simple logical graph with the forms/contracts/protocols
    from the .lith files without requiring the full catalog machinery.
    """
    nexus_payload = lith_nexus_index if isinstance(lith_nexus_index, dict) else {}
    lith_nodes_raw = nexus_payload.get("nodes", [])
    if not isinstance(lith_nodes_raw, list):
        lith_nodes_raw = []
    lith_edges_raw = nexus_payload.get("edges", [])
    if not isinstance(lith_edges_raw, list):
        lith_edges_raw = []

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for lith_node in lith_nodes_raw:
        if not isinstance(lith_node, dict):
            continue
        node_id = str(lith_node.get("id", "")).strip()
        if not node_id:
            continue
        node_kind = str(lith_node.get("kind", "")).strip().lower()
        node_title = str(lith_node.get("title", "")).strip()
        node_path = ""
        prov = lith_node.get("provenance")
        if isinstance(prov, dict):
            node_path = str(prov.get("path", "") or "").strip()

        logical_node: dict[str, Any] = {
            "id": node_id,
            "node_type": node_kind,
            "kind": node_kind,
            "label": node_title or node_id,
            "x": 0.5,
            "y": 0.5,
            "hue": _hue_for_kind(node_kind),
            "importance": 0.5,
            "provenance": {"origin_graph": "lith_nexus"},
        }
        if node_path:
            logical_node["path"] = node_path
        nodes.append(logical_node)

    node_id_set = {n["id"] for n in nodes}

    for lith_edge in lith_edges_raw:
        if not isinstance(lith_edge, dict):
            continue
        source_id = str(lith_edge.get("source", "")).strip()
        target_id = str(lith_edge.get("target", "")).strip()
        if not source_id or not target_id:
            continue
        if source_id not in node_id_set or target_id not in node_id_set:
            continue
        edge_kind = str(lith_edge.get("kind", "relates")).strip().lower() or "relates"
        edge_id = (
            str(lith_edge.get("id", "")).strip() or f"edge:{source_id}|{target_id}"
        )
        edges.append(
            {
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "kind": edge_kind,
                "weight": 0.5,
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "joins": {},
        "provenance": {"origin_graph": "lith_nexus"},
    }


def _hue_for_kind(kind: str) -> int:
    """Return a consistent hue for a node kind."""
    kind_lower = kind.lower()
    if kind_lower in ("contract", "契"):
        return 280
    if kind_lower == "protocol":
        return 200
    if kind_lower == "packet":
        return 160
    if kind_lower == "form":
        return 120
    if kind_lower == "spec":
        return 60
    if kind_lower == "fact":
        return 30
    if kind_lower == "tag":
        return 340
    return 198
