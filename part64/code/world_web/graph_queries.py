from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_query_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return "overview"
    aliases = {
        "summary": "overview",
        "stats": "overview",
        "neighbor": "neighbors",
        "node_neighbors": "neighbors",
        "roles": "role_slice",
        "role": "role_slice",
        "recent": "recently_updated",
    }
    return aliases.get(text, text)


def _node_role(node: dict[str, Any]) -> str:
    role = str(node.get("role", "") or "").strip().lower()
    if role:
        return role
    web_role = str(node.get("web_node_role", "") or "").strip().lower()
    if web_role:
        return web_role
    return (
        str(node.get("node_type", "unknown") or "unknown").strip().lower() or "unknown"
    )


def _graph_rows(
    nexus_graph: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = nexus_graph if isinstance(nexus_graph, dict) else {}
    nodes = [row for row in graph.get("nodes", []) if isinstance(row, dict)]
    edges = [row for row in graph.get("edges", []) if isinstance(row, dict)]
    return nodes, edges


def _node_index(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in nodes:
        node_id = str(row.get("id", "")).strip()
        if node_id and node_id not in by_id:
            by_id[node_id] = row
    return by_id


def _query_overview(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    role_counts: dict[str, int] = {}
    for node in nodes:
        role = _node_role(node)
        role_counts[role] = role_counts.get(role, 0) + 1

    degree: dict[str, int] = {}
    for edge in edges:
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if source_id:
            degree[source_id] = degree.get(source_id, 0) + 1
        if target_id:
            degree[target_id] = degree.get(target_id, 0) + 1

    top_degree_nodes = sorted(
        [{"id": node_id, "degree": count} for node_id, count in degree.items()],
        key=lambda row: (-_safe_int(row.get("degree", 0), 0), str(row.get("id", ""))),
    )[:24]

    edge_kind_counts: dict[str, int] = {}
    for edge in edges:
        kind = str(edge.get("kind", "relates") or "relates").strip().lower()
        edge_kind_counts[kind] = edge_kind_counts.get(kind, 0) + 1

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "role_counts": role_counts,
        "edge_kind_counts": edge_kind_counts,
        "top_degree_nodes": top_degree_nodes,
    }


def _query_neighbors(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    node_id = str(args.get("node_id", "")).strip()
    if not node_id and nodes:
        node_id = str(nodes[0].get("id", "")).strip()
    edge_role = str(args.get("edge_role", "")).strip().lower()

    rows: list[dict[str, Any]] = []
    for edge in edges:
        kind = str(edge.get("kind", "relates") or "relates").strip().lower()
        if edge_role and kind != edge_role:
            continue
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if source_id == node_id and target_id:
            rows.append({"node_id": target_id, "edge_kind": kind, "direction": "out"})
        elif target_id == node_id and source_id:
            rows.append({"node_id": source_id, "edge_kind": kind, "direction": "in"})

    rows.sort(
        key=lambda row: (
            str(row.get("direction", "")),
            str(row.get("edge_kind", "")),
            str(row.get("node_id", "")),
        )
    )
    return {
        "node_id": node_id,
        "edge_role": edge_role,
        "neighbor_count": len(rows),
        "neighbors": rows[:128],
    }


def _query_role_slice(
    nodes: list[dict[str, Any]], args: dict[str, Any]
) -> dict[str, Any]:
    role = str(args.get("role", "file") or "file").strip().lower()
    rows = [
        {
            "id": str(node.get("id", "")).strip(),
            "label": str(node.get("label", "") or "").strip(),
            "importance": round(_safe_float(node.get("importance", 0.0), 0.0), 6),
        }
        for node in nodes
        if _node_role(node) == role
    ]
    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("importance", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return {
        "role": role,
        "count": len(rows),
        "nodes": rows[:128],
    }


def _query_search(nodes: list[dict[str, Any]], args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("q", args.get("query", "")) or "").strip().lower()
    if not query:
        return {"query": query, "count": 0, "nodes": []}

    rows: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        label = str(node.get("label", "") or "").strip()
        canonical_url = str(node.get("canonical_url", "") or "").strip()
        haystack = " ".join(
            [node_id.lower(), label.lower(), canonical_url.lower()]
        ).strip()
        if query in haystack:
            rows.append(
                {
                    "id": node_id,
                    "role": _node_role(node),
                    "label": label,
                    "canonical_url": canonical_url,
                }
            )

    rows.sort(key=lambda row: (str(row.get("role", "")), str(row.get("id", ""))))
    return {"query": query, "count": len(rows), "nodes": rows[:128]}


def _query_url_status(
    nodes: list[dict[str, Any]], args: dict[str, Any]
) -> dict[str, Any]:
    target = str(
        args.get("target", args.get("url", args.get("url_id", ""))) or ""
    ).strip()
    target_lower = target.lower()

    found: dict[str, Any] | None = None
    for node in nodes:
        role = _node_role(node)
        if role != "web:url":
            continue
        node_id = str(node.get("id", "")).strip()
        canonical_url = str(node.get("canonical_url", "") or "").strip()
        if target and target_lower not in {node_id.lower(), canonical_url.lower()}:
            continue
        found = {
            "id": node_id,
            "canonical_url": canonical_url,
            "next_allowed_fetch_ts": round(
                max(0.0, _safe_float(node.get("next_allowed_fetch_ts", 0.0), 0.0)),
                6,
            ),
            "fail_count": max(0, _safe_int(node.get("fail_count", 0), 0)),
            "last_fetch_ts": round(
                max(0.0, _safe_float(node.get("last_fetch_ts", 0.0), 0.0)), 6
            ),
            "last_status": str(node.get("last_status", "") or "").strip(),
        }
        break

    return {"target": target, "found": found}


def _query_resource_for_url(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    target = str(
        args.get("target", args.get("url_id", args.get("url", ""))) or ""
    ).strip()
    if not target:
        return {"target": "", "count": 0, "resources": []}

    node_by_id = _node_index(nodes)
    target_url_id = ""
    for node in nodes:
        if _node_role(node) != "web:url":
            continue
        node_id = str(node.get("id", "")).strip()
        canonical_url = str(node.get("canonical_url", "") or "").strip()
        if target.lower() in {node_id.lower(), canonical_url.lower()}:
            target_url_id = node_id
            break

    if not target_url_id:
        return {"target": target, "count": 0, "resources": []}

    resources: list[dict[str, Any]] = []
    for edge in edges:
        if str(edge.get("kind", "")).strip().lower() != "web:source_of":
            continue
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if target_id != target_url_id:
            continue
        source_node = node_by_id.get(source_id, {})
        resources.append(
            {
                "id": source_id,
                "canonical_url": str(
                    source_node.get("canonical_url", "") or ""
                ).strip(),
                "fetched_ts": round(
                    max(0.0, _safe_float(source_node.get("fetched_ts", 0.0), 0.0)), 6
                ),
                "content_hash": str(source_node.get("content_hash", "") or "").strip(),
            }
        )

    resources.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return {
        "target": target_url_id,
        "count": len(resources),
        "resources": resources[:64],
    }


def _query_recently_updated(
    nodes: list[dict[str, Any]], args: dict[str, Any]
) -> dict[str, Any]:
    limit = max(1, min(128, _safe_int(args.get("limit", 24), 24)))
    rows: list[dict[str, Any]] = []
    for node in nodes:
        role = _node_role(node)
        fetched_ts = max(
            0.0,
            _safe_float(node.get("fetched_ts", node.get("last_fetch_ts", 0.0)), 0.0),
        )
        if fetched_ts <= 0.0:
            continue
        rows.append(
            {
                "id": str(node.get("id", "")).strip(),
                "role": role,
                "canonical_url": str(node.get("canonical_url", "") or "").strip(),
                "fetched_ts": round(fetched_ts, 6),
            }
        )
    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return {"count": len(rows), "nodes": rows[:limit]}


def run_named_graph_query(
    nexus_graph: dict[str, Any] | None,
    query_name: str,
    *,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes, edges = _graph_rows(nexus_graph)
    query = _normalize_query_name(query_name)
    query_args = dict(args) if isinstance(args, dict) else {}

    handlers = {
        "overview": lambda: _query_overview(nodes, edges),
        "neighbors": lambda: _query_neighbors(nodes, edges, query_args),
        "role_slice": lambda: _query_role_slice(nodes, query_args),
        "search": lambda: _query_search(nodes, query_args),
        "url_status": lambda: _query_url_status(nodes, query_args),
        "resource_for_url": lambda: _query_resource_for_url(nodes, edges, query_args),
        "recently_updated": lambda: _query_recently_updated(nodes, query_args),
    }

    if query in handlers:
        result = handlers[query]()
    else:
        result = {
            "error": "unknown_query",
            "query": query,
            "supported": [
                "overview",
                "neighbors",
                "search",
                "url_status",
                "resource_for_url",
                "recently_updated",
                "role_slice",
            ],
        }

    payload = {
        "record": "eta-mu.graph-query.v1",
        "schema_version": "graph.query.v1",
        "generated_at": _now_iso(),
        "query": query,
        "args": query_args,
        "result": result,
    }
    payload["snapshot_hash"] = _canonical_hash(payload)
    return payload


def _facts_counts(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    nodes_by_role: dict[str, int] = {}
    edges_by_kind: dict[str, int] = {}
    for node in nodes:
        role = _node_role(node)
        nodes_by_role[role] = nodes_by_role.get(role, 0) + 1
    for edge in edges:
        kind = str(edge.get("kind", "relates") or "relates").strip().lower()
        edges_by_kind[kind] = edges_by_kind.get(kind, 0) + 1
    return {
        "nodes_by_role": nodes_by_role,
        "edges_by_kind": edges_by_kind,
    }


def _facts_web_section(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    node_by_id = _node_index(nodes)
    inbound_links: dict[str, int] = {}
    source_of_count: dict[str, int] = {}
    links_to_count: dict[str, int] = {}

    for edge in edges:
        kind = str(edge.get("kind", "")).strip().lower()
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if kind == "web:links_to" and target_id:
            inbound_links[target_id] = inbound_links.get(target_id, 0) + 1
            links_to_count[source_id] = links_to_count.get(source_id, 0) + 1
        if kind == "web:source_of" and source_id:
            source_of_count[source_id] = source_of_count.get(source_id, 0) + 1

    now_epoch = time.time()
    urls: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []

    for node in nodes:
        role = _node_role(node)
        if role == "web:url":
            node_id = str(node.get("id", "")).strip()
            next_allowed = max(
                0.0, _safe_float(node.get("next_allowed_fetch_ts", 0.0), 0.0)
            )
            urls.append(
                {
                    "url_id": node_id,
                    "canonical_url": str(node.get("canonical_url", "") or "").strip(),
                    "next_allowed_fetch_ts": round(next_allowed, 6),
                    "cooldown_active": bool(next_allowed > now_epoch),
                    "fail_count": max(0, _safe_int(node.get("fail_count", 0), 0)),
                    "last_status": str(node.get("last_status", "") or "").strip(),
                    "last_fetch_ts": round(
                        max(0.0, _safe_float(node.get("last_fetch_ts", 0.0), 0.0)), 6
                    ),
                    "inbound_links": max(0, inbound_links.get(node_id, 0)),
                }
            )
        elif role == "web:resource":
            node_id = str(node.get("id", "")).strip()
            resources.append(
                {
                    "res_id": node_id,
                    "canonical_url": str(node.get("canonical_url", "") or "").strip(),
                    "fetched_ts": round(
                        max(0.0, _safe_float(node.get("fetched_ts", 0.0), 0.0)), 6
                    ),
                    "content_hash": str(node.get("content_hash", "") or "").strip(),
                    "link_count": max(0, links_to_count.get(node_id, 0)),
                    "source_count": max(0, source_of_count.get(node_id, 0)),
                }
            )

    urls.sort(
        key=lambda row: (
            -_safe_int(row.get("inbound_links", 0), 0),
            -_safe_float(row.get("last_fetch_ts", 0.0), 0.0),
            str(row.get("url_id", "")),
        )
    )
    resources.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("res_id", "")),
        )
    )

    return {
        "urls": urls[:64],
        "resources": resources[:64],
        "resource_lookup": {
            str(row.get("res_id", "")): dict(row)
            for row in resources
            if str(row.get("res_id", ""))
        },
        "node_lookup": node_by_id,
    }


def _facts_recent_events(
    simulation: dict[str, Any], *, limit: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dynamics = (
        simulation.get("presence_dynamics", {}) if isinstance(simulation, dict) else {}
    )
    if isinstance(dynamics, dict):
        trails = dynamics.get("daimoi_outcome_trails", [])
        if isinstance(trails, list):
            for row in trails[-limit:]:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "id": f"daimoi:{_safe_int(row.get('seq', 0), 0)}",
                        "kind": "daimoi_outcome",
                        "ts": str(row.get("ts", "") or ""),
                        "outcome": str(row.get("outcome", "") or "").strip().lower(),
                        "target": str(row.get("graph_node_id", "") or "").strip(),
                        "reason": str(row.get("reason", "") or "").strip(),
                    }
                )

    crawler_graph = (
        simulation.get("crawler_graph", {}) if isinstance(simulation, dict) else {}
    )
    crawler_events = (
        crawler_graph.get("events", []) if isinstance(crawler_graph, dict) else []
    )
    if isinstance(crawler_events, list):
        for row in crawler_events[-limit:]:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "id": str(row.get("id", "") or "").strip(),
                    "kind": str(row.get("kind", "") or "").strip(),
                    "ts": str(row.get("ts", "") or ""),
                    "url_id": str(row.get("url_id", "") or "").strip(),
                    "reason": str(row.get("reason", "") or "").strip(),
                    "status": str(row.get("status", "") or "").strip(),
                }
            )

    rows.sort(
        key=lambda row: (
            str(row.get("ts", "")),
            str(row.get("kind", "")),
            str(row.get("id", "")),
        )
    )
    return rows[-limit:]


def _facts_invariants(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    node_by_id = _node_index(nodes)
    url_ids = {
        str(node.get("id", "")).strip()
        for node in nodes
        if _node_role(node) == "web:url" and str(node.get("id", "")).strip()
    }
    violations: list[dict[str, Any]] = []

    source_edges_by_resource: dict[str, int] = {}
    for edge in edges:
        kind = str(edge.get("kind", "")).strip().lower()
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if kind == "web:source_of":
            source_edges_by_resource[source_id] = (
                source_edges_by_resource.get(source_id, 0) + 1
            )
            if target_id not in url_ids:
                violations.append(
                    {
                        "kind": "web_source_target_not_url",
                        "resource_id": source_id,
                        "target_id": target_id,
                    }
                )
        if kind == "web:links_to" and target_id not in url_ids:
            violations.append(
                {
                    "kind": "web_links_target_not_url",
                    "source_id": source_id,
                    "target_id": target_id,
                }
            )

    for node in nodes:
        role = _node_role(node)
        node_id = str(node.get("id", "")).strip()
        if role == "web:resource":
            source_count = source_edges_by_resource.get(node_id, 0)
            if source_count != 1:
                violations.append(
                    {
                        "kind": "resource_source_of_count_invalid",
                        "resource_id": node_id,
                        "count": source_count,
                    }
                )
        if role == "web:url":
            canonical_url = str(node.get("canonical_url", "") or "").strip()
            if not canonical_url:
                violations.append(
                    {"kind": "url_missing_canonical_url", "url_id": node_id}
                )
            next_allowed = _safe_float(node.get("next_allowed_fetch_ts", 0.0), 0.0)
            fail_count = _safe_int(node.get("fail_count", 0), 0)
            if next_allowed < 0.0 or fail_count < 0:
                violations.append(
                    {
                        "kind": "url_cooldown_invalid",
                        "url_id": node_id,
                        "next_allowed_fetch_ts": round(next_allowed, 6),
                        "fail_count": fail_count,
                    }
                )

    # deterministic dedupe
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(
        violations,
        key=lambda item: _canonical_json(item if isinstance(item, dict) else {}),
    ):
        token = _canonical_json(row)
        if token in seen:
            continue
        seen.add(token)
        deduped.append(row)
    return deduped


def build_facts_snapshot(
    simulation: dict[str, Any] | None,
    *,
    part_root: Path | str | None = None,
) -> dict[str, Any]:
    payload = simulation if isinstance(simulation, dict) else {}
    nexus_graph = (
        payload.get("nexus_graph", {})
        if isinstance(payload.get("nexus_graph", {}), dict)
        else {}
    )
    nodes, edges = _graph_rows(nexus_graph)
    counts = _facts_counts(nodes, edges)
    web_section = _facts_web_section(nodes, edges)
    invariants = _facts_invariants(nodes, edges)
    recent_events = _facts_recent_events(payload, limit=48)

    dynamics = (
        payload.get("presence_dynamics", {})
        if isinstance(payload.get("presence_dynamics", {}), dict)
        else {}
    )
    outcomes = (
        dynamics.get("daimoi_outcome_summary", {})
        if isinstance(dynamics.get("daimoi_outcome_summary", {}), dict)
        else {}
    )

    facts: dict[str, Any] = {
        "record": "eta-mu.facts-snapshot.v1",
        "schema_version": "facts.snapshot.v1",
        "ts": _now_iso(),
        "tick": max(0, _safe_int(dynamics.get("tick", 0), 0)),
        "counts": counts,
        "recent_events": recent_events,
        "web": {
            "urls": web_section.get("urls", []),
            "resources": web_section.get("resources", []),
        },
        "dynamics": {
            "daimoi_outcomes": {
                "food": max(0, _safe_int(outcomes.get("food", 0), 0)),
                "death": max(0, _safe_int(outcomes.get("death", 0), 0)),
                "total": max(0, _safe_int(outcomes.get("total", 0), 0)),
            }
        },
        "invariants_violations": invariants,
    }

    snapshot_hash = _canonical_hash(facts)
    facts["snapshot_hash"] = snapshot_hash

    snapshot_path = ""
    if part_root is not None:
        base = Path(part_root)
        ts_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = base / "world_state" / "facts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ts_token}_{snapshot_hash[:12]}.json"
        out_path.write_text(
            json.dumps(facts, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
        )
        snapshot_path = str(out_path)
    facts["snapshot_path"] = snapshot_path
    return facts
