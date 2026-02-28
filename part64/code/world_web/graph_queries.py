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
        "recently_touched": "recently_updated",
        "touched": "recently_updated",
        "explain": "explain_daimoi",
        "outcomes": "recent_outcomes",
        "crawler": "crawler_status",
        "arxiv": "arxiv_papers",
        "arxiv_papers": "arxiv_papers",
        "resource_summary": "web_resource_summary",
        "graph_summary": "graph_summary",
        "github": "github_status",
        "github_repo": "github_repo_summary",
        "github_search": "github_find",
        "github_recent": "github_recent_changes",
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


def _node_extension(node: dict[str, Any]) -> dict[str, Any]:
    extension = node.get("extension", {}) if isinstance(node, dict) else {}
    return extension if isinstance(extension, dict) else {}


def _node_value(node: dict[str, Any], key: str, default: Any = "") -> Any:
    if key in node and node.get(key) is not None:
        value = node.get(key)
        if isinstance(value, str):
            if value.strip():
                return value
        else:
            return value
    extension = _node_extension(node)
    if key in extension:
        return extension.get(key)
    return default


def _node_list(node: dict[str, Any], key: str) -> list[Any]:
    value = _node_value(node, key, [])
    return value if isinstance(value, list) else []


def _node_canonical_url(node: dict[str, Any]) -> str:
    for key in ("canonical_url", "url", "href", "source_url"):
        value = str(_node_value(node, key, "") or "").strip()
        if value:
            return value
    return ""


def _node_title(node: dict[str, Any]) -> str:
    for key in ("title", "label", "name"):
        value = str(_node_value(node, key, "") or "").strip()
        if value:
            return value
    return ""


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
        canonical_url = _node_canonical_url(node)
        title = _node_title(node)
        haystack = " ".join(
            [node_id.lower(), label.lower(), canonical_url.lower(), title.lower()]
        ).strip()
        if query in haystack:
            rows.append(
                {
                    "id": node_id,
                    "role": _node_role(node),
                    "label": label,
                    "canonical_url": canonical_url,
                    "title": title,
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
        canonical_url = _node_canonical_url(node)
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
            "last_status": str(
                node.get("last_status", node.get("status", "")) or ""
            ).strip(),
            "title": _node_title(node),
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
        canonical_url = _node_canonical_url(node)
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
                "canonical_url": _node_canonical_url(source_node),
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
                "canonical_url": _node_canonical_url(node),
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


def _simulation_sections(
    simulation: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = simulation if isinstance(simulation, dict) else {}
    dynamics = (
        payload.get("presence_dynamics", {})
        if isinstance(payload.get("presence_dynamics", {}), dict)
        else {}
    )
    crawler_graph = (
        payload.get("crawler_graph", {})
        if isinstance(payload.get("crawler_graph", {}), dict)
        else {}
    )
    field_particles = [
        row for row in dynamics.get("field_particles", []) if isinstance(row, dict)
    ]
    outcome_rows = [
        row
        for row in dynamics.get("daimoi_outcome_trails", [])
        if isinstance(row, dict)
    ]
    return dynamics, crawler_graph, field_particles, outcome_rows


def _query_explain_daimoi(
    simulation: dict[str, Any] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    dynamics, _, field_particles, outcome_rows = _simulation_sections(simulation)
    target_id = str(
        args.get("daimoi_id", args.get("id", args.get("target", ""))) or ""
    ).strip()

    live_state: dict[str, Any] | None = None
    if target_id:
        for row in field_particles:
            if str(row.get("id", "") or "").strip() == target_id:
                live_state = row
                break
    elif field_particles:
        live_state = field_particles[0]
        target_id = str(live_state.get("id", "") or "").strip()

    matched_outcomes: list[dict[str, Any]] = []
    for row in outcome_rows:
        row_daimoi_id = str(row.get("daimoi_id", "") or "").strip()
        if target_id and row_daimoi_id != target_id:
            continue
        matched_outcomes.append(row)

    matched_outcomes.sort(
        key=lambda row: (
            _safe_int(row.get("tick", row.get("seq", 0)), 0),
            _safe_int(row.get("seq", 0), 0),
        )
    )
    last_outcome = matched_outcomes[-1] if matched_outcomes else None
    last_outcome_kind = (
        str(last_outcome.get("outcome", "") or "").strip().lower()
        if isinstance(last_outcome, dict)
        else ""
    )

    status = "unknown"
    if isinstance(live_state, dict) and live_state:
        status = "alive"
    elif last_outcome_kind in {"food", "death"}:
        status = last_outcome_kind

    reinforcement_direction = "none"
    if last_outcome_kind == "food":
        reinforcement_direction = "forward"
    elif last_outcome_kind == "death":
        reinforcement_direction = "inverse"

    return {
        "daimoi_id": target_id,
        "status": status,
        "live_state": (
            {
                "present": True,
                "owner": str(
                    live_state.get(
                        "presence_id",
                        live_state.get(
                            "owner_presence_id", live_state.get("owner", "")
                        ),
                    )
                    or ""
                ).strip(),
                "graph_node_id": str(live_state.get("graph_node_id", "") or "").strip(),
                "top_job": str(live_state.get("top_job", "") or "").strip(),
                "collision_count": max(
                    0,
                    _safe_int(
                        live_state.get(
                            "collision_count", live_state.get("collisions", 0)
                        ),
                        0,
                    ),
                ),
                "message_probability": round(
                    max(
                        0.0,
                        _safe_float(live_state.get("message_probability", 0.0), 0.0),
                    ),
                    6,
                ),
            }
            if isinstance(live_state, dict)
            else {"present": False}
        ),
        "last_outcome": (
            {
                "outcome": last_outcome_kind,
                "reason": str(last_outcome.get("reason", "") or "").strip(),
                "target_id": str(last_outcome.get("graph_node_id", "") or "").strip(),
                "intensity": round(
                    max(0.0, _safe_float(last_outcome.get("intensity", 0.0), 0.0)),
                    6,
                ),
                "tick": max(
                    0,
                    _safe_int(last_outcome.get("tick", last_outcome.get("seq", 0)), 0),
                ),
                "trail_steps": max(
                    0,
                    _safe_int(last_outcome.get("trail_steps", 0), 0),
                ),
                "ts": str(last_outcome.get("ts", "") or "").strip(),
            }
            if isinstance(last_outcome, dict)
            else None
        ),
        "reinforcement": {
            "direction": reinforcement_direction,
            "intensity": round(
                max(
                    0.0,
                    _safe_float(
                        (last_outcome or {}).get("intensity", 0.0)
                        if isinstance(last_outcome, dict)
                        else 0.0,
                        0.0,
                    ),
                ),
                6,
            ),
        },
        "outcome_count": len(matched_outcomes),
        "tick": max(0, _safe_int(dynamics.get("tick", 0), 0)),
    }


def _query_recent_outcomes(
    simulation: dict[str, Any] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    _, _, _, outcome_rows = _simulation_sections(simulation)
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 360), 360)))
    limit = max(1, min(256, _safe_int(args.get("limit", 48), 48)))

    normalized_rows = [
        {
            "tick": max(0, _safe_int(row.get("tick", row.get("seq", 0)), 0)),
            "seq": max(0, _safe_int(row.get("seq", 0), 0)),
            "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
            "outcome": str(row.get("outcome", "") or "").strip().lower(),
            "reason": str(row.get("reason", "") or "").strip(),
            "target_id": str(row.get("graph_node_id", "") or "").strip(),
            "intensity": round(
                max(0.0, _safe_float(row.get("intensity", 0.0), 0.0)), 6
            ),
            "ts": str(row.get("ts", "") or "").strip(),
        }
        for row in outcome_rows
    ]
    normalized_rows.sort(
        key=lambda row: (
            _safe_int(row.get("tick", 0), 0),
            _safe_int(row.get("seq", 0), 0),
        )
    )

    max_tick = max(
        [max(0, _safe_int(row.get("tick", 0), 0)) for row in normalized_rows],
        default=0,
    )
    min_tick = max(0, max_tick - window_ticks + 1)
    windowed = [
        row
        for row in normalized_rows
        if max(0, _safe_int(row.get("tick", 0), 0)) >= min_tick
    ]
    return {
        "window_ticks": int(window_ticks),
        "count": len(windowed),
        "outcomes": windowed[-limit:],
    }


def _query_crawler_status(simulation: dict[str, Any] | None) -> dict[str, Any]:
    _, crawler_graph, _, _ = _simulation_sections(simulation)
    stats = crawler_graph.get("stats", {}) if isinstance(crawler_graph, dict) else {}
    status = crawler_graph.get("status", {}) if isinstance(crawler_graph, dict) else {}
    nodes = [row for row in crawler_graph.get("nodes", []) if isinstance(row, dict)]
    url_nodes = [row for row in nodes if _node_role(row) == "web:url"]
    now_epoch = time.time()

    cooldown_active = 0
    fail_total = 0
    for row in url_nodes:
        next_allowed = max(0.0, _safe_float(row.get("next_allowed_fetch_ts", 0.0), 0.0))
        if next_allowed > now_epoch:
            cooldown_active += 1
        fail_total += max(0, _safe_int(row.get("fail_count", 0), 0))

    last_fetch_rows = [
        {
            "url_id": str(row.get("id", "") or "").strip(),
            "canonical_url": _node_canonical_url(row),
            "title": _node_title(row),
            "last_fetch_ts": round(
                max(0.0, _safe_float(row.get("last_fetch_ts", 0.0), 0.0)),
                6,
            ),
            "last_status": str(row.get("last_status", "") or "").strip(),
            "next_allowed_fetch_ts": round(
                max(0.0, _safe_float(row.get("next_allowed_fetch_ts", 0.0), 0.0)),
                6,
            ),
        }
        for row in url_nodes
    ]
    last_fetch_rows.sort(
        key=lambda row: (
            -_safe_float(row.get("last_fetch_ts", 0.0), 0.0),
            str(row.get("url_id", "")),
        )
    )

    queue_length = max(
        0,
        _safe_int(
            status.get(
                "queue_length", status.get("pending_count", status.get("queue", 0))
            ),
            0,
        ),
    )

    arxiv_abs_rows = [
        row for row in url_nodes if "arxiv.org/abs/" in _node_canonical_url(row)
    ]
    arxiv_abs_fetched = [
        row
        for row in arxiv_abs_rows
        if str(row.get("last_status", "") or "").strip().lower() == "ok"
        and _safe_float(row.get("last_fetch_ts", 0.0), 0.0) > 0.0
    ]
    arxiv_recent_fetches = [
        row
        for row in last_fetch_rows
        if "arxiv.org/abs/" in str(row.get("canonical_url", ""))
    ]

    return {
        "queue_length": int(queue_length),
        "cooldown_active": int(cooldown_active),
        "url_node_count": len(url_nodes),
        "resource_node_count": max(
            0,
            _safe_int(
                stats.get(
                    "web_role_counts",
                    {},
                ).get("web:resource", 0)
                if isinstance(stats.get("web_role_counts", {}), dict)
                else 0,
                0,
            ),
        ),
        "fail_count_total": int(fail_total),
        "event_count": max(0, _safe_int(stats.get("event_count", 0), 0)),
        "arxiv_abs_count": len(arxiv_abs_rows),
        "arxiv_abs_fetched": len(arxiv_abs_fetched),
        "arxiv_recent_fetches": arxiv_recent_fetches[:8],
        "last_fetches": last_fetch_rows[:8],
    }


def _query_arxiv_papers(
    simulation: dict[str, Any] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    _, crawler_graph, _, _ = _simulation_sections(simulation)
    nodes = [row for row in crawler_graph.get("nodes", []) if isinstance(row, dict)]
    limit = max(
        1,
        min(
            64,
            _safe_int(args.get("limit", args.get("n", args.get("count", 8))), 8),
        ),
    )

    papers: list[dict[str, Any]] = []
    for node in nodes:
        if _node_role(node) != "web:url":
            continue
        canonical_url = _node_canonical_url(node)
        if "arxiv.org/abs/" not in canonical_url:
            continue
        arxiv_id = ""
        if "/abs/" in canonical_url:
            arxiv_id = (
                canonical_url.split("/abs/", 1)[1].split("?", 1)[0].split("#", 1)[0]
            )

        last_status = str(node.get("last_status", node.get("status", "")) or "").strip()
        last_fetch_ts = max(
            0.0,
            _safe_float(node.get("last_fetch_ts", node.get("fetched_ts", 0.0)), 0.0),
        )
        fetched = last_status.lower() == "ok" and last_fetch_ts > 0.0
        papers.append(
            {
                "url_id": str(node.get("id", "") or "").strip(),
                "arxiv_id": arxiv_id,
                "canonical_url": canonical_url,
                "title": _node_title(node),
                "last_status": last_status,
                "last_fetch_ts": round(last_fetch_ts, 6),
                "fetched": bool(fetched),
            }
        )

    papers.sort(
        key=lambda row: (
            0 if bool(row.get("fetched", False)) else 1,
            -_safe_float(row.get("last_fetch_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )
    fetched_count = sum(1 for row in papers if bool(row.get("fetched", False)))
    return {
        "count_total": len(papers),
        "count_fetched": fetched_count,
        "papers": papers[:limit],
    }


def _github_resource_rows(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in nodes:
        if _node_role(node) != "web:resource":
            continue
        kind = str(_node_value(node, "kind", "") or "").strip().lower()
        if not kind.startswith("github:"):
            continue

        labels = [
            str(item).strip()
            for item in _node_list(node, "labels")
            if str(item).strip()
        ]
        authors = [
            str(item).strip()
            for item in _node_list(node, "authors")
            if str(item).strip()
        ]
        atoms = [row for row in _node_list(node, "atoms") if isinstance(row, dict)]
        rows.append(
            {
                "id": str(node.get("id", "") or "").strip(),
                "kind": kind,
                "canonical_url": _node_canonical_url(node),
                "title": _node_title(node),
                "repo": str(_node_value(node, "repo", "") or "").strip(),
                "number": max(0, _safe_int(_node_value(node, "number", 0), 0)),
                "labels": labels,
                "authors": authors,
                "updated_at": str(_node_value(node, "updated_at", "") or "").strip(),
                "fetched_ts": round(
                    max(0.0, _safe_float(_node_value(node, "fetched_ts", 0.0), 0.0)),
                    6,
                ),
                "content_hash": str(
                    _node_value(node, "content_hash", "") or ""
                ).strip(),
                "importance_score": max(
                    0,
                    _safe_int(_node_value(node, "importance_score", 0), 0),
                ),
                "atoms": atoms,
                "state": str(_node_value(node, "state", "") or "").strip().lower(),
                "merged_at": str(_node_value(node, "merged_at", "") or "").strip(),
            }
        )

    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return rows


def _query_github_status(
    nodes: list[dict[str, Any]],
    simulation: dict[str, Any] | None,
) -> dict[str, Any]:
    resources = _github_resource_rows(nodes)
    repo_set = {
        str(row.get("repo", "")).strip()
        for row in resources
        if str(row.get("repo", "")).strip()
    }

    now_epoch = time.time()
    github_url_nodes = []
    for node in nodes:
        if _node_role(node) != "web:url":
            continue
        canonical_url = _node_canonical_url(node).lower()
        source_hint = str(_node_value(node, "source_hint", "") or "").strip().lower()
        if (
            source_hint == "github"
            or "github.com/" in canonical_url
            or "raw.githubusercontent.com/" in canonical_url
        ):
            github_url_nodes.append(node)

    cooldown_active = 0
    for node in github_url_nodes:
        next_allowed = max(
            0.0,
            _safe_float(_node_value(node, "next_allowed_fetch_ts", 0.0), 0.0),
        )
        if next_allowed > now_epoch:
            cooldown_active += 1

    crawler_graph = (
        simulation.get("crawler_graph", {}) if isinstance(simulation, dict) else {}
    )
    status = crawler_graph.get("status", {}) if isinstance(crawler_graph, dict) else {}
    github_status = status.get("github", {}) if isinstance(status, dict) else {}
    github_status = github_status if isinstance(github_status, dict) else {}

    events = crawler_graph.get("events", []) if isinstance(crawler_graph, dict) else []
    github_events = [
        row
        for row in events
        if isinstance(row, dict)
        and str(row.get("event", row.get("kind", "")))
        .strip()
        .lower()
        .startswith("github_")
    ]

    monitored_repos = [
        str(item).strip()
        for item in github_status.get("monitored_repos", [])
        if str(item).strip()
    ]
    if not monitored_repos:
        monitored_repos = sorted(repo_set)

    return {
        "monitored_repos": monitored_repos,
        "queue_length": max(0, _safe_int(github_status.get("queue_length", 0), 0)),
        "active_fetches": max(0, _safe_int(github_status.get("active_fetches", 0), 0)),
        "cooldown_blocks": max(
            0,
            _safe_int(
                github_status.get(
                    "cooldown_blocks",
                    github_status.get("cooldown_active", cooldown_active),
                ),
                cooldown_active,
            ),
        ),
        "url_node_count": len(github_url_nodes),
        "resource_node_count": len(resources),
        "event_count": len(github_events),
        "last_fetches": [
            {
                "repo": row.get("repo", ""),
                "kind": row.get("kind", ""),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "fetched_ts": row.get("fetched_ts", 0.0),
                "content_hash": row.get("content_hash", ""),
            }
            for row in resources[:8]
        ],
    }


def _query_github_repo_summary(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    resources = _github_resource_rows(nodes)
    repo = str(args.get("repo", "") or "").strip()
    if not repo and resources:
        repo = str(resources[0].get("repo", "") or "").strip()

    filtered = [
        row
        for row in resources
        if not repo or str(row.get("repo", "")).strip().lower() == repo.lower()
    ]

    ranked = list(filtered)
    ranked.sort(
        key=lambda row: (
            -_safe_int(row.get("importance_score", 0), 0),
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )

    return {
        "repo": repo,
        "count": len(filtered),
        "last_updated": str(filtered[0].get("updated_at", "")) if filtered else "",
        "recent_resources": [
            {
                "id": row.get("id", ""),
                "kind": row.get("kind", ""),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "number": row.get("number", 0),
                "updated_at": row.get("updated_at", ""),
                "fetched_ts": row.get("fetched_ts", 0.0),
                "importance_score": row.get("importance_score", 0),
            }
            for row in filtered[:24]
        ],
        "top_items": [
            {
                "id": row.get("id", ""),
                "kind": row.get("kind", ""),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "importance_score": row.get("importance_score", 0),
                "content_hash": row.get("content_hash", ""),
            }
            for row in ranked[:16]
        ],
    }


def _query_github_find(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    term = (
        str(args.get("term", args.get("q", args.get("query", ""))) or "")
        .strip()
        .lower()
    )
    repo = str(args.get("repo", "") or "").strip().lower()
    limit = max(1, min(128, _safe_int(args.get("limit", 24), 24)))

    if not term:
        return {"term": term, "repo": repo, "count": 0, "matches": []}

    matches: list[dict[str, Any]] = []
    for row in _github_resource_rows(nodes):
        row_repo = str(row.get("repo", "")).strip().lower()
        if repo and row_repo != repo:
            continue

        title = str(row.get("title", "") or "")
        canonical_url = str(row.get("canonical_url", "") or "")
        haystack = " ".join(
            [
                title.lower(),
                canonical_url.lower(),
                str(row.get("kind", "")).lower(),
            ]
        )

        row_matched = term in haystack
        if row_matched:
            matches.append(
                {
                    "repo": row.get("repo", ""),
                    "kind": row.get("kind", ""),
                    "canonical_url": canonical_url,
                    "title": title,
                    "res_id": row.get("id", ""),
                    "atom": None,
                    "fetched_ts": row.get("fetched_ts", 0.0),
                }
            )

        for atom in row.get("atoms", []):
            if not isinstance(atom, dict):
                continue
            atom_blob = json.dumps(atom, ensure_ascii=False, sort_keys=True).lower()
            if term not in atom_blob:
                continue
            matches.append(
                {
                    "repo": row.get("repo", ""),
                    "kind": row.get("kind", ""),
                    "canonical_url": canonical_url,
                    "title": title,
                    "res_id": row.get("id", ""),
                    "atom": atom,
                    "fetched_ts": row.get("fetched_ts", 0.0),
                }
            )

    matches.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("repo", "")),
            str(row.get("canonical_url", "")),
            json.dumps(row.get("atom") or {}, ensure_ascii=False, sort_keys=True),
        )
    )

    return {
        "term": term,
        "repo": repo,
        "count": len(matches),
        "matches": matches[:limit],
    }


def _query_github_recent_changes(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 360), 360)))
    limit = max(1, min(128, _safe_int(args.get("limit", 32), 32)))

    changes: list[dict[str, Any]] = []
    for row in _github_resource_rows(nodes):
        atoms = row.get("atoms", []) if isinstance(row.get("atoms", []), list) else []
        atom_kinds = {
            str(atom.get("kind", "")).strip().lower()
            for atom in atoms
            if isinstance(atom, dict)
        }
        change_kind = ""
        if row.get("kind") == "github:pr" and str(row.get("merged_at", "")).strip():
            change_kind = "pr_merged"
        elif "changes_dependency" in atom_kinds:
            change_kind = "dependency_change"
        elif row.get("kind") == "github:advisory" or "references_cve" in atom_kinds:
            change_kind = "security_signal"
        elif row.get("kind") in {"github:release", "github:issue"}:
            change_kind = str(row.get("kind", "")).replace("github:", "")

        if not change_kind:
            continue

        changes.append(
            {
                "change_kind": change_kind,
                "repo": row.get("repo", ""),
                "kind": row.get("kind", ""),
                "number": row.get("number", 0),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "fetched_ts": row.get("fetched_ts", 0.0),
                "content_hash": row.get("content_hash", ""),
            }
        )

    changes.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )
    return {
        "window_ticks": int(window_ticks),
        "count": len(changes),
        "changes": changes[:limit],
    }


def _query_web_resource_summary(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    target = str(
        args.get(
            "target",
            args.get(
                "res_id",
                args.get("resource_id", args.get("url_id", args.get("url", ""))),
            ),
        )
        or ""
    ).strip()
    if not target:
        return {"target": "", "found": None, "link_count": 0, "links": []}

    node_by_id = _node_index(nodes)
    target_lower = target.lower()
    resource_node: dict[str, Any] | None = None

    for node in nodes:
        if _node_role(node) != "web:resource":
            continue
        node_id = str(node.get("id", "") or "").strip()
        canonical_url = str(node.get("canonical_url", "") or "").strip()
        if target_lower in {node_id.lower(), canonical_url.lower()}:
            resource_node = node
            break

    if resource_node is None:
        matched_url_id = ""
        for node in nodes:
            if _node_role(node) != "web:url":
                continue
            node_id = str(node.get("id", "") or "").strip()
            canonical_url = str(node.get("canonical_url", "") or "").strip()
            if target_lower in {node_id.lower(), canonical_url.lower()}:
                matched_url_id = node_id
                break
        if matched_url_id:
            for edge in edges:
                kind = str(edge.get("kind", "") or "").strip().lower()
                if kind != "web:source_of":
                    continue
                if str(edge.get("target", "") or "").strip() != matched_url_id:
                    continue
                source_id = str(edge.get("source", "") or "").strip()
                candidate = node_by_id.get(source_id)
                if (
                    isinstance(candidate, dict)
                    and _node_role(candidate) == "web:resource"
                ):
                    resource_node = candidate
                    break

    if not isinstance(resource_node, dict):
        return {"target": target, "found": None, "link_count": 0, "links": []}

    resource_id = str(resource_node.get("id", "") or "").strip()
    links: list[dict[str, Any]] = []
    for edge in edges:
        kind = str(edge.get("kind", "") or "").strip().lower()
        if kind != "web:links_to":
            continue
        source_id = str(edge.get("source", "") or "").strip()
        if source_id != resource_id:
            continue
        target_id = str(edge.get("target", "") or "").strip()
        target_node = node_by_id.get(target_id, {})
        links.append(
            {
                "url_id": target_id,
                "canonical_url": str(
                    target_node.get("canonical_url", "") or ""
                ).strip(),
                "last_status": str(target_node.get("last_status", "") or "").strip(),
                "next_allowed_fetch_ts": round(
                    max(
                        0.0,
                        _safe_float(target_node.get("next_allowed_fetch_ts", 0.0), 0.0),
                    ),
                    6,
                ),
            }
        )
    links.sort(
        key=lambda row: (str(row.get("canonical_url", "")), str(row.get("url_id", "")))
    )

    return {
        "target": target,
        "found": {
            "res_id": resource_id,
            "canonical_url": str(resource_node.get("canonical_url", "") or "").strip(),
            "fetched_ts": round(
                max(0.0, _safe_float(resource_node.get("fetched_ts", 0.0), 0.0)),
                6,
            ),
            "content_hash": str(resource_node.get("content_hash", "") or "").strip(),
            "source_url_id": str(resource_node.get("source_url_id", "") or "").strip(),
        },
        "link_count": len(links),
        "links": links[:128],
    }


def _query_graph_summary(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    simulation: dict[str, Any] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    scope = str(args.get("scope", "all") or "all").strip().lower()
    top_n = max(1, min(64, _safe_int(args.get("n", 12), 12)))
    overview = _query_overview(nodes, edges)
    top_degree_nodes = list(overview.get("top_degree_nodes", []))[:top_n]

    filtered_nodes = nodes
    if scope in {"web", "crawler"}:
        filtered_nodes = [
            row
            for row in nodes
            if _node_role(row) in {"web:url", "web:resource"}
            or str(row.get("crawler_kind", "")).strip()
        ]
    elif scope in {"nexus", "graph"}:
        filtered_nodes = [row for row in nodes if str(row.get("id", "")).strip()]

    role_counts: dict[str, int] = {}
    for node in filtered_nodes:
        role = _node_role(node)
        role_counts[role] = role_counts.get(role, 0) + 1

    _, _, _, outcome_rows = _simulation_sections(simulation)
    outcome_counts = {"food": 0, "death": 0}
    for row in outcome_rows:
        outcome_kind = str(row.get("outcome", "") or "").strip().lower()
        if outcome_kind in outcome_counts:
            outcome_counts[outcome_kind] += 1

    return {
        "scope": scope,
        "node_count": len(filtered_nodes),
        "edge_count": len(edges),
        "role_counts": role_counts,
        "edge_kind_counts": overview.get("edge_kind_counts", {}),
        "top_nodes": top_degree_nodes,
        "outcomes": outcome_counts,
    }


def run_named_graph_query(
    nexus_graph: dict[str, Any] | None,
    query_name: str,
    *,
    args: dict[str, Any] | None = None,
    simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes, edges = _graph_rows(nexus_graph)
    query = _normalize_query_name(query_name)
    query_args = dict(args) if isinstance(args, dict) else {}

    handlers = {
        "overview": lambda: _query_overview(nodes, edges),
        "graph_summary": lambda: _query_graph_summary(
            nodes,
            edges,
            simulation,
            query_args,
        ),
        "neighbors": lambda: _query_neighbors(nodes, edges, query_args),
        "role_slice": lambda: _query_role_slice(nodes, query_args),
        "search": lambda: _query_search(nodes, query_args),
        "url_status": lambda: _query_url_status(nodes, query_args),
        "resource_for_url": lambda: _query_resource_for_url(nodes, edges, query_args),
        "recently_updated": lambda: _query_recently_updated(nodes, query_args),
        "explain_daimoi": lambda: _query_explain_daimoi(simulation, query_args),
        "recent_outcomes": lambda: _query_recent_outcomes(simulation, query_args),
        "crawler_status": lambda: _query_crawler_status(simulation),
        "arxiv_papers": lambda: _query_arxiv_papers(simulation, query_args),
        "github_status": lambda: _query_github_status(nodes, simulation),
        "github_repo_summary": lambda: _query_github_repo_summary(nodes, query_args),
        "github_find": lambda: _query_github_find(nodes, query_args),
        "github_recent_changes": lambda: _query_github_recent_changes(
            nodes,
            query_args,
        ),
        "web_resource_summary": lambda: _query_web_resource_summary(
            nodes,
            edges,
            query_args,
        ),
    }

    if query in handlers:
        result = handlers[query]()
    else:
        result = {
            "error": "unknown_query",
            "query": query,
            "supported": [
                "overview",
                "graph_summary",
                "neighbors",
                "search",
                "url_status",
                "resource_for_url",
                "recently_updated",
                "role_slice",
                "explain_daimoi",
                "recent_outcomes",
                "crawler_status",
                "arxiv_papers",
                "github_status",
                "github_repo_summary",
                "github_find",
                "github_recent_changes",
                "web_resource_summary",
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


def _facts_github_section(
    nodes: list[dict[str, Any]],
    simulation: dict[str, Any],
    *,
    invariants: list[dict[str, Any]],
) -> dict[str, Any]:
    resources = _github_resource_rows(nodes)

    crawler_graph = (
        simulation.get("crawler_graph", {}) if isinstance(simulation, dict) else {}
    )
    status = crawler_graph.get("status", {}) if isinstance(crawler_graph, dict) else {}
    github_status = status.get("github", {}) if isinstance(status, dict) else {}
    github_status = github_status if isinstance(github_status, dict) else {}

    monitored_repos = [
        str(item).strip()
        for item in github_status.get("monitored_repos", [])
        if str(item).strip()
    ]
    if not monitored_repos:
        monitored_repos = sorted(
            {
                str(row.get("repo", "")).strip()
                for row in resources
                if str(row.get("repo", "")).strip()
            }
        )

    atom_counts: dict[str, int] = {}
    atom_lookup: dict[str, dict[str, Any]] = {}
    for row in resources:
        for atom in (
            row.get("atoms", []) if isinstance(row.get("atoms", []), list) else []
        ):
            if not isinstance(atom, dict):
                continue
            token = _canonical_json(atom)
            atom_counts[token] = atom_counts.get(token, 0) + 1
            if token not in atom_lookup:
                atom_lookup[token] = atom

    top_atoms = sorted(
        atom_counts.items(),
        key=lambda item: (-_safe_int(item[1], 0), str(item[0])),
    )

    github_invariants = [
        row
        for row in invariants
        if isinstance(row, dict)
        and str(row.get("kind", "")).strip().lower().startswith("github_")
    ]

    return {
        "monitored_repos": monitored_repos,
        "recent_resources": [
            {
                "repo": row.get("repo", ""),
                "kind": row.get("kind", ""),
                "number": row.get("number", 0),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "updated_at": row.get("updated_at", ""),
                "fetched_ts": row.get("fetched_ts", 0.0),
                "content_hash": row.get("content_hash", ""),
                "importance_score": row.get("importance_score", 0),
            }
            for row in resources[:24]
        ],
        "top_atoms": [
            {
                "atom": atom_lookup.get(token, {}),
                "count": count,
            }
            for token, count in top_atoms[:20]
        ],
        "invariants_violations": github_invariants,
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
    allowed_github_kinds = {
        "github:repo",
        "github:pr",
        "github:issue",
        "github:release",
        "github:advisory",
        "github:compare",
        "github:diff",
        "github:file",
    }

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
            canonical_url = _node_canonical_url(node)
            kind_value = str(_node_value(node, "kind", "") or "").strip().lower()
            is_github_resource = (
                "github.com/" in canonical_url.lower()
                or "raw.githubusercontent.com/" in canonical_url.lower()
                or kind_value.startswith("github:")
            )
            if is_github_resource and kind_value not in allowed_github_kinds:
                violations.append(
                    {
                        "kind": "github_resource_kind_invalid",
                        "resource_id": node_id,
                        "value": kind_value,
                    }
                )
        if role == "web:url":
            canonical_url = _node_canonical_url(node)
            if not canonical_url:
                violations.append(
                    {"kind": "url_missing_canonical_url", "url_id": node_id}
                )
            next_allowed = _safe_float(
                _node_value(node, "next_allowed_fetch_ts", 0.0), 0.0
            )
            fail_count = _safe_int(_node_value(node, "fail_count", 0), 0)
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


def _facts_table_rows(
    simulation: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    dynamics, _, field_particles, outcome_rows = _simulation_sections(simulation)
    node_by_id = _node_index(nodes)
    tick_hint = max(0, _safe_int(dynamics.get("tick", 0), 0))

    node_table = [
        {
            "node_id": str(node.get("id", "") or "").strip(),
            "role": _node_role(node),
            "status": str(node.get("status", "") or "").strip(),
            "label": str(node.get("label", "") or "").strip(),
            "canonical_url": str(node.get("canonical_url", "") or "").strip(),
            "importance": round(_safe_float(node.get("importance", 0.0), 0.0), 6),
        }
        for node in nodes
        if str(node.get("id", "") or "").strip()
    ]
    node_table.sort(key=lambda row: str(row.get("node_id", "")))

    edge_table = [
        {
            "src": str(edge.get("source", "") or "").strip(),
            "role": str(edge.get("kind", "") or "").strip().lower() or "relates",
            "dst": str(edge.get("target", "") or "").strip(),
            "weight": round(_safe_float(edge.get("weight", 0.0), 0.0), 6),
        }
        for edge in edges
        if str(edge.get("source", "") or "").strip()
        and str(edge.get("target", "") or "").strip()
    ]
    edge_table.sort(
        key=lambda row: (
            str(row.get("src", "")),
            str(row.get("role", "")),
            str(row.get("dst", "")),
        )
    )

    daimoi_table = [
        {
            "daimoi_id": str(row.get("id", "") or "").strip(),
            "owner": str(
                row.get(
                    "presence_id",
                    row.get("owner_presence_id", row.get("owner", "")),
                )
                or ""
            ).strip(),
            "intent": str(row.get("top_job", "") or "").strip(),
            "born_tick": 0,
            "ttl": max(0, _safe_int(row.get("ttl", row.get("ttl_steps", 0)), 0)),
            "age": max(0, _safe_int(row.get("age", 0), 0)),
            "graph_node_id": str(row.get("graph_node_id", "") or "").strip(),
            "route_node_id": str(row.get("route_node_id", "") or "").strip(),
            "message_probability": round(
                max(0.0, _safe_float(row.get("message_probability", 0.0), 0.0)),
                6,
            ),
        }
        for row in field_particles
        if str(row.get("id", "") or "").strip()
    ]
    daimoi_table.sort(key=lambda row: str(row.get("daimoi_id", "")))

    event_collision_table = []
    for row in field_particles:
        daimoi_id = str(row.get("id", "") or "").strip()
        if not daimoi_id:
            continue
        collisions = max(
            0,
            _safe_int(row.get("collision_count", row.get("collisions", 0)), 0),
        )
        if collisions <= 0:
            continue
        event_collision_table.append(
            {
                "tick": max(0, _safe_int(row.get("age", tick_hint), tick_hint)),
                "daimoi_id": daimoi_id,
                "target_id": str(row.get("graph_node_id", "") or "").strip(),
                "collision_count": collisions,
            }
        )
    event_collision_table.sort(
        key=lambda row: (
            _safe_int(row.get("tick", 0), 0),
            str(row.get("daimoi_id", "")),
        )
    )

    outcome_table: list[dict[str, Any]] = []
    for row in outcome_rows:
        outcome_kind = str(row.get("outcome", "") or "").strip().lower()
        if outcome_kind not in {"food", "death"}:
            continue
        outcome_table.append(
            {
                "tick": max(0, _safe_int(row.get("tick", row.get("seq", 0)), 0)),
                "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
                "target_id": str(row.get("graph_node_id", "") or "").strip(),
                "reason": str(row.get("reason", "") or "").strip(),
                "intensity": round(
                    max(0.0, _safe_float(row.get("intensity", 0.0), 0.0)),
                    6,
                ),
                "outcome": outcome_kind,
            }
        )
    outcome_table.sort(
        key=lambda row: (
            _safe_int(row.get("tick", 0), 0),
            str(row.get("daimoi_id", "")),
            str(row.get("outcome", "")),
        )
    )

    event_timeout_table = [
        {
            "tick": max(0, _safe_int(row.get("tick", 0), 0)),
            "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
            "reason": str(row.get("reason", "") or "").strip(),
        }
        for row in outcome_table
        if str(row.get("outcome", "")).strip() == "death"
        and any(
            token in str(row.get("reason", "")).strip().lower()
            for token in ("timeout", "ttl")
        )
    ]

    heartbeat = (
        dynamics.get("resource_heartbeat", {})
        if isinstance(dynamics.get("resource_heartbeat", {}), dict)
        else {}
    )
    devices = heartbeat.get("devices", {}) if isinstance(heartbeat, dict) else {}
    capacity_table = []
    if isinstance(devices, dict):
        for device_id, payload in devices.items():
            device_row = payload if isinstance(payload, dict) else {}
            utilization = max(
                0.0,
                min(
                    100.0,
                    _safe_float(
                        device_row.get(
                            "utilization", device_row.get("usage_percent", 0.0)
                        ),
                        0.0,
                    ),
                ),
            )
            capacity_table.append(
                {
                    "tick": tick_hint,
                    "target_id": f"device:{str(device_id).strip().lower()}",
                    "cap": 100.0,
                    "used": round(utilization, 6),
                }
            )
    capacity_table.sort(key=lambda row: str(row.get("target_id", "")))

    web_url_table = [
        {
            "url_id": str(node.get("id", "") or "").strip(),
            "canonical_url": str(node.get("canonical_url", "") or "").strip(),
            "next_allowed_fetch_ts": round(
                max(0.0, _safe_float(node.get("next_allowed_fetch_ts", 0.0), 0.0)),
                6,
            ),
            "last_fetch_ts": round(
                max(0.0, _safe_float(node.get("last_fetch_ts", 0.0), 0.0)),
                6,
            ),
            "fail_count": max(0, _safe_int(node.get("fail_count", 0), 0)),
            "last_status": str(node.get("last_status", "") or "").strip(),
        }
        for node in nodes
        if _node_role(node) == "web:url"
    ]
    web_url_table.sort(key=lambda row: str(row.get("url_id", "")))

    web_resource_table = [
        {
            "res_id": str(node.get("id", "") or "").strip(),
            "canonical_url": str(node.get("canonical_url", "") or "").strip(),
            "content_hash": str(node.get("content_hash", "") or "").strip(),
            "fetched_ts": round(
                max(0.0, _safe_float(node.get("fetched_ts", 0.0), 0.0)),
                6,
            ),
            "source_url_id": str(node.get("source_url_id", "") or "").strip(),
        }
        for node in nodes
        if _node_role(node) == "web:resource"
    ]
    web_resource_table.sort(key=lambda row: str(row.get("res_id", "")))

    web_link_table = []
    for edge in edges:
        if str(edge.get("kind", "") or "").strip().lower() != "web:links_to":
            continue
        source_id = str(edge.get("source", "") or "").strip()
        target_id = str(edge.get("target", "") or "").strip()
        if not source_id or not target_id:
            continue
        source_node = node_by_id.get(source_id, {})
        target_node = node_by_id.get(target_id, {})
        if (
            _node_role(source_node) != "web:resource"
            or _node_role(target_node) != "web:url"
        ):
            continue
        web_link_table.append({"res_id": source_id, "url_id": target_id})
    web_link_table.sort(
        key=lambda row: (str(row.get("res_id", "")), str(row.get("url_id", "")))
    )

    food_table = [
        {
            "tick": max(0, _safe_int(row.get("tick", 0), 0)),
            "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
            "target_id": str(row.get("target_id", "") or "").strip(),
        }
        for row in outcome_table
        if str(row.get("outcome", "")).strip() == "food"
    ]
    death_table = [
        {
            "tick": max(0, _safe_int(row.get("tick", 0), 0)),
            "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
        }
        for row in outcome_table
        if str(row.get("outcome", "")).strip() == "death"
    ]

    return {
        "node": node_table,
        "edge": edge_table,
        "daimoi": daimoi_table,
        "event_collision": event_collision_table,
        "event_timeout": event_timeout_table,
        "capacity": capacity_table,
        "web_url": web_url_table,
        "web_resource": web_resource_table,
        "web_link": web_link_table,
        "food": food_table,
        "death": death_table,
    }


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        _canonical_json(row if isinstance(row, dict) else {})
        for row in rows
        if isinstance(row, dict)
    ]
    path.write_text(
        ("\n".join(lines) + "\n") if lines else "",
        "utf-8",
    )


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
    github_section = _facts_github_section(nodes, payload, invariants=invariants)
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
    tables = _facts_table_rows(payload, nodes=nodes, edges=edges)

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
        "github": github_section,
        "dynamics": {
            "daimoi_outcomes": {
                "food": max(0, _safe_int(outcomes.get("food", 0), 0)),
                "death": max(0, _safe_int(outcomes.get("death", 0), 0)),
                "total": max(0, _safe_int(outcomes.get("total", 0), 0)),
            }
        },
        "tables": {
            table_name: {
                "row_count": len(table_rows),
                "path": "",
            }
            for table_name, table_rows in tables.items()
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
        table_dir = out_dir / "current"
        table_dir.mkdir(parents=True, exist_ok=True)
        table_meta = facts.get("tables", {})
        if isinstance(table_meta, dict):
            for table_name, table_rows in tables.items():
                table_path = table_dir / f"{table_name}.jsonl"
                _write_jsonl_rows(table_path, table_rows)
                row_meta = table_meta.get(table_name)
                if isinstance(row_meta, dict):
                    row_meta["path"] = str(table_path)
    facts["snapshot_path"] = snapshot_path
    return facts
