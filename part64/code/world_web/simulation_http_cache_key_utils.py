"""Simulation HTTP cache key helpers."""

from __future__ import annotations

import hashlib
from typing import Any

from .metrics import _safe_float


def simulation_http_cache_key(
    *,
    perspective: str,
    catalog: dict[str, Any],
    queue_snapshot: dict[str, Any],
    influence_snapshot: dict[str, Any],
    cache_ignore_queue: bool,
    cache_ignore_influence: bool,
    config_version: int,
) -> str:
    file_graph = catalog.get("file_graph", {}) if isinstance(catalog, dict) else {}
    crawler_graph = (
        catalog.get("crawler_graph", {}) if isinstance(catalog, dict) else {}
    )
    file_stats = file_graph.get("stats", {}) if isinstance(file_graph, dict) else {}
    crawler_stats = (
        crawler_graph.get("stats", {}) if isinstance(crawler_graph, dict) else {}
    )

    file_count = int(_safe_float(file_stats.get("file_count", 0), 0.0))
    file_edge_count = int(_safe_float(file_stats.get("edge_count", 0), 0.0))
    crawler_count = int(_safe_float(crawler_stats.get("crawler_count", 0), 0.0))
    crawler_edge_count = int(_safe_float(crawler_stats.get("edge_count", 0), 0.0))

    if file_count <= 0:
        file_nodes = (
            file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
        )
        if isinstance(file_nodes, list):
            file_count = len(file_nodes)
    if file_edge_count <= 0:
        file_edges = file_graph.get("edges", []) if isinstance(file_graph, dict) else []
        if isinstance(file_edges, list):
            file_edge_count = len(file_edges)
    if crawler_count <= 0:
        crawler_nodes = (
            crawler_graph.get("crawler_nodes", [])
            if isinstance(crawler_graph, dict)
            else []
        )
        if isinstance(crawler_nodes, list):
            crawler_count = len(crawler_nodes)
    if crawler_edge_count <= 0:
        crawler_edges = (
            crawler_graph.get("edges", []) if isinstance(crawler_graph, dict) else []
        )
        if isinstance(crawler_edges, list):
            crawler_edge_count = len(crawler_edges)

    fingerprint = (
        f"{max(0, file_count)}:{max(0, file_edge_count)}:"
        f"{max(0, crawler_count)}:{max(0, crawler_edge_count)}"
    )
    if fingerprint == "0:0:0:0":
        file_graph_generated_at = (
            str(file_graph.get("generated_at", "")).strip()
            if isinstance(file_graph, dict)
            else ""
        )
        crawler_generated_at = (
            str(crawler_graph.get("generated_at", "")).strip()
            if isinstance(crawler_graph, dict)
            else ""
        )
        fingerprint = f"ts:{file_graph_generated_at}:{crawler_generated_at}"

    queue_pending = 0
    queue_events = 0
    if not cache_ignore_queue:
        queue_pending = int(_safe_float(queue_snapshot.get("pending_count", 0), 0.0))
        queue_events = int(_safe_float(queue_snapshot.get("event_count", 0), 0.0))
    clicks_recent = 0
    user_inputs_recent = 0
    user_signal = ""
    if not cache_ignore_influence:
        clicks_recent = int(_safe_float(influence_snapshot.get("clicks_45s", 0), 0.0))
        user_inputs_recent = int(
            _safe_float(influence_snapshot.get("user_inputs_120s", 0), 0.0)
        )
        user_rows = (
            influence_snapshot.get("recent_user_inputs", [])
            if isinstance(influence_snapshot, dict)
            else []
        )
        if isinstance(user_rows, list) and user_rows:
            newest = user_rows[0] if isinstance(user_rows[0], dict) else {}
            user_signal = "|".join(
                [
                    str(newest.get("kind", "")),
                    str(newest.get("target", "")),
                    str(newest.get("message", ""))[:48],
                    str(newest.get("x_ratio", "")),
                    str(newest.get("y_ratio", "")),
                ]
            )
    user_signal_hash = hashlib.sha1(user_signal.encode("utf-8")).hexdigest()[:10]
    config_snapshot = max(0, int(_safe_float(config_version, 0.0)))
    return (
        f"{perspective}|{fingerprint}|"
        f"q:{max(0, queue_pending)}:{max(0, queue_events)}|"
        f"i:{max(0, clicks_recent)}:{max(0, user_inputs_recent)}:{user_signal_hash}|"
        f"cfg:{config_snapshot}|simulation"
    )
