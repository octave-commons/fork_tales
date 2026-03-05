from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any, Callable

from .muse_runtime import _slugify as _muse_slugify


def _normalize_muse_id(raw: Any) -> str:
    return _muse_slugify(str(raw or "").strip(), fallback="")


def _muse_error_status(error: str) -> HTTPStatus:
    return HTTPStatus.NOT_FOUND if error == "muse_not_found" else HTTPStatus.BAD_REQUEST


def _header_value(headers: Any, name: str) -> str:
    try:
        return str(headers.get(name, "") or "").strip()
    except Exception:
        return ""


def _normalize_threat_radar(raw: str) -> str:
    radar = str(raw or "").strip().lower()
    if radar in {"github", "local"}:
        return "local"
    if radar in {"global", "hormuz"}:
        return radar
    if radar in {"cyber", "regime"}:
        return "cyber"
    return ""


def _threat_row_is_github_like(row: dict[str, Any]) -> bool:
    kind_value = str(row.get("kind", "") or "").strip().lower()
    if kind_value.startswith("github:"):
        return True
    canonical_url = str(row.get("canonical_url", "") or "").strip().lower()
    if (
        "github.com/" in canonical_url
        or "githubusercontent.com/" in canonical_url
        or "githubassets.com/" in canonical_url
    ):
        return True
    domain_value = str(row.get("domain", "") or "").strip().lower()
    if (
        domain_value == "github.com"
        or domain_value.endswith(".github.com")
        or domain_value == "githubusercontent.com"
        or domain_value.endswith(".githubusercontent.com")
        or domain_value == "githubassets.com"
        or domain_value.endswith(".githubassets.com")
    ):
        return True
    return False


def handle_muse_threat_report_get_route(
    *,
    handler: Any,
    path: str,
    params: dict[str, list[str]],
    send_json: Callable[..., None],
    server_module: Any,
) -> bool:
    parsed_path = str(path or "").strip()
    if parsed_path != "/api/muse/threat-radar/report":
        return False

    window_ticks = max(
        1,
        min(
            20_000,
            int(
                server_module._safe_float(
                    str(params.get("window_ticks", ["1440"])[0] or "1440"),
                    1440.0,
                )
            ),
        ),
    )
    limit = max(
        1,
        min(
            128,
            int(
                server_module._safe_float(
                    str(params.get("limit", ["24"])[0] or "24"),
                    24.0,
                )
            ),
        ),
    )
    requested_radar = str(params.get("radar", ["global"])[0] or "").strip().lower()
    radar = _normalize_threat_radar(requested_radar)
    if not radar:
        send_json(
            {
                "ok": False,
                "error": "invalid_radar",
                "supported": ["local", "global", "hormuz", "cyber", "regime"],
            },
            status=HTTPStatus.BAD_REQUEST,
        )
        return True

    repo = str(params.get("repo", [""])[0] or "").strip().lower()
    kind = str(params.get("kind", [""])[0] or "").strip().lower()
    state_bins = max(
        3,
        min(
            48,
            int(
                server_module._safe_float(
                    str(params.get("state_bins", ["8"])[0] or "8"),
                    8.0,
                )
            ),
        ),
    )
    threat_limit = max(
        16,
        min(
            512,
            int(
                server_module._safe_float(
                    str(
                        params.get("threat_limit", [str(max(64, limit * 4))])[0]
                        or str(max(64, limit * 4))
                    ),
                    float(max(64, limit * 4)),
                )
            ),
        ),
    )
    apply_regime_threshold = server_module._safe_bool_query(
        str(params.get("apply_regime_threshold", ["true"])[0] or "true"),
        default=True,
    )
    since_snapshot_hash = str(params.get("since_snapshot_hash", [""])[0] or "").strip()

    catalog = handler._runtime_catalog_base(
        allow_inline_collect=False,
        strict_collect=False,
    )
    file_graph = catalog.get("file_graph", {}) if isinstance(catalog, dict) else {}
    crawler_graph = (
        catalog.get("crawler_graph", {}) if isinstance(catalog, dict) else {}
    )
    logical_graph = (
        catalog.get("logical_graph", {}) if isinstance(catalog, dict) else {}
    )

    query_args: dict[str, Any] = {
        "window_ticks": window_ticks,
        "limit": limit,
    }
    query_name = "github_threat_radar"
    if radar == "local":
        min_weak_label_score = max(
            -16,
            min(
                16,
                int(
                    server_module._safe_float(
                        str(params.get("min_weak_label_score", ["1"])[0] or "1"),
                        1.0,
                    )
                ),
            ),
        )
        if repo:
            query_args["repo"] = repo
        query_args["min_weak_label_score"] = int(min_weak_label_score)
    elif radar == "global":
        query_name = "geopolitical_news_radar"
        query_args["include_provisional"] = False
        domain = str(params.get("domain", [""])[0] or "").strip().lower()
        if domain:
            query_args["domain"] = domain
        if kind:
            query_args["kind"] = kind
    elif radar == "hormuz":
        query_name = "hormuz_threat_radar"
        if kind:
            query_args["kind"] = kind
    else:
        query_name = "cyber_risk_radar"
        min_weak_label_score = max(
            -16,
            min(
                16,
                int(
                    server_module._safe_float(
                        str(params.get("min_weak_label_score", ["1"])[0] or "1"),
                        1.0,
                    )
                ),
            ),
        )
        if repo:
            query_args["repo"] = repo
        query_args["state_bins"] = state_bins
        query_args["threat_limit"] = threat_limit
        query_args["apply_regime_threshold"] = bool(apply_regime_threshold)
        query_args["min_weak_label_score"] = int(min_weak_label_score)

    def _canonical_nexus_for_threat_radar(*, include_logical: bool) -> dict[str, Any]:
        payload = server_module.simulation_module._build_canonical_nexus_graph(
            file_graph if isinstance(file_graph, dict) else None,
            crawler_graph if isinstance(crawler_graph, dict) else None,
            logical_graph
            if include_logical and isinstance(logical_graph, dict)
            else None,
            include_crawler=True,
            include_logical=include_logical,
        )
        if not isinstance(payload, dict):
            return {"nodes": [], "edges": []}
        return payload

    prefer_crawler_direct = radar in {"local", "hormuz"}
    nexus_graph: dict[str, Any]
    if prefer_crawler_direct and isinstance(crawler_graph, dict):
        crawler_nodes_raw = crawler_graph.get("nodes", [])
        if not isinstance(crawler_nodes_raw, list) or not crawler_nodes_raw:
            crawler_nodes_raw = crawler_graph.get("crawler_nodes", [])
        crawler_edges_raw = crawler_graph.get("edges", [])
        crawler_nodes = (
            [row for row in crawler_nodes_raw if isinstance(row, dict)]
            if isinstance(crawler_nodes_raw, list)
            else []
        )
        crawler_edges = (
            [row for row in crawler_edges_raw if isinstance(row, dict)]
            if isinstance(crawler_edges_raw, list)
            else []
        )
        if crawler_nodes:
            nexus_graph = {
                "nodes": crawler_nodes,
                "edges": crawler_edges,
            }
        else:
            nexus_graph = _canonical_nexus_for_threat_radar(include_logical=False)
    else:
        nexus_graph = _canonical_nexus_for_threat_radar(
            include_logical=(radar == "cyber")
        )

    simulation_payload: dict[str, Any] = {
        "nexus_graph": nexus_graph,
        "generated_at": str(
            catalog.get("generated_at", "") if isinstance(catalog, dict) else ""
        ),
    }
    if isinstance(crawler_graph, dict):
        simulation_payload["crawler_graph"] = crawler_graph

    cache_key_payload = {
        "radar": radar,
        "query": query_name,
        "args": query_args,
        "catalog_generated_at": simulation_payload.get("generated_at", ""),
    }
    cache_key = hashlib.sha256(
        json.dumps(
            cache_key_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    cached_payload: dict[str, Any] | None = None
    with server_module._MUSE_THREAT_RADAR_REPORT_CACHE_LOCK:
        cached_row = server_module._MUSE_THREAT_RADAR_REPORT_CACHE.get(cache_key)
        if isinstance(cached_row, dict):
            cached_payload = {
                "snapshot_hash": str(cached_row.get("snapshot_hash", "") or ""),
                "result": cached_row.get("result", {}),
            }

    query_payload = cached_payload
    if not isinstance(query_payload, dict):
        query_payload = server_module.run_named_graph_query(
            nexus_graph,
            query_name,
            args=query_args,
            simulation=simulation_payload,
        )
        with server_module._MUSE_THREAT_RADAR_REPORT_CACHE_LOCK:
            server_module._MUSE_THREAT_RADAR_REPORT_CACHE[cache_key] = {
                "snapshot_hash": str(
                    query_payload.get("snapshot_hash", "") or ""
                ).strip(),
                "result": query_payload.get("result", {}),
                "updated_monotonic": round(time.monotonic(), 6),
            }
            if (
                len(server_module._MUSE_THREAT_RADAR_REPORT_CACHE)
                > server_module._MUSE_THREAT_RADAR_REPORT_CACHE_MAX
            ):
                overflow = (
                    len(server_module._MUSE_THREAT_RADAR_REPORT_CACHE)
                    - server_module._MUSE_THREAT_RADAR_REPORT_CACHE_MAX
                )
                oldest_keys = sorted(
                    server_module._MUSE_THREAT_RADAR_REPORT_CACHE.items(),
                    key=lambda item: server_module._safe_float(
                        item[1].get("updated_monotonic", 0.0),
                        0.0,
                    ),
                )[: max(0, overflow)]
                for key, _value in oldest_keys:
                    server_module._MUSE_THREAT_RADAR_REPORT_CACHE.pop(key, None)

    result = (
        query_payload.get("result", {})
        if isinstance(query_payload.get("result", {}), dict)
        else {}
    )

    if isinstance(result, dict):
        raw_threat_rows = (
            result.get("threats", [])
            if isinstance(result.get("threats", []), list)
            else []
        )
        threat_rows = [row for row in raw_threat_rows if isinstance(row, dict)]
        scoped_rows = threat_rows
        if radar == "local":
            scoped_rows = [
                row for row in threat_rows if _threat_row_is_github_like(row)
            ]
        elif radar in {"global", "hormuz"}:
            scoped_rows = [
                row for row in threat_rows if not _threat_row_is_github_like(row)
            ]

        if len(scoped_rows) != len(threat_rows):
            result = dict(result)
            result["threats"] = scoped_rows

            level_counts = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }
            for row in scoped_rows:
                level = str(row.get("risk_level", "") or "").strip().lower()
                if level == "critical":
                    level_counts["critical"] += 1
                elif level == "high":
                    level_counts["high"] += 1
                elif level == "medium":
                    level_counts["medium"] += 1
                else:
                    level_counts["low"] += 1

            result["count"] = len(scoped_rows)
            result["critical_count"] = int(level_counts["critical"])
            result["high_count"] = int(level_counts["high"])
            result["medium_count"] = int(level_counts["medium"])
            result["low_count"] = int(level_counts["low"])

    if radar == "global" and isinstance(result, dict):
        global_rows_raw = result.get("threats", [])
        global_rows = (
            [row for row in global_rows_raw if isinstance(row, dict)]
            if isinstance(global_rows_raw, list)
            else []
        )
        provisional_count = sum(
            1 for row in global_rows if bool(row.get("provisional", False))
        )
        seed_only = len(global_rows) > 0 and provisional_count == len(global_rows)
        now_iso = datetime.now(timezone.utc).isoformat()
        with server_module._MUSE_THREAT_RADAR_LOCK:
            prior_streak = max(
                0,
                server_module._safe_int(
                    server_module._MUSE_THREAT_RADAR_STATE.get(
                        "global_seed_only_streak",
                        0,
                    ),
                    0,
                ),
            )
            streak = prior_streak + 1 if seed_only else 0
            alert = bool(
                streak >= server_module._MUSE_THREAT_RADAR_GLOBAL_SEED_ONLY_ALERT_STREAK
            )
            server_module._MUSE_THREAT_RADAR_STATE["global_seed_only_streak"] = int(
                streak
            )
            server_module._MUSE_THREAT_RADAR_STATE["global_seed_only_alert"] = bool(
                alert
            )
            if global_rows and not seed_only:
                server_module._MUSE_THREAT_RADAR_STATE[
                    "last_non_provisional_global_at"
                ] = now_iso

        result = dict(result)
        quality = result.get("quality", {})
        quality_dict = quality if isinstance(quality, dict) else {}
        quality_dict = dict(quality_dict)
        quality_dict["seed_only"] = bool(seed_only)
        quality_dict["seed_only_streak"] = int(streak)
        quality_dict["seed_only_alert"] = bool(alert)
        quality_dict["needs_crawl_evidence"] = bool(
            quality_dict.get("needs_crawl_evidence", False) or seed_only
        )
        result["quality"] = quality_dict
        result["provisional_count"] = int(provisional_count)
        result["non_provisional_count"] = int(
            max(0, len(global_rows) - provisional_count)
        )

    resolved_snapshot_hash = str(
        query_payload.get("snapshot_hash", "")
        if isinstance(query_payload, dict)
        else ""
    ).strip()
    runtime_status = handler._muse_threat_radar_status()
    if isinstance(runtime_status, dict):
        runtime_status = dict(runtime_status)
        runtime_status["scope"] = radar
        if radar == "local":
            runtime_status["label"] = "Local Cyber Threat Radar"
        elif radar == "global":
            runtime_status["label"] = "Global Geopolitical Feed"
        elif radar == "hormuz":
            runtime_status["label"] = "Hormuz Maritime Watch"
        else:
            runtime_status["label"] = "Cyber Risk Radar"

    if (
        since_snapshot_hash
        and resolved_snapshot_hash
        and since_snapshot_hash == resolved_snapshot_hash
    ):
        send_json(
            {
                "ok": True,
                "record": "eta-mu.muse-threat-radar-report.v1",
                "snapshot_hash": resolved_snapshot_hash,
                "not_modified": True,
                "radar": radar,
                "query": query_name,
                "runtime": runtime_status,
            }
        )
        return True

    handler._update_active_threats(radar, result)
    send_json(
        {
            "ok": True,
            "record": "eta-mu.muse-threat-radar-report.v1",
            "snapshot_hash": resolved_snapshot_hash,
            "radar": radar,
            "query": query_name,
            "runtime": runtime_status,
            "result": result,
        }
    )
    return True


def handle_muse_get_route(
    *,
    handler: Any,
    path: str,
    params: dict[str, list[str]],
    send_json: Callable[..., None],
    server_module: Any,
) -> bool:
    parsed_path = str(path or "").strip()

    if parsed_path == "/api/muse/runtime":
        manager = handler._muse_manager()
        send_json({"ok": True, "runtime": manager.snapshot()})
        return True

    if parsed_path == "/api/muse/threat-radar/status":
        send_json(
            {
                "ok": True,
                "record": "eta-mu.muse-threat-radar-status.v1",
                "runtime": handler._muse_threat_radar_status(),
            }
        )
        return True

    if parsed_path == "/api/muse/threat-radar/tick":
        force = server_module._safe_bool_query(
            str(params.get("force", ["false"])[0] or "false"),
            default=False,
        )
        payload = handler._muse_threat_radar_tick(
            force=force,
            reason="api.tick",
        )
        status = (
            HTTPStatus.OK if bool(payload.get("ok", False)) else HTTPStatus.BAD_GATEWAY
        )
        send_json(payload, status=status)
        return True

    if parsed_path == "/api/muse/events":
        manager = handler._muse_manager()
        muse_id = _normalize_muse_id(str(params.get("muse_id", [""])[0] or ""))
        since_seq = max(
            0,
            int(
                server_module._safe_float(
                    str(params.get("since_seq", ["0"])[0] or "0"),
                    0.0,
                )
            ),
        )
        limit = max(
            1,
            min(
                512,
                int(
                    server_module._safe_float(
                        str(params.get("limit", ["96"])[0] or "96"),
                        96.0,
                    )
                ),
            ),
        )
        events = manager.list_events(
            muse_id=muse_id,
            since_seq=since_seq,
            limit=limit,
        )
        next_seq = since_seq
        if events:
            next_seq = max(
                int(row.get("seq", since_seq))
                for row in events
                if isinstance(row, dict)
            )
        send_json(
            {
                "ok": True,
                "record": "eta-mu.muse-event-page.v1",
                "muse_id": muse_id,
                "since_seq": since_seq,
                "next_seq": next_seq,
                "events": events,
            }
        )
        return True

    if parsed_path == "/api/muse/context":
        manager = handler._muse_manager()
        muse_id = _normalize_muse_id(str(params.get("muse_id", [""])[0] or ""))
        turn_id = str(params.get("turn_id", [""])[0] or "").strip()
        if not muse_id or not turn_id:
            send_json(
                {"ok": False, "error": "muse_id_and_turn_id_required"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return True
        manifest = manager.get_context_manifest(muse_id, turn_id)
        if not isinstance(manifest, dict):
            send_json(
                {"ok": False, "error": "context_manifest_not_found"},
                status=HTTPStatus.NOT_FOUND,
            )
            return True
        send_json({"ok": True, "manifest": manifest})
        return True

    return False


def handle_muse_post_route(
    *,
    handler: Any,
    path: str,
    read_json_body: Callable[[], dict[str, Any] | None],
    send_json: Callable[..., None],
    headers: Any,
    server_module: Any,
) -> bool:
    parsed_path = str(path or "").strip()

    if parsed_path == "/api/muse/create":
        req = read_json_body() or {}
        manager = handler._muse_manager()
        payload = manager.create_muse(
            muse_id=str(req.get("muse_id", "") or "").strip(),
            label=str(req.get("label", "") or "").strip(),
            anchor=req.get("anchor") if isinstance(req.get("anchor"), dict) else None,
            user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
        )
        status = (
            HTTPStatus.OK if bool(payload.get("ok", False)) else HTTPStatus.BAD_REQUEST
        )
        send_json(payload, status=status)
        return True

    if parsed_path == "/api/muse/pause":
        req = read_json_body() or {}
        manager = handler._muse_manager()
        muse_id = _normalize_muse_id(str(req.get("muse_id", "") or ""))
        paused = server_module._safe_bool_query(
            str(req.get("paused", "true") or "true"),
            default=True,
        )
        payload = manager.set_pause(
            muse_id,
            paused=paused,
            reason=str(req.get("reason", "") or "").strip(),
            user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
        )
        if not bool(payload.get("ok", False)):
            send_json(
                payload,
                status=_muse_error_status(str(payload.get("error", ""))),
            )
            return True
        send_json(payload)
        return True

    if parsed_path == "/api/muse/pin":
        req = read_json_body() or {}
        manager = handler._muse_manager()
        payload = manager.pin_node(
            _normalize_muse_id(str(req.get("muse_id", "") or "")),
            node_id=str(req.get("node_id", "") or "").strip(),
            user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
            reason=str(req.get("reason", "") or "").strip(),
        )
        if not bool(payload.get("ok", False)):
            send_json(
                payload,
                status=_muse_error_status(str(payload.get("error", ""))),
            )
            return True
        send_json(payload)
        return True

    if parsed_path == "/api/muse/unpin":
        req = read_json_body() or {}
        manager = handler._muse_manager()
        payload = manager.unpin_node(
            _normalize_muse_id(str(req.get("muse_id", "") or "")),
            node_id=str(req.get("node_id", "") or "").strip(),
            user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
        )
        if not bool(payload.get("ok", False)):
            send_json(
                payload,
                status=_muse_error_status(str(payload.get("error", ""))),
            )
            return True
        send_json(payload)
        return True

    if parsed_path == "/api/muse/bind-nexus":
        req = read_json_body() or {}
        manager = handler._muse_manager()
        payload = manager.bind_nexus(
            _normalize_muse_id(str(req.get("muse_id", "") or "")),
            nexus_id=str(req.get("nexus_id", "") or "").strip(),
            reason=str(req.get("reason", "") or "").strip(),
            user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
        )
        if not bool(payload.get("ok", False)):
            send_json(
                payload,
                status=_muse_error_status(str(payload.get("error", ""))),
            )
            return True
        send_json(payload)
        return True

    if parsed_path == "/api/muse/sync-pins":
        req = read_json_body() or {}
        manager = handler._muse_manager()
        pinned_node_ids = req.get("pinned_node_ids", [])
        payload = manager.sync_workspace_pins(
            _normalize_muse_id(str(req.get("muse_id", "") or "")),
            pinned_node_ids=pinned_node_ids
            if isinstance(pinned_node_ids, list)
            else [],
            reason=str(req.get("reason", "") or "").strip(),
            user_intent_id=str(req.get("user_intent_id", "") or "").strip(),
        )
        if not bool(payload.get("ok", False)):
            send_json(
                payload,
                status=_muse_error_status(str(payload.get("error", ""))),
            )
            return True
        send_json(payload)
        return True

    if parsed_path == "/api/muse/message":
        req = read_json_body() or {}
        manager = handler._muse_manager()
        handler._muse_tool_cache = {}
        catalog = handler._runtime_catalog_base()
        graph_revision = str(
            req.get("graph_revision", catalog.get("generated_at", ""))
            or catalog.get("generated_at", "")
        ).strip()
        idempotency_key = str(
            req.get("idempotency_key", _header_value(headers, "Idempotency-Key")) or ""
        ).strip()
        surrounding_nodes = req.get("surrounding_nodes", [])
        if not isinstance(surrounding_nodes, list):
            surrounding_nodes = []

        muse_id = _normalize_muse_id(str(req.get("muse_id", "") or ""))
        if not muse_id:
            send_json(
                {"ok": False, "error": "muse_id_required"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return True

        if muse_id in {"witness_thread", "github_security_review"}:
            threat_nodes = handler._get_active_threat_nodes("local")
            if threat_nodes:
                surrounding_nodes = list(surrounding_nodes) + threat_nodes
        elif muse_id == "chaos":
            threat_nodes = handler._get_active_threat_nodes("global")
            if threat_nodes:
                surrounding_nodes = list(surrounding_nodes) + threat_nodes

        payload = manager.send_message(
            muse_id=muse_id,
            text=str(req.get("text", "") or "").strip(),
            mode=str(req.get("mode", "stochastic") or "stochastic").strip(),
            token_budget=max(
                320,
                min(
                    8192,
                    int(
                        server_module._safe_float(
                            str(req.get("token_budget", 2048) or 2048),
                            2048.0,
                        )
                    ),
                ),
            ),
            idempotency_key=idempotency_key,
            graph_revision=graph_revision,
            surrounding_nodes=surrounding_nodes,
            tool_callback=handler._muse_tool_callback,
            reply_builder=handler._muse_reply_builder,
            seed=str(req.get("seed", "") or "").strip(),
        )
        if not bool(payload.get("ok", False)):
            status_code = int(payload.get("status_code", HTTPStatus.BAD_REQUEST))
            send_json(payload, status=status_code)
            return True
        send_json(payload)
        return True

    return False
