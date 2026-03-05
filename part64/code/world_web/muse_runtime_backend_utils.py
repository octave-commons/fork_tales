from __future__ import annotations

from typing import Any


def muse_tool_callback(
    *,
    handler: Any,
    tool_name: str,
    server_module: Any,
) -> dict[str, Any]:
    clean_tool = str(tool_name or "").strip().lower()
    cache = getattr(handler, "_muse_tool_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(handler, "_muse_tool_cache", cache)

    def _cached_simulation() -> dict[str, Any]:
        cached = cache.get("simulation")
        if isinstance(cached, dict):
            return cached
        simulation_payload = server_module.build_simulation_state(
            handler._collect_catalog_fast()
        )
        cache["simulation"] = simulation_payload
        return simulation_payload

    if clean_tool == "facts_snapshot":
        payload = cache.get("facts_snapshot")
        if not isinstance(payload, dict):
            payload = server_module.build_facts_snapshot(
                _cached_simulation(),
                part_root=handler.part_root,
            )
            cache["facts_snapshot"] = payload
        counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
        node_count = max(
            0,
            int(
                server_module._safe_float(
                    sum(
                        int(server_module._safe_float(value, 0.0))
                        for value in (
                            counts.get("nodes_by_role", {}).values()
                            if isinstance(counts.get("nodes_by_role", {}), dict)
                            else []
                        )
                    ),
                    0.0,
                )
            ),
        )
        edge_count = max(
            0,
            int(
                server_module._safe_float(
                    sum(
                        int(server_module._safe_float(value, 0.0))
                        for value in (
                            counts.get("edges_by_kind", {}).values()
                            if isinstance(counts.get("edges_by_kind", {}), dict)
                            else []
                        )
                    ),
                    0.0,
                )
            ),
        )
        return {
            "ok": True,
            "summary": "facts snapshot generated",
            "record": str(payload.get("record", "")),
            "snapshot_hash": str(payload.get("snapshot_hash", "")),
            "snapshot_path": str(payload.get("snapshot_path", "")),
            "node_count": node_count,
            "edge_count": edge_count,
        }
    if (
        clean_tool.startswith("graph:")
        or clean_tool.startswith("graph_query:")
        or clean_tool == "graph_query"
    ):
        graph_tail = ""
        if clean_tool in {"graph", "graph_query"}:
            graph_tail = "overview"
        elif clean_tool.startswith("graph_query:"):
            graph_tail = str(clean_tool.split(":", 1)[1]).strip()
        elif clean_tool.startswith("graph:"):
            graph_tail = str(clean_tool.split(":", 1)[1]).strip()
        tail_parts = [piece for piece in graph_tail.split(" ") if piece]
        query_name = str(tail_parts[0] if tail_parts else "").strip().lower()
        query_arg = " ".join(tail_parts[1:]).strip()
        if not query_name:
            query_name = "overview"
        query_args: dict[str, Any] = {}
        if query_name in {"neighbors", "node_neighbors"} and query_arg:
            query_args["node_id"] = query_arg
        elif query_name == "search" and query_arg:
            query_args["q"] = query_arg
        elif query_name in {"url_status", "resource_for_url"} and query_arg:
            query_args["target"] = query_arg
        elif query_name == "recently_updated" and query_arg:
            query_args["limit"] = max(
                1, int(server_module._safe_float(query_arg, 24.0))
            )
        elif query_name == "role_slice" and query_arg:
            query_args["role"] = query_arg
        elif query_name == "explain_daimoi" and query_arg:
            query_args["daimoi_id"] = query_arg
        elif query_name == "recent_outcomes":
            if query_arg:
                parts = [piece for piece in query_arg.split(" ") if piece]
                if parts:
                    query_args["window_ticks"] = max(
                        1,
                        int(server_module._safe_float(parts[0], 360.0)),
                    )
                if len(parts) > 1:
                    query_args["limit"] = max(
                        1,
                        int(server_module._safe_float(parts[1], 48.0)),
                    )
        elif query_name in {"web_resource_summary", "graph_summary"} and query_arg:
            if query_name == "web_resource_summary":
                query_args["target"] = query_arg
            else:
                parts = [piece for piece in query_arg.split(" ") if piece]
                if parts:
                    query_args["scope"] = parts[0]
                if len(parts) > 1:
                    query_args["n"] = max(
                        1,
                        int(server_module._safe_float(parts[1], 12.0)),
                    )
        elif query_name == "arxiv_papers" and query_arg:
            parts = [piece for piece in query_arg.split(" ") if piece]
            if parts:
                query_args["limit"] = max(
                    1,
                    int(server_module._safe_float(parts[0], 8.0)),
                )
        elif query_name == "github_repo_summary" and query_arg:
            query_args["repo"] = query_arg
        elif query_name == "github_find" and query_arg:
            parts = [piece for piece in query_arg.split(" ") if piece]
            if parts:
                query_args["term"] = parts[0]
            if len(parts) > 1:
                query_args["repo"] = parts[1]
            if len(parts) > 2:
                query_args["limit"] = max(
                    1,
                    int(server_module._safe_float(parts[2], 24.0)),
                )
        elif query_name == "github_recent_changes" and query_arg:
            parts = [piece for piece in query_arg.split(" ") if piece]
            if parts:
                query_args["window_ticks"] = max(
                    1,
                    int(server_module._safe_float(parts[0], 360.0)),
                )
            if len(parts) > 1:
                query_args["limit"] = max(
                    1,
                    int(server_module._safe_float(parts[1], 32.0)),
                )
        elif query_name == "github_threat_radar" and query_arg:
            parts = [piece for piece in query_arg.split(" ") if piece]
            if parts:
                if parts[0].isdigit():
                    query_args["window_ticks"] = max(
                        1,
                        int(server_module._safe_float(parts[0], 1440.0)),
                    )
                else:
                    query_args["repo"] = parts[0]
            if len(parts) > 1:
                if parts[1].isdigit():
                    query_args["limit"] = max(
                        1,
                        int(server_module._safe_float(parts[1], 24.0)),
                    )
                elif "repo" not in query_args:
                    query_args["repo"] = parts[1]
            if len(parts) > 2 and "repo" not in query_args:
                query_args["repo"] = parts[2]
        elif query_name == "hormuz_threat_radar" and query_arg:
            parts = [piece for piece in query_arg.split(" ") if piece]
            if parts and parts[0].isdigit():
                query_args["window_ticks"] = max(
                    1,
                    int(server_module._safe_float(parts[0], 1440.0)),
                )
            if len(parts) > 1 and parts[1].isdigit():
                query_args["limit"] = max(
                    1,
                    int(server_module._safe_float(parts[1], 24.0)),
                )
            for token in parts:
                if not token.isdigit():
                    query_args["kind"] = token.strip().lower()
                    break
        elif query_name == "multi_threat_radar" and query_arg:
            parts = [piece for piece in query_arg.split(" ") if piece]
            if parts and parts[0].isdigit():
                query_args["window_ticks"] = max(
                    1,
                    int(server_module._safe_float(parts[0], 1440.0)),
                )
            if len(parts) > 1 and parts[1].isdigit():
                query_args["limit"] = max(
                    1,
                    int(server_module._safe_float(parts[1], 24.0)),
                )
        elif query_name in {"cyber_regime_state", "cyber_regime", "regime"}:
            parts = [piece for piece in query_arg.split(" ") if piece]
            numeric_parts = [piece for piece in parts if piece.isdigit()]
            if numeric_parts:
                query_args["window_ticks"] = max(
                    1,
                    int(server_module._safe_float(numeric_parts[0], 1440.0)),
                )
            if len(numeric_parts) > 1:
                query_args["state_bins"] = max(
                    3,
                    int(server_module._safe_float(numeric_parts[1], 8.0)),
                )
            for token in parts:
                lowered = str(token or "").strip().lower()
                if not lowered or lowered.isdigit():
                    continue
                query_args["repo"] = lowered
                break
        elif query_name in {"cyber_risk_radar", "regime_radar"}:
            parts = [piece for piece in query_arg.split(" ") if piece]
            numeric_parts = [piece for piece in parts if piece.isdigit()]
            if numeric_parts:
                query_args["window_ticks"] = max(
                    1,
                    int(server_module._safe_float(numeric_parts[0], 1440.0)),
                )
            if len(numeric_parts) > 1:
                query_args["limit"] = max(
                    1,
                    int(server_module._safe_float(numeric_parts[1], 24.0)),
                )
            if len(numeric_parts) > 2:
                query_args["state_bins"] = max(
                    3,
                    int(server_module._safe_float(numeric_parts[2], 8.0)),
                )
            if len(numeric_parts) > 3:
                query_args["threat_limit"] = max(
                    16,
                    int(server_module._safe_float(numeric_parts[3], 256.0)),
                )
            for token in parts:
                lowered = str(token or "").strip().lower()
                if not lowered:
                    continue
                if lowered in {"true", "yes", "on", "false", "no", "off"}:
                    query_args["apply_regime_threshold"] = (
                        server_module._safe_bool_query(
                            lowered,
                            default=True,
                        )
                    )
                    continue
                if lowered.isdigit():
                    continue
                if "repo" not in query_args:
                    query_args["repo"] = lowered
        simulation = _cached_simulation()
        nexus_graph = (
            simulation.get("nexus_graph", {})
            if isinstance(simulation.get("nexus_graph", {}), dict)
            else {}
        )
        payload = server_module.run_named_graph_query(
            nexus_graph,
            query_name,
            args=query_args,
            simulation=simulation,
        )
        result = payload.get("result", {})
        if isinstance(result, dict) and result.get("error"):
            return {
                "ok": False,
                "error": str(result.get("error", "unknown_query")),
                "query": query_name,
            }
        result_count = 0
        if isinstance(result, dict):
            for count_key in (
                "count",
                "neighbor_count",
                "node_count",
                "edge_count",
            ):
                if count_key in result:
                    result_count = max(
                        result_count,
                        int(server_module._safe_float(result.get(count_key, 0), 0.0)),
                    )
        return {
            "ok": True,
            "summary": f"graph query {query_name} generated",
            "query": query_name,
            "snapshot_hash": str(payload.get("snapshot_hash", "")),
            "result_count": int(result_count),
            "result": result if isinstance(result, dict) else {},
        }
    if clean_tool == "study_snapshot":
        payload = server_module.build_study_snapshot(
            handler.part_root,
            handler.vault_root,
            queue_snapshot=handler.task_queue.snapshot(include_pending=True),
            council_snapshot=handler.council_chamber.snapshot(
                include_decisions=True,
                limit=16,
            ),
            drift_payload=server_module.build_drift_scan_payload(
                handler.part_root,
                handler.vault_root,
            ),
            truth_gate_blocked=None,
            resource_snapshot=server_module._resource_monitor_snapshot(
                part_root=handler.part_root
            ),
        )
        return {
            "ok": True,
            "summary": "study snapshot generated",
            "record": str(payload.get("record", "")),
        }
    if clean_tool == "drift_scan":
        payload = server_module.build_drift_scan_payload(
            handler.part_root, handler.vault_root
        )
        return {
            "ok": True,
            "summary": "drift scan generated",
            "blocked_gates": len(payload.get("blocked_gates", [])),
        }
    if clean_tool == "push_truth_dry_run":
        payload = server_module.build_push_truth_dry_run_payload(
            handler.part_root,
            handler.vault_root,
        )
        gate = payload.get("gate", {}) if isinstance(payload, dict) else {}
        return {
            "ok": True,
            "summary": "push-truth dry run generated",
            "blocked": bool(gate.get("blocked", False))
            if isinstance(gate, dict)
            else False,
        }
    return {"ok": False, "error": "unsupported_tool"}


def muse_reply_builder(
    *,
    handler: Any,
    messages: list[dict[str, Any]],
    context_block: str,
    mode: str,
    muse_id: str = "",
    turn_id: str = "",
    server_module: Any,
) -> dict[str, Any]:
    del turn_id
    from .muse_runtime import _muse_system_prompt

    model_mode = "canonical" if str(mode).strip().lower() == "deterministic" else "llm"
    clean_muse_id = str(muse_id or "").strip() or "witness_thread"

    muse_prompt = _muse_system_prompt(clean_muse_id)
    if muse_prompt:
        context_block = f"{muse_prompt}\n\n{context_block}"

    response = server_module.build_chat_reply(
        messages=[
            {"role": "system", "text": context_block},
            *messages,
        ],
        mode=model_mode,
        context=server_module.build_world_payload(handler.part_root),
        multi_entity=True,
        presence_ids=[clean_muse_id],
    )
    if not isinstance(response, dict):
        return {"reply": "", "mode": "canonical", "model": None}
    return {
        "reply": str(response.get("reply", "") or "").strip(),
        "mode": str(response.get("mode", model_mode) or model_mode),
        "model": response.get("model"),
    }
