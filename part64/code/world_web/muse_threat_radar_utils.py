from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any


def muse_threat_radar_status(*, server_module: Any) -> dict[str, Any]:
    with server_module._MUSE_THREAT_RADAR_LOCK:
        state = dict(server_module._MUSE_THREAT_RADAR_STATE)
    return {
        "enabled": bool(server_module._MUSE_THREAT_RADAR_ENABLED),
        "muse_id": server_module._MUSE_THREAT_RADAR_MUSE_ID,
        "label": server_module._MUSE_THREAT_RADAR_LABEL,
        "interval_seconds": round(server_module._MUSE_THREAT_RADAR_INTERVAL_SECONDS, 3),
        "token_budget": int(server_module._MUSE_THREAT_RADAR_TOKEN_BUDGET),
        "prompt": server_module._MUSE_THREAT_RADAR_PROMPT,
        "state": state,
    }


def update_active_threats(
    *,
    server_module: Any,
    radar: str,
    result: dict[str, Any],
) -> None:
    """Update active threat cache for muse context injection."""
    if not isinstance(result, dict):
        return
    threats = [row for row in result.get("threats", []) if isinstance(row, dict)][
        : server_module._MUSE_ACTIVE_THREATS_MAX_PER_RADAR
    ]

    now_iso = datetime.now(timezone.utc).isoformat()
    radar_key = "local" if radar in {"local", "github"} else "global"
    muse_tags = (
        ["witness_thread", "github_security_review"]
        if radar_key == "local"
        else ["chaos"]
    )
    hot_repos = [
        {
            "repo": row.get("repo", ""),
            "max_risk_score": row.get("max_risk_score", 0),
        }
        for row in (result.get("hot_repos") or [])
        if isinstance(row, dict)
    ][: server_module._MUSE_ACTIVE_THREATS_MAX_PER_RADAR]

    watch_sources = [
        {
            "url": row.get("url", ""),
            "kind": row.get("kind", ""),
            "title": row.get("title", ""),
        }
        for row in (result.get("sources") or [])
        if isinstance(row, dict)
    ][: server_module._MUSE_ACTIVE_THREATS_MAX_PER_RADAR]

    context_nodes: list[dict[str, Any]] = []
    for idx, threat in enumerate(threats):
        risk_level = str(threat.get("risk_level", "low") or "low").upper()
        risk_score = max(0, server_module._safe_int(threat.get("risk_score", 0), 0))
        title = str(threat.get("title", "") or threat.get("kind", "threat")).strip()[
            :180
        ]
        kind = str(threat.get("kind", "") or "unknown").strip()
        canonical_url = str(threat.get("canonical_url", "") or "").strip()
        repo = str(threat.get("repo", "") or "").strip()
        domain = str(threat.get("domain", "") or "").strip()
        cves = [
            str(cve).upper() for cve in (threat.get("cves") or []) if str(cve).strip()
        ][:4]
        signals = [
            str(s).strip()
            for s in (threat.get("signals") or threat.get("labels") or [])
            if str(s).strip()
        ][:6]

        hash_source = canonical_url or f"{title}|{kind}|{repo}|{domain}|{idx}"
        url_hash = hashlib.sha1(hash_source.encode("utf-8")).hexdigest()[:8]
        node_id = f"threat:{radar}:{idx}:{url_hash}"
        label = f"[{risk_level}:{risk_score}] {title[:80]}"
        text_parts = [f"Threat: {title}", f"Risk: {risk_level} ({risk_score})"]
        if repo:
            text_parts.append(f"Repo: {repo}")
        if domain:
            text_parts.append(f"Domain: {domain}")
        if kind:
            text_parts.append(f"Kind: {kind}")
        if cves:
            text_parts.append(f"CVEs: {', '.join(cves[:3])}")
        if canonical_url:
            text_parts.append(f"URL: {canonical_url}")
        if signals:
            text_parts.append(f"Signals: {' '.join(signals[:4])}")

        context_nodes.append(
            {
                "id": node_id,
                "kind": "threat",
                "label": label,
                "text": "\n".join(text_parts),
                "x": 0.5 + (idx * 0.02),
                "y": 0.3 + (idx * 0.05),
                "visibility": "public",
                "tags": [
                    radar,
                    f"risk_{risk_level.lower()}",
                    kind,
                    "threat",
                    *muse_tags,
                    *signals[:2],
                ],
                "risk_score": risk_score,
                "risk_level": risk_level,
                "radar": radar,
                "ts": now_iso,
            }
        )

    if radar_key == "local":
        for idx, hot in enumerate(hot_repos[:4]):
            repo = str(hot.get("repo", "") or "").strip()
            if not repo:
                continue
            score = max(0, server_module._safe_int(hot.get("max_risk_score", 0), 0))
            repo_hash = hashlib.sha1(repo.encode("utf-8")).hexdigest()[:8]
            context_nodes.append(
                {
                    "id": f"threat-source:repo:{repo_hash}",
                    "kind": "threat-source",
                    "label": f"Hot Repo [{score}] {repo}",
                    "text": f"Hot repo by threat score: {repo} (max_risk_score={score})",
                    "x": 0.62,
                    "y": 0.18 + (idx * 0.06),
                    "visibility": "public",
                    "tags": ["local", "hot-repo", "threat", *muse_tags],
                    "risk_score": score,
                    "risk_level": "HOT",
                    "radar": "local",
                    "ts": now_iso,
                }
            )
    else:
        for idx, source in enumerate(watch_sources[:4]):
            url = str(source.get("url", "") or "").strip()
            title = str(source.get("title", "") or source.get("kind", "source")).strip()
            if not url and not title:
                continue
            source_key = url or title
            source_hash = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:8]
            context_nodes.append(
                {
                    "id": f"threat-source:site:{source_hash}",
                    "kind": "threat-source",
                    "label": f"Hot Site {title[:70]}",
                    "text": (
                        f"Watch source: {title}. URL: {url}"
                        if url
                        else f"Watch source: {title}"
                    ),
                    "x": 0.62,
                    "y": 0.18 + (idx * 0.06),
                    "visibility": "public",
                    "tags": ["global", "hot-site", "threat", *muse_tags],
                    "risk_score": 4,
                    "risk_level": "WATCH",
                    "radar": "global",
                    "ts": now_iso,
                }
            )

    with server_module._MUSE_ACTIVE_THREATS_LOCK:
        server_module._MUSE_ACTIVE_THREATS[radar_key] = {
            "threats": context_nodes,
            "hot_repos": hot_repos if radar_key == "local" else [],
            "hot_domains": hot_repos if radar_key == "global" else [],
            "watch_sources": watch_sources if radar_key == "global" else [],
            "updated_at": now_iso,
        }


def get_active_threat_nodes(*, server_module: Any, radar: str) -> list[dict[str, Any]]:
    radar_key = "local" if radar in {"local", "github"} else "global"
    with server_module._MUSE_ACTIVE_THREATS_LOCK:
        data = dict(server_module._MUSE_ACTIVE_THREATS.get(radar_key, {}))
    return data.get("threats", []) if isinstance(data.get("threats"), list) else []


def muse_threat_radar_tick(
    *,
    handler: Any,
    server_module: Any,
    now_monotonic: float | None = None,
    force: bool = False,
    reason: str = "manual",
) -> dict[str, Any]:
    now_mono = (
        max(0.0, server_module._safe_float(now_monotonic, 0.0))
        if now_monotonic is not None
        else time.monotonic()
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    if not server_module._MUSE_THREAT_RADAR_ENABLED:
        with server_module._MUSE_THREAT_RADAR_LOCK:
            server_module._MUSE_THREAT_RADAR_STATE["last_skipped_reason"] = "disabled"
        return {
            "ok": True,
            "status": "disabled",
            "runtime": handler._muse_threat_radar_status(),
        }

    with server_module._MUSE_THREAT_RADAR_LOCK:
        next_due = max(
            0.0,
            server_module._safe_float(
                server_module._MUSE_THREAT_RADAR_STATE.get("next_run_monotonic", 0.0),
                0.0,
            ),
        )
        if not force and now_mono < next_due:
            server_module._MUSE_THREAT_RADAR_STATE["last_skipped_reason"] = (
                "interval_not_elapsed"
            )
            skipped = True
        else:
            skipped = False
            server_module._MUSE_THREAT_RADAR_STATE["next_run_monotonic"] = round(
                now_mono + server_module._MUSE_THREAT_RADAR_INTERVAL_SECONDS,
                6,
            )
            server_module._MUSE_THREAT_RADAR_STATE["last_run_monotonic"] = round(
                now_mono,
                6,
            )
            server_module._MUSE_THREAT_RADAR_STATE["last_run_at"] = now_iso
            server_module._MUSE_THREAT_RADAR_STATE["last_reason"] = str(
                reason or "manual"
            )
            server_module._MUSE_THREAT_RADAR_STATE["last_skipped_reason"] = ""

    if skipped:
        return {
            "ok": True,
            "status": "skipped",
            "reason": "interval_not_elapsed",
            "runtime": handler._muse_threat_radar_status(),
        }

    manager = handler._muse_manager()
    runtime = manager.snapshot()
    muse_rows = runtime.get("muses", []) if isinstance(runtime, dict) else []
    muse_ids = {
        str(row.get("id", "")).strip()
        for row in muse_rows
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }
    if server_module._MUSE_THREAT_RADAR_MUSE_ID not in muse_ids:
        manager.create_muse(
            muse_id=server_module._MUSE_THREAT_RADAR_MUSE_ID,
            label=server_module._MUSE_THREAT_RADAR_LABEL,
            anchor={"x": 0.82, "y": 0.5, "zoom": 1.0, "kind": "threat-radar"},
            user_intent_id="runtime:threat-radar-bootstrap",
        )

    bucket = int(now_mono / max(1.0, server_module._MUSE_THREAT_RADAR_INTERVAL_SECONDS))
    idempotency_key = (
        f"threat-radar:{bucket}"
        if not force
        else f"threat-radar:force:{time.time_ns()}"
    )
    try:
        handler._muse_tool_cache = {}
        catalog = handler._runtime_catalog_base()
        graph_revision = str(catalog.get("generated_at", "") or "").strip()
        payload = manager.send_message(
            muse_id=server_module._MUSE_THREAT_RADAR_MUSE_ID,
            text=server_module._MUSE_THREAT_RADAR_PROMPT,
            mode="deterministic",
            token_budget=server_module._MUSE_THREAT_RADAR_TOKEN_BUDGET,
            idempotency_key=idempotency_key,
            graph_revision=graph_revision,
            surrounding_nodes=[],
            tool_callback=handler._muse_tool_callback,
            reply_builder=handler._muse_reply_builder,
            seed=f"threat-radar|{bucket}",
        )
    except Exception as exc:
        error_text = f"{exc.__class__.__name__}:{exc}"
        with server_module._MUSE_THREAT_RADAR_LOCK:
            server_module._MUSE_THREAT_RADAR_STATE["last_status"] = "error"
            server_module._MUSE_THREAT_RADAR_STATE["last_turn_id"] = ""
            server_module._MUSE_THREAT_RADAR_STATE["last_error"] = error_text
        return {
            "ok": False,
            "status": "error",
            "error": error_text,
            "runtime": handler._muse_threat_radar_status(),
        }

    ok = bool(payload.get("ok", False)) if isinstance(payload, dict) else False
    turn_id = str(payload.get("turn_id", "") or "") if isinstance(payload, dict) else ""
    error = str(payload.get("error", "") or "") if isinstance(payload, dict) else ""

    with server_module._MUSE_THREAT_RADAR_LOCK:
        server_module._MUSE_THREAT_RADAR_STATE["last_status"] = "ok" if ok else "error"
        server_module._MUSE_THREAT_RADAR_STATE["last_turn_id"] = turn_id
        server_module._MUSE_THREAT_RADAR_STATE["last_error"] = error

    return {
        "ok": ok,
        "status": "triggered" if ok else "error",
        "muse_id": server_module._MUSE_THREAT_RADAR_MUSE_ID,
        "turn_id": turn_id,
        "error": error,
        "runtime": handler._muse_threat_radar_status(),
    }
