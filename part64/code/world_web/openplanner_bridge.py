from __future__ import annotations

import json
import os
import time
import urllib.error
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any
from urllib.request import Request, urlopen


OPENPLANNER_URL = str(
    os.getenv("OPENPLANNER_URL", "http://127.0.0.1:7777") or "http://127.0.0.1:7777"
).rstrip("/")
OPENPLANNER_API_KEY = str(
    os.getenv("OPENPLANNER_API_KEY", "change-me") or "change-me"
).strip()
OPENPLANNER_PROJECT = str(
    os.getenv("OPENPLANNER_PROJECT", "eta-mu") or "eta-mu"
).strip()
OPENPLANNER_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("OPENPLANNER_TIMEOUT_SECONDS", "3.5") or "3.5"),
)
OPENPLANNER_ENABLED = str(
    os.getenv("OPENPLANNER_ENABLED", "1") or "1"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

OPENPLANNER_ACTIVE_PRESENCE_IDS = {
    "receipt_river",
    "witness_thread",
    "fork_tax_canticle",
    "mage_of_receipts",
    "keeper_of_receipts",
    "anchor_registry",
    "gates_of_truth",
    "file_sentinel",
    "change_fog",
    "path_ward",
    "manifest_lith",
    "core_pulse",
    "health_sentinel_cpu",
    "health_sentinel_gpu1",
    "health_sentinel_gpu2",
    "health_sentinel_npu0",
    "presence.user.operator",
    "github_security_review",
    "chaos",
}


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENPLANNER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _openplanner_request(
    path: str,
    *,
    payload: dict[str, Any],
    timeout_s: float | None = None,
) -> tuple[bool, dict[str, Any], int, str]:
    if not OPENPLANNER_ENABLED:
        return False, {}, 0, "disabled"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(
        f"{OPENPLANNER_URL}{path}",
        data=body,
        headers=_auth_headers(),
        method="POST",
    )
    try:
        with urlopen(
            req, timeout=max(0.2, float(timeout_s or OPENPLANNER_TIMEOUT_SECONDS))
        ) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip() else {}
            return (
                True,
                parsed if isinstance(parsed, dict) else {},
                int(response.status),
                "",
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except Exception:
            parsed = {}
        return (
            False,
            parsed if isinstance(parsed, dict) else {},
            int(exc.code),
            f"http:{exc.code}",
        )
    except Exception as exc:
        return False, {}, 0, f"error:{exc.__class__.__name__}"


def _event_id(prefix: str, parts: list[str]) -> str:
    payload = "|".join(parts)
    return f"{prefix}:{sha1(payload.encode('utf-8')).hexdigest()[:20]}"


def ingest_openplanner_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    clean_events = [row for row in events if isinstance(row, dict)]
    if not clean_events:
        return {"ok": False, "error": "no_events"}
    ok, body, status, error = _openplanner_request(
        "/v1/events",
        payload={"events": clean_events},
    )
    return {
        "ok": ok,
        "status": status,
        "error": error,
        "count": int(body.get("count", len(clean_events)))
        if isinstance(body, dict)
        else len(clean_events),
        "body": body if isinstance(body, dict) else {},
    }


def build_user_input_openplanner_events(
    processed_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in processed_events:
        if not isinstance(row, dict):
            continue
        row_id = str(row.get("id", "")).strip()
        ts = str(row.get("ts", "")).strip() or datetime.now(timezone.utc).isoformat()
        kind = str(row.get("kind", "input")).strip() or "input"
        target = str(row.get("target", "simulation")).strip() or "simulation"
        message = str(row.get("message", "")).strip() or f"{kind} on {target}"
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        output.append(
            {
                "schema": "openplanner.event.v1",
                "id": _event_id("eta-mu-user", [row_id, ts, kind, target, message]),
                "ts": ts,
                "source": "eta-mu-user",
                "kind": kind,
                "source_ref": {
                    "project": OPENPLANNER_PROJECT,
                    "session": str(
                        row.get("presence_id", "presence.user.operator")
                        or "presence.user.operator"
                    ),
                    "message": row_id,
                },
                "text": message,
                "meta": {
                    "role": "user",
                    "author": "operator",
                    "target": target,
                    "record": str(
                        row.get("record", "ημ.user-input.v1") or "ημ.user-input.v1"
                    ),
                    "tags": ["eta-mu", "user-input", kind],
                },
                "extra": {
                    "eta_mu": row,
                    "meta": meta,
                },
            }
        )
    return output


def build_presence_say_openplanner_event(
    *,
    presence_id: str,
    text: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    clean_presence_id = str(presence_id or "").strip()
    clean_text = str(text or "").strip()
    if not clean_presence_id or not clean_text:
        return None
    if (
        clean_presence_id not in OPENPLANNER_ACTIVE_PRESENCE_IDS
        and not clean_presence_id.startswith("presence:concept:")
    ):
        return None
    rendered = str(payload.get("rendered_text", "")).strip() or clean_text
    ts = (
        str(payload.get("generated_at", "")).strip()
        or datetime.now(timezone.utc).isoformat()
    )
    return {
        "schema": "openplanner.event.v1",
        "id": _event_id(
            "eta-mu-presence", [clean_presence_id, ts, clean_text, rendered]
        ),
        "ts": ts,
        "source": "eta-mu-presence",
        "kind": "presence_say",
        "source_ref": {
            "project": OPENPLANNER_PROJECT,
            "session": clean_presence_id,
            "message": str(
                payload.get("presence_id", clean_presence_id) or clean_presence_id
            ),
        },
        "text": rendered,
        "meta": {
            "role": "assistant",
            "author": clean_presence_id,
            "tags": ["eta-mu", "presence", "say"],
        },
        "extra": {
            "user_text": clean_text,
            "say_intent": payload.get("say_intent", {}),
            "presence_name": payload.get("presence_name", {}),
        },
    }


def build_muse_event_openplanner_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return None
    muse_id = str(event.get("muse_id", "")).strip()
    if not muse_id:
        return None
    ts = str(event.get("ts", "")).strip() or datetime.now(timezone.utc).isoformat()
    kind = str(event.get("kind", "muse.event")).strip() or "muse.event"
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    text = str(
        payload.get("reply") or payload.get("summary") or payload.get("node_id") or kind
    ).strip()
    if not text:
        text = kind
    return {
        "schema": "openplanner.event.v1",
        "id": _event_id(
            "eta-mu-muse", [str(event.get("event_id", "")), muse_id, ts, kind, text]
        ),
        "ts": ts,
        "source": "eta-mu-muse",
        "kind": kind.replace(".", "_"),
        "source_ref": {
            "project": OPENPLANNER_PROJECT,
            "session": muse_id,
            "message": str(event.get("event_id", "") or ""),
            "turn": str(event.get("turn_id", "") or ""),
        },
        "text": text,
        "meta": {
            "role": "assistant",
            "author": muse_id,
            "tags": ["eta-mu", "muse", kind],
        },
        "extra": {
            "status": str(event.get("status", "ok") or "ok"),
            "payload": payload,
        },
    }


def search_openplanner_memory(
    *,
    query_text: str,
    session: str,
    limit: int = 4,
    vector: bool = True,
) -> dict[str, Any]:
    clean_query = str(query_text or "").strip()
    clean_session = str(session or "").strip()
    if not clean_query:
        return {"ok": False, "error": "query_required", "rows": []}

    payload: dict[str, Any] = {
        "q": clean_query,
        "limit": max(1, min(12, int(limit))),
        "project": OPENPLANNER_PROJECT,
    }
    if clean_session:
        payload["session"] = clean_session

    path = "/v1/search/vector" if vector else "/v1/search/fts"
    ok, body, status, error = _openplanner_request(path, payload=payload)
    rows: list[dict[str, Any]] = []
    if vector and isinstance(body, dict):
        result = body.get("result") if isinstance(body.get("result"), dict) else {}
        ids = result.get("ids") if isinstance(result.get("ids"), list) else []
        docs = (
            result.get("documents") if isinstance(result.get("documents"), list) else []
        )
        metas = (
            result.get("metadatas") if isinstance(result.get("metadatas"), list) else []
        )
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if docs and isinstance(docs[0], list):
            docs = docs[0]
        if metas and isinstance(metas[0], list):
            metas = metas[0]
        for idx, item_id in enumerate(ids):
            rows.append(
                {
                    "id": str(item_id or ""),
                    "text": str(docs[idx] if idx < len(docs) else ""),
                    "meta": metas[idx]
                    if idx < len(metas) and isinstance(metas[idx], dict)
                    else {},
                }
            )
    elif isinstance(body, dict):
        raw_rows = body.get("rows") if isinstance(body.get("rows"), list) else []
        rows = [row for row in raw_rows if isinstance(row, dict)]
    return {
        "ok": ok,
        "status": status,
        "error": error,
        "rows": rows,
    }


def summarize_openplanner_memory(
    *,
    query_text: str,
    session: str,
    limit: int = 3,
) -> list[str]:
    response = search_openplanner_memory(
        query_text=query_text,
        session=session,
        limit=limit,
        vector=True,
    )
    if not bool(response.get("ok", False)):
        response = search_openplanner_memory(
            query_text=query_text,
            session=session,
            limit=limit,
            vector=False,
        )
    rows = response.get("rows") if isinstance(response.get("rows"), list) else []
    summary: list[str] = []
    for row in rows[: max(1, min(6, int(limit)))]:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("snippet") or "").strip()
        if not text:
            continue
        compact = " ".join(text.split())
        summary.append(compact[:160])
    return summary


def ingest_single_muse_event(event: dict[str, Any]) -> dict[str, Any]:
    row = build_muse_event_openplanner_event(event)
    if row is None:
        return {"ok": False, "error": "invalid_muse_event"}
    return ingest_openplanner_events([row])
