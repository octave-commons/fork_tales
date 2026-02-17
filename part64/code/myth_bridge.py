from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


MYTH_EVENT_TYPE = "cover_field_presence"


def decay_ledger(
    ledger: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    decayed: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in ledger.items():
        next_value = dict(value)
        next_value["buzz"] = float(next_value.get("buzz", 0.0)) * 0.90
        next_value["tradition"] = float(next_value.get("tradition", 0.0)) * 0.995
        decayed[key] = next_value
    return decayed


def add_mention(
    ledger: dict[tuple[str, str], dict[str, Any]],
    mention: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    event_type = str(mention.get("event-type", "unknown"))
    claim = str(mention.get("claim", "unknown"))
    key = (event_type, claim)
    weight = float(mention.get("weight", 0.0))
    event_instance = mention.get("event-instance")

    updated = dict(ledger)
    row = dict(updated.get(key, {}))
    row["buzz"] = float(row.get("buzz", 0.0)) + weight
    row["tradition"] = float(row.get("tradition", 0.0)) + (
        0.12 * math.log(1.0 + weight)
    )
    row["mentions"] = int(row.get("mentions", 0)) + 1

    if event_instance:
        instances = set(row.get("event-instances", set()))
        instances.add(event_instance)
        row["event-instances"] = instances

    updated[key] = row
    return updated


def attribution(
    ledger: dict[tuple[str, str], dict[str, Any]], event_type: str
) -> dict[str, float]:
    rows: list[tuple[str, float]] = []
    for (current_event_type, claim), row in ledger.items():
        if current_event_type == event_type:
            rows.append((claim, float(row.get("tradition", 0.0))))

    total = max(sum(value for _claim, value in rows), 1.0e-9)
    return {claim: (value / total) for claim, value in rows}


def build_mentions_from_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    cover_fields = catalog.get("cover_fields", [])
    for cover in cover_fields:
        mentions.append(
            {
                "event-type": MYTH_EVENT_TYPE,
                "claim": str(cover.get("id", "unknown")),
                "weight": 1.0,
                "event-instance": f"{cover.get('part', 'part')}-{cover.get('id', 'unknown')}",
            }
        )

    counts = catalog.get("counts", {})
    for kind in ("audio", "image", "video"):
        value = int(counts.get(kind, 0))
        if value > 0:
            mentions.append(
                {
                    "event-type": "media_presence",
                    "claim": kind,
                    "weight": float(min(value, 12)) / 12.0,
                }
            )

    return mentions


def fetch_remote_myth_snapshot() -> dict[str, Any] | None:
    source_url = str(os.getenv("MYTH_BRIDGE_URL", "")).strip()
    if not source_url:
        return None

    req = Request(source_url, method="GET")
    try:
        with urlopen(req, timeout=0.8) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if isinstance(payload, dict):
                return payload
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None

    return None


class MythStateTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ledger: dict[tuple[str, str], dict[str, Any]] = {}

    def snapshot(self, catalog: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._ledger = decay_ledger(self._ledger)
            for mention in build_mentions_from_catalog(catalog):
                self._ledger = add_mention(self._ledger, mention)

            cover_probs = attribution(self._ledger, MYTH_EVENT_TYPE)
            media_probs = attribution(self._ledger, "media_presence")

            top_cover_claim = ""
            top_cover_weight = 0.0
            if cover_probs:
                top_cover_claim, top_cover_weight = max(
                    cover_probs.items(), key=lambda item: item[1]
                )

            remote_snapshot = fetch_remote_myth_snapshot()

            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "event_type": MYTH_EVENT_TYPE,
                "ledger_size": len(self._ledger),
                "top_cover_claim": top_cover_claim,
                "top_cover_weight": round(float(top_cover_weight), 6),
                "cover_attribution": cover_probs,
                "media_attribution": media_probs,
                "remote": remote_snapshot,
            }
