from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def build_shared_stream_bootstrap_payload(*, perspective: str) -> dict[str, Any]:
    timestamp_value = datetime.now(timezone.utc).isoformat()
    return {
        "type": "simulation",
        "simulation": {
            "ok": True,
            "generated_at": timestamp_value,
            "timestamp": timestamp_value,
            "perspective": perspective,
            "total": 0,
            "points": [],
            "presence_dynamics": {
                "generated_at": timestamp_value,
                "field_particles": [],
                "daimoi_probabilistic": {
                    "record": "ημ.daimoi-probabilistic.v1",
                    "schema_version": "daimoi.probabilistic.v1",
                    "active": 0,
                    "spawned": 0,
                    "collisions": 0,
                    "deflects": 0,
                    "diffuses": 0,
                    "handoffs": 0,
                    "deliveries": 0,
                    "job_triggers": {},
                    "disabled": True,
                    "disabled_reason": "shared_ws_cache_miss",
                },
            },
        },
        "projection": {
            "record": "家_映.v1",
            "perspective": perspective,
            "ts": int(time.time() * 1000),
            "name": perspective,
            "summary": {
                "view": perspective,
                "active_entities": 0,
                "total_items": 0,
                "active_queue": 0,
            },
            "highlights": [],
            "narrative": {
                "tone": "bootstrap",
                "line_en": "stream bootstrap",
                "line_ja": "stream bootstrap",
            },
        },
    }


def collect_shared_stream_frame(
    *,
    part_root: Path,
    perspective: str,
    payload_mode: str,
    load_cached_payload: Callable[..., tuple[dict[str, Any], dict[str, Any]] | None],
    json_compact: Callable[[Any], str],
) -> tuple[dict[str, Any], str]:
    cached_payload = load_cached_payload(
        part_root=part_root,
        perspective=perspective,
        payload_mode=payload_mode,
        allow_disabled_particle_dynamics=True,
    )
    if cached_payload is None:
        fallback_payload = build_shared_stream_bootstrap_payload(
            perspective=perspective
        )
        return fallback_payload, "shared-ws-cache-miss"

    simulation_payload, projection_payload = cached_payload
    timestamp_value = str(
        simulation_payload.get("timestamp", "")
        or simulation_payload.get("generated_at", "")
        or ""
    )
    payload: dict[str, Any] = {
        "type": "simulation",
        "simulation": simulation_payload,
        "projection": projection_payload,
    }
    if timestamp_value:
        return payload, timestamp_value

    digest = hashlib.sha1(
        json_compact(payload).encode("utf-8", errors="ignore")
    ).hexdigest()
    return payload, f"shared-ws:{digest}"
