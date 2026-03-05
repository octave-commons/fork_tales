# SPDX-License-Identifier: GPL-3.0-or-later
# This file is part of Fork Tales.
# Copyright (C) 2024-2025 Fork Tales Contributors

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable


WS_WIRE_ARRAY_SCHEMA = "eta-mu.ws.arr.v1"
_WS_PACK_TAG_OBJECT = -1
_WS_PACK_TAG_ARRAY = -2
_WS_PACK_TAG_STRING = -3
_WS_PACK_TAG_BOOL = -4
_WS_PACK_TAG_NULL = -5

DEFAULT_SIMULATION_WS_CHUNK_CHARS = 96000
DEFAULT_SIMULATION_WS_CHUNK_DELTA_MIN_CHARS = 262144
DEFAULT_SIMULATION_WS_CHUNK_MAX_CHUNKS = 512
DEFAULT_SIMULATION_WS_CHUNK_MESSAGE_TYPES = frozenset(
    {
        "simulation",
        "simulation_delta",
        "catalog",
        "muse_events",
    }
)


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _json_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalize_ws_wire_mode(mode: str, *, default_mode: str = "json") -> str:
    clean = str(mode or "").strip().lower()
    if clean in {"arr", "array", "arrays", "packed", "compact"}:
        return "arr"
    if clean in {"json", "object", "objects", "legacy"}:
        return "json"

    clean_default = str(default_mode or "json").strip().lower()
    if clean_default in {"arr", "array", "arrays", "packed", "compact"}:
        return "arr"
    return "json"


def _ws_pack_value(
    value: Any,
    *,
    key_table: list[str],
    key_index: dict[str, int],
) -> Any:
    if value is None:
        return [_WS_PACK_TAG_NULL]
    if isinstance(value, bool):
        return [_WS_PACK_TAG_BOOL, 1 if value else 0]
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return [_WS_PACK_TAG_STRING, "0"]
        return value
    if isinstance(value, str):
        return [_WS_PACK_TAG_STRING, value]
    if isinstance(value, list):
        encoded_items = [
            _ws_pack_value(item, key_table=key_table, key_index=key_index)
            for item in value
        ]
        return [_WS_PACK_TAG_ARRAY, *encoded_items]
    if isinstance(value, dict):
        encoded_pairs: list[Any] = [_WS_PACK_TAG_OBJECT]
        for key, nested in value.items():
            key_name = str(key)
            key_slot = key_index.get(key_name)
            if key_slot is None:
                key_slot = len(key_table)
                key_index[key_name] = key_slot
                key_table.append(key_name)
            encoded_pairs.append(key_slot)
            encoded_pairs.append(
                _ws_pack_value(nested, key_table=key_table, key_index=key_index)
            )
        return encoded_pairs
    return [_WS_PACK_TAG_STRING, str(value)]


def ws_pack_message(
    payload: dict[str, Any],
    *,
    schema: str = WS_WIRE_ARRAY_SCHEMA,
) -> list[Any]:
    key_table: list[str] = []
    key_index: dict[str, int] = {}
    encoded = _ws_pack_value(payload, key_table=key_table, key_index=key_index)
    return [str(schema or WS_WIRE_ARRAY_SCHEMA), key_table, encoded]


def simulation_ws_chunk_plan(
    payload: dict[str, Any],
    *,
    chunk_chars: int,
    message_seq: int,
    allowed_types: Iterable[str] | None = None,
    default_chunk_chars: int = DEFAULT_SIMULATION_WS_CHUNK_CHARS,
    delta_min_chars: int = DEFAULT_SIMULATION_WS_CHUNK_DELTA_MIN_CHARS,
    max_chunks: int = DEFAULT_SIMULATION_WS_CHUNK_MAX_CHUNKS,
) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(payload, dict):
        return [], None

    payload_type = str(payload.get("type", "") or "").strip().lower()
    if not payload_type or payload_type == "ws_chunk":
        return [], None

    if allowed_types is None:
        allowed = set(DEFAULT_SIMULATION_WS_CHUNK_MESSAGE_TYPES)
    else:
        allowed = {
            str(row or "").strip().lower()
            for row in allowed_types
            if str(row or "").strip()
        }
    if payload_type not in allowed:
        return [], None

    chunk_size = max(4096, int(_safe_float(chunk_chars, float(default_chunk_chars))))
    payload_text = _json_compact(payload)
    payload_chars = len(payload_text)

    if payload_type == "simulation_delta":
        target_delta_min_chars = max(chunk_size, int(max(1, delta_min_chars)))
        if payload_chars <= target_delta_min_chars:
            return [], payload_text
    if payload_chars <= chunk_size:
        return [], payload_text

    chunk_total = int(math.ceil(payload_chars / float(chunk_size)))
    if chunk_total <= 1:
        return [], payload_text

    allowed_max_chunks = max(1, int(max_chunks))
    if chunk_total > allowed_max_chunks:
        chunk_total = allowed_max_chunks
        chunk_size = int(math.ceil(payload_chars / float(chunk_total)))

    digest = hashlib.sha1(payload_text.encode("utf-8")).hexdigest()[:16]
    chunk_id = f"ws:{payload_type}:{int(message_seq)}:{digest}"

    rows: list[dict[str, Any]] = []
    for index in range(chunk_total):
        start = index * chunk_size
        end = min(payload_chars, (index + 1) * chunk_size)
        rows.append(
            {
                "type": "ws_chunk",
                "record": "eta-mu.ws.chunk.v1",
                "schema_version": "ws.chunk.v1",
                "chunk_id": chunk_id,
                "chunk_payload_type": payload_type,
                "chunk_index": index,
                "chunk_total": chunk_total,
                "payload_chars": payload_chars,
                "payload": payload_text[start:end],
            }
        )
    return rows, None


def simulation_ws_chunk_messages(
    payload: dict[str, Any],
    *,
    chunk_chars: int,
    message_seq: int,
    allowed_types: Iterable[str] | None = None,
    default_chunk_chars: int = DEFAULT_SIMULATION_WS_CHUNK_CHARS,
    delta_min_chars: int = DEFAULT_SIMULATION_WS_CHUNK_DELTA_MIN_CHARS,
    max_chunks: int = DEFAULT_SIMULATION_WS_CHUNK_MAX_CHUNKS,
) -> list[dict[str, Any]]:
    rows, _ = simulation_ws_chunk_plan(
        payload,
        chunk_chars=chunk_chars,
        message_seq=message_seq,
        allowed_types=allowed_types,
        default_chunk_chars=default_chunk_chars,
        delta_min_chars=delta_min_chars,
        max_chunks=max_chunks,
    )
    return rows


def simulation_ws_worker_for_top_level_key(key: str) -> str:
    clean = str(key or "").strip().lower()
    if clean in {
        "timestamp",
        "total",
        "audio",
        "image",
        "video",
        "points",
        "truth_state",
        "perspective",
        "myth",
        "world",
        "tick_elapsed_ms",
        "slack_ms",
        "ingestion_pressure",
        "ws_particle_max",
        "particle_payload_mode",
        "network_send_pressure",
        "network_send_ema_ms",
        "network_particle_cap",
        "graph_node_positions_truncated",
        "graph_node_positions_total",
        "presence_anchor_positions_truncated",
        "presence_anchor_positions_total",
    }:
        return "sim-core"
    if clean in {"projection"}:
        return "sim-projection"
    if clean in {"field_particles", "pain_field"}:
        return "sim-particles"
    if clean in {"daimoi", "entities", "echoes"}:
        return "sim-daimoi"
    if clean in {"presence_dynamics"}:
        return "sim-presence"
    return "sim-misc"


def simulation_ws_worker_for_presence_key(key: str) -> str:
    clean = str(key or "").strip().lower()
    if clean in {
        "field_particles",
        "field_particles_page",
        "nooi_field",
        "graph_node_positions",
        "presence_anchor_positions",
    }:
        return "sim-particles"
    if clean in {
        "simulation_budget",
        "resource_heartbeat",
        "compute_jobs_180s",
        "compute_summary",
        "compute_jobs",
        "resource_daimoi",
        "resource_consumption",
        "daimoi_collision_events",
        "daimoi_collision_events_record",
        "daimoi_collision_event_seq",
        "river_flow",
    }:
        return "sim-resource"
    if clean in {
        "click_events",
        "file_events",
        "recent_click_targets",
        "recent_file_paths",
        "user_presence",
        "user_input_messages",
        "user_embedded_daimoi_count",
    }:
        return "sim-interaction"
    if clean in {
        "daimoi_probabilistic",
        "daimoi_probabilistic_record",
        "daimoi_behavior_defaults",
        "ghost",
        "fork_tax",
    }:
        return "sim-daimoi"
    return "sim-presence"


def simulation_ws_split_delta_by_worker(delta: dict[str, Any]) -> list[dict[str, Any]]:
    patch = delta.get("patch") if isinstance(delta, dict) else None
    if not isinstance(patch, dict) or not patch:
        return []

    worker_patch: dict[str, dict[str, Any]] = {}
    worker_changed_keys: dict[str, list[str]] = {}

    def record_worker_patch(
        *,
        worker_id: str,
        key: str,
        value: Any,
        changed_key: str,
    ) -> None:
        bucket = worker_patch.get(worker_id)
        if not isinstance(bucket, dict):
            bucket = {}
            worker_patch[worker_id] = bucket
        bucket[key] = value

        changed = worker_changed_keys.get(worker_id)
        if not isinstance(changed, list):
            changed = []
            worker_changed_keys[worker_id] = changed
        if changed_key not in changed:
            changed.append(changed_key)

    for key, value in patch.items():
        if key == "presence_dynamics":
            if isinstance(value, dict):
                grouped_dynamics: dict[str, dict[str, Any]] = {}
                for dynamics_key, dynamics_value in value.items():
                    worker_id = simulation_ws_worker_for_presence_key(dynamics_key)
                    grouped = grouped_dynamics.get(worker_id)
                    if not isinstance(grouped, dict):
                        grouped = {}
                        grouped_dynamics[worker_id] = grouped
                    grouped[dynamics_key] = dynamics_value

                for worker_id, dynamics_patch in grouped_dynamics.items():
                    bucket = worker_patch.get(worker_id)
                    if not isinstance(bucket, dict):
                        bucket = {}
                        worker_patch[worker_id] = bucket
                    existing_dynamics = bucket.get("presence_dynamics")
                    if not isinstance(existing_dynamics, dict):
                        existing_dynamics = {}
                        bucket["presence_dynamics"] = existing_dynamics
                    existing_dynamics.update(dynamics_patch)

                    changed = worker_changed_keys.get(worker_id)
                    if not isinstance(changed, list):
                        changed = []
                        worker_changed_keys[worker_id] = changed
                    for dynamics_key in dynamics_patch:
                        changed_key = f"presence_dynamics.{dynamics_key}"
                        if changed_key not in changed:
                            changed.append(changed_key)
            else:
                record_worker_patch(
                    worker_id="sim-presence",
                    key="presence_dynamics",
                    value=value,
                    changed_key="presence_dynamics",
                )
            continue

        record_worker_patch(
            worker_id=simulation_ws_worker_for_top_level_key(key),
            key=key,
            value=value,
            changed_key=key,
        )

    timestamp_value = patch.get("timestamp")
    if timestamp_value is not None:
        for row in worker_patch.values():
            if isinstance(row, dict) and "timestamp" not in row:
                row["timestamp"] = timestamp_value

    worker_priority = {
        "sim-core": 0,
        "sim-particles": 1,
        "sim-resource": 2,
        "sim-interaction": 3,
        "sim-presence": 4,
        "sim-daimoi": 5,
        "sim-projection": 6,
        "sim-misc": 7,
    }

    rows: list[dict[str, Any]] = []
    for worker_id, worker_row_patch in worker_patch.items():
        changed = worker_changed_keys.get(worker_id, [])
        rows.append(
            {
                "worker_id": worker_id,
                "patch": worker_row_patch,
                "changed_keys": changed,
            }
        )
    rows.sort(
        key=lambda row: (
            worker_priority.get(str(row.get("worker_id", "")), 99),
            str(row.get("worker_id", "")),
        )
    )
    return rows
