from __future__ import annotations

import fnmatch
import hashlib
import math
import time
from typing import Any

from .constants import (
    CANONICAL_NAMED_FIELD_IDS,
    ENTITY_MANIFEST,
    FIELD_TO_PRESENCE,
    PROJECTION_DEFAULT_PERSPECTIVE,
    PROJECTION_ELEMENTS,
    PROJECTION_FIELD_SCHEMAS,
    PROJECTION_PERSPECTIVES,
    SIM_TICK_SECONDS,
)
from .metrics import _clamp01, _safe_float


def projection_perspective_options() -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for key in ("hybrid", "causal-time", "swimlanes"):
        meta = PROJECTION_PERSPECTIVES[key]
        options.append(
            {
                "id": key,
                "symbol": str(meta.get("id", f"perspective.{key}")),
                "name": str(meta.get("name", key.title())),
                "merge": str(meta.get("merge", key)),
                "description": str(meta.get("description", "")),
                "default": key == PROJECTION_DEFAULT_PERSPECTIVE,
            }
        )
    return options


def normalize_projection_perspective(raw: str | None) -> str:
    text = str(raw or "").strip().lower().replace("_", "-")
    aliases = {
        "": PROJECTION_DEFAULT_PERSPECTIVE,
        "hybrid": "hybrid",
        "causal": "causal-time",
        "causal-time": "causal-time",
        "causaltime": "causal-time",
        "swimlane": "swimlanes",
        "swimlanes": "swimlanes",
        "lanes": "swimlanes",
    }
    resolved = aliases.get(text, text)
    if resolved not in PROJECTION_PERSPECTIVES:
        return PROJECTION_DEFAULT_PERSPECTIVE
    return resolved


def _projection_source_influence(
    simulation: dict[str, Any] | None,
    catalog: dict[str, Any],
    influence_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(influence_snapshot, dict) and influence_snapshot:
        return influence_snapshot
    if isinstance(simulation, dict):
        dynamics = simulation.get("presence_dynamics")
        if isinstance(dynamics, dict) and dynamics:
            return dynamics
    runtime = catalog.get("presence_runtime")
    if isinstance(runtime, dict):
        return runtime
    return {}


def _projection_presence_impacts(
    simulation: dict[str, Any] | None,
    influence: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(simulation, dict):
        dynamics = simulation.get("presence_dynamics")
        if isinstance(dynamics, dict):
            rows = dynamics.get("presence_impacts")
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]

    clicks = int(
        _safe_float(
            influence.get("click_events", influence.get("clicks_45s", 0)),
            0.0,
        )
    )
    files = int(
        _safe_float(
            influence.get("file_events", influence.get("file_changes_120s", 0)),
            0.0,
        )
    )
    click_ratio = _clamp01(clicks / 18.0)
    file_ratio = _clamp01(files / 24.0)
    impacts: list[dict[str, Any]] = []
    by_id = {str(item.get("id", "")): item for item in ENTITY_MANIFEST}
    for idx, presence_id in enumerate(CANONICAL_NAMED_FIELD_IDS):
        meta = by_id.get(presence_id, {})
        file_signal = _clamp01(file_ratio * (0.55 + (idx % 3) * 0.12))
        click_signal = _clamp01(click_ratio * (0.62 + (idx % 4) * 0.08))
        world_signal = _clamp01((file_signal * 0.58) + (click_signal * 0.42))
        impacts.append(
            {
                "id": presence_id,
                "en": str(meta.get("en", presence_id.replace("_", " ").title())),
                "ja": str(meta.get("ja", "")),
                "affected_by": {
                    "files": round(file_signal, 4),
                    "clicks": round(click_signal, 4),
                },
                "affects": {
                    "world": round(world_signal, 4),
                    "ledger": round(_clamp01(world_signal * 0.86), 4),
                },
                "notes_en": "Fallback projection impact synthesized from runtime pressure.",
                "notes_ja": "ランタイム圧から合成したフォールバック影響。",
            }
        )
    return impacts


def _semantic_xy_from_embedding(vector: list[float]) -> tuple[float, float] | None:
    from .db import _normalize_embedding_vector

    normalized = _normalize_embedding_vector(vector)
    if normalized is None:
        return None

    x = 0.0
    y = 0.0
    for idx, value in enumerate(normalized):
        angle = (((idx * 2654435761) % 360) / 360.0) * math.tau
        x += value * math.cos(angle)
        y += value * math.sin(angle)

    magnitude = math.sqrt((x * x) + (y * y))
    if magnitude > 0.0:
        x /= magnitude
        y /= magnitude

    return (
        round(_clamp01(0.5 + (x * 0.28)), 4),
        round(_clamp01(0.5 + (y * 0.28)), 4),
    )


def _field_scores_from_position(
    x: float,
    y: float,
    field_anchors: dict[str, tuple[float, float]],
) -> dict[str, float]:
    from .metrics import _normalize_field_scores

    raw: dict[str, float] = {}
    for field_id in FIELD_TO_PRESENCE:
        anchor_x, anchor_y = field_anchors.get(field_id, (0.5, 0.5))
        dx = x - anchor_x
        dy = y - anchor_y
        distance = math.sqrt((dx * dx) + (dy * dy))
        raw[field_id] = 1.0 / (0.04 + (distance * distance * 6.0))
    return _normalize_field_scores(raw)


def _blend_field_scores(
    base_scores: dict[str, float],
    positional_scores: dict[str, float],
    *,
    position_weight: float,
    fallback_field: str,
) -> dict[str, float]:
    from .metrics import _normalize_field_scores

    blend = _clamp01(_safe_float(position_weight, 0.0))
    if blend <= 0.0:
        return _normalize_field_scores(base_scores, fallback_field=fallback_field)
    if blend >= 1.0:
        return _normalize_field_scores(positional_scores, fallback_field=fallback_field)

    mixed: dict[str, float] = {}
    for field_id in FIELD_TO_PRESENCE:
        left = _safe_float(base_scores.get(field_id, 0.0), 0.0)
        right = _safe_float(positional_scores.get(field_id, 0.0), 0.0)
        mixed[field_id] = (left * (1.0 - blend)) + (right * blend)
    return _normalize_field_scores(mixed, fallback_field=fallback_field)


def _eta_mu_embed_layer_identity(
    *,
    collection: str,
    space_id: str,
    space_signature: str,
    model_name: str,
) -> dict[str, str]:
    from .constants import ETA_MU_INGEST_VECSTORE_COLLECTION

    clean_collection = str(collection or ETA_MU_INGEST_VECSTORE_COLLECTION).strip()
    clean_space_id = str(space_id or "").strip()
    clean_signature = str(space_signature or "").strip()
    clean_model = str(model_name or "").strip()
    layer_key = "|".join(
        [clean_collection, clean_space_id, clean_signature, clean_model]
    )
    layer_id = "layer:" + hashlib.sha1(layer_key.encode("utf-8")).hexdigest()[:12]

    label_parts = [clean_collection]
    if clean_space_id:
        label_parts.append(clean_space_id)
    if clean_model:
        label_parts.append(clean_model)
    elif clean_signature:
        label_parts.append(clean_signature[:10])

    return {
        "id": layer_id,
        "key": layer_key,
        "label": " · ".join(part for part in label_parts if part),
        "collection": clean_collection,
        "space_id": clean_space_id,
        "space_signature": clean_signature,
        "model_name": clean_model,
    }


def _eta_mu_embed_layer_matches(pattern: str, layer: dict[str, Any]) -> bool:
    candidate_values = [
        str(layer.get("id", "")),
        str(layer.get("key", "")),
        str(layer.get("collection", "")),
        str(layer.get("space_id", "")),
        str(layer.get("space_signature", "")),
        str(layer.get("model_name", "")),
        str(layer.get("label", "")),
    ]
    clean_pattern = str(pattern or "").strip()
    if not clean_pattern:
        return False
    if clean_pattern == "*":
        return True
    for value in candidate_values:
        if not value:
            continue
        if fnmatch.fnmatch(value, clean_pattern):
            return True
    return False


def _eta_mu_embed_layer_is_active(
    layer: dict[str, Any],
    active_patterns: list[str],
) -> bool:
    if not active_patterns or active_patterns == ["*"]:
        return True
    return any(
        _eta_mu_embed_layer_matches(pattern, layer) for pattern in active_patterns
    )


def _eta_mu_embed_layer_order_index(
    layer: dict[str, Any],
    order_patterns: list[str],
) -> int:
    if not order_patterns:
        return 0
    for index, pattern in enumerate(order_patterns):
        if _eta_mu_embed_layer_matches(pattern, layer):
            return index
    return len(order_patterns) + 1


def _eta_mu_embed_layer_point(
    *,
    node_x: float,
    node_y: float,
    layer_seed: str,
    index: int,
    semantic_xy: tuple[float, float] | None,
) -> tuple[float, float, str]:
    from .constants import ETA_MU_FILE_GRAPH_LAYER_BLEND
    from .metrics import _stable_ratio

    if semantic_xy is None:
        target_x = _stable_ratio(layer_seed, index * 13 + 3)
        target_y = _stable_ratio(layer_seed, index * 13 + 7)
        source = "deterministic"
    else:
        target_x, target_y = semantic_xy
        source = "semantic"

    orbit = 0.008 + (index % 4) * 0.004
    angle = _stable_ratio(layer_seed, index * 17 + 11) * math.tau
    target_x = _clamp01(target_x + math.cos(angle) * orbit)
    target_y = _clamp01(target_y + math.sin(angle) * orbit)

    blend = _clamp01(ETA_MU_FILE_GRAPH_LAYER_BLEND)
    layered_x = _clamp01((node_x * (1.0 - blend)) + (target_x * blend))
    layered_y = _clamp01((node_y * (1.0 - blend)) + (target_y * blend))
    return round(layered_x, 4), round(layered_y, 4), source


def _build_projection_field_snapshot(
    catalog: dict[str, Any],
    simulation: dict[str, Any] | None,
    *,
    perspective: str,
    queue_snapshot: dict[str, Any] | None,
    influence_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    queue = queue_snapshot or catalog.get("task_queue", {})
    influence = _projection_source_influence(simulation, catalog, influence_snapshot)

    counts = catalog.get("counts", {})
    item_count = len(catalog.get("items", []))
    promptdb = catalog.get("promptdb", {})

    audio_ratio = _clamp01(_safe_float(counts.get("audio", 0), 0.0) / 24.0)
    image_ratio = _clamp01(_safe_float(counts.get("image", 0), 0.0) / 220.0)
    text_ratio = _clamp01(_safe_float(counts.get("text", 0), 0.0) / 90.0)
    promptdb_ratio = _clamp01(_safe_float(promptdb.get("packet_count", 0), 0.0) / 120.0)

    clicks = int(
        _safe_float(
            influence.get("click_events", influence.get("clicks_45s", 0)),
            0.0,
        )
    )
    files = int(
        _safe_float(
            influence.get("file_events", influence.get("file_changes_120s", 0)),
            0.0,
        )
    )
    click_ratio = _clamp01(clicks / 18.0)
    file_ratio = _clamp01(files / 24.0)

    pending_count = int(_safe_float(queue.get("pending_count", 0), 0.0))
    queue_events = int(_safe_float(queue.get("event_count", 0), 0.0))
    queue_pending_ratio = _clamp01(pending_count / 8.0)
    queue_events_ratio = _clamp01(queue_events / 48.0)

    fork_tax = influence.get("fork_tax", {}) if isinstance(influence, dict) else {}
    paid_ratio = _clamp01(_safe_float(fork_tax.get("paid_ratio", 0.5), 0.5))
    balance_raw = max(0.0, _safe_float(fork_tax.get("balance", 0.0), 0.0))
    debt_raw = max(0.0, _safe_float(fork_tax.get("debt", balance_raw), balance_raw))
    fork_balance_ratio = _clamp01(
        1.0
        - (
            balance_raw
            / max(1.0, debt_raw + _safe_float(fork_tax.get("paid", 0.0), 0.0))
        )
    )

    witness_continuity = _safe_float(
        (
            ((simulation or {}).get("presence_dynamics") or {}).get("witness_thread")
            or {}
        ).get("continuity_index", click_ratio),
        click_ratio,
    )
    witness_ratio = _clamp01(max(click_ratio, witness_continuity))

    drift_index = _clamp01((file_ratio * 0.58) + (queue_pending_ratio * 0.42))
    proof_gap = _clamp01((1.0 - paid_ratio) * 0.64 + drift_index * 0.36)
    artifact_flux = _clamp01(
        (audio_ratio * 0.49)
        + (image_ratio * 0.11)
        + (file_ratio * 0.24)
        + (promptdb_ratio * 0.16)
    )
    coherence_focus = _clamp01(
        1.0
        - (abs(audio_ratio - text_ratio) * 0.74)
        - (drift_index * 0.24)
        + (witness_ratio * 0.14)
    )
    curiosity_drive = _clamp01(
        (text_ratio * 0.26)
        + (promptdb_ratio * 0.42)
        + (click_ratio * 0.2)
        + (artifact_flux * 0.12)
    )
    gate_pressure = _clamp01(
        (drift_index * 0.48) + (proof_gap * 0.3) + (queue_pending_ratio * 0.22)
    )
    council_heat = _clamp01(
        (queue_events_ratio * 0.56)
        + (queue_pending_ratio * 0.29)
        + (gate_pressure * 0.15)
    )

    if perspective_key == "causal-time":
        witness_ratio = _clamp01((witness_ratio * 0.76) + (gate_pressure * 0.24))
        drift_index = _clamp01((drift_index * 0.74) + (gate_pressure * 0.26))
    elif perspective_key == "swimlanes":
        council_heat = _clamp01((council_heat * 0.76) + (queue_pending_ratio * 0.24))
        coherence_focus = _clamp01((coherence_focus * 0.82) + (curiosity_drive * 0.18))

    impacts = _projection_presence_impacts(simulation, influence)
    applied_reiso = [
        f"rei.{str(item.get('id', 'presence')).strip()}.impulse"
        for item in sorted(
            impacts,
            key=lambda row: _safe_float(
                (row.get("affects", {}) or {}).get("world", 0.0), 0.0
            ),
            reverse=True,
        )[:4]
        if str(item.get("id", "")).strip()
    ]

    world_tick = int(
        _safe_float(((simulation or {}).get("world") or {}).get("tick", 0), 0.0)
    )
    witness_lineage = (
        ((simulation or {}).get("presence_dynamics") or {}).get("witness_thread") or {}
    ).get("lineage", []) or []
    vectors = {
        "f1": {
            "energy": round(artifact_flux, 4),
            "audio_ratio": round(audio_ratio, 4),
            "file_ratio": round(file_ratio, 4),
            "mix_sources": int(
                _safe_float((catalog.get("mix") or {}).get("sources", 0), 0.0)
            ),
        },
        "f2": {
            "energy": round(witness_ratio, 4),
            "click_ratio": round(click_ratio, 4),
            "continuity_index": round(witness_ratio, 4),
            "lineage_links": len(witness_lineage),
        },
        "f3": {
            "energy": round(coherence_focus, 4),
            "balance_ratio": round(_clamp01(1.0 - abs(audio_ratio - text_ratio)), 4),
            "promptdb_ratio": round(promptdb_ratio, 4),
            "catalog_entropy": round(_clamp01(item_count / 400.0), 4),
        },
        "f4": {
            "energy": round(drift_index, 4),
            "file_ratio": round(file_ratio, 4),
            "queue_pending_ratio": round(queue_pending_ratio, 4),
            "drift_index": round(drift_index, 4),
        },
        "f5": {
            "energy": round(_clamp01(1.0 - paid_ratio), 4),
            "paid_ratio": round(paid_ratio, 4),
            "balance_ratio": round(fork_balance_ratio, 4),
            "debt": round(debt_raw, 4),
        },
        "f6": {
            "energy": round(curiosity_drive, 4),
            "text_ratio": round(text_ratio, 4),
            "promptdb_ratio": round(promptdb_ratio, 4),
            "interaction_ratio": round(click_ratio, 4),
        },
        "f7": {
            "energy": round(gate_pressure, 4),
            "queue_ratio": round(queue_pending_ratio, 4),
            "drift_index": round(drift_index, 4),
            "proof_gap": round(proof_gap, 4),
        },
        "f8": {
            "energy": round(council_heat, 4),
            "queue_events_ratio": round(queue_events_ratio, 4),
            "pending_ratio": round(queue_pending_ratio, 4),
            "decision_load": round(_clamp01((queue_events + pending_count) / 52.0), 4),
        },
    }

    merge_mode = {
        "hybrid": "hybrid",
        "causal-time": "causal",
        "swimlanes": "wallclock",
    }.get(perspective_key, "hybrid")

    snapshot_seed = (
        f"{perspective_key}|{ts_ms}|{item_count}|{queue_events}|{pending_count}"
    )
    snapshot_id = (
        f"field.snapshot.{hashlib.sha1(snapshot_seed.encode('utf-8')).hexdigest()[:12]}"
    )
    return {
        "record": "場/snapshot.v1",
        "id": snapshot_id,
        "ts": ts_ms,
        "vectors": vectors,
        "applied_reiso": applied_reiso,
        "merge_mode": merge_mode,
        "ticks": [
            f"tick.world.{world_tick}",
            f"tick.queue.{queue_events}",
            f"tick.catalog.{item_count}",
        ],
    }


def _projection_field_levels(field_snapshot: dict[str, Any]) -> dict[str, float]:
    vectors = field_snapshot.get("vectors", {})
    levels: dict[str, float] = {}
    for field_name in ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]:
        field_row = vectors.get(field_name, {})
        levels[field_name] = _clamp01(_safe_float(field_row.get("energy", 0.0), 0.0))
    return levels


def _projection_dominant_field(
    field_levels: dict[str, float],
    field_bindings: dict[str, Any],
) -> tuple[str, float]:
    if not field_bindings:
        return "f3", field_levels.get("f3", 0.0)
    best_field = "f3"
    best_score = -1.0
    for key, weight in field_bindings.items():
        score = _clamp01(_safe_float(weight, 0.0)) * field_levels.get(str(key), 0.0)
        if score > best_score:
            best_score = score
            best_field = str(key)
    return best_field, field_levels.get(best_field, 0.0)


def _build_projection_coherence_state(
    field_snapshot: dict[str, Any],
    *,
    perspective: str,
) -> dict[str, Any]:
    levels = _projection_field_levels(field_snapshot)
    ts_ms = int(time.time() * 1000)
    centroid = {
        "x": round((levels["f3"] * 0.6) + (levels["f6"] * 0.4), 4),
        "y": round((levels["f2"] * 0.52) + (levels["f4"] * 0.48), 4),
        "z": round((levels["f1"] * 0.45) + (levels["f7"] * 0.55), 4),
    }
    tension = _clamp01(
        (levels["f2"] * 0.4) + (levels["f4"] * 0.25) + (levels["f7"] * 0.35)
    )
    drift = _clamp01(
        (levels["f4"] * 0.45) + (levels["f5"] * 0.25) + (levels["f8"] * 0.3)
    )
    entropy = _clamp01((1.0 - levels["f3"]) * 0.58 + levels["f8"] * 0.42)

    perspective_scores = {
        "hybrid": _clamp01(
            (levels["f1"] * 0.28) + (levels["f3"] * 0.34) + (levels["f6"] * 0.38)
        ),
        "causal-time": _clamp01(
            (levels["f2"] * 0.37) + (levels["f4"] * 0.28) + (levels["f7"] * 0.35)
        ),
        "swimlanes": _clamp01(
            (levels["f8"] * 0.42) + (levels["f4"] * 0.24) + (levels["f5"] * 0.34)
        ),
    }
    dominant_perspective = max(
        perspective_scores.keys(),
        key=lambda key: perspective_scores.get(key, 0.0),
    )
    coherence_seed = f"{field_snapshot.get('id', '')}|{perspective}|{ts_ms}"
    return {
        "record": "心/state.v1",
        "id": f"coherence.{hashlib.sha1(coherence_seed.encode('utf-8')).hexdigest()[:12]}",
        "ts": ts_ms,
        "centroid": centroid,
        "tension": round(tension, 4),
        "drift": round(drift, 4),
        "entropy": round(entropy, 4),
        "dominant_perspective": dominant_perspective,
        "perspective_score": round(
            perspective_scores.get(normalize_projection_perspective(perspective), 0.0),
            4,
        ),
    }


def _build_projection_element_states(
    field_snapshot: dict[str, Any],
    coherence_state: dict[str, Any],
    *,
    perspective: str,
    simulation: dict[str, Any] | None,
    influence_snapshot: dict[str, Any] | None,
    catalog: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    perspective_key = normalize_projection_perspective(perspective)
    levels = _projection_field_levels(field_snapshot)
    influence = influence_snapshot or {}

    impact_rows = _projection_presence_impacts(simulation, influence)
    impact_map = {str(item.get("id", "")): item for item in impact_rows}
    ts_ms = int(time.time() * 1000)
    clamps = {
        "record": "映/clamp.v1",
        "min_area": 0.1,
        "max_area": 0.36,
        "max_pulse": 0.92,
        "decay_half_life": {
            "mass_ms": 2400,
            "priority_ms": 1800,
            "pulse_ms": 1200,
        },
    }

    elements: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []

    logical_graph: dict[str, Any] = {}
    if isinstance(simulation, dict):
        logical_graph = simulation.get("logical_graph", {})
    if not logical_graph and isinstance(catalog, dict):
        logical_graph = catalog.get("logical_graph", {})
    if not isinstance(logical_graph, dict):
        logical_graph = {}

    graph_nodes = (
        logical_graph.get("nodes", []) if isinstance(logical_graph, dict) else []
    )

    test_nodes = [
        node
        for node in graph_nodes
        if isinstance(node, dict) and str(node.get("kind", "")) == "test"
    ]

    for template in PROJECTION_ELEMENTS:
        element_id = str(template.get("id", "")).strip()
        if not element_id:
            continue
        field_bindings = dict(template.get("field_bindings", {}))
        binding_sum = sum(
            max(0.0, _safe_float(weight, 0.0)) for weight in field_bindings.values()
        )
        if binding_sum <= 0:
            field_signal = levels.get("f3", 0.0)
        else:
            weighted = 0.0
            for field_key, weight in field_bindings.items():
                weighted += levels.get(str(field_key), 0.0) * max(
                    0.0, _safe_float(weight, 0.0)
                )
            field_signal = _clamp01(weighted / binding_sum)

        presence_id = str(template.get("presence", "")).strip()
        impact = impact_map.get(presence_id, {}) if presence_id else {}
        affected = impact.get("affected_by", {}) if isinstance(impact, dict) else {}
        affects = impact.get("affects", {}) if isinstance(impact, dict) else {}
        presence_signal = _clamp01(
            (_safe_float(affects.get("world", 0.0), 0.0) * 0.6)
            + (_safe_float(affected.get("files", 0.0), 0.0) * 0.23)
            + (_safe_float(affected.get("clicks", 0.0), 0.0) * 0.17)
        )
        queue_signal = levels.get("f8", 0.0)
        causal_signal = levels.get("f7", 0.0)

        if perspective_key == "causal-time":
            field_signal = _clamp01(
                (field_signal * 0.68)
                + (levels.get("f2", 0.0) * 0.19)
                + (causal_signal * 0.13)
            )
        elif perspective_key == "swimlanes":
            field_signal = _clamp01(
                (field_signal * 0.72)
                + (queue_signal * 0.2)
                + (levels.get("f4", 0.0) * 0.08)
            )

        kind = str(template.get("kind", "panel"))
        mass = _clamp01(
            (field_signal * 0.56) + (presence_signal * 0.24) + (queue_signal * 0.2)
        )
        priority = _clamp01(
            (mass * 0.67) + (causal_signal * 0.23) + (levels.get("f2", 0.0) * 0.1)
        )
        area = 0.1 + (mass * 0.24) + (priority * 0.04)
        if kind == "chat-lens":
            area += 0.05
            priority = _clamp01(priority + 0.08)
        if "governance" in (template.get("tags") or []):
            priority = _clamp01(priority + (levels.get("f7", 0.0) * 0.1))
        if kind == "test":
            priority = _clamp01(priority + 0.12)
            mass = _clamp01(mass + 0.15)
            if str(template.get("status", "")) == "failed":
                pulse = 0.95
                mass = _clamp01(mass + 0.2)
        if perspective_key == "swimlanes":
            area *= 0.88
        area = max(clamps["min_area"], min(clamps["max_area"], area))

        opacity = _clamp01(0.42 + (mass * 0.46) + (presence_signal * 0.12))
        pulse = min(
            clamps["max_pulse"],
            _clamp01(
                (field_signal * 0.42) + (presence_signal * 0.31) + (queue_signal * 0.27)
            ),
        )

        dominant_field, dominant_level = _projection_dominant_field(
            levels, field_bindings
        )
        explain = {
            "field_signal": round(field_signal, 4),
            "presence_signal": round(presence_signal, 4),
            "queue_signal": round(queue_signal, 4),
            "causal_signal": round(causal_signal, 4),
            "dominant_field": dominant_field,
            "dominant_level": round(dominant_level, 4),
            "field_bindings": {
                str(key): round(_safe_float(value, 0.0), 4)
                for key, value in field_bindings.items()
            },
            "reason_en": (
                f"{template.get('title', 'Element')} expands on {dominant_field} "
                f"under {perspective_key} perspective coupling."
            ),
            "reason_ja": "投影は場の優勢軸に従って拡縮される。",
            "coherence_tension": round(
                _safe_float(coherence_state.get("tension", 0.0), 0.0), 4
            ),
        }
        sources = [
            str(field_snapshot.get("id", "")),
            str(coherence_state.get("id", "")),
            f"perspective:{perspective_key}",
        ]
        if presence_id:
            sources.append(f"presence:{presence_id}")

        elements.append(
            {
                "record": "映/element.v1",
                "id": element_id,
                "kind": kind,
                "title": str(template.get("title", element_id)),
                "binds_to": list(template.get("binds_to", [])),
                "field_bindings": {
                    str(key): round(_safe_float(value, 0.0), 4)
                    for key, value in field_bindings.items()
                },
                "presence": presence_id,
                "tags": list(template.get("tags", [])),
                "lane": str(template.get("lane", "voice")),
                "memory_scope": str(template.get("memory_scope", "shared")),
            }
        )
        states.append(
            {
                "record": "映/state.v1",
                "element_id": element_id,
                "ts": ts_ms,
                "mass": round(mass, 4),
                "priority": round(priority, 4),
                "area": round(area, 4),
                "opacity": round(opacity, 4),
                "pulse": round(pulse, 4),
                "sources": sources,
                "explain": explain,
            }
        )

    for test_node in test_nodes:
        test_id = str(test_node.get("id", ""))
        label = str(test_node.get("label", "Test"))
        status = str(test_node.get("status", "failed"))
        if not test_id:
            continue

        element_id = f"ui:test:{test_id}"
        kind = "test"
        lane = "council" if perspective_key == "causal-time" else "senses"

        field_signal = levels.get("f7", 0.0)
        presence_signal = 0.8
        queue_signal = 0.5

        mass = _clamp01(0.7 + (field_signal * 0.2))
        priority = _clamp01(0.85 + (field_signal * 0.1))
        area = 0.18
        opacity = 0.95
        pulse = 0.95

        sources = [str(field_snapshot.get("id", "")), test_id]

        explain = {
            "reason_en": f"Active test entity: {label} ({status})",
            "reason_ja": f"アクティブなテスト実体: {label} ({status})",
            "status": status,
        }

        elements.append(
            {
                "record": "映/element.v1",
                "id": element_id,
                "kind": kind,
                "title": label,
                "glyph": "試",
                "status": status,
                "lane": lane,
                "tags": ["test", "quality", "signal"],
                "binds_to": [],
                "field_bindings": {"f7": 0.8, "f4": 0.2},
            }
        )

        states.append(
            {
                "record": "映/state.v1",
                "element_id": element_id,
                "ts": ts_ms,
                "mass": round(mass, 4),
                "priority": round(priority, 4),
                "area": round(area, 4),
                "opacity": round(opacity, 4),
                "pulse": round(pulse, 4),
                "sources": sources,
                "explain": explain,
            }
        )

    return elements, states, clamps


def _build_projection_rects_swimlanes(
    states: list[dict[str, Any]],
    elements: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    state_map = {str(state.get("element_id", "")): state for state in states}
    lane_order = ["senses", "voice", "memory", "council"]
    lane_to_elements: dict[str, list[dict[str, Any]]] = {
        lane: [] for lane in lane_order
    }
    for element in elements:
        lane = str(element.get("lane", "voice"))
        if lane not in lane_to_elements:
            lane_to_elements[lane] = []
            lane_order.append(lane)
        lane_to_elements[lane].append(element)

    lane_count = max(1, len(lane_order))
    lane_width = 1.0 / lane_count
    rects: dict[str, dict[str, float]] = {}
    for lane_index, lane in enumerate(lane_order):
        lane_elements = lane_to_elements.get(lane, [])
        lane_elements.sort(
            key=lambda element: _safe_float(
                (state_map.get(str(element.get("id", "")), {}) or {}).get(
                    "priority", 0.0
                ),
                0.0,
            ),
            reverse=True,
        )
        y_cursor = 0.0
        for element in lane_elements:
            element_id = str(element.get("id", ""))
            state = state_map.get(element_id, {})
            height = max(
                0.12, min(0.38, _safe_float(state.get("area", 0.12), 0.12) * 1.85)
            )
            if y_cursor + height > 1.0:
                height = max(0.08, 1.0 - y_cursor)
            if height <= 0.0:
                break
            rects[element_id] = {
                "x": round((lane_index * lane_width) + 0.008, 4),
                "y": round(min(0.98, y_cursor + 0.01), 4),
                "w": round(max(0.05, lane_width - 0.016), 4),
                "h": round(max(0.08, height - 0.018), 4),
            }
            y_cursor += height
    return rects


def _build_projection_rects_grid(
    states: list[dict[str, Any]],
    *,
    perspective: str,
) -> dict[str, dict[str, float]]:
    if perspective == "causal-time":
        ordered = sorted(
            states,
            key=lambda state: (
                _safe_float(
                    (state.get("explain", {}) or {}).get("causal_signal", 0.0), 0.0
                ),
                _safe_float(state.get("priority", 0.0), 0.0),
            ),
            reverse=True,
        )
    else:
        ordered = sorted(
            states,
            key=lambda state: _safe_float(state.get("priority", 0.0), 0.0),
            reverse=True,
        )

    placements: list[dict[str, Any]] = []
    cursor_x = 0
    cursor_y = 0
    row_height = 0

    for state in ordered:
        area = _safe_float(state.get("area", 0.1), 0.1)
        mass = _safe_float(state.get("mass", 0.0), 0.0)
        pulse = _safe_float(state.get("pulse", 0.0), 0.0)
        width_units = max(3, min(12, int(round(area * 22))))
        height_units = max(2, min(5, int(round(1 + (mass * 2.4) + (pulse * 1.3)))))
        if cursor_x + width_units > 12:
            cursor_y += row_height
            cursor_x = 0
            row_height = 0
        placements.append(
            {
                "id": str(state.get("element_id", "")),
                "x": cursor_x,
                "y": cursor_y,
                "w": width_units,
                "h": height_units,
            }
        )
        cursor_x += width_units
        row_height = max(row_height, height_units)

    total_rows = max(1, cursor_y + row_height)
    rects: dict[str, dict[str, float]] = {}
    for placement in placements:
        element_id = placement["id"]
        if not element_id:
            continue
        rects[element_id] = {
            "x": round(placement["x"] / 12.0, 4),
            "y": round(placement["y"] / total_rows, 4),
            "w": round(placement["w"] / 12.0, 4),
            "h": round(placement["h"] / total_rows, 4),
        }
    return rects


def _build_projection_layout(
    elements: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    perspective: str,
    clamps: dict[str, Any],
) -> dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    if perspective_key == "swimlanes":
        rects = _build_projection_rects_swimlanes(states, elements)
    else:
        rects = _build_projection_rects_grid(states, perspective=perspective_key)

    for element in elements:
        element_id = str(element.get("id", ""))
        if not element_id or element_id in rects:
            continue
        rects[element_id] = {"x": 0.0, "y": 0.0, "w": 0.25, "h": 0.2}

    layout_seed = f"layout|{perspective_key}|{ts_ms}|{len(elements)}"
    return {
        "record": "映/layout.v1",
        "id": f"layout.{hashlib.sha1(layout_seed.encode('utf-8')).hexdigest()[:12]}",
        "ts": ts_ms,
        "perspective": perspective_key,
        "elements": [str(element.get("id", "")) for element in elements],
        "rects": rects,
        "states": states,
        "clamps": clamps,
        "notes": "Derived projection from field vectors + presence impacts + queue pressure.",
    }


def _build_projection_chat_sessions(
    elements: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    perspective: str,
) -> list[dict[str, Any]]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    state_map = {str(state.get("element_id", "")): state for state in states}
    sessions: list[dict[str, Any]] = []
    for element in elements:
        if str(element.get("kind", "")) != "chat-lens":
            continue
        element_id = str(element.get("id", ""))
        if not element_id:
            continue
        presence = str(element.get("presence", "witness_thread"))
        state = state_map.get(element_id, {})
        mass = _safe_float(state.get("mass", 0.0), 0.0)
        if perspective_key == "causal-time":
            memory_scope = "council"
        elif perspective_key == "swimlanes":
            memory_scope = "local"
        else:
            memory_scope = str(element.get("memory_scope", "shared"))
        sessions.append(
            {
                "record": "映/chat-session.v1",
                "id": f"chat-session:{presence}",
                "ts": ts_ms,
                "presence": presence,
                "lens_element": element_id,
                "field_bindings": dict(element.get("field_bindings", {})),
                "memory_scope": memory_scope,
                "tags": ["chat", "field-bound", perspective_key],
                "status": "active" if mass >= 0.42 else "listening",
            }
        )
    return sessions


def _build_projection_vector_view(
    field_snapshot: dict[str, Any],
    elements: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    perspective: str,
) -> dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    mode = {
        "hybrid": "axes",
        "causal-time": "barycentric-slice",
        "swimlanes": "cluster",
    }.get(perspective_key, "axes")
    state_by_id = {str(item.get("element_id", "")): item for item in states}
    ranked_elements = sorted(
        elements,
        key=lambda element: _safe_float(
            (state_by_id.get(str(element.get("id", "")), {}) or {}).get(
                "priority", 0.0
            ),
            0.0,
        ),
        reverse=True,
    )
    overlay_presences = [
        str(item.get("presence", ""))
        for item in ranked_elements
        if str(item.get("presence", "")).strip()
    ][:4]
    return {
        "record": "映/vector-view.v1",
        "id": f"vector-view:{perspective_key}",
        "ts": ts_ms,
        "field_snapshot": str(field_snapshot.get("id", "")),
        "mode": mode,
        "axes": {
            "x": "f3.coherence_focus",
            "y": "f2.witness_tension",
            "z": "f7.gate_pressure",
        },
        "overlay_reiso": list(field_snapshot.get("applied_reiso", []))[:6],
        "overlay_presences": overlay_presences,
        "show_causality": perspective_key != "swimlanes",
    }


def _build_projection_tick_view(
    field_snapshot: dict[str, Any],
    queue_snapshot: dict[str, Any],
    *,
    perspective: str,
) -> dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    merge = PROJECTION_PERSPECTIVES[perspective_key]["merge"]
    return {
        "record": "映/tick-view.v1",
        "id": f"tick-view:{perspective_key}",
        "ts": ts_ms,
        "sources": [
            *list(field_snapshot.get("ticks", [])),
            f"queue.pending.{int(_safe_float(queue_snapshot.get('pending_count', 0), 0.0))}",
            f"queue.events.{int(_safe_float(queue_snapshot.get('event_count', 0), 0.0))}",
        ],
        "window": {
            "lookback_seconds": 120,
            "sample_ms": int(max(50, SIM_TICK_SECONDS * 1000)),
        },
        "show_causal": perspective_key != "hybrid",
        "merge": merge,
    }


def build_ui_projection(
    catalog: dict[str, Any],
    simulation: dict[str, Any] | None = None,
    *,
    perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
    queue_snapshot: dict[str, Any] | None = None,
    influence_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    perspective_key = normalize_projection_perspective(perspective)
    queue = queue_snapshot or catalog.get("task_queue", {})
    influence = _projection_source_influence(simulation, catalog, influence_snapshot)
    field_snapshot = _build_projection_field_snapshot(
        catalog,
        simulation,
        perspective=perspective_key,
        queue_snapshot=queue,
        influence_snapshot=influence,
    )
    coherence_state = _build_projection_coherence_state(
        field_snapshot,
        perspective=perspective_key,
    )
    elements, states, clamps = _build_projection_element_states(
        field_snapshot,
        coherence_state,
        perspective=perspective_key,
        simulation=simulation,
        influence_snapshot=influence,
        catalog=catalog,
    )
    layout = _build_projection_layout(
        elements,
        states,
        perspective=perspective_key,
        clamps=clamps,
    )
    chat_sessions = _build_projection_chat_sessions(
        elements,
        states,
        perspective=perspective_key,
    )
    vector_view = _build_projection_vector_view(
        field_snapshot,
        elements,
        states,
        perspective=perspective_key,
    )
    tick_view = _build_projection_tick_view(
        field_snapshot,
        queue,
        perspective=perspective_key,
    )

    return {
        "record": "家_映.v1",
        "contract": "家_映.v1",
        "ts": int(time.time() * 1000),
        "perspective": perspective_key,
        "default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
        "perspectives": projection_perspective_options(),
        "field_schemas": PROJECTION_FIELD_SCHEMAS,
        "field_snapshot": field_snapshot,
        "coherence": coherence_state,
        "elements": elements,
        "states": states,
        "layout": layout,
        "chat_sessions": chat_sessions,
        "vector_view": vector_view,
        "tick_view": tick_view,
        "queue": {
            "pending_count": int(_safe_float(queue.get("pending_count", 0), 0.0)),
            "event_count": int(_safe_float(queue.get("event_count", 0), 0.0)),
        },
    }


def attach_ui_projection(
    catalog: dict[str, Any],
    simulation: dict[str, Any] | None = None,
    *,
    perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
    queue_snapshot: dict[str, Any] | None = None,
    influence_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    projection = build_ui_projection(
        catalog,
        simulation,
        perspective=perspective,
        queue_snapshot=queue_snapshot,
        influence_snapshot=influence_snapshot,
    )
    catalog["ui_default_perspective"] = PROJECTION_DEFAULT_PERSPECTIVE
    catalog["ui_perspectives"] = projection_perspective_options()
    catalog["ui_projection"] = projection
    return projection
