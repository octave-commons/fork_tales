"""Backend field particle synthesis helpers for simulation."""

from __future__ import annotations

import colorsys
import math
import time
from typing import Any

from .constants import (
    ENTITY_MANIFEST,
    FIELD_TO_PRESENCE,
    _DAIMO_DYNAMICS_LOCK,
    _DAIMO_DYNAMICS_CACHE,
)
from .metrics import _safe_float, _clamp01, _stable_ratio
from .daimoi_probabilistic import _simplex_noise_2d


def _build_backend_field_particles(
    *,
    file_graph: dict[str, Any] | None,
    presence_impacts: list[dict[str, Any]],
    resource_heartbeat: dict[str, Any],
    compute_jobs: list[dict[str, Any]],
    now: float,
) -> list[dict[str, float | str]]:
    if not presence_impacts:
        return []

    file_nodes_raw = (
        file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
    )
    file_nodes = [row for row in file_nodes_raw if isinstance(row, dict)]
    embedding_nodes_raw = (
        file_graph.get("embedding_particles", [])
        if isinstance(file_graph, dict)
        else []
    )
    embedding_nodes = [row for row in embedding_nodes_raw if isinstance(row, dict)]

    manifest_by_id = {
        str(row.get("id", "")).strip(): row
        for row in ENTITY_MANIFEST
        if str(row.get("id", "")).strip()
    }
    presence_to_field: dict[str, str] = {}
    for field_id, presence_id in FIELD_TO_PRESENCE.items():
        pid = str(presence_id).strip()
        if pid and pid not in presence_to_field:
            presence_to_field[pid] = str(field_id).strip()

    devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    if not isinstance(devices, dict):
        devices = {}
    resource_pressure = 0.0
    for device_key in ("cpu", "gpu1", "gpu2", "npu0"):
        row = devices.get(device_key, {})
        util = _safe_float(
            (row if isinstance(row, dict) else {}).get("utilization", 0.0), 0.0
        )
        resource_pressure = max(resource_pressure, _clamp01(util / 100.0))

    compute_pressure = _clamp01(len(compute_jobs) / 24.0)

    field_particles: list[dict[str, float | str]] = []
    now_mono = time.monotonic()
    live_ids: set[str] = set()

    def _node_field_similarity(
        node: dict[str, Any], target_field_id: str, target_presence_id: str
    ) -> float:
        if not target_field_id:
            return 0.0
        score = 0.0
        field_scores = node.get("field_scores", {})
        if isinstance(field_scores, dict):
            score = _clamp01(_safe_float(field_scores.get(target_field_id, 0.0), 0.0))
        dominant_field = str(node.get("dominant_field", "")).strip()
        if dominant_field and dominant_field == target_field_id:
            score = max(score, 0.85)
        dominant_presence = str(node.get("dominant_presence", "")).strip()
        if dominant_presence and dominant_presence == target_presence_id:
            score = max(score, 1.0)
        return _clamp01(score)

    with _DAIMO_DYNAMICS_LOCK:
        runtime = _DAIMO_DYNAMICS_CACHE.get("field_particles", {})
        if not isinstance(runtime, dict):
            runtime = {}
        # Handle nested runtime structure: {'particles': {...}, 'surfaces': {...}}
        particle_cache = runtime.get("particles", {})
        if not isinstance(particle_cache, dict):
            particle_cache = {}

        for impact in presence_impacts:
            presence_id = str(impact.get("id", "")).strip()
            if not presence_id:
                continue

            presence_meta = manifest_by_id.get(presence_id, {})
            anchor_x = _clamp01(
                _safe_float(
                    presence_meta.get("x", _stable_ratio(f"{presence_id}|anchor", 3)),
                    _stable_ratio(f"{presence_id}|anchor", 3),
                )
            )
            anchor_y = _clamp01(
                _safe_float(
                    presence_meta.get("y", _stable_ratio(f"{presence_id}|anchor", 9)),
                    _stable_ratio(f"{presence_id}|anchor", 9),
                )
            )
            base_hue = _safe_float(presence_meta.get("hue", 200.0), 200.0)
            target_field_id = presence_to_field.get(presence_id, "")
            presence_role, particle_mode = _particle_role_and_mode_for_presence(
                presence_id
            )

            affected_by = (
                impact.get("affected_by", {}) if isinstance(impact, dict) else {}
            )
            affects = impact.get("affects", {}) if isinstance(impact, dict) else {}
            file_influence = _clamp01(
                _safe_float(
                    (affected_by if isinstance(affected_by, dict) else {}).get(
                        "files", 0.0
                    ),
                    0.0,
                )
            )
            world_influence = _clamp01(
                _safe_float(
                    (affects if isinstance(affects, dict) else {}).get("world", 0.0),
                    0.0,
                )
            )
            ledger_influence = _clamp01(
                _safe_float(
                    (affects if isinstance(affects, dict) else {}).get("ledger", 0.0),
                    0.0,
                )
            )

            node_signals: list[dict[str, float]] = []
            cluster_map: dict[tuple[int, int], dict[str, float]] = {}
            cluster_bucket_size = 0.18
            local_density_score = 0.0
            for node in file_nodes:
                nx = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
                ny = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
                field_similarity = _node_field_similarity(
                    node, target_field_id, presence_id
                )
                embed_signal = _clamp01(
                    (_safe_float(node.get("embed_layer_count", 0.0), 0.0) / 3.0)
                    + (
                        0.35
                        if str(node.get("vecstore_collection", "")).strip()
                        else 0.0
                    )
                )
                signed_similarity = max(
                    -1.0,
                    min(
                        1.0,
                        (field_similarity * 0.72) + (embed_signal * 0.34) - 0.43,
                    ),
                )
                node_importance = _clamp01(
                    _safe_float(node.get("importance", 0.25), 0.25)
                )
                distance_to_anchor = math.sqrt(
                    ((nx - anchor_x) * (nx - anchor_x))
                    + ((ny - anchor_y) * (ny - anchor_y))
                )
                anchor_proximity = _clamp01(1.0 - (distance_to_anchor / 0.55))
                relevance = (
                    (abs(signed_similarity) * 0.62)
                    + (node_importance * 0.24)
                    + (anchor_proximity * 0.14)
                )
                if relevance < 0.12 and anchor_proximity <= 0.04:
                    continue

                if distance_to_anchor <= 0.24:
                    local_density_score += _clamp01(
                        1.0 - (distance_to_anchor / 0.24)
                    ) * (0.35 + (node_importance * 0.65))

                node_signals.append(
                    {
                        "x": nx,
                        "y": ny,
                        "signed": signed_similarity,
                        "importance": node_importance,
                        "relevance": relevance,
                    }
                )

                cluster_key = (
                    int(nx / cluster_bucket_size),
                    int(ny / cluster_bucket_size),
                )
                cluster_weight = (
                    0.24 + (node_importance * 0.64) + (abs(signed_similarity) * 0.82)
                )
                cluster_row = cluster_map.setdefault(
                    cluster_key,
                    {
                        "xw": 0.0,
                        "yw": 0.0,
                        "signed": 0.0,
                        "weight_raw": 0.0,
                    },
                )
                cluster_row["xw"] += nx * cluster_weight
                cluster_row["yw"] += ny * cluster_weight
                cluster_row["signed"] += signed_similarity * cluster_weight
                cluster_row["weight_raw"] += cluster_weight

            if len(node_signals) > 140:
                node_signals.sort(
                    key=lambda row: _safe_float(row.get("relevance", 0.0), 0.0),
                    reverse=True,
                )
                node_signals = node_signals[:140]

            clusters: list[dict[str, float]] = []
            for cluster_row in cluster_map.values():
                weight_raw = _safe_float(cluster_row.get("weight_raw", 0.0), 0.0)
                if weight_raw <= 1e-8:
                    continue
                clusters.append(
                    {
                        "x": _clamp01(
                            _safe_float(cluster_row.get("xw", 0.0), 0.0) / weight_raw
                        ),
                        "y": _clamp01(
                            _safe_float(cluster_row.get("yw", 0.0), 0.0) / weight_raw
                        ),
                        "signed": max(
                            -1.0,
                            min(
                                1.0,
                                _safe_float(cluster_row.get("signed", 0.0), 0.0)
                                / weight_raw,
                            ),
                        ),
                        "weight_raw": weight_raw,
                        "weight": 0.0,
                    }
                )
            clusters.sort(
                key=lambda row: _safe_float(row.get("weight_raw", 0.0), 0.0),
                reverse=True,
            )
            if len(clusters) > 8:
                clusters = clusters[:8]

            cluster_weight_total = 0.0
            for row in clusters:
                cluster_weight_total += _safe_float(row.get("weight_raw", 0.0), 0.0)
            if cluster_weight_total > 1e-8:
                for row in clusters:
                    row["weight"] = _clamp01(
                        _safe_float(row.get("weight_raw", 0.0), 0.0)
                        / cluster_weight_total
                    )

            local_density_ratio = _clamp01(local_density_score / 3.0)
            cluster_ratio = _clamp01(len(clusters) / 6.0)

            field_center_x = anchor_x
            field_center_y = anchor_y
            if clusters:
                primary_cluster = clusters[0]
                cluster_pull = _clamp01(
                    0.22
                    + (local_density_ratio * 0.42)
                    + (file_influence * 0.28)
                    + (cluster_ratio * 0.2)
                )
                field_center_x = _clamp01(
                    (anchor_x * (1.0 - cluster_pull))
                    + (
                        _safe_float(primary_cluster.get("x", anchor_x), anchor_x)
                        * cluster_pull
                    )
                )
                field_center_y = _clamp01(
                    (anchor_y * (1.0 - cluster_pull))
                    + (
                        _safe_float(primary_cluster.get("y", anchor_y), anchor_y)
                        * cluster_pull
                    )
                )

            raw_count = (
                4.0
                + (world_influence * 4.0)
                + (file_influence * 4.2)
                + (local_density_ratio * 8.6)
                + (cluster_ratio * 2.2)
                - (resource_pressure * 1.2)
            )
            particle_count = max(4, min(22, int(round(raw_count))))

            short_range_radius = 0.16 + (local_density_ratio * 0.04)
            interaction_radius = 0.36
            long_range_radius = 0.92
            spread = max(
                0.028,
                min(
                    0.14,
                    0.072 + (local_density_ratio * 0.056) + (cluster_ratio * 0.022),
                ),
            )
            peer_repulsion_radius = max(0.032, min(0.18, spread * 1.35))
            peer_repulsion_strength = (
                0.00022 + (local_density_ratio * 0.00088) + (cluster_ratio * 0.00034)
            )

            for local_index in range(particle_count):
                particle_id = f"field:{presence_id}:{local_index}"
                live_ids.add(particle_id)
                cache_row = particle_cache.get(particle_id, {})
                if not isinstance(cache_row, dict):
                    cache_row = {}

                seed_ratio = _stable_ratio(f"{particle_id}|seed", local_index + 11)
                home_dx = (
                    (_stable_ratio(f"{particle_id}|home-x", local_index + 19) * 2.0)
                    - 1.0
                ) * spread
                home_dy = (
                    (
                        (_stable_ratio(f"{particle_id}|home-y", local_index + 29) * 2.0)
                        - 1.0
                    )
                    * spread
                    * 0.82
                )
                home_x = _clamp01(field_center_x + home_dx)
                home_y = _clamp01(field_center_y + home_dy)

                px = _clamp01(_safe_float(cache_row.get("x", home_x), home_x))
                py = _clamp01(_safe_float(cache_row.get("y", home_y), home_y))
                pvx = _safe_float(cache_row.get("vx", 0.0), 0.0)
                pvy = _safe_float(cache_row.get("vy", 0.0), 0.0)

                fx = (home_x - px) * (0.18 + (ledger_influence * 0.18))
                fy = (home_y - py) * (0.18 + (ledger_influence * 0.18))

                if particle_count > 1 and peer_repulsion_strength > 1e-9:
                    for peer_index in range(particle_count):
                        if peer_index == local_index:
                            continue
                        peer_id = f"field:{presence_id}:{peer_index}"
                        peer_home_dx = (
                            (
                                _stable_ratio(
                                    f"{peer_id}|home-x",
                                    peer_index + 19,
                                )
                                * 2.0
                            )
                            - 1.0
                        ) * spread
                        peer_home_dy = (
                            (
                                (
                                    _stable_ratio(
                                        f"{peer_id}|home-y",
                                        peer_index + 29,
                                    )
                                    * 2.0
                                )
                                - 1.0
                            )
                            * spread
                            * 0.82
                        )
                        peer_home_x = _clamp01(field_center_x + peer_home_dx)
                        peer_home_y = _clamp01(field_center_y + peer_home_dy)
                        peer_cache_row = particle_cache.get(peer_id, {})
                        if isinstance(peer_cache_row, dict):
                            peer_x = _clamp01(
                                _safe_float(
                                    peer_cache_row.get("x", peer_home_x),
                                    peer_home_x,
                                )
                            )
                            peer_y = _clamp01(
                                _safe_float(
                                    peer_cache_row.get("y", peer_home_y),
                                    peer_home_y,
                                )
                            )
                        else:
                            peer_x = peer_home_x
                            peer_y = peer_home_y

                        repel_dx = px - peer_x
                        repel_dy = py - peer_y
                        repel_distance = math.sqrt(
                            (repel_dx * repel_dx) + (repel_dy * repel_dy)
                        )
                        if repel_distance <= 1e-8:
                            repel_angle = (
                                (
                                    seed_ratio
                                    + _stable_ratio(
                                        f"{particle_id}|peer:{peer_index}",
                                        peer_index + 53,
                                    )
                                )
                                % 1.0
                            ) * (math.pi * 2.0)
                            fx += math.cos(repel_angle) * (
                                peer_repulsion_strength * 0.9
                            )
                            fy += math.sin(repel_angle) * (
                                peer_repulsion_strength * 0.9
                            )
                            continue
                        if repel_distance >= peer_repulsion_radius:
                            continue
                        repel_falloff = _clamp01(
                            1.0 - (repel_distance / peer_repulsion_radius)
                        )
                        repel_strength = peer_repulsion_strength * (
                            repel_falloff * repel_falloff
                        )
                        fx += (repel_dx / repel_distance) * repel_strength
                        fy += (repel_dy / repel_distance) * repel_strength

                for node in node_signals:
                    dx = _safe_float(node.get("x", 0.5), 0.5) - px
                    dy = _safe_float(node.get("y", 0.5), 0.5) - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= 1e-8 or distance > interaction_radius:
                        continue

                    signed_similarity = max(
                        -1.0,
                        min(1.0, _safe_float(node.get("signed", 0.0), 0.0)),
                    )
                    if abs(signed_similarity) <= 0.03:
                        continue
                    node_importance = _clamp01(
                        _safe_float(node.get("importance", 0.25), 0.25)
                    )

                    if distance <= short_range_radius:
                        falloff = _clamp01(1.0 - (distance / short_range_radius))
                        strength = (
                            (0.00125 + (node_importance * 0.00245))
                            * (falloff * falloff)
                            * (0.78 + (abs(signed_similarity) * 0.94))
                            * (0.72 + (file_influence * 0.58))
                        )
                    else:
                        transition = max(1e-8, interaction_radius - short_range_radius)
                        band = _clamp01((interaction_radius - distance) / transition)
                        strength = (
                            (0.00024 + (node_importance * 0.00082))
                            * band
                            * (0.46 + (abs(signed_similarity) * 0.54))
                        )

                    direction = 1.0 if signed_similarity >= 0.0 else -1.0
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                for cluster in clusters:
                    dx = _safe_float(cluster.get("x", 0.5), 0.5) - px
                    dy = _safe_float(cluster.get("y", 0.5), 0.5) - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= short_range_radius or distance > long_range_radius:
                        continue

                    cluster_signed = max(
                        -1.0,
                        min(1.0, _safe_float(cluster.get("signed", 0.0), 0.0)),
                    )
                    if abs(cluster_signed) <= 0.04:
                        continue
                    cluster_weight = _clamp01(
                        _safe_float(cluster.get("weight", 0.0), 0.0)
                    )
                    range_span = max(1e-8, long_range_radius - short_range_radius)
                    falloff = _clamp01((long_range_radius - distance) / range_span)
                    strength = (
                        (0.00012 + (cluster_weight * 0.00044))
                        * falloff
                        * (0.54 + (abs(cluster_signed) * 0.56))
                        * (0.6 + (cluster_ratio * 0.5))
                    )
                    direction = 1.0 if cluster_signed >= 0.0 else -1.0
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                for embed in embedding_nodes:
                    ex = _clamp01(_safe_float(embed.get("x", 0.5), 0.5))
                    ey = _clamp01(_safe_float(embed.get("y", 0.5), 0.5))
                    dx = ex - px
                    dy = ey - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= 1e-8 or distance > 0.23:
                        continue
                    falloff = _clamp01(1.0 - (distance / 0.23))
                    if falloff <= 0.0:
                        continue
                    cohesion = _clamp01(_safe_float(embed.get("cohesion", 0.0), 0.0))
                    density = _clamp01(
                        _safe_float(embed.get("text_density", 0.45), 0.45)
                    )
                    signed = (
                        (file_influence * 0.74)
                        + (cohesion * 0.52)
                        + (density * 0.26)
                        - 0.58
                    )
                    direction = 1.0 if signed >= 0.0 else -1.0
                    strength = (0.00042 + (abs(signed) * 0.00108)) * (falloff * falloff)
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                jitter_angle = (now * (0.34 + (compute_pressure * 0.4))) + (
                    local_index * 0.93
                )
                jitter_power = (
                    0.00006
                    + ((1.0 - resource_pressure) * 0.0001)
                    + (local_density_ratio * 0.00005)
                )
                fx += math.cos(jitter_angle) * jitter_power
                fy += math.sin(jitter_angle) * jitter_power

                simplex_phase = now * (0.28 + (compute_pressure * 0.24))
                simplex_amp = (
                    0.00005
                    + ((1.0 - resource_pressure) * 0.00007)
                    + (local_density_ratio * 0.00004)
                )
                simplex_seed = (local_index + 1) * 73 + len(presence_id)
                simplex_x = _simplex_noise_2d(
                    (px * 4.8) + simplex_phase + (local_index * 0.23),
                    (py * 4.8) + (simplex_phase * 0.69),
                    seed=simplex_seed,
                )
                simplex_y = _simplex_noise_2d(
                    (px * 4.8) + 13.0 + (simplex_phase * 0.57),
                    (py * 4.8) + 29.0 + simplex_phase,
                    seed=simplex_seed + 41,
                )
                fx += simplex_x * simplex_amp
                fy += simplex_y * simplex_amp

                damping = max(0.74, 0.91 - (resource_pressure * 0.13))
                vx = (pvx * damping) + fx
                vy = (pvy * damping) + fy
                speed = math.sqrt((vx * vx) + (vy * vy))
                speed_limit = (
                    0.0042
                    + ((1.0 - resource_pressure) * 0.0021)
                    + (local_density_ratio * 0.0018)
                )
                if speed > speed_limit and speed > 1e-8:
                    scale = speed_limit / speed
                    vx *= scale
                    vy *= scale

                nx = _clamp01(px + vx)
                ny = _clamp01(py + vy)
                particle_cache[particle_id] = {
                    "x": nx,
                    "y": ny,
                    "vx": vx,
                    "vy": vy,
                    "ts": now_mono,
                }

                saturation = max(
                    0.32,
                    min(
                        0.58,
                        0.4 + (world_influence * 0.16) + (local_density_ratio * 0.06),
                    ),
                )
                value = max(
                    0.38,
                    min(
                        0.68,
                        0.48
                        + (ledger_influence * 0.12)
                        + (local_density_ratio * 0.06)
                        - (resource_pressure * 0.12),
                    ),
                )
                r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                    (base_hue % 360.0) / 360.0,
                    saturation,
                    value,
                )
                particle_size = (
                    0.9
                    + (world_influence * 1.0)
                    + (file_influence * 0.8)
                    + (local_density_ratio * 0.9)
                )

                field_particles.append(
                    {
                        "id": particle_id,
                        "presence_id": presence_id,
                        "presence_role": presence_role,
                        "particle_mode": particle_mode,
                        "x": round(nx, 5),
                        "y": round(ny, 5),
                        "size": round(particle_size, 5),
                        "r": round(_clamp01(r_raw), 5),
                        "g": round(_clamp01(g_raw), 5),
                        "b": round(_clamp01(b_raw), 5),
                    }
                )

        # RENDER CHAOS BUTTERFLIES - convert chaos particles to field particles
        chaos_hue = 300.0  # Purple for chaos
        for pid, particle_state in particle_cache.items():
            if not isinstance(particle_state, dict):
                continue
            if not bool(particle_state.get("is_chaos_butterfly", False)):
                continue

            # Add to live_ids so they don't get cleaned up
            live_ids.add(pid)

            nx = _clamp01(_safe_float(particle_state.get("x", 0.5), 0.5))
            ny = _clamp01(_safe_float(particle_state.get("y", 0.5), 0.5))
            particle_size = _safe_float(particle_state.get("size", 0.5), 0.5)

            # Chaos butterflies have distinct purple color with high saturation
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (chaos_hue % 360.0) / 360.0,
                0.85,  # High saturation
                0.92,  # High brightness
            )

            field_particles.append(
                {
                    "id": pid,
                    "presence_id": "chaos_butterfly",
                    "presence_role": "chaos-agent",
                    "particle_mode": "noise-spreader",
                    "x": round(nx, 5),
                    "y": round(ny, 5),
                    "size": round(particle_size, 5),
                    "r": round(_clamp01(r_raw), 5),
                    "g": round(_clamp01(g_raw), 5),
                    "b": round(_clamp01(b_raw), 5),
                }
            )

        stale_before = now_mono - 180.0
        for pid in list(particle_cache.keys()):
            if pid in live_ids:
                continue
            row = particle_cache.get(pid, {})
            ts_value = _safe_float(
                (row if isinstance(row, dict) else {}).get("ts", 0.0), 0.0
            )
            if ts_value < stale_before:
                particle_cache.pop(pid, None)

        # Preserve nested runtime structure when saving back
        runtime["particles"] = particle_cache
        _DAIMO_DYNAMICS_CACHE["field_particles"] = runtime

    field_particles.sort(
        key=lambda row: (
            str(row.get("presence_id", "")),
            str(row.get("id", "")),
        )
    )
    return field_particles


_PARTICLE_ROLE_BY_PRESENCE: dict[str, str] = {
    "witness_thread": "crawl-routing",
    "keeper_of_receipts": "file-analysis",
    "mage_of_receipts": "image-captioning",
    "anchor_registry": "council-orchestration",
    "gates_of_truth": "compliance-gating",
}


def _particle_role_and_mode_for_presence(presence_id: str) -> tuple[str, str]:
    clean_presence_id = str(presence_id).strip()
    if not clean_presence_id:
        return "neutral", "neutral"
    role = str(_PARTICLE_ROLE_BY_PRESENCE.get(clean_presence_id, "")).strip()
    if not role:
        return "neutral", "neutral"
    return role, "role-bound"
