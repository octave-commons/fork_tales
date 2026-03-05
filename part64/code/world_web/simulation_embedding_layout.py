"""Embedding/layout projection helpers for simulation file graphs."""

from __future__ import annotations

import colorsys
import math
import time
from typing import Any

from .metrics import _clamp01, _safe_float, _stable_ratio
from .daimoi_probabilistic import _simplex_noise_2d
from . import simulation_document_layout as simulation_document_layout_module


def apply_file_graph_document_similarity_layout(
    file_graph: dict[str, Any],
    *,
    now: float | None = None,
    summary_chars: int = 300,
    excerpt_chars: int = 520,
) -> list[dict[str, float]]:
    file_nodes_raw = file_graph.get("file_nodes", [])
    if not isinstance(file_nodes_raw, list) or len(file_nodes_raw) <= 0:
        file_graph["embedding_particles"] = []
        return []

    entries: list[dict[str, Any]] = []
    for index, node in enumerate(file_nodes_raw):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip() or f"file:{index}"
        x = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        importance = _clamp01(_safe_float(node.get("importance", 0.2), 0.2))
        local_range = (
            simulation_document_layout_module.document_layout_range_from_importance(
                importance
            )
        )
        tokens = simulation_document_layout_module.document_layout_tokens(
            node,
            summary_chars=summary_chars,
            excerpt_chars=excerpt_chars,
        )
        semantic_vector = (
            simulation_document_layout_module.document_layout_semantic_vector(
                node,
                tokens,
            )
        )
        text_density = simulation_document_layout_module.document_layout_text_density(
            node,
            tokens,
        )
        entries.append(
            {
                "id": node_id,
                "index": len(entries),
                "node": node,
                "x": x,
                "y": y,
                "importance": importance,
                "range": local_range,
                "embedded": simulation_document_layout_module.document_layout_is_embedded(
                    node
                ),
                "tokens": tokens,
                "vector": semantic_vector,
                "text_density": text_density,
            }
        )

    if not entries:
        file_graph["embedding_particles"] = []
        return []

    cell_size = 0.08
    grid: dict[tuple[int, int], list[int]] = {}
    for index, entry in enumerate(entries):
        gx = int(entry["x"] / cell_size)
        gy = int(entry["y"] / cell_size)
        grid.setdefault((gx, gy), []).append(index)

    offsets: list[list[float]] = [[0.0, 0.0] for _ in entries]
    if len(entries) > 1:
        for index, left in enumerate(entries):
            gx = int(left["x"] / cell_size)
            gy = int(left["y"] / cell_size)
            radius_cells = max(1, int(math.ceil(left["range"] / cell_size)))

            for oy in range(-radius_cells, radius_cells + 1):
                for ox in range(-radius_cells, radius_cells + 1):
                    bucket = grid.get((gx + ox, gy + oy), [])
                    for other_index in bucket:
                        if other_index <= index:
                            continue
                        right = entries[other_index]
                        pair_range = max(left["range"], right["range"])
                        dx = right["x"] - left["x"]
                        dy = right["y"] - left["y"]
                        distance = math.sqrt((dx * dx) + (dy * dy))
                        if distance <= 1e-8 or distance > pair_range:
                            continue

                        similarity = simulation_document_layout_module.document_layout_similarity(
                            left["node"],
                            right["node"],
                            left.get("tokens", []),
                            right.get("tokens", []),
                        )
                        semantic_signed = max(
                            -1.0, min(1.0, (similarity - 0.52) / 0.48)
                        )
                        mixed_embedding = bool(left["embedded"]) != bool(
                            right["embedded"]
                        )
                        signed_similarity = (
                            -max(0.46, abs(semantic_signed) * 0.72)
                            if mixed_embedding
                            else semantic_signed
                        )
                        if abs(signed_similarity) < 0.22:
                            continue

                        falloff = _clamp01(1.0 - (distance / max(pair_range, 1e-6)))
                        importance_mix = (
                            left["importance"] + right["importance"]
                        ) * 0.5
                        density_mix = (
                            _safe_float(left.get("text_density"), 0.45)
                            + _safe_float(right.get("text_density"), 0.45)
                        ) * 0.5
                        strength = (
                            falloff
                            * abs(signed_similarity)
                            * (1.2 if mixed_embedding else 1.0)
                            * (0.00145 + (importance_mix * 0.0022))
                            * (0.66 + (density_mix * 0.3))
                        )
                        if strength <= 0.0:
                            continue

                        ux = dx / distance
                        uy = dy / distance
                        direction = 1.0 if signed_similarity >= 0.0 else -1.0
                        fx = ux * strength * direction
                        fy = uy * strength * direction

                        offsets[index][0] += fx
                        offsets[index][1] += fy
                        offsets[other_index][0] -= fx
                        offsets[other_index][1] -= fy

    embedding_particle_points: list[dict[str, float]] = []
    embedding_particle_nodes: list[dict[str, float | str]] = []
    embedded_entries = [entry for entry in entries if bool(entry.get("embedded"))]
    if embedded_entries:
        now_seconds = _safe_float(now, time.time()) if now is not None else time.time()
        particle_count = max(6, min(42, int(round(len(embedded_entries) * 1.8))))
        particles: list[dict[str, Any]] = []
        source_weights = [
            max(0.08, _safe_float(entry.get("text_density", 0.45), 0.45))
            for entry in embedded_entries
        ]
        source_weight_total = sum(source_weights)

        for index in range(particle_count):
            source = embedded_entries[index % len(embedded_entries)]
            if source_weight_total > 1e-8 and len(embedded_entries) > 1:
                ratio_slot = (float(index) + 0.5) / float(max(1, particle_count))
                cumulative = 0.0
                for entry, weight in zip(embedded_entries, source_weights):
                    cumulative += weight / source_weight_total
                    if ratio_slot <= cumulative:
                        source = entry
                        break
            seed = f"{source['id']}|particle|{index}"
            scatter = 0.006 + (
                _stable_ratio(seed, 31)
                * max(0.018, _safe_float(source["range"], 0.03) * 0.64)
            )
            seed_x = (_stable_ratio(seed, 47) * 2.0) - 1.0
            seed_y = (_stable_ratio(seed, 53) * 2.0) - 1.0
            x = _clamp01(_safe_float(source["x"], 0.5) + (seed_x * scatter))
            y = _clamp01(_safe_float(source["y"], 0.5) + (seed_y * scatter * 0.82))
            particles.append(
                {
                    "id": f"embed-particle:{index}",
                    "x": x,
                    "y": y,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vector": list(source.get("vector", [])),
                    "text_density": _safe_float(source.get("text_density"), 0.45),
                    "focus_x": x,
                    "focus_y": y,
                    "cohesion": 0.0,
                    "drift": (_stable_ratio(seed, 41) * 2.0) - 1.0,
                }
            )

        for _ in range(4):
            particle_forces: list[list[float]] = [[0.0, 0.0] for _ in particles]

            for particle_index, particle in enumerate(particles):
                influence_total = 0.0
                avg_x = 0.0
                avg_y = 0.0
                avg_vector = [0.0 for _ in particle.get("vector", [])]
                doc_radius = 0.22

                for entry in embedded_entries:
                    dx = _safe_float(entry["x"], 0.5) - _safe_float(particle["x"], 0.5)
                    dy = _safe_float(entry["y"], 0.5) - _safe_float(particle["y"], 0.5)
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance > doc_radius:
                        continue
                    if distance <= 1e-8:
                        jitter = (
                            _stable_ratio(
                                f"{particle['id']}|{entry['id']}|jitter",
                                particle_index + 1,
                            )
                            - 0.5
                        ) * 0.0012
                        dx += jitter
                        dy -= jitter
                        distance = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))

                    similarity = (
                        simulation_document_layout_module.semantic_vector_cosine(
                            particle.get("vector", []),
                            entry.get("vector", []),
                        )
                    )
                    distance_weight = _clamp01(1.0 - (distance / doc_radius))
                    density_weight = 0.24 + (
                        _safe_float(entry.get("text_density"), 0.45) * 0.92
                    )
                    similarity_weight = 0.28 + ((similarity + 1.0) * 0.36)
                    influence_weight = (
                        distance_weight
                        * distance_weight
                        * density_weight
                        * similarity_weight
                    )
                    if influence_weight <= 0.0:
                        continue

                    influence_total += influence_weight
                    avg_x += _safe_float(entry["x"], 0.5) * influence_weight
                    avg_y += _safe_float(entry["y"], 0.5) * influence_weight
                    entry_vector = entry.get("vector", [])
                    for axis in range(min(len(avg_vector), len(entry_vector))):
                        avg_vector[axis] += (
                            _safe_float(entry_vector[axis], 0.0) * influence_weight
                        )

                    direction = 1.0 if similarity >= 0.0 else -1.0
                    force_strength = (
                        (0.00072 + (abs(similarity) * 0.0024))
                        * distance_weight
                        * density_weight
                    )
                    particle_forces[particle_index][0] += (
                        (dx / distance) * force_strength * direction
                    )
                    particle_forces[particle_index][1] += (
                        (dy / distance) * force_strength * direction
                    )

                if influence_total > 0.0:
                    target_x = avg_x / influence_total
                    target_y = avg_y / influence_total
                    particle["focus_x"] = target_x
                    particle["focus_y"] = target_y
                    particle["cohesion"] = _clamp01(
                        (_safe_float(particle.get("cohesion", 0.0), 0.0) * 0.55)
                        + min(1.0, influence_total * 0.48)
                    )
                    pull_strength = min(0.0052, 0.0012 + (influence_total * 0.0019))
                    particle_forces[particle_index][0] += (
                        target_x - _safe_float(particle["x"], 0.5)
                    ) * pull_strength
                    particle_forces[particle_index][1] += (
                        target_y - _safe_float(particle["y"], 0.5)
                    ) * pull_strength

                    if avg_vector:
                        avg_magnitude = math.sqrt(
                            sum(value * value for value in avg_vector)
                        )
                        if avg_magnitude > 1e-8:
                            normalized_avg = [
                                value / avg_magnitude for value in avg_vector
                            ]
                            particle["vector"] = (
                                simulation_document_layout_module.semantic_vector_blend(
                                    list(particle.get("vector", [])),
                                    normalized_avg,
                                    0.26,
                                )
                            )
                else:
                    particle["cohesion"] = _clamp01(
                        _safe_float(particle.get("cohesion", 0.0), 0.0) * 0.86
                    )

            for left_index in range(len(particles)):
                left = particles[left_index]
                for right_index in range(left_index + 1, len(particles)):
                    right = particles[right_index]
                    dx = _safe_float(right["x"], 0.5) - _safe_float(left["x"], 0.5)
                    dy = _safe_float(right["y"], 0.5) - _safe_float(left["y"], 0.5)
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance > 0.2:
                        continue
                    if distance <= 1e-8:
                        jitter = (
                            _stable_ratio(
                                f"{left['id']}|{right['id']}|pair", left_index + 3
                            )
                            - 0.5
                        ) * 0.001
                        dx += jitter
                        dy -= jitter
                        distance = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))

                    similarity = (
                        simulation_document_layout_module.semantic_vector_cosine(
                            left.get("vector", []),
                            right.get("vector", []),
                        )
                    )
                    falloff = _clamp01(1.0 - (distance / 0.2))
                    pair_strength = (0.00044 + (abs(similarity) * 0.00186)) * falloff
                    direction = 1.0 if similarity >= 0.0 else -1.0
                    fx = (dx / distance) * pair_strength * direction
                    fy = (dy / distance) * pair_strength * direction

                    particle_forces[left_index][0] += fx
                    particle_forces[left_index][1] += fy
                    particle_forces[right_index][0] -= fx
                    particle_forces[right_index][1] -= fy

            for particle_index, particle in enumerate(particles):
                drift_phase = (
                    now_seconds
                    * (0.62 + abs(_safe_float(particle.get("drift", 0.0), 0.0)) * 0.42)
                ) + (particle_index * 0.41)
                particle_forces[particle_index][0] += math.cos(drift_phase) * 0.00021
                particle_forces[particle_index][1] += math.sin(drift_phase) * 0.00017

                particle_x = _safe_float(particle.get("x", 0.5), 0.5)
                particle_y = _safe_float(particle.get("y", 0.5), 0.5)
                simplex_amp = 0.00011 + (
                    abs(_safe_float(particle.get("drift", 0.0), 0.0)) * 0.00017
                )
                simplex_phase = now_seconds * 0.31
                simplex_x = _simplex_noise_2d(
                    (particle_x * 4.6) + (particle_index * 0.19) + simplex_phase,
                    (particle_y * 4.6) + (simplex_phase * 0.71),
                    seed=particle_index + 17,
                )
                simplex_y = _simplex_noise_2d(
                    (particle_x * 4.6) + 17.0 + (simplex_phase * 0.59),
                    (particle_y * 4.6) + 11.0 + simplex_phase,
                    seed=particle_index + 29,
                )
                particle_forces[particle_index][0] += simplex_x * simplex_amp
                particle_forces[particle_index][1] += simplex_y * simplex_amp

                vx = (
                    _safe_float(particle.get("vx", 0.0), 0.0)
                    + particle_forces[particle_index][0]
                ) * 0.84
                vy = (
                    _safe_float(particle.get("vy", 0.0), 0.0)
                    + particle_forces[particle_index][1]
                ) * 0.84
                speed = math.sqrt((vx * vx) + (vy * vy))
                speed_limit = 0.0062 + (
                    _safe_float(particle.get("text_density", 0.45), 0.45) * 0.0024
                )
                if speed > speed_limit and speed > 1e-8:
                    scale = speed_limit / speed
                    vx *= scale
                    vy *= scale

                particle["vx"] = vx
                particle["vy"] = vy
                particle["x"] = _clamp01(_safe_float(particle.get("x", 0.5), 0.5) + vx)
                particle["y"] = _clamp01(_safe_float(particle.get("y", 0.5), 0.5) + vy)

        if len(embedded_entries) > 1:
            for entry in embedded_entries:
                entry_index = int(entry.get("index", 0))
                if entry_index < 0 or entry_index >= len(offsets):
                    continue
                influence_x = 0.0
                influence_y = 0.0
                influence_radius = max(
                    0.08,
                    min(
                        0.26,
                        (_safe_float(entry.get("range", 0.03), 0.03) * 2.4) + 0.05,
                    ),
                )

                for particle in particles:
                    dx = _safe_float(particle.get("x", 0.5), 0.5) - _safe_float(
                        entry.get("x", 0.5), 0.5
                    )
                    dy = _safe_float(particle.get("y", 0.5), 0.5) - _safe_float(
                        entry.get("y", 0.5), 0.5
                    )
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance > influence_radius:
                        continue
                    if distance <= 1e-8:
                        continue

                    similarity = (
                        simulation_document_layout_module.semantic_vector_cosine(
                            entry.get("vector", []),
                            particle.get("vector", []),
                        )
                    )
                    falloff = _clamp01(1.0 - (distance / influence_radius))
                    density_mix = 0.58 + (
                        (
                            _safe_float(entry.get("text_density"), 0.45)
                            + _safe_float(particle.get("text_density"), 0.45)
                        )
                        * 0.24
                    )
                    strength = (
                        (0.00016 + (abs(similarity) * 0.00052)) * falloff * density_mix
                    )
                    direction = 1.0 if similarity >= 0.0 else -1.0
                    influence_x += (dx / distance) * strength * direction
                    influence_y += (dy / distance) * strength * direction

                max_influence = 0.0032 + (
                    _safe_float(entry.get("importance", 0.2), 0.2) * 0.0048
                )
                offsets[entry_index][0] += max(
                    -max_influence, min(max_influence, influence_x)
                )
                offsets[entry_index][1] += max(
                    -max_influence, min(max_influence, influence_y)
                )

        density_center_weight_total = sum(source_weights)
        if density_center_weight_total > 1e-8:
            density_center_x = (
                sum(
                    _safe_float(entry.get("x", 0.5), 0.5) * weight
                    for entry, weight in zip(embedded_entries, source_weights)
                )
                / density_center_weight_total
            )
            density_center_y = (
                sum(
                    _safe_float(entry.get("y", 0.5), 0.5) * weight
                    for entry, weight in zip(embedded_entries, source_weights)
                )
                / density_center_weight_total
            )
            density_spread = max(source_weights) - min(source_weights)
            center_pull = min(0.18, 0.06 + (density_spread * 0.09))
            for particle in particles:
                particle_x = _safe_float(particle.get("x", 0.5), 0.5)
                particle_y = _safe_float(particle.get("y", 0.5), 0.5)
                particle["x"] = _clamp01(
                    particle_x + ((density_center_x - particle_x) * center_pull)
                )
                particle["y"] = _clamp01(
                    particle_y + ((density_center_y - particle_y) * center_pull)
                )

        for particle in particles[:48]:
            hue = simulation_document_layout_module.semantic_vector_hue(
                list(particle.get("vector", []))
            )
            cohesion = _clamp01(_safe_float(particle.get("cohesion", 0.0), 0.0))
            saturation = max(0.52, min(0.92, 0.64 + (cohesion * 0.2)))
            value = max(0.72, min(0.98, 0.84 + (cohesion * 0.14)))
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (hue % 360.0) / 360.0,
                saturation,
                value,
            )
            size = (
                1.8
                + (_safe_float(particle.get("text_density", 0.45), 0.45) * 1.1)
                + (cohesion * 1.8)
            )
            x_norm = _clamp01(_safe_float(particle.get("x", 0.5), 0.5))
            y_norm = _clamp01(_safe_float(particle.get("y", 0.5), 0.5))
            embedding_particle_points.append(
                {
                    "x": round((x_norm * 2.0) - 1.0, 5),
                    "y": round(1.0 - (y_norm * 2.0), 5),
                    "size": round(size, 5),
                    "r": round(r_raw, 5),
                    "g": round(g_raw, 5),
                    "b": round(b_raw, 5),
                }
            )
            embedding_particle_nodes.append(
                {
                    "id": str(particle.get("id", "")),
                    "x": round(x_norm, 5),
                    "y": round(y_norm, 5),
                    "hue": round(hue, 4),
                    "cohesion": round(cohesion, 5),
                    "text_density": round(
                        _safe_float(particle.get("text_density", 0.45), 0.45), 5
                    ),
                }
            )

    file_graph["embedding_particles"] = embedding_particle_nodes

    position_by_id: dict[str, tuple[float, float]] = {}
    for index, entry in enumerate(entries):
        max_offset = 0.008 + (entry["importance"] * 0.014)
        offset_x = max(-max_offset, min(max_offset, offsets[index][0]))
        offset_y = max(-max_offset, min(max_offset, offsets[index][1]))
        x = round(_clamp01(entry["x"] + offset_x), 6)
        y = round(_clamp01(entry["y"] + offset_y), 6)
        entry["node"]["x"] = x
        entry["node"]["y"] = y
        position_by_id[entry["id"]] = (x, y)

    graph_nodes = file_graph.get("nodes", [])
    if isinstance(graph_nodes, list):
        for node in graph_nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("node_type", "")).strip().lower() != "file":
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            position = position_by_id.get(node_id)
            if position is None:
                continue
            node["x"] = position[0]
            node["y"] = position[1]

    return embedding_particle_points
