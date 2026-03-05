"""Collision and collision-noise helpers for simulation stream particles."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import defaultdict
from typing import Any, Callable


_SEMANTIC_COLLISION_BUFFER_LOCAL = threading.local()


def stream_particle_effective_mass(
    row: dict[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
) -> float:
    semantic_text_chars = max(0.0, safe_float(row.get("semantic_text_chars", 0.0), 0.0))
    semantic_mass = max(0.0, safe_float(row.get("semantic_mass", 0.0), 0.0))
    daimoi_energy = max(0.0, safe_float(row.get("daimoi_energy", 0.0), 0.0))
    message_probability = max(0.0, safe_float(row.get("message_probability", 0.0), 0.0))
    package_entropy = max(0.0, safe_float(row.get("package_entropy", 0.0), 0.0))

    text_term = math.log1p(semantic_text_chars) * 0.32
    energy_term = math.log1p((daimoi_energy * 2.8) + (message_probability * 3.5)) * 0.42
    entropy_term = package_entropy * 0.08
    mass_term = semantic_mass * 0.15
    return max(0.35, min(8.5, 0.5 + text_term + energy_term + entropy_term + mass_term))


def stream_particle_collision_radius(
    row: dict[str, Any],
    mass_value: float,
    *,
    safe_float: Callable[[Any, float], float],
) -> float:
    size_value = max(0.35, safe_float(row.get("size", 1.0), 1.0))
    return max(
        0.004, min(0.035, (size_value * 0.0044) + (math.sqrt(mass_value) * 0.0014))
    )


def apply_stream_collision_behavior_variation(
    particle_rows: list[dict[str, Any]],
    *,
    now_seconds: float | None = None,
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    clamp01: Callable[[float], float],
    simplex_noise_2d: Callable[..., float],
    stream_noise_amplitude: float,
    stream_collision_static: float,
    stream_ant_influence: float,
) -> None:
    if not isinstance(particle_rows, list) or not particle_rows:
        return
    now_value = safe_float(now_seconds, time.time())
    amplitude_ratio = max(0.0, stream_noise_amplitude / 10.0)
    for index, row in enumerate(particle_rows):
        if not isinstance(row, dict):
            continue
        collisions = max(0, safe_int(row.get("collision_count", 0), 0))
        if collisions <= 0:
            continue

        is_nexus = bool(row.get("is_nexus", False))
        collision_signal = clamp01(
            safe_float(collisions, 0.0) / max(1.0, stream_collision_static)
        )
        if collision_signal <= 1e-8:
            continue

        vx_value = safe_float(row.get("vx", 0.0), 0.0)
        vy_value = safe_float(row.get("vy", 0.0), 0.0)
        x_value = clamp01(safe_float(row.get("x", 0.5), 0.5))
        y_value = clamp01(safe_float(row.get("y", 0.5), 0.5))
        particle_id = str(row.get("id", "") or f"particle:{index}")
        seed = int(hashlib.sha1(particle_id.encode("utf-8")).hexdigest()[:8], 16)

        coupling_damp = 1.0 - (
            collision_signal
            * (0.13 if not is_nexus else 0.08)
            * max(0.2, stream_ant_influence)
        )
        coupling_damp = max(0.68 if not is_nexus else 0.78, min(1.0, coupling_damp))
        vx_value *= coupling_damp
        vy_value *= coupling_damp

        phase = now_value * (0.67 + (collision_signal * 0.29))
        noise_x = simplex_noise_2d(
            (x_value * 6.4) + phase + (index * 0.021),
            (y_value * 6.1) + (phase * 0.73),
            seed=(seed % 251) + 17,
        )
        noise_y = simplex_noise_2d(
            (x_value * 6.0) + 109.0 + (phase * 0.61),
            (y_value * 6.3) + phase + (index * 0.019),
            seed=(seed % 251) + 73,
        )
        kick_gain = (
            (0.00056 + (collision_signal * 0.00242))
            * max(0.2, amplitude_ratio)
            * (1.0 if not is_nexus else 0.46)
        )
        vx_value += noise_x * kick_gain
        vy_value += noise_y * kick_gain

        speed = math.hypot(vx_value, vy_value)
        min_escape_speed = (
            (0.00062 + (collision_signal * 0.00155))
            * max(0.2, amplitude_ratio)
            * (1.0 if not is_nexus else 0.45)
        )
        if speed < min_escape_speed:
            if speed > 1e-8:
                ux = vx_value / speed
                uy = vy_value / speed
            else:
                ux = noise_x
                uy = noise_y
                unorm = math.hypot(ux, uy)
                if unorm <= 1e-8:
                    ux, uy = 1.0, 0.0
                else:
                    ux /= unorm
                    uy /= unorm
            vx_value += ux * (min_escape_speed - speed)
            vy_value += uy * (min_escape_speed - speed)

        row["vx"] = round(vx_value, 6)
        row["vy"] = round(vy_value, 6)
        row["collision_escape_signal"] = round(collision_signal, 6)


def semantic_collision_buffer_pool() -> dict[str, list[Any]]:
    state = getattr(_SEMANTIC_COLLISION_BUFFER_LOCAL, "state", None)
    if isinstance(state, dict):
        return state
    state = {
        "x": [],
        "y": [],
        "vx": [],
        "vy": [],
        "radius": [],
        "mass": [],
        "collisions": [],
    }
    _SEMANTIC_COLLISION_BUFFER_LOCAL.state = state
    return state


def resolve_semantic_particle_collisions_native(
    particle_rows: list[dict[str, Any]],
    *,
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[float], float],
    stream_particle_effective_mass_fn: Callable[[dict[str, Any]], float],
    stream_particle_collision_radius_fn: Callable[[dict[str, Any], float], float],
    apply_stream_collision_behavior_variation_fn: Callable[
        [list[dict[str, Any]]], None
    ],
) -> bool:
    if len(particle_rows) < 2:
        return True

    try:
        from . import c_double_buffer_backend
    except Exception:
        return False

    resolver_inplace = getattr(
        c_double_buffer_backend,
        "resolve_semantic_collisions_native_inplace",
        None,
    )
    resolver = getattr(
        c_double_buffer_backend, "resolve_semantic_collisions_native", None
    )
    if not callable(resolver_inplace) and not callable(resolver):
        return False

    pool = semantic_collision_buffer_pool()
    x_values = pool.get("x", [])
    y_values = pool.get("y", [])
    vx_values = pool.get("vx", [])
    vy_values = pool.get("vy", [])
    radius_values = pool.get("radius", [])
    mass_values = pool.get("mass", [])
    collisions = pool.get("collisions", [])
    if not (
        isinstance(x_values, list)
        and isinstance(y_values, list)
        and isinstance(vx_values, list)
        and isinstance(vy_values, list)
        and isinstance(radius_values, list)
        and isinstance(mass_values, list)
        and isinstance(collisions, list)
    ):
        return False

    x_values.clear()
    y_values.clear()
    vx_values.clear()
    vy_values.clear()
    radius_values.clear()
    mass_values.clear()
    collisions.clear()

    for row in particle_rows:
        x_value = clamp01(safe_float(row.get("x", 0.5), 0.5))
        y_value = clamp01(safe_float(row.get("y", 0.5), 0.5))
        vx_value = safe_float(row.get("vx", 0.0), 0.0)
        vy_value = safe_float(row.get("vy", 0.0), 0.0)
        mass_value = stream_particle_effective_mass_fn(row)
        radius_value = stream_particle_collision_radius_fn(row, mass_value)
        x_values.append(x_value)
        y_values.append(y_value)
        vx_values.append(vx_value)
        vy_values.append(vy_value)
        mass_values.append(mass_value)
        radius_values.append(radius_value)

    if callable(resolver_inplace):
        resolved_inplace = bool(
            resolver_inplace(
                x=x_values,
                y=y_values,
                vx=vx_values,
                vy=vy_values,
                radius=radius_values,
                mass=mass_values,
                collisions_out=collisions,
                restitution=0.91,
                separation_percent=0.84,
                cell_size=0.04,
            )
        )
        if not resolved_inplace or len(collisions) != len(particle_rows):
            return False

        for idx, row in enumerate(particle_rows):
            row["x"] = round(clamp01(safe_float(x_values[idx], x_values[idx])), 5)
            row["y"] = round(clamp01(safe_float(y_values[idx], y_values[idx])), 5)
            row["vx"] = round(safe_float(vx_values[idx], vx_values[idx]), 6)
            row["vy"] = round(safe_float(vy_values[idx], vy_values[idx]), 6)
            row["collision_count"] = max(0, int(safe_float(collisions[idx], 0.0)))
        apply_stream_collision_behavior_variation_fn(particle_rows)
        return True

    if not callable(resolver):
        return False

    resolved = resolver(
        x=x_values,
        y=y_values,
        vx=vx_values,
        vy=vy_values,
        radius=radius_values,
        mass=mass_values,
        restitution=0.91,
        separation_percent=0.84,
        cell_size=0.04,
    )
    if not (isinstance(resolved, tuple) and len(resolved) == 5):
        return False

    x_next, y_next, vx_next, vy_next, collisions = resolved
    count = len(particle_rows)
    if not (
        isinstance(x_next, list)
        and isinstance(y_next, list)
        and isinstance(vx_next, list)
        and isinstance(vy_next, list)
        and isinstance(collisions, list)
        and len(x_next) == count
        and len(y_next) == count
        and len(vx_next) == count
        and len(vy_next) == count
        and len(collisions) == count
    ):
        return False

    for idx, row in enumerate(particle_rows):
        row["x"] = round(clamp01(safe_float(x_next[idx], x_values[idx])), 5)
        row["y"] = round(clamp01(safe_float(y_next[idx], y_values[idx])), 5)
        row["vx"] = round(safe_float(vx_next[idx], vx_values[idx]), 6)
        row["vy"] = round(safe_float(vy_next[idx], vy_values[idx]), 6)
        row["collision_count"] = max(0, int(safe_float(collisions[idx], 0.0)))
    apply_stream_collision_behavior_variation_fn(particle_rows)
    return True


def resolve_semantic_particle_collisions(
    rows: list[dict[str, Any]],
    *,
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[float], float],
    stream_particle_effective_mass_fn: Callable[[dict[str, Any]], float],
    stream_particle_collision_radius_fn: Callable[[dict[str, Any], float], float],
    apply_stream_collision_behavior_variation_fn: Callable[
        [list[dict[str, Any]]], None
    ],
    resolve_semantic_particle_collisions_native_fn: Callable[
        [list[dict[str, Any]]], bool
    ],
) -> None:
    if not isinstance(rows, list):
        return
    particle_rows = [row for row in rows if isinstance(row, dict)]
    if len(particle_rows) < 2:
        return

    if resolve_semantic_particle_collisions_native_fn(particle_rows):
        return

    mass_by_id: dict[str, float] = {}
    radius_by_id: dict[str, float] = {}
    for row in particle_rows:
        particle_id = str(row.get("id", "") or id(row))
        mass_value = stream_particle_effective_mass_fn(row)
        mass_by_id[particle_id] = mass_value
        radius_by_id[particle_id] = stream_particle_collision_radius_fn(row, mass_value)

    cell_size = 0.04
    grid: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in particle_rows:
        x_value = clamp01(safe_float(row.get("x", 0.5), 0.5))
        y_value = clamp01(safe_float(row.get("y", 0.5), 0.5))
        gx = int(x_value / cell_size)
        gy = int(y_value / cell_size)
        grid[(gx, gy)].append(row)

    restitution = 0.91
    separation_percent = 0.84
    collision_count_updates: dict[str, int] = defaultdict(int)

    visited_pairs: set[tuple[str, str]] = set()
    for (gx, gy), bucket in grid.items():
        neighbors: list[dict[str, Any]] = []
        for nx in (gx - 1, gx, gx + 1):
            for ny in (gy - 1, gy, gy + 1):
                neighbors.extend(grid.get((nx, ny), []))

        for row_a in bucket:
            id_a = str(row_a.get("id", "") or id(row_a))
            x_a = clamp01(safe_float(row_a.get("x", 0.5), 0.5))
            y_a = clamp01(safe_float(row_a.get("y", 0.5), 0.5))
            vx_a = safe_float(row_a.get("vx", 0.0), 0.0)
            vy_a = safe_float(row_a.get("vy", 0.0), 0.0)
            mass_a = max(0.2, safe_float(mass_by_id.get(id_a, 1.0), 1.0))
            inv_mass_a = 1.0 / mass_a
            radius_a = safe_float(radius_by_id.get(id_a, 0.01), 0.01)

            for row_b in neighbors:
                if row_a is row_b:
                    continue
                id_b = str(row_b.get("id", "") or id(row_b))
                pair = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                if pair in visited_pairs:
                    continue
                visited_pairs.add(pair)

                x_b = clamp01(safe_float(row_b.get("x", 0.5), 0.5))
                y_b = clamp01(safe_float(row_b.get("y", 0.5), 0.5))
                dx = x_b - x_a
                dy = y_b - y_a
                distance = math.hypot(dx, dy)

                mass_b = max(0.2, safe_float(mass_by_id.get(id_b, 1.0), 1.0))
                inv_mass_b = 1.0 / mass_b
                radius_b = safe_float(radius_by_id.get(id_b, 0.01), 0.01)
                min_distance = radius_a + radius_b
                if distance >= min_distance:
                    continue

                if distance < 1e-6:
                    seed = int(
                        hashlib.sha1(f"{id_a}|{id_b}".encode("utf-8")).hexdigest()[:8],
                        16,
                    )
                    theta = float(seed % 6283) / 1000.0
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    distance = 1e-6
                else:
                    nx = dx / distance
                    ny = dy / distance

                vx_b = safe_float(row_b.get("vx", 0.0), 0.0)
                vy_b = safe_float(row_b.get("vy", 0.0), 0.0)
                rel_vx = vx_a - vx_b
                rel_vy = vy_a - vy_b
                vel_normal = (rel_vx * nx) + (rel_vy * ny)

                if vel_normal < 0.0:
                    impulse = (-(1.0 + restitution) * vel_normal) / max(
                        1e-6, inv_mass_a + inv_mass_b
                    )
                    impulse_x = impulse * nx
                    impulse_y = impulse * ny
                    vx_a += impulse_x * inv_mass_a
                    vy_a += impulse_y * inv_mass_a
                    vx_b -= impulse_x * inv_mass_b
                    vy_b -= impulse_y * inv_mass_b

                    tangent_x = rel_vx - (vel_normal * nx)
                    tangent_y = rel_vy - (vel_normal * ny)
                    tangent_norm = math.hypot(tangent_x, tangent_y)
                    if tangent_norm > 1e-6:
                        tangent_x /= tangent_norm
                        tangent_y /= tangent_norm
                        tangent_impulse = min(
                            abs(impulse) * 0.1,
                            abs((rel_vx * tangent_x) + (rel_vy * tangent_y)),
                        )
                        vx_a -= tangent_impulse * tangent_x * inv_mass_a
                        vy_a -= tangent_impulse * tangent_y * inv_mass_a
                        vx_b += tangent_impulse * tangent_x * inv_mass_b
                        vy_b += tangent_impulse * tangent_y * inv_mass_b

                penetration = min_distance - distance
                correction = (
                    max(0.0, penetration) / max(1e-6, inv_mass_a + inv_mass_b)
                ) * separation_percent
                correction_x = correction * nx
                correction_y = correction * ny

                x_a -= correction_x * inv_mass_a
                y_a -= correction_y * inv_mass_a
                x_b += correction_x * inv_mass_b
                y_b += correction_y * inv_mass_b

                row_b["x"] = round(clamp01(x_b), 5)
                row_b["y"] = round(clamp01(y_b), 5)
                row_b["vx"] = round(vx_b, 6)
                row_b["vy"] = round(vy_b, 6)
                collision_count_updates[id_b] += 1

                collision_count_updates[id_a] += 1

            row_a["x"] = round(clamp01(x_a), 5)
            row_a["y"] = round(clamp01(y_a), 5)
            row_a["vx"] = round(vx_a, 6)
            row_a["vy"] = round(vy_a, 6)

    for row in particle_rows:
        particle_id = str(row.get("id", "") or id(row))
        collisions = int(collision_count_updates.get(particle_id, 0))
        row["collision_count"] = collisions

    apply_stream_collision_behavior_variation_fn(particle_rows)
