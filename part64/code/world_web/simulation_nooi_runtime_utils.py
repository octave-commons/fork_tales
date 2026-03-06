from __future__ import annotations

import math
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable


def build_reset_nooi_runtime_state(
    *,
    nooi_field_factory: Callable[[], Any],
) -> dict[str, Any]:
    return {
        "nooi_field": nooi_field_factory(),
        "nooi_random_boot_applied": False,
        "motion_history": {},
        "weaver_interaction_cooldown_until": {},
        "weaver_interaction_health": {
            "checked_monotonic": 0.0,
            "healthy": False,
        },
        "daimoi_crawl_search_state": {},
    }


def record_daimoi_motion_trail(
    row: dict[str, Any],
    *,
    tick: int,
    motion_history: dict[str, list[dict[str, Any]]],
    motion_history_lock: Any,
    trail_steps: int,
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    clamp01: Callable[[Any], float],
) -> tuple[str, list[dict[str, Any]]]:
    daimoi_id = str(row.get("id", "") or "").strip()
    if not daimoi_id:
        return "", []
    sample = {
        "x": clamp01(safe_float(row.get("x", 0.5), 0.5)),
        "y": clamp01(safe_float(row.get("y", 0.5), 0.5)),
        "vx": safe_float(row.get("vx", 0.0), 0.0),
        "vy": safe_float(row.get("vy", 0.0), 0.0),
        "tick": max(0, safe_int(tick, 0)),
    }
    with motion_history_lock:
        history = list(motion_history.get(daimoi_id, []))
        history.append(sample)
        if len(history) > trail_steps:
            history = history[-trail_steps:]
        motion_history[daimoi_id] = history
        return daimoi_id, [dict(step) for step in history]


def prune_daimoi_motion_history(
    active_ids: set[str],
    *,
    motion_history: dict[str, list[dict[str, Any]]],
    motion_history_lock: Any,
) -> None:
    with motion_history_lock:
        stale_ids = [
            daimoi_id
            for daimoi_id in list(motion_history.keys())
            if daimoi_id not in active_ids
        ]
        for daimoi_id in stale_ids:
            motion_history.pop(daimoi_id, None)


def maybe_seed_random_nooi_field_vectors(
    *,
    force: bool,
    nooi_random_boot_applied: bool,
    nooi_random_boot_lock: Any,
    nooi_field: Any,
    random_field_vectors_on_boot: float,
    random_field_vector_count: int,
    random_field_vector_magnitude: float,
    random_field_vector_seed: int,
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    clamp01: Callable[[Any], float],
) -> bool:
    if safe_float(random_field_vectors_on_boot, 0.0) < 0.5:
        return nooi_random_boot_applied
    with nooi_random_boot_lock:
        if nooi_random_boot_applied and not force:
            return nooi_random_boot_applied
        count = max(0, safe_int(random_field_vector_count, 0))
        magnitude = max(0.0, safe_float(random_field_vector_magnitude, 0.0))
        if count <= 0 or magnitude <= 0.0:
            return True
        seed = safe_int(random_field_vector_seed, 0)
        if seed <= 0:
            seed = int(time.time_ns() & 0xFFFFFFFF)
        rng = random.Random(seed)
        for _ in range(count):
            x_value = clamp01(rng.random())
            y_value = clamp01(rng.random())
            theta = rng.random() * math.tau
            speed = magnitude * (0.35 + (rng.random() * 0.65))
            nooi_field.deposit(
                x_value,
                y_value,
                math.cos(theta) * speed,
                math.sin(theta) * speed,
            )
        return True


def particle_influences_nooi(row: dict[str, Any]) -> bool:
    return not bool(row.get("is_nexus", False))


def nooi_flow_at(
    *,
    nooi_field: Any,
    x_value: float,
    y_value: float,
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[Any], float],
) -> tuple[float, float, float]:
    flow_x, flow_y = nooi_field.sample_vector(
        clamp01(safe_float(x_value, 0.5)),
        clamp01(safe_float(y_value, 0.5)),
    )
    magnitude = math.hypot(flow_x, flow_y)
    if magnitude <= 1e-8:
        return (0.0, 0.0, 0.0)
    return (flow_x / magnitude, flow_y / magnitude, min(1.0, magnitude))


def nooi_outcome_from_particle(
    row: dict[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
) -> dict[str, Any] | None:
    interaction_status = (
        str(row.get("crawler_interaction_status", "") or "").strip().lower()
    )
    if interaction_status == "accepted":
        return {
            "outcome": "food",
            "intensity": min(
                1.0,
                0.45
                + max(0.0, safe_float(row.get("message_probability", 0.0), 0.0) * 0.4),
            ),
            "reason": "crawler_interaction_accepted",
        }
    if interaction_status == "deadline_expired":
        return {
            "outcome": "death",
            "intensity": 0.78,
            "reason": "crawler_deadline_exceeded",
        }
    if interaction_status in {"cooldown_blocked", "rate_limited", "unreachable"}:
        return None

    consumed = max(0.0, safe_float(row.get("resource_consume_amount", 0.0), 0.0))
    blocked = bool(row.get("resource_action_blocked", False))
    collisions = max(0, safe_int(row.get("collision_count", 0), 0))
    if consumed >= 0.04 and not blocked:
        return {
            "outcome": "food",
            "intensity": min(1.0, 0.25 + consumed),
            "reason": "resource_consumed",
        }
    if blocked or collisions > 0:
        return {
            "outcome": "death",
            "intensity": min(1.0, 0.3 + (collisions * 0.1)),
            "reason": "blocked_or_collision",
        }
    return None


def _trail_weights(step_intensity: float) -> list[float]:
    return [
        step_intensity,
        step_intensity * 0.85,
        step_intensity * 0.72,
        step_intensity * 0.58,
        step_intensity * 0.46,
        step_intensity * 0.35,
        step_intensity * 0.24,
        step_intensity * 0.16,
    ]


def apply_nooi_from_particles(
    particles: list[dict[str, Any]],
    *,
    dt_seconds: float,
    tick: int,
    nooi_field: Any,
    motion_history: dict[str, list[dict[str, Any]]],
    motion_history_lock: Any,
    trail_steps: int,
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    clamp01: Callable[[Any], float],
) -> tuple[dict[str, Any], dict[str, int]]:
    nooi_rows = [
        row
        for row in particles
        if isinstance(row, dict) and particle_influences_nooi(row)
    ]
    summary = {"food": 0, "death": 0, "total": 0}
    nooi_field.decay(dt_seconds)
    now_iso = datetime.now(timezone.utc).isoformat()
    active_ids: set[str] = set()
    for row in nooi_rows:
        x_value = safe_float(row.get("x", 0.5), 0.5)
        y_value = safe_float(row.get("y", 0.5), 0.5)
        vx_value = safe_float(row.get("vx", 0.0), 0.0)
        vy_value = safe_float(row.get("vy", 0.0), 0.0)
        row_tick = max(0, safe_int(row.get("age", tick), tick))
        daimoi_id, motion_trail = record_daimoi_motion_trail(
            row,
            tick=row_tick,
            motion_history=motion_history,
            motion_history_lock=motion_history_lock,
            trail_steps=trail_steps,
            safe_float=safe_float,
            safe_int=safe_int,
            clamp01=clamp01,
        )
        if daimoi_id:
            active_ids.add(daimoi_id)
        nooi_field.deposit(x_value, y_value, vx_value, vy_value)

        outcome = nooi_outcome_from_particle(
            row,
            safe_float=safe_float,
            safe_int=safe_int,
        )
        if not isinstance(outcome, dict):
            continue
        outcome_kind = str(outcome.get("outcome", "")).strip().lower()
        if outcome_kind not in {"food", "death"}:
            continue
        intensity = max(0.05, safe_float(outcome.get("intensity", 0.2), 0.2))
        direction_scale = 1.0 if outcome_kind == "food" else -1.0
        trail_rows = motion_trail or [
            {
                "x": clamp01(x_value),
                "y": clamp01(y_value),
                "vx": vx_value,
                "vy": vy_value,
                "tick": row_tick,
            }
        ]
        step_count = max(1, len(trail_rows))
        for step_index, step in enumerate(trail_rows):
            weight_scale = 0.45 + (((step_index + 1) / float(step_count)) * 0.55)
            step_intensity = max(
                0.05,
                intensity
                * weight_scale
                * min(
                    1.0,
                    max(
                        0.35,
                        math.hypot(
                            safe_float(step.get("vx", 0.0), 0.0),
                            safe_float(step.get("vy", 0.0), 0.0),
                        )
                        * 160.0,
                    ),
                ),
            )
            nooi_field.deposit(
                safe_float(step.get("x", x_value), x_value),
                safe_float(step.get("y", y_value), y_value),
                safe_float(step.get("vx", vx_value), vx_value) * direction_scale,
                safe_float(step.get("vy", vy_value), vy_value) * direction_scale,
                layer_weights=_trail_weights(step_intensity),
            )
        nooi_field.append_outcome_trail(
            outcome=outcome_kind,
            x=x_value,
            y=y_value,
            vx=vx_value,
            vy=vy_value,
            intensity=intensity,
            presence_id=str(row.get("presence_id", row.get("owner", "")) or ""),
            daimoi_id=daimoi_id,
            reason=str(outcome.get("reason", "") or ""),
            graph_node_id=str(row.get("graph_node_id", "") or ""),
            tick=row_tick,
            trail_steps=step_count,
            ts=now_iso,
        )
        summary[outcome_kind] = summary.get(outcome_kind, 0) + 1
        summary["total"] = summary.get("total", 0) + 1

    prune_daimoi_motion_history(
        active_ids,
        motion_history=motion_history,
        motion_history_lock=motion_history_lock,
    )
    return nooi_field.get_grid_snapshot(nooi_rows), summary
