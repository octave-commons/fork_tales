"""Stream motion overlay helpers for simulation dynamics."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable


def stream_motion_tick_scale(
    dt_seconds: float,
    *,
    safe_float: Callable[[Any, float], float],
) -> float:
    dt = max(0.001, safe_float(dt_seconds, 0.08))
    return max(0.55, min(3.0, dt / 0.0166667))


def particle_origin_presence_id(row: dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    explicit_origin = str(row.get("origin_presence_id", "") or "").strip()
    if explicit_origin:
        return explicit_origin
    particle_id = str(row.get("id", "") or "").strip()
    if particle_id.startswith("field:"):
        body = particle_id[6:]
        if body:
            return str(body.rsplit(":", 1)[0]).strip()
    owner_presence = str(row.get("owner_presence_id", "") or "").strip()
    if owner_presence:
        return owner_presence
    return str(row.get("presence_id", "") or "").strip()


def update_stream_motion_overlays(
    presence_dynamics: dict[str, Any],
    *,
    dt_seconds: float,
    now_seconds: float | None = None,
    policy: dict[str, Any] | None = None,
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    clamp01: Callable[[float], float],
    simplex_noise_2d: Callable[..., float],
    nooi_flow_at: Callable[[float, float], tuple[float, float, float]],
    dynamics_lock: Any,
    dynamics_cache: dict[str, Any],
    stream_overlay_nooi_gain: float,
    stream_overlay_anchor_nooi_gain: float,
) -> None:
    if not isinstance(presence_dynamics, dict):
        return

    policy_obj = policy if isinstance(policy, dict) else {}
    if policy_obj:
        tick_signals = presence_dynamics.get("tick_signals", {})
        if not isinstance(tick_signals, dict):
            tick_signals = {}

        slack_ms = safe_float(policy_obj.get("slack_ms", float("nan")), float("nan"))
        tick_budget_ms = safe_float(
            policy_obj.get("tick_budget_ms", float("nan")),
            float("nan"),
        )
        ingestion_pressure = max(
            0.0,
            min(1.0, safe_float(policy_obj.get("ingestion_pressure", 0.0), 0.0)),
        )
        ws_particle_max = max(
            0,
            int(safe_float(policy_obj.get("ws_particle_max", 0.0), 0.0)),
        )
        guard_mode = str(policy_obj.get("guard_mode", "") or "").strip()

        if math.isfinite(slack_ms):
            tick_signals["slack_ms"] = slack_ms
        if math.isfinite(tick_budget_ms):
            tick_signals["tick_budget_ms"] = tick_budget_ms
        tick_signals["ingestion_pressure"] = ingestion_pressure
        if ws_particle_max > 0:
            tick_signals["ws_particle_max"] = ws_particle_max
        if guard_mode:
            tick_signals["guard_mode"] = guard_mode

        presence_dynamics["tick_signals"] = tick_signals

    rows = presence_dynamics.get("field_particles", [])
    if not isinstance(rows, list) or not rows:
        presence_dynamics.pop("graph_node_positions", None)
        presence_dynamics.pop("presence_anchor_positions", None)
        return

    now_mono = safe_float(now_seconds, 0.0)
    if now_mono <= 0.0:
        now_mono = time.monotonic()
    frame_scale = stream_motion_tick_scale(dt_seconds, safe_float=safe_float)

    node_acc: dict[str, dict[str, float]] = {}
    presence_acc: dict[str, dict[str, float]] = {}
    max_nodes = 2200

    for row in rows:
        if not isinstance(row, dict):
            continue

        x_value = clamp01(safe_float(row.get("x", 0.5), 0.5))
        y_value = clamp01(safe_float(row.get("y", 0.5), 0.5))
        vx_value = safe_float(row.get("vx", 0.0), 0.0)
        vy_value = safe_float(row.get("vy", 0.0), 0.0)

        presence_id = str(row.get("presence_id", "") or "").strip()
        if presence_id:
            presence_bucket = presence_acc.get(presence_id)
            if not isinstance(presence_bucket, dict):
                presence_bucket = {
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "count": 0.0,
                }
                presence_acc[presence_id] = presence_bucket
            presence_bucket["sum_x"] += x_value
            presence_bucket["sum_y"] += y_value
            presence_bucket["count"] += 1.0

        route_probability = clamp01(safe_float(row.get("route_probability", 0.0), 0.0))
        influence_power = clamp01(safe_float(row.get("influence_power", 0.0), 0.0))
        semantic_signal = clamp01(
            abs(safe_float(row.get("drift_cost_semantic_term", 0.0), 0.0))
            + (safe_float(row.get("message_probability", 0.0), 0.0) * 0.4)
            + (safe_float(row.get("package_entropy", 0.0), 0.0) * 0.15)
        )
        base_weight = max(
            0.08,
            0.2
            + (route_probability * 0.38)
            + (influence_power * 0.28)
            + (semantic_signal * 0.2),
        )

        node_refs = (
            (
                str(row.get("route_node_id", "") or "").strip(),
                safe_float(row.get("route_x", x_value), x_value),
                safe_float(row.get("route_y", y_value), y_value),
                1.0,
            ),
            (
                str(row.get("graph_node_id", "") or "").strip(),
                safe_float(row.get("graph_x", x_value), x_value),
                safe_float(row.get("graph_y", y_value), y_value),
                0.76,
            ),
        )

        for node_id, anchor_x_raw, anchor_y_raw, role_weight in node_refs:
            if not node_id:
                continue
            if node_id not in node_acc and len(node_acc) >= max_nodes:
                continue

            anchor_x = clamp01(anchor_x_raw if math.isfinite(anchor_x_raw) else x_value)
            anchor_y = clamp01(anchor_y_raw if math.isfinite(anchor_y_raw) else y_value)
            weight = max(0.05, base_weight * max(0.05, safe_float(role_weight, 1.0)))

            bucket = node_acc.get(node_id)
            if not isinstance(bucket, dict):
                bucket = {
                    "sum_x": 0.0,
                    "sum_y": 0.0,
                    "weight": 0.0,
                    "flow_x": 0.0,
                    "flow_y": 0.0,
                    "flow_weight": 0.0,
                    "anchor_x": 0.0,
                    "anchor_y": 0.0,
                    "anchor_weight": 0.0,
                    "samples": 0.0,
                }
                node_acc[node_id] = bucket

            bucket["sum_x"] += x_value * weight
            bucket["sum_y"] += y_value * weight
            bucket["weight"] += weight
            bucket["flow_x"] += vx_value * weight
            bucket["flow_y"] += vy_value * weight
            bucket["flow_weight"] += weight
            bucket["anchor_x"] += anchor_x * weight
            bucket["anchor_y"] += anchor_y * weight
            bucket["anchor_weight"] += weight
            bucket["samples"] += 1.0

    graph_positions: dict[str, dict[str, Any]] = {}
    presence_positions: dict[str, dict[str, Any]] = {}

    with dynamics_lock:
        graph_cache = dynamics_cache.get("graph_nodes", {})
        if not isinstance(graph_cache, dict):
            graph_cache = {}

        ranked_nodes = sorted(
            node_acc.items(),
            key=lambda item: (-safe_float(item[1].get("samples", 0.0), 0.0), item[0]),
        )
        for node_id, acc in ranked_nodes:
            weight_total = max(1e-6, safe_float(acc.get("weight", 0.0), 0.0))
            target_x = clamp01(safe_float(acc.get("sum_x", 0.0), 0.0) / weight_total)
            target_y = clamp01(safe_float(acc.get("sum_y", 0.0), 0.0) / weight_total)

            anchor_weight = max(1e-6, safe_float(acc.get("anchor_weight", 0.0), 0.0))
            anchor_x = clamp01(
                safe_float(acc.get("anchor_x", 0.0), 0.0) / anchor_weight
            )
            anchor_y = clamp01(
                safe_float(acc.get("anchor_y", 0.0), 0.0) / anchor_weight
            )

            flow_weight = max(1e-6, safe_float(acc.get("flow_weight", 0.0), 0.0))
            flow_x = safe_float(acc.get("flow_x", 0.0), 0.0) / flow_weight
            flow_y = safe_float(acc.get("flow_y", 0.0), 0.0) / flow_weight

            state = graph_cache.get(node_id, {})
            if not isinstance(state, dict):
                state = {}

            x_value = clamp01(safe_float(state.get("x", anchor_x), anchor_x))
            y_value = clamp01(safe_float(state.get("y", anchor_y), anchor_y))
            vx_value = safe_float(state.get("vx", 0.0), 0.0)
            vy_value = safe_float(state.get("vy", 0.0), 0.0)

            sample_count = max(1.0, safe_float(acc.get("samples", 1.0), 1.0))
            density_signal = clamp01(sample_count / 24.0)
            node_seed = safe_int(state.get("seed", 0), 0)
            if node_seed <= 0:
                node_seed = int(
                    hashlib.sha1(f"stream-node:{node_id}".encode("utf-8")).hexdigest()[
                        :8
                    ],
                    16,
                )

            drift_scale = 0.0012 + (density_signal * 0.002)
            drift_time = now_mono * (0.18 + (density_signal * 0.16))
            drift_x = simplex_noise_2d(
                (anchor_x * 7.6) + (sample_count * 0.033),
                drift_time,
                seed=(node_seed % 251) + 17,
            )
            drift_y = simplex_noise_2d(
                (anchor_y * 7.2) + 41.0 + (sample_count * 0.027),
                drift_time * 1.13,
                seed=(node_seed % 251) + 79,
            )
            target_x = clamp01(target_x + (drift_x * drift_scale))
            target_y = clamp01(target_y + (drift_y * drift_scale))

            spring_gain = 0.028 + (density_signal * 0.018)
            anchor_gain = 0.015 + ((1.0 - density_signal) * 0.01)
            flow_gain = 0.39 + (density_signal * 0.18)

            vx_value += (
                ((target_x - x_value) * spring_gain)
                + ((anchor_x - x_value) * anchor_gain)
                + (flow_x * flow_gain)
            ) * frame_scale
            vy_value += (
                ((target_y - y_value) * spring_gain)
                + ((anchor_y - y_value) * anchor_gain)
                + (flow_y * flow_gain)
            ) * frame_scale

            nooi_dir_x, nooi_dir_y, nooi_signal = nooi_flow_at(x_value, y_value)
            if nooi_signal > 0.0:
                nooi_gain = (
                    safe_float(stream_overlay_nooi_gain, 0.0)
                    * nooi_signal
                    * frame_scale
                )
                vx_value += nooi_dir_x * nooi_gain
                vy_value += nooi_dir_y * nooi_gain

            damping_tick = math.pow(0.81, frame_scale)
            vx_value *= damping_tick
            vy_value *= damping_tick

            speed = math.hypot(vx_value, vy_value)
            max_speed = 0.0036 + (density_signal * 0.0039)
            if speed > max_speed and speed > 0.0:
                speed_scale = max_speed / speed
                vx_value *= speed_scale
                vy_value *= speed_scale

            x_value = clamp01(x_value + (vx_value * frame_scale))
            y_value = clamp01(y_value + (vy_value * frame_scale))

            graph_cache[node_id] = {
                "x": x_value,
                "y": y_value,
                "vx": vx_value,
                "vy": vy_value,
                "samples": int(sample_count),
                "seed": node_seed,
                "ts": now_mono,
            }
            graph_positions[node_id] = {
                "x": round(x_value, 5),
                "y": round(y_value, 5),
                "vx": round(vx_value, 6),
                "vy": round(vy_value, 6),
                "samples": int(sample_count),
            }

        graph_stale_before = now_mono - 120.0
        for node_id in list(graph_cache.keys()):
            state = graph_cache.get(node_id)
            if not isinstance(state, dict):
                graph_cache.pop(node_id, None)
                continue
            ts_value = safe_float(state.get("ts", now_mono), now_mono)
            if node_id not in node_acc and ts_value < graph_stale_before:
                graph_cache.pop(node_id, None)

        dynamics_cache["graph_nodes"] = graph_cache

        anchor_cache = dynamics_cache.get("presence_anchors", {})
        if not isinstance(anchor_cache, dict):
            anchor_cache = {}

        ranked_presences = sorted(
            presence_acc.items(),
            key=lambda item: (-safe_float(item[1].get("count", 0.0), 0.0), item[0]),
        )
        for presence_id, acc in ranked_presences[:240]:
            count_value = max(1.0, safe_float(acc.get("count", 1.0), 1.0))
            target_x = clamp01(safe_float(acc.get("sum_x", 0.0), 0.0) / count_value)
            target_y = clamp01(safe_float(acc.get("sum_y", 0.0), 0.0) / count_value)

            state = anchor_cache.get(presence_id, {})
            if not isinstance(state, dict):
                state = {}

            x_value = clamp01(safe_float(state.get("x", target_x), target_x))
            y_value = clamp01(safe_float(state.get("y", target_y), target_y))
            vx_value = safe_float(state.get("vx", 0.0), 0.0)
            vy_value = safe_float(state.get("vy", 0.0), 0.0)
            presence_seed = safe_int(state.get("seed", 0), 0)
            if presence_seed <= 0:
                presence_seed = int(
                    hashlib.sha1(
                        f"stream-presence:{presence_id}".encode("utf-8")
                    ).hexdigest()[:8],
                    16,
                )

            density_signal = clamp01(count_value / 40.0)
            drift_scale = 0.0011 + (density_signal * 0.0018)
            drift_time = now_mono * (0.14 + (density_signal * 0.11))
            target_x = clamp01(
                target_x
                + (
                    simplex_noise_2d(
                        (target_x * 6.3) + 17.0,
                        drift_time,
                        seed=(presence_seed % 251) + 23,
                    )
                    * drift_scale
                )
            )
            target_y = clamp01(
                target_y
                + (
                    simplex_noise_2d(
                        (target_y * 6.1) + 53.0,
                        drift_time * 1.09,
                        seed=(presence_seed % 251) + 61,
                    )
                    * drift_scale
                )
            )
            pull_gain = 0.18 + (density_signal * 0.12)
            vx_value += (target_x - x_value) * pull_gain * frame_scale
            vy_value += (target_y - y_value) * pull_gain * frame_scale

            nooi_dir_x, nooi_dir_y, nooi_signal = nooi_flow_at(x_value, y_value)
            if nooi_signal > 0.0:
                nooi_gain = (
                    safe_float(stream_overlay_anchor_nooi_gain, 0.0)
                    * nooi_signal
                    * frame_scale
                )
                vx_value += nooi_dir_x * nooi_gain
                vy_value += nooi_dir_y * nooi_gain

            vx_value *= math.pow(0.84, frame_scale)
            vy_value *= math.pow(0.84, frame_scale)

            speed = math.hypot(vx_value, vy_value)
            max_speed = 0.016 + (density_signal * 0.01)
            if speed > max_speed and speed > 0.0:
                speed_scale = max_speed / speed
                vx_value *= speed_scale
                vy_value *= speed_scale

            x_value = clamp01(x_value + (vx_value * frame_scale))
            y_value = clamp01(y_value + (vy_value * frame_scale))

            anchor_cache[presence_id] = {
                "x": x_value,
                "y": y_value,
                "vx": vx_value,
                "vy": vy_value,
                "count": int(count_value),
                "seed": presence_seed,
                "ts": now_mono,
            }
            presence_positions[presence_id] = {
                "x": round(x_value, 5),
                "y": round(y_value, 5),
                "count": int(count_value),
            }

        anchor_stale_before = now_mono - 90.0
        for presence_id in list(anchor_cache.keys()):
            state = anchor_cache.get(presence_id)
            if not isinstance(state, dict):
                anchor_cache.pop(presence_id, None)
                continue
            ts_value = safe_float(state.get("ts", now_mono), now_mono)
            if presence_id not in presence_acc and ts_value < anchor_stale_before:
                anchor_cache.pop(presence_id, None)

        dynamics_cache["presence_anchors"] = anchor_cache

    if graph_positions:
        presence_dynamics["graph_node_positions"] = graph_positions
    else:
        presence_dynamics.pop("graph_node_positions", None)

    if presence_positions:
        presence_dynamics["presence_anchor_positions"] = presence_positions
    else:
        presence_dynamics.pop("presence_anchor_positions", None)
