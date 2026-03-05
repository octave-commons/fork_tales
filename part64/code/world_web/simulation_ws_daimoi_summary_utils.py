"""Websocket daimoi summary helper utilities."""

from __future__ import annotations

import math
from typing import Any, Callable


def simulation_ws_graph_node_position_map(
    node_positions: Any,
    *,
    node_limit: int,
    clamp01: Callable[[float], float],
    safe_float: Callable[[Any, float], float],
) -> dict[str, tuple[float, float]]:
    if not isinstance(node_positions, dict):
        return {}

    mapped: dict[str, tuple[float, float]] = {}
    limit = max(1, int(node_limit))
    for node_id, row in node_positions.items():
        if len(mapped) >= limit:
            break
        node_key = str(node_id or "").strip()
        if not node_key or not isinstance(row, dict):
            continue
        mapped[node_key] = (
            clamp01(safe_float(row.get("x", 0.5), 0.5)),
            clamp01(safe_float(row.get("y", 0.5), 0.5)),
        )
    return mapped


def simulation_ws_graph_variability_update(
    node_positions: Any,
    *,
    graph_variability_lock: Any,
    graph_variability_state: dict[str, Any],
    node_limit: int,
    distance_ref: float,
    ema_alpha: float,
    clamp01: Callable[[float], float],
    safe_float: Callable[[Any, float], float],
) -> dict[str, Any]:
    current_positions = simulation_ws_graph_node_position_map(
        node_positions,
        node_limit=node_limit,
        clamp01=clamp01,
        safe_float=safe_float,
    )

    with graph_variability_lock:
        previous_state = dict(graph_variability_state)
        previous_positions_raw = previous_state.get("positions", {})
        previous_positions = (
            dict(previous_positions_raw)
            if isinstance(previous_positions_raw, dict)
            else {}
        )

        displacements: list[float] = []
        moved_count = 0
        moved_threshold = max(0.0005, distance_ref * 0.35)
        for node_id, (cx, cy) in current_positions.items():
            prior = previous_positions.get(node_id)
            if not isinstance(prior, tuple) or len(prior) < 2:
                continue
            px = clamp01(safe_float(prior[0], cx))
            py = clamp01(safe_float(prior[1], cy))
            displacement = math.hypot(cx - px, cy - py)
            displacements.append(displacement)
            if displacement >= moved_threshold:
                moved_count += 1

        shared_nodes = len(displacements)
        mean_displacement = (
            (sum(displacements) / float(shared_nodes)) if shared_nodes > 0 else 0.0
        )
        if len(displacements) >= 2:
            ordered = sorted(displacements)
            p90_index = max(
                0,
                min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.9))),
            )
            p90_displacement = ordered[p90_index]
        else:
            p90_displacement = mean_displacement
        active_share = (
            (float(moved_count) / float(shared_nodes)) if shared_nodes > 0 else 0.0
        )

        mean_term = clamp01(mean_displacement / max(1e-6, distance_ref))
        p90_term = clamp01(p90_displacement / max(1e-6, distance_ref * 1.8))
        active_term = clamp01(active_share / 0.45)
        raw_score = clamp01(
            (mean_term * 0.55) + (p90_term * 0.25) + (active_term * 0.2)
        )

        alpha = clamp01(safe_float(ema_alpha, 0.2))
        previous_score = clamp01(
            safe_float(previous_state.get("score", raw_score), raw_score)
        )
        score = clamp01((previous_score * (1.0 - alpha)) + (raw_score * alpha))
        peak_score = max(
            score,
            clamp01(safe_float(previous_state.get("peak_score", score), score) * 0.92),
        )

        graph_variability_state.clear()
        graph_variability_state.update(
            {
                "positions": dict(current_positions),
                "score": score,
                "raw_score": raw_score,
                "peak_score": peak_score,
                "mean_displacement": mean_displacement,
                "p90_displacement": p90_displacement,
                "active_share": active_share,
                "shared_nodes": shared_nodes,
                "sampled_nodes": len(current_positions),
            }
        )

        return {
            "score": score,
            "raw_score": raw_score,
            "peak_score": peak_score,
            "mean_displacement": mean_displacement,
            "p90_displacement": p90_displacement,
            "active_share": active_share,
            "shared_nodes": shared_nodes,
            "sampled_nodes": len(current_positions),
        }


def simulation_ws_daimoi_live_metrics(
    rows: Any,
    *,
    default_target: float,
    daimoi_probabilistic_module: Any,
    clamp01: Callable[[float], float],
    safe_float: Callable[[Any, float], float],
) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        return {}

    positions_fn = getattr(
        daimoi_probabilistic_module,
        "_anti_clump_positions_from_particles",
        None,
    )
    metrics_fn = getattr(daimoi_probabilistic_module, "_anti_clump_metrics", None)
    if not callable(positions_fn) or not callable(metrics_fn):
        return {}

    try:
        positions = positions_fn(rows)
        try:
            metrics = metrics_fn(
                positions,
                previous_collision_count=0,
                particles=rows,
            )
        except TypeError:
            metrics = metrics_fn(positions, previous_collision_count=0)
    except Exception:
        return {}

    if not isinstance(metrics, dict):
        return {}

    clump_score = clamp01(safe_float(metrics.get("clump_score", 0.0), 0.0))
    target = clamp01(safe_float(default_target, 0.38))
    snr_valid = safe_float(metrics.get("snr_valid", 0.0), 0.0) > 0.5
    snr_low_gap = max(0.0, safe_float(metrics.get("snr_low_gap", 0.0), 0.0))
    snr_high_gap = max(0.0, safe_float(metrics.get("snr_high_gap", 0.0), 0.0))
    if snr_valid:
        drive_estimate = max(-1.0, min(1.0, snr_low_gap + (snr_high_gap * 0.35)))
    else:
        drive_estimate = max(-1.0, min(1.0, (clump_score - target) * 2.2))

    return {
        "clump_score": clump_score,
        "drive_estimate": drive_estimate,
        "metrics": {
            "nn_term": clamp01(safe_float(metrics.get("nn_term", 0.0), 0.0)),
            "entropy_norm": clamp01(safe_float(metrics.get("entropy_norm", 1.0), 1.0)),
            "hotspot_term": clamp01(safe_float(metrics.get("hotspot_term", 0.0), 0.0)),
            "collision_term": clamp01(
                safe_float(metrics.get("collision_term", 0.0), 0.0)
            ),
            "collision_rate": max(
                0.0,
                safe_float(metrics.get("collision_rate", 0.0), 0.0),
            ),
            "median_distance": max(
                0.0,
                safe_float(metrics.get("median_distance", 0.0), 0.0),
            ),
            "target_distance": max(
                0.0,
                safe_float(metrics.get("target_distance", 0.0), 0.0),
            ),
            "top_share": clamp01(safe_float(metrics.get("top_share", 0.0), 0.0)),
            "mean_spacing": max(
                0.0,
                safe_float(metrics.get("mean_spacing", 0.0), 0.0),
            ),
            "fano_factor": max(
                0.0,
                safe_float(metrics.get("fano_factor", 0.0), 0.0),
            ),
            "fano_excess": max(
                0.0,
                safe_float(metrics.get("fano_excess", 0.0), 0.0),
            ),
            "spatial_noise": max(
                0.0,
                safe_float(metrics.get("spatial_noise", 0.0), 0.0),
            ),
            "motion_signal": max(
                0.0,
                safe_float(metrics.get("motion_signal", 0.0), 0.0),
            ),
            "motion_noise": max(
                0.0,
                safe_float(metrics.get("motion_noise", 0.0), 0.0),
            ),
            "motion_samples": max(
                0,
                int(safe_float(metrics.get("motion_samples", 0), 0.0)),
            ),
            "semantic_noise": max(
                0.0,
                safe_float(metrics.get("semantic_noise", 0.0), 0.0),
            ),
            "snr_signal": max(
                0.0,
                safe_float(metrics.get("snr_signal", 0.0), 0.0),
            ),
            "snr_noise": max(
                0.0,
                safe_float(metrics.get("snr_noise", 0.0), 0.0),
            ),
            "snr": max(0.0, safe_float(metrics.get("snr", 0.0), 0.0)),
            "snr_valid": bool(snr_valid),
            "snr_low_gap": snr_low_gap,
            "snr_high_gap": snr_high_gap,
            "snr_min": max(
                0.05,
                safe_float(metrics.get("snr_min", 0.85), 0.85),
            ),
            "snr_max": max(
                0.1,
                safe_float(metrics.get("snr_max", 1.65), 1.65),
            ),
            "snr_in_band": bool(safe_float(metrics.get("snr_in_band", 0.0), 0.0) > 0.5),
        },
    }


def simulation_ws_ensure_daimoi_summary(
    payload: dict[str, Any],
    *,
    include_live_metrics: bool = True,
    include_graph_variability: bool = True,
    daimoi_probabilistic_module: Any,
    graph_variability_noise_gain: float,
    graph_variability_route_damp: float,
    live_metrics_builder: Callable[..., dict[str, Any]],
    graph_variability_builder: Callable[[Any], dict[str, Any]],
    clamp01: Callable[[float], float],
    safe_float: Callable[[Any, float], float],
) -> None:
    if not isinstance(payload, dict):
        return
    dynamics = payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return

    summary_raw = dynamics.get("daimoi_probabilistic", {})
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}

    anti_raw = summary.get("anti_clump", {})
    anti = dict(anti_raw) if isinstance(anti_raw, dict) else {}

    metrics_raw = anti.get("metrics", {})
    metrics = dict(metrics_raw) if isinstance(metrics_raw, dict) else {}
    scales_raw = anti.get("scales", {})
    scales = dict(scales_raw) if isinstance(scales_raw, dict) else {}

    default_target = safe_float(
        getattr(daimoi_probabilistic_module, "DAIMOI_ANTI_CLUMP_TARGET", 0.33),
        0.33,
    )
    anti_target = clamp01(
        safe_float(anti.get("target", default_target), default_target)
    )

    rows = dynamics.get("field_particles", [])
    live_metrics: dict[str, Any] = {}
    if include_live_metrics:
        live_metrics = live_metrics_builder(
            rows,
            default_target=anti_target,
        )

    clump_score = clamp01(safe_float(summary.get("clump_score", 0.0), 0.0))
    if isinstance(live_metrics, dict) and live_metrics:
        clump_score = clamp01(
            safe_float(live_metrics.get("clump_score", clump_score), clump_score)
        )

    drive_default = max(-1.0, min(1.0, (clump_score - anti_target) * 2.2))
    anti_drive = drive_default
    if isinstance(live_metrics, dict) and live_metrics:
        anti_drive = max(
            -1.0,
            min(
                1.0,
                safe_float(
                    live_metrics.get("drive_estimate", drive_default), drive_default
                ),
            ),
        )
    else:
        anti_drive = max(
            -1.0,
            min(
                1.0,
                safe_float(
                    summary.get("anti_clump_drive", anti.get("drive", drive_default)),
                    drive_default,
                ),
            ),
        )

    graph_variability_raw = anti.get("graph_variability", {})
    graph_variability: dict[str, Any] = (
        dict(graph_variability_raw) if isinstance(graph_variability_raw, dict) else {}
    )
    if include_graph_variability:
        graph_variability = graph_variability_builder(
            dynamics.get("graph_node_positions", {})
        )
    graph_score = clamp01(
        safe_float(
            graph_variability.get("score", 0.0)
            if isinstance(graph_variability, dict)
            else 0.0,
            0.0,
        )
    )
    noise_gain = max(
        1.0,
        min(
            2.2,
            1.0 + (graph_score * graph_variability_noise_gain),
        ),
    )
    route_damp = max(
        0.55,
        min(
            1.0,
            1.0 - (graph_score * graph_variability_route_damp),
        ),
    )

    anti["target"] = round(anti_target, 6)
    anti["drive"] = round(anti_drive, 6)
    anti["clump_score"] = round(clump_score, 6)
    live_metrics_map = (
        dict(live_metrics.get("metrics", {}))
        if isinstance(live_metrics, dict)
        and isinstance(live_metrics.get("metrics", {}), dict)
        else {}
    )
    anti["metrics"] = {
        "nn_term": max(
            0.0,
            safe_float(
                live_metrics_map.get("nn_term", metrics.get("nn_term", 0.0)),
                0.0,
            ),
        ),
        "entropy_norm": max(
            0.0,
            safe_float(
                live_metrics_map.get(
                    "entropy_norm",
                    metrics.get("entropy_norm", 1.0),
                ),
                1.0,
            ),
        ),
        "hotspot_term": max(
            0.0,
            safe_float(
                live_metrics_map.get("hotspot_term", metrics.get("hotspot_term", 0.0)),
                0.0,
            ),
        ),
        "collision_term": max(
            0.0,
            safe_float(
                live_metrics_map.get(
                    "collision_term",
                    metrics.get("collision_term", 0.0),
                ),
                0.0,
            ),
        ),
        "collision_rate": max(
            0.0,
            safe_float(
                live_metrics_map.get(
                    "collision_rate",
                    metrics.get("collision_rate", 0.0),
                ),
                0.0,
            ),
        ),
        "median_distance": max(
            0.0,
            safe_float(
                live_metrics_map.get(
                    "median_distance",
                    metrics.get("median_distance", 0.0),
                ),
                0.0,
            ),
        ),
        "target_distance": max(
            0.0,
            safe_float(
                live_metrics_map.get(
                    "target_distance",
                    metrics.get("target_distance", 0.0),
                ),
                0.0,
            ),
        ),
        "top_share": clamp01(
            safe_float(
                live_metrics_map.get("top_share", metrics.get("top_share", 0.0)),
                0.0,
            )
        ),
        "mean_spacing": max(
            0.0,
            safe_float(
                live_metrics_map.get("mean_spacing", metrics.get("mean_spacing", 0.0)),
                0.0,
            ),
        ),
        "fano_factor": max(
            0.0,
            safe_float(
                live_metrics_map.get("fano_factor", metrics.get("fano_factor", 0.0)),
                0.0,
            ),
        ),
        "fano_excess": max(
            0.0,
            safe_float(
                live_metrics_map.get("fano_excess", metrics.get("fano_excess", 0.0)),
                0.0,
            ),
        ),
        "spatial_noise": max(
            0.0,
            safe_float(
                live_metrics_map.get(
                    "spatial_noise",
                    metrics.get("spatial_noise", 0.0),
                ),
                0.0,
            ),
        ),
        "motion_signal": max(
            0.0,
            safe_float(
                live_metrics_map.get(
                    "motion_signal",
                    metrics.get("motion_signal", 0.0),
                ),
                0.0,
            ),
        ),
        "motion_noise": max(
            0.0,
            safe_float(
                live_metrics_map.get("motion_noise", metrics.get("motion_noise", 0.0)),
                0.0,
            ),
        ),
        "motion_samples": max(
            0,
            int(
                safe_float(
                    live_metrics_map.get(
                        "motion_samples",
                        metrics.get("motion_samples", 0),
                    ),
                    0.0,
                )
            ),
        ),
        "semantic_noise": max(
            0.0,
            safe_float(
                live_metrics_map.get(
                    "semantic_noise",
                    metrics.get("semantic_noise", 0.0),
                ),
                0.0,
            ),
        ),
        "snr_signal": max(
            0.0,
            safe_float(
                live_metrics_map.get("snr_signal", metrics.get("snr_signal", 0.0)),
                0.0,
            ),
        ),
        "snr_noise": max(
            0.0,
            safe_float(
                live_metrics_map.get("snr_noise", metrics.get("snr_noise", 0.0)),
                0.0,
            ),
        ),
        "snr": max(
            0.0,
            safe_float(live_metrics_map.get("snr", metrics.get("snr", 0.0)), 0.0),
        ),
        "snr_valid": bool(
            safe_float(
                live_metrics_map.get("snr_valid", metrics.get("snr_valid", 0.0)),
                0.0,
            )
            > 0.5
        ),
        "snr_low_gap": max(
            0.0,
            safe_float(
                live_metrics_map.get("snr_low_gap", metrics.get("snr_low_gap", 0.0)),
                0.0,
            ),
        ),
        "snr_high_gap": max(
            0.0,
            safe_float(
                live_metrics_map.get("snr_high_gap", metrics.get("snr_high_gap", 0.0)),
                0.0,
            ),
        ),
        "snr_min": max(
            0.05,
            safe_float(
                live_metrics_map.get("snr_min", metrics.get("snr_min", 0.85)),
                0.85,
            ),
        ),
        "snr_max": max(
            0.1,
            safe_float(
                live_metrics_map.get("snr_max", metrics.get("snr_max", 1.65)),
                1.65,
            ),
        ),
        "snr_in_band": bool(
            safe_float(
                live_metrics_map.get("snr_in_band", metrics.get("snr_in_band", 0.0)),
                0.0,
            )
            > 0.5
        ),
    }
    anti["snr"] = round(
        max(0.0, safe_float(anti["metrics"].get("snr", 0.0), 0.0)),
        6,
    )
    anti["snr_valid"] = bool(anti["metrics"].get("snr_valid", False))
    snr_band_min = max(0.05, safe_float(anti["metrics"].get("snr_min", 0.85), 0.85))
    snr_band_max = max(
        snr_band_min + 0.05,
        safe_float(anti["metrics"].get("snr_max", 1.65), 1.65),
    )
    anti["snr_band"] = {
        "min": round(snr_band_min, 6),
        "max": round(snr_band_max, 6),
        "low_gap": round(
            max(0.0, safe_float(anti["metrics"].get("snr_low_gap", 0.0), 0.0)),
            6,
        ),
        "high_gap": round(
            max(0.0, safe_float(anti["metrics"].get("snr_high_gap", 0.0), 0.0)),
            6,
        ),
        "in_band": bool(anti["metrics"].get("snr_in_band", False)),
    }
    anti["scales"] = {
        "spawn": max(0.0, safe_float(scales.get("spawn", 1.0), 1.0)),
        "anchor": max(0.0, safe_float(scales.get("anchor", 1.0), 1.0)),
        "semantic": max(0.0, safe_float(scales.get("semantic", 1.0), 1.0)),
        "edge": max(0.0, safe_float(scales.get("edge", 1.0), 1.0)),
        "tangent": max(0.0, safe_float(scales.get("tangent", 1.0), 1.0)),
        "friction_slip": max(
            0.0,
            safe_float(scales.get("friction_slip", 1.0), 1.0),
        ),
        "simplex_gain": max(
            0.0,
            safe_float(scales.get("simplex_gain", 1.0), 1.0),
        ),
        "simplex_scale": max(
            0.0,
            safe_float(scales.get("simplex_scale", 1.0), 1.0),
        ),
        "noise_gain": round(noise_gain, 6),
        "route_damp": round(route_damp, 6),
    }
    if isinstance(graph_variability, dict):
        anti["graph_variability"] = {
            "score": round(
                clamp01(safe_float(graph_variability.get("score", 0.0), 0.0)), 6
            ),
            "raw_score": round(
                clamp01(safe_float(graph_variability.get("raw_score", 0.0), 0.0)),
                6,
            ),
            "peak_score": round(
                clamp01(safe_float(graph_variability.get("peak_score", 0.0), 0.0)),
                6,
            ),
            "mean_displacement": round(
                max(
                    0.0,
                    safe_float(graph_variability.get("mean_displacement", 0.0), 0.0),
                ),
                6,
            ),
            "p90_displacement": round(
                max(
                    0.0,
                    safe_float(graph_variability.get("p90_displacement", 0.0), 0.0),
                ),
                6,
            ),
            "active_share": round(
                clamp01(safe_float(graph_variability.get("active_share", 0.0), 0.0)),
                6,
            ),
            "shared_nodes": int(
                max(0, safe_float(graph_variability.get("shared_nodes", 0), 0.0))
            ),
            "sampled_nodes": int(
                max(0, safe_float(graph_variability.get("sampled_nodes", 0), 0.0))
            ),
        }

    summary["clump_score"] = round(clump_score, 6)
    summary["anti_clump_drive"] = round(anti_drive, 6)
    summary["snr"] = round(max(0.0, safe_float(anti.get("snr", 0.0), 0.0)), 6)
    summary["anti_clump"] = anti
    dynamics["daimoi_probabilistic"] = summary
    payload["presence_dynamics"] = dynamics


def simulation_ws_payload_missing_daimoi_summary(
    payload: dict[str, Any],
    *,
    ensure_summary: Callable[[dict[str, Any]], None],
) -> bool:
    if not isinstance(payload, dict):
        return True
    if "presence_dynamics" not in payload:
        return True
    dynamics = payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict) or not dynamics:
        return True
    ensure_summary(payload)
    dynamics = payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return True
    summary = dynamics.get("daimoi_probabilistic", {})
    if not isinstance(summary, dict) or not summary:
        return True
    if "clump_score" not in summary and "anti_clump" not in summary:
        return True
    return False
