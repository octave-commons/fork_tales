from __future__ import annotations

import math
from typing import Any

from .metrics import _clamp01, _safe_float, _safe_int


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, float):
        return value if math.isfinite(value) else default
    if isinstance(value, int):
        return float(value)
    parsed = _safe_float(value, default)
    if not math.isfinite(parsed):
        return default
    return parsed


def _clamp_range(value: Any, lower: float, upper: float) -> float:
    lo = _finite_float(lower, 0.0)
    hi = _finite_float(upper, 1.0)
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, _finite_float(value, lo)))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(_finite_float(value, 0.0) for value in values)
    length = len(ordered)
    mid = length // 2
    if length % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) * 0.5


def _cfg_float(config: dict[str, Any] | None, key: str, default: float) -> float:
    if isinstance(config, dict):
        return _safe_float(config.get(key, default), default)
    return default


def _cfg_int(config: dict[str, Any] | None, key: str, default: int) -> int:
    if isinstance(config, dict):
        return _safe_int(config.get(key, default), default)
    return default


def anti_clump_observer_config(
    *,
    sample_limit: Any = 96,
    grid_cap: Any = 96,
    collision_rate_ref: Any = 2.2,
    snr_min: Any = 0.85,
    snr_max: Any = 1.65,
    snr_alpha: Any = 0.4,
    snr_beta: Any = 0.2,
    snr_eps: Any = 1e-6,
) -> dict[str, Any]:
    snr_min_value = max(0.05, _safe_float(snr_min, 0.85))
    snr_max_value = max(snr_min_value + 0.05, _safe_float(snr_max, 1.65))
    return {
        "sample_limit": max(8, _safe_int(sample_limit, 96)),
        "grid_cap": max(4, _safe_int(grid_cap, 96)),
        "collision_rate_ref": max(0.25, _safe_float(collision_rate_ref, 2.2)),
        "snr_min": snr_min_value,
        "snr_max": snr_max_value,
        "snr_alpha": max(0.0, _safe_float(snr_alpha, 0.4)),
        "snr_beta": max(0.0, _safe_float(snr_beta, 0.2)),
        "snr_eps": max(1e-9, _safe_float(snr_eps, 1e-6)),
    }


def anti_clump_controller_config(
    *,
    observer_config: dict[str, Any] | None = None,
    drive_limit: Any = 1.0,
    integral_limit: Any = 1.5,
    target: Any = 0.38,
    kp: Any = 0.22,
    ki: Any = 0.04,
    smoothing: Any = 0.15,
    update_stride: Any = 10,
    min_particles: Any = 24,
    high_snr_perturb_gain: Any = 0.28,
) -> dict[str, Any]:
    config = dict(observer_config) if isinstance(observer_config, dict) else {}
    config.update(
        {
            "drive_limit": max(0.25, _safe_float(drive_limit, 1.0)),
            "integral_limit": max(0.25, _safe_float(integral_limit, 1.5)),
            "target": _clamp01(_safe_float(target, 0.38)),
            "kp": max(0.0, _safe_float(kp, 0.22)),
            "ki": max(0.0, _safe_float(ki, 0.04)),
            "smoothing": _clamp01(_safe_float(smoothing, 0.15)),
            "update_stride": max(1, _safe_int(update_stride, 10)),
            "min_particles": max(8, _safe_int(min_particles, 24)),
            "high_snr_perturb_gain": max(
                0.0,
                _safe_float(high_snr_perturb_gain, 0.28),
            ),
        }
    )
    return config


DEFAULT_ANTI_CLUMP_SCALE_ORDER = (
    "semantic",
    "edge",
    "anchor",
    "spawn",
    "tangent",
    "friction_slip",
    "simplex_gain",
    "simplex_scale",
)


def anti_clump_positions_from_particles(particles: Any) -> list[tuple[float, float]]:
    if isinstance(particles, dict):
        rows = list(particles.values())
    elif isinstance(particles, list):
        rows = particles
    else:
        rows = []

    positions: list[tuple[float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if bool(row.get("is_nexus", False)):
            continue
        x = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        positions.append((x, y))
    return positions


def anti_clump_field_vector(row: dict[str, Any]) -> tuple[float, float]:
    field_x = _safe_float(row.get("field_fx", 0.0), 0.0)
    field_y = _safe_float(row.get("field_fy", 0.0), 0.0)
    field_mag = math.sqrt((field_x * field_x) + (field_y * field_y))
    if field_mag > 1e-9:
        return field_x, field_y

    route_x = _safe_float(row.get("route_x", float("nan")), float("nan"))
    route_y = _safe_float(row.get("route_y", float("nan")), float("nan"))
    if math.isfinite(route_x) and math.isfinite(route_y):
        px = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        py = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        return route_x - px, route_y - py

    return 0.0, 0.0


def anti_clump_semantic_charge(row: dict[str, Any]) -> float:
    direct = max(0.0, _safe_float(row.get("semantic_charge", 0.0), 0.0))
    bundle_charge = max(0.0, _safe_float(row.get("semantic_bundle_charge", 0.0), 0.0))
    bundle_mass = max(0.0, _safe_float(row.get("semantic_bundle_mass", 0.0), 0.0))
    bundle_gravity = max(0.0, _safe_float(row.get("semantic_bundle_gravity", 0.0), 0.0))
    semantic_mass = max(0.0, _safe_float(row.get("semantic_mass", 0.0), 0.0))
    semantic_mass_term = _clamp01(math.log1p(semantic_mass) / math.log1p(12.0))
    bundle_term = _clamp01(
        (bundle_charge * 0.58) + (bundle_mass * 0.24) + (bundle_gravity * 0.18)
    )
    return max(direct, bundle_charge, bundle_term, semantic_mass_term)


def anti_clump_metrics(
    positions: list[tuple[float, float]],
    *,
    previous_collision_count: int,
    particles: Any | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    count = len(positions)

    snr_min = max(0.05, _cfg_float(config, "snr_min", 0.85))
    snr_max = max(snr_min + 0.05, _cfg_float(config, "snr_max", 1.65))
    snr_eps = max(1e-9, _cfg_float(config, "snr_eps", 1e-6))
    if count <= 1:
        return {
            "count": float(count),
            "clump_score": 0.0,
            "nn_term": 0.0,
            "entropy_norm": 1.0,
            "hotspot_term": 0.0,
            "collision_term": 0.0,
            "collision_rate": 0.0,
            "median_distance": 0.0,
            "target_distance": 0.0,
            "top_share": 0.0,
            "mean_spacing": 1.0,
            "fano_factor": 0.0,
            "fano_excess": 0.0,
            "spatial_noise": 0.0,
            "motion_signal": 0.0,
            "motion_noise": 0.0,
            "motion_samples": 0.0,
            "semantic_noise": 0.0,
            "snr_signal": 0.0,
            "snr_noise": snr_eps,
            "snr": 0.0,
            "snr_valid": 0.0,
            "snr_low_gap": 0.0,
            "snr_high_gap": 0.0,
            "snr_min": snr_min,
            "snr_max": snr_max,
            "snr_in_band": 0.0,
        }

    density = max(1e-9, float(count))
    mean_spacing = max(1e-6, 1.0 / math.sqrt(density))
    density_grid = int(math.ceil(1.0 / mean_spacing))
    grid_cap = max(4, _cfg_int(config, "grid_cap", 96))
    grid = max(4, min(grid_cap, max(2, density_grid)))
    sample_limit = max(8, _cfg_int(config, "sample_limit", 96))
    cell_counts = [0 for _ in range(grid * grid)]
    indices_by_cell: dict[int, list[int]] = {}
    coords: list[tuple[int, int]] = []

    for index, (x_raw, y_raw) in enumerate(positions):
        x = _clamp01(_finite_float(x_raw, 0.5))
        y = _clamp01(_finite_float(y_raw, 0.5))
        cx = min(grid - 1, max(0, int(math.floor(x * grid))))
        cy = min(grid - 1, max(0, int(math.floor(y * grid))))
        cell_id = (cy * grid) + cx
        cell_counts[cell_id] += 1
        bucket = indices_by_cell.get(cell_id)
        if bucket is None:
            indices_by_cell[cell_id] = [index]
        else:
            bucket.append(index)
        coords.append((cx, cy))

    non_zero_counts = [value for value in cell_counts if value > 0]
    entropy = 0.0
    for value in non_zero_counts:
        ratio = _finite_float(value / float(count), 0.0)
        if ratio > 1e-12:
            entropy -= ratio * math.log(ratio)
    max_entropy = math.log(max(1, min(len(non_zero_counts), count)))
    entropy_norm = _clamp01(entropy / max_entropy) if max_entropy > 1e-12 else 1.0

    mean_bin_count = float(count) / float(max(1, len(cell_counts)))
    variance = 0.0
    if mean_bin_count > 1e-12:
        for value in cell_counts:
            delta = float(value) - mean_bin_count
            variance += delta * delta
        variance /= float(max(1, len(cell_counts)))
    fano_factor = variance / mean_bin_count if mean_bin_count > 1e-12 else 0.0
    fano_excess = max(0.0, fano_factor - 1.0)
    spatial_noise = fano_excess

    top_share = 0.0
    hotspot_term = 0.0
    if non_zero_counts:
        ordered_counts = sorted(non_zero_counts, reverse=True)
        top_k = max(1, int(math.ceil(len(ordered_counts) * 0.1)))
        top_share = sum(ordered_counts[:top_k]) / float(max(1, count))
        uniform_share = top_k / float(max(1, len(ordered_counts)))
        hotspot_term = _clamp01(
            (top_share - uniform_share) / max(1e-6, 1.0 - uniform_share)
        )

    sample_step = max(1, int(math.floor(count / float(sample_limit))))
    sample_indices = list(range(0, count, sample_step))[:sample_limit]
    if not sample_indices:
        sample_indices = [0]
    neighbor_distances: list[float] = []

    for index in sample_indices:
        x = _clamp01(_finite_float(positions[index][0], 0.5))
        y = _clamp01(_finite_float(positions[index][1], 0.5))
        cx, cy = coords[index]
        nearest_sq = float("inf")
        for ny in range(max(0, cy - 1), min(grid - 1, cy + 1) + 1):
            for nx in range(max(0, cx - 1), min(grid - 1, cx + 1) + 1):
                neighbor_cell_id = (ny * grid) + nx
                candidates = indices_by_cell.get(neighbor_cell_id, [])
                for other_index in candidates:
                    if other_index == index:
                        continue
                    ox = _clamp01(_finite_float(positions[other_index][0], 0.5))
                    oy = _clamp01(_finite_float(positions[other_index][1], 0.5))
                    dx = ox - x
                    dy = oy - y
                    dist_sq = (dx * dx) + (dy * dy)
                    if dist_sq < nearest_sq:
                        nearest_sq = dist_sq
        if math.isfinite(nearest_sq):
            neighbor_distances.append(math.sqrt(max(0.0, nearest_sq)))
        else:
            neighbor_distances.append(math.sqrt(2.0) / float(grid))

    median_distance = _median(neighbor_distances)
    target_distance = max(1e-6, 0.5 * mean_spacing)
    nn_ratio = median_distance / max(1e-6, target_distance)
    nn_term = _clamp01(max(0.0, (1.0 / max(1e-6, nn_ratio)) - 1.0))

    collision_rate = max(
        0.0,
        _safe_float(previous_collision_count, 0.0) / float(max(1, count)),
    )
    collision_rate_ref = max(0.25, _cfg_float(config, "collision_rate_ref", 2.2))
    collision_term = _clamp01(collision_rate / collision_rate_ref)

    particle_rows: list[dict[str, Any]] = []
    if isinstance(particles, dict):
        particle_rows = [row for row in particles.values() if isinstance(row, dict)]
    elif isinstance(particles, list):
        particle_rows = [row for row in particles if isinstance(row, dict)]

    motion_signal = 0.0
    motion_noise = 0.0
    motion_signal_sum = 0.0
    motion_noise_sum = 0.0
    motion_samples = 0

    semantic_charge_bins = [0.0 for _ in range(grid * grid)]
    semantic_charge_total = 0.0

    for row in particle_rows:
        px = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        py = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        cx = min(grid - 1, max(0, int(math.floor(px * grid))))
        cy = min(grid - 1, max(0, int(math.floor(py * grid))))
        cell_id = (cy * grid) + cx

        semantic_charge = anti_clump_semantic_charge(row)
        if semantic_charge > 1e-12:
            semantic_charge_bins[cell_id] = (
                semantic_charge_bins[cell_id] + semantic_charge
            )
            semantic_charge_total += semantic_charge

        if bool(row.get("is_nexus", False)):
            continue

        vx = _safe_float(row.get("vx", 0.0), 0.0)
        vy = _safe_float(row.get("vy", 0.0), 0.0)
        field_x, field_y = anti_clump_field_vector(row)
        field_mag = math.sqrt((field_x * field_x) + (field_y * field_y))
        if field_mag <= 1e-9:
            continue

        fhat_x = field_x / field_mag
        fhat_y = field_y / field_mag
        projection = (vx * fhat_x) + (vy * fhat_y)
        projection_sq = projection * projection
        perp_x = vx - (projection * fhat_x)
        perp_y = vy - (projection * fhat_y)
        perp_sq = (perp_x * perp_x) + (perp_y * perp_y)

        motion_signal_sum += projection_sq
        motion_noise_sum += perp_sq
        motion_samples += 1

    if motion_samples > 0:
        motion_signal = motion_signal_sum / float(motion_samples)
        motion_noise = motion_noise_sum / float(motion_samples)

    semantic_noise = 0.0
    if semantic_charge_total > 1e-12:
        inv_count = 1.0 / float(count)
        inv_semantic = 1.0 / semantic_charge_total
        kl_divergence = 0.0
        eps = max(1e-12, _cfg_float(config, "snr_eps", 1e-6))
        for cell_id, observed_count in enumerate(cell_counts):
            if observed_count <= 0:
                continue
            rho = max(eps, float(observed_count) * inv_count)
            tau = max(eps, semantic_charge_bins[cell_id] * inv_semantic)
            kl_divergence += rho * math.log(rho / tau)
        semantic_noise = max(0.0, kl_divergence)

    alpha = max(0.0, _cfg_float(config, "snr_alpha", 0.4))
    beta = max(0.0, _cfg_float(config, "snr_beta", 0.2))
    eps = max(1e-9, _cfg_float(config, "snr_eps", 1e-6))

    snr_noise = (
        (motion_noise * (1.0 + (alpha * spatial_noise))) + (beta * semantic_noise) + eps
    )
    snr_signal = motion_signal
    snr_valid = motion_samples > 0
    snr = (snr_signal / max(eps, snr_noise)) if snr_valid else 0.0
    snr_low_gap = max(0.0, (snr_min - snr) / max(1e-6, snr_min)) if snr_valid else 0.0
    snr_high_gap = max(0.0, (snr - snr_max) / max(1e-6, snr_max)) if snr_valid else 0.0
    snr_in_band = snr_valid and snr_low_gap <= 1e-12 and snr_high_gap <= 1e-12

    fano_term = _clamp01(fano_excess / (1.0 + fano_excess))
    off_field_term = (
        _clamp01(motion_noise / max(eps, motion_signal + motion_noise))
        if motion_samples > 0
        else 0.0
    )
    semantic_noise_norm = _clamp01(1.0 - math.exp(-semantic_noise))

    clump_score = _clamp01(
        (fano_term * 0.26)
        + (nn_term * 0.20)
        + ((1.0 - entropy_norm) * 0.16)
        + (hotspot_term * 0.12)
        + (collision_term * 0.10)
        + (off_field_term * 0.10)
        + (semantic_noise_norm * 0.06)
    )

    return {
        "count": float(count),
        "clump_score": clump_score,
        "nn_term": nn_term,
        "entropy_norm": entropy_norm,
        "hotspot_term": hotspot_term,
        "collision_term": collision_term,
        "collision_rate": collision_rate,
        "median_distance": median_distance,
        "target_distance": target_distance,
        "top_share": _clamp01(top_share),
        "mean_spacing": mean_spacing,
        "fano_factor": max(0.0, fano_factor),
        "fano_excess": fano_excess,
        "spatial_noise": spatial_noise,
        "motion_signal": max(0.0, motion_signal),
        "motion_noise": max(0.0, motion_noise),
        "motion_samples": float(max(0, motion_samples)),
        "semantic_noise": semantic_noise,
        "snr_signal": max(0.0, snr_signal),
        "snr_noise": max(eps, snr_noise),
        "snr": max(0.0, snr),
        "snr_valid": 1.0 if snr_valid else 0.0,
        "snr_low_gap": snr_low_gap,
        "snr_high_gap": snr_high_gap,
        "snr_min": snr_min,
        "snr_max": snr_max,
        "snr_in_band": 1.0 if snr_in_band else 0.0,
    }


def anti_clump_scales(drive: float) -> dict[str, float]:
    limited_drive = _clamp_range(drive, -1.0, 1.0)
    return {
        "semantic": _clamp_range(math.exp(-0.8 * limited_drive), 0.35, 1.2),
        "edge": _clamp_range(math.exp(-0.6 * limited_drive), 0.4, 1.1),
        "anchor": _clamp_range(math.exp(-0.7 * limited_drive), 0.45, 1.1),
        "spawn": _clamp_range(math.exp(-0.5 * limited_drive), 0.5, 1.05),
        "tangent": _clamp_range(math.exp(0.5 * limited_drive), 0.8, 1.8),
        "friction_slip": _clamp_range(math.exp(0.45 * limited_drive), 0.8, 1.24),
        "simplex_gain": _clamp_range(math.exp(0.62 * limited_drive), 0.72, 2.2),
        "simplex_scale": _clamp_range(math.exp(0.24 * limited_drive), 0.82, 1.34),
    }


def anti_clump_summary_from_snapshot(
    snapshot: dict[str, Any] | None,
    *,
    target: float,
    drive: float,
    scales: dict[str, Any] | None = None,
    snr_default_min: float = 0.85,
    snr_default_max: float = 1.65,
    scale_order: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    row = snapshot if isinstance(snapshot, dict) else {}
    snr = max(0.0, _safe_float(row.get("snr", 0.0), 0.0))
    snr_valid = bool(row.get("snr_valid", False))

    snr_min_default = max(0.05, _safe_float(snr_default_min, 0.85))
    snr_max_default = max(
        snr_min_default + 0.05,
        _safe_float(snr_default_max, max(snr_min_default + 0.05, 1.65)),
    )
    snr_min = max(
        0.05,
        _safe_float(row.get("snr_min", snr_min_default), snr_min_default),
    )
    snr_max = max(
        snr_min + 0.05,
        _safe_float(row.get("snr_max", snr_max_default), snr_max_default),
    )
    snr_low_gap = max(0.0, _safe_float(row.get("snr_low_gap", 0.0), 0.0))
    snr_high_gap = max(0.0, _safe_float(row.get("snr_high_gap", 0.0), 0.0))
    snr_in_band = bool(row.get("snr_in_band", False))

    selected_scale_order = (
        tuple(scale_order)
        if isinstance(scale_order, (tuple, list)) and len(scale_order) > 0
        else DEFAULT_ANTI_CLUMP_SCALE_ORDER
    )
    scale_row = scales if isinstance(scales, dict) else {}
    scale_summary: dict[str, float] = {}
    for key in selected_scale_order:
        token = str(key or "").strip()
        if not token:
            continue
        scale_summary[token] = round(_safe_float(scale_row.get(token, 1.0), 1.0), 6)

    clump_raw = _clamp01(_safe_float(row.get("clump_score", 0.0), 0.0))
    clump_score = _clamp01(_safe_float(row.get("score_ema", clump_raw), clump_raw))

    return {
        "target": round(_clamp01(_safe_float(target, 0.38)), 6),
        "clump_score": round(clump_score, 6),
        "raw_clump_score": round(clump_raw, 6),
        "drive": round(_clamp_range(drive, -1.0, 1.0), 6),
        "error": round(_safe_float(row.get("error", 0.0), 0.0), 6),
        "integral": round(_safe_float(row.get("integral", 0.0), 0.0), 6),
        "updated": bool(row.get("updated", False)),
        "tick": max(0, _safe_int(row.get("tick", 0), 0)),
        "particle_count": max(0, _safe_int(row.get("particle_count", 0), 0)),
        "snr": round(snr, 6),
        "snr_valid": snr_valid,
        "snr_band": {
            "min": round(snr_min, 6),
            "max": round(snr_max, 6),
            "low_gap": round(snr_low_gap, 6),
            "high_gap": round(snr_high_gap, 6),
            "in_band": snr_in_band,
        },
        "metrics": {
            "nn_term": round(_clamp01(_safe_float(row.get("nn_term", 0.0), 0.0)), 6),
            "entropy_norm": round(
                _clamp01(_safe_float(row.get("entropy_norm", 1.0), 1.0)),
                6,
            ),
            "hotspot_term": round(
                _clamp01(_safe_float(row.get("hotspot_term", 0.0), 0.0)),
                6,
            ),
            "collision_term": round(
                _clamp01(_safe_float(row.get("collision_term", 0.0), 0.0)),
                6,
            ),
            "collision_rate": round(
                max(0.0, _safe_float(row.get("collision_rate", 0.0), 0.0)),
                6,
            ),
            "median_distance": round(
                max(0.0, _safe_float(row.get("median_distance", 0.0), 0.0)),
                6,
            ),
            "target_distance": round(
                max(0.0, _safe_float(row.get("target_distance", 0.0), 0.0)),
                6,
            ),
            "top_share": round(
                _clamp01(_safe_float(row.get("top_share", 0.0), 0.0)),
                6,
            ),
            "mean_spacing": round(
                max(0.0, _safe_float(row.get("mean_spacing", 0.0), 0.0)),
                6,
            ),
            "fano_factor": round(
                max(0.0, _safe_float(row.get("fano_factor", 0.0), 0.0)),
                6,
            ),
            "fano_excess": round(
                max(0.0, _safe_float(row.get("fano_excess", 0.0), 0.0)),
                6,
            ),
            "spatial_noise": round(
                max(0.0, _safe_float(row.get("spatial_noise", 0.0), 0.0)),
                6,
            ),
            "motion_signal": round(
                max(0.0, _safe_float(row.get("motion_signal", 0.0), 0.0)),
                6,
            ),
            "motion_noise": round(
                max(0.0, _safe_float(row.get("motion_noise", 0.0), 0.0)),
                6,
            ),
            "motion_samples": max(0, _safe_int(row.get("motion_samples", 0), 0)),
            "semantic_noise": round(
                max(0.0, _safe_float(row.get("semantic_noise", 0.0), 0.0)),
                6,
            ),
            "snr_signal": round(
                max(0.0, _safe_float(row.get("snr_signal", 0.0), 0.0)),
                6,
            ),
            "snr_noise": round(
                max(0.0, _safe_float(row.get("snr_noise", 0.0), 0.0)),
                6,
            ),
            "snr": round(snr, 6),
            "snr_low_gap": round(snr_low_gap, 6),
            "snr_high_gap": round(snr_high_gap, 6),
            "snr_min": round(snr_min, 6),
            "snr_max": round(snr_max, 6),
            "snr_in_band": snr_in_band,
        },
        "scales": scale_summary,
    }


def anti_clump_controller_update(
    controller_state: dict[str, Any] | None,
    *,
    particles: dict[str, Any],
    previous_collision_count: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = dict(controller_state) if isinstance(controller_state, dict) else {}
    tick = max(0, _safe_int(state.get("tick", 0), 0)) + 1

    drive_limit = max(0.25, _cfg_float(config, "drive_limit", 1.0))
    integral_limit = max(0.25, _cfg_float(config, "integral_limit", 1.5))
    target = _clamp01(_cfg_float(config, "target", 0.38))
    kp = max(0.0, _cfg_float(config, "kp", 0.22))
    ki = max(0.0, _cfg_float(config, "ki", 0.04))
    high_snr_perturb_gain = max(0.0, _cfg_float(config, "high_snr_perturb_gain", 0.28))
    smoothing = _clamp01(_cfg_float(config, "smoothing", 0.15))
    update_stride = max(1, _cfg_int(config, "update_stride", 10))
    min_particles = max(8, _cfg_int(config, "min_particles", 24))

    previous_drive = _clamp_range(
        _safe_float(state.get("drive", 0.0), 0.0),
        -drive_limit,
        drive_limit,
    )
    integral = _clamp_range(
        _safe_float(state.get("integral", 0.0), 0.0),
        -integral_limit,
        integral_limit,
    )
    score_ema = _clamp01(_safe_float(state.get("score_ema", target), target))

    should_update = (tick % update_stride == 0) or ("clump_score" not in state)
    if should_update:
        positions = anti_clump_positions_from_particles(particles)
        metrics = anti_clump_metrics(
            positions,
            previous_collision_count=previous_collision_count,
            particles=particles,
            config=config,
        )
        particle_count = max(0, _safe_int(metrics.get("count", 0.0), 0))
        score_raw = _clamp01(
            _safe_float(metrics.get("clump_score", score_ema), score_ema)
        )
        score_ema = _clamp01((score_ema * 0.82) + (score_raw * 0.18))

        snr_valid = _safe_float(metrics.get("snr_valid", 0.0), 0.0) > 0.5
        snr_low_gap = max(0.0, _safe_float(metrics.get("snr_low_gap", 0.0), 0.0))
        snr_high_gap = max(0.0, _safe_float(metrics.get("snr_high_gap", 0.0), 0.0))
        snr_in_band = _safe_float(metrics.get("snr_in_band", 0.0), 0.0) > 0.5
        snr_min_value = max(0.05, _safe_float(metrics.get("snr_min", 0.85), 0.85))
        snr_max_value = max(
            snr_min_value + 0.05,
            _safe_float(metrics.get("snr_max", 1.65), 1.65),
        )

        if particle_count < min_particles:
            error = score_ema - target
            integral = _clamp_range(integral * 0.82, -integral_limit, integral_limit)
            raw_drive = 0.0
        elif snr_valid:
            error = snr_low_gap
            if snr_in_band:
                integral = _clamp_range(
                    integral * 0.84, -integral_limit, integral_limit
                )
            else:
                integral = _clamp_range(
                    integral + error,
                    -integral_limit,
                    integral_limit,
                )
            raw_drive = (
                (kp * error) + (ki * integral) + (snr_high_gap * high_snr_perturb_gain)
            )
        else:
            error = score_ema - target
            integral = _clamp_range(
                integral + error,
                -integral_limit,
                integral_limit,
            )
            raw_drive = (kp * error) + (ki * integral)

        drive = _clamp_range(
            ((previous_drive * (1.0 - smoothing)) + (raw_drive * smoothing)),
            -drive_limit,
            drive_limit,
        )
        state.update(
            {
                "tick": tick,
                "updated": True,
                "drive": drive,
                "integral": integral,
                "error": error,
                "score_ema": score_ema,
                "particle_count": particle_count,
                "clump_score": score_raw,
                "nn_term": _clamp01(_safe_float(metrics.get("nn_term", 0.0), 0.0)),
                "entropy_norm": _clamp01(
                    _safe_float(metrics.get("entropy_norm", 1.0), 1.0)
                ),
                "hotspot_term": _clamp01(
                    _safe_float(metrics.get("hotspot_term", 0.0), 0.0)
                ),
                "collision_term": _clamp01(
                    _safe_float(metrics.get("collision_term", 0.0), 0.0)
                ),
                "collision_rate": max(
                    0.0,
                    _safe_float(metrics.get("collision_rate", 0.0), 0.0),
                ),
                "median_distance": max(
                    0.0,
                    _safe_float(metrics.get("median_distance", 0.0), 0.0),
                ),
                "target_distance": max(
                    0.0,
                    _safe_float(metrics.get("target_distance", 0.0), 0.0),
                ),
                "top_share": _clamp01(_safe_float(metrics.get("top_share", 0.0), 0.0)),
                "mean_spacing": max(
                    0.0,
                    _safe_float(metrics.get("mean_spacing", 0.0), 0.0),
                ),
                "fano_factor": max(
                    0.0,
                    _safe_float(metrics.get("fano_factor", 0.0), 0.0),
                ),
                "fano_excess": max(
                    0.0,
                    _safe_float(metrics.get("fano_excess", 0.0), 0.0),
                ),
                "spatial_noise": max(
                    0.0,
                    _safe_float(metrics.get("spatial_noise", 0.0), 0.0),
                ),
                "motion_signal": max(
                    0.0,
                    _safe_float(metrics.get("motion_signal", 0.0), 0.0),
                ),
                "motion_noise": max(
                    0.0,
                    _safe_float(metrics.get("motion_noise", 0.0), 0.0),
                ),
                "motion_samples": max(
                    0,
                    _safe_int(metrics.get("motion_samples", 0.0), 0),
                ),
                "semantic_noise": max(
                    0.0,
                    _safe_float(metrics.get("semantic_noise", 0.0), 0.0),
                ),
                "snr_signal": max(
                    0.0,
                    _safe_float(metrics.get("snr_signal", 0.0), 0.0),
                ),
                "snr_noise": max(
                    0.0,
                    _safe_float(metrics.get("snr_noise", 0.0), 0.0),
                ),
                "snr": max(0.0, _safe_float(metrics.get("snr", 0.0), 0.0)),
                "snr_valid": bool(snr_valid),
                "snr_low_gap": snr_low_gap,
                "snr_high_gap": snr_high_gap,
                "snr_min": snr_min_value,
                "snr_max": snr_max_value,
                "snr_in_band": bool(snr_in_band),
            }
        )
        return state

    state["tick"] = tick
    state["updated"] = False
    return state
