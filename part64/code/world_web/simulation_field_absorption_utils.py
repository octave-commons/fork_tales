"""Resource absorption helpers for simulation field particles."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable


def _resource_mix_vector(
    row: dict[str, Any],
    *,
    safe_float: Callable[[Any, float], float],
    canonical_resource_type: Callable[[str], str],
) -> tuple[dict[str, float], str]:
    amount = safe_float(row.get("resource_emit_amount", 0.0), 0.0)
    resource_type = str(row.get("resource_type", "cpu") or "cpu")
    mix_raw = row.get("resource_mix_vector", {})
    mix_vector: dict[str, float] = {}
    if isinstance(mix_raw, dict):
        for key, value in mix_raw.items():
            resource_name = canonical_resource_type(str(key or ""))
            if not resource_name:
                continue
            mix_amount = max(0.0, safe_float(value, 0.0))
            if mix_amount <= 1e-9:
                continue
            mix_vector[resource_name] = mix_vector.get(resource_name, 0.0) + mix_amount

    if not mix_vector:
        resource_name = canonical_resource_type(resource_type) or "cpu"
        mix_vector[resource_name] = max(0.0, amount)

    primary_resource = canonical_resource_type(resource_type)
    if (
        not primary_resource
        or primary_resource not in mix_vector
        or safe_float(mix_vector.get(primary_resource, 0.0), 0.0) <= 1e-9
    ):
        primary_resource = max(
            mix_vector.keys(),
            key=lambda name: safe_float(mix_vector.get(name, 0.0), 0.0),
        )
    return mix_vector, primary_resource


def apply_resource_daimoi_absorption(
    row: dict[str, Any],
    *,
    presence_centers: dict[str, tuple[float, float]],
    now_value: float,
    next_x: float,
    next_y: float,
    vx_value: float,
    vy_value: float,
    get_presence_runtime_manager: Callable[[], Any],
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[float], float],
    canonical_resource_type: Callable[[str], str],
    resource_daimoi_wallet_floor: dict[str, float],
    normalize_resource_wallet_denoms: Callable[[dict[str, Any]], list[dict[str, Any]]],
    wallet_denoms_add_vector: Callable[[list[dict[str, Any]], dict[str, float]], None],
) -> tuple[float, float]:
    if not bool(row.get("resource_daimoi", False)):
        return next_x, next_y

    target_id = str(row.get("resource_target_presence_id", "") or "").strip()
    if not target_id:
        return next_x, next_y

    tx, ty = presence_centers.get(target_id, (0.5, 0.5))
    dist_sq = ((tx - next_x) ** 2) + ((ty - next_y) ** 2)
    if dist_sq >= 0.0036:
        return next_x, next_y

    manager = get_presence_runtime_manager()
    state = manager.get_state(target_id)
    wallet = state.setdefault("resource_wallet", {})
    if not isinstance(wallet, dict):
        wallet = {}
        state["resource_wallet"] = wallet

    mix_vector, primary_resource = _resource_mix_vector(
        row,
        safe_float=safe_float,
        canonical_resource_type=canonical_resource_type,
    )

    current_bal = safe_float(wallet.get(primary_resource, 0.0), 0.0)
    wallet_floor = max(
        0.1,
        safe_float(resource_daimoi_wallet_floor.get(primary_resource, 4.0), 4.0),
    )
    pressure = clamp01(current_bal / max(1e-6, current_bal + wallet_floor))
    absorb_prob = 1.0 - (pressure * 0.85)

    seed_val = int(
        hashlib.sha1(f"{row.get('id')}|{now_value}".encode("utf-8")).hexdigest()[:8],
        16,
    )
    rng_val = (seed_val % 1000) / 1000.0

    if rng_val < absorb_prob:
        already_credited = bool(row.get("resource_emit_wallet_credit", False))
        if not already_credited:
            for resource_name, mix_amount in mix_vector.items():
                prior = max(0.0, safe_float(wallet.get(resource_name, 0.0), 0.0))
                wallet[resource_name] = round(prior + mix_amount, 6)
            denoms = normalize_resource_wallet_denoms(state)
            wallet_denoms_add_vector(denoms, mix_vector)
            state["resource_wallet_denoms"] = denoms
        row["_absorbed"] = True
        row["resource_wallet_credit_reused"] = bool(already_credited)
        return next_x, next_y

    row["_deflected"] = True
    row["vx"] = -vx_value * 0.6
    row["vy"] = -vy_value * 0.6
    dx = next_x - tx
    dy = next_y - ty
    magnitude = math.hypot(dx, dy)
    if magnitude > 1e-6:
        next_x = clamp01(next_x + ((dx / magnitude) * 0.03))
        next_y = clamp01(next_y + ((dy / magnitude) * 0.03))
    return next_x, next_y
