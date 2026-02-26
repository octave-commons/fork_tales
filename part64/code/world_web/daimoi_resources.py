from __future__ import annotations

import math
from typing import Any

from .metrics import _safe_float


DAIMOI_RESOURCE_ALIASES: dict[str, str] = {
    "cpu": "cpu",
    "gpu": "gpu",
    "gpu0": "gpu",
    "gpu1": "gpu",
    "gpu2": "gpu",
    "npu": "npu",
    "npu0": "npu",
    "ram": "ram",
    "mem": "ram",
    "memory": "ram",
    "disk": "disk",
    "storage": "disk",
    "network": "network",
    "net": "network",
}


DAIMOI_WALLET_FLOOR: dict[str, float] = {
    "cpu": 6.0,
    "gpu": 5.0,
    "npu": 4.0,
    "ram": 8.0,
    "disk": 7.0,
    "network": 7.0,
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, float):
        return value if math.isfinite(value) else default
    if isinstance(value, int):
        return float(value)
    parsed = _safe_float(value, default)
    if not math.isfinite(parsed):
        return default
    return parsed


def _clamp01_finite(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _finite_float(value, default)))


def resource_wallet_by_type(
    wallet: dict[str, Any] | None,
    *,
    resource_keys: tuple[str, ...],
    resource_aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    values = {resource: 0.0 for resource in resource_keys}
    if not isinstance(wallet, dict):
        return values

    alias_map = (
        resource_aliases
        if isinstance(resource_aliases, dict)
        else DAIMOI_RESOURCE_ALIASES
    )
    for key, raw in wallet.items():
        token = str(key or "").strip().lower()
        if not token:
            continue
        resource = alias_map.get(token)
        if not resource or resource not in values:
            continue
        values[resource] = values[resource] + max(0.0, _finite_float(raw, 0.0))
    return values


def presence_need_by_resource(
    impact: dict[str, Any] | None,
    *,
    queue_ratio: float,
    resource_keys: tuple[str, ...],
    wallet_floor: dict[str, float] | None = None,
    resource_aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    impact_row = impact if isinstance(impact, dict) else {}
    affected_by = impact_row.get("affected_by", {})
    if not isinstance(affected_by, dict):
        affected_by = {}

    resource_signal = _clamp01_finite(affected_by.get("resource", 0.0), 0.0)
    queue_signal = _clamp01_finite(queue_ratio, 0.0)

    wallet = resource_wallet_by_type(
        impact_row.get("resource_wallet", {}),
        resource_keys=resource_keys,
        resource_aliases=resource_aliases,
    )
    floor_map = wallet_floor if isinstance(wallet_floor, dict) else DAIMOI_WALLET_FLOOR

    needs: dict[str, float] = {}
    for resource in resource_keys:
        floor = max(
            0.1,
            _finite_float(floor_map.get(resource, 6.0), 6.0),
        )
        balance = max(0.0, _finite_float(wallet.get(resource, 0.0), 0.0))
        deficit = _clamp01_finite(1.0 - (balance / floor), 0.0)
        queue_push = queue_signal * (0.18 if resource in {"network", "disk"} else 0.08)
        needs[resource] = _clamp01_finite(
            (deficit * 0.64) + (resource_signal * 0.26) + queue_push,
            0.0,
        )
    return needs
