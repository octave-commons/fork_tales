# SPDX-License-Identifier: GPL-3.0-or-later
# This file is part of Fork Tales.
# Copyright (C) 2024-2025 Fork Tales Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any

from .constants import ENTITY_MANIFEST, USER_PRESENCE_ID
from .metrics import _safe_float, _safe_int, _clamp01, _stable_ratio

_RESOURCE_DAIMOI_TYPES: tuple[str, ...] = (
    "cpu",
    "ram",
    "disk",
    "network",
    "gpu",
    "npu",
)
_RESOURCE_DAIMOI_TYPE_ALIASES: dict[str, str] = {
    "gpu1": "gpu",
    "gpu2": "gpu",
    "gpu_intel": "gpu",
    "intel": "gpu",
    "npu0": "npu",
    "net": "network",
    "netup": "network",
    "netdown": "network",
}
_RESOURCE_DAIMOI_WALLET_FLOOR: dict[str, float] = {
    "cpu": 6.0,
    "ram": 6.0,
    "disk": 4.0,
    "network": 4.0,
    "gpu": 5.0,
    "npu": 5.0,
}
_RESOURCE_DAIMOI_WALLET_CAP: dict[str, float] = {
    "cpu": 48.0,
    "ram": 48.0,
    "disk": 32.0,
    "network": 32.0,
    "gpu": 40.0,
    "npu": 40.0,
}
_RESOURCE_DAIMOI_DENOM_QUANTUM = max(
    1e-6,
    min(
        0.1,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_DENOM_QUANTUM", "0.000001")
            or "0.000001",
            0.000001,
        ),
    ),
)
_RESOURCE_DAIMOI_DENOM_OVERPAY_PENALTY = max(
    0.0,
    min(
        2.0,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_DENOM_OVERPAY_PENALTY", "0.18")
            or "0.18",
            0.18,
        ),
    ),
)
_RESOURCE_DAIMOI_DENOM_GREEDY_STEP_LIMIT = max(
    8,
    min(
        4096,
        int(
            _safe_float(
                os.getenv("SIMULATION_RESOURCE_DAIMOI_DENOM_GREEDY_STEP_LIMIT", "256")
                or "256",
                256.0,
            )
        ),
    ),
)
_RESOURCE_DAIMOI_MIX_EPSILON_BASE = max(
    0.001,
    min(
        0.45,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_MIX_EPSILON", "0.12") or "0.12",
            0.12,
        ),
    ),
)
_RESOURCE_DAIMOI_MIX_PRESSURE_GAIN = max(
    0.0,
    min(
        1.0,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_MIX_PRESSURE_GAIN", "0.35") or "0.35",
            0.35,
        ),
    ),
)
_RESOURCE_DAIMOI_PRESSURE_SOFT_PERCENT_DEFAULT = max(
    0.0,
    min(
        100.0,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_PRESSURE_SOFT_PERCENT", "50") or "50",
            50.0,
        ),
    ),
)
_RESOURCE_DAIMOI_PRESSURE_HARD_PERCENT_DEFAULT = max(
    0.0,
    min(
        100.0,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_PRESSURE_HARD_PERCENT", "80") or "80",
            80.0,
        ),
    ),
)
_RESOURCE_DAIMOI_DEBT_LOCK = threading.Lock()
_RESOURCE_DAIMOI_DEBT_STATE: dict[str, float] = {
    resource_type: 0.0 for resource_type in _RESOURCE_DAIMOI_TYPES
}
_RESOURCE_DAIMOI_DEBT_LAST_MONOTONIC = time.monotonic()
_RESOURCE_DAIMOI_VELOCITY_LOCK = threading.Lock()
_RESOURCE_DAIMOI_VELOCITY_STATE: dict[str, Any] = {
    "last_monotonic": time.monotonic(),
    "usage_prev": {resource_type: 0.0 for resource_type in _RESOURCE_DAIMOI_TYPES},
}
_RESOURCE_DAIMOI_EMIT_BASE_RATE = max(
    0.0,
    min(
        0.2,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_EMIT_BASE_RATE", "0.0025")
            or "0.0025",
            0.0025,
        ),
    ),
)
_RESOURCE_DAIMOI_EMIT_ALPHA = max(
    0.0,
    min(
        0.8,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_EMIT_ALPHA", "0.016") or "0.016",
            0.016,
        ),
    ),
)
_RESOURCE_DAIMOI_EMIT_BETA = max(
    0.0,
    min(
        0.8,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_EMIT_BETA", "0.010") or "0.010",
            0.010,
        ),
    ),
)
_RESOURCE_DAIMOI_EMIT_GAMMA = max(
    0.0,
    min(
        0.8,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_EMIT_GAMMA", "0.012") or "0.012",
            0.012,
        ),
    ),
)
_RESOURCE_DAIMOI_ACTION_BASE_COST = 0.00001
_RESOURCE_DAIMOI_ACTION_COST_MAX = 0.0028
_RESOURCE_DAIMOI_ACTION_SATISFIED_RATIO = 0.85
_RESOURCE_DAIMOI_ACTION_RISK_PREMIUM = max(
    0.0,
    min(
        4.0,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_ACTION_RISK_PREMIUM", "0.55")
            or "0.55",
            0.55,
        ),
    ),
)
_RESOURCE_DAIMOI_ACTION_UTILITY_ETA = max(
    0.0,
    min(
        8.0,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_ACTION_UTILITY_ETA", "1.0") or "1.0",
            1.0,
        ),
    ),
)
_RESOURCE_DAIMOI_ACTION_UTILITY_XI = max(
    0.0,
    min(
        8.0,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_ACTION_UTILITY_XI", "1.0") or "1.0",
            1.0,
        ),
    ),
)
_RESOURCE_DAIMOI_ACTION_UTILITY_KAPPA = max(
    0.0,
    min(
        8.0,
        _safe_float(
            os.getenv("SIMULATION_RESOURCE_DAIMOI_ACTION_UTILITY_KAPPA", "0.55")
            or "0.55",
            0.55,
        ),
    ),
)
_RESOURCE_CTL_BUDGET_CAP_DEFAULT: dict[str, float] = {
    "cpu": 1.0,
    "ram": 0.85,
    "disk": 0.6,
    "network": 0.6,
    "gpu": 0.7,
    "npu": 0.7,
}
_RESOURCE_CTL_BUDGET_RECHARGE_DEFAULT: dict[str, float] = {
    "cpu": 0.42,
    "ram": 0.34,
    "disk": 0.24,
    "network": 0.24,
    "gpu": 0.28,
    "npu": 0.28,
}
_RESOURCE_CTL_BUDGET_EVAL_COST: dict[str, float] = {
    "cpu": 0.0032,
    "ram": 0.0018,
    "disk": 0.0011,
    "network": 0.0011,
    "gpu": 0.0014,
    "npu": 0.0014,
}
_RESOURCE_CTL_BUDGET_DENOM_EXTRA_COST: dict[str, float] = {
    "cpu": 0.0024,
    "ram": 0.0015,
    "disk": 0.0009,
    "network": 0.0009,
    "gpu": 0.0012,
    "npu": 0.0012,
}
_RESOURCE_CTL_BUDGET_QUEUE_TAX: dict[str, float] = {
    "cpu": 0.02,
    "ram": 0.012,
    "disk": 0.018,
    "network": 0.022,
    "gpu": 0.008,
    "npu": 0.008,
}
_RESOURCE_CTL_BUDGET_LOCK = threading.Lock()
_RESOURCE_CTL_BUDGET_STATE: dict[str, Any] = {
    "last_monotonic": time.monotonic(),
    "budget": dict(_RESOURCE_CTL_BUDGET_CAP_DEFAULT),
}
_RESOURCE_DAIMOI_CPU_SENTINEL_ID = "health_sentinel_cpu"
_RESOURCE_DAIMOI_SENTINEL_RESOURCE_BY_ID: dict[str, str] = {
    "health_sentinel_cpu": "cpu",
    "health_sentinel_gpu1": "gpu",
    "health_sentinel_gpu2": "gpu",
    "health_sentinel_npu0": "npu",
}
_RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT = max(
    0.0,
    min(
        100.0,
        _safe_float(
            os.getenv("SIMULATION_CPU_SENTINEL_BURN_START_PERCENT", "80") or "80",
            80.0,
        ),
    ),
)
_RESOURCE_DAIMOI_CPU_SENTINEL_BURN_MAX_MULTIPLIER = max(
    1.0,
    min(
        128.0,
        _safe_float(
            os.getenv("SIMULATION_CPU_SENTINEL_BURN_MAX_MULTIPLIER", "12.0") or "12.0",
            12.0,
        ),
    ),
)
_RESOURCE_DAIMOI_CPU_SENTINEL_BURN_COST_MAX = max(
    _RESOURCE_DAIMOI_ACTION_COST_MAX,
    min(
        4.0,
        _safe_float(
            os.getenv("SIMULATION_CPU_SENTINEL_BURN_COST_MAX", "0.4") or "0.4",
            0.4,
        ),
    ),
)
_RESOURCE_DAIMOI_CPU_SENTINEL_ATTRACTOR_START_PERCENT = max(
    0.0,
    min(
        100.0,
        _safe_float(
            os.getenv(
                "SIMULATION_CPU_SENTINEL_ATTRACTOR_START_PERCENT",
                str(_RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT),
            )
            or str(_RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT),
            _RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT,
        ),
    ),
)
_RESOURCE_DAIMOI_CPU_SENTINEL_ATTRACTOR_GAIN = max(
    0.0,
    min(
        8.0,
        _safe_float(
            os.getenv("SIMULATION_CPU_SENTINEL_ATTRACTOR_GAIN", "1.8") or "1.8",
            1.8,
        ),
    ),
)
_RESOURCE_DAIMOI_CPU_SENTINEL_ATTRACTOR_RESOURCE_BOOST = max(
    1.0,
    min(
        24.0,
        _safe_float(
            os.getenv("SIMULATION_CPU_SENTINEL_ATTRACTOR_RESOURCE_BOOST", "4.0")
            or "4.0",
            4.0,
        ),
    ),
)
_RESOURCE_DAIMOI_CPU_SENTINEL_ATTRACTOR_ALL_DAIMOI = str(
    os.getenv("SIMULATION_CPU_SENTINEL_ATTRACTOR_ALL_DAIMOI", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}


def _canonical_resource_type(resource_type: str) -> str:
    key = str(resource_type or "").strip().lower()
    if not key:
        return ""
    if key in _RESOURCE_DAIMOI_TYPES:
        return key
    return str(_RESOURCE_DAIMOI_TYPE_ALIASES.get(key, "")).strip().lower()


def _core_resource_type_from_presence_id(presence_id: str) -> str:
    pid = str(presence_id or "").strip().lower()
    prefix = "presence.core."
    if not pid.startswith(prefix):
        return ""
    return _canonical_resource_type(pid[len(prefix) :])


def _normalize_resource_wallet(
    impact: dict[str, Any],
) -> dict[str, float]:
    wallet_raw = impact.get("resource_wallet", {})
    wallet: dict[str, float] = {}
    if isinstance(wallet_raw, dict):
        for key, value in wallet_raw.items():
            name_raw = str(key or "").strip().lower()
            if not name_raw:
                continue
            amount = max(0.0, _safe_float(value, 0.0))
            wallet[name_raw] = amount
            canonical = _canonical_resource_type(name_raw)
            if canonical:
                wallet[canonical] = max(amount, wallet.get(canonical, 0.0))
    impact["resource_wallet"] = wallet
    return wallet


def _resource_vector_normalized(raw_value: Any) -> dict[str, float]:
    vector: dict[str, float] = {}
    if isinstance(raw_value, dict):
        for key, value in raw_value.items():
            resource_name = _canonical_resource_type(str(key or ""))
            if not resource_name:
                continue
            amount = max(0.0, _safe_float(value, 0.0))
            if amount <= 1e-12:
                continue
            vector[resource_name] = vector.get(resource_name, 0.0) + amount
    return vector


def _resource_vector_total(vector: dict[str, float]) -> float:
    return sum(max(0.0, _safe_float(value, 0.0)) for value in vector.values())


def _resource_vector_quantized(vector: dict[str, float]) -> dict[str, float]:
    quantum = max(1e-9, _safe_float(_RESOURCE_DAIMOI_DENOM_QUANTUM, 0.000001))
    quantized: dict[str, float] = {}
    for resource_name in _RESOURCE_DAIMOI_TYPES:
        value = max(0.0, _safe_float(vector.get(resource_name, 0.0), 0.0))
        if value <= 1e-12:
            continue
        snapped = round(round(value / quantum) * quantum, 6)
        if snapped <= 1e-12:
            continue
        quantized[resource_name] = snapped
    return quantized


def _normalize_resource_wallet_denoms(
    impact: dict[str, Any],
) -> list[dict[str, Any]]:
    denoms_raw = impact.get("resource_wallet_denoms", [])
    normalized: list[dict[str, Any]] = []
    if isinstance(denoms_raw, list):
        for row in denoms_raw:
            if not isinstance(row, dict):
                continue
            vector = _resource_vector_quantized(
                _resource_vector_normalized(row.get("vector", {}))
            )
            if _resource_vector_total(vector) <= 1e-12:
                continue
            count = max(0, int(_safe_float(row.get("count", 0.0), 0.0)))
            if count <= 0:
                continue
            normalized.append({"vector": vector, "count": count})

    impact["resource_wallet_denoms"] = normalized
    return normalized


def _wallet_denoms_add_vector(
    denoms: list[dict[str, Any]],
    vector: dict[str, float],
) -> None:
    quantized = _resource_vector_quantized(vector)
    if _resource_vector_total(quantized) <= 1e-12:
        return
    for bucket in denoms:
        if not isinstance(bucket, dict):
            continue
        existing = _resource_vector_quantized(
            _resource_vector_normalized(bucket.get("vector", {}))
        )
        if existing == quantized:
            count = max(0, int(_safe_float(bucket.get("count", 0.0), 0.0)))
            bucket["count"] = count + 1
            bucket["vector"] = existing
            return
    denoms.append({"vector": quantized, "count": 1})


def _resource_required_payment_vector(
    *,
    focus_resource: str,
    desired_cost: float,
    pressure: dict[str, float],
) -> dict[str, float]:
    desired = max(0.0, _safe_float(desired_cost, 0.0))
    if desired <= 1e-12:
        return {}

    weights = _resource_mixing_weights(focus_resource, pressure=pressure)
    total_weight = sum(max(0.0, _safe_float(value, 0.0)) for value in weights.values())
    if total_weight <= 1e-12:
        focus = _canonical_resource_type(focus_resource) or "cpu"
        return {focus: desired}

    vector: dict[str, float] = {}
    for resource_name, value in sorted(weights.items()):
        weight = max(0.0, _safe_float(value, 0.0))
        if weight <= 1e-12:
            continue
        share = desired * (weight / total_weight)
        if share <= 1e-12:
            continue
        vector[resource_name] = share
    return _resource_vector_quantized(vector)


def _wallet_denoms_payment_plan(
    *,
    denoms: list[dict[str, Any]],
    required_vector: dict[str, float],
) -> dict[str, Any]:
    required = _resource_vector_quantized(required_vector)
    if _resource_vector_total(required) <= 1e-12:
        return {
            "affordable": True,
            "spent_vector": {},
            "selected": {},
            "required_vector": required,
            "remaining_vector": {},
            "overpay": 0.0,
        }

    remaining = {k: max(0.0, _safe_float(v, 0.0)) for k, v in required.items()}
    spent_vector: dict[str, float] = {
        resource_name: 0.0 for resource_name in _RESOURCE_DAIMOI_TYPES
    }
    selected_by_index: dict[int, int] = {}
    epsilon = max(1e-9, _safe_float(_RESOURCE_DAIMOI_DENOM_QUANTUM, 0.000001) * 0.5)
    overpay_penalty = max(
        0.0, _safe_float(_RESOURCE_DAIMOI_DENOM_OVERPAY_PENALTY, 0.18)
    )
    step_limit = max(
        8, int(_safe_float(_RESOURCE_DAIMOI_DENOM_GREEDY_STEP_LIMIT, 256.0))
    )
    steps = 0

    while steps < step_limit and any(value > epsilon for value in remaining.values()):
        steps += 1
        best_idx = -1
        best_score = -1.0
        best_total = 0.0
        best_vector: dict[str, float] = {}

        for idx, bucket in enumerate(denoms):
            if not isinstance(bucket, dict):
                continue
            count = max(0, int(_safe_float(bucket.get("count", 0.0), 0.0)))
            used_count = max(0, int(selected_by_index.get(idx, 0)))
            if used_count >= count:
                continue

            vector = _resource_vector_quantized(
                _resource_vector_normalized(bucket.get("vector", {}))
            )
            if _resource_vector_total(vector) <= 1e-12:
                continue

            useful_cover = 0.0
            overpay = 0.0
            total = 0.0
            for resource_name in _RESOURCE_DAIMOI_TYPES:
                amount = max(0.0, _safe_float(vector.get(resource_name, 0.0), 0.0))
                if amount <= 1e-12:
                    continue
                need = max(0.0, _safe_float(remaining.get(resource_name, 0.0), 0.0))
                useful_cover += min(need, amount)
                overpay += max(0.0, amount - need)
                total += amount

            score = useful_cover / max(1e-9, 1.0 + (overpay * overpay_penalty))
            if score <= 1e-12:
                continue
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12 and total < best_total
            ):
                best_idx = idx
                best_score = score
                best_total = total
                best_vector = vector

        if best_idx < 0 or _resource_vector_total(best_vector) <= 1e-12:
            break

        selected_by_index[best_idx] = selected_by_index.get(best_idx, 0) + 1
        for resource_name in _RESOURCE_DAIMOI_TYPES:
            amount = max(0.0, _safe_float(best_vector.get(resource_name, 0.0), 0.0))
            if amount <= 1e-12:
                continue
            spent_vector[resource_name] = spent_vector.get(resource_name, 0.0) + amount
            remaining[resource_name] = max(
                0.0,
                _safe_float(remaining.get(resource_name, 0.0), 0.0) - amount,
            )

    affordable = all(value <= epsilon for value in remaining.values())
    spent_quantized = _resource_vector_quantized(spent_vector)
    overpay = max(
        0.0,
        _resource_vector_total(spent_quantized) - _resource_vector_total(required),
    )
    return {
        "affordable": bool(affordable),
        "spent_vector": spent_quantized if affordable else {},
        "selected": selected_by_index if affordable else {},
        "required_vector": required,
        "remaining_vector": _resource_vector_quantized(remaining),
        "overpay": overpay if affordable else 0.0,
    }


def _presence_anchor_position(
    presence_id: str,
    impact: dict[str, Any],
    *,
    manifest_by_id: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    x_value = impact.get("x")
    y_value = impact.get("y")
    if x_value is not None and y_value is not None:
        return (
            _clamp01(_safe_float(x_value, 0.5)),
            _clamp01(_safe_float(y_value, 0.5)),
        )

    meta = manifest_by_id.get(presence_id, {})
    if isinstance(meta, dict):
        return (
            _clamp01(
                _safe_float(
                    meta.get("x", _stable_ratio(f"{presence_id}|anchor", 3)),
                    _stable_ratio(f"{presence_id}|anchor", 3),
                )
            ),
            _clamp01(
                _safe_float(
                    meta.get("y", _stable_ratio(f"{presence_id}|anchor", 11)),
                    _stable_ratio(f"{presence_id}|anchor", 11),
                )
            ),
        )

    return (
        _clamp01(_stable_ratio(f"{presence_id}|anchor", 3)),
        _clamp01(_stable_ratio(f"{presence_id}|anchor", 11)),
    )


def _resource_availability_ratio(
    resource_type: str,
    resource_heartbeat: dict[str, Any],
) -> float:
    usage_clamped = _resource_usage_percent(resource_type, resource_heartbeat)
    return _clamp01((100.0 - usage_clamped) / 100.0)


def _resource_usage_percent(
    resource_type: str,
    resource_heartbeat: dict[str, Any],
) -> float:
    kind = _canonical_resource_type(resource_type)
    if not kind:
        return 100.0

    devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    if not isinstance(devices, dict):
        devices = {}
    monitor = (
        resource_heartbeat.get("resource_monitor", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    if not isinstance(monitor, dict):
        monitor = {}

    usage_percent = 100.0
    if kind == "cpu":
        usage_percent = _safe_float(
            (
                devices.get("cpu", {}) if isinstance(devices.get("cpu"), dict) else {}
            ).get("utilization", monitor.get("cpu_percent", 100.0)),
            _safe_float(monitor.get("cpu_percent", 100.0), 100.0),
        )
    elif kind == "ram":
        usage_percent = _safe_float(monitor.get("memory_percent", 100.0), 100.0)
    elif kind == "disk":
        usage_percent = _safe_float(monitor.get("disk_percent", 100.0), 100.0)
    elif kind == "network":
        usage_percent = _safe_float(monitor.get("network_percent", 100.0), 100.0)
    elif kind == "gpu":
        gpu1 = _safe_float(
            (
                devices.get("gpu1", {}) if isinstance(devices.get("gpu1"), dict) else {}
            ).get("utilization", 100.0),
            100.0,
        )
        gpu2 = _safe_float(
            (
                devices.get("gpu2", {}) if isinstance(devices.get("gpu2"), dict) else {}
            ).get("utilization", 100.0),
            100.0,
        )
        usage_percent = min(gpu1, gpu2)
    elif kind == "npu":
        usage_percent = _safe_float(
            (
                devices.get("npu0", {}) if isinstance(devices.get("npu0"), dict) else {}
            ).get("utilization", 100.0),
            100.0,
        )

    return max(0.0, min(100.0, usage_percent))


def _resource_pressure_thresholds(resource_type: str) -> tuple[float, float]:
    kind = _canonical_resource_type(resource_type)
    if not kind:
        kind = "cpu"

    if kind == "cpu":
        soft_default = max(
            0.0,
            min(
                100.0,
                _safe_float(
                    os.getenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "50") or "50",
                    50.0,
                ),
            ),
        )
        hard_default = max(
            0.0,
            min(
                100.0,
                _safe_float(
                    os.getenv("SIMULATION_CPU_SENTINEL_BURN_START_PERCENT", "80")
                    or "80",
                    80.0,
                ),
            ),
        )
    else:
        soft_default = max(
            0.0,
            min(
                100.0,
                _safe_float(
                    os.getenv(
                        "SIMULATION_RESOURCE_PRESSURE_SOFT_PERCENT",
                        str(_RESOURCE_DAIMOI_PRESSURE_SOFT_PERCENT_DEFAULT),
                    )
                    or str(_RESOURCE_DAIMOI_PRESSURE_SOFT_PERCENT_DEFAULT),
                    _RESOURCE_DAIMOI_PRESSURE_SOFT_PERCENT_DEFAULT,
                ),
            ),
        )
        hard_default = max(
            0.0,
            min(
                100.0,
                _safe_float(
                    os.getenv(
                        "SIMULATION_RESOURCE_PRESSURE_HARD_PERCENT",
                        str(_RESOURCE_DAIMOI_PRESSURE_HARD_PERCENT_DEFAULT),
                    )
                    or str(_RESOURCE_DAIMOI_PRESSURE_HARD_PERCENT_DEFAULT),
                    _RESOURCE_DAIMOI_PRESSURE_HARD_PERCENT_DEFAULT,
                ),
            ),
        )

    key = str(kind or "cpu").upper()
    soft = max(
        0.0,
        min(
            100.0,
            _safe_float(
                os.getenv(f"SIMULATION_{key}_PRESSURE_SOFT_PERCENT", str(soft_default))
                or str(soft_default),
                soft_default,
            ),
        ),
    )
    hard = max(
        0.0,
        min(
            100.0,
            _safe_float(
                os.getenv(f"SIMULATION_{key}_PRESSURE_HARD_PERCENT", str(hard_default))
                or str(hard_default),
                hard_default,
            ),
        ),
    )
    if hard <= soft:
        hard = min(100.0, soft + 1.0)
        if hard <= soft:
            soft = max(0.0, hard - 1.0)
    return soft, hard


def _resource_pressure_ratio(
    usage_percent: float,
    *,
    soft_percent: float,
    hard_percent: float,
) -> float:
    usage = max(0.0, min(100.0, _safe_float(usage_percent, 0.0)))
    soft = max(0.0, min(100.0, _safe_float(soft_percent, 50.0)))
    hard = max(0.0, min(100.0, _safe_float(hard_percent, 80.0)))
    if hard <= soft:
        hard = min(100.0, soft + 1.0)
    return _clamp01((usage - soft) / max(1.0, hard - soft))


def _resource_debt_vector_update(
    resource_heartbeat: dict[str, Any],
) -> dict[str, float]:
    global _RESOURCE_DAIMOI_DEBT_LAST_MONOTONIC
    now = time.monotonic()
    with _RESOURCE_DAIMOI_DEBT_LOCK:
        dt_seconds = max(
            0.001,
            min(2.0, now - _safe_float(_RESOURCE_DAIMOI_DEBT_LAST_MONOTONIC, now)),
        )
        _RESOURCE_DAIMOI_DEBT_LAST_MONOTONIC = now

        updated: dict[str, float] = {}
        for resource_type in _RESOURCE_DAIMOI_TYPES:
            usage = _resource_usage_percent(resource_type, resource_heartbeat)
            soft, hard = _resource_pressure_thresholds(resource_type)
            overload = max(0.0, usage - soft)
            overload_norm = overload / max(1.0, hard - soft)
            next_value = max(
                0.0,
                _safe_float(_RESOURCE_DAIMOI_DEBT_STATE.get(resource_type, 0.0), 0.0)
                + (dt_seconds * overload_norm),
            )
            _RESOURCE_DAIMOI_DEBT_STATE[resource_type] = next_value
            updated[resource_type] = next_value
        return updated


def _resource_debt_snapshot() -> dict[str, float]:
    with _RESOURCE_DAIMOI_DEBT_LOCK:
        return {
            resource_type: max(
                0.0,
                _safe_float(_RESOURCE_DAIMOI_DEBT_STATE.get(resource_type, 0.0), 0.0),
            )
            for resource_type in _RESOURCE_DAIMOI_TYPES
        }


def _resource_velocity_vector(
    resource_heartbeat: dict[str, Any],
    *,
    queue_ratio: float,
) -> dict[str, float]:
    now = time.monotonic()
    queue_push = _clamp01(_safe_float(queue_ratio, 0.0))
    with _RESOURCE_DAIMOI_VELOCITY_LOCK:
        last = _safe_float(
            _RESOURCE_DAIMOI_VELOCITY_STATE.get("last_monotonic", now),
            now,
        )
        dt_seconds = max(0.001, min(2.0, now - last))
        _RESOURCE_DAIMOI_VELOCITY_STATE["last_monotonic"] = now

        previous_raw = _RESOURCE_DAIMOI_VELOCITY_STATE.get("usage_prev", {})
        previous_usage = previous_raw if isinstance(previous_raw, dict) else {}
        next_usage: dict[str, float] = {}
        velocity: dict[str, float] = {}

        for resource_type in _RESOURCE_DAIMOI_TYPES:
            usage_now = _resource_usage_percent(resource_type, resource_heartbeat)
            usage_prev = _safe_float(
                previous_usage.get(resource_type, usage_now), usage_now
            )
            delta_per_second = abs(usage_now - usage_prev) / max(
                1.0, dt_seconds * 100.0
            )
            queue_weight = 0.58 if resource_type in {"cpu", "disk", "network"} else 0.34
            velocity_signal = _clamp01(
                (delta_per_second * 0.78) + (queue_push * queue_weight)
            )
            velocity[resource_type] = velocity_signal
            next_usage[resource_type] = usage_now

        _RESOURCE_DAIMOI_VELOCITY_STATE["usage_prev"] = next_usage
        return velocity


def _resource_emission_rate(
    *,
    resource_type: str,
    pressure_signal: float,
    debt_value: float,
    velocity_signal: float,
) -> float:
    resource_name = _canonical_resource_type(resource_type)
    if not resource_name:
        resource_name = "cpu"

    debt_norm = _clamp01(
        max(0.0, _safe_float(debt_value, 0.0))
        / max(1.0, max(0.0, _safe_float(debt_value, 0.0)) + 1.0)
    )
    pressure = _clamp01(_safe_float(pressure_signal, 0.0))
    velocity = _clamp01(_safe_float(velocity_signal, 0.0))

    base_rate = max(
        0.0,
        min(
            0.2,
            _safe_float(
                os.getenv(
                    f"SIMULATION_{resource_name.upper()}_DAIMOI_EMIT_BASE_RATE",
                    str(_RESOURCE_DAIMOI_EMIT_BASE_RATE),
                )
                or str(_RESOURCE_DAIMOI_EMIT_BASE_RATE),
                _RESOURCE_DAIMOI_EMIT_BASE_RATE,
            ),
        ),
    )
    alpha = max(
        0.0,
        min(
            0.8,
            _safe_float(
                os.getenv(
                    f"SIMULATION_{resource_name.upper()}_DAIMOI_EMIT_ALPHA",
                    str(_RESOURCE_DAIMOI_EMIT_ALPHA),
                )
                or str(_RESOURCE_DAIMOI_EMIT_ALPHA),
                _RESOURCE_DAIMOI_EMIT_ALPHA,
            ),
        ),
    )
    beta = max(
        0.0,
        min(
            0.8,
            _safe_float(
                os.getenv(
                    f"SIMULATION_{resource_name.upper()}_DAIMOI_EMIT_BETA",
                    str(_RESOURCE_DAIMOI_EMIT_BETA),
                )
                or str(_RESOURCE_DAIMOI_EMIT_BETA),
                _RESOURCE_DAIMOI_EMIT_BETA,
            ),
        ),
    )
    gamma = max(
        0.0,
        min(
            0.8,
            _safe_float(
                os.getenv(
                    f"SIMULATION_{resource_name.upper()}_DAIMOI_EMIT_GAMMA",
                    str(_RESOURCE_DAIMOI_EMIT_GAMMA),
                )
                or str(_RESOURCE_DAIMOI_EMIT_GAMMA),
                _RESOURCE_DAIMOI_EMIT_GAMMA,
            ),
        ),
    )

    return max(
        0.0, base_rate + (alpha * pressure) + (beta * debt_norm) + (gamma * velocity)
    )


def _resource_ctl_budget_cap_vector() -> dict[str, float]:
    global_cap = max(
        0.05,
        min(
            32.0,
            _safe_float(
                os.getenv("SIMULATION_RESOURCE_CTL_BUDGET_CAP", "1.0") or "1.0",
                1.0,
            ),
        ),
    )
    caps: dict[str, float] = {}
    for resource_name in _RESOURCE_DAIMOI_TYPES:
        default_cap = max(
            0.01,
            _safe_float(_RESOURCE_CTL_BUDGET_CAP_DEFAULT.get(resource_name, 1.0), 1.0),
        )
        cap = max(
            0.01,
            min(
                32.0,
                _safe_float(
                    os.getenv(
                        f"SIMULATION_RESOURCE_CTL_BUDGET_CAP_{resource_name.upper()}",
                        str(default_cap),
                    )
                    or str(default_cap),
                    default_cap,
                ),
            ),
        )
        caps[resource_name] = cap * global_cap
    return caps


def _resource_ctl_budget_recharge_vector(caps: dict[str, float]) -> dict[str, float]:
    global_recharge = max(
        0.0,
        min(
            8.0,
            _safe_float(
                os.getenv("SIMULATION_RESOURCE_CTL_BUDGET_RECHARGE", "1.0") or "1.0",
                1.0,
            ),
        ),
    )
    recharge: dict[str, float] = {}
    for resource_name in _RESOURCE_DAIMOI_TYPES:
        default_rate = max(
            0.0,
            _safe_float(
                _RESOURCE_CTL_BUDGET_RECHARGE_DEFAULT.get(resource_name, 0.2), 0.2
            ),
        )
        rate = max(
            0.0,
            min(
                32.0,
                _safe_float(
                    os.getenv(
                        f"SIMULATION_RESOURCE_CTL_BUDGET_RECHARGE_{resource_name.upper()}",
                        str(default_rate),
                    )
                    or str(default_rate),
                    default_rate,
                ),
            ),
        )
        recharge[resource_name] = rate * global_recharge
    return recharge


def _resource_ctl_budget_prepare(
    *,
    queue_push: float,
    candidate_count: int,
) -> dict[str, Any]:
    now = time.monotonic()
    caps = _resource_ctl_budget_cap_vector()
    recharge = _resource_ctl_budget_recharge_vector(caps)
    queue_signal = _clamp01(_safe_float(queue_push, 0.0))

    with _RESOURCE_CTL_BUDGET_LOCK:
        last = _safe_float(_RESOURCE_CTL_BUDGET_STATE.get("last_monotonic", now), now)
        dt_seconds = max(0.001, min(2.0, now - last))
        _RESOURCE_CTL_BUDGET_STATE["last_monotonic"] = now

        budget_raw = _RESOURCE_CTL_BUDGET_STATE.get("budget", {})
        budget = budget_raw if isinstance(budget_raw, dict) else {}
        next_budget: dict[str, float] = {}
        for resource_name in _RESOURCE_DAIMOI_TYPES:
            current = max(
                0.0,
                _safe_float(
                    budget.get(resource_name, caps.get(resource_name, 1.0)),
                    caps.get(resource_name, 1.0),
                ),
            )
            replenished = current + (
                max(0.0, _safe_float(recharge.get(resource_name, 0.0), 0.0))
                * dt_seconds
            )
            tax = queue_signal * max(
                0.0,
                _safe_float(
                    _RESOURCE_CTL_BUDGET_QUEUE_TAX.get(resource_name, 0.0), 0.0
                ),
            )
            next_budget[resource_name] = max(
                0.0,
                min(_safe_float(caps.get(resource_name, 1.0), 1.0), replenished - tax),
            )
        _RESOURCE_CTL_BUDGET_STATE["budget"] = dict(next_budget)

    ratio_values = [
        _clamp01(
            _safe_float(next_budget.get(resource_name, 0.0), 0.0)
            / max(1e-6, _safe_float(caps.get(resource_name, 1.0), 1.0))
        )
        for resource_name in _RESOURCE_DAIMOI_TYPES
    ]
    budget_ratio = min(ratio_values) if ratio_values else 0.0

    # Scale scheduling by live candidate density so emitter/burn thresholds remain
    # the primary regulation gates instead of a fixed global row cap.
    if budget_ratio < 0.08:
        mode = "minimal"
        max_actions = max(12, int(round(candidate_count * 0.08)))
        allow_denom = False
    elif budget_ratio < 0.16:
        mode = "reduced"
        max_actions = max(24, int(round(candidate_count * 0.2)))
        allow_denom = False
    elif budget_ratio < 0.3:
        mode = "moderate"
        max_actions = max(48, int(round(candidate_count * 0.55)))
        allow_denom = True
    else:
        mode = "full"
        max_actions = max(96, int(round(candidate_count * 0.95)))
        allow_denom = True

    max_actions = max(1, min(max_actions, max(1, int(candidate_count))))
    return {
        "cap": {k: round(_safe_float(v, 0.0), 6) for k, v in sorted(caps.items())},
        "before": {
            k: round(_safe_float(v, 0.0), 6) for k, v in sorted(next_budget.items())
        },
        "mode": mode,
        "ratio": round(_clamp01(budget_ratio), 6),
        "max_actions": int(max_actions),
        "allow_denom": bool(allow_denom),
    }


def _resource_ctl_budget_commit(overhead_vector: dict[str, float]) -> dict[str, float]:
    with _RESOURCE_CTL_BUDGET_LOCK:
        budget_raw = _RESOURCE_CTL_BUDGET_STATE.get("budget", {})
        budget = budget_raw if isinstance(budget_raw, dict) else {}
        updated: dict[str, float] = {}
        for resource_name in _RESOURCE_DAIMOI_TYPES:
            current = max(0.0, _safe_float(budget.get(resource_name, 0.0), 0.0))
            spent = max(0.0, _safe_float(overhead_vector.get(resource_name, 0.0), 0.0))
            updated[resource_name] = max(0.0, current - spent)
        _RESOURCE_CTL_BUDGET_STATE["budget"] = dict(updated)
        return updated


def _resource_need_ratio(
    impact: dict[str, Any],
    resource_type: str,
    *,
    queue_ratio: float,
) -> float:
    kind = _canonical_resource_type(resource_type)
    if not kind:
        return 0.0
    affected_by = impact.get("affected_by", {})
    if not isinstance(affected_by, dict):
        affected_by = {}

    wallet = _normalize_resource_wallet(impact)
    balance = max(0.0, _safe_float(wallet.get(kind, 0.0), 0.0))
    floor = max(0.1, _safe_float(_RESOURCE_DAIMOI_WALLET_FLOOR.get(kind, 4.0), 4.0))
    deficit_ratio = _clamp01((floor - balance) / floor)
    base_need = _clamp01(_safe_float(affected_by.get("resource", 0.0), 0.0))
    queue_push = _clamp01(_safe_float(queue_ratio, 0.0))
    is_sub_sim = bool(
        str(impact.get("presence_type", "")).strip() == "sub-sim"
        or str(impact.get("id", "")).strip().startswith("presence.sim.")
    )
    sub_sim_boost = 0.22 if is_sub_sim else 0.0
    return _clamp01(
        (base_need * 0.34)
        + (deficit_ratio * 0.46)
        + (queue_push * 0.18)
        + sub_sim_boost
    )


def _resource_pressure_vector(resource_heartbeat: dict[str, Any]) -> dict[str, float]:
    pressures: dict[str, float] = {}
    for resource_type in _RESOURCE_DAIMOI_TYPES:
        usage = _resource_usage_percent(resource_type, resource_heartbeat)
        soft, hard = _resource_pressure_thresholds(resource_type)
        pressures[resource_type] = _resource_pressure_ratio(
            usage,
            soft_percent=soft,
            hard_percent=hard,
        )
    return pressures


def _resource_mixing_weights(
    focus_resource: str,
    *,
    pressure: dict[str, float] | None = None,
) -> dict[str, float]:
    focus = _canonical_resource_type(focus_resource)
    if not focus:
        focus = "cpu"

    pressure_map = pressure if isinstance(pressure, dict) else {}
    mix_epsilon = max(
        0.001,
        min(
            0.45,
            _safe_float(
                os.getenv(
                    "SIMULATION_RESOURCE_DAIMOI_MIX_EPSILON",
                    str(_RESOURCE_DAIMOI_MIX_EPSILON_BASE),
                )
                or str(_RESOURCE_DAIMOI_MIX_EPSILON_BASE),
                _RESOURCE_DAIMOI_MIX_EPSILON_BASE,
            ),
        ),
    )
    pressure_gain = max(
        0.0,
        min(
            1.0,
            _safe_float(
                os.getenv(
                    "SIMULATION_RESOURCE_DAIMOI_MIX_PRESSURE_GAIN",
                    str(_RESOURCE_DAIMOI_MIX_PRESSURE_GAIN),
                )
                or str(_RESOURCE_DAIMOI_MIX_PRESSURE_GAIN),
                _RESOURCE_DAIMOI_MIX_PRESSURE_GAIN,
            ),
        ),
    )
    focus_pressure = _clamp01(_safe_float(pressure_map.get(focus, 0.0), 0.0))
    coupled_weight = max(
        0.001,
        min(0.45, mix_epsilon * (1.0 + (math.sqrt(focus_pressure) * pressure_gain))),
    )

    weights: dict[str, float] = {}
    for resource_type in _RESOURCE_DAIMOI_TYPES:
        resource_pressure = _clamp01(
            _safe_float(pressure_map.get(resource_type, 0.0), 0.0)
        )
        value = (
            1.0
            if resource_type == focus
            else (coupled_weight * (0.5 + (resource_pressure * 0.5)))
        )
        if value > 1e-9:
            weights[resource_type] = value

    # Keep direct spend path available even if matrix is misconfigured.
    weights[focus] = max(1.0, _safe_float(weights.get(focus, 0.0), 0.0))
    return weights


def _resource_payment_plan(
    *,
    wallet: dict[str, float],
    focus_resource: str,
    desired_cost: float,
    pressure: dict[str, float],
    prefer_high_pressure: bool,
) -> dict[str, Any]:
    desired = max(0.0, _safe_float(desired_cost, 0.0))
    if desired <= 1e-12:
        return {
            "desired": desired,
            "effective_credit": 0.0,
            "affordable_credit": 0.0,
            "debt": 0.0,
            "spent_total": 0.0,
            "breakdown": {},
            "affordability": 1.0,
        }

    weights = _resource_mixing_weights(focus_resource, pressure=pressure)
    balances: dict[str, float] = {
        resource_type: max(0.0, _safe_float(wallet.get(resource_type, 0.0), 0.0))
        for resource_type in _RESOURCE_DAIMOI_TYPES
    }

    affordable_credit = 0.0
    ranked_sources: list[tuple[float, str, float, float]] = []
    for resource_type, mix_weight in weights.items():
        unit_credit = max(0.0, _safe_float(mix_weight, 0.0))
        if unit_credit <= 1e-9:
            continue
        balance = balances.get(resource_type, 0.0)
        if balance <= 1e-12:
            continue
        resource_pressure = _clamp01(_safe_float(pressure.get(resource_type, 1.0), 1.0))
        if prefer_high_pressure:
            priority = unit_credit * (0.35 + (resource_pressure * 1.05))
        else:
            priority = unit_credit * (1.2 - (resource_pressure * 0.65))
        ranked_sources.append((priority, resource_type, unit_credit, balance))
        affordable_credit += balance * unit_credit

    ranked_sources.sort(key=lambda row: (-row[0], row[1]))
    target_credit = min(desired, affordable_credit)
    remaining_credit = target_credit
    spent_by_resource: dict[str, float] = {}
    effective_credit = 0.0

    for _, resource_type, unit_credit, balance in ranked_sources:
        if remaining_credit <= 1e-12:
            break
        max_credit = balance * unit_credit
        if max_credit <= 1e-12:
            continue
        taken_credit = min(remaining_credit, max_credit)
        spent_amount = taken_credit / max(1e-9, unit_credit)
        if spent_amount <= 1e-12:
            continue
        spent_by_resource[resource_type] = (
            spent_by_resource.get(resource_type, 0.0) + spent_amount
        )
        effective_credit += taken_credit
        remaining_credit -= taken_credit

    spent_total = sum(spent_by_resource.values())
    debt = max(0.0, desired - effective_credit)
    affordability = _clamp01(effective_credit / max(1e-9, desired))
    return {
        "desired": desired,
        "effective_credit": effective_credit,
        "affordable_credit": affordable_credit,
        "debt": debt,
        "spent_total": spent_total,
        "breakdown": spent_by_resource,
        "affordability": affordability,
    }


def _resource_action_contract_estimate(
    *,
    row: dict[str, Any],
    presence_id: str,
    resource_pressure: dict[str, float],
    resource_debt: dict[str, float],
    queue_push: float,
    sentinel_usage_by_presence: dict[str, float] | None = None,
    cpu_sentinel_burn_threshold: float | None = None,
) -> dict[str, Any]:
    sentinel_resource = _RESOURCE_DAIMOI_SENTINEL_RESOURCE_BY_ID.get(presence_id, "")
    is_resource_sentinel = bool(sentinel_resource)
    focus_resource = sentinel_resource if is_resource_sentinel else "cpu"
    threshold = (
        _safe_float(
            cpu_sentinel_burn_threshold,
            _RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT,
        )
        if cpu_sentinel_burn_threshold is not None
        else _RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT
    )
    sentinel_usage_map = (
        sentinel_usage_by_presence
        if isinstance(sentinel_usage_by_presence, dict)
        else {}
    )
    sentinel_usage = max(
        0.0, _safe_float(sentinel_usage_map.get(presence_id, 0.0), 0.0)
    )

    focus_pressure = _clamp01(
        _safe_float(resource_pressure.get(focus_resource, 0.0), 0.0)
    )
    focus_debt = max(0.0, _safe_float(resource_debt.get(focus_resource, 0.0), 0.0))
    focus_debt_norm = _clamp01(focus_debt / max(1.0, focus_debt + 1.0))

    influence_power = _clamp01(
        _safe_float(
            row.get("influence_power", row.get("message_probability", 0.0)), 0.0
        )
    )
    message_probability = _clamp01(
        _safe_float(row.get("message_probability", 0.0), 0.0)
    )
    route_probability = _clamp01(_safe_float(row.get("route_probability", 0.0), 0.0))
    drift_signal = _clamp01(abs(_safe_float(row.get("drift_score", 0.0), 0.0)))

    base_cost = (
        _RESOURCE_DAIMOI_ACTION_BASE_COST
        + (influence_power * 0.00086)
        + (message_probability * 0.00054)
        + (route_probability * 0.00032)
        + (drift_signal * 0.00024)
        + (queue_push * 0.00028)
    )
    base_cost = min(
        _RESOURCE_DAIMOI_ACTION_COST_MAX,
        max(_RESOURCE_DAIMOI_ACTION_BASE_COST, base_cost),
    )

    sentinel_burn_intensity = 0.0
    sentinel_burn_multiplier = 1.0
    if is_resource_sentinel:
        sentinel_burn_intensity = _clamp01(
            (sentinel_usage - threshold) / max(1.0, (100.0 - threshold))
        )
        sentinel_burn_multiplier = 1.0 + (
            sentinel_burn_intensity
            * (_RESOURCE_DAIMOI_CPU_SENTINEL_BURN_MAX_MULTIPLIER - 1.0)
        )
        base_cost = min(
            _RESOURCE_DAIMOI_CPU_SENTINEL_BURN_COST_MAX,
            max(
                _RESOURCE_DAIMOI_ACTION_BASE_COST, base_cost * sentinel_burn_multiplier
            ),
        )

    action_risk = _clamp01(
        (drift_signal * 0.46)
        + ((1.0 - message_probability) * 0.18)
        + (route_probability * 0.22)
        + (queue_push * 0.14)
    )
    risk_multiplier = 1.0 + (
        _RESOURCE_DAIMOI_ACTION_RISK_PREMIUM * action_risk * focus_pressure
    )
    debt_multiplier = 1.0 + (focus_debt_norm * 0.38)
    desired_cost = min(
        _RESOURCE_DAIMOI_CPU_SENTINEL_BURN_COST_MAX
        if is_resource_sentinel
        else _RESOURCE_DAIMOI_ACTION_COST_MAX,
        max(
            _RESOURCE_DAIMOI_ACTION_BASE_COST,
            base_cost * risk_multiplier * debt_multiplier,
        ),
    )

    expected_cost_vector = _resource_required_payment_vector(
        focus_resource=focus_resource,
        desired_cost=desired_cost,
        pressure=resource_pressure,
    )

    risk_component_factor = (
        _RESOURCE_DAIMOI_ACTION_RISK_PREMIUM * action_risk * desired_cost
    )
    payment_vector_raw: dict[str, float] = {}
    for resource_name in _RESOURCE_DAIMOI_TYPES:
        expected_component = max(
            0.0,
            _safe_float(expected_cost_vector.get(resource_name, 0.0), 0.0),
        )
        pressure_component = _clamp01(
            _safe_float(resource_pressure.get(resource_name, 0.0), 0.0)
        )
        risk_component = max(0.0, risk_component_factor * pressure_component)
        total_component = expected_component + risk_component
        if total_component <= 1e-12:
            continue
        payment_vector_raw[resource_name] = total_component
    payment_vector = _resource_vector_quantized(payment_vector_raw)
    if _resource_vector_total(payment_vector) <= 1e-12 and desired_cost > 1e-12:
        payment_vector = _resource_vector_quantized({focus_resource: desired_cost})

    reclaim_estimate = focus_pressure * (
        0.58 + (route_probability * 0.22) + (influence_power * 0.2)
    )
    reclaim_vector = _resource_vector_quantized(
        {focus_resource: max(0.0, desired_cost * reclaim_estimate)}
    )

    reclaim_term = sum(
        (
            _RESOURCE_DAIMOI_ACTION_UTILITY_ETA
            * _clamp01(_safe_float(resource_pressure.get(resource_name, 0.0), 0.0))
            * max(0.0, _safe_float(reclaim_vector.get(resource_name, 0.0), 0.0))
        )
        for resource_name in _RESOURCE_DAIMOI_TYPES
    )
    cost_term = sum(
        (
            _RESOURCE_DAIMOI_ACTION_UTILITY_XI
            * _clamp01(_safe_float(resource_pressure.get(resource_name, 0.0), 0.0))
            * max(0.0, _safe_float(payment_vector.get(resource_name, 0.0), 0.0))
        )
        for resource_name in _RESOURCE_DAIMOI_TYPES
    )
    utility = (
        reclaim_term - cost_term - (_RESOURCE_DAIMOI_ACTION_UTILITY_KAPPA * action_risk)
    )

    return {
        "focus_resource": focus_resource,
        "is_resource_sentinel": is_resource_sentinel,
        "sentinel_usage": sentinel_usage,
        "sentinel_threshold": threshold,
        "sentinel_burn_intensity": sentinel_burn_intensity,
        "sentinel_burn_multiplier": sentinel_burn_multiplier,
        "focus_pressure": focus_pressure,
        "focus_debt": focus_debt,
        "action_risk": action_risk,
        "risk_multiplier": risk_multiplier,
        "debt_multiplier": debt_multiplier,
        "desired_cost": desired_cost,
        "expected_cost_vector": expected_cost_vector,
        "required_payment_vector": payment_vector,
        "reclaim_vector": reclaim_vector,
        "utility": utility,
    }


def _resource_action_utility(
    *,
    row: dict[str, Any],
    presence_id: str,
    resource_pressure: dict[str, float],
    queue_push: float,
    resource_debt: dict[str, float] | None = None,
    sentinel_usage_by_presence: dict[str, float] | None = None,
    cpu_sentinel_burn_threshold: float | None = None,
) -> float:
    contract = _resource_action_contract_estimate(
        row=row,
        presence_id=presence_id,
        resource_pressure=resource_pressure,
        resource_debt=resource_debt if isinstance(resource_debt, dict) else {},
        queue_push=queue_push,
        sentinel_usage_by_presence=sentinel_usage_by_presence,
        cpu_sentinel_burn_threshold=cpu_sentinel_burn_threshold,
    )

    return _safe_float(contract.get("utility", 0.0), 0.0)


def _apply_resource_daimoi_emissions(
    *,
    field_particles: list[dict[str, Any]],
    presence_impacts: list[dict[str, Any]],
    resource_heartbeat: dict[str, Any],
    queue_ratio: float,
) -> dict[str, Any]:
    resource_devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    cpu_utilization = max(
        0.0,
        min(
            100.0,
            _safe_float(
                (
                    resource_devices.get("cpu", {})
                    if isinstance(resource_devices.get("cpu", {}), dict)
                    else {}
                ).get("utilization", 0.0),
                0.0,
            ),
        ),
    )
    cpu_sentinel_attractor_active = (
        cpu_utilization >= _RESOURCE_DAIMOI_CPU_SENTINEL_ATTRACTOR_START_PERCENT
    )
    resource_pressure_by_type = _resource_pressure_vector(resource_heartbeat)
    resource_debt = _resource_debt_vector_update(resource_heartbeat)
    resource_velocity = _resource_velocity_vector(
        resource_heartbeat,
        queue_ratio=queue_ratio,
    )

    summary: dict[str, Any] = {
        "record": "eta-mu.resource-daimoi-flow.v1",
        "schema_version": "resource.daimoi.flow.v1",
        "emitter_rows": 0,
        "delivered_packets": 0,
        "total_transfer": 0.0,
        "by_resource": {},
        "recipients": [],
        "queue_ratio": round(_clamp01(_safe_float(queue_ratio, 0.0)), 6),
        "cpu_utilization": round(cpu_utilization, 2),
        "cpu_sentinel_id": _RESOURCE_DAIMOI_CPU_SENTINEL_ID,
        "cpu_sentinel_attractor_threshold": round(
            _RESOURCE_DAIMOI_CPU_SENTINEL_ATTRACTOR_START_PERCENT,
            2,
        ),
        "cpu_sentinel_attractor_active": bool(cpu_sentinel_attractor_active),
        "cpu_sentinel_forced_packets": 0,
    }
    if not isinstance(field_particles, list) or not isinstance(presence_impacts, list):
        return summary

    manifest_by_id = {
        str(row.get("id", "")).strip(): row
        for row in ENTITY_MANIFEST
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }

    recipient_impacts: list[dict[str, Any]] = []
    fallback_recipients: list[dict[str, Any]] = []
    anchor_by_presence: dict[str, tuple[float, float]] = {}
    impact_by_id: dict[str, dict[str, Any]] = {}
    for impact in presence_impacts:
        if not isinstance(impact, dict):
            continue
        presence_id = str(impact.get("id", "")).strip()
        if not presence_id:
            continue
        _normalize_resource_wallet(impact)
        impact_by_id[presence_id] = impact
        if _core_resource_type_from_presence_id(presence_id):
            continue

        anchor_by_presence[presence_id] = _presence_anchor_position(
            presence_id,
            impact,
            manifest_by_id=manifest_by_id,
        )
        fallback_recipients.append(impact)
        # All presences with anchors are valid recipients
        recipient_impacts.append(impact)

    if not recipient_impacts:
        recipient_impacts = fallback_recipients
    if not recipient_impacts:
        return summary

    cpu_emitter_stop_percent = max(
        0.0,
        min(
            100.0,
            _safe_float(
                os.getenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "50") or "50",
                50.0,
            ),
        ),
    )
    cpu_emitter_cutoff_active = cpu_utilization >= cpu_emitter_stop_percent
    summary["cpu_emitter_stop_percent"] = round(cpu_emitter_stop_percent, 2)
    summary["cpu_emitter_cutoff_active"] = bool(cpu_emitter_cutoff_active)
    summary["resource_pressure"] = {
        resource_type: round(_clamp01(_safe_float(value, 0.0)), 6)
        for resource_type, value in sorted(resource_pressure_by_type.items())
    }
    summary["resource_debt"] = {
        resource_type: round(max(0.0, _safe_float(value, 0.0)), 6)
        for resource_type, value in sorted(resource_debt.items())
    }
    summary["resource_velocity"] = {
        resource_type: round(_clamp01(_safe_float(value, 0.0)), 6)
        for resource_type, value in sorted(resource_velocity.items())
    }

    # Ambient mint for core emitters. This is intentionally slow and pressure-aware:
    # emitters replenish from live headroom/pressure/debt and do not use hard caps.
    core_minted_totals: dict[str, float] = {key: 0.0 for key in _RESOURCE_DAIMOI_TYPES}
    for impact in presence_impacts:
        if not isinstance(impact, dict):
            continue
        presence_id = str(impact.get("id", "")).strip()
        resource_type = _core_resource_type_from_presence_id(presence_id)
        if not resource_type:
            continue
        wallet = _normalize_resource_wallet(impact)
        current = max(0.0, _safe_float(wallet.get(resource_type, 0.0), 0.0))
        availability = _resource_availability_ratio(resource_type, resource_heartbeat)
        pressure_signal = _clamp01(
            _safe_float(resource_pressure_by_type.get(resource_type, 0.0), 0.0)
        )
        debt_signal = _clamp01(
            _safe_float(resource_debt.get(resource_type, 0.0), 0.0)
            / max(1.0, _safe_float(resource_debt.get(resource_type, 0.0), 0.0) + 1.0)
        )
        mint_amount = 0.0
        if not cpu_emitter_cutoff_active:
            mint_amount = (
                0.002
                + (availability * 0.018)
                + (pressure_signal * 0.010)
                + (debt_signal * 0.014)
            ) * max(
                0.2,
                1.0 - (_clamp01(_safe_float(queue_ratio, 0.0)) * 0.55),
            )
        next_balance = current + max(0.0, mint_amount)
        minted = max(0.0, next_balance - current)
        wallet[resource_type] = round(next_balance, 6)
        impact["resource_wallet"] = wallet
        if minted > 1e-8:
            core_minted_totals[resource_type] = (
                core_minted_totals.get(resource_type, 0.0) + minted
            )
    summary["core_minted"] = {
        key: round(value, 6)
        for key, value in sorted(core_minted_totals.items())
        if value > 1e-8
    }

    resource_totals: dict[str, float] = {key: 0.0 for key in _RESOURCE_DAIMOI_TYPES}
    recipient_totals: dict[str, float] = {}
    lambda_totals: dict[str, float] = {key: 0.0 for key in _RESOURCE_DAIMOI_TYPES}
    lambda_counts: dict[str, int] = {key: 0 for key in _RESOURCE_DAIMOI_TYPES}
    packet_count = 0
    emitter_rows = 0
    cpu_sentinel_forced_packets = 0
    cpu_sentinel_impact = impact_by_id.get(_RESOURCE_DAIMOI_CPU_SENTINEL_ID)

    for row in field_particles:
        if not isinstance(row, dict):
            continue
        if bool(row.get("is_nexus", False)):
            continue
        presence_id = str(row.get("presence_id", "")).strip()
        if presence_id == USER_PRESENCE_ID:
            continue
        if presence_id == _RESOURCE_DAIMOI_CPU_SENTINEL_ID:
            row["resource_emit_disabled"] = True
            row["resource_emit_disabled_reason"] = "cpu_sentinel_sink"
            continue
        resource_type = _core_resource_type_from_presence_id(presence_id)
        if not resource_type:
            row["resource_emit_disabled"] = True
            row["resource_emit_disabled_reason"] = "non_core_presence"
            continue
        if cpu_emitter_cutoff_active:
            row["resource_emit_disabled"] = True
            row["resource_emit_disabled_reason"] = "global_cpu_cutoff"
            continue

        emitter_cpu_cost = 0.0
        emitter_cpu_payment: dict[str, Any] | None = None
        emitter_impact = impact_by_id.get(presence_id)
        if not isinstance(emitter_impact, dict):
            continue
        emitter_wallet = _normalize_resource_wallet(emitter_impact)
        source_balance = max(
            0.0, _safe_float(emitter_wallet.get(resource_type, 0.0), 0.0)
        )

        resource_floor = max(
            0.1,
            _safe_float(_RESOURCE_DAIMOI_WALLET_FLOOR.get(resource_type, 4.0), 4.0),
        )
        resource_pressure = _clamp01(source_balance / max(1e-6, resource_floor))
        if source_balance <= 1e-8:
            continue

        if resource_type != "cpu":
            emitter_cpu_cost = _RESOURCE_DAIMOI_ACTION_BASE_COST
            emitter_cpu_payment = _resource_payment_plan(
                wallet=emitter_wallet,
                focus_resource="cpu",
                desired_cost=emitter_cpu_cost,
                pressure=resource_pressure_by_type,
                prefer_high_pressure=False,
            )
            if _safe_float(
                emitter_cpu_payment.get("effective_credit", 0.0), 0.0
            ) + 1e-9 < (emitter_cpu_cost * _RESOURCE_DAIMOI_ACTION_SATISFIED_RATIO):
                row["resource_action_blocked"] = True
                row["resource_block_reason"] = "resource_wallet_required_for_emit"
                row["resource_emit_affordability"] = round(
                    _clamp01(
                        _safe_float(emitter_cpu_payment.get("affordability", 0.0), 0.0)
                    ),
                    6,
                )
                row["top_job"] = "resource_starved"
                continue

        emitter_rows += 1

        availability = _resource_availability_ratio(resource_type, resource_heartbeat)
        pressure_signal = _clamp01(
            _safe_float(resource_pressure_by_type.get(resource_type, 0.0), 0.0)
        )
        debt_signal = max(0.0, _safe_float(resource_debt.get(resource_type, 0.0), 0.0))
        velocity_signal = _clamp01(
            _safe_float(resource_velocity.get(resource_type, 0.0), 0.0)
        )
        emission_lambda = _resource_emission_rate(
            resource_type=resource_type,
            pressure_signal=pressure_signal,
            debt_value=debt_signal,
            velocity_signal=velocity_signal,
        )
        lambda_totals[resource_type] = (
            lambda_totals.get(resource_type, 0.0) + emission_lambda
        )
        lambda_counts[resource_type] = lambda_counts.get(resource_type, 0) + 1

        influence_power = _clamp01(
            _safe_float(
                row.get(
                    "influence_power",
                    row.get("message_probability", 0.0),
                ),
                0.0,
            )
        )
        route_probability = _clamp01(
            _safe_float(row.get("route_probability", 0.5), 0.5)
        )
        drift_score = _clamp01(abs(_safe_float(row.get("drift_score", 0.0), 0.0)))
        gravity_potential = max(
            0.0, _safe_float(row.get("gravity_potential", 0.0), 0.0)
        )
        gravity_signal = _clamp01(gravity_potential / (gravity_potential + 1.0))
        row_signal = _clamp01(
            (influence_power * 0.45)
            + (route_probability * 0.24)
            + (drift_score * 0.16)
            + (gravity_signal * 0.15)
        )

        emit_amount = emission_lambda
        emit_amount *= 0.5 + (row_signal * 0.5)
        emit_amount *= 0.2 + (availability * 0.8)
        emit_amount *= max(0.2, 1.0 - (_clamp01(_safe_float(queue_ratio, 0.0)) * 0.45))
        emit_amount = min(source_balance, emit_amount)
        emit_amount = max(0.0, emit_amount)
        if emit_amount <= 1e-7:
            continue

        px = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        py = _clamp01(_safe_float(row.get("y", 0.5), 0.5))

        best_target: dict[str, Any] | None = None
        best_target_id = ""
        best_score = -1.0
        forced_cpu_target = False
        if (
            cpu_sentinel_attractor_active
            and resource_type == "cpu"
            and isinstance(cpu_sentinel_impact, dict)
            and presence_id != _RESOURCE_DAIMOI_CPU_SENTINEL_ID
        ):
            forced_target_id = str(cpu_sentinel_impact.get("id", "")).strip()
            if forced_target_id:
                best_target = cpu_sentinel_impact
                best_target_id = forced_target_id
                best_score = 1.0
                forced_cpu_target = True

        if best_target is None:
            for impact in recipient_impacts:
                target_id = str(impact.get("id", "")).strip()
                if not target_id:
                    continue
                need_ratio = _resource_need_ratio(
                    impact,
                    resource_type,
                    queue_ratio=queue_ratio,
                )
                ax, ay = anchor_by_presence.get(target_id, (0.5, 0.5))
                distance = math.sqrt(((ax - px) * (ax - px)) + ((ay - py) * (ay - py)))
                proximity = _clamp01(1.0 - min(1.0, distance / 1.15))
                score = (need_ratio * 0.72) + (proximity * 0.28)
                if score > best_score:
                    best_score = score
                    best_target = impact
                    best_target_id = target_id

        if best_target is None or best_score <= 1e-8:
            continue

        credited = max(0.0, emit_amount)
        if credited <= 1e-8:
            continue

        mix_weights = _resource_mixing_weights(
            resource_type,
            pressure=resource_pressure_by_type,
        )
        mix_vector: dict[str, float] = {}
        for mix_resource, mix_weight in sorted(mix_weights.items()):
            resource_name = _canonical_resource_type(mix_resource)
            if not resource_name:
                continue
            component = max(0.0, credited * max(0.0, _safe_float(mix_weight, 0.0)))
            if component <= 1e-9:
                continue
            mix_vector[resource_name] = component
        credit_total = sum(mix_vector.values())
        if credit_total <= 1e-8:
            continue

        target_wallet = _normalize_resource_wallet(best_target)
        target_denoms = _normalize_resource_wallet_denoms(best_target)
        for mix_resource, mix_amount in mix_vector.items():
            prior = max(0.0, _safe_float(target_wallet.get(mix_resource, 0.0), 0.0))
            target_wallet[mix_resource] = round(prior + mix_amount, 6)
        _wallet_denoms_add_vector(target_denoms, mix_vector)
        best_target["resource_wallet"] = target_wallet
        best_target["resource_wallet_denoms"] = target_denoms

        packet_count += 1
        for mix_resource, mix_amount in mix_vector.items():
            resource_totals[mix_resource] = (
                resource_totals.get(mix_resource, 0.0) + mix_amount
            )
        recipient_totals[best_target_id] = (
            recipient_totals.get(best_target_id, 0.0) + credit_total
        )

        row["resource_daimoi"] = True
        row["resource_type"] = resource_type
        row["resource_emit_amount"] = round(credited, 6)
        row["resource_emit_credit_total"] = round(credit_total, 6)
        row["resource_mix_vector"] = {
            mix_resource: round(max(0.0, _safe_float(mix_amount, 0.0)), 6)
            for mix_resource, mix_amount in sorted(mix_vector.items())
            if _safe_float(mix_amount, 0.0) > 1e-8
        }
        row["resource_emit_wallet_credit"] = True
        row["resource_target_presence_id"] = best_target_id
        row["resource_availability"] = round(availability, 6)
        row["resource_lambda"] = round(max(0.0, emission_lambda), 6)
        row["resource_pressure"] = round(pressure_signal, 6)
        row["resource_debt"] = round(debt_signal, 6)
        row["resource_velocity"] = round(velocity_signal, 6)
        row["resource_action_blocked"] = False
        row["cpu_sentinel_attractor_active"] = bool(
            cpu_sentinel_attractor_active and resource_type == "cpu"
        )
        if forced_cpu_target:
            row["resource_forced_target"] = "cpu_sentinel_attractor"
            cpu_sentinel_forced_packets += 1
        row["top_job"] = "emit_resource_packet"
        row["job_probabilities"] = {
            "emit_resource_packet": round(0.74, 6),
            "invoke_resource_probe": round(0.16, 6),
            "deliver_message": round(0.10, 6),
        }
        # Decrement payload from source
        source_after = max(0.0, source_balance - credited)
        emitter_wallet[resource_type] = round(source_after, 6)

        if emitter_cpu_cost > 0.0 and isinstance(emitter_impact, dict):
            cost_breakdown = (
                emitter_cpu_payment.get("breakdown", {})
                if isinstance(emitter_cpu_payment, dict)
                else {}
            )
            if isinstance(cost_breakdown, dict):
                for cost_resource, cost_value in cost_breakdown.items():
                    resource_name = _canonical_resource_type(cost_resource)
                    if not resource_name:
                        continue
                    balance = max(
                        0.0,
                        _safe_float(emitter_wallet.get(resource_name, 0.0), 0.0),
                    )
                    reduced = max(0.0, balance - max(0.0, _safe_float(cost_value, 0.0)))
                    emitter_wallet[resource_name] = round(reduced, 6)

            row["resource_emit_cpu_cost"] = round(emitter_cpu_cost, 6)
            row["resource_emit_payment_vector"] = {
                resource_name: round(max(0.0, _safe_float(cost, 0.0)), 6)
                for resource_name, cost in sorted(cost_breakdown.items())
                if _safe_float(cost, 0.0) > 1e-8
            }
            row["resource_emit_affordability"] = round(
                _clamp01(
                    _safe_float(
                        (emitter_cpu_payment or {}).get("affordability", 0.0),
                        0.0,
                    )
                ),
                6,
            )
            row["resource_emit_cpu_balance_after"] = round(
                max(0.0, _safe_float(emitter_wallet.get("cpu", 0.0), 0.0)),
                6,
            )

        emitter_impact["resource_wallet"] = emitter_wallet

    summary["emitter_rows"] = int(emitter_rows)
    summary["delivered_packets"] = int(packet_count)
    summary["total_transfer"] = round(sum(resource_totals.values()), 6)
    summary["cpu_sentinel_forced_packets"] = int(cpu_sentinel_forced_packets)
    summary["lambda_by_resource"] = {
        key: round(
            _safe_float(lambda_totals.get(key, 0.0), 0.0)
            / float(max(1, int(lambda_counts.get(key, 0)))),
            6,
        )
        for key in sorted(lambda_totals.keys())
        if int(lambda_counts.get(key, 0)) > 0
    }
    summary["by_resource"] = {
        key: round(value, 6)
        for key, value in sorted(resource_totals.items())
        if value > 1e-8
    }
    summary["recipients"] = [
        {
            "presence_id": key,
            "credited": round(value, 6),
        }
        for key, value in sorted(
            recipient_totals.items(),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )[:16]
    ]
    return summary


def _apply_resource_daimoi_action_consumption(
    *,
    field_particles: list[dict[str, Any]],
    presence_impacts: list[dict[str, Any]],
    resource_heartbeat: dict[str, Any],
    queue_ratio: float,
) -> dict[str, Any]:
    resource_devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    cpu_utilization = max(
        0.0,
        min(
            100.0,
            _safe_float(
                (
                    resource_devices.get("cpu", {})
                    if isinstance(resource_devices.get("cpu", {}), dict)
                    else {}
                ).get("utilization", 0.0),
                0.0,
            ),
        ),
    )
    cpu_sentinel_burn_threshold = max(
        0.0,
        min(
            100.0,
            _safe_float(
                os.getenv(
                    "SIMULATION_CPU_SENTINEL_BURN_START_PERCENT",
                    str(_RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT),
                )
                or str(_RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT),
                _RESOURCE_DAIMOI_CPU_SENTINEL_BURN_START_PERCENT,
            ),
        ),
    )
    sentinel_usage_by_presence: dict[str, float] = {}
    sentinel_burn_active_by_presence: dict[str, bool] = {}
    for (
        sentinel_id,
        sentinel_resource,
    ) in _RESOURCE_DAIMOI_SENTINEL_RESOURCE_BY_ID.items():
        usage = _resource_usage_percent(sentinel_resource, resource_heartbeat)
        sentinel_usage_by_presence[sentinel_id] = usage
        sentinel_burn_active_by_presence[sentinel_id] = (
            usage >= cpu_sentinel_burn_threshold
        )
    cpu_sentinel_burn_active = bool(
        sentinel_burn_active_by_presence.get(_RESOURCE_DAIMOI_CPU_SENTINEL_ID, False)
    )
    resource_pressure = _resource_pressure_vector(resource_heartbeat)
    resource_debt = _resource_debt_snapshot()

    summary: dict[str, Any] = {
        "record": "eta-mu.resource-daimoi-consumption.v1",
        "schema_version": "resource.daimoi.consumption.v1",
        "action_packets": 0,
        "blocked_packets": 0,
        "consumed_total": 0.0,
        "debt_total": 0.0,
        "by_resource": {},
        "starved_presences": [],
        "active_presences": [],
        "queue_ratio": round(_clamp01(_safe_float(queue_ratio, 0.0)), 6),
        "cpu_utilization": round(cpu_utilization, 2),
        "cpu_sentinel_id": _RESOURCE_DAIMOI_CPU_SENTINEL_ID,
        "cpu_sentinel_burn_threshold": round(
            cpu_sentinel_burn_threshold,
            2,
        ),
        "cpu_sentinel_burn_max_multiplier": round(
            _RESOURCE_DAIMOI_CPU_SENTINEL_BURN_MAX_MULTIPLIER,
            6,
        ),
        "cpu_sentinel_burn_cost_max": round(
            _RESOURCE_DAIMOI_CPU_SENTINEL_BURN_COST_MAX,
            6,
        ),
        "cpu_sentinel_burn_active": bool(cpu_sentinel_burn_active),
        "sentinel_burn_threshold": round(
            cpu_sentinel_burn_threshold,
            2,
        ),
        "sentinel_burn_active": {
            sentinel_id: bool(active)
            for sentinel_id, active in sorted(sentinel_burn_active_by_presence.items())
        },
        "sentinel_resource_usage": {
            sentinel_id: round(_safe_float(usage, 0.0), 2)
            for sentinel_id, usage in sorted(sentinel_usage_by_presence.items())
        },
        "resource_pressure": {
            resource_type: round(_clamp01(_safe_float(value, 0.0)), 6)
            for resource_type, value in sorted(resource_pressure.items())
        },
        "resource_debt": {
            resource_type: round(max(0.0, _safe_float(value, 0.0)), 6)
            for resource_type, value in sorted(resource_debt.items())
        },
    }
    if not isinstance(field_particles, list) or not isinstance(presence_impacts, list):
        return summary

    impact_by_id: dict[str, dict[str, Any]] = {}
    for impact in presence_impacts:
        if not isinstance(impact, dict):
            continue
        presence_id = str(impact.get("id", "")).strip()
        if not presence_id:
            continue
        _normalize_resource_wallet(impact)
        impact_by_id[presence_id] = impact

    if not impact_by_id:
        return summary

    queue_push = _clamp01(_safe_float(queue_ratio, 0.0))
    consumed_by_resource: dict[str, float] = {
        key: 0.0 for key in _RESOURCE_DAIMOI_TYPES
    }
    consumed_by_presence: dict[str, float] = {}
    debt_by_presence: dict[str, float] = {}
    blocked_by_presence: dict[str, int] = {}
    blocked_packets = 0
    action_packets = 0
    ctl_overhead: dict[str, float] = {key: 0.0 for key in _RESOURCE_DAIMOI_TYPES}

    ordered_rows = [row for row in field_particles if isinstance(row, dict)]
    control_budget = _resource_ctl_budget_prepare(
        queue_push=queue_push,
        candidate_count=len(ordered_rows),
    )
    ordered_rows.sort(
        key=lambda row: _resource_action_utility(
            row=row,
            presence_id=str(
                (row if isinstance(row, dict) else {}).get("presence_id", "")
            ).strip(),
            resource_pressure=resource_pressure,
            queue_push=queue_push,
            resource_debt=resource_debt,
            sentinel_usage_by_presence=sentinel_usage_by_presence,
            cpu_sentinel_burn_threshold=cpu_sentinel_burn_threshold,
        ),
        reverse=True,
    )
    max_actions = max(
        1, int(_safe_float(control_budget.get("max_actions", 256), 256.0))
    )
    allow_denom = bool(control_budget.get("allow_denom", True))
    candidate_rows = len(ordered_rows)
    if len(ordered_rows) > max_actions:
        ordered_rows = ordered_rows[:max_actions]
    summary["control_budget"] = {
        "mode": str(control_budget.get("mode", "full")),
        "ratio": round(_safe_float(control_budget.get("ratio", 0.0), 0.0), 6),
        "allow_denom": bool(allow_denom),
        "max_actions": int(max_actions),
        "candidate_rows": int(candidate_rows),
        "scheduled_rows": int(len(ordered_rows)),
        "cap": {
            key: round(_safe_float(value, 0.0), 6)
            for key, value in sorted(
                (
                    control_budget.get("cap", {})
                    if isinstance(control_budget, dict)
                    else {}
                ).items()
            )
        },
        "before": {
            key: round(_safe_float(value, 0.0), 6)
            for key, value in sorted(
                (
                    control_budget.get("before", {})
                    if isinstance(control_budget, dict)
                    else {}
                ).items()
            )
        },
    }

    for row in ordered_rows:
        presence_id = str(row.get("presence_id", "")).strip()
        for resource_name, eval_cost in _RESOURCE_CTL_BUDGET_EVAL_COST.items():
            ctl_overhead[resource_name] = ctl_overhead.get(resource_name, 0.0) + max(
                0.0,
                _safe_float(eval_cost, 0.0),
            )
        if not presence_id:
            continue
        if presence_id == USER_PRESENCE_ID:
            row["resource_action_blocked"] = False
            continue
        if _core_resource_type_from_presence_id(presence_id):
            continue

        impact = impact_by_id.get(presence_id)
        if not isinstance(impact, dict):
            continue

        sentinel_resource = _RESOURCE_DAIMOI_SENTINEL_RESOURCE_BY_ID.get(
            presence_id,
            "",
        )
        is_resource_sentinel = bool(sentinel_resource)
        sentinel_usage = _safe_float(
            sentinel_usage_by_presence.get(
                presence_id,
                cpu_utilization
                if presence_id == _RESOURCE_DAIMOI_CPU_SENTINEL_ID
                else 0.0,
            ),
            0.0,
        )
        sentinel_burn_active = bool(
            sentinel_burn_active_by_presence.get(
                presence_id,
                False,
            )
        )

        if is_resource_sentinel and not sentinel_burn_active:
            row["resource_action_blocked"] = False
            row["resource_sentinel_idle"] = True
            row["resource_sentinel_resource_type"] = sentinel_resource
            row["resource_sentinel_usage_percent"] = round(sentinel_usage, 2)
            row["resource_sentinel_burn_threshold"] = round(
                cpu_sentinel_burn_threshold,
                2,
            )
            if presence_id == _RESOURCE_DAIMOI_CPU_SENTINEL_ID:
                row["resource_sentinel_cpu_utilization"] = round(cpu_utilization, 2)
            top_job = str(row.get("top_job", "")).strip()
            if top_job in {"", "observe"}:
                row["top_job"] = "observe"
            continue

        wallet = _normalize_resource_wallet(impact)
        denoms = _normalize_resource_wallet_denoms(impact)
        contract = _resource_action_contract_estimate(
            row=row,
            presence_id=presence_id,
            resource_pressure=resource_pressure,
            resource_debt=resource_debt,
            queue_push=queue_push,
            sentinel_usage_by_presence=sentinel_usage_by_presence,
            cpu_sentinel_burn_threshold=cpu_sentinel_burn_threshold,
        )
        focus_resource = (
            _canonical_resource_type(
                str(contract.get("focus_resource", "cpu") or "cpu")
            )
            or "cpu"
        )
        desired_cost = max(
            _RESOURCE_DAIMOI_ACTION_BASE_COST,
            _safe_float(contract.get("desired_cost", 0.0), 0.0),
        )
        action_risk = _clamp01(_safe_float(contract.get("action_risk", 0.0), 0.0))
        focus_pressure = _clamp01(_safe_float(contract.get("focus_pressure", 0.0), 0.0))
        focus_debt = max(0.0, _safe_float(contract.get("focus_debt", 0.0), 0.0))
        risk_multiplier = max(
            1.0,
            _safe_float(contract.get("risk_multiplier", 1.0), 1.0),
        )
        debt_multiplier = max(
            1.0,
            _safe_float(contract.get("debt_multiplier", 1.0), 1.0),
        )
        row["resource_action_risk"] = round(action_risk, 6)
        row["resource_action_focus_pressure"] = round(focus_pressure, 6)
        row["resource_action_focus_debt"] = round(focus_debt, 6)
        row["resource_action_risk_multiplier"] = round(risk_multiplier, 6)
        row["resource_action_debt_multiplier"] = round(debt_multiplier, 6)
        row["resource_action_utility"] = round(
            _safe_float(contract.get("utility", 0.0), 0.0),
            6,
        )

        if is_resource_sentinel:
            row["resource_sentinel_idle"] = False
            row["resource_sentinel_resource_type"] = focus_resource
            row["resource_sentinel_usage_percent"] = round(
                _safe_float(
                    contract.get("sentinel_usage", sentinel_usage), sentinel_usage
                ),
                2,
            )
            row["resource_sentinel_burn_intensity"] = round(
                _clamp01(
                    _safe_float(contract.get("sentinel_burn_intensity", 0.0), 0.0)
                ),
                6,
            )
            row["resource_sentinel_burn_multiplier"] = round(
                max(
                    1.0, _safe_float(contract.get("sentinel_burn_multiplier", 1.0), 1.0)
                ),
                6,
            )
            if presence_id == _RESOURCE_DAIMOI_CPU_SENTINEL_ID:
                row["resource_sentinel_cpu_utilization"] = round(cpu_utilization, 2)
            row["resource_sentinel_burn_threshold"] = round(
                _safe_float(
                    contract.get("sentinel_threshold", cpu_sentinel_burn_threshold),
                    cpu_sentinel_burn_threshold,
                ),
                2,
            )

        expected_cost_vector = _resource_vector_quantized(
            _resource_vector_normalized(contract.get("expected_cost_vector", {}))
        )
        reclaim_vector = _resource_vector_quantized(
            _resource_vector_normalized(contract.get("reclaim_vector", {}))
        )
        required_payment_vector = _resource_vector_quantized(
            _resource_vector_normalized(contract.get("required_payment_vector", {}))
        )
        if _resource_vector_total(required_payment_vector) <= 1e-12:
            required_payment_vector = _resource_required_payment_vector(
                focus_resource=focus_resource,
                desired_cost=desired_cost,
                pressure=resource_pressure,
            )
        desired_cost = max(
            desired_cost, _resource_vector_total(required_payment_vector)
        )

        row["resource_contract_cost_vector"] = {
            resource_name: round(max(0.0, _safe_float(amount, 0.0)), 6)
            for resource_name, amount in sorted(expected_cost_vector.items())
            if _safe_float(amount, 0.0) > 1e-8
        }
        row["resource_contract_reclaim_vector"] = {
            resource_name: round(max(0.0, _safe_float(amount, 0.0)), 6)
            for resource_name, amount in sorted(reclaim_vector.items())
            if _safe_float(amount, 0.0) > 1e-8
        }
        row["resource_required_payment_vector"] = {
            resource_name: round(max(0.0, _safe_float(amount, 0.0)), 6)
            for resource_name, amount in sorted(required_payment_vector.items())
            if _safe_float(amount, 0.0) > 1e-8
        }

        consume_breakdown: dict[str, float] = {}
        consumed = 0.0
        effective_credit = 0.0
        action_debt = max(0.0, desired_cost)
        satisfied = False
        overpay = 0.0

        if denoms and allow_denom:
            row["resource_burn_strategy"] = "denom_knapsack"
            for (
                resource_name,
                extra_cost,
            ) in _RESOURCE_CTL_BUDGET_DENOM_EXTRA_COST.items():
                ctl_overhead[resource_name] = ctl_overhead.get(
                    resource_name, 0.0
                ) + max(
                    0.0,
                    _safe_float(extra_cost, 0.0),
                )
            denom_plan = _wallet_denoms_payment_plan(
                denoms=denoms,
                required_vector=required_payment_vector,
            )
            satisfied = bool(
                desired_cost <= 1e-9 or bool(denom_plan.get("affordable", False))
            )
            if satisfied:
                selected_by_index_raw = (
                    denom_plan.get("selected", {})
                    if isinstance(denom_plan, dict)
                    else {}
                )
                selected_by_index: dict[int, int] = {}
                if isinstance(selected_by_index_raw, dict):
                    for key, value in selected_by_index_raw.items():
                        index = int(
                            _safe_float(key, key if isinstance(key, int) else 0)
                        )
                        count = max(0, int(_safe_float(value, 0.0)))
                        if count <= 0:
                            continue
                        selected_by_index[index] = (
                            selected_by_index.get(index, 0) + count
                        )

                spent_vector = (
                    denom_plan.get("spent_vector", {})
                    if isinstance(denom_plan, dict)
                    else {}
                )
                if isinstance(spent_vector, dict):
                    consume_breakdown = _resource_vector_quantized(
                        _resource_vector_normalized(spent_vector)
                    )

                for index in sorted(selected_by_index.keys(), reverse=True):
                    used_count = max(0, int(selected_by_index.get(index, 0)))
                    if used_count <= 0:
                        continue
                    if index < 0 or index >= len(denoms):
                        continue
                    bucket = denoms[index]
                    if not isinstance(bucket, dict):
                        continue
                    bucket_count = max(
                        0, int(_safe_float(bucket.get("count", 0.0), 0.0))
                    )
                    next_count = max(0, bucket_count - used_count)
                    if next_count <= 0:
                        denoms.pop(index)
                    else:
                        bucket["count"] = next_count
                        denoms[index] = bucket

                for resource_name, spend_amount in consume_breakdown.items():
                    balance = max(0.0, _safe_float(wallet.get(resource_name, 0.0), 0.0))
                    wallet[resource_name] = round(
                        max(0.0, balance - max(0.0, _safe_float(spend_amount, 0.0))),
                        6,
                    )

                consumed = max(0.0, _resource_vector_total(consume_breakdown))
                effective_credit = max(0.0, desired_cost)
                action_debt = max(0.0, desired_cost - effective_credit)
                overpay = max(0.0, _safe_float(denom_plan.get("overpay", 0.0), 0.0))
            else:
                consume_breakdown = {}
                consumed = 0.0
                effective_credit = 0.0
                action_debt = max(0.0, desired_cost)
        else:
            row["resource_burn_strategy"] = "aggregate_mix"
            payment = _resource_payment_plan(
                wallet=wallet,
                focus_resource=focus_resource,
                desired_cost=desired_cost,
                pressure=resource_pressure,
                prefer_high_pressure=is_resource_sentinel,
            )
            effective_credit_raw = max(
                0.0,
                _safe_float(payment.get("effective_credit", 0.0), 0.0),
            )
            consume_breakdown_raw = payment.get("breakdown", {})
            satisfied = desired_cost <= 1e-9 or effective_credit_raw >= (
                desired_cost * _RESOURCE_DAIMOI_ACTION_SATISFIED_RATIO
            )
            if satisfied and isinstance(consume_breakdown_raw, dict):
                for cost_resource, cost_value in consume_breakdown_raw.items():
                    resource_name = _canonical_resource_type(cost_resource)
                    if not resource_name:
                        continue
                    spend_amount = max(0.0, _safe_float(cost_value, 0.0))
                    if spend_amount <= 1e-12:
                        continue
                    current_balance = max(
                        0.0,
                        _safe_float(wallet.get(resource_name, 0.0), 0.0),
                    )
                    next_balance = max(0.0, current_balance - spend_amount)
                    wallet[resource_name] = round(next_balance, 6)
                    consume_breakdown[resource_name] = (
                        consume_breakdown.get(resource_name, 0.0) + spend_amount
                    )

                consumed = max(0.0, sum(consume_breakdown.values()))
                effective_credit = effective_credit_raw
                action_debt = max(0.0, desired_cost - effective_credit)
            else:
                consume_breakdown = {}
                consumed = 0.0
                effective_credit = 0.0
                action_debt = max(0.0, desired_cost)
        remaining = max(0.0, _safe_float(wallet.get(focus_resource, 0.0), 0.0))
        impact["resource_wallet"] = wallet
        impact["resource_wallet_denoms"] = denoms

        row["resource_consume_type"] = focus_resource
        row["resource_consume_amount"] = round(consumed, 6)
        row["resource_action_cost"] = round(desired_cost, 6)
        row["resource_effective_credit"] = round(effective_credit, 6)
        row["resource_action_debt"] = round(action_debt, 6)
        row["resource_affordability"] = round(
            _clamp01(effective_credit / max(1e-9, desired_cost)),
            6,
        )
        row["resource_payment_overpay"] = round(max(0.0, overpay), 6)
        row["resource_payment_vector"] = {
            resource_name: round(max(0.0, _safe_float(cost, 0.0)), 6)
            for resource_name, cost in sorted(consume_breakdown.items())
            if _safe_float(cost, 0.0) > 1e-8
        }
        row["resource_balance_after"] = round(remaining, 6)

        action_packets += 1
        for resource_name, cost_value in consume_breakdown.items():
            consumed_by_resource[resource_name] = consumed_by_resource.get(
                resource_name, 0.0
            ) + max(0.0, _safe_float(cost_value, 0.0))
        consumed_by_presence[presence_id] = (
            consumed_by_presence.get(presence_id, 0.0) + consumed
        )
        debt_by_presence[presence_id] = (
            debt_by_presence.get(presence_id, 0.0) + action_debt
        )
        if not satisfied:
            blocked_packets += 1
            blocked_by_presence[presence_id] = (
                blocked_by_presence.get(presence_id, 0) + 1
            )
            row["resource_action_blocked"] = True
            row["top_job"] = "resource_starved"
            row["message_probability"] = round(
                _clamp01(_safe_float(row.get("message_probability", 0.0), 0.0) * 0.22),
                6,
            )
            row["route_probability"] = round(
                _clamp01(_safe_float(row.get("route_probability", 0.0), 0.0) * 0.32),
                6,
            )
            row["influence_power"] = round(
                _clamp01(_safe_float(row.get("influence_power", 0.0), 0.0) * 0.28),
                6,
            )
            row["vx"] = round(_safe_float(row.get("vx", 0.0), 0.0) * 0.4, 6)
            row["vy"] = round(_safe_float(row.get("vy", 0.0), 0.0) * 0.4, 6)
            row["r"] = round(
                _clamp01((_safe_float(row.get("r", 0.4), 0.4) * 0.78) + 0.16),
                5,
            )
            row["g"] = round(
                _clamp01(_safe_float(row.get("g", 0.4), 0.4) * 0.42),
                5,
            )
            row["b"] = round(
                _clamp01(_safe_float(row.get("b", 0.4), 0.4) * 0.42),
                5,
            )
        else:
            row["resource_action_blocked"] = False
            if is_resource_sentinel:
                row["top_job"] = "burn_resource_packet"
            else:
                top_job = str(row.get("top_job", "")).strip()
                if top_job in {"", "observe"}:
                    row["top_job"] = "consume_resource_packet"

    summary["action_packets"] = int(action_packets)
    summary["blocked_packets"] = int(blocked_packets)
    summary["consumed_total"] = round(sum(consumed_by_resource.values()), 6)
    summary["debt_total"] = round(sum(debt_by_presence.values()), 6)
    summary["by_resource"] = {
        resource: round(amount, 6)
        for resource, amount in sorted(consumed_by_resource.items())
        if amount > 1e-8
    }
    summary["starved_presences"] = [
        {
            "presence_id": presence_id,
            "blocked_packets": blocked,
        }
        for presence_id, blocked in sorted(
            blocked_by_presence.items(),
            key=lambda item: (-item[1], item[0]),
        )[:16]
    ]
    summary["active_presences"] = [
        {
            "presence_id": presence_id,
            "consumed": round(amount, 6),
        }
        for presence_id, amount in sorted(
            consumed_by_presence.items(),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )[:16]
        if amount > 1e-8
    ]
    summary["debt_by_presence"] = [
        {
            "presence_id": presence_id,
            "debt": round(amount, 6),
        }
        for presence_id, amount in sorted(
            debt_by_presence.items(),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )[:16]
        if amount > 1e-8
    ]
    ctl_after = _resource_ctl_budget_commit(ctl_overhead)
    control_budget_row = summary.get("control_budget", {})
    if isinstance(control_budget_row, dict):
        control_budget_row["overhead"] = {
            key: round(_safe_float(value, 0.0), 6)
            for key, value in sorted(ctl_overhead.items())
            if _safe_float(value, 0.0) > 1e-8
        }
        control_budget_row["after"] = {
            key: round(_safe_float(value, 0.0), 6)
            for key, value in sorted(ctl_after.items())
        }
        summary["control_budget"] = control_budget_row
    return summary


def reset_resource_runtime_state() -> None:
    global _RESOURCE_DAIMOI_DEBT_LAST_MONOTONIC
    with _RESOURCE_DAIMOI_DEBT_LOCK:
        _RESOURCE_DAIMOI_DEBT_LAST_MONOTONIC = time.monotonic()
        for resource_type in _RESOURCE_DAIMOI_TYPES:
            _RESOURCE_DAIMOI_DEBT_STATE[resource_type] = 0.0
    with _RESOURCE_DAIMOI_VELOCITY_LOCK:
        _RESOURCE_DAIMOI_VELOCITY_STATE["last_monotonic"] = time.monotonic()
        _RESOURCE_DAIMOI_VELOCITY_STATE["usage_prev"] = {
            resource_type: 0.0 for resource_type in _RESOURCE_DAIMOI_TYPES
        }
    with _RESOURCE_CTL_BUDGET_LOCK:
        _RESOURCE_CTL_BUDGET_STATE["last_monotonic"] = time.monotonic()
        _RESOURCE_CTL_BUDGET_STATE["budget"] = _resource_ctl_budget_cap_vector()
