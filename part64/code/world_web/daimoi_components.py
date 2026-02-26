from __future__ import annotations

import math
from typing import Any

from .metrics import _safe_float


DAIMOI_COMPONENT_RESOURCE_REQ: dict[str, dict[str, float]] = {
    "deliver_message": {
        "cpu": 0.22,
        "gpu": 0.05,
        "npu": 0.02,
        "ram": 0.28,
        "disk": 0.16,
        "network": 0.86,
    },
    "invoke_receipt_audit": {
        "cpu": 0.48,
        "gpu": 0.08,
        "npu": 0.04,
        "ram": 0.41,
        "disk": 0.58,
        "network": 0.22,
    },
    "invoke_truth_gate": {
        "cpu": 0.55,
        "gpu": 0.09,
        "npu": 0.08,
        "ram": 0.46,
        "disk": 0.28,
        "network": 0.3,
    },
    "invoke_anchor_register": {
        "cpu": 0.44,
        "gpu": 0.06,
        "npu": 0.03,
        "ram": 0.36,
        "disk": 0.3,
        "network": 0.52,
    },
    "invoke_file_organize": {
        "cpu": 0.37,
        "gpu": 0.06,
        "npu": 0.03,
        "ram": 0.34,
        "disk": 0.83,
        "network": 0.2,
    },
    "invoke_graph_crawl": {
        "cpu": 0.5,
        "gpu": 0.07,
        "npu": 0.04,
        "ram": 0.4,
        "disk": 0.33,
        "network": 0.64,
    },
    "invoke_resource_probe": {
        "cpu": 0.54,
        "gpu": 0.26,
        "npu": 0.22,
        "ram": 0.38,
        "disk": 0.3,
        "network": 0.24,
    },
    "invoke_diffuse_field": {
        "cpu": 0.31,
        "gpu": 0.18,
        "npu": 0.15,
        "ram": 0.26,
        "disk": 0.2,
        "network": 0.34,
    },
    "emit_resource_packet": {
        "cpu": 0.66,
        "gpu": 0.38,
        "npu": 0.34,
        "ram": 0.46,
        "disk": 0.4,
        "network": 0.44,
    },
    "absorb_resource": {
        "cpu": 0.42,
        "gpu": 0.31,
        "npu": 0.27,
        "ram": 0.54,
        "disk": 0.43,
        "network": 0.39,
    },
}


DAIMOI_COMPONENT_COST: dict[str, float] = {
    "deliver_message": 0.18,
    "invoke_receipt_audit": 0.52,
    "invoke_truth_gate": 0.58,
    "invoke_anchor_register": 0.34,
    "invoke_file_organize": 0.47,
    "invoke_graph_crawl": 0.44,
    "invoke_resource_probe": 0.42,
    "invoke_diffuse_field": 0.26,
    "emit_resource_packet": 0.36,
    "absorb_resource": 0.31,
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


def component_resource_req(
    job_key: str,
    *,
    resource_keys: tuple[str, ...],
    component_resource_req_map: dict[str, dict[str, float]] | None = None,
    token_boost: float = 0.58,
) -> dict[str, float]:
    token = str(job_key or "").strip()
    source_map = (
        component_resource_req_map
        if isinstance(component_resource_req_map, dict)
        else DAIMOI_COMPONENT_RESOURCE_REQ
    )
    base = source_map.get(token, {})
    req = {
        resource: _clamp01_finite(base.get(resource, 0.0), 0.0)
        for resource in resource_keys
    }
    lowered = token.lower()
    boost_value = _clamp01_finite(token_boost, 0.58)
    for resource in resource_keys:
        if resource in lowered:
            req[resource] = max(req[resource], boost_value)
    return req


def component_cost(
    job_key: str,
    *,
    component_cost_map: dict[str, float] | None = None,
    default_cost: float = 0.3,
) -> float:
    source_map = (
        component_cost_map
        if isinstance(component_cost_map, dict)
        else DAIMOI_COMPONENT_COST
    )
    default_value = max(0.0, _finite_float(default_cost, 0.3))
    return max(
        0.0,
        _finite_float(source_map.get(str(job_key or ""), default_value), default_value),
    )
