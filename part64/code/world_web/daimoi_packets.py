from __future__ import annotations

import math
from typing import Any, Callable

from .metrics import _safe_float


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


def packet_components_from_job_probabilities(
    probabilities: dict[str, float],
    *,
    component_resource_req: Callable[[str], dict[str, float]],
    component_cost: Callable[[str], float],
    component_embedding: Callable[[str], list[float]],
) -> list[dict[str, Any]]:
    if not isinstance(probabilities, dict):
        return []

    sanitized: dict[str, float] = {}
    total = 0.0
    for key, raw in probabilities.items():
        token = str(key or "").strip()
        if not token:
            continue
        value = max(0.0, _finite_float(raw, 0.0))
        if value <= 1e-12:
            continue
        sanitized[token] = value
        total += value

    if total <= 1e-12:
        return []

    components: list[dict[str, Any]] = []
    for component_id in sorted(sanitized.keys()):
        p_i = max(1e-12, sanitized[component_id] / total)
        req = component_resource_req(component_id)
        components.append(
            {
                "component_id": component_id,
                "p_i": p_i,
                "req": req,
                "cost_i": component_cost(component_id),
                "embedding": component_embedding(component_id),
            }
        )

    components.sort(
        key=lambda row: (
            -_finite_float(row.get("p_i", 0.0), 0.0),
            str(row.get("component_id", "")),
        )
    )
    return components


def packet_resource_signature(
    components: list[dict[str, Any]],
    *,
    resource_keys: tuple[str, ...],
) -> dict[str, float]:
    rho = {resource: 0.0 for resource in resource_keys}
    for row in components:
        if not isinstance(row, dict):
            continue
        p_i = _clamp01_finite(row.get("p_i", 0.0), 0.0)
        req = row.get("req", {})
        req_map = req if isinstance(req, dict) else {}
        for resource in resource_keys:
            rho[resource] = rho[resource] + (
                p_i * _clamp01_finite(req_map.get(resource, 0.0), 0.0)
            )
    return {resource: _clamp01_finite(value, 0.0) for resource, value in rho.items()}


def packet_component_contract(
    components: list[dict[str, Any]],
    *,
    resource_keys: tuple[str, ...],
    record: str,
    schema_version: str,
    top_k: int = 4,
) -> dict[str, Any]:
    resource_signature = packet_resource_signature(
        components,
        resource_keys=resource_keys,
    )

    if top_k > 0:
        visible_rows = components[: int(top_k)]
    else:
        visible_rows = components

    visible = [
        {
            "component_id": str(row.get("component_id", "")),
            "p_i": round(_clamp01_finite(row.get("p_i", 0.0), 0.0), 6),
            "req": {
                resource: round(_clamp01_finite(req_value, 0.0), 6)
                for resource, req_value in (
                    row.get("req", {}) if isinstance(row.get("req", {}), dict) else {}
                ).items()
                if str(resource).strip()
            },
            "cost_i": round(max(0.0, _finite_float(row.get("cost_i", 0.0), 0.0)), 6),
        }
        for row in visible_rows
        if isinstance(row, dict)
    ]

    return {
        "record": record,
        "schema_version": schema_version,
        "component_count": int(len(components)),
        "components": visible,
        "resource_signature": {
            resource: round(_clamp01_finite(value, 0.0), 6)
            for resource, value in resource_signature.items()
        },
    }
