from __future__ import annotations

import math
from typing import Any, Callable


def softmax_probabilities(
    values: list[float],
    *,
    finite_float: Callable[[Any, float], float],
) -> list[float]:
    if not values:
        return []
    finite_values = [finite_float(value, 0.0) for value in values]
    max_value = max(finite_values)
    exps = [math.exp(value - max_value) for value in finite_values]
    total = sum(exps)
    if total <= 1e-12:
        uniform = 1.0 / float(len(values))
        return [uniform for _ in values]
    return [value / total for value in exps]


def sample_absorb_component(
    *,
    components: list[dict[str, Any]],
    lens_embedding: list[float],
    need_by_resource: dict[str, float],
    context: dict[str, Any],
    seed: str,
    resource_keys: tuple[str, ...],
    component_embedding: Callable[[str], list[float]],
    component_cost: Callable[[str], float],
    clamp01_finite: Callable[[Any, float], float],
    finite_float: Callable[[Any, float], float],
    coerce_vector: Callable[[Any], list[float]],
    normalize_vector: Callable[[list[float]], list[float]],
    safe_cosine_unit: Callable[[list[float], list[float]], float],
    stable_ratio: Callable[[str, int], float],
    softplus: Callable[[float], float],
    beta_weights: tuple[float, ...],
    temp_weights: tuple[float, ...],
    beta_max: float,
    temp_min: float,
    temp_max: float,
    zeta: float,
    lambda_cost: float,
    record: str,
    schema_version: str,
    method: str,
) -> dict[str, Any]:
    feature_vector = [
        clamp01_finite(context.get("pressure", 0.0), 0.0),
        clamp01_finite(context.get("congestion", 0.0), 0.0),
        clamp01_finite(context.get("wallet_pressure", 0.0), 0.0),
        clamp01_finite(context.get("message_entropy", 0.0), 0.0),
        clamp01_finite(context.get("queue", 0.0), 0.0),
        clamp01_finite(context.get("contact", 0.0), 0.0),
    ]
    beta_raw = sum(
        weight * feature for weight, feature in zip(beta_weights, feature_vector)
    )
    temp_raw = sum(
        weight * feature for weight, feature in zip(temp_weights, feature_vector)
    )
    beta = min(beta_max, max(0.0, softplus(beta_raw)))
    temperature = min(
        temp_max,
        max(temp_min, temp_min + softplus(temp_raw)),
    )

    need = {
        resource: clamp01_finite(
            (need_by_resource if isinstance(need_by_resource, dict) else {}).get(
                resource,
                0.0,
            ),
            0.0,
        )
        for resource in resource_keys
    }
    lens_unit = normalize_vector(coerce_vector(lens_embedding))

    scored_rows: list[dict[str, Any]] = []
    scaled_logits: list[float] = []
    for index, row in enumerate(components):
        if not isinstance(row, dict):
            continue
        component_id = str(row.get("component_id", "")).strip()
        if not component_id:
            continue
        p_i = max(1e-12, finite_float(row.get("p_i", 0.0), 0.0))
        req_raw = row.get("req", {})
        req_map = req_raw if isinstance(req_raw, dict) else {}
        req = {
            resource: clamp01_finite(req_map.get(resource, 0.0), 0.0)
            for resource in resource_keys
        }
        embedding = coerce_vector(
            row.get("embedding", component_embedding(component_id))
        )
        s_i = safe_cosine_unit(lens_unit, normalize_vector(embedding))
        q_i = sum(need[resource] * req[resource] for resource in resource_keys)
        cost_i = max(
            0.0, finite_float(row.get("cost_i", component_cost(component_id)), 0.0)
        )
        logit = math.log(p_i) + (beta * s_i) + (zeta * q_i) - (lambda_cost * cost_i)
        scaled_logit = logit / max(temp_min, temperature)
        scaled_logits.append(scaled_logit)
        scored_rows.append(
            {
                "index": int(index),
                "component_id": component_id,
                "p_i": p_i,
                "req": req,
                "s_i": s_i,
                "q_i": q_i,
                "cost_i": cost_i,
                "logit": logit,
                "scaled_logit": scaled_logit,
            }
        )

    if not scored_rows:
        return {
            "record": record,
            "schema_version": schema_version,
            "method": method,
            "beta": round(beta, 6),
            "temperature": round(temperature, 6),
            "zeta": zeta,
            "lambda_cost": lambda_cost,
            "feature_vector": [round(value, 6) for value in feature_vector],
            "selected_component_id": "",
            "selected_probability": 0.0,
            "components": [],
        }

    probs = softmax_probabilities(scaled_logits, finite_float=finite_float)
    selected: dict[str, Any] | None = None
    for index, row in enumerate(scored_rows):
        prob = probs[index] if index < len(probs) else 0.0
        row["probability"] = prob
        uniform = stable_ratio(
            f"{seed}|absorb|{row['component_id']}|{index}",
            index + 11,
        )
        uniform = min(1.0 - 1e-9, max(1e-9, finite_float(uniform, 0.5)))
        gumbel = -math.log(-math.log(uniform))
        row["gumbel"] = gumbel
        row["gumbel_score"] = finite_float(row.get("scaled_logit", 0.0), 0.0) + gumbel
        if selected is None or (
            finite_float(row.get("gumbel_score", 0.0), 0.0)
            > finite_float(selected.get("gumbel_score", 0.0), 0.0)
        ):
            selected = row

    selected_row = selected if isinstance(selected, dict) else scored_rows[0]
    selected_probability = clamp01_finite(selected_row.get("probability", 0.0), 0.0)
    return {
        "record": record,
        "schema_version": schema_version,
        "method": method,
        "beta": round(beta, 6),
        "temperature": round(temperature, 6),
        "zeta": zeta,
        "lambda_cost": lambda_cost,
        "feature_vector": [round(value, 6) for value in feature_vector],
        "selected_component_id": str(selected_row.get("component_id", "")),
        "selected_probability": round(selected_probability, 6),
        "components": [
            {
                "component_id": str(row.get("component_id", "")),
                "p_i": round(clamp01_finite(row.get("p_i", 0.0), 0.0), 6),
                "req": {
                    resource: round(clamp01_finite(value, 0.0), 6)
                    for resource, value in (
                        row.get("req", {})
                        if isinstance(row.get("req", {}), dict)
                        else {}
                    ).items()
                },
                "s_i": round(finite_float(row.get("s_i", 0.0), 0.0), 6),
                "q_i": round(clamp01_finite(row.get("q_i", 0.0), 0.0), 6),
                "cost_i": round(
                    max(0.0, finite_float(row.get("cost_i", 0.0), 0.0)),
                    6,
                ),
                "logit": round(finite_float(row.get("logit", 0.0), 0.0), 6),
                "probability": round(
                    clamp01_finite(row.get("probability", 0.0), 0.0),
                    6,
                ),
                "gumbel": round(finite_float(row.get("gumbel", 0.0), 0.0), 6),
                "gumbel_score": round(
                    finite_float(row.get("gumbel_score", 0.0), 0.0),
                    6,
                ),
            }
            for row in scored_rows
        ],
    }
