from __future__ import annotations

import math
from typing import Any, Callable


def seed_curr_matrix(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    state_unit_vector: Callable[[dict[str, Any], str], list[float]],
    safe_cosine_unit: Callable[[list[float], list[float]], float],
) -> dict[str, float]:
    left_seed = state_unit_vector(left, "e_seed")
    left_curr = state_unit_vector(left, "e_curr")
    right_seed = state_unit_vector(right, "e_seed")
    right_curr = state_unit_vector(right, "e_curr")
    return {
        "ss": safe_cosine_unit(left_seed, right_seed),
        "sc": safe_cosine_unit(left_seed, right_curr),
        "cs": safe_cosine_unit(left_curr, right_seed),
        "cc": safe_cosine_unit(left_curr, right_curr),
        "self_left": safe_cosine_unit(left_seed, left_curr),
        "self_right": safe_cosine_unit(right_seed, right_curr),
    }


def collision_semantic_update(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    impulse: float,
    seed_curr_matrix_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, float]],
    finite_float: Callable[[Any, float], float],
    clamp01: Callable[[Any], float],
    clamp01_finite: Callable[[Any, float], float],
    sigmoid: Callable[[float], float],
    safe_float: Callable[[Any, float], float],
    state_unit_vector: Callable[[dict[str, Any], str], list[float]],
    blend_vectors: Callable[[list[float], list[float], float], list[float]],
    normalize_vector: Callable[[list[float]], list[float]],
    dirichlet_transfer_fn: Callable[..., dict[str, float]],
    job_keys_set: Any,
    job_keys_sorted: tuple[str, ...],
    job_keys: tuple[str, ...],
    alpha_baseline: float,
    alpha_max: float,
    transfer_lambda: float,
    repulsion_mu: float,
    impulse_reference: float,
    size_bias_beta: float,
    collision_repulsion_boost: float,
    collision_coupling_gain: float,
) -> dict[str, Any]:
    matrix = seed_curr_matrix_fn(left, right)
    semantic_affinity = (
        (matrix["cc"] * 0.5)
        + (((matrix["sc"] + matrix["cs"]) * 0.5) * 0.3)
        + (matrix["ss"] * 0.2)
    )
    semantic_affinity = finite_float(semantic_affinity, 0.0)
    transfer_t = clamp01_finite((semantic_affinity + 1.0) * 0.5, 0.5)
    repulsion_u = clamp01_finite(((-semantic_affinity) + 1.0) * 0.5, 0.5)
    if semantic_affinity < -0.5:
        repulsion_u = min(
            1.0,
            repulsion_u * safe_float(collision_repulsion_boost, 1.9),
        )
    intensity = clamp01_finite(finite_float(impulse, 0.0) / impulse_reference, 0.0)

    left_size = max(1e-8, finite_float(left.get("size", 1.0), 1.0))
    right_size = max(1e-8, finite_float(right.get("size", 1.0), 1.0))
    bias_left = sigmoid(size_bias_beta * math.log(right_size / left_size))
    bias_right = sigmoid(size_bias_beta * math.log(left_size / right_size))
    coupling_gain = safe_float(collision_coupling_gain, 0.62)
    coupling_left = clamp01(intensity * bias_left * coupling_gain)
    coupling_right = clamp01(intensity * bias_right * coupling_gain)
    coupling_left_01 = clamp01_finite(coupling_left, 0.0)
    coupling_right_01 = clamp01_finite(coupling_right, 0.0)
    transfer_t_01 = clamp01_finite(transfer_t, 0.0)
    repulsion_u_01 = clamp01_finite(repulsion_u, 0.0)
    left_delta = transfer_lambda * coupling_left_01 * transfer_t_01
    left_rho = repulsion_mu * coupling_left_01 * repulsion_u_01
    right_delta = transfer_lambda * coupling_right_01 * transfer_t_01
    right_rho = repulsion_mu * coupling_right_01 * repulsion_u_01

    left_seed = state_unit_vector(left, "e_seed")
    left_curr = state_unit_vector(left, "e_curr")
    right_seed = state_unit_vector(right, "e_seed")
    right_curr = state_unit_vector(right, "e_curr")

    trust_left = clamp01((matrix["self_left"] + 1.0) * 0.5)
    trust_right = clamp01((matrix["self_right"] + 1.0) * 0.5)
    left_export = blend_vectors(left_seed, left_curr, trust_left)
    right_export = blend_vectors(right_seed, right_curr, trust_right)

    next_left_curr = normalize_vector(
        [
            (left_curr[idx] * (1.0 - coupling_left))
            + (right_export[idx] * coupling_left)
            for idx in range(min(len(left_curr), len(right_export)))
        ]
    )
    next_right_curr = normalize_vector(
        [
            (right_curr[idx] * (1.0 - coupling_right))
            + (left_export[idx] * coupling_right)
            for idx in range(min(len(right_curr), len(left_export)))
        ]
    )

    left_alpha_pkg_raw = left.get("alpha_pkg", {})
    right_alpha_pkg_raw = right.get("alpha_pkg", {})

    resource_transfer: dict[str, Any] = {}

    left_emit = finite_float(left_alpha_pkg_raw.get("emit_resource_packet", 0.0), 0.0)
    left_absorb = finite_float(left_alpha_pkg_raw.get("absorb_resource", 0.0), 0.0)
    right_emit = finite_float(right_alpha_pkg_raw.get("emit_resource_packet", 0.0), 0.0)
    right_absorb = finite_float(right_alpha_pkg_raw.get("absorb_resource", 0.0), 0.0)

    action_threshold = alpha_baseline * 1.5

    if left_emit > action_threshold and right_absorb > action_threshold:
        owner_id = str(left.get("owner", ""))
        if "presence.core." in owner_id:
            res_type = owner_id.replace("presence.core.", "")
            amount = max(0.1, intensity * 5.0)
            resource_transfer["left_to_right"] = {res_type: amount}

    if right_emit > action_threshold and left_absorb > action_threshold:
        owner_id = str(right.get("owner", ""))
        if "presence.core." in owner_id:
            res_type = owner_id.replace("presence.core.", "")
            amount = max(0.1, intensity * 5.0)
            resource_transfer["right_to_left"] = {res_type: amount}

    left_alpha_pkg = left_alpha_pkg_raw if isinstance(left_alpha_pkg_raw, dict) else {}
    right_alpha_pkg = (
        right_alpha_pkg_raw if isinstance(right_alpha_pkg_raw, dict) else {}
    )

    if left_alpha_pkg.keys() <= job_keys_set and right_alpha_pkg.keys() <= job_keys_set:
        left_alpha_pkg_next: dict[str, float] = {}
        right_alpha_pkg_next: dict[str, float] = {}
        for key in job_keys_sorted:
            src = max(
                1e-8,
                finite_float(left_alpha_pkg.get(key, alpha_baseline), alpha_baseline),
            )
            tgt = max(
                1e-8,
                finite_float(right_alpha_pkg.get(key, alpha_baseline), alpha_baseline),
            )
            left_shifted = ((1.0 - left_rho) * (src + (left_delta * tgt))) + (
                left_rho * alpha_baseline
            )
            right_shifted = ((1.0 - right_rho) * (tgt + (right_delta * src))) + (
                right_rho * alpha_baseline
            )
            left_alpha_pkg_next[key] = min(
                alpha_max,
                max(1e-8, finite_float(left_shifted, alpha_baseline)),
            )
            right_alpha_pkg_next[key] = min(
                alpha_max,
                max(1e-8, finite_float(right_shifted, alpha_baseline)),
            )
    else:
        left_alpha_pkg_safe = {
            str(key): max(1e-8, finite_float(value, alpha_baseline))
            for key, value in dict(left_alpha_pkg).items()
        }
        right_alpha_pkg_safe = {
            str(key): max(1e-8, finite_float(value, alpha_baseline))
            for key, value in dict(right_alpha_pkg).items()
        }
        package_keys = tuple(
            sorted(
                set(
                    [
                        *left_alpha_pkg_safe.keys(),
                        *right_alpha_pkg_safe.keys(),
                        *job_keys,
                    ]
                )
            )
        )
        left_alpha_pkg_next = dirichlet_transfer_fn(
            left_alpha_pkg_safe,
            right_alpha_pkg_safe,
            coupling=coupling_left,
            transfer_t=transfer_t,
            repulsion_u=repulsion_u,
            keys=package_keys,
        )
        right_alpha_pkg_next = dirichlet_transfer_fn(
            right_alpha_pkg_safe,
            left_alpha_pkg_safe,
            coupling=coupling_right,
            transfer_t=transfer_t,
            repulsion_u=repulsion_u,
            keys=package_keys,
        )

    left_alpha_msg_raw = dict(left.get("alpha_msg", {}))
    right_alpha_msg_raw = dict(right.get("alpha_msg", {}))
    left_alpha_msg = {
        "deliver": max(
            1e-8,
            finite_float(
                left_alpha_msg_raw.get("deliver", alpha_baseline),
                alpha_baseline,
            ),
        ),
        "hold": max(
            1e-8,
            finite_float(
                left_alpha_msg_raw.get("hold", alpha_baseline),
                alpha_baseline,
            ),
        ),
    }
    right_alpha_msg = {
        "deliver": max(
            1e-8,
            finite_float(
                right_alpha_msg_raw.get("deliver", alpha_baseline),
                alpha_baseline,
            ),
        ),
        "hold": max(
            1e-8,
            finite_float(
                right_alpha_msg_raw.get("hold", alpha_baseline),
                alpha_baseline,
            ),
        ),
    }
    left_alpha_msg_next = {
        "deliver": min(
            alpha_max,
            max(
                1e-8,
                finite_float(
                    (
                        (1.0 - left_rho)
                        * (
                            left_alpha_msg["deliver"]
                            + (left_delta * right_alpha_msg["deliver"])
                        )
                    )
                    + (left_rho * alpha_baseline),
                    alpha_baseline,
                ),
            ),
        ),
        "hold": min(
            alpha_max,
            max(
                1e-8,
                finite_float(
                    (
                        (1.0 - left_rho)
                        * (
                            left_alpha_msg["hold"]
                            + (left_delta * right_alpha_msg["hold"])
                        )
                    )
                    + (left_rho * alpha_baseline),
                    alpha_baseline,
                ),
            ),
        ),
    }
    right_alpha_msg_next = {
        "deliver": min(
            alpha_max,
            max(
                1e-8,
                finite_float(
                    (
                        (1.0 - right_rho)
                        * (
                            right_alpha_msg["deliver"]
                            + (right_delta * left_alpha_msg["deliver"])
                        )
                    )
                    + (right_rho * alpha_baseline),
                    alpha_baseline,
                ),
            ),
        ),
        "hold": min(
            alpha_max,
            max(
                1e-8,
                finite_float(
                    (
                        (1.0 - right_rho)
                        * (
                            right_alpha_msg["hold"]
                            + (right_delta * left_alpha_msg["hold"])
                        )
                    )
                    + (right_rho * alpha_baseline),
                    alpha_baseline,
                ),
            ),
        ),
    }

    left["e_curr"] = next_left_curr
    right["e_curr"] = next_right_curr
    left["alpha_pkg"] = left_alpha_pkg_next
    right["alpha_pkg"] = right_alpha_pkg_next
    left["alpha_msg"] = left_alpha_msg_next
    right["alpha_msg"] = right_alpha_msg_next
    left["last_collision_matrix"] = {
        "ss": round(matrix["ss"], 6),
        "sc": round(matrix["sc"], 6),
        "cs": round(matrix["cs"], 6),
        "cc": round(matrix["cc"], 6),
    }
    right["last_collision_matrix"] = dict(left["last_collision_matrix"])

    return {
        "ss": matrix["ss"],
        "sc": matrix["sc"],
        "cs": matrix["cs"],
        "cc": matrix["cc"],
        "semantic_affinity": semantic_affinity,
        "transfer": transfer_t,
        "repulsion": repulsion_u,
        "intensity": intensity,
        "resource_transfer": resource_transfer,
    }
