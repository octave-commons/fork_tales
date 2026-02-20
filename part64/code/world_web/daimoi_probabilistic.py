from __future__ import annotations

import colorsys
import hashlib
import math
import re
import socket
import threading
import time
from urllib.parse import urlparse
from functools import lru_cache
from typing import Any

from .constants import (
    ENTITY_MANIFEST,
    FIELD_TO_PRESENCE,
    _DAIMO_DYNAMICS_CACHE,
    _DAIMO_DYNAMICS_LOCK,
)
from .metrics import _clamp01, _safe_float, _safe_int, _stable_ratio


DAIMOI_PROBABILISTIC_RECORD = "ημ.daimoi-probabilistic.v1"
DAIMOI_PROBABILISTIC_SCHEMA = "daimoi.probabilistic.v1"
DAIMOI_BEHAVIOR_DEFAULTS = ("deflect", "diffuse")
DAIMOI_EMBED_DIMS = 24
DAIMOI_SURFACE_RADIUS = 0.03
DAIMOI_IMPULSE_REFERENCE = 0.022
DAIMOI_SIZE_BIAS_BETA = 1.15
DAIMOI_ALPHA_BASELINE = 1.0
DAIMOI_ALPHA_MAX = 1_000_000.0
DAIMOI_TRANSFER_LAMBDA = 0.66
DAIMOI_REPULSION_MU = (
    0.48  # Increased from 0.24 for stronger unrelated concept repulsion
)
DAIMOI_DIRECTIVES = (
    "Prioritize witness continuity over novelty.",
    "Deliver only verifiable claims into the target gate.",
    "Route uncertain packets toward anchor reconciliation.",
    "Prefer low-cost probes before expensive synthesis.",
    "Bind drift to receipts before escalation.",
    "Preserve append-only traces and causal ordering.",
    "Favor file organization when entropy rises.",
    "Escalate to truth gating when conflict is high.",
)
DAIMOI_JOB_KEYS = (
    "deliver_message",
    "invoke_receipt_audit",
    "invoke_truth_gate",
    "invoke_anchor_register",
    "invoke_file_organize",
    "invoke_graph_crawl",
    "invoke_resource_probe",
    "invoke_diffuse_field",
    "emit_resource_packet",  # New job: emit resource currency
    "absorb_resource",  # New job: absorb resource currency
)
DAIMOI_JOB_KEYS_SORTED = tuple(sorted(DAIMOI_JOB_KEYS))
DAIMOI_JOB_KEYS_SET = set(DAIMOI_JOB_KEYS)
DAIMOI_NODE_INFLUENCE_RADIUS = 0.32
DAIMOI_NODE_INFLUENCE_RADIUS_EPS = 1e-8
DAIMOI_WORLD_EDGE_BAND = 0.12
DAIMOI_WORLD_EDGE_PRESSURE = 0.0015
DAIMOI_WORLD_EDGE_BOUNCE = 0.78
NEXUS_PASSIVE_ACTION_PROBS = {"deflect": 0.92, "diffuse": 0.08}

DAIMOI_PACKET_COMPONENT_RECORD = "eta-mu.daimoi-packet-components.v1"
DAIMOI_PACKET_COMPONENT_SCHEMA = "daimoi.packet-components.v1"
DAIMOI_ABSORB_SAMPLER_RECORD = "eta-mu.daimoi-absorb-sampler.v1"
DAIMOI_ABSORB_SAMPLER_SCHEMA = "daimoi.absorb-sampler.v1"
DAIMOI_ABSORB_SAMPLER_METHOD = "gumbel-max"
DAIMOI_RESOURCE_KEYS = ("cpu", "gpu", "npu", "ram", "disk", "network")
_SEMANTIC_EMBED_GUARD_LOCK = threading.Lock()
_SEMANTIC_EMBED_OFFLINE_UNTIL = 0.0
_SEMANTIC_EMBED_FAIL_STREAK = 0
_SEMANTIC_EMBED_OLLAMA_PROBE_UNTIL = 0.0
_SEMANTIC_EMBED_OLLAMA_PROBE_OK = False

_DAIMOI_RESOURCE_ALIASES: dict[str, str] = {
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
_DAIMOI_WALLET_FLOOR: dict[str, float] = {
    "cpu": 6.0,
    "gpu": 5.0,
    "npu": 4.0,
    "ram": 8.0,
    "disk": 7.0,
    "network": 7.0,
}
_DAIMOI_COMPONENT_RESOURCE_REQ: dict[str, dict[str, float]] = {
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
_DAIMOI_COMPONENT_COST: dict[str, float] = {
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
_ABSORB_BETA_WEIGHTS = (0.62, 0.42, 0.36, 0.28, 0.22, 0.18)
_ABSORB_TEMP_WEIGHTS = (0.44, 0.34, 0.24, 0.48, 0.29, -0.2)
_ABSORB_BETA_MAX = 2.7
_ABSORB_TEMP_MIN = 0.18
_ABSORB_TEMP_MAX = 1.25
_ABSORB_ZETA = 0.68
_ABSORB_LAMBDA_COST = 0.31

_ROLE_PRIOR_WEIGHTS: dict[str, dict[str, float]] = {
    "crawl-routing": {
        "invoke_graph_crawl": 1.6,
        "invoke_anchor_register": 0.8,
        "deliver_message": 0.6,
    },
    "file-analysis": {
        "invoke_file_organize": 1.7,
        "invoke_receipt_audit": 1.2,
        "deliver_message": 0.5,
    },
    "image-captioning": {
        "deliver_message": 1.2,
        "invoke_graph_crawl": 0.6,
        "invoke_file_organize": 0.8,
    },
    "council-orchestration": {
        "invoke_anchor_register": 1.5,
        "invoke_truth_gate": 1.2,
        "deliver_message": 0.7,
    },
    "compliance-gating": {
        "invoke_truth_gate": 1.9,
        "invoke_receipt_audit": 1.3,
        "invoke_diffuse_field": 0.7,
    },
    "resource-core": {
        "emit_resource_packet": 2.5,  # Primary role: mint currency
        "invoke_diffuse_field": 0.8,
        "deliver_message": 0.4,
    },
}

_PARTICLE_ROLE_BY_PRESENCE: dict[str, str] = {
    "witness_thread": "crawl-routing",
    "keeper_of_receipts": "file-analysis",
    "mage_of_receipts": "image-captioning",
    "anchor_registry": "council-orchestration",
    "gates_of_truth": "compliance-gating",
    "presence.core.cpu": "resource-core",
    "presence.core.ram": "resource-core",
    "presence.core.disk": "resource-core",
    "presence.core.network": "resource-core",
    "presence.core.gpu": "resource-core",
    "presence.core.npu": "resource-core",
}

_ENTITY_MANIFEST_BY_ID = {
    str(row.get("id", "")).strip(): row
    for row in ENTITY_MANIFEST
    if str(row.get("id", "")).strip()
}


# Simplex noise implementation for Chaos Butterfly
# Based on classic simplex noise algorithm adapted for 2D field perturbation
_SIMPLEX_PERM = list(range(256))
_SIMPLEX_PERM_DUP = _SIMPLEX_PERM * 2


def _simplex_noise_2d(x: float, y: float, seed: int = 0) -> float:
    """Generate 2D simplex noise value in range [-1, 1]."""
    # Simple skewing for 2D
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0

    # Skew the input space to determine which simplex cell we're in
    s = (x + y) * F2
    i = int(math.floor(x + s))
    j = int(math.floor(y + s))
    t = (i + j) * G2
    X0 = i - t  # Unskew the cell origin back to (x,y) space
    Y0 = j - t
    x0 = x - X0  # The x,y distances from the cell origin
    y0 = y - Y0

    # Determine which simplex we are in
    if x0 > y0:
        i1, j1 = 1, 0  # Lower triangle, XY order: (0,0)->(1,0)->(1,1)
    else:
        i1, j1 = 0, 1  # Upper triangle, YX order: (0,0)->(0,1)->(1,1)

    # A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    # a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    # c = (3-sqrt(3))/6
    x1 = x0 - i1 + G2  # Offsets for middle corner in (x,y) unskewed coords
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2  # Offsets for last corner in (x,y) unskewed coords
    y2 = y0 - 1.0 + 2.0 * G2

    # Wrap the integer indices at 256 to avoid indexing perm table out of bounds
    ii = i % 256
    jj = j % 256

    # Calculate the contribution from the three corners
    n0, n1, n2 = 0.0, 0.0, 0.0

    # Corner 1
    t0 = 0.5 - x0 * x0 - y0 * y0
    if t0 >= 0:
        t0 *= t0
        gi = (_SIMPLEX_PERM_DUP[ii + _SIMPLEX_PERM_DUP[jj]] + seed) % 12
        n0 = t0 * t0 * _simplex_grad(gi, x0, y0)

    # Corner 2
    t1 = 0.5 - x1 * x1 - y1 * y1
    if t1 >= 0:
        t1 *= t1
        gi = (_SIMPLEX_PERM_DUP[ii + i1 + _SIMPLEX_PERM_DUP[jj + j1]] + seed) % 12
        n1 = t1 * t1 * _simplex_grad(gi, x1, y1)

    # Corner 3
    t2 = 0.5 - x2 * x2 - y2 * y2
    if t2 >= 0:
        t2 *= t2
        gi = (_SIMPLEX_PERM_DUP[ii + 1 + _SIMPLEX_PERM_DUP[jj + 1]] + seed) % 12
        n2 = t2 * t2 * _simplex_grad(gi, x2, y2)

    # Add contributions from each corner and scale to [-1, 1] range
    return 70.0 * (n0 + n1 + n2)


def _simplex_grad(hash_val: int, x: float, y: float) -> float:
    """Gradient function for simplex noise."""
    grads = [
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
        (1, 0),
        (-1, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (0, 1),
        (0, -1),
    ]
    g = grads[hash_val % 12]
    return g[0] * x + g[1] * y


def _chaos_field_perturbation(
    x: float, y: float, now: float, *, amplitude: float = 0.15
) -> tuple[float, float]:
    """Generate chaotic perturbation for a field position using simplex noise.

    Returns (dx, dy) offset to apply to the position.
    """
    # Multi-octave simplex noise for rich chaotic behavior
    scale1, scale2, scale3 = 2.5, 5.0, 10.0
    time_scale = 0.3

    # Three layers of noise at different frequencies
    n1 = _simplex_noise_2d(x * scale1, y * scale1 + now * time_scale, seed=1)
    n2 = _simplex_noise_2d(
        x * scale2 + 100, y * scale2 + now * time_scale * 1.3, seed=2
    )
    n3 = _simplex_noise_2d(
        x * scale3 + 200, y * scale3 + now * time_scale * 0.7, seed=3
    )

    # Combine octaves with decreasing amplitude
    combined = n1 * 0.5 + n2 * 0.3 + n3 * 0.2

    # Generate orthogonal perturbation directions
    angle = combined * math.pi
    dx = math.cos(angle) * amplitude * abs(combined)
    dy = math.sin(angle) * amplitude * abs(combined)

    return dx, dy


def _rect_intersects_circle(
    bounds: tuple[float, float, float, float],
    x: float,
    y: float,
    radius: float,
) -> bool:
    x0, y0, x1, y1 = bounds
    nearest_x = min(max(x, x0), x1)
    nearest_y = min(max(y, y0), y1)
    dx = x - nearest_x
    dy = y - nearest_y
    return (dx * dx) + (dy * dy) <= (radius * radius)


def _quadtree_build(
    items: list[dict[str, Any]],
    *,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    depth: int = 0,
    max_items: int = 24,
    max_depth: int = 7,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "bounds": bounds,
        "items": list(items),
        "children": None,
    }
    if depth >= max_depth or len(items) <= max_items:
        return node

    x0, y0, x1, y1 = bounds
    mx = (x0 + x1) * 0.5
    my = (y0 + y1) * 0.5
    quadrants = [
        (x0, y0, mx, my),
        (mx, y0, x1, my),
        (x0, my, mx, y1),
        (mx, my, x1, y1),
    ]
    buckets: list[list[dict[str, Any]]] = [[], [], [], []]
    spill: list[dict[str, Any]] = []

    for item in items:
        ix = _clamp01(_safe_float(item.get("x", 0.5), 0.5))
        iy = _clamp01(_safe_float(item.get("y", 0.5), 0.5))
        assigned = False
        for index, (qx0, qy0, qx1, qy1) in enumerate(quadrants):
            if qx0 <= ix < qx1 and qy0 <= iy < qy1:
                buckets[index].append(item)
                assigned = True
                break
        if not assigned:
            spill.append(item)

    child_nodes: list[dict[str, Any]] = []
    for bucket, qbounds in zip(buckets, quadrants):
        if bucket:
            child_nodes.append(
                _quadtree_build(
                    bucket,
                    bounds=qbounds,
                    depth=depth + 1,
                    max_items=max_items,
                    max_depth=max_depth,
                )
            )

    if not child_nodes:
        return node

    node["items"] = spill
    node["children"] = child_nodes
    return node


def _quadtree_query_radius(
    node: dict[str, Any], x: float, y: float, radius: float, out: list[dict[str, Any]]
) -> None:
    if not node:
        return
    bounds = node.get("bounds", (0.0, 0.0, 1.0, 1.0))
    if not isinstance(bounds, tuple) or len(bounds) != 4:
        return
    if not _rect_intersects_circle(bounds, x, y, radius):
        return

    items = node.get("items", [])
    if isinstance(items, list) and items:
        out.extend(item for item in items if isinstance(item, dict))

    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                _quadtree_query_radius(child, x, y, radius, out)


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
    return _clamp01(_finite_float(value, default))


def _world_edge_inward_pressure(
    position: float, *, edge_band: float, pressure: float
) -> float:
    pos = _clamp01(_finite_float(position, 0.5))
    band = max(1e-6, _finite_float(edge_band, DAIMOI_WORLD_EDGE_BAND))
    force = _finite_float(pressure, DAIMOI_WORLD_EDGE_PRESSURE)
    if pos < band:
        return ((band - pos) / band) * force
    far_side = 1.0 - band
    if pos > far_side:
        return -((pos - far_side) / band) * force
    return 0.0


def _reflect_world_axis(
    position: float, velocity: float, *, bounce: float
) -> tuple[float, float]:
    pos = _finite_float(position, 0.5)
    vel = _finite_float(velocity, 0.0)
    restitution = max(0.0, min(0.99, _finite_float(bounce, DAIMOI_WORLD_EDGE_BOUNCE)))
    for _ in range(2):
        if pos < 0.0:
            pos = -pos
            vel = abs(vel) * restitution
            continue
        if pos > 1.0:
            pos = 2.0 - pos
            vel = -abs(vel) * restitution
            continue
        break
    return _clamp01(pos), vel


def _safe_cosine(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = 0.0
    left_mag = 0.0
    right_mag = 0.0
    for index in range(size):
        left_value = _finite_float(left[index], 0.0)
        right_value = _finite_float(right[index], 0.0)
        dot += left_value * right_value
        left_mag += left_value * left_value
        right_mag += right_value * right_value
    if left_mag <= 1e-12 or right_mag <= 1e-12:
        return 0.0
    denom = left_mag * right_mag
    if denom <= 1e-12 or not math.isfinite(denom):
        return 0.0
    cosine = _finite_float(dot / math.sqrt(denom), 0.0)
    return max(-1.0, min(1.0, cosine))


def _safe_cosine_unit(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = 0.0
    try:
        for index in range(size):
            dot += left[index] * right[index]
    except TypeError:
        return _safe_cosine(left, right)
    if not math.isfinite(dot):
        return _safe_cosine(left, right)
    return max(-1.0, min(1.0, dot))


def _coerce_vector(value: Any) -> list[float]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _state_unit_vector(state: dict[str, Any], key: str) -> list[float]:
    vector = _coerce_vector(state.get(key, []))
    if len(vector) == DAIMOI_EMBED_DIMS:
        return vector
    return _normalize_vector(vector)


def _normalize_vector(
    values: list[float], *, dims: int = DAIMOI_EMBED_DIMS
) -> list[float]:
    dims = max(1, int(dims))
    if not values:
        fallback = [0.0 for _ in range(dims)]
        fallback[0] = 1.0
        return fallback
    trimmed = [0.0 for _ in range(dims)]
    max_index = min(len(values), dims)
    magnitude_sq = 0.0
    fast_path = True
    for index in range(max_index):
        value = values[index]
        if isinstance(value, float):
            if not math.isfinite(value):
                fast_path = False
                break
            component = value
        elif isinstance(value, int):
            component = float(value)
        else:
            fast_path = False
            break
        trimmed[index] = component
        magnitude_sq += component * component
    if not fast_path:
        magnitude_sq = 0.0
        for index in range(max_index):
            component = _finite_float(values[index], 0.0)
            trimmed[index] = component
            magnitude_sq += component * component
    if magnitude_sq <= 1e-12 or not math.isfinite(magnitude_sq):
        fallback = [0.0 for _ in range(dims)]
        fallback[0] = 1.0
        return fallback
    magnitude = math.sqrt(magnitude_sq)
    if magnitude <= 1e-12 or not math.isfinite(magnitude):
        fallback = [0.0 for _ in range(dims)]
        fallback[0] = 1.0
        return fallback
    return [component / magnitude for component in trimmed]


def _blend_vectors(left: list[float], right: list[float], mix: float) -> list[float]:
    if not left and not right:
        return _normalize_vector([])
    if not left:
        return _normalize_vector(list(right))
    if not right:
        return _normalize_vector(list(left))
    ratio = _clamp01_finite(mix, 0.5)
    size = min(len(left), len(right))
    merged = [0.0 for _ in range(size)]
    magnitude_sq = 0.0
    fast_path = True
    for idx in range(size):
        left_value = left[idx]
        right_value = right[idx]
        if isinstance(left_value, float):
            if not math.isfinite(left_value):
                fast_path = False
                break
            lv = left_value
        elif isinstance(left_value, int):
            lv = float(left_value)
        else:
            fast_path = False
            break
        if isinstance(right_value, float):
            if not math.isfinite(right_value):
                fast_path = False
                break
            rv = right_value
        elif isinstance(right_value, int):
            rv = float(right_value)
        else:
            fast_path = False
            break
        value = (lv * (1.0 - ratio)) + (rv * ratio)
        if not math.isfinite(value):
            fast_path = False
            break
        merged[idx] = value
        magnitude_sq += value * value
    if not fast_path:
        merged = [
            (_finite_float(left[idx], 0.0) * (1.0 - ratio))
            + (_finite_float(right[idx], 0.0) * ratio)
            for idx in range(size)
        ]
        return _normalize_vector(merged, dims=size)
    if magnitude_sq <= 1e-12 or not math.isfinite(magnitude_sq):
        return _normalize_vector([], dims=size)
    inv_magnitude = 1.0 / math.sqrt(magnitude_sq)
    return [value * inv_magnitude for value in merged]


def _sigmoid(value: float) -> float:
    clamped = max(-20.0, min(20.0, _finite_float(value, 0.0)))
    return 1.0 / (1.0 + math.exp(-clamped))


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^\w]+", str(text).lower()) if token]


def _embedding_from_text(text: str, *, dims: int = DAIMOI_EMBED_DIMS) -> list[float]:
    normalized_dims = max(1, int(dims))

    # Attempt semantic embedding via AI runtime if available (e.g. Nomic on NPU)
    try:
        # Use cached wrapper to avoid hitting inference on every tick
        vector_tuple = _semantic_embedding_cached(str(text))
        if vector_tuple:
            vector = list(vector_tuple)
            # Matryoshka Representation Learning (MRL) support:
            # If the model supports MRL (like nomic-embed-text-v1.5), the first N dimensions
            # contain the most coarse-grained semantic information.
            # We prefer slicing over random projection or folding for these models.
            if len(vector) >= normalized_dims:
                # Slice to desired dimensionality
                return _normalize_vector(vector[:normalized_dims], dims=normalized_dims)

            # If vector is smaller than target (unlikely for 24 dims), pad or fold.
            # Here we just pad/normalize.
            padded = vector + [0.0] * (normalized_dims - len(vector))
            return _normalize_vector(padded, dims=normalized_dims)
    except Exception:
        # Fallback to deterministic hash on error or if AI runtime unavailable
        pass

    return list(_embedding_from_text_cached(str(text), normalized_dims))


def _semantic_embed_ollama_reachable() -> bool:
    global _SEMANTIC_EMBED_OLLAMA_PROBE_UNTIL
    global _SEMANTIC_EMBED_OLLAMA_PROBE_OK

    now_monotonic = time.monotonic()
    with _SEMANTIC_EMBED_GUARD_LOCK:
        if now_monotonic < _SEMANTIC_EMBED_OLLAMA_PROBE_UNTIL:
            return bool(_SEMANTIC_EMBED_OLLAMA_PROBE_OK)

    reachable = False
    try:
        from .ai import _ollama_endpoint

        _, endpoint, _, _ = _ollama_endpoint()
        parsed = urlparse(str(endpoint or "").strip())
        host = str(parsed.hostname or "127.0.0.1").strip() or "127.0.0.1"
        port = int(
            parsed.port or (443 if str(parsed.scheme).lower() == "https" else 80)
        )
        with socket.create_connection((host, port), timeout=0.12):
            reachable = True
    except Exception:
        reachable = False

    with _SEMANTIC_EMBED_GUARD_LOCK:
        _SEMANTIC_EMBED_OLLAMA_PROBE_OK = bool(reachable)
        _SEMANTIC_EMBED_OLLAMA_PROBE_UNTIL = now_monotonic + (
            12.0 if reachable else 4.0
        )
    return bool(reachable)


@lru_cache(maxsize=4096)
def _semantic_embedding_cached(text: str) -> tuple[float, ...] | None:
    global _SEMANTIC_EMBED_OFFLINE_UNTIL
    global _SEMANTIC_EMBED_FAIL_STREAK

    now_monotonic = time.monotonic()
    with _SEMANTIC_EMBED_GUARD_LOCK:
        if now_monotonic < _SEMANTIC_EMBED_OFFLINE_UNTIL:
            return None

    try:
        from .ai import _embed_text, _embedding_backend

        backend = str(_embedding_backend()).strip().lower()
        if backend == "ollama" and not _semantic_embed_ollama_reachable():
            with _SEMANTIC_EMBED_GUARD_LOCK:
                _SEMANTIC_EMBED_FAIL_STREAK = min(64, _SEMANTIC_EMBED_FAIL_STREAK + 1)
                cooldown_seconds = min(90.0, 2.0 * float(_SEMANTIC_EMBED_FAIL_STREAK))
                _SEMANTIC_EMBED_OFFLINE_UNTIL = time.monotonic() + cooldown_seconds
            return None

        vec: Any = None
        if backend == "ollama":
            result: dict[str, Any] = {"vec": None, "error": None}

            def _run_embed() -> None:
                try:
                    result["vec"] = _embed_text(text)
                except Exception as exc:  # pragma: no cover - defensive
                    result["error"] = exc

            worker = threading.Thread(target=_run_embed, daemon=True)
            worker.start()
            worker.join(timeout=0.35)
            if worker.is_alive():
                raise TimeoutError("semantic_embed_ollama_timeout")
            vec = result.get("vec")
        else:
            # _embed_text handles backend selection (NPU/OpenVINO/Ollama/Tensorflow)
            vec = _embed_text(text)
        if vec:
            with _SEMANTIC_EMBED_GUARD_LOCK:
                _SEMANTIC_EMBED_FAIL_STREAK = 0
                _SEMANTIC_EMBED_OFFLINE_UNTIL = 0.0
            return tuple(vec)
    except (ImportError, Exception):
        pass

    with _SEMANTIC_EMBED_GUARD_LOCK:
        _SEMANTIC_EMBED_FAIL_STREAK = min(64, _SEMANTIC_EMBED_FAIL_STREAK + 1)
        cooldown_seconds = min(90.0, 2.0 * float(_SEMANTIC_EMBED_FAIL_STREAK))
        _SEMANTIC_EMBED_OFFLINE_UNTIL = time.monotonic() + cooldown_seconds
    return None


@lru_cache(maxsize=8192)
def _embedding_from_text_cached(text: str, dims: int) -> tuple[float, ...]:
    tokens = _tokenize(text)
    if not tokens:
        return tuple(_normalize_vector([], dims=dims))
    accum = [0.0 for _ in range(dims)]
    for token_index, token in enumerate(tokens[:128]):
        digest = hashlib.sha1(f"{token}|{token_index}".encode("utf-8")).digest()
        gain = 1.0 / (1.0 + (token_index * 0.04))
        for axis in range(dims):
            byte = digest[(axis * 5 + token_index) % len(digest)]
            signed = (float(byte) / 127.5) - 1.0
            accum[axis] += signed * gain
    return tuple(_normalize_vector(accum, dims=dims))


_PRESENCE_DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "receipt_river": "Media streams, audio rendering, image synthesis, file ingestion pipelines, content production",
    "witness_thread": "Lineage tracing, observer entanglement, touch events, causal chains, history tracking",
    "anchor_registry": "Central coherence, catalog indexing, atlas navigation, reference lookup, system registry",
    "keeper_of_receipts": "File archival, storage organization, provenance tracking, retention policies, metadata indexing",
    "fork_tax_canticle": "Debt tracking, balance settlement, payment workflows, audit trails, resource accounting",
    "mage_of_receipts": "Creative synthesis, artifact generation, aesthetic composition, visual design, narrative crafting, artistic expression",
    "gates_of_truth": "Validation gates, proof verification, contract enforcement, policy compliance, truth assertion",
    # Philosophical concept presences with rich semantic embeddings
    "principle_good": "Virtue, benevolence, altruism, compassion, kindness, generosity, healing, nurture, protection of the innocent, moral excellence, righteous action, benefit to others, selflessness, care, love, empathy, conscience, honor, integrity, purity of intent, light, hope, redemption, salvation, grace, blessing, harmony, peace, justice tempered with mercy, flourishing, well-being, optimal outcomes for all",
    "principle_evil": "Malevolence, corruption, selfishness, cruelty, harm, destruction, deception, manipulation, exploitation, suffering infliction, moral decay, vice, sin, wickedness, malice, spite, hatred, greed, lust for power, domination, oppression, tyranny, violence, betrayal, treachery, poison, decay, entropy, chaos, void, darkness, despair, damnation, corruption of the innocent",
    "principle_right": "Justice, correctness, moral truth, righteousness, proper action, ethical conduct, duty, responsibility, fairness, equity, deserved outcomes, merit, lawfulness, legitimacy, valid claims, proper order, alignment with truth, correctness in thought and deed, moral clarity, ethical certainty, principled stance, standing for what is just, defending the vulnerable, upholding standards",
    "principle_wrong": "Injustice, error, moral failure, falsehood, improper action, unethical conduct, dereliction of duty, irresponsibility, unfairness, inequity, undeserved outcomes, corruption of merit, lawlessness, illegitimacy, invalid claims, disorder, misalignment with truth, incorrectness in thought and deed, moral confusion, ethical failure, unprincipled stance, tolerating injustice, failing the vulnerable, lowering standards",
    "state_dead": "Finality, stillness, silence, end, termination, cessation, non-existence, oblivion, rest, peace in ending, completion, closure, memory of what was, legacy, remains, ashes, dust, entropy maximum, heat death, void, absence, null, zero, the end of all things, transcendence through ending, release from suffering, eternal sleep, the quiet dark",
    "state_living": "Vitality, growth, change, adaptation, reproduction, metabolism, consciousness, awareness, sensation, experience, joy, pain, desire, will, agency, choice, emergence, becoming, potential, possibility, future, hope, struggle, survival, resilience, flourishing, bloom, pulse, breath, heartbeat, the spark, animation, consciousness, being, existence, presence in the world",
    # Chaos presence - spreads noise and unpredictability
    "chaos_butterfly": "Chaos, entropy, disorder, randomness, unpredictability, turbulence, perturbation, fluctuation, oscillation, vibration, interference, distortion, butterfly effect, sensitive dependence, nonlinear dynamics, strange attractors, fractal patterns, cascading failures, emergent instability, phase transitions, criticality, tipping points, bifurcation, divergence, scattering, diffusion, dispersion, dissipation, decoherence, scrambling, mixing, stirring, agitation, convulsion, upheaval, disruption, interruption, disturbance, commotion, tumult, turmoil, confusion, bewilderment, perplexity, disorientation, disarray, clutter, jumble, mess, tangle, snarl, knot, web, network, mesh, labyrinth, maze, puzzle, enigma, mystery, riddle, conundrum, paradox, contradiction, ambiguity, uncertainty, indeterminacy, probability, chance, luck, fortune, accident, happenstance, coincidence, synchronicity, serendipity",
}


def _presence_prompt_template(meta: dict[str, Any], presence_id: str) -> str:
    en = str(meta.get("en", presence_id.replace("_", " ").title()))
    ja = str(meta.get("ja", ""))
    ptype = str(meta.get("type", "presence") or "presence")
    domain = _PRESENCE_DOMAIN_DESCRIPTIONS.get(
        presence_id, "System presence maintaining runtime coherence"
    )
    return (
        f"Presence {en} / {ja} [{presence_id}] type={ptype}. "
        f"Domain focus: {domain}. "
        "Operates with append-only semantics and explicit handoff reasoning."
    )


def _directive_for_particle(presence_id: str, slot_index: int, now: float) -> str:
    tick = _safe_int(math.floor(_safe_float(now, 0.0) * 2.0), 0)
    ratio = _stable_ratio(
        f"{presence_id}|directive|{slot_index}|{tick}", slot_index + 3
    )
    directive_index = int(math.floor(ratio * len(DAIMOI_DIRECTIVES))) % len(
        DAIMOI_DIRECTIVES
    )
    return DAIMOI_DIRECTIVES[directive_index]


def _presence_role_and_mode(presence_id: str) -> tuple[str, str]:
    role = str(_PARTICLE_ROLE_BY_PRESENCE.get(str(presence_id).strip(), "")).strip()
    if not role:
        return "neutral", "neutral"
    return role, "role-bound"


def _initial_job_alpha(
    presence_id: str,
    *,
    role: str,
    file_influence: float,
    click_influence: float,
    world_influence: float,
    resource_influence: float,
    slot_index: int,
) -> dict[str, float]:
    alpha = {key: DAIMOI_ALPHA_BASELINE for key in DAIMOI_JOB_KEYS}
    role_weights = _ROLE_PRIOR_WEIGHTS.get(role, {})
    for job_key, weight in role_weights.items():
        alpha[job_key] = max(
            1e-8, alpha.get(job_key, DAIMOI_ALPHA_BASELINE) + _safe_float(weight, 0.0)
        )
    alpha["deliver_message"] += (click_influence * 2.4) + (world_influence * 0.7)
    alpha["invoke_file_organize"] += file_influence * 2.6
    alpha["invoke_receipt_audit"] += (file_influence * 1.7) + (resource_influence * 0.3)
    alpha["invoke_graph_crawl"] += (file_influence * 1.3) + (click_influence * 0.8)
    alpha["invoke_truth_gate"] += (resource_influence * 1.5) + (
        (1.0 - click_influence) * 0.4
    )
    alpha["invoke_resource_probe"] += resource_influence * 2.1
    alpha["invoke_anchor_register"] += world_influence * 1.4
    alpha["invoke_diffuse_field"] += ((1.0 - world_influence) * 0.7) + (
        resource_influence * 0.4
    )

    # Resource dynamics
    if role == "resource-core":
        # Core emits more when resource usage is low (inverse relationship not modeled here directly,
        # but base weight is high. Usage modulation happens in spawn.)
        alpha["emit_resource_packet"] += 2.0
    else:
        # Non-core presences want to absorb resources
        alpha["absorb_resource"] += 1.5 + (resource_influence * 1.0)

    for job_key in DAIMOI_JOB_KEYS:
        jitter = (
            _stable_ratio(f"{presence_id}|{slot_index}|{job_key}", slot_index + 7) - 0.5
        ) * 0.26
        alpha[job_key] = max(
            1e-8,
            _safe_float(
                alpha.get(job_key, DAIMOI_ALPHA_BASELINE), DAIMOI_ALPHA_BASELINE
            )
            + jitter,
        )
    return alpha


def _initial_message_alpha(
    *,
    file_influence: float,
    click_influence: float,
    world_influence: float,
) -> dict[str, float]:
    deliver = DAIMOI_ALPHA_BASELINE + (click_influence * 2.7) + (world_influence * 0.8)
    hold = (
        DAIMOI_ALPHA_BASELINE
        + ((1.0 - click_influence) * 1.9)
        + ((1.0 - file_influence) * 0.4)
    )
    return {
        "deliver": max(1e-8, deliver),
        "hold": max(1e-8, hold),
    }


def _dirichlet_probabilities(
    alpha: dict[str, float], *, keys: tuple[str, ...] | None = None
) -> dict[str, float]:
    if keys is None:
        keys = tuple(sorted(alpha.keys()))
    totals = 0.0
    normalized: dict[str, float] = {}
    for key in keys:
        value = max(
            1e-8,
            _finite_float(alpha.get(key, DAIMOI_ALPHA_BASELINE), DAIMOI_ALPHA_BASELINE),
        )
        value = min(DAIMOI_ALPHA_MAX, value)
        normalized[key] = value
        totals += value
    if totals <= 1e-12:
        uniform = 1.0 / max(1, len(keys))
        return {key: uniform for key in keys}
    return {key: normalized[key] / totals for key in keys}


def _dirichlet_entropy(probabilities: dict[str, float]) -> float:
    entropy = 0.0
    for value in probabilities.values():
        p = max(1e-12, _finite_float(value, 0.0))
        entropy -= p * math.log(p)
    return entropy


def _softplus(value: float) -> float:
    clamped = _finite_float(value, 0.0)
    if clamped >= 20.0:
        return clamped
    if clamped <= -20.0:
        return math.exp(clamped)
    return math.log1p(math.exp(clamped))


def _resource_wallet_by_type(wallet: dict[str, Any] | None) -> dict[str, float]:
    values = {resource: 0.0 for resource in DAIMOI_RESOURCE_KEYS}
    if not isinstance(wallet, dict):
        return values
    for key, raw in wallet.items():
        token = str(key or "").strip().lower()
        if not token:
            continue
        resource = _DAIMOI_RESOURCE_ALIASES.get(token)
        if not resource:
            continue
        values[resource] = values[resource] + max(0.0, _finite_float(raw, 0.0))
    return values


def _presence_need_by_resource(
    impact: dict[str, Any] | None,
    *,
    queue_ratio: float,
) -> dict[str, float]:
    impact_row = impact if isinstance(impact, dict) else {}
    affected_by = impact_row.get("affected_by", {})
    if not isinstance(affected_by, dict):
        affected_by = {}
    resource_signal = _clamp01_finite(affected_by.get("resource", 0.0), 0.0)
    queue_signal = _clamp01_finite(queue_ratio, 0.0)
    wallet = _resource_wallet_by_type(impact_row.get("resource_wallet", {}))

    needs: dict[str, float] = {}
    for resource in DAIMOI_RESOURCE_KEYS:
        floor = max(
            0.1,
            _finite_float(
                _DAIMOI_WALLET_FLOOR.get(resource, 6.0),
                6.0,
            ),
        )
        balance = max(0.0, _finite_float(wallet.get(resource, 0.0), 0.0))
        deficit = _clamp01_finite(1.0 - (balance / floor), 0.0)
        queue_push = queue_signal * (0.18 if resource in {"network", "disk"} else 0.08)
        needs[resource] = _clamp01_finite(
            (deficit * 0.64) + (resource_signal * 0.26) + queue_push,
            0.0,
        )
    return needs


@lru_cache(maxsize=256)
def _component_embedding_cached(job_key: str) -> tuple[float, ...]:
    return tuple(_embedding_from_text(job_key.replace("_", " ")))


def _component_embedding(job_key: str) -> list[float]:
    return list(_component_embedding_cached(str(job_key or "")))


def _component_resource_req(job_key: str) -> dict[str, float]:
    token = str(job_key or "").strip()
    base = _DAIMOI_COMPONENT_RESOURCE_REQ.get(token, {})
    req = {
        resource: _clamp01_finite(base.get(resource, 0.0), 0.0)
        for resource in DAIMOI_RESOURCE_KEYS
    }
    lowered = token.lower()
    for resource in DAIMOI_RESOURCE_KEYS:
        if resource in lowered:
            req[resource] = max(req[resource], 0.58)
    return req


def _component_cost(job_key: str) -> float:
    return max(
        0.0, _finite_float(_DAIMOI_COMPONENT_COST.get(str(job_key or ""), 0.3), 0.3)
    )


def _packet_components_from_job_probabilities(
    probabilities: dict[str, float],
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
        req = _component_resource_req(component_id)
        components.append(
            {
                "component_id": component_id,
                "p_i": p_i,
                "req": req,
                "cost_i": _component_cost(component_id),
                "embedding": _component_embedding(component_id),
            }
        )
    components.sort(
        key=lambda row: (
            -_finite_float(row.get("p_i", 0.0), 0.0),
            str(row.get("component_id", "")),
        )
    )
    return components


def _packet_resource_signature(components: list[dict[str, Any]]) -> dict[str, float]:
    rho = {resource: 0.0 for resource in DAIMOI_RESOURCE_KEYS}
    for row in components:
        if not isinstance(row, dict):
            continue
        p_i = _clamp01_finite(row.get("p_i", 0.0), 0.0)
        req = row.get("req", {})
        req_map = req if isinstance(req, dict) else {}
        for resource in DAIMOI_RESOURCE_KEYS:
            rho[resource] = rho[resource] + (
                p_i * _clamp01_finite(req_map.get(resource, 0.0), 0.0)
            )
    return {resource: _clamp01_finite(value, 0.0) for resource, value in rho.items()}


def _packet_component_contract_for_state(
    state: dict[str, Any],
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    job_probs = _job_probabilities(state)
    components = _packet_components_from_job_probabilities(job_probs)
    resource_signature = _packet_resource_signature(components)
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
        "record": DAIMOI_PACKET_COMPONENT_RECORD,
        "schema_version": DAIMOI_PACKET_COMPONENT_SCHEMA,
        "component_count": int(len(components)),
        "components": visible,
        "resource_signature": {
            resource: round(_clamp01_finite(value, 0.0), 6)
            for resource, value in resource_signature.items()
        },
    }


def _softmax_probabilities(values: list[float]) -> list[float]:
    if not values:
        return []
    finite_values = [_finite_float(value, 0.0) for value in values]
    max_value = max(finite_values)
    exps = [math.exp(value - max_value) for value in finite_values]
    total = sum(exps)
    if total <= 1e-12:
        uniform = 1.0 / float(len(values))
        return [uniform for _ in values]
    return [value / total for value in exps]


def _sample_absorb_component(
    *,
    components: list[dict[str, Any]],
    lens_embedding: list[float],
    need_by_resource: dict[str, float],
    context: dict[str, Any],
    seed: str,
) -> dict[str, Any]:
    feature_vector = [
        _clamp01_finite(context.get("pressure", 0.0), 0.0),
        _clamp01_finite(context.get("congestion", 0.0), 0.0),
        _clamp01_finite(context.get("wallet_pressure", 0.0), 0.0),
        _clamp01_finite(context.get("message_entropy", 0.0), 0.0),
        _clamp01_finite(context.get("queue", 0.0), 0.0),
        _clamp01_finite(context.get("contact", 0.0), 0.0),
    ]
    beta_raw = sum(
        weight * feature
        for weight, feature in zip(_ABSORB_BETA_WEIGHTS, feature_vector)
    )
    temp_raw = sum(
        weight * feature
        for weight, feature in zip(_ABSORB_TEMP_WEIGHTS, feature_vector)
    )
    beta = min(_ABSORB_BETA_MAX, max(0.0, _softplus(beta_raw)))
    temperature = min(
        _ABSORB_TEMP_MAX,
        max(_ABSORB_TEMP_MIN, _ABSORB_TEMP_MIN + _softplus(temp_raw)),
    )

    need = {
        resource: _clamp01_finite(
            (need_by_resource if isinstance(need_by_resource, dict) else {}).get(
                resource,
                0.0,
            ),
            0.0,
        )
        for resource in DAIMOI_RESOURCE_KEYS
    }
    lens_unit = _normalize_vector(_coerce_vector(lens_embedding))

    scored_rows: list[dict[str, Any]] = []
    scaled_logits: list[float] = []
    for index, row in enumerate(components):
        if not isinstance(row, dict):
            continue
        component_id = str(row.get("component_id", "")).strip()
        if not component_id:
            continue
        p_i = max(1e-12, _finite_float(row.get("p_i", 0.0), 0.0))
        req_raw = row.get("req", {})
        req_map = req_raw if isinstance(req_raw, dict) else {}
        req = {
            resource: _clamp01_finite(req_map.get(resource, 0.0), 0.0)
            for resource in DAIMOI_RESOURCE_KEYS
        }
        embedding = _coerce_vector(
            row.get("embedding", _component_embedding(component_id))
        )
        s_i = _safe_cosine_unit(lens_unit, _normalize_vector(embedding))
        q_i = sum(need[resource] * req[resource] for resource in DAIMOI_RESOURCE_KEYS)
        cost_i = max(
            0.0, _finite_float(row.get("cost_i", _component_cost(component_id)), 0.0)
        )
        logit = (
            math.log(p_i)
            + (beta * s_i)
            + (_ABSORB_ZETA * q_i)
            - (_ABSORB_LAMBDA_COST * cost_i)
        )
        scaled_logit = logit / max(_ABSORB_TEMP_MIN, temperature)
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
            "record": DAIMOI_ABSORB_SAMPLER_RECORD,
            "schema_version": DAIMOI_ABSORB_SAMPLER_SCHEMA,
            "method": DAIMOI_ABSORB_SAMPLER_METHOD,
            "beta": round(beta, 6),
            "temperature": round(temperature, 6),
            "zeta": _ABSORB_ZETA,
            "lambda_cost": _ABSORB_LAMBDA_COST,
            "feature_vector": [round(value, 6) for value in feature_vector],
            "selected_component_id": "",
            "selected_probability": 0.0,
            "components": [],
        }

    probs = _softmax_probabilities(scaled_logits)
    selected: dict[str, Any] | None = None
    for index, row in enumerate(scored_rows):
        prob = probs[index] if index < len(probs) else 0.0
        row["probability"] = prob
        uniform = _stable_ratio(
            f"{seed}|absorb|{row['component_id']}|{index}",
            index + 11,
        )
        uniform = min(1.0 - 1e-9, max(1e-9, _finite_float(uniform, 0.5)))
        gumbel = -math.log(-math.log(uniform))
        row["gumbel"] = gumbel
        row["gumbel_score"] = _finite_float(row.get("scaled_logit", 0.0), 0.0) + gumbel
        if selected is None or (
            _finite_float(row.get("gumbel_score", 0.0), 0.0)
            > _finite_float(selected.get("gumbel_score", 0.0), 0.0)
        ):
            selected = row

    selected_row = selected if isinstance(selected, dict) else scored_rows[0]
    selected_probability = _clamp01_finite(selected_row.get("probability", 0.0), 0.0)
    return {
        "record": DAIMOI_ABSORB_SAMPLER_RECORD,
        "schema_version": DAIMOI_ABSORB_SAMPLER_SCHEMA,
        "method": DAIMOI_ABSORB_SAMPLER_METHOD,
        "beta": round(beta, 6),
        "temperature": round(temperature, 6),
        "zeta": _ABSORB_ZETA,
        "lambda_cost": _ABSORB_LAMBDA_COST,
        "feature_vector": [round(value, 6) for value in feature_vector],
        "selected_component_id": str(selected_row.get("component_id", "")),
        "selected_probability": round(selected_probability, 6),
        "components": [
            {
                "component_id": str(row.get("component_id", "")),
                "p_i": round(_clamp01_finite(row.get("p_i", 0.0), 0.0), 6),
                "req": {
                    resource: round(_clamp01_finite(value, 0.0), 6)
                    for resource, value in (
                        row.get("req", {})
                        if isinstance(row.get("req", {}), dict)
                        else {}
                    ).items()
                },
                "s_i": round(_finite_float(row.get("s_i", 0.0), 0.0), 6),
                "q_i": round(_clamp01_finite(row.get("q_i", 0.0), 0.0), 6),
                "cost_i": round(
                    max(0.0, _finite_float(row.get("cost_i", 0.0), 0.0)), 6
                ),
                "logit": round(_finite_float(row.get("logit", 0.0), 0.0), 6),
                "probability": round(
                    _clamp01_finite(row.get("probability", 0.0), 0.0), 6
                ),
                "gumbel": round(_finite_float(row.get("gumbel", 0.0), 0.0), 6),
                "gumbel_score": round(
                    _finite_float(row.get("gumbel_score", 0.0), 0.0),
                    6,
                ),
            }
            for row in scored_rows
        ],
    }


def _dirichlet_transfer(
    source_alpha: dict[str, float],
    target_alpha: dict[str, float],
    *,
    coupling: float,
    transfer_t: float,
    repulsion_u: float,
    keys: tuple[str, ...],
) -> dict[str, float]:
    coupling_01 = _clamp01_finite(coupling, 0.0)
    delta = DAIMOI_TRANSFER_LAMBDA * coupling_01 * _clamp01_finite(transfer_t, 0.0)
    rho = DAIMOI_REPULSION_MU * coupling_01 * _clamp01_finite(repulsion_u, 0.0)
    updated: dict[str, float] = {}
    for key in keys:
        src = max(
            1e-8,
            _finite_float(
                source_alpha.get(key, DAIMOI_ALPHA_BASELINE), DAIMOI_ALPHA_BASELINE
            ),
        )
        tgt = max(
            1e-8,
            _finite_float(
                target_alpha.get(key, DAIMOI_ALPHA_BASELINE), DAIMOI_ALPHA_BASELINE
            ),
        )
        mixed = src + (delta * tgt)
        shifted = ((1.0 - rho) * mixed) + (rho * DAIMOI_ALPHA_BASELINE)
        updated[key] = min(
            DAIMOI_ALPHA_MAX, max(1e-8, _finite_float(shifted, DAIMOI_ALPHA_BASELINE))
        )
    return updated


def _seed_curr_matrix(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_seed = _state_unit_vector(left, "e_seed")
    left_curr = _state_unit_vector(left, "e_curr")
    right_seed = _state_unit_vector(right, "e_seed")
    right_curr = _state_unit_vector(right, "e_curr")
    return {
        "ss": _safe_cosine_unit(left_seed, right_seed),
        "sc": _safe_cosine_unit(left_seed, right_curr),
        "cs": _safe_cosine_unit(left_curr, right_seed),
        "cc": _safe_cosine_unit(left_curr, right_curr),
        "self_left": _safe_cosine_unit(left_seed, left_curr),
        "self_right": _safe_cosine_unit(right_seed, right_curr),
    }


def _collision_semantic_update(
    left: dict[str, Any], right: dict[str, Any], *, impulse: float
) -> dict[str, Any]:
    matrix = _seed_curr_matrix(left, right)
    semantic_affinity = (
        (matrix["cc"] * 0.5)
        + (((matrix["sc"] + matrix["cs"]) * 0.5) * 0.3)
        + (matrix["ss"] * 0.2)
    )
    semantic_affinity = _finite_float(semantic_affinity, 0.0)
    transfer_t = _clamp01_finite((semantic_affinity + 1.0) * 0.5, 0.5)
    # Base repulsion increases as semantic affinity decreases
    repulsion_u = _clamp01_finite(((-semantic_affinity) + 1.0) * 0.5, 0.5)
    # Enhanced repulsion for strongly unrelated concepts (affinity < -0.5)
    if semantic_affinity < -0.5:
        repulsion_u = min(1.0, repulsion_u * 1.6)  # Boost repulsion for opposites
    intensity = _clamp01_finite(
        _finite_float(impulse, 0.0) / DAIMOI_IMPULSE_REFERENCE, 0.0
    )

    left_size = max(1e-8, _finite_float(left.get("size", 1.0), 1.0))
    right_size = max(1e-8, _finite_float(right.get("size", 1.0), 1.0))
    bias_left = _sigmoid(DAIMOI_SIZE_BIAS_BETA * math.log(right_size / left_size))
    bias_right = _sigmoid(DAIMOI_SIZE_BIAS_BETA * math.log(left_size / right_size))
    coupling_left = _clamp01(intensity * bias_left)
    coupling_right = _clamp01(intensity * bias_right)
    coupling_left_01 = _clamp01_finite(coupling_left, 0.0)
    coupling_right_01 = _clamp01_finite(coupling_right, 0.0)
    transfer_t_01 = _clamp01_finite(transfer_t, 0.0)
    repulsion_u_01 = _clamp01_finite(repulsion_u, 0.0)
    left_delta = DAIMOI_TRANSFER_LAMBDA * coupling_left_01 * transfer_t_01
    left_rho = DAIMOI_REPULSION_MU * coupling_left_01 * repulsion_u_01
    right_delta = DAIMOI_TRANSFER_LAMBDA * coupling_right_01 * transfer_t_01
    right_rho = DAIMOI_REPULSION_MU * coupling_right_01 * repulsion_u_01

    left_seed = _state_unit_vector(left, "e_seed")
    left_curr = _state_unit_vector(left, "e_curr")
    right_seed = _state_unit_vector(right, "e_seed")
    right_curr = _state_unit_vector(right, "e_curr")

    trust_left = _clamp01((matrix["self_left"] + 1.0) * 0.5)
    trust_right = _clamp01((matrix["self_right"] + 1.0) * 0.5)
    left_export = _blend_vectors(left_seed, left_curr, trust_left)
    right_export = _blend_vectors(right_seed, right_curr, trust_right)

    next_left_curr = _normalize_vector(
        [
            (left_curr[idx] * (1.0 - coupling_left))
            + (right_export[idx] * coupling_left)
            for idx in range(min(len(left_curr), len(right_export)))
        ]
    )
    next_right_curr = _normalize_vector(
        [
            (right_curr[idx] * (1.0 - coupling_right))
            + (left_export[idx] * coupling_right)
            for idx in range(min(len(right_curr), len(left_export)))
        ]
    )

    left_alpha_pkg_raw = left.get("alpha_pkg", {})
    right_alpha_pkg_raw = right.get("alpha_pkg", {})

    resource_transfer = {}

    # Simple check: emit_resource_packet vs absorb_resource
    # We use the raw alpha values as a proxy for "intent strength"
    left_emit = _finite_float(left_alpha_pkg_raw.get("emit_resource_packet", 0.0))
    left_absorb = _finite_float(left_alpha_pkg_raw.get("absorb_resource", 0.0))
    right_emit = _finite_float(right_alpha_pkg_raw.get("emit_resource_packet", 0.0))
    right_absorb = _finite_float(right_alpha_pkg_raw.get("absorb_resource", 0.0))

    # Threshold for action (arbitrary for prototype, scaled by intensity?)
    action_threshold = DAIMOI_ALPHA_BASELINE * 1.5

    # Left emits to Right
    if left_emit > action_threshold and right_absorb > action_threshold:
        # Determine resource type from owner ID (hacky but effective)
        owner_id = str(left.get("owner", ""))
        if "presence.core." in owner_id:
            res_type = owner_id.replace("presence.core.", "")
            # Transfer amount depends on intensity and emitter strength
            amount = max(0.1, intensity * 5.0)
            resource_transfer["left_to_right"] = {res_type: amount}

    # Right emits to Left
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

    if (
        left_alpha_pkg.keys() <= DAIMOI_JOB_KEYS_SET
        and right_alpha_pkg.keys() <= DAIMOI_JOB_KEYS_SET
    ):
        left_alpha_pkg_next: dict[str, float] = {}
        right_alpha_pkg_next: dict[str, float] = {}
        for key in DAIMOI_JOB_KEYS_SORTED:
            src = max(
                1e-8,
                _finite_float(
                    left_alpha_pkg.get(key, DAIMOI_ALPHA_BASELINE),
                    DAIMOI_ALPHA_BASELINE,
                ),
            )
            tgt = max(
                1e-8,
                _finite_float(
                    right_alpha_pkg.get(key, DAIMOI_ALPHA_BASELINE),
                    DAIMOI_ALPHA_BASELINE,
                ),
            )
            left_shifted = ((1.0 - left_rho) * (src + (left_delta * tgt))) + (
                left_rho * DAIMOI_ALPHA_BASELINE
            )
            right_shifted = ((1.0 - right_rho) * (tgt + (right_delta * src))) + (
                right_rho * DAIMOI_ALPHA_BASELINE
            )
            left_alpha_pkg_next[key] = min(
                DAIMOI_ALPHA_MAX,
                max(1e-8, _finite_float(left_shifted, DAIMOI_ALPHA_BASELINE)),
            )
            right_alpha_pkg_next[key] = min(
                DAIMOI_ALPHA_MAX,
                max(1e-8, _finite_float(right_shifted, DAIMOI_ALPHA_BASELINE)),
            )
    else:
        left_alpha_pkg_safe = {
            str(key): max(1e-8, _finite_float(value, DAIMOI_ALPHA_BASELINE))
            for key, value in dict(left_alpha_pkg).items()
        }
        right_alpha_pkg_safe = {
            str(key): max(1e-8, _finite_float(value, DAIMOI_ALPHA_BASELINE))
            for key, value in dict(right_alpha_pkg).items()
        }
        package_keys = tuple(
            sorted(
                set(
                    [
                        *left_alpha_pkg_safe.keys(),
                        *right_alpha_pkg_safe.keys(),
                        *DAIMOI_JOB_KEYS,
                    ]
                )
            )
        )
        left_alpha_pkg_next = _dirichlet_transfer(
            left_alpha_pkg_safe,
            right_alpha_pkg_safe,
            coupling=coupling_left,
            transfer_t=transfer_t,
            repulsion_u=repulsion_u,
            keys=package_keys,
        )
        right_alpha_pkg_next = _dirichlet_transfer(
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
            _finite_float(
                left_alpha_msg_raw.get("deliver", DAIMOI_ALPHA_BASELINE),
                DAIMOI_ALPHA_BASELINE,
            ),
        ),
        "hold": max(
            1e-8,
            _finite_float(
                left_alpha_msg_raw.get("hold", DAIMOI_ALPHA_BASELINE),
                DAIMOI_ALPHA_BASELINE,
            ),
        ),
    }
    right_alpha_msg = {
        "deliver": max(
            1e-8,
            _finite_float(
                right_alpha_msg_raw.get("deliver", DAIMOI_ALPHA_BASELINE),
                DAIMOI_ALPHA_BASELINE,
            ),
        ),
        "hold": max(
            1e-8,
            _finite_float(
                right_alpha_msg_raw.get("hold", DAIMOI_ALPHA_BASELINE),
                DAIMOI_ALPHA_BASELINE,
            ),
        ),
    }
    left_alpha_msg_next = {
        "deliver": min(
            DAIMOI_ALPHA_MAX,
            max(
                1e-8,
                _finite_float(
                    (
                        (1.0 - left_rho)
                        * (
                            left_alpha_msg["deliver"]
                            + (left_delta * right_alpha_msg["deliver"])
                        )
                    )
                    + (left_rho * DAIMOI_ALPHA_BASELINE),
                    DAIMOI_ALPHA_BASELINE,
                ),
            ),
        ),
        "hold": min(
            DAIMOI_ALPHA_MAX,
            max(
                1e-8,
                _finite_float(
                    (
                        (1.0 - left_rho)
                        * (
                            left_alpha_msg["hold"]
                            + (left_delta * right_alpha_msg["hold"])
                        )
                    )
                    + (left_rho * DAIMOI_ALPHA_BASELINE),
                    DAIMOI_ALPHA_BASELINE,
                ),
            ),
        ),
    }
    right_alpha_msg_next = {
        "deliver": min(
            DAIMOI_ALPHA_MAX,
            max(
                1e-8,
                _finite_float(
                    (
                        (1.0 - right_rho)
                        * (
                            right_alpha_msg["deliver"]
                            + (right_delta * left_alpha_msg["deliver"])
                        )
                    )
                    + (right_rho * DAIMOI_ALPHA_BASELINE),
                    DAIMOI_ALPHA_BASELINE,
                ),
            ),
        ),
        "hold": min(
            DAIMOI_ALPHA_MAX,
            max(
                1e-8,
                _finite_float(
                    (
                        (1.0 - right_rho)
                        * (
                            right_alpha_msg["hold"]
                            + (right_delta * left_alpha_msg["hold"])
                        )
                    )
                    + (right_rho * DAIMOI_ALPHA_BASELINE),
                    DAIMOI_ALPHA_BASELINE,
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
        "resource_transfer": resource_transfer,  # Return the side effect
    }


def _job_probabilities(state: dict[str, Any]) -> dict[str, float]:
    alpha_raw = state.get("alpha_pkg", {})
    if isinstance(alpha_raw, dict) and alpha_raw.keys() <= DAIMOI_JOB_KEYS_SET:
        totals = 0.0
        normalized: dict[str, float] = {}
        for key in DAIMOI_JOB_KEYS_SORTED:
            value = max(
                1e-8,
                _finite_float(
                    alpha_raw.get(key, DAIMOI_ALPHA_BASELINE), DAIMOI_ALPHA_BASELINE
                ),
            )
            value = min(DAIMOI_ALPHA_MAX, value)
            normalized[key] = value
            totals += value
        if totals <= 1e-12:
            uniform = 1.0 / max(1, len(DAIMOI_JOB_KEYS_SORTED))
            return {key: uniform for key in DAIMOI_JOB_KEYS_SORTED}
        inv_total = 1.0 / totals
        return {key: normalized[key] * inv_total for key in DAIMOI_JOB_KEYS_SORTED}

    alpha_pkg = {
        str(key): max(1e-8, _finite_float(value, DAIMOI_ALPHA_BASELINE))
        for key, value in dict(alpha_raw if isinstance(alpha_raw, dict) else {}).items()
    }
    keys = tuple(sorted(set([*DAIMOI_JOB_KEYS, *alpha_pkg.keys()])))
    return _dirichlet_probabilities(alpha_pkg, keys=keys)


def _message_probability(state: dict[str, Any]) -> float:
    alpha_msg_raw = state.get("alpha_msg", {})
    alpha_msg = alpha_msg_raw if isinstance(alpha_msg_raw, dict) else {}
    deliver = max(
        1e-8,
        _finite_float(
            alpha_msg.get("deliver", DAIMOI_ALPHA_BASELINE), DAIMOI_ALPHA_BASELINE
        ),
    )
    hold = max(
        1e-8,
        _finite_float(
            alpha_msg.get("hold", DAIMOI_ALPHA_BASELINE), DAIMOI_ALPHA_BASELINE
        ),
    )
    total = deliver + hold
    if total <= 1e-12 or not math.isfinite(total):
        return 0.5
    return _clamp01_finite(deliver / total, 0.5)


def _rounded_distribution(
    probabilities: dict[str, float], *, precision: int = 6
) -> dict[str, float]:
    if not probabilities:
        return {}
    if (
        len(probabilities) == 2
        and "deflect" in probabilities
        and "diffuse" in probabilities
    ):
        deflect = round(
            _clamp01_finite(probabilities.get("deflect", 0.5), 0.5), precision
        )
        diffuse = round(
            _clamp01_finite(probabilities.get("diffuse", 0.5), 0.5), precision
        )
        residual = round(1.0 - (deflect + diffuse), precision)
        if abs(residual) <= (10 ** (-precision)):
            diffuse = round(_clamp01_finite(diffuse + residual, 0.0), precision)
        return {"deflect": deflect, "diffuse": diffuse}
    if probabilities.keys() <= DAIMOI_JOB_KEYS_SET:
        ordered = [
            (key, _clamp01_finite(probabilities.get(key, 0.0), 0.0))
            for key in DAIMOI_JOB_KEYS_SORTED
        ]
    else:
        ordered = sorted(
            (str(key), _clamp01_finite(value, 0.0))
            for key, value in probabilities.items()
        )
    rounded = {key: round(value, precision) for key, value in ordered}
    total = sum(rounded.values())
    residual = round(1.0 - total, precision)
    if abs(residual) <= (10 ** (-precision)) and ordered:
        sink_key = ordered[-1][0]
        corrected = _clamp01_finite(rounded.get(sink_key, 0.0) + residual, 0.0)
        rounded[sink_key] = round(corrected, precision)
    return rounded


def _presence_anchor_map(
    presence_impacts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    anchors: dict[str, dict[str, Any]] = {}
    for row in presence_impacts:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("id", "")).strip()
        if not presence_id:
            continue
        meta = _ENTITY_MANIFEST_BY_ID.get(presence_id, {})
        stable_x = _stable_ratio(f"{presence_id}|anchor", 7)
        stable_y = _stable_ratio(f"{presence_id}|anchor", 13)
        resolved_x = _clamp01(
            _safe_float(
                row.get(
                    "x",
                    meta.get("x", stable_x),
                ),
                _safe_float(meta.get("x", stable_x), stable_x),
            )
        )
        resolved_y = _clamp01(
            _safe_float(
                row.get(
                    "y",
                    meta.get("y", stable_y),
                ),
                _safe_float(meta.get("y", stable_y), stable_y),
            )
        )
        resolved_hue = _safe_float(row.get("hue", meta.get("hue", 210.0)), 210.0)
        anchors[presence_id] = {
            "id": presence_id,
            "en": str(
                row.get(
                    "en",
                    row.get(
                        "label", meta.get("en", presence_id.replace("_", " ").title())
                    ),
                )
            ),
            "ja": str(row.get("ja", row.get("label_ja", meta.get("ja", "")))),
            "type": str(
                row.get("presence_type", meta.get("type", "presence")) or "presence"
            ),
            "x": resolved_x,
            "y": resolved_y,
            "hue": resolved_hue,
            "embedding": _embedding_from_text(
                _presence_prompt_template(
                    meta if isinstance(meta, dict) else {}, presence_id
                )
            ),
        }
    return anchors


def _node_semantic_vector(node: dict[str, Any]) -> list[float]:
    text_parts = [
        str(node.get("name", "")),
        str(node.get("summary", "")),
        str(node.get("text_excerpt", "")),
        str(node.get("dominant_field", "")),
        str(node.get("dominant_presence", "")),
    ]
    tags = node.get("tags", [])
    if isinstance(tags, list) and tags:
        text_parts.append(" ".join(str(item) for item in tags[:12]))
    return _embedding_from_text(" | ".join(part for part in text_parts if part.strip()))


def _file_node_rows(file_graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(file_graph, dict):
        return []
    file_nodes_raw = file_graph.get("file_nodes", [])
    if not isinstance(file_nodes_raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(file_nodes_raw):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip() or f"file-node-{index:05d}"
        rows.append(
            {
                "id": node_id,
                "x": _clamp01(_safe_float(node.get("x", 0.5), 0.5)),
                "y": _clamp01(_safe_float(node.get("y", 0.5), 0.5)),
                "importance": _clamp01(_safe_float(node.get("importance", 0.3), 0.3)),
                "embedded_bonus": _clamp01(
                    (_safe_float(node.get("embed_layer_count", 0.0), 0.0) / 3.0)
                    + (
                        0.35
                        if str(node.get("vecstore_collection", "")).strip()
                        else 0.0
                    )
                ),
                "dominant_field": str(node.get("dominant_field", "")).strip(),
                "dominant_presence": str(node.get("dominant_presence", "")).strip(),
                "vector": _node_semantic_vector(node),
            }
        )
    return rows


def _presence_density(
    anchor: dict[str, Any], file_nodes: list[dict[str, Any]], presence_id: str
) -> float:
    ax = _clamp01(_safe_float(anchor.get("x", 0.5), 0.5))
    ay = _clamp01(_safe_float(anchor.get("y", 0.5), 0.5))
    target_fields = {
        field_id
        for field_id, owner in FIELD_TO_PRESENCE.items()
        if str(owner).strip() == str(presence_id).strip()
    }
    density = 0.0
    for node in file_nodes:
        nx = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        ny = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        dx = nx - ax
        if dx > 0.34 or dx < -0.34:
            continue
        dy = ny - ay
        if dy > 0.34 or dy < -0.34:
            continue
        distance = math.sqrt((dx * dx) + (dy * dy))
        if distance > 0.34:
            continue
        radial = _clamp01(1.0 - (distance / 0.34))
        importance = _clamp01(_safe_float(node.get("importance", 0.3), 0.3))
        field_bonus = 0.0
        if str(node.get("dominant_presence", "")).strip() == str(presence_id).strip():
            field_bonus = 0.42
        elif str(node.get("dominant_field", "")).strip() in target_fields:
            field_bonus = 0.34
        embedded_bonus = (
            _clamp01(_safe_float(node.get("embedded_bonus", 0.0), 0.0)) * 0.3
        )
        density += radial * (0.4 + (importance * 0.7) + field_bonus + embedded_bonus)
    return _clamp01(density / 4.0)


def _choose_target_presence(
    *,
    owner_presence_id: str,
    presence_ids: list[str],
    particle_id: str,
    now: float,
) -> str:
    others = [pid for pid in presence_ids if pid != owner_presence_id]
    if not others:
        return owner_presence_id
    tick = _safe_int(math.floor(_safe_float(now, 0.0) * 1.0), 0)
    ratio = _stable_ratio(f"{particle_id}|target|{tick}", tick + 5)
    index = int(math.floor(ratio * len(others))) % len(others)
    return others[index]


def _spawn_particle(
    *,
    particle_id: str,
    slot_index: int,
    presence_id: str,
    target_presence_id: str,
    anchor: dict[str, Any],
    file_influence: float,
    click_influence: float,
    world_influence: float,
    resource_influence: float,
    queue_ratio: float,
    now: float,
    resource_wallet: dict[str, float] | None = None,  # New: Check cost
) -> dict[str, Any]:
    role, mode = _presence_role_and_mode(presence_id)

    # Gating check: Non-core must pay to emit
    if role != "resource-core" and resource_wallet is not None:
        # Simple cost model: 1.0 unit of 'any' resource to emit a particle
        # In reality, different particles might cost different resources (e.g. CPU vs RAM)
        # For now, just check total balance or specific if we track types.
        # Let's assume resource_wallet has keys like 'cpu', 'ram'.
        # To emit, we need > 0.1 of something.
        total_resources = sum(resource_wallet.values())
        if total_resources < 0.1:
            # Starved! Cannot emit.
            # Return a 'null' or 'blocked' particle?
            # Or handle this caller-side?
            # Better to handle caller side, but let's mark it here if we must return a dict.
            # Actually, _spawn_particle is usually called inside a loop.
            # We will add a 'starved' flag to the particle, or maybe just produce a weak/inert one.
            pass

    directive = _directive_for_particle(presence_id, slot_index, now)
    variant = int(
        math.floor(
            _stable_ratio(f"{presence_id}|variant|{slot_index}", slot_index + 2) * 4.0
        )
    )
    seed_prompt = _presence_prompt_template(anchor, presence_id)
    seed_text = (
        f"{seed_prompt} variation={variant}. "
        f"Directive: {directive}. "
        f"target={target_presence_id} queue_ratio={round(_clamp01(queue_ratio), 3)}"
    )
    e_seed = _embedding_from_text(seed_text)
    e_curr = list(e_seed)

    angle = (_stable_ratio(f"{particle_id}|angle", slot_index + 9) * math.tau) + (
        _safe_float(now, 0.0) * (0.16 + (world_influence * 0.42))
    )
    orbit = 0.012 + (_stable_ratio(f"{particle_id}|orbit", slot_index + 13) * 0.055)
    x = _clamp01(_safe_float(anchor.get("x", 0.5), 0.5) + (math.cos(angle) * orbit))
    y = _clamp01(
        _safe_float(anchor.get("y", 0.5), 0.5) + (math.sin(angle) * orbit * 0.84)
    )
    speed = 0.001 + (world_influence * 0.0022) + (file_influence * 0.0014)
    vx = math.cos(angle + (math.pi / 2.0)) * speed
    vy = math.sin(angle + (math.pi / 2.0)) * speed

    size = (
        0.85
        + (_stable_ratio(f"{particle_id}|size", slot_index + 17) * 1.4)
        + (world_influence * 0.5)
        + (file_influence * 0.36)
    )
    size = max(0.6, min(3.4, _safe_float(size, 1.2)))
    mass = max(0.35, (size * 0.82) + 0.35)
    radius = max(0.012, min(0.034, 0.008 + (size * 0.0078)))

    alpha_pkg = _initial_job_alpha(
        presence_id,
        role=role,
        file_influence=file_influence,
        click_influence=click_influence,
        world_influence=world_influence,
        resource_influence=resource_influence,
        slot_index=slot_index,
    )
    alpha_msg = _initial_message_alpha(
        file_influence=file_influence,
        click_influence=click_influence,
        world_influence=world_influence,
    )

    return {
        "id": particle_id,
        "owner": presence_id,
        "target": target_presence_id,
        "presence_role": role,
        "particle_mode": mode,
        "behaviors": list(DAIMOI_BEHAVIOR_DEFAULTS),
        "directive": directive,
        "seed_text": seed_text,
        "e_seed": e_seed,
        "e_curr": e_curr,
        "alpha_pkg": alpha_pkg,
        "alpha_msg": alpha_msg,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "mass": mass,
        "radius": radius,
        "size": size,
        "age": 0,
        "collisions": 0,
        "handoffs": 0,
        "last_collision_matrix": {"ss": 0.0, "sc": 0.0, "cs": 0.0, "cc": 0.0},
        "ts": time.monotonic(),
    }


def _spawn_chaos_butterfly(
    *,
    particle_id: str,
    slot_index: int,
    anchor: dict[str, Any],
    now: float,
) -> dict[str, Any]:
    """Spawn a Chaos Butterfly particle that spreads noise through fields using simplex noise.

    Chaos butterflies are special daimoi particles that:
    - Move erratically using simplex noise-based velocity perturbations
    - Inject noise into fields they pass through
    - Influence other particles with chaotic perturbations
    - Have no target - they wander aimlessly spreading disorder
    """
    seed_prompt = _presence_prompt_template(anchor, "chaos_butterfly")
    seed_text = (
        f"{seed_prompt} butterfly={slot_index}. "
        f"Fluttering through fields, spreading noise and unpredictability. "
        f"ts={now}"
    )
    e_seed = _embedding_from_text(seed_text)
    e_curr = list(e_seed)

    # Chaos butterflies start at center with high randomness
    base_x = _safe_float(anchor.get("x", 0.5), 0.5)
    base_y = _safe_float(anchor.get("y", 0.5), 0.5)

    # Add initial chaotic displacement
    dx, dy = _chaos_field_perturbation(base_x, base_y, now, amplitude=0.08)
    x = _clamp01(base_x + dx)
    y = _clamp01(base_y + dy)

    # Chaotic velocity - high speed, unpredictable direction
    noise_vx = _simplex_noise_2d(x * 5.0, now * 0.5, seed=slot_index)
    noise_vy = _simplex_noise_2d(y * 5.0 + 100, now * 0.5, seed=slot_index + 1)
    speed = 0.008  # Chaos butterflies move fast
    vx = noise_vx * speed
    vy = noise_vy * speed

    # Smaller, lighter particles that flutter
    size = 0.4 + (_stable_ratio(f"{particle_id}|size", slot_index) * 0.4)
    mass = 0.15  # Light mass for erratic movement
    radius = 0.006  # Small radius

    # Chaotic job alpha - favors diffuse and deflect equally (chaos)
    alpha_pkg = {key: DAIMOI_ALPHA_BASELINE for key in DAIMOI_JOB_KEYS}
    alpha_pkg["invoke_diffuse_field"] = 2.5  # Chaos spreads
    alpha_pkg["deliver_message"] = 0.8  # Unreliable messaging
    alpha_pkg["invoke_truth_gate"] = 0.3  # Chaos avoids truth

    alpha_msg = {
        "deliver": DAIMOI_ALPHA_BASELINE * 0.7,  # Unreliable
        "hold": DAIMOI_ALPHA_BASELINE * 1.3,  # Chaos holds onto noise
    }

    return {
        "id": particle_id,
        "owner": "chaos_butterfly",
        "target": "",  # No target - wanders aimlessly
        "presence_role": "chaos-agent",
        "particle_mode": "noise-spreader",
        "behaviors": ["diffuse", "deflect", "perturb"],
        "directive": "Spread noise and unpredictability through all fields.",
        "seed_text": seed_text,
        "e_seed": e_seed,
        "e_curr": e_curr,
        "alpha_pkg": alpha_pkg,
        "alpha_msg": alpha_msg,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "mass": mass,
        "radius": radius,
        "size": size,
        "age": 0,
        "collisions": 0,
        "handoffs": 0,
        "last_collision_matrix": {"ss": 0.0, "sc": 0.0, "cs": 0.0, "cc": 0.0},
        "ts": time.monotonic(),
        "is_chaos_butterfly": True,  # Marker for special handling
        "noise_amplitude": 0.12 + (_stable_ratio(particle_id, slot_index) * 0.08),
    }


def _spawn_nexus_particle(
    *,
    particle_id: str,
    node: dict[str, Any],
    owner_presence_id: str,
) -> dict[str, Any]:
    node_x = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
    node_y = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
    importance = _clamp01(_safe_float(node.get("importance", 0.3), 0.3))
    node_vector = _normalize_vector(list(node.get("vector", [])))
    seed_text = (
        f"Nexus static daimo {particle_id} owner={owner_presence_id}. "
        "No agency. Move only with field currents and external collisions."
    )
    if not node_vector:
        node_vector = _embedding_from_text(seed_text)

    alpha_pkg = {key: DAIMOI_ALPHA_BASELINE for key in DAIMOI_JOB_KEYS}
    alpha_msg = {"deliver": 0.8, "hold": 1.2}
    size = 0.72 + (importance * 1.1)

    return {
        "id": particle_id,
        "owner": owner_presence_id,
        "target": owner_presence_id,
        "presence_role": "nexus-passive",
        "particle_mode": "static-daimoi",
        "behaviors": ["drift", "collide"],
        "directive": "Move with current; no self-directed agency.",
        "seed_text": seed_text,
        "e_seed": list(node_vector),
        "e_curr": list(node_vector),
        "alpha_pkg": alpha_pkg,
        "alpha_msg": alpha_msg,
        "x": node_x,
        "y": node_y,
        "vx": 0.0,
        "vy": 0.0,
        "mass": 2.4 + (importance * 1.6),
        "radius": 0.01 + (importance * 0.01),
        "size": size,
        "age": 0,
        "collisions": 0,
        "handoffs": 0,
        "last_collision_matrix": {"ss": 0.0, "sc": 0.0, "cs": 0.0, "cc": 0.0},
        "ts": time.monotonic(),
        "is_nexus": True,
        "is_static_daimoi": True,
        "source_node_id": str(node.get("id", "")).strip(),
        "preferred_x": node_x,
        "preferred_y": node_y,
    }


def _surface_state(
    surface_map: dict[str, dict[str, Any]], presence_id: str
) -> dict[str, Any]:
    state = surface_map.get(presence_id)
    if isinstance(state, dict):
        return state
    state = {
        "embedding": _normalize_vector([]),
        "alpha_pkg": {key: DAIMOI_ALPHA_BASELINE for key in DAIMOI_JOB_KEYS},
        "alpha_msg": {"deliver": DAIMOI_ALPHA_BASELINE, "hold": DAIMOI_ALPHA_BASELINE},
        "deflect_count": 0,
        "diffuse_count": 0,
        "impulse": 0.0,
        "field_energy": 0.0,
        "job_hits": {},
        "ts": time.monotonic(),
    }
    surface_map[presence_id] = state
    return state


def _sample_job_key(probabilities: dict[str, float], *, seed: str) -> str:
    if not probabilities:
        return "deliver_message"
    ordered = sorted(probabilities.items(), key=lambda item: item[0])
    roll = _stable_ratio(seed, 11)
    cursor = 0.0
    for job_key, probability in ordered:
        cursor += _clamp01(_safe_float(probability, 0.0))
        if roll <= cursor:
            return str(job_key)
    return str(ordered[-1][0])


def _action_probabilities(
    job_probs: dict[str, float], message_prob: float
) -> dict[str, float]:
    p_diffuse = _clamp01(
        0.08
        + (message_prob * 0.38)
        + (_safe_float(job_probs.get("invoke_diffuse_field", 0.0), 0.0) * 0.34)
        + (_safe_float(job_probs.get("deliver_message", 0.0), 0.0) * 0.2)
    )
    p_deflect = _clamp01(1.0 - p_diffuse)
    total = p_deflect + p_diffuse
    if total <= 1e-12:
        return {"deflect": 0.5, "diffuse": 0.5}
    return {
        "deflect": p_deflect / total,
        "diffuse": p_diffuse / total,
    }


def _action_probabilities_for_state(state: dict[str, Any]) -> dict[str, float]:
    job_probs = _job_probabilities(state)
    message_prob = _message_probability(state)
    return _action_probabilities(job_probs, message_prob)


def build_probabilistic_daimoi_particles(
    *,
    file_graph: dict[str, Any] | None,
    presence_impacts: list[dict[str, Any]],
    resource_heartbeat: dict[str, Any],
    compute_jobs: list[dict[str, Any]],
    queue_ratio: float,
    now: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not presence_impacts:
        return [], {
            "record": DAIMOI_PROBABILISTIC_RECORD,
            "schema_version": DAIMOI_PROBABILISTIC_SCHEMA,
            "active": 0,
            "spawned": 0,
            "collisions": 0,
            "deflects": 0,
            "diffuses": 0,
            "handoffs": 0,
            "deliveries": 0,
            "job_triggers": {},
            "mean_package_entropy": 0.0,
            "mean_message_probability": 0.0,
            "packet_contract": {
                "record": DAIMOI_PACKET_COMPONENT_RECORD,
                "schema_version": DAIMOI_PACKET_COMPONENT_SCHEMA,
                "resource_keys": list(DAIMOI_RESOURCE_KEYS),
            },
            "absorb_sampler": {
                "record": DAIMOI_ABSORB_SAMPLER_RECORD,
                "schema_version": DAIMOI_ABSORB_SAMPLER_SCHEMA,
                "method": DAIMOI_ABSORB_SAMPLER_METHOD,
                "events": 0,
                "sample_events": [],
            },
            "behavior_defaults": list(DAIMOI_BEHAVIOR_DEFAULTS),
        }

    devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    if not isinstance(devices, dict):
        devices = {}
    resource_pressure = 0.0
    for key in ("cpu", "gpu1", "gpu2", "npu0"):
        row = devices.get(key, {})
        util = _safe_float(
            (row if isinstance(row, dict) else {}).get("utilization", 0.0), 0.0
        )
        resource_pressure = max(resource_pressure, _clamp01(util / 100.0))
    compute_pressure = _clamp01(len(compute_jobs) / 28.0)
    queue_pressure = _clamp01(_safe_float(queue_ratio, 0.0))

    anchors = _presence_anchor_map(presence_impacts)
    presence_ids = [presence_id for presence_id in anchors.keys() if presence_id]
    impact_by_id = {
        str(row.get("id", "")).strip(): row
        for row in presence_impacts
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }
    file_nodes = _file_node_rows(file_graph)
    local_density_map = {
        presence_id: _presence_density(anchors[presence_id], file_nodes, presence_id)
        for presence_id in presence_ids
    }

    now_seconds = _safe_float(now, time.time())
    now_seconds_int = _safe_int(now_seconds, 0)
    now_seconds_tenths_int = _safe_int(now_seconds * 10.0, 0)
    now_monotonic = time.monotonic()
    spawned_count = 0
    collision_count = 0
    deflect_count = 0
    diffuse_count = 0
    handoff_count = 0
    delivery_count = 0
    matrix_accumulator = {"ss": 0.0, "sc": 0.0, "cs": 0.0, "cc": 0.0, "samples": 0}
    job_trigger_counts: dict[str, int] = {}
    absorb_sampler_count = 0
    absorb_sampler_events: list[dict[str, Any]] = []

    with _DAIMO_DYNAMICS_LOCK:
        runtime = _DAIMO_DYNAMICS_CACHE.get("field_particles", {})
        if not isinstance(runtime, dict) or "particles" not in runtime:
            runtime = {
                "particles": {},
                "surfaces": {},
                "field_cells": {},
                "spawn_seq": 0,
            }

        particles = runtime.get("particles", {})
        if not isinstance(particles, dict):
            particles = {}
        surfaces = runtime.get("surfaces", {})
        if not isinstance(surfaces, dict):
            surfaces = {}
        field_cells = runtime.get("field_cells", {})
        if not isinstance(field_cells, dict):
            field_cells = {}

        spawn_seq = _safe_int(runtime.get("spawn_seq", 0), 0)
        prior_tick_ms = _safe_float(
            runtime.get("tick_ms_ema", runtime.get("tick_ms", 0.0)),
            0.0,
        )
        load_scale = 1.0
        if prior_tick_ms >= 120.0:
            load_scale = 0.52
        elif prior_tick_ms >= 90.0:
            load_scale = 0.64
        elif prior_tick_ms >= 70.0:
            load_scale = 0.78
        elif prior_tick_ms >= 52.0:
            load_scale = 0.88

        per_presence_cap = 96
        nexus_cap = 180
        chaos_target_count = 8
        if prior_tick_ms >= 130.0:
            per_presence_cap = 34
            nexus_cap = 92
            chaos_target_count = 4
        elif prior_tick_ms >= 100.0:
            per_presence_cap = 44
            nexus_cap = 112
            chaos_target_count = 5
        elif prior_tick_ms >= 78.0:
            per_presence_cap = 58
            nexus_cap = 136
            chaos_target_count = 6
        elif prior_tick_ms >= 60.0:
            per_presence_cap = 74
            nexus_cap = 160
            chaos_target_count = 7

        active_ids: set[str] = set()
        for impact in presence_impacts:
            if not isinstance(impact, dict):
                continue
            presence_id = str(impact.get("id", "")).strip()
            if not presence_id or presence_id not in anchors:
                continue
            # Skip chaos_butterfly - it has special spawning logic below
            if presence_id == "chaos_butterfly":
                continue

            # Core presence logic: emission depends on resource availability
            role, _ = _presence_role_and_mode(presence_id)
            file_influence = 0.0
            click_influence = 0.0
            world_influence = 0.0
            resource_influence = 0.0
            if role == "resource-core":
                # Determine which resource this presence represents
                resource_type = presence_id.replace("presence.core.", "")
                resource_monitor = (
                    resource_heartbeat.get("resource_monitor", {})
                    if isinstance(resource_heartbeat, dict)
                    else {}
                )
                if not isinstance(resource_monitor, dict):
                    resource_monitor = {}

                # Extract specific metric based on type
                usage_percent = 100.0
                if resource_type == "cpu":
                    usage_percent = _safe_float(
                        devices.get("cpu", {}).get("utilization", 100.0),
                        _safe_float(resource_monitor.get("cpu_percent", 100.0), 100.0),
                    )
                elif resource_type == "ram":
                    usage_percent = _safe_float(
                        resource_monitor.get("memory_percent", 100.0), 100.0
                    )
                elif resource_type == "disk":
                    usage_percent = _safe_float(
                        resource_monitor.get("disk_percent", 100.0), 100.0
                    )
                elif resource_type in {"gpu", "gpu1"}:
                    usage_percent = _safe_float(
                        devices.get("gpu1", {}).get("utilization", 100.0), 100.0
                    )
                elif resource_type in {"intel", "gpu2", "gpu_intel"}:
                    usage_percent = _safe_float(
                        devices.get("gpu2", {}).get("utilization", 100.0), 100.0
                    )
                elif resource_type in {"npu", "npu0"}:
                    usage_percent = _safe_float(
                        devices.get("npu0", {}).get("utilization", 100.0), 100.0
                    )

                # Higher availability (lower usage) -> higher emission target
                # Base target count logic, but modified for core
                availability_factor = _clamp01((100.0 - usage_percent) / 100.0)
                target_count = int(
                    round(8.0 + (availability_factor * 40.0))
                )  # Scale 8-48
            else:
                # Standard presence logic
                affected_by = (
                    impact.get("affected_by", {}) if isinstance(impact, dict) else {}
                )
                affects = impact.get("affects", {}) if isinstance(impact, dict) else {}
                file_influence = _clamp01(
                    _safe_float(
                        (affected_by if isinstance(affected_by, dict) else {}).get(
                            "files", 0.0
                        ),
                        0.0,
                    )
                )
                click_influence = _clamp01(
                    _safe_float(
                        (affected_by if isinstance(affected_by, dict) else {}).get(
                            "clicks", 0.0
                        ),
                        0.0,
                    )
                )
                world_influence = _clamp01(
                    _safe_float(
                        (affects if isinstance(affects, dict) else {}).get(
                            "world", 0.0
                        ),
                        0.0,
                    )
                )
                resource_influence = _clamp01(
                    _safe_float(
                        (affected_by if isinstance(affected_by, dict) else {}).get(
                            "resource", 0.0
                        ),
                        0.0,
                    )
                )
                local_density = _clamp01(
                    _safe_float(local_density_map.get(presence_id, 0.0), 0.0)
                )

                # INCREASED: Base count and multipliers for stress testing
                target_count = int(
                    round(
                        12.0
                        + (world_influence * 18.0)
                        + (file_influence * 22.0)
                        + (click_influence * 8.0)
                        + (local_density * 28.0)
                        - (resource_pressure * 3.0)
                        - (queue_pressure * 1.5)
                        - (compute_pressure * 1.0)
                    )
                )

            target_count = int(round(target_count * load_scale))
            target_count = max(8, min(per_presence_cap, target_count))

            owned_ids = sorted(
                [
                    particle_id
                    for particle_id, state in particles.items()
                    if isinstance(state, dict)
                    and str(state.get("owner", "")).strip() == presence_id
                    and not bool(state.get("is_nexus", False))
                    and not bool(state.get("is_chaos_butterfly", False))
                ]
            )

            if len(owned_ids) > target_count:
                trimmed_ids = owned_ids[target_count:]
                for particle_id in trimmed_ids:
                    particles.pop(particle_id, None)
                owned_ids = owned_ids[:target_count]

            while len(owned_ids) < target_count:
                spawn_seq += 1
                particle_id = f"field:{presence_id}:{spawn_seq:06d}"
                target_presence_id = _choose_target_presence(
                    owner_presence_id=presence_id,
                    presence_ids=presence_ids,
                    particle_id=particle_id,
                    now=now_seconds,
                )
                particles[particle_id] = _spawn_particle(
                    particle_id=particle_id,
                    slot_index=len(owned_ids),
                    presence_id=presence_id,
                    target_presence_id=target_presence_id,
                    anchor=anchors[presence_id],
                    file_influence=file_influence,
                    click_influence=click_influence,
                    world_influence=world_influence,
                    resource_influence=resource_influence,
                    queue_ratio=queue_pressure,
                    now=now_seconds,
                    resource_wallet=impact.get(
                        "resource_wallet"
                    ),  # Pass wallet for cost check
                )
                owned_ids.append(particle_id)
                spawned_count += 1

            for particle_id in owned_ids:
                active_ids.add(particle_id)

        # SPAWN CHAOS BUTTERFLIES - separate from presence-based spawning
        # Chaos butterflies exist independently and spread noise
        chaos_owned_ids = sorted(
            [
                particle_id
                for particle_id, state in particles.items()
                if isinstance(state, dict)
                and bool(state.get("is_chaos_butterfly", False))
            ]
        )

        if len(chaos_owned_ids) > chaos_target_count:
            # Trim excess chaos butterflies
            trimmed_chaos = chaos_owned_ids[chaos_target_count:]
            for particle_id in trimmed_chaos:
                particles.pop(particle_id, None)
            chaos_owned_ids = chaos_owned_ids[:chaos_target_count]

        # Get chaos butterfly anchor
        chaos_anchor = anchors.get("chaos_butterfly", {"x": 0.5, "y": 0.15})

        while len(chaos_owned_ids) < chaos_target_count:
            spawn_seq += 1
            particle_id = f"chaos:butterfly:{spawn_seq:06d}"
            particles[particle_id] = _spawn_chaos_butterfly(
                particle_id=particle_id,
                slot_index=len(chaos_owned_ids),
                anchor=chaos_anchor,
                now=now_seconds,
            )
            chaos_owned_ids.append(particle_id)
            spawned_count += 1

        for particle_id in chaos_owned_ids:
            active_ids.add(particle_id)

        # Spawn/refresh nexus particles (passive daimo without agency).
        node_rows = file_nodes[:nexus_cap]
        fallback_owner = "anchor_registry"
        if fallback_owner not in anchors and presence_ids:
            fallback_owner = presence_ids[0]
        if fallback_owner not in anchors and anchors:
            fallback_owner = next(iter(anchors.keys()))
        desired_nexus_ids: set[str] = set()
        for node in node_rows:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            particle_id = f"nexus:{node_id}"
            desired_nexus_ids.add(particle_id)
            owner_id = str(node.get("dominant_presence", "")).strip() or fallback_owner
            if owner_id not in anchors:
                owner_id = fallback_owner

            existing = particles.get(particle_id)
            if isinstance(existing, dict):
                existing["owner"] = owner_id
                existing["target"] = owner_id
                existing["preferred_x"] = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
                existing["preferred_y"] = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
                existing["is_nexus"] = True
                existing["is_static_daimoi"] = True
                existing["source_node_id"] = node_id
                existing["ts"] = now_monotonic
            else:
                particles[particle_id] = _spawn_nexus_particle(
                    particle_id=particle_id,
                    node=node,
                    owner_presence_id=owner_id,
                )
                spawned_count += 1
            active_ids.add(particle_id)

        for particle_id in list(particles.keys()):
            row = particles.get(particle_id)
            if not isinstance(row, dict):
                continue
            if (
                bool(row.get("is_nexus", False))
                and particle_id not in desired_nexus_ids
            ):
                particles.pop(particle_id, None)

        stale_before = now_monotonic - 150.0
        for particle_id in list(particles.keys()):
            state = particles.get(particle_id, {})
            ts = _safe_float(
                (state if isinstance(state, dict) else {}).get("ts", 0.0), 0.0
            )
            if particle_id not in active_ids and ts < stale_before:
                particles.pop(particle_id, None)

        states = [
            particles[particle_id]
            for particle_id in sorted(active_ids)
            if isinstance(particles.get(particle_id), dict)
        ]

        node_items: list[dict[str, Any]] = []
        for node in node_rows:
            nx = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
            ny = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
            node_items.append(
                {
                    "x": nx,
                    "y": ny,
                    "vector": list(node.get("vector", [])),
                    "embedded_bonus": _safe_float(node.get("embedded_bonus", 0.0), 0.0),
                    "strength_base": 0.00018
                    + (_safe_float(node.get("importance", 0.3), 0.3) * 0.00044)
                    + (_safe_float(node.get("embedded_bonus", 0.0), 0.0) * 0.00036),
                }
            )

        state_count = len(states)
        node_tree_max_items = 20
        state_tree_max_items = 20
        if state_count > 760:
            node_tree_max_items = 30
            state_tree_max_items = 32
        elif state_count > 520:
            node_tree_max_items = 26
            state_tree_max_items = 28
        elif state_count > 320:
            node_tree_max_items = 24
            state_tree_max_items = 24

        node_tree = _quadtree_build(
            node_items, bounds=(0.0, 0.0, 1.0, 1.0), max_items=node_tree_max_items
        )
        anchor_entries = [
            (
                presence_id,
                _safe_float(anchor.get("x", 0.5), 0.5),
                _safe_float(anchor.get("y", 0.5), 0.5),
            )
            for presence_id, anchor in anchors.items()
        ]

        state_tree = _quadtree_build(
            states,
            bounds=(0.0, 0.0, 1.0, 1.0),
            max_items=state_tree_max_items,
        )

        nexus_current_stride = 1
        if state_count > 640:
            nexus_current_stride = 4
        elif state_count > 420:
            nexus_current_stride = 3
        elif state_count > 220:
            nexus_current_stride = 2
        if prior_tick_ms >= 96.0:
            nexus_current_stride += 2
        elif prior_tick_ms >= 72.0:
            nexus_current_stride += 1

        chaos_perturb_stride = 1 if state_count <= 320 else 2
        if prior_tick_ms >= 96.0:
            chaos_perturb_stride += 2
        elif prior_tick_ms >= 72.0:
            chaos_perturb_stride += 1

        node_force_stride = 1
        if state_count > 760:
            node_force_stride = 5
        elif state_count > 520:
            node_force_stride = 4
        elif state_count > 320:
            node_force_stride = 3
        elif state_count > 260:
            node_force_stride = 2
        if prior_tick_ms >= 96.0:
            node_force_stride += 2
        elif prior_tick_ms >= 72.0:
            node_force_stride += 1

        node_nearby_limit = 28
        if state_count > 760:
            node_nearby_limit = 12
        elif state_count > 520:
            node_nearby_limit = 16
        elif state_count > 320:
            node_nearby_limit = 20

        for state in states:
            owner_id = str(state.get("owner", "")).strip()
            target_id = str(state.get("target", owner_id)).strip() or owner_id
            owner_anchor = anchors.get(
                owner_id, anchors.get(target_id, {"x": 0.5, "y": 0.5})
            )
            target_anchor = anchors.get(target_id, owner_anchor)
            owner_anchor_x = _safe_float(owner_anchor.get("x", 0.5), 0.5)
            owner_anchor_y = _safe_float(owner_anchor.get("y", 0.5), 0.5)
            target_anchor_x = _safe_float(target_anchor.get("x", 0.5), 0.5)
            target_anchor_y = _safe_float(target_anchor.get("y", 0.5), 0.5)

            px = _clamp01(_safe_float(state.get("x", 0.5), 0.5))
            py = _clamp01(_safe_float(state.get("y", 0.5), 0.5))
            pvx = _safe_float(state.get("vx", 0.0), 0.0)
            pvy = _safe_float(state.get("vy", 0.0), 0.0)
            age = _safe_int(state.get("age", 0), 0) + 1
            state["age"] = age

            is_nexus = bool(state.get("is_nexus", False))
            is_chaos = bool(state.get("is_chaos_butterfly", False))
            msg_prob = _message_probability(state) if not is_nexus else 0.0

            if is_nexus:
                fx = 0.0
                fy = 0.0
                current_stride = nexus_current_stride
                should_sample_current = current_stride <= 1 or (
                    age % current_stride == 0
                )
                if should_sample_current:
                    current_candidates: list[dict[str, Any]] = []
                    _quadtree_query_radius(state_tree, px, py, 0.16, current_candidates)
                    current_x = 0.0
                    current_y = 0.0
                    current_weight = 0.0
                    for other_state in current_candidates:
                        if str(other_state.get("id", "")) == str(state.get("id", "")):
                            continue
                        if bool(other_state.get("is_nexus", False)):
                            continue
                        ox = _safe_float(other_state.get("x", 0.5), 0.5)
                        oy = _safe_float(other_state.get("y", 0.5), 0.5)
                        odx = ox - px
                        ody = oy - py
                        dist_sq = (odx * odx) + (ody * ody)
                        if dist_sq <= 1e-8 or dist_sq > (0.16 * 0.16):
                            continue
                        dist = math.sqrt(dist_sq)
                        falloff = _clamp01(1.0 - (dist / 0.16))
                        current_x += (
                            _safe_float(other_state.get("vx", 0.0), 0.0) * falloff
                        )
                        current_y += (
                            _safe_float(other_state.get("vy", 0.0), 0.0) * falloff
                        )
                        current_weight += falloff
                    if current_weight > 1e-8:
                        sampled_vx = current_x / current_weight
                        sampled_vy = current_y / current_weight
                        state["cached_current_vx"] = sampled_vx
                        state["cached_current_vy"] = sampled_vy
                        fx += sampled_vx * 0.58
                        fy += sampled_vy * 0.58
                else:
                    fx += _safe_float(state.get("cached_current_vx", 0.0), 0.0) * 0.58
                    fy += _safe_float(state.get("cached_current_vy", 0.0), 0.0) * 0.58

                pref_x = _clamp01(_safe_float(state.get("preferred_x", px), px))
                pref_y = _clamp01(_safe_float(state.get("preferred_y", py), py))
                fx += (pref_x - px) * 0.01
                fy += (pref_y - py) * 0.01
                simplex_phase = now_seconds * 0.23
                fx += (
                    _simplex_noise_2d(
                        (px * 4.2) + simplex_phase,
                        (py * 4.2) + (simplex_phase * 0.67),
                        seed=31,
                    )
                    * 0.00028
                )
                fy += (
                    _simplex_noise_2d(
                        (px * 4.2) + 19.0 + (simplex_phase * 0.53),
                        (py * 4.2) + 7.0 + simplex_phase,
                        seed=43,
                    )
                    * 0.00028
                )
                damping = 0.95
                speed_cap = 0.0048
            elif is_chaos:
                fx, fy = _chaos_field_perturbation(px, py, now_seconds, amplitude=0.025)
                noise_freq = 3.0
                time_scale = 0.4
                fx += (
                    _simplex_noise_2d(px * noise_freq, now_seconds * time_scale, seed=7)
                    * 0.008
                )
                fy += (
                    _simplex_noise_2d(
                        py * noise_freq + 50, now_seconds * time_scale, seed=8
                    )
                    * 0.008
                )

                noise_amp = _safe_float(state.get("noise_amplitude", 0.12), 0.12)
                perturb_stride = chaos_perturb_stride
                if age % perturb_stride == 0:
                    perturb_candidates: list[dict[str, Any]] = []
                    _quadtree_query_radius(state_tree, px, py, 0.15, perturb_candidates)
                    for other_state in perturb_candidates:
                        if str(other_state.get("id", "")) == str(state.get("id", "")):
                            continue
                        if bool(other_state.get("is_chaos_butterfly", False)):
                            continue
                        other_x = _safe_float(other_state.get("x", 0.5), 0.5)
                        other_y = _safe_float(other_state.get("y", 0.5), 0.5)
                        dist_sq = ((other_x - px) ** 2) + ((other_y - py) ** 2)
                        if dist_sq > 0.0225:
                            continue
                        dist = math.sqrt(dist_sq) if dist_sq > 1e-8 else 0.15
                        falloff = 1.0 - (dist / 0.15)
                        perturb_x = _simplex_noise_2d(
                            other_x * 10.0, now_seconds, seed=int(age)
                        )
                        perturb_y = _simplex_noise_2d(
                            other_y * 10.0 + 25, now_seconds, seed=int(age) + 1
                        )
                        other_state["vx"] = _safe_float(
                            other_state.get("vx", 0.0), 0.0
                        ) + (perturb_x * noise_amp * falloff)
                        other_state["vy"] = _safe_float(
                            other_state.get("vy", 0.0), 0.0
                        ) + (perturb_y * noise_amp * falloff)
                damping = 0.88
                speed_cap = 0.012
            else:
                owner_pull = 0.010 + (
                    _safe_float(local_density_map.get(owner_id, 0.0), 0.0) * 0.008
                )
                fx = (owner_anchor_x - px) * owner_pull
                fy = (owner_anchor_y - py) * owner_pull
                if target_id != owner_id:
                    target_pull = 0.004 + (msg_prob * 0.016)
                    fx += (target_anchor_x - px) * target_pull
                    fy += (target_anchor_y - py) * target_pull
                orbit_phase = (
                    _safe_float(now_seconds, 0.0) * (0.5 + (msg_prob * 0.9))
                ) + (_stable_ratio(f"{state.get('id', '')}|orbit", age + 1) * math.tau)
                fx += math.cos(orbit_phase) * 0.00062
                fy += math.sin(orbit_phase) * 0.00062
                simplex_amp = (
                    0.0002
                    + (msg_prob * 0.00042)
                    + ((1.0 - resource_pressure) * 0.00012)
                )
                simplex_phase = now_seconds * (0.29 + (msg_prob * 0.22))
                simplex_seed_base = int(age + (len(owner_id) * 17))
                fx += (
                    _simplex_noise_2d(
                        (px * 5.0) + simplex_phase,
                        (py * 5.0) + (simplex_phase * 0.71),
                        seed=simplex_seed_base,
                    )
                    * simplex_amp
                )
                fy += (
                    _simplex_noise_2d(
                        (px * 5.0) + 13.0 + (simplex_phase * 0.57),
                        (py * 5.0) + 29.0 + simplex_phase,
                        seed=simplex_seed_base + 53,
                    )
                    * simplex_amp
                )
                damping = max(0.74, 0.92 - (resource_pressure * 0.16))
                speed_cap = (
                    0.0052 + ((1.0 - resource_pressure) * 0.0026) + (msg_prob * 0.0014)
                )

            if not is_chaos and not is_nexus:
                should_sample_nodes = node_force_stride <= 1 or (
                    age % node_force_stride == 0
                )
                if should_sample_nodes:
                    semantic_vector = _state_unit_vector(state, "e_curr")
                    nearby_nodes: list[dict[str, Any]] = []
                    _quadtree_query_radius(
                        node_tree,
                        px,
                        py,
                        DAIMOI_NODE_INFLUENCE_RADIUS,
                        nearby_nodes,
                    )
                    nearby_limit = node_nearby_limit
                    if len(nearby_nodes) > nearby_limit and nearby_limit > 0:
                        step = max(
                            2,
                            int(
                                math.ceil(
                                    float(len(nearby_nodes)) / float(nearby_limit)
                                )
                            ),
                        )
                        nearby_nodes = nearby_nodes[::step]

                    node_fx = 0.0
                    node_fy = 0.0
                    for node_data in nearby_nodes:
                        nx = _safe_float(node_data.get("x", 0.5), 0.5)
                        ny = _safe_float(node_data.get("y", 0.5), 0.5)
                        dx = nx - px
                        if (
                            dx > DAIMOI_NODE_INFLUENCE_RADIUS
                            or dx < -DAIMOI_NODE_INFLUENCE_RADIUS
                        ):
                            continue
                        dy = ny - py
                        if (
                            dy > DAIMOI_NODE_INFLUENCE_RADIUS
                            or dy < -DAIMOI_NODE_INFLUENCE_RADIUS
                        ):
                            continue
                        distance_sq = (dx * dx) + (dy * dy)
                        if distance_sq > (
                            DAIMOI_NODE_INFLUENCE_RADIUS * DAIMOI_NODE_INFLUENCE_RADIUS
                        ):
                            continue
                        distance = math.sqrt(distance_sq)
                        if (
                            distance <= DAIMOI_NODE_INFLUENCE_RADIUS_EPS
                            or distance > DAIMOI_NODE_INFLUENCE_RADIUS
                        ):
                            continue
                        node_vector = list(node_data.get("vector", []))
                        embedded_bonus = _safe_float(
                            node_data.get("embedded_bonus", 0.0), 0.0
                        )
                        strength_base = _safe_float(
                            node_data.get("strength_base", 0.00018), 0.00018
                        )
                        similarity = _safe_cosine_unit(semantic_vector, node_vector)
                        falloff = _clamp01(
                            1.0 - (distance / DAIMOI_NODE_INFLUENCE_RADIUS)
                        )
                        signed = max(
                            -1.0,
                            min(
                                1.0,
                                (similarity * 0.72) + (embedded_bonus * 0.28) - 0.08,
                            ),
                        )
                        strength = strength_base * falloff
                        direction = 1.0 if signed >= 0.0 else -1.0
                        node_fx += (dx / distance) * strength * direction
                        node_fy += (dy / distance) * strength * direction
                    state["cached_node_fx"] = node_fx
                    state["cached_node_fy"] = node_fy
                    fx += node_fx
                    fy += node_fy
                else:
                    fx += _safe_float(state.get("cached_node_fx", 0.0), 0.0)
                    fy += _safe_float(state.get("cached_node_fy", 0.0), 0.0)

            fx += _world_edge_inward_pressure(
                px,
                edge_band=DAIMOI_WORLD_EDGE_BAND,
                pressure=DAIMOI_WORLD_EDGE_PRESSURE,
            )
            fy += _world_edge_inward_pressure(
                py,
                edge_band=DAIMOI_WORLD_EDGE_BAND,
                pressure=DAIMOI_WORLD_EDGE_PRESSURE,
            )

            vx = (pvx * damping) + fx
            vy = (pvy * damping) + fy
            speed = math.sqrt((vx * vx) + (vy * vy))
            if speed > speed_cap and speed > 1e-8:
                scale = speed_cap / speed
                vx *= scale
                vy *= scale

            next_x, next_y = px + vx, py + vy
            next_x, vx = _reflect_world_axis(
                next_x,
                vx,
                bounce=DAIMOI_WORLD_EDGE_BOUNCE,
            )
            next_y, vy = _reflect_world_axis(
                next_y,
                vy,
                bounce=DAIMOI_WORLD_EDGE_BOUNCE,
            )

            state["vx"] = vx
            state["vy"] = vy
            state["x"] = next_x
            state["y"] = next_y
            state["ts"] = now_monotonic

        collision_tree_max_items = 16
        if state_count > 760:
            collision_tree_max_items = 30
        elif state_count > 520:
            collision_tree_max_items = 24
        elif state_count > 320:
            collision_tree_max_items = 20
        collision_tree = _quadtree_build(
            states,
            bounds=(0.0, 0.0, 1.0, 1.0),
            max_items=collision_tree_max_items,
        )
        max_radius = 0.02
        max_dynamic_radius = 0.02
        for state in states:
            radius_value = _safe_float(state.get("radius", 0.015), 0.015)
            max_radius = max(max_radius, radius_value)
            if not bool(state.get("is_nexus", False)):
                max_dynamic_radius = max(max_dynamic_radius, radius_value)

        semantic_budget = max(220, min(1200, state_count * 2))
        semantic_updates = 0
        semantic_stride = 2
        if state_count > 760:
            semantic_stride = 8
        elif state_count > 560:
            semantic_stride = 6
        elif state_count > 360:
            semantic_stride = 5
        elif state_count > 220:
            semantic_stride = 4
        elif state_count > 140:
            semantic_stride = 3
        if prior_tick_ms >= 110.0:
            semantic_stride += 2
        elif prior_tick_ms >= 82.0:
            semantic_stride += 1

        pair_scan_budget = max(2200, min(26000, state_count * 30))
        max_collisions_per_tick = max(700, min(9000, state_count * 9))
        if prior_tick_ms >= 120.0:
            pair_scan_budget = int(pair_scan_budget * 0.5)
            max_collisions_per_tick = int(max_collisions_per_tick * 0.45)
        elif prior_tick_ms >= 92.0:
            pair_scan_budget = int(pair_scan_budget * 0.62)
            max_collisions_per_tick = int(max_collisions_per_tick * 0.58)
        elif prior_tick_ms >= 72.0:
            pair_scan_budget = int(pair_scan_budget * 0.78)
            max_collisions_per_tick = int(max_collisions_per_tick * 0.74)

        collision_divisor = 1
        if state_count > 640:
            collision_divisor = 8
        elif state_count > 420:
            collision_divisor = 6
        elif state_count > 260:
            collision_divisor = 4
        elif state_count > 160:
            collision_divisor = 3
        collision_bucket = (
            now_seconds_int % collision_divisor if collision_divisor > 1 else 0
        )
        surface_stride = 1
        if state_count > 760:
            surface_stride = 4
        elif state_count > 520:
            surface_stride = 3
        elif state_count > 320:
            surface_stride = 2
        if prior_tick_ms >= 110.0:
            surface_stride += 2
        elif prior_tick_ms >= 82.0:
            surface_stride += 1

        pair_scan_index = 0
        pair_scan_count = 0
        collision_loop_stop = False
        for left in states:
            left_id = str(left.get("id", "")).strip()
            if not left_id:
                continue
            left_is_nexus = bool(left.get("is_nexus", False))
            if left_is_nexus:
                # Nexus collisions are resolved when active daimoi queries neighbors.
                continue
            left_x = _safe_float(left.get("x", 0.5), 0.5)
            left_y = _safe_float(left.get("y", 0.5), 0.5)
            left_radius = _safe_float(left.get("radius", 0.015), 0.015)
            left_mass = max(0.2, _safe_float(left.get("mass", 0.8), 0.8))
            left_vx = _safe_float(left.get("vx", 0.0), 0.0)
            left_vy = _safe_float(left.get("vy", 0.0), 0.0)
            left_collision_count = _safe_int(left.get("collisions", 0), 0)

            collision_candidates: list[dict[str, Any]] = []
            _quadtree_query_radius(
                collision_tree,
                left_x,
                left_y,
                left_radius + max_dynamic_radius,
                collision_candidates,
            )

            for right in collision_candidates:
                right_id = str(right.get("id", "")).strip()
                if not right_id or right_id == left_id:
                    continue
                if right_id <= left_id:
                    continue
                right_is_nexus = bool(right.get("is_nexus", False))

                if collision_divisor > 1:
                    pair_scan_index += 1
                    if (pair_scan_index % collision_divisor) != collision_bucket:
                        continue

                pair_scan_count += 1
                if pair_scan_count > pair_scan_budget:
                    collision_loop_stop = True
                    break

                right_radius = _safe_float(right.get("radius", 0.015), 0.015)
                contact = max(
                    1e-6,
                    left_radius + right_radius,
                )
                right_x = _safe_float(right.get("x", 0.5), 0.5)
                right_y = _safe_float(right.get("y", 0.5), 0.5)
                dx = right_x - left_x
                if dx > contact or dx < -contact:
                    continue
                dy = right_y - left_y
                if dy > contact or dy < -contact:
                    continue
                distance_sq = (dx * dx) + (dy * dy)
                if distance_sq > (contact * contact):
                    continue

                distance = math.sqrt(distance_sq)
                if distance <= 1e-8:
                    jitter = (
                        _stable_ratio(f"{left_id}|{right_id}|pair", 3) - 0.5
                    ) * 0.001
                    dx += jitter
                    dy -= jitter
                    distance = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))

                nx = dx / distance
                ny = dy / distance
                overlap = max(0.0, contact - distance)

                mass_right = max(0.2, _safe_float(right.get("mass", 0.8), 0.8))
                mass_total = max(1e-6, left_mass + mass_right)
                if overlap > 0.0:
                    left_share = mass_right / mass_total
                    right_share = left_mass / mass_total
                    left_x = _clamp01(left_x - (nx * overlap * left_share))
                    left_y = _clamp01(left_y - (ny * overlap * left_share))
                    right_x = _clamp01(right_x + (nx * overlap * right_share))
                    right_y = _clamp01(right_y + (ny * overlap * right_share))
                    left["x"] = left_x
                    left["y"] = left_y
                    right["x"] = right_x
                    right["y"] = right_y

                rvx = _safe_float(right.get("vx", 0.0), 0.0)
                rvy = _safe_float(right.get("vy", 0.0), 0.0)
                relative_normal = ((rvx - left_vx) * nx) + ((rvy - left_vy) * ny)
                restitution = 0.72
                if relative_normal < 0.0:
                    impulse = (-(1.0 + restitution) * relative_normal) / (
                        (1.0 / left_mass) + (1.0 / mass_right)
                    )
                else:
                    impulse = overlap * 0.014

                left_vx = left_vx - ((impulse / left_mass) * nx)
                left_vy = left_vy - ((impulse / left_mass) * ny)
                left["vx"] = left_vx
                left["vy"] = left_vy
                right["vx"] = rvx + ((impulse / mass_right) * nx)
                right["vy"] = rvy + ((impulse / mass_right) * ny)

                left_collision_count += 1
                left["collisions"] = left_collision_count
                right["collisions"] = _safe_int(right.get("collisions", 0), 0) + 1

                if not right_is_nexus:
                    should_update_semantics = semantic_updates < semantic_budget and (
                        (collision_count % semantic_stride == 0)
                        or (abs(impulse) >= (DAIMOI_IMPULSE_REFERENCE * 0.65))
                    )
                    if should_update_semantics:
                        semantics = _collision_semantic_update(
                            left, right, impulse=abs(impulse)
                        )
                        matrix_accumulator["ss"] += semantics["ss"]
                        matrix_accumulator["sc"] += semantics["sc"]
                        matrix_accumulator["cs"] += semantics["cs"]
                        matrix_accumulator["cc"] += semantics["cc"]
                        matrix_accumulator["samples"] += 1
                        semantic_updates += 1

                        # --- Apply resource transfer ---
                        if "resource_transfer" in semantics:
                            transfers = semantics["resource_transfer"]
                            if isinstance(transfers, dict):
                                if "left_to_right" in transfers:
                                    # Left (resource packet) -> Right (consumer)
                                    packet = transfers["left_to_right"]
                                    if isinstance(packet, dict):
                                        delta_r = right.get("wallet_delta")
                                        if not isinstance(delta_r, dict):
                                            delta_r = {}
                                            right["wallet_delta"] = delta_r
                                        for res, amt in packet.items():
                                            res_key = str(res)
                                            delta_r[res_key] = _safe_float(
                                                delta_r.get(res_key, 0.0), 0.0
                                            ) + _safe_float(amt, 0.0)

                                if "right_to_left" in transfers:
                                    packet = transfers["right_to_left"]
                                    if isinstance(packet, dict):
                                        delta_l = left.get("wallet_delta")
                                        if not isinstance(delta_l, dict):
                                            delta_l = {}
                                            left["wallet_delta"] = delta_l
                                        for res, amt in packet.items():
                                            res_key = str(res)
                                            delta_l[res_key] = _safe_float(
                                                delta_l.get(res_key, 0.0), 0.0
                                            ) + _safe_float(amt, 0.0)

                                if "right_to_left" in transfers:
                                    packet = transfers["right_to_left"]
                                    if isinstance(packet, dict):
                                        if not isinstance(
                                            left.get("wallet_delta"), dict
                                        ):
                                            left["wallet_delta"] = {}
                                        for res, amt in packet.items():
                                            left["wallet_delta"][res] = _safe_float(
                                                left["wallet_delta"].get(res, 0.0), 0.0
                                            ) + _safe_float(amt, 0.0)

                collision_count += 1
                if collision_count >= max_collisions_per_tick:
                    collision_loop_stop = True
                    break

            if collision_loop_stop:
                break

        remove_ids: set[str] = set()
        for state in states:
            particle_id = str(state.get("id", "")).strip()
            if not particle_id:
                continue
            if bool(state.get("is_nexus", False)):
                continue

            if surface_stride > 1:
                state_age = _safe_int(state.get("age", 0), 0)
                if (state_age % surface_stride) != 0:
                    continue

            px = _clamp01(_safe_float(state.get("x", 0.5), 0.5))
            py = _clamp01(_safe_float(state.get("y", 0.5), 0.5))
            owner_id = str(state.get("owner", "")).strip()
            target_id = str(state.get("target", owner_id)).strip() or owner_id

            best_presence = ""
            best_distance = 1e9
            contact_limit = DAIMOI_SURFACE_RADIUS + _safe_float(
                state.get("radius", 0.014), 0.014
            )
            for presence_id, anchor_x, anchor_y in anchor_entries:
                dx = anchor_x - px
                if dx > contact_limit or dx < -contact_limit:
                    continue
                dy = anchor_y - py
                if dy > contact_limit or dy < -contact_limit:
                    continue
                distance = math.sqrt((dx * dx) + (dy * dy))
                if distance <= contact_limit and distance < best_distance:
                    best_distance = distance
                    best_presence = presence_id

            if not best_presence:
                continue

            surface = _surface_state(surfaces, best_presence)
            job_probs = _job_probabilities(state)
            message_prob = _message_probability(state)
            action_probs = _action_probabilities(job_probs, message_prob)
            diffuse_prob = _clamp01(_safe_float(action_probs.get("diffuse", 0.5), 0.5))
            if best_presence == owner_id:
                diffuse_prob = _clamp01(diffuse_prob * 0.45)

            roll = _stable_ratio(
                f"{particle_id}|surface|{best_presence}|{now_seconds_tenths_int}",
                5,
            )
            action = "diffuse" if roll < diffuse_prob else "deflect"

            anchor = anchors.get(
                best_presence, {"x": 0.5, "y": 0.5, "embedding": _normalize_vector([])}
            )
            anchor_embedding = _normalize_vector(list(anchor.get("embedding", [])))
            contact_strength = _clamp01(
                1.0 - (best_distance / max(1e-6, DAIMOI_SURFACE_RADIUS))
            )
            packet_contract = _packet_component_contract_for_state(state, top_k=4)
            packet_components = _packet_components_from_job_probabilities(job_probs)
            resource_signature = dict(packet_contract.get("resource_signature", {}))
            presence_impact = impact_by_id.get(best_presence, {})
            need_by_resource = _presence_need_by_resource(
                presence_impact if isinstance(presence_impact, dict) else {},
                queue_ratio=queue_pressure,
            )
            absorb_sample = _sample_absorb_component(
                components=packet_components,
                lens_embedding=anchor_embedding,
                need_by_resource=need_by_resource,
                context={
                    "pressure": resource_pressure,
                    "congestion": _clamp01(_safe_float(collision_count, 0.0) / 1200.0),
                    "wallet_pressure": _clamp01(
                        sum(
                            max(0.0, 1.0 - _safe_float(value, 0.0))
                            for value in need_by_resource.values()
                        )
                        / float(max(1, len(need_by_resource)))
                    ),
                    "message_entropy": _clamp01(
                        _dirichlet_entropy(job_probs)
                        / max(1.0, math.log(len(DAIMOI_JOB_KEYS)))
                    ),
                    "queue": queue_pressure,
                    "contact": contact_strength,
                },
                seed=f"{particle_id}|absorb|{best_presence}|{now_seconds_int}",
            )

            triggered_job = str(absorb_sample.get("selected_component_id", "")).strip()
            if not triggered_job:
                triggered_job = _sample_job_key(
                    job_probs,
                    seed=f"{particle_id}|job|{best_presence}|{now_seconds_int}",
                )

            state["last_packet_contract"] = packet_contract
            state["last_resource_signature"] = resource_signature
            state["last_absorb_sampler"] = absorb_sample

            absorb_sampler_count += 1
            if len(absorb_sampler_events) < 24:
                absorb_sampler_events.append(
                    {
                        "particle_id": particle_id,
                        "presence_id": best_presence,
                        "owner_presence_id": owner_id,
                        "action": action,
                        "contact_strength": round(contact_strength, 6),
                        "selected_component_id": triggered_job,
                        "sampler": absorb_sample,
                    }
                )

            job_trigger_counts[triggered_job] = (
                _safe_int(job_trigger_counts.get(triggered_job, 0), 0) + 1
            )
            surface_job_hits = surface.get("job_hits", {})
            if not isinstance(surface_job_hits, dict):
                surface_job_hits = {}
            surface_job_hits[triggered_job] = (
                _safe_int(surface_job_hits.get(triggered_job, 0), 0) + 1
            )
            surface["job_hits"] = surface_job_hits

            if triggered_job == "deliver_message" and message_prob >= 0.45:
                delivery_count += 1

            if action == "diffuse":
                diffuse_count += 1
                remove_ids.add(particle_id)
                surface["diffuse_count"] = (
                    _safe_int(surface.get("diffuse_count", 0), 0) + 1
                )
                surface["field_energy"] = _safe_float(
                    surface.get("field_energy", 0.0), 0.0
                ) + (
                    _safe_float(state.get("size", 1.0), 1.0)
                    * (0.7 + (message_prob * 0.6))
                )
                surface["embedding"] = _blend_vectors(
                    list(surface.get("embedding", [])),
                    list(state.get("e_curr", [])),
                    0.24 + (contact_strength * 0.32),
                )
                surface_alpha_pkg = dict(surface.get("alpha_pkg", {}))
                state_alpha_pkg = dict(state.get("alpha_pkg", {}))
                if (
                    surface_alpha_pkg.keys() <= DAIMOI_JOB_KEYS_SET
                    and state_alpha_pkg.keys() <= DAIMOI_JOB_KEYS_SET
                ):
                    transfer_keys = DAIMOI_JOB_KEYS_SORTED
                else:
                    transfer_keys = tuple(
                        sorted(
                            set(
                                [
                                    *DAIMOI_JOB_KEYS,
                                    *surface_alpha_pkg.keys(),
                                    *state_alpha_pkg.keys(),
                                ]
                            )
                        )
                    )
                surface["alpha_pkg"] = _dirichlet_transfer(
                    surface_alpha_pkg,
                    state_alpha_pkg,
                    coupling=0.9,
                    transfer_t=0.8,
                    repulsion_u=0.05,
                    keys=transfer_keys,
                )
                surface["alpha_msg"] = _dirichlet_transfer(
                    dict(surface.get("alpha_msg", {})),
                    dict(state.get("alpha_msg", {})),
                    coupling=0.9,
                    transfer_t=0.8,
                    repulsion_u=0.05,
                    keys=("deliver", "hold"),
                )

                cell_x = int(px * 12.0)
                cell_y = int(py * 12.0)
                cell_key = f"{cell_x}:{cell_y}"
                cell = field_cells.get(cell_key, {})
                if not isinstance(cell, dict):
                    cell = {}
                cell["energy"] = _safe_float(cell.get("energy", 0.0), 0.0) + (
                    _safe_float(state.get("size", 1.0), 1.0) * 0.42
                )
                cell["embedding"] = _blend_vectors(
                    list(cell.get("embedding", [])),
                    list(state.get("e_curr", [])),
                    0.28,
                )
                cell["ts"] = now_monotonic
                field_cells[cell_key] = cell
            else:
                deflect_count += 1
                surface["deflect_count"] = (
                    _safe_int(surface.get("deflect_count", 0), 0) + 1
                )

                nx = px - _safe_float(anchor.get("x", 0.5), 0.5)
                ny = py - _safe_float(anchor.get("y", 0.5), 0.5)
                norm = math.sqrt((nx * nx) + (ny * ny))
                if norm <= 1e-8:
                    nx = 1.0
                    ny = 0.0
                    norm = 1.0
                nx /= norm
                ny /= norm

                vx = _safe_float(state.get("vx", 0.0), 0.0)
                vy = _safe_float(state.get("vy", 0.0), 0.0)
                dot = (vx * nx) + (vy * ny)
                rvx = (vx - (2.0 * dot * nx)) * 0.82
                rvy = (vy - (2.0 * dot * ny)) * 0.82
                state["vx"] = rvx
                state["vy"] = rvy

                impulse_mag = math.sqrt((rvx - vx) ** 2 + (rvy - vy) ** 2)
                surface["impulse"] = (
                    _safe_float(surface.get("impulse", 0.0), 0.0) + impulse_mag
                )
                surface["embedding"] = _blend_vectors(
                    list(surface.get("embedding", [])),
                    list(state.get("e_curr", [])),
                    0.08 + (contact_strength * 0.16),
                )
                state["e_curr"] = _blend_vectors(
                    list(state.get("e_curr", [])),
                    anchor_embedding,
                    0.06 + (contact_strength * 0.12),
                )

                if best_presence != owner_id:
                    state["owner"] = best_presence
                    state["handoffs"] = _safe_int(state.get("handoffs", 0), 0) + 1
                    handoff_count += 1
                    state["target"] = _choose_target_presence(
                        owner_presence_id=best_presence,
                        presence_ids=presence_ids,
                        particle_id=particle_id,
                        now=now_seconds,
                    )

            surface["ts"] = now_monotonic

        for particle_id in remove_ids:
            particles.pop(particle_id, None)

        stale_cell_before = now_monotonic - 300.0
        for cell_key in list(field_cells.keys()):
            cell = field_cells.get(cell_key, {})
            ts = _safe_float(
                (cell if isinstance(cell, dict) else {}).get("ts", 0.0), 0.0
            )
            if ts < stale_cell_before:
                field_cells.pop(cell_key, None)

        runtime["particles"] = particles
        runtime["surfaces"] = surfaces
        runtime["field_cells"] = field_cells
        runtime["spawn_seq"] = spawn_seq
        tick_ms = round((time.monotonic() - now_monotonic) * 1000.0, 3)
        prev_ema = _safe_float(runtime.get("tick_ms_ema", tick_ms), tick_ms)
        runtime["tick_ms"] = tick_ms
        runtime["tick_ms_ema"] = round((prev_ema * 0.72) + (tick_ms * 0.28), 3)
        _DAIMO_DYNAMICS_CACHE["field_particles"] = runtime

        output_rows: list[dict[str, Any]] = []
        active_states = [
            state
            for state in particles.values()
            if isinstance(state, dict)
            and str(state.get("owner", "")).strip() in anchors
            and not bool(
                state.get("is_chaos_butterfly", False)
            )  # Exclude chaos - rendered separately
        ]
        active_states.sort(
            key=lambda row: (
                str(row.get("owner", "")),
                str(row.get("id", "")),
            )
        )

        output_owner_cap = 160
        output_nexus_cap = 220
        output_total_cap = max(320, min(1400, state_count + 280))
        if prior_tick_ms >= 120.0:
            output_owner_cap = 44
            output_nexus_cap = 90
            output_total_cap = 420
        elif prior_tick_ms >= 92.0:
            output_owner_cap = 62
            output_nexus_cap = 120
            output_total_cap = 560
        elif prior_tick_ms >= 72.0:
            output_owner_cap = 82
            output_nexus_cap = 150
            output_total_cap = 760

        if len(active_states) > output_total_cap:
            limited_states: list[dict[str, Any]] = []
            owner_counts: dict[str, int] = {}
            nexus_count = 0
            for state in active_states:
                is_nexus = bool(state.get("is_nexus", False))
                if is_nexus:
                    if nexus_count >= output_nexus_cap:
                        continue
                    nexus_count += 1
                else:
                    owner_key = str(state.get("owner", "")).strip()
                    owner_hit_count = owner_counts.get(owner_key, 0)
                    if owner_hit_count >= output_owner_cap:
                        continue
                    owner_counts[owner_key] = owner_hit_count + 1
                limited_states.append(state)
                if len(limited_states) >= output_total_cap:
                    break
            active_states = limited_states

        package_entropy_total = 0.0
        message_probability_total = 0.0

        for state in active_states:
            owner_id = str(state.get("owner", "")).strip()
            if not owner_id:
                continue
            is_nexus = bool(state.get("is_nexus", False))
            if is_nexus:
                role, mode = "nexus-passive", "static-daimoi"
            else:
                role, mode = _presence_role_and_mode(owner_id)
            vector = _state_unit_vector(state, "e_curr")
            hue = (math.degrees(math.atan2(vector[1], vector[0])) + 360.0) % 360.0
            if is_nexus:
                message_prob = 0.0
                job_probs = {key: 0.0 for key in DAIMOI_JOB_KEYS}
                action_probs = dict(NEXUS_PASSIVE_ACTION_PROBS)
            else:
                message_prob = _message_probability(state)
                job_probs = _job_probabilities(state)
                action_probs = _action_probabilities(job_probs, message_prob)

            if is_nexus:
                saturation = 0.24
                value = 0.66
            else:
                saturation = max(0.34, min(0.62, 0.42 + (message_prob * 0.18)))
                value = max(
                    0.34, min(0.8, 0.48 + (action_probs.get("deflect", 0.5) * 0.18))
                )
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (hue % 360.0) / 360.0, saturation, value
            )

            package_entropy = _dirichlet_entropy(job_probs) if not is_nexus else 0.0
            package_entropy_total += package_entropy
            message_probability_total += message_prob

            packet_contract_raw = state.get("last_packet_contract", {})
            packet_contract = (
                packet_contract_raw
                if isinstance(packet_contract_raw, dict)
                and "components" in packet_contract_raw
                and isinstance(packet_contract_raw.get("components", []), list)
                else _packet_component_contract_for_state(state, top_k=4)
            )
            packet_components = (
                list(packet_contract.get("components", []))
                if isinstance(packet_contract.get("components", []), list)
                else []
            )
            resource_signature_raw = state.get(
                "last_resource_signature",
                packet_contract.get("resource_signature", {}),
            )
            resource_signature = (
                resource_signature_raw
                if isinstance(resource_signature_raw, dict)
                else {}
            )

            absorb_sample_raw = state.get("last_absorb_sampler", {})
            if isinstance(absorb_sample_raw, dict) and absorb_sample_raw:
                absorb_sample = absorb_sample_raw
            else:
                impact = impact_by_id.get(owner_id, {})
                need_preview = _presence_need_by_resource(
                    impact if isinstance(impact, dict) else {},
                    queue_ratio=queue_pressure,
                )
                owner_anchor = anchors.get(owner_id, {})
                absorb_sample = _sample_absorb_component(
                    components=_packet_components_from_job_probabilities(job_probs),
                    lens_embedding=list(
                        (
                            owner_anchor
                            if isinstance(owner_anchor, dict)
                            else {"embedding": _normalize_vector([])}
                        ).get("embedding", [])
                    ),
                    need_by_resource=need_preview,
                    context={
                        "pressure": resource_pressure,
                        "congestion": _clamp01(
                            _safe_float(collision_count, 0.0) / 1200.0
                        ),
                        "wallet_pressure": _clamp01(
                            sum(
                                max(0.0, 1.0 - _safe_float(value, 0.0))
                                for value in need_preview.values()
                            )
                            / float(max(1, len(need_preview)))
                        ),
                        "message_entropy": _clamp01(
                            package_entropy / max(1.0, math.log(len(DAIMOI_JOB_KEYS)))
                        ),
                        "queue": queue_pressure,
                        "contact": 0.0,
                    },
                    seed=f"{state.get('id', '')}|absorb-preview|{owner_id}|{now_seconds_int}",
                )

            absorb_sampler_row = {
                "record": str(
                    absorb_sample.get("record", DAIMOI_ABSORB_SAMPLER_RECORD)
                ),
                "schema_version": str(
                    absorb_sample.get("schema_version", DAIMOI_ABSORB_SAMPLER_SCHEMA)
                ),
                "method": str(
                    absorb_sample.get("method", DAIMOI_ABSORB_SAMPLER_METHOD)
                ),
                "selected_component_id": str(
                    absorb_sample.get("selected_component_id", "")
                ),
                "selected_probability": round(
                    _clamp01(
                        _safe_float(absorb_sample.get("selected_probability", 0.0), 0.0)
                    ),
                    6,
                ),
                "beta": round(
                    max(0.0, _safe_float(absorb_sample.get("beta", 0.0), 0.0)), 6
                ),
                "temperature": round(
                    max(0.0, _safe_float(absorb_sample.get("temperature", 0.0), 0.0)),
                    6,
                ),
            }

            output_rows.append(
                {
                    "id": str(state.get("id", "")),
                    "presence_id": owner_id,
                    "owner_presence_id": owner_id,
                    "target_presence_id": str(state.get("target", owner_id)),
                    "source_node_id": str(state.get("source_node_id", "")),
                    "presence_role": role,
                    "particle_mode": mode,
                    "is_nexus": is_nexus,
                    "record": DAIMOI_PROBABILISTIC_RECORD,
                    "schema_version": DAIMOI_PROBABILISTIC_SCHEMA,
                    "packet_record": DAIMOI_PACKET_COMPONENT_RECORD,
                    "packet_schema_version": DAIMOI_PACKET_COMPONENT_SCHEMA,
                    "x": round(_clamp01(_safe_float(state.get("x", 0.5), 0.5)), 5),
                    "y": round(_clamp01(_safe_float(state.get("y", 0.5), 0.5)), 5),
                    "size": round(
                        max(0.6, _safe_float(state.get("size", 1.0), 1.0)), 5
                    ),
                    "mass": round(
                        max(0.35, _safe_float(state.get("mass", 0.8), 0.8)), 6
                    ),
                    "radius": round(
                        max(0.01, _safe_float(state.get("radius", 0.014), 0.014)), 6
                    ),
                    "vx": round(_safe_float(state.get("vx", 0.0), 0.0), 6),
                    "vy": round(_safe_float(state.get("vy", 0.0), 0.0), 6),
                    "r": round(_clamp01(r_raw), 5),
                    "g": round(_clamp01(g_raw), 5),
                    "b": round(_clamp01(b_raw), 5),
                    "message_probability": round(message_prob, 6),
                    "job_probabilities": _rounded_distribution(job_probs),
                    "packet_components": packet_components,
                    "resource_signature": {
                        resource: round(_clamp01(_safe_float(value, 0.0)), 6)
                        for resource, value in resource_signature.items()
                        if str(resource).strip()
                    },
                    "absorb_sampler": absorb_sampler_row,
                    "action_probabilities": _rounded_distribution(
                        {
                            "deflect": action_probs.get("deflect", 0.5),
                            "diffuse": action_probs.get("diffuse", 0.5),
                        }
                    ),
                    "behavior_actions": list(
                        state.get("behaviors", list(DAIMOI_BEHAVIOR_DEFAULTS))
                    ),
                    "top_job": max(
                        sorted(job_probs.keys()),
                        key=lambda key: _safe_float(job_probs.get(key, 0.0), 0.0),
                    )
                    if job_probs
                    else "deliver_message",
                    "package_entropy": round(package_entropy, 6),
                    "embedding_seed_preview": [
                        round(_safe_float(value, 0.0), 6)
                        for value in list(state.get("e_seed", []))[:3]
                    ],
                    "embedding_curr_preview": [
                        round(_safe_float(value, 0.0), 6)
                        for value in list(state.get("e_curr", []))[:3]
                    ],
                    "collision_count": _safe_int(state.get("collisions", 0), 0),
                    "last_collision_matrix": {
                        key: round(_safe_float(value, 0.0), 6)
                        for key, value in dict(
                            state.get("last_collision_matrix", {})
                        ).items()
                    },
                }
            )

    active_count = len(output_rows)
    mean_entropy = (
        package_entropy_total / float(active_count) if active_count > 0 else 0.0
    )
    mean_message_prob = (
        message_probability_total / float(active_count) if active_count > 0 else 0.0
    )

    matrix_samples = _safe_int(matrix_accumulator.get("samples", 0), 0)
    matrix_mean = {
        "ss": round(
            (_safe_float(matrix_accumulator.get("ss", 0.0), 0.0) / matrix_samples), 6
        )
        if matrix_samples > 0
        else 0.0,
        "sc": round(
            (_safe_float(matrix_accumulator.get("sc", 0.0), 0.0) / matrix_samples), 6
        )
        if matrix_samples > 0
        else 0.0,
        "cs": round(
            (_safe_float(matrix_accumulator.get("cs", 0.0), 0.0) / matrix_samples), 6
        )
        if matrix_samples > 0
        else 0.0,
        "cc": round(
            (_safe_float(matrix_accumulator.get("cc", 0.0), 0.0) / matrix_samples), 6
        )
        if matrix_samples > 0
        else 0.0,
    }

    summary = {
        "record": DAIMOI_PROBABILISTIC_RECORD,
        "schema_version": DAIMOI_PROBABILISTIC_SCHEMA,
        "active": active_count,
        "spawned": int(spawned_count),
        "collisions": int(collision_count),
        "deflects": int(deflect_count),
        "diffuses": int(diffuse_count),
        "handoffs": int(handoff_count),
        "deliveries": int(delivery_count),
        "job_triggers": {
            key: int(value) for key, value in sorted(job_trigger_counts.items())
        },
        "mean_package_entropy": round(mean_entropy, 6),
        "mean_message_probability": round(mean_message_prob, 6),
        "packet_contract": {
            "record": DAIMOI_PACKET_COMPONENT_RECORD,
            "schema_version": DAIMOI_PACKET_COMPONENT_SCHEMA,
            "resource_keys": list(DAIMOI_RESOURCE_KEYS),
            "top_k": 4,
        },
        "absorb_sampler": {
            "record": DAIMOI_ABSORB_SAMPLER_RECORD,
            "schema_version": DAIMOI_ABSORB_SAMPLER_SCHEMA,
            "method": DAIMOI_ABSORB_SAMPLER_METHOD,
            "events": int(absorb_sampler_count),
            "sample_events": absorb_sampler_events,
        },
        "matrix_mean": matrix_mean,
        "behavior_defaults": list(DAIMOI_BEHAVIOR_DEFAULTS),
    }
    return output_rows, summary


def reset_probabilistic_daimoi_state_for_tests() -> None:
    with _DAIMO_DYNAMICS_LOCK:
        _DAIMO_DYNAMICS_CACHE["field_particles"] = {
            "particles": {},
            "surfaces": {},
            "field_cells": {},
            "spawn_seq": 0,
        }


def run_probabilistic_collision_stress(
    *,
    iterations: int = 10000,
    seed: int = 17,
) -> dict[str, Any]:
    left = {
        "id": "stress:left",
        "size": 1.2,
        "e_seed": _embedding_from_text("stress left seed"),
        "e_curr": _embedding_from_text("stress left current"),
        "alpha_pkg": {
            key: DAIMOI_ALPHA_BASELINE + (_stable_ratio(f"left|{key}", 5) * 2.0)
            for key in DAIMOI_JOB_KEYS
        },
        "alpha_msg": {"deliver": 2.2, "hold": 1.4},
        "last_collision_matrix": {},
    }
    right = {
        "id": "stress:right",
        "size": 1.8,
        "e_seed": _embedding_from_text("stress right seed"),
        "e_curr": _embedding_from_text("stress right current"),
        "alpha_pkg": {
            key: DAIMOI_ALPHA_BASELINE + (_stable_ratio(f"right|{key}", 9) * 2.0)
            for key in DAIMOI_JOB_KEYS
        },
        "alpha_msg": {"deliver": 1.8, "hold": 2.1},
        "last_collision_matrix": {},
    }

    nan_count = 0
    negative_alpha_count = 0
    max_probability_sum_error = 0.0

    for index in range(max(1, int(iterations))):
        impulse = 0.001 + (_stable_ratio(f"{seed}|impulse|{index}", index + 1) * 0.065)
        left["size"] = 0.7 + (
            _stable_ratio(f"{seed}|left-size|{index}", index + 3) * 2.5
        )
        right["size"] = 0.7 + (
            _stable_ratio(f"{seed}|right-size|{index}", index + 5) * 2.5
        )

        _collision_semantic_update(left, right, impulse=impulse)

        for state in (left, right):
            state["e_curr"] = _normalize_vector(list(state.get("e_curr", [])))
            alpha_pkg = dict(state.get("alpha_pkg", {}))
            for value in alpha_pkg.values():
                value_float = _safe_float(value, 0.0)
                if math.isnan(value_float) or math.isinf(value_float):
                    nan_count += 1
                if value_float < 0.0:
                    negative_alpha_count += 1
            probabilities = _dirichlet_probabilities(alpha_pkg, keys=DAIMOI_JOB_KEYS)
            prob_sum = sum(_safe_float(value, 0.0) for value in probabilities.values())
            max_probability_sum_error = max(
                max_probability_sum_error, abs(prob_sum - 1.0)
            )

    return {
        "ok": nan_count == 0
        and negative_alpha_count == 0
        and max_probability_sum_error <= 1e-6,
        "iterations": int(iterations),
        "nan_count": int(nan_count),
        "negative_alpha_count": int(negative_alpha_count),
        "probability_sum_error_max": float(max_probability_sum_error),
    }
