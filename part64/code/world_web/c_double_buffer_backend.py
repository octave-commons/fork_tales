from __future__ import annotations

import atexit
import colorsys
import ctypes
import heapq
import math
import os
import shutil
import site
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from . import daimoi_observer as daimoi_observer_module
from .daimoi_probabilistic import (
    DAIMOI_ANTI_CLUMP_TARGET,
    _anti_clump_controller_update,
    _anti_clump_scales,
)

CDB_FLAG_NEXUS = 0x1
CDB_FLAG_CHAOS = 0x2
_CDB_NOOI_FLOAT_COUNT = 64 * 64 * 8 * 2
_CDB_EMBED_DIM = 24

_CDB_GRAPH_RUNTIME_RECORD = "eta-mu.graph-runtime.cdb.v1"
_CDB_GRAPH_RUNTIME_SCHEMA = "graph.runtime.cdb.v1"

_EDGE_HEALTH_DECAY = 0.985
_EDGE_HEALTH_REPAIR_GAIN = 0.085
_EDGE_HEALTH_QUEUE_PENALTY = 0.045
_EDGE_HEALTH_FLOOR = 0.05
_EDGE_HEALTH_REGISTRY: dict[str, float] = {}
try:
    _EDGE_HEALTH_REGISTRY_MAX = max(
        1024,
        int(float(os.getenv("CDB_EDGE_HEALTH_REGISTRY_MAX", "32768") or "32768")),
    )
except Exception:
    _EDGE_HEALTH_REGISTRY_MAX = 32768
_EDGE_HEALTH_REGISTRY_TRIM_TARGET = max(
    768,
    min(
        _EDGE_HEALTH_REGISTRY_MAX,
        int(float(_EDGE_HEALTH_REGISTRY_MAX) * 0.88),
    ),
)

_VALVE_ALPHA_PRESSURE = 0.44
_VALVE_ALPHA_GRAVITY = 1.0
_VALVE_ALPHA_AFFINITY = 0.36
_VALVE_ALPHA_SATURATION = 0.52
_VALVE_ALPHA_HEALTH = 0.34

try:
    _CDB_GRAPH_VARIABILITY_NODE_LIMIT = max(
        64,
        int(float(os.getenv("CDB_GRAPH_VARIABILITY_NODE_LIMIT", "4096") or "4096")),
    )
except Exception:
    _CDB_GRAPH_VARIABILITY_NODE_LIMIT = 4096
try:
    _CDB_GRAPH_VARIABILITY_DISTANCE_REF = max(
        0.001,
        float(os.getenv("CDB_GRAPH_VARIABILITY_DISTANCE_REF", "0.02") or "0.02"),
    )
except Exception:
    _CDB_GRAPH_VARIABILITY_DISTANCE_REF = 0.02
try:
    _CDB_GRAPH_VARIABILITY_SCORE_EMA_ALPHA = max(
        0.01,
        min(
            1.0,
            float(os.getenv("CDB_GRAPH_VARIABILITY_SCORE_EMA_ALPHA", "0.2") or "0.2"),
        ),
    )
except Exception:
    _CDB_GRAPH_VARIABILITY_SCORE_EMA_ALPHA = 0.2
try:
    _CDB_GRAPH_VARIABILITY_NOISE_GAIN = max(
        0.0,
        min(
            2.0,
            float(os.getenv("CDB_GRAPH_VARIABILITY_NOISE_GAIN", "1.0") or "1.0"),
        ),
    )
except Exception:
    _CDB_GRAPH_VARIABILITY_NOISE_GAIN = 1.0
try:
    _CDB_GRAPH_VARIABILITY_ROUTE_DAMP = max(
        0.0,
        min(
            0.8,
            float(os.getenv("CDB_GRAPH_VARIABILITY_ROUTE_DAMP", "0.35") or "0.35"),
        ),
    )
except Exception:
    _CDB_GRAPH_VARIABILITY_ROUTE_DAMP = 0.35

_RESOURCE_TYPES: tuple[str, ...] = (
    "cpu",
    "gpu",
    "npu",
    "ram",
    "disk",
    "network",
)
_RESOURCE_ALIASES: dict[str, str] = {
    "cpu": "cpu",
    "gpu": "gpu",
    "gpu0": "gpu",
    "gpu1": "gpu",
    "gpu2": "gpu",
    "npu": "npu",
    "npu0": "npu",
    "ram": "ram",
    "memory": "ram",
    "mem": "ram",
    "disk": "disk",
    "storage": "disk",
    "network": "network",
    "net": "network",
    "netup": "network",
    "netdown": "network",
}
_RESOURCE_WALLET_FLOOR: dict[str, float] = {
    "cpu": 6.0,
    "gpu": 5.0,
    "npu": 4.0,
    "ram": 8.0,
    "disk": 7.0,
    "network": 7.0,
}
_RESOURCE_NEED_MIN = 0.03
_RESOURCE_NEED_EMA_ALPHA = 0.24
_RESOURCE_NEED_THRESHOLDS: dict[str, float] = {
    "cpu": 0.42,
    "gpu": 0.38,
    "npu": 0.36,
    "ram": 0.45,
    "disk": 0.4,
    "network": 0.41,
}
_RESOURCE_NEED_STEEPNESS: dict[str, float] = {
    "cpu": 6.4,
    "gpu": 7.1,
    "npu": 6.8,
    "ram": 5.4,
    "disk": 5.8,
    "network": 6.0,
}

_GROWTH_MODE_MAP = {
    0: "normal",
    1: "watch",
    2: "critical",
}

_NATIVE_DIR = Path(__file__).resolve().parent / "native"
_NATIVE_SOURCE = _NATIVE_DIR / "c_double_buffer_sim.c"
_NATIVE_LIB = _NATIVE_DIR / "libc_double_buffer_sim.so"
_EMBED_NATIVE_SOURCE = _NATIVE_DIR / "c_embed_runtime.cpp"
_EMBED_NATIVE_LIB = _NATIVE_DIR / "libc_embed_runtime.so"
_EMBED_NATIVE_BUILDINFO = _NATIVE_DIR / "libc_embed_runtime.buildinfo"

_LIB_LOCK = threading.Lock()
_ENGINE_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_ENGINE: "_CDBEngine | None" = None

_EMBED_LIB_LOCK = threading.Lock()
_EMBED_RUNTIME_LOCK = threading.Lock()
_EMBED_LIB: ctypes.CDLL | None = None
_EMBED_RUNTIME: Any = None
_EMBED_RUNTIME_ERROR: str = ""
_EMBED_RUNTIME_SOURCE: str = "pending"
_EMBED_RUNTIME_CPU_FALLBACK: bool = False
_EMBED_RUNTIME_CPU_FALLBACK_DETAIL: str = ""
_LEVEL_ZERO_PRELOAD_LOCK = threading.Lock()
_LEVEL_ZERO_PRELOADED = False
_LEVEL_ZERO_HANDLES: list[Any] = []
_EMBED_MODEL_DOWNLOAD_LOCK = threading.Lock()
_EMBED_MODEL_DOWNLOAD_ATTEMPTED = False
_EMBED_VECTOR_CACHE_LOCK = threading.Lock()
_EMBED_VECTOR_CACHE: dict[str, tuple[float, ...]] = {}
_COLLISION_CTYPE_SCRATCH = threading.local()
_COLLISION_SCRATCH_SHRINK_FACTOR = 4
_COLLISION_SCRATCH_SHRINK_RUNS = 24
_COLLISION_SCRATCH_SHRINK_MIN_CAPACITY = 1024
_COLLISION_NATIVE_RELEASE_COOLDOWN_SECONDS = 45.0

_CDB_MAX_PRESENCE_SLOTS = 64
_DEFAULT_PRESENCE_LAYOUT: tuple[tuple[str, float, float, float], ...] = (
    ("receipt_river", 0.22, 0.38, 212.0),
    ("witness_thread", 0.63, 0.33, 262.0),
    ("fork_tax_canticle", 0.44, 0.62, 34.0),
    ("mage_of_receipts", 0.33, 0.71, 286.0),
    ("keeper_of_receipts", 0.57, 0.72, 124.0),
    ("anchor_registry", 0.49, 0.5, 184.0),
    ("gates_of_truth", 0.76, 0.54, 52.0),
    ("file_sentinel", 0.68, 0.43, 168.0),
    ("change_fog", 0.71, 0.6, 204.0),
    ("path_ward", 0.31, 0.44, 142.0),
    ("manifest_lith", 0.56, 0.27, 78.0),
    ("resolution_weaver", 0.82, 0.36, 314.0),
    ("core_pulse", 0.5, 0.5, 0.0),
)


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(max(0.0, _safe_float(value, 0.0)) for value in values)
    if not ordered:
        return 0.0
    q_clamped = _clamp01(_safe_float(q, 0.0))
    if len(ordered) == 1:
        return ordered[0]
    index = int(round((len(ordered) - 1) * q_clamped))
    index = max(0, min(len(ordered) - 1, index))
    return ordered[index]


def _graph_node_position_map(
    *,
    node_ids: list[Any],
    node_positions: list[Any],
) -> dict[str, tuple[float, float]]:
    if not isinstance(node_ids, list) or not isinstance(node_positions, list):
        return {}

    node_count = min(
        len(node_ids),
        len(node_positions),
        max(1, int(_CDB_GRAPH_VARIABILITY_NODE_LIMIT)),
    )
    if node_count <= 0:
        return {}

    mapped: dict[str, tuple[float, float]] = {}
    for index in range(node_count):
        node_id = str(node_ids[index] or "").strip()
        if not node_id:
            continue
        row = node_positions[index]
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        mapped[node_id] = (
            _clamp01(_safe_float(row[0], 0.5)),
            _clamp01(_safe_float(row[1], 0.5)),
        )
    return mapped


def _graph_node_variability_update(
    engine: Any,
    *,
    node_ids: list[Any],
    node_positions: list[Any],
) -> dict[str, Any]:
    current_positions = _graph_node_position_map(
        node_ids=node_ids,
        node_positions=node_positions,
    )
    previous_positions_raw = getattr(engine, "_graph_node_positions_prev", {})
    previous_positions = (
        dict(previous_positions_raw) if isinstance(previous_positions_raw, dict) else {}
    )

    displacements: list[float] = []
    moved_count = 0
    moved_threshold = max(0.0005, _CDB_GRAPH_VARIABILITY_DISTANCE_REF * 0.35)
    for node_id, (cx, cy) in current_positions.items():
        prior = previous_positions.get(node_id)
        if not isinstance(prior, tuple) or len(prior) < 2:
            continue
        px = _clamp01(_safe_float(prior[0], cx))
        py = _clamp01(_safe_float(prior[1], cy))
        displacement = math.hypot(cx - px, cy - py)
        displacements.append(displacement)
        if displacement >= moved_threshold:
            moved_count += 1

    shared_nodes = len(displacements)
    mean_displacement = (
        (sum(displacements) / float(shared_nodes)) if shared_nodes > 0 else 0.0
    )
    p90_displacement = _quantile(displacements, 0.9)
    active_share = (
        (float(moved_count) / float(shared_nodes)) if shared_nodes > 0 else 0.0
    )

    mean_term = _clamp01(
        mean_displacement / max(1e-6, _CDB_GRAPH_VARIABILITY_DISTANCE_REF)
    )
    p90_term = _clamp01(
        p90_displacement / max(1e-6, _CDB_GRAPH_VARIABILITY_DISTANCE_REF * 1.8)
    )
    active_term = _clamp01(active_share / 0.45)
    raw_score = _clamp01((mean_term * 0.55) + (p90_term * 0.25) + (active_term * 0.2))

    previous_state_raw = getattr(engine, "_graph_node_variability_state", {})
    previous_state = (
        dict(previous_state_raw) if isinstance(previous_state_raw, dict) else {}
    )
    alpha = _clamp01(_safe_float(_CDB_GRAPH_VARIABILITY_SCORE_EMA_ALPHA, 0.2))
    previous_score = _clamp01(
        _safe_float(previous_state.get("score", raw_score), raw_score)
    )
    score = _clamp01((previous_score * (1.0 - alpha)) + (raw_score * alpha))
    peak_score = max(
        score,
        _clamp01(_safe_float(previous_state.get("peak_score", score), score) * 0.92),
    )

    state = {
        "score": score,
        "raw_score": raw_score,
        "peak_score": peak_score,
        "mean_displacement": mean_displacement,
        "p90_displacement": p90_displacement,
        "active_share": active_share,
        "shared_nodes": shared_nodes,
        "sampled_nodes": len(current_positions),
    }
    setattr(engine, "_graph_node_variability_state", dict(state))
    setattr(engine, "_graph_node_positions_prev", dict(current_positions))
    return state


def _is_cpu_fallback_signal(text: str) -> bool:
    probe = str(text or "").strip().lower()
    if not probe:
        return False

    if "cpu" in probe and (
        "fallback" in probe or "fall back" in probe or "falling back" in probe
    ):
        return True
    if "ov cpu" in probe and ("fallback" in probe or "unsupported" in probe):
        return True
    if "ze_result_error_unsupported_feature" in probe and "cpu" in probe:
        return True
    if "unsupported" in probe and "npu" in probe and "cpu" in probe:
        return True
    if "cpuexecutionprovider" in probe:
        return True
    if "cpu execution provider" in probe:
        return True
    if "selected_device=cpu" in probe:
        return True
    return False


def _normalize_embed_device(raw: str | None) -> str:
    value = str(raw or "").strip().upper()
    if not value:
        return "AUTO"
    if value in {"NPU", "INTEL_NPU"}:
        return "NPU"
    if value in {"GPU", "CUDA", "NVIDIA", "NVIDIA_GPU"}:
        return "GPU"
    if value == "AUTO":
        return "AUTO"
    return "AUTO"


def _embed_device_candidates(raw: str | None) -> list[str]:
    normalized = _normalize_embed_device(raw)
    if normalized == "NPU":
        return ["NPU"]
    if normalized == "GPU":
        return ["GPU"]
    return ["NPU", "GPU"]


def _is_hardware_embed_device(value: str) -> bool:
    probe = str(value or "").strip().upper()
    if not probe:
        return False
    return probe in {"NPU", "GPU", "CUDA"}


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        expo = math.exp(-value)
        return 1.0 / (1.0 + expo)
    expo = math.exp(value)
    return expo / (1.0 + expo)


def _simplex_grad(hash_val: int, x: float, y: float) -> float:
    grads: tuple[tuple[float, float], ...] = (
        (1.0, 1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (0.0, -1.0),
    )
    gx, gy = grads[int(hash_val) % len(grads)]
    return (gx * x) + (gy * y)


def _simplex_hash(i: int, j: int, seed: int) -> int:
    mix = (
        ((int(i) + 1) * 0x9E3779B1)
        ^ ((int(j) + 1) * 0x85EBCA77)
        ^ ((int(seed) + 1) * 0xC2B2AE3D)
    ) & 0xFFFFFFFF
    mix ^= mix >> 16
    mix = (mix * 0x7FEB352D) & 0xFFFFFFFF
    mix ^= mix >> 15
    mix = (mix * 0x846CA68B) & 0xFFFFFFFF
    mix ^= mix >> 16
    return int(mix % 12)


def _simplex_noise_2d(x: float, y: float, *, seed: int = 0) -> float:
    f2 = 0.5 * (math.sqrt(3.0) - 1.0)
    g2 = (3.0 - math.sqrt(3.0)) / 6.0

    s = (x + y) * f2
    i = int(math.floor(x + s))
    j = int(math.floor(y + s))
    t = (i + j) * g2
    x0 = x - (i - t)
    y0 = y - (j - t)

    if x0 > y0:
        i1, j1 = 1, 0
    else:
        i1, j1 = 0, 1

    x1 = x0 - i1 + g2
    y1 = y0 - j1 + g2
    x2 = x0 - 1.0 + (2.0 * g2)
    y2 = y0 - 1.0 + (2.0 * g2)

    n0 = 0.0
    n1 = 0.0
    n2 = 0.0

    t0 = 0.5 - (x0 * x0) - (y0 * y0)
    if t0 >= 0.0:
        t0 *= t0
        n0 = t0 * t0 * _simplex_grad(_simplex_hash(i, j, seed), x0, y0)

    t1 = 0.5 - (x1 * x1) - (y1 * y1)
    if t1 >= 0.0:
        t1 *= t1
        n1 = (
            t1
            * t1
            * _simplex_grad(
                _simplex_hash(i + i1, j + j1, seed),
                x1,
                y1,
            )
        )

    t2 = 0.5 - (x2 * x2) - (y2 * y2)
    if t2 >= 0.0:
        t2 *= t2
        n2 = t2 * t2 * _simplex_grad(_simplex_hash(i + 1, j + 1, seed), x2, y2)

    return 70.0 * (n0 + n1 + n2)


def _simplex_motion_delta(
    *,
    x: float,
    y: float,
    now: float,
    seed: int,
    amplitude: float,
    space_scale: float = 4.4,
    phase_scale: float = 0.28,
) -> tuple[float, float]:
    amp = max(0.0, _safe_float(amplitude, 0.0))
    if amp <= 1e-10:
        return (0.0, 0.0)

    scale = max(0.25, _safe_float(space_scale, 4.4))
    phase = _safe_float(now, 0.0) * max(0.01, _safe_float(phase_scale, 0.28))
    seed_value = int(seed)
    noise_x = _simplex_noise_2d(
        (x * scale) + phase,
        (y * scale) + (phase * 0.73),
        seed=seed_value,
    )
    noise_y = _simplex_noise_2d(
        (x * scale) + 17.0 + (phase * 0.61),
        (y * scale) + 11.0 + phase,
        seed=seed_value + 101,
    )
    return (noise_x * amp, noise_y * amp)


def _safe_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _edge_health_default(affinity: float) -> float:
    return _clamp01(0.42 + (_clamp01(_safe_float(affinity, 0.5)) * 0.5))


def _edge_health_key(node_ids: list[str], source_index: int, target_index: int) -> str:
    if (
        source_index < 0
        or target_index < 0
        or source_index >= len(node_ids)
        or target_index >= len(node_ids)
    ):
        return f"edge:{source_index}->{target_index}"
    return f"{node_ids[source_index]}->{node_ids[target_index]}"


def _prune_edge_health_registry(*, active_keys: set[str]) -> None:
    if len(_EDGE_HEALTH_REGISTRY) <= _EDGE_HEALTH_REGISTRY_MAX:
        return
    trim_target = max(
        1,
        min(_EDGE_HEALTH_REGISTRY_MAX, _EDGE_HEALTH_REGISTRY_TRIM_TARGET),
    )
    for key in list(_EDGE_HEALTH_REGISTRY.keys()):
        if len(_EDGE_HEALTH_REGISTRY) <= trim_target:
            break
        if key in active_keys:
            continue
        _EDGE_HEALTH_REGISTRY.pop(key, None)
    if len(_EDGE_HEALTH_REGISTRY) <= trim_target:
        return
    for key in list(_EDGE_HEALTH_REGISTRY.keys()):
        if len(_EDGE_HEALTH_REGISTRY) <= trim_target:
            break
        _EDGE_HEALTH_REGISTRY.pop(key, None)


def _derive_edge_cost_components(
    *,
    edge_sources: list[int],
    edge_targets: list[int],
    edge_affinity: list[float],
    node_count: int,
    queue_ratio: float,
    cpu_ratio: float,
    cost_w_l: float,
    cost_w_c: float,
    cost_w_s: float,
) -> dict[str, Any]:
    edge_count = len(edge_sources)
    global_saturation = _clamp01(
        (_clamp01(_safe_float(queue_ratio, 0.0)) * 0.62)
        + (_clamp01(_safe_float(cpu_ratio, 0.0)) * 0.38)
    )
    if edge_count <= 0 or node_count <= 0:
        return {
            "global_saturation": global_saturation,
            "edge_saturation": [],
            "edge_latency_component": [],
            "edge_congestion_component": [],
            "edge_semantic_component": [],
            "edge_base_cost": [],
        }

    out_degree = [0 for _ in range(node_count)]
    in_degree = [0 for _ in range(node_count)]
    for edge_index in range(edge_count):
        src = int(edge_sources[edge_index]) if edge_index < len(edge_sources) else -1
        dst = int(edge_targets[edge_index]) if edge_index < len(edge_targets) else -1
        if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
            continue
        out_degree[src] += 1
        in_degree[dst] += 1

    mean_degree = math.sqrt((edge_count / max(1.0, float(node_count))) + 1.0)
    degree_norm = max(1.0, mean_degree * 2.0)

    latency_component_const = max(0.0001, _safe_float(cost_w_l, 1.0))
    congestion_weight = max(0.0, _safe_float(cost_w_c, 2.0))
    semantic_weight = max(0.0, _safe_float(cost_w_s, 1.0))

    edge_saturation: list[float] = []
    edge_latency_component: list[float] = []
    edge_congestion_component: list[float] = []
    edge_semantic_component: list[float] = []
    edge_base_cost: list[float] = []

    for edge_index in range(edge_count):
        src = int(edge_sources[edge_index]) if edge_index < len(edge_sources) else -1
        dst = int(edge_targets[edge_index]) if edge_index < len(edge_targets) else -1
        affinity = _clamp01(
            _safe_float(
                edge_affinity[edge_index] if edge_index < len(edge_affinity) else 0.5,
                0.5,
            )
        )
        if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
            sat = global_saturation
        else:
            degree_pressure = _clamp01(
                (float(out_degree[src]) + float(in_degree[dst])) / degree_norm
            )
            sat = _clamp01((global_saturation * 0.58) + (degree_pressure * 0.42))

        latency_component = latency_component_const
        congestion_component = congestion_weight * sat
        semantic_component = semantic_weight * (1.0 - affinity)
        base_cost = max(
            0.0001,
            latency_component + congestion_component + semantic_component,
        )

        edge_saturation.append(sat)
        edge_latency_component.append(latency_component)
        edge_congestion_component.append(congestion_component)
        edge_semantic_component.append(semantic_component)
        edge_base_cost.append(base_cost)

    return {
        "global_saturation": global_saturation,
        "edge_saturation": edge_saturation,
        "edge_latency_component": edge_latency_component,
        "edge_congestion_component": edge_congestion_component,
        "edge_semantic_component": edge_semantic_component,
        "edge_base_cost": edge_base_cost,
    }


def _route_terms_for_edge(
    *,
    source: int,
    target: int,
    edge_index: int | None,
    gravity: list[float],
    node_price: list[float],
    edge_cost: list[float],
    edge_health: list[float],
    edge_affinity: list[float],
    edge_saturation: list[float],
    edge_latency_component: list[float],
    edge_congestion_component: list[float],
    edge_semantic_component: list[float],
    edge_upkeep_penalty: list[float],
    resource_gravity_maps: dict[str, list[float]] | None = None,
    resource_signature: dict[str, float] | None = None,
    eta: float,
    upsilon: float,
) -> dict[str, Any]:
    eta_value = _safe_float(eta, 1.0)
    upsilon_value = _safe_float(upsilon, 0.72)

    gravity_delta_scalar = _safe_float(
        gravity[target] if target < len(gravity) else 0.0,
        0.0,
    ) - _safe_float(
        gravity[source] if source < len(gravity) else 0.0,
        0.0,
    )
    gravity_delta = gravity_delta_scalar
    route_gravity_mode = "scalar-gravity"
    route_resource_focus = ""
    route_resource_focus_weight = 0.0
    route_resource_focus_delta = 0.0
    route_resource_focus_contribution = 0.0

    maps = resource_gravity_maps if isinstance(resource_gravity_maps, dict) else {}
    signature = resource_signature if isinstance(resource_signature, dict) else {}
    if maps and signature:
        weighted_delta = 0.0
        focus_abs = 0.0
        matched = False
        for raw_resource, raw_weight in signature.items():
            resource = _canonical_resource_type(str(raw_resource)) or str(raw_resource)
            map_values = maps.get(resource)
            if not isinstance(map_values, list):
                continue
            if source >= len(map_values) or target >= len(map_values):
                continue
            weight = max(0.0, _safe_float(raw_weight, 0.0))
            if weight <= 1e-8:
                continue
            delta = _safe_float(map_values[target], 0.0) - _safe_float(
                map_values[source],
                0.0,
            )
            contribution = weight * delta
            weighted_delta += contribution
            matched = True
            if abs(contribution) > focus_abs:
                focus_abs = abs(contribution)
                route_resource_focus = resource
                route_resource_focus_weight = weight
                route_resource_focus_delta = delta
                route_resource_focus_contribution = contribution
        if matched:
            gravity_delta = weighted_delta
            route_gravity_mode = "resource-signature"

    drift_gravity_term = eta_value * gravity_delta

    if edge_index is None:
        latency_component = 0.0
        congestion_component = 0.0
        semantic_component = 0.0
        upkeep_penalty = 0.0
        selected_edge_cost = 0.0
        selected_edge_health = 1.0
        selected_edge_affinity = 0.5
        selected_edge_saturation = 0.0
    else:
        latency_component = max(
            0.0,
            _safe_float(
                edge_latency_component[edge_index]
                if edge_index < len(edge_latency_component)
                else 0.0,
                0.0,
            ),
        )
        congestion_component = max(
            0.0,
            _safe_float(
                edge_congestion_component[edge_index]
                if edge_index < len(edge_congestion_component)
                else 0.0,
                0.0,
            ),
        )
        semantic_component = max(
            0.0,
            _safe_float(
                edge_semantic_component[edge_index]
                if edge_index < len(edge_semantic_component)
                else 0.0,
                0.0,
            ),
        )
        upkeep_penalty = max(
            0.0,
            _safe_float(
                edge_upkeep_penalty[edge_index]
                if edge_index < len(edge_upkeep_penalty)
                else 0.0,
                0.0,
            ),
        )
        selected_edge_cost = max(
            0.0,
            _safe_float(
                edge_cost[edge_index] if edge_index < len(edge_cost) else 0.0, 0.0
            ),
        )
        selected_edge_health = _clamp01(
            _safe_float(
                edge_health[edge_index] if edge_index < len(edge_health) else 1.0, 1.0
            )
        )
        selected_edge_affinity = _clamp01(
            _safe_float(
                edge_affinity[edge_index] if edge_index < len(edge_affinity) else 0.5,
                0.5,
            )
        )
        selected_edge_saturation = _clamp01(
            _safe_float(
                edge_saturation[edge_index]
                if edge_index < len(edge_saturation)
                else 0.0,
                0.0,
            )
        )

    drift_cost_latency_term = -(upsilon_value * latency_component)
    drift_cost_congestion_term = -(upsilon_value * congestion_component)
    drift_cost_semantic_term = -(upsilon_value * semantic_component)
    drift_cost_upkeep_term = -(upsilon_value * upkeep_penalty)
    drift_cost_term = (
        drift_cost_latency_term
        + drift_cost_congestion_term
        + drift_cost_semantic_term
        + drift_cost_upkeep_term
    )

    pressure_delta = _safe_float(
        node_price[source] if source < len(node_price) else 0.0,
        0.0,
    ) - _safe_float(
        node_price[target] if target < len(node_price) else 0.0,
        0.0,
    )
    valve_pressure_term = _VALVE_ALPHA_PRESSURE * pressure_delta
    valve_gravity_term = _VALVE_ALPHA_GRAVITY * gravity_delta
    valve_affinity_term = _VALVE_ALPHA_AFFINITY * selected_edge_affinity
    valve_saturation_term = -(_VALVE_ALPHA_SATURATION * selected_edge_saturation)
    valve_health_term = -(_VALVE_ALPHA_HEALTH * (1.0 - selected_edge_health))
    valve_score_proxy = (
        valve_pressure_term
        + valve_gravity_term
        + valve_affinity_term
        + valve_saturation_term
        + valve_health_term
    )

    return {
        "drift_gravity_term": drift_gravity_term,
        "drift_cost_term": drift_cost_term,
        "drift_gravity_delta": gravity_delta,
        "drift_gravity_delta_scalar": gravity_delta_scalar,
        "drift_cost_latency_term": drift_cost_latency_term,
        "drift_cost_congestion_term": drift_cost_congestion_term,
        "drift_cost_semantic_term": drift_cost_semantic_term,
        "drift_cost_upkeep_term": drift_cost_upkeep_term,
        "route_gravity_mode": route_gravity_mode,
        "route_resource_focus": route_resource_focus,
        "route_resource_focus_weight": route_resource_focus_weight,
        "route_resource_focus_delta": route_resource_focus_delta,
        "route_resource_focus_contribution": route_resource_focus_contribution,
        "selected_edge_cost": selected_edge_cost,
        "selected_edge_health": selected_edge_health,
        "selected_edge_affinity": selected_edge_affinity,
        "selected_edge_saturation": selected_edge_saturation,
        "selected_edge_upkeep_penalty": upkeep_penalty,
        "valve_pressure_term": valve_pressure_term,
        "valve_gravity_term": valve_gravity_term,
        "valve_affinity_term": valve_affinity_term,
        "valve_saturation_term": valve_saturation_term,
        "valve_health_term": valve_health_term,
        "valve_score_proxy": valve_score_proxy,
    }


def _route_score_for_edge(
    *,
    source: int,
    target: int,
    edge_index: int | None,
    gravity: list[float],
    edge_latency_component: list[float],
    edge_congestion_component: list[float],
    edge_semantic_component: list[float],
    edge_upkeep_penalty: list[float],
    resource_gravity_maps: dict[str, list[float]] | None = None,
    resource_signature: dict[str, float] | None = None,
    eta: float,
    upsilon: float,
) -> float:
    eta_value = _safe_float(eta, 1.0)
    upsilon_value = _safe_float(upsilon, 0.72)

    gravity_delta = _safe_float(
        gravity[target] if target < len(gravity) else 0.0,
        0.0,
    ) - _safe_float(
        gravity[source] if source < len(gravity) else 0.0,
        0.0,
    )

    maps = resource_gravity_maps if isinstance(resource_gravity_maps, dict) else {}
    signature = resource_signature if isinstance(resource_signature, dict) else {}
    if maps and signature:
        weighted_delta = 0.0
        matched = False
        for raw_resource, raw_weight in signature.items():
            resource = _canonical_resource_type(str(raw_resource)) or str(raw_resource)
            map_values = maps.get(resource)
            if not isinstance(map_values, list):
                continue
            if source >= len(map_values) or target >= len(map_values):
                continue
            weight = max(0.0, _safe_float(raw_weight, 0.0))
            if weight <= 1e-8:
                continue
            delta = _safe_float(map_values[target], 0.0) - _safe_float(
                map_values[source],
                0.0,
            )
            weighted_delta += weight * delta
            matched = True
        if matched:
            gravity_delta = weighted_delta

    if edge_index is None:
        latency_component = 0.0
        congestion_component = 0.0
        semantic_component = 0.0
        upkeep_component = 0.0
    else:
        latency_component = max(
            0.0,
            _safe_float(
                edge_latency_component[edge_index]
                if edge_index < len(edge_latency_component)
                else 0.0,
                0.0,
            ),
        )
        congestion_component = max(
            0.0,
            _safe_float(
                edge_congestion_component[edge_index]
                if edge_index < len(edge_congestion_component)
                else 0.0,
                0.0,
            ),
        )
        semantic_component = max(
            0.0,
            _safe_float(
                edge_semantic_component[edge_index]
                if edge_index < len(edge_semantic_component)
                else 0.0,
                0.0,
            ),
        )
        upkeep_component = max(
            0.0,
            _safe_float(
                edge_upkeep_penalty[edge_index]
                if edge_index < len(edge_upkeep_penalty)
                else 0.0,
                0.0,
            ),
        )

    drift_gravity_term = eta_value * gravity_delta
    drift_cost_term = -(
        upsilon_value
        * (
            latency_component
            + congestion_component
            + semantic_component
            + upkeep_component
        )
    )
    return drift_gravity_term + drift_cost_term


def _canonical_resource_type(resource_type: str) -> str:
    return _RESOURCE_ALIASES.get(str(resource_type or "").strip().lower(), "")


def _core_resource_type_from_presence_id(presence_id: str) -> str:
    token = str(presence_id or "").strip().lower()
    if token.startswith("presence.core."):
        return _canonical_resource_type(token.rsplit(".", 1)[-1])
    return ""


def _presence_priority_rank(presence_id: str) -> int:
    token = str(presence_id or "").strip().lower()
    if token.startswith("presence.core."):
        return 0
    if token.startswith("health_sentinel_"):
        return 1
    if token == "presence.user.operator":
        return 2
    return 3


def _resource_wallet_by_type(wallet: dict[str, Any] | None) -> dict[str, float]:
    totals = {resource: 0.0 for resource in _RESOURCE_TYPES}
    if not isinstance(wallet, dict):
        return totals
    for key, value in wallet.items():
        resource = _canonical_resource_type(str(key))
        if not resource:
            continue
        totals[resource] += max(0.0, _safe_float(value, 0.0))
    return totals


def _normalize_resource_signature(values: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {
        resource: max(0.0, _safe_float(values.get(resource, 0.0), 0.0))
        for resource in _RESOURCE_TYPES
    }
    total = sum(normalized.values())
    if total <= 1e-8:
        uniform = 1.0 / float(len(_RESOURCE_TYPES))
        return {resource: uniform for resource in _RESOURCE_TYPES}
    return {resource: (weight / total) for resource, weight in normalized.items()}


def _default_resource_signature(queue_ratio: float) -> dict[str, float]:
    queue_clamped = _clamp01(_safe_float(queue_ratio, 0.0))
    base = {
        "cpu": 0.21 + (queue_clamped * 0.06),
        "gpu": 0.14,
        "npu": 0.1,
        "ram": 0.18,
        "disk": 0.14 + (queue_clamped * 0.08),
        "network": 0.17 + (queue_clamped * 0.12),
    }
    return _normalize_resource_signature(base)


def _presence_resource_need_model(
    *,
    presence_id: str,
    impact: dict[str, Any],
    queue_ratio: float,
    base_need: float | None = None,
) -> dict[str, Any]:
    impact_row = impact if isinstance(impact, dict) else {}
    affected_by = impact_row.get("affected_by", {})
    if not isinstance(affected_by, dict):
        affected_by = {}
    affects = impact_row.get("affects", {})
    if not isinstance(affects, dict):
        affects = {}

    resource_signal = _clamp01(_safe_float(affected_by.get("resource", 0.0), 0.0))
    file_signal = _clamp01(_safe_float(affected_by.get("files", 0.0), 0.0))
    click_signal = _clamp01(_safe_float(affected_by.get("clicks", 0.0), 0.0))
    world_signal = _clamp01(_safe_float(affects.get("world", 0.0), 0.0))
    queue_clamped = _clamp01(_safe_float(queue_ratio, 0.0))
    if base_need is None:
        base_need = _clamp01(
            (resource_signal * 0.55)
            + (file_signal * 0.2)
            + (click_signal * 0.13)
            + (queue_clamped * 0.12)
        )
    base_need_value = _clamp01(_safe_float(base_need, 0.0))

    priority_signal = _clamp01(
        (resource_signal * 0.42)
        + (world_signal * 0.24)
        + (file_signal * 0.18)
        + (click_signal * 0.08)
        + (queue_clamped * 0.08)
    )
    priority = 0.72 + (0.78 * priority_signal)

    wallet_by_type = _resource_wallet_by_type(impact_row.get("resource_wallet", {}))
    core_resource = _core_resource_type_from_presence_id(presence_id)
    token = str(presence_id or "").strip().lower()
    util_ema_prev_raw = impact_row.get("_resource_util_ema", {})
    util_ema_prev = util_ema_prev_raw if isinstance(util_ema_prev_raw, dict) else {}

    needs: dict[str, float] = {}
    util_ema: dict[str, float] = {}
    raw_utilization: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    steepness: dict[str, float] = {}

    for resource in _RESOURCE_TYPES:
        floor = max(0.1, _safe_float(_RESOURCE_WALLET_FLOOR.get(resource, 6.0), 6.0))
        balance = max(0.0, _safe_float(wallet_by_type.get(resource, 0.0), 0.0))
        balance_ratio = _clamp01(balance / floor)
        deficit = _clamp01(1.0 - balance_ratio)
        if resource == "network":
            queue_push = queue_clamped * 0.22
        elif resource == "disk":
            queue_push = queue_clamped * 0.14
        elif resource == "cpu":
            queue_push = queue_clamped * 0.1
        else:
            queue_push = queue_clamped * 0.05

        id_hint = 0.08 if resource in token else 0.0
        util_raw = _clamp01(
            (deficit * 0.58) + (base_need_value * 0.24) + queue_push + id_hint
        )
        util_prev = _clamp01(
            _safe_float(util_ema_prev.get(resource, util_raw), util_raw)
        )
        util_next = _clamp01(
            (util_prev * (1.0 - _RESOURCE_NEED_EMA_ALPHA))
            + (util_raw * _RESOURCE_NEED_EMA_ALPHA)
        )

        theta = _clamp01(_safe_float(_RESOURCE_NEED_THRESHOLDS.get(resource, 0.4), 0.4))
        slope = max(0.1, _safe_float(_RESOURCE_NEED_STEEPNESS.get(resource, 6.0), 6.0))
        logistic = _sigmoid(slope * (util_next - theta))

        needs[resource] = _clamp01(priority * logistic)
        util_ema[resource] = util_next
        raw_utilization[resource] = util_raw
        thresholds[resource] = theta
        steepness[resource] = slope

    if core_resource:
        for resource in _RESOURCE_TYPES:
            if resource == core_resource:
                needs[resource] = max(needs[resource], 0.38 + (resource_signal * 0.28))
            else:
                needs[resource] = _clamp01(needs[resource] * 0.24)

    if sum(needs.values()) <= 1e-8:
        signature = _default_resource_signature(queue_clamped)
        needs = {
            resource: _safe_float(signature.get(resource, 0.0), 0.0)
            for resource in _RESOURCE_TYPES
        }

    impact_row["_resource_util_ema"] = {
        resource: _clamp01(_safe_float(value, 0.0))
        for resource, value in util_ema.items()
    }

    return {
        "needs": {
            resource: _clamp01(_safe_float(needs.get(resource, 0.0), 0.0))
            for resource in _RESOURCE_TYPES
        },
        "util_ema": {
            resource: _clamp01(_safe_float(util_ema.get(resource, 0.0), 0.0))
            for resource in _RESOURCE_TYPES
        },
        "util_raw": {
            resource: _clamp01(_safe_float(raw_utilization.get(resource, 0.0), 0.0))
            for resource in _RESOURCE_TYPES
        },
        "priority": max(0.0, _safe_float(priority, 1.0)),
        "alpha": _RESOURCE_NEED_EMA_ALPHA,
        "thresholds": thresholds,
        "steepness": steepness,
    }


def _presence_resource_need_vector(
    *,
    presence_id: str,
    impact: dict[str, Any],
    queue_ratio: float,
    base_need: float | None = None,
) -> dict[str, float]:
    model = _presence_resource_need_model(
        presence_id=presence_id,
        impact=impact,
        queue_ratio=queue_ratio,
        base_need=base_need,
    )
    needs = model.get("needs", {}) if isinstance(model, dict) else {}
    if isinstance(needs, dict) and needs:
        return {
            resource: _clamp01(_safe_float(needs.get(resource, 0.0), 0.0))
            for resource in _RESOURCE_TYPES
        }
    signature = _default_resource_signature(queue_ratio)
    return {
        resource: _safe_float(signature.get(resource, 0.0), 0.0)
        for resource in _RESOURCE_TYPES
    }


def _bounded_dijkstra_distances(
    *,
    node_count: int,
    adjacency: list[list[tuple[int, float]]],
    source: int,
    radius_cost: float,
) -> list[float]:
    distances = [math.inf for _ in range(max(0, node_count))]
    if node_count <= 0 or source < 0 or source >= node_count:
        return distances
    radius = max(0.1, _safe_float(radius_cost, 6.0))

    distances[source] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(source))]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > radius:
            continue
        if dist > distances[node] + 1e-12:
            continue
        for neighbor, step_cost in adjacency[node]:
            step = max(0.0001, _safe_float(step_cost, 1.0))
            candidate = dist + step
            if candidate >= distances[neighbor] or candidate > radius:
                continue
            distances[neighbor] = candidate
            heapq.heappush(heap, (candidate, neighbor))
    return distances


def _compute_resource_gravity_maps(
    *,
    node_count: int,
    edge_sources: list[int],
    edge_targets: list[int],
    edge_cost: list[float],
    source_nodes: list[int],
    source_mass: list[float],
    source_need_by_resource: list[dict[str, float]],
    radius_cost: float,
    gravity_const: float,
    epsilon: float,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    maps: dict[str, list[float]] = {
        resource: [0.0 for _ in range(max(0, node_count))]
        for resource in _RESOURCE_TYPES
    }
    if node_count <= 0:
        return maps, {resource: 0.0 for resource in _RESOURCE_TYPES}

    edge_count = min(len(edge_sources), len(edge_targets), len(edge_cost))
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for edge_index in range(edge_count):
        src = int(edge_sources[edge_index])
        dst = int(edge_targets[edge_index])
        if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
            continue
        adjacency[src].append(
            (dst, max(0.0001, _safe_float(edge_cost[edge_index], 1.0)))
        )

    radius = max(0.1, _safe_float(radius_cost, 6.0))
    grav_const = max(0.1, _safe_float(gravity_const, 1.0))
    grav_eps = max(1e-6, _safe_float(epsilon, 0.001))

    distance_cache: dict[int, list[float]] = {}
    for source_index, source in enumerate(source_nodes):
        source_node = int(source)
        if source_node < 0 or source_node >= node_count:
            continue
        mass = max(
            0.0,
            _safe_float(
                source_mass[source_index] if source_index < len(source_mass) else 0.0,
                0.0,
            ),
        )
        if mass <= 1e-8:
            continue
        need_vector = (
            source_need_by_resource[source_index]
            if source_index < len(source_need_by_resource)
            and isinstance(source_need_by_resource[source_index], dict)
            else {}
        )
        active_needs = [
            (resource, _clamp01(_safe_float(need_vector.get(resource, 0.0), 0.0)))
            for resource in _RESOURCE_TYPES
            if _safe_float(need_vector.get(resource, 0.0), 0.0) > _RESOURCE_NEED_MIN
        ]
        if not active_needs:
            continue

        distances = distance_cache.get(source_node)
        if distances is None:
            distances = _bounded_dijkstra_distances(
                node_count=node_count,
                adjacency=adjacency,
                source=source_node,
                radius_cost=radius,
            )
            distance_cache[source_node] = distances

        for node in range(node_count):
            distance = _safe_float(distances[node], math.inf)
            if not math.isfinite(distance) or distance < 0.0 or distance > radius:
                continue
            potential = (grav_const * mass) / ((distance * distance) + grav_eps)
            if not math.isfinite(potential) or potential <= 0.0:
                continue
            for resource, need_weight in active_needs:
                maps[resource][node] += need_weight * potential

    peaks = {
        resource: max(
            (max(0.0, _safe_float(value, 0.0)) for value in values),
            default=0.0,
        )
        for resource, values in maps.items()
    }
    return maps, peaks


def _build_presence_resource_signatures(
    *,
    presence_ids: list[str],
    presence_impacts: list[dict[str, Any]] | None,
    queue_ratio: float,
) -> dict[str, dict[str, float]]:
    impact_by_id = {
        str(row.get("id", "")).strip(): row
        for row in (presence_impacts if isinstance(presence_impacts, list) else [])
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }
    signatures: dict[str, dict[str, float]] = {}
    default_signature = _default_resource_signature(queue_ratio)
    for presence_id in presence_ids:
        impact = impact_by_id.get(presence_id, {})
        needs = _presence_resource_need_vector(
            presence_id=presence_id,
            impact=impact,
            queue_ratio=queue_ratio,
            base_need=None,
        )
        signatures[presence_id] = _normalize_resource_signature(needs)
    if not signatures and presence_ids:
        signatures[presence_ids[0]] = dict(default_signature)
    return signatures


def _presence_layout(
    *,
    presence_impacts: list[dict[str, Any]] | None,
    entity_manifest: list[dict[str, Any]] | None,
) -> tuple[list[str], dict[str, tuple[float, float, float | None]]]:
    layout_map: dict[str, tuple[float, float, float | None]] = {}
    manifest_order: list[str] = []

    for row in entity_manifest if isinstance(entity_manifest, list) else []:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("id", "") or "").strip()
        if not presence_id:
            continue
        x = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        hue = _safe_optional_float(row.get("hue"))
        if presence_id not in manifest_order:
            manifest_order.append(presence_id)
        layout_map[presence_id] = (x, y, hue)

    if not layout_map:
        for presence_id, x, y, hue in _DEFAULT_PRESENCE_LAYOUT:
            if presence_id not in manifest_order:
                manifest_order.append(presence_id)
            layout_map[presence_id] = (x, y, hue)

    impact_ids: list[str] = []
    for row in presence_impacts if isinstance(presence_impacts, list) else []:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("id", "") or "").strip()
        if not presence_id or presence_id in impact_ids:
            continue
        impact_ids.append(presence_id)
        current_x, current_y, current_hue = layout_map.get(
            presence_id, (0.5, 0.5, None)
        )
        impact_x = _clamp01(_safe_float(row.get("x", current_x), current_x))
        impact_y = _clamp01(_safe_float(row.get("y", current_y), current_y))
        impact_hue = _safe_optional_float(row.get("hue"))
        layout_map[presence_id] = (
            impact_x,
            impact_y,
            impact_hue if impact_hue is not None else current_hue,
        )

    selected_ids: list[str] = []
    for presence_id, _, _, _ in _DEFAULT_PRESENCE_LAYOUT:
        if (
            presence_id in layout_map
            and (not impact_ids or presence_id in impact_ids)
            and presence_id not in selected_ids
        ):
            selected_ids.append(presence_id)

    for presence_id in impact_ids:
        if presence_id in layout_map and presence_id not in selected_ids:
            selected_ids.append(presence_id)

    if not selected_ids:
        for presence_id in manifest_order:
            if presence_id in layout_map and presence_id not in selected_ids:
                selected_ids.append(presence_id)

    if not selected_ids:
        selected_ids = ["anchor_registry"]
        layout_map["anchor_registry"] = (0.5, 0.5, 184.0)

    return selected_ids[:_CDB_MAX_PRESENCE_SLOTS], layout_map


def _mask_nodes_for_anchor(
    *,
    node_ids: list[str],
    node_positions: list[tuple[float, float]],
    anchor_x: float,
    anchor_y: float,
    k: int = 3,
) -> list[dict[str, Any]]:
    if not node_ids or not node_positions:
        return []
    node_count = min(len(node_ids), len(node_positions))
    if node_count <= 0:
        return []
    mask_size = max(1, min(int(k), node_count))
    nearest: list[tuple[float, int]] = []
    for index in range(node_count):
        nx, ny = node_positions[index]
        dx = _safe_float(nx, 0.5) - _safe_float(anchor_x, 0.5)
        dy = _safe_float(ny, 0.5) - _safe_float(anchor_y, 0.5)
        nearest.append(((dx * dx) + (dy * dy), index))
    nearest.sort(key=lambda row: (row[0], row[1]))
    selected = nearest[:mask_size]

    raw_weights = [
        1.0 / max(0.0004, _safe_float(dist_sq, 0.0)) for dist_sq, _ in selected
    ]
    weight_total = sum(raw_weights)
    if weight_total <= 1e-12:
        normalized_weights = [1.0 / float(len(selected)) for _ in selected]
    else:
        normalized_weights = [weight / weight_total for weight in raw_weights]

    return [
        {
            "node_id": str(node_ids[index]),
            "weight": round(_clamp01(_safe_float(weight, 0.0)), 6),
            "distance": round(math.sqrt(max(0.0, _safe_float(dist_sq, 0.0))), 6),
        }
        for weight, (dist_sq, index) in zip(normalized_weights, selected)
    ]


def _seed_from_layout(*, presence_ids: list[str], file_node_count: int) -> int:
    seed = 0x9E3779B9
    seed ^= (max(0, int(file_node_count)) * 2654435761) & 0xFFFFFFFF
    seed ^= (len(presence_ids) * 2246822519) & 0xFFFFFFFF
    for pid in presence_ids[:12]:
        for char in pid[:64]:
            seed = ((seed * 16777619) ^ ord(char)) & 0xFFFFFFFF
    return seed or 1


def _build_shared_library() -> None:
    if not _NATIVE_SOURCE.exists():
        raise RuntimeError(f"missing native source: {_NATIVE_SOURCE}")
    _NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "gcc",
        "-O3",
        "-fPIC",
        "-shared",
        "-std=c11",
        "-pthread",
        str(_NATIVE_SOURCE),
        "-lm",
        "-o",
        str(_NATIVE_LIB),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _load_native_lib() -> ctypes.CDLL:
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        if (not _NATIVE_LIB.exists()) or (
            _NATIVE_LIB.stat().st_mtime < _NATIVE_SOURCE.stat().st_mtime
        ):
            _build_shared_library()

        lib = ctypes.CDLL(str(_NATIVE_LIB))
        lib.cdb_engine_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        lib.cdb_engine_create.restype = ctypes.c_void_p

        lib.cdb_engine_start.argtypes = [ctypes.c_void_p]
        lib.cdb_engine_start.restype = ctypes.c_int

        lib.cdb_engine_stop.argtypes = [ctypes.c_void_p]
        lib.cdb_engine_stop.restype = ctypes.c_int

        lib.cdb_engine_destroy.argtypes = [ctypes.c_void_p]
        lib.cdb_engine_destroy.restype = None

        if hasattr(lib, "cdb_release_thread_scratch"):
            lib.cdb_release_thread_scratch.argtypes = []
            lib.cdb_release_thread_scratch.restype = None

        lib.cdb_engine_snapshot.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.cdb_engine_snapshot.restype = ctypes.c_uint32

        if hasattr(lib, "cdb_engine_update_nooi"):
            lib.cdb_engine_update_nooi.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.cdb_engine_update_nooi.restype = ctypes.c_int

        if hasattr(lib, "cdb_engine_update_embeddings"):
            lib.cdb_engine_update_embeddings.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.cdb_engine_update_embeddings.restype = ctypes.c_int

        if hasattr(lib, "cdb_growth_guard_scores"):
            lib.cdb_growth_guard_scores.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.cdb_growth_guard_scores.restype = ctypes.c_uint32

        if hasattr(lib, "cdb_growth_guard_pressure"):
            lib.cdb_growth_guard_pressure.argtypes = [
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.cdb_growth_guard_pressure.restype = ctypes.c_int

        if hasattr(lib, "cdb_graph_runtime_maps"):
            lib.cdb_graph_runtime_maps.argtypes = [
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint32,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.cdb_graph_runtime_maps.restype = ctypes.c_int

        if hasattr(lib, "cdb_engine_set_flags"):
            lib.cdb_engine_set_flags.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            lib.cdb_engine_set_flags.restype = None

        if hasattr(lib, "cdb_graph_route_step"):
            lib.cdb_graph_route_step.argtypes = [
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.cdb_graph_route_step.restype = ctypes.c_int

        if hasattr(lib, "cdb_resolve_semantic_collisions"):
            lib.cdb_resolve_semantic_collisions.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.cdb_resolve_semantic_collisions.restype = ctypes.c_uint32

        _LIB = lib
        return lib


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = int(float(raw))
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _resolve_force_runtime_config(*, particle_count: int) -> dict[str, Any]:
    count = max(1, int(particle_count))

    cpu_count = int(os.cpu_count() or 1)
    cpu_count = max(1, min(64, cpu_count))
    default_workers = cpu_count
    if default_workers > 4:
        default_workers -= 3
    default_workers = max(1, min(32, default_workers))

    requested_workers = _int_env(
        "CDB_FORCE_WORKERS",
        default_workers,
        minimum=1,
        maximum=32,
    )
    effective_workers = max(1, min(requested_workers, count))
    requested_collision_workers = _int_env(
        "CDB_COLLISION_WORKERS",
        requested_workers,
        minimum=1,
        maximum=32,
    )
    effective_collision_workers = max(1, min(requested_collision_workers, count))

    bh_theta = _safe_float(os.getenv("CDB_BH_THETA", "0.62"), 0.62)
    bh_theta = max(0.2, min(1.4, bh_theta))

    bh_leaf_capacity = _int_env("CDB_BH_LEAF_CAP", 8, minimum=1, maximum=64)
    bh_max_depth = _int_env("CDB_BH_MAX_DEPTH", 12, minimum=4, maximum=24)

    collision_spring = _safe_float(os.getenv("CDB_COLLISION_SPRING", "80.0"), 80.0)
    collision_spring = max(1.0, min(250.0, collision_spring))

    cluster_theta = _safe_float(os.getenv("CDB_CLUSTER_THETA", "0.68"), 0.68)
    cluster_theta = max(0.2, min(1.6, cluster_theta))
    cluster_rest_length = _safe_float(
        os.getenv("CDB_CLUSTER_REST_LENGTH", "0.08"),
        0.08,
    )
    cluster_rest_length = max(0.01, min(0.4, cluster_rest_length))
    cluster_stiffness = _safe_float(
        os.getenv("CDB_CLUSTER_STIFFNESS", "1.0"),
        1.0,
    )
    cluster_stiffness = max(0.1, min(4.0, cluster_stiffness))

    quadtree_capacity_estimate = max(256, (count * 16) + 128)

    return {
        "barnes_hut_theta": round(bh_theta, 6),
        "barnes_hut_leaf_capacity": int(bh_leaf_capacity),
        "barnes_hut_max_depth": int(bh_max_depth),
        "collision_spring": round(collision_spring, 6),
        "cluster_theta": round(cluster_theta, 6),
        "cluster_rest_length": round(cluster_rest_length, 6),
        "cluster_stiffness": round(cluster_stiffness, 6),
        "force_workers_requested": int(requested_workers),
        "force_workers_effective": int(effective_workers),
        "collision_workers_requested": int(requested_collision_workers),
        "collision_workers_effective": int(effective_collision_workers),
        "quadtree_capacity_estimate": int(quadtree_capacity_estimate),
    }


def _maybe_download_embed_model_path() -> Path | None:
    global _EMBED_MODEL_DOWNLOAD_ATTEMPTED

    if not _bool_env("CDB_EMBED_AUTO_DOWNLOAD_MODEL", True):
        return None

    with _EMBED_MODEL_DOWNLOAD_LOCK:
        if _EMBED_MODEL_DOWNLOAD_ATTEMPTED:
            return None
        _EMBED_MODEL_DOWNLOAD_ATTEMPTED = True

    try:
        from huggingface_hub import snapshot_download  # type: ignore

        snapshot = snapshot_download(
            repo_id="nomic-ai/nomic-embed-text-v1.5",
            allow_patterns=["onnx/model.onnx", "tokenizer.json"],
            local_files_only=False,
        )
    except Exception:
        return None

    model_path = Path(str(snapshot)) / "onnx" / "model.onnx"
    if model_path.exists() and model_path.is_file():
        return model_path
    return None


def _resolve_embed_model_path() -> Path | None:
    explicit = str(os.getenv("CDB_EMBED_MODEL_PATH", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists() and path.is_file():
            return path

    part_root = Path(__file__).resolve().parents[2]
    candidates = [
        part_root
        / "world_state"
        / "models"
        / "nomic-embed-text-v1.5"
        / "onnx"
        / "model.onnx",
        part_root / "models" / "nomic-embed-text-v1.5" / "onnx" / "model.onnx",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    hf_home = Path(
        str(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")) or "")
    )
    snapshot_root = (
        hf_home / "hub" / "models--nomic-ai--nomic-embed-text-v1.5" / "snapshots"
    )
    if snapshot_root.exists() and snapshot_root.is_dir():
        snapshots = sorted(
            snapshot_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for snapshot in snapshots:
            model_path = snapshot / "onnx" / "model.onnx"
            if model_path.exists() and model_path.is_file():
                return model_path

    downloaded = _maybe_download_embed_model_path()
    if downloaded is not None:
        return downloaded
    return None


def _resolve_embed_tokenizer_path(model_path: Path) -> Path | None:
    explicit = str(os.getenv("CDB_EMBED_TOKENIZER_PATH", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists() and path.is_file():
            return path

    model_parent = model_path.parent
    candidates = [
        model_parent / "tokenizer.json",
        model_parent.parent / "tokenizer.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    for parent in model_path.parents:
        if parent.name == "models--nomic-ai--nomic-embed-text-v1.5":
            snapshots = parent / "snapshots"
            if snapshots.exists() and snapshots.is_dir():
                snapshot_paths = sorted(
                    snapshots.iterdir(),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for snapshot in snapshot_paths:
                    tok = snapshot / "tokenizer.json"
                    if tok.exists() and tok.is_file():
                        return tok
            break
    return None


def _resolve_ort_include_dir() -> Path | None:
    explicit = str(os.getenv("CDB_ORT_INCLUDE_DIR", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path

    part_root = Path(__file__).resolve().parents[2]
    candidates = sorted(
        [
            path
            for path in part_root.glob("onnxruntime-linux-x64-*/include")
            if path.exists()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _resolve_ort_capi_dir() -> Path | None:
    explicit_lib = str(os.getenv("CDB_ORT_LIB_DIR", "") or "").strip()
    if explicit_lib:
        path = Path(explicit_lib).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path

    explicit = str(os.getenv("CDB_ORT_CAPI_DIR", "") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path

    site_candidates: list[Path] = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str) and user_site:
            site_candidates.append(Path(user_site))
    except Exception:
        pass
    try:
        for row in site.getsitepackages():
            site_candidates.append(Path(str(row)))
    except Exception:
        pass

    discovered: list[Path] = []
    for site_root in site_candidates:
        capi = site_root / "onnxruntime" / "capi"
        if capi.exists() and capi.is_dir():
            discovered.append(capi.resolve())

    part_root = Path(__file__).resolve().parents[2]
    for local_dir in part_root.glob("onnxruntime-linux-x64-*/lib"):
        if local_dir.exists() and local_dir.is_dir():
            discovered.append(local_dir.resolve())

    if not discovered:
        return None

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in discovered:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    def _candidate_score(path: Path) -> tuple[int, float]:
        score = 0
        if (path / "libonnxruntime.so").exists():
            score += 1
        if (path / "libonnxruntime_providers_openvino.so").exists():
            score += 10
        if (path / "libopenvino_intel_npu_plugin.so").exists():
            score += 10
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (score, mtime)

    ordered = sorted(unique, key=_candidate_score, reverse=True)
    return ordered[0] if ordered else None


def _resolve_ort_soname(capi_dir: Path) -> str | None:
    direct = capi_dir / "libonnxruntime.so"
    if direct.exists() and direct.is_file():
        return "libonnxruntime.so"

    candidates = sorted(
        [
            path
            for path in capi_dir.glob("libonnxruntime.so.*")
            if path.exists() and path.is_file()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    return candidates[0].name


def _ensure_ort_soname_links(capi_dir: Path | None) -> None:
    if capi_dir is None or not capi_dir.exists() or not capi_dir.is_dir():
        return

    direct = capi_dir / "libonnxruntime.so"
    so1 = capi_dir / "libonnxruntime.so.1"

    if direct.exists() and so1.exists():
        return

    versioned = sorted(
        [
            path
            for path in capi_dir.glob("libonnxruntime.so.*")
            if path.exists()
            and path.is_file()
            and path.name not in {"libonnxruntime.so.1"}
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not versioned:
        return

    target = versioned[0].name
    try:
        if not so1.exists():
            so1.symlink_to(target)
    except Exception:
        pass
    try:
        if not direct.exists():
            direct.symlink_to(target)
    except Exception:
        pass


def _preload_level_zero_library(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    try:
        handle = ctypes.CDLL(str(path), mode=mode)
    except Exception:
        return
    _LEVEL_ZERO_HANDLES.append(handle)


def _shared_library_loadable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    try:
        ctypes.CDLL(str(path), mode=mode)
    except Exception:
        return False
    return True


def _preload_ort_core_runtime(capi_dir: Path | None) -> None:
    if not _bool_env("CDB_EMBED_PRELOAD_ORT_CORE", False):
        return
    if capi_dir is None or not capi_dir.exists() or not capi_dir.is_dir():
        return

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for stem in ("libonnxruntime.so", "libonnxruntime_providers_shared.so"):
        direct = capi_dir / stem
        candidate: Path | None = (
            direct if direct.exists() and direct.is_file() else None
        )
        if candidate is None:
            matches = sorted(
                [
                    path
                    for path in capi_dir.glob(f"{stem}.*")
                    if path.exists() and path.is_file()
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            candidate = matches[0] if matches else None
        if candidate is None:
            continue
        try:
            handle = ctypes.CDLL(str(candidate), mode=mode)
        except Exception:
            continue
        _LEVEL_ZERO_HANDLES.append(handle)


def _resolve_level_zero_loader_path(directory: Path) -> Path | None:
    candidates = [
        directory / "libze_loader.so",
        directory / "libze_loader.so.1",
    ]
    for candidate in candidates:
        if _shared_library_loadable(candidate):
            return candidate

    try:
        for candidate in sorted(directory.glob("libze_loader.so.*")):
            if _shared_library_loadable(candidate):
                return candidate
    except Exception:
        return None
    return None


def _preload_level_zero_runtime(
    *, loader_path: Path | None, alt_driver: Path | None
) -> None:
    global _LEVEL_ZERO_PRELOADED
    with _LEVEL_ZERO_PRELOAD_LOCK:
        if _LEVEL_ZERO_PRELOADED:
            return
        if loader_path is not None:
            _preload_level_zero_library(loader_path)
        if alt_driver is not None:
            _preload_level_zero_library(alt_driver)
        _LEVEL_ZERO_PRELOADED = True


def _prepare_npu_level_zero_env() -> None:
    if _bool_env("CDB_EMBED_SKIP_LEVEL_ZERO_SETUP", False):
        return

    part_root = Path(__file__).resolve().parents[2]
    libze_candidates = [
        Path(str(os.getenv("CDB_LIBZE_LOADER_DIR", "") or "").strip()),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/lib/x86_64-linux-gnu"),
        part_root
        / ".cache-npu"
        / "libze"
        / "extracted"
        / "usr"
        / "lib"
        / "x86_64-linux-gnu",
    ]

    selected_loader_path: Path | None = None
    for candidate in libze_candidates:
        if not candidate:
            continue
        if not candidate.exists():
            continue
        loader_path = _resolve_level_zero_loader_path(candidate)
        if loader_path is not None:
            selected_loader_path = loader_path
            break

    existing = str(os.getenv("LD_LIBRARY_PATH", "") or "")
    parts = [part for part in existing.split(":") if part]

    capi_dir = _resolve_ort_capi_dir()
    if capi_dir is not None:
        _ensure_ort_soname_links(capi_dir)
        _preload_ort_core_runtime(capi_dir)
        capi_str = str(capi_dir)
        if capi_str not in parts:
            parts.insert(0, capi_str)

    if selected_loader_path is not None:
        selected_dir = selected_loader_path.parent
        selected_str = str(selected_dir)
        if (
            selected_str
            not in {
                "/usr/lib/x86_64-linux-gnu",
                "/lib/x86_64-linux-gnu",
            }
            and selected_str not in parts
        ):
            parts.append(selected_str)

    if parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    selected_alt_driver: Path | None = None
    alt_driver_env = str(os.getenv("ZE_ENABLE_ALT_DRIVERS", "") or "").strip()
    if alt_driver_env:
        candidate = Path(alt_driver_env)
        if _shared_library_loadable(candidate):
            selected_alt_driver = candidate
        else:
            os.environ.pop("ZE_ENABLE_ALT_DRIVERS", None)
    else:
        candidate = Path("/usr/lib/x86_64-linux-gnu/libze_intel_npu.so")
        if _shared_library_loadable(candidate):
            os.environ["ZE_ENABLE_ALT_DRIVERS"] = str(candidate)
            selected_alt_driver = candidate

    _preload_level_zero_runtime(
        loader_path=selected_loader_path,
        alt_driver=selected_alt_driver,
    )


def _embed_source_signature() -> str:
    try:
        stat = _EMBED_NATIVE_SOURCE.stat()
    except Exception:
        return "missing"
    return f"{int(stat.st_mtime_ns)}:{int(stat.st_size)}"


def _embed_buildinfo_matches(
    *, include_dir: Path, capi_dir: Path, soname: str, source_signature: str
) -> bool:
    if not _EMBED_NATIVE_BUILDINFO.exists() or not _EMBED_NATIVE_BUILDINFO.is_file():
        return False
    try:
        rows = _EMBED_NATIVE_BUILDINFO.read_text(encoding="utf-8").splitlines()
    except Exception:
        return False
    expected = [
        str(include_dir.resolve()),
        str(capi_dir.resolve()),
        str(soname),
        str(source_signature),
    ]
    return rows[:4] == expected


def _write_embed_buildinfo(
    *, include_dir: Path, capi_dir: Path, soname: str, source_signature: str
) -> None:
    payload = (
        f"{include_dir.resolve()}\n{capi_dir.resolve()}\n{soname}\n{source_signature}\n"
    )
    _EMBED_NATIVE_BUILDINFO.write_text(payload, encoding="utf-8")


def _build_embed_shared_library() -> None:
    if not _EMBED_NATIVE_SOURCE.exists():
        raise RuntimeError(f"missing embed native source: {_EMBED_NATIVE_SOURCE}")

    include_dir = _resolve_ort_include_dir()
    capi_dir = _resolve_ort_capi_dir()
    if include_dir is None:
        raise RuntimeError("missing ONNX Runtime include directory for c_embed_runtime")
    if capi_dir is None:
        raise RuntimeError("missing ONNX Runtime capi directory for c_embed_runtime")

    soname = _resolve_ort_soname(capi_dir)
    if not soname:
        raise RuntimeError("missing libonnxruntime.so in capi directory")
    source_signature = _embed_source_signature()

    command = [
        "g++",
        "-O3",
        "-fPIC",
        "-shared",
        "-std=c++17",
        str(_EMBED_NATIVE_SOURCE),
        "-I",
        str(include_dir),
        "-L",
        str(capi_dir),
        f"-l:{soname}",
        f"-Wl,-rpath,{capi_dir}",
        "-o",
        str(_EMBED_NATIVE_LIB),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    _write_embed_buildinfo(
        include_dir=include_dir,
        capi_dir=capi_dir,
        soname=soname,
        source_signature=source_signature,
    )


def _load_embed_lib() -> ctypes.CDLL:
    global _EMBED_LIB
    with _EMBED_LIB_LOCK:
        if _EMBED_LIB is not None:
            return _EMBED_LIB

        capi_dir_for_links = _resolve_ort_capi_dir()
        _ensure_ort_soname_links(capi_dir_for_links)

        can_build = shutil.which("g++") is not None

        needs_build = (not _EMBED_NATIVE_LIB.exists()) or (
            _EMBED_NATIVE_LIB.stat().st_mtime < _EMBED_NATIVE_SOURCE.stat().st_mtime
        )
        if not needs_build:
            include_dir = _resolve_ort_include_dir()
            capi_dir = _resolve_ort_capi_dir()
            soname = _resolve_ort_soname(capi_dir) if capi_dir is not None else None
            if include_dir is None or capi_dir is None or not soname:
                needs_build = True
            elif not _embed_buildinfo_matches(
                include_dir=include_dir,
                capi_dir=capi_dir,
                soname=soname,
                source_signature=_embed_source_signature(),
            ):
                needs_build = True

        if needs_build and not can_build and _EMBED_NATIVE_LIB.exists():
            needs_build = False

        if needs_build:
            try:
                _build_embed_shared_library()
            except Exception:
                if not _EMBED_NATIVE_LIB.exists():
                    raise

        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        lib = ctypes.CDLL(str(_EMBED_NATIVE_LIB), mode=mode)
        lib.c_embed_runtime_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.c_embed_runtime_create.restype = ctypes.c_void_p

        lib.c_embed_runtime_last_create_error.argtypes = []
        lib.c_embed_runtime_last_create_error.restype = ctypes.c_char_p

        lib.c_embed_runtime_embed.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.c_embed_runtime_embed.restype = ctypes.c_int32

        lib.c_embed_runtime_last_error.argtypes = [ctypes.c_void_p]
        lib.c_embed_runtime_last_error.restype = ctypes.c_char_p

        lib.c_embed_runtime_selected_device.argtypes = [ctypes.c_void_p]
        lib.c_embed_runtime_selected_device.restype = ctypes.c_char_p

        lib.c_embed_runtime_cpu_fallback_detected.argtypes = [ctypes.c_void_p]
        lib.c_embed_runtime_cpu_fallback_detected.restype = ctypes.c_int32

        lib.c_embed_runtime_cpu_fallback_detail.argtypes = [ctypes.c_void_p]
        lib.c_embed_runtime_cpu_fallback_detail.restype = ctypes.c_char_p

        lib.c_embed_runtime_output_dim.argtypes = [ctypes.c_void_p]
        lib.c_embed_runtime_output_dim.restype = ctypes.c_int32

        lib.c_embed_runtime_seq_len.argtypes = [ctypes.c_void_p]
        lib.c_embed_runtime_seq_len.restype = ctypes.c_int64

        lib.c_embed_runtime_destroy.argtypes = [ctypes.c_void_p]
        lib.c_embed_runtime_destroy.restype = None

        _EMBED_LIB = lib
        return lib


class _CEmbedRuntime:
    def __init__(self, *, requested_device: str) -> None:
        self._lib = _load_embed_lib()
        self._handle: ctypes.c_void_p | None = None
        self._tokenizer: Any = None
        self._seq_len = _int_env("CDB_EMBED_SEQ_LEN", 128, minimum=8, maximum=8192)
        self._out_dim = 24
        normalized = _normalize_embed_device(requested_device)
        self._device = "NPU" if normalized == "AUTO" else normalized
        self._threads = _int_env("CDB_EMBED_THREADS", 1, minimum=1, maximum=32)
        self._strict = _bool_env("CDB_EMBED_STRICT_DEVICE", True)

        model_path = _resolve_embed_model_path()
        if model_path is None:
            raise RuntimeError(
                "nomic onnx model path not found for c runtime embedding"
            )
        tokenizer_path = _resolve_embed_tokenizer_path(model_path)
        if tokenizer_path is None:
            raise RuntimeError("tokenizer.json not found for c runtime embedding")

        try:
            from tokenizers import Tokenizer  # type: ignore

            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            raise RuntimeError(
                f"failed to load tokenizer for c runtime embedding: {exc}"
            ) from exc

        handle = self._lib.c_embed_runtime_create(
            str(model_path).encode("utf-8"),
            self._device.encode("utf-8"),
            ctypes.c_int64(self._seq_len),
            ctypes.c_int32(self._out_dim),
            ctypes.c_int32(self._threads),
            ctypes.c_int32(1 if self._strict else 0),
        )
        if not handle:
            detail = ""
            try:
                raw = self._lib.c_embed_runtime_last_create_error()
                if raw:
                    detail = str(raw.decode("utf-8", errors="ignore") or "").strip()
            except Exception:
                detail = ""
            if detail:
                raise RuntimeError(f"failed to create c embed runtime handle: {detail}")
            raise RuntimeError("failed to create c embed runtime handle")
        self._handle = handle

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._lib.c_embed_runtime_destroy(self._handle)
        finally:
            self._handle = None

    def selected_device(self) -> str:
        if self._handle is None:
            return "invalid"
        raw = self._lib.c_embed_runtime_selected_device(self._handle)
        if not raw:
            return "unknown"
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return "unknown"

    def cpu_fallback_detected(self) -> bool:
        if self._handle is None:
            return False
        try:
            return bool(
                int(self._lib.c_embed_runtime_cpu_fallback_detected(self._handle))
            )
        except Exception:
            return False

    def cpu_fallback_detail(self) -> str:
        if self._handle is None:
            return ""
        try:
            raw = self._lib.c_embed_runtime_cpu_fallback_detail(self._handle)
        except Exception:
            return ""
        if not raw:
            return ""
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def last_error(self) -> str:
        if self._handle is None:
            return "invalid_handle"
        raw = self._lib.c_embed_runtime_last_error(self._handle)
        if not raw:
            return ""
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return "unknown_error"

    def embed_24(self, text: str) -> list[float] | None:
        if self._handle is None:
            return None

        encoded = self._tokenizer.encode(str(text or ""))
        token_ids = list(getattr(encoded, "ids", []) or [])
        if not token_ids:
            token_ids = [101, 102]

        token_ids = token_ids[: self._seq_len]
        attention_mask = [1] * len(token_ids)
        token_type_ids = [0] * len(token_ids)

        if len(token_ids) < self._seq_len:
            pad = self._seq_len - len(token_ids)
            token_ids.extend([0] * pad)
            attention_mask.extend([0] * pad)
            token_type_ids.extend([0] * pad)

        ids_array = (ctypes.c_int64 * self._seq_len)(*token_ids)
        mask_array = (ctypes.c_int64 * self._seq_len)(*attention_mask)
        type_array = (ctypes.c_int64 * self._seq_len)(*token_type_ids)
        out_array = (ctypes.c_float * self._out_dim)(*([0.0] * self._out_dim))

        ok = int(
            self._lib.c_embed_runtime_embed(
                self._handle,
                ids_array,
                mask_array,
                type_array,
                out_array,
            )
        )
        if ok != 1:
            return None
        return [float(out_array[i]) for i in range(self._out_dim)]


def _c_embed_runtime_enabled() -> bool:
    return _bool_env("CDB_EMBED_IN_C", True)


def _c_embed_runtime_required() -> bool:
    return _bool_env("CDB_EMBED_REQUIRE_C", True)


def _get_c_embed_runtime() -> Any:
    global _EMBED_RUNTIME
    global _EMBED_RUNTIME_ERROR
    global _EMBED_RUNTIME_SOURCE
    global _EMBED_RUNTIME_CPU_FALLBACK
    global _EMBED_RUNTIME_CPU_FALLBACK_DETAIL

    if not _c_embed_runtime_enabled():
        _EMBED_RUNTIME_SOURCE = "disabled"
        _EMBED_RUNTIME_ERROR = "c runtime embedding disabled"
        _EMBED_RUNTIME_CPU_FALLBACK = False
        _EMBED_RUNTIME_CPU_FALLBACK_DETAIL = ""
        return None

    with _EMBED_RUNTIME_LOCK:
        if _EMBED_RUNTIME is not None:
            return _EMBED_RUNTIME

        requested_device = os.getenv("CDB_EMBED_DEVICE", "AUTO")
        device_candidates = _embed_device_candidates(requested_device)
        failures: list[str] = []

        for candidate in device_candidates:
            runtime: _CEmbedRuntime | None = None
            try:
                if candidate == "NPU":
                    _prepare_npu_level_zero_env()

                runtime = _CEmbedRuntime(requested_device=candidate)

                selected_device = str(runtime.selected_device() or "").strip().upper()
                if not _is_hardware_embed_device(selected_device):
                    raise RuntimeError(
                        f"invalid_runtime_device:selected_device={selected_device or 'unknown'}"
                    )
                if candidate == "GPU" and _bool_env("CDB_EMBED_GPU_REQUIRE_CUDA", True):
                    if selected_device != "CUDA":
                        raise RuntimeError(
                            f"cuda_required_for_gpu:selected_device={selected_device or 'unknown'}"
                        )

                fallback_detected = bool(runtime.cpu_fallback_detected())
                fallback_detail = str(runtime.cpu_fallback_detail() or "").strip()
                if fallback_detected:
                    detail = (
                        fallback_detail or "cpu_fallback_detected_during_runtime_init"
                    )
                    raise RuntimeError(detail)

                probe_vec = runtime.embed_24("eta-mu hardware embedding probe")
                runtime_error = str(runtime.last_error() or "").strip()
                fallback_detected = bool(runtime.cpu_fallback_detected())
                fallback_detail = str(runtime.cpu_fallback_detail() or "").strip()
                if fallback_detected:
                    detail = fallback_detail or runtime_error or "cpu_fallback_detected"
                    raise RuntimeError(detail)
                if _is_cpu_fallback_signal(runtime_error):
                    raise RuntimeError(runtime_error)
                if _is_cpu_fallback_signal(fallback_detail):
                    raise RuntimeError(fallback_detail)
                if not (isinstance(probe_vec, list) and len(probe_vec) == 24):
                    detail = runtime_error or "hardware_probe_failed"
                    raise RuntimeError(detail)

                _EMBED_RUNTIME = runtime
                _EMBED_RUNTIME_SOURCE = f"c-onnxruntime:{selected_device}"
                _EMBED_RUNTIME_ERROR = ""
                _EMBED_RUNTIME_CPU_FALLBACK = False
                _EMBED_RUNTIME_CPU_FALLBACK_DETAIL = ""
                return runtime
            except Exception as exc:
                if runtime is not None:
                    try:
                        runtime.close()
                    except Exception:
                        pass
                failures.append(f"{candidate}:{exc}")

        _EMBED_RUNTIME = None
        _EMBED_RUNTIME_SOURCE = "c-unavailable"
        _EMBED_RUNTIME_ERROR = (
            " | ".join(failures) or "hardware_embedding_runtime_unavailable"
        )
        _EMBED_RUNTIME_CPU_FALLBACK = _is_cpu_fallback_signal(_EMBED_RUNTIME_ERROR)
        _EMBED_RUNTIME_CPU_FALLBACK_DETAIL = (
            _EMBED_RUNTIME_ERROR if _EMBED_RUNTIME_CPU_FALLBACK else ""
        )
        return None


def _reset_c_embed_runtime() -> None:
    global _EMBED_RUNTIME
    global _EMBED_RUNTIME_ERROR
    global _EMBED_RUNTIME_SOURCE
    global _EMBED_RUNTIME_CPU_FALLBACK
    global _EMBED_RUNTIME_CPU_FALLBACK_DETAIL

    runtime: Any = None
    with _EMBED_RUNTIME_LOCK:
        runtime = _EMBED_RUNTIME
        _EMBED_RUNTIME = None
        _EMBED_RUNTIME_ERROR = ""
        _EMBED_RUNTIME_SOURCE = "pending"
        _EMBED_RUNTIME_CPU_FALLBACK = False
        _EMBED_RUNTIME_CPU_FALLBACK_DETAIL = ""
    if runtime is not None:
        try:
            runtime.close()
        except Exception:
            pass
    _clear_embed_seed_cache()


def embed_runtime_snapshot() -> dict[str, Any]:
    runtime = _get_c_embed_runtime()
    selected_device = ""
    if runtime is not None:
        try:
            selected_device = str(runtime.selected_device() or "").strip().upper()
        except Exception:
            selected_device = ""
    return {
        "source": str(_EMBED_RUNTIME_SOURCE or "pending"),
        "error": str(_EMBED_RUNTIME_ERROR or ""),
        "cpu_fallback": bool(_EMBED_RUNTIME_CPU_FALLBACK),
        "cpu_fallback_detail": str(_EMBED_RUNTIME_CPU_FALLBACK_DETAIL or ""),
        "selected_device": selected_device,
    }


def embed_text_24_local(
    text: str,
    *,
    requested_device: str | None = None,
) -> list[float] | None:
    prompt = str(text or "").strip()
    if not prompt:
        return None

    requested = _normalize_embed_device(
        requested_device or os.getenv("CDB_EMBED_DEVICE", "AUTO")
    )
    active = _normalize_embed_device(os.getenv("CDB_EMBED_DEVICE", "AUTO"))
    if requested != active:
        os.environ["CDB_EMBED_DEVICE"] = requested
        _reset_c_embed_runtime()

    runtime = _get_c_embed_runtime()
    if runtime is None:
        return None

    try:
        vec = runtime.embed_24(prompt)
    except Exception:
        vec = None
    if not isinstance(vec, list) or len(vec) != 24:
        return None
    try:
        return [float(item) for item in vec]
    except Exception:
        return None


_EMBED_CACHE_HITS = 0
_EMBED_CACHE_MISSES = 0


def _embed_seed_vector_24(text: str) -> tuple[float, ...]:
    global _EMBED_RUNTIME_ERROR
    global _EMBED_RUNTIME_CPU_FALLBACK
    global _EMBED_RUNTIME_CPU_FALLBACK_DETAIL
    global _EMBED_CACHE_HITS, _EMBED_CACHE_MISSES

    key = str(text or "")
    with _EMBED_VECTOR_CACHE_LOCK:
        cached = _EMBED_VECTOR_CACHE.get(key)
    if cached is not None:
        _EMBED_CACHE_HITS += 1
        return cached

    _EMBED_CACHE_MISSES += 1

    runtime = _get_c_embed_runtime()
    if runtime is not None:
        runtime_error = ""
        fallback_detected = False
        fallback_detail = ""
        try:
            vec = runtime.embed_24(key)
        except Exception:
            vec = None
        try:
            runtime_error = str(runtime.last_error() or "").strip()
        except Exception:
            runtime_error = ""

        try:
            fallback_detected = bool(runtime.cpu_fallback_detected())
            fallback_detail = str(runtime.cpu_fallback_detail() or "").strip()
        except Exception:
            fallback_detected = False
            fallback_detail = ""

        if fallback_detected:
            detail = fallback_detail or runtime_error or "npu_cpu_fallback_detected"
            _EMBED_RUNTIME_CPU_FALLBACK = True
            _EMBED_RUNTIME_CPU_FALLBACK_DETAIL = detail
            _EMBED_RUNTIME_ERROR = detail
            vec = None
        elif runtime_error:
            _EMBED_RUNTIME_ERROR = runtime_error
            if _is_cpu_fallback_signal(runtime_error):
                _EMBED_RUNTIME_CPU_FALLBACK = True
                _EMBED_RUNTIME_CPU_FALLBACK_DETAIL = runtime_error

        packed: tuple[float, ...] | None = None
        if isinstance(vec, list) and len(vec) == 24:
            _EMBED_RUNTIME_ERROR = ""
            _EMBED_RUNTIME_CPU_FALLBACK = False
            _EMBED_RUNTIME_CPU_FALLBACK_DETAIL = ""
            packed = tuple(float(v) for v in vec)
        elif _EMBED_RUNTIME_CPU_FALLBACK:
            # Cache zeros for fallback to avoid repeated slow NPU->CPU failure paths
            packed = tuple(0.0 for _ in range(24))

        if packed is not None:
            with _EMBED_VECTOR_CACHE_LOCK:
                if len(_EMBED_VECTOR_CACHE) >= 16384:
                    _EMBED_VECTOR_CACHE.clear()
                _EMBED_VECTOR_CACHE[key] = packed
            return packed

    if not _EMBED_RUNTIME_ERROR:
        _EMBED_RUNTIME_ERROR = "hardware_embedding_runtime_unavailable"
    packed = tuple(0.0 for _ in range(24))
    with _EMBED_VECTOR_CACHE_LOCK:
        if len(_EMBED_VECTOR_CACHE) >= 16384:
            _EMBED_VECTOR_CACHE.clear()
        _EMBED_VECTOR_CACHE[key] = packed
    return packed


def _clear_embed_seed_cache() -> None:
    with _EMBED_VECTOR_CACHE_LOCK:
        _EMBED_VECTOR_CACHE.clear()


def _copy_float_data_to_c_buffer(
    *, source: list[float], target: Any, target_size: int
) -> None:
    limit = min(len(source), max(0, int(target_size)))
    for index in range(limit):
        target[index] = _safe_float(source[index], 0.0)
    for index in range(limit, target_size):
        target[index] = 0.0


class _CDBEngine:
    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib
        self._ptr: ctypes.c_void_p | None = None
        self._count = 0
        self._seed = 0
        self._x: Any = None
        self._y: Any = None
        self._vx: Any = None
        self._vy: Any = None
        self._deflect: Any = None
        self._message: Any = None
        self._entropy: Any = None
        self._owner: Any = None
        self._flags: Any = None
        self._nooi_update_buffer: Any = None
        self._embedding_update_buffer: Any = None
        self._embedding_update_capacity = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._ptr is None:
                self._nooi_update_buffer = None
                self._embedding_update_buffer = None
                self._embedding_update_capacity = 0
                return
            self._lib.cdb_engine_stop(self._ptr)
            self._lib.cdb_engine_destroy(self._ptr)
            self._ptr = None
            self._count = 0
            self._nooi_update_buffer = None
            self._embedding_update_buffer = None
            self._embedding_update_capacity = 0

    def ensure(self, *, count: int, seed: int) -> None:
        target_count = max(64, int(count))
        target_seed = max(1, int(seed))
        with self._lock:
            if (
                self._ptr is not None
                and self._count == target_count
                and self._seed == target_seed
            ):
                return

            if self._ptr is not None:
                self._lib.cdb_engine_stop(self._ptr)
                self._lib.cdb_engine_destroy(self._ptr)
                self._ptr = None

            ptr = self._lib.cdb_engine_create(
                ctypes.c_uint32(target_count), ctypes.c_uint32(target_seed)
            )
            if not ptr:
                raise RuntimeError("failed to create c double-buffer simulation engine")
            start_code = int(self._lib.cdb_engine_start(ptr))
            if start_code != 0:
                self._lib.cdb_engine_destroy(ptr)
                raise RuntimeError("failed to start c double-buffer simulation engine")

            self._ptr = ptr
            self._count = target_count
            self._seed = target_seed

            float_array = ctypes.c_float * target_count
            int_array = ctypes.c_uint32 * target_count
            self._x = float_array()
            self._y = float_array()
            self._vx = float_array()
            self._vy = float_array()
            self._deflect = float_array()
            self._message = float_array()
            self._entropy = float_array()
            self._owner = int_array()
            self._flags = int_array()
            self._nooi_update_buffer = (ctypes.c_float * _CDB_NOOI_FLOAT_COUNT)()
            embedding_size = max(1, self._count * _CDB_EMBED_DIM)
            self._embedding_update_buffer = (ctypes.c_float * embedding_size)()
            self._embedding_update_capacity = embedding_size

    def snapshot(
        self,
    ) -> tuple[
        int,
        int,
        int,
        int,
        int,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
    ]:
        with self._lock:
            if self._ptr is None:
                raise RuntimeError("c double-buffer engine is not initialized")
            frame = ctypes.c_uint64(0)
            force_frame = ctypes.c_uint64(0)
            chaos_frame = ctypes.c_uint64(0)
            semantic_frame = ctypes.c_uint64(0)
            count = int(
                self._lib.cdb_engine_snapshot(
                    self._ptr,
                    self._x,
                    self._y,
                    self._vx,
                    self._vy,
                    self._deflect,
                    self._message,
                    self._entropy,
                    self._owner,
                    self._flags,
                    ctypes.c_uint32(self._count),
                    ctypes.byref(frame),
                    ctypes.byref(force_frame),
                    ctypes.byref(chaos_frame),
                    ctypes.byref(semantic_frame),
                )
            )
            return (
                count,
                int(frame.value),
                int(force_frame.value),
                int(chaos_frame.value),
                int(semantic_frame.value),
                self._x,
                self._y,
                self._vx,
                self._vy,
                self._deflect,
                self._message,
                self._entropy,
                self._owner,
                self._flags,
            )

    def update_nooi(self, data: list[float]) -> None:
        if not data:
            return
        with self._lock:
            if self._ptr is None:
                return
            expected_size = _CDB_NOOI_FLOAT_COUNT
            if self._nooi_update_buffer is None:
                self._nooi_update_buffer = (ctypes.c_float * expected_size)()
            _copy_float_data_to_c_buffer(
                source=data,
                target=self._nooi_update_buffer,
                target_size=expected_size,
            )
            self._lib.cdb_engine_update_nooi(self._ptr, self._nooi_update_buffer)

    def update_embeddings(self, data: list[float]) -> None:
        if not data:
            return
        with self._lock:
            if self._ptr is None:
                return
            expected_size = max(1, self._count * _CDB_EMBED_DIM)
            if (
                self._embedding_update_buffer is None
                or self._embedding_update_capacity != expected_size
            ):
                self._embedding_update_buffer = (ctypes.c_float * expected_size)()
                self._embedding_update_capacity = expected_size
            _copy_float_data_to_c_buffer(
                source=data,
                target=self._embedding_update_buffer,
                target_size=expected_size,
            )
            self._lib.cdb_engine_update_embeddings(
                self._ptr, self._embedding_update_buffer
            )

    def set_flags(self, flags: int) -> None:
        with self._lock:
            if self._ptr is None:
                return
            self._lib.cdb_engine_set_flags(self._ptr, ctypes.c_uint32(flags))


def _get_engine(*, count: int, seed: int) -> _CDBEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = _CDBEngine(_load_native_lib())
        _ENGINE.ensure(count=count, seed=seed)
        return _ENGINE


def shutdown_c_double_buffer_backend() -> None:
    global _ENGINE
    global _EMBED_RUNTIME
    with _ENGINE_LOCK:
        if _ENGINE is None:
            pass
        else:
            _ENGINE.close()
            _ENGINE = None

    with _EMBED_RUNTIME_LOCK:
        runtime = _EMBED_RUNTIME
        _EMBED_RUNTIME = None
    if runtime is not None:
        try:
            runtime.close()
        except Exception:
            pass

    with _LIB_LOCK:
        lib = _LIB
    if lib is not None and hasattr(lib, "cdb_release_thread_scratch"):
        try:
            lib.cdb_release_thread_scratch()
        except Exception:
            pass

    try:
        _clear_embed_seed_cache()
    except Exception:
        pass


atexit.register(shutdown_c_double_buffer_backend)


def _collision_worker_count(*, count: int, worker_count: int | None) -> int:
    if worker_count is None:
        cpu_count = int(os.cpu_count() or 1)
        cpu_count = max(1, min(64, cpu_count))
        default_workers = cpu_count
        if default_workers > 4:
            default_workers -= 3
        default_workers = max(1, min(32, default_workers))
        worker_count = _int_env(
            "CDB_COLLISION_WORKERS",
            _int_env("CDB_FORCE_WORKERS", default_workers, minimum=1, maximum=32),
            minimum=1,
            maximum=32,
        )
    workers = max(1, min(32, int(worker_count)))
    return max(1, min(workers, count))


def _collision_ctype_state(capacity: int) -> dict[str, Any]:
    float_array = ctypes.c_float * capacity
    uint32_array = ctypes.c_uint32 * capacity
    return {
        "capacity": capacity,
        "x": float_array(),
        "y": float_array(),
        "vx": float_array(),
        "vy": float_array(),
        "radius": float_array(),
        "mass": float_array(),
        "collision": uint32_array(),
        "shrink_candidate_runs": 0,
        "last_native_release_at": 0.0,
    }


def _collision_release_native_thread_scratch_if_needed(
    *,
    prior_capacity: int,
    next_capacity: int,
    state: dict[str, Any],
) -> None:
    if prior_capacity < (_COLLISION_SCRATCH_SHRINK_MIN_CAPACITY * 2):
        return
    if next_capacity > max(2, prior_capacity // _COLLISION_SCRATCH_SHRINK_FACTOR):
        return
    now_value = time.monotonic()
    last_release = _safe_float(state.get("last_native_release_at", 0.0), 0.0)
    if (now_value - last_release) < _COLLISION_NATIVE_RELEASE_COOLDOWN_SECONDS:
        return
    try:
        lib = _load_native_lib()
    except Exception:
        state["last_native_release_at"] = now_value
        return
    if hasattr(lib, "cdb_release_thread_scratch"):
        try:
            lib.cdb_release_thread_scratch()
        except Exception:
            pass
    state["last_native_release_at"] = now_value


def _collision_ctype_scratch(count: int) -> dict[str, Any]:
    cap_target = max(2, int(count))
    cap_target = 1 << (cap_target - 1).bit_length()

    state = getattr(_COLLISION_CTYPE_SCRATCH, "state", None)
    if not isinstance(state, dict):
        state = {}
        _COLLISION_CTYPE_SCRATCH.state = state

    capacity = int(_safe_float(state.get("capacity", 0), 0.0))
    if capacity >= cap_target:
        low_water_runs = int(_safe_float(state.get("shrink_candidate_runs", 0), 0.0))
        if capacity >= _COLLISION_SCRATCH_SHRINK_MIN_CAPACITY and cap_target <= max(
            2, capacity // _COLLISION_SCRATCH_SHRINK_FACTOR
        ):
            low_water_runs += 1
            state["shrink_candidate_runs"] = low_water_runs
            if low_water_runs >= _COLLISION_SCRATCH_SHRINK_RUNS:
                next_state = _collision_ctype_state(cap_target)
                next_state["last_native_release_at"] = _safe_float(
                    state.get("last_native_release_at", 0.0),
                    0.0,
                )
                _collision_release_native_thread_scratch_if_needed(
                    prior_capacity=capacity,
                    next_capacity=cap_target,
                    state=next_state,
                )
                _COLLISION_CTYPE_SCRATCH.state = next_state
                return next_state
        else:
            state["shrink_candidate_runs"] = 0
        return state

    next_state = _collision_ctype_state(cap_target)
    next_state["last_native_release_at"] = _safe_float(
        state.get("last_native_release_at", 0.0),
        0.0,
    )
    _COLLISION_CTYPE_SCRATCH.state = next_state
    return next_state


def _resolve_semantic_collisions_native_buffers(
    *,
    x: list[float],
    y: list[float],
    vx: list[float],
    vy: list[float],
    radius: list[float],
    mass: list[float],
    restitution: float,
    separation_percent: float,
    cell_size: float,
    worker_count: int | None,
) -> tuple[int, Any, Any, Any, Any, Any] | None:
    count = len(x)
    if not (
        len(y) == count
        and len(vx) == count
        and len(vy) == count
        and len(radius) == count
        and len(mass) == count
    ):
        return None

    try:
        lib = _load_native_lib()
    except Exception:
        return None
    if not hasattr(lib, "cdb_resolve_semantic_collisions"):
        return None

    workers = _collision_worker_count(count=count, worker_count=worker_count)

    bounded_restitution = max(0.0, min(1.0, _safe_float(restitution, 0.88)))
    bounded_separation = max(0.0, min(1.2, _safe_float(separation_percent, 0.72)))
    bounded_cell_size = max(0.005, min(0.25, _safe_float(cell_size, 0.04)))

    scratch = _collision_ctype_scratch(count)
    x_arr = scratch.get("x")
    y_arr = scratch.get("y")
    vx_arr = scratch.get("vx")
    vy_arr = scratch.get("vy")
    radius_arr = scratch.get("radius")
    mass_arr = scratch.get("mass")
    collision_arr = scratch.get("collision")

    if (
        x_arr is None
        or y_arr is None
        or vx_arr is None
        or vy_arr is None
        or radius_arr is None
        or mass_arr is None
        or collision_arr is None
    ):
        return None

    for idx in range(count):
        x_arr[idx] = _clamp01(_safe_float(x[idx], 0.5))
        y_arr[idx] = _clamp01(_safe_float(y[idx], 0.5))
        vx_arr[idx] = _safe_float(vx[idx], 0.0)
        vy_arr[idx] = _safe_float(vy[idx], 0.0)
        radius_arr[idx] = max(0.0, _safe_float(radius[idx], 0.0))
        mass_arr[idx] = max(0.2, _safe_float(mass[idx], 1.0))
        collision_arr[idx] = 0

    try:
        lib.cdb_resolve_semantic_collisions(
            ctypes.c_uint32(count),
            x_arr,
            y_arr,
            vx_arr,
            vy_arr,
            radius_arr,
            mass_arr,
            ctypes.c_float(bounded_restitution),
            ctypes.c_float(bounded_separation),
            ctypes.c_float(bounded_cell_size),
            ctypes.c_uint32(workers),
            collision_arr,
        )
    except Exception:
        return None

    return count, x_arr, y_arr, vx_arr, vy_arr, collision_arr


def resolve_semantic_collisions_native(
    *,
    x: list[float],
    y: list[float],
    vx: list[float],
    vy: list[float],
    radius: list[float],
    mass: list[float],
    restitution: float = 0.88,
    separation_percent: float = 0.72,
    cell_size: float = 0.04,
    worker_count: int | None = None,
) -> tuple[list[float], list[float], list[float], list[float], list[int]] | None:
    count = len(x)
    if count <= 1:
        if not (
            len(y) == count
            and len(vx) == count
            and len(vy) == count
            and len(radius) == count
            and len(mass) == count
        ):
            return None
        return (
            list(x),
            list(y),
            list(vx),
            list(vy),
            [0 for _ in range(max(0, count))],
        )
    resolved = _resolve_semantic_collisions_native_buffers(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        radius=radius,
        mass=mass,
        restitution=restitution,
        separation_percent=separation_percent,
        cell_size=cell_size,
        worker_count=worker_count,
    )
    if not (isinstance(resolved, tuple) and len(resolved) == 6):
        return None
    count, x_arr, y_arr, vx_arr, vy_arr, collision_arr = resolved

    return (
        [float(x_arr[i]) for i in range(count)],
        [float(y_arr[i]) for i in range(count)],
        [float(vx_arr[i]) for i in range(count)],
        [float(vy_arr[i]) for i in range(count)],
        [int(collision_arr[i]) for i in range(count)],
    )


def resolve_semantic_collisions_native_inplace(
    *,
    x: list[float],
    y: list[float],
    vx: list[float],
    vy: list[float],
    radius: list[float],
    mass: list[float],
    collisions_out: list[int] | None = None,
    restitution: float = 0.88,
    separation_percent: float = 0.72,
    cell_size: float = 0.04,
    worker_count: int | None = None,
) -> bool:
    count = len(x)
    if not (
        len(y) == count
        and len(vx) == count
        and len(vy) == count
        and len(radius) == count
        and len(mass) == count
    ):
        return False

    if count <= 1:
        if collisions_out is not None:
            if len(collisions_out) > count:
                del collisions_out[count:]
            while len(collisions_out) < count:
                collisions_out.append(0)
            for idx in range(count):
                collisions_out[idx] = 0
        return True

    resolved = _resolve_semantic_collisions_native_buffers(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        radius=radius,
        mass=mass,
        restitution=restitution,
        separation_percent=separation_percent,
        cell_size=cell_size,
        worker_count=worker_count,
    )
    if not (isinstance(resolved, tuple) and len(resolved) == 6):
        return False
    count, x_arr, y_arr, vx_arr, vy_arr, collision_arr = resolved

    for idx in range(count):
        x[idx] = float(x_arr[idx])
        y[idx] = float(y_arr[idx])
        vx[idx] = float(vx_arr[idx])
        vy[idx] = float(vy_arr[idx])

    if collisions_out is not None:
        if len(collisions_out) > count:
            del collisions_out[count:]
        while len(collisions_out) < count:
            collisions_out.append(0)
        for idx in range(count):
            collisions_out[idx] = int(collision_arr[idx])

    return True


def compute_growth_guard_scores_native(
    *,
    importance: list[float],
    layer_counts: list[int],
    has_collection: list[bool],
    recent_hit: list[bool],
) -> list[float] | None:
    count = len(importance)
    if count <= 0:
        return []
    if not (
        len(layer_counts) == count
        and len(has_collection) == count
        and len(recent_hit) == count
    ):
        return None

    try:
        lib = _load_native_lib()
    except Exception:
        return None
    if not hasattr(lib, "cdb_growth_guard_scores"):
        return None

    float_array = ctypes.c_float * count
    uint32_array = ctypes.c_uint32 * count
    uint8_array = ctypes.c_uint8 * count

    importance_arr = float_array(
        *[_clamp01(_safe_float(value, 0.25)) for value in importance]
    )
    layer_arr = uint32_array(
        *[max(0, int(_safe_float(value, 0.0))) for value in layer_counts]
    )
    collection_arr = uint8_array(*[1 if bool(value) else 0 for value in has_collection])
    recent_arr = uint8_array(*[1 if bool(value) else 0 for value in recent_hit])
    out_scores = float_array()

    try:
        written = int(
            lib.cdb_growth_guard_scores(
                importance_arr,
                layer_arr,
                collection_arr,
                recent_arr,
                ctypes.c_uint32(count),
                out_scores,
            )
        )
    except Exception:
        return None

    if written <= 0:
        return None
    limit = min(count, written)
    return [_clamp01(_safe_float(out_scores[index], 0.0)) for index in range(limit)] + [
        0.0 for _ in range(limit, count)
    ]


def compute_growth_guard_pressure_native(
    *,
    file_count: int,
    edge_count: int,
    crawler_count: int,
    item_count: int,
    sim_point_budget: int,
    queue_pending_count: int,
    queue_event_count: int,
    cpu_utilization: float,
    weaver_graph_node_limit: float,
    watch_threshold: float,
    critical_threshold: float,
) -> dict[str, Any] | None:
    try:
        lib = _load_native_lib()
    except Exception:
        return None
    if not hasattr(lib, "cdb_growth_guard_pressure"):
        return None

    blend = ctypes.c_float(0.0)
    point_ratio = ctypes.c_float(0.0)
    file_ratio = ctypes.c_float(0.0)
    edge_ratio = ctypes.c_float(0.0)
    crawler_ratio = ctypes.c_float(0.0)
    queue_ratio = ctypes.c_float(0.0)
    resource_ratio = ctypes.c_float(0.0)
    target_file_nodes = ctypes.c_uint32(0)
    target_edge_count = ctypes.c_uint32(0)
    mode_value = ctypes.c_uint32(0)

    try:
        status = int(
            lib.cdb_growth_guard_pressure(
                ctypes.c_uint32(max(0, int(file_count))),
                ctypes.c_uint32(max(0, int(edge_count))),
                ctypes.c_uint32(max(0, int(crawler_count))),
                ctypes.c_uint32(max(0, int(item_count))),
                ctypes.c_uint32(max(0, int(sim_point_budget))),
                ctypes.c_uint32(max(0, int(queue_pending_count))),
                ctypes.c_uint32(max(0, int(queue_event_count))),
                ctypes.c_float(_safe_float(cpu_utilization, 0.0)),
                ctypes.c_float(max(1.0, _safe_float(weaver_graph_node_limit, 1.0))),
                ctypes.c_float(_safe_float(watch_threshold, 0.0)),
                ctypes.c_float(_safe_float(critical_threshold, 1.0)),
                ctypes.byref(blend),
                ctypes.byref(point_ratio),
                ctypes.byref(file_ratio),
                ctypes.byref(edge_ratio),
                ctypes.byref(crawler_ratio),
                ctypes.byref(queue_ratio),
                ctypes.byref(resource_ratio),
                ctypes.byref(target_file_nodes),
                ctypes.byref(target_edge_count),
                ctypes.byref(mode_value),
            )
        )
    except Exception:
        return None

    if status != 0:
        return None

    mode = _GROWTH_MODE_MAP.get(int(mode_value.value), "normal")
    return {
        "mode": mode,
        "blend": _clamp01(_safe_float(blend.value, 0.0)),
        "point_ratio": _clamp01(_safe_float(point_ratio.value, 0.0)),
        "file_ratio": _clamp01(_safe_float(file_ratio.value, 0.0)),
        "edge_ratio": _clamp01(_safe_float(edge_ratio.value, 0.0)),
        "crawler_ratio": _clamp01(_safe_float(crawler_ratio.value, 0.0)),
        "queue_ratio": _clamp01(_safe_float(queue_ratio.value, 0.0)),
        "resource_ratio": _clamp01(_safe_float(resource_ratio.value, 0.0)),
        "target_file_nodes": max(0, int(target_file_nodes.value)),
        "target_edge_count": max(0, int(target_edge_count.value)),
    }


def compute_graph_runtime_maps_native(
    *,
    file_graph: dict[str, Any] | None,
    presence_ids: list[str],
    presence_layout: dict[str, tuple[float, float, float | None]],
    presence_impacts: list[dict[str, Any]] | None,
    queue_ratio: float,
    cpu_utilization: float,
    cost_weights: tuple[float, float, float] = (1.0, 2.0, 1.0),
    radius_cost: float = 6.0,
    gravity_const: float = 1.0,
    epsilon: float = 0.001,
) -> dict[str, Any] | None:
    graph = file_graph if isinstance(file_graph, dict) else {}
    graph_nodes_raw = graph.get("nodes", [])
    if not isinstance(graph_nodes_raw, list) or not graph_nodes_raw:
        graph_nodes_raw = [
            *(
                graph.get("field_nodes", [])
                if isinstance(graph.get("field_nodes", []), list)
                else []
            ),
            *(
                graph.get("tag_nodes", [])
                if isinstance(graph.get("tag_nodes", []), list)
                else []
            ),
            *(
                graph.get("file_nodes", [])
                if isinstance(graph.get("file_nodes", []), list)
                else []
            ),
        ]

    node_ids: list[str] = []
    node_positions: list[tuple[float, float]] = []
    node_index_by_id: dict[str, int] = {}
    for node in graph_nodes_raw:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        if not node_id or node_id in node_index_by_id:
            continue
        node_index_by_id[node_id] = len(node_ids)
        node_ids.append(node_id)
        node_positions.append(
            (
                _clamp01(_safe_float(node.get("x", 0.5), 0.5)),
                _clamp01(_safe_float(node.get("y", 0.5), 0.5)),
            )
        )

    node_count = len(node_ids)
    if node_count <= 0:
        return None

    raw_edges = graph.get("edges", [])
    if not isinstance(raw_edges, list) or not raw_edges:
        return None

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_affinity: list[float] = []
    seen_edges: set[tuple[int, int]] = set()
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if not source_id or not target_id:
            continue
        source_index = node_index_by_id.get(source_id)
        target_index = node_index_by_id.get(target_id)
        if source_index is None or target_index is None or source_index == target_index:
            continue
        affinity_value = _clamp01(
            _safe_float(
                edge.get(
                    "semantic_affinity",
                    edge.get("a_e", edge.get("weight", 0.5)),
                ),
                0.5,
            )
        )
        forward = (int(source_index), int(target_index))
        reverse = (int(target_index), int(source_index))
        if forward not in seen_edges:
            seen_edges.add(forward)
            edge_sources.append(forward[0])
            edge_targets.append(forward[1])
            edge_affinity.append(float(affinity_value))
        if reverse not in seen_edges:
            seen_edges.add(reverse)
            edge_sources.append(reverse[0])
            edge_targets.append(reverse[1])
            edge_affinity.append(float(affinity_value))

    edge_count = len(edge_sources)
    if edge_count <= 0:
        return None

    impact_by_id = {
        str(row.get("id", "")).strip(): row
        for row in (presence_impacts if isinstance(presence_impacts, list) else [])
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }

    source_nodes: list[int] = []
    source_mass: list[float] = []
    source_need: list[float] = []
    source_need_by_resource: list[dict[str, float]] = []
    source_profiles: list[dict[str, Any]] = []
    source_node_index_by_presence: dict[str, int] = {}
    queue_clamped = _clamp01(_safe_float(queue_ratio, 0.0))
    node_source_pressure: list[float] = [0.0 for _ in range(node_count)]

    for presence_id in presence_ids:
        anchor_x, anchor_y, _ = presence_layout.get(presence_id, (0.5, 0.5, None))
        nearest_index = min(
            range(node_count),
            key=lambda idx: (
                ((node_positions[idx][0] - anchor_x) ** 2)
                + ((node_positions[idx][1] - anchor_y) ** 2)
            ),
        )
        source_node_index_by_presence[presence_id] = int(nearest_index)

        impact = impact_by_id.get(presence_id, {})
        affected_by = impact.get("affected_by", {}) if isinstance(impact, dict) else {}
        affects = impact.get("affects", {}) if isinstance(impact, dict) else {}
        wallet = impact.get("resource_wallet", {}) if isinstance(impact, dict) else {}

        file_signal = _clamp01(_safe_float(affected_by.get("files", 0.0), 0.0))
        click_signal = _clamp01(_safe_float(affected_by.get("clicks", 0.0), 0.0))
        resource_signal = _clamp01(_safe_float(affected_by.get("resource", 0.0), 0.0))
        world_signal = _clamp01(_safe_float(affects.get("world", 0.0), 0.0))
        wallet_total = 0.0
        if isinstance(wallet, dict):
            wallet_total = sum(
                max(0.0, _safe_float(value, 0.0)) for value in wallet.values()
            )
        wallet_ratio = _clamp01(wallet_total / 8.0)

        need_value = _clamp01(
            (resource_signal * 0.4)
            + (file_signal * 0.2)
            + (click_signal * 0.1)
            + (queue_clamped * 0.15)
            + ((1.0 - wallet_ratio) * 0.15)
        )
        mass_value = max(
            0.2,
            0.6 + (world_signal * 1.2) + (file_signal * 0.35) + (wallet_ratio * 0.25),
        )

        source_nodes.append(int(nearest_index))
        source_mass.append(float(mass_value))
        source_need.append(float(max(0.01, need_value)))
        need_model = _presence_resource_need_model(
            presence_id=presence_id,
            impact=impact if isinstance(impact, dict) else {},
            queue_ratio=queue_clamped,
            base_need=need_value,
        )
        need_by_resource_raw = (
            need_model.get("needs", {}) if isinstance(need_model, dict) else {}
        )
        need_by_resource = {
            resource: _clamp01(
                _safe_float(need_by_resource_raw.get(resource, 0.0), 0.0)
            )
            for resource in _RESOURCE_TYPES
        }
        source_need_by_resource.append(need_by_resource)
        node_source_pressure[int(nearest_index)] += float(max(0.01, need_value))

        mask_nodes = _mask_nodes_for_anchor(
            node_ids=node_ids,
            node_positions=node_positions,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            k=3,
        )
        influence_strength = _clamp01(
            (world_signal * 0.55) + (file_signal * 0.25) + (click_signal * 0.2)
        )
        source_profiles.append(
            {
                "presence_id": presence_id,
                "source_node_id": node_ids[int(nearest_index)],
                "mask": {
                    "mode": "nearest-k",
                    "k": int(len(mask_nodes)),
                    "nodes": mask_nodes,
                },
                "influence": {
                    "mode": "anchor-mask",
                    "strength": round(influence_strength, 6),
                    "anchor": {
                        "x": round(_clamp01(_safe_float(anchor_x, 0.5)), 6),
                        "y": round(_clamp01(_safe_float(anchor_y, 0.5)), 6),
                    },
                },
                "need_scalar": round(max(0.01, need_value), 6),
                "need_by_resource": {
                    resource: round(
                        _clamp01(_safe_float(need_by_resource.get(resource, 0.0), 0.0)),
                        6,
                    )
                    for resource in _RESOURCE_TYPES
                },
                "need_model": {
                    "kind": "ema-logistic.v1",
                    "priority": round(
                        max(0.0, _safe_float(need_model.get("priority", 1.0), 1.0)),
                        6,
                    ),
                    "alpha": round(
                        max(0.0, _safe_float(need_model.get("alpha", 0.0), 0.0)),
                        6,
                    ),
                    "util_raw": {
                        resource: round(
                            _clamp01(
                                _safe_float(
                                    (
                                        need_model.get("util_raw", {})
                                        if isinstance(need_model, dict)
                                        else {}
                                    ).get(resource, 0.0),
                                    0.0,
                                )
                            ),
                            6,
                        )
                        for resource in _RESOURCE_TYPES
                    },
                    "util_ema": {
                        resource: round(
                            _clamp01(
                                _safe_float(
                                    (
                                        need_model.get("util_ema", {})
                                        if isinstance(need_model, dict)
                                        else {}
                                    ).get(resource, 0.0),
                                    0.0,
                                )
                            ),
                            6,
                        )
                        for resource in _RESOURCE_TYPES
                    },
                    "thresholds": {
                        resource: round(
                            _clamp01(
                                _safe_float(
                                    (
                                        need_model.get("thresholds", {})
                                        if isinstance(need_model, dict)
                                        else {}
                                    ).get(resource, 0.0),
                                    0.0,
                                )
                            ),
                            6,
                        )
                        for resource in _RESOURCE_TYPES
                    },
                },
                "mass": round(mass_value, 6),
            }
        )

    if not source_nodes:
        source_nodes = [0]
        source_mass = [1.0]
        source_need = [max(0.05, queue_clamped)]
        fallback_need = _default_resource_signature(queue_clamped)
        source_need_by_resource = [fallback_need]
        source_profiles = [
            {
                "presence_id": "anchor_registry",
                "source_node_id": node_ids[0],
                "mask": {
                    "mode": "nearest-k",
                    "k": 1,
                    "nodes": [{"node_id": node_ids[0], "weight": 1.0, "distance": 0.0}],
                },
                "influence": {
                    "mode": "fallback",
                    "strength": 1.0,
                    "anchor": {
                        "x": round(_safe_float(node_positions[0][0], 0.5), 6),
                        "y": round(_safe_float(node_positions[0][1], 0.5), 6),
                    },
                },
                "need_scalar": round(max(0.05, queue_clamped), 6),
                "need_by_resource": {
                    resource: round(
                        _clamp01(_safe_float(fallback_need.get(resource, 0.0), 0.0)), 6
                    )
                    for resource in _RESOURCE_TYPES
                },
                "need_model": {
                    "kind": "ema-logistic.v1",
                    "priority": 1.0,
                    "alpha": round(_RESOURCE_NEED_EMA_ALPHA, 6),
                    "util_raw": {
                        resource: round(
                            _clamp01(
                                _safe_float(fallback_need.get(resource, 0.0), 0.0)
                            ),
                            6,
                        )
                        for resource in _RESOURCE_TYPES
                    },
                    "util_ema": {
                        resource: round(
                            _clamp01(
                                _safe_float(fallback_need.get(resource, 0.0), 0.0)
                            ),
                            6,
                        )
                        for resource in _RESOURCE_TYPES
                    },
                    "thresholds": {
                        resource: round(
                            _clamp01(
                                _safe_float(
                                    _RESOURCE_NEED_THRESHOLDS.get(resource, 0.4),
                                    0.4,
                                )
                            ),
                            6,
                        )
                        for resource in _RESOURCE_TYPES
                    },
                },
                "mass": 1.0,
            }
        ]

    try:
        lib = _load_native_lib()
    except Exception:
        return None
    if not hasattr(lib, "cdb_graph_runtime_maps"):
        return None

    edge_src_arr = (ctypes.c_uint32 * edge_count)(*edge_sources)
    edge_dst_arr = (ctypes.c_uint32 * edge_count)(*edge_targets)
    edge_aff_arr = (ctypes.c_float * edge_count)(*edge_affinity)

    source_count = len(source_nodes)
    source_nodes_arr = (ctypes.c_uint32 * source_count)(*source_nodes)
    source_mass_arr = (ctypes.c_float * source_count)(*source_mass)
    source_need_arr = (ctypes.c_float * source_count)(*source_need)

    out_min_dist = (ctypes.c_float * node_count)()
    out_gravity = (ctypes.c_float * node_count)()
    out_edge_cost = (ctypes.c_float * edge_count)()
    out_node_sat = (ctypes.c_float * node_count)()
    out_node_price = (ctypes.c_float * node_count)()

    cost_w_l = _safe_float(cost_weights[0] if len(cost_weights) >= 1 else 1.0, 1.0)
    cost_w_c = _safe_float(cost_weights[1] if len(cost_weights) >= 2 else 2.0, 2.0)
    cost_w_s = _safe_float(cost_weights[2] if len(cost_weights) >= 3 else 1.0, 1.0)
    cpu_ratio = _clamp01(_safe_float(cpu_utilization, 0.0) / 100.0)
    radius_cost_value = max(0.1, _safe_float(radius_cost, 6.0))
    gravity_const_value = max(0.1, _safe_float(gravity_const, 1.0))
    epsilon_value = max(1e-6, _safe_float(epsilon, 0.001))
    cost_components = _derive_edge_cost_components(
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_affinity=edge_affinity,
        node_count=node_count,
        queue_ratio=queue_clamped,
        cpu_ratio=cpu_ratio,
        cost_w_l=cost_w_l,
        cost_w_c=cost_w_c,
        cost_w_s=cost_w_s,
    )
    edge_saturation = list(cost_components.get("edge_saturation", []))
    edge_latency_component = list(cost_components.get("edge_latency_component", []))
    edge_congestion_component = list(
        cost_components.get("edge_congestion_component", [])
    )
    edge_semantic_component = list(cost_components.get("edge_semantic_component", []))
    edge_base_cost = list(cost_components.get("edge_base_cost", []))
    global_saturation = _clamp01(
        _safe_float(cost_components.get("global_saturation", 0.0), 0.0)
    )

    try:
        status = int(
            lib.cdb_graph_runtime_maps(
                ctypes.c_uint32(node_count),
                ctypes.c_uint32(edge_count),
                edge_src_arr,
                edge_dst_arr,
                edge_aff_arr,
                ctypes.c_float(queue_clamped),
                ctypes.c_float(cpu_ratio),
                ctypes.c_float(cost_w_l),
                ctypes.c_float(cost_w_c),
                ctypes.c_float(cost_w_s),
                source_nodes_arr,
                source_mass_arr,
                source_need_arr,
                ctypes.c_uint32(source_count),
                ctypes.c_float(radius_cost_value),
                ctypes.c_float(gravity_const_value),
                ctypes.c_float(epsilon_value),
                out_min_dist,
                out_gravity,
                out_edge_cost,
                out_node_sat,
                out_node_price,
            )
        )
    except Exception:
        return None

    if status != 0:
        return None

    min_distance = [
        _safe_float(out_min_dist[index], -1.0) for index in range(node_count)
    ]
    gravity = [
        max(0.0, _safe_float(out_gravity[index], 0.0)) for index in range(node_count)
    ]
    node_saturation = [
        _clamp01(_safe_float(out_node_sat[index], queue_clamped))
        for index in range(node_count)
    ]
    node_price = [
        max(0.0, _safe_float(out_node_price[index], 1.0)) for index in range(node_count)
    ]
    edge_cost = [
        max(0.0, _safe_float(out_edge_cost[index], 0.0)) for index in range(edge_count)
    ]

    edge_health: list[float] = []
    edge_upkeep_penalty: list[float] = []
    adjusted_edge_cost: list[float] = []
    active_edge_health_keys: set[str] = set()
    for edge_index in range(edge_count):
        src = max(0, min(node_count - 1, int(edge_sources[edge_index])))
        dst = max(0, min(node_count - 1, int(edge_targets[edge_index])))
        key = _edge_health_key(node_ids, src, dst)
        active_edge_health_keys.add(key)
        previous = _EDGE_HEALTH_REGISTRY.get(
            key,
            _edge_health_default(
                _safe_float(
                    edge_affinity[edge_index]
                    if edge_index < len(edge_affinity)
                    else 0.5,
                    0.5,
                )
            ),
        )
        source_pressure = _clamp01(
            _safe_float(
                node_source_pressure[src] if src < len(node_source_pressure) else 0.0,
                0.0,
            )
        )
        source_relief = 1.0 - _clamp01(
            _safe_float(
                node_saturation[src] if src < len(node_saturation) else queue_clamped,
                queue_clamped,
            )
        )
        target_relief = 1.0 - _clamp01(
            _safe_float(
                node_saturation[dst] if dst < len(node_saturation) else queue_clamped,
                queue_clamped,
            )
        )
        sponsor_signal = _clamp01(
            (source_pressure * 0.68) + (source_relief * 0.2) + (target_relief * 0.12)
        )
        next_health = _clamp01(
            max(_EDGE_HEALTH_FLOOR, previous * _EDGE_HEALTH_DECAY)
            - (queue_clamped * _EDGE_HEALTH_QUEUE_PENALTY)
            + (sponsor_signal * _EDGE_HEALTH_REPAIR_GAIN)
        )
        _EDGE_HEALTH_REGISTRY[key] = next_health
        edge_health.append(next_health)

        base_cost = max(
            0.0001,
            _safe_float(
                edge_base_cost[edge_index]
                if edge_index < len(edge_base_cost)
                else edge_cost[edge_index],
                _safe_float(edge_cost[edge_index], 0.0),
            ),
        )
        upkeep_penalty = (1.0 - next_health) * 0.35
        edge_upkeep_penalty.append(upkeep_penalty)
        adjusted_edge_cost.append(base_cost + upkeep_penalty)

    _prune_edge_health_registry(active_keys=active_edge_health_keys)

    edge_cost = adjusted_edge_cost

    resource_gravity_maps, resource_gravity_peaks = _compute_resource_gravity_maps(
        node_count=node_count,
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_cost=edge_cost,
        source_nodes=source_nodes,
        source_mass=source_mass,
        source_need_by_resource=source_need_by_resource,
        radius_cost=radius_cost_value,
        gravity_const=gravity_const_value,
        epsilon=epsilon_value,
    )
    resource_peak_max = max(resource_gravity_peaks.values(), default=0.0)
    active_resource_types = [
        resource
        for resource in _RESOURCE_TYPES
        if _safe_float(resource_gravity_peaks.get(resource, 0.0), 0.0) > 1e-8
    ]

    edge_cost_mean = (sum(edge_cost) / edge_count) if edge_count > 0 else 0.0
    gravity_mean = (sum(gravity) / node_count) if node_count > 0 else 0.0
    price_mean = (sum(node_price) / node_count) if node_count > 0 else 0.0
    edge_health_mean = (sum(edge_health) / edge_count) if edge_count > 0 else 0.0
    edge_saturation_mean = (
        (sum(edge_saturation) / edge_count)
        if edge_count > 0 and edge_saturation
        else 0.0
    )
    edge_saturation_max = max(edge_saturation) if edge_saturation else 0.0
    edge_affinity_mean = (
        (sum(edge_affinity) / edge_count) if edge_count > 0 and edge_affinity else 0.0
    )
    edge_upkeep_penalty_mean = (
        (sum(edge_upkeep_penalty) / edge_count)
        if edge_count > 0 and edge_upkeep_penalty
        else 0.0
    )
    gravity_max = max(gravity) if gravity else 0.0
    edge_cost_max = max(edge_cost) if edge_cost else 0.0
    price_max = max(node_price) if node_price else 0.0
    edge_health_max = max(edge_health) if edge_health else 0.0
    edge_health_min = min(edge_health) if edge_health else 0.0

    top_indices = sorted(range(node_count), key=lambda idx: gravity[idx], reverse=True)[
        :3
    ]
    top_nodes = [
        {
            "node_id": node_ids[index],
            "gravity": round(gravity[index], 6),
            "local_price": round(node_price[index], 6),
            "distance_cost": round(min_distance[index], 6),
        }
        for index in top_indices
    ]

    return {
        "record": _CDB_GRAPH_RUNTIME_RECORD,
        "schema_version": _CDB_GRAPH_RUNTIME_SCHEMA,
        "node_ids": node_ids,
        "node_positions": node_positions,
        "edge_src_index": edge_sources,
        "edge_dst_index": edge_targets,
        "source_node_index_by_presence": source_node_index_by_presence,
        "source_profiles": source_profiles,
        "presence_source_count": int(len(source_profiles)),
        "presence_model": {
            "mask": "nearest-k.v1",
            "need": "ema-logistic-resource-need.v2",
            "mass": "signal-wallet.v1",
        },
        "min_distance": min_distance,
        "gravity": gravity,
        "node_saturation": node_saturation,
        "node_price": node_price,
        "edge_cost": edge_cost,
        "edge_health": edge_health,
        "edge_affinity": edge_affinity,
        "edge_saturation": edge_saturation,
        "edge_latency_component": edge_latency_component,
        "edge_congestion_component": edge_congestion_component,
        "edge_semantic_component": edge_semantic_component,
        "edge_upkeep_penalty": edge_upkeep_penalty,
        "resource_types": list(_RESOURCE_TYPES),
        "resource_gravity_maps": resource_gravity_maps,
        "resource_gravity_peaks": {
            resource: round(
                _safe_float(resource_gravity_peaks.get(resource, 0.0), 0.0), 6
            )
            for resource in _RESOURCE_TYPES
        },
        "resource_gravity_peak_max": round(resource_peak_max, 6),
        "active_resource_types": active_resource_types,
        "node_count": node_count,
        "edge_count": edge_count,
        "source_count": source_count,
        "global_saturation": round(global_saturation, 6),
        "valve_weights": {
            "pressure": round(_VALVE_ALPHA_PRESSURE, 6),
            "gravity": round(_VALVE_ALPHA_GRAVITY, 6),
            "affinity": round(_VALVE_ALPHA_AFFINITY, 6),
            "saturation": round(_VALVE_ALPHA_SATURATION, 6),
            "health": round(_VALVE_ALPHA_HEALTH, 6),
        },
        "radius_cost": round(radius_cost_value, 6),
        "cost_weights": {
            "latency": round(cost_w_l, 6),
            "congestion": round(cost_w_c, 6),
            "semantic": round(cost_w_s, 6),
        },
        "edge_cost_mean": round(edge_cost_mean, 6),
        "edge_cost_max": round(edge_cost_max, 6),
        "edge_health_mean": round(edge_health_mean, 6),
        "edge_health_max": round(edge_health_max, 6),
        "edge_health_min": round(edge_health_min, 6),
        "edge_saturation_mean": round(edge_saturation_mean, 6),
        "edge_saturation_max": round(edge_saturation_max, 6),
        "edge_affinity_mean": round(edge_affinity_mean, 6),
        "edge_upkeep_penalty_mean": round(edge_upkeep_penalty_mean, 6),
        "gravity_mean": round(gravity_mean, 6),
        "gravity_max": round(gravity_max, 6),
        "price_mean": round(price_mean, 6),
        "price_max": round(price_max, 6),
        "top_nodes": top_nodes,
    }


def compute_graph_route_step_native(
    *,
    graph_runtime: dict[str, Any] | None,
    particle_source_nodes: list[int],
    particle_resource_signature: list[dict[str, float]] | None = None,
    eta: float = 1.0,
    upsilon: float = 0.72,
    temperature: float = 0.35,
    step_seed: int = 0,
) -> dict[str, Any] | None:
    runtime = graph_runtime if isinstance(graph_runtime, dict) else {}
    node_count = max(0, int(_safe_float(runtime.get("node_count", 0), 0.0)))
    edge_count = max(0, int(_safe_float(runtime.get("edge_count", 0), 0.0)))
    if node_count <= 0 or edge_count <= 0 or not particle_source_nodes:
        return None

    edge_src = _coerce_list(runtime.get("edge_src_index", []))
    edge_dst = _coerce_list(runtime.get("edge_dst_index", []))
    edge_cost = _coerce_list(runtime.get("edge_cost", []))
    edge_health = _coerce_list(runtime.get("edge_health", []))
    edge_affinity = _coerce_list(runtime.get("edge_affinity", []))
    edge_saturation = _coerce_list(runtime.get("edge_saturation", []))
    edge_latency_component = _coerce_list(runtime.get("edge_latency_component", []))
    edge_congestion_component = _coerce_list(
        runtime.get("edge_congestion_component", [])
    )
    edge_semantic_component = _coerce_list(runtime.get("edge_semantic_component", []))
    edge_upkeep_penalty = _coerce_list(runtime.get("edge_upkeep_penalty", []))
    gravity = _coerce_list(runtime.get("gravity", []))
    node_price = _coerce_list(runtime.get("node_price", []))
    if (
        len(edge_src) != edge_count
        or len(edge_dst) != edge_count
        or len(edge_cost) != edge_count
        or len(gravity) != node_count
    ):
        return None
    if len(edge_health) != edge_count:
        edge_health = [1.0 for _ in range(edge_count)]
    if len(edge_affinity) != edge_count:
        edge_affinity = [0.5 for _ in range(edge_count)]
    if len(node_price) != node_count:
        node_price = [1.0 for _ in range(node_count)]

    weight_defaults = runtime.get("cost_weights", {})
    if not isinstance(weight_defaults, dict):
        weight_defaults = {}
    latency_default = max(
        0.0001,
        _safe_float(weight_defaults.get("latency", 1.0), 1.0),
    )
    saturation_default = _clamp01(
        _safe_float(runtime.get("global_saturation", 0.0), 0.0)
    )
    if len(edge_saturation) != edge_count:
        edge_saturation = [saturation_default for _ in range(edge_count)]
    if len(edge_latency_component) != edge_count:
        edge_latency_component = [latency_default for _ in range(edge_count)]
    if len(edge_congestion_component) != edge_count:
        edge_congestion_component = [0.0 for _ in range(edge_count)]
    if len(edge_semantic_component) != edge_count:
        edge_semantic_component = [0.0 for _ in range(edge_count)]
    if len(edge_upkeep_penalty) != edge_count:
        edge_upkeep_penalty = [
            max(
                0.0,
                _safe_float(edge_cost[idx], 0.0)
                - _safe_float(edge_latency_component[idx], 0.0)
                - _safe_float(edge_congestion_component[idx], 0.0)
                - _safe_float(edge_semantic_component[idx], 0.0),
            )
            for idx in range(edge_count)
        ]

    particle_count = len(particle_source_nodes)
    normalized_sources = [
        max(0, min(node_count - 1, int(index))) for index in particle_source_nodes
    ]

    resource_gravity_maps: dict[str, list[float]] = {}
    resource_maps_raw = runtime.get("resource_gravity_maps", {})
    if isinstance(resource_maps_raw, dict):
        for resource in _RESOURCE_TYPES:
            values = resource_maps_raw.get(resource)
            if not isinstance(values, list) or len(values) != node_count:
                continue
            resource_gravity_maps[resource] = [
                max(0.0, _safe_float(value, 0.0)) for value in values
            ]

    normalized_signatures: list[dict[str, float]] = []
    raw_signature_rows = (
        particle_resource_signature
        if isinstance(particle_resource_signature, list)
        else []
    )
    for index in range(particle_count):
        signature_row = (
            raw_signature_rows[index] if index < len(raw_signature_rows) else {}
        )
        if not isinstance(signature_row, dict):
            normalized_signatures.append({})
            continue
        mapped: dict[str, float] = {}
        for key, value in signature_row.items():
            resource = _canonical_resource_type(str(key))
            if not resource:
                continue
            mapped[resource] = mapped.get(resource, 0.0) + max(
                0.0, _safe_float(value, 0.0)
            )
        if sum(mapped.values()) <= 1e-8:
            normalized_signatures.append({})
            continue
        normalized_signatures.append(_normalize_resource_signature(mapped))

    resource_aware_routing = bool(resource_gravity_maps) and any(
        bool(signature) for signature in normalized_signatures
    )

    try:
        lib = _load_native_lib()
    except Exception:
        lib = None

    if (
        resource_aware_routing
        or lib is None
        or not hasattr(lib, "cdb_graph_route_step")
    ):
        next_nodes = []
        drift_scores = []
        route_probabilities = []
        drift_gravity_terms = []
        drift_cost_terms = []
        drift_gravity_delta = []
        drift_gravity_delta_scalar = []
        drift_cost_latency_terms = []
        drift_cost_congestion_terms = []
        drift_cost_semantic_terms = []
        drift_cost_upkeep_terms = []
        selected_edge_cost = []
        selected_edge_health = []
        selected_edge_affinity = []
        selected_edge_saturation = []
        selected_edge_upkeep_penalty = []
        valve_pressure_terms = []
        valve_gravity_terms = []
        valve_affinity_terms = []
        valve_saturation_terms = []
        valve_health_terms = []
        valve_score_proxies = []
        route_resource_focus = []
        route_resource_focus_weight = []
        route_resource_focus_delta = []
        route_resource_focus_contribution = []
        route_gravity_mode = []
        for particle_index, source in enumerate(normalized_sources):
            particle_signature = (
                normalized_signatures[particle_index]
                if particle_index < len(normalized_signatures)
                else {}
            )
            candidate_nodes: list[int] = []
            candidate_scores: list[float] = []
            candidate_edge_indices: list[int] = []
            best_node = source
            best_score = -10_000.0
            second_score = -10_000.0
            best_edge_index: int | None = None
            for edge_index, src in enumerate(edge_src):
                if int(src) != source:
                    continue
                dst = max(0, min(node_count - 1, int(edge_dst[edge_index])))
                score = _route_score_for_edge(
                    source=source,
                    target=dst,
                    edge_index=edge_index,
                    gravity=gravity,
                    edge_latency_component=edge_latency_component,
                    edge_congestion_component=edge_congestion_component,
                    edge_semantic_component=edge_semantic_component,
                    edge_upkeep_penalty=edge_upkeep_penalty,
                    resource_gravity_maps=resource_gravity_maps,
                    resource_signature=particle_signature,
                    eta=eta,
                    upsilon=upsilon,
                )
                candidate_nodes.append(dst)
                candidate_scores.append(score)
                candidate_edge_indices.append(edge_index)
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_node = dst
                    best_edge_index = edge_index
                elif score > second_score:
                    second_score = score

            if not candidate_nodes:
                if step_seed > 0 and node_count > 1:
                    spin = (
                        ((step_seed + 1) * 1103515245)
                        ^ ((particle_index + 1) * 2246822519)
                        ^ ((source + 1) * 3266489917)
                    ) & 0xFFFFFFFF
                    hop = 1 + int(spin % max(1, node_count - 1))
                    best_node = (source + hop) % node_count
                    best_score = 0.0
                    second_score = -0.6
                else:
                    best_node = source
                    best_score = 0.0
                    second_score = -0.4
                best_edge_index = None
            else:
                margin_seed = (
                    ((step_seed + 1) * 1103515245)
                    ^ ((particle_index + 1) * 2246822519)
                    ^ ((source + 1) * 3266489917)
                ) & 0xFFFFFFFF
                random_01 = float(margin_seed & 0xFFFFFF) / float(0xFFFFFF)
                margin = best_score - second_score if second_score > -9999.0 else 0.0
                explore_chance = 0.0
                if best_score < -0.12:
                    explore_chance += 0.28
                if margin < 0.1:
                    explore_chance += 0.18
                if (
                    step_seed > 0
                    and len(candidate_nodes) > 1
                    and random_01 < min(0.65, explore_chance)
                ):
                    rank = int((margin_seed >> 8) % len(candidate_nodes))
                    best_node = candidate_nodes[rank]
                    best_score = candidate_scores[rank]
                    best_edge_index = candidate_edge_indices[rank]

            best_terms = _route_terms_for_edge(
                source=source,
                target=best_node,
                edge_index=best_edge_index,
                gravity=gravity,
                node_price=node_price,
                edge_cost=edge_cost,
                edge_health=edge_health,
                edge_affinity=edge_affinity,
                edge_saturation=edge_saturation,
                edge_latency_component=edge_latency_component,
                edge_congestion_component=edge_congestion_component,
                edge_semantic_component=edge_semantic_component,
                edge_upkeep_penalty=edge_upkeep_penalty,
                resource_gravity_maps=resource_gravity_maps,
                resource_signature=particle_signature,
                eta=eta,
                upsilon=upsilon,
            )

            margin = best_score - second_score if second_score > -9999.0 else 0.0
            probability = _sigmoid(margin / max(0.05, _safe_float(temperature, 0.35)))
            next_nodes.append(best_node)
            drift_scores.append(math.tanh(best_score * 0.45))
            route_probabilities.append(_clamp01(probability))
            drift_gravity_terms.append(
                _safe_float(best_terms.get("drift_gravity_term", 0.0), 0.0)
            )
            drift_cost_terms.append(
                _safe_float(best_terms.get("drift_cost_term", 0.0), 0.0)
            )
            drift_gravity_delta.append(
                _safe_float(best_terms.get("drift_gravity_delta", 0.0), 0.0)
            )
            drift_gravity_delta_scalar.append(
                _safe_float(best_terms.get("drift_gravity_delta_scalar", 0.0), 0.0)
            )
            drift_cost_latency_terms.append(
                _safe_float(best_terms.get("drift_cost_latency_term", 0.0), 0.0)
            )
            drift_cost_congestion_terms.append(
                _safe_float(best_terms.get("drift_cost_congestion_term", 0.0), 0.0)
            )
            drift_cost_semantic_terms.append(
                _safe_float(best_terms.get("drift_cost_semantic_term", 0.0), 0.0)
            )
            drift_cost_upkeep_terms.append(
                _safe_float(best_terms.get("drift_cost_upkeep_term", 0.0), 0.0)
            )
            selected_edge_cost.append(
                max(0.0, _safe_float(best_terms.get("selected_edge_cost", 0.0), 0.0))
            )
            selected_edge_health.append(
                _clamp01(_safe_float(best_terms.get("selected_edge_health", 1.0), 1.0))
            )
            selected_edge_affinity.append(
                _clamp01(
                    _safe_float(best_terms.get("selected_edge_affinity", 0.5), 0.5)
                )
            )
            selected_edge_saturation.append(
                _clamp01(
                    _safe_float(best_terms.get("selected_edge_saturation", 0.0), 0.0)
                )
            )
            selected_edge_upkeep_penalty.append(
                max(
                    0.0,
                    _safe_float(
                        best_terms.get("selected_edge_upkeep_penalty", 0.0), 0.0
                    ),
                )
            )
            valve_pressure_terms.append(
                _safe_float(best_terms.get("valve_pressure_term", 0.0), 0.0)
            )
            valve_gravity_terms.append(
                _safe_float(best_terms.get("valve_gravity_term", 0.0), 0.0)
            )
            valve_affinity_terms.append(
                _safe_float(best_terms.get("valve_affinity_term", 0.0), 0.0)
            )
            valve_saturation_terms.append(
                _safe_float(best_terms.get("valve_saturation_term", 0.0), 0.0)
            )
            valve_health_terms.append(
                _safe_float(best_terms.get("valve_health_term", 0.0), 0.0)
            )
            valve_score_proxies.append(
                _safe_float(best_terms.get("valve_score_proxy", 0.0), 0.0)
            )
            route_resource_focus.append(str(best_terms.get("route_resource_focus", "")))
            route_resource_focus_weight.append(
                _clamp01(
                    _safe_float(best_terms.get("route_resource_focus_weight", 0.0), 0.0)
                )
            )
            route_resource_focus_delta.append(
                _safe_float(best_terms.get("route_resource_focus_delta", 0.0), 0.0)
            )
            route_resource_focus_contribution.append(
                _safe_float(
                    best_terms.get("route_resource_focus_contribution", 0.0), 0.0
                )
            )
            route_gravity_mode.append(
                str(best_terms.get("route_gravity_mode", "scalar-gravity"))
            )

        return {
            "next_node_index": next_nodes,
            "drift_score": drift_scores,
            "route_probability": route_probabilities,
            "drift_gravity_term": drift_gravity_terms,
            "drift_cost_term": drift_cost_terms,
            "drift_gravity_delta": drift_gravity_delta,
            "drift_gravity_delta_scalar": drift_gravity_delta_scalar,
            "drift_cost_latency_term": drift_cost_latency_terms,
            "drift_cost_congestion_term": drift_cost_congestion_terms,
            "drift_cost_semantic_term": drift_cost_semantic_terms,
            "drift_cost_upkeep_term": drift_cost_upkeep_terms,
            "selected_edge_cost": selected_edge_cost,
            "selected_edge_health": selected_edge_health,
            "selected_edge_affinity": selected_edge_affinity,
            "selected_edge_saturation": selected_edge_saturation,
            "selected_edge_upkeep_penalty": selected_edge_upkeep_penalty,
            "valve_pressure_term": valve_pressure_terms,
            "valve_gravity_term": valve_gravity_terms,
            "valve_affinity_term": valve_affinity_terms,
            "valve_saturation_term": valve_saturation_terms,
            "valve_health_term": valve_health_terms,
            "valve_score_proxy": valve_score_proxies,
            "route_resource_focus": route_resource_focus,
            "route_resource_focus_weight": route_resource_focus_weight,
            "route_resource_focus_delta": route_resource_focus_delta,
            "route_resource_focus_contribution": route_resource_focus_contribution,
            "route_gravity_mode": route_gravity_mode,
            "resource_routing_mode": "resource-signature"
            if resource_aware_routing
            else "scalar-gravity",
            "resource_types": list(resource_gravity_maps.keys()),
        }

    edge_src_arr = (ctypes.c_uint32 * edge_count)(*[int(value) for value in edge_src])
    edge_dst_arr = (ctypes.c_uint32 * edge_count)(*[int(value) for value in edge_dst])
    edge_cost_arr = (ctypes.c_float * edge_count)(
        *[_safe_float(value, 1.0) for value in edge_cost]
    )
    gravity_arr = (ctypes.c_float * node_count)(
        *[_safe_float(value, 0.0) for value in gravity]
    )
    source_arr = (ctypes.c_uint32 * particle_count)(
        *[int(value) for value in normalized_sources]
    )

    out_next = (ctypes.c_uint32 * particle_count)()
    out_drift = (ctypes.c_float * particle_count)()
    out_probability = (ctypes.c_float * particle_count)()

    try:
        status = int(
            lib.cdb_graph_route_step(
                ctypes.c_uint32(node_count),
                ctypes.c_uint32(edge_count),
                edge_src_arr,
                edge_dst_arr,
                edge_cost_arr,
                gravity_arr,
                source_arr,
                ctypes.c_uint32(particle_count),
                ctypes.c_float(max(0.01, _safe_float(eta, 1.0))),
                ctypes.c_float(max(0.01, _safe_float(upsilon, 0.72))),
                ctypes.c_float(max(0.05, _safe_float(temperature, 0.35))),
                ctypes.c_uint32(max(0, int(step_seed)) & 0xFFFFFFFF),
                out_next,
                out_drift,
                out_probability,
            )
        )
    except Exception:
        return None

    if status != 0:
        return None

    next_nodes = [
        max(0, min(node_count - 1, int(out_next[index])))
        for index in range(particle_count)
    ]
    drift_scores = [
        _safe_float(out_drift[index], 0.0) for index in range(particle_count)
    ]
    route_probabilities = [
        _clamp01(_safe_float(out_probability[index], 1.0))
        for index in range(particle_count)
    ]

    edge_lookup: dict[tuple[int, int], int] = {}
    for edge_index in range(edge_count):
        src = max(0, min(node_count - 1, int(edge_src[edge_index])))
        dst = max(0, min(node_count - 1, int(edge_dst[edge_index])))
        key = (src, dst)
        existing_index = edge_lookup.get(key)
        if existing_index is None:
            edge_lookup[key] = edge_index
            continue
        if _safe_float(edge_cost[edge_index], 1.0) < _safe_float(
            edge_cost[existing_index],
            1.0,
        ):
            edge_lookup[key] = edge_index

    drift_gravity_terms: list[float] = []
    drift_cost_terms: list[float] = []
    drift_gravity_delta: list[float] = []
    drift_gravity_delta_scalar: list[float] = []
    drift_cost_latency_terms: list[float] = []
    drift_cost_congestion_terms: list[float] = []
    drift_cost_semantic_terms: list[float] = []
    drift_cost_upkeep_terms: list[float] = []
    selected_edge_cost: list[float] = []
    selected_edge_health: list[float] = []
    selected_edge_affinity: list[float] = []
    selected_edge_saturation: list[float] = []
    selected_edge_upkeep_penalty: list[float] = []
    valve_pressure_terms: list[float] = []
    valve_gravity_terms: list[float] = []
    valve_affinity_terms: list[float] = []
    valve_saturation_terms: list[float] = []
    valve_health_terms: list[float] = []
    valve_score_proxies: list[float] = []
    route_resource_focus: list[str] = []
    route_resource_focus_weight: list[float] = []
    route_resource_focus_delta: list[float] = []
    route_resource_focus_contribution: list[float] = []
    route_gravity_mode: list[str] = []
    for index, source in enumerate(normalized_sources):
        target = next_nodes[index] if index < len(next_nodes) else source
        edge_index = edge_lookup.get((source, target))
        particle_signature = (
            normalized_signatures[index] if index < len(normalized_signatures) else {}
        )
        terms = _route_terms_for_edge(
            source=source,
            target=target,
            edge_index=edge_index,
            gravity=gravity,
            node_price=node_price,
            edge_cost=edge_cost,
            edge_health=edge_health,
            edge_affinity=edge_affinity,
            edge_saturation=edge_saturation,
            edge_latency_component=edge_latency_component,
            edge_congestion_component=edge_congestion_component,
            edge_semantic_component=edge_semantic_component,
            edge_upkeep_penalty=edge_upkeep_penalty,
            resource_gravity_maps=resource_gravity_maps,
            resource_signature=particle_signature,
            eta=eta,
            upsilon=upsilon,
        )
        drift_gravity_terms.append(
            _safe_float(terms.get("drift_gravity_term", 0.0), 0.0)
        )
        drift_cost_terms.append(_safe_float(terms.get("drift_cost_term", 0.0), 0.0))
        drift_gravity_delta.append(
            _safe_float(terms.get("drift_gravity_delta", 0.0), 0.0)
        )
        drift_gravity_delta_scalar.append(
            _safe_float(terms.get("drift_gravity_delta_scalar", 0.0), 0.0)
        )
        drift_cost_latency_terms.append(
            _safe_float(terms.get("drift_cost_latency_term", 0.0), 0.0)
        )
        drift_cost_congestion_terms.append(
            _safe_float(terms.get("drift_cost_congestion_term", 0.0), 0.0)
        )
        drift_cost_semantic_terms.append(
            _safe_float(terms.get("drift_cost_semantic_term", 0.0), 0.0)
        )
        drift_cost_upkeep_terms.append(
            _safe_float(terms.get("drift_cost_upkeep_term", 0.0), 0.0)
        )
        selected_edge_cost.append(
            max(0.0, _safe_float(terms.get("selected_edge_cost", 0.0), 0.0))
        )
        selected_edge_health.append(
            _clamp01(_safe_float(terms.get("selected_edge_health", 1.0), 1.0))
        )
        selected_edge_affinity.append(
            _clamp01(_safe_float(terms.get("selected_edge_affinity", 0.5), 0.5))
        )
        selected_edge_saturation.append(
            _clamp01(_safe_float(terms.get("selected_edge_saturation", 0.0), 0.0))
        )
        selected_edge_upkeep_penalty.append(
            max(0.0, _safe_float(terms.get("selected_edge_upkeep_penalty", 0.0), 0.0))
        )
        valve_pressure_terms.append(
            _safe_float(terms.get("valve_pressure_term", 0.0), 0.0)
        )
        valve_gravity_terms.append(
            _safe_float(terms.get("valve_gravity_term", 0.0), 0.0)
        )
        valve_affinity_terms.append(
            _safe_float(terms.get("valve_affinity_term", 0.0), 0.0)
        )
        valve_saturation_terms.append(
            _safe_float(terms.get("valve_saturation_term", 0.0), 0.0)
        )
        valve_health_terms.append(_safe_float(terms.get("valve_health_term", 0.0), 0.0))
        valve_score_proxies.append(
            _safe_float(terms.get("valve_score_proxy", 0.0), 0.0)
        )
        route_resource_focus.append(str(terms.get("route_resource_focus", "")))
        route_resource_focus_weight.append(
            _clamp01(_safe_float(terms.get("route_resource_focus_weight", 0.0), 0.0))
        )
        route_resource_focus_delta.append(
            _safe_float(terms.get("route_resource_focus_delta", 0.0), 0.0)
        )
        route_resource_focus_contribution.append(
            _safe_float(terms.get("route_resource_focus_contribution", 0.0), 0.0)
        )
        route_gravity_mode.append(
            str(terms.get("route_gravity_mode", "scalar-gravity"))
        )

    return {
        "next_node_index": next_nodes,
        "drift_score": drift_scores,
        "route_probability": route_probabilities,
        "drift_gravity_term": drift_gravity_terms,
        "drift_cost_term": drift_cost_terms,
        "drift_gravity_delta": drift_gravity_delta,
        "drift_gravity_delta_scalar": drift_gravity_delta_scalar,
        "drift_cost_latency_term": drift_cost_latency_terms,
        "drift_cost_congestion_term": drift_cost_congestion_terms,
        "drift_cost_semantic_term": drift_cost_semantic_terms,
        "drift_cost_upkeep_term": drift_cost_upkeep_terms,
        "selected_edge_cost": selected_edge_cost,
        "selected_edge_health": selected_edge_health,
        "selected_edge_affinity": selected_edge_affinity,
        "selected_edge_saturation": selected_edge_saturation,
        "selected_edge_upkeep_penalty": selected_edge_upkeep_penalty,
        "valve_pressure_term": valve_pressure_terms,
        "valve_gravity_term": valve_gravity_terms,
        "valve_affinity_term": valve_affinity_terms,
        "valve_saturation_term": valve_saturation_terms,
        "valve_health_term": valve_health_terms,
        "valve_score_proxy": valve_score_proxies,
        "route_resource_focus": route_resource_focus,
        "route_resource_focus_weight": route_resource_focus_weight,
        "route_resource_focus_delta": route_resource_focus_delta,
        "route_resource_focus_contribution": route_resource_focus_contribution,
        "route_gravity_mode": route_gravity_mode,
        "resource_routing_mode": "resource-signature"
        if resource_aware_routing
        else "scalar-gravity",
        "resource_types": list(resource_gravity_maps.keys()),
    }


def build_double_buffer_field_particles(
    *,
    file_graph: dict[str, Any] | None,
    presence_impacts: list[dict[str, Any]] | None,
    resource_heartbeat: dict[str, Any] | None,
    compute_jobs: list[dict[str, Any]] | None,
    queue_ratio: float,
    now: float,
    entity_manifest: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    presence_rows = presence_impacts if isinstance(presence_impacts, list) else []
    presence_ids, presence_layout = _presence_layout(
        presence_impacts=presence_rows,
        entity_manifest=entity_manifest,
    )

    file_nodes = (
        list((file_graph or {}).get("file_nodes", []))
        if isinstance(file_graph, dict)
        else []
    )
    file_node_position_by_id: dict[str, tuple[float, float]] = {}

    def _file_node_semantic_payload(row: dict[str, Any]) -> tuple[float, float]:
        text_chars = 0
        for key in (
            "summary",
            "text_excerpt",
            "name",
            "label",
            "title",
            "source_rel_path",
            "archive_rel_path",
            "archived_rel_path",
            "url",
        ):
            value = row.get(key)
            if isinstance(value, str):
                text_chars += len(value)
            elif isinstance(value, list):
                text_chars += sum(len(str(item)) for item in value)

        tags = row.get("tags", [])
        if isinstance(tags, list):
            text_chars += sum(len(str(tag)) for tag in tags)
        labels = row.get("labels", [])
        if isinstance(labels, list):
            text_chars += sum(len(str(label)) for label in labels)

        embed_layer_count = max(
            0.0, _safe_float(row.get("embed_layer_count", 0.0), 0.0)
        )
        embedding_links = row.get("embedding_links", [])
        link_count = len(embedding_links) if isinstance(embedding_links, list) else 0
        importance = _clamp01(_safe_float(row.get("importance", 0.24), 0.24))

        semantic_mass = max(
            0.4,
            min(
                12.0,
                0.44
                + (math.log1p(max(0, text_chars)) * 0.22)
                + (embed_layer_count * 0.13)
                + (float(link_count) * 0.06)
                + (importance * 1.6),
            ),
        )
        return float(max(0, text_chars)), float(semantic_mass)

    file_node_semantic_by_id: dict[str, tuple[float, float]] = {}
    for row in file_nodes:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        if not node_id:
            continue
        file_node_position_by_id[node_id] = (
            _clamp01(_safe_float(row.get("x", 0.5), 0.5)),
            _clamp01(_safe_float(row.get("y", 0.5), 0.5)),
        )
        file_node_semantic_by_id[node_id] = _file_node_semantic_payload(row)

    wallet_total_by_presence: dict[str, float] = {}
    for impact in presence_rows:
        if not isinstance(impact, dict):
            continue
        presence_id = str(impact.get("id", "")).strip()
        if not presence_id:
            continue
        wallet = impact.get("resource_wallet", {})
        wallet_total = 0.0
        if isinstance(wallet, dict):
            wallet_total = sum(
                max(0.0, _safe_float(value, 0.0)) for value in wallet.values()
            )
        wallet_total_by_presence[presence_id] = max(0.0, wallet_total)
    cpu_util = _safe_float(
        (
            (resource_heartbeat or {}).get("devices", {})
            if isinstance(resource_heartbeat, dict)
            else {}
        )
        .get("cpu", {})
        .get("utilization", 0.0),
        0.0,
    )

    cpu_daimoi_stop_percent = max(
        0.0,
        min(
            100.0,
            _safe_float(
                os.getenv("SIMULATION_CPU_DAIMOI_STOP_PERCENT", "50") or "50",
                50.0,
            ),
        ),
    )
    cpu_core_emitter_enabled = cpu_util < cpu_daimoi_stop_percent
    if not cpu_core_emitter_enabled:
        presence_ids = [
            pid for pid in presence_ids if not str(pid).startswith("presence.core.")
        ]

    if presence_ids:
        order_index = {
            presence_id: index for index, presence_id in enumerate(presence_ids)
        }
        presence_ids = sorted(
            presence_ids,
            key=lambda pid: (
                _presence_priority_rank(pid),
                order_index.get(pid, 0),
            ),
        )

    if not presence_ids:
        fallback_presence_id = "health_sentinel_cpu"
        presence_ids = [fallback_presence_id]
        if fallback_presence_id not in presence_layout:
            presence_layout[fallback_presence_id] = (0.5, 0.5, None)

    particles_per_presence = max(
        6,
        min(
            4096,
            int(
                _safe_float(
                    os.getenv("CDB_TARGET_PARTICLES_PER_PRESENCE", "10") or "10",
                    10.0,
                )
            ),
        ),
    )
    particles_file_sqrt_factor = max(
        0.0,
        min(
            4096.0,
            _safe_float(
                os.getenv("CDB_TARGET_FILE_SQRT_FACTOR", "6") or "6",
                6.0,
            ),
        ),
    )
    particles_base_offset = max(
        24,
        min(
            200000,
            int(
                _safe_float(
                    os.getenv("CDB_TARGET_BASE_OFFSET", "64") or "64",
                    64.0,
                )
            ),
        ),
    )

    target_count = (
        (max(1, len(presence_ids)) * particles_per_presence)
        + int((len(file_nodes) ** 0.5) * particles_file_sqrt_factor)
        + particles_base_offset
    )
    queue_clamped = _clamp01(_safe_float(queue_ratio, 0.0))
    if queue_clamped >= 0.9:
        target_count = int(target_count * 0.65)
    elif queue_clamped >= 0.75:
        target_count = int(target_count * 0.78)
    elif queue_clamped >= 0.55:
        target_count = int(target_count * 0.9)

    compute_count = int(len(compute_jobs) if isinstance(compute_jobs, list) else 0)
    if compute_count >= 96:
        target_count = int(target_count * 0.7)
    elif compute_count >= 64:
        target_count = int(target_count * 0.78)
    elif compute_count >= 32:
        target_count = int(target_count * 0.88)

    if cpu_util >= 95.0:
        target_count = int(target_count * 0.62)
    elif cpu_util >= 85.0:
        target_count = int(target_count * 0.74)
    elif cpu_util >= 75.0:
        target_count = int(target_count * 0.86)

    min_count_floor = max(
        64,
        int(_safe_float(os.getenv("CDB_TARGET_MIN_COUNT", "96") or "96", 96.0)),
    )
    max_count_floor = max(
        min_count_floor,
        int(_safe_float(os.getenv("CDB_TARGET_MAX_COUNT", "720") or "720", 720.0)),
    )
    min_count = max(
        min_count_floor, len(presence_ids) * max(4, particles_per_presence // 2)
    )
    target_count = max(min_count, min(max_count_floor, target_count))
    semantic_wallet_scale = max(
        1.0,
        _safe_float(os.getenv("CDB_SEMANTIC_WALLET_SCALE", "24") or "24", 24.0),
    )

    seed = _seed_from_layout(
        presence_ids=presence_ids,
        file_node_count=len(file_nodes),
    )

    # Try to inject field data before engine creation/snapshot if possible,
    # or rather update the engine.
    # Note: _get_engine returns a persistent singleton if parameters match.
    engine = _get_engine(count=target_count, seed=seed)

    anti_clump_runtime = getattr(engine, "_anti_clump_runtime", {})
    if not isinstance(anti_clump_runtime, dict):
        anti_clump_runtime = {}
    anti_clump_prev_drive = max(
        -1.0,
        min(1.0, _safe_float(anti_clump_runtime.get("drive", 0.0), 0.0)),
    )
    anti_clump_scale_map = _anti_clump_scales(anti_clump_prev_drive)
    anti_clump_spawn_scale = max(
        0.35,
        min(1.0, _safe_float(anti_clump_scale_map.get("spawn", 1.0), 1.0)),
    )
    anti_clump_anchor_scale = max(
        0.45,
        min(1.2, _safe_float(anti_clump_scale_map.get("anchor", 1.0), 1.0)),
    )
    anti_clump_semantic_scale = max(
        0.35,
        min(1.2, _safe_float(anti_clump_scale_map.get("semantic", 1.0), 1.0)),
    )
    anti_clump_edge_scale = max(
        0.4,
        min(1.2, _safe_float(anti_clump_scale_map.get("edge", 1.0), 1.0)),
    )
    anti_clump_tangent_scale_base = max(
        0.8,
        min(1.8, _safe_float(anti_clump_scale_map.get("tangent", 1.0), 1.0)),
    )
    anti_clump_friction_slip = max(
        0.8,
        min(1.24, _safe_float(anti_clump_scale_map.get("friction_slip", 1.0), 1.0)),
    )
    anti_clump_simplex_gain = max(
        0.72,
        min(2.2, _safe_float(anti_clump_scale_map.get("simplex_gain", 1.0), 1.0)),
    )
    anti_clump_simplex_scale = max(
        0.82,
        min(1.34, _safe_float(anti_clump_scale_map.get("simplex_scale", 1.0), 1.0)),
    )
    anti_clump_tangent_scale = anti_clump_tangent_scale_base
    graph_variability_score = 0.0
    graph_variability_raw_score = 0.0
    graph_variability_peak_score = 0.0
    graph_variability_mean_displacement = 0.0
    graph_variability_p90_displacement = 0.0
    graph_variability_active_share = 0.0
    graph_variability_shared_nodes = 0
    graph_variability_sampled_nodes = 0
    graph_noise_gain = 1.0
    graph_route_damp = 1.0

    # Set simulation flags from env
    # 1 = Spatial Collision (Particle-Particle)
    # 2 = Mean Field (Field-Particle Interaction/Drift)
    # 3 = Both
    sim_flags_env = int(os.getenv("CDB_SIM_FLAGS", "0"))
    set_flags = getattr(engine, "set_flags", None)
    if callable(set_flags):
        try:
            set_flags(sim_flags_env)
        except Exception:
            pass

    # Resolve embeddings for all particles using C inference runtime when available.
    # Python only receives 24d folded vectors from the runtime boundary.
    _prof_emb_start = time.perf_counter()

    nexus_stride_env = str(os.getenv("CDB_NEXUS_STRIDE", "") or "").strip()
    if nexus_stride_env:
        nexus_stride = int(_safe_float(nexus_stride_env, 11.0))
        if nexus_stride <= 0:
            nexus_stride = 11
    else:
        if file_nodes:
            target_nexus_count = max(
                24,
                min(len(file_nodes), max(24, target_count // 3)),
            )
        else:
            target_nexus_count = max(12, target_count // 8)
        nexus_stride = max(1, target_count // max(1, target_nexus_count))
    use_virtual_nexus = target_count >= 96

    nexus_slot_mask: list[bool] = [False for _ in range(target_count)]
    nexus_seed_node_ids: list[str] = ["" for _ in range(target_count)]

    flat_embeddings: list[float] = []
    for i in range(target_count):
        owner_id = i % max(1, len(presence_ids))
        presence_id = presence_ids[owner_id]

        is_nexus_slot = use_virtual_nexus and ((i % nexus_stride) == 0)
        nexus_slot_mask[i] = is_nexus_slot
        if is_nexus_slot:
            # Map this nexus particle to a file from .ημ (or the general file graph)
            # file_nodes are already sorted by recency (ingested_at descending).
            file_index = i // nexus_stride
            if file_index < len(file_nodes):
                file_node = file_nodes[file_index]
                file_node_id = str(file_node.get("id", "")).strip()
                if file_node_id:
                    nexus_seed_node_ids[i] = file_node_id
                name = str(file_node.get("name", "unknown"))
                # Use name and excerpt to create a semantically meaningful seed for the embedding.
                # The hardware runtime will generate a 24d vector that guides Semantic Gravity.
                excerpt = str(file_node.get("text_excerpt", ""))[:240]
                seed_text = f"NEXUS File {name} {excerpt}"
            else:
                seed_text = f"CDB Slot {i} Owner {presence_id} NEXUS-UNUSED"
        else:
            seed_text = f"CDB Slot {i} Owner {presence_id}"

        emb = list(_embed_seed_vector_24(seed_text))
        if len(emb) < 24:
            emb = emb + [0.0] * (24 - len(emb))
        elif len(emb) > 24:
            emb = emb[:24]
        flat_embeddings.extend(emb)
    _prof_emb_end = time.perf_counter()
    if os.getenv("SIM_PROFILE_INTERNAL") == "1":
        print(
            f"CDB PROFILE: embeddings={(_prof_emb_end - _prof_emb_start) * 1000:.2f}ms "
            f"count={target_count} hits={_EMBED_CACHE_HITS} misses={_EMBED_CACHE_MISSES}",
            flush=True,
        )
    if flat_embeddings:
        engine.update_embeddings(flat_embeddings)

    embedding_runtime_source = _EMBED_RUNTIME_SOURCE
    embedding_runtime_error = _EMBED_RUNTIME_ERROR
    embedding_runtime_cpu_fallback = bool(_EMBED_RUNTIME_CPU_FALLBACK)
    embedding_runtime_cpu_fallback_detail = str(
        _EMBED_RUNTIME_CPU_FALLBACK_DETAIL or ""
    )

    try:
        from .simulation import _NOOI_FIELD

        if _NOOI_FIELD:
            flat_field = []
            for layer in _NOOI_FIELD.layers:
                flat_field.extend(layer)
            engine.update_nooi(flat_field)
    except ImportError:
        pass

    (
        count,
        frame_id,
        force_frame,
        chaos_frame,
        semantic_frame,
        x_arr,
        y_arr,
        vx_arr,
        vy_arr,
        deflect_arr,
        message_arr,
        entropy_arr,
        owner_arr,
        flags_arr,
    ) = engine.snapshot()

    palette: tuple[tuple[float, float, float], ...] = (
        (0.89, 0.33, 0.24),
        (0.21, 0.72, 0.36),
        (0.24, 0.41, 0.91),
        (0.82, 0.67, 0.22),
        (0.62, 0.27, 0.85),
        (0.18, 0.75, 0.79),
    )

    graph_runtime = compute_graph_runtime_maps_native(
        file_graph=file_graph,
        presence_ids=presence_ids,
        presence_layout=presence_layout,
        presence_impacts=presence_rows,
        queue_ratio=queue_clamped,
        cpu_utilization=cpu_util,
    )
    graph_node_ids = (
        _coerce_list(graph_runtime.get("node_ids", []))
        if isinstance(graph_runtime, dict)
        else []
    )
    graph_node_index_by_id = {
        str(node_id): idx
        for idx, node_id in enumerate(graph_node_ids)
        if str(node_id).strip()
    }
    source_node_index_by_presence = (
        dict(graph_runtime.get("source_node_index_by_presence", {}))
        if isinstance(graph_runtime, dict)
        else {}
    )
    graph_distance = (
        _coerce_list(graph_runtime.get("min_distance", []))
        if isinstance(graph_runtime, dict)
        else []
    )
    graph_gravity = (
        _coerce_list(graph_runtime.get("gravity", []))
        if isinstance(graph_runtime, dict)
        else []
    )
    graph_node_price = (
        _coerce_list(graph_runtime.get("node_price", []))
        if isinstance(graph_runtime, dict)
        else []
    )
    graph_node_saturation = (
        _coerce_list(graph_runtime.get("node_saturation", []))
        if isinstance(graph_runtime, dict)
        else []
    )
    graph_node_positions = (
        _coerce_list(graph_runtime.get("node_positions", []))
        if isinstance(graph_runtime, dict)
        else []
    )
    graph_gravity_max = max(
        1e-6,
        max(
            (max(0.0, _safe_float(value, 0.0)) for value in graph_gravity), default=0.0
        ),
    )

    graph_variability_state = _graph_node_variability_update(
        engine,
        node_ids=graph_node_ids,
        node_positions=graph_node_positions,
    )
    graph_variability_score = _clamp01(
        _safe_float(graph_variability_state.get("score", 0.0), 0.0)
    )
    graph_variability_raw_score = _clamp01(
        _safe_float(
            graph_variability_state.get("raw_score", graph_variability_score), 0.0
        )
    )
    graph_variability_peak_score = _clamp01(
        _safe_float(
            graph_variability_state.get("peak_score", graph_variability_score), 0.0
        )
    )
    graph_variability_mean_displacement = max(
        0.0,
        _safe_float(graph_variability_state.get("mean_displacement", 0.0), 0.0),
    )
    graph_variability_p90_displacement = max(
        0.0,
        _safe_float(graph_variability_state.get("p90_displacement", 0.0), 0.0),
    )
    graph_variability_active_share = _clamp01(
        _safe_float(graph_variability_state.get("active_share", 0.0), 0.0)
    )
    graph_variability_shared_nodes = max(
        0,
        int(_safe_float(graph_variability_state.get("shared_nodes", 0), 0.0)),
    )
    graph_variability_sampled_nodes = max(
        0,
        int(_safe_float(graph_variability_state.get("sampled_nodes", 0), 0.0)),
    )
    graph_noise_gain = max(
        1.0,
        min(
            2.2,
            1.0 + (graph_variability_score * _CDB_GRAPH_VARIABILITY_NOISE_GAIN),
        ),
    )
    graph_route_damp = max(
        0.55,
        min(
            1.0,
            1.0 - (graph_variability_score * _CDB_GRAPH_VARIABILITY_ROUTE_DAMP),
        ),
    )
    anti_clump_tangent_scale = max(
        0.8,
        min(2.4, anti_clump_tangent_scale_base * graph_noise_gain),
    )

    presence_resource_signatures = _build_presence_resource_signatures(
        presence_ids=presence_ids,
        presence_impacts=presence_rows,
        queue_ratio=queue_clamped,
    )
    default_signature = _default_resource_signature(queue_clamped)

    particle_source_nodes: list[int] = [0 for _ in range(count)]
    particle_resource_signature: list[dict[str, float]] = [
        default_signature for _ in range(count)
    ]
    if graph_node_ids:
        source_count = len(graph_node_ids)
        for index in range(count):
            owner_id = int(owner_arr[index])
            presence_id = presence_ids[owner_id % len(presence_ids)]
            mapped_node_index = source_node_index_by_presence.get(
                presence_id,
                owner_id % source_count,
            )
            particle_source_nodes[index] = max(
                0,
                min(source_count - 1, int(mapped_node_index)),
            )
            particle_resource_signature[index] = presence_resource_signatures.get(
                presence_id,
                default_signature,
            )

    route_runtime = compute_graph_route_step_native(
        graph_runtime=graph_runtime,
        particle_source_nodes=particle_source_nodes,
        particle_resource_signature=particle_resource_signature,
        eta=1.0,
        upsilon=0.72,
        temperature=0.35,
        step_seed=int(frame_id),
    )
    route_indices = (
        _coerce_list(route_runtime.get("next_node_index", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_scores = (
        _coerce_list(route_runtime.get("drift_score", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_probabilities = (
        _coerce_list(route_runtime.get("route_probability", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_gravity_terms = (
        _coerce_list(route_runtime.get("drift_gravity_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_cost_terms = (
        _coerce_list(route_runtime.get("drift_cost_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_gravity_deltas = (
        _coerce_list(route_runtime.get("drift_gravity_delta", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_gravity_delta_scalar = (
        _coerce_list(route_runtime.get("drift_gravity_delta_scalar", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_edge_costs = (
        _coerce_list(route_runtime.get("selected_edge_cost", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_edge_health = (
        _coerce_list(route_runtime.get("selected_edge_health", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_cost_latency_terms = (
        _coerce_list(route_runtime.get("drift_cost_latency_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_cost_congestion_terms = (
        _coerce_list(route_runtime.get("drift_cost_congestion_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_cost_semantic_terms = (
        _coerce_list(route_runtime.get("drift_cost_semantic_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_cost_upkeep_terms = (
        _coerce_list(route_runtime.get("drift_cost_upkeep_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_edge_affinity = (
        _coerce_list(route_runtime.get("selected_edge_affinity", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_edge_saturation = (
        _coerce_list(route_runtime.get("selected_edge_saturation", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_edge_upkeep_penalty = (
        _coerce_list(route_runtime.get("selected_edge_upkeep_penalty", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_valve_pressure_terms = (
        _coerce_list(route_runtime.get("valve_pressure_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_valve_gravity_terms = (
        _coerce_list(route_runtime.get("valve_gravity_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_valve_affinity_terms = (
        _coerce_list(route_runtime.get("valve_affinity_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_valve_saturation_terms = (
        _coerce_list(route_runtime.get("valve_saturation_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_valve_health_terms = (
        _coerce_list(route_runtime.get("valve_health_term", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_valve_score_proxy = (
        _coerce_list(route_runtime.get("valve_score_proxy", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_resource_focus = (
        _coerce_list(route_runtime.get("route_resource_focus", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_resource_focus_weight = (
        _coerce_list(route_runtime.get("route_resource_focus_weight", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_resource_focus_delta = (
        _coerce_list(route_runtime.get("route_resource_focus_delta", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_resource_focus_contribution = (
        _coerce_list(route_runtime.get("route_resource_focus_contribution", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_gravity_mode = (
        _coerce_list(route_runtime.get("route_gravity_mode", []))
        if isinstance(route_runtime, dict)
        else []
    )
    route_resource_routing_mode = (
        str(route_runtime.get("resource_routing_mode", "scalar-gravity"))
        if isinstance(route_runtime, dict)
        else "scalar-gravity"
    )

    rows: list[dict[str, Any]] = []
    deflect_count = 0
    diffuse_count = 0
    entropy_total = 0.0
    message_total = 0.0
    drift_score_total = 0.0
    route_probability_total = 0.0
    route_switch_count = 0
    influence_power_total = 0.0
    drift_gravity_term_total = 0.0
    drift_cost_term_total = 0.0
    edge_health_total = 0.0
    edge_affinity_total = 0.0
    edge_saturation_total = 0.0
    drift_cost_upkeep_term_total = 0.0
    valve_score_proxy_total = 0.0
    route_resource_focus_weight_total = 0.0
    route_resource_focus_contribution_total = 0.0
    resource_route_count = 0
    virtual_nexus_count = 0
    for index in range(count):
        owner_id = int(owner_arr[index])
        flags = int(flags_arr[index])
        virtual_nexus = (
            count >= 64 and index < len(nexus_slot_mask) and nexus_slot_mask[index]
        )
        if virtual_nexus:
            virtual_nexus_count += 1
        is_nexus = ((flags & CDB_FLAG_NEXUS) != 0) or virtual_nexus
        is_chaos = ((flags & CDB_FLAG_CHAOS) != 0) and not is_nexus

        presence_id = presence_ids[owner_id % len(presence_ids)]
        anchor_x, anchor_y, anchor_hue = presence_layout.get(
            presence_id,
            (0.5, 0.5, None),
        )

        if isinstance(anchor_hue, (int, float)):
            hue_norm = (float(anchor_hue) % 360.0) / 360.0
            saturation = 0.56 if is_nexus else (0.86 if is_chaos else 0.72)
            value = 0.82 if is_nexus else (0.95 if is_chaos else 0.9)
            r, g, b = colorsys.hsv_to_rgb(hue_norm, saturation, value)
        else:
            r, g, b = palette[owner_id % len(palette)]

        raw_x = _clamp01(_safe_float(x_arr[index], 0.5))
        raw_y = _clamp01(_safe_float(y_arr[index], 0.5))
        vel_x = _safe_float(vx_arr[index], 0.0)
        vel_y = _safe_float(vy_arr[index], 0.0)

        anchor_gain = (
            1.16 if is_nexus else (1.48 if is_chaos else 1.42)
        ) * anti_clump_anchor_scale
        anchor_focus_x = _clamp01(0.5 + ((anchor_x - 0.5) * anchor_gain))
        anchor_focus_y = _clamp01(0.5 + ((anchor_y - 0.5) * anchor_gain))

        spread = (0.22 if is_nexus else (0.44 if is_chaos else 0.34)) * max(
            0.85, min(1.5, 1.0 + (max(0.0, anti_clump_prev_drive) * 0.45))
        )
        cluster_x = anchor_focus_x + ((raw_x - 0.5) * spread)
        cluster_y = anchor_focus_y + ((raw_y - 0.5) * spread)

        free_float_blend = max(
            0.08,
            min(
                0.76,
                (0.2 if is_nexus else (0.42 if is_chaos else 0.36))
                + ((anti_clump_tangent_scale - 1.0) * 0.08),
            ),
        )
        velocity_influence = (0.07 if is_nexus else (0.2 if is_chaos else 0.16)) * max(
            0.8,
            min(1.35, anti_clump_edge_scale),
        )
        phase = (
            (float(frame_id % 8192) * 0.009)
            + (float(index) * 0.131)
            + (float(owner_id) * 0.17)
        )
        orbit = 0.0031 if is_nexus else (0.009 if is_chaos else 0.006)
        x_norm = _clamp01(
            (cluster_x * (1.0 - free_float_blend))
            + (raw_x * free_float_blend)
            + (vel_x * velocity_influence)
            + (math.cos(phase) * orbit)
        )
        y_norm = _clamp01(
            (cluster_y * (1.0 - free_float_blend))
            + (raw_y * free_float_blend)
            + (vel_y * velocity_influence)
            + (math.sin(phase * 1.11) * orbit)
        )

        particle_mode = (
            "static-daimoi"
            if is_nexus
            else ("chaos-butterfly" if is_chaos else "neutral")
        )
        presence_role = (
            "nexus-passive" if is_nexus else ("chaos-noise" if is_chaos else "active")
        )
        size = 1.95 if is_nexus else (1.65 if is_chaos else 1.32)

        deflect_value = _clamp01(_safe_float(deflect_arr[index], 0.66))
        if is_nexus:
            deflect_value = 0.92

        package_entropy = round(
            _clamp01(_safe_float(entropy_arr[index], 0.41)),
            6,
        )
        message_probability = round(
            _clamp01(_safe_float(message_arr[index], 0.0 if is_nexus else 0.27)),
            6,
        )

        graph_node_id = ""
        route_node_id = ""
        graph_x = float("nan")
        graph_y = float("nan")
        route_x = float("nan")
        route_y = float("nan")
        graph_distance_cost = -1.0
        gravity_potential = 0.0
        local_price = 1.0
        node_saturation = queue_clamped
        drift_score = 0.0
        route_probability = 1.0
        drift_gravity_term = 0.0
        drift_cost_term = 0.0
        drift_gravity_delta_value = 0.0
        drift_gravity_delta_scalar_value = 0.0
        drift_cost_latency_term = 0.0
        drift_cost_congestion_term = 0.0
        drift_cost_semantic_term = 0.0
        drift_cost_upkeep_term = 0.0
        selected_edge_cost_value = 0.0
        selected_edge_health_value = 1.0
        selected_edge_affinity_value = 0.5
        selected_edge_saturation_value = 0.0
        selected_edge_upkeep_penalty_value = 0.0
        valve_pressure_term = 0.0
        valve_gravity_term = 0.0
        valve_affinity_term = 0.0
        valve_saturation_term = 0.0
        valve_health_term = 0.0
        valve_score_proxy = 0.0
        route_resource_focus_value = ""
        route_resource_focus_weight_value = 0.0
        route_resource_focus_delta_value = 0.0
        route_resource_focus_contribution_value = 0.0
        route_gravity_mode_value = "scalar-gravity"
        motion_vx = vel_x
        motion_vy = vel_y
        semantic_text_chars_value = 0.0
        semantic_mass_value = 0.0
        semantic_payload_signal_value = 0.0
        resource_wallet_total_value = max(
            0.0,
            _safe_float(wallet_total_by_presence.get(presence_id, 0.0), 0.0),
        )
        preferred_nexus_node_id = (
            nexus_seed_node_ids[index]
            if virtual_nexus and index < len(nexus_seed_node_ids)
            else ""
        )

        if graph_node_ids:
            mapped_node_index_value: Any = source_node_index_by_presence.get(
                presence_id,
                owner_id % len(graph_node_ids),
            )
            if preferred_nexus_node_id:
                mapped_node_index_value = graph_node_index_by_id.get(
                    preferred_nexus_node_id,
                    mapped_node_index_value,
                )
            mapped_node_index_fallback = owner_id % len(graph_node_ids)
            mapped_node_index_float = _safe_float(
                mapped_node_index_value,
                float(mapped_node_index_fallback),
            )
            mapped_node_index_int = int(mapped_node_index_float)
            mapped_node_index = max(
                0,
                min(
                    len(graph_node_ids) - 1,
                    mapped_node_index_int,
                ),
            )

            graph_node_id = str(graph_node_ids[mapped_node_index])
            if mapped_node_index < len(graph_distance):
                graph_distance_cost = _safe_float(
                    graph_distance[mapped_node_index], -1.0
                )
            if mapped_node_index < len(graph_gravity):
                gravity_potential = max(
                    0.0, _safe_float(graph_gravity[mapped_node_index], 0.0)
                )
            if mapped_node_index < len(graph_node_price):
                local_price = max(
                    0.0, _safe_float(graph_node_price[mapped_node_index], 1.0)
                )
            if mapped_node_index < len(graph_node_saturation):
                node_saturation = _clamp01(
                    _safe_float(graph_node_saturation[mapped_node_index], queue_clamped)
                )
            if mapped_node_index < len(graph_node_positions):
                graph_row = graph_node_positions[mapped_node_index]
                if isinstance(graph_row, (list, tuple)) and len(graph_row) >= 2:
                    graph_x = _clamp01(_safe_float(graph_row[0], x_norm))
                    graph_y = _clamp01(_safe_float(graph_row[1], y_norm))

            route_node_index = mapped_node_index
            if index < len(route_indices):
                route_node_index = max(
                    0, min(len(graph_node_ids) - 1, int(route_indices[index]))
                )
            if index < len(route_scores):
                drift_score = _safe_float(route_scores[index], 0.0)
            if index < len(route_probabilities):
                route_probability = _clamp01(
                    _safe_float(route_probabilities[index], 1.0)
                )
            if index < len(route_gravity_terms):
                drift_gravity_term = _safe_float(route_gravity_terms[index], 0.0)
            if index < len(route_cost_terms):
                drift_cost_term = _safe_float(route_cost_terms[index], 0.0)
            if index < len(route_gravity_deltas):
                drift_gravity_delta_value = _safe_float(
                    route_gravity_deltas[index], 0.0
                )
            if index < len(route_gravity_delta_scalar):
                drift_gravity_delta_scalar_value = _safe_float(
                    route_gravity_delta_scalar[index],
                    0.0,
                )
            if index < len(route_cost_latency_terms):
                drift_cost_latency_term = _safe_float(
                    route_cost_latency_terms[index],
                    0.0,
                )
            if index < len(route_cost_congestion_terms):
                drift_cost_congestion_term = _safe_float(
                    route_cost_congestion_terms[index],
                    0.0,
                )
            if index < len(route_cost_semantic_terms):
                drift_cost_semantic_term = _safe_float(
                    route_cost_semantic_terms[index],
                    0.0,
                )
            if index < len(route_cost_upkeep_terms):
                drift_cost_upkeep_term = _safe_float(
                    route_cost_upkeep_terms[index],
                    0.0,
                )
            if index < len(route_edge_costs):
                selected_edge_cost_value = max(
                    0.0, _safe_float(route_edge_costs[index], 0.0)
                )
            if index < len(route_edge_health):
                selected_edge_health_value = _clamp01(
                    _safe_float(route_edge_health[index], 1.0)
                )
            if index < len(route_edge_affinity):
                selected_edge_affinity_value = _clamp01(
                    _safe_float(route_edge_affinity[index], 0.5)
                )
            if index < len(route_edge_saturation):
                selected_edge_saturation_value = _clamp01(
                    _safe_float(route_edge_saturation[index], 0.0)
                )
            if index < len(route_edge_upkeep_penalty):
                selected_edge_upkeep_penalty_value = max(
                    0.0,
                    _safe_float(route_edge_upkeep_penalty[index], 0.0),
                )
            if index < len(route_valve_pressure_terms):
                valve_pressure_term = _safe_float(
                    route_valve_pressure_terms[index], 0.0
                )
            if index < len(route_valve_gravity_terms):
                valve_gravity_term = _safe_float(route_valve_gravity_terms[index], 0.0)
            if index < len(route_valve_affinity_terms):
                valve_affinity_term = _safe_float(
                    route_valve_affinity_terms[index], 0.0
                )
            if index < len(route_valve_saturation_terms):
                valve_saturation_term = _safe_float(
                    route_valve_saturation_terms[index],
                    0.0,
                )
            if index < len(route_valve_health_terms):
                valve_health_term = _safe_float(route_valve_health_terms[index], 0.0)
            if index < len(route_valve_score_proxy):
                valve_score_proxy = _safe_float(route_valve_score_proxy[index], 0.0)
            if index < len(route_resource_focus):
                route_resource_focus_value = str(route_resource_focus[index] or "")
            if index < len(route_resource_focus_weight):
                route_resource_focus_weight_value = _clamp01(
                    _safe_float(route_resource_focus_weight[index], 0.0)
                )
            if index < len(route_resource_focus_delta):
                route_resource_focus_delta_value = _safe_float(
                    route_resource_focus_delta[index],
                    0.0,
                )
            if index < len(route_resource_focus_contribution):
                route_resource_focus_contribution_value = _safe_float(
                    route_resource_focus_contribution[index],
                    0.0,
                )
            if index < len(route_gravity_mode):
                route_gravity_mode_value = (
                    str(route_gravity_mode[index] or "") or "scalar-gravity"
                )

            route_node_id = str(graph_node_ids[route_node_index])
            if route_node_id and graph_node_id and route_node_id != graph_node_id:
                route_switch_count += 1
            if route_node_index < len(graph_node_positions):
                route_row = graph_node_positions[route_node_index]
                if isinstance(route_row, (list, tuple)) and len(route_row) >= 2:
                    route_x = _clamp01(_safe_float(route_row[0], x_norm))
                    route_y = _clamp01(_safe_float(route_row[1], y_norm))
                    route_dx = route_x - x_norm
                    route_dy = route_y - y_norm
                    route_mix = (
                        _clamp01(
                            (route_probability * 0.24)
                            + (_clamp01((drift_score + 1.0) * 0.5) * 0.12)
                        )
                        * anti_clump_semantic_scale
                        * graph_route_damp
                    )
                    escape_gain = _clamp01(1.0 - route_probability)
                    roam_radius = (0.0016 if is_nexus else 0.003) + (
                        escape_gain * (0.006 if is_nexus else 0.009)
                    )
                    roam_radius *= anti_clump_tangent_scale
                    swirl_phase = (
                        phase * 1.9
                        + (route_probability * 3.1)
                        + (_safe_float(index, 0.0) * 0.17)
                    )
                    x_norm = _clamp01(
                        (x_norm * (1.0 - route_mix))
                        + (route_x * route_mix)
                        + (math.cos(swirl_phase) * roam_radius)
                    )
                    y_norm = _clamp01(
                        (y_norm * (1.0 - route_mix))
                        + (route_y * route_mix)
                        + (math.sin(swirl_phase * 1.07) * roam_radius)
                    )
                    route_motion_gain = (
                        (0.22 if is_nexus else 0.32)
                        + (route_probability * (0.16 if is_nexus else 0.2))
                    ) * anti_clump_edge_scale
                    motion_vx = vel_x + (route_dx * route_motion_gain)
                    motion_vy = vel_y + (route_dy * route_motion_gain)

        if is_nexus and preferred_nexus_node_id:
            graph_node_id = preferred_nexus_node_id
            fallback_pos = file_node_position_by_id.get(preferred_nexus_node_id)
            if isinstance(fallback_pos, tuple) and len(fallback_pos) == 2:
                fallback_x = _clamp01(_safe_float(fallback_pos[0], 0.5))
                fallback_y = _clamp01(_safe_float(fallback_pos[1], 0.5))
                if not (math.isfinite(graph_x) and math.isfinite(graph_y)):
                    graph_x = fallback_x
                    graph_y = fallback_y
            if not route_node_id:
                route_node_id = graph_node_id
            if not (math.isfinite(route_x) and math.isfinite(route_y)):
                route_x = graph_x
                route_y = graph_y

            if not is_nexus:
                gravity_signal = _clamp01(gravity_potential / graph_gravity_max)
                price_signal = _clamp01((local_price - 1.0) / 4.0)
                route_signal = _clamp01(route_probability)
                deflect_value = _clamp01(
                    deflect_value
                    + (price_signal * 0.16)
                    - (gravity_signal * 0.12)
                    - (route_signal * 0.06)
                )
                message_probability = round(
                    _clamp01(
                        message_probability
                        + (gravity_signal * 0.12)
                        - (price_signal * 0.08)
                        + (_clamp01(drift_score) * 0.06)
                        + ((1.0 - route_signal) * 0.04)
                    ),
                    6,
                )

        semantic_node_candidates = [route_node_id, graph_node_id]
        if preferred_nexus_node_id:
            semantic_node_candidates.append(preferred_nexus_node_id)
        for semantic_node_id in semantic_node_candidates:
            node_id = str(semantic_node_id or "").strip()
            if not node_id:
                continue
            semantic_payload = file_node_semantic_by_id.get(node_id)
            if not isinstance(semantic_payload, tuple) or len(semantic_payload) < 2:
                continue
            semantic_text_chars_value = max(
                semantic_text_chars_value,
                max(0.0, _safe_float(semantic_payload[0], 0.0)),
            )
            semantic_mass_value = max(
                semantic_mass_value,
                max(0.0, _safe_float(semantic_payload[1], 0.0)),
            )

        wallet_signal_value = _clamp01(
            resource_wallet_total_value / semantic_wallet_scale
        )
        semantic_mass_floor = (1.1 if is_nexus else 0.55) + (
            wallet_signal_value * (2.6 if is_nexus else 1.2)
        )
        semantic_mass_value = max(semantic_mass_value, semantic_mass_floor)
        if semantic_text_chars_value <= 0.0 and wallet_signal_value > 0.0:
            semantic_text_chars_value = math.log1p(resource_wallet_total_value) * 18.0
        semantic_payload_signal_value = _clamp01(
            (_clamp01(math.log1p(max(0.0, semantic_text_chars_value)) / 8.0) * 0.52)
            + (_clamp01(semantic_mass_value / 7.5) * 0.63)
            + (wallet_signal_value * 0.55)
        )

        mass_base = 1.6 if is_nexus else 1.0
        mass_value = max(
            mass_base,
            min(
                6.0 if is_nexus else 3.0,
                mass_base + (semantic_mass_value * (0.22 if is_nexus else 0.08)),
            ),
        )

        simplex_seed = (
            ((int(frame_id) + 1) * 1315423911)
            ^ ((index + 1) * 2654435761)
            ^ ((owner_id + 1) * 2246822519)
        ) & 0xFFFFFFFF
        simplex_amp = (
            (0.0009 if is_nexus else (0.0026 if is_chaos else 0.0018))
            * anti_clump_tangent_scale
            * anti_clump_simplex_gain
        )
        simplex_dx, simplex_dy = _simplex_motion_delta(
            x=x_norm,
            y=y_norm,
            now=now,
            seed=simplex_seed,
            amplitude=simplex_amp,
            space_scale=4.4 * anti_clump_simplex_scale,
            phase_scale=0.28 * anti_clump_simplex_scale,
        )
        x_norm = _clamp01(x_norm + simplex_dx)
        y_norm = _clamp01(y_norm + simplex_dy)
        motion_vx += simplex_dx * 0.92
        motion_vy += simplex_dy * 0.92
        if is_nexus:
            nexus_motion_damping = max(
                0.42,
                min(0.78, 0.62 - (semantic_payload_signal_value * 0.08)),
            )
            if anti_clump_friction_slip > 1.0:
                nexus_motion_damping = min(
                    0.9,
                    nexus_motion_damping * anti_clump_friction_slip,
                )
            motion_vx *= nexus_motion_damping
            motion_vy *= nexus_motion_damping
        else:
            motion_vx *= anti_clump_friction_slip
            motion_vy *= anti_clump_friction_slip

        gravity_signal = _clamp01(gravity_potential / graph_gravity_max)
        price_signal = _clamp01((local_price - 1.0) / 4.0)
        route_signal = _clamp01(route_probability)
        motion_signal = _clamp01(
            math.sqrt((motion_vx * motion_vx) + (motion_vy * motion_vy)) * 4.0
        )
        influence_power = _clamp01(
            0.24
            + (message_probability * 0.38)
            + (abs(drift_score) * 0.22)
            + (route_signal * 0.12)
            + (gravity_signal * 0.16)
            + (price_signal * 0.08)
            + (motion_signal * 0.12)
            + (0.16 if is_nexus else 0.0)
        )

        action_probabilities = {
            "deflect": round(deflect_value, 6),
            "diffuse": round(_clamp01(1.0 - deflect_value), 6),
        }

        if (not is_nexus) and anti_clump_spawn_scale < 0.999:
            keep_seed = (
                ((index + 1) * 1103515245)
                ^ ((int(frame_id) + 1) * 12345)
                ^ ((owner_id + 1) * 2654435761)
            ) & 0xFFFFFFFF
            keep_ratio = keep_seed / 4294967295.0
            if keep_ratio > anti_clump_spawn_scale:
                continue

        if action_probabilities["deflect"] >= action_probabilities["diffuse"]:
            deflect_count += 1
        else:
            diffuse_count += 1

        entropy_total += package_entropy
        message_total += message_probability
        drift_score_total += _safe_float(drift_score, 0.0)
        route_probability_total += _safe_float(route_probability, 1.0)
        influence_power_total += _safe_float(influence_power, 0.0)
        drift_gravity_term_total += _safe_float(drift_gravity_term, 0.0)
        drift_cost_term_total += _safe_float(drift_cost_term, 0.0)
        edge_health_total += _safe_float(selected_edge_health_value, 1.0)
        edge_affinity_total += _safe_float(selected_edge_affinity_value, 0.5)
        edge_saturation_total += _safe_float(selected_edge_saturation_value, 0.0)
        drift_cost_upkeep_term_total += _safe_float(drift_cost_upkeep_term, 0.0)
        valve_score_proxy_total += _safe_float(valve_score_proxy, 0.0)
        route_resource_focus_weight_total += _safe_float(
            route_resource_focus_weight_value,
            0.0,
        )
        route_resource_focus_contribution_total += abs(
            _safe_float(route_resource_focus_contribution_value, 0.0)
        )
        if route_gravity_mode_value == "resource-signature":
            resource_route_count += 1

        rows.append(
            {
                "id": f"cdb:{index}",
                "presence_id": presence_id,
                "presence_role": presence_role,
                "particle_mode": particle_mode,
                "x": round(x_norm, 5),
                "y": round(y_norm, 5),
                "size": round(size, 5),
                "r": round(r, 5),
                "g": round(g, 5),
                "b": round(b, 5),
                "record": "eta-mu.field-particles.cdb.v1",
                "schema_version": "daimoi.probabilistic.cdb.v1",
                "owner_presence_id": presence_id,
                "target_presence_id": "",
                "top_job": "observe",
                "package_entropy": package_entropy,
                "message_probability": message_probability,
                "semantic_text_chars": round(semantic_text_chars_value, 3),
                "semantic_mass": round(semantic_mass_value, 6),
                "resource_wallet_total": round(resource_wallet_total_value, 6),
                "mass": round(mass_value, 5),
                "radius": round(0.02 if is_nexus else 0.012, 5),
                "collision_count": 0,
                "job_probabilities": {
                    "observe": 0.54,
                    "compile": 0.19,
                    "route": 0.27,
                },
                "action_probabilities": action_probabilities,
                "behavior_actions": ["deflect", "diffuse"],
                "is_nexus": is_nexus,
                "is_static_daimoi": is_nexus,
                "source_node_id": (
                    graph_node_id
                    if graph_node_id
                    else (f"node:{owner_id}" if is_nexus else "")
                ),
                "graph_node_id": graph_node_id,
                "graph_x": round(graph_x, 5) if math.isfinite(graph_x) else None,
                "graph_y": round(graph_y, 5) if math.isfinite(graph_y) else None,
                "graph_distance_cost": round(graph_distance_cost, 6),
                "gravity_potential": round(gravity_potential, 6),
                "local_price": round(local_price, 6),
                "node_saturation": round(node_saturation, 6),
                "route_node_id": route_node_id,
                "route_x": round(route_x, 5) if math.isfinite(route_x) else None,
                "route_y": round(route_y, 5) if math.isfinite(route_y) else None,
                "drift_score": round(drift_score, 6),
                "drift_gravity_term": round(drift_gravity_term, 6),
                "drift_cost_term": round(drift_cost_term, 6),
                "drift_gravity_delta": round(drift_gravity_delta_value, 6),
                "drift_gravity_delta_scalar": round(
                    drift_gravity_delta_scalar_value,
                    6,
                ),
                "drift_cost_latency_term": round(drift_cost_latency_term, 6),
                "drift_cost_congestion_term": round(drift_cost_congestion_term, 6),
                "drift_cost_semantic_term": round(drift_cost_semantic_term, 6),
                "drift_cost_upkeep_term": round(drift_cost_upkeep_term, 6),
                "route_gravity_mode": route_gravity_mode_value,
                "route_resource_focus": route_resource_focus_value,
                "route_resource_focus_weight": round(
                    route_resource_focus_weight_value,
                    6,
                ),
                "route_resource_focus_delta": round(
                    route_resource_focus_delta_value, 6
                ),
                "route_resource_focus_contribution": round(
                    route_resource_focus_contribution_value,
                    6,
                ),
                "route_probability": round(route_probability, 6),
                "selected_edge_cost": round(selected_edge_cost_value, 6),
                "selected_edge_health": round(selected_edge_health_value, 6),
                "selected_edge_affinity": round(selected_edge_affinity_value, 6),
                "selected_edge_saturation": round(selected_edge_saturation_value, 6),
                "selected_edge_upkeep_penalty": round(
                    selected_edge_upkeep_penalty_value,
                    6,
                ),
                "valve_pressure_term": round(valve_pressure_term, 6),
                "valve_gravity_term": round(valve_gravity_term, 6),
                "valve_affinity_term": round(valve_affinity_term, 6),
                "valve_saturation_term": round(valve_saturation_term, 6),
                "valve_health_term": round(valve_health_term, 6),
                "valve_score_proxy": round(valve_score_proxy, 6),
                "influence_power": round(influence_power, 6),
                "vx": round(_safe_float(motion_vx, 0.0), 6),
                "vy": round(_safe_float(motion_vy, 0.0), 6),
            }
        )

    collision_count = 0
    anti_clump_particles = {
        str(row.get("id", f"cdb:{index}")): row
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    }
    anti_clump_prev_collisions = int(
        _safe_float(getattr(engine, "_anti_clump_last_collisions", 0), 0.0)
    )
    anti_clump_runtime = _anti_clump_controller_update(
        anti_clump_runtime,
        particles=anti_clump_particles,
        previous_collision_count=anti_clump_prev_collisions,
    )
    setattr(engine, "_anti_clump_runtime", dict(anti_clump_runtime))
    setattr(engine, "_anti_clump_last_collisions", int(collision_count))

    anti_clump_drive = max(
        -1.0,
        min(1.0, _safe_float(anti_clump_runtime.get("drive", 0.0), 0.0)),
    )
    anti_clump_scale_snapshot = _anti_clump_scales(anti_clump_drive)
    anti_clump_scale_summary = {
        "semantic": anti_clump_scale_snapshot.get("semantic", 1.0),
        "edge": anti_clump_scale_snapshot.get("edge", 1.0),
        "anchor": anti_clump_scale_snapshot.get("anchor", 1.0),
        "spawn": anti_clump_scale_snapshot.get("spawn", 1.0),
        "tangent": anti_clump_scale_snapshot.get("tangent", 1.0),
        "friction_slip": anti_clump_scale_snapshot.get("friction_slip", 1.0),
        "simplex_gain": anti_clump_scale_snapshot.get("simplex_gain", 1.0),
        "simplex_scale": anti_clump_scale_snapshot.get("simplex_scale", 1.0),
        "tangent_base": anti_clump_tangent_scale_base,
        "tangent_effective": anti_clump_tangent_scale,
        "noise_gain": graph_noise_gain,
        "route_damp": graph_route_damp,
    }
    anti_clump_summary = daimoi_observer_module.anti_clump_summary_from_snapshot(
        anti_clump_runtime,
        target=DAIMOI_ANTI_CLUMP_TARGET,
        drive=anti_clump_drive,
        scales=anti_clump_scale_summary,
        scale_order=(
            "semantic",
            "edge",
            "anchor",
            "spawn",
            "tangent",
            "friction_slip",
            "simplex_gain",
            "simplex_scale",
            "tangent_base",
            "tangent_effective",
            "noise_gain",
            "route_damp",
        ),
    )
    anti_clump_summary["graph_variability"] = {
        "score": round(_safe_float(graph_variability_score, 0.0), 6),
        "raw_score": round(_safe_float(graph_variability_raw_score, 0.0), 6),
        "peak_score": round(_safe_float(graph_variability_peak_score, 0.0), 6),
        "mean_displacement": round(
            _safe_float(graph_variability_mean_displacement, 0.0),
            6,
        ),
        "p90_displacement": round(
            _safe_float(graph_variability_p90_displacement, 0.0),
            6,
        ),
        "active_share": round(
            _clamp01(_safe_float(graph_variability_active_share, 0.0)),
            6,
        ),
        "shared_nodes": int(max(0, graph_variability_shared_nodes)),
        "sampled_nodes": int(max(0, graph_variability_sampled_nodes)),
    }

    active_count = len(rows)
    summary = {
        "record": "eta-mu.daimoi-probabilistic.cdb.v1",
        "schema_version": "daimoi.probabilistic.cdb.v1",
        "active": int(active_count),
        "spawned": 0,
        "collisions": int(collision_count),
        "deflects": int(deflect_count),
        "diffuses": int(diffuse_count),
        "handoffs": 0,
        "deliveries": 0,
        "job_triggers": {},
        "mean_package_entropy": round(
            (entropy_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_message_probability": round(
            (message_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_drift_score": round(
            (drift_score_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_route_probability": round(
            (route_probability_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_drift_gravity_term": round(
            (drift_gravity_term_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_drift_cost_term": round(
            (drift_cost_term_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_drift_cost_upkeep_term": round(
            (drift_cost_upkeep_term_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_selected_edge_health": round(
            (edge_health_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_selected_edge_affinity": round(
            (edge_affinity_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_selected_edge_saturation": round(
            (edge_saturation_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_valve_score_proxy": round(
            (valve_score_proxy_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "mean_route_resource_focus_weight": round(
            (route_resource_focus_weight_total / active_count)
            if active_count > 0
            else 0.0,
            6,
        ),
        "mean_route_resource_focus_contribution": round(
            (route_resource_focus_contribution_total / active_count)
            if active_count > 0
            else 0.0,
            6,
        ),
        "graph_node_variability_score": round(
            _safe_float(graph_variability_score, 0.0),
            6,
        ),
        "graph_node_variability_raw_score": round(
            _safe_float(graph_variability_raw_score, 0.0),
            6,
        ),
        "graph_node_variability_mean_displacement": round(
            _safe_float(graph_variability_mean_displacement, 0.0),
            6,
        ),
        "graph_node_variability_shared_nodes": int(
            max(0, graph_variability_shared_nodes)
        ),
        "graph_node_variability_sampled_nodes": int(
            max(0, graph_variability_sampled_nodes)
        ),
        "mean_influence_power": round(
            (influence_power_total / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "route_switch_ratio": round(
            (route_switch_count / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "resource_routing_mode": route_resource_routing_mode,
        "resource_route_ratio": round(
            (resource_route_count / active_count) if active_count > 0 else 0.0,
            6,
        ),
        "nexus_stride": int(max(1, nexus_stride)),
        "virtual_nexus_enabled": bool(use_virtual_nexus),
        "virtual_nexus_count": int(virtual_nexus_count),
        "target_count": int(target_count),
        "presence_count": int(len(presence_ids)),
        "presence_profile": str(
            os.getenv("SIMULATION_PRESENCE_PROFILE", "full") or "full"
        ),
        "cpu_utilization": round(_safe_float(cpu_util, 0.0), 2),
        "cpu_core_emitter_enabled": bool(cpu_core_emitter_enabled),
        "cpu_daimoi_stop_percent": round(cpu_daimoi_stop_percent, 2),
        "matrix_mean": {"ss": 0.0, "sc": 0.0, "cs": 0.0, "cc": 0.0},
        "behavior_defaults": ["deflect", "diffuse"],
        "clump_score": anti_clump_summary["clump_score"],
        "anti_clump_drive": anti_clump_summary["drive"],
        "snr": anti_clump_summary["snr"],
        "anti_clump": anti_clump_summary,
        "backend": "c-double-buffer",
        "embedding_runtime_source": str(embedding_runtime_source or "unknown"),
        "embedding_runtime_error": str(embedding_runtime_error or ""),
        "embedding_runtime_cpu_fallback": bool(embedding_runtime_cpu_fallback),
        "embedding_runtime_cpu_fallback_detail": str(
            embedding_runtime_cpu_fallback_detail or ""
        ),
        "frame_id": int(frame_id),
        "force_frame": int(force_frame),
        "chaos_frame": int(chaos_frame),
        "semantic_frame": int(semantic_frame),
        "double_buffer": True,
        "systems": ["force", "chaos", "semantic", "integrate"],
        "force_runtime": _resolve_force_runtime_config(particle_count=count),
        "resource_types": list(_RESOURCE_TYPES),
        "queue_ratio": round(_safe_float(queue_ratio, 0.0), 6),
        "compute_jobs": int(len(compute_jobs) if isinstance(compute_jobs, list) else 0),
    }

    if isinstance(graph_runtime, dict):
        summary["graph_runtime"] = {
            "record": str(graph_runtime.get("record", _CDB_GRAPH_RUNTIME_RECORD)),
            "schema_version": str(
                graph_runtime.get("schema_version", _CDB_GRAPH_RUNTIME_SCHEMA)
            ),
            "node_count": int(_safe_float(graph_runtime.get("node_count", 0), 0.0)),
            "edge_count": int(_safe_float(graph_runtime.get("edge_count", 0), 0.0)),
            "source_count": int(_safe_float(graph_runtime.get("source_count", 0), 0.0)),
            "presence_source_count": int(
                _safe_float(graph_runtime.get("presence_source_count", 0), 0.0)
            ),
            "presence_model": dict(graph_runtime.get("presence_model", {})),
            "source_profiles": [
                {
                    "presence_id": str(row.get("presence_id", "")),
                    "source_node_id": str(row.get("source_node_id", "")),
                    "mask": dict(row.get("mask", {})),
                    "influence": dict(row.get("influence", {})),
                    "need_scalar": round(
                        _safe_float(row.get("need_scalar", 0.0), 0.0),
                        6,
                    ),
                    "need_by_resource": {
                        str(resource): round(_safe_float(value, 0.0), 6)
                        for resource, value in (
                            row.get("need_by_resource", {})
                            if isinstance(row.get("need_by_resource", {}), dict)
                            else {}
                        ).items()
                        if str(resource).strip()
                    },
                    "need_model": {
                        "kind": str(
                            (
                                row.get("need_model", {})
                                if isinstance(row.get("need_model", {}), dict)
                                else {}
                            ).get("kind", "")
                        ),
                        "priority": round(
                            _safe_float(
                                (
                                    row.get("need_model", {})
                                    if isinstance(row.get("need_model", {}), dict)
                                    else {}
                                ).get("priority", 0.0),
                                0.0,
                            ),
                            6,
                        ),
                        "alpha": round(
                            _safe_float(
                                (
                                    row.get("need_model", {})
                                    if isinstance(row.get("need_model", {}), dict)
                                    else {}
                                ).get("alpha", 0.0),
                                0.0,
                            ),
                            6,
                        ),
                        "util_raw": {
                            str(resource): round(_safe_float(value, 0.0), 6)
                            for resource, value in (
                                (
                                    row.get("need_model", {})
                                    if isinstance(row.get("need_model", {}), dict)
                                    else {}
                                ).get("util_raw", {})
                                if isinstance(
                                    (
                                        row.get("need_model", {})
                                        if isinstance(row.get("need_model", {}), dict)
                                        else {}
                                    ).get("util_raw", {}),
                                    dict,
                                )
                                else {}
                            ).items()
                            if str(resource).strip()
                        },
                        "util_ema": {
                            str(resource): round(_safe_float(value, 0.0), 6)
                            for resource, value in (
                                (
                                    row.get("need_model", {})
                                    if isinstance(row.get("need_model", {}), dict)
                                    else {}
                                ).get("util_ema", {})
                                if isinstance(
                                    (
                                        row.get("need_model", {})
                                        if isinstance(row.get("need_model", {}), dict)
                                        else {}
                                    ).get("util_ema", {}),
                                    dict,
                                )
                                else {}
                            ).items()
                            if str(resource).strip()
                        },
                        "thresholds": {
                            str(resource): round(_safe_float(value, 0.0), 6)
                            for resource, value in (
                                (
                                    row.get("need_model", {})
                                    if isinstance(row.get("need_model", {}), dict)
                                    else {}
                                ).get("thresholds", {})
                                if isinstance(
                                    (
                                        row.get("need_model", {})
                                        if isinstance(row.get("need_model", {}), dict)
                                        else {}
                                    ).get("thresholds", {}),
                                    dict,
                                )
                                else {}
                            ).items()
                            if str(resource).strip()
                        },
                    },
                    "mass": round(_safe_float(row.get("mass", 0.0), 0.0), 6),
                }
                for row in list(graph_runtime.get("source_profiles", []))[:8]
                if isinstance(row, dict)
            ],
            "radius_cost": round(
                _safe_float(graph_runtime.get("radius_cost", 6.0), 6.0),
                6,
            ),
            "cost_weights": dict(graph_runtime.get("cost_weights", {})),
            "edge_cost_mean": round(
                _safe_float(graph_runtime.get("edge_cost_mean", 0.0), 0.0),
                6,
            ),
            "edge_cost_max": round(
                _safe_float(graph_runtime.get("edge_cost_max", 0.0), 0.0),
                6,
            ),
            "edge_health_mean": round(
                _safe_float(graph_runtime.get("edge_health_mean", 0.0), 0.0),
                6,
            ),
            "edge_health_max": round(
                _safe_float(graph_runtime.get("edge_health_max", 0.0), 0.0),
                6,
            ),
            "edge_health_min": round(
                _safe_float(graph_runtime.get("edge_health_min", 0.0), 0.0),
                6,
            ),
            "edge_saturation_mean": round(
                _safe_float(graph_runtime.get("edge_saturation_mean", 0.0), 0.0),
                6,
            ),
            "edge_saturation_max": round(
                _safe_float(graph_runtime.get("edge_saturation_max", 0.0), 0.0),
                6,
            ),
            "edge_affinity_mean": round(
                _safe_float(graph_runtime.get("edge_affinity_mean", 0.0), 0.0),
                6,
            ),
            "edge_upkeep_penalty_mean": round(
                _safe_float(graph_runtime.get("edge_upkeep_penalty_mean", 0.0), 0.0),
                6,
            ),
            "global_saturation": round(
                _safe_float(graph_runtime.get("global_saturation", 0.0), 0.0),
                6,
            ),
            "gravity_mean": round(
                _safe_float(graph_runtime.get("gravity_mean", 0.0), 0.0),
                6,
            ),
            "gravity_max": round(
                _safe_float(graph_runtime.get("gravity_max", 0.0), 0.0),
                6,
            ),
            "price_mean": round(
                _safe_float(graph_runtime.get("price_mean", 0.0), 0.0),
                6,
            ),
            "price_max": round(
                _safe_float(graph_runtime.get("price_max", 0.0), 0.0),
                6,
            ),
            "valve_weights": dict(graph_runtime.get("valve_weights", {})),
            "resource_types": [
                str(resource)
                for resource in list(graph_runtime.get("resource_types", []))
                if str(resource).strip()
            ],
            "resource_gravity_peaks": {
                str(resource): round(_safe_float(value, 0.0), 6)
                for resource, value in (
                    graph_runtime.get("resource_gravity_peaks", {})
                    if isinstance(graph_runtime.get("resource_gravity_peaks", {}), dict)
                    else {}
                ).items()
                if str(resource).strip()
            },
            "resource_gravity_peak_max": round(
                _safe_float(graph_runtime.get("resource_gravity_peak_max", 0.0), 0.0),
                6,
            ),
            "active_resource_types": [
                str(resource)
                for resource in list(graph_runtime.get("active_resource_types", []))
                if str(resource).strip()
            ],
            "top_nodes": list(graph_runtime.get("top_nodes", []))[:3],
        }
        graph_systems = [
            "edge-cost",
            "edge-upkeep-health",
            "bounded-gravity",
            "local-price",
            "valve-score-diagnostics",
        ]
        if isinstance(route_runtime, dict):
            graph_systems.append("graph-route-step")
            if route_resource_routing_mode == "resource-signature":
                graph_systems.append("resource-signature-routing")
        summary["graph_systems"] = graph_systems

    return rows, summary
