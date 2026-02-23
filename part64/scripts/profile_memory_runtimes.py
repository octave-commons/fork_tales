#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import psutil


PART_ROOT = Path(__file__).resolve().parents[1]
if str(PART_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.world_web import c_double_buffer_backend


@dataclass
class CallCounters:
    route_terms_calls: int = 0
    route_score_calls: int = 0


@dataclass
class ProfileResult:
    scenario: str
    iterations: int
    warmup: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float
    rss_delta_mb: float
    py_current_mb: float
    py_peak_mb: float
    terms_calls_per_iter: float
    score_calls_per_iter: float
    routing_mode: str


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * max(0.0, min(1.0, float(p)))
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] + ((values[upper] - values[lower]) * weight)


def _build_graph_runtime(
    *,
    node_count: int,
    fanout: int,
    seed: int,
    resource_aware: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_latency_component: list[float] = []
    edge_congestion_component: list[float] = []
    edge_semantic_component: list[float] = []
    edge_upkeep_penalty: list[float] = []
    edge_cost: list[float] = []
    edge_health: list[float] = []
    edge_affinity: list[float] = []
    edge_saturation: list[float] = []

    fanout_value = max(1, min(node_count - 1, int(fanout))) if node_count > 1 else 1
    for src in range(node_count):
        used: set[int] = set()
        for step in range(1, fanout_value + 1):
            dst = (src + step + ((src * 17 + step * 31) % node_count)) % node_count
            if dst == src:
                dst = (dst + 1) % node_count
            if dst in used:
                continue
            used.add(dst)

            edge_src.append(src)
            edge_dst.append(dst)

            latency = 0.35 + (rng.random() * 1.8)
            congestion = rng.random() * 0.9
            semantic = rng.random() * 0.7
            upkeep = rng.random() * 0.5
            edge_latency_component.append(latency)
            edge_congestion_component.append(congestion)
            edge_semantic_component.append(semantic)
            edge_upkeep_penalty.append(upkeep)
            edge_cost.append(latency + congestion + semantic + upkeep)
            edge_health.append(0.5 + (rng.random() * 0.5))
            edge_affinity.append(rng.random())
            edge_saturation.append(rng.random() * 0.8)

    gravity = [rng.random() * 2.0 for _ in range(node_count)]
    node_price = [0.4 + (rng.random() * 1.8) for _ in range(node_count)]
    runtime: dict[str, Any] = {
        "node_count": node_count,
        "edge_count": len(edge_src),
        "edge_src_index": edge_src,
        "edge_dst_index": edge_dst,
        "edge_cost": edge_cost,
        "edge_health": edge_health,
        "edge_affinity": edge_affinity,
        "edge_saturation": edge_saturation,
        "edge_latency_component": edge_latency_component,
        "edge_congestion_component": edge_congestion_component,
        "edge_semantic_component": edge_semantic_component,
        "edge_upkeep_penalty": edge_upkeep_penalty,
        "gravity": gravity,
        "node_price": node_price,
        "global_saturation": 0.42,
        "cost_weights": {
            "latency": 1.0,
            "congestion": 1.0,
            "semantic": 1.0,
        },
    }

    if resource_aware:
        resource_maps: dict[str, list[float]] = {}
        for resource in c_double_buffer_backend._RESOURCE_TYPES:
            center = rng.randrange(max(1, node_count))
            spread = max(8.0, float(node_count) * 0.16)
            values: list[float] = []
            for node_index in range(node_count):
                dist = abs(node_index - center)
                folded = min(dist, node_count - dist)
                profile = math.exp(-((folded * folded) / (2.0 * spread * spread)))
                values.append(max(0.0, profile + (rng.random() * 0.08)))
            resource_maps[resource] = values
        runtime["resource_gravity_maps"] = resource_maps

    return runtime


def _build_particle_inputs(
    *,
    node_count: int,
    particle_count: int,
    seed: int,
    resource_aware: bool,
) -> tuple[list[int], list[dict[str, float]] | None]:
    rng = random.Random(seed)
    sources = [rng.randrange(max(1, node_count)) for _ in range(max(1, particle_count))]
    if not resource_aware:
        return sources, None

    signatures: list[dict[str, float]] = []
    resource_types = list(c_double_buffer_backend._RESOURCE_TYPES)
    resource_count = max(1, len(resource_types))
    for idx in range(len(sources)):
        primary = resource_types[idx % resource_count]
        secondary = resource_types[(idx * 5 + 1) % resource_count]
        tertiary = resource_types[(idx * 7 + 2) % resource_count]
        signatures.append({primary: 0.72, secondary: 0.2, tertiary: 0.08})
    return sources, signatures


@contextmanager
def _instrument_backend(
    *,
    counters: CallCounters,
    force_python_fallback: bool,
) -> Iterator[None]:
    original_terms = c_double_buffer_backend._route_terms_for_edge
    original_score = c_double_buffer_backend._route_score_for_edge
    original_load_native = c_double_buffer_backend._load_native_lib

    def _counted_terms(**kwargs: Any) -> dict[str, Any]:
        counters.route_terms_calls += 1
        return original_terms(**kwargs)

    def _counted_score(**kwargs: Any) -> float:
        counters.route_score_calls += 1
        return original_score(**kwargs)

    c_double_buffer_backend._route_terms_for_edge = _counted_terms
    c_double_buffer_backend._route_score_for_edge = _counted_score
    if force_python_fallback:
        c_double_buffer_backend._load_native_lib = lambda: None

    try:
        yield
    finally:
        c_double_buffer_backend._route_terms_for_edge = original_terms
        c_double_buffer_backend._route_score_for_edge = original_score
        c_double_buffer_backend._load_native_lib = original_load_native


def _profile_loop(
    *,
    scenario: str,
    iterations: int,
    warmup: int,
    runner: Callable[[int], str],
    counters: CallCounters,
) -> ProfileResult:
    process = psutil.Process()
    routing_mode = "unknown"

    for index in range(max(0, warmup)):
        routing_mode = runner(index)

    gc.collect()
    rss_start = float(process.memory_info().rss)
    rss_peak = rss_start
    timed_terms_start = counters.route_terms_calls
    timed_score_start = counters.route_score_calls

    durations_ms: list[float] = []
    tracemalloc.start(25)
    try:
        for index in range(max(1, iterations)):
            started = time.perf_counter()
            routing_mode = runner(index + 10_000)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            durations_ms.append(elapsed_ms)

            current_rss = float(process.memory_info().rss)
            if current_rss > rss_peak:
                rss_peak = current_rss
        py_current, py_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    gc.collect()
    rss_end = float(process.memory_info().rss)
    ordered = sorted(durations_ms)
    iter_count = max(1, iterations)
    timed_terms_calls = max(0, counters.route_terms_calls - timed_terms_start)
    timed_score_calls = max(0, counters.route_score_calls - timed_score_start)

    return ProfileResult(
        scenario=scenario,
        iterations=iter_count,
        warmup=max(0, warmup),
        mean_ms=statistics.fmean(durations_ms),
        median_ms=statistics.median(durations_ms),
        p95_ms=_percentile(ordered, 0.95),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        rss_start_mb=(rss_start / (1024.0 * 1024.0)),
        rss_end_mb=(rss_end / (1024.0 * 1024.0)),
        rss_peak_mb=(rss_peak / (1024.0 * 1024.0)),
        rss_delta_mb=((rss_end - rss_start) / (1024.0 * 1024.0)),
        py_current_mb=(py_current / (1024.0 * 1024.0)),
        py_peak_mb=(py_peak / (1024.0 * 1024.0)),
        terms_calls_per_iter=(float(timed_terms_calls) / float(iter_count)),
        score_calls_per_iter=(float(timed_score_calls) / float(iter_count)),
        routing_mode=str(routing_mode or "unknown"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile memory and latency across frontend-facing Python/C runtime hot paths"
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--fanout", type=int, default=12)
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--engine-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    iterations = max(1, int(args.iterations))
    warmup = max(0, int(args.warmup))
    node_count = max(16, int(args.nodes))
    fanout = max(1, int(args.fanout))
    particle_count = max(1, int(args.particles))
    seed = int(args.seed)

    results: list[ProfileResult] = []

    # Scenario 1: Python resource-signature fallback.
    runtime_resource = _build_graph_runtime(
        node_count=node_count,
        fanout=fanout,
        seed=seed,
        resource_aware=True,
    )
    sources_resource, signatures_resource = _build_particle_inputs(
        node_count=node_count,
        particle_count=particle_count,
        seed=seed + 17,
        resource_aware=True,
    )
    counters = CallCounters()
    with _instrument_backend(counters=counters, force_python_fallback=False):

        def _run_resource(step_seed: int) -> str:
            route = c_double_buffer_backend.compute_graph_route_step_native(
                graph_runtime=runtime_resource,
                particle_source_nodes=sources_resource,
                particle_resource_signature=signatures_resource,
                step_seed=step_seed & 0xFFFFFFFF,
            )
            if not isinstance(route, dict):
                raise RuntimeError("resource-signature route returned invalid payload")
            return str(route.get("resource_routing_mode", "unknown"))

        results.append(
            _profile_loop(
                scenario="python-resource-signature-fallback",
                iterations=iterations,
                warmup=warmup,
                runner=_run_resource,
                counters=counters,
            )
        )

    # Scenario 2: Forced scalar fallback (Python path only).
    runtime_scalar = _build_graph_runtime(
        node_count=node_count,
        fanout=fanout,
        seed=seed + 101,
        resource_aware=False,
    )
    sources_scalar, signatures_scalar = _build_particle_inputs(
        node_count=node_count,
        particle_count=particle_count,
        seed=seed + 151,
        resource_aware=False,
    )
    counters = CallCounters()
    with _instrument_backend(counters=counters, force_python_fallback=True):

        def _run_scalar_forced(step_seed: int) -> str:
            route = c_double_buffer_backend.compute_graph_route_step_native(
                graph_runtime=runtime_scalar,
                particle_source_nodes=sources_scalar,
                particle_resource_signature=signatures_scalar,
                step_seed=step_seed & 0xFFFFFFFF,
            )
            if not isinstance(route, dict):
                raise RuntimeError("forced scalar fallback returned invalid payload")
            return str(route.get("resource_routing_mode", "unknown"))

        results.append(
            _profile_loop(
                scenario="python-scalar-fallback-forced",
                iterations=iterations,
                warmup=warmup,
                runner=_run_scalar_forced,
                counters=counters,
            )
        )

    # Scenario 3: Scalar auto (native C route step when available).
    counters = CallCounters()
    with _instrument_backend(counters=counters, force_python_fallback=False):

        def _run_scalar_auto(step_seed: int) -> str:
            route = c_double_buffer_backend.compute_graph_route_step_native(
                graph_runtime=runtime_scalar,
                particle_source_nodes=sources_scalar,
                particle_resource_signature=signatures_scalar,
                step_seed=step_seed & 0xFFFFFFFF,
            )
            if not isinstance(route, dict):
                raise RuntimeError("scalar auto route returned invalid payload")
            return str(route.get("resource_routing_mode", "unknown"))

        results.append(
            _profile_loop(
                scenario="scalar-auto-native-or-fallback",
                iterations=iterations,
                warmup=warmup,
                runner=_run_scalar_auto,
                counters=counters,
            )
        )

    # Scenario 4: C engine update loop (native engine buffers + snapshot).
    c_double_buffer_backend.shutdown_c_double_buffer_backend()
    engine_count = max(64, int(args.engine_count))
    engine = c_double_buffer_backend._get_engine(count=engine_count, seed=seed + 999)
    rng = random.Random(seed + 12345)
    nooi_data = [
        rng.random() * 0.5 for _ in range(c_double_buffer_backend._CDB_NOOI_FLOAT_COUNT)
    ]
    embedding_data = [
        ((rng.random() * 2.0) - 1.0)
        for _ in range(max(1, engine_count * c_double_buffer_backend._CDB_EMBED_DIM))
    ]
    counters = CallCounters()
    with _instrument_backend(counters=counters, force_python_fallback=False):

        def _run_engine(_: int) -> str:
            engine.update_nooi(nooi_data)
            engine.update_embeddings(embedding_data)
            engine.snapshot()
            return "c-engine-update"

        results.append(
            _profile_loop(
                scenario="c-engine-update-loop",
                iterations=iterations,
                warmup=warmup,
                runner=_run_engine,
                counters=counters,
            )
        )

    c_double_buffer_backend.shutdown_c_double_buffer_backend()

    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "results": [asdict(row) for row in results],
                },
                indent=2,
            )
        )
        return 0

    for row in results:
        print(
            f"{row.scenario}: "
            f"n={row.iterations} warmup={row.warmup} "
            f"mean={row.mean_ms:.2f}ms median={row.median_ms:.2f}ms p95={row.p95_ms:.2f}ms "
            f"min={row.min_ms:.2f}ms max={row.max_ms:.2f}ms | "
            f"rss start/end/peak={row.rss_start_mb:.2f}/{row.rss_end_mb:.2f}/{row.rss_peak_mb:.2f}MB "
            f"delta={row.rss_delta_mb:+.2f}MB | "
            f"py current/peak={row.py_current_mb:.2f}/{row.py_peak_mb:.2f}MB | "
            f"terms/iter={row.terms_calls_per_iter:.2f} score/iter={row.score_calls_per_iter:.2f} "
            f"mode={row.routing_mode}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
