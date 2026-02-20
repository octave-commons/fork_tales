#!/usr/bin/env python3
"""
CDB Kernel Benchmark Script

Benchmarks individual C simulation kernels in isolation to track performance
regressions and validate optimizations.

Usage:
    python bench_cdb_kernels.py [--nodes N] [--fanout F] [--sources S] [--particles P]
                                [--iters I] [--warmup W] [--json]

Examples:
    # Default benchmark
    python bench_cdb_kernels.py

    # Large graph stress test
    python bench_cdb_kernels.py --nodes 420 --fanout 8 --particles 1600

    # CI mode with JSON output
    python bench_cdb_kernels.py --json --output baseline.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Resolve library path
_LIB_PATH = (
    Path(__file__).parent.parent
    / "code"
    / "world_web"
    / "native"
    / "libc_double_buffer_sim.so"
)


@dataclass
class KernelResult:
    name: str
    iterations: int
    warmup: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    config: dict[str, Any]


def _load_lib() -> ctypes.CDLL:
    if not _LIB_PATH.exists():
        raise FileNotFoundError(f"Native library not found: {_LIB_PATH}")
    lib = ctypes.CDLL(str(_LIB_PATH))

    # Set up argtypes for graph runtime maps
    lib.cdb_graph_runtime_maps.argtypes = [
        ctypes.c_uint32,  # node_count
        ctypes.c_uint32,  # edge_count
        ctypes.POINTER(ctypes.c_uint32),  # edge_src
        ctypes.POINTER(ctypes.c_uint32),  # edge_dst
        ctypes.POINTER(ctypes.c_float),  # edge_affinity
        ctypes.c_float,  # wl
        ctypes.c_float,  # wc
        ctypes.c_float,  # ws
        ctypes.c_float,  # global_sat
        ctypes.c_float,  # base_cost
        ctypes.POINTER(ctypes.c_uint32),  # source_nodes
        ctypes.POINTER(ctypes.c_float),  # source_mass
        ctypes.POINTER(ctypes.c_float),  # source_need
        ctypes.c_uint32,  # source_count
        ctypes.c_float,  # bounded_radius
        ctypes.c_float,  # grav_const
        ctypes.c_float,  # grav_eps
        ctypes.POINTER(ctypes.c_float),  # out_min_dist
        ctypes.POINTER(ctypes.c_float),  # out_gravity
        ctypes.POINTER(ctypes.c_float),  # out_edge_cost
        ctypes.POINTER(ctypes.c_float),  # out_node_saturation
        ctypes.POINTER(ctypes.c_float),  # out_node_price
    ]
    lib.cdb_graph_runtime_maps.restype = ctypes.c_int

    # Set up argtypes for route step
    lib.cdb_graph_route_step.argtypes = [
        ctypes.c_uint32,  # node_count
        ctypes.c_uint32,  # edge_count
        ctypes.POINTER(ctypes.c_uint32),  # edge_src
        ctypes.POINTER(ctypes.c_uint32),  # edge_dst
        ctypes.POINTER(ctypes.c_float),  # edge_cost
        ctypes.POINTER(ctypes.c_float),  # node_gravity
        ctypes.POINTER(ctypes.c_uint32),  # particle_current_node
        ctypes.c_uint32,  # particle_count
        ctypes.c_float,  # eta
        ctypes.c_float,  # upsilon
        ctypes.c_float,  # temperature
        ctypes.c_uint32,  # step_seed
        ctypes.POINTER(ctypes.c_uint32),  # out_next_node
        ctypes.POINTER(ctypes.c_float),  # out_drift_score
        ctypes.POINTER(ctypes.c_float),  # out_route_probability
    ]
    lib.cdb_graph_route_step.restype = ctypes.c_int

    # Set up argtypes for CSR builder
    lib.cdb_build_csr_edges.argtypes = [
        ctypes.c_uint32,  # node_count
        ctypes.c_uint32,  # edge_count
        ctypes.POINTER(ctypes.c_uint32),  # edge_src
        ctypes.POINTER(ctypes.c_uint32),  # edge_dst
        ctypes.POINTER(ctypes.c_uint32),  # csr_node_offsets
        ctypes.POINTER(ctypes.c_uint32),  # csr_edge_indices
    ]
    lib.cdb_build_csr_edges.restype = ctypes.c_int

    # Set up argtypes for CSR route step
    lib.cdb_graph_route_step_csr.argtypes = [
        ctypes.c_uint32,  # node_count
        ctypes.c_uint32,  # edge_count
        ctypes.POINTER(ctypes.c_uint32),  # edge_src
        ctypes.POINTER(ctypes.c_uint32),  # edge_dst
        ctypes.POINTER(ctypes.c_float),  # edge_cost
        ctypes.POINTER(ctypes.c_float),  # node_gravity
        ctypes.POINTER(ctypes.c_uint32),  # csr_node_offsets
        ctypes.POINTER(ctypes.c_uint32),  # csr_edge_indices
        ctypes.POINTER(ctypes.c_uint32),  # particle_current_node
        ctypes.c_uint32,  # particle_count
        ctypes.c_float,  # eta
        ctypes.c_float,  # upsilon
        ctypes.c_float,  # temperature
        ctypes.c_uint32,  # step_seed
        ctypes.POINTER(ctypes.c_uint32),  # out_next_node
        ctypes.POINTER(ctypes.c_float),  # out_drift_score
        ctypes.POINTER(ctypes.c_float),  # out_route_probability
    ]
    lib.cdb_graph_route_step_csr.restype = ctypes.c_int

    return lib


def _build_graph(
    node_count: int, fanout: int, seed: int = 42
) -> tuple[list[int], list[int], list[float]]:
    """Build a deterministic regular graph for benchmarking."""
    random.seed(seed)
    edge_src = []
    edge_dst = []
    edge_aff = []

    for src in range(node_count):
        for k in range(1, fanout + 1):
            dst = (src + k) % node_count
            edge_src.append(src)
            edge_dst.append(dst)
            edge_aff.append(0.5 + random.random() * 0.3)

    return edge_src, edge_dst, edge_aff


def _percentile(data: list[float], p: float) -> float:
    """Calculate percentile without numpy dependency."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def bench_graph_maps(
    lib: ctypes.CDLL,
    *,
    node_count: int,
    edge_src: list[int],
    edge_dst: list[int],
    edge_aff: list[float],
    source_count: int,
    iters: int,
    warmup: int,
) -> KernelResult:
    """Benchmark cdb_graph_runtime_maps kernel."""
    edge_count = len(edge_src)

    # Build sources
    source_nodes = [(i * 17) % node_count for i in range(source_count)]
    source_mass = [1.0 + (i % 5) * 0.1 for i in range(source_count)]
    source_need = [0.2 + (i % 7) * 0.05 for i in range(source_count)]

    # Convert to ctypes arrays
    edge_src_arr = (ctypes.c_uint32 * edge_count)(*edge_src)
    edge_dst_arr = (ctypes.c_uint32 * edge_count)(*edge_dst)
    edge_aff_arr = (ctypes.c_float * edge_count)(*edge_aff)
    source_nodes_arr = (ctypes.c_uint32 * source_count)(*source_nodes)
    source_mass_arr = (ctypes.c_float * source_count)(*source_mass)
    source_need_arr = (ctypes.c_float * source_count)(*source_need)

    # Output arrays
    out_min = (ctypes.c_float * node_count)()
    out_grav = (ctypes.c_float * node_count)()
    out_edge = (ctypes.c_float * edge_count)()
    out_sat = (ctypes.c_float * node_count)()
    out_price = (ctypes.c_float * node_count)()

    # Warmup
    for _ in range(warmup):
        lib.cdb_graph_runtime_maps(
            ctypes.c_uint32(node_count),
            ctypes.c_uint32(edge_count),
            edge_src_arr,
            edge_dst_arr,
            edge_aff_arr,
            ctypes.c_float(0.2),
            ctypes.c_float(0.45),
            ctypes.c_float(1.0),
            ctypes.c_float(2.0),
            ctypes.c_float(1.0),
            source_nodes_arr,
            source_mass_arr,
            source_need_arr,
            ctypes.c_uint32(source_count),
            ctypes.c_float(6.0),
            ctypes.c_float(1.0),
            ctypes.c_float(0.001),
            out_min,
            out_grav,
            out_edge,
            out_sat,
            out_price,
        )

    # Timed iterations
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        lib.cdb_graph_runtime_maps(
            ctypes.c_uint32(node_count),
            ctypes.c_uint32(edge_count),
            edge_src_arr,
            edge_dst_arr,
            edge_aff_arr,
            ctypes.c_float(0.2),
            ctypes.c_float(0.45),
            ctypes.c_float(1.0),
            ctypes.c_float(2.0),
            ctypes.c_float(1.0),
            source_nodes_arr,
            source_mass_arr,
            source_need_arr,
            ctypes.c_uint32(source_count),
            ctypes.c_float(6.0),
            ctypes.c_float(1.0),
            ctypes.c_float(0.001),
            out_min,
            out_grav,
            out_edge,
            out_sat,
            out_price,
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return KernelResult(
        name="cdb_graph_runtime_maps",
        iterations=iters,
        warmup=warmup,
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        p95_ms=_percentile(times, 95),
        min_ms=min(times),
        max_ms=max(times),
        stddev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        config={
            "node_count": node_count,
            "edge_count": edge_count,
            "source_count": source_count,
        },
    )


def bench_graph_route(
    lib: ctypes.CDLL,
    *,
    node_count: int,
    edge_src: list[int],
    edge_dst: list[int],
    gravity: list[float],
    particle_count: int,
    iters: int,
    warmup: int,
) -> KernelResult:
    """Benchmark cdb_graph_route_step kernel."""
    edge_count = len(edge_src)

    # Build particles
    particle_sources = [(i * 19) % node_count for i in range(particle_count)]

    # Build edge costs (use gravity-derived)
    edge_cost = [1.0] * edge_count

    # Convert to ctypes arrays
    edge_src_arr = (ctypes.c_uint32 * edge_count)(*edge_src)
    edge_dst_arr = (ctypes.c_uint32 * edge_count)(*edge_dst)
    edge_cost_arr = (ctypes.c_float * edge_count)(*edge_cost)
    gravity_arr = (ctypes.c_float * node_count)(*gravity)
    particle_sources_arr = (ctypes.c_uint32 * particle_count)(*particle_sources)

    # Output arrays
    out_next = (ctypes.c_uint32 * particle_count)()
    out_drift = (ctypes.c_float * particle_count)()
    out_prob = (ctypes.c_float * particle_count)()

    # Warmup
    for _ in range(warmup):
        lib.cdb_graph_route_step(
            ctypes.c_uint32(node_count),
            ctypes.c_uint32(edge_count),
            edge_src_arr,
            edge_dst_arr,
            edge_cost_arr,
            gravity_arr,
            particle_sources_arr,
            ctypes.c_uint32(particle_count),
            ctypes.c_float(1.0),
            ctypes.c_float(0.72),
            ctypes.c_float(0.35),
            ctypes.c_uint32(12345),
            out_next,
            out_drift,
            out_prob,
        )

    # Timed iterations
    times = []
    for seed in range(iters):
        t0 = time.perf_counter()
        lib.cdb_graph_route_step(
            ctypes.c_uint32(node_count),
            ctypes.c_uint32(edge_count),
            edge_src_arr,
            edge_dst_arr,
            edge_cost_arr,
            gravity_arr,
            particle_sources_arr,
            ctypes.c_uint32(particle_count),
            ctypes.c_float(1.0),
            ctypes.c_float(0.72),
            ctypes.c_float(0.35),
            ctypes.c_uint32(seed + 1000),
            out_next,
            out_drift,
            out_prob,
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return KernelResult(
        name="cdb_graph_route_step",
        iterations=iters,
        warmup=warmup,
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        p95_ms=_percentile(times, 95),
        min_ms=min(times),
        max_ms=max(times),
        stddev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        config={
            "node_count": node_count,
            "edge_count": edge_count,
            "particle_count": particle_count,
        },
    )


def bench_graph_route_csr(
    lib: ctypes.CDLL,
    *,
    node_count: int,
    edge_src: list[int],
    edge_dst: list[int],
    gravity: list[float],
    particle_count: int,
    iters: int,
    warmup: int,
) -> KernelResult:
    """Benchmark cdb_graph_route_step_csr kernel (CSR-optimized)."""
    edge_count = len(edge_src)

    # Build particles
    particle_sources = [(i * 19) % node_count for i in range(particle_count)]

    # Build edge costs
    edge_cost = [1.0] * edge_count

    # Convert to ctypes arrays
    edge_src_arr = (ctypes.c_uint32 * edge_count)(*edge_src)
    edge_dst_arr = (ctypes.c_uint32 * edge_count)(*edge_dst)
    edge_cost_arr = (ctypes.c_float * edge_count)(*edge_cost)
    gravity_arr = (ctypes.c_float * node_count)(*gravity)
    particle_sources_arr = (ctypes.c_uint32 * particle_count)(*particle_sources)

    # Build CSR structures
    csr_offsets = (ctypes.c_uint32 * (node_count + 1))()
    csr_indices = (ctypes.c_uint32 * edge_count)()
    rc = lib.cdb_build_csr_edges(
        ctypes.c_uint32(node_count),
        ctypes.c_uint32(edge_count),
        edge_src_arr,
        edge_dst_arr,
        csr_offsets,
        csr_indices,
    )
    if rc != 0:
        raise RuntimeError("cdb_build_csr_edges failed")

    # Output arrays
    out_next = (ctypes.c_uint32 * particle_count)()
    out_drift = (ctypes.c_float * particle_count)()
    out_prob = (ctypes.c_float * particle_count)()

    # Warmup
    for _ in range(warmup):
        lib.cdb_graph_route_step_csr(
            ctypes.c_uint32(node_count),
            ctypes.c_uint32(edge_count),
            edge_src_arr,
            edge_dst_arr,
            edge_cost_arr,
            gravity_arr,
            csr_offsets,
            csr_indices,
            particle_sources_arr,
            ctypes.c_uint32(particle_count),
            ctypes.c_float(1.0),
            ctypes.c_float(0.72),
            ctypes.c_float(0.35),
            ctypes.c_uint32(12345),
            out_next,
            out_drift,
            out_prob,
        )

    # Timed iterations
    times = []
    for seed in range(iters):
        t0 = time.perf_counter()
        lib.cdb_graph_route_step_csr(
            ctypes.c_uint32(node_count),
            ctypes.c_uint32(edge_count),
            edge_src_arr,
            edge_dst_arr,
            edge_cost_arr,
            gravity_arr,
            csr_offsets,
            csr_indices,
            particle_sources_arr,
            ctypes.c_uint32(particle_count),
            ctypes.c_float(1.0),
            ctypes.c_float(0.72),
            ctypes.c_float(0.35),
            ctypes.c_uint32(seed + 1000),
            out_next,
            out_drift,
            out_prob,
        )
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return KernelResult(
        name="cdb_graph_route_step_csr",
        iterations=iters,
        warmup=warmup,
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        p95_ms=_percentile(times, 95),
        min_ms=min(times),
        max_ms=max(times),
        stddev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        config={
            "node_count": node_count,
            "edge_count": edge_count,
            "particle_count": particle_count,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CDB simulation kernels")
    parser.add_argument("--nodes", type=int, default=220, help="Graph node count")
    parser.add_argument("--fanout", type=int, default=6, help="Edge fanout per node")
    parser.add_argument("--sources", type=int, default=12, help="Gravity source count")
    parser.add_argument(
        "--particles", type=int, default=800, help="Particle count for routing"
    )
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, help="Output file path (for JSON mode)")
    args = parser.parse_args()

    lib = _load_lib()

    # Build graph
    edge_src, edge_dst, edge_aff = _build_graph(args.nodes, args.fanout)
    edge_count = len(edge_src)

    # Run gravity maps first to get gravity values
    maps_result = bench_graph_maps(
        lib,
        node_count=args.nodes,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_aff=edge_aff,
        source_count=args.sources,
        iters=args.iters,
        warmup=args.warmup,
    )

    # Use gravity output for route benchmark
    gravity = [0.1 + (i % 17) * 0.03 for i in range(args.nodes)]

    route_result = bench_graph_route(
        lib,
        node_count=args.nodes,
        edge_src=edge_src,
        edge_dst=edge_dst,
        gravity=gravity,
        particle_count=args.particles,
        iters=args.iters,
        warmup=args.warmup,
    )

    # Run CSR version for comparison
    route_csr_result = bench_graph_route_csr(
        lib,
        node_count=args.nodes,
        edge_src=edge_src,
        edge_dst=edge_dst,
        gravity=gravity,
        particle_count=args.particles,
        iters=args.iters,
        warmup=args.warmup,
    )

    if args.json:
        output = {
            "config": {
                "nodes": args.nodes,
                "fanout": args.fanout,
                "sources": args.sources,
                "particles": args.particles,
                "iters": args.iters,
                "warmup": args.warmup,
            },
            "results": [
                asdict(maps_result),
                asdict(route_result),
                asdict(route_csr_result),
            ],
        }
        json_str = json.dumps(output, indent=2)
        if args.output:
            Path(args.output).write_text(json_str)
        else:
            print(json_str)
    else:
        print(f"CDB Kernel Benchmark Results")
        print(f"============================")
        print(
            f"Config: nodes={args.nodes} fanout={args.fanout} sources={args.sources} particles={args.particles}"
        )
        print(f"Iters: {args.iters} warmup: {args.warmup}")
        print()
        print(f"cdb_graph_runtime_maps:")
        print(f"  mean:   {maps_result.mean_ms:.4f} ms")
        print(f"  median: {maps_result.median_ms:.4f} ms")
        print(f"  p95:    {maps_result.p95_ms:.4f} ms")
        print(f"  min:    {maps_result.min_ms:.4f} ms")
        print(f"  max:    {maps_result.max_ms:.4f} ms")
        print(f"  stddev: {maps_result.stddev_ms:.4f} ms")
        print()
        print(f"cdb_graph_route_step (original):")
        print(f"  mean:   {route_result.mean_ms:.4f} ms")
        print(f"  median: {route_result.median_ms:.4f} ms")
        print(f"  p95:    {route_result.p95_ms:.4f} ms")
        print(f"  min:    {route_result.min_ms:.4f} ms")
        print(f"  max:    {route_result.max_ms:.4f} ms")
        print(f"  stddev: {route_result.stddev_ms:.4f} ms")
        print()
        print(f"cdb_graph_route_step_csr (optimized):")
        print(f"  mean:   {route_csr_result.mean_ms:.4f} ms")
        print(f"  median: {route_csr_result.median_ms:.4f} ms")
        print(f"  p95:    {route_csr_result.p95_ms:.4f} ms")
        print(f"  min:    {route_csr_result.min_ms:.4f} ms")
        print(f"  max:    {route_csr_result.max_ms:.4f} ms")
        print(f"  stddev: {route_csr_result.stddev_ms:.4f} ms")
        print()
        # Show speedup
        speedup = (
            route_result.mean_ms / route_csr_result.mean_ms
            if route_csr_result.mean_ms > 0
            else 0
        )
        improvement = (
            (
                (route_result.mean_ms - route_csr_result.mean_ms)
                / route_result.mean_ms
                * 100
            )
            if route_result.mean_ms > 0
            else 0
        )
        print(f"CSR speedup: {speedup:.2f}x ({improvement:.1f}% faster)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
