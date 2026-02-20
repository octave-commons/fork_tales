#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import statistics
import time
from pathlib import Path
from typing import Any


PART_ROOT = Path(__file__).resolve().parents[1]
if str(PART_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.world_web.c_double_buffer_backend import (
    build_double_buffer_field_particles,
    shutdown_c_double_buffer_backend,
)
from code.world_web.daimoi_probabilistic import (
    build_probabilistic_daimoi_particles,
    reset_probabilistic_daimoi_state_for_tests,
)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] + ((values[upper] - values[lower]) * weight)


def _sample_inputs() -> dict[str, Any]:
    presence_ids = [
        "witness_thread",
        "keeper_of_receipts",
        "mage_of_receipts",
        "anchor_registry",
        "gates_of_truth",
        "receipt_river",
        "manifest_lith",
        "file_sentinel",
        "chaos_butterfly",
    ]
    presence_impacts = [
        {
            "id": pid,
            "affected_by": {
                "files": 0.24 + ((index % 4) * 0.17),
                "clicks": 0.19 + ((index % 5) * 0.13),
                "resource": 0.2,
            },
            "affects": {
                "world": 0.34 + ((index % 6) * 0.11),
                "ledger": 0.21 + ((index % 5) * 0.09),
            },
        }
        for index, pid in enumerate(presence_ids)
    ]
    file_nodes = [
        {
            "id": f"file:{index}",
            "x": ((index % 34) / 33.0),
            "y": (((index * 11) % 34) / 33.0),
            "importance": 0.22 + ((index % 8) * 0.08),
            "embedded_bonus": 0.2 if (index % 3 == 0) else 0.0,
            "dominant_presence": presence_ids[index % len(presence_ids)],
        }
        for index in range(220)
    ]
    return {
        "file_graph": {"file_nodes": file_nodes},
        "presence_impacts": presence_impacts,
        "resource_heartbeat": {
            "devices": {
                "cpu": {"utilization": 48.0},
                "gpu1": {"utilization": 31.0},
                "npu0": {"utilization": 14.0},
            }
        },
        "compute_jobs": [
            {
                "id": "compute:bench:1",
                "trigger": "benchmark",
                "emitter_presence_id": "health_sentinel_gpu1",
                "metric": "gpu1_util",
                "value": 0.31,
                "severity": "medium",
                "latency_ms": 42.0,
            }
        ],
        "queue_ratio": 0.18,
    }


def _run_benchmark(
    *,
    label: str,
    iterations: int,
    warmup: int,
    runner: Any,
) -> tuple[list[float], int]:
    for _ in range(max(0, warmup)):
        runner()

    durations_ms: list[float] = []
    last_count = 0
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        rows, _ = runner()
        durations_ms.append((time.perf_counter() - started) * 1000.0)
        last_count = len(rows) if isinstance(rows, list) else 0

    sorted_durations = sorted(durations_ms)
    print(
        f"{label}: n={len(durations_ms)} mean={statistics.fmean(durations_ms):.2f}ms "
        f"median={statistics.median(durations_ms):.2f}ms p95={_percentile(sorted_durations, 0.95):.2f}ms "
        f"min={min(durations_ms):.2f}ms max={max(durations_ms):.2f}ms rows={last_count}"
    )
    return durations_ms, last_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark C double-buffer simulation backend versus Python probabilistic backend"
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    sample = _sample_inputs()
    now_seed = 1_700_900_000.0

    reset_probabilistic_daimoi_state_for_tests()

    py_counter = {"i": 0}

    def run_python() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        py_counter["i"] += 1
        return build_probabilistic_daimoi_particles(
            file_graph=sample["file_graph"],
            presence_impacts=sample["presence_impacts"],
            resource_heartbeat=sample["resource_heartbeat"],
            compute_jobs=sample["compute_jobs"],
            queue_ratio=sample["queue_ratio"],
            now=now_seed + py_counter["i"],
        )

    c_counter = {"i": 0}

    def run_cdb() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        c_counter["i"] += 1
        return build_double_buffer_field_particles(
            file_graph=sample["file_graph"],
            presence_impacts=sample["presence_impacts"],
            resource_heartbeat=sample["resource_heartbeat"],
            compute_jobs=sample["compute_jobs"],
            queue_ratio=sample["queue_ratio"],
            now=now_seed + c_counter["i"],
        )

    py_durations, _ = _run_benchmark(
        label="python-probabilistic",
        iterations=max(1, args.iterations),
        warmup=max(0, args.warmup),
        runner=run_python,
    )
    c_durations, _ = _run_benchmark(
        label="c-double-buffer",
        iterations=max(1, args.iterations),
        warmup=max(0, args.warmup),
        runner=run_cdb,
    )

    py_mean = statistics.fmean(py_durations)
    c_mean = statistics.fmean(c_durations)
    delta_ms = py_mean - c_mean
    speedup = (py_mean / c_mean) if c_mean > 1e-9 else 0.0
    percent = ((delta_ms / py_mean) * 100.0) if py_mean > 1e-9 else 0.0
    direction = "faster" if delta_ms > 0.0 else "slower"
    print(
        f"delta: {abs(delta_ms):.2f}ms {direction} "
        f"({percent:+.2f}% vs python, speedup x{speedup:.3f})"
    )

    shutdown_c_double_buffer_backend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
