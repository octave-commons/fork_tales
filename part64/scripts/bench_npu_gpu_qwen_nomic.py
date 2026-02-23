#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    _ai_module = importlib.import_module("code.world_web.ai")
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    CODE_ROOT = ROOT / "code"
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    _ai_module = importlib.import_module("world_web.ai")

_openvino_embed = getattr(_ai_module, "_openvino_embed")
_torch_embed = getattr(_ai_module, "_torch_embed")


EMBED_SENTENCES = [
    "Fork tax receipts align with witness continuity.",
    "Presence routing should prefer lower latency accelerators.",
    "Nomic embedding vectors support semantic clustering.",
    "Qwen vision responses should stay concise and deterministic.",
    "OpenVINO endpoint health determines NPU readiness.",
]


def _latency_stats(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"avg_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(values_ms)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return {
        "avg_ms": round(statistics.mean(ordered), 3),
        "p95_ms": round(float(ordered[idx]), 3),
    }


def _bench_openvino_embed(device: str, rounds: int) -> dict[str, Any]:
    os.environ["OPENVINO_EMBED_DEVICE"] = device
    os.environ["OPENVINO_EMBED_MODEL"] = "nomic-embed-text"
    latencies: list[float] = []
    ok = 0
    for idx in range(rounds):
        text = EMBED_SENTENCES[idx % len(EMBED_SENTENCES)]
        started = time.perf_counter()
        vec = _openvino_embed(text, model="nomic-embed-text", device=device)
        latencies.append((time.perf_counter() - started) * 1000.0)
        if vec:
            ok += 1
    stats = _latency_stats(latencies)
    return {
        "kind": "embedding",
        "backend": "openvino",
        "model": "nomic-embed-text",
        "device": device,
        "success": ok,
        "rounds": rounds,
        **stats,
    }


def _bench_torch_embed(rounds: int) -> dict[str, Any]:
    os.environ["TORCH_EMBED_MODEL"] = "nomic-ai/nomic-embed-text-v1.5"
    latencies: list[float] = []
    ok = 0
    for idx in range(rounds):
        text = EMBED_SENTENCES[idx % len(EMBED_SENTENCES)]
        started = time.perf_counter()
        vec = _torch_embed(text, model=None)
        latencies.append((time.perf_counter() - started) * 1000.0)
        if vec:
            ok += 1
    stats = _latency_stats(latencies)
    return {
        "kind": "embedding",
        "backend": "torch",
        "model": "nomic-embed-text",
        "device": "GPU",
        "success": ok,
        "rounds": rounds,
        **stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark local C embedding runtime on NPU vs GPU"
    )
    parser.add_argument("--rounds", type=int, default=12, help="Rounds per benchmark")
    parser.add_argument(
        "--output",
        default="runs/model-bench/npu-gpu-c-embed.latest.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    part_root = Path(".").resolve()

    os.environ["CDB_EMBED_IN_C"] = "1"
    os.environ["CDB_EMBED_REQUIRE_C"] = "1"

    rounds = max(1, int(args.rounds))
    results = [
        _bench_openvino_embed("NPU", rounds),
        _bench_torch_embed(rounds),
    ]

    payload = {
        "record": "eta_mu.benchmark.c_embed_npu_gpu.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (part_root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
