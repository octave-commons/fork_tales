#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen


@dataclass
class BenchStats:
    label: str
    request_count: int
    attempt_count: int
    failure_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    slice_latency_mean_ms: float
    slice_fallback_count: int
    slice_sources: dict[str, int]
    payload_bytes_mean: float


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


def _request_with_timing(
    *,
    url: str,
    timeout_seconds: float,
) -> tuple[float, dict[str, Any], int]:
    started = time.perf_counter()
    with urlopen(url, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected object payload from {url}")
    if data.get("ok") is False:
        raise RuntimeError(f"benchmark target returned non-ok payload for {url}")

    slice_meta = (
        (data.get("presence_dynamics") or {}).get("simulation_budget") or {}
    ).get("slice_offload") or {}
    if not isinstance(slice_meta, dict):
        slice_meta = {}
    return elapsed_ms, slice_meta, len(payload)


def _request_with_retry(
    *,
    url: str,
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
) -> tuple[float, dict[str, Any], int]:
    attempts = max(0, int(retries)) + 1
    last_error: Exception | None = None
    for attempt_index in range(attempts):
        try:
            return _request_with_timing(
                url=url,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            if attempt_index >= attempts - 1:
                break
            if retry_delay_seconds > 0.0:
                time.sleep(retry_delay_seconds)
    if last_error is None:
        raise RuntimeError(f"request failed without error for {url}")
    raise last_error


def summarize_endpoint(
    *,
    label: str,
    attempt_count: int,
    failure_count: int,
    durations_ms: list[float],
    slice_meta_rows: list[dict[str, Any]],
    payload_sizes: list[int],
) -> BenchStats:
    if not durations_ms:
        raise RuntimeError(f"no samples for {label}")

    sorted_durations = sorted(durations_ms)
    slice_latencies = [
        float(row.get("latency_ms", 0.0))
        for row in slice_meta_rows
        if isinstance(row, dict)
    ]
    fallback_count = sum(
        1
        for row in slice_meta_rows
        if isinstance(row, dict) and bool(row.get("fallback", False))
    )
    source_counts: dict[str, int] = {}
    for row in slice_meta_rows:
        source = str((row if isinstance(row, dict) else {}).get("source", "")).strip()
        if not source:
            source = "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

    return BenchStats(
        label=label,
        request_count=len(durations_ms),
        attempt_count=max(0, int(attempt_count)),
        failure_count=max(0, int(failure_count)),
        mean_ms=statistics.fmean(durations_ms),
        median_ms=statistics.median(durations_ms),
        p95_ms=_percentile(sorted_durations, 0.95),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        slice_latency_mean_ms=(
            statistics.fmean(slice_latencies) if slice_latencies else 0.0
        ),
        slice_fallback_count=fallback_count,
        slice_sources=source_counts,
        payload_bytes_mean=(statistics.fmean(payload_sizes) if payload_sizes else 0.0),
    )


def _collect_samples(
    *,
    url: str,
    timeout_seconds: float,
    request_count: int,
    retries: int,
    retry_delay_seconds: float,
    max_attempts: int,
) -> tuple[list[float], list[dict[str, Any]], list[int], int, int]:
    durations: list[float] = []
    metas: list[dict[str, Any]] = []
    sizes: list[int] = []
    failures = 0
    attempts = 0
    while len(durations) < request_count and attempts < max_attempts:
        attempts += 1
        try:
            elapsed, meta, size = _request_with_retry(
                url=url,
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_delay_seconds=retry_delay_seconds,
            )
            durations.append(elapsed)
            metas.append(meta)
            sizes.append(size)
        except Exception:  # pragma: no cover - network/runtime dependent
            failures += 1
            continue

    if not durations:
        raise RuntimeError(f"no successful samples for {url}")
    return durations, metas, sizes, attempts, failures


def _warmup_target(
    url: str,
    count: int,
    timeout: float,
    retries: int,
    delay: float,
) -> int:
    failures = 0
    for _ in range(count):
        try:
            _request_with_retry(
                url=url,
                timeout_seconds=timeout,
                retries=retries,
                retry_delay_seconds=delay,
            )
        except Exception:
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare /api/simulation latency with and without C worker offload"
    )
    parser.add_argument(
        "--baseline-url",
        default="http://127.0.0.1:18877/api/simulation",
        help="URL for baseline (no C worker) runtime",
    )
    parser.add_argument(
        "--offload-url",
        default="http://127.0.0.1:18879/api/simulation",
        help="URL for offloaded runtime (UDS/Redis+C worker)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Warmup requests per endpoint before timing",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=30,
        help="Timed request count per endpoint",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Per-request timeout seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count per request on transient failures",
    )
    parser.add_argument(
        "--retry-delay-ms",
        type=float,
        default=700.0,
        help="Delay between retries in milliseconds",
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=8,
        help="Max endpoint attempts as requests * multiplier",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    timeout_seconds = max(1.0, args.timeout)
    retries = max(0, int(args.retries))
    retry_delay_seconds = max(0.0, float(args.retry_delay_ms) / 1000.0)
    request_count = max(1, int(args.requests))
    max_attempts = max(
        request_count, request_count * max(1, int(args.max_attempts_multiplier))
    )

    warmup_failures: dict[str, int] = {
        "baseline": 0,
        "offload": 0,
    }
    warmup_count = max(0, args.warmup)
    if warmup_count > 0:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    _warmup_target,
                    str(args.baseline_url),
                    warmup_count,
                    timeout_seconds,
                    retries,
                    retry_delay_seconds,
                ): "baseline",
                executor.submit(
                    _warmup_target,
                    str(args.offload_url),
                    warmup_count,
                    timeout_seconds,
                    retries,
                    retry_delay_seconds,
                ): "offload",
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    warmup_failures[key] = future.result()
                except Exception:
                    warmup_failures[key] = warmup_count

    (
        baseline_durations,
        baseline_meta,
        baseline_sizes,
        baseline_attempts,
        baseline_failures,
    ) = _collect_samples(
        url=str(args.baseline_url),
        timeout_seconds=timeout_seconds,
        request_count=request_count,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        max_attempts=max_attempts,
    )
    (
        offload_durations,
        offload_meta,
        offload_sizes,
        offload_attempts,
        offload_failures,
    ) = _collect_samples(
        url=str(args.offload_url),
        timeout_seconds=timeout_seconds,
        request_count=request_count,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        max_attempts=max_attempts,
    )

    baseline = summarize_endpoint(
        label="baseline-local",
        attempt_count=baseline_attempts,
        failure_count=baseline_failures,
        durations_ms=baseline_durations,
        slice_meta_rows=baseline_meta,
        payload_sizes=baseline_sizes,
    )
    offload = summarize_endpoint(
        label="offload-c-worker",
        attempt_count=offload_attempts,
        failure_count=offload_failures,
        durations_ms=offload_durations,
        slice_meta_rows=offload_meta,
        payload_sizes=offload_sizes,
    )

    delta_ms = baseline.mean_ms - offload.mean_ms
    if offload.mean_ms > 1e-9:
        speedup = baseline.mean_ms / offload.mean_ms
    else:
        speedup = 0.0
    percent = (delta_ms / baseline.mean_ms) * 100.0 if baseline.mean_ms > 1e-9 else 0.0

    direction = "faster" if delta_ms > 0 else "slower"
    delta_text = (
        "delta: "
        f"{abs(delta_ms):.2f}ms {direction} "
        f"({percent:+.2f}% vs baseline, speedup x{speedup:.3f})"
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": True,
                    "baseline": baseline.__dict__,
                    "offload": offload.__dict__,
                    "warmup_failures": warmup_failures,
                    "delta_ms": delta_ms,
                    "speedup": speedup,
                    "percent": percent,
                    "direction": direction,
                    "delta_text": delta_text,
                },
                indent=2,
            )
        )
    else:
        print(
            f"{baseline.label}: n={baseline.request_count} "
            f"attempts={baseline.attempt_count} failures={baseline.failure_count} "
            f"mean={baseline.mean_ms:.2f}ms median={baseline.median_ms:.2f}ms "
            f"p95={baseline.p95_ms:.2f}ms min={baseline.min_ms:.2f}ms max={baseline.max_ms:.2f}ms "
            f"payload={baseline.payload_bytes_mean:.0f}B "
            f"slice_mean={baseline.slice_latency_mean_ms:.2f}ms fallbacks={baseline.slice_fallback_count}/{baseline.request_count} "
            f"sources={baseline.slice_sources}"
        )
        print(
            f"{offload.label}: n={offload.request_count} "
            f"attempts={offload.attempt_count} failures={offload.failure_count} "
            f"mean={offload.mean_ms:.2f}ms median={offload.median_ms:.2f}ms "
            f"p95={offload.p95_ms:.2f}ms min={offload.min_ms:.2f}ms max={offload.max_ms:.2f}ms "
            f"payload={offload.payload_bytes_mean:.0f}B "
            f"slice_mean={offload.slice_latency_mean_ms:.2f}ms fallbacks={offload.slice_fallback_count}/{offload.request_count} "
            f"sources={offload.slice_sources}"
        )
        if warmup_failures["baseline"] > 0 or warmup_failures["offload"] > 0:
            print(
                "warmup_failures: "
                f"baseline={warmup_failures['baseline']} "
                f"offload={warmup_failures['offload']}"
            )
        print(delta_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
