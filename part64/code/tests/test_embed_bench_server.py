from __future__ import annotations

import math

from code.world_web import embed_bench_server as bench


def test_latency_summary_percentiles_are_stable() -> None:
    summary = bench._latency_summary([10.0, 20.0, 30.0, 40.0, 50.0])
    assert summary["count"] == 5.0
    assert summary["p50_ms"] == 30.0
    assert summary["p90_ms"] == 46.0
    assert summary["p99_ms"] == 49.6
    assert summary["max_ms"] == 50.0


def test_state_tracks_window_and_success_counts() -> None:
    state = bench.BenchmarkState(
        bench.BenchConfig(window_size=3, interval_ms=0, target_dim=128)
    )
    state.start(reset=True)
    state.record_sample(
        ok=True,
        latency_ms=11.0,
        vector_dim=128,
        error="",
        sample_text="alpha",
    )
    state.record_sample(
        ok=True,
        latency_ms=13.0,
        vector_dim=128,
        error="",
        sample_text="beta",
    )
    state.record_sample(
        ok=True,
        latency_ms=17.0,
        vector_dim=128,
        error="",
        sample_text="gamma",
    )
    state.record_sample(
        ok=True,
        latency_ms=19.0,
        vector_dim=128,
        error="",
        sample_text="delta",
    )
    state.record_sample(
        ok=False,
        latency_ms=5.0,
        vector_dim=0,
        error="boom",
        sample_text="epsilon",
    )

    snapshot = state.snapshot(include_recent=True)
    stats = snapshot["stats"]
    latency = snapshot["latency"]

    assert stats["total_samples"] == 5
    assert stats["success_samples"] == 4
    assert stats["error_samples"] == 1
    assert stats["last_error"] == "boom"
    assert latency["count"] == 3
    assert latency["min_ms"] == 13.0
    assert latency["max_ms"] == 19.0


def test_run_embedding_once_applies_target_dim_and_normalizes(monkeypatch) -> None:
    def fake_embed(_text: str, **_kwargs):
        return [1.0, 2.0, 3.0, 4.0]

    monkeypatch.setattr(bench, "_embed_text", fake_embed)
    config = bench.BenchConfig(backend="openvino", target_dim=2)
    vector = bench.run_embedding_once("hello", config)

    assert isinstance(vector, list)
    assert len(vector) == 2
    magnitude = math.sqrt(sum(value * value for value in vector))
    assert abs(magnitude - 1.0) < 1e-6
