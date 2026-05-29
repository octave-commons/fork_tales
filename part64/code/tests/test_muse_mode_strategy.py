from __future__ import annotations

from code.world_web.muse_mode_strategy import (
    normalize_muse_runtime_mode,
    resolve_muse_reply_backend_mode,
    select_muse_surround_rows,
)


def test_normalize_muse_runtime_mode_falls_back_to_stochastic() -> None:
    assert normalize_muse_runtime_mode(" deterministic ") == "deterministic"
    assert normalize_muse_runtime_mode("weird-mode") == "stochastic"
    assert normalize_muse_runtime_mode(None) == "stochastic"


def test_resolve_muse_reply_backend_mode_maps_modes() -> None:
    assert resolve_muse_reply_backend_mode("deterministic") == "canonical"
    assert resolve_muse_reply_backend_mode("stochastic") == "llm"
    assert resolve_muse_reply_backend_mode("bogus") == "llm"


def test_select_muse_surround_rows_keeps_rank_for_deterministic_mode() -> None:
    scored_rows = [
        ({"id": "node:1"}, 0.9),
        ({"id": "node:2"}, 0.7),
        ({"id": "node:3"}, 0.4),
    ]
    selected = select_muse_surround_rows(
        scored_rows,
        mode="deterministic",
        seed="seed-a",
        tau=0.82,
    )
    assert [row[0]["id"] for row in selected] == ["node:1", "node:2", "node:3"]


def test_select_muse_surround_rows_is_seed_stable_for_stochastic_mode() -> None:
    scored_rows = [
        ({"id": "node:a"}, 1.4),
        ({"id": "node:b"}, 0.8),
        ({"id": "node:c"}, 0.3),
    ]
    one = select_muse_surround_rows(
        scored_rows,
        mode="stochastic",
        seed="seed-b",
        tau=0.82,
    )
    two = select_muse_surround_rows(
        scored_rows,
        mode="stochastic",
        seed="seed-b",
        tau=0.82,
    )
    ids_one = [row[0]["id"] for row in one]
    ids_two = [row[0]["id"] for row in two]
    assert ids_one == ids_two
    assert sorted(ids_one) == ["node:a", "node:b", "node:c"]
