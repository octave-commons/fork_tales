from __future__ import annotations

import threading

from code.world_web.nooi import NooiField
from code.world_web.simulation_nooi_runtime_utils import (
    apply_nooi_from_particles,
    build_reset_nooi_runtime_state,
    maybe_seed_random_nooi_field_vectors,
    nooi_outcome_from_particle,
    record_daimoi_motion_trail,
)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _clamp01(value: object) -> float:
    numeric = _safe_float(value, 0.0)
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def test_build_reset_nooi_runtime_state_resets_structures() -> None:
    state = build_reset_nooi_runtime_state(nooi_field_factory=NooiField)
    assert isinstance(state["nooi_field"], NooiField)
    assert state["motion_history"] == {}
    assert state["weaver_interaction_health"]["healthy"] is False


def test_record_motion_and_outcome_helpers_work() -> None:
    motion_history: dict[str, list[dict[str, object]]] = {}
    daimoi_id, trail = record_daimoi_motion_trail(
        {"id": "d1", "x": 0.2, "y": 0.3, "vx": 0.1, "vy": 0.05},
        tick=7,
        motion_history=motion_history,
        motion_history_lock=threading.Lock(),
        trail_steps=4,
        safe_float=_safe_float,
        safe_int=_safe_int,
        clamp01=_clamp01,
    )
    assert daimoi_id == "d1"
    assert len(trail) == 1

    outcome = nooi_outcome_from_particle(
        {"crawler_interaction_status": "accepted", "message_probability": 0.8},
        safe_float=_safe_float,
        safe_int=_safe_int,
    )
    assert outcome is not None
    assert outcome["outcome"] == "food"


def test_apply_nooi_from_particles_records_summary_and_seed_vectors() -> None:
    nooi_field = NooiField()
    boot_applied = maybe_seed_random_nooi_field_vectors(
        force=True,
        nooi_random_boot_applied=False,
        nooi_random_boot_lock=threading.Lock(),
        nooi_field=nooi_field,
        random_field_vectors_on_boot=1.0,
        random_field_vector_count=3,
        random_field_vector_magnitude=0.2,
        random_field_vector_seed=42,
        safe_float=_safe_float,
        safe_int=_safe_int,
        clamp01=_clamp01,
    )
    assert boot_applied is True

    motion_history: dict[str, list[dict[str, object]]] = {}
    grid, summary = apply_nooi_from_particles(
        [
            {
                "id": "d1",
                "x": 0.4,
                "y": 0.6,
                "vx": 0.1,
                "vy": 0.0,
                "age": 9,
                "resource_consume_amount": 0.08,
            },
            {
                "id": "d2",
                "x": 0.7,
                "y": 0.2,
                "vx": 0.05,
                "vy": 0.03,
                "age": 10,
                "resource_action_blocked": True,
            },
        ],
        dt_seconds=0.2,
        tick=11,
        nooi_field=nooi_field,
        motion_history=motion_history,
        motion_history_lock=threading.Lock(),
        trail_steps=6,
        safe_float=_safe_float,
        safe_int=_safe_int,
        clamp01=_clamp01,
    )
    assert isinstance(grid, dict)
    assert int(summary["food"]) >= 1
    assert int(summary["death"]) >= 1
