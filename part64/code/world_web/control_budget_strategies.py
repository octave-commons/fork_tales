from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .control_budget_policy import (
    ControlBudgetSnapshot,
    control_budget_mode_label,
    resolve_control_budget_tier,
)


@dataclass(frozen=True)
class ThreatBudgetStrategy:
    allow_classifier: bool
    allow_llm: bool
    llm_item_cap_limit: int


@dataclass(frozen=True)
class InteractionBudgetStrategy:
    per_tick_cap_limit: int | None
    cooldown_floor_seconds: float | None


_THREAT_BUDGET_STRATEGIES: dict[str, ThreatBudgetStrategy] = {
    "minimal": ThreatBudgetStrategy(
        allow_classifier=False,
        allow_llm=False,
        llm_item_cap_limit=0,
    ),
    "reduced": ThreatBudgetStrategy(
        allow_classifier=True,
        allow_llm=False,
        llm_item_cap_limit=0,
    ),
    "moderate": ThreatBudgetStrategy(
        allow_classifier=True,
        allow_llm=True,
        llm_item_cap_limit=2,
    ),
    "full": ThreatBudgetStrategy(
        allow_classifier=True,
        allow_llm=True,
        llm_item_cap_limit=6,
    ),
}


_INTERACTION_BUDGET_STRATEGIES: dict[str, InteractionBudgetStrategy] = {
    "minimal": InteractionBudgetStrategy(
        per_tick_cap_limit=1, cooldown_floor_seconds=24.0
    ),
    "reduced": InteractionBudgetStrategy(
        per_tick_cap_limit=2, cooldown_floor_seconds=14.0
    ),
    "moderate": InteractionBudgetStrategy(
        per_tick_cap_limit=3, cooldown_floor_seconds=9.0
    ),
    "full": InteractionBudgetStrategy(
        per_tick_cap_limit=None, cooldown_floor_seconds=None
    ),
}


def _apply_interaction_budget_adjustment(
    *,
    cap: int,
    cooldown_seconds: float,
    per_tick_cap_limit: int | None,
    cooldown_floor_seconds: float | None,
) -> tuple[int, float]:
    updated_cap = int(cap)
    updated_cooldown_seconds = float(cooldown_seconds)
    if per_tick_cap_limit is not None:
        updated_cap = min(updated_cap, int(per_tick_cap_limit))
    if cooldown_floor_seconds is not None:
        updated_cooldown_seconds = max(
            updated_cooldown_seconds,
            float(cooldown_floor_seconds),
        )
    return updated_cap, updated_cooldown_seconds


def apply_threat_compute_budget_policy(
    *,
    snapshot: ControlBudgetSnapshot,
    llm_item_cap_default: int,
) -> dict[str, Any]:
    tier = resolve_control_budget_tier(snapshot)
    strategy = _THREAT_BUDGET_STRATEGIES.get(
        tier,
        _THREAT_BUDGET_STRATEGIES["full"],
    )

    allow_classifier = strategy.allow_classifier
    allow_llm = strategy.allow_llm
    llm_item_cap = max(1, int(llm_item_cap_default))
    reason = tier

    if strategy.llm_item_cap_limit <= 0:
        llm_item_cap = 0
    else:
        llm_item_cap = min(llm_item_cap, int(strategy.llm_item_cap_limit))

    if snapshot.queue_ratio >= 0.90:
        allow_llm = False
        llm_item_cap = 0
        reason = f"{reason}+queue"
    elif snapshot.queue_ratio >= 0.75 and allow_llm and llm_item_cap > 0:
        llm_item_cap = max(1, min(llm_item_cap, 2))
        reason = f"{reason}+queue_soft"

    if snapshot.compute_jobs_180s >= 48 and allow_llm and llm_item_cap > 0:
        llm_item_cap = 1
        reason = f"{reason}+compute"

    return {
        "bound": bool(snapshot.bound),
        "mode": control_budget_mode_label(snapshot),
        "ratio": round(snapshot.ratio, 6),
        "queue_ratio": round(snapshot.queue_ratio, 6),
        "cpu_sentinel_burn_active": bool(snapshot.cpu_sentinel_burn_active),
        "compute_jobs_180s": int(snapshot.compute_jobs_180s),
        "allow_classifier": bool(allow_classifier),
        "allow_llm": bool(allow_llm),
        "llm_item_cap": int(max(0, llm_item_cap)),
        "reason": str(reason),
    }


def apply_weaver_interaction_budget_policy(
    *,
    snapshot: ControlBudgetSnapshot,
    base_per_tick_cap: int,
    base_cooldown_seconds: float,
) -> dict[str, Any]:
    tier = resolve_control_budget_tier(snapshot)
    tier_strategy = _INTERACTION_BUDGET_STRATEGIES.get(
        tier,
        _INTERACTION_BUDGET_STRATEGIES["full"],
    )

    cap = max(0, int(base_per_tick_cap))
    cooldown_seconds = max(0.2, float(base_cooldown_seconds))
    cap, cooldown_seconds = _apply_interaction_budget_adjustment(
        cap=cap,
        cooldown_seconds=cooldown_seconds,
        per_tick_cap_limit=tier_strategy.per_tick_cap_limit,
        cooldown_floor_seconds=tier_strategy.cooldown_floor_seconds,
    )

    if snapshot.queue_ratio >= 0.90:
        cap, cooldown_seconds = _apply_interaction_budget_adjustment(
            cap=cap,
            cooldown_seconds=cooldown_seconds,
            per_tick_cap_limit=1,
            cooldown_floor_seconds=24.0,
        )
    elif snapshot.queue_ratio >= 0.75:
        cap, cooldown_seconds = _apply_interaction_budget_adjustment(
            cap=cap,
            cooldown_seconds=cooldown_seconds,
            per_tick_cap_limit=2,
            cooldown_floor_seconds=12.0,
        )

    if snapshot.compute_jobs_180s >= 48:
        cap, cooldown_seconds = _apply_interaction_budget_adjustment(
            cap=cap,
            cooldown_seconds=cooldown_seconds,
            per_tick_cap_limit=1,
            cooldown_floor_seconds=16.0,
        )
    elif snapshot.compute_jobs_180s >= 28:
        cap, cooldown_seconds = _apply_interaction_budget_adjustment(
            cap=cap,
            cooldown_seconds=cooldown_seconds,
            per_tick_cap_limit=2,
            cooldown_floor_seconds=10.0,
        )

    return {
        "mode": control_budget_mode_label(
            snapshot,
            fallback_when_unbound="full",
            fallback_when_bound="full",
        ),
        "ratio": round(snapshot.ratio, 6),
        "queue_ratio": round(snapshot.queue_ratio, 6),
        "compute_jobs_180s": int(snapshot.compute_jobs_180s),
        "cpu_sentinel_burn_active": bool(snapshot.cpu_sentinel_burn_active),
        "per_tick_cap": int(max(0, cap)),
        "local_cooldown_seconds": round(max(0.2, cooldown_seconds), 3),
    }
