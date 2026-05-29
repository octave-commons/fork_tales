from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class ControlBudgetSnapshot:
    mode: str
    ratio: float
    queue_ratio: float
    compute_jobs_180s: int
    cpu_sentinel_burn_active: bool
    bound: bool


def build_control_budget_snapshot(
    presence_dynamics: dict[str, Any] | None,
) -> ControlBudgetSnapshot:
    dynamics = presence_dynamics if isinstance(presence_dynamics, dict) else {}
    resource_consumption = (
        dynamics.get("resource_consumption", {})
        if isinstance(dynamics.get("resource_consumption", {}), dict)
        else {}
    )
    control_budget = (
        resource_consumption.get("control_budget", {})
        if isinstance(resource_consumption.get("control_budget", {}), dict)
        else {}
    )
    mode = str(control_budget.get("mode", "") or "").strip().lower()
    ratio = _clamp01(_safe_float(control_budget.get("ratio", 1.0), 1.0))
    queue_ratio = _clamp01(
        _safe_float(resource_consumption.get("queue_ratio", 0.0), 0.0)
    )
    compute_jobs_180s = max(0, _safe_int(dynamics.get("compute_jobs_180s", 0), 0))
    cpu_sentinel_burn_active = bool(
        resource_consumption.get("cpu_sentinel_burn_active", False)
    )
    return ControlBudgetSnapshot(
        mode=mode,
        ratio=ratio,
        queue_ratio=queue_ratio,
        compute_jobs_180s=compute_jobs_180s,
        cpu_sentinel_burn_active=cpu_sentinel_burn_active,
        bound=bool(control_budget),
    )


def resolve_control_budget_tier(snapshot: ControlBudgetSnapshot) -> str:
    if (
        snapshot.cpu_sentinel_burn_active
        or snapshot.mode == "minimal"
        or snapshot.ratio < 0.08
    ):
        return "minimal"
    if snapshot.mode == "reduced" or snapshot.ratio < 0.16:
        return "reduced"
    if snapshot.mode == "moderate" or snapshot.ratio < 0.30:
        return "moderate"
    return "full"


def control_budget_mode_label(
    snapshot: ControlBudgetSnapshot,
    *,
    fallback_when_unbound: str = "full",
    fallback_when_bound: str = "unknown",
) -> str:
    if snapshot.mode:
        return snapshot.mode
    if snapshot.bound:
        return fallback_when_bound
    return fallback_when_unbound
