from __future__ import annotations

from typing import Any


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def resolve_threat_proximity_strategy(
    *,
    p_active_max: float,
    p_critical_max: float,
) -> dict[str, Any]:
    boost = 0
    if p_critical_max >= 0.75:
        boost = 2
    elif p_critical_max >= 0.55:
        boost = 1
    if p_active_max >= 0.70 and boost < 2:
        boost += 1

    signal = ""
    if p_critical_max >= 0.55:
        signal = "proximity_critical_state"
    elif p_active_max >= 0.55:
        signal = "proximity_active_state"

    return {
        "boost": max(0, min(2, int(boost))),
        "signal": signal,
        "p_active_max": round(_clamp01(p_active_max), 6),
        "p_critical_max": round(_clamp01(p_critical_max), 6),
    }


def apply_threat_signal_strategy(
    *,
    signals: list[str],
    source_tier_boost: int,
    corroboration_boost: int,
    proximity_signal: str,
) -> list[str]:
    merged = set(str(token or "") for token in signals if str(token or "").strip())
    if source_tier_boost > 0:
        merged.add("source_tier_boost")
    if corroboration_boost > 0:
        merged.add("corroborated_signal")
    if proximity_signal:
        merged.add(str(proximity_signal))
    return sorted(merged)


def build_threat_llm_fallback(
    *,
    llm_requested: bool,
    allow_llm: bool,
    llm_item_cap: int,
    llm_model: str,
) -> dict[str, Any]:
    llm_error = ""
    if llm_requested and not allow_llm:
        llm_error = "disabled_by_compute_budget"
    elif llm_requested and llm_item_cap <= 0:
        llm_error = "disabled_by_compute_budget"
    elif not llm_requested:
        llm_error = "disabled_by_query"

    return {
        "enabled": False,
        "applied": False,
        "model": str(llm_model or ""),
        "error": llm_error,
        "metrics": {},
    }


def resolve_threat_risk_level(score: int) -> str:
    level_score = int(score)
    if level_score >= 11:
        return "critical"
    if level_score >= 8:
        return "high"
    if level_score >= 5:
        return "medium"
    return "low"


def resolve_threat_scoring_mode(
    *,
    llm_enabled: bool,
    llm_applied: bool,
    classifier_enabled: bool,
) -> str:
    if llm_enabled and llm_applied:
        return "llm_blend"
    if classifier_enabled:
        return "classifier"
    return "deterministic"
