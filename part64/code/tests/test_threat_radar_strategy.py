from __future__ import annotations

from code.world_web.threat_radar_strategy import (
    apply_threat_signal_strategy,
    build_threat_llm_fallback,
    resolve_threat_proximity_strategy,
    resolve_threat_risk_level,
    resolve_threat_scoring_mode,
)


def test_resolve_threat_proximity_strategy_prefers_critical_signal() -> None:
    strategy = resolve_threat_proximity_strategy(
        p_active_max=0.82,
        p_critical_max=0.61,
    )
    assert strategy["boost"] == 2
    assert strategy["signal"] == "proximity_critical_state"


def test_apply_threat_signal_strategy_merges_signal_markers() -> None:
    signals = apply_threat_signal_strategy(
        signals=["credential", "credential"],
        source_tier_boost=1,
        corroboration_boost=1,
        proximity_signal="proximity_active_state",
    )
    assert signals == [
        "corroborated_signal",
        "credential",
        "proximity_active_state",
        "source_tier_boost",
    ]


def test_build_threat_llm_fallback_uses_query_and_budget_reasons() -> None:
    disabled_by_budget = build_threat_llm_fallback(
        llm_requested=True,
        allow_llm=False,
        llm_item_cap=4,
        llm_model="stub-llm",
    )
    disabled_by_query = build_threat_llm_fallback(
        llm_requested=False,
        allow_llm=True,
        llm_item_cap=4,
        llm_model="stub-llm",
    )
    assert disabled_by_budget["error"] == "disabled_by_compute_budget"
    assert disabled_by_query["error"] == "disabled_by_query"


def test_threat_risk_and_scoring_modes_follow_thresholds() -> None:
    assert resolve_threat_risk_level(11) == "critical"
    assert resolve_threat_risk_level(8) == "high"
    assert resolve_threat_risk_level(5) == "medium"
    assert resolve_threat_risk_level(4) == "low"
    assert (
        resolve_threat_scoring_mode(
            llm_enabled=True,
            llm_applied=True,
            classifier_enabled=True,
        )
        == "llm_blend"
    )
    assert (
        resolve_threat_scoring_mode(
            llm_enabled=False,
            llm_applied=False,
            classifier_enabled=True,
        )
        == "classifier"
    )
    assert (
        resolve_threat_scoring_mode(
            llm_enabled=False,
            llm_applied=False,
            classifier_enabled=False,
        )
        == "deterministic"
    )
