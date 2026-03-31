from __future__ import annotations

from code.world_web.daimoi_observer_control_strategy import (
    resolve_anti_clump_can_integrate,
    resolve_anti_clump_control_mode,
    resolve_anti_clump_deadband_active,
    resolve_anti_clump_integral_leak,
    resolve_anti_clump_raw_drive,
    step_anti_clump_integrals_for_mode,
)


def test_resolve_anti_clump_control_mode_prioritizes_underpop() -> None:
    state = resolve_anti_clump_control_mode(
        particle_count=8,
        min_particles=24,
        snr_valid=True,
        snr_low_gap=0.4,
        score_ema=0.9,
        target=0.4,
        snr_score_blend=0.6,
    )
    assert state.mode == "underpop"
    assert abs(state.error - 0.5) < 1e-9


def test_resolve_anti_clump_control_mode_uses_snr_blended_error() -> None:
    state = resolve_anti_clump_control_mode(
        particle_count=36,
        min_particles=24,
        snr_valid=True,
        snr_low_gap=0.08,
        score_ema=0.9,
        target=0.4,
        snr_score_blend=0.6,
    )
    assert state.mode == "snr"
    assert abs(state.error - 0.3) < 1e-9


def test_resolve_anti_clump_deadband_active_for_snr_and_score() -> None:
    snr_deadband = resolve_anti_clump_deadband_active(
        mode="snr",
        error=0.01,
        snr_high_gap=0.01,
        deadband_enabled=True,
        snr_low_gap_deadband=0.03,
        score_deadband=0.02,
    )
    score_deadband = resolve_anti_clump_deadband_active(
        mode="score",
        error=0.015,
        snr_high_gap=0.2,
        deadband_enabled=True,
        snr_low_gap_deadband=0.03,
        score_deadband=0.02,
    )
    assert snr_deadband is True
    assert score_deadband is True


def test_integral_leak_and_can_integrate_respect_mode_constraints() -> None:
    leak = resolve_anti_clump_integral_leak(
        mode="underpop",
        deadband_active=False,
        snr_in_band=False,
        integral_leak=0.04,
        deadband_integral_leak_boost=0.0,
    )
    assert abs(leak - 0.18) < 1e-9

    can_integrate = resolve_anti_clump_can_integrate(
        mode="snr",
        deadband_active=False,
        snr_in_band=False,
        integral_enabled=True,
        integral_freeze_on_saturation=True,
        is_saturated=True,
        integral_freeze_on_sign_flip=True,
        error_sign_streak=2,
        integral_sign_stable_updates=2,
    )
    assert can_integrate is False


def test_step_integrals_and_raw_drive_follow_mode_strategy() -> None:
    def _step(
        *,
        value: float,
        error: float,
        integrate: bool,
        leak: float,
        integral_limit: float,
    ) -> float:
        if integrate:
            return value + error
        return value * (1.0 - leak)

    snr_snr, snr_score, active = step_anti_clump_integrals_for_mode(
        mode="snr",
        error=0.3,
        can_integrate=True,
        leak=0.1,
        integral_limit=1.5,
        integral_snr=0.2,
        integral_score=0.4,
        integral_step=_step,
    )
    assert abs(snr_snr - 0.5) < 1e-9
    assert abs(snr_score - 0.36) < 1e-9
    assert abs(active - snr_snr) < 1e-9

    raw_drive = resolve_anti_clump_raw_drive(
        mode="snr",
        deadband_active=False,
        kp_eff=0.2,
        ki_eff=0.1,
        error=0.5,
        active_integral=0.5,
        snr_high_gap=0.25,
        high_snr_perturb_gain=0.3,
    )
    assert abs(raw_drive - 0.225) < 1e-9
