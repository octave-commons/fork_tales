from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class AntiClumpControlMode:
    mode: str
    error: float
    score_error: float


def resolve_anti_clump_control_mode(
    *,
    particle_count: int,
    min_particles: int,
    snr_valid: bool,
    snr_low_gap: float,
    score_ema: float,
    target: float,
    snr_score_blend: float,
) -> AntiClumpControlMode:
    score_error = score_ema - target
    mode = "score"
    error = score_error

    if particle_count < min_particles:
        mode = "underpop"
    elif snr_valid:
        mode = "snr"
        error = snr_low_gap
        score_excess = max(0.0, score_error)
        error = max(error, score_excess * snr_score_blend)

    return AntiClumpControlMode(
        mode=mode,
        error=error,
        score_error=score_error,
    )


def resolve_anti_clump_deadband_active(
    *,
    mode: str,
    error: float,
    snr_high_gap: float,
    deadband_enabled: bool,
    snr_low_gap_deadband: float,
    score_deadband: float,
) -> bool:
    if not deadband_enabled:
        return False
    if (
        mode == "snr"
        and error < snr_low_gap_deadband
        and snr_high_gap < snr_low_gap_deadband
    ):
        return True
    if mode == "score" and abs(error) < score_deadband:
        return True
    return False


def resolve_anti_clump_integral_leak(
    *,
    mode: str,
    deadband_active: bool,
    snr_in_band: bool,
    integral_leak: float,
    deadband_integral_leak_boost: float,
) -> float:
    leak = _clamp01(
        integral_leak + (deadband_integral_leak_boost if deadband_active else 0.0)
    )
    if mode == "underpop":
        return max(leak, 0.18)
    if mode == "snr" and snr_in_band:
        return max(leak, 0.16)
    return leak


def resolve_anti_clump_can_integrate(
    *,
    mode: str,
    deadband_active: bool,
    snr_in_band: bool,
    integral_enabled: bool,
    integral_freeze_on_saturation: bool,
    is_saturated: bool,
    integral_freeze_on_sign_flip: bool,
    error_sign_streak: int,
    integral_sign_stable_updates: int,
) -> bool:
    can_integrate = (
        integral_enabled
        and mode in {"snr", "score"}
        and not deadband_active
        and not (mode == "snr" and snr_in_band)
    )
    if can_integrate and integral_freeze_on_saturation and is_saturated:
        return False
    if (
        can_integrate
        and integral_freeze_on_sign_flip
        and error_sign_streak < integral_sign_stable_updates
    ):
        return False
    return can_integrate


def step_anti_clump_integrals_for_mode(
    *,
    mode: str,
    error: float,
    can_integrate: bool,
    leak: float,
    integral_limit: float,
    integral_snr: float,
    integral_score: float,
    integral_step: Callable[..., float],
) -> tuple[float, float, float]:
    if mode == "snr":
        next_integral_snr = integral_step(
            value=integral_snr,
            error=error,
            integrate=can_integrate,
            leak=leak,
            integral_limit=integral_limit,
        )
        next_integral_score = integral_step(
            value=integral_score,
            error=0.0,
            integrate=False,
            leak=leak,
            integral_limit=integral_limit,
        )
        return next_integral_snr, next_integral_score, next_integral_snr

    if mode == "score":
        next_integral_score = integral_step(
            value=integral_score,
            error=error,
            integrate=can_integrate,
            leak=leak,
            integral_limit=integral_limit,
        )
        next_integral_snr = integral_step(
            value=integral_snr,
            error=0.0,
            integrate=False,
            leak=leak,
            integral_limit=integral_limit,
        )
        return next_integral_snr, next_integral_score, next_integral_score

    next_integral_snr = integral_step(
        value=integral_snr,
        error=0.0,
        integrate=False,
        leak=leak,
        integral_limit=integral_limit,
    )
    next_integral_score = integral_step(
        value=integral_score,
        error=0.0,
        integrate=False,
        leak=leak,
        integral_limit=integral_limit,
    )
    return next_integral_snr, next_integral_score, next_integral_score


def resolve_anti_clump_raw_drive(
    *,
    mode: str,
    deadband_active: bool,
    kp_eff: float,
    ki_eff: float,
    error: float,
    active_integral: float,
    snr_high_gap: float,
    high_snr_perturb_gain: float,
) -> float:
    if mode == "underpop" or deadband_active:
        return 0.0

    raw_drive = (kp_eff * error) + (ki_eff * active_integral)
    if mode == "snr":
        raw_drive += snr_high_gap * high_snr_perturb_gain
    return raw_drive
