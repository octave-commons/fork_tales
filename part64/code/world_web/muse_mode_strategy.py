from __future__ import annotations

import math
import random
from hashlib import sha1
from typing import Any


_MUSE_RUNTIME_MODES: set[str] = {"deterministic", "stochastic"}


def _seed_to_int(seed: str) -> int:
    return int(sha1(str(seed or "").encode("utf-8")).hexdigest()[:16], 16)


def normalize_muse_runtime_mode(value: Any, *, default: str = "stochastic") -> str:
    normalized_default = str(default or "stochastic").strip().lower() or "stochastic"
    if normalized_default not in _MUSE_RUNTIME_MODES:
        normalized_default = "stochastic"
    mode = str(value or normalized_default).strip().lower()
    if mode not in _MUSE_RUNTIME_MODES:
        return normalized_default
    return mode


def resolve_muse_reply_backend_mode(value: Any) -> str:
    if normalize_muse_runtime_mode(value) == "deterministic":
        return "canonical"
    return "llm"


def select_muse_surround_rows(
    scored_rows: list[tuple[dict[str, Any], float]],
    *,
    mode: Any,
    seed: str,
    tau: float,
) -> list[tuple[dict[str, Any], float]]:
    clean_mode = normalize_muse_runtime_mode(mode)
    if clean_mode == "deterministic":
        return list(scored_rows)

    selected_rows: list[tuple[dict[str, Any], float]] = []
    pool = list(scored_rows)
    while pool:
        weights = [math.exp(score / max(0.1, float(tau))) for _node, score in pool]
        total = sum(weights)
        if total <= 0.0:
            break
        mark = (
            random.Random(_seed_to_int(f"{seed}|{len(selected_rows)}")).random() * total
        )
        cursor = 0.0
        chosen_index = 0
        for idx, weight in enumerate(weights):
            cursor += weight
            if cursor >= mark:
                chosen_index = idx
                break
        selected_rows.append(pool.pop(chosen_index))
    return selected_rows
