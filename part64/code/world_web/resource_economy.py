from __future__ import annotations

import time
from typing import Any

from .metrics import _safe_float, _safe_int, _clamp01
from .docker_runtime import update_container_resources

# Economy Constants
MIN_CPU_NANO = 10_000_000  # 0.01 CPU
MAX_CPU_NANO = 4_000_000_000  # 4.0 CPU (cap)
MIN_MEM_BYTES = 64 * 1024 * 1024  # 64 MB
MAX_MEM_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB
DEFAULT_BASE_CPU_NANO = 250_000_000  # 0.25 CPU floor when limits unavailable
DEFAULT_BASE_MEM_BYTES = 512 * 1024 * 1024  # 512 MB floor when limits unavailable

# Conversion Rates
DAIMOI_TO_CPU_NANO = 100_000_000  # 1 CPU Daimoi = 0.1 CPU core
DAIMOI_TO_MEM_BYTES = 128 * 1024 * 1024  # 1 RAM Daimoi = 128 MB RAM

# Consumption Rates (per tick)
CPU_DECAY_RATE = 0.005
RAM_DECAY_RATE = 0.002

_UPDATE_COOLDOWN_SECONDS = 15.0
_LAST_UPDATE_TIME: dict[str, float] = {}


def _base_cpu_nano_from_sim(sim: dict[str, Any]) -> int:
    resources = (
        sim.get("resources", {}) if isinstance(sim.get("resources"), dict) else {}
    )
    limits = (
        resources.get("limits", {}) if isinstance(resources.get("limits"), dict) else {}
    )

    nano_cpus = max(0, _safe_int(limits.get("nano_cpus"), 0))
    if nano_cpus > 0:
        return max(MIN_CPU_NANO, nano_cpus)

    cpu_limit_cores = max(0.0, _safe_float(limits.get("cpu_limit_cores"), 0.0))
    if cpu_limit_cores > 0.0:
        return max(MIN_CPU_NANO, int(round(cpu_limit_cores * 1_000_000_000.0)))

    return max(MIN_CPU_NANO, DEFAULT_BASE_CPU_NANO)


def _base_mem_bytes_from_sim(sim: dict[str, Any]) -> int:
    resources = (
        sim.get("resources", {}) if isinstance(sim.get("resources"), dict) else {}
    )
    limits = (
        resources.get("limits", {}) if isinstance(resources.get("limits"), dict) else {}
    )
    usage = (
        resources.get("usage", {}) if isinstance(resources.get("usage"), dict) else {}
    )

    memory_limit_bytes = max(0, _safe_int(limits.get("memory_limit_bytes"), 0))
    if memory_limit_bytes > 0:
        return max(MIN_MEM_BYTES, memory_limit_bytes)

    observed_limit = max(0, _safe_int(usage.get("memory_limit_bytes"), 0))
    if observed_limit > 0:
        return max(MIN_MEM_BYTES, observed_limit)

    return max(MIN_MEM_BYTES, DEFAULT_BASE_MEM_BYTES)


def sync_sub_sim_presences(
    presence_impacts: list[dict[str, Any]],
    docker_snapshot: dict[str, Any],
) -> None:
    """
    Injects presences for Docker simulations into the impact list.
    """
    if not isinstance(docker_snapshot, dict):
        return

    simulations = docker_snapshot.get("simulations", [])
    if not isinstance(simulations, list):
        return

    existing_by_id: dict[str, dict[str, Any]] = {}
    for item in presence_impacts:
        if not isinstance(item, dict):
            continue
        presence_id = str(item.get("id", "")).strip()
        if not presence_id:
            continue
        existing_by_id[presence_id] = item

    for sim in simulations:
        if not isinstance(sim, dict):
            continue

        container_id = str(sim.get("id", "")).strip()
        name = str(sim.get("name", "")).strip()
        if not container_id or not name:
            continue

        # Construct presence ID
        presence_id = f"presence.sim.{name}"

        # Extract current resources for initial state
        resources = (
            sim.get("resources", {}) if isinstance(sim.get("resources"), dict) else {}
        )
        usage = (
            resources.get("usage", {})
            if isinstance(resources.get("usage"), dict)
            else {}
        )
        cpu_percent = _safe_float(usage.get("cpu_percent", 0.0), 0.0)
        mem_bytes = _safe_int(usage.get("memory_usage_bytes", 0), 0)

        base_cpu_nano = _base_cpu_nano_from_sim(sim)
        base_mem_bytes = _base_mem_bytes_from_sim(sim)

        existing = existing_by_id.get(presence_id)
        if isinstance(existing, dict):
            economy_raw = existing.get("economy")
            if not isinstance(economy_raw, dict):
                economy_raw = {}
                existing["economy"] = economy_raw

            prior_cpu = max(0, _safe_int(economy_raw.get("base_cpu_nano"), 0))
            prior_mem = max(0, _safe_int(economy_raw.get("base_mem_bytes"), 0))
            economy_raw["base_cpu_nano"] = max(base_cpu_nano, prior_cpu)
            economy_raw["base_mem_bytes"] = max(base_mem_bytes, prior_mem)
            economy_raw["cpu_usage"] = cpu_percent
            economy_raw["mem_usage"] = mem_bytes
            continue

        # Create presence entry
        impact = {
            "id": presence_id,
            "presence_type": "sub-sim",
            "container_id": container_id,
            "label": name,
            "en": f"Sim: {name}",
            "ja": f"シム: {name}",
            # Base influence from actual usage
            "affected_by": {
                "resource": _clamp01(cpu_percent / 100.0),
                "files": 0.5,  # Default
            },
            "affects": {
                "world": 0.5,
            },
            # Wallet placeholder (will be populated from runtime state if available)
            "resource_wallet": {},
            # Meta for economy processing
            "economy": {
                "cpu_usage": cpu_percent,
                "mem_usage": mem_bytes,
                "base_cpu_nano": base_cpu_nano,
                "base_mem_bytes": base_mem_bytes,
            },
        }
        presence_impacts.append(impact)


def process_resource_cycle(
    presence_impacts: list[dict[str, Any]],
    now: float,
) -> None:
    """
    Decays wallets, calculates target limits, and updates Docker containers.
    """
    for presence in presence_impacts:
        if presence.get("presence_type") != "sub-sim":
            continue

        wallet = presence.get("resource_wallet")
        if not isinstance(wallet, dict):
            wallet = {}
            presence["resource_wallet"] = wallet

        container_id = presence.get("container_id")
        if not container_id:
            continue

        # 1. Decay (Consumption)
        cpu_credit = max(0.0, _safe_float(wallet.get("cpu", 0.0), 0.0))
        ram_credit = max(0.0, _safe_float(wallet.get("ram", 0.0), 0.0))

        # Burn rate increases with higher balance (maintenance cost)
        cpu_burn = CPU_DECAY_RATE * (1.0 + (cpu_credit * 0.1))
        ram_burn = RAM_DECAY_RATE * (1.0 + (ram_credit * 0.1))

        cpu_credit = max(0.0, cpu_credit - cpu_burn)
        ram_credit = max(0.0, ram_credit - ram_burn)

        # Update wallet in place (will be persisted by particle/presence sync)
        wallet["cpu"] = cpu_credit
        wallet["ram"] = ram_credit

        # 2. Calculate Targets
        # Base limit + Credit-based expansion
        economy_raw = presence.get("economy")
        if not isinstance(economy_raw, dict):
            economy_raw = {}
            presence["economy"] = economy_raw
        base_cpu_nano = max(
            MIN_CPU_NANO,
            _safe_int(economy_raw.get("base_cpu_nano"), DEFAULT_BASE_CPU_NANO),
        )
        base_mem_bytes = max(
            MIN_MEM_BYTES,
            _safe_int(economy_raw.get("base_mem_bytes"), DEFAULT_BASE_MEM_BYTES),
        )

        target_cpu_nano = int(MIN_CPU_NANO + (cpu_credit * DAIMOI_TO_CPU_NANO))
        target_mem_bytes = int(MIN_MEM_BYTES + (ram_credit * DAIMOI_TO_MEM_BYTES))
        target_cpu_nano = max(base_cpu_nano, target_cpu_nano)
        target_mem_bytes = max(base_mem_bytes, target_mem_bytes)

        # Clamp
        target_cpu_nano = min(max(base_cpu_nano, MAX_CPU_NANO), target_cpu_nano)
        target_mem_bytes = min(max(base_mem_bytes, MAX_MEM_BYTES), target_mem_bytes)

        # 3. Check Cooldown & Apply
        last_update = _LAST_UPDATE_TIME.get(container_id, 0.0)
        if now - last_update > _UPDATE_COOLDOWN_SECONDS:
            # We don't read current limits here easily (expensive to inspect every tick).
            # We just optimistically set. Docker handles no-op if unchanged efficiently?
            # Actually docker_runtime.update_container_resources checks if updates dict is empty,
            # but we are constructing it.
            # To avoid spam, maybe we should cache the LAST SET limit and only update if diff > threshold.

            # Simple threshold check logic could live here if we cached last_set_limits.
            # For now, rely on cooldown.

            success, msg = update_container_resources(
                container_id,
                nano_cpus=target_cpu_nano,
                memory_bytes=target_mem_bytes,
            )
            if success:
                _LAST_UPDATE_TIME[container_id] = now
                presence["economy_last_update"] = "ok"
            else:
                presence["economy_last_update"] = f"error:{msg}"
