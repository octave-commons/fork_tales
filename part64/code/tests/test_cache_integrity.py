from __future__ import annotations

from pathlib import Path

from code.world_web import constants as constants_module
from code.world_web import simulation as simulation_module


def test_mix_cache_defaults_keep_required_keys() -> None:
    assert isinstance(constants_module._MIX_CACHE, dict)
    assert {"fingerprint", "wav", "meta"}.issubset(constants_module._MIX_CACHE)


def test_build_mix_stream_recovers_from_empty_cache_shape() -> None:
    with simulation_module._MIX_CACHE_LOCK:
        simulation_module._MIX_CACHE.clear()

    wav, meta = simulation_module.build_mix_stream(
        {"artifacts": [], "items": []}, Path(".")
    )
    assert isinstance(wav, bytes)
    assert isinstance(meta, dict)
    assert "fingerprint" in meta
