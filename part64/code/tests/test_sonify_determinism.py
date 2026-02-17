"""
Determinism test: manifest -> WAV hash snapshot.

This is a *receipt* test: if it changes, update the snapshot intentionally.
Run: python -m code.tests.test_sonify_determinism
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from code.sonify.ledger_sonify import sonify_manifest, manifest_seed, write_wav

EXPECTED_SHA256 = "48620bf025c8ff62414830bb6d5d7ed5eb4066bb98175119afa07784c413954f"

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def test():
    manifest = {"part": 64, "seed_label": "ημ_part_64", "files": [{"path":"x","sha256":"0"*64}]}
    seed = manifest_seed(manifest)
    stereo = sonify_manifest(manifest, seconds=1.0)
    out = Path(__file__).parent / "_tmp_receipt.wav"
    write_wav(out, stereo)
    h = sha256_file(out)
    assert h == EXPECTED_SHA256, {"seed": seed, "got": h, "expected": EXPECTED_SHA256}
    out.unlink(missing_ok=True)

if __name__ == "__main__":
    test()
    print("ok")
