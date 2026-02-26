# SPDX-License-Identifier: GPL-3.0-or-later
# This file is part of Fork Tales.
# Copyright (C) 2024-2025 Fork Tales Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
ledger_sonify.py — deterministic "receipt audio" generator

Purpose
-------
Turn a manifest (list of files + sha256) into a deterministic WAV fingerprint.
This is NOT "music"; it's a replayable acoustic checksum that can be embedded
in other tracks or used for regression tests.

Constraints supported
---------------------
- seed derivation is explicit (no hidden RNG)
- WAV is canonical output
"""

from __future__ import annotations

import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Dict, List, Tuple

SR = 44100


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest_seed(manifest: Dict) -> int:
    """
    Derive a 32-bit seed from the manifest content (stable across runs).
    """
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(blob).digest()
    return int.from_bytes(h[:4], "big", signed=False)


def write_wav(path: Path, stereo_f32, sr: int = SR) -> None:
    import numpy as np

    data = np.clip(stereo_f32, -1, 1)
    ints = (data * 32767).astype("int16")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(ints.tobytes())


def sonify_manifest(manifest: Dict, seconds: float = 3.0):
    """
    Generate a short stereo fingerprint:
    - three tones whose frequencies come from the manifest hash
    - gated pulses at prime intervals
    """
    import numpy as np

    seed = manifest_seed(manifest)
    n = int(seconds * SR)
    t = np.arange(n) / SR

    h = hashlib.sha256(str(seed).encode("utf-8")).digest()
    f1 = 220 + (h[0] / 255.0) * 220
    f2 = 330 + (h[1] / 255.0) * 330
    f3 = 110 + (h[2] / 255.0) * 440

    tone = (
        0.22 * np.sin(2 * math.pi * f1 * t)
        + 0.18 * np.sin(2 * math.pi * f2 * t)
        + 0.12 * np.sin(2 * math.pi * f3 * t)
    )

    # prime-ish pulse gates
    gate = (np.sin(2 * math.pi * 2.0 * t) > 0).astype(float) * 0.6 + (
        np.sin(2 * math.pi * 3.0 * t) > 0
    ).astype(float) * 0.4
    sig = np.tanh(tone * gate)

    # stereo: tiny delay
    d = int(0.007 * SR)
    left = sig
    right = np.concatenate([np.zeros(d), sig[:-d]])
    stereo = np.stack([left, right], axis=1) * 0.9
    return stereo


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("usage: ledger_sonify.py <manifest.json> <out.wav>")
        return 2
    mpath = Path(argv[1])
    out = Path(argv[2])
    manifest = json.loads(mpath.read_text("utf-8"))
    stereo = sonify_manifest(manifest)
    write_wav(out, stereo)
    print("seed:", manifest_seed(manifest))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv))
