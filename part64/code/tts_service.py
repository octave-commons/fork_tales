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

import os
import re
import time
import math
import torch
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple
from melo.api import TTS
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="eta-mu TTS sidecar")

# Config
CACHE_DIR = Path("world_state/tts_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FFMPEG_BIN = shutil.which("ffmpeg") or ""


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


NARRATOR_UNIFIER_ENABLED = _env_bool("TTS_NARRATOR_UNIFIER_ENABLED", True)
NARRATOR_PROFILE_VERSION = "narrator-v1"
NARRATOR_EN_PITCH = _env_float("TTS_NARRATOR_EN_PITCH", 1.02, 0.9, 1.1)
NARRATOR_JP_PITCH = _env_float("TTS_NARRATOR_JP_PITCH", 0.97, 0.9, 1.1)
NARRATOR_EN_VARIANCE_DEPTH = _env_float(
    "TTS_NARRATOR_EN_VARIANCE_DEPTH", 0.02, 0.0, 0.2
)
NARRATOR_VARIANCE_FREQ_HZ = _env_float("TTS_NARRATOR_VARIANCE_FREQ_HZ", 4.5, 0.1, 12.0)
NARRATOR_ROOM_PREDELAY_MS = _env_int("TTS_NARRATOR_ROOM_PREDELAY_MS", 12, 0, 250)
NARRATOR_SWITCH_FADE_OUT_MS = _env_int("TTS_NARRATOR_SWITCH_FADE_OUT_MS", 50, 0, 250)
NARRATOR_SWITCH_FADE_IN_MS = _env_int("TTS_NARRATOR_SWITCH_FADE_IN_MS", 50, 0, 250)
NARRATOR_SWITCH_GAP_MS = _env_int("TTS_NARRATOR_SWITCH_GAP_MS", 100, 0, 400)
NARRATOR_ENVELOPE_WINDOW_MS = _env_int("TTS_NARRATOR_ENVELOPE_WINDOW_MS", 60, 20, 400)
NARRATOR_ENVELOPE_STRENGTH = _env_float(
    "TTS_NARRATOR_ENVELOPE_STRENGTH", 0.12, 0.0, 0.5
)
NARRATOR_ENVELOPE_MAX_GAIN_DB = _env_float(
    "TTS_NARRATOR_ENVELOPE_MAX_GAIN_DB", 1.2, 0.2, 8.0
)
NARRATOR_TARGET_DBFS = _env_float("TTS_NARRATOR_TARGET_DBFS", -18.0, -30.0, -8.0)

print(f"[tts] Initializing MeloTTS on {DEVICE}...")

# Warmup / Load Models
# We'll load them lazily or at start?
# Loading at start is better for Tier 0 latency.
MODELS = {
    "EN": TTS(language="EN", device=DEVICE),
    "JP": TTS(language="JP", device=DEVICE),
}


def _narrator_profile_signature() -> str:
    return (
        f"enabled={int(NARRATOR_UNIFIER_ENABLED)}"
        f";profile={NARRATOR_PROFILE_VERSION}"
        f";en_pitch={NARRATOR_EN_PITCH:.4f}"
        f";jp_pitch={NARRATOR_JP_PITCH:.4f}"
        f";en_var={NARRATOR_EN_VARIANCE_DEPTH:.4f}"
        f";var_f={NARRATOR_VARIANCE_FREQ_HZ:.3f}"
        f";room_pd={NARRATOR_ROOM_PREDELAY_MS}"
        f";fade_out={NARRATOR_SWITCH_FADE_OUT_MS}"
        f";gap={NARRATOR_SWITCH_GAP_MS}"
        f";fade_in={NARRATOR_SWITCH_FADE_IN_MS}"
        f";env_w={NARRATOR_ENVELOPE_WINDOW_MS}"
        f";env_s={NARRATOR_ENVELOPE_STRENGTH:.3f}"
        f";env_g={NARRATOR_ENVELOPE_MAX_GAIN_DB:.2f}"
    )


def _build_narrator_filter_chain(lang: str) -> str:
    pitch = NARRATOR_EN_PITCH if lang == "EN" else NARRATOR_JP_PITCH
    filters = [f"rubberband=pitch={pitch:.6f}"]
    if lang == "EN" and NARRATOR_EN_VARIANCE_DEPTH > 0.0:
        filters.append(
            "vibrato="
            f"f={NARRATOR_VARIANCE_FREQ_HZ:.3f}"
            f":d={NARRATOR_EN_VARIANCE_DEPTH:.3f}"
        )

    # Keep the narrator chain conservative. The broader EQ/compressor/echo/exciter
    # stack that once worked can produce clipped, DC-heavy output on current ffmpeg
    # builds, so only apply the parts we have verified remain speech-safe.
    return ",".join(filters)


def _render_with_ffmpeg(input_path: Path, output_path: Path, filters: str) -> bool:
    if not FFMPEG_BIN:
        return False
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-af",
        filters,
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=40)
    except (OSError, subprocess.SubprocessError):
        return False
    if not output_path.exists():
        return False
    return output_path.stat().st_size > 44


def _apply_envelope_shape(segment):
    if NARRATOR_ENVELOPE_STRENGTH <= 0.0 or len(segment) < NARRATOR_ENVELOPE_WINDOW_MS:
        return segment

    chunk_ms = max(1, NARRATOR_ENVELOPE_WINDOW_MS)
    chunk_count = max(1, math.ceil(len(segment) / chunk_ms))
    chunks = []
    rms_values = []
    for idx in range(chunk_count):
        start = idx * chunk_ms
        end = min(len(segment), start + chunk_ms)
        chunk = segment[start:end]
        chunks.append(chunk)
        rms_values.append(float(max(chunk.rms, 1)))

    if not rms_values:
        return segment
    mean_rms = sum(rms_values) / len(rms_values)
    if mean_rms <= 0:
        return segment

    shaped = segment[:0]
    for chunk, rms in zip(chunks, rms_values):
        normalized = (rms / mean_rms) - 1.0
        scale = 1.0 + (normalized * NARRATOR_ENVELOPE_STRENGTH)
        scale = max(0.7, min(1.3, scale))
        gain_db = 20.0 * math.log10(scale)
        gain_db = max(
            -NARRATOR_ENVELOPE_MAX_GAIN_DB, min(NARRATOR_ENVELOPE_MAX_GAIN_DB, gain_db)
        )
        shaped += chunk.apply_gain(gain_db)
    return shaped


def _normalize_segment_level(segment):
    dbfs = float(getattr(segment, "dBFS", float("-inf")))
    if dbfs == float("-inf"):
        return segment
    gain = NARRATOR_TARGET_DBFS - dbfs
    gain = max(-18.0, min(18.0, gain))
    return segment.apply_gain(gain)


def _apply_language_switch_smoothing(
    combined, segment, switched_language: bool, pydub_module
):
    if not switched_language:
        return combined + segment

    fade_out_ms = min(NARRATOR_SWITCH_FADE_OUT_MS, len(combined))
    if fade_out_ms > 0:
        combined = combined[:-fade_out_ms] + combined[-fade_out_ms:].fade_out(
            fade_out_ms
        )

    if NARRATOR_SWITCH_GAP_MS > 0:
        combined += pydub_module.AudioSegment.silent(duration=NARRATOR_SWITCH_GAP_MS)

    fade_in_ms = min(NARRATOR_SWITCH_FADE_IN_MS, len(segment))
    if fade_in_ms > 0:
        segment = segment.fade_in(fade_in_ms)

    return combined + segment


def split_text_by_language(text: str) -> List[Tuple[str, str]]:
    """
    Splits text into chunks of English and Japanese.
    Returns list of (text, lang_code) tuples.
    """
    # Pattern to catch Japanese characters (Kanji, Hiragana, Katakana)
    jp_pattern = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")

    parts = []
    last_end = 0

    for match in jp_pattern.finditer(text):
        start, end = match.start(), match.end()
        # Add preceding English/Other part if exists
        if start > last_end:
            parts.append((text[last_end:start], "EN"))
        # Add Japanese part
        parts.append((text[start:end], "JP"))
        last_end = end

    # Add remaining part
    if last_end < len(text):
        parts.append((text[last_end:], "EN"))

    # Filter empty or whitespace-only
    return [(t.strip(), l) for t, l in parts if t.strip()]


def get_cache_key(text: str, speed: float) -> str:
    spec = f"{text}|{speed}|v3|{_narrator_profile_signature()}|{_build_narrator_filter_chain('EN')}|{_build_narrator_filter_chain('JP')}"
    return hashlib.sha1(spec.encode("utf-8")).hexdigest()


@app.get("/tts")
async def generate_tts(text: str = Query(...), speed: float = 1.0):
    start_time = time.time()

    # 1. Check Cache (Tier 1)
    cache_id = get_cache_key(text, speed)
    cache_path = CACHE_DIR / f"{cache_id}.wav"

    if cache_path.exists():
        return FileResponse(
            cache_path, media_type="audio/wav", headers={"X-TTS-Cache": "HIT"}
        )

    # 2. Tier 0 Synthesis
    try:
        segments = split_text_by_language(text)
        if not segments:
            return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)

        temp_files = set()
        rendered_files = []
        for i, (chunk, lang) in enumerate(segments):
            model = MODELS.get(lang, MODELS["EN"])

            # Robust Speaker ID lookup
            spk_id = 0
            try:
                # Access the underlying dict if it exists
                spk2id = {}
                hps = getattr(model, "hps", None)
                data = getattr(hps, "data", None)
                data_spk2id = getattr(data, "spk2id", None)
                hps_spk2id = getattr(hps, "spk2id", None)

                if isinstance(data_spk2id, dict):
                    spk2id = data_spk2id
                elif isinstance(hps_spk2id, dict):
                    spk2id = hps_spk2id

                target_key = f"{lang}-Default"
                if target_key in spk2id:
                    spk_id = spk2id[target_key]
                elif spk2id:
                    spk_id = list(spk2id.values())[0]
            except Exception:
                spk_id = 0

            seg_path = CACHE_DIR / f"seg_{cache_id}_{i}.wav"
            model.tts_to_file(chunk, spk_id, str(seg_path), speed=speed)
            temp_files.add(seg_path)

            rendered_path = seg_path
            if NARRATOR_UNIFIER_ENABLED:
                styled_path = CACHE_DIR / f"seg_{cache_id}_{i}_styled.wav"
                temp_files.add(styled_path)
                if _render_with_ffmpeg(
                    seg_path, styled_path, _build_narrator_filter_chain(lang)
                ):
                    rendered_path = styled_path
            rendered_files.append((rendered_path, lang))

        # 3. Concatenate (using sox or pydub if needed, but we'll use simple wave joining for Tier 0 speed)
        # Actually, let's use pydub for safer merging if available.
        import pydub

        combined = pydub.AudioSegment.empty()

        previous_lang = ""
        for segment_path, lang in rendered_files:
            audio = pydub.AudioSegment.from_wav(str(segment_path))
            if NARRATOR_UNIFIER_ENABLED:
                audio = _apply_envelope_shape(audio)
                audio = _normalize_segment_level(audio)

            if len(combined) == 0:
                fade_in_ms = min(NARRATOR_SWITCH_FADE_IN_MS, len(audio))
                if fade_in_ms > 0:
                    audio = audio.fade_in(fade_in_ms)
                combined += audio
            else:
                switched = previous_lang != lang
                if NARRATOR_UNIFIER_ENABLED:
                    combined = _apply_language_switch_smoothing(
                        combined,
                        audio,
                        switched_language=switched,
                        pydub_module=pydub,
                    )
                else:
                    combined += audio
            previous_lang = lang

        combined.export(str(cache_path), format="wav")

        for file_path in temp_files:
            file_path.unlink(missing_ok=True)

        duration = time.time() - start_time
        print(f"[tts] Synthesized '{text[:20]}...' in {duration:.2f}s")

        return FileResponse(
            cache_path,
            media_type="audio/wav",
            headers={"X-TTS-Cache": "MISS", "X-TTS-Latency": f"{duration:.3f}s"},
        )

    except Exception as e:
        for pattern in [f"seg_{cache_id}_*.wav", f"seg_{cache_id}_*_styled.wav"]:
            for candidate in CACHE_DIR.glob(pattern):
                candidate.unlink(missing_ok=True)
        print(f"[tts] Error: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8788)
