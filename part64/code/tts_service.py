import os
import re
import time
import torch
import hashlib
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

print(f"[tts] Initializing MeloTTS on {DEVICE}...")

# Warmup / Load Models
# We'll load them lazily or at start?
# Loading at start is better for Tier 0 latency.
MODELS = {
    "EN": TTS(language="EN", device=DEVICE),
    "JP": TTS(language="JP", device=DEVICE),
}


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
    spec = f"{text}|{speed}|v1"
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

        temp_files = []
        for i, (chunk, lang) in enumerate(segments):
            model = MODELS.get(lang, MODELS["EN"])

            # Robust Speaker ID lookup
            spk_id = 0
            try:
                # Access the underlying dict if it exists
                spk2id = {}
                if hasattr(model.hps, "data") and hasattr(model.hps.data, "spk2id"):
                    spk2id = model.hps.data.spk2id
                elif hasattr(model.hps, "spk2id"):
                    spk2id = model.hps.spk2id

                target_key = f"{lang}-Default"
                if target_key in spk2id:
                    spk_id = spk2id[target_key]
                elif spk2id:
                    spk_id = list(spk2id.values())[0]
            except Exception:
                spk_id = 0

            seg_path = CACHE_DIR / f"seg_{cache_id}_{i}.wav"
            model.tts_to_file(chunk, spk_id, str(seg_path), speed=speed)
            temp_files.append(seg_path)

        # 3. Concatenate (using sox or pydub if needed, but we'll use simple wave joining for Tier 0 speed)
        # Actually, let's use pydub for safer merging if available.
        import pydub

        combined = pydub.AudioSegment.empty()
        for f in temp_files:
            combined += pydub.AudioSegment.from_wav(str(f))
            f.unlink()  # clean up segment

        combined.export(str(cache_path), format="wav")

        duration = time.time() - start_time
        print(f"[tts] Synthesized '{text[:20]}...' in {duration:.2f}s")

        return FileResponse(
            cache_path,
            media_type="audio/wav",
            headers={"X-TTS-Cache": "MISS", "X-TTS-Latency": f"{duration:.3f}s"},
        )

    except Exception as e:
        print(f"[tts] Error: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8788)
