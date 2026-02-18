from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


FIELD_ORDER = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
FIELD_NAMES = {
    "f1": "artifact_flux",
    "f2": "witness_tension",
    "f3": "coherence_focus",
    "f4": "drift_pressure",
    "f5": "fork_tax_balance",
    "f6": "curiosity_drive",
    "f7": "gate_pressure",
    "f8": "council_heat",
}


def _discover_export_dir() -> Path:
    explicit = os.getenv("TF_IMAGE_ROUTER_EXPORT_DIR", "").strip()
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate

    export_root = Path(os.getenv("TF_IMAGE_ROUTER_EXPORT_ROOT", "/output/export"))
    if export_root.exists() and export_root.is_dir():
        candidates = sorted(
            [p for p in export_root.glob("qwen3-vl-2b-image*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    raise RuntimeError(
        "No image-router export found. Set TF_IMAGE_ROUTER_EXPORT_DIR or run image training first."
    )


def _resolve_image_size(export_dir: Path) -> int:
    meta_path = export_dir / "field_router_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            image_size = int(meta.get("image_size", 224))
            return max(64, min(image_size, 1024))
        except Exception:
            return 224
    return 224


def _normalize_image(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    data = np.asarray(image, dtype=np.float32) / 255.0
    return data


def _decode_data_uri(data_uri: str) -> Optional[Image.Image]:
    marker = ";base64,"
    if marker not in data_uri:
        return None
    _, payload = data_uri.split(marker, 1)
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None
    try:
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


def _extract_image_source(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "image_url":
                    continue
                image_payload = block.get("image_url")
                if isinstance(image_payload, dict):
                    url = image_payload.get("url")
                    if isinstance(url, str) and url.strip():
                        return url.strip(), "image_url"
                elif isinstance(image_payload, str) and image_payload.strip():
                    return image_payload.strip(), "image_url"
        elif isinstance(content, str) and content.startswith("file://"):
            return content.strip(), "content_file_uri"
    return None, None


class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False


app = FastAPI(title="TF Image Router Bridge")

API_KEY = os.getenv("TF_IMAGE_ROUTER_API_KEY", "local-tensorflow-token")
MODEL_ID = os.getenv("TF_IMAGE_ROUTER_MODEL_ID", "qwen3-vl-2b-image")
EXPORT_DIR = _discover_export_dir()
IMAGE_SIZE = _resolve_image_size(EXPORT_DIR)
SAVED_MODEL = tf.saved_model.load(str(EXPORT_DIR))
INFER = SAVED_MODEL.signatures.get("serving_default")
if INFER is None:
    raise RuntimeError("SavedModel missing serving_default signature")
INPUT_KEYS = list(INFER.structured_input_signature[1].keys())
OUTPUT_KEYS = list(INFER.structured_outputs.keys())
INPUT_KEY = INPUT_KEYS[0] if INPUT_KEYS else "inputs"


def _verify_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _load_image_from_source(source: str) -> Optional[Image.Image]:
    if source.startswith("data:image/"):
        return _decode_data_uri(source)

    if source.startswith("file://"):
        source = source[7:]

    path = Path(source)
    if not path.is_absolute():
        path = Path("/vault") / path
    if not path.exists() or not path.is_file():
        return None

    try:
        return Image.open(path)
    except Exception:
        return None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "export_dir": str(EXPORT_DIR),
        "image_size": IMAGE_SIZE,
    }


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _verify_auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "tensorflow",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    payload: ChatRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _verify_auth(authorization)

    if payload.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported")
    if payload.model != MODEL_ID:
        raise HTTPException(
            status_code=400, detail=f"Unsupported model: {payload.model}"
        )

    image_source, source_kind = _extract_image_source(payload.messages)
    if not image_source:
        raise HTTPException(
            status_code=400,
            detail="No image source found. Pass an image via messages[].content[].image_url.url",
        )

    image = _load_image_from_source(image_source)
    if image is None:
        raise HTTPException(
            status_code=400, detail=f"Cannot load image: {image_source}"
        )

    pixels = _normalize_image(image, IMAGE_SIZE)
    batch = tf.convert_to_tensor(np.expand_dims(pixels, axis=0), dtype=tf.float32)
    outputs = INFER(**{INPUT_KEY: batch})

    logits = outputs.get("field_logits")
    if logits is None and OUTPUT_KEYS:
        logits = outputs.get(OUTPUT_KEYS[0])
    if logits is None:
        # Fallback: take the first tensor if signature key differs.
        first_value = next(iter(outputs.values()))
        logits = first_value

    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    probs = probs / max(float(np.sum(probs)), 1e-9)
    top_index = int(np.argmax(probs))
    dominant_field = FIELD_ORDER[top_index]

    field_scores = {
        field_id: float(round(float(probs[idx]), 6))
        for idx, field_id in enumerate(FIELD_ORDER)
    }
    summary = {
        "dominant_field": dominant_field,
        "dominant_name": FIELD_NAMES[dominant_field],
        "field_scores": field_scores,
        "source_kind": source_kind,
    }

    return {
        "id": "chatcmpl-tf-image-router",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(summary, ensure_ascii=True),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
