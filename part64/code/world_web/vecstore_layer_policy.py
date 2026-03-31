from __future__ import annotations

from dataclasses import dataclass
import re


_LAYER_MODE_DISABLED: set[str] = {"", "single", "none", "off"}
_LAYER_MODE_SPACE_SIGNATURE: set[str] = {"space-signature", "space_signature"}
_LAYER_MODE_SPACE_MODEL: set[str] = {"space-model", "space_model"}


@dataclass(frozen=True)
class VecstoreLayerContext:
    base_collection: str
    mode: str
    space_id: str
    space_signature: str
    model_name: str


def sanitize_collection_name_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip().lower())
    cleaned = cleaned.strip("._-")
    return cleaned or "layer"


def resolve_vecstore_layer_token(
    *,
    mode: str,
    space_id: str,
    space_signature: str,
    model_name: str,
) -> str:
    mode_key = str(mode or "").strip()
    if mode_key in _LAYER_MODE_DISABLED:
        return ""
    if mode_key == "space":
        return sanitize_collection_name_token(space_id)
    if mode_key == "signature":
        return sanitize_collection_name_token(space_signature[:12])
    if mode_key == "model":
        return sanitize_collection_name_token(model_name)
    if mode_key in _LAYER_MODE_SPACE_SIGNATURE:
        return sanitize_collection_name_token(f"{space_id}_{space_signature[:10]}")
    if mode_key in _LAYER_MODE_SPACE_MODEL:
        return sanitize_collection_name_token(f"{space_id}_{model_name}")
    return sanitize_collection_name_token(mode_key)


def resolve_vecstore_collection_name(context: VecstoreLayerContext) -> str:
    base = str(context.base_collection or "").strip()
    token = resolve_vecstore_layer_token(
        mode=context.mode,
        space_id=context.space_id,
        space_signature=context.space_signature,
        model_name=context.model_name,
    )
    if not token:
        return base
    return f"{base}__{token}"
