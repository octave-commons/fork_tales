"""Document-layout helper utilities extracted from simulation.py."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from .metrics import _clamp01, _safe_float, _safe_int
from .simulation_file_graph_prep import bounded_text


def clean_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_-]+", text.lower()) if token]


def document_layout_range_from_importance(importance: float) -> float:
    normalized = _clamp01(_safe_float(importance, 0.2))
    return 0.018 + (normalized * 0.055)


def document_layout_tokens(
    node: dict[str, Any],
    *,
    summary_chars: int,
    excerpt_chars: int,
) -> list[str]:
    values: list[str] = []
    tags = node.get("tags", [])
    labels = node.get("labels", [])
    if isinstance(tags, list):
        values.extend(str(tag) for tag in tags)
    if isinstance(labels, list):
        values.extend(str(label) for label in labels)
    values.extend(
        [
            bounded_text(node.get("summary", ""), limit=summary_chars),
            bounded_text(node.get("text_excerpt", ""), limit=excerpt_chars),
            bounded_text(node.get("source_rel_path", ""), limit=160),
            bounded_text(node.get("archived_rel_path", ""), limit=160),
            bounded_text(node.get("archive_rel_path", ""), limit=160),
            bounded_text(node.get("name", ""), limit=160),
            bounded_text(node.get("kind", ""), limit=64),
            bounded_text(node.get("dominant_field", ""), limit=32),
            bounded_text(node.get("vecstore_collection", ""), limit=96),
        ]
    )

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in clean_tokens(value):
            if len(token) < 3:
                continue
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= 80:
                return deduped
    return deduped


def document_layout_text_density(node: dict[str, Any], tokens: list[str]) -> float:
    token_density = min(1.0, len(tokens) / 42.0)
    summary_len = len(str(node.get("summary", "")).strip())
    excerpt_len = len(str(node.get("text_excerpt", "")).strip())
    label_len = len(str(node.get("name", "")).strip()) + len(
        str(node.get("label", "")).strip()
    )
    char_density = min(1.0, (summary_len + excerpt_len + label_len) / 760.0)

    tags = node.get("tags", [])
    labels = node.get("labels", [])
    embedding_links = node.get("embedding_links", [])
    tag_count = len(tags) if isinstance(tags, list) else 0
    label_count = len(labels) if isinstance(labels, list) else 0
    link_count = len(embedding_links) if isinstance(embedding_links, list) else 0
    layer_count = _safe_int(node.get("embed_layer_count", 0), 0)
    structural = min(
        1.0,
        (tag_count * 0.08)
        + (label_count * 0.05)
        + (layer_count * 0.22)
        + (link_count * 0.04),
    )

    density = 0.2 + (token_density * 0.42) + (char_density * 0.26) + (structural * 0.36)
    return max(0.12, min(1.9, density))


def document_layout_semantic_vector(
    node: dict[str, Any],
    tokens: list[str],
    *,
    dimensions: int = 8,
) -> list[float]:
    if dimensions <= 0:
        return []

    raw_tokens = list(tokens)
    if not raw_tokens:
        raw_tokens = clean_tokens(
            " ".join(
                [
                    str(node.get("dominant_field", "")),
                    str(node.get("kind", "")),
                    str(node.get("vecstore_collection", "")),
                    str(node.get("name", "")),
                    str(node.get("label", "")),
                ]
            )
        )
    if not raw_tokens:
        raw_tokens = ["eta", "mu", "field"]

    accum = [0.0 for _ in range(dimensions)]
    for token_index, token in enumerate(raw_tokens[:96]):
        weight = 0.8 + min(1.6, len(token) / 6.5)
        digest = hashlib.sha1(
            f"{token}|{token_index}|{dimensions}".encode("utf-8")
        ).digest()
        for axis in range(dimensions):
            byte = digest[axis % len(digest)]
            signed = (float(byte) / 127.5) - 1.0
            accum[axis] += signed * weight

    field_token = str(node.get("dominant_field", "")).strip()
    kind_token = str(node.get("kind", "")).strip().lower()
    for marker, gain in ((field_token, 0.36), (kind_token, 0.22)):
        if not marker:
            continue
        digest = hashlib.sha1(f"marker:{marker}".encode("utf-8")).digest()
        for axis in range(dimensions):
            byte = digest[(axis * 3) % len(digest)]
            signed = (float(byte) / 127.5) - 1.0
            accum[axis] += signed * gain

    magnitude = math.sqrt(sum(value * value for value in accum))
    if magnitude <= 1e-8:
        fallback = [0.0 for _ in range(dimensions)]
        fallback[0] = 1.0
        return fallback
    return [value / magnitude for value in accum]


def semantic_vector_blend(
    base: list[float], target: list[float], blend: float
) -> list[float]:
    if not base and not target:
        return []
    if not base:
        return list(target)
    if not target:
        return list(base)

    mix = max(0.0, min(1.0, _safe_float(blend, 0.5)))
    size = min(len(base), len(target))
    if size <= 0:
        return list(base)

    merged = [(base[i] * (1.0 - mix)) + (target[i] * mix) for i in range(size)]
    magnitude = math.sqrt(sum(value * value for value in merged))
    if magnitude <= 1e-8:
        return [0.0 for _ in range(size)]
    return [value / magnitude for value in merged]


def semantic_vector_cosine(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(size))
    left_mag = sum(left[i] * left[i] for i in range(size))
    right_mag = sum(right[i] * right[i] for i in range(size))
    if left_mag <= 1e-12 or right_mag <= 1e-12:
        return 0.0
    cosine = dot / math.sqrt(left_mag * right_mag)
    return max(-1.0, min(1.0, cosine))


def semantic_vector_hue(vector: list[float]) -> float:
    if not vector:
        return 210.0
    vx = _safe_float(vector[0], 0.0)
    vy = _safe_float(vector[1], 0.0) if len(vector) > 1 else 0.0
    if abs(vx) <= 1e-8 and abs(vy) <= 1e-8:
        return 210.0
    return (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0


def document_layout_similarity(
    left_node: dict[str, Any],
    right_node: dict[str, Any],
    left_tokens: list[str],
    right_tokens: list[str],
) -> float:
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set.intersection(right_set))
    union = max(1, len(left_set) + len(right_set) - overlap)
    token_jaccard = overlap / float(union)

    left_field = str(left_node.get("dominant_field", "")).strip()
    right_field = str(right_node.get("dominant_field", "")).strip()
    same_field = 1.0 if left_field and left_field == right_field else 0.0

    field_repulsion = 0.0
    if left_field and right_field and left_field != right_field:
        field_repulsion = -0.35
        opposite_pairs = [
            ("f9", "f10"),
            ("f10", "f9"),
            ("f11", "f12"),
            ("f12", "f11"),
            ("f13", "f14"),
            ("f14", "f13"),
        ]
        if (left_field, right_field) in opposite_pairs:
            field_repulsion = -0.62

    same_kind = (
        1.0
        if str(left_node.get("kind", "")).strip().lower()
        and str(left_node.get("kind", "")).strip().lower()
        == str(right_node.get("kind", "")).strip().lower()
        else 0.0
    )
    left_collection = str(left_node.get("vecstore_collection", "")).strip()
    right_collection = str(right_node.get("vecstore_collection", "")).strip()
    same_collection = (
        1.0 if left_collection and left_collection == right_collection else 0.0
    )

    score = (
        (token_jaccard * 0.78)
        + (same_field * 0.12)
        + (same_kind * 0.06)
        + (same_collection * 0.04)
        + field_repulsion
    )

    if token_jaccard < 0.05 and same_field <= 0.0 and same_kind <= 0.0:
        score *= 0.25

    return _clamp01(score)


def document_layout_is_embedded(node: dict[str, Any]) -> bool:
    if _safe_int(node.get("embed_layer_count", 0), 0) > 0:
        return True

    layer_points = node.get("embed_layer_points", [])
    if isinstance(layer_points, list):
        for row in layer_points:
            if not isinstance(row, dict):
                continue
            if bool(row.get("active", True)):
                return True

    if str(node.get("vecstore_collection", "")).strip():
        return True

    embedding_links = node.get("embedding_links", [])
    if isinstance(embedding_links, list) and len(embedding_links) > 0:
        return True

    return False
