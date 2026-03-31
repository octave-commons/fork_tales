from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import re
from typing import Any


_MEDIA_AUDIO_CANDIDATE_TOKENS = {"song", "track", "music", "audio", "bpm"}
_MEDIA_IMAGE_CANDIDATE_TOKENS = {
    "image",
    "photo",
    "picture",
    "cover",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
    "svg",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _hash_unit(value: str) -> float:
    token = sha1(str(value or "").encode("utf-8")).hexdigest()[:10]
    raw = int(token, 16)
    return raw / float(0xFFFFFFFFFF)


@dataclass(frozen=True)
class MuseMediaIntent:
    requested_kind: str
    strict_kind: str
    token_list: list[str]
    token_set: set[str]
    query_tokens_by_kind: dict[str, list[str]]


@dataclass(frozen=True)
class MuseMediaClassifierState:
    classifier_bias: dict[str, float]
    focus_node_bias: dict[str, dict[str, float]]
    classifier_presence_ids: list[str]


def detect_muse_media_intent(
    *,
    text: str,
    audio_suffixes: tuple[str, ...],
    image_suffixes: tuple[str, ...],
    audio_action_tokens: set[str],
    image_action_tokens: set[str],
    audio_hint_tokens: set[str],
    image_hint_tokens: set[str],
    audio_stop_tokens: set[str],
    image_stop_tokens: set[str],
) -> MuseMediaIntent | None:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None

    token_list = re.findall(r"[a-z0-9_./:-]+", normalized)
    token_set = set(token_list)

    audio_suffix_hit = any(
        token.endswith(audio_suffixes) or token in {"mp3", "wav", "ogg", "m4a", "flac"}
        for token in token_set
    )
    image_suffix_hit = any(
        token.endswith(image_suffixes)
        or token in {"png", "jpg", "jpeg", "webp", "gif", "svg", "bmp"}
        for token in token_set
    )
    audio_action_hit = bool(token_set & set(audio_action_tokens))
    image_action_hit = bool(token_set & set(image_action_tokens))
    audio_hint_hit = bool(token_set & set(audio_hint_tokens))
    image_hint_hit = bool(token_set & set(image_hint_tokens))

    explicit_audio = (
        normalized.startswith("/play")
        or normalized.startswith("play ")
        or ("play" in token_set and bool(token_set & set(audio_hint_tokens)))
    )
    explicit_image = (
        normalized.startswith("/image")
        or normalized.startswith("/open-image")
        or normalized.startswith("open image")
        or (("open" in token_set or "show" in token_set) and image_hint_hit)
    )

    audio_signal = 0
    image_signal = 0
    if audio_action_hit:
        audio_signal += 2
    if audio_hint_hit:
        audio_signal += 3
    if audio_suffix_hit:
        audio_signal += 3
    if explicit_audio:
        audio_signal += 4
    if image_action_hit:
        image_signal += 2
    if image_hint_hit:
        image_signal += 3
    if image_suffix_hit:
        image_signal += 3
    if explicit_image:
        image_signal += 4

    if audio_signal <= 0 and image_signal <= 0:
        return None

    requested_kind = ""
    if audio_signal > image_signal:
        requested_kind = "audio"
    elif image_signal > audio_signal:
        requested_kind = "image"

    strict_kind = ""
    if explicit_audio and not explicit_image:
        strict_kind = "audio"
    elif explicit_image and not explicit_audio:
        strict_kind = "image"

    return MuseMediaIntent(
        requested_kind=requested_kind,
        strict_kind=strict_kind,
        token_list=token_list,
        token_set=token_set,
        query_tokens_by_kind={
            "audio": [
                token
                for token in token_list
                if token not in set(audio_stop_tokens) and len(token) > 1
            ][:12],
            "image": [
                token
                for token in token_list
                if token not in set(image_stop_tokens) and len(token) > 1
            ][:12],
        },
    )


def build_muse_media_classifier_state(
    *,
    surrounding_nodes: list[dict[str, Any]],
    token_set: set[str],
) -> MuseMediaClassifierState:
    classifier_bias = {"audio": 0.0, "image": 0.0}
    focus_node_bias = {"audio": {}, "image": {}}
    classifier_presence_ids: list[str] = []

    for raw_row in surrounding_nodes if isinstance(surrounding_nodes, list) else []:
        if not isinstance(raw_row, dict):
            continue
        row_id = str(
            raw_row.get(
                "id", raw_row.get("node_id", raw_row.get("source_rel_path", ""))
            )
            or ""
        ).strip()
        row_kind = str(raw_row.get("kind", "") or "").strip().lower()
        presence_type = str(raw_row.get("presence_type", "") or "").strip().lower()
        tags = {
            str(item).strip().lower()
            for item in (
                raw_row.get("tags", []) if isinstance(raw_row.get("tags"), list) else []
            )
            if str(item).strip()
        }
        seed_terms = {
            str(item).strip().lower()
            for item in (
                raw_row.get("seed_terms", [])
                if isinstance(raw_row.get("seed_terms"), list)
                else []
            )
            if str(item).strip()
        }
        if not seed_terms:
            inferred_terms = re.findall(
                r"[a-z0-9_./:-]+",
                " ".join(
                    [
                        str(raw_row.get("label", "") or ""),
                        str(raw_row.get("text", "") or ""),
                        str(raw_row.get("source_rel_path", "") or ""),
                    ]
                ).lower(),
            )
            seed_terms = {token for token in inferred_terms if len(token) > 1}

        focus_ids = [
            str(item).strip()
            for item in (
                raw_row.get("focus_node_ids", [])
                if isinstance(raw_row.get("focus_node_ids"), list)
                else []
            )
            if str(item).strip()
        ]
        default_kind = (
            str(raw_row.get("default_media_kind", raw_row.get("media_kind", "")) or "")
            .strip()
            .lower()
        )
        if default_kind not in {"audio", "image"}:
            continue

        overlap = len(seed_terms.intersection(token_set))
        is_classifier_row = bool(
            row_kind == "classifier"
            or presence_type
            in {"classifier", "modality_classifier", "baseline_classifier"}
            or "classifier" in tags
        )
        if is_classifier_row:
            classifier_bias[default_kind] += 0.14 + min(0.42, overlap * 0.1)
            if row_id and row_id not in classifier_presence_ids:
                classifier_presence_ids.append(row_id)

        if focus_ids:
            focus_boost = 0.32 + min(1.4, overlap * 0.28)
            bias_map = focus_node_bias[default_kind]
            for focus_id in focus_ids:
                current = _safe_float(bias_map.get(focus_id, 0.0), 0.0)
                bias_map[focus_id] = current + focus_boost
        elif "concept-seed" in tags and overlap > 0:
            classifier_bias[default_kind] += min(0.24, overlap * 0.08)

    return MuseMediaClassifierState(
        classifier_bias=classifier_bias,
        focus_node_bias=focus_node_bias,
        classifier_presence_ids=classifier_presence_ids,
    )


def resolve_muse_media_requested_kind(
    *,
    requested_kind: str,
    strict_kind: str,
    classifier_bias: dict[str, float],
) -> str:
    if strict_kind or requested_kind:
        return str(requested_kind or "")
    audio_bias = _safe_float(classifier_bias.get("audio", 0.0), 0.0)
    image_bias = _safe_float(classifier_bias.get("image", 0.0), 0.0)
    if audio_bias > image_bias + 0.04:
        return "audio"
    if image_bias > audio_bias + 0.04:
        return "image"
    return ""


def resolve_muse_media_classifier_default_kind(
    *, classifier_bias: dict[str, float]
) -> str:
    audio_bias = _safe_float(classifier_bias.get("audio", 0.0), 0.0)
    image_bias = _safe_float(classifier_bias.get("image", 0.0), 0.0)
    if audio_bias > image_bias:
        return "audio"
    if image_bias > audio_bias:
        return "image"
    return ""


def _resolved_media_url(raw_url: str, source_rel_path: str) -> str:
    resolved_url = str(raw_url or "").strip()
    clean_source_rel_path = str(source_rel_path or "").strip()
    if resolved_url and not resolved_url.startswith(("http://", "https://", "/")):
        return "/" + resolved_url.lstrip("./")
    if resolved_url:
        return resolved_url
    if not clean_source_rel_path:
        return ""
    clean_rel = clean_source_rel_path.lstrip("/")
    if clean_rel.startswith("library/"):
        return "/" + clean_rel
    return "/library/" + clean_rel


def build_muse_media_candidates(
    *,
    surrounding_nodes: list[dict[str, Any]],
    muse_id: str,
    requested_kind: str,
    strict_kind: str,
    query_tokens_by_kind: dict[str, list[str]],
    explicit_selected: set[str],
    tet_distance: dict[str, float],
    classifier_bias: dict[str, float],
    focus_node_bias: dict[str, dict[str, float]],
    audio_suffixes: tuple[str, ...],
    image_suffixes: tuple[str, ...],
    audio_target_presence_id: str,
    image_target_presence_id: str,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    clean_muse_id = str(muse_id or "").strip()

    for raw_row in surrounding_nodes if isinstance(surrounding_nodes, list) else []:
        if not isinstance(raw_row, dict):
            continue
        node_id = str(
            raw_row.get(
                "id", raw_row.get("node_id", raw_row.get("source_rel_path", ""))
            )
            or ""
        ).strip()
        if not node_id or node_id in seen_ids:
            continue
        seen_ids.add(node_id)

        kind = str(raw_row.get("kind", "resource") or "resource").strip().lower()
        label = str(raw_row.get("label", node_id) or node_id).strip() or node_id
        text_blob = str(raw_row.get("text", "") or "")
        source_rel_path = str(raw_row.get("source_rel_path", "") or "").strip()
        raw_url = str(raw_row.get("url", "") or "").strip()
        tags = [
            str(item).strip().lower()
            for item in (
                raw_row.get("tags", []) if isinstance(raw_row.get("tags"), list) else []
            )
            if str(item).strip()
        ]
        joined = " ".join(
            [node_id, label, text_blob, source_rel_path, raw_url, " ".join(tags)]
        ).lower()
        corpus_tokens = set(re.findall(r"[a-z0-9_./:-]+", joined))

        kind_audio = (
            kind == "audio"
            or kind.startswith("audio/")
            or "music" in kind
            or "song" in kind
        )
        suffix_audio = any(
            str(field).strip().lower().endswith(audio_suffixes)
            for field in (node_id, label, source_rel_path, raw_url)
            if str(field).strip()
        )
        path_audio = "/artifacts/audio/" in joined or "artifacts/audio/" in joined
        lexical_audio = bool(_MEDIA_AUDIO_CANDIDATE_TOKENS & corpus_tokens)
        is_audio_candidate = bool(
            kind_audio or suffix_audio or path_audio or lexical_audio
        )

        kind_image = (
            kind == "image"
            or kind.startswith("image/")
            or kind == "cover_art"
            or "image" in kind
        )
        suffix_image = any(
            str(field).strip().lower().endswith(image_suffixes)
            for field in (node_id, label, source_rel_path, raw_url)
            if str(field).strip()
        )
        path_image = "/artifacts/images/" in joined or "artifacts/images/" in joined
        lexical_image = bool(_MEDIA_IMAGE_CANDIDATE_TOKENS & corpus_tokens)
        is_image_candidate = bool(
            kind_image or suffix_image or path_image or lexical_image
        )

        if strict_kind == "audio":
            is_image_candidate = False
        elif strict_kind == "image":
            is_audio_candidate = False

        if requested_kind == "audio" and not strict_kind:
            is_image_candidate = False
        elif requested_kind == "image" and not strict_kind:
            is_audio_candidate = False

        resolved_url = _resolved_media_url(raw_url, source_rel_path)

        if is_audio_candidate:
            audio_overlap = len(
                corpus_tokens.intersection(set(query_tokens_by_kind.get("audio", [])))
            )
            audio_score = 0.0
            if kind_audio:
                audio_score += 1.0
            if suffix_audio:
                audio_score += 0.72
            if path_audio:
                audio_score += 0.42
            if node_id in explicit_selected:
                audio_score += 0.78
            if "workspace-pin" in tags:
                audio_score += 0.34
            if clean_muse_id.lower() in tags:
                audio_score += 0.24
            if node_id in tet_distance:
                audio_score += (
                    1.0 - _clamp01(_safe_float(tet_distance[node_id], 1.0))
                ) * 0.26
            audio_score += min(0.84, audio_overlap * 0.18)
            audio_score += min(
                0.55, _safe_float(classifier_bias.get("audio", 0.0), 0.0)
            )
            audio_score += min(
                0.9,
                _safe_float(
                    focus_node_bias.get("audio", {}).get(node_id, 0.0)
                    if isinstance(focus_node_bias.get("audio"), dict)
                    else 0.0,
                    0.0,
                ),
            )
            audio_score += _hash_unit(f"audio|{clean_muse_id}|{node_id}") * 0.04
            candidates.append(
                {
                    "media_kind": "audio",
                    "intent": "play_music",
                    "node_id": node_id,
                    "label": label,
                    "kind": kind,
                    "score": audio_score,
                    "url": resolved_url,
                    "source_rel_path": source_rel_path,
                    "target_presence_id": audio_target_presence_id,
                }
            )

        if is_image_candidate:
            image_overlap = len(
                corpus_tokens.intersection(set(query_tokens_by_kind.get("image", [])))
            )
            image_score = 0.0
            if kind_image:
                image_score += 1.0
            if suffix_image:
                image_score += 0.72
            if path_image:
                image_score += 0.42
            if node_id in explicit_selected:
                image_score += 0.78
            if "workspace-pin" in tags:
                image_score += 0.34
            if clean_muse_id.lower() in tags:
                image_score += 0.24
            if node_id in tet_distance:
                image_score += (
                    1.0 - _clamp01(_safe_float(tet_distance[node_id], 1.0))
                ) * 0.26
            image_score += min(0.84, image_overlap * 0.18)
            image_score += min(
                0.55, _safe_float(classifier_bias.get("image", 0.0), 0.0)
            )
            image_score += min(
                0.9,
                _safe_float(
                    focus_node_bias.get("image", {}).get(node_id, 0.0)
                    if isinstance(focus_node_bias.get("image"), dict)
                    else 0.0,
                    0.0,
                ),
            )
            image_score += _hash_unit(f"image|{clean_muse_id}|{node_id}") * 0.04
            candidates.append(
                {
                    "media_kind": "image",
                    "intent": "open_image",
                    "node_id": node_id,
                    "label": label,
                    "kind": kind,
                    "score": image_score,
                    "url": resolved_url,
                    "source_rel_path": source_rel_path,
                    "target_presence_id": image_target_presence_id,
                }
            )

    candidates.sort(
        key=lambda row: (
            -_safe_float(row.get("score", 0.0), 0.0),
            str(row.get("media_kind", "")),
            str(row.get("node_id", "")),
        )
    )
    limited = candidates[: max(0, int(max_candidates))]
    if strict_kind:
        return [row for row in limited if str(row.get("media_kind", "")) == strict_kind]
    return limited


def resolve_muse_media_block_reason(*, strict_kind: str, resolved_kind: str) -> str:
    if strict_kind == "audio" or resolved_kind == "audio":
        return "no_audio_candidates"
    if strict_kind == "image" or resolved_kind == "image":
        return "no_image_candidates"
    return "no_media_candidates"
