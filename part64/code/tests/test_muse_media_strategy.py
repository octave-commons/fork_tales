from __future__ import annotations

from code.world_web.muse_media_strategy import (
    build_muse_media_candidates,
    build_muse_media_classifier_state,
    detect_muse_media_intent,
    resolve_muse_media_block_reason,
    resolve_muse_media_classifier_default_kind,
    resolve_muse_media_requested_kind,
)
from code.world_web.muse_runtime import (
    _AUDIO_ACTION_TOKENS,
    _AUDIO_HINT_TOKENS,
    _AUDIO_STOP_TOKENS,
    _AUDIO_SUFFIXES,
    _IMAGE_ACTION_TOKENS,
    _IMAGE_HINT_TOKENS,
    _IMAGE_STOP_TOKENS,
    _IMAGE_SUFFIXES,
)


def test_detect_muse_media_intent_finds_explicit_audio_request() -> None:
    intent = detect_muse_media_intent(
        text="play witness thread song mp3 now",
        audio_suffixes=_AUDIO_SUFFIXES,
        image_suffixes=_IMAGE_SUFFIXES,
        audio_action_tokens=_AUDIO_ACTION_TOKENS,
        image_action_tokens=_IMAGE_ACTION_TOKENS,
        audio_hint_tokens=_AUDIO_HINT_TOKENS,
        image_hint_tokens=_IMAGE_HINT_TOKENS,
        audio_stop_tokens=_AUDIO_STOP_TOKENS,
        image_stop_tokens=_IMAGE_STOP_TOKENS,
    )
    assert intent is not None
    assert intent.requested_kind == "audio"
    assert intent.strict_kind == "audio"
    assert "witness" in intent.query_tokens_by_kind["audio"]


def test_classifier_state_biases_default_kind_from_classifier_rows() -> None:
    state = build_muse_media_classifier_state(
        surrounding_nodes=[
            {
                "id": "presence.modality.baseline",
                "kind": "presence",
                "presence_type": "classifier",
                "default_media_kind": "image",
                "seed_terms": ["open", "play"],
                "tags": ["classifier", "image"],
            }
        ],
        token_set={"open", "play"},
    )
    assert state.classifier_bias["image"] > 0.0
    assert state.classifier_presence_ids == ["presence.modality.baseline"]
    assert (
        resolve_muse_media_requested_kind(
            requested_kind="",
            strict_kind="",
            classifier_bias=state.classifier_bias,
        )
        == "image"
    )
    assert (
        resolve_muse_media_classifier_default_kind(
            classifier_bias=state.classifier_bias,
        )
        == "image"
    )


def test_build_muse_media_candidates_applies_focus_bias_to_target_node() -> None:
    candidates = build_muse_media_candidates(
        surrounding_nodes=[
            {
                "id": "audio:target-canticle",
                "kind": "audio",
                "label": "Fork Tax Canticle",
                "text": "ritual canticle for fork tax",
                "source_rel_path": "artifacts/audio/fork_tax_canticle.mp3",
                "url": "/library/artifacts/audio/fork_tax_canticle.mp3",
                "tags": ["audio"],
            },
            {
                "id": "audio:distractor-loop",
                "kind": "audio",
                "label": "Chaos Loop",
                "text": "generic loop",
                "source_rel_path": "artifacts/audio/chaos_loop.mp3",
                "url": "/library/artifacts/audio/chaos_loop.mp3",
                "tags": ["workspace-pin", "chaos"],
            },
        ],
        muse_id="chaos",
        requested_kind="audio",
        strict_kind="audio",
        query_tokens_by_kind={
            "audio": ["fork", "tax", "canticle", "music"],
            "image": [],
        },
        explicit_selected=set(),
        tet_distance={},
        classifier_bias={"audio": 0.2, "image": 0.0},
        focus_node_bias={"audio": {"audio:target-canticle": 1.2}, "image": {}},
        audio_suffixes=_AUDIO_SUFFIXES,
        image_suffixes=_IMAGE_SUFFIXES,
        audio_target_presence_id="receipt_river",
        image_target_presence_id="mage_of_receipts",
        max_candidates=8,
    )
    assert candidates[0]["node_id"] == "audio:target-canticle"
    assert candidates[0]["media_kind"] == "audio"


def test_resolve_muse_media_block_reason_prefers_kind_specific_reasons() -> None:
    assert (
        resolve_muse_media_block_reason(strict_kind="audio", resolved_kind="")
        == "no_audio_candidates"
    )
    assert (
        resolve_muse_media_block_reason(strict_kind="", resolved_kind="image")
        == "no_image_candidates"
    )
    assert (
        resolve_muse_media_block_reason(strict_kind="", resolved_kind="")
        == "no_media_candidates"
    )
