from __future__ import annotations

from code.world_web.muse_threat_fallback_strategy import (
    THREAT_FOCUSED_MUSE_IDS,
    build_muse_threat_fallback_reply,
)


def test_build_muse_threat_fallback_reply_formats_threats_and_sources() -> None:
    reply = build_muse_threat_fallback_reply(
        muse_id="chaos",
        user_text="what threats are active right now",
        manifest={
            "tet_units": [
                {
                    "kind": "threat",
                    "text": "Threat: Regional maritime incident\nRisk: CRITICAL (9)\nDomain: strait.example",
                },
                {
                    "kind": "threat-source",
                    "text": "Watch source: UKMTO advisory feed. URL: https://example.test/feed",
                },
            ]
        },
    )
    assert "Active global threat snapshot" in reply
    assert "CRITICAL(9) Regional maritime incident" in reply
    assert "Hot sources:" in reply
    assert "UKMTO advisory feed" in reply


def test_build_muse_threat_fallback_reply_rejects_non_threat_muse() -> None:
    assert "chaos" in THREAT_FOCUSED_MUSE_IDS
    reply = build_muse_threat_fallback_reply(
        muse_id="stability",
        user_text="what threats are active",
        manifest={
            "tet_units": [{"kind": "threat", "text": "Threat: sample\nRisk: HIGH (8)"}]
        },
    )
    assert reply == ""


def test_build_muse_threat_fallback_reply_requires_prompt_for_source_only_rows() -> (
    None
):
    promptless = build_muse_threat_fallback_reply(
        muse_id="witness_thread",
        user_text="hello there",
        manifest={
            "tet_units": [
                {"kind": "threat-source", "text": "Watch source: Feed Alpha."}
            ]
        },
    )
    prompted = build_muse_threat_fallback_reply(
        muse_id="witness_thread",
        user_text="show active security alerts",
        manifest={
            "tet_units": [
                {"kind": "threat-source", "text": "Watch source: Feed Alpha."}
            ]
        },
    )
    assert promptless == ""
    assert "Hot sources:" in prompted
    assert "Feed Alpha" in prompted
