from __future__ import annotations

import importlib
from unittest.mock import patch

myth_bridge = importlib.import_module("code.myth_bridge")
MYTH_EVENT_TYPE = myth_bridge.MYTH_EVENT_TYPE
MythStateTracker = myth_bridge.MythStateTracker
add_mention = myth_bridge.add_mention
attribution = myth_bridge.attribution
build_mentions_from_catalog = myth_bridge.build_mentions_from_catalog
decay_ledger = myth_bridge.decay_ledger
fetch_remote_myth_snapshot = myth_bridge.fetch_remote_myth_snapshot


def approx(expected: float, actual: float) -> bool:
    return abs(float(expected) - float(actual)) <= 1.0e-9


def test_decay_and_addition_match_expected_coefficients() -> None:
    ledger = {("winter", "claim"): {"buzz": 10.0, "tradition": 5.0}}
    decayed = decay_ledger(ledger)
    assert approx(9.0, decayed[("winter", "claim")]["buzz"])
    assert approx(4.975, decayed[("winter", "claim")]["tradition"])

    updated = add_mention(
        {},
        {
            "event-type": "winter-pyre",
            "claim": "claim/one",
            "weight": 1.0,
            "event-instance": "evt-1",
        },
    )
    row = updated[("winter-pyre", "claim/one")]
    assert approx(1.0, row["buzz"])
    assert row["mentions"] == 1
    assert row["event-instances"] == {"evt-1"}


def test_attribution_normalizes_probabilities() -> None:
    ledger = {}
    ledger = add_mention(ledger, {"event-type": "winter", "claim": "a", "weight": 1.0})
    ledger = add_mention(ledger, {"event-type": "winter", "claim": "b", "weight": 3.0})
    probs = attribution(ledger, "winter")
    assert approx(1.0, probs["a"] + probs["b"])
    assert probs["b"] > probs["a"]


def test_mentions_builder_emits_cover_and_media_claims() -> None:
    catalog = {
        "cover_fields": [
            {"id": "receipt_river", "part": "64"},
            {"id": "witness_thread", "part": "64"},
        ],
        "counts": {"audio": 3, "image": 2, "video": 0},
    }
    mentions = build_mentions_from_catalog(catalog)
    cover_mentions = [m for m in mentions if m["event-type"] == MYTH_EVENT_TYPE]
    media_mentions = [m for m in mentions if m["event-type"] == "media_presence"]
    assert len(cover_mentions) == 2
    assert {m["claim"] for m in media_mentions} == {"audio", "image"}


def test_remote_snapshot_is_optional() -> None:
    with patch("os.getenv", return_value=""):
        assert fetch_remote_myth_snapshot() is None


def test_tracker_snapshot_contains_stable_shape() -> None:
    tracker = MythStateTracker()
    catalog = {
        "cover_fields": [{"id": "receipt_river", "part": "64"}],
        "counts": {"audio": 1, "image": 1, "video": 0},
    }
    snapshot = tracker.snapshot(catalog)
    assert snapshot["event_type"] == MYTH_EVENT_TYPE
    assert snapshot["ledger_size"] >= 1
    assert "cover_attribution" in snapshot
    assert "media_attribution" in snapshot
