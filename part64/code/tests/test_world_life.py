from __future__ import annotations

import importlib


world_life = importlib.import_module("code.world_life")
LifeStateTracker = world_life.LifeStateTracker
build_interaction_response = world_life.build_interaction_response


def test_world_life_snapshot_contains_people_songs_and_books() -> None:
    tracker = LifeStateTracker()
    catalog = {
        "counts": {"audio": 3, "image": 1, "video": 0},
        "cover_fields": [{"id": "receipt_river", "part": "64"}],
    }
    myth = {"top_cover_claim": "receipt_river", "top_cover_weight": 0.7}
    entities = [
        {
            "id": "receipt_river",
            "en": "Receipt River",
            "ja": "領収書の川",
            "type": "flow",
        },
        {
            "id": "gates_of_truth",
            "en": "Gates of Truth",
            "ja": "真理の門",
            "type": "portal",
        },
    ]

    snapshot = tracker.snapshot(catalog, myth, entities)

    assert snapshot["tick"] >= 1
    assert isinstance(snapshot["people"], list)
    assert len(snapshot["people"]) >= 3
    assert isinstance(snapshot["songs"], list)
    assert len(snapshot["songs"]) == len(snapshot["people"])
    assert isinstance(snapshot["presences"], list)
    assert len(snapshot["presences"]) == 2


def test_world_life_writes_books_over_time() -> None:
    tracker = LifeStateTracker()
    catalog = {"counts": {"audio": 1, "image": 0, "video": 0}, "cover_fields": []}
    myth = {"top_cover_claim": "witness_thread", "top_cover_weight": 0.4}
    entities: list[dict[str, str]] = []

    last = {}
    for _ in range(9):
        last = tracker.snapshot(catalog, myth, entities)

    assert len(last["books"]) >= 1
    assert last["books"][-1]["title"]["en"].startswith("Chronicle of")


def test_world_life_interaction_response_uses_presence_and_action() -> None:
    tracker = LifeStateTracker()
    catalog = {"counts": {"audio": 2, "image": 0, "video": 0}, "cover_fields": []}
    myth = {"top_cover_claim": "gates_of_truth", "top_cover_weight": 0.55}
    entities = [
        {
            "id": "witness_thread",
            "en": "Witness Thread",
            "ja": "証人の糸",
            "type": "network",
        }
    ]
    snapshot = tracker.snapshot(catalog, myth, entities)

    response = build_interaction_response(snapshot, "scribe_aya", "pray")

    assert response["ok"] is True
    assert response["action"] == "pray"
    assert "Witness Thread" in response["line_en"]
    assert "証人の糸" in response["line_ja"]


def test_world_life_interaction_handles_empty_world() -> None:
    response = build_interaction_response({}, "unknown", "speak")
    assert response["ok"] is False
    assert response["error"] == "world_has_no_people"
