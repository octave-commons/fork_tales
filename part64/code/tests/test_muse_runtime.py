from __future__ import annotations

from typing import Any

from code.world_web.muse_runtime import (
    BOOTSTRAP_MUSE_SPECS,
    DEFAULT_MUSE_ID,
    InMemoryMuseStorage,
    MuseRuntimeManager,
)


def _manager(*, turns_per_minute: int = 24) -> MuseRuntimeManager:
    return MuseRuntimeManager(
        storage=InMemoryMuseStorage(),
        enabled=True,
        backend_name="memory",
        turns_per_minute=turns_per_minute,
        max_particles_per_turn=8,
        max_events=512,
        max_history_messages=96,
        max_manifests_per_muse=48,
        audio_intent_enabled=True,
        audio_min_score=0.55,
        audio_daimoi_fanout=3,
        audio_max_candidates=32,
        audio_target_presence_id="receipt_river",
        image_target_presence_id="mage_of_receipts",
    )


def _reply_builder(**_: Any) -> dict[str, Any]:
    return {
        "reply": "stable muse reply",
        "mode": "canonical",
        "model": "unit",
    }


def test_muse_pause_resume_and_rate_limit() -> None:
    manager = _manager(turns_per_minute=1)

    paused = manager.set_pause(
        DEFAULT_MUSE_ID,
        paused=True,
        reason="unit-test",
        user_intent_id="intent:pause",
    )
    assert paused["ok"] is True

    blocked = manager.send_message(
        muse_id=DEFAULT_MUSE_ID,
        text="hello while paused",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="",
        graph_revision="graph:v1",
        surrounding_nodes=[],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="s1",
    )
    assert blocked["ok"] is False
    assert blocked["error"] == "muse_paused"
    assert blocked["status_code"] == 409

    resumed = manager.set_pause(
        DEFAULT_MUSE_ID,
        paused=False,
        reason="unit-test",
        user_intent_id="intent:resume",
    )
    assert resumed["ok"] is True

    first = manager.send_message(
        muse_id=DEFAULT_MUSE_ID,
        text="hello after resume",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="k1",
        graph_revision="graph:v2",
        surrounding_nodes=[],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="s2",
    )
    assert first["ok"] is True

    limited = manager.send_message(
        muse_id=DEFAULT_MUSE_ID,
        text="second turn in same minute",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="k2",
        graph_revision="graph:v2",
        surrounding_nodes=[],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="s3",
    )
    assert limited["ok"] is False
    assert limited["error"] == "rate_limited"
    assert limited["status_code"] == 429

    kinds = [row["kind"] for row in manager.list_events(limit=200)]
    assert "muse.paused" in kinds
    assert "muse.resumed" in kinds
    assert "muse.rate_limited" in kinds


def test_muse_dedupe_tool_events_and_sealed_compliance_drop() -> None:
    manager = _manager()
    tool_calls: list[str] = []

    def _tool_callback(*, tool_name: str) -> dict[str, Any]:
        tool_calls.append(tool_name)
        return {"ok": True, "summary": f"ran:{tool_name}"}

    payload = manager.send_message(
        muse_id=DEFAULT_MUSE_ID,
        text="/study then report drift",
        mode="deterministic",
        token_budget=900,
        idempotency_key="idem-1",
        graph_revision="graph:v1",
        surrounding_nodes=[
            {
                "id": "node:sealed",
                "kind": "resource",
                "text": "should never be selected",
                "visibility": "sealed",
                "x": 0.5,
                "y": 0.5,
            },
            {
                "id": "node:public",
                "kind": "resource",
                "text": "public evidence",
                "visibility": "public",
                "x": 0.51,
                "y": 0.51,
            },
        ],
        tool_callback=_tool_callback,
        reply_builder=_reply_builder,
        seed="seed-1",
    )
    assert payload["ok"] is True
    assert payload["reply"] == "stable muse reply"
    assert tool_calls
    dropped = payload["manifest"].get("dropped", [])
    assert any(
        str(row.get("node_id", "")) == "node:sealed"
        and str(row.get("reason", "")) == "sealed_visibility"
        for row in dropped
        if isinstance(row, dict)
    )

    deduped = manager.send_message(
        muse_id=DEFAULT_MUSE_ID,
        text="same idempotency should replay",
        mode="deterministic",
        token_budget=900,
        idempotency_key="idem-1",
        graph_revision="graph:v1",
        surrounding_nodes=[],
        tool_callback=_tool_callback,
        reply_builder=_reply_builder,
        seed="seed-2",
    )
    assert deduped["ok"] is True
    assert deduped["turn_id"] == payload["turn_id"]

    kinds = [row["kind"] for row in manager.list_events(limit=240)]
    assert "muse.tool.requested" in kinds
    assert "muse.tool.result" in kinds
    assert "muse.message.deduped" in kinds


def test_muse_tool_request_parses_facts_and_graph_commands() -> None:
    manager = _manager()
    tool_calls: list[str] = []

    def _tool_callback(*, tool_name: str) -> dict[str, Any]:
        tool_calls.append(tool_name)
        if tool_name == "facts_snapshot":
            return {
                "ok": True,
                "summary": "facts snapshot generated",
                "snapshot_hash": "f" * 64,
                "snapshot_path": "/tmp/facts.json",
                "node_count": 3,
                "edge_count": 2,
            }
        if tool_name.startswith("graph:"):
            return {
                "ok": True,
                "summary": "graph query generated",
                "query": "overview",
                "snapshot_hash": "g" * 64,
                "result_count": 2,
            }
        return {"ok": True, "summary": f"ran:{tool_name}"}

    payload = manager.send_message(
        muse_id=DEFAULT_MUSE_ID,
        text="/facts then /graph overview",
        mode="deterministic",
        token_budget=900,
        idempotency_key="facts-graph-1",
        graph_revision="graph:v-facts",
        surrounding_nodes=[],
        tool_callback=_tool_callback,
        reply_builder=_reply_builder,
        seed="seed-facts-graph",
    )
    assert payload["ok"] is True
    assert "facts_snapshot" in tool_calls
    assert "graph:overview" in tool_calls
    assert str(payload.get("reply", "")).startswith("Facts grounded at")
    receipts = payload.get("grounded_receipts", {})
    assert receipts.get("snapshot_hash") == "f" * 64
    assert "overview" in receipts.get("queries_used", [])


def test_muse_graph_neighbors_command_carries_argument_tail() -> None:
    manager = _manager()
    tool_calls: list[str] = []

    def _tool_callback(*, tool_name: str) -> dict[str, Any]:
        tool_calls.append(tool_name)
        return {"ok": True, "summary": f"ran:{tool_name}"}

    payload = manager.send_message(
        muse_id=DEFAULT_MUSE_ID,
        text="/graph neighbors url:aaaa",
        mode="deterministic",
        token_budget=700,
        idempotency_key="graph-neighbors-1",
        graph_revision="graph:v-neighbors",
        surrounding_nodes=[],
        tool_callback=_tool_callback,
        reply_builder=_reply_builder,
        seed="seed-neighbors",
    )
    assert payload["ok"] is True
    assert "graph:neighbors url:aaaa" in tool_calls


def test_bootstrap_fixed_muses_are_present_and_typed() -> None:
    manager = _manager()
    rows = manager.list_muses()
    muse_by_id = {
        str(row.get("id", "")).strip(): row for row in rows if isinstance(row, dict)
    }
    expected_ids = {
        str(spec.get("id", "")).strip()
        for spec in BOOTSTRAP_MUSE_SPECS
        if str(spec.get("id", "")).strip()
    }
    assert expected_ids.issubset(set(muse_by_id.keys()))
    for muse_id in expected_ids:
        row = muse_by_id[muse_id]
        assert row.get("presence_type") == "muse"
        assert row.get("panel_id") == f"nexus.ui.chat.{muse_id}"


def test_chaos_pinned_node_is_explicit_context() -> None:
    manager = _manager()
    pin_result = manager.pin_node(
        "chaos",
        node_id="node:seeded:chaos",
        user_intent_id="intent:pin",
        reason="unit-test",
    )
    assert pin_result["ok"] is True

    payload = manager.send_message(
        muse_id="chaos",
        text="use my pinned node",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="",
        graph_revision="graph:v3",
        surrounding_nodes=[],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="seed-chaos",
    )
    assert payload["ok"] is True
    explicit_selected = payload["manifest"].get("explicit_selected", [])
    assert "node:seeded:chaos" in explicit_selected


def test_play_song_path_requests_audio_and_routes_daimoi() -> None:
    manager = _manager()
    payload = manager.send_message(
        muse_id="chaos",
        text="play witness thread song mp3 now",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="song-1",
        graph_revision="graph:v-song",
        surrounding_nodes=[
            {
                "id": "audio:witness-thread",
                "kind": "audio",
                "label": "Witness Thread 78 BPM",
                "text": "artifact track for witness thread",
                "x": 0.5,
                "y": 0.22,
                "source_rel_path": "artifacts/audio/witness_thread_78bpm.mp3",
                "url": "/library/artifacts/audio/witness_thread_78bpm.mp3",
                "tags": ["chaos", "workspace-pin"],
            },
            {
                "id": "note:non-audio",
                "kind": "resource",
                "label": "note",
                "text": "not a song",
                "x": 0.2,
                "y": 0.4,
            },
        ],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="song-seed",
    )
    assert payload["ok"] is True
    media_actions = payload.get("media_actions", [])
    assert isinstance(media_actions, list)
    assert len(media_actions) == 1
    assert media_actions[0].get("media_kind") == "audio"
    audio_actions = payload.get("audio_actions", [])
    assert isinstance(audio_actions, list)
    assert len(audio_actions) == 1
    action = audio_actions[0]
    assert action.get("status") == "requested"
    assert action.get("selected_node_id") == "audio:witness-thread"
    assert str(action.get("selected_url", "")).endswith("witness_thread_78bpm.mp3")
    assert str(action.get("target_presence_id", "")) == "receipt_river"
    assert int(action.get("collision_count", 0)) > 0

    assert any(
        str(row.get("intent", "")) == "audio.play"
        and str(row.get("target_node_id", "")) == "audio:witness-thread"
        for row in payload.get("daimoi", [])
        if isinstance(row, dict)
    )

    kinds = [row["kind"] for row in manager.list_events(limit=400)]
    assert "muse.audio.intent.detected" in kinds
    assert "muse.daimoi.audio.collided" in kinds
    assert "audio.play.requested" in kinds


def test_open_image_path_requests_image_and_routes_daimoi() -> None:
    manager = _manager()
    payload = manager.send_message(
        muse_id="stability",
        text="open image cover now",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="image-1",
        graph_revision="graph:v-image",
        surrounding_nodes=[
            {
                "id": "image:cover",
                "kind": "image",
                "label": "Stability Cover",
                "text": "cover image for stability",
                "x": 0.52,
                "y": 0.2,
                "source_rel_path": "artifacts/images/stability_cover.png",
                "url": "/library/artifacts/images/stability_cover.png",
                "tags": ["stability", "workspace-pin"],
            },
            {
                "id": "audio:reference",
                "kind": "audio",
                "label": "Reference Song",
                "text": "supporting audio",
                "x": 0.43,
                "y": 0.42,
                "source_rel_path": "artifacts/audio/reference.mp3",
                "url": "/library/artifacts/audio/reference.mp3",
            },
        ],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="image-seed",
    )
    assert payload["ok"] is True
    media_actions = payload.get("media_actions", [])
    assert isinstance(media_actions, list)
    assert len(media_actions) == 1
    action = media_actions[0]
    assert action.get("status") == "requested"
    assert action.get("media_kind") == "image"
    assert action.get("selected_node_id") == "image:cover"
    assert str(action.get("selected_url", "")).endswith("stability_cover.png")
    assert str(action.get("target_presence_id", "")) == "mage_of_receipts"
    assert int(action.get("collision_count", 0)) > 0
    assert payload.get("audio_actions", []) == []

    assert any(
        str(row.get("intent", "")) == "image.open"
        and str(row.get("target_node_id", "")) == "image:cover"
        for row in payload.get("daimoi", [])
        if isinstance(row, dict)
    )

    kinds = [row["kind"] for row in manager.list_events(limit=400)]
    assert "muse.image.intent.detected" in kinds
    assert "muse.daimoi.image.collided" in kinds
    assert "image.open.requested" in kinds


def test_play_song_path_blocks_when_no_audio_candidates() -> None:
    manager = _manager()
    payload = manager.send_message(
        muse_id="stability",
        text="play a song",
        mode="deterministic",
        token_budget=900,
        idempotency_key="song-2",
        graph_revision="graph:v-song",
        surrounding_nodes=[
            {
                "id": "note:study",
                "kind": "resource",
                "label": "study note",
                "text": "drift and receipts",
                "x": 0.3,
                "y": 0.4,
            }
        ],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="song-seed-2",
    )
    assert payload["ok"] is True
    audio_actions = payload.get("audio_actions", [])
    assert isinstance(audio_actions, list)
    assert len(audio_actions) == 1
    action = audio_actions[0]
    assert action.get("status") == "blocked"
    assert action.get("reason") == "no_audio_candidates"

    kinds = [row["kind"] for row in manager.list_events(limit=400)]
    assert "muse.audio.intent.detected" in kinds
    assert "audio.play.blocked" in kinds


def test_open_image_path_blocks_when_no_image_candidates() -> None:
    manager = _manager()
    payload = manager.send_message(
        muse_id="chaos",
        text="open image",
        mode="deterministic",
        token_budget=900,
        idempotency_key="image-2",
        graph_revision="graph:v-image",
        surrounding_nodes=[
            {
                "id": "audio:only",
                "kind": "audio",
                "label": "song only",
                "text": "audio resource",
                "x": 0.31,
                "y": 0.44,
                "source_rel_path": "artifacts/audio/only.mp3",
                "url": "/library/artifacts/audio/only.mp3",
            }
        ],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="image-seed-2",
    )
    assert payload["ok"] is True
    media_actions = payload.get("media_actions", [])
    assert isinstance(media_actions, list)
    assert len(media_actions) == 1
    action = media_actions[0]
    assert action.get("status") == "blocked"
    assert action.get("media_kind") == "image"
    assert action.get("reason") == "no_image_candidates"
    assert payload.get("audio_actions", []) == []

    kinds = [row["kind"] for row in manager.list_events(limit=400)]
    assert "muse.image.intent.detected" in kinds
    assert "image.open.blocked" in kinds


def test_classifier_presence_biases_ambiguous_prompt_to_default_kind() -> None:
    manager = _manager()
    payload = manager.send_message(
        muse_id="chaos",
        text="open play",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="classifier-1",
        graph_revision="graph:v-classifier",
        surrounding_nodes=[
            {
                "id": "audio:loop",
                "kind": "audio",
                "label": "Loop",
                "text": "generic loop",
                "x": 0.43,
                "y": 0.48,
                "source_rel_path": "artifacts/audio/generic_loop.mp3",
                "url": "/library/artifacts/audio/generic_loop.mp3",
            },
            {
                "id": "image:cover",
                "kind": "image",
                "label": "Cover",
                "text": "generic cover",
                "x": 0.58,
                "y": 0.22,
                "source_rel_path": "artifacts/images/generic_cover.png",
                "url": "/library/artifacts/images/generic_cover.png",
            },
            {
                "id": "presence.modality.baseline",
                "kind": "presence",
                "presence_type": "classifier",
                "label": "Baseline Classifier",
                "text": "base modality classifier",
                "default_media_kind": "image",
                "seed_terms": ["open", "play"],
                "x": 0.51,
                "y": 0.51,
                "tags": ["classifier", "image"],
            },
        ],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="classifier-seed",
    )
    assert payload["ok"] is True
    actions = payload.get("media_actions", [])
    assert isinstance(actions, list)
    assert len(actions) == 1
    action = actions[0]
    assert action.get("status") == "requested"
    assert action.get("media_kind") == "image"
    assert action.get("selected_node_id") == "image:cover"
    assert action.get("classifier_default_kind") == "image"
    classifier_presence_ids = action.get("classifier_presence_ids", [])
    assert isinstance(classifier_presence_ids, list)
    assert "presence.modality.baseline" in classifier_presence_ids


def test_concept_seed_focus_boosts_simple_prompt_targeting() -> None:
    manager = _manager()
    base_nodes = [
        {
            "id": "audio:target-canticle",
            "kind": "audio",
            "label": "Fork Tax Canticle",
            "text": "ritual canticle for fork tax",
            "x": 0.44,
            "y": 0.18,
            "source_rel_path": "artifacts/audio/fork_tax_canticle.mp3",
            "url": "/library/artifacts/audio/fork_tax_canticle.mp3",
            "tags": ["audio"],
        },
        {
            "id": "audio:distractor-loop",
            "kind": "audio",
            "label": "Chaos Loop",
            "text": "generic loop",
            "x": 0.63,
            "y": 0.32,
            "source_rel_path": "artifacts/audio/chaos_loop.mp3",
            "url": "/library/artifacts/audio/chaos_loop.mp3",
            "tags": ["workspace-pin", "chaos"],
        },
    ]

    without_seed = manager.send_message(
        muse_id="chaos",
        text="play fork tax canticle music now",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="focus-1",
        graph_revision="graph:v-focus",
        surrounding_nodes=base_nodes,
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="focus-seed-1",
    )
    assert without_seed["ok"] is True
    action_without = without_seed.get("media_actions", [])[0]
    assert action_without.get("selected_node_id") == "audio:distractor-loop"

    with_seed = manager.send_message(
        muse_id="chaos",
        text="play fork tax canticle music now",
        mode="deterministic",
        token_budget=1024,
        idempotency_key="focus-2",
        graph_revision="graph:v-focus",
        surrounding_nodes=[
            *base_nodes,
            {
                "id": "seed.audio.ritual",
                "kind": "daimon",
                "presence_type": "concept_seed",
                "label": "Audio Ritual Seed",
                "text": "concept seed for canticle",
                "media_kind": "audio",
                "seed_terms": ["fork", "tax", "canticle", "play", "music"],
                "focus_node_ids": ["audio:target-canticle"],
                "x": 0.49,
                "y": 0.41,
                "tags": ["concept-seed", "audio"],
            },
        ],
        tool_callback=None,
        reply_builder=_reply_builder,
        seed="focus-seed-2",
    )
    assert with_seed["ok"] is True
    action_with = with_seed.get("media_actions", [])[0]
    assert action_with.get("selected_node_id") == "audio:target-canticle"
    assert action_with.get("media_kind") == "audio"
