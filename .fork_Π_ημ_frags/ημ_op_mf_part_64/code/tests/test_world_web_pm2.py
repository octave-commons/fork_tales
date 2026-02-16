from __future__ import annotations

import json
import tempfile
import wave
from array import array
from pathlib import Path

from code.world_pm2 import parse_args as parse_pm2_args
from code.world_web import (
    RuntimeInfluenceTracker,
    TaskQueue,
    analyze_utterance,
    attach_ui_projection,
    build_chat_reply,
    build_drift_scan_payload,
    build_ui_projection,
    build_voice_lines,
    build_world_payload,
    build_mix_stream,
    build_presence_say_payload,
    build_push_truth_dry_run_payload,
    build_simulation_state,
    collect_catalog,
    detect_artifact_refs,
    normalize_projection_perspective,
    projection_perspective_options,
    render_index,
    resolve_artifact_path,
    resolve_library_path,
    transcribe_audio_bytes,
    utterances_to_ledger_rows,
    websocket_accept_value,
    websocket_frame_text,
)


def _create_fixture_tree(root: Path) -> None:
    manifest = {
        "part": 64,
        "seed_label": "eta_mu_part_64",
        "files": [
            {"path": "artifacts/audio/test.wav", "role": "audio/canonical"},
            {"path": "world_state/constraints.md", "role": "world_state"},
        ],
    }
    (root / "artifacts" / "audio").mkdir(parents=True)
    (root / "world_state").mkdir(parents=True)
    sample = array("h", [0, 1200, -1200, 300, -300, 0])
    with wave.open(str(root / "artifacts" / "audio" / "test.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(sample.tobytes())
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "world_state" / "constraints.md").write_text(
        "# Constraints\n\n- C-64-world-snapshot: active\n",
        encoding="utf-8",
    )


def test_world_payload_and_artifact_resolution() -> None:
    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        _create_fixture_tree(part_root)

        payload = build_world_payload(part_root)
        assert payload["part"] == 64
        assert payload["roles"]["audio/canonical"] == 1
        assert payload["constraints"] == ["C-64-world-snapshot: active"]
        assert "generated_at" in payload

        ok_path = resolve_artifact_path(part_root, "/artifacts/audio/test.wav")
        assert ok_path is not None
        assert ok_path.name == "test.wav"

        blocked = resolve_artifact_path(part_root, "/artifacts/../manifest.json")
        assert blocked is None


def test_pm2_parse_args_defaults() -> None:
    args = parse_pm2_args(["start"])
    assert args.command == "start"
    assert args.port == 8787
    assert args.host == "127.0.0.1"
    assert args.name == "eta-mu-world"


def test_voice_lines_canonical_payload() -> None:
    payload = build_voice_lines("canonical")
    assert payload["mode"] == "canonical"
    assert payload["model"] is None
    assert len(payload["lines"]) >= 7
    ids = {line["id"] for line in payload["lines"]}
    assert {
        "receipt_river",
        "witness_thread",
        "fork_tax_canticle",
        "mage_of_receipts",
        "keeper_of_receipts",
        "anchor_registry",
        "gates_of_truth",
    }.issubset(ids)
    first = payload["lines"][0]
    assert "line_en" in first and "line_ja" in first


def test_chat_reply_canonical_fallback() -> None:
    payload = build_chat_reply(
        [{"role": "user", "text": "Anchor me to the registry"}],
        mode="canonical",
    )
    assert payload["mode"] == "canonical"
    assert payload["model"] is None
    assert isinstance(payload["reply"], str)
    assert len(payload["reply"]) > 8


def test_chat_reply_multi_entity_canonical_trace() -> None:
    payload = build_chat_reply(
        [{"role": "user", "text": "Fork tax meets anchor registry"}],
        mode="canonical",
        multi_entity=True,
        presence_ids=["fork_tax_canticle", "anchor_registry"],
    )

    assert payload["mode"] == "canonical"
    assert payload["model"] is None
    assert isinstance(payload["reply"], str)
    assert "\u266a" in payload["reply"]
    assert "trace" in payload
    assert payload["trace"]["version"] == "v1"
    assert payload["trace"]["multi_entity"] is True
    assert len(payload["trace"]["entities"]) == 2
    assert payload["trace"]["entities"][0]["status"] == "ok"


def test_chat_reply_multi_entity_ollama_trace_with_stub() -> None:
    def fake_generate(_prompt: str) -> tuple[str | None, str]:
        return "[[PULSE]] I sing in mirrored witness.", "test-model"

    payload = build_chat_reply(
        [{"role": "user", "text": "witness this"}],
        mode="ollama",
        multi_entity=True,
        presence_ids=["witness_thread"],
        generate_text_fn=fake_generate,
    )

    assert payload["mode"] == "ollama"
    assert payload["model"] == "test-model"
    assert "trace" in payload
    assert payload["trace"]["entities"][0]["mode"] == "ollama"
    assert payload["trace"]["entities"][0]["status"] == "ok"
    assert "[[PULSE]]" in payload["trace"]["overlay_tags"]


def test_chat_reply_multi_entity_ollama_falls_back_per_presence() -> None:
    def fake_generate(_prompt: str) -> tuple[str | None, str]:
        return None, "test-model"

    payload = build_chat_reply(
        [{"role": "user", "text": "tell me about receipts"}],
        mode="ollama",
        multi_entity=True,
        presence_ids=["receipt_river"],
        generate_text_fn=fake_generate,
    )

    assert payload["mode"] == "canonical"
    assert payload["model"] is None
    assert payload["trace"]["entities"][0]["status"] == "fallback"
    assert len(payload["trace"]["failures"]) == 1
    assert payload["trace"]["failures"][0]["fallback_used"] is True


def test_transcribe_audio_empty_payload() -> None:
    payload = transcribe_audio_bytes(b"", mime="audio/webm", language="ja")
    assert payload["ok"] is False
    assert payload["engine"] == "none"
    assert payload["error"] == "empty audio payload"


def test_catalog_library_and_dashboard_render() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        part_a = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part_a)

        part_b = vault / "ημ_op_mf_part_63"
        (part_b / "artifacts" / "audio").mkdir(parents=True)
        (part_b / "artifacts" / "images").mkdir(parents=True)
        (part_b / "artifacts" / "audio" / "receipt_river_78bpm.mp3").write_bytes(b"mp3")
        (
            part_b / "artifacts" / "images" / "Receipt_River_cover_ημ_part63.png"
        ).write_bytes(b"png")
        (part_b / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "ημ_op_mf_part_63",
                    "artifacts": [
                        {
                            "path": "artifacts/audio/receipt_river_78bpm.mp3",
                            "role": "audio",
                        },
                        {
                            "path": "artifacts/images/Receipt_River_cover_ημ_part63.png",
                            "role": "cover_art",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        catalog = collect_catalog(part_a, vault)
        assert catalog["counts"]["audio"] == 2
        assert catalog["counts"]["image"] == 1
        assert len(catalog["canonical_terms"]) >= 5
        assert len(catalog["named_fields"]) == 7
        assert catalog["ui_default_perspective"] == "hybrid"
        assert [item["id"] for item in catalog["ui_perspectives"]] == [
            "hybrid",
            "causal-time",
            "swimlanes",
        ]
        assert [field["id"] for field in catalog["named_fields"]] == [
            "receipt_river",
            "witness_thread",
            "fork_tax_canticle",
            "mage_of_receipts",
            "keeper_of_receipts",
            "anchor_registry",
            "gates_of_truth",
        ]
        assert all("gradient" in field for field in catalog["named_fields"])
        assert len(catalog["cover_fields"]) == 1
        assert catalog["cover_fields"][0]["display_role"]["en"] == "Cover Art"
        assert any(
            item["name"] == "receipt_river_78bpm.mp3" for item in catalog["items"]
        )
        rr = next(
            item
            for item in catalog["items"]
            if item["name"] == "receipt_river_78bpm.mp3"
        )
        assert rr["display_name"]["en"] == "Receipt River"
        assert rr["display_name"]["ja"] == "領収書の川"

        payload = build_world_payload(part_a)
        html_doc = render_index(payload, catalog)
        assert html_doc == ""

        library_ok = resolve_library_path(
            vault, "/library/ημ_op_mf_part_63/artifacts/audio/receipt_river_78bpm.mp3"
        )
        assert library_ok is not None
        assert library_ok.name == "receipt_river_78bpm.mp3"

        library_blocked = resolve_library_path(vault, "/library/../manifest.json")
        assert library_blocked is None

        mix_wav, mix_meta = build_mix_stream(catalog, vault)
        assert mix_wav.startswith(b"RIFF")
        assert mix_meta["sources"] >= 1

        simulation = build_simulation_state(catalog)
        assert simulation["total"] >= 1
        assert len(simulation["points"]) == simulation["total"]
        assert isinstance(simulation.get("myth"), dict)
        assert isinstance(simulation.get("world"), dict)
        first = simulation["points"][0]
        assert "x" in first and "y" in first and "size" in first


def test_catalog_includes_promptdb_packets() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        promptdb = vault / ".opencode" / "promptdb"
        promptdb.mkdir(parents=True)
        (promptdb / "demo.intent.lisp").write_text(
            """
(packet
  (v "opencode.packet/v1")
  (id "demo:packet")
  (kind :intent)
  (title "Demo")
  (tags [:demo])
  (routing (target :eta-mu-world) (handler :orchestrate) (mode :dry-run)))
            """.strip(),
            encoding="utf-8",
        )
        (promptdb / "diagrams").mkdir(parents=True)
        (promptdb / "diagrams" / "demo.packet.lisp").write_text(
            """
(packet
  (v "opencode.packet/v1")
  (id "diagram:demo")
  (kind :diagram)
  (title "Demo Diagram")
  (tags [:diagram]))
            """.strip(),
            encoding="utf-8",
        )
        (promptdb / "contracts").mkdir(parents=True)
        (promptdb / "contracts" / "demo.contract.lisp").write_text(
            '(contract "demo.contract/v1" (line-format :receipt-line))',
            encoding="utf-8",
        )

        catalog = collect_catalog(part, vault)
        promptdb_index = catalog.get("promptdb", {})
        assert promptdb_index.get("packet_count") == 2
        assert promptdb_index.get("contract_count") == 1
        refresh = promptdb_index.get("refresh", {})
        assert refresh.get("strategy") == "polling+debounce"
        assert isinstance(refresh.get("debounce_ms"), int)
        packets = promptdb_index.get("packets", [])
        assert len(packets) == 2
        ids = {packet.get("id") for packet in packets}
        assert ids == {"demo:packet", "diagram:demo"}


def test_projection_perspective_normalization_and_defaults() -> None:
    assert normalize_projection_perspective("hybrid") == "hybrid"
    assert normalize_projection_perspective("causal") == "causal-time"
    assert normalize_projection_perspective("causal_time") == "causal-time"
    assert normalize_projection_perspective("swimlane") == "swimlanes"
    assert normalize_projection_perspective("unknown-mode") == "hybrid"

    options = projection_perspective_options()
    assert len(options) == 3
    assert options[0]["id"] == "hybrid"
    assert any(option["default"] for option in options)


def test_ui_projection_builds_layout_and_chat_lens() -> None:
    catalog = {
        "counts": {"audio": 5, "image": 8, "text": 7},
        "items": [
            {
                "rel_path": "artifacts/audio/demo.wav",
                "part": "64",
                "kind": "audio",
                "bytes": 1024,
                "mtime_utc": "2026-02-15T00:00:00+00:00",
            }
        ],
        "promptdb": {"packet_count": 3},
        "task_queue": {"pending_count": 2, "event_count": 5},
    }
    simulation = {
        "world": {"tick": 12},
        "presence_dynamics": {
            "click_events": 3,
            "file_events": 4,
            "witness_thread": {
                "continuity_index": 0.67,
                "lineage": [{"ref": "receipts.log"}],
            },
            "fork_tax": {
                "paid_ratio": 0.52,
                "balance": 3.0,
                "debt": 8.0,
                "paid": 5.0,
            },
            "presence_impacts": [
                {
                    "id": "witness_thread",
                    "affected_by": {"files": 0.4, "clicks": 0.7},
                    "affects": {"world": 0.66, "ledger": 0.7},
                }
            ],
        },
    }

    projection = build_ui_projection(
        catalog,
        simulation,
        perspective="causal-time",
        queue_snapshot={"pending_count": 2, "event_count": 5},
        influence_snapshot=simulation["presence_dynamics"],
    )
    assert projection["contract"] == "家_映.v1"
    assert projection["perspective"] == "causal-time"
    assert projection["default_perspective"] == "hybrid"
    assert len(projection["field_schemas"]) == 8
    assert projection["layout"]["perspective"] == "causal-time"
    assert len(projection["elements"]) >= 7
    assert len(projection["states"]) == len(projection["elements"])
    assert any(item["kind"] == "chat-lens" for item in projection["elements"])
    assert projection["chat_sessions"][0]["memory_scope"] == "council"
    assert all("explain" in state for state in projection["states"])


def test_attach_ui_projection_mutates_catalog_with_projection_bundle() -> None:
    catalog = {
        "counts": {"audio": 2, "image": 1, "text": 1},
        "items": [],
        "promptdb": {"packet_count": 1},
    }
    projection = attach_ui_projection(
        catalog,
        perspective="swimlanes",
        queue_snapshot={"pending_count": 1, "event_count": 2},
        influence_snapshot={"clicks_45s": 1, "file_changes_120s": 2},
    )
    assert catalog["ui_default_perspective"] == "hybrid"
    assert catalog["ui_projection"]["perspective"] == "swimlanes"
    assert projection["layout"]["perspective"] == "swimlanes"
    rects = projection["layout"]["rects"]
    assert isinstance(rects, dict)
    assert len(rects) == len(projection["elements"])


def test_presence_say_payload_returns_intent_and_rendered_text() -> None:
    catalog = {
        "items": [{"name": "demo.wav"}],
        "promptdb": {"packet_count": 2},
    }
    payload = build_presence_say_payload(
        catalog,
        text="repair drift now",
        requested_presence_id="witness_thread",
    )

    assert payload["ok"] is True
    assert payload["presence_id"] == "witness_thread"
    assert isinstance(payload["rendered_text"], str)
    assert payload["rendered_text"]
    say_intent = payload["say_intent"]
    assert isinstance(say_intent["facts"], list)
    assert isinstance(say_intent["asks"], list)
    assert isinstance(say_intent["repairs"], list)
    assert say_intent["constraints"]["no_new_facts"] is True


def test_task_queue_persists_dedupes_and_emits_receipts() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        queue_log = root / ".opencode" / "runtime" / "task_queue.v1.jsonl"
        receipts_log = root / "receipts.log"
        queue = TaskQueue(
            queue_log,
            receipts_log,
            owner="Err",
            host="127.0.0.1:8787",
        )

        first = queue.enqueue(
            kind="drift-scan",
            payload={"scope": [".opencode/promptdb"]},
            dedupe_key="drift:promptdb",
        )
        assert first["ok"] is True
        assert first["deduped"] is False

        second = queue.enqueue(
            kind="drift-scan",
            payload={"scope": [".opencode/promptdb"]},
            dedupe_key="drift:promptdb",
        )
        assert second["ok"] is True
        assert second["deduped"] is True

        popped = queue.dequeue()
        assert popped["ok"] is True
        assert popped["task"]["kind"] == "drift-scan"

        receipt_rows = receipts_log.read_text("utf-8").splitlines()
        assert len(receipt_rows) == 2
        assert "task-queue:enqueue" in receipt_rows[0]
        assert "task-queue:dequeue" in receipt_rows[1]

        replay = TaskQueue(
            queue_log,
            receipts_log,
            owner="Err",
            host="127.0.0.1:8787",
        )
        assert replay.snapshot()["pending_count"] == 0


def test_drift_scan_reports_open_questions_and_keeper_signal() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        promptdb = vault / ".opencode" / "promptdb" / "diagrams"
        promptdb.mkdir(parents=True)
        (promptdb / "part64_runtime_system.packet.lisp").write_text(
            """
(packet
  (v "opencode.packet/v1")
  (id "diagram:part64-runtime-system:v1")
  (kind :diagram)
  (title "Demo")
  (tags [:demo])
  (body
    (diagram
      (open-questions
        (q (id q.task-queue) (text "queue?"))
        (q (id q.refresh-rate) (text "refresh?"))))))
            """.strip(),
            encoding="utf-8",
        )

        (vault / "receipts.log").write_text(
            (
                "ts=2026-02-15T00:00:00Z | kind=:decision | origin=unit-test | owner=Err | "
                "dod=test | pi=part64 | host=127.0.0.1:8787 | manifest=manifest.lith | "
                "refs=.opencode/promptdb/00_wire_world.intent.lisp"
            ),
            encoding="utf-8",
        )

        payload = build_drift_scan_payload(part, vault)
        assert payload["ok"] is True
        assert payload["open_questions"]["total"] == 2
        assert payload["open_questions"]["unresolved_count"] == 2
        assert any(
            gate.get("reason") == "open-questions-unresolved"
            for gate in payload["blocked_gates"]
        )
        keeper = payload.get("keeper_of_contracts")
        assert isinstance(keeper, dict)
        assert keeper["presence_id"] == "keeper_of_contracts"


def test_push_truth_dry_run_uses_manifest_proof_schema() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        promptdb = vault / ".opencode" / "promptdb"
        promptdb.mkdir(parents=True)
        (promptdb / "00_wire_world.intent.lisp").write_text(
            "(packet)", encoding="utf-8"
        )

        (vault / "receipts.log").write_text(
            (
                "ts=2026-02-15T00:00:00Z | kind=:decision | origin=unit-test | owner=Err | "
                "dod=test | pi=part64 | host=127.0.0.1:8787 | manifest=manifest.lith | "
                "refs=.opencode/promptdb/00_wire_world.intent.lisp"
            ),
            encoding="utf-8",
        )

        (vault / "manifest.lith").write_text(
            """
(manifest
  (v "promethean.manifest/v1")
  (proof-schema
    (required-refs [".opencode/promptdb/00_wire_world.intent.lisp" "receipts.log"])
    (required-hashes ["sha256:pi_zip" "sha256:manifest"])
    (host-handle "github:err")))
            """.strip(),
            encoding="utf-8",
        )

        (vault / "Π.test.zip").write_bytes(b"zip")

        payload = build_push_truth_dry_run_payload(part, vault)
        assert payload["ok"] is True
        assert payload["proof_schema"]["host_handle"] == "github:err"
        assert payload["proof_schema"]["required_hashes"] == [
            "sha256:pi_zip",
            "sha256:manifest",
        ]
        assert payload["artifacts"]["host_has_github_gist"] is True
        assert payload["gate"]["blocked"] is False


def test_websocket_helpers() -> None:
    accept = websocket_accept_value("dGhlIHNhbXBsZSBub25jZQ==")
    assert accept == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    frame = websocket_frame_text('{"ok":true}')
    assert frame[0] == 0x81
    assert frame[1] == len('{"ok":true}')


def test_simulation_state_includes_world_summary_slot() -> None:
    simulation = build_simulation_state({"items": [], "counts": {}})
    assert isinstance(simulation.get("world"), dict)


def test_runtime_influence_tracker_reports_bilingual_fork_tax_and_ghost() -> None:
    tracker = RuntimeInfluenceTracker()
    tracker.record_witness(event_type="touch", target="particle_field")
    tracker.record_file_delta(
        {
            "added_count": 1,
            "updated_count": 2,
            "removed_count": 0,
            "sample_paths": ["artifacts/audio/test.wav", "receipts.log"],
        }
    )

    snapshot = tracker.snapshot(queue_snapshot={"pending_count": 1, "event_count": 3})
    assert snapshot["clicks_45s"] == 1
    assert snapshot["file_changes_120s"] == 3
    assert snapshot["fork_tax"]["law_ja"].startswith("フォーク税")
    assert snapshot["ghost"]["id"] == "file_sentinel"
    assert snapshot["ghost"]["ja"] == "ファイルの哨戒者"


def test_runtime_influence_tracker_accepts_manual_fork_tax_payment() -> None:
    tracker = RuntimeInfluenceTracker()
    tracker.record_file_delta(
        {
            "added_count": 0,
            "updated_count": 3,
            "removed_count": 0,
            "sample_paths": ["receipts.log"],
        }
    )
    before = tracker.snapshot(queue_snapshot={"pending_count": 0, "event_count": 0})
    payment = tracker.pay_fork_tax(
        amount=2.5,
        source="test-suite",
        target="fork_tax_canticle",
    )
    after = tracker.snapshot(queue_snapshot={"pending_count": 0, "event_count": 0})

    assert payment["applied"] == 2.5
    assert payment["target"] == "fork_tax_canticle"
    assert after["fork_tax"]["paid"] > before["fork_tax"]["paid"]
    assert after["fork_tax"]["balance"] <= before["fork_tax"]["balance"]
    assert "fork_tax_canticle" in after["recent_click_targets"]


def test_simulation_state_includes_presence_dynamics_and_file_sentinel() -> None:
    simulation = build_simulation_state(
        {"items": [], "counts": {"audio": 3}},
        influence_snapshot={
            "clicks_45s": 2,
            "file_changes_120s": 4,
            "recent_click_targets": ["particle_field"],
            "recent_file_paths": ["receipts.log"],
            "fork_tax": {
                "law_en": "Pay the fork tax.",
                "law_ja": "フォーク税は法。",
                "debt": 5.0,
                "paid": 2.0,
                "balance": 3.0,
                "paid_ratio": 0.4,
            },
            "ghost": {
                "id": "file_sentinel",
                "en": "File Sentinel",
                "ja": "ファイルの哨戒者",
                "auto_commit_pulse": 0.5,
                "queue_pending": 1,
                "status_en": "staging receipts",
                "status_ja": "領収書を段取り中",
            },
        },
        queue_snapshot={"pending_count": 1, "event_count": 4},
    )
    dynamics = simulation.get("presence_dynamics", {})
    assert dynamics.get("fork_tax", {}).get("law_ja") == "フォーク税は法。"
    assert dynamics.get("ghost", {}).get("id") == "file_sentinel"
    witness = dynamics.get("witness_thread", {})
    assert witness.get("id") == "witness_thread"
    assert witness.get("ja") == "証人の糸"
    lineage = witness.get("lineage", [])
    assert any(item.get("ref") == "particle_field" for item in lineage)
    assert any(item.get("ref") == "receipts.log" for item in lineage)
    impacts = dynamics.get("presence_impacts", [])
    assert any(item.get("id") == "receipt_river" for item in impacts)
    assert any(item.get("id") == "file_sentinel" for item in impacts)


def test_simulation_state_witness_thread_uses_idle_lineage_without_events() -> None:
    simulation = build_simulation_state(
        {"items": [], "counts": {}},
        influence_snapshot={
            "clicks_45s": 0,
            "file_changes_120s": 0,
            "recent_click_targets": [],
            "recent_file_paths": [],
            "fork_tax": {
                "law_en": "Pay the fork tax.",
                "law_ja": "フォーク税は法。",
                "debt": 0.0,
                "paid": 0.0,
                "balance": 0.0,
                "paid_ratio": 1.0,
            },
            "ghost": {
                "id": "file_sentinel",
                "en": "File Sentinel",
                "ja": "ファイルの哨戒者",
                "auto_commit_pulse": 0.0,
                "queue_pending": 0,
                "status_en": "gate idle",
                "status_ja": "門前で待機中",
            },
        },
        queue_snapshot={"pending_count": 0, "event_count": 0},
    )
    witness = simulation.get("presence_dynamics", {}).get("witness_thread", {})
    lineage = witness.get("lineage", [])
    assert isinstance(lineage, list)
    assert lineage[0]["kind"] == "idle"
    assert lineage[0]["ref"] == "awaiting-touch"


def test_eta_mu_ledger_helpers() -> None:
    refs = detect_artifact_refs(
        "proof in artifacts/audio/test.wav PR #12 commit abcdef1"
    )
    assert refs == ["artifacts/audio/test.wav", "PR #12", "abcdef1"]

    row = analyze_utterance("we should prove this", idx=1)
    assert row["classification"] == "eta-claim"
    assert row["eta_claim"] is True
    assert row["mu_proof"] is False

    rows = utterances_to_ledger_rows(
        ["", "we will ship", "evidence artifacts/audio/test.wav"]
    )
    assert len(rows) == 2
    assert rows[0]["idx"] == 1
    assert rows[0]["classification"] == "eta-claim"
    assert rows[1]["idx"] == 2
    assert rows[1]["classification"] == "mu-proof"
