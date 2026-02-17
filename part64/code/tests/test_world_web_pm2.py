from __future__ import annotations

import json
import subprocess
import tempfile
import wave
import zipfile
from array import array
from pathlib import Path
from typing import Any

import code.world_web as world_web_module

from code.world_pm2 import parse_args as parse_pm2_args
from code.world_web import (
    CouncilChamber,
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
    build_study_snapshot,
    build_pi_archive_payload,
    build_simulation_state,
    collect_catalog,
    detect_artifact_refs,
    normalize_projection_perspective,
    projection_perspective_options,
    validate_pi_archive_portable,
    render_index,
    resolve_artifact_path,
    resolve_library_member,
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


def test_library_resolution_uses_eta_mu_substrate_root() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vault = root / "vault"
        vault.mkdir(parents=True)

        archive_dir = (
            root / ".opencode" / "knowledge" / "archive" / "2026" / "02" / "16"
        )
        archive_dir.mkdir(parents=True)
        artifact = archive_dir / "demo.txt"
        artifact.write_text("ok", encoding="utf-8")

        resolved = resolve_library_path(
            vault,
            "/library/.opencode/knowledge/archive/2026/02/16/demo.txt",
        )
        assert resolved is not None
        assert resolved == artifact.resolve()


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


def test_embedding_backend_accepts_tensorflow(monkeypatch: Any) -> None:
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "tensorflow")
    assert world_web_module._embedding_backend() == "tensorflow"


def test_text_generation_backend_accepts_tensorflow(monkeypatch: Any) -> None:
    monkeypatch.setenv("TEXT_GENERATION_BACKEND", "tensorflow")
    assert world_web_module._text_generation_backend() == "tensorflow"


def test_tensorflow_generate_text_returns_none_when_module_missing(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module, "_load_tensorflow_module", lambda: None)
    text, model = world_web_module._tensorflow_generate_text("gate blocked")
    assert text is None
    assert model == "tensorflow-hash-v1"


def test_ollama_generate_text_uses_tensorflow_backend_when_selected(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TEXT_GENERATION_BACKEND", "tensorflow")

    def _fake_tf_generate(
        prompt: str, model: str | None = None, timeout_s: float | None = None
    ) -> tuple[str | None, str]:
        assert "blocked" in prompt
        assert model is None
        assert timeout_s is None
        return "tf line", "tensorflow-hash-v1"

    monkeypatch.setattr(
        world_web_module, "_tensorflow_generate_text", _fake_tf_generate
    )
    text, model = world_web_module._ollama_generate_text("blocked gate signal")
    assert text == "tf line"
    assert model == "tensorflow-hash-v1"


def test_ollama_embed_uses_tensorflow_backend_when_selected(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "tensorflow")
    expected = [0.1, 0.2, 0.3]

    def _fake_tf_embed(text: str, model: str | None = None) -> list[float] | None:
        assert text == "drift gate"
        assert model is None
        return expected

    monkeypatch.setattr(world_web_module, "_tensorflow_embed", _fake_tf_embed)
    result = world_web_module._ollama_embed("drift gate")
    assert result == expected


def test_resource_monitor_snapshot_reports_devices_and_auto_backend(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("ETA_MU_GPU1_UTILIZATION", "82")
    monkeypatch.setenv("ETA_MU_GPU2_UTILIZATION", "17")
    monkeypatch.setenv("ETA_MU_NPU0_UTILIZATION", "64")
    monkeypatch.setenv("ETA_MU_NPU0_QUEUE_DEPTH", "4")

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        (part_root / "world_state").mkdir(parents=True)
        (part_root / "world_state" / "weaver-error.log").write_text(
            "error: gate blocked\nwarning: retry\n",
            encoding="utf-8",
        )

        snapshot = world_web_module._resource_monitor_snapshot(part_root=part_root)

    assert snapshot.get("record") == "ημ.resource-heartbeat.v1"
    devices = snapshot.get("devices", {})
    assert devices.get("cpu", {}).get("status") in {"ok", "watch", "hot"}
    assert devices.get("gpu1", {}).get("utilization") == 82.0
    assert devices.get("npu0", {}).get("queue_depth") == 4
    auto_backend = snapshot.get("auto_backend", {})
    assert isinstance(auto_backend.get("embeddings_order", []), list)
    assert isinstance(auto_backend.get("text_order", []), list)


def test_ollama_embed_auto_uses_resource_selected_order(monkeypatch: Any) -> None:
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "auto")
    calls: list[str] = []

    monkeypatch.setattr(
        world_web_module,
        "_resource_auto_embedding_order",
        lambda snapshot=None: ["tensorflow", "openvino", "ollama"],
    )

    def _fake_tf_embed(text: str, model: str | None = None) -> list[float] | None:
        assert text == "resource route"
        assert model is None
        calls.append("tensorflow")
        return [0.6, 0.8]

    def _fake_openvino_embed(
        text: str,
        model: str | None = None,
    ) -> list[float] | None:
        calls.append("openvino")
        return None

    def _fake_remote_embed(text: str, model: str | None = None) -> list[float] | None:
        calls.append("ollama")
        return None

    monkeypatch.setattr(world_web_module, "_tensorflow_embed", _fake_tf_embed)
    monkeypatch.setattr(world_web_module, "_openvino_embed", _fake_openvino_embed)
    monkeypatch.setattr(world_web_module, "_ollama_embed_remote", _fake_remote_embed)

    result = world_web_module._ollama_embed("resource route")
    assert result == [0.6, 0.8]
    assert calls == ["tensorflow"]


def test_ollama_generate_text_auto_uses_resource_selected_order(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TEXT_GENERATION_BACKEND", "auto")
    calls: list[str] = []

    monkeypatch.setattr(
        world_web_module,
        "_resource_auto_text_order",
        lambda snapshot=None: ["ollama", "tensorflow"],
    )

    def _fake_remote(
        prompt: str, model: str | None = None, timeout_s: float | None = None
    ) -> tuple[str | None, str]:
        assert "route" in prompt
        calls.append("ollama")
        return None, "ollama-route"

    def _fake_tf(
        prompt: str, model: str | None = None, timeout_s: float | None = None
    ) -> tuple[str | None, str]:
        calls.append("tensorflow")
        return "tf-route", "tensorflow-hash-v1"

    monkeypatch.setattr(world_web_module, "_ollama_generate_text_remote", _fake_remote)
    monkeypatch.setattr(world_web_module, "_tensorflow_generate_text", _fake_tf)

    text, model = world_web_module._ollama_generate_text("route this prompt")
    assert text == "tf-route"
    assert model == "tensorflow-hash-v1"
    assert calls == ["ollama", "tensorflow"]


def test_presence_say_payload_supports_health_sentinel_resource_facts() -> None:
    payload = build_presence_say_payload(
        {
            "items": [],
            "promptdb": {"packet_count": 0},
            "council": {"pending_count": 0, "decision_count": 0, "approved_count": 0},
            "presence_runtime": {
                "resource_heartbeat": {
                    "devices": {
                        "cpu": {"utilization": 73.5},
                        "gpu1": {"utilization": 41.0},
                        "gpu2": {"utilization": 12.0},
                        "npu0": {"utilization": 28.5},
                    },
                    "hot_devices": ["cpu"],
                    "log_watch": {
                        "error_count": 2,
                        "latest": "error: timeout while syncing council queue",
                    },
                }
            },
        },
        text="resource heartbeat status",
        requested_presence_id="cpu",
    )

    assert payload["presence_id"] == "health_sentinel_cpu"
    repairs = payload.get("say_intent", {}).get("repairs", [])
    assert any("backend" in str(item).lower() for item in repairs)
    facts = payload.get("say_intent", {}).get("facts", [])
    assert any("cpu_utilization=73.5" in str(item) for item in facts)
    assert any("resource_hot_devices=cpu" in str(item) for item in facts)


def test_build_study_snapshot_includes_resource_hot_and_log_warnings() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "part64"
        part.mkdir(parents=True)

        payload = build_study_snapshot(
            part,
            root,
            queue_snapshot={
                "queue_log": str(root / "queue.jsonl"),
                "pending_count": 0,
                "dedupe_keys": 0,
                "event_count": 0,
                "pending": [],
            },
            council_snapshot={
                "decision_log": str(root / "council.jsonl"),
                "decision_count": 0,
                "pending_count": 0,
                "approved_count": 0,
                "auto_restart_enabled": True,
                "require_council": True,
                "cooldown_seconds": 25,
                "decisions": [],
            },
            drift_payload={
                "active_drifts": [],
                "blocked_gates": [],
                "open_questions": {
                    "total": 0,
                    "resolved_count": 0,
                    "unresolved_count": 0,
                },
                "receipts_parse": {
                    "path": str(root / "receipts.log"),
                    "ok": True,
                    "rows": 0,
                    "has_intent_ref": False,
                },
            },
            truth_gate_blocked=False,
            resource_snapshot={
                "record": "ημ.resource-heartbeat.v1",
                "devices": {},
                "hot_devices": ["gpu1"],
                "log_watch": {
                    "error_ratio": 0.56,
                    "error_count": 7,
                },
                "auto_backend": {
                    "embeddings_order": ["tensorflow", "ollama"],
                    "text_order": ["ollama", "tensorflow"],
                },
            },
        )

    signals = payload.get("signals", {})
    assert signals.get("resource_hot_count") == 1
    assert signals.get("resource_log_error_ratio") == 0.56
    warnings = payload.get("warnings", [])
    assert any(item.get("code") == "runtime.resource_hot" for item in warnings)
    assert any(item.get("code") == "runtime.log_error_ratio" for item in warnings)


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
        heat_values = catalog.get("heat_values", {})
        assert heat_values.get("record") == "ημ.heat-values.v1"
        heat_regions = heat_values.get("regions", [])
        assert isinstance(heat_regions, list)
        assert len(heat_regions) == len(world_web_module.FIELD_TO_PRESENCE)
        heat_facts = heat_values.get("facts", [])
        assert {
            str(row.get("region_id", "")) for row in heat_facts if isinstance(row, dict)
        } == set(world_web_module.FIELD_TO_PRESENCE.keys())
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

        library_with_member = resolve_library_path(
            vault,
            "/library/ημ_op_mf_part_63/artifacts/audio/receipt_river_78bpm.mp3?member=payload/test.wav",
        )
        assert library_with_member is not None
        assert (
            resolve_library_member("/library/sample.zip?member=payload/test.wav")
            == "payload/test.wav"
        )
        assert (
            resolve_library_member("/library/sample.zip?member=../manifest.json")
            is None
        )

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


def test_library_archive_member_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        archive_path = Path(td) / "sample.zip"
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive_zip:
            archive_zip.writestr("payload/test.md", "hello archive")
            archive_zip.writestr("manifest.json", '{"record":"ημ.archive-manifest.v1"}')

        member_payload = world_web_module._read_library_archive_member(
            archive_path,
            "payload/test.md",
        )
        assert member_payload is not None
        payload, content_type = member_payload
        assert payload == b"hello archive"
        assert content_type.startswith("text/")
        assert (
            world_web_module._read_library_archive_member(
                archive_path,
                "../manifest.json",
            )
            is None
        )


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
    element_ids = {str(item.get("id", "")) for item in projection["elements"]}
    assert "nexus.ui.projection_ledger" in element_ids
    assert "nexus.ui.stability_observatory" in element_ids
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


def test_presence_say_payload_supports_file_organizer_and_concept_presence() -> None:
    catalog = {
        "items": [{"name": "guide.md"}, {"name": "ledger.md"}],
        "promptdb": {"packet_count": 3},
        "file_graph": {
            "stats": {
                "concept_presence_count": 2,
                "organized_file_count": 2,
            },
            "concept_presences": [
                {
                    "id": "presence:concept:demo12345678",
                    "label": "Concept: ledger / witness",
                    "label_ja": "概念: ledger / witness",
                    "terms": ["ledger", "witness"],
                    "file_count": 2,
                }
            ],
        },
    }

    organizer_payload = build_presence_say_payload(
        catalog,
        text="sort files by concept",
        requested_presence_id="file_organizer",
    )
    assert organizer_payload["presence_id"] == "file_organizer"
    organizer_repairs = organizer_payload["say_intent"]["repairs"]
    assert any("concept presences" in str(item) for item in organizer_repairs)

    concept_payload = build_presence_say_payload(
        catalog,
        text="refine this cluster",
        requested_presence_id="presence:concept:demo12345678",
    )
    assert concept_payload["presence_id"] == "presence:concept:demo12345678"
    assert concept_payload["presence_name"]["en"] == "Concept: ledger / witness"
    concept_facts = concept_payload["say_intent"]["facts"]
    assert any("concept_terms=ledger,witness" in str(item) for item in concept_facts)


def test_presence_say_payload_supports_the_council_profile() -> None:
    payload = build_presence_say_payload(
        {
            "items": [],
            "promptdb": {"packet_count": 0},
            "council": {
                "pending_count": 2,
                "decision_count": 4,
                "approved_count": 1,
            },
        },
        text="why is the gate blocked?",
        requested_presence_id="the_council",
    )
    assert payload["presence_id"] == "the_council"
    assert payload["presence_name"]["en"] == "The Council"
    asks = payload["say_intent"]["asks"]
    repairs = payload["say_intent"]["repairs"]
    facts = payload["say_intent"]["facts"]
    assert any("gate reason" in str(item).lower() for item in asks)
    assert any("quorum" in str(item).lower() for item in repairs)
    assert any("council_pending=2" in str(item) for item in facts)
    assert any("council_decisions=4" in str(item) for item in facts)


def test_chat_reply_multi_entity_normalizes_aliases_and_dedupes() -> None:
    payload = build_chat_reply(
        [{"role": "user", "text": "Diagnose blocked gate and restart policy"}],
        mode="canonical",
        multi_entity=True,
        presence_ids=["council", "the-council", "file-sentinel"],
    )
    entities = payload.get("trace", {}).get("entities", [])
    entity_ids = [str(row.get("presence_id", "")) for row in entities]
    assert "the_council" in entity_ids
    assert "file_sentinel" in entity_ids
    assert entity_ids.count("the_council") == 1


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


def test_catalog_includes_pi_archive_summary() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        catalog = collect_catalog(part, vault)
        pi_archive = catalog.get("pi_archive", {})
        assert pi_archive.get("record") == "ημ.pi-archive.v1"
        assert (
            len(str((pi_archive.get("hash") or {}).get("canonical_sha256", ""))) == 64
        )
        assert isinstance(pi_archive.get("portable"), dict)


def test_pi_archive_hash_is_deterministic_for_stable_payload() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        (vault / "receipts.log").write_text(
            (
                "ts=2026-02-15T00:00:00Z | kind=:decision | origin=unit-test | owner=Err | "
                "dod=test | pi=part64 | host=127.0.0.1:8787 | manifest=manifest.lith | "
                "refs=.opencode/promptdb/00_wire_world.intent.lisp"
            ),
            encoding="utf-8",
        )

        catalog = collect_catalog(part, vault)
        queue_snapshot = {"pending_count": 0, "event_count": 0, "pending": []}
        archive_a = build_pi_archive_payload(
            part,
            vault,
            catalog=catalog,
            queue_snapshot=queue_snapshot,
        )
        archive_b = build_pi_archive_payload(
            part,
            vault,
            catalog=catalog,
            queue_snapshot=queue_snapshot,
        )

        assert (
            archive_a["hash"]["canonical_sha256"]
            == archive_b["hash"]["canonical_sha256"]
        )
        assert archive_a["signature"]["value"] == archive_b["signature"]["value"]
        assert archive_a["portable"]["ok"] is True


def test_validate_pi_archive_portable_rejects_missing_sections() -> None:
    payload = validate_pi_archive_portable({"record": "ημ.pi-archive.v1"})
    assert payload["ok"] is False
    assert any(item.startswith("missing:snapshot") for item in payload["errors"])


def test_websocket_helpers() -> None:
    accept = websocket_accept_value("dGhlIHNhbXBsZSBub25jZQ==")
    assert accept == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    frame = websocket_frame_text('{"ok":true}')
    assert frame[0] == 0x81
    assert frame[1] == len('{"ok":true}')


def test_simulation_state_includes_world_summary_slot() -> None:
    simulation = build_simulation_state({"items": [], "counts": {}})
    assert isinstance(simulation.get("world"), dict)


def test_eta_mu_inbox_is_ingested_and_graphed() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "new_witness_note.md").write_text(
            "Witness thread continuity, gate proof, and fork tax notes.",
            encoding="utf-8",
        )
        (inbox / "new_receipt.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        catalog = collect_catalog(part, vault)
        inbox_state = catalog.get("eta_mu_inbox", {})
        assert inbox_state.get("is_empty") is True
        assert inbox_state.get("pending_count") == 0
        assert inbox_state.get("processed_count", 0) >= 2

        index_path = vault / ".opencode" / "runtime" / "eta_mu_knowledge.v1.jsonl"
        assert index_path.exists()
        entries = [
            json.loads(line)
            for line in index_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        assert any(row.get("name") == "new_witness_note.md" for row in entries)

        note_entry = next(
            row for row in entries if row.get("name") == "new_witness_note.md"
        )
        assert note_entry.get("archive_kind") == "zip"
        assert str(note_entry.get("archived_rel_path", "")).endswith(".zip")
        assert str(note_entry.get("archive_member_path", "")).startswith("payload/")
        assert note_entry.get("archive_manifest_path") == "manifest.json"
        assert isinstance(note_entry.get("archive_container_id"), str)
        assert note_entry.get("archive_container_id")
        assert any(
            isinstance(link, dict) and link.get("kind") == "stored_in_archive"
            for link in note_entry.get("embedding_links", [])
        )

        note_zip_path = vault / str(note_entry.get("archived_rel_path", ""))
        assert note_zip_path.exists()
        with zipfile.ZipFile(note_zip_path, "r") as archive_zip:
            members = set(archive_zip.namelist())
            assert str(note_entry.get("archive_member_path", "")) in members
            assert "manifest.json" in members
            manifest_payload = json.loads(
                archive_zip.read("manifest.json").decode("utf-8")
            )
            assert manifest_payload.get("record") == "ημ.archive-manifest.v1"
            assert manifest_payload.get("archive_rel_path") == note_entry.get(
                "archived_rel_path"
            )
            assert str(manifest_payload.get("source_rel_path", "")).endswith(
                "new_witness_note.md"
            )

        file_graph = catalog.get("file_graph", {})
        assert file_graph.get("record") == "ημ.file-graph.v1"
        assert file_graph.get("stats", {}).get("file_count", 0) >= 2
        assert len(file_graph.get("edges", [])) >= 2
        assert file_graph.get("stats", {}).get("archive_count", 0) >= 2
        assert any(
            str(node.get("url", "")).startswith("/library/.opencode/knowledge/archive/")
            for node in file_graph.get("file_nodes", [])
        )
        note_node = next(
            node
            for node in file_graph.get("file_nodes", [])
            if node.get("name") == "new_witness_note.md"
        )
        assert note_node.get("archive_kind") == "zip"
        assert note_node.get("archive_member_path") == note_entry.get(
            "archive_member_path"
        )
        assert note_node.get("archive_container_id") == note_entry.get(
            "archive_container_id"
        )
        assert isinstance(note_node.get("embed_layer_points", []), list)
        assert note_node.get("embed_layer_count", 0) >= 1
        embed_layers = file_graph.get("embed_layers", [])
        assert isinstance(embed_layers, list)
        assert len(embed_layers) >= 1
        assert file_graph.get("stats", {}).get("embed_layer_count", 0) >= 1
        organizer = file_graph.get("organizer_presence", {})
        assert organizer.get("id") == "presence:file_organizer"
        concept_presences = file_graph.get("concept_presences", [])
        assert isinstance(concept_presences, list)
        assert len(concept_presences) >= 1
        assert any(
            str(edge.get("kind", "")) == "spawns_presence"
            for edge in file_graph.get("edges", [])
            if isinstance(edge, dict)
        )
        assert isinstance(note_node.get("concept_presence_id", ""), str)
        assert note_node.get("organized_by") == "file_organizer"


def test_eta_mu_inbox_registry_skips_unchanged_duplicate(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        note_body = "Witness thread continuity, gate proof, and fork tax notes."
        (inbox / "new_witness_note.md").write_text(note_body, encoding="utf-8")

        first_catalog = collect_catalog(part, vault)
        first_inbox = first_catalog.get("eta_mu_inbox", {})
        assert first_inbox.get("processed_count", 0) >= 1
        assert first_inbox.get("skipped_count", 0) == 0

        index_path = vault / ".opencode" / "runtime" / "eta_mu_knowledge.v1.jsonl"
        assert index_path.exists()

        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "new_witness_note.md").write_text(note_body, encoding="utf-8")

        second_catalog = collect_catalog(part, vault)
        second_inbox = second_catalog.get("eta_mu_inbox", {})
        assert second_inbox.get("processed_count", 0) == 0
        assert second_inbox.get("skipped_count", 0) >= 1
        assert second_inbox.get("is_empty") is True

        entries = [
            json.loads(line)
            for line in index_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        note_entries = [
            row for row in entries if row.get("name") == "new_witness_note.md"
        ]
        assert len(note_entries) == 1

        registry_path = vault / ".Π" / "ημ_registry.jsonl"
        assert registry_path.exists()
        registry_rows = [
            json.loads(line)
            for line in registry_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        note_registry_rows = [
            row
            for row in registry_rows
            if row.get("source_name") == "new_witness_note.md"
        ]
        assert any(row.get("event") == "ingested" for row in note_registry_rows)

        ingested_keys = {
            str(row.get("registry_key", ""))
            for row in note_registry_rows
            if row.get("event") == "ingested"
        }
        assert any(
            row.get("event") == "skipped"
            and row.get("reason") == "duplicate_unchanged"
            and str(row.get("registry_key", "")) in ingested_keys
            for row in note_registry_rows
        )


def test_eta_mu_ingest_rejects_unsupported_payloads(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "unsupported.bin").write_bytes(b"\x00\x01\x02\x03")

        catalog = collect_catalog(part, vault)
        inbox_state = catalog.get("eta_mu_inbox", {})

        assert inbox_state.get("rejected_count", 0) >= 1
        assert inbox_state.get("is_empty") is True

        rejected_root = inbox / "_rejected"
        assert rejected_root.exists()
        assert any(path.is_file() for path in rejected_root.rglob("*"))

        registry_path = vault / ".Π" / "ημ_registry.jsonl"
        assert registry_path.exists()
        registry_rows = [
            json.loads(line)
            for line in registry_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        rejected_rows = [
            row for row in registry_rows if row.get("source_name") == "unsupported.bin"
        ]
        assert any(row.get("event") == "rejected" for row in rejected_rows)
        assert any(row.get("status") == "reject" for row in rejected_rows)


def test_eta_mu_ingest_writes_manifest_stats_snapshot_artifacts(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "ingest_note.md").write_text(
            "Witness thread continuity and gate proof.",
            encoding="utf-8",
        )
        (inbox / "ingest_image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        catalog = collect_catalog(part, vault)
        inbox_state = catalog.get("eta_mu_inbox", {})

        assert inbox_state.get("processed_count", 0) >= 1
        artifacts = inbox_state.get("artifacts", {})
        for key in ("manifest", "stats", "snapshot"):
            rel_path = str(artifacts.get(key, ""))
            assert rel_path
            artifact_path = vault / rel_path
            assert artifact_path.exists()
            assert artifact_path.read_text("utf-8").startswith("(artifact ")

        registry_path = vault / ".Π" / "ημ_registry.jsonl"
        assert registry_path.exists()
        registry_rows = [
            json.loads(line)
            for line in registry_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        note_rows = [
            row for row in registry_rows if row.get("source_name") == "ingest_note.md"
        ]
        assert any(row.get("event") == "ingested" for row in note_rows)
        assert any(str(row.get("idempotence_key", "")).strip() for row in note_rows)
        assert any(str(row.get("embed_id", "")).strip() for row in note_rows)


def test_embeddings_db_upsert_query_delete_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        upsert = world_web_module._embedding_db_upsert(
            vault,
            entry_id="demo-emb-1",
            text="fork tax witness continuity",
            embedding=[1.0, 0.0, 0.0],
            metadata={"kind": "demo"},
            model="test-embed",
        )
        assert upsert.get("ok") is True
        assert upsert.get("status") == "upserted"

        status = world_web_module._embedding_db_status(vault)
        assert status.get("ok") is True
        assert status.get("entry_count") == 1
        assert status.get("dimensions") == [3]

        listing = world_web_module._embedding_db_list(vault, limit=10)
        assert listing.get("ok") is True
        assert listing.get("count") == 1
        assert listing.get("entries", [])[0].get("id") == "demo-emb-1"

        query = world_web_module._embedding_db_query(
            vault,
            query_embedding=[0.95, 0.05, 0.0],
            top_k=3,
        )
        assert query.get("ok") is True
        assert query.get("match_count") == 1
        first = query.get("results", [])[0]
        assert first.get("id") == "demo-emb-1"
        assert float(first.get("score", 0.0)) > 0.9

        deleted = world_web_module._embedding_db_delete(vault, entry_id="demo-emb-1")
        assert deleted.get("ok") is True
        assert deleted.get("status") == "deleted"

        query_after = world_web_module._embedding_db_query(
            vault,
            query_embedding=[0.95, 0.05, 0.0],
            top_k=3,
        )
        assert query_after.get("ok") is True
        assert query_after.get("match_count") == 0


def test_embeddings_db_upsert_is_idempotent_for_same_payload() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        first = world_web_module._embedding_db_upsert(
            vault,
            entry_id="same-id",
            text="anchor registry",
            embedding=[0.2, 0.8, 0.1],
            metadata={"source": "unit"},
            model="test-embed",
        )
        second = world_web_module._embedding_db_upsert(
            vault,
            entry_id="same-id",
            text="anchor registry",
            embedding=[0.2, 0.8, 0.1],
            metadata={"source": "unit"},
            model="test-embed",
        )

        assert first.get("ok") is True
        assert first.get("status") == "upserted"
        assert second.get("ok") is True
        assert second.get("status") == "unchanged"

        db_path = world_web_module._embeddings_db_path(vault)
        assert db_path.exists()
        rows = [
            json.loads(line)
            for line in db_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        upsert_rows = [
            row
            for row in rows
            if row.get("record") == "ημ.embedding-db.v1"
            and row.get("event") == "upsert"
            and row.get("id") == "same-id"
        ]
        assert len(upsert_rows) == 1


def test_ollama_embed_falls_back_to_api_embed(monkeypatch: Any) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

    calls: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        payload = json.loads((req.data or b"{}").decode("utf-8"))
        calls.append(
            {
                "url": str(req.full_url),
                "timeout": timeout,
                "payload": payload,
            }
        )
        if str(req.full_url).endswith("/api/embeddings"):
            raise world_web_module.URLError("not found")
        if str(req.full_url).endswith("/api/embed"):
            return _FakeResponse({"embeddings": [[0.1, 0.2, 0.3]]})
        raise AssertionError(f"unexpected url: {req.full_url}")

    monkeypatch.setattr(world_web_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://stub.local:11434")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "stub-embed-model")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SEC", "2")

    vector = world_web_module._ollama_embed("witness probe")

    assert vector == [0.1, 0.2, 0.3]
    assert len(calls) == 2
    assert calls[0]["url"].endswith("/api/embeddings")
    assert calls[0]["payload"] == {
        "model": "stub-embed-model",
        "prompt": "witness probe",
    }
    assert calls[1]["url"].endswith("/api/embed")
    assert calls[1]["payload"] == {
        "model": "stub-embed-model",
        "input": "witness probe",
    }


def test_ollama_embed_prefers_explicit_model_over_env(monkeypatch: Any) -> None:
    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"embedding": [0.4, 0.5, 0.6]}).encode("utf-8")

    seen_models: list[str] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        _ = timeout
        payload = json.loads((req.data or b"{}").decode("utf-8"))
        seen_models.append(str(payload.get("model", "")))
        return _FakeResponse()

    monkeypatch.setattr(world_web_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://stub.local:11434")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "env-model")

    vector = world_web_module._ollama_embed("gate probe", model="arg-model")

    assert vector == [0.4, 0.5, 0.6]
    assert seen_models == ["arg-model"]


def test_eta_mu_space_forms_support_layered_collections(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        world_web_module, "ETA_MU_INGEST_VECSTORE_LAYER_MODE", "signature"
    )
    monkeypatch.setattr(
        world_web_module, "ETA_MU_INGEST_VECSTORE_COLLECTION", "eta_mu_nexus_v1"
    )

    spaces = world_web_module._eta_mu_space_forms()

    text_collection = str(spaces.get("text", {}).get("collection", ""))
    image_collection = str(spaces.get("image", {}).get("collection", ""))
    assert text_collection.startswith("eta_mu_nexus_v1__")
    assert image_collection.startswith("eta_mu_nexus_v1__")
    assert text_collection != image_collection

    vecstore = spaces.get("vecstore", {})
    assert vecstore.get("layer_mode") == "signature"
    collections = vecstore.get("collections", {})
    assert collections.get("text") == text_collection
    assert collections.get("image") == image_collection


def test_simulation_state_includes_file_graph_nodes() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 2,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 2,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:abc",
                    "name": "new_witness_note.md",
                    "kind": "text",
                    "x": 0.22,
                    "y": 0.31,
                    "hue": 212,
                    "importance": 0.7,
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 7,
                "file_count": 1,
                "edge_count": 1,
                "kind_counts": {"text": 1},
                "field_counts": {"f6": 1},
                "knowledge_entries": 1,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    assert simulation.get("file_graph", {}).get("record") == "ημ.file-graph.v1"
    assert simulation.get("total", 0) >= 1
    assert len(simulation.get("points", [])) == simulation.get("total", 0)


def test_catalog_includes_crawler_graph_nodes_and_edges(monkeypatch: Any) -> None:
    def fake_fetch(_part_root: Path) -> dict[str, Any]:
        return {
            "ok": True,
            "source": "http://127.0.0.1:8793/api/weaver/graph",
            "status": {"alive": 1, "queue_size": 2},
            "graph": {
                "nodes": [
                    {
                        "id": "url:https://example.org/guide",
                        "kind": "url",
                        "url": "https://example.org/guide",
                        "domain": "example.org",
                        "title": "Guide",
                        "status": "fetched",
                        "depth": 1,
                        "compliance": "allowed",
                    },
                    {
                        "id": "domain:example.org",
                        "kind": "domain",
                        "domain": "example.org",
                    },
                ],
                "edges": [
                    {
                        "id": "edge:1",
                        "source": "url:https://example.org/guide",
                        "target": "domain:example.org",
                        "kind": "domain_membership",
                    }
                ],
                "counts": {"nodes_total": 2, "edges_total": 1, "url_nodes_total": 1},
            },
        }

    monkeypatch.setattr(world_web_module, "_fetch_weaver_graph_payload", fake_fetch)

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        catalog = collect_catalog(part, vault)
        crawler_graph = catalog.get("crawler_graph", {})
        assert crawler_graph.get("record") == "ημ.crawler-graph.v1"
        assert crawler_graph.get("stats", {}).get("crawler_count", 0) >= 2
        assert crawler_graph.get("stats", {}).get("edge_count", 0) >= 2
        assert crawler_graph.get("status", {}).get("alive") == 1
        assert any(
            str(node.get("url", "")).startswith("https://")
            for node in crawler_graph.get("crawler_nodes", [])
        )


def test_simulation_state_includes_crawler_graph_nodes() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "crawler_graph": {
            "record": "ημ.crawler-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "source": {
                "endpoint": "http://127.0.0.1:8793/api/weaver/graph",
                "service": "web-graph-weaver",
            },
            "status": {"alive": 1, "queue_size": 0},
            "nodes": [],
            "field_nodes": [],
            "crawler_nodes": [
                {
                    "id": "crawler:abc",
                    "node_id": "url:https://example.org",
                    "node_type": "crawler",
                    "crawler_kind": "url",
                    "label": "https://example.org",
                    "x": 0.68,
                    "y": 0.32,
                    "hue": 200,
                    "importance": 0.8,
                    "url": "https://example.org",
                    "dominant_field": "f2",
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 7,
                "crawler_count": 1,
                "edge_count": 0,
                "kind_counts": {"url": 1},
                "field_counts": {"f2": 1},
                "nodes_total": 1,
                "edges_total": 0,
                "url_nodes_total": 1,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    assert simulation.get("crawler_graph", {}).get("record") == "ημ.crawler-graph.v1"
    assert simulation.get("total", 0) >= 1
    assert len(simulation.get("points", [])) == simulation.get("total", 0)


def test_catalog_includes_truth_state_snapshot() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        catalog = collect_catalog(part, vault)
        truth_state = catalog.get("truth_state", {})
        assert truth_state.get("record") == "ημ.truth-state.v1"
        assert truth_state.get("claim", {}).get("id") == "claim.push_truth_gate_ready"
        assert isinstance(truth_state.get("claims"), list)
        assert isinstance(truth_state.get("proof", {}).get("entries"), list)


def test_simulation_state_includes_truth_state() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "truth_state": {
            "record": "ημ.truth-state.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "name_binding": {
                "id": "gates_of_truth",
                "symbol": "Gates_of_Truth",
                "glyph": "真",
                "ascii": "TRUTH",
                "law": "Truth requires world scope (ω) + proof refs + receipts.",
            },
            "world": {
                "id": "127.0.0.1:8787",
                "ctx/ω-world": "127.0.0.1:8787",
                "ctx_omega_world": "127.0.0.1:8787",
            },
            "claim": {
                "id": "claim.push_truth_gate_ready",
                "text": "push-truth gate is ready for apply",
                "status": "proved",
                "kappa": 0.86,
                "world": "127.0.0.1:8787",
                "proof_refs": ["runtime:/api/push-truth/dry-run"],
                "theta": 0.72,
            },
            "claims": [
                {
                    "id": "claim.push_truth_gate_ready",
                    "text": "push-truth gate is ready for apply",
                    "status": "proved",
                    "kappa": 0.86,
                    "world": "127.0.0.1:8787",
                    "proof_refs": ["runtime:/api/push-truth/dry-run"],
                    "theta": 0.72,
                }
            ],
            "guard": {"theta": 0.72, "passes": True},
            "gate": {"target": "push-truth", "blocked": False, "reasons": []},
            "invariants": {
                "world_scoped": True,
                "proof_required": True,
                "proof_kind_subset": True,
                "receipts_parse_ok": True,
                "sim_bead_mint_blocked": True,
                "truth_binding_registered": True,
            },
            "proof": {
                "required_kinds": [":logic/bridge"],
                "entries": [
                    {
                        "kind": ":logic/bridge",
                        "ref": "manifest.lith",
                        "present": True,
                        "detail": "manifest proof-schema source",
                    }
                ],
                "counts": {"total": 1, "present": 1, "by_kind": {":logic/bridge": 1}},
            },
            "artifacts": {
                "pi_zip_count": 1,
                "host_handle": "github:err",
                "host_has_github_gist": True,
                "truth_receipt_count": 1,
                "decision_receipt_count": 1,
            },
            "schema": {
                "source": "manifest.lith",
                "required_refs": ["receipts.log"],
                "required_hashes": ["sha256:manifest"],
                "host_handle": "github:err",
                "missing_refs": [],
                "missing_hashes": [],
            },
            "needs": [],
        },
    }

    simulation = build_simulation_state(catalog)
    assert simulation.get("truth_state", {}).get("record") == "ημ.truth-state.v1"
    assert simulation.get("truth_state", {}).get("claim", {}).get("status") == "proved"
    assert simulation.get("total", 0) >= 1
    assert len(simulation.get("points", [])) == simulation.get("total", 0)


def test_simulation_state_exposes_logical_graph_and_pain_field() -> None:
    source_path = "notes/proof.md"
    file_id = world_web_module._file_id_for_path(source_path)
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "/tmp/.ημ",
                "pending_count": 0,
                "processed_count": 1,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 1,
                "last_ingested_at": "2026-02-16T00:00:00+00:00",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:proof",
                    "name": "proof.md",
                    "label": "proof.md",
                    "kind": "text",
                    "x": 0.31,
                    "y": 0.42,
                    "hue": 212,
                    "importance": 0.7,
                    "source_rel_path": source_path,
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 7,
                "file_count": 1,
                "edge_count": 0,
                "kind_counts": {"text": 1},
                "field_counts": {"f6": 1},
                "knowledge_entries": 1,
            },
        },
        "truth_state": {
            "record": "ημ.truth-state.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "claim": {
                "id": "claim.push_truth_gate_ready",
                "text": "push-truth gate is ready for apply",
                "status": "undecided",
                "kappa": 0.5,
                "world": "127.0.0.1:8787",
                "proof_refs": [source_path],
                "theta": 0.72,
            },
            "claims": [
                {
                    "id": "claim.push_truth_gate_ready",
                    "text": "push-truth gate is ready for apply",
                    "status": "undecided",
                    "kappa": 0.5,
                    "world": "127.0.0.1:8787",
                    "proof_refs": [source_path],
                    "theta": 0.72,
                }
            ],
            "proof": {
                "required_kinds": [":logic/bridge"],
                "entries": [
                    {
                        "kind": ":logic/bridge",
                        "ref": source_path,
                        "present": True,
                        "detail": "manifest proof-schema source",
                    }
                ],
            },
            "gate": {
                "target": "push-truth",
                "blocked": True,
                "reasons": ["missing-receipt"],
            },
        },
        "test_failures": [
            {
                "name": "test_push_truth_gate",
                "status": "failed",
                "message": "missing receipt",
                "covered_files": [source_path],
                "severity": 0.9,
            }
        ],
    }

    simulation = build_simulation_state(catalog)
    logical_graph = simulation.get("logical_graph", {})
    assert logical_graph.get("record") == "ημ.logical-graph.v1"
    assert (
        logical_graph.get("joins", {}).get("file_index", {}).get(source_path) == file_id
    )
    assert logical_graph.get("stats", {}).get("fact_nodes", 0) >= 1

    pain_field = simulation.get("pain_field", {})
    assert pain_field.get("record") == "ημ.pain-field.v1"
    assert pain_field.get("active") is True
    failing = pain_field.get("failing_tests", [])
    assert isinstance(failing, list)
    assert failing and file_id in failing[0].get("file_ids", [])
    assert any(row.get("heat", 0) > 0.5 for row in pain_field.get("node_heat", []))
    debug_target = pain_field.get("debug", {})
    assert debug_target.get("meaning") == "DEBUG"
    assert debug_target.get("grounded") is True
    assert debug_target.get("file_id") == file_id
    assert debug_target.get("reason") == "points-to-hottest-file"


def test_pain_field_ingests_test_covers_span_relations() -> None:
    source_path = "notes/proof.md"
    file_id = world_web_module._file_id_for_path(source_path)
    logical_graph = {
        "nodes": [
            {
                "id": "logical:file:proof",
                "kind": "file",
                "file_id": file_id,
                "path": source_path,
                "label": "proof.md",
                "x": 0.31,
                "y": 0.42,
            }
        ],
        "edges": [],
        "joins": {
            "file_index": {source_path: file_id},
        },
    }
    catalog = {
        "test_failures": [
            {
                "name": "test_push_truth_gate",
                "status": "failed",
                "message": "missing receipt",
                "severity": 1.0,
            }
        ],
        "test_coverage": {
            "by_test_spans": {
                "test_push_truth_gate": [
                    {
                        "file": source_path,
                        "start_line": 12,
                        "end_line": 20,
                        "weight": 0.75,
                    },
                    {
                        "file": source_path,
                        "start_line": 28,
                        "end_line": 34,
                        "weight": 0.25,
                    },
                ]
            }
        },
    }

    pain_field = world_web_module._build_pain_field(catalog, logical_graph)
    relations = pain_field.get("relations", {})
    test_covers = relations.get("覆/test-covers-span", [])
    span_maps = relations.get("覆/span-maps-to-region", [])

    assert len(test_covers) == 2
    assert len(span_maps) == 2
    assert all(float(row.get("w", 0.0)) > 0.0 for row in test_covers)

    failing = pain_field.get("failing_tests", [])
    assert len(failing) == 1
    assert len(failing[0].get("span_ids", [])) == 2
    assert len(failing[0].get("region_ids", [])) == 1

    spans = pain_field.get("spans", [])
    assert len(spans) == 2
    assert {int(row.get("start_line", 0)) for row in spans} == {12, 28}

    heat_regions = pain_field.get("heat_regions", [])
    assert heat_regions and float(heat_regions[0].get("heat", 0.0)) > 0.0
    assert any(
        row.get("node_id") == "logical:file:proof"
        for row in pain_field.get("node_heat", [])
    )

    heat_values = world_web_module._materialize_heat_values(
        {
            "logical_graph": logical_graph,
            "pain_field": pain_field,
        },
        pain_field,
    )
    assert heat_values.get("record") == "ημ.heat-values.v1"
    assert heat_values.get("active") is True
    facts = heat_values.get("facts", [])
    assert any(str(row.get("kind", "")) == "熱/value" for row in facts)
    assert any(float(row.get("value", 0.0)) > 0.0 for row in facts)
    locate_rows = heat_values.get("locate", [])
    assert isinstance(locate_rows, list)
    assert any(str(row.get("kind", "")) == "址" for row in locate_rows)


def test_load_test_signal_artifacts_prefers_lcov_and_failing_test_list() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part_root = root / "part"
        vault_root = root / "vault"
        (part_root / "world_state").mkdir(parents=True)
        (part_root / "coverage").mkdir(parents=True)
        (vault_root / ".opencode" / "runtime").mkdir(parents=True)

        (part_root / "world_state" / "failing_tests.txt").write_text(
            "test_receipt_gate\ntest_truth_flow | code/world_web.py code/lore.py\n",
            encoding="utf-8",
        )
        (part_root / "coverage" / "lcov.info").write_text(
            "TN:test_receipt_gate\n"
            "SF:code/world_web.py\n"
            "DA:10,0\n"
            "DA:11,0\n"
            "DA:12,1\n"
            "LF:3\n"
            "LH:1\n"
            "end_of_record\n"
            "TN:test_truth_flow\n"
            "SF:code/lore.py\n"
            "DA:1,1\n"
            "DA:2,1\n"
            "LF:2\n"
            "LH:2\n"
            "end_of_record\n",
            encoding="utf-8",
        )

        failures, coverage = world_web_module._load_test_signal_artifacts(
            part_root,
            vault_root,
        )

        assert [row.get("name") for row in failures] == [
            "test_receipt_gate",
            "test_truth_flow",
        ]
        assert failures[1].get("covered_files") == ["code/world_web.py", "code/lore.py"]

        assert coverage.get("source") == "lcov"
        by_test = coverage.get("by_test", {})
        assert isinstance(by_test, dict)
        assert "test_receipt_gate" in by_test
        assert "code/world_web.py" in by_test["test_receipt_gate"]
        by_test_spans = coverage.get("by_test_spans", {})
        assert isinstance(by_test_spans, dict)
        assert "test_receipt_gate" in by_test_spans
        receipt_spans = by_test_spans.get("test_receipt_gate", [])
        assert isinstance(receipt_spans, list)
        assert receipt_spans
        assert receipt_spans[0].get("file") == "code/world_web.py"
        assert int(receipt_spans[0].get("start_line", 0)) == 12
        assert int(receipt_spans[0].get("end_line", 0)) == 12
        assert float(receipt_spans[0].get("weight", 0.0)) > 0.0

        truth_spans = by_test_spans.get("test_truth_flow", [])
        assert isinstance(truth_spans, list)
        assert truth_spans
        assert truth_spans[0].get("file") == "code/lore.py"
        assert int(truth_spans[0].get("start_line", 0)) == 1
        assert int(truth_spans[0].get("end_line", 0)) == 2
        assert coverage.get("hottest_files", [""])[0] == "code/world_web.py"


def test_pain_field_uses_hottest_coverage_file_when_failure_has_no_mapping() -> None:
    hot_path = "code/world_web.py"
    cool_path = "code/lore.py"
    hot_id = world_web_module._file_id_for_path(hot_path)
    cool_id = world_web_module._file_id_for_path(cool_path)

    logical_graph = {
        "nodes": [
            {
                "id": "logical:file:hot",
                "kind": "file",
                "file_id": hot_id,
                "x": 0.22,
                "y": 0.33,
                "label": "world_web.py",
            },
            {
                "id": "logical:file:cool",
                "kind": "file",
                "file_id": cool_id,
                "x": 0.67,
                "y": 0.58,
                "label": "lore.py",
            },
        ],
        "edges": [],
        "joins": {
            "file_index": {
                hot_path: hot_id,
                cool_path: cool_id,
            }
        },
    }
    catalog = {
        "test_failures": [
            {
                "name": "test_receipt_gate",
                "status": "failed",
                "message": "assertion failed",
            }
        ],
        "test_coverage": {
            "files": {
                hot_path: {"line_rate": 0.12, "lines_found": 120},
                cool_path: {"line_rate": 0.96, "lines_found": 120},
            }
        },
    }

    pain_field = world_web_module._build_pain_field(catalog, logical_graph)
    failing_tests = pain_field.get("failing_tests", [])

    assert failing_tests
    assert hot_path in failing_tests[0].get("covered_files", [])
    assert hot_id in failing_tests[0].get("file_ids", [])
    debug_target = pain_field.get("debug", {})
    assert debug_target.get("meaning") == "DEBUG"
    assert debug_target.get("grounded") is True
    assert debug_target.get("path") == hot_path
    assert debug_target.get("file_id") == hot_id
    assert debug_target.get("source") in {
        "pain_field.max_heat",
        "coverage.hottest_files",
    }
    assert any(
        row.get("file_id") == hot_id and row.get("heat", 0) > 0.0
        for row in pain_field.get("node_heat", [])
    )


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


def test_council_consider_event_blocks_when_gate_is_blocked(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "DOCKER_AUTORESTART_ENABLED", True)
    monkeypatch.setattr(world_web_module, "COUNCIL_MIN_OVERLAP_MEMBERS", 2)
    monkeypatch.setattr(world_web_module, "DOCKER_AUTORESTART_INCLUDE_GLOBS", "**/*.md")
    monkeypatch.setattr(world_web_module, "DOCKER_AUTORESTART_EXCLUDE_GLOBS", "")
    monkeypatch.setattr(
        world_web_module,
        "build_drift_scan_payload",
        lambda _part_root, _vault_root: {
            "blocked_gates": [{"reason": "missing-receipt"}]
        },
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "part64"
        part.mkdir(parents=True)
        (part / "docker-compose.yml").write_text(
            "services:\n  eta-mu-system:\n    image: busybox\n",
            encoding="utf-8",
        )

        chamber = CouncilChamber(
            root / "council.jsonl",
            root / "receipts.log",
            owner="Err",
            host="127.0.0.1:8787",
            part_root=part,
            vault_root=root,
        )
        result = chamber.consider_event(
            event_type="file_changed",
            data={"path": "notes/proof.md"},
            catalog={
                "file_graph": {
                    "file_nodes": [
                        {
                            "id": "file:proof",
                            "name": "proof.md",
                            "source_rel_path": "notes/proof.md",
                            "field_scores": {"f7": 0.7},
                        }
                    ]
                }
            },
            influence_snapshot={"task_queue": {"pending_count": 0}},
        )

        assert result["ok"] is True
        decision = result["decision"]
        assert decision["status"] == "blocked"
        assert decision["gate"]["blocked"] is True
        assert decision["action"]["attempted"] is False


def test_council_manual_vote_can_execute_restart(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "DOCKER_AUTORESTART_ENABLED", True)
    monkeypatch.setattr(world_web_module, "DOCKER_AUTORESTART_REQUIRE_COUNCIL", True)
    monkeypatch.setattr(world_web_module, "COUNCIL_MIN_OVERLAP_MEMBERS", 2)
    monkeypatch.setattr(world_web_module, "DOCKER_AUTORESTART_INCLUDE_GLOBS", "**/*.md")
    monkeypatch.setattr(world_web_module, "DOCKER_AUTORESTART_EXCLUDE_GLOBS", "")
    monkeypatch.setattr(
        world_web_module, "DOCKER_AUTORESTART_SERVICES", "eta-mu-system"
    )
    monkeypatch.setattr(
        world_web_module,
        "build_drift_scan_payload",
        lambda _part_root, _vault_root: {"blocked_gates": []},
    )
    base_auto_vote = world_web_module._council_auto_vote

    def _forced_auto_vote(member_id: str, **kwargs: Any) -> tuple[str, str]:
        if str(member_id) == "the_council":
            return "no", "forced no vote for manual override path"
        return base_auto_vote(member_id, **kwargs)

    monkeypatch.setattr(world_web_module, "_council_auto_vote", _forced_auto_vote)

    run_calls: list[list[str]] = []

    def _fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        run_calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(world_web_module.subprocess, "run", _fake_run)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "part64"
        part.mkdir(parents=True)
        (part / "docker-compose.yml").write_text(
            "services:\n  eta-mu-system:\n    image: busybox\n",
            encoding="utf-8",
        )

        chamber = CouncilChamber(
            root / "council.jsonl",
            root / "receipts.log",
            owner="Err",
            host="127.0.0.1:8787",
            part_root=part,
            vault_root=root,
        )

        considered = chamber.consider_event(
            event_type="file_changed",
            data={"path": "notes/core.md"},
            catalog={
                "file_graph": {
                    "file_nodes": [
                        {
                            "id": "file:core",
                            "name": "core.md",
                            "source_rel_path": "notes/core.md",
                            "field_scores": {},
                        }
                    ]
                }
            },
            influence_snapshot={"task_queue": {"pending_count": 0}},
        )

        decision = considered["decision"]
        assert decision["status"] == "awaiting-votes"
        decision_id = str(decision["id"])

        voted = chamber.vote(
            decision_id=decision_id,
            member_id="the_council",
            vote="yes",
            reason="manual approval",
            actor="tester",
        )
        assert voted["ok"] is True
        final_decision = voted["decision"]
        assert final_decision["status"] == "executed"
        assert final_decision["action"]["ok"] is True
        assert run_calls


def test_build_study_snapshot_reports_stability_and_warnings() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "part64"
        part.mkdir(parents=True)
        receipts_path = root / "receipts.log"
        receipts_path.write_text("ts=demo\n", encoding="utf-8")

        payload = build_study_snapshot(
            part,
            root,
            queue_snapshot={
                "queue_log": str(root / "queue.jsonl"),
                "pending_count": 3,
                "dedupe_keys": 2,
                "event_count": 9,
                "pending": [{"id": "task:1", "kind": "repair"}],
            },
            council_snapshot={
                "decision_log": str(root / "council.jsonl"),
                "decision_count": 3,
                "pending_count": 2,
                "approved_count": 1,
                "auto_restart_enabled": True,
                "require_council": True,
                "cooldown_seconds": 25,
                "decisions": [
                    {"id": "decision:1", "status": "blocked"},
                    {"id": "decision:2", "status": "executed"},
                ],
            },
            drift_payload={
                "active_drifts": [
                    {
                        "id": "open_questions_unresolved",
                        "severity": "medium",
                        "detail": "2 open questions unresolved",
                    }
                ],
                "blocked_gates": [
                    {
                        "target": "push-truth",
                        "reason": "open-questions-unresolved",
                    }
                ],
                "open_questions": {
                    "total": 2,
                    "resolved_count": 0,
                    "unresolved_count": 2,
                },
                "receipts_parse": {
                    "path": str(receipts_path),
                    "ok": True,
                    "rows": 12,
                    "has_intent_ref": True,
                },
            },
            truth_gate_blocked=True,
        )

        assert payload.get("ok") is True
        assert payload.get("record") == "ημ.study-snapshot.v1"
        assert payload.get("signals", {}).get("blocked_gate_count") == 1
        assert payload.get("signals", {}).get("queue_pending_count") == 3
        assert payload.get("signals", {}).get("council_pending_count") == 2
        assert payload.get("signals", {}).get("truth_gate_blocked") is True
        assert payload.get("stability", {}).get("label") in {
            "stable",
            "watch",
            "unstable",
        }
        assert payload.get("runtime", {}).get("receipts_path_within_vault") is True
        warnings = payload.get("warnings", [])
        assert any(item.get("code") == "gate.blocked" for item in warnings)
        assert any(item.get("code") == "drift.open_questions" for item in warnings)


def test_build_study_snapshot_flags_receipts_path_outside_vault() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "part64"
        part.mkdir(parents=True)

        payload = build_study_snapshot(
            part,
            root,
            queue_snapshot={
                "queue_log": str(root / "queue.jsonl"),
                "pending_count": 0,
                "dedupe_keys": 0,
                "event_count": 0,
                "pending": [],
            },
            council_snapshot={
                "decision_log": str(root / "council.jsonl"),
                "decision_count": 0,
                "pending_count": 0,
                "approved_count": 0,
                "auto_restart_enabled": True,
                "require_council": True,
                "cooldown_seconds": 25,
                "decisions": [],
            },
            drift_payload={
                "active_drifts": [],
                "blocked_gates": [],
                "open_questions": {
                    "total": 0,
                    "resolved_count": 0,
                    "unresolved_count": 0,
                },
                "receipts_parse": {
                    "path": "/receipts.log",
                    "ok": True,
                    "rows": 0,
                    "has_intent_ref": False,
                },
            },
            truth_gate_blocked=False,
        )

        assert payload.get("runtime", {}).get("receipts_path_within_vault") is False
        warnings = payload.get("warnings", [])
        assert any(
            item.get("code") == "runtime.receipts_path_outside_vault"
            for item in warnings
        )


def test_export_study_snapshot_roundtrip_appends_history_and_receipt(
    monkeypatch: Any,
) -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "part64"
        part.mkdir(parents=True)
        receipts_path = root / "receipts.log"
        monkeypatch.setattr(
            world_web_module,
            "_ensure_receipts_log_path",
            lambda _vault_root, _part_root: receipts_path,
        )

        result = world_web_module.export_study_snapshot(
            part,
            root,
            queue_snapshot={
                "queue_log": str(root / "queue.jsonl"),
                "pending_count": 1,
                "dedupe_keys": 1,
                "event_count": 2,
                "pending": [{"id": "task:1", "kind": "study"}],
            },
            council_snapshot={
                "decision_log": str(root / "council.jsonl"),
                "decision_count": 1,
                "pending_count": 1,
                "approved_count": 0,
                "auto_restart_enabled": True,
                "require_council": True,
                "cooldown_seconds": 25,
                "decisions": [{"id": "decision:one", "status": "blocked"}],
            },
            drift_payload={
                "active_drifts": [
                    {
                        "id": "open_questions_unresolved",
                        "severity": "medium",
                        "detail": "1 unresolved",
                    }
                ],
                "blocked_gates": [
                    {
                        "target": "push-truth",
                        "reason": "open-questions-unresolved",
                    }
                ],
                "open_questions": {
                    "total": 1,
                    "resolved_count": 0,
                    "unresolved_count": 1,
                },
                "receipts_parse": {
                    "path": str(root / "receipts.log"),
                    "ok": True,
                    "rows": 0,
                    "has_intent_ref": False,
                },
            },
            truth_gate_blocked=True,
            label="unit-export",
            owner="Tester",
            host="127.0.0.1:8791",
        )

        assert result.get("ok") is True
        event = result.get("event", {})
        assert str(event.get("id", "")).startswith("study:")
        assert event.get("label") == "unit-export"
        assert event.get("owner") == "Tester"

        history = result.get("history", {})
        assert history.get("record") == "ημ.study-history.v1"
        latest = history.get("latest", [])
        assert isinstance(latest, list)
        assert latest
        assert latest[0].get("id") == event.get("id")

        log_path = Path(str(history.get("path", "")))
        assert log_path.exists()

        loaded = world_web_module._load_study_snapshot_events(root, limit=4)
        assert loaded
        assert loaded[0].get("id") == event.get("id")

        receipts = receipts_path.read_text("utf-8")
        assert "origin=study" in receipts
        assert "study:export" in receipts


def test_study_snapshot_history_limit_returns_latest_first() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        event_a = {
            "v": "eta-mu.study/v1",
            "ts": "2026-02-16T10:00:00+00:00",
            "op": "export",
            "id": "study:a",
        }
        event_b = {
            "v": "eta-mu.study/v1",
            "ts": "2026-02-16T10:00:01+00:00",
            "op": "export",
            "id": "study:b",
        }

        world_web_module._append_study_snapshot_event(root, event_a)
        world_web_module._append_study_snapshot_event(root, event_b)

        rows = world_web_module._load_study_snapshot_events(root, limit=1)
        assert len(rows) == 1
        assert rows[0].get("id") == "study:b"
