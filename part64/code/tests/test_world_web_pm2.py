from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import wave
import zipfile
from array import array
from pathlib import Path
from typing import Any, cast

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
    build_witness_lineage_payload,
    build_ui_projection,
    build_voice_lines,
    build_world_payload,
    build_mix_stream,
    build_presence_say_payload,
    build_push_truth_dry_run_payload,
    build_world_log_payload,
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


def _css_rule_block(css_source: str, selector: str) -> str:
    marker = f"{selector} {{"
    start = css_source.find(marker)
    assert start >= 0, f"missing css selector: {selector}"
    end = css_source.find("}\n", start)
    if end < 0:
        end = css_source.find("}", start)
    assert end >= 0, f"unterminated css selector: {selector}"
    return css_source[start : end + 1]


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


def test_library_resolution_falls_back_to_archived_eta_mu_source_path() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vault = root / "vault"
        vault.mkdir(parents=True)

        archive_rel = ".opencode/knowledge/archive/2026/02/16/demo.png"
        archive_path = (root / archive_rel).resolve()
        archive_path.parent.mkdir(parents=True)
        archive_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        index_path = root / ".opencode" / "runtime" / "eta_mu_knowledge.v1.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_row = {
            "record": "ημ.knowledge.v1",
            "id": "knowledge.demo",
            "source_rel_path": ".ημ/ChatGPT Image Feb 15, 2026, 01_50_05 PM.png",
            "archive_rel_path": archive_rel,
            "ingested_at": "2026-02-16T01:51:13.553990+00:00",
        }
        index_path.write_text(json.dumps(index_row) + "\n", encoding="utf-8")

        resolved = resolve_library_path(
            vault,
            "/library/.%CE%B7%CE%BC/ChatGPT%20Image%20Feb%2015%2C%202026%2C%2001_50_05%20PM.png",
        )

        assert resolved is not None
        assert resolved == archive_path


def test_pm2_parse_args_defaults() -> None:
    args = parse_pm2_args(["start"])
    assert args.command == "start"
    assert args.port == 8787
    assert args.host == "127.0.0.1"
    assert args.name == "eta-mu-world"


def test_world_panel_layering_interaction_guards_present() -> None:
    part_root = Path(__file__).resolve().parents[2]
    viewport_path = (
        part_root
        / "frontend"
        / "src"
        / "components"
        / "App"
        / "WorldPanelsViewport.tsx"
    )
    css_path = part_root / "frontend" / "src" / "index.css"

    viewport_source = viewport_path.read_text("utf-8")
    css_source = css_path.read_text("utf-8")

    assert "renderFocusPane" in viewport_source
    assert "world-council-grid" in viewport_source
    assert "onAdjustPanelCouncilRank" in viewport_source
    assert "onPinPanelToTertiary" in viewport_source
    assert "world-unity-sidebar" in viewport_source
    assert "world-unity-icon-strip" in viewport_source
    assert "world-unity-icon-cluster" in viewport_source
    assert "world-unity-inline-card" in viewport_source
    assert "world-glass-pan-pad" in viewport_source
    assert "mouse lock: pane holds this card" in viewport_source
    assert "pending rank on release" in viewport_source
    assert "focusCandidateIds" in viewport_source
    assert "if (!panelById.has(trayPanelId)" in viewport_source
    assert "isGlassForwardCandidate" in viewport_source
    assert "preferred simulation frame" in viewport_source

    body_block = _css_rule_block(css_source, ".world-panel-body")
    assert "overscroll-behavior: contain;" in body_block
    assert "-webkit-overflow-scrolling: touch;" in body_block

    edit_tag_block = _css_rule_block(css_source, ".world-panel-edit-tag")
    assert "pointer-events: none;" in edit_tag_block

    rack_block = _css_rule_block(css_source, ".world-panel-chip-rack")
    assert "pointer-events: none;" in rack_block

    chip_block = _css_rule_block(css_source, ".world-panel-chip")
    assert "pointer-events: auto;" in chip_block


def test_world_panel_world_space_state_space_contract_present() -> None:
    part_root = Path(__file__).resolve().parents[2]
    app_path = part_root / "frontend" / "src" / "App.tsx"
    viewport_path = (
        part_root
        / "frontend"
        / "src"
        / "components"
        / "App"
        / "WorldPanelsViewport.tsx"
    )
    css_path = part_root / "frontend" / "src" / "index.css"

    app_source = app_path.read_text("utf-8")
    viewport_source = viewport_path.read_text("utf-8")
    css_source = css_path.read_text("utf-8")

    assert "const [panelWorldBiases, setPanelWorldBiases]" in app_source
    assert "const [panelWindowStates, setPanelWindowStates]" in app_source
    assert (
        "const panelWindowStateById = useMemo<Record<string, PanelWindowState>>"
        in app_source
    )
    assert (
        "const activatePanelWindow = useCallback((panelId: string) => {" in app_source
    )
    assert (
        "const minimizePanelWindow = useCallback((panelId: string) => {" in app_source
    )
    assert "const closePanelWindow = useCallback((panelId: string) => {" in app_source
    assert "const panelStateSpaceBiases = useMemo(() => {" in app_source
    assert "const worldDeltaX = info.offset.x / pixelsPerWorldX" in app_source
    assert "function shouldRouteWheelToCore" in app_source
    assert (
        'window.addEventListener("wheel", onGlobalWheel, { passive: false, capture: true });'
        in app_source
    )
    assert "transition-colors pointer-events-none" in app_source
    assert "w-full" in app_source
    assert "shadow-[0_12px_30px_rgba(2,8,14,0.34)] pointer-events-auto" in app_source
    assert "panelWorldX" in app_source
    assert "pixelsPerWorldX" in app_source
    assert 'id: "nexus.ui.world_log"' in app_source
    assert "id: GLASS_VIEWPORT_PANEL_ID" in app_source
    assert "<WorldLogPanel catalog={catalog} />" in app_source
    assert "onNudgeCameraPan={nudgeCameraPan}" in app_source
    assert "isGlassPrimaryPanelId" in app_source
    assert "Glass lane preferred" in app_source
    assert "if (!isWideViewport)" not in app_source

    assert 'className="world-council-root"' in viewport_source
    assert "world-council-grid" in viewport_source
    assert "world-council-shell" in viewport_source
    assert "world-unity-sidebar" in viewport_source
    assert "world-unity-icon-strip" in viewport_source
    assert "world-unity-icon-cluster" in viewport_source
    assert "world-unity-inline-card" in viewport_source
    assert "world-task-tray-glass" in viewport_source
    assert "world-orbital-dock" not in viewport_source
    assert 'renderFocusPane("primary", primaryPanelId)' in viewport_source
    assert "onAdjustPanelCouncilRank" in viewport_source
    assert "onPinPanelToTertiary" in viewport_source
    assert "onMinimizePanel" in viewport_source
    assert "onClosePanel" in viewport_source
    assert "overlay-constellation" not in viewport_source
    assert "onGridPanelDragEnd" not in viewport_source

    council_root_block = _css_rule_block(css_source, ".world-council-root")
    assert "pointer-events: auto;" in council_root_block

    focus_block = _css_rule_block(css_source, ".world-focus-pane")
    assert "max-height: calc(100vh - 12.2rem);" in focus_block

    unity_sidebar_block = _css_rule_block(css_source, ".world-unity-sidebar")
    assert "display: grid;" in unity_sidebar_block

    unity_strip_block = _css_rule_block(css_source, ".world-unity-icon-strip")
    assert "overflow: auto;" in unity_strip_block
    assert "max-height:" in unity_strip_block

    unity_cluster_block = _css_rule_block(css_source, ".world-unity-icon-cluster")
    assert "display: grid;" in unity_cluster_block

    assert ".world-smart-card-actions" in css_source
    assert ".world-unity-icon" in css_source
    assert ".world-focus-action-close" in css_source
    assert ".world-focus-pane-glass" in css_source


def test_hologram_canvas_remote_resource_metadata_contract() -> None:
    part_root = Path(__file__).resolve().parents[2]
    backdrop_path = (
        part_root / "frontend" / "src" / "components" / "App" / "CoreBackdrop.tsx"
    )
    canvas_path = (
        part_root / "frontend" / "src" / "components" / "Simulation" / "Canvas.tsx"
    )
    css_path = part_root / "frontend" / "src" / "index.css"

    backdrop_source = backdrop_path.read_text("utf-8")
    canvas_source = canvas_path.read_text("utf-8")
    css_source = css_path.read_text("utf-8")

    assert "interactive={false}" not in backdrop_source
    assert "\n          interactive\n" in backdrop_source

    assert (
        'type GraphWorldscreenView = "website" | "editor" | "video" | "metadata";'
        in canvas_source
    )
    assert (
        'type GraphWorldscreenMode = "overview" | "conversation" | "stats";'
        in canvas_source
    )
    assert "anchorRatioX?: number;" in canvas_source
    assert "anchorRatioY?: number;" in canvas_source
    assert "function resolveWorldscreenPlacement(" in canvas_source
    assert "anchorRatioX: clamp01(xRatio)," in canvas_source
    assert "anchorRatioY: clamp01(yRatio)," in canvas_source
    assert "const worldscreenPlacement = worldscreen" in canvas_source
    assert "transformOrigin: worldscreenPlacement.transformOrigin," in canvas_source
    assert 'worldscreen.view === "metadata"' in canvas_source
    assert 'worldscreenMode === "overview"' in canvas_source
    assert 'worldscreenMode === "conversation"' in canvas_source
    assert 'worldscreenMode === "stats"' in canvas_source
    assert "isRemoteHttpUrl(worldscreenUrl)" in canvas_source
    assert "shouldOpenWorldscreen" in canvas_source
    assert "event.stopPropagation();" in canvas_source
    assert "Math.max(0.012, hit.radiusNorm * 1.8)" in canvas_source
    assert "single tap centers nexus in glass lane" in canvas_source
    assert 'resourceKind === "image"' in canvas_source
    assert "onNexusInteraction?.({" in canvas_source
    assert "/api/image/commentary" in canvas_source
    assert "/api/image/comments" in canvas_source
    assert "/api/presence/accounts/upsert" in canvas_source
    assert "parent_comment_id" in canvas_source
    assert "true_graph_embed" in canvas_source
    assert "compacted_into_nexus" in canvas_source

    assert "html,\nbody,\n#root {" in css_source
    assert "overscroll-behavior: none;" in css_source


def test_canvas_file_graph_positions_are_server_authoritative() -> None:
    part_root = Path(__file__).resolve().parents[2]
    canvas_path = (
        part_root / "frontend" / "src" / "components" / "Simulation" / "Canvas.tsx"
    )

    canvas_source = canvas_path.read_text("utf-8")

    assert "computeDocumentLayout" not in canvas_source
    assert "documentLayoutState" not in canvas_source
    assert "fileLayoutById" not in canvas_source
    assert (
        "const graphPositionForNode = (node: any): { x: number; y: number } => ({"
        in canvas_source
    )
    assert "x: clamp01(Number(node?.x ?? 0.5))" in canvas_source
    assert "y: clamp01(Number(node?.y ?? 0.5))" in canvas_source
    assert "p += vec3(" not in canvas_source
    assert "clip.xy += forceDir" not in canvas_source
    assert "vec3 p = aPos;" in canvas_source
    assert "updatePresenceParticles" not in canvas_source
    assert "drawPresenceParticles" not in canvas_source
    assert "drawPresenceParticleTelemetry" not in canvas_source
    assert "const orbitTime =" not in canvas_source
    assert "currentPositions[i] +=" not in canvas_source
    assert "currentPositions[i] = targetPositions[i];" in canvas_source
    assert "state?.presence_dynamics?.field_particles" in canvas_source


def test_world_log_panel_contract_present() -> None:
    part_root = Path(__file__).resolve().parents[2]
    app_path = part_root / "frontend" / "src" / "App.tsx"
    layout_path = part_root / "frontend" / "src" / "app" / "worldPanelLayout.ts"
    panel_path = (
        part_root / "frontend" / "src" / "components" / "Panels" / "WorldLogPanel.tsx"
    )

    app_source = app_path.read_text("utf-8")
    layout_source = layout_path.read_text("utf-8")
    panel_source = panel_path.read_text("utf-8")

    assert 'id: "nexus.ui.world_log"' in app_source
    assert "<WorldLogPanel catalog={catalog} />" in app_source
    assert '"nexus.ui.world_log": {' in layout_source
    assert 'anchorId: "witness_thread"' in layout_source

    assert 'runtimeApiUrl("/api/world/events?limit=180")' in panel_source
    assert 'runtimeApiUrl("/api/eta-mu/sync")' in panel_source
    assert "ingest .ημ now" in panel_source
    assert 'if (source !== "nasa_gibs")' in panel_source
    assert 'loading="lazy"' in panel_source
    assert 'referrerPolicy="no-referrer"' in panel_source


def test_witness_thread_ledger_panel_contract_present() -> None:
    part_root = Path(__file__).resolve().parents[2]
    app_path = part_root / "frontend" / "src" / "App.tsx"
    layout_path = part_root / "frontend" / "src" / "app" / "worldPanelLayout.ts"
    panel_path = part_root / "frontend" / "src" / "components" / "Panels" / "Chat.tsx"
    server_path = part_root / "code" / "world_web" / "server.py"

    app_source = app_path.read_text("utf-8")
    layout_source = layout_path.read_text("utf-8")
    panel_source = panel_path.read_text("utf-8")
    server_source = server_path.read_text("utf-8")

    assert 'id: "nexus.ui.chat.witness_thread"' in app_source
    assert "multi_entity: true" in app_source
    assert 'presence_ids: [resolvedMusePresenceId || "witness_thread"]' in app_source
    assert "activeMusePresenceId={activeMusePresenceId}" in app_source
    assert "onMusePresenceChange={setActiveMusePresenceId}" in app_source
    assert "emitWitnessChatReply" in app_source
    assert '"nexus.ui.chat.witness_thread": {' in layout_source
    assert 'anchorId: "witness_thread"' in layout_source

    assert "Witness Thread Ledger / 証人の糸 台帳" in panel_source
    assert "Particles Made Clear / 粒子明瞭化" in panel_source
    assert 'runtimeApiUrl("/api/witness/lineage")' in panel_source
    assert "/say witness_thread ${trimmed}" in panel_source
    assert (
        'const [chatChannel, setChatChannel] = useState<ChatChannel>("ledger")'
        in panel_source
    )

    assert 'if parsed.path == "/api/witness/lineage":' in server_source
    assert "build_witness_lineage_payload(self.part_root)" in server_source


def test_stability_observatory_panel_npu_widget_contract_present() -> None:
    part_root = Path(__file__).resolve().parents[2]
    panel_path = (
        part_root
        / "frontend"
        / "src"
        / "components"
        / "Panels"
        / "StabilityObservatoryPanel.tsx"
    )

    panel_source = panel_path.read_text("utf-8")

    assert "function resourceStatusClass(status: string): string" in panel_source
    assert "const npuDevice = runtimeResource?.devices?.npu0;" in panel_source
    assert "const npuQueueDepth =" in panel_source
    assert "npu lane" in panel_source
    assert "queue <code>{npuQueueDepth}</code>" in panel_source


def test_world_web_module_entrypoint_help() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "code.world_web", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--host" in proc.stdout
    assert "--port" in proc.stdout


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


def test_witness_lineage_payload_raises_missing_upstream_drift(
    monkeypatch: Any,
) -> None:
    def fake_run(
        cmd: list[str],
        cwd: str,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, check, capture_output, text, timeout
        args = cmd[1:]
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")
        if args == ["rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(cmd, 0, "/tmp/repo\n", "")
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, "feature/witness-ledger\n", "")
        if args == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return subprocess.CompletedProcess(cmd, 128, "", "no upstream configured")
        if args == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                "M  part64/frontend/src/App.tsx\n M receipts.log\n?? specs/drafts/witness.md\n",
                "",
            )
        if args == ["log", "-1", "--pretty=%h %s"]:
            return subprocess.CompletedProcess(cmd, 0, "abc1234 witness baseline\n", "")
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(world_web_module.subprocess, "run", fake_run)

    payload = build_witness_lineage_payload(Path("."))

    assert payload["ok"] is True
    assert payload["record"] == "ημ.witness-lineage.v1"
    assert payload["checkpoint"]["branch"] == "feature/witness-ledger"
    assert payload["checkpoint"]["upstream"] == ""
    assert payload["push_obligation"] is False
    assert payload["push_obligation_unknown"] is True
    assert payload["working_tree"]["dirty"] is True
    assert payload["working_tree"]["staged"] == 1
    assert payload["working_tree"]["unstaged"] == 1
    assert payload["working_tree"]["untracked"] == 1
    assert payload["continuity_drift"]["active"] is True
    assert payload["continuity_drift"]["code"] == "missing_upstream"


def test_witness_lineage_payload_reports_ahead_behind_when_upstream_present(
    monkeypatch: Any,
) -> None:
    def fake_run(
        cmd: list[str],
        cwd: str,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, check, capture_output, text, timeout
        args = cmd[1:]
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")
        if args == ["rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(cmd, 0, "/tmp/repo\n", "")
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, "main\n", "")
        if args == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
            return subprocess.CompletedProcess(cmd, 0, "origin/main\n", "")
        if args == ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"]:
            return subprocess.CompletedProcess(cmd, 0, "3\t2\n", "")
        if args == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if args == ["log", "-1", "--pretty=%h %s"]:
            return subprocess.CompletedProcess(cmd, 0, "def5678 synced\n", "")
        if args == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(
                cmd, 0, "git@github.com:err/fork_tales.git\n", ""
            )
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(world_web_module.subprocess, "run", fake_run)

    payload = build_witness_lineage_payload(Path("."))

    assert payload["ok"] is True
    assert payload["checkpoint"]["branch"] == "main"
    assert payload["checkpoint"]["upstream"] == "origin/main"
    assert payload["checkpoint"]["ahead"] == 3
    assert payload["checkpoint"]["behind"] == 2
    assert payload["push_obligation"] is True
    assert payload["push_obligation_unknown"] is False
    assert payload["repo"]["remote"] == "origin"
    assert payload["repo"]["remote_url"] == "git@github.com:err/fork_tales.git"
    assert payload["continuity_drift"]["active"] is True
    assert payload["continuity_drift"]["code"] == "behind_upstream"


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


def test_ollama_embed_records_compute_job_event(monkeypatch: Any) -> None:
    tracker = RuntimeInfluenceTracker()
    monkeypatch.setattr(world_web_module, "_INFLUENCE_TRACKER", tracker)
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "tensorflow")
    monkeypatch.setattr(
        world_web_module,
        "_tensorflow_embed",
        lambda text, model=None: [0.2, 0.4] if text == "job probe" else None,
    )

    result = world_web_module._ollama_embed("job probe")
    assert result == [0.2, 0.4]
    snapshot = tracker.snapshot(queue_snapshot={"pending_count": 0, "event_count": 0})
    assert snapshot.get("compute_jobs_180s") == 1
    jobs = snapshot.get("compute_jobs", [])
    assert isinstance(jobs, list)
    assert jobs[0].get("kind") == "embedding"
    assert str(jobs[0].get("op", "")).startswith("embed.")


def test_ollama_generate_text_records_compute_job_event(monkeypatch: Any) -> None:
    tracker = RuntimeInfluenceTracker()
    monkeypatch.setattr(world_web_module, "_INFLUENCE_TRACKER", tracker)
    monkeypatch.setenv("TEXT_GENERATION_BACKEND", "tensorflow")

    def _fake_tf_generate(
        prompt: str, model: str | None = None, timeout_s: float | None = None
    ) -> tuple[str | None, str]:
        del model, timeout_s
        if "gate" not in prompt:
            return None, "tensorflow-hash-v1"
        return "tf guidance", "tensorflow-hash-v1"

    monkeypatch.setattr(
        world_web_module, "_tensorflow_generate_text", _fake_tf_generate
    )
    text, model = world_web_module._ollama_generate_text("gate remains blocked")
    assert text == "tf guidance"
    assert model == "tensorflow-hash-v1"
    snapshot = tracker.snapshot(queue_snapshot={"pending_count": 0, "event_count": 0})
    assert snapshot.get("compute_summary", {}).get("llm_jobs") == 1
    jobs = snapshot.get("compute_jobs", [])
    assert isinstance(jobs, list)
    assert jobs[0].get("kind") == "llm"
    assert str(jobs[0].get("op", "")).startswith("text_generate.")


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


def test_presence_accounts_and_image_comments_persist_in_runtime_db() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        account = world_web_module._upsert_presence_account(
            vault,
            presence_id="witness_thread",
            display_name="Witness Thread",
            handle="witness_thread",
            avatar="",
            bio="Tracks continuity.",
            tags=["witness", "continuity"],
        )
        assert account["ok"] is True
        listing = world_web_module._list_presence_accounts(vault, limit=10)
        assert listing["ok"] is True
        assert listing["count"] >= 1
        assert any(
            str(row.get("presence_id", "")) == "witness_thread"
            for row in listing.get("entries", [])
        )

        created = world_web_module._create_image_comment(
            vault,
            image_ref="artifacts/images/demo.png",
            presence_id="witness_thread",
            comment="The edge contrast highlights gate boundaries.",
            metadata={"model": "qwen3-vl:2b-instruct", "backend": "test"},
        )
        assert created["ok"] is True
        comments = world_web_module._list_image_comments(
            vault,
            image_ref="artifacts/images/demo.png",
            limit=20,
        )
        assert comments["ok"] is True
        assert comments["count"] >= 1
        assert comments["entries"][0]["presence_id"] == "witness_thread"


def test_world_web_server_exposes_image_commentary_and_account_endpoints() -> None:
    part_root = Path(__file__).resolve().parents[2]
    server_path = part_root / "code" / "world_web" / "server.py"
    source = server_path.read_text("utf-8")

    assert 'if parsed.path == "/api/presence/accounts":' in source
    assert 'if parsed.path == "/api/presence/accounts/upsert":' in source
    assert 'if parsed.path == "/api/image/comments":' in source
    assert 'if parsed.path == "/api/image/commentary":' in source
    assert 'if parsed.path == "/api/world/events":' in source
    assert 'if parsed.path == "/api/eta-mu/sync":' in source
    assert "build_image_commentary" in source
    assert "build_world_log_payload" in source
    assert "sync_eta_mu_inbox" in source


def test_build_image_commentary_falls_back_without_ollama(
    monkeypatch: Any,
) -> None:
    from code.world_web import ai as ai_module

    monkeypatch.setattr(
        ai_module,
        "_tensorflow_image_fingerprint",
        lambda _bytes: {
            "ok": True,
            "error": "",
            "width": 320,
            "height": 200,
            "channels": 3,
            "mean_luma": 0.42,
        },
    )

    def _raise_unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("ollama unavailable")

    monkeypatch.setattr(world_web_module, "urlopen", _raise_unavailable)

    payload = world_web_module.build_image_commentary(
        image_bytes=b"\x89PNG\r\n\x1a\n",
        mime="image/png",
        image_ref="artifacts/images/demo.png",
        presence_id="witness_thread",
        prompt="Summarize this image",
    )
    assert payload["ok"] is True
    assert payload["backend"] == "tensorflow-fallback"
    assert "witness_thread" in str(payload.get("commentary", ""))


def test_eta_mu_image_derive_segment_adds_vllm_caption_for_embedding(
    monkeypatch: Any,
) -> None:
    from code.world_web import ai as ai_module

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
        headers = {str(k).lower(): str(v) for k, v in req.header_items()}
        calls.append(
            {
                "url": str(req.full_url),
                "timeout": timeout,
                "payload": payload,
                "headers": headers,
            }
        )
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "A rocky coastline with a red lighthouse near rough water."
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(world_web_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("ETA_MU_IMAGE_VISION_ENABLED", "1")
    monkeypatch.setenv("ETA_MU_IMAGE_VISION_BASE_URL", "http://vllm.local:8001")
    monkeypatch.setenv("ETA_MU_IMAGE_VISION_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    monkeypatch.setenv("ETA_MU_IMAGE_VISION_API_KEY", "test-key")
    monkeypatch.setenv("ETA_MU_IMAGE_VISION_TIMEOUT_SECONDS", "7")

    segment = ai_module._eta_mu_image_derive_segment(
        source_hash="abc123",
        source_bytes=8,
        source_rel_path=".ημ/demo.png",
        mime="image/png",
        image_bytes=b"\x89PNG\r\n\x1a\n",
    )

    assert segment.get("vision_backend") == "vllm"
    assert segment.get("vision_model") == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert "vision-caption=" in str(segment.get("text", ""))
    assert "rocky coastline" in str(segment.get("vision_caption", "")).lower()

    assert len(calls) == 1
    assert calls[0]["url"] == "http://vllm.local:8001/v1/chat/completions"
    assert calls[0]["headers"].get("authorization") == "Bearer test-key"
    assert calls[0]["timeout"] == 7.0
    content_rows = calls[0]["payload"]["messages"][0]["content"]
    assert content_rows[1]["type"] == "image_url"
    assert str(content_rows[1]["image_url"]["url"]).startswith("data:image/png;base64,")


def test_eta_mu_image_derive_segment_reports_vllm_unconfigured_when_enabled(
    monkeypatch: Any,
) -> None:
    from code.world_web import ai as ai_module

    monkeypatch.setenv("ETA_MU_IMAGE_VISION_ENABLED", "1")
    monkeypatch.delenv("ETA_MU_IMAGE_VISION_BASE_URL", raising=False)

    segment = ai_module._eta_mu_image_derive_segment(
        source_hash="abc123",
        source_bytes=8,
        source_rel_path=".ημ/demo.png",
        mime="image/png",
        image_bytes=b"\x89PNG\r\n\x1a\n",
    )

    assert segment.get("vision_backend") == "vllm"
    assert segment.get("vision_error") == "vllm_base_url_unset"
    assert "vision-caption=" not in str(segment.get("text", ""))


def test_eta_mu_embed_vector_for_image_uses_text_embedding_backend(
    monkeypatch: Any,
) -> None:
    from code.world_web import ai as ai_module

    calls: list[dict[str, Any]] = []

    def fake_embed_text(
        text: str,
        model: str | None = None,
        **_kwargs: Any,
    ) -> list[float] | None:
        calls.append({"text": text, "model": model})
        return [1.0, 0.0, 0.0]

    def fail_ollama(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("_ollama_embed should not be called for image modality")

    monkeypatch.setattr(ai_module, "_embed_text", fake_embed_text)
    monkeypatch.setattr(ai_module, "_ollama_embed", fail_ollama)

    vector, model_name, fallback_used = ai_module._eta_mu_embed_vector_for_segment(
        modality="image",
        segment_text="vision-caption=lighthouse over rough sea",
        space={
            "model": {"name": "nomic-embed-text"},
            "dims": 3,
        },
        embed_id="img:demo",
    )

    assert calls
    assert calls[0]["model"] == "nomic-embed-text"
    assert "vision-caption" in str(calls[0]["text"])
    assert vector == [1.0, 0.0, 0.0]
    assert model_name == "nomic-embed-text"
    assert fallback_used is False


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


def test_collect_catalog_runtime_fast_mode_skips_pi_archive_build() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        catalog = collect_catalog(
            part,
            vault,
            sync_inbox=False,
            include_pi_archive=False,
        )

        inbox = catalog.get("eta_mu_inbox", {})
        assert inbox.get("record") == "ημ.inbox.v1"
        assert str(inbox.get("sync_status", "")) in {"", "skipped", "failed"}

        pi_archive = catalog.get("pi_archive", {})
        assert pi_archive.get("record") == "ημ.pi-archive.v1"
        assert pi_archive.get("status") == "deferred"
        assert pi_archive.get("ledger_count") == 0


def test_collect_catalog_runtime_fast_mode_defers_world_log(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    called = {"world_log": False}

    def _build_world_log_payload(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        called["world_log"] = True
        return {"ok": True, "record": "ημ.world-log.v1", "events": []}

    monkeypatch.setattr(
        chamber_module,
        "build_world_log_payload",
        _build_world_log_payload,
    )

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        catalog = collect_catalog(
            part,
            vault,
            sync_inbox=False,
            include_pi_archive=False,
            include_world_log=False,
        )

        assert called["world_log"] is False
        world_log = catalog.get("world_log", {})
        assert world_log.get("ok") is False
        assert world_log.get("record") == "ημ.world-log.v1"
        assert world_log.get("error") == "world_log_deferred:runtime_fast_path"


def test_collect_catalog_retains_items_when_world_log_collection_fails(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    def _raise_world_log(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise PermissionError("read-only world log stream")

    monkeypatch.setattr(
        chamber_module,
        "build_world_log_payload",
        _raise_world_log,
    )

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        catalog = collect_catalog(
            part,
            vault,
            sync_inbox=False,
            include_pi_archive=False,
        )

        items = catalog.get("items", [])
        assert len(items) == 2

        world_log = catalog.get("world_log", {})
        assert world_log.get("ok") is False
        assert world_log.get("record") == "ημ.world-log.v1"
        assert world_log.get("events") == []
        assert world_log.get("error") == "world_log_unavailable:PermissionError"


def test_world_log_payload_emits_stream_error_event_when_collector_fails(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        def _raise_emsc(_vault: Path) -> None:
            raise PermissionError("read-only stream log")

        monkeypatch.setattr(chamber_module, "_collect_emsc_stream_rows", _raise_emsc)
        monkeypatch.setattr(
            chamber_module, "_collect_nws_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_swpc_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_eonet_event_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module,
            "_collect_wikimedia_stream_rows",
            lambda _vault: None,
        )

        payload = build_world_log_payload(part, vault, limit=60)
        assert payload.get("ok") is True
        events = payload.get("events", [])
        emsc_errors = [
            row
            for row in events
            if isinstance(row, dict)
            and str(row.get("source", "")) == "emsc_stream"
            and str(row.get("kind", "")) == "emsc.stream.error"
        ]
        assert emsc_errors
        assert "PermissionError" in str(emsc_errors[0].get("detail", ""))


def test_world_log_payload_degrades_event_when_embedding_persist_fails(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        runtime_dir = vault / ".opencode" / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "gibs_layers.v1.jsonl").write_text(
            json.dumps(
                {
                    "record": "eta-mu.gibs-layer.v1",
                    "id": "gibs:fixture-embedding-failure",
                    "ts": "2030-01-01T00:00:00Z",
                    "source": "nasa_gibs",
                    "kind": "gibs.layer.fixture",
                    "status": "recorded",
                    "title": "Fixture Layer",
                    "detail": "fixture layer row",
                    "refs": ["https://gibs.earthdata.nasa.gov/fixture.jpg"],
                    "tags": ["gibs", "nasa"],
                    "meta": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with chamber_module._WORLD_LOG_EMBEDDING_IDS_LOCK:
            chamber_module._WORLD_LOG_EMBEDDING_IDS.clear()

        def _raise_upsert(*_args: Any, **_kwargs: Any) -> None:
            raise PermissionError("read-only embeddings db")

        monkeypatch.setattr(
            chamber_module,
            "_embedding_db_upsert_append_only",
            _raise_upsert,
        )
        monkeypatch.setattr(
            chamber_module, "_collect_emsc_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_nws_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_swpc_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_eonet_event_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module,
            "_collect_wikimedia_stream_rows",
            lambda _vault: None,
        )

        payload = build_world_log_payload(part, vault, limit=30)
        assert payload.get("ok") is True
        events = payload.get("events", [])
        degraded = [
            row
            for row in events
            if isinstance(row, dict)
            and "embedding_error=PermissionError" in str(row.get("detail", ""))
            and str(row.get("status", "")) == "degraded"
        ]
        assert degraded


def test_collect_catalog_accepts_part_root_outside_vault() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "mounted_part"
        vault = root / "vault"
        part.mkdir(parents=True)
        vault.mkdir(parents=True)
        _create_fixture_tree(part)

        catalog = collect_catalog(
            part,
            vault,
            sync_inbox=False,
            include_pi_archive=False,
        )

        items = catalog.get("items", [])
        assert len(items) == 2
        rel_paths = {str(row.get("rel_path", "")) for row in items}
        assert "artifacts/audio/test.wav" in rel_paths
        assert "world_state/constraints.md" in rel_paths


def test_runtime_library_path_resolves_part_root_relative_files() -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "mounted_part"
        vault = root / "vault"
        part.mkdir(parents=True)
        vault.mkdir(parents=True)
        _create_fixture_tree(part)

        resolved = server_module._resolve_runtime_library_path(
            vault,
            part,
            "/library/artifacts/audio/test.wav",
        )
        assert resolved is not None
        assert resolved.name == "test.wav"


def test_runtime_catalog_refresh_is_not_blocked_by_inbox_sync(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "mounted_part"
        vault = root / "vault"
        part.mkdir(parents=True)
        vault.mkdir(parents=True)
        _create_fixture_tree(part)

        with server_module._RUNTIME_CATALOG_CACHE_LOCK:
            server_module._RUNTIME_CATALOG_CACHE["catalog"] = None
            server_module._RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = 0.0
            server_module._RUNTIME_CATALOG_CACHE["last_error"] = ""
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = 0.0
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = None
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""

        def _slow_sync(_vault_root: Path) -> dict[str, Any]:
            time.sleep(0.9)
            return {
                "record": "ημ.inbox.v1",
                "pending_count": 0,
                "processed_count": 0,
                "is_empty": True,
            }

        def _fast_collect(
            part_root: Path,
            vault_root: Path,
            *,
            sync_inbox: bool,
            include_pi_archive: bool,
            include_world_log: bool,
        ) -> dict[str, Any]:
            assert sync_inbox is False
            assert include_pi_archive is False
            assert include_world_log is False
            return {
                "generated_at": "2026-02-18T00:00:00+00:00",
                "part_roots": [str(part_root), str(vault_root)],
                "counts": {"audio": 1},
                "items": [
                    {
                        "part": "64",
                        "name": "test.wav",
                        "role": "audio/canonical",
                        "kind": "audio",
                        "rel_path": "artifacts/audio/test.wav",
                        "url": "/library/artifacts/audio/test.wav",
                    }
                ],
                "eta_mu_inbox": {
                    "record": "ημ.inbox.v1",
                    "pending_count": 0,
                    "processed_count": 0,
                    "is_empty": True,
                },
            }

        monkeypatch.setattr(server_module, "sync_eta_mu_inbox", _slow_sync)
        monkeypatch.setattr(server_module, "collect_catalog", _fast_collect)
        monkeypatch.setattr(server_module, "_RUNTIME_ETA_MU_SYNC_SECONDS", 0.0)

        handler_cls = server_module.make_handler(part, vault, "127.0.0.1", 8787)
        handler = handler_cls.__new__(handler_cls)

        started = time.monotonic()
        handler._schedule_runtime_catalog_refresh()

        cached_catalog: dict[str, Any] | None = None
        deadline = started + 1.5
        while time.monotonic() < deadline:
            with server_module._RUNTIME_CATALOG_CACHE_LOCK:
                snapshot = server_module._RUNTIME_CATALOG_CACHE.get("catalog")
                if isinstance(snapshot, dict):
                    cached_catalog = dict(snapshot)
                    break
            time.sleep(0.02)

        elapsed = time.monotonic() - started
        assert isinstance(cached_catalog, dict)
        assert elapsed < 0.5
        assert len(cached_catalog.get("items", [])) == 1


def test_runtime_catalog_refresh_collects_before_scheduling_sync(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "mounted_part"
        vault = root / "vault"
        part.mkdir(parents=True)
        vault.mkdir(parents=True)
        _create_fixture_tree(part)

        with server_module._RUNTIME_CATALOG_CACHE_LOCK:
            server_module._RUNTIME_CATALOG_CACHE["catalog"] = None
            server_module._RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = 0.0
            server_module._RUNTIME_CATALOG_CACHE["last_error"] = ""
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = 0.0
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = None
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""

        handler_cls = server_module.make_handler(part, vault, "127.0.0.1", 8787)
        handler = handler_cls.__new__(handler_cls)
        order: list[str] = []

        def _collect_catalog_fast() -> dict[str, Any]:
            order.append("collect")
            return {
                "generated_at": "2026-02-18T00:00:00+00:00",
                "part_roots": [str(part)],
                "counts": {},
                "items": [],
                "eta_mu_inbox": {},
            }

        def _schedule_runtime_inbox_sync() -> None:
            order.append("sync")

        monkeypatch.setattr(server_module, "_RUNTIME_ETA_MU_SYNC_SECONDS", 0.0)
        monkeypatch.setattr(handler, "_collect_catalog_fast", _collect_catalog_fast)
        monkeypatch.setattr(
            handler,
            "_schedule_runtime_inbox_sync",
            _schedule_runtime_inbox_sync,
        )

        handler._schedule_runtime_catalog_refresh()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if len(order) >= 2:
                break
            time.sleep(0.01)

        assert len(order) >= 2
        assert order[0] == "collect"
        assert order[1] == "sync"


def test_runtime_catalog_base_warms_cache_inline_when_empty(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        part = root / "mounted_part"
        vault = root / "vault"
        part.mkdir(parents=True)
        vault.mkdir(parents=True)
        _create_fixture_tree(part)

        with server_module._RUNTIME_CATALOG_CACHE_LOCK:
            server_module._RUNTIME_CATALOG_CACHE["catalog"] = None
            server_module._RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = 0.0
            server_module._RUNTIME_CATALOG_CACHE["last_error"] = ""
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = 0.0
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = None
            server_module._RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""

        handler_cls = server_module.make_handler(part, vault, "127.0.0.1", 8787)
        handler = handler_cls.__new__(handler_cls)

        monkeypatch.setattr(
            handler,
            "_collect_catalog_fast",
            lambda: {
                "generated_at": "2026-02-18T00:00:00+00:00",
                "part_roots": [str(part)],
                "counts": {"audio": 1},
                "items": [{"name": "test.wav", "rel_path": "artifacts/audio/test.wav"}],
                "eta_mu_inbox": {"record": "ημ.inbox.v1"},
            },
        )
        monkeypatch.setattr(
            server_module,
            "_collect_runtime_catalog_isolated",
            lambda *_args, **_kwargs: (None, "catalog_subprocess_disabled"),
        )
        monkeypatch.setattr(handler, "_schedule_runtime_inbox_sync", lambda: None)

        catalog = handler._runtime_catalog_base()
        assert len(catalog.get("items", [])) == 1
        with server_module._RUNTIME_CATALOG_CACHE_LOCK:
            cached = server_module._RUNTIME_CATALOG_CACHE.get("catalog")
        assert isinstance(cached, dict)
        assert len(cached.get("items", [])) == 1


def test_runtime_catalog_fallback_bootstraps_particles_from_manifest() -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        fallback_catalog = server_module._runtime_catalog_fallback(part, vault)
        assert fallback_catalog.get("runtime_state") == "fallback"
        assert len(fallback_catalog.get("items", [])) >= 2
        counts = fallback_catalog.get("counts", {})
        assert int(counts.get("audio", 0)) >= 1

        simulation = build_simulation_state(fallback_catalog)
        assert int(simulation.get("total", 0)) > 0
        assert len(simulation.get("points", [])) == simulation.get("total", 0)


def test_world_log_payload_tracks_pending_eta_mu_and_embeddings() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        (vault / "receipts.log").write_text(
            (
                "ts=2026-02-15T00:00:00Z | kind=:decision | origin=unit-test | owner=Err | "
                "dod=world-log-check | pi=part64 | host=127.0.0.1:8787 | manifest=manifest.lith | "
                "refs=.opencode/promptdb/00_wire_world.intent.lisp"
            ),
            encoding="utf-8",
        )
        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "pending_note.md").write_text(
            "pending note waiting for ingest",
            encoding="utf-8",
        )

        payload = build_world_log_payload(part, vault, limit=80)
        assert payload.get("ok") is True
        assert payload.get("record") == "ημ.world-log.v1"
        assert int(payload.get("count", 0)) >= 1
        assert int(payload.get("pending_inbox", 0)) >= 1

        events = payload.get("events", [])
        pending_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("kind", "")) == "eta_mu.pending"
        ]
        assert pending_events
        sample = pending_events[0]
        assert str(sample.get("embedding_id", "")).startswith("world-event:")
        assert isinstance(sample.get("x"), float)
        assert isinstance(sample.get("y"), float)
        assert isinstance(sample.get("relations", []), list)

        listing = world_web_module._embedding_db_list(vault, limit=500)
        ids = {
            str(row.get("id", ""))
            for row in listing.get("entries", [])
            if isinstance(row, dict)
        }
        assert str(sample.get("embedding_id", "")) in ids


def test_wikimedia_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._WIKIMEDIA_STREAM_LOCK:
            chamber_module._WIKIMEDIA_STREAM_CACHE.clear()
            chamber_module._WIKIMEDIA_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "WIKIMEDIA_EVENTSTREAMS_ENABLED", True)
        monkeypatch.setattr(
            chamber_module, "WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS", 0.0
        )
        monkeypatch.setattr(
            chamber_module, "WIKIMEDIA_EVENTSTREAMS_RATE_LIMIT_PER_POLL", 1
        )
        monkeypatch.setattr(
            chamber_module, "WIKIMEDIA_EVENTSTREAMS_STREAMS", "recentchange,page-create"
        )

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [
                    {
                        "meta": {
                            "id": "evt-1",
                            "stream": "recentchange",
                            "dt": "2026-02-18T00:00:00Z",
                        },
                        "type": "edit",
                        "title": "Alpha",
                        "comment": "first",
                        "wiki": "enwiki",
                        "server_url": "https://en.wikipedia.org",
                    },
                    {
                        "meta": {
                            "id": "evt-1",
                            "stream": "recentchange",
                            "dt": "2026-02-18T00:00:00Z",
                        },
                        "type": "edit",
                        "title": "Alpha",
                        "comment": "duplicate",
                        "wiki": "enwiki",
                        "server_url": "https://en.wikipedia.org",
                    },
                    {
                        "meta": {
                            "id": "evt-2",
                            "stream": "page-create",
                            "dt": "2026-02-18T00:00:01Z",
                        },
                        "type": "create",
                        "title": "Beta",
                        "comment": "second",
                        "wiki": "enwiki",
                        "server_url": "https://en.wikipedia.org",
                    },
                ],
                "parse_errors": 2,
                "bytes_read": 321,
            }

        monkeypatch.setattr(chamber_module, "_wikimedia_fetch_sse_events", _fake_fetch)

        chamber_module._collect_wikimedia_stream_rows(vault)
        log_path, rows = chamber_module._load_wikimedia_stream_rows(vault, limit=200)
        assert log_path.endswith("wikimedia_stream.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "wikimedia.stream.connected" in kinds
        assert "wikimedia.stream.poll" in kinds
        assert "wikimedia.stream.parse-error" in kinds
        assert "wikimedia.stream.dedupe" in kinds
        assert "wikimedia.stream.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.WIKIMEDIA_EVENT_RECORD
        ]
        assert len(accepted) == 1


def test_wikimedia_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._WIKIMEDIA_STREAM_LOCK:
            chamber_module._WIKIMEDIA_STREAM_CACHE.clear()
            chamber_module._WIKIMEDIA_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "WIKIMEDIA_EVENTSTREAMS_ENABLED", False)
        chamber_module._collect_wikimedia_stream_rows(vault)

        monkeypatch.setattr(chamber_module, "WIKIMEDIA_EVENTSTREAMS_ENABLED", True)
        monkeypatch.setattr(
            chamber_module, "WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS", 0.0
        )

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_wikimedia_fetch_sse_events", _empty_fetch)
        chamber_module._collect_wikimedia_stream_rows(vault)

        _, rows = chamber_module._load_wikimedia_stream_rows(vault, limit=200)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "wikimedia.stream.paused" in kinds
        assert "wikimedia.stream.resumed" in kinds


def test_nws_alert_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._NWS_ALERTS_LOCK:
            chamber_module._NWS_ALERTS_CACHE.clear()
            chamber_module._NWS_ALERTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "NWS_ALERTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "NWS_ALERTS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "NWS_ALERTS_RATE_LIMIT_PER_POLL", 1)

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "alerts": [
                    {
                        "id": "https://api.weather.gov/alerts/ALPHA",
                        "properties": {
                            "event": "Severe Thunderstorm Warning",
                            "headline": "Severe thunderstorm warning",
                            "areaDesc": "King County",
                            "severity": "Severe",
                            "urgency": "Immediate",
                            "certainty": "Observed",
                            "status": "Actual",
                            "sent": "2026-02-18T00:00:00Z",
                        },
                    },
                    {
                        "id": "https://api.weather.gov/alerts/ALPHA",
                        "properties": {
                            "event": "Severe Thunderstorm Warning",
                            "headline": "duplicate",
                            "areaDesc": "King County",
                            "status": "Actual",
                            "sent": "2026-02-18T00:00:00Z",
                        },
                    },
                    {
                        "id": "https://api.weather.gov/alerts/BETA",
                        "properties": {
                            "event": "Flood Watch",
                            "headline": "Flood watch",
                            "areaDesc": "Pierce County",
                            "status": "Actual",
                            "sent": "2026-02-18T00:01:00Z",
                        },
                    },
                ],
                "parse_errors": 1,
                "bytes_read": 432,
            }

        monkeypatch.setattr(chamber_module, "_nws_fetch_active_alerts", _fake_fetch)

        chamber_module._collect_nws_alert_rows(vault)
        log_path, rows = chamber_module._load_nws_alert_rows(vault, limit=220)
        assert log_path.endswith("nws_alerts.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "nws.alerts.connected" in kinds
        assert "nws.alerts.poll" in kinds
        assert "nws.alerts.parse-error" in kinds
        assert "nws.alerts.dedupe" in kinds
        assert "nws.alerts.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.NWS_ALERT_RECORD
        ]
        assert len(accepted) == 1


def test_nws_alert_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._NWS_ALERTS_LOCK:
            chamber_module._NWS_ALERTS_CACHE.clear()
            chamber_module._NWS_ALERTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "NWS_ALERTS_ENABLED", False)
        chamber_module._collect_nws_alert_rows(vault)

        monkeypatch.setattr(chamber_module, "NWS_ALERTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "NWS_ALERTS_POLL_INTERVAL_SECONDS", 0.0)

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "alerts": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_nws_fetch_active_alerts", _empty_fetch)
        chamber_module._collect_nws_alert_rows(vault)

        _, rows = chamber_module._load_nws_alert_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "nws.alerts.paused" in kinds
        assert "nws.alerts.resumed" in kinds


def test_swpc_alert_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._SWPC_ALERTS_LOCK:
            chamber_module._SWPC_ALERTS_CACHE.clear()
            chamber_module._SWPC_ALERTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_RATE_LIMIT_PER_POLL", 1)

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "alerts": [
                    {
                        "message_type": "ALERT",
                        "message_code": "ALTK08",
                        "serial_number": "101",
                        "issue_datetime": "2026-02-18T00:00:00Z",
                        "message": "Geomagnetic conditions expected.",
                    },
                    {
                        "message_type": "ALERT",
                        "message_code": "ALTK08",
                        "serial_number": "101",
                        "issue_datetime": "2026-02-18T00:00:00Z",
                        "message": "duplicate",
                    },
                    {
                        "message_type": "WATCH",
                        "message_code": "WATA20",
                        "serial_number": "102",
                        "issue_datetime": "2026-02-18T00:01:00Z",
                        "message": "Solar flare watch.",
                    },
                ],
                "parse_errors": 1,
                "bytes_read": 384,
            }

        monkeypatch.setattr(chamber_module, "_swpc_fetch_alert_rows", _fake_fetch)

        chamber_module._collect_swpc_alert_rows(vault)
        log_path, rows = chamber_module._load_swpc_alert_rows(vault, limit=220)
        assert log_path.endswith("swpc_alerts.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "swpc.alerts.connected" in kinds
        assert "swpc.alerts.poll" in kinds
        assert "swpc.alerts.parse-error" in kinds
        assert "swpc.alerts.dedupe" in kinds
        assert "swpc.alerts.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.SWPC_ALERT_RECORD
        ]
        assert len(accepted) == 1


def test_swpc_alert_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._SWPC_ALERTS_LOCK:
            chamber_module._SWPC_ALERTS_CACHE.clear()
            chamber_module._SWPC_ALERTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_ENABLED", False)
        chamber_module._collect_swpc_alert_rows(vault)

        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_POLL_INTERVAL_SECONDS", 0.0)

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "alerts": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_swpc_fetch_alert_rows", _empty_fetch)
        chamber_module._collect_swpc_alert_rows(vault)

        _, rows = chamber_module._load_swpc_alert_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "swpc.alerts.paused" in kinds
        assert "swpc.alerts.resumed" in kinds


def test_eonet_event_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._EONET_EVENTS_LOCK:
            chamber_module._EONET_EVENTS_CACHE.clear()
            chamber_module._EONET_EVENTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "EONET_EVENTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "EONET_EVENTS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "EONET_EVENTS_RATE_LIMIT_PER_POLL", 1)

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [
                    {
                        "id": "EONET-1",
                        "title": "Volcano unrest",
                        "link": "https://eonet.gsfc.nasa.gov/api/v3/events/EONET-1",
                        "categories": [{"id": "volcanoes", "title": "Volcanoes"}],
                        "geometry": [
                            {
                                "date": "2026-02-18T00:00:00Z",
                                "type": "Point",
                                "coordinates": [14.0, 40.8],
                            }
                        ],
                        "sources": [{"id": "USGS", "url": "https://example.test/usgs"}],
                    },
                    {
                        "id": "EONET-1",
                        "title": "Volcano unrest duplicate",
                        "categories": [{"id": "volcanoes", "title": "Volcanoes"}],
                        "geometry": [
                            {
                                "date": "2026-02-18T00:00:00Z",
                                "type": "Point",
                                "coordinates": [14.0, 40.8],
                            }
                        ],
                    },
                    {
                        "id": "EONET-2",
                        "title": "Wildfire activity",
                        "link": "https://eonet.gsfc.nasa.gov/api/v3/events/EONET-2",
                        "categories": [{"id": "wildfires", "title": "Wildfires"}],
                        "geometry": [
                            {
                                "date": "2026-02-18T00:05:00Z",
                                "type": "Point",
                                "coordinates": [-120.0, 45.0],
                            }
                        ],
                    },
                ],
                "parse_errors": 2,
                "bytes_read": 512,
            }

        monkeypatch.setattr(chamber_module, "_eonet_fetch_events", _fake_fetch)

        chamber_module._collect_eonet_event_rows(vault)
        log_path, rows = chamber_module._load_eonet_event_rows(vault, limit=220)
        assert log_path.endswith("eonet_events.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "eonet.events.connected" in kinds
        assert "eonet.events.poll" in kinds
        assert "eonet.events.parse-error" in kinds
        assert "eonet.events.dedupe" in kinds
        assert "eonet.events.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.EONET_EVENT_RECORD
        ]
        assert len(accepted) == 1


def test_eonet_event_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._EONET_EVENTS_LOCK:
            chamber_module._EONET_EVENTS_CACHE.clear()
            chamber_module._EONET_EVENTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "EONET_EVENTS_ENABLED", False)
        chamber_module._collect_eonet_event_rows(vault)

        monkeypatch.setattr(chamber_module, "EONET_EVENTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "EONET_EVENTS_POLL_INTERVAL_SECONDS", 0.0)

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_eonet_fetch_events", _empty_fetch)
        chamber_module._collect_eonet_event_rows(vault)

        _, rows = chamber_module._load_eonet_event_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "eonet.events.paused" in kinds
        assert "eonet.events.resumed" in kinds


def test_gibs_capabilities_parser_extracts_target_layer_metadata() -> None:
    from code.world_web import chamber as chamber_module

    capabilities_xml = """
<Capabilities xmlns="http://www.opengis.net/wmts/1.0">
  <Contents>
    <Layer>
      <Title>MODIS Terra True Color</Title>
      <Identifier>MODIS_Terra_CorrectedReflectance_TrueColor</Identifier>
      <Format>image/jpeg</Format>
      <Dimension>
        <Identifier>Time</Identifier>
        <Default>2026-02-18</Default>
        <Value>2026-02-16/2026-02-18/P1D</Value>
      </Dimension>
      <TileMatrixSetLink>
        <TileMatrixSet>250m</TileMatrixSet>
      </TileMatrixSetLink>
    </Layer>
    <Layer>
      <Title>Ignored Layer</Title>
      <Identifier>Other_Layer</Identifier>
      <Format>image/png</Format>
      <TileMatrixSetLink>
        <TileMatrixSet>500m</TileMatrixSet>
      </TileMatrixSetLink>
    </Layer>
  </Contents>
</Capabilities>
"""

    layers, parse_errors = chamber_module._gibs_layers_from_capabilities_xml(
        capabilities_xml,
        target_layers=["MODIS_Terra_CorrectedReflectance_TrueColor"],
        max_layers=4,
    )
    assert parse_errors == 0
    assert len(layers) == 1
    layer = layers[0]
    assert layer.get("layer_id") == "MODIS_Terra_CorrectedReflectance_TrueColor"
    assert layer.get("title") == "MODIS Terra True Color"
    assert layer.get("default_time") == "2026-02-18"
    assert layer.get("latest_time") == "2026-02-18"
    assert layer.get("tile_matrix_set") == "250m"


def test_gibs_layer_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._GIBS_LAYERS_LOCK:
            chamber_module._GIBS_LAYERS_CACHE.clear()
            chamber_module._GIBS_LAYERS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_RATE_LIMIT_PER_POLL", 1)
        monkeypatch.setattr(
            chamber_module,
            "GIBS_LAYERS_CAPABILITIES_ENDPOINT",
            "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities",
        )

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "layers": [
                    {
                        "layer_id": "MODIS_Terra_CorrectedReflectance_TrueColor",
                        "title": "MODIS Terra True Color",
                        "default_time": "2026-02-18",
                        "latest_time": "2026-02-18",
                        "tile_matrix_set": "250m",
                        "formats": ["image/jpeg"],
                    },
                    {
                        "layer_id": "MODIS_Terra_CorrectedReflectance_TrueColor",
                        "title": "duplicate",
                        "default_time": "2026-02-18",
                        "latest_time": "2026-02-18",
                        "tile_matrix_set": "250m",
                        "formats": ["image/jpeg"],
                    },
                    {
                        "layer_id": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
                        "title": "VIIRS SNPP True Color",
                        "default_time": "2026-02-18",
                        "latest_time": "2026-02-18",
                        "tile_matrix_set": "250m",
                        "formats": ["image/png"],
                    },
                ],
                "parse_errors": 1,
                "bytes_read": 654,
            }

        monkeypatch.setattr(
            chamber_module, "_gibs_fetch_capabilities_layers", _fake_fetch
        )

        chamber_module._collect_gibs_layer_rows(vault)
        log_path, rows = chamber_module._load_gibs_layer_rows(vault, limit=220)
        assert log_path.endswith("gibs_layers.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "gibs.layers.connected" in kinds
        assert "gibs.layers.poll" in kinds
        assert "gibs.layers.parse-error" in kinds
        assert "gibs.layers.dedupe" in kinds
        assert "gibs.layers.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.GIBS_LAYER_RECORD
        ]
        assert len(accepted) == 1
        refs = accepted[0].get("refs", [])
        assert isinstance(refs, list)
        assert any("/wmts/epsg4326/best/" in str(value) for value in refs)


def test_gibs_layer_stream_collect_emits_pause_resume_and_compliance_events(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._GIBS_LAYERS_LOCK:
            chamber_module._GIBS_LAYERS_CACHE.clear()
            chamber_module._GIBS_LAYERS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_ENABLED", False)
        chamber_module._collect_gibs_layer_rows(vault)

        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(
            chamber_module,
            "GIBS_LAYERS_CAPABILITIES_ENDPOINT",
            "http://example.com/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities",
        )

        calls = {"count": 0}

        def _should_not_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            calls["count"] += 1
            return {
                "ok": True,
                "error": "",
                "layers": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(
            chamber_module,
            "_gibs_fetch_capabilities_layers",
            _should_not_fetch,
        )

        chamber_module._collect_gibs_layer_rows(vault)

        _, rows = chamber_module._load_gibs_layer_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "gibs.layers.paused" in kinds
        assert "gibs.layers.resumed" in kinds
        assert "gibs.layers.compliance" in kinds
        compliance_rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("kind", "")) == "gibs.layers.compliance"
        ]
        assert compliance_rows
        assert str(compliance_rows[-1].get("status", "")) == "blocked"
        assert calls["count"] == 0


def test_emsc_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._EMSC_STREAM_LOCK:
            chamber_module._EMSC_STREAM_CACHE.clear()
            chamber_module._EMSC_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "EMSC_STREAM_ENABLED", True)
        monkeypatch.setattr(chamber_module, "EMSC_STREAM_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "EMSC_STREAM_RATE_LIMIT_PER_POLL", 1)

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [
                    {
                        "action": "create",
                        "data": {
                            "properties": {
                                "unid": "eq-1",
                                "time": "2026-02-18T00:00:00Z",
                                "mag": 4.8,
                                "flynn_region": "Ionian Sea",
                            },
                            "geometry": {"coordinates": [20.2, 37.8, 10.0]},
                        },
                    },
                    {
                        "action": "create",
                        "data": {
                            "properties": {
                                "unid": "eq-1",
                                "time": "2026-02-18T00:00:00Z",
                                "mag": 4.8,
                                "flynn_region": "Ionian Sea",
                            },
                            "geometry": {"coordinates": [20.2, 37.8, 10.0]},
                        },
                    },
                    {
                        "action": "create",
                        "data": {
                            "properties": {
                                "unid": "eq-2",
                                "time": "2026-02-18T00:00:03Z",
                                "mag": 5.1,
                                "flynn_region": "Aegean Sea",
                            },
                            "geometry": {"coordinates": [25.0, 38.5, 8.0]},
                        },
                    },
                ],
                "parse_errors": 2,
                "bytes_read": 512,
            }

        monkeypatch.setattr(chamber_module, "_emsc_fetch_ws_events", _fake_fetch)

        chamber_module._collect_emsc_stream_rows(vault)
        log_path, rows = chamber_module._load_emsc_stream_rows(vault, limit=220)
        assert log_path.endswith("emsc_stream.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "emsc.stream.connected" in kinds
        assert "emsc.stream.poll" in kinds
        assert "emsc.stream.parse-error" in kinds
        assert "emsc.stream.dedupe" in kinds
        assert "emsc.stream.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.EMSC_EVENT_RECORD
        ]
        assert len(accepted) == 1


def test_emsc_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._EMSC_STREAM_LOCK:
            chamber_module._EMSC_STREAM_CACHE.clear()
            chamber_module._EMSC_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "EMSC_STREAM_ENABLED", False)
        chamber_module._collect_emsc_stream_rows(vault)

        monkeypatch.setattr(chamber_module, "EMSC_STREAM_ENABLED", True)
        monkeypatch.setattr(chamber_module, "EMSC_STREAM_POLL_INTERVAL_SECONDS", 0.0)

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_emsc_fetch_ws_events", _empty_fetch)
        chamber_module._collect_emsc_stream_rows(vault)

        _, rows = chamber_module._load_emsc_stream_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "emsc.stream.paused" in kinds
        assert "emsc.stream.resumed" in kinds


def test_world_log_payload_includes_nws_and_emsc_rows_and_embeddings(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "eta_mu_part_64"
        _create_fixture_tree(part)

        chamber_module._append_nws_alert_rows(
            vault,
            [
                {
                    "record": chamber_module.NWS_ALERT_RECORD,
                    "id": "nws:test-1",
                    "ts": "2026-02-18T00:00:00Z",
                    "source": "nws_alerts",
                    "kind": "nws.alert.flood-watch",
                    "status": "actual",
                    "title": "Flood watch",
                    "detail": "fixture alert",
                    "refs": ["https://api.weather.gov/alerts/TEST"],
                    "tags": ["nws", "flood-watch"],
                    "meta": {},
                }
            ],
        )
        chamber_module._append_emsc_stream_rows(
            vault,
            [
                {
                    "record": chamber_module.EMSC_EVENT_RECORD,
                    "id": "emsc:test-1",
                    "ts": "2026-02-18T00:00:01Z",
                    "source": "emsc_stream",
                    "kind": "emsc.earthquake",
                    "status": "recorded",
                    "title": "M 4.9 earthquake",
                    "detail": "fixture quake",
                    "refs": [
                        "https://www.seismicportal.eu/eventdetails.html?unid=eq-1"
                    ],
                    "tags": ["emsc", "earthquake"],
                    "meta": {},
                }
            ],
        )

        monkeypatch.setattr(
            chamber_module, "_collect_emsc_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_nws_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_wikimedia_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )

        payload = build_world_log_payload(part, vault, limit=80)
        events = payload.get("events", [])
        nws_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "nws_alerts"
        ]
        emsc_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "emsc_stream"
        ]
        assert nws_events
        assert emsc_events
        assert str(nws_events[0].get("embedding_id", "")).startswith("world-event:")
        assert str(emsc_events[0].get("embedding_id", "")).startswith("world-event:")


def test_world_log_payload_includes_swpc_and_eonet_rows_and_embeddings(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "eta_mu_part_64"
        _create_fixture_tree(part)

        chamber_module._append_swpc_alert_rows(
            vault,
            [
                {
                    "record": chamber_module.SWPC_ALERT_RECORD,
                    "id": "swpc:test-1",
                    "ts": "2026-02-18T00:02:00Z",
                    "source": "swpc_alerts",
                    "kind": "swpc.alert.alert",
                    "status": "actual",
                    "title": "ALERT ALTK08 #101",
                    "detail": "fixture swpc alert",
                    "refs": ["https://services.swpc.noaa.gov/products/alerts.json"],
                    "tags": ["swpc", "space-weather"],
                    "meta": {},
                }
            ],
        )
        chamber_module._append_eonet_event_rows(
            vault,
            [
                {
                    "record": chamber_module.EONET_EVENT_RECORD,
                    "id": "eonet:test-1",
                    "ts": "2026-02-18T00:03:00Z",
                    "source": "nasa_eonet",
                    "kind": "eonet.event.wildfires",
                    "status": "open",
                    "title": "Wildfire activity",
                    "detail": "fixture eonet event",
                    "refs": ["https://eonet.gsfc.nasa.gov/api/v3/events/EONET-1"],
                    "tags": ["eonet", "wildfires"],
                    "meta": {},
                }
            ],
        )

        monkeypatch.setattr(
            chamber_module, "_collect_swpc_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_eonet_event_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_wikimedia_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_nws_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_emsc_stream_rows", lambda _vault: None
        )

        payload = build_world_log_payload(part, vault, limit=80)
        events = payload.get("events", [])
        swpc_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "swpc_alerts"
        ]
        eonet_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "nasa_eonet"
        ]
        assert swpc_events
        assert eonet_events
        assert str(swpc_events[0].get("embedding_id", "")).startswith("world-event:")
        assert str(eonet_events[0].get("embedding_id", "")).startswith("world-event:")


def test_world_log_payload_includes_wikimedia_rows_and_embeddings(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "eta_mu_part_64"
        _create_fixture_tree(part)

        with chamber_module._WIKIMEDIA_STREAM_LOCK:
            chamber_module._WIKIMEDIA_STREAM_CACHE.clear()
            chamber_module._WIKIMEDIA_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        chamber_module._append_wikimedia_stream_rows(
            vault,
            [
                {
                    "record": chamber_module.WIKIMEDIA_EVENT_RECORD,
                    "id": "wikimedia:test-1",
                    "ts": "2026-02-18T00:00:00Z",
                    "source": "wikimedia_eventstreams",
                    "kind": "wikimedia.edit",
                    "status": "recorded",
                    "title": "Alpha",
                    "detail": "fixture event",
                    "refs": ["https://en.wikipedia.org/wiki/Alpha"],
                    "tags": ["enwiki", "edit"],
                    "meta": {},
                }
            ],
        )

        monkeypatch.setattr(
            chamber_module, "_collect_wikimedia_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )

        payload = build_world_log_payload(part, vault, limit=60)
        events = payload.get("events", [])
        wiki_events = [
            row
            for row in events
            if isinstance(row, dict)
            and str(row.get("source", "")) == "wikimedia_eventstreams"
        ]
        assert wiki_events
        sample = wiki_events[0]
        assert str(sample.get("embedding_id", "")).startswith("world-event:")


def test_world_log_payload_includes_gibs_rows_and_embeddings(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "eta_mu_part_64"
        _create_fixture_tree(part)

        with chamber_module._GIBS_LAYERS_LOCK:
            chamber_module._GIBS_LAYERS_CACHE.clear()
            chamber_module._GIBS_LAYERS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        chamber_module._append_gibs_layer_rows(
            vault,
            [
                {
                    "record": chamber_module.GIBS_LAYER_RECORD,
                    "id": "gibs:test-1",
                    "ts": "2026-02-18T00:00:00Z",
                    "source": "nasa_gibs",
                    "kind": "gibs.layer.modis-terra-correctedreflectance-truecolor",
                    "status": "recorded",
                    "title": "MODIS Terra True Color",
                    "detail": "fixture layer event",
                    "refs": [
                        "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2026-02-18/250m/2/1/1.jpg"
                    ],
                    "tags": ["gibs", "nasa", "satellite"],
                    "meta": {},
                }
            ],
        )

        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )

        payload = build_world_log_payload(part, vault, limit=60)
        events = payload.get("events", [])
        gibs_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "nasa_gibs"
        ]
        assert gibs_events
        sample = gibs_events[0]
        assert str(sample.get("embedding_id", "")).startswith("world-event:")


def test_eta_mu_vecstore_upsert_batch_mirrors_embeddings_with_chroma(
    monkeypatch: Any,
) -> None:
    from code.world_web import db as db_module

    class FakeCollection:
        def upsert(self, **_kwargs: Any) -> None:
            return

    monkeypatch.setattr(
        db_module,
        "_get_eta_mu_chroma_collection",
        lambda _collection_name: FakeCollection(),
    )

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        result = db_module._eta_mu_vecstore_upsert_batch(
            vault,
            rows=[
                {
                    "id": "emb_test_world_log_1",
                    "embedding": [0.2, 0.1, -0.3, 0.4],
                    "metadata": {"source.loc": "sample/path.md"},
                    "document": "sample vector row",
                    "model": "stub-model",
                }
            ],
            collection_name="eta_mu_test_collection",
            space_set_signature="space.sig.test",
        )

        assert result.get("ok") is True
        assert result.get("backend") == "chroma"
        assert int(result.get("mirrored", 0)) >= 1

        listing = world_web_module._embedding_db_list(vault, limit=20)
        ids = [
            str(row.get("id", ""))
            for row in listing.get("entries", [])
            if isinstance(row, dict)
        ]
        assert "emb_test_world_log_1" in ids


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


def test_simulation_ws_normalize_delta_stream_mode_aliases() -> None:
    from code.world_web import server as server_module

    assert (
        server_module._simulation_ws_normalize_delta_stream_mode("workers") == "workers"
    )
    assert (
        server_module._simulation_ws_normalize_delta_stream_mode("thread") == "workers"
    )
    assert server_module._simulation_ws_normalize_delta_stream_mode("world") == "world"


def test_ws_wire_mode_normalization_aliases() -> None:
    from code.world_web import server as server_module

    assert server_module._normalize_ws_wire_mode("arr") == "arr"
    assert server_module._normalize_ws_wire_mode("packed") == "arr"
    assert server_module._normalize_ws_wire_mode("json") == "json"


def test_ws_pack_message_uses_array_numeric_wire_nodes() -> None:
    from code.world_web import server as server_module

    payload = {
        "type": "simulation_delta",
        "ok": True,
        "nullable": None,
        "delta": {
            "patch": {
                "timestamp": "2026-02-21T18:10:00Z",
                "values": [1, "alpha", 2.5, False, None],
            }
        },
    }

    packed = server_module._ws_pack_message(payload)
    assert isinstance(packed, list)
    assert packed[0] == server_module._WS_WIRE_ARRAY_SCHEMA
    assert isinstance(packed[1], list)
    assert isinstance(packed[2], list)

    def assert_only_arrays_strings_numbers(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                assert_only_arrays_strings_numbers(item)
            return
        assert isinstance(value, (str, int, float))
        assert not isinstance(value, bool)

    assert_only_arrays_strings_numbers(packed)

    key_table = cast(list[str], packed[1])

    def decode_node(node: Any) -> Any:
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            return node
        assert isinstance(node, list)
        assert node
        tag = int(cast(int | float, node[0]))
        if tag == server_module._WS_PACK_TAG_NULL:
            return None
        if tag == server_module._WS_PACK_TAG_BOOL:
            return int(cast(int | float, node[1])) != 0
        if tag == server_module._WS_PACK_TAG_STRING:
            return str(node[1])
        if tag == server_module._WS_PACK_TAG_ARRAY:
            return [decode_node(item) for item in node[1:]]
        if tag == server_module._WS_PACK_TAG_OBJECT:
            out: dict[str, Any] = {}
            index = 1
            while index + 1 < len(node):
                key_slot = int(cast(int | float, node[index]))
                key_name = key_table[key_slot]
                out[key_name] = decode_node(node[index + 1])
                index += 2
            return out
        raise AssertionError(f"unexpected node tag: {tag}")

    decoded = decode_node(packed[2])
    assert decoded == payload


def test_simulation_ws_split_delta_by_worker_splits_presence_dynamics() -> None:
    from code.world_web import server as server_module

    delta = {
        "patch": {
            "timestamp": "2026-02-21T17:55:00Z",
            "total": 2,
            "daimoi": {"active": 1},
            "presence_dynamics": {
                "field_particles": [{"id": "dm-1"}],
                "resource_heartbeat": {"devices": {"cpu": {"utilization": 12.0}}},
                "user_presence": {"id": "user"},
            },
        },
        "changed_keys": [
            "timestamp",
            "total",
            "daimoi",
            "presence_dynamics",
        ],
    }

    rows = server_module._simulation_ws_split_delta_by_worker(delta)
    by_worker = {
        str(row.get("worker_id", "")): row
        for row in rows
        if isinstance(row, dict) and row.get("worker_id")
    }

    assert "sim-core" in by_worker
    assert "sim-daimoi" in by_worker
    assert "sim-particles" in by_worker
    assert "sim-resource" in by_worker
    assert "sim-interaction" in by_worker

    assert by_worker["sim-core"]["patch"].get("total") == 2
    assert by_worker["sim-daimoi"]["patch"].get("daimoi") == {"active": 1}
    assert by_worker["sim-particles"]["patch"].get("presence_dynamics", {}).get(
        "field_particles", []
    ) == [{"id": "dm-1"}]
    assert by_worker["sim-resource"]["patch"].get("presence_dynamics", {}).get(
        "resource_heartbeat", {}
    ) == {"devices": {"cpu": {"utilization": 12.0}}}
    assert by_worker["sim-interaction"]["patch"].get("presence_dynamics", {}).get(
        "user_presence", {}
    ) == {"id": "user"}

    assert by_worker["sim-daimoi"]["patch"].get("timestamp") == "2026-02-21T17:55:00Z"


class _MockWebSocketTransport:
    def __init__(self, payload: bytes) -> None:
        self._buffer = payload
        self.sent: list[bytes] = []

    def recv(self, size: int) -> bytes:
        if not self._buffer:
            return b""
        chunk = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return chunk

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)


def _masked_ws_frame(opcode: int, payload: bytes) -> bytes:
    mask = bytes([0x23, 0x45, 0x67, 0x89])
    masked_payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return bytes([0x80 | (opcode & 0x0F), 0x80 | len(payload)]) + mask + masked_payload


def test_consume_ws_client_frame_replies_to_ping() -> None:
    from code.world_web import server as server_module

    transport = _MockWebSocketTransport(_masked_ws_frame(0x9, b"ping"))
    keep_open = server_module._consume_ws_client_frame(cast(Any, transport))

    assert keep_open is True
    assert transport.sent == [b"\x8a\x04ping"]


def test_consume_ws_client_frame_handles_close() -> None:
    from code.world_web import server as server_module

    payload = struct.pack("!H", 1000)
    transport = _MockWebSocketTransport(_masked_ws_frame(0x8, payload))
    keep_open = server_module._consume_ws_client_frame(cast(Any, transport))

    assert keep_open is False
    assert transport.sent == [b"\x88\x02" + payload]


def test_compact_simulation_payload_keeps_presence_particles_and_drops_heavy_graphs() -> (
    None
):
    from code.world_web import server as server_module

    payload = {
        "timestamp": "2026-02-20T22:20:00Z",
        "field_particles": [{"id": "root-dm-1"}],
        "presence_dynamics": {
            "field_particles": [{"id": "dyn-dm-1", "presence_id": "witness_thread"}],
        },
        "nexus_graph": {"nodes": [1]},
        "logical_graph": {"nodes": [1]},
        "pain_field": {"nodes": [1]},
        "file_graph": {"file_nodes": [1]},
        "crawler_graph": {"crawler_nodes": [1]},
    }

    compact = server_module._simulation_http_compact_simulation_payload(payload)

    assert "nexus_graph" not in compact
    assert "logical_graph" not in compact
    assert "pain_field" not in compact
    assert "file_graph" not in compact
    assert "crawler_graph" not in compact
    assert "field_particles" not in compact
    assert compact.get("presence_dynamics", {}).get("field_particles", []) == [
        {"id": "dyn-dm-1", "presence_id": "witness_thread"}
    ]


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


def test_docmeta_tags_create_first_class_tag_nodes(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    from code.world_web import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "ETA_MU_DOCMETA_CYCLE_SECONDS", 0.0)
    monkeypatch.setattr(catalog_module, "ETA_MU_DOCMETA_MAX_PER_CYCLE", 256)

    monkeypatch.setattr(
        catalog_module,
        "_ollama_generate_text",
        lambda *_args, **_kwargs: ("", ""),
    )

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
        (inbox / "new_anchor_note.md").write_text(
            "Anchor registry updates tie proofs to runtime gates.",
            encoding="utf-8",
        )

        catalog = collect_catalog(part, vault)
        file_graph = catalog.get("file_graph", {})
        assert isinstance(file_graph, dict)

        tag_nodes = file_graph.get("tag_nodes", [])
        assert isinstance(tag_nodes, list)
        assert len(tag_nodes) >= 1

        stats = file_graph.get("stats", {})
        assert isinstance(stats, dict)
        assert stats.get("tag_count", 0) == len(tag_nodes)
        assert stats.get("docmeta_enriched_count", 0) >= 1
        assert stats.get("tag_edge_count", 0) >= 1

        note_node = next(
            node
            for node in file_graph.get("file_nodes", [])
            if node.get("name") == "new_witness_note.md"
        )
        assert str(note_node.get("summary", "")).strip()
        assert len(note_node.get("tags", [])) >= 1
        assert len(note_node.get("labels", [])) >= 1

        note_id = str(note_node.get("id", "")).strip()
        assert note_id
        assert any(
            isinstance(edge, dict)
            and str(edge.get("kind", "")) == "labeled_as"
            and str(edge.get("source", "")) == note_id
            for edge in file_graph.get("edges", [])
        )

        simulation = build_simulation_state(catalog)
        logical_graph = simulation.get("logical_graph", {})
        assert isinstance(logical_graph, dict)
        assert any(
            isinstance(node, dict) and str(node.get("kind", "")) == "tag"
            for node in logical_graph.get("nodes", [])
        )
        assert any(
            isinstance(edge, dict)
            and str(edge.get("kind", "")) in {"labeled_as", "relates_tag"}
            for edge in logical_graph.get("edges", [])
        )


def test_docmeta_index_is_idempotent_and_refreshes_on_source_change(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    from code.world_web import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "ETA_MU_DOCMETA_CYCLE_SECONDS", 0.0)
    monkeypatch.setattr(catalog_module, "ETA_MU_DOCMETA_MAX_PER_CYCLE", 256)
    monkeypatch.setattr(
        catalog_module,
        "_ollama_generate_text",
        lambda *_args, **_kwargs: (
            '{"summary":"auto metadata summary","tags":["witness","gate"]}',
            "stub-llm",
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        note_name = "docmeta_refresh_note.md"
        (inbox / note_name).write_text(
            "Witness thread continuity, gate proof, and fork tax notes.",
            encoding="utf-8",
        )

        collect_catalog(part, vault)
        docmeta_path = vault / ".opencode" / "runtime" / "eta_mu_docmeta.v1.jsonl"
        assert docmeta_path.exists()
        first_rows = [
            json.loads(line)
            for line in docmeta_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        first_count = len(first_rows)
        assert first_count >= 1

        collect_catalog(part, vault)
        second_rows = [
            json.loads(line)
            for line in docmeta_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        second_count = len(second_rows)
        assert second_count == first_count

        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / note_name).write_text(
            "Witness thread continuity with updated anchor registry links.",
            encoding="utf-8",
        )
        collect_catalog(part, vault)

        third_rows = [
            json.loads(line)
            for line in docmeta_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        assert len(third_rows) > second_count
        latest = next(
            row
            for row in sorted(
                third_rows,
                key=lambda item: str(item.get("generated_at", "")),
                reverse=True,
            )
            if str(row.get("source_rel_path", "")).endswith(note_name)
        )
        assert str(latest.get("summary", "")).strip()
        assert latest.get("strategy") in {"llm", "heuristic"}
        assert len(latest.get("tags", [])) >= 1


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


def test_ollama_embed_force_nomic_with_small_context(monkeypatch: Any) -> None:
    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"embedding": [0.7, 0.8, 0.9]}).encode("utf-8")

    seen_payloads: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        _ = timeout
        payload = json.loads((req.data or b"{}").decode("utf-8"))
        seen_payloads.append(payload)
        return _FakeResponse()

    monkeypatch.setattr(world_web_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://stub.local:11434")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:8b")
    monkeypatch.setenv("OLLAMA_EMBED_FORCE_NOMIC", "1")
    monkeypatch.setenv("OLLAMA_EMBED_NUM_CTX", "512")
    monkeypatch.setenv("OLLAMA_EMBED_MAX_CHARS", "6")

    vector = world_web_module._ollama_embed("abcdefghi", model="arg-model")

    assert vector == [0.7, 0.8, 0.9]
    assert seen_payloads[0]["model"] == "nomic-embed-text"
    assert seen_payloads[0]["prompt"] == "abcdef"
    assert seen_payloads[0]["options"] == {"num_ctx": 512}


def test_ollama_embed_includes_gpu_options_when_configured(monkeypatch: Any) -> None:
    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"embedding": [0.3, 0.4, 0.5]}).encode("utf-8")

    seen_payloads: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        _ = timeout
        payload = json.loads((req.data or b"{}").decode("utf-8"))
        seen_payloads.append(payload)
        return _FakeResponse()

    monkeypatch.setattr(world_web_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://stub.local:11434")
    monkeypatch.setenv("OLLAMA_NUM_GPU", "2")
    monkeypatch.setenv("OLLAMA_MAIN_GPU", "1")
    monkeypatch.setenv("OLLAMA_EMBED_NUM_CTX", "256")

    vector = world_web_module._ollama_embed("gpu configured")

    assert vector == [0.3, 0.4, 0.5]
    assert seen_payloads[0].get("options") == {
        "num_gpu": 2,
        "main_gpu": 1,
        "num_ctx": 256,
    }


def test_ollama_generate_remote_includes_gpu_options_when_configured(
    monkeypatch: Any,
) -> None:
    class _FakeResponse:
        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"response": "gpu path active"}).encode("utf-8")

    seen_payloads: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        _ = timeout
        payload = json.loads((req.data or b"{}").decode("utf-8"))
        seen_payloads.append(payload)
        return _FakeResponse()

    monkeypatch.setattr(world_web_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://stub.local:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
    monkeypatch.setenv("OLLAMA_NUM_GPU", "1")
    monkeypatch.setenv("OLLAMA_MAIN_GPU", "0")

    text, model = world_web_module._ollama_generate_text_remote("gpu prompt")

    assert text == "gpu path active"
    assert model == "qwen3-vl:4b-instruct"
    assert seen_payloads[0].get("options") == {"num_gpu": 1, "main_gpu": 0}


def test_resource_auto_embedding_order_prefers_hardware_backends(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("EMBED_ALLOW_CPU_FALLBACK", raising=False)
    order = world_web_module._resource_auto_embedding_order(
        snapshot={
            "devices": {
                "npu0": {"status": "ok"},
                "cpu": {"utilization": 22.0},
                "gpu1": {"utilization": 31.0},
            }
        }
    )

    assert order[0] == "openvino"
    assert "ollama" in order
    assert "tensorflow" not in order


def test_effective_request_embed_model_honors_force_nomic(monkeypatch: Any) -> None:
    from code.world_web import server as server_module

    monkeypatch.setenv("OLLAMA_EMBED_FORCE_NOMIC", "1")
    assert (
        server_module._effective_request_embed_model("qwen3-embedding:8b")
        == "nomic-embed-text"
    )

    monkeypatch.setenv("OLLAMA_EMBED_FORCE_NOMIC", "0")
    assert (
        server_module._effective_request_embed_model("qwen3-embedding:8b")
        == "qwen3-embedding:8b"
    )


def test_openvino_embed_uses_local_endpoint_device_and_normalization(
    monkeypatch: Any,
) -> None:
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
        headers = {str(k).lower(): str(v) for k, v in req.header_items()}
        calls.append(
            {
                "url": str(req.full_url),
                "timeout": timeout,
                "payload": payload,
                "headers": headers,
            }
        )
        return _FakeResponse({"embedding": [3.0, 4.0]})

    monkeypatch.setattr(world_web_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENVINO_EMBED_ENDPOINT", "http://ov.local:18000/v1/embeddings")
    monkeypatch.setenv("OPENVINO_EMBED_DEVICE", "NPU")
    monkeypatch.setenv("OPENVINO_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OPENVINO_EMBED_TIMEOUT_SEC", "7")
    monkeypatch.setenv("OPENVINO_EMBED_MAX_CHARS", "5")
    monkeypatch.setenv("OPENVINO_EMBED_NORMALIZE", "1")
    monkeypatch.setenv("OPENVINO_EMBED_BEARER_TOKEN", "secret-token")

    vector = world_web_module._openvino_embed("abcdefghi")

    assert vector is not None
    assert len(calls) >= 1
    assert calls[0]["url"].endswith("/v1/embeddings")
    assert calls[0]["payload"]["model"] == "nomic-embed-text"
    assert calls[0]["payload"]["input"] == ["abcde"]
    assert calls[0]["payload"]["options"]["device"] == "NPU"
    assert calls[0]["headers"]["authorization"] == "Bearer secret-token"
    assert calls[0]["timeout"] == 7.0
    assert abs(vector[0] - 0.6) < 1e-6
    assert abs(vector[1] - 0.8) < 1e-6


def test_apply_embedding_provider_options_supports_gpu_and_npu_presets(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_EMBED_FORCE_NOMIC", "0")
    monkeypatch.setenv("OPENVINO_EMBED_DEVICE", "CPU")

    gpu_result = world_web_module._apply_embedding_provider_options(
        {
            "preset": "gpu_local",
            "ollama_embed_num_ctx": 384,
        }
    )
    assert gpu_result.get("ok") is True
    assert os.getenv("EMBEDDINGS_BACKEND") == "ollama"
    assert os.getenv("OLLAMA_EMBED_FORCE_NOMIC") == "1"
    assert os.getenv("OLLAMA_EMBED_NUM_CTX") == "384"

    npu_result = world_web_module._apply_embedding_provider_options(
        {
            "preset": "npu_local",
            "openvino_endpoint": "http://ov.local:18000/v1/embeddings",
            "openvino_bearer_token": "ov-token",
        }
    )
    assert npu_result.get("ok") is True
    assert os.getenv("EMBEDDINGS_BACKEND") == "openvino"
    assert os.getenv("OPENVINO_EMBED_DEVICE") == "NPU"
    assert os.getenv("OPENVINO_EMBED_BEARER_TOKEN") == "ov-token"
    options = npu_result.get("options", {}).get("config", {})
    assert options.get("openvino_auth_mode") == "bearer"
    assert options.get("openvino_auth_header_name") == "Authorization"


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


def test_simulation_state_applies_document_similarity_layout_to_points() -> None:
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
                    "id": "file:left",
                    "name": "alpha_notes.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "x": 0.5,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
                {
                    "id": "file:right",
                    "name": "alpha_archive.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "x": 0.54,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
            ],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 2},
                "field_counts": {"f6": 2},
                "knowledge_entries": 2,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    sim_graph = simulation.get("file_graph", {})
    sim_nodes = sim_graph.get("file_nodes", [])
    assert isinstance(sim_nodes, list)
    assert len(sim_nodes) == 2

    left = sim_nodes[0]
    right = sim_nodes[1]
    assert float(left.get("x", 0.0)) > 0.5
    assert float(right.get("x", 1.0)) < 0.54

    points = simulation.get("points", [])
    assert len(points) >= 2
    left_point = points[0]
    right_point = points[1]
    expected_left_x = round((float(left.get("x", 0.5)) * 2.0) - 1.0, 5)
    expected_right_x = round((float(right.get("x", 0.5)) * 2.0) - 1.0, 5)
    assert abs(float(left_point.get("x", 0.0)) - expected_left_x) <= 1e-5
    assert abs(float(right_point.get("x", 0.0)) - expected_right_x) <= 1e-5


def test_simulation_state_document_similarity_layout_is_subtle() -> None:
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
                    "id": "file:left",
                    "name": "alpha_notes.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "x": 0.4,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
                {
                    "id": "file:right",
                    "name": "gamma_image.png",
                    "kind": "image",
                    "summary": "solar flare telemetry",
                    "tags": ["solar", "flare"],
                    "dominant_field": "f1",
                    "x": 0.44,
                    "y": 0.5,
                    "hue": 320,
                    "importance": 1.0,
                },
            ],
            "edges": [],
            "stats": {
                "field_count": 2,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 1, "image": 1},
                "field_counts": {"f1": 1, "f6": 1},
                "knowledge_entries": 2,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    sim_nodes = simulation.get("file_graph", {}).get("file_nodes", [])
    assert isinstance(sim_nodes, list)
    assert len(sim_nodes) == 2

    left_x = float(sim_nodes[0].get("x", 0.4))
    right_x = float(sim_nodes[1].get("x", 0.44))
    assert left_x < 0.4
    assert right_x > 0.44

    assert abs(left_x - 0.4) < 0.03
    assert abs(right_x - 0.44) < 0.03


def test_simulation_state_embedded_nodes_repel_non_embedded_nodes() -> None:
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
                    "id": "file:embedded",
                    "name": "alpha_embed.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "vecstore_collection": "eta_mu_nexus_v1",
                    "embed_layer_count": 1,
                    "x": 0.5,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
                {
                    "id": "file:plain",
                    "name": "alpha_plain.md",
                    "kind": "text",
                    "summary": "alpha witness archive",
                    "tags": ["alpha", "witness"],
                    "dominant_field": "f6",
                    "x": 0.52,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 1.0,
                },
            ],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 2},
                "field_counts": {"f6": 2},
                "knowledge_entries": 2,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    sim_nodes = simulation.get("file_graph", {}).get("file_nodes", [])
    assert isinstance(sim_nodes, list)
    assert len(sim_nodes) == 2

    embedded_x = float(sim_nodes[0].get("x", 0.5))
    plain_x = float(sim_nodes[1].get("x", 0.52))
    assert embedded_x < 0.5
    assert plain_x > 0.52
    assert abs(embedded_x - 0.5) < 0.03
    assert abs(plain_x - 0.52) < 0.03


def test_simulation_state_emits_embedding_particles_for_embedded_files(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module.time, "time", lambda: 1_700_001_234.0)
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
                    "id": "file:embed-left",
                    "name": "left.md",
                    "kind": "text",
                    "summary": "alpha witness archive stream",
                    "tags": ["alpha", "witness", "stream"],
                    "dominant_field": "f6",
                    "vecstore_collection": "eta_mu_nexus_v1",
                    "embed_layer_count": 1,
                    "x": 0.45,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 0.9,
                },
                {
                    "id": "file:embed-right",
                    "name": "right.md",
                    "kind": "text",
                    "summary": "alpha witness archive stream",
                    "tags": ["alpha", "witness", "archive"],
                    "dominant_field": "f6",
                    "vecstore_collection": "eta_mu_nexus_v1",
                    "embed_layer_count": 1,
                    "x": 0.55,
                    "y": 0.5,
                    "hue": 210,
                    "importance": 0.9,
                },
            ],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 2},
                "field_counts": {"f6": 2},
                "knowledge_entries": 2,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    embedding_particles = simulation.get("embedding_particles", [])
    assert isinstance(embedding_particles, list)
    assert len(embedding_particles) >= 6
    assert any(float(row.get("size", 0.0)) > 2.0 for row in embedding_particles)

    for row in embedding_particles[:3]:
        assert -1.0 <= float(row.get("x", 0.0)) <= 1.0
        assert -1.0 <= float(row.get("y", 0.0)) <= 1.0

    graph_particles = simulation.get("file_graph", {}).get("embedding_particles", [])
    assert isinstance(graph_particles, list)
    assert len(graph_particles) == len(embedding_particles)


def test_embedding_particles_bias_toward_denser_nearby_documents(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module.time, "time", lambda: 1_700_009_999.0)

    dense_text = "alpha witness archive stream " * 80
    sparse_text = "alpha witness"

    def _catalog_for_density(left_summary: str, right_summary: str) -> dict[str, Any]:
        return {
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
                        "id": "file:left",
                        "name": "alpha_note.md",
                        "kind": "text",
                        "summary": left_summary,
                        "text_excerpt": left_summary,
                        "tags": ["alpha", "witness", "archive"],
                        "dominant_field": "f6",
                        "vecstore_collection": "eta_mu_nexus_v1",
                        "embed_layer_count": 1,
                        "x": 0.44,
                        "y": 0.5,
                        "hue": 210,
                        "importance": 0.9,
                    },
                    {
                        "id": "file:right",
                        "name": "alpha_note.md",
                        "kind": "text",
                        "summary": right_summary,
                        "text_excerpt": right_summary,
                        "tags": ["alpha", "witness", "archive"],
                        "dominant_field": "f6",
                        "vecstore_collection": "eta_mu_nexus_v1",
                        "embed_layer_count": 1,
                        "x": 0.56,
                        "y": 0.5,
                        "hue": 210,
                        "importance": 0.9,
                    },
                ],
                "edges": [],
                "stats": {
                    "field_count": 1,
                    "file_count": 2,
                    "edge_count": 0,
                    "kind_counts": {"text": 2},
                    "field_counts": {"f6": 2},
                    "knowledge_entries": 2,
                },
            },
        }

    left_dense = build_simulation_state(_catalog_for_density(dense_text, sparse_text))
    right_dense = build_simulation_state(_catalog_for_density(sparse_text, dense_text))

    left_particles = left_dense.get("embedding_particles", [])
    right_particles = right_dense.get("embedding_particles", [])
    assert isinstance(left_particles, list)
    assert isinstance(right_particles, list)
    assert left_particles
    assert right_particles

    left_mean_x = sum(
        (float(row.get("x", 0.0)) + 1.0) * 0.5 for row in left_particles
    ) / len(left_particles)
    right_mean_x = sum(
        (float(row.get("x", 0.0)) + 1.0) * 0.5 for row in right_particles
    ) / len(right_particles)

    assert left_mean_x < right_mean_x
    assert abs(right_mean_x - left_mean_x) > 0.0004


def test_logical_graph_includes_world_log_event_nodes() -> None:
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
                "processed_count": 0,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 1,
                "last_ingested_at": "",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [
                {
                    "id": "file:abc",
                    "node_id": "knowledge:abc",
                    "node_type": "file",
                    "label": "pending_note.md",
                    "source_rel_path": ".ημ/pending_note.md",
                    "x": 0.24,
                    "y": 0.31,
                    "hue": 210,
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 0,
                "file_count": 1,
                "edge_count": 0,
                "kind_counts": {},
                "field_counts": {},
                "knowledge_entries": 1,
            },
        },
        "world_log": {
            "ok": True,
            "record": "ημ.world-log.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "count": 2,
            "limit": 180,
            "pending_inbox": 1,
            "sources": {"eta_mu_inbox": 1, "receipt": 1},
            "kinds": {"eta_mu.pending": 1, ":decision": 1},
            "relation_count": 1,
            "events": [
                {
                    "id": "evt_a",
                    "source": "eta_mu_inbox",
                    "kind": "eta_mu.pending",
                    "status": "pending",
                    "title": "pending inbox file",
                    "detail": "awaiting ingest",
                    "refs": [".ημ/pending_note.md"],
                    "x": 0.4,
                    "y": 0.5,
                    "relations": [{"event_id": "evt_b", "score": 0.8}],
                },
                {
                    "id": "evt_b",
                    "source": "receipt",
                    "kind": ":decision",
                    "status": "recorded",
                    "title": "decision",
                    "detail": "recorded",
                    "refs": ["receipts.log"],
                    "x": 0.6,
                    "y": 0.55,
                    "relations": [{"event_id": "evt_a", "score": 0.8}],
                },
            ],
        },
    }

    simulation = build_simulation_state(catalog)
    logical_graph = simulation.get("logical_graph", {})
    nodes = logical_graph.get("nodes", [])
    edges = logical_graph.get("edges", [])

    assert any(
        isinstance(node, dict) and str(node.get("kind", "")) == "event"
        for node in nodes
    )
    assert any(
        isinstance(edge, dict) and str(edge.get("kind", "")) == "mentions"
        for edge in edges
    )
    assert any(
        isinstance(edge, dict) and str(edge.get("kind", "")) == "correlates"
        for edge in edges
    )


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


def test_simulation_state_unifies_crawler_nodes_into_nexus_graph() -> None:
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-16T00:00:00+00:00",
            "inbox": {},
            "nodes": [
                {
                    "id": "field:gates_of_truth",
                    "node_type": "field",
                    "node_id": "field:gates_of_truth",
                    "x": 0.2,
                    "y": 0.2,
                    "hue": 52,
                    "field": "f2",
                    "label": "gates_of_truth",
                },
                {
                    "id": "file:1",
                    "node_type": "file",
                    "x": 0.4,
                    "y": 0.4,
                    "hue": 200,
                    "importance": 0.5,
                    "source_rel_path": "notes/a.md",
                    "dominant_field": "f2",
                },
            ],
            "field_nodes": [
                {
                    "id": "field:gates_of_truth",
                    "node_type": "field",
                    "node_id": "field:gates_of_truth",
                    "x": 0.2,
                    "y": 0.2,
                    "hue": 52,
                    "field": "f2",
                    "label": "gates_of_truth",
                }
            ],
            "file_nodes": [
                {
                    "id": "file:1",
                    "node_type": "file",
                    "x": 0.4,
                    "y": 0.4,
                    "hue": 200,
                    "importance": 0.5,
                    "source_rel_path": "notes/a.md",
                    "dominant_field": "f2",
                }
            ],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 1,
                "edge_count": 0,
                "kind_counts": {},
                "field_counts": {"f2": 1},
                "knowledge_entries": 0,
            },
        },
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
            "edges": [
                {
                    "id": "edge:field-to-crawler",
                    "source": "crawler-field:gates_of_truth",
                    "target": "crawler:abc",
                    "kind": "hyperlink",
                    "weight": 0.4,
                }
            ],
            "stats": {
                "field_count": 1,
                "crawler_count": 1,
                "edge_count": 1,
                "kind_counts": {"url": 1},
                "field_counts": {"f2": 1},
                "nodes_total": 1,
                "edges_total": 1,
                "url_nodes_total": 1,
            },
        },
    }

    simulation = build_simulation_state(catalog)
    file_graph = simulation.get("file_graph", {})
    graph_nodes = file_graph.get("nodes", []) if isinstance(file_graph, dict) else []
    graph_edges = file_graph.get("edges", []) if isinstance(file_graph, dict) else []
    crawler_rows = (
        file_graph.get("crawler_nodes", []) if isinstance(file_graph, dict) else []
    )

    assert any(str(node.get("id", "")) == "crawler:abc" for node in graph_nodes)
    assert any(str(node.get("id", "")) == "crawler:abc" for node in crawler_rows)
    assert any(
        str(edge.get("source", "")) == "field:gates_of_truth"
        and str(edge.get("target", "")) == "crawler:abc"
        for edge in graph_edges
    )


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


def test_simulation_state_includes_canonical_nexus_graph_and_field_registry() -> None:
    """Test that simulation state includes the unified canonical model types.

    See specs/drafts/part64-deep-research-09-unified-nexus-graph.md
    See specs/drafts/part64-deep-research-10-shared-fields-daimoi-dynamics.md
    """
    source_path = "docs/canonical_test.md"
    file_id = world_web_module._file_id_for_path(source_path)
    catalog = {
        "items": [],
        "counts": {},
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "generated_at": "2026-02-20T00:00:00+00:00",
            "nodes": [],
            "field_nodes": [
                {
                    "id": "field:logos",
                    "node_id": "field:logos",
                    "node_type": "field",
                    "field": "logos",
                    "label": "Logos",
                    "x": 0.5,
                    "y": 0.5,
                    "hue": 200,
                }
            ],
            "tag_nodes": [
                {
                    "id": "tag:test",
                    "node_id": "tag:test",
                    "node_type": "tag",
                    "tag": "test",
                    "label": "Test",
                    "x": 0.4,
                    "y": 0.6,
                    "hue": 180,
                }
            ],
            "file_nodes": [
                {
                    "id": "file:canonical_test",
                    "node_id": "file:canonical_test",
                    "node_type": "file",
                    "name": "canonical_test.md",
                    "label": "Canonical Test",
                    "x": 0.31,
                    "y": 0.42,
                    "hue": 212,
                    "importance": 0.7,
                    "source_rel_path": source_path,
                }
            ],
            "edges": [
                {
                    "id": "edge:test:file:field",
                    "source": "file:canonical_test",
                    "target": "field:logos",
                    "kind": "belongs_to",
                    "weight": 0.8,
                }
            ],
            "stats": {
                "field_count": 1,
                "file_count": 1,
                "edge_count": 1,
            },
        },
        "crawler_graph": {
            "record": "ημ.crawler-graph.v1",
            "generated_at": "2026-02-20T00:00:00+00:00",
            "crawler_nodes": [
                {
                    "id": "crawler:example",
                    "node_type": "crawler",
                    "crawler_kind": "url",
                    "label": "Example",
                    "url": "https://example.com",
                    "x": 0.7,
                    "y": 0.3,
                    "hue": 150,
                }
            ],
            "edges": [],
            "stats": {},
        },
        "truth_state": {
            "record": "ημ.truth-state.v1",
            "generated_at": "2026-02-20T00:00:00+00:00",
            "claim": {
                "id": "claim.canonical",
                "text": "Canonical model is unified",
                "status": "proved",
                "kappa": 0.9,
            },
            "claims": [],
        },
    }

    simulation = build_simulation_state(catalog)

    # Check canonical nexus_graph
    nexus_graph = simulation.get("nexus_graph", {})
    assert nexus_graph.get("record") == "ημ.nexus-graph.v1"
    assert nexus_graph.get("schema_version") == "nexus.graph.v1"

    # Check nodes
    nodes = nexus_graph.get("nodes", [])
    assert isinstance(nodes, list)
    assert len(nodes) >= 3  # At least field, tag, file

    # Check node roles
    roles = {n.get("role") for n in nodes}
    assert "field" in roles
    assert "file" in roles
    assert "tag" in roles

    # Check node structure
    file_node = next((n for n in nodes if n.get("role") == "file"), None)
    assert file_node is not None
    assert file_node.get("id") == "file:canonical_test"
    assert file_node.get("label") == "Canonical Test"
    assert file_node.get("provenance", {}).get("path") == source_path

    # Check edges
    edges = nexus_graph.get("edges", [])
    assert isinstance(edges, list)
    assert len(edges) >= 1

    # Check joins
    joins = nexus_graph.get("joins", {})
    assert isinstance(joins.get("by_role", {}), dict)
    assert isinstance(joins.get("by_path", {}), dict)
    assert source_path in joins.get("by_path", {})

    # Check stats
    stats = nexus_graph.get("stats", {})
    assert stats.get("node_count") >= 3
    assert stats.get("edge_count") >= 1
    assert isinstance(stats.get("role_counts", {}), dict)
    assert stats.get("role_counts", {}).get("file", 0) >= 1

    # Check canonical field_registry
    field_registry = simulation.get("field_registry", {})
    assert field_registry.get("record") == "ημ.field-registry.v1"
    assert field_registry.get("bounded") is True
    assert field_registry.get("field_count") == 4  # demand, flow, entropy, graph

    # Check fields exist
    fields = field_registry.get("fields", {})
    assert "demand" in fields
    assert "flow" in fields
    assert "entropy" in fields
    assert "graph" in fields

    # Check field structure
    for field_name, field in fields.items():
        assert field.get("kind") == field_name
        assert field.get("record") == "ημ.shared-field.v1"
        assert isinstance(field.get("samples", []), list)
        assert isinstance(field.get("stats", {}), dict)

    # Check weights
    weights = field_registry.get("weights", {})
    assert weights.get("demand", 0) > 0
    assert weights.get("flow", 0) >= 0
    assert weights.get("entropy", 0) >= 0
    assert weights.get("graph", 0) > 0

    # Verify backward compatibility: legacy graph payloads still exist
    assert "file_graph" in simulation
    assert "crawler_graph" in simulation
    assert "logical_graph" in simulation
    assert simulation["file_graph"].get("record") == "ημ.file-graph.v1"


def test_canonical_nexus_node_builder_maps_legacy_types() -> None:
    """Test that _build_canonical_nexus_node correctly maps legacy node types."""
    # Test file node
    file_legacy = {
        "id": "test-file",
        "node_type": "file",
        "label": "Test File",
        "x": 0.5,
        "y": 0.3,
        "hue": 200,
        "source_rel_path": "test.md",
    }
    file_canonical = world_web_module._build_canonical_nexus_node(
        file_legacy, origin_graph="test"
    )
    assert file_canonical.get("role") == "file"
    assert file_canonical.get("label") == "Test File"
    assert file_canonical.get("provenance", {}).get("path") == "test.md"

    # Test crawler node
    crawler_legacy = {
        "id": "test-crawler",
        "node_type": "crawler",
        "crawler_kind": "url",
        "label": "Test URL",
        "x": 0.7,
        "y": 0.4,
        "hue": 150,
        "url": "https://example.com",
    }
    crawler_canonical = world_web_module._build_canonical_nexus_node(
        crawler_legacy, origin_graph="crawler_graph"
    )
    assert crawler_canonical.get("role") == "crawler"
    assert crawler_canonical.get("extension", {}).get("url") == "https://example.com"

    # Test field node
    field_legacy = {
        "id": "field:test",
        "node_type": "field",
        "field": "test",
        "label": "Test Field",
        "x": 0.2,
        "y": 0.8,
        "hue": 180,
    }
    field_canonical = world_web_module._build_canonical_nexus_node(
        field_legacy, origin_graph="file_graph"
    )
    assert field_canonical.get("role") == "field"


def test_canonical_field_registry_is_bounded() -> None:
    """Test that field registry has bounded field count (no per-presence fields)."""
    from code.world_web.constants import FIELD_KINDS, MAX_FIELD_COUNT

    # Field count must be bounded
    assert len(FIELD_KINDS) == 4
    assert MAX_FIELD_COUNT == 4
    assert set(FIELD_KINDS) == {"demand", "flow", "entropy", "graph"}

    # Build field registry and verify bounded
    field_registry = world_web_module._build_field_registry({}, None)

    assert field_registry.get("bounded") is True
    assert field_registry.get("field_count") == 4
    assert len(field_registry.get("fields", {})) == 4


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


def test_runtime_influence_tracker_records_compute_jobs() -> None:
    tracker = RuntimeInfluenceTracker()
    tracker.record_compute_job(
        kind="embedding",
        op="embed.ollama",
        backend="ollama",
        resource="gpu",
        emitter_presence_id="health_sentinel_gpu1",
        target_presence_id="file_organizer",
        model="nomic-embed-text",
        status="ok",
        latency_ms=24.7,
    )
    tracker.record_compute_job(
        kind="llm",
        op="text_generate.ollama",
        backend="ollama",
        resource="gpu",
        emitter_presence_id="health_sentinel_gpu1",
        target_presence_id="witness_thread",
        model="qwen3-vl:2b-instruct",
        status="error",
        latency_ms=131.0,
        error="timeout",
    )

    snapshot = tracker.snapshot(queue_snapshot={"pending_count": 0, "event_count": 0})
    assert snapshot.get("compute_jobs_180s") == 2
    summary = snapshot.get("compute_summary", {})
    assert summary.get("llm_jobs") == 1
    assert summary.get("embedding_jobs") == 1
    assert summary.get("error_count") == 1
    assert summary.get("resource_counts", {}).get("gpu") == 2
    jobs = snapshot.get("compute_jobs", [])
    assert isinstance(jobs, list)
    assert len(jobs) >= 2
    assert str(jobs[0].get("id", "")).startswith("compute:")
    assert jobs[0].get("resource") == "gpu"


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
            "compute_jobs_180s": 2,
            "compute_summary": {
                "llm_jobs": 1,
                "embedding_jobs": 1,
                "ok_count": 1,
                "error_count": 1,
                "resource_counts": {"gpu": 2},
            },
            "compute_jobs": [
                {
                    "id": "compute:test-1",
                    "at": "2026-02-18T00:00:00+00:00",
                    "ts": 1_700_000_000.0,
                    "kind": "llm",
                    "op": "text_generate.ollama",
                    "backend": "ollama",
                    "resource": "gpu",
                    "emitter_presence_id": "health_sentinel_gpu1",
                    "target_presence_id": "witness_thread",
                    "model": "qwen3-vl:2b-instruct",
                    "status": "ok",
                    "latency_ms": 84.2,
                    "error": "",
                }
            ],
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
    assert dynamics.get("compute_jobs_180s") == 2
    assert dynamics.get("compute_summary", {}).get("llm_jobs") == 1
    compute_jobs = dynamics.get("compute_jobs", [])
    assert isinstance(compute_jobs, list)
    assert compute_jobs[0].get("id") == "compute:test-1"
    assert compute_jobs[0].get("emitter_presence_id") == "health_sentinel_gpu1"
    simulation_budget = dynamics.get("simulation_budget", {})
    assert int(simulation_budget.get("point_limit", 0)) <= int(
        simulation_budget.get("point_limit_max", 0)
    )
    slice_offload = simulation_budget.get("slice_offload", {})
    assert str(slice_offload.get("source", "")).strip()
    assert "fallback" in slice_offload
    field_particles = dynamics.get("field_particles", [])
    assert isinstance(field_particles, list)
    assert dynamics.get("field_particles_record") == "ημ.field-particles.v1"
    assert simulation.get("field_particles") == field_particles
    resource_daimoi = dynamics.get("resource_daimoi", {})
    assert resource_daimoi.get("record") == "eta-mu.resource-daimoi-flow.v1"
    assert "delivered_packets" in resource_daimoi
    assert "total_transfer" in resource_daimoi
    resource_consumption = dynamics.get("resource_consumption", {})
    assert resource_consumption.get("record") == "eta-mu.resource-daimoi-consumption.v1"
    assert "action_packets" in resource_consumption
    assert "blocked_packets" in resource_consumption
    assert "consumed_total" in resource_consumption
    if field_particles:
        first_particle = field_particles[0]
        assert str(first_particle.get("presence_id", "")).strip()
        assert str(first_particle.get("presence_role", "")).strip()
        assert first_particle.get("particle_mode") in {"neutral", "role-bound"}
        assert 0.0 <= float(first_particle.get("r", 0.0)) <= 0.8
        assert 0.0 <= float(first_particle.get("g", 0.0)) <= 0.8
        assert 0.0 <= float(first_particle.get("b", 0.0)) <= 0.8
        if first_particle.get("resource_consume_amount") is not None:
            assert float(first_particle.get("resource_consume_amount", 0.0)) >= 0.0
        if first_particle.get("resource_action_blocked") is not None:
            assert isinstance(first_particle.get("resource_action_blocked"), bool)


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


def test_backend_field_particles_shift_toward_embedded_similarity(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module.time, "time", lambda: 1_700_100_000.0)

    base_catalog = {
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
            "file_nodes": [],
            "edges": [],
            "stats": {
                "field_count": 1,
                "file_count": 2,
                "edge_count": 0,
                "kind_counts": {"text": 2},
                "field_counts": {"f2": 2},
                "knowledge_entries": 2,
            },
        },
    }

    embed_catalog = json.loads(json.dumps(base_catalog))
    embed_catalog["file_graph"]["file_nodes"] = [
        {
            "id": "file:witness-embed",
            "name": "witness_trace.md",
            "kind": "text",
            "summary": "witness trace continuity lineage",
            "text_excerpt": "witness trace continuity lineage",
            "tags": ["witness", "trace", "lineage"],
            "dominant_field": "f2",
            "field_scores": {"f2": 0.92, "f7": 0.08},
            "vecstore_collection": "eta_mu_nexus_v1",
            "embed_layer_count": 1,
            "x": 0.7,
            "y": 0.3,
            "hue": 250,
            "importance": 0.95,
        },
        {
            "id": "file:witness-support",
            "name": "witness_context.md",
            "kind": "text",
            "summary": "witness field context",
            "tags": ["witness", "field"],
            "dominant_field": "f2",
            "field_scores": {"f2": 0.86},
            "x": 0.64,
            "y": 0.34,
            "hue": 242,
            "importance": 0.78,
        },
    ]

    plain_catalog = json.loads(json.dumps(base_catalog))
    plain_catalog["file_graph"]["file_nodes"] = [
        {
            "id": "file:witness-plain",
            "name": "witness_trace.md",
            "kind": "text",
            "summary": "witness trace continuity lineage",
            "text_excerpt": "witness trace continuity lineage",
            "tags": ["witness", "trace", "lineage"],
            "dominant_field": "f2",
            "field_scores": {"f2": 0.92, "f7": 0.08},
            "x": 0.7,
            "y": 0.3,
            "hue": 250,
            "importance": 0.95,
        },
        {
            "id": "file:witness-support",
            "name": "witness_context.md",
            "kind": "text",
            "summary": "witness field context",
            "tags": ["witness", "field"],
            "dominant_field": "f2",
            "field_scores": {"f2": 0.86},
            "x": 0.64,
            "y": 0.34,
            "hue": 242,
            "importance": 0.78,
        },
    ]

    cache = getattr(world_web_module, "_DAIMO_DYNAMICS_CACHE", {})
    if isinstance(cache, dict):
        cache["field_particles"] = {}
    embed_simulation = build_simulation_state(embed_catalog)

    if isinstance(cache, dict):
        cache["field_particles"] = {}
    plain_simulation = build_simulation_state(plain_catalog)

    def _witness_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        rows = payload.get("presence_dynamics", {}).get("field_particles", [])
        return [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("presence_id", "")).strip() == "witness_thread"
        ]

    witness_embed = _witness_rows(embed_simulation)
    witness_plain = _witness_rows(plain_simulation)

    assert witness_embed
    assert witness_plain

    target_x = 0.67
    target_y = 0.32

    def _mean_distance(rows: list[dict[str, Any]]) -> float:
        total = 0.0
        for row in rows:
            dx = float(row.get("x", 0.5)) - target_x
            dy = float(row.get("y", 0.5)) - target_y
            total += (dx * dx + dy * dy) ** 0.5
        return total / max(1, len(rows))

    assert embed_simulation.get("embedding_particles")
    assert plain_simulation.get("embedding_particles") == []
    assert witness_embed != witness_plain
    assert abs(_mean_distance(witness_embed) - _mean_distance(witness_plain)) > 0.0001


def test_backend_field_particles_scale_with_local_cluster_density(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module.time, "time", lambda: 1_700_200_000.0)

    def _catalog_with_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "items": [],
            "counts": {},
            "file_graph": {
                "record": "ημ.file-graph.v1",
                "generated_at": "2026-02-18T00:00:00+00:00",
                "inbox": {
                    "record": "ημ.inbox.v1",
                    "path": "/tmp/.ημ",
                    "pending_count": 0,
                    "processed_count": len(nodes),
                    "failed_count": 0,
                    "is_empty": True,
                    "knowledge_entries": len(nodes),
                    "last_ingested_at": "2026-02-18T00:00:00+00:00",
                    "errors": [],
                },
                "nodes": [],
                "field_nodes": [],
                "file_nodes": nodes,
                "edges": [],
                "stats": {
                    "field_count": 1,
                    "file_count": len(nodes),
                    "edge_count": 0,
                    "kind_counts": {"text": len(nodes)},
                    "field_counts": {"f2": len(nodes)},
                    "knowledge_entries": len(nodes),
                },
            },
        }

    dense_nodes: list[dict[str, Any]] = []
    for idx in range(5):
        dense_nodes.append(
            {
                "id": f"file:dense:{idx}",
                "name": f"witness_dense_{idx}.md",
                "kind": "text",
                "summary": "witness continuity cluster",
                "tags": ["witness", "trace", "cluster"],
                "dominant_field": "f2",
                "field_scores": {"f2": 0.9, "f7": 0.1},
                "x": 0.62 + (idx * 0.012),
                "y": 0.31 + (idx * 0.008),
                "importance": 0.72,
            }
        )

    sparse_nodes: list[dict[str, Any]] = []
    for idx in range(5):
        sparse_nodes.append(
            {
                "id": f"file:sparse:{idx}",
                "name": f"witness_sparse_{idx}.md",
                "kind": "text",
                "summary": "witness continuity distributed",
                "tags": ["witness", "trace", "cluster"],
                "dominant_field": "f2",
                "field_scores": {"f2": 0.9, "f7": 0.1},
                "x": 0.08 + (idx * 0.18),
                "y": 0.92 - (idx * 0.18),
                "importance": 0.72,
            }
        )

    cache = getattr(world_web_module, "_DAIMO_DYNAMICS_CACHE", {})
    if isinstance(cache, dict):
        cache["field_particles"] = {}
    dense_simulation = build_simulation_state(_catalog_with_nodes(dense_nodes))

    if isinstance(cache, dict):
        cache["field_particles"] = {}
    sparse_simulation = build_simulation_state(_catalog_with_nodes(sparse_nodes))

    def _witness_count(payload: dict[str, Any]) -> int:
        rows = payload.get("presence_dynamics", {}).get("field_particles", [])
        return len(
            [
                row
                for row in rows
                if isinstance(row, dict)
                and str(row.get("presence_id", "")).strip() == "witness_thread"
            ]
        )

    dense_count = _witness_count(dense_simulation)
    sparse_count = _witness_count(sparse_simulation)

    assert dense_count > sparse_count
    assert dense_count >= 6


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
