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
import code.world_web.server as world_web_server

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


def test_simulation_ws_payload_missing_daimoi_summary_guard() -> None:
    assert world_web_server._simulation_ws_payload_missing_daimoi_summary({}) is True
    assert (
        world_web_server._simulation_ws_payload_missing_daimoi_summary(
            {"presence_dynamics": {"daimoi_probabilistic": {}}}
        )
        is False
    )
    assert (
        world_web_server._simulation_ws_payload_missing_daimoi_summary(
            {
                "presence_dynamics": {
                    "daimoi_probabilistic": {
                        "clump_score": 0.42,
                        "anti_clump_drive": -0.08,
                        "anti_clump": {"target": 0.38},
                    }
                }
            }
        )
        is False
    )


def test_simulation_ws_ensure_daimoi_summary_backfills_legacy_payload() -> None:
    payload: dict[str, Any] = {
        "presence_dynamics": {
            "daimoi_probabilistic": {
                "active": 128,
                "collisions": 7,
            }
        }
    }

    world_web_server._simulation_ws_ensure_daimoi_summary(payload)

    summary = payload["presence_dynamics"]["daimoi_probabilistic"]
    anti = summary.get("anti_clump")
    assert isinstance(anti, dict)
    assert isinstance(anti.get("metrics"), dict)
    assert isinstance(anti.get("scales"), dict)
    assert "clump_score" in summary
    assert "anti_clump_drive" in summary
    assert "snr" in summary
    assert anti.get("target") is not None


def test_simulation_ws_ensure_daimoi_summary_tracks_live_clump_and_graph_variability() -> (
    None
):
    with world_web_server._SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_LOCK:
        world_web_server._SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_STATE.clear()
        world_web_server._SIMULATION_WS_DAIMOI_GRAPH_VARIABILITY_STATE.update(
            {
                "positions": {},
                "score": 0.0,
                "raw_score": 0.0,
                "peak_score": 0.0,
                "mean_displacement": 0.0,
                "p90_displacement": 0.0,
                "active_share": 0.0,
                "shared_nodes": 0,
                "sampled_nodes": 0,
            }
        )

    payload: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {"id": "p:1", "x": 0.20, "y": 0.20},
                {"id": "p:2", "x": 0.205, "y": 0.205},
                {"id": "p:3", "x": 0.21, "y": 0.21},
                {"id": "p:4", "x": 0.8, "y": 0.8},
            ],
            "graph_node_positions": {
                "node:a": {"x": 0.2, "y": 0.2},
                "node:b": {"x": 0.8, "y": 0.8},
            },
            "daimoi_probabilistic": {},
        }
    }

    world_web_server._simulation_ws_ensure_daimoi_summary(payload)
    summary_one = payload["presence_dynamics"]["daimoi_probabilistic"]
    anti_one = summary_one.get("anti_clump", {})
    assert float(summary_one.get("clump_score", 0.0)) > 0.0
    assert isinstance(anti_one.get("metrics"), dict)
    assert isinstance(anti_one.get("scales"), dict)
    assert float((anti_one.get("metrics", {}) or {}).get("fano_factor", 0.0)) >= 0.0
    assert anti_one.get("graph_variability", {}).get("shared_nodes") == 0

    payload["presence_dynamics"]["graph_node_positions"] = {
        "node:a": {"x": 0.35, "y": 0.22},
        "node:b": {"x": 0.66, "y": 0.77},
    }
    world_web_server._simulation_ws_ensure_daimoi_summary(payload)
    summary_two = payload["presence_dynamics"]["daimoi_probabilistic"]
    anti_two = summary_two.get("anti_clump", {})
    graph_two = anti_two.get("graph_variability", {})
    scales_two = anti_two.get("scales", {})

    assert float(graph_two.get("score", 0.0)) > 0.0
    assert int(graph_two.get("shared_nodes", 0)) >= 2
    assert float(scales_two.get("noise_gain", 1.0)) >= 1.0
    assert float(scales_two.get("route_damp", 1.0)) <= 1.0


def test_simulation_ws_ensure_daimoi_summary_can_skip_live_refresh_work(
    monkeypatch: Any,
) -> None:
    def fail_live_metrics(*_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("live metrics should not be recomputed")

    def fail_graph_variability(*_: Any, **__: Any) -> dict[str, Any]:
        raise AssertionError("graph variability should not be recomputed")

    monkeypatch.setattr(
        world_web_server,
        "_simulation_ws_daimoi_live_metrics",
        fail_live_metrics,
    )
    monkeypatch.setattr(
        world_web_server,
        "_simulation_ws_graph_variability_update",
        fail_graph_variability,
    )

    payload: dict[str, Any] = {
        "presence_dynamics": {
            "field_particles": [
                {"id": "p:1", "x": 0.2, "y": 0.2},
                {"id": "p:2", "x": 0.8, "y": 0.8},
            ],
            "graph_node_positions": {
                "node:a": {"x": 0.2, "y": 0.2},
                "node:b": {"x": 0.8, "y": 0.8},
            },
            "daimoi_probabilistic": {
                "clump_score": 0.41,
                "anti_clump_drive": -0.09,
                "anti_clump": {
                    "target": 0.38,
                    "metrics": {"nn_term": 0.12},
                    "scales": {"spawn": 0.98, "semantic": 0.92},
                    "graph_variability": {
                        "score": 0.22,
                        "raw_score": 0.24,
                        "peak_score": 0.29,
                        "mean_displacement": 0.011,
                        "p90_displacement": 0.019,
                        "active_share": 0.5,
                        "shared_nodes": 2,
                        "sampled_nodes": 2,
                    },
                },
            },
        }
    }

    world_web_server._simulation_ws_ensure_daimoi_summary(
        payload,
        include_live_metrics=False,
        include_graph_variability=False,
    )

    summary = payload["presence_dynamics"]["daimoi_probabilistic"]
    anti = summary.get("anti_clump", {})
    graph = anti.get("graph_variability", {})

    assert float(summary.get("clump_score", 0.0)) == 0.41
    assert float(summary.get("anti_clump_drive", 0.0)) == -0.09
    assert float(graph.get("score", 0.0)) == 0.22
    assert int(graph.get("shared_nodes", 0)) == 2


def test_simulation_ws_lite_field_particles_respects_max_rows() -> None:
    rows = [
        {"id": "dm:1", "presence_id": "witness_thread", "x": 0.2, "y": 0.3},
        {"id": "dm:2", "presence_id": "anchor_registry", "x": 0.4, "y": 0.5},
        {"id": "dm:3", "presence_id": "gates_of_truth", "x": 0.6, "y": 0.7},
    ]

    compact = world_web_server._simulation_ws_lite_field_particles(rows, max_rows=2)

    assert len(compact) == 2
    assert compact[0].get("id") == "dm:1"
    assert compact[1].get("id") == "dm:2"


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
    assert "normalizeGraphNodePositionMap(" in canvas_source
    assert "normalizePresenceAnchorPositionMap(" in canvas_source
    assert "currentSimulation?.presence_dynamics?.graph_node_positions" in canvas_source
    assert (
        "simulationRef.current?.presence_dynamics?.presence_anchor_positions"
        in canvas_source
    )
    assert (
        "const backendNodePosition = graphNodePositionMap.get(nodeId);" in canvas_source
    )
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


def test_chat_reply_multi_entity_llm_trace_with_stub() -> None:
    def fake_generate(_prompt: str) -> tuple[str | None, str]:
        return "[[PULSE]] I sing in mirrored witness.", "test-model"

    payload = build_chat_reply(
        [{"role": "user", "text": "witness this"}],
        mode="llm",
        multi_entity=True,
        presence_ids=["witness_thread"],
        generate_text_fn=fake_generate,
    )

    assert payload["mode"] == "llm"
    assert payload["model"] == "test-model"
    assert "trace" in payload
    assert payload["trace"]["entities"][0]["mode"] == "llm"
    assert payload["trace"]["entities"][0]["status"] == "ok"
    assert "[[PULSE]]" in payload["trace"]["overlay_tags"]


def test_chat_reply_multi_entity_llm_falls_back_per_presence() -> None:
    def fake_generate(_prompt: str) -> tuple[str | None, str]:
        return None, "test-model"

    payload = build_chat_reply(
        [{"role": "user", "text": "tell me about receipts"}],
        mode="llm",
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


def test_embedding_backend_maps_legacy_values_to_auto(monkeypatch: Any) -> None:
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "tensorflow")
    assert world_web_module._embedding_backend() == "auto"
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "ollama")
    assert world_web_module._embedding_backend() == "auto"


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

    def _fake_tf(
        prompt: str,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> tuple[str | None, str]:
        assert prompt == "blocked gate signal"
        assert model is None
        assert timeout_s is None
        return "tf-route", "tensorflow-hash-v1"

    monkeypatch.setattr(world_web_module, "_tensorflow_generate_text", _fake_tf)

    text, model = world_web_module._ollama_generate_text("blocked gate signal")
    assert text == "tf-route"
    assert model == "tensorflow-hash-v1"


def test_ollama_embed_uses_torch_backend_when_selected(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "torch")
    expected = [0.1, 0.2, 0.3]

    def _fake_torch_embed(
        text: str,
        model: str | None = None,
        *,
        record_job: bool = True,
    ) -> list[float] | None:
        assert text == "drift gate"
        assert model is None
        assert record_job is False
        return expected

    monkeypatch.setattr(
        world_web_module,
        "_torch_embed",
        _fake_torch_embed,
        raising=False,
    )
    result = world_web_module._ollama_embed("drift gate")
    assert result == expected


def test_ollama_embed_records_compute_job_event(monkeypatch: Any) -> None:
    tracker = RuntimeInfluenceTracker()
    monkeypatch.setattr(world_web_module, "_INFLUENCE_TRACKER", tracker)
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "torch")
    monkeypatch.setattr(
        world_web_module,
        "_torch_embed",
        lambda text, model=None, record_job=True: (
            [0.2, 0.4] if text == "job probe" else None
        ),
        raising=False,
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

    monkeypatch.setattr(
        world_web_module,
        "_tensorflow_generate_text",
        lambda prompt, model=None, timeout_s=None: ("tf-route", "tensorflow-hash-v1"),
    )

    text, model = world_web_module._ollama_generate_text("gate remains blocked")
    assert text == "tf-route"
    assert model == "tensorflow-hash-v1"
    snapshot = tracker.snapshot(queue_snapshot={"pending_count": 0, "event_count": 0})
    assert snapshot.get("compute_summary", {}).get("llm_jobs") == 1
    jobs = snapshot.get("compute_jobs", [])
    assert isinstance(jobs, list)
    assert jobs[0].get("kind") == "llm"
    assert jobs[0].get("op") == "text_generate.tensorflow"
    assert jobs[0].get("status") == "ok"


def test_ollama_generate_text_uses_vllm_backend_when_selected(monkeypatch: Any) -> None:
    monkeypatch.setenv("TEXT_GENERATION_BACKEND", "vllm")
    monkeypatch.setenv("TEXT_GENERATION_MODEL", "qwen3-vl:local")

    def _fake_remote(
        prompt: str,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> tuple[str | None, str]:
        assert prompt == "vllm runtime prompt"
        assert model is None
        assert timeout_s is None
        return "vllm-runtime-output", "qwen3-vl:local"

    monkeypatch.setattr(world_web_module, "_ollama_generate_text_remote", _fake_remote)

    text, model = world_web_module._ollama_generate_text("vllm runtime prompt")
    assert text == "vllm-runtime-output"
    assert model == "qwen3-vl:local"


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
        lambda snapshot=None: ["torch", "openvino"],
    )

    def _fake_torch_embed(
        text: str,
        model: str | None = None,
        *,
        record_job: bool = True,
    ) -> list[float] | None:
        assert text == "resource route"
        assert model is None
        assert record_job is False
        calls.append("torch")
        return [0.6, 0.8]

    def _fake_openvino_embed(
        text: str,
        model: str | None = None,
    ) -> list[float] | None:
        calls.append("openvino")
        return None

    monkeypatch.setattr(
        world_web_module,
        "_torch_embed",
        _fake_torch_embed,
        raising=False,
    )
    monkeypatch.setattr(world_web_module, "_openvino_embed", _fake_openvino_embed)

    result = world_web_module._ollama_embed("resource route")
    assert result == [0.6, 0.8]
    assert calls == ["torch"]


def test_ollama_generate_text_auto_uses_resource_selected_order(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TEXT_GENERATION_BACKEND", "auto")
    monkeypatch.setenv("TEXT_GENERATION_MODEL", "qwen3-vl:local")
    calls: list[str] = []

    monkeypatch.setattr(
        world_web_module,
        "_resource_auto_text_order",
        lambda snapshot=None: ["openvino", "tensorflow"],
    )

    def _fake_remote(
        prompt: str, model: str | None = None, timeout_s: float | None = None
    ) -> tuple[str | None, str]:
        assert "route" in prompt
        calls.append("openvino")
        return None, "openvino-route"

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
    assert calls == ["openvino", "tensorflow"]


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
    assert payload["backend"] == "vllm-fallback"
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
    assert segment.get("vision_error") in {"", None}
    assert "rocky coastline" in str(segment.get("vision_caption", "")).lower()
    assert "vision-caption=" in str(segment.get("text", ""))
    assert len(calls) == 1
    assert calls[0]["url"] == "http://vllm.local:8001/v1/chat/completions"
    assert calls[0]["timeout"] == 7.0
    assert calls[0]["payload"].get("model") == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert calls[0]["headers"].get("x-api-key") == "test-key"


def test_eta_mu_image_derive_segment_reports_vllm_unconfigured_when_enabled(
    monkeypatch: Any,
) -> None:
    from code.world_web import ai as ai_module

    monkeypatch.setenv("ETA_MU_IMAGE_VISION_ENABLED", "1")
    monkeypatch.delenv("ETA_MU_IMAGE_VISION_BASE_URL", raising=False)
    monkeypatch.delenv("TEXT_GENERATION_BASE_URL", raising=False)
    monkeypatch.delenv("OPENVINO_CHAT_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENVINO_EMBED_ENDPOINT", raising=False)

    segment = ai_module._eta_mu_image_derive_segment(
        source_hash="abc123",
        source_bytes=8,
        source_rel_path=".ημ/demo.png",
        mime="image/png",
        image_bytes=b"\x89PNG\r\n\x1a\n",
    )

    assert segment.get("vision_backend") == "vllm"
    assert segment.get("vision_error") == "vision_unconfigured"
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

        catalog = collect_catalog(
            part,
            vault,
            include_world_log=False,
            include_pi_archive=False,
        )
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

        catalog = collect_catalog(
            part,
            vault,
            sync_inbox=False,
            include_world_log=False,
        )
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


def test_collect_catalog_runtime_fast_mode_emits_progress_events() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        events: list[str] = []

        def _progress(stage: str, detail: dict[str, Any] | None) -> None:
            del detail
            events.append(str(stage or "").strip())

        catalog = collect_catalog(
            part,
            vault,
            sync_inbox=False,
            include_pi_archive=False,
            include_world_log=False,
            progress_callback=_progress,
        )

        assert isinstance(catalog, dict)
        assert events
        assert "catalog_begin" in events
        assert "file_graph_start" in events
        assert "file_graph_done" in events
        assert "catalog_done" in events


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

        catalog = {
            "counts": {"audio": 1},
            "items": [
                {
                    "name": "test.wav",
                    "kind": "audio",
                }
            ],
            "promptdb": {"packet_count": 0},
            "truth_state": {"gate": {"blocked": False}},
        }
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
