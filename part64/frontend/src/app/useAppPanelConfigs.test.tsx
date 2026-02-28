/* @vitest-environment jsdom */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GLASS_VIEWPORT_PANEL_ID } from "./appShellConstants";
import { useAppPanelConfigs } from "./useAppPanelConfigs";

type UseAppPanelConfigsArgs = Parameters<typeof useAppPanelConfigs>[0];

vi.mock("../components/Simulation/Canvas", () => ({
  OVERLAY_VIEW_OPTIONS: [
    { id: "omni", label: "Omni", description: "all overlays" },
    { id: "presence", label: "Presence", description: "presence lane" },
    { id: "true-graph", label: "True Graph", description: "truth lane" },
  ],
  SimulationCanvas: ({ defaultOverlayView }: { defaultOverlayView: string }) => (
    <div data-testid={`sim-canvas-${defaultOverlayView}`}>{defaultOverlayView}</div>
  ),
}));

vi.mock("../components/Panels/MusePresencePanel", () => ({
  MusePresencePanel: ({ museId }: { museId: string }) => <div data-testid={`muse-presence-${museId}`}>{museId}</div>,
}));

vi.mock("../components/Panels/ProjectionLedgerPanel", () => ({
  ProjectionLedgerPanel: () => <div data-testid="projection-ledger-panel" />,
}));

vi.mock("../components/Panels/Vitals", () => ({
  VitalsPanel: () => <div data-testid="lazy-vitals-panel" />,
}));

vi.mock("../components/Panels/Catalog", () => ({
  CatalogPanel: () => <div data-testid="lazy-catalog-panel" />,
}));

vi.mock("../components/Panels/Omni", () => ({
  OmniPanel: () => <div data-testid="lazy-omni-panel" />,
}));

vi.mock("../components/Panels/MythWorld", () => ({
  MythWorldPanel: () => <div data-testid="lazy-myth-panel" />,
}));

vi.mock("../components/Panels/WebGraphWeaverPanel", () => ({
  WebGraphWeaverPanel: () => <div data-testid="lazy-web-graph-panel" />,
}));

vi.mock("../components/Panels/ThreatRadarPanel", () => ({
  ThreatRadarPanel: () => <div data-testid="lazy-threat-radar-panel" />,
}));

vi.mock("../components/Panels/InspirationAtlasPanel", () => ({
  InspirationAtlasPanel: () => <div data-testid="lazy-inspiration-panel" />,
}));

vi.mock("../components/Panels/StabilityObservatoryPanel", () => ({
  StabilityObservatoryPanel: () => <div data-testid="lazy-stability-observatory-panel" />,
}));

vi.mock("../components/Panels/RuntimeConfigPanel", () => ({
  RuntimeConfigPanel: () => <div data-testid="lazy-runtime-config-panel" />,
}));

vi.mock("../components/Panels/DaimoiPresencePanel", () => ({
  DaimoiPresencePanel: () => <div data-testid="lazy-daimoi-panel" />,
}));

vi.mock("../components/Panels/WorldLogPanel", () => ({
  WorldLogPanel: () => <div data-testid="lazy-world-log-panel" />,
}));

function createArgs(overrides: Partial<UseAppPanelConfigsArgs> = {}): UseAppPanelConfigsArgs {
  return {
    activeMusePresenceId: "witness_thread",
    activeProjection: null,
    autopilotEvents: [],
    catalog: null,
    deferredCoreSimulationTuning: {
      particleDensity: 0.5,
      particleScale: 1,
      motionSpeed: 1,
      mouseInfluence: 1,
      layerDepth: 1,
      graphNodeSmoothness: 1,
      graphNodeStepScale: 1,
    },
    deferredPanelsReady: false,
    flyCameraToAnchor: vi.fn(),
    handleMuseWorkspaceBindingsChange: vi.fn(),
    handleMuseWorkspaceContextChange: vi.fn(),
    handleMuseWorkspaceSend: vi.fn(),
    handleRecord: vi.fn(async () => undefined),
    handleSendVoice: vi.fn(async () => undefined),
    handleTranscribe: vi.fn(async () => "voice text"),
    handleUserPresenceInput: vi.fn(),
    handleWorldInteract: vi.fn(async () => undefined),
    interactingPersonId: null,
    isRecording: false,
    isThinking: false,
    museWorkspaceBindings: {
      witness_thread: ["file:1", "file:2"],
      chaos: [],
      stability: [],
      symmetry: [],
    },
    museWorkspaceContexts: {
      witness_thread: {
        pinnedFileNodeIds: ["file:1", "file:2"],
        searchQuery: "focus",
        pinnedNexusSummaries: [],
      },
      chaos: { pinnedFileNodeIds: [], searchQuery: "", pinnedNexusSummaries: [] },
      stability: { pinnedFileNodeIds: [], searchQuery: "", pinnedNexusSummaries: [] },
      symmetry: { pinnedFileNodeIds: [], searchQuery: "", pinnedNexusSummaries: [] },
    },
    projectionStateByElement: new Map([
      [
        "nexus.ui.chat.witness_thread",
        {
          element_id: "nexus.ui.chat.witness_thread",
          opacity: 0.5,
          pulse: 10,
        },
      ],
    ]) as unknown as UseAppPanelConfigsArgs["projectionStateByElement"],
    setActiveMusePresenceId: vi.fn(),
    simulation: null,
    voiceInputMeta: "voice idle",
    worldInteraction: null,
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
});

describe("useAppPanelConfigs", () => {
  it("builds expected panel ids and renders dedicated + muse panels", () => {
    const { result } = renderHook(() => useAppPanelConfigs(createArgs()));

    const panelIds = result.current.map((entry) => entry.id);
    expect(panelIds).toContain("nexus.ui.dedicated_views");
    expect(panelIds).toContain(GLASS_VIEWPORT_PANEL_ID);
    expect(panelIds).toContain("nexus.ui.chat.witness_thread");
    expect(panelIds).toContain("nexus.ui.runtime_config");
    expect(panelIds).toContain("nexus.ui.threat_radar");

    const dedicated = result.current.find((entry) => entry.id === "nexus.ui.dedicated_views");
    if (!dedicated) {
      throw new Error("dedicated panel missing");
    }
    render(<>{dedicated.render()}</>);
    expect(screen.queryByTestId("sim-canvas-omni")).toBeNull();
    expect(screen.getByTestId("sim-canvas-presence")).toBeTruthy();
    expect(screen.getByTestId("sim-canvas-true-graph")).toBeTruthy();

    cleanup();
    const witness = result.current.find((entry) => entry.id === "nexus.ui.chat.witness_thread");
    if (!witness) {
      throw new Error("witness panel missing");
    }
    const view = render(<>{witness.render()}</>);
    expect(screen.getByTestId("muse-presence-witness_thread")).toBeTruthy();
    expect(view.container.textContent).toContain("workspace binds");
    expect(view.container.textContent).toContain("2");
    expect(view.container.innerHTML).toContain("opacity: 0.96");
    expect(view.container.innerHTML).toContain("scale(1.100)");
  });

  it("keeps witness and world log panel contracts wired", async () => {
    const { result } = renderHook(() => useAppPanelConfigs(createArgs({ deferredPanelsReady: true })));

    const witness = result.current.find((entry) => entry.id === "nexus.ui.chat.witness_thread");
    if (!witness) {
      throw new Error("witness panel missing");
    }
    expect(witness.anchorKind).toBe("node");
    expect(witness.anchorId).toBe("witness_thread");
    expect(witness.worldSize).toBe("m");

    const worldLog = result.current.find((entry) => entry.id === "nexus.ui.world_log");
    if (!worldLog) {
      throw new Error("world log panel missing");
    }
    expect(worldLog.className).toContain("card");
    render(<>{worldLog.render()}</>);

    await waitFor(() => {
      expect(screen.getByTestId("lazy-world-log-panel")).toBeTruthy();
    });
  });

  it("renders placeholders for deferred lazy panels when not ready", () => {
    const { result } = renderHook(() => useAppPanelConfigs(createArgs()));

    const webGraphPanel = result.current.find((entry) => entry.id === "nexus.ui.web_graph_weaver");
    if (!webGraphPanel) {
      throw new Error("web graph panel missing");
    }
    render(<>{webGraphPanel.render()}</>);
    expect(screen.getByText("warming up panel...")).toBeTruthy();

    cleanup();
    const autopilotLedger = result.current.find((entry) => entry.id === "nexus.ui.autopilot_ledger");
    if (!autopilotLedger) {
      throw new Error("autopilot ledger panel missing");
    }
    render(<>{autopilotLedger.render()}</>);
    expect(screen.getByText("No autopilot events yet.")).toBeTruthy();
  });

  it("renders lazy panel modules when deferred panels are ready", async () => {
    const { result } = renderHook(() => useAppPanelConfigs(createArgs({ deferredPanelsReady: true })));

    const webGraphPanel = result.current.find((entry) => entry.id === "nexus.ui.web_graph_weaver");
    const threatRadarPanel = result.current.find((entry) => entry.id === "nexus.ui.threat_radar");
    const runtimeConfigPanel = result.current.find((entry) => entry.id === "nexus.ui.runtime_config");
    const mythCommonsPanel = result.current.find((entry) => entry.id === "nexus.ui.myth_commons");
    if (!webGraphPanel || !threatRadarPanel || !runtimeConfigPanel || !mythCommonsPanel) {
      throw new Error("missing deferred panel config");
    }

    render(
      <>
        {webGraphPanel.render()}
        {threatRadarPanel.render()}
        {runtimeConfigPanel.render()}
        {mythCommonsPanel.render()}
      </>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("lazy-web-graph-panel")).toBeTruthy();
      expect(screen.getByTestId("lazy-threat-radar-panel")).toBeTruthy();
      expect(screen.getByTestId("lazy-runtime-config-panel")).toBeTruthy();
      expect(screen.getByTestId("lazy-myth-panel")).toBeTruthy();
    });
  });

  it("renders autopilot event details", () => {
    const args = createArgs({
      autopilotEvents: [
        {
          ts: "2026-02-26T20:00:00Z",
          actionId: "scan-drift",
          intent: "study",
          confidence: 0.91,
          risk: 0.32,
          perms: ["drift:scan"],
          result: "ok",
          summary: "scan complete",
          gate: "confidence",
        },
      ],
    });
    const { result } = renderHook(() => useAppPanelConfigs(args));
    const autopilotLedger = result.current.find((entry) => entry.id === "nexus.ui.autopilot_ledger");
    if (!autopilotLedger) {
      throw new Error("autopilot ledger panel missing");
    }

    render(<>{autopilotLedger.render()}</>);
    expect(screen.getByText("scan complete")).toBeTruthy();
    expect(screen.getByText(/confidence 0\.91/)).toBeTruthy();
    expect(screen.getByText(/gate/)).toBeTruthy();
    expect(screen.getByText(/drift:scan/)).toBeTruthy();
  });
});
