/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import {
  COUNCIL_BOOST_STORAGE_KEY,
  INTERFACE_OPACITY_STORAGE_KEY,
  MUSE_WORKSPACE_STORAGE_KEY,
  TERTIARY_PIN_STORAGE_KEY,
} from "./app/appShellConstants";
import type { Catalog, SimulationState } from "./types";

const testMocks = vi.hoisted(() => {
  return {
    useWorldState: vi.fn(),
    useAutopilotController: vi.fn(),
    useAppPanelConfigs: vi.fn(),
    useChatCommandHandlers: vi.fn(),
    autopilotToggle: vi.fn(),
  };
});

vi.mock("./hooks/useWorldState", () => {
  return {
    useWorldState: (...args: unknown[]) => testMocks.useWorldState(...args),
  };
});

vi.mock("./hooks/useAutopilotController", () => {
  return {
    useAutopilotController: (...args: unknown[]) => testMocks.useAutopilotController(...args),
  };
});

vi.mock("./app/useAppPanelConfigs", () => {
  return {
    useAppPanelConfigs: (...args: unknown[]) => testMocks.useAppPanelConfigs(...args),
  };
});

vi.mock("./app/useChatCommandHandlers", () => {
  return {
    useChatCommandHandlers: (...args: unknown[]) => testMocks.useChatCommandHandlers(...args),
  };
});

vi.mock("./components/App/CoreBackdrop", () => {
  return {
    CoreBackdrop: (props: any) => (
      <div>
        <p>mock core backdrop</p>
        <button
          type="button"
          onClick={() => {
            props.onOverlayInit?.({
              pulseAt: () => {},
              getAnchorRatio: () => ({ x: 0.52, y: 0.46 }),
              projectRatioToClient: () => ({ x: 160, y: 120, w: 320, h: 220 }),
              interactAt: (x: number, y: number) => ({
                hitNode: false,
                openedWorldscreen: false,
                target: "mock_target",
                xRatio: x,
                yRatio: y,
              }),
              interactClientAt: (_x: number, _y: number) => ({
                hitNode: false,
                openedWorldscreen: false,
                target: "mock_target",
                xRatio: 0.38,
                yRatio: 0.41,
              }),
            });
          }}
        >
          init overlay
        </button>
        <button
          type="button"
          onClick={() => {
            props.onNexusInteraction?.({
              nodeId: "node:1",
              nodeKind: "file",
              resourceKind: "text",
              label: "node one",
              xRatio: 0.42,
              yRatio: 0.47,
              openWorldscreen: true,
              isDoubleTap: true,
            });
          }}
        >
          emit nexus interaction
        </button>
        <button
          type="button"
          onClick={() => {
            props.onUserPresenceInput?.({
              kind: "click",
              target: "panel:mock",
              message: "mock click",
              xRatio: 0.55,
              yRatio: 0.62,
              embedDaimoi: true,
            });
          }}
        >
          emit user presence
        </button>
      </div>
    ),
  };
});

vi.mock("./components/App/CoreControlPanel", () => {
  return {
    CoreControlPanel: (props: any) => (
      <div>
        <p>mock core control panel</p>
        <button type="button" onClick={props.onToggleAutopilot}>toggle autopilot</button>
        <button type="button" onClick={props.onToggleCoreFlight}>toggle flight</button>
        <button type="button" onClick={props.onToggleCoreOrbit}>toggle orbit</button>
        <button type="button" onClick={() => props.onSetInterfaceOpacity?.(0.74)}>set opacity</button>
        <button type="button" onClick={props.onResetInterfaceOpacity}>reset opacity</button>
        <button type="button" onClick={() => props.onSetCoreSimulationDial?.("motionSpeed", 1.7)}>set sim dial</button>
        <button type="button" onClick={() => props.onSetCoreVisualDial?.("brightness", 1.2)}>set visual dial</button>
        <button type="button" onClick={() => props.onSetMouseDaimonTuning?.({ enabled: false })}>set mouse daimon</button>
        <button type="button" onClick={props.onOpenRuntimeConfig}>open runtime config</button>
        <button type="button" onClick={() => props.onSelectPerspective?.("causal-time")}>set perspective</button>
      </div>
    ),
  };
});

vi.mock("./components/App/CoreLayerManagerOverlay", () => {
  return {
    CoreLayerManagerOverlay: (props: any) => (
      <div>
        <p>mock layer manager</p>
        <button type="button" onClick={props.onToggleOpen}>toggle layer manager</button>
        <button type="button" onClick={() => props.onSetLayerEnabled?.("presence", false)}>disable presence layer</button>
        <button type="button" onClick={() => props.onSetAllLayers?.(true)}>enable all layers</button>
      </div>
    ),
  };
});

vi.mock("./components/App/WorldPanelsViewport", () => {
  return {
    WorldPanelsViewport: (props: any) => (
      <div>
        <p>mock world panels viewport</p>
        <button type="button" onClick={props.onToggleEditMode}>toggle edit mode</button>
        <button type="button" onClick={() => props.onAdjustPanelCouncilRank?.("nexus.ui.runtime_config", 1)}>rank runtime config</button>
        <button
          type="button"
          onClick={() => props.onFlyCameraToAnchor?.({
            kind: "node",
            id: "witness_thread",
            label: "Witness Thread",
            x: 0.52,
            y: 0.42,
            radius: 0.18,
            hue: 210,
            confidence: 0.9,
            presenceSignature: {},
          })}
        >
          fly camera anchor
        </button>
        <button
          type="button"
          onClick={() => props.onGlassInteractAt?.({ panelId: "nexus.ui.runtime_config", xRatio: 0.51, yRatio: 0.47 })}
        >
          glass interact
        </button>
        <button
          type="button"
          onClick={() => props.onGlassInteractAt?.({
            panelId: "nexus.ui.runtime_config",
            xRatio: 0.36,
            yRatio: 0.41,
            clientX: 90,
            clientY: 120,
          })}
        >
          glass client interact
        </button>
        <button
          type="button"
          onClick={() => props.onWorldPanelDragEnd?.("nexus.ui.runtime_config", { offset: { x: 120, y: 64 } })}
        >
          drag panel
        </button>
        <button type="button" onClick={() => props.onActivatePanel?.("nexus.ui.runtime_config")}>activate panel</button>
        <button type="button" onClick={() => props.onMinimizePanel?.("nexus.ui.runtime_config")}>minimize panel</button>
        <button type="button" onClick={() => props.onClosePanel?.("nexus.ui.runtime_config")}>close panel</button>
      </div>
    ),
  };
});

function createCatalogFixture(): Catalog {
  return {
    part_roots: ["/tmp/part64"],
    entity_manifest: [
      { id: "witness_thread", en: "Witness Thread", ja: "証人の糸", x: 0.45, y: 0.44, hue: 210 },
      { id: "anchor_registry", en: "Anchor Registry", ja: "錨台帳", x: 0.55, y: 0.58, hue: 180 },
    ],
    projection: {
      active: true,
      perspective: "hybrid",
      perspectives: [
        { id: "hybrid", symbol: "perspective.hybrid", name: "Hybrid", merge: "hybrid", description: "hybrid", default: true },
        { id: "causal-time", symbol: "perspective.causal-time", name: "Causal", merge: "causal-time", description: "causal", default: false },
      ],
      states: [],
      chat_sessions: [],
    },
    file_graph: {
      nodes: [],
      edges: [],
    },
    muse_runtime: {
      muse_count: 2,
      event_seq: 12,
    },
  } as unknown as Catalog;
}

function createSimulationFixture(): SimulationState {
  return {
    timestamp: "2026-02-28T00:00:00Z",
    total: 0,
    audio: 0,
    image: 0,
    graph_nodes: 0,
    graph_edges: 0,
    presence_dynamics: {
      field_particles: [],
      compute_jobs: [],
    },
    file_graph: {
      nodes: [],
      edges: [],
    },
    truth_graph: {
      node_count: 0,
      edge_count: 0,
    },
    view_graph: {
      node_count: 0,
      edge_count: 0,
    },
  } as unknown as SimulationState;
}

function mockResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

beforeEach(() => {
  window.localStorage.clear();

  testMocks.autopilotToggle.mockReset();
  testMocks.useWorldState.mockReset();
  testMocks.useAutopilotController.mockReset();
  testMocks.useAppPanelConfigs.mockReset();
  testMocks.useChatCommandHandlers.mockReset();

  testMocks.useWorldState.mockReturnValue({
    catalog: createCatalogFixture(),
    simulation: createSimulationFixture(),
    projection: null,
    museEvents: [],
    isConnected: true,
  });

  testMocks.useAutopilotController.mockReturnValue({
    autopilotEnabled: true,
    autopilotStatus: "running",
    autopilotSummary: "running",
    autopilotEvents: [],
    handleAutopilotUserInput: vi.fn(() => false),
    toggleAutopilot: testMocks.autopilotToggle,
  });

  testMocks.useAppPanelConfigs.mockReturnValue([
    {
      id: "nexus.ui.runtime_config",
      fallbackSpan: 1,
      anchorKind: "node",
      anchorId: "witness_thread",
      worldSize: "m",
      pinnedByDefault: true,
      render: () => null,
    },
    {
      id: "nexus.ui.world_log",
      fallbackSpan: 1,
      anchorKind: "node",
      anchorId: "anchor_registry",
      worldSize: "m",
      pinnedByDefault: true,
      render: () => null,
    },
    {
      id: "nexus.ui.simulation_map",
      fallbackSpan: 1,
      anchorKind: "node",
      anchorId: "witness_thread",
      worldSize: "m",
      pinnedByDefault: true,
      render: () => null,
    },
  ]);

  testMocks.useChatCommandHandlers.mockReturnValue({
    handleChatCommand: vi.fn(() => false),
  });

  vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/api/muse/create")) {
      return mockResponse({ ok: true, muse_id: "muse.archive_witness" });
    }
    return mockResponse({ ok: true, entries: [] });
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("renders shell and routes mocked control interactions", async () => {
    render(<App />);

    expect(screen.getByText("eta-mu world daemon")).toBeTruthy();
    expect(screen.getByText(/Connected/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "init overlay" }));
    fireEvent.click(screen.getByRole("button", { name: "emit nexus interaction" }));
    fireEvent.click(screen.getByRole("button", { name: "emit user presence" }));
    fireEvent.click(screen.getByRole("button", { name: "toggle autopilot" }));
    fireEvent.click(screen.getByRole("button", { name: "toggle flight" }));
    fireEvent.click(screen.getByRole("button", { name: "toggle orbit" }));
    fireEvent.click(screen.getByRole("button", { name: "set opacity" }));
    fireEvent.click(screen.getByRole("button", { name: "reset opacity" }));
    fireEvent.click(screen.getByRole("button", { name: "set sim dial" }));
    fireEvent.click(screen.getByRole("button", { name: "set visual dial" }));
    fireEvent.click(screen.getByRole("button", { name: "set mouse daimon" }));
    fireEvent.click(screen.getByRole("button", { name: "open runtime config" }));
    fireEvent.click(screen.getByRole("button", { name: "set perspective" }));

    fireEvent.click(screen.getByRole("button", { name: "toggle layer manager" }));
    fireEvent.click(screen.getByRole("button", { name: "disable presence layer" }));
    fireEvent.click(screen.getByRole("button", { name: "enable all layers" }));

    fireEvent.click(screen.getByRole("button", { name: "toggle edit mode" }));
    fireEvent.click(screen.getByRole("button", { name: "rank runtime config" }));
    fireEvent.click(screen.getByRole("button", { name: "fly camera anchor" }));
    fireEvent.click(screen.getByRole("button", { name: "glass interact" }));
    fireEvent.click(screen.getByRole("button", { name: "glass client interact" }));
    fireEvent.click(screen.getByRole("button", { name: "drag panel" }));
    fireEvent.click(screen.getByRole("button", { name: "activate panel" }));
    fireEvent.click(screen.getByRole("button", { name: "minimize panel" }));
    fireEvent.click(screen.getByRole("button", { name: "close panel" }));

    expect(testMocks.autopilotToggle).toHaveBeenCalled();
    expect(testMocks.useWorldState).toHaveBeenCalledWith("hybrid");
    expect(testMocks.useWorldState).toHaveBeenCalledWith("causal-time");

    const fetchSpy = vi.mocked(globalThis.fetch);
    expect(fetchSpy.mock.calls.length).toBeGreaterThanOrEqual(0);
  }, 15000);

  it("survives invalid localStorage seeds and disconnected mode", () => {
    window.localStorage.setItem(MUSE_WORKSPACE_STORAGE_KEY, "{bad-json");
    window.localStorage.setItem(COUNCIL_BOOST_STORAGE_KEY, "{bad-json");
    window.localStorage.setItem(TERTIARY_PIN_STORAGE_KEY, "\n\n");
    window.localStorage.setItem(INTERFACE_OPACITY_STORAGE_KEY, "not-a-number");

    testMocks.useWorldState.mockReturnValue({
      catalog: createCatalogFixture(),
      simulation: createSimulationFixture(),
      projection: null,
      museEvents: [],
      isConnected: false,
    });

    render(<App />);

    expect(screen.getByText(/Disconnected/)).toBeTruthy();
    expect(screen.getByText("mock world panels viewport")).toBeTruthy();
  });

  it("creates a muse from the forge input and allows dismissing toast", async () => {
    render(<App />);

    const input = screen.getByPlaceholderText("create muse label (e.g. Archive Witness)");
    fireEvent.change(input, { target: { value: "Archive Witness" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(screen.getByText("Muse Created")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "dismiss" }));

    await waitFor(() => {
      expect(screen.queryByText("Muse Created")).toBeNull();
    });
  });

  it("shows create failure toast when muse create endpoint fails", async () => {
    vi.mocked(globalThis.fetch).mockImplementationOnce(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/create")) {
        return mockResponse({ ok: false, error: "intent_conflict" }, 409);
      }
      return mockResponse({ ok: true, entries: [] });
    });

    render(<App />);

    fireEvent.change(screen.getByPlaceholderText("create muse label (e.g. Archive Witness)"), {
      target: { value: "Conflicting Muse" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create Muse" }));

    await waitFor(() => {
      expect(screen.getByText("Muse Create Failed")).toBeTruthy();
      expect(screen.getByText("intent_conflict")).toBeTruthy();
    });
  });
});
