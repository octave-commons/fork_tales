/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { WorldAnchorTarget, WorldPanelNexusEntry } from "../../app/worldPanelLayout";
import { WorldPanelsViewport } from "./WorldPanelsViewport";

type WorldPanelsViewportProps = ComponentProps<typeof WorldPanelsViewport>;
type SortedPanel = WorldPanelsViewportProps["sortedPanels"][number];

function makeAnchor(
  id: string,
  label: string,
  overrides: Partial<WorldAnchorTarget> = {},
): WorldAnchorTarget {
  return {
    kind: "node",
    id,
    label,
    x: 0.56,
    y: 0.42,
    radius: 0.18,
    hue: 192,
    confidence: 0.74,
    presenceSignature: {},
    ...overrides,
  };
}

function makePanel(overrides: Partial<SortedPanel> = {}): SortedPanel {
  const id = overrides.id ?? "nexus.ui.chat.witness_thread";
  return {
    id,
    fallbackSpan: 1,
    render: () => <div data-testid={`panel-${id}`}>panel::{id}</div>,
    priority: 1,
    depth: 1,
    councilScore: 0.82,
    councilBoost: 1,
    councilReason: "running lane",
    presenceId: "witness_thread",
    presenceLabel: "Witness Thread",
    presenceLabelJa: "証人の糸",
    presenceRole: "camera-guidance",
    particleDisposition: "role-bound",
    particleCount: 28,
    toolHints: ["focus", "guide", "sync"],
    ...overrides,
  };
}

function makeNexusEntry(panel: SortedPanel, anchor: WorldAnchorTarget): WorldPanelNexusEntry {
  return {
    panelId: panel.id,
    panelLabel: panel.presenceLabel,
    anchor,
    x: anchor.x,
    y: anchor.y,
    hue: anchor.hue,
    confidence: anchor.confidence,
    open: true,
    minimized: false,
    selected: false,
  };
}

function makeProps(overrides: Partial<WorldPanelsViewportProps> = {}): WorldPanelsViewportProps {
  const sortedPanels =
    overrides.sortedPanels ??
    [
      makePanel({
        id: "nexus.ui.chat.witness_thread",
        presenceLabel: "Witness Thread",
        presenceId: "witness_thread",
        presenceRole: "camera-guidance",
      }),
      makePanel({
        id: "nexus.ui.runtime_config",
        presenceLabel: "Runtime Config",
        presenceId: "health_sentinel_cpu",
        presenceRole: "compute-scheduler",
        councilScore: 0.76,
        councilReason: "waiting for cycle",
      }),
    ];

  const panelNexusLayout =
    overrides.panelNexusLayout ??
    sortedPanels.map((panel) =>
      makeNexusEntry(
        panel,
        makeAnchor(panel.presenceId || panel.id, panel.presenceLabel, {
          hue: panel.id === "nexus.ui.runtime_config" ? 156 : 202,
        }),
      ),
    );

  return {
    viewportWidth: 1480,
    viewportHeight: 920,
    worldPanelLayout: [],
    panelNexusLayout,
    sortedPanels,
    panelWindowStateById: {},
    tertiaryPinnedPanelId: null,
    pinnedPanels: {},
    selectedPanelId: sortedPanels[0]?.id ?? null,
    isEditMode: false,
    coreFlightSpeed: 1.2,
    onToggleEditMode: vi.fn(),
    onHoverPanel: vi.fn(),
    onSelectPanel: vi.fn(),
    onTogglePanelPin: vi.fn(),
    onActivatePanel: vi.fn(),
    onMinimizePanel: vi.fn(),
    onClosePanel: vi.fn(),
    onAdjustPanelCouncilRank: vi.fn(),
    onPinPanelToTertiary: vi.fn(),
    onFlyCameraToAnchor: vi.fn(),
    onGlassInteractAt: vi.fn(),
    onNudgeCameraPan: vi.fn(),
    onWorldPanelDragEnd: vi.fn(),
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("WorldPanelsViewport", () => {
  it("routes toolbar actions for runtime config and edit mode", () => {
    const props = makeProps();
    render(<WorldPanelsViewport {...props} />);

    fireEvent.click(screen.getByRole("button", { name: "config" }));
    fireEvent.click(screen.getByRole("button", { name: "edit rank" }));

    expect(props.onActivatePanel).toHaveBeenCalledWith("nexus.ui.runtime_config");
    expect(props.onSelectPanel).toHaveBeenCalledWith("nexus.ui.runtime_config");
    expect(props.onToggleEditMode).toHaveBeenCalledTimes(1);
  });

  it("opens tray cards and runs file-analysis job requests", async () => {
    const analysisPanel = makePanel({
      id: "nexus.ui.chat.anchor_registry",
      presenceId: "anchor_registry",
      presenceLabel: "Anchor Registry",
      presenceRole: "file-analysis",
      councilReason: "active analysis",
    });
    const props = makeProps({
      sortedPanels: [analysisPanel],
      panelNexusLayout: [
        makeNexusEntry(
          analysisPanel,
          makeAnchor("anchor_registry", "Anchor Registry", { x: 0.61, y: 0.44, hue: 176 }),
        ),
      ],
      selectedPanelId: analysisPanel.id,
    });

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      {
        ok: true,
        status: 200,
        json: async () => ({ ok: true, sync_status: "queued" }),
      } as Response,
    );

    const view = render(<WorldPanelsViewport {...props} />);

    fireEvent.click(screen.getByRole("button", { name: /Anchor Registry/ }));
    const trayCard = view.container.querySelector(".world-unity-tray-card");
    if (!(trayCard instanceof HTMLElement)) {
      throw new Error("tray card did not open");
    }

    fireEvent.click(within(trayCard).getByRole("button", { name: "job" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });

    const syncCall = fetchMock.mock.calls.find(([url]) => String(url).includes("/api/eta-mu/sync"));
    expect(syncCall).toBeTruthy();
    expect((syncCall?.[1] as RequestInit | undefined)?.method).toBe("POST");
    expect((syncCall?.[1] as RequestInit | undefined)?.body).toBe("{}");

    fireEvent.click(within(trayCard).getByRole("button", { name: "guide" }));

    expect(props.onActivatePanel).toHaveBeenCalledWith(analysisPanel.id);
    expect(props.onSelectPanel).toHaveBeenCalledWith(analysisPanel.id);
    expect(props.onFlyCameraToAnchor).toHaveBeenCalled();
  });

  it("filters the presence rail by status and search query", async () => {
    const runningPanel = makePanel({
      id: "nexus.ui.chat.running",
      presenceId: "presence.running",
      presenceLabel: "Running Lens",
      presenceRole: "compute-scheduler",
      councilReason: "stable operational state",
    });
    const pausedPanel = makePanel({
      id: "nexus.ui.chat.paused",
      presenceId: "presence.paused",
      presenceLabel: "Paused Lens",
      presenceRole: "file-analysis",
      councilReason: "waiting for queue",
    });
    const blockedPanel = makePanel({
      id: "nexus.ui.chat.blocked",
      presenceId: "presence.blocked",
      presenceLabel: "Blocked Lens",
      presenceRole: "compliance-gating",
      councilReason: "permission denied by gate",
    });

    const props = makeProps({
      sortedPanels: [runningPanel, pausedPanel, blockedPanel],
      panelNexusLayout: [
        makeNexusEntry(runningPanel, makeAnchor("presence.running", "Running Lens", { hue: 180 })),
        makeNexusEntry(pausedPanel, makeAnchor("presence.paused", "Paused Lens", { hue: 90 })),
        makeNexusEntry(blockedPanel, makeAnchor("presence.blocked", "Blocked Lens", { hue: 8 })),
      ],
      selectedPanelId: runningPanel.id,
    });

    render(<WorldPanelsViewport {...props} />);

    fireEvent.click(screen.getByRole("button", { name: /block\s+1/i }));
    expect(screen.getByRole("button", { name: /Blocked Lens/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Running Lens/ })).toBeNull();

    fireEvent.change(screen.getByPlaceholderText("presence, role, panel"), {
      target: { value: "zzz-no-match" },
    });

    await waitFor(() => {
      expect(screen.getByText("no presences match filters")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "reset filters" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Running Lens/ })).toBeTruthy();
      expect(screen.getByRole("button", { name: /Paused Lens/ })).toBeTruthy();
      expect(screen.getByRole("button", { name: /Blocked Lens/ })).toBeTruthy();
    });
  });
});
