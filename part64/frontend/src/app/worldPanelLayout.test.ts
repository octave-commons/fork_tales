import { describe, expect, it } from "vitest";

import {
  PANEL_ANCHOR_PRESETS,
  anchorOffsetForSide,
  containsAnchorNoCoverZone,
  defaultPinnedPanelMap,
  normalizeUnit,
  overlapAmount,
  panelSizeForWorld,
  preferredSideForAnchor,
  type WorldPanelLayoutEntry,
} from "./worldPanelLayout";

function makeEntry(overrides: Partial<WorldPanelLayoutEntry> = {}): WorldPanelLayoutEntry {
  return {
    id: "panel-a",
    panel: {
      id: "panel-a",
      fallbackSpan: 6,
      render: () => null,
      priority: 0.4,
      depth: 50,
    },
    anchor: {
      kind: "node",
      id: "anchor-a",
      label: "Anchor A",
      x: 0.5,
      y: 0.5,
      radius: 0.1,
      hue: 180,
      confidence: 0.8,
      presenceSignature: { anchor_a: 1 },
    },
    anchorScreenX: 100,
    anchorScreenY: 80,
    side: "left",
    x: 20,
    y: 20,
    width: 120,
    height: 90,
    panelWorldX: 0,
    panelWorldY: 0,
    panelWorldZ: 0,
    pixelsPerWorldX: 100,
    pixelsPerWorldY: 100,
    tetherX: 0,
    tetherY: 0,
    glow: 0.6,
    collapse: false,
    ...overrides,
  };
}

describe("worldPanelLayout helpers", () => {
  it("builds default pinned map from presets", () => {
    const pinned = defaultPinnedPanelMap([
      "nexus.ui.chat.witness_thread",
      "nexus.ui.web_graph_weaver",
      "custom.panel",
    ]);

    expect(pinned["nexus.ui.chat.witness_thread"]).toBe(true);
    expect(pinned["nexus.ui.web_graph_weaver"]).toBe(false);
    expect(pinned["custom.panel"]).toBe(false);
    expect(PANEL_ANCHOR_PRESETS["nexus.ui.glass_viewport"]?.anchorId).toBe("view_lens_keeper");
  });

  it("normalizes unit values with fallback and clamping", () => {
    expect(normalizeUnit(undefined, 0.3)).toBe(0.3);
    expect(normalizeUnit(-10)).toBe(0);
    expect(normalizeUnit(7)).toBe(1);
    expect(normalizeUnit(0.42)).toBe(0.42);
  });

  it("computes panel sizes and collapse mode from motion", () => {
    const calm = panelSizeForWorld("m", 0.4, 1.1, 0.1);
    const fast = panelSizeForWorld("m", 0.4, 1.1, 0.9);

    expect(calm.collapse).toBe(false);
    expect(calm.width).toBeGreaterThan(240);
    expect(calm.height).toBeGreaterThan(162);

    expect(fast.collapse).toBe(true);
    expect(fast.height).toBe(56);
    expect(fast.width).toBeLessThan(calm.width);
  });

  it("chooses and stabilizes preferred side per panel", () => {
    const sideByPanel = new Map<string, "left" | "right" | "top" | "bottom">();
    const initial = preferredSideForAnchor("panel-a", 20, 100, 300, 200, sideByPanel);
    expect(initial).toBe("left");

    const sticky = preferredSideForAnchor("panel-a", 130, 100, 300, 200, sideByPanel);
    expect(sticky).toBe("left");

    const flipped = preferredSideForAnchor("panel-a", 270, 100, 300, 200, sideByPanel);
    expect(flipped).toBe("right");
  });

  it("returns offsets by preferred side", () => {
    expect(anchorOffsetForSide("left")).toEqual({ x: -34, y: -8 });
    expect(anchorOffsetForSide("right")).toEqual({ x: 34, y: -8 });
    expect(anchorOffsetForSide("top")).toEqual({ x: 0, y: -32 });
    expect(anchorOffsetForSide("bottom")).toEqual({ x: 0, y: 32 });
  });

  it("computes overlap amounts and anchor no-cover zones", () => {
    const a = makeEntry({ x: 20, y: 20, width: 100, height: 80, anchorScreenX: 70, anchorScreenY: 60 });
    const b = makeEntry({ id: "panel-b", x: 90, y: 50, width: 90, height: 60 });
    const c = makeEntry({ id: "panel-c", x: 260, y: 240, width: 50, height: 40 });

    expect(overlapAmount(a, b)).toEqual({ x: 30, y: 50 });
    expect(overlapAmount(a, c)).toBeNull();
    expect(containsAnchorNoCoverZone(a, 20)).toBe(true);
    expect(containsAnchorNoCoverZone({ ...a, anchorScreenX: 400, anchorScreenY: 300 }, 20)).toBe(false);
  });
});
