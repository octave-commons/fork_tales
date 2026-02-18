import type { ReactNode } from "react";

export type PanelAnchorKind = "node" | "cluster" | "region";
export type PanelPreferredSide = "left" | "right" | "top" | "bottom";
export type PanelWorldSize = "s" | "m" | "l" | "xl";

export interface PanelConfig {
  id: string;
  fallbackSpan: number;
  className?: string;
  anchorKind?: PanelAnchorKind;
  anchorId?: string;
  worldSize?: PanelWorldSize;
  pinnedByDefault?: boolean;
  render: () => ReactNode;
}

export interface WorldAnchorTarget {
  kind: PanelAnchorKind;
  id: string;
  label: string;
  x: number;
  y: number;
  radius: number;
  hue: number;
  confidence: number;
  presenceSignature: Record<string, number>;
}

export interface WorldPanelLayoutEntry {
  id: string;
  panel: PanelConfig & { priority: number; depth: number };
  anchor: WorldAnchorTarget;
  anchorScreenX: number;
  anchorScreenY: number;
  side: PanelPreferredSide;
  x: number;
  y: number;
  width: number;
  height: number;
  panelWorldX: number;
  panelWorldY: number;
  panelWorldZ: number;
  pixelsPerWorldX: number;
  pixelsPerWorldY: number;
  tetherX: number;
  tetherY: number;
  glow: number;
  collapse: boolean;
}

export interface PanelWindowState {
  open: boolean;
  minimized: boolean;
}

export interface WorldPanelNexusEntry {
  panelId: string;
  panelLabel: string;
  anchor: WorldAnchorTarget;
  x: number;
  y: number;
  hue: number;
  confidence: number;
  open: boolean;
  minimized: boolean;
  selected: boolean;
}

interface PanelAnchorPreset {
  kind: PanelAnchorKind;
  worldSize: PanelWorldSize;
  pinnedByDefault?: boolean;
  anchorId?: string;
}

export const MAX_WORLD_PANELS_VISIBLE = 5;
export const WORLD_PANEL_MARGIN = 16;

export const PANEL_ANCHOR_PRESETS: Record<string, PanelAnchorPreset> = {
  "nexus.ui.command_center": {
    kind: "node",
    worldSize: "xl",
    pinnedByDefault: true,
    anchorId: "gates_of_truth",
  },
  "nexus.ui.chat.witness_thread": {
    kind: "node",
    worldSize: "l",
    pinnedByDefault: true,
    anchorId: "witness_thread",
  },
  "nexus.ui.web_graph_weaver": {
    kind: "cluster",
    worldSize: "m",
  },
  "nexus.ui.inspiration_atlas": {
    kind: "cluster",
    worldSize: "m",
  },
  "nexus.ui.entity_vitals": {
    kind: "node",
    worldSize: "m",
  },
  "nexus.ui.projection_ledger": {
    kind: "region",
    worldSize: "m",
    pinnedByDefault: false,
  },
  "nexus.ui.autopilot_ledger": {
    kind: "region",
    worldSize: "m",
  },
  "nexus.ui.world_log": {
    kind: "node",
    worldSize: "l",
    pinnedByDefault: true,
    anchorId: "witness_thread",
  },
  "nexus.ui.stability_observatory": {
    kind: "node",
    worldSize: "l",
    pinnedByDefault: true,
    anchorId: "fork_tax_canticle",
  },
  "nexus.ui.omni_archive": {
    kind: "node",
    worldSize: "l",
    pinnedByDefault: true,
    anchorId: "receipt_river",
  },
  "nexus.ui.myth_commons": {
    kind: "node",
    worldSize: "m",
    pinnedByDefault: true,
    anchorId: "anchor_registry",
  },
  "nexus.ui.dedicated_views": {
    kind: "region",
    worldSize: "xl",
    anchorId: "anchor_registry",
  },
};

export function defaultPinnedPanelMap(panelIds: string[]): Record<string, boolean> {
  const pinned: Record<string, boolean> = {};
  panelIds.forEach((panelId) => {
    pinned[panelId] = Boolean(PANEL_ANCHOR_PRESETS[panelId]?.pinnedByDefault);
  });
  return pinned;
}

function clampValue(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function normalizeUnit(raw: number | undefined | null, fallback = 0.5): number {
  if (typeof raw !== "number" || Number.isNaN(raw)) {
    return fallback;
  }
  return clampValue(raw, 0, 1);
}

export function panelSizeForWorld(
  worldSize: PanelWorldSize,
  priority: number,
  zoom: number,
  speedNorm: number,
): { width: number; height: number; collapse: boolean } {
  const bySize: Record<PanelWorldSize, { w: number; h: number }> = {
    s: { w: 306, h: 206 },
    m: { w: 396, h: 268 },
    l: { w: 484, h: 332 },
    xl: { w: 586, h: 404 },
  };
  const base = bySize[worldSize];
  const zoomScale = clampValue(0.98 + ((zoom - 1) * 0.36), 0.88, 1.32);
  const priorityScale = 1.02 + (priority * 0.3);
  const motionScale = 1 - (speedNorm * 0.18);
  const width = clampValue(Math.round(base.w * zoomScale * priorityScale * motionScale), 240, 780);
  const height = clampValue(Math.round(base.h * zoomScale * (1 + priority * 0.22) * motionScale), 162, 620);
  const collapse = speedNorm > 0.62;
  return { width: collapse ? Math.round(width * 0.58) : width, height: collapse ? 56 : height, collapse };
}

export function preferredSideForAnchor(
  panelId: string,
  px: number,
  py: number,
  viewportWidth: number,
  viewportHeight: number,
  sideByPanel: Map<string, PanelPreferredSide>,
): PanelPreferredSide {
  const dx = px - (viewportWidth / 2);
  const dy = py - (viewportHeight / 2);
  const axisDominant = Math.abs(dx) > Math.abs(dy) * 1.2;
  const suggested: PanelPreferredSide = axisDominant
    ? (dx < 0 ? "left" : "right")
    : (dy < 0 ? "top" : "bottom");
  const previous = sideByPanel.get(panelId);
  if (!previous) {
    sideByPanel.set(panelId, suggested);
    return suggested;
  }
  const shouldFlip =
    (previous === "left" && dx > 88)
    || (previous === "right" && dx < -88)
    || (previous === "top" && dy > 64)
    || (previous === "bottom" && dy < -64);
  if (shouldFlip) {
    sideByPanel.set(panelId, suggested);
    return suggested;
  }
  return previous;
}

export function anchorOffsetForSide(side: PanelPreferredSide): { x: number; y: number } {
  if (side === "left") {
    return { x: -34, y: -8 };
  }
  if (side === "right") {
    return { x: 34, y: -8 };
  }
  if (side === "top") {
    return { x: 0, y: -32 };
  }
  return { x: 0, y: 32 };
}

export function overlapAmount(a: WorldPanelLayoutEntry, b: WorldPanelLayoutEntry): { x: number; y: number } | null {
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;
  const overlapX = Math.min(ax2, bx2) - Math.max(a.x, b.x);
  const overlapY = Math.min(ay2, by2) - Math.max(a.y, b.y);
  if (overlapX <= 0 || overlapY <= 0) {
    return null;
  }
  return { x: overlapX, y: overlapY };
}

export function containsAnchorNoCoverZone(panel: WorldPanelLayoutEntry, radius = 28): boolean {
  const cx = panel.anchorScreenX;
  const cy = panel.anchorScreenY;
  return (
    cx >= panel.x - radius
    && cx <= panel.x + panel.width + radius
    && cy >= panel.y - radius
    && cy <= panel.y + panel.height + radius
  );
}
