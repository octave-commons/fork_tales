import { describe, expect, it } from "vitest";

import {
  APP_WORKSPACE_NORMALIZE_OPTIONS,
  DEFAULT_INTERFACE_OPACITY,
  FIXED_MUSE_PRESENCES,
  GLASS_VIEWPORT_PANEL_ID,
  INTERFACE_OPACITY_MAX,
  INTERFACE_OPACITY_MIN,
  PANEL_TOOL_HINTS,
  PRESENCE_OPERATIONAL_ROLE_BY_ID,
  isGlassPrimaryPanelId,
} from "./appShellConstants";

describe("appShellConstants", () => {
  it("keeps interface opacity defaults in bounds", () => {
    expect(INTERFACE_OPACITY_MIN).toBeLessThan(INTERFACE_OPACITY_MAX);
    expect(DEFAULT_INTERFACE_OPACITY).toBeGreaterThanOrEqual(INTERFACE_OPACITY_MIN);
    expect(DEFAULT_INTERFACE_OPACITY).toBeLessThanOrEqual(INTERFACE_OPACITY_MAX);
  });

  it("exposes fixed muse lanes and panel hints", () => {
    expect(FIXED_MUSE_PRESENCES.map((entry) => entry.presenceId)).toEqual([
      "witness_thread",
      "chaos",
      "stability",
      "symmetry",
    ]);
    expect(PANEL_TOOL_HINTS[GLASS_VIEWPORT_PANEL_ID]).toEqual(["glass", "camera", "pan"]);
    expect(PANEL_TOOL_HINTS["nexus.ui.threat_radar"]).toEqual(["threat", "security", "review"]);
  });

  it("maps known presence roles and glass panel detection", () => {
    expect(PRESENCE_OPERATIONAL_ROLE_BY_ID.witness_thread).toBe("crawl-routing");
    expect(PRESENCE_OPERATIONAL_ROLE_BY_ID.health_sentinel_cpu).toBe("compute-scheduler");

    expect(isGlassPrimaryPanelId(GLASS_VIEWPORT_PANEL_ID)).toBe(true);
    expect(isGlassPrimaryPanelId("nexus.ui.dedicated_views")).toBe(true);
    expect(isGlassPrimaryPanelId("nexus.ui.world_log")).toBe(false);
  });

  it("sets workspace normalization limits", () => {
    expect(APP_WORKSPACE_NORMALIZE_OPTIONS.maxPinnedFileNodeIds).toBe(48);
    expect(APP_WORKSPACE_NORMALIZE_OPTIONS.maxSearchQueryLength).toBe(180);
    expect(APP_WORKSPACE_NORMALIZE_OPTIONS.maxPinnedNexusSummaries).toBe(24);
  });
});
