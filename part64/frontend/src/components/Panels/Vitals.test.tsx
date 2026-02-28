/* @vitest-environment jsdom */

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { VitalsPanel } from "./Vitals";
import type { Catalog, EntityState, PresenceDynamics } from "../../types";

function makeCatalog(): Catalog {
  return {
    generated_at: "2026-02-28T00:00:00Z",
    part_roots: ["part64"],
    counts: {},
    canonical_terms: [],
    cover_fields: [],
    entity_manifest: [
      { id: "witness_thread", en: "Witness Thread", ja: "証人の糸", hue: 196 },
      { id: "chaos", en: "Chaos", ja: "カオス", hue: 328 },
    ],
  } as unknown as Catalog;
}

function makeEntities(): EntityState[] {
  return [
    {
      id: "witness_thread",
      bpm: 96,
      stability: 94,
      resonance: 7.2,
      vitals: { queue_depth: 3, drift_signal: "stable" },
    },
    {
      id: "chaos",
      bpm: 84,
      stability: 67,
      resonance: 4.8,
      vitals: { queue_depth: 6 },
    },
  ] as unknown as EntityState[];
}

function makePresenceDynamics(): PresenceDynamics {
  return {
    witness_thread: {
      en: "Witness Thread",
      ja: "証人の糸",
      notes_en: "Keeps continuity anchored.",
      notes_ja: "連続性を固定する。",
      continuity_index: 0.82,
      click_pressure: 0.21,
      file_pressure: 0.33,
      linked_presences: ["chaos"],
      lineage: [{ kind: "edge", ref: "evt:11", why_ja: "証跡リンク" }],
    },
    growth_guard: {
      mode: "balanced",
      pressure: { blend: 0.37 },
      action: {
        kind: "compact",
        collapsed_file_nodes: 2,
        collapsed_edges: 4,
        clusters: 1,
      },
    },
    presence_impacts: [
      {
        id: "witness_thread",
        affected_by: { files: 0.45, clicks: 0.2, resource: 0.3 },
        affects: { world: 0.41 },
        notes_ja: "証人の糸に資源圧が加算される。",
      },
    ],
  } as unknown as PresenceDynamics;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("VitalsPanel", () => {
  it("shows empty-state copy when entities are missing", () => {
    render(<VitalsPanel entities={[]} catalog={null} presenceDynamics={null} />);
    expect(screen.getByText("No vitals signal / バイタル信号なし")).toBeTruthy();
  });

  it("renders witness continuity, growth guard, and per-entity metrics", () => {
    render(
      <VitalsPanel
        entities={makeEntities()}
        catalog={makeCatalog()}
        presenceDynamics={makePresenceDynamics()}
      />,
    );

    expect(screen.getByText("Witness Thread / 証人の糸")).toBeTruthy();
    expect(screen.getByText("Growth Guard / 増殖監視")).toBeTruthy();
    expect(screen.getByText("mode=balanced pressure=37% action=compact")).toBeTruthy();
    expect(screen.getByText(/Linked presences: Chaos \/ カオス/)).toBeTruthy();
    expect(screen.getByText("File Influence / ファイル影響")).toBeTruthy();
    expect(screen.getByText("Resource Influence / 資源影響")).toBeTruthy();
    expect(screen.getAllByText("Pulse / 脈拍").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Stability / 安定性").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Resonance / 共鳴").length).toBeGreaterThan(0);
    expect(screen.getByText("96 BPM")).toBeTruthy();
    expect(screen.getByText("94%")).toBeTruthy();
    expect(screen.getByText("7.2Hz")).toBeTruthy();
  });
});
