/* @vitest-environment jsdom */

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { InspirationAtlasPanel } from "./InspirationAtlasPanel";
import type { SimulationState } from "../../types";

function makeSimulation(): SimulationState {
  return {
    presence_dynamics: {
      river_flow: { rate: 6 },
      witness_thread: { continuity_index: 0.8 },
      ghost: { auto_commit_pulse: 0.25 },
      fork_tax: { paid_ratio: 0.4 },
      presence_impacts: [
        {
          id: "receipt_river",
          affected_by: { files: 0.4, clicks: 0.2 },
          affects: { world: 0.5, ledger: 0.3 },
        },
        {
          id: "anchor_registry",
          affected_by: { files: 0.3, clicks: 0.1 },
          affects: { world: 0.2, ledger: 0.2 },
        },
        {
          id: "mage_of_receipts",
          affected_by: { files: 0.2, clicks: 0.3 },
          affects: { world: 0.4, ledger: 0.1 },
        },
      ],
    },
  } as unknown as SimulationState;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("InspirationAtlasPanel", () => {
  it("renders default forces when simulation is absent", () => {
    render(<InspirationAtlasPanel simulation={null} />);

    expect(screen.getByText("Inspiration Atlas / .eta-mu")).toBeTruthy();
    expect(screen.getByText("Web Search Sync")).toBeTruthy();
    expect(screen.getByText("force 0% · anchor receipt_river")).toBeTruthy();
  });

  it("computes board force and signal summaries from presence dynamics", () => {
    render(<InspirationAtlasPanel simulation={makeSimulation()} />);

    expect(screen.getByText("Web Search Sync")).toBeTruthy();
    expect(screen.getByText("Part 64 Runtime System")).toBeTruthy();
    expect(screen.getByText("Operation Mindfuck :: Inner Jam")).toBeTruthy();
    expect(screen.getByText("force 52% · anchor receipt_river")).toBeTruthy();
    expect(screen.getByText("river flow")).toBeTruthy();
    expect(screen.getByText("witness continuity")).toBeTruthy();
    expect(screen.getByText("ghost pulse")).toBeTruthy();
    expect(screen.getByText("fork tax paid")).toBeTruthy();
    expect(screen.getByRole("img", { name: /web search sync inspiration board/i })).toBeTruthy();
  });
});
