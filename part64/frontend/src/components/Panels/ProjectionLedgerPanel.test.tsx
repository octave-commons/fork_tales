/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ProjectionLedgerPanel } from "./ProjectionLedgerPanel";
import type { UIProjectionBundle } from "../../types";

function makeProjection(): UIProjectionBundle {
  return {
    perspective: "ops",
    default_perspective: "ops",
    coherence: {
      tension: 0.41,
      drift: 0.12,
      entropy: 0.3,
    },
    queue: {
      pending_count: 2,
      event_count: 11,
    },
    elements: [
      {
        id: "panel.alpha",
        title: "Alpha Panel",
        presence: "witness_thread",
        lane: "north",
      },
      {
        id: "panel.beta",
        title: "Beta Panel",
        presence: "chaos",
        lane: "south",
      },
    ],
    states: [
      {
        element_id: "panel.alpha",
        priority: 0.7,
        mass: 0.6,
        area: 0.3,
        pulse: 0.5,
        explain: {
          reason_en: "Alpha reasoning",
          field_signal: 0.4,
          presence_signal: 0.5,
          field_bindings: {
            river: 0.8,
          },
        },
      },
      {
        element_id: "panel.beta",
        priority: 0.9,
        mass: 0.8,
        area: 0.5,
        pulse: 0.3,
        explain: {
          reason_en: "Beta reasoning",
          field_signal: 0.9,
          presence_signal: 0.7,
          field_bindings: {
            river: 0.2,
          },
        },
      },
    ],
    field_schemas: [
      {
        field: "river",
        name: "River",
        interpretation: {
          en: "Flow tendency",
        },
      },
    ],
    layout: {
      rects: {
        "panel.alpha": { x: 0.1, y: 0.2, w: 0.3, h: 0.4 },
        "panel.beta": { x: 0.4, y: 0.1, w: 0.2, h: 0.3 },
      },
    },
  } as unknown as UIProjectionBundle;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ProjectionLedgerPanel", () => {
  it("shows fallback text when projection feed is missing", () => {
    render(<ProjectionLedgerPanel projection={null} />);
    expect(screen.getByText("Projection feed is not available yet.")).toBeTruthy();
  });

  it("renders projection diagnostics and supports focus switching", () => {
    render(<ProjectionLedgerPanel projection={makeProjection()} />);

    expect(screen.getByText("Perspective")).toBeTruthy();
    expect(screen.getByText("ops")).toBeTruthy();
    expect(screen.getByText("2 / 2 boxes routed")).toBeTruthy();
    expect(screen.getByText("Field Leadership")).toBeTruthy();
    expect(
      screen.getByText((_, node) => String(node?.textContent ?? "").trim() === "north lane"),
    ).toBeTruthy();
    expect(
      screen.getByText((_, node) => String(node?.textContent ?? "").trim() === "south lane"),
    ).toBeTruthy();

    const alphaButtons = screen.getAllByRole("button", { name: "Alpha Panel" });
    fireEvent.click(alphaButtons[0]);

    expect(screen.getByText("Alpha reasoning")).toBeTruthy();
    expect(
      screen
        .getAllByText((_, node) => String(node?.textContent ?? "").includes("id panel.alpha"))
        .length,
    ).toBeGreaterThan(0);
  });
});
