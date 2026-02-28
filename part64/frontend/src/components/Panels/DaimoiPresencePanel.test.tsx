/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DaimoiPresencePanel } from "./DaimoiPresencePanel";
import type { Catalog, SimulationState } from "../../types";

function createCatalogFixture(): Catalog {
  return {
    entity_manifest: [
      { id: "witness_thread", en: "Witness Thread", hue: 200 },
      { id: "gates_of_truth", en: "Gates of Truth", hue: 120 },
    ],
  } as unknown as Catalog;
}

function createSimulationFixture(): SimulationState {
  return {
    presence_dynamics: {
      field_particles: [
        {
          id: "daimoi-1",
          owner_presence_id: "witness_thread",
          x: 0.2,
          y: 0.4,
          message_probability: 0.8,
          route_probability: 0.6,
          drift_score: 0.4,
          action_probabilities: { deflect: 0.5, diffuse: 0.2 },
          job_probabilities: { route_message: 0.7, stabilize: 0.3 },
        },
        {
          id: "daimoi-2",
          owner_presence_id: "witness_thread",
          x: 0.3,
          y: 0.6,
          message_probability: 0.5,
          route_probability: 0.4,
          drift_score: 0.2,
          action_probabilities: { deflect: 0.3, diffuse: 0.1 },
          job_probabilities: { route_message: 0.4 },
        },
        {
          id: "daimoi-3",
          owner_presence_id: "gates_of_truth",
          x: 0.8,
          y: 0.2,
          message_probability: 0.2,
          route_probability: 0.3,
          drift_score: 0.1,
          action_probabilities: { deflect: 0.1, diffuse: 0.4 },
          job_probabilities: { audit: 0.6 },
        },
      ],
      user_query_transient_edges: [
        { id: "edge-t1", target: "nexus", query: "alpha", hits: 2, life: 0.5, strength: 0.7 },
      ],
      user_query_promoted_edges: [
        { id: "edge-p1", target: "witness_thread", query: "beta", hits: 4, life: 0.9, strength: 0.8 },
      ],
      daimoi_probabilistic: {
        active: 3,
        collisions: 1,
        mean_message_probability: 0.51,
        mean_package_entropy: 0.777,
        clump_score: 0.62,
        anti_clump_drive: 0.34,
        snr: 0.9,
        snr_band: { min: 0.4, max: 1.2 },
        anti_clump: {
          target: 0.3,
          metrics: {
            nn_term: 0.2,
            fano_factor: 1.1,
            motion_noise: 0.005,
            semantic_noise: 0.3,
          },
          scales: {
            spawn: 1.1,
            anchor: 1.0,
            semantic: 0.8,
            edge: 1.2,
            tangent: 0.7,
          },
        },
        job_triggers: {
          route_message: 9,
          audit: 3,
          stabilize: 2,
        },
        deflects: 5,
        diffuses: 3,
        handoffs: 1,
        deliveries: 7,
      },
    },
  } as unknown as SimulationState;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("DaimoiPresencePanel", () => {
  it("renders waiting state when simulation is missing", () => {
    render(
      <DaimoiPresencePanel
        catalog={null}
        simulation={null}
        onFocusAnchor={vi.fn()}
        onEmitUserInput={vi.fn()}
      />,
    );

    expect(screen.getByText("Waiting for simulation payload...")).toBeTruthy();
  });

  it("renders anti-clump, edge, and presence summaries", async () => {
    render(
      <DaimoiPresencePanel
        catalog={createCatalogFixture()}
        simulation={createSimulationFixture()}
        onFocusAnchor={vi.fn()}
        onEmitUserInput={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Anti-clump controller")).toBeTruthy();
      expect(screen.getByText("clumped")).toBeTruthy();
      expect(screen.getByText(/target 0.300 \| score 0.620 \| drive \+0.340/)).toBeTruthy();
      expect(screen.getByText(/transient edges:/)).toBeTruthy();
      expect(screen.getByText("nexus · h2 · life 50%")).toBeTruthy();
      expect(screen.getByText("witness_thread · h4 · s 80%")).toBeTruthy();
      expect(screen.getByText(/Witness Thread/)).toBeTruthy();
      expect(screen.getAllByText(/route_message/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("daimoi-1")).toBeTruthy();
    });
  });

  it("emits search queries and focus callbacks", async () => {
    const onFocusAnchor = vi.fn();
    const onEmitUserInput = vi.fn();

    render(
      <DaimoiPresencePanel
        catalog={createCatalogFixture()}
        simulation={createSimulationFixture()}
        onFocusAnchor={onFocusAnchor}
        onEmitUserInput={onEmitUserInput}
      />,
    );

    fireEvent.click(screen.getByText("emit"));
    expect(screen.getByText("Type a query first.")).toBeTruthy();

    fireEvent.change(screen.getByPlaceholderText("search query -> emits query daimoi + variants"), {
      target: { value: "trace anomaly" },
    });
    fireEvent.change(screen.getByPlaceholderText("target: nexus or presence_id"), {
      target: { value: "gates_of_truth" },
    });

    fireEvent.click(screen.getByText("emit"));

    await waitFor(() => {
      expect(onEmitUserInput).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: "search_query",
          target: "gates_of_truth",
          message: "trace anomaly",
          embedDaimoi: true,
        }),
      );
      expect(screen.getByText("Query emitted: trace anomaly")).toBeTruthy();
    });

    const presenceLabel = screen.getAllByText("Witness Thread").find((element) => element.closest("button"));
    const presenceButton = presenceLabel?.closest("button");
    if (!(presenceButton instanceof HTMLButtonElement)) {
      throw new Error("presence focus button not found");
    }
    fireEvent.click(presenceButton);

    await waitFor(() => {
      expect(onFocusAnchor).toHaveBeenCalledWith(expect.objectContaining({
        kind: "node",
        id: "witness_thread",
      }));
      expect(screen.getByText("focus locked ->")).toBeTruthy();
      expect(screen.getByText("presence Witness Thread")).toBeTruthy();
    });

    const daimoiLabel = screen.getAllByText("daimoi-1").find((element) => element.closest("button"));
    const daimoiButton = daimoiLabel?.closest("button");
    if (!(daimoiButton instanceof HTMLButtonElement)) {
      throw new Error("daimoi focus button not found");
    }
    fireEvent.click(daimoiButton);

    await waitFor(() => {
      const lastCallArg = onFocusAnchor.mock.calls.at(-1)?.[0];
      expect(lastCallArg).toMatchObject({
        kind: "node",
        id: "daimoi-1",
      });
      expect(screen.getByText("daimoi daimoi-1")).toBeTruthy();
    });
  });
});
