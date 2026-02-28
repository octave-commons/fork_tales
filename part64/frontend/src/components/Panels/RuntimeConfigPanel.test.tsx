/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RuntimeConfigPanel } from "./RuntimeConfigPanel";

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function createConfigPayload() {
  return {
    ok: true,
    record: "cfg-001",
    runtime_config_version: 7,
    generated_at: "2026-02-20T12:00:00.000Z",
    available_modules: ["physics"],
    module_count: 1,
    constant_count: 2,
    numeric_leaf_count: 4,
    modules: {
      physics: {
        constants: {
          DRIFT_FRICTION: 0.4,
          DAMPING: {
            base: 0.6,
            lanes: [0.1, 0.2],
          },
        },
        constant_count: 2,
        numeric_leaf_count: 4,
      },
    },
  };
}

function createBootstrapPayload() {
  return {
    ok: true,
    record: "bootstrap-001",
    generated_at: "2026-02-20T12:00:00.000Z",
    job: {
      status: "completed",
      job_id: "job-123",
      phase: "catalog_ready",
      phase_started_at: "2026-02-20T11:59:30.000Z",
      updated_at: "2026-02-20T12:00:00.000Z",
    },
    report: {
      generated_at: "2026-02-20T12:00:00.000Z",
      selection: {
        graph_surface: "hybrid",
        projection_reason: "smoke check",
        embed_layer_count: 3,
        active_embed_layer_count: 2,
      },
      compression: {
        before_edges: 200,
        after_edges: 150,
        collapsed_edges: 50,
      },
      graph_diff: {
        truth_file_node_count: 40,
        view_file_node_count: 36,
        truth_file_nodes_missing_from_view_count: 1,
        truth_file_nodes_missing_from_view: [
          {
            id: "f-1",
            path: "docs/guide.md",
            reason: "projection_hidden",
            projection_group_refs: [{ group_id: "g-1" }],
          },
        ],
        view_projection_overflow_node_count: 2,
        projection_group_count: 4,
        projection_surface_visible_group_count: 3,
        projection_hidden_group_count: 1,
        ingested_item_count: 50,
        ingested_items_missing_from_truth_graph_count: 1,
        ingested_items_missing_from_truth_graph: [
          {
            path: "tmp/orphan.md",
            reason: "orphan",
            kind: "doc",
          },
        ],
        compaction_mode: "edge_cap",
        view_graph_reconstructable_from_truth_graph: true,
        notes: ["graph diff stable"],
      },
      phase_ms: {
        catalog: 1200,
        simulation: 2400,
        cache_store: 350,
      },
    },
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("RuntimeConfigPanel", () => {
  it("loads config, edits a leaf, and applies edited values", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method || "GET").toUpperCase();
      if (url.includes("/api/config/update") && method === "POST") {
        return mockJsonResponse({ ok: true, current: 0.45 });
      }
      if (url.includes("/api/config")) {
        return mockJsonResponse(createConfigPayload());
      }
      if (url.includes("/api/simulation/bootstrap")) {
        return mockJsonResponse(createBootstrapPayload());
      }
      return mockJsonResponse({ ok: true });
    });

    render(<RuntimeConfigPanel />);

    await waitFor(() => {
      expect(screen.getByText("Runtime Config Interface")).toBeTruthy();
      expect(screen.getAllByText("physics").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("graph diff stable")).toBeTruthy();
    });

    const frictionLabels = screen.getAllByText("DRIFT_FRICTION");
    fireEvent.click(frictionLabels[0]);

    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: "x2" }).length).toBeGreaterThanOrEqual(1);
    });
    fireEvent.click(screen.getAllByRole("button", { name: "x2" })[0]);

    await waitFor(() => {
      expect(screen.getByText("Apply Edited (1)")).toBeTruthy();
    });

    fireEvent.click(screen.getByText("Apply Edited (1)"));

    await waitFor(() => {
      expect(screen.getByText("applied 1 edited values")).toBeTruthy();
    });

    const updateCall = fetchSpy.mock.calls.find(([url, init]) => {
      return String(url).includes("/api/config/update") && String((init as RequestInit | undefined)?.method || "GET").toUpperCase() === "POST";
    });
    expect(updateCall).toBeTruthy();
    const updateBody = JSON.parse(String((updateCall?.[1] as RequestInit | undefined)?.body || "{}"));
    expect(updateBody.module).toBe("physics");
    expect(typeof updateBody.key).toBe("string");
    expect(typeof updateBody.value).toBe("number");
  });

  it("queues bootstrap and reports stream probe failures", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method || "GET").toUpperCase();
      if (url.includes("/api/simulation/bootstrap") && method === "POST") {
        return mockJsonResponse({
          ok: true,
          status: "queued",
          job: {
            status: "queued",
            job_id: "job-queued-99",
          },
        });
      }
      if (url.includes("/api/config")) {
        return mockJsonResponse(createConfigPayload());
      }
      if (url.includes("/api/simulation/bootstrap")) {
        return mockJsonResponse(createBootstrapPayload());
      }
      if (url.includes("/api/catalog/stream")) {
        return mockJsonResponse({ ok: false, error: "stream down" }, 500);
      }
      return mockJsonResponse({ ok: true });
    });

    render(<RuntimeConfigPanel />);

    await waitFor(() => {
      expect(screen.getByText("queue bootstrap")).toBeTruthy();
    });

    fireEvent.click(screen.getByText("queue bootstrap"));

    await waitFor(() => {
      expect(screen.getByText("bootstrap queued · job-queued-99")).toBeTruthy();
    });

    fireEvent.click(screen.getByText("start stream"));

    await waitFor(() => {
      expect(screen.getByText("catalog stream failed: catalog stream failed (500)")).toBeTruthy();
    });

    const queueCall = fetchSpy.mock.calls.find(([url, init]) => {
      return String(url).includes("/api/simulation/bootstrap") && String((init as RequestInit | undefined)?.method || "GET").toUpperCase() === "POST";
    });
    expect(queueCall).toBeTruthy();
    const queueBody = JSON.parse(String((queueCall?.[1] as RequestInit | undefined)?.body || "{}"));
    expect(queueBody).toMatchObject({
      perspective: "hybrid",
      include_simulation: false,
      wait: false,
    });

    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/catalog/stream")),
    ).toBe(true);
  });
});
