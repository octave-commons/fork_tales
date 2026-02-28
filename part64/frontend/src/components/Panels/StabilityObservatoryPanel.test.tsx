/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { StabilityObservatoryPanel } from "./StabilityObservatoryPanel";

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("StabilityObservatoryPanel", () => {
  it("renders study-v1 snapshot metrics", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/study?limit=10")) {
        return mockJsonResponse({
          generated_at: "2026-02-20T12:00:00.000Z",
          signals: {
            blocked_gate_count: 2,
            active_drift_count: 3,
            queue_pending_count: 4,
            council_pending_count: 1,
            truth_gate_blocked: true,
            resource_hot_count: 2,
            resource_log_error_ratio: 0.125,
          },
          stability: {
            score: 0.82,
            label: "stable",
          },
          runtime: {
            receipts_path: "receipts.log",
            receipts_path_within_vault: true,
            resource: {
              devices: {
                cpu: { utilization: 67 },
                npu0: {
                  status: "watch",
                  utilization: 58,
                  queue_depth: 3,
                  temperature: 58.2,
                  device: "NPU0",
                },
              },
              auto_backend: {
                embeddings_order: ["npu0", "cpu"],
                text_order: ["cpu"],
              },
            },
          },
          drift: {
            blocked_gates: [{ target: "truth", reason: "missing_receipt" }],
            active_drifts: [
              { severity: "high" },
              { severity: "medium" },
              { severity: "low" },
            ],
            open_questions: { unresolved_count: 2 },
          },
          council: {
            pending_count: 1,
            decisions: [],
          },
          queue: {
            pending_count: 4,
            dedupe_keys: 2,
            event_count: 9,
            pending: [],
          },
          warnings: [{ code: "gate.blocked", severity: "high", message: "truth: missing_receipt" }],
        });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<StabilityObservatoryPanel catalog={null} simulation={null} />);

    await waitFor(() => {
      expect(screen.getByText("Stability Observatory")).toBeTruthy();
      expect(screen.getByText("82% (stable)")).toBeTruthy();
      expect(screen.getByText("blocked")).toBeTruthy();
      expect(screen.getByText("high 1 | medium 1 | low 1")).toBeTruthy();
      expect(screen.getByText(/58.2C/)).toBeTruthy();
      expect(screen.getByText(/NPU0/)).toBeTruthy();
      expect(screen.getByText("npu0 -> cpu")).toBeTruthy();
    });
  });

  it("falls back to legacy endpoints when study API is unavailable", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/study?limit=10")) {
        return mockJsonResponse({ error: "not found" }, 404);
      }
      if (url.includes("/api/council?limit=10")) {
        return mockJsonResponse({
          ok: true,
          council: {
            pending_count: 2,
            decisions: [],
          },
        });
      }
      if (url.includes("/api/task/queue")) {
        return mockJsonResponse({
          ok: true,
          queue: {
            pending_count: 3,
            dedupe_keys: 5,
            event_count: 11,
            pending: [{ id: "task-1", kind: "study.refresh" }],
          },
        });
      }
      if (url.includes("/api/drift/scan")) {
        return mockJsonResponse({
          blocked_gates: [{ target: "truth", reason: "pending_decision" }],
          active_drifts: [{ severity: "high" }],
          open_questions: { unresolved_count: 1 },
        });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<StabilityObservatoryPanel catalog={null} simulation={null} />);

    await waitFor(() => {
      expect(screen.getByText("legacy")).toBeTruthy();
      expect(screen.getByText(/task-1/)).toBeTruthy();
      expect(screen.getByText(/pending_decision/)).toBeTruthy();
    });

    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/council?limit=10")),
    ).toBe(true);
    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/task/queue")),
    ).toBe(true);
    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/drift/scan")),
    ).toBe(true);
  });

  it("exports evidence and refreshes snapshot", async () => {
    let studyCalls = 0;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/study?limit=10")) {
        studyCalls += 1;
        return mockJsonResponse({
          generated_at: "2026-02-20T12:00:00.000Z",
          signals: {
            blocked_gate_count: 0,
            active_drift_count: 0,
            queue_pending_count: 0,
            council_pending_count: 0,
          },
          council: { pending_count: 0, decisions: [] },
          queue: { pending_count: 0, dedupe_keys: 0, event_count: 0, pending: [] },
          drift: { blocked_gates: [], active_drifts: [], open_questions: { unresolved_count: 0 } },
          warnings: [],
        });
      }
      if (url.includes("/api/study/export")) {
        return mockJsonResponse({
          ok: true,
          event: { id: "study-evt-123" },
          history: { count: 7 },
        });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<StabilityObservatoryPanel catalog={null} simulation={null} />);

    await waitFor(() => {
      expect(screen.getByText("Export Evidence")).toBeTruthy();
    });

    fireEvent.click(screen.getByText("Export Evidence"));

    await waitFor(() => {
      expect(screen.getByText("evidence exported study-evt-123 (history=7)")).toBeTruthy();
    });

    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/study/export")),
    ).toBe(true);
    expect(studyCalls).toBeGreaterThanOrEqual(2);
  });
});
