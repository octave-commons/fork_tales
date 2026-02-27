/* @vitest-environment jsdom */

import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useChatCommandHandlers } from "./useChatCommandHandlers";

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function createHookHarness() {
  const emitSystemMessage = vi.fn<(text: string) => void>();
  const emitWitnessChatReply = vi.fn<(
    payload: Record<string, unknown>,
    source: string,
    requestedMusePresenceId?: string,
  ) => void>();
  const buildMuseSurroundingNodes = vi.fn((...args: [string, unknown]) => {
    void args;
    return [{ id: "node:1", kind: "file", label: "demo" }];
  });

  const hook = renderHook(() => useChatCommandHandlers({
    activeMusePresenceId: "witness_thread",
    catalogGeneratedAt: "2026-02-26T00:00:00Z",
    catalogTruthGateBlocked: false,
    simulationTimestamp: "2026-02-26T00:01:00Z",
    simulationTruthGateBlocked: false,
    buildMuseSurroundingNodes: (musePresenceId, workspace) => buildMuseSurroundingNodes(musePresenceId, workspace),
    emitSystemMessage: (text) => emitSystemMessage(text),
    emitWitnessChatReply: (payload, source, requestedMusePresenceId) =>
      emitWitnessChatReply(payload, source, requestedMusePresenceId),
  }));

  return {
    hook,
    emitSystemMessage,
    emitWitnessChatReply,
    buildMuseSurroundingNodes,
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn() as unknown as typeof fetch);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useChatCommandHandlers", () => {
  it("ignores non-command chat text", async () => {
    const { hook } = createHookHarness();
    const consumed = await hook.result.current.handleChatCommand("hello world");

    expect(consumed).toBe(false);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("handles /ledger and emits ledger output", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(mockJsonResponse({ jsonl: "line-1\nline-2" }));
    const { hook, emitSystemMessage } = createHookHarness();

    const consumed = await hook.result.current.handleChatCommand("/ledger first | second ");

    expect(consumed).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(vi.mocked(fetch).mock.calls[0]?.[0]).toBe("/api/eta-mu-ledger");
    const payload = JSON.parse(String((vi.mocked(fetch).mock.calls[0]?.[1] as RequestInit).body));
    expect(payload).toEqual({ utterances: ["first", "second"] });
    expect(emitSystemMessage).toHaveBeenCalledWith("eta/mu ledger\nline-1\nline-2");
  });

  it("handles /say and emits witness + system messages", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(mockJsonResponse({
      muse: { label: "Chaos" },
      turn_id: "turn-7",
      reply: "field acknowledged",
      manifest: {
        explicit_selected: ["a"],
        surround_selected: ["b", "c"],
      },
    }));

    const {
      hook,
      buildMuseSurroundingNodes,
      emitSystemMessage,
      emitWitnessChatReply,
    } = createHookHarness();

    const consumed = await hook.result.current.handleChatCommand("/say chaos drift now");

    expect(consumed).toBe(true);
    expect(buildMuseSurroundingNodes).toHaveBeenCalledWith("chaos", null);
    expect(emitWitnessChatReply).toHaveBeenCalledWith(
      expect.objectContaining({ reply: "field acknowledged" }),
      "command:/say",
      "chaos",
    );
    expect(emitSystemMessage.mock.calls.at(-1)?.[0]).toContain("Chaos / muse turn turn-7");
    expect(emitSystemMessage.mock.calls.at(-1)?.[0]).toContain("explicit=1 surrounding=2");
  });

  it("handles /push-truth --dry-run", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(mockJsonResponse({
      gate: { blocked: true },
      needs: ["receipt", "alignment"],
    }));
    const { hook, emitSystemMessage } = createHookHarness();

    const consumed = await hook.result.current.handleChatCommand("/push-truth --dry-run");

    expect(consumed).toBe(true);
    expect(emitSystemMessage).toHaveBeenCalledWith(
      "push-truth dry-run\ngate=blocked\nneeds=receipt, alignment",
    );
  });

  it("handles /study export", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(mockJsonResponse({
      event: { id: "study-1", label: "release-gate" },
      history: { count: 4 },
    }));
    const { hook, emitSystemMessage } = createHookHarness();

    const consumed = await hook.result.current.handleChatCommand("/study export release-gate");

    expect(consumed).toBe(true);
    expect(vi.mocked(fetch).mock.calls[0]?.[0]).toBe("/api/study/export");
    expect(emitSystemMessage).toHaveBeenCalledWith(
      "study export\nid=study-1\nlabel=release-gate\nhistory=4",
    );
  });

  it("handles /study snapshot via /api/study endpoint", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(mockJsonResponse({
      stability: { score: 0.84, label: "steady" },
      signals: {
        truth_gate_blocked: false,
        blocked_gate_count: 1,
        active_drift_count: 2,
        queue_pending_count: 3,
        queue_event_count: 7,
        council_pending_count: 1,
        council_approved_count: 2,
        council_decision_count: 6,
      },
      council: {
        decisions: [{
          status: "approved",
          id: "decision-9",
          resource: { source_rel_path: "receipts.log" },
        }],
      },
      drift: {
        blocked_gates: [{ reason: "truth_gate" }],
      },
      runtime: {
        receipts_path_within_vault: true,
      },
      warnings: [{ code: "warn", message: "minor" }],
    }));

    const { hook, emitSystemMessage } = createHookHarness();
    const consumed = await hook.result.current.handleChatCommand("/study");

    expect(consumed).toBe(true);
    const last = String(emitSystemMessage.mock.calls.at(-1)?.[0] ?? "");
    expect(last).toContain("study snapshot");
    expect(last).toContain("stability=84% (steady)");
    expect(last).toContain("runtime_receipts_within_vault=true");
  });

  it("falls back to legacy study mode when /api/study is 404", async () => {
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/study?limit=6")) {
        return mockJsonResponse({}, 404);
      }
      if (url.includes("/api/council?limit=6")) {
        return mockJsonResponse({
          council: {
            pending_count: 2,
            approved_count: 3,
            decision_count: 4,
            decisions: [{ status: "pending", id: "d1", resource: { source_rel_path: "doc.md" } }],
          },
        });
      }
      if (url.includes("/api/task/queue")) {
        return mockJsonResponse({
          ok: true,
          queue: {
            pending_count: 3,
            event_count: 11,
          },
        });
      }
      if (url.includes("/api/drift/scan")) {
        return mockJsonResponse({
          active_drifts: [{ id: "drift-1" }],
          blocked_gates: [{ reason: "gates_of_truth" }],
        });
      }
      return mockJsonResponse({}, 500);
    });

    const { hook, emitSystemMessage } = createHookHarness();
    const consumed = await hook.result.current.handleChatCommand("/study now");

    expect(consumed).toBe(true);
    const last = String(emitSystemMessage.mock.calls.at(-1)?.[0] ?? "");
    expect(last).toContain("runtime_receipts_within_vault=(unknown:legacy-mode)");
    expect(last).toContain("blocked_gates=1 active_drifts=1");
  });
});
