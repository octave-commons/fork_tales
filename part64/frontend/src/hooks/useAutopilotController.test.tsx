/* @vitest-environment jsdom */

import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { autopilotInstances, MockAutopilot } = vi.hoisted(() => {
  interface SenseContextLike {
    isConnected: boolean;
    blockedGateCount: number;
    activeDriftCount: number;
    queuePendingCount: number;
    truthGateBlocked: boolean;
    health: "green" | "yellow" | "red";
    healthReasons: string[];
    permissions: Record<string, boolean>;
  }

  interface IntentHypothesisLike {
    goal: string;
    confidence: number;
  }

  interface PlannedActionLike {
    id: string;
    label: string;
    goal: string;
    risk: number;
    cost: number;
    requiredPerms: string[];
    run: () => Promise<{ ok: boolean; summary: string }>;
  }

  interface AskPayloadLike {
    gate?: string;
    reason: string;
    need: string;
    context?: Record<string, unknown>;
  }

  interface ActionEventLike {
    ts: string;
    actionId: string;
    intent: string;
    confidence: number;
    risk: number;
    perms: string[];
    result: "ok" | "failed" | "skipped";
    summary: string;
  }

  interface HookSetLike {
    sense: () => Promise<SenseContextLike>;
    hypothesize: (ctx: SenseContextLike) => Promise<IntentHypothesisLike>;
    plan: (ctx: SenseContextLike, goal: string) => Promise<PlannedActionLike>;
    gate: (
      ctx: SenseContextLike,
      hypothesis: IntentHypothesisLike,
      action: PlannedActionLike,
    ) => { ok: true; action: PlannedActionLike } | { ok: false; ask: AskPayloadLike };
    onActionEvent?: (event: ActionEventLike) => void;
    onAsk?: (ask: AskPayloadLike) => void;
    onTickError?: (error: unknown) => void;
  }

  const instances: Array<{
    hooks: HookSetLike;
    start: ReturnType<typeof vi.fn>;
    stop: ReturnType<typeof vi.fn>;
    resume: ReturnType<typeof vi.fn>;
    isRunning: ReturnType<typeof vi.fn>;
    isWaitingForInput: ReturnType<typeof vi.fn>;
    setWaitingForInput: (next: boolean) => void;
  }> = [];

  class HoistedMockAutopilot {
    readonly hooks: HookSetLike;

    private waitingForInput = false;

    private running = false;

    start = vi.fn(() => {
      this.running = true;
    });

    stop = vi.fn(() => {
      this.running = false;
      this.waitingForInput = false;
    });

    resume = vi.fn(() => {
      this.running = true;
      this.waitingForInput = false;
    });

    isRunning = vi.fn(() => this.running);

    isWaitingForInput = vi.fn(() => this.waitingForInput);

    constructor(hooks: HookSetLike) {
      this.hooks = hooks;
      instances.push(this as unknown as typeof instances[number]);
    }

    setWaitingForInput(next: boolean): void {
      this.waitingForInput = next;
    }
  }

  return {
    autopilotInstances: instances,
    MockAutopilot: HoistedMockAutopilot,
  };
});

vi.mock("../autopilot", () => ({
  Autopilot: MockAutopilot,
}));

import { useAutopilotController } from "./useAutopilotController";

function latestAutopilotInstance() {
  const instance = autopilotInstances.at(-1);
  if (!instance) {
    throw new Error("expected mock autopilot instance");
  }
  return instance;
}

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

beforeEach(() => {
  autopilotInstances.length = 0;
  vi.stubGlobal("fetch", vi.fn(async () => mockJsonResponse({ ok: false }, 503)) as unknown as typeof fetch);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useAutopilotController", () => {
  it("creates autopilot controller and starts autopilot by default", () => {
    const emitSystemMessage = vi.fn();
    renderHook(() => useAutopilotController({
      catalog: null,
      simulation: null,
      isConnected: true,
      emitSystemMessage,
    }));

    const instance = latestAutopilotInstance();
    expect(instance.start).toHaveBeenCalledTimes(1);
  });

  it("derives context, plans clear-gates action, and runs dry-run command", async () => {
    const emitSystemMessage = vi.fn();
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/study?limit=4")) {
        return mockJsonResponse({
          signals: {
            blocked_gate_count: 2,
            active_drift_count: 5,
            queue_pending_count: 9,
            truth_gate_blocked: true,
          },
        });
      }
      if (url.includes("/api/push-truth/dry-run")) {
        return mockJsonResponse({
          gate: { blocked: true },
          needs: ["receipt", "alignment"],
        });
      }
      return mockJsonResponse({ ok: false }, 503);
    });

    renderHook(() => useAutopilotController({
      catalog: null,
      simulation: null,
      isConnected: true,
      emitSystemMessage,
    }));
    const instance = latestAutopilotInstance();

    const context = await instance.hooks.sense();
    expect(context).toMatchObject({
      blockedGateCount: 2,
      activeDriftCount: 5,
      queuePendingCount: 9,
      truthGateBlocked: true,
      health: "yellow",
    });

    const hypothesis = await instance.hooks.hypothesize(context);
    expect(hypothesis.goal).toBe("clear-gates");

    const action = await instance.hooks.plan(context, hypothesis.goal);
    expect(action.id).toBe("autopilot.push-truth-dry-run");
    expect(action.requiredPerms).toEqual(["runtime.read", "truth.push.dry-run"]);

    const permissionGate = instance.hooks.gate(context, hypothesis, action);
    expect(permissionGate.ok).toBe(false);
    if (permissionGate.ok) {
      throw new Error("expected permission gate block");
    }
    expect(permissionGate.ask.gate).toBe("permission");

    const approvedContext = {
      ...context,
      permissions: {
        ...context.permissions,
        "truth.push.dry-run": true,
      },
    };
    const riskGate = instance.hooks.gate(approvedContext, hypothesis, action);
    expect(riskGate.ok).toBe(false);
    if (riskGate.ok) {
      throw new Error("expected risk gate block");
    }
    expect(riskGate.ask.gate).toBe("risk");

    const fullyApprovedContext = {
      ...approvedContext,
      permissions: {
        ...approvedContext.permissions,
        "risk:autopilot.push-truth-dry-run": true,
      },
    };
    const runGate = instance.hooks.gate(fullyApprovedContext, hypothesis, action);
    expect(runGate.ok).toBe(true);

    const result = await action.run();
    expect(result).toMatchObject({ ok: true, summary: "push-truth dry-run gate=blocked" });
    expect(emitSystemMessage).toHaveBeenCalledWith(
      "autopilot /push-truth --dry-run\ngate=blocked\nneeds=receipt, alignment",
    );
  });

  it("handles ask responses for permission grants, directives, and pause", async () => {
    const emitSystemMessage = vi.fn();
    const { result } = renderHook(() => useAutopilotController({
      catalog: null,
      simulation: null,
      isConnected: true,
      emitSystemMessage,
    }));
    const instance = latestAutopilotInstance();

    act(() => {
      instance.setWaitingForInput(true);
      instance.hooks.onAsk?.({
        gate: "permission",
        reason: "need perm",
        need: "Grant truth.push.dry-run so I can continue?",
        context: { permission: "truth.push.dry-run" },
      });
    });

    const granted = result.current.handleAutopilotUserInput("grant permission");
    expect(granted).toBe(true);
    expect(instance.resume).toHaveBeenCalledTimes(1);
    expect(emitSystemMessage).toHaveBeenCalledWith("autopilot permission granted: truth.push.dry-run");

    act(() => {
      instance.setWaitingForInput(true);
      instance.hooks.onAsk?.({
        gate: "confidence",
        reason: "low confidence",
        need: "Pick next move",
        context: {},
      });
    });

    const directed = result.current.handleAutopilotUserInput("run drift scan now");
    expect(directed).toBe(true);

    const inferred = await instance.hooks.hypothesize({
      isConnected: true,
      blockedGateCount: 0,
      activeDriftCount: 0,
      queuePendingCount: 0,
      truthGateBlocked: false,
      health: "green",
      healthReasons: [],
      permissions: { "runtime.read": true, "truth.push.dry-run": true },
    });
    expect(inferred).toMatchObject({ goal: "scan-drift", confidence: 0.99 });

    act(() => {
      instance.setWaitingForInput(true);
      instance.hooks.onAsk?.({
        gate: "health",
        reason: "runtime unstable",
        need: "pause autopilot?",
      });
    });

    const paused = result.current.handleAutopilotUserInput("pause autopilot");
    expect(paused).toBe(true);
    expect(instance.stop).toHaveBeenCalled();
    await waitFor(() => {
      expect(result.current.autopilotEnabled).toBe(false);
      expect(result.current.autopilotStatus).toBe("stopped");
      expect(result.current.autopilotSummary).toBe("paused by user");
    });
    expect(emitSystemMessage).toHaveBeenCalledWith("autopilot paused by user");
  });

  it("records autopilot events and tick errors in summary", async () => {
    const emitSystemMessage = vi.fn();
    const { result } = renderHook(() => useAutopilotController({
      catalog: null,
      simulation: null,
      isConnected: true,
      emitSystemMessage,
    }));
    const instance = latestAutopilotInstance();

    act(() => {
      instance.hooks.onActionEvent?.({
        ts: "2026-02-26T20:00:00Z",
        actionId: "autopilot.study-snapshot",
        intent: "maintain-observability",
        confidence: 0.72,
        risk: 0.19,
        perms: ["runtime.read"],
        result: "ok",
        summary: "study snapshot sampled",
      });
    });

    expect(result.current.autopilotEvents).toHaveLength(1);
    expect(result.current.autopilotSummary).toBe("maintain-observability: study snapshot sampled");

    act(() => {
      instance.hooks.onTickError?.(new Error("tick failed"));
    });
    expect(result.current.autopilotSummary).toBe("tick error; waiting for next cycle");
  });
});
