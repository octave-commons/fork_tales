/* @vitest-environment jsdom */

import { waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Autopilot, requestUserInput, type AskPayload, type AutopilotHooks, type PlannedAction } from "./autopilot";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  document.body.innerHTML = "";
});

describe("requestUserInput", () => {
  it("scrolls chat, focuses input, and dispatches ask + toast events", () => {
    const panel = document.createElement("section");
    panel.id = "chat-panel";
    panel.scrollIntoView = vi.fn();
    document.body.appendChild(panel);

    const input = document.createElement("textarea");
    input.id = "chat-input";
    input.focus = vi.fn();
    document.body.appendChild(input);

    const askEvents: AskPayload[] = [];
    const toasts: Array<{ title?: string; body?: string }> = [];
    const askListener = (event: Event) => {
      const payload = (event as CustomEvent<AskPayload>).detail;
      askEvents.push(payload);
    };
    const toastListener = (event: Event) => {
      const payload = (event as CustomEvent<{ title?: string; body?: string }>).detail;
      toasts.push(payload);
    };

    window.addEventListener("autopilot:ask", askListener as EventListener);
    window.addEventListener("ui:toast", toastListener as EventListener);

    const ask: AskPayload = {
      reason: "permission required",
      need: "approve write",
      gate: "permission",
      urgency: "high",
    };
    requestUserInput(ask);

    expect(panel.scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "end" });
    expect(input.focus).toHaveBeenCalledTimes(1);
    expect(askEvents).toEqual([ask]);
    expect(toasts).toEqual([{ title: "Need your input", body: "approve write" }]);

    window.removeEventListener("autopilot:ask", askListener as EventListener);
    window.removeEventListener("ui:toast", toastListener as EventListener);
  });
});

describe("Autopilot", () => {
  it("stops and asks for input when gate blocks action", async () => {
    const blockedAsk: AskPayload = {
      reason: "risk too high",
      need: "confirm dry run",
      gate: "risk",
      urgency: "med",
    };

    const action: PlannedAction<{ phase: string }> = {
      id: "push-truth",
      label: "Push Truth",
      goal: "submit",
      risk: 0.82,
      cost: 0.2,
      requiredPerms: ["truth:push"],
      run: vi.fn(async () => ({ ok: true, summary: "should not run" })),
    };

    const onAsk = vi.fn();
    const onActionEvent = vi.fn();
    const hooks: AutopilotHooks<{ phase: string }> = {
      sense: async () => ({ phase: "before" }),
      hypothesize: async () => ({ goal: "submit", confidence: 0.61 }),
      plan: async () => action,
      gate: () => ({ ok: false, ask: blockedAsk }),
      onAsk,
      onActionEvent,
      tickDelayMs: 10,
    };

    const pilot = new Autopilot(hooks);
    pilot.start();

    await waitFor(() => {
      expect(onAsk).toHaveBeenCalledWith(blockedAsk);
      expect(onActionEvent).toHaveBeenCalledWith(expect.objectContaining({
        actionId: "push-truth",
        result: "skipped",
        gate: "risk",
      }));
    });

    expect(action.run).not.toHaveBeenCalled();
    expect(pilot.isRunning()).toBe(false);
    expect(pilot.isWaitingForInput()).toBe(true);
  });

  it("runs action and reports failed when verification fails", async () => {
    vi.useFakeTimers();

    const action: PlannedAction<{ marker: string }> = {
      id: "sync-ledger",
      label: "Sync Ledger",
      goal: "sync",
      risk: 0.2,
      cost: 0.3,
      requiredPerms: ["ledger:write"],
      run: vi.fn(async () => ({ ok: true, summary: "sync complete" })),
      verify: vi.fn(async () => false),
    };

    const sense = vi
      .fn<() => Promise<{ marker: string }>>()
      .mockResolvedValueOnce({ marker: "before" })
      .mockResolvedValueOnce({ marker: "after" });

    const pilotRef: { current: Autopilot<{ marker: string }> | null } = { current: null };
    const onActionEvent = vi.fn(() => {
      pilotRef.current?.stop();
    });

    const hooks: AutopilotHooks<{ marker: string }> = {
      sense,
      hypothesize: async () => ({ goal: "sync", confidence: 0.93 }),
      plan: async () => action,
      gate: () => ({ ok: true, action }),
      onActionEvent,
      tickDelayMs: 10,
    };

    const pilot = new Autopilot(hooks);
    pilotRef.current = pilot;
    pilot.start();
    await vi.runAllTimersAsync();

    expect(action.run).toHaveBeenCalledTimes(1);
    expect(action.verify).toHaveBeenCalledWith(
      { marker: "before" },
      { marker: "after" },
      { ok: true, summary: "sync complete" },
    );
    expect(onActionEvent).toHaveBeenCalledWith(expect.objectContaining({
      actionId: "sync-ledger",
      result: "failed",
      summary: "sync complete",
      perms: ["ledger:write"],
    }));
  });

  it("reports tick errors via onTickError", async () => {
    vi.useFakeTimers();

    const error = new Error("sense failed");
    const pilotRef: { current: Autopilot<{ ok: boolean }> | null } = { current: null };
    const onTickError = vi.fn((received: unknown) => {
      expect(received).toBe(error);
      pilotRef.current?.stop();
    });

    const hooks: AutopilotHooks<{ ok: boolean }> = {
      sense: async () => {
        throw error;
      },
      hypothesize: async () => ({ goal: "noop", confidence: 0.5 }),
      plan: async () => ({
        id: "noop",
        label: "No-op",
        goal: "noop",
        risk: 0,
        cost: 0,
        requiredPerms: [],
        run: async () => ({ ok: true, summary: "ok" }),
      }),
      gate: (_ctx, _hyp, action) => ({ ok: true, action }),
      onTickError,
      tickDelayMs: 10,
    };

    const pilot = new Autopilot(hooks);
    pilotRef.current = pilot;
    pilot.start();
    await vi.runAllTimersAsync();

    expect(onTickError).toHaveBeenCalledTimes(1);
  });
});
