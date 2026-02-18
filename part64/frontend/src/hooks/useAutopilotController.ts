import { useCallback, useEffect, useRef, useState } from "react";
import {
  Autopilot,
  type AskPayload,
  type AutopilotActionEvent,
  type AutopilotActionResult,
  type GateVerdict,
  type IntentHypothesis,
  type PlannedAction,
} from "../autopilot";
import { runtimeBaseUrl } from "../runtime/endpoints";
import type { Catalog, DriftScanPayload, SimulationState, StudySnapshotPayload } from "../types";

type AutopilotHealth = "green" | "yellow" | "red";

interface AutopilotSenseContext {
  isConnected: boolean;
  blockedGateCount: number;
  activeDriftCount: number;
  queuePendingCount: number;
  truthGateBlocked: boolean;
  health: AutopilotHealth;
  healthReasons: string[];
  permissions: Record<string, boolean>;
}

interface UseAutopilotControllerArgs {
  catalog: Catalog | null;
  simulation: SimulationState | null;
  isConnected: boolean;
  emitSystemMessage: (text: string) => void;
}

interface UseAutopilotControllerResult {
  autopilotEnabled: boolean;
  autopilotStatus: "running" | "waiting" | "stopped";
  autopilotSummary: string;
  autopilotEvents: AutopilotActionEvent[];
  handleAutopilotUserInput: (text: string) => boolean;
  toggleAutopilot: () => void;
}

const AUTOPILOT_C_MIN = 0.72;
const AUTOPILOT_R_MAX = 0.45;

const DEFAULT_AUTOPILOT_PERMISSIONS: Record<string, boolean> = {
  "runtime.read": true,
  "truth.push.dry-run": false,
};

function clampValue(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function directiveToGoal(input: string): string {
  const normalized = input.trim().toLowerCase();
  if (normalized.includes("drift")) {
    return "scan-drift";
  }
  if (normalized.includes("queue") || normalized.includes("study")) {
    return "reduce-queue";
  }
  if (normalized.includes("truth") || normalized.includes("push")) {
    return "clear-gates";
  }
  return "maintain-observability";
}

function isAffirmativeResponse(input: string): boolean {
  const normalized = input.trim().toLowerCase();
  return (
    normalized === "yes"
    || normalized === "y"
    || normalized.includes("grant")
    || normalized.includes("allow")
    || normalized.includes("approve")
  );
}

export function useAutopilotController({
  catalog,
  simulation,
  isConnected,
  emitSystemMessage,
}: UseAutopilotControllerArgs): UseAutopilotControllerResult {
  const [autopilotEnabled, setAutopilotEnabled] = useState(true);
  const [autopilotStatus, setAutopilotStatus] = useState<"running" | "waiting" | "stopped">("running");
  const [autopilotSummary, setAutopilotSummary] = useState("running");
  const [autopilotEvents, setAutopilotEvents] = useState<AutopilotActionEvent[]>([]);
  const [autopilotPermissions, setAutopilotPermissions] = useState<Record<string, boolean>>(
    DEFAULT_AUTOPILOT_PERMISSIONS,
  );

  const autopilotRef = useRef<Autopilot<AutopilotSenseContext> | null>(null);
  const autopilotPendingAskRef = useRef<AskPayload | null>(null);
  const autopilotDirectiveRef = useRef<string | null>(null);
  const autopilotPermissionsRef = useRef(autopilotPermissions);
  const autopilotLastActionRef = useRef<{ id: string; ts: number } | null>(null);
  const runtimeSnapshotRef = useRef({ catalog, simulation, isConnected });

  useEffect(() => {
    runtimeSnapshotRef.current = { catalog, simulation, isConnected };
  }, [catalog, isConnected, simulation]);

  useEffect(() => {
    autopilotPermissionsRef.current = autopilotPermissions;
  }, [autopilotPermissions]);

  const runAutopilotStudySnapshot = useCallback(async (): Promise<AutopilotActionResult> => {
    const baseUrl = runtimeBaseUrl();
    try {
      const response = await fetch(`${baseUrl}/api/study?limit=6`);
      if (!response.ok) {
        return {
          ok: false,
          summary: `study snapshot failed (${response.status})`,
        };
      }
      const study = (await response.json()) as StudySnapshotPayload;
      emitSystemMessage(
        [
          "autopilot /study",
          `stability=${Math.round(study.stability.score * 100)}%`,
          `blocked_gates=${study.signals.blocked_gate_count}`,
          `active_drifts=${study.signals.active_drift_count}`,
          `queue_pending=${study.signals.queue_pending_count}`,
        ].join("\n"),
      );
      return {
        ok: true,
        summary: `study snapshot sampled (stability=${Math.round(study.stability.score * 100)}%)`,
        meta: { queue_pending: study.signals.queue_pending_count },
      };
    } catch {
      return {
        ok: false,
        summary: "study snapshot request crashed",
      };
    }
  }, [emitSystemMessage]);

  const runAutopilotDriftScan = useCallback(async (): Promise<AutopilotActionResult> => {
    const baseUrl = runtimeBaseUrl();
    try {
      const response = await fetch(`${baseUrl}/api/drift/scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        return {
          ok: false,
          summary: `drift scan failed (${response.status})`,
        };
      }
      const payload = (await response.json()) as DriftScanPayload;
      emitSystemMessage(
        [
          "autopilot /drift",
          `active_drifts=${payload.active_drifts.length}`,
          `blocked_gates=${payload.blocked_gates.length}`,
        ].join("\n"),
      );
      return {
        ok: true,
        summary: `drift scanned (blocked=${payload.blocked_gates.length})`,
      };
    } catch {
      return {
        ok: false,
        summary: "drift scan request crashed",
      };
    }
  }, [emitSystemMessage]);

  const runAutopilotPushTruthDryRun = useCallback(async (): Promise<AutopilotActionResult> => {
    const baseUrl = runtimeBaseUrl();
    try {
      const response = await fetch(`${baseUrl}/api/push-truth/dry-run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        return {
          ok: false,
          summary: `push-truth dry-run failed (${response.status})`,
        };
      }
      const payload = (await response.json()) as {
        gate?: { blocked?: boolean };
        needs?: string[];
      };
      const blocked = payload?.gate?.blocked ? "blocked" : "pass";
      const needs = Array.isArray(payload?.needs) ? payload.needs.join(", ") : "(none)";
      emitSystemMessage(`autopilot /push-truth --dry-run\ngate=${blocked}\nneeds=${needs}`);
      return {
        ok: true,
        summary: `push-truth dry-run gate=${blocked}`,
      };
    } catch {
      return {
        ok: false,
        summary: "push-truth dry-run request crashed",
      };
    }
  }, [emitSystemMessage]);

  const senseAutopilotContext = useCallback(async (): Promise<AutopilotSenseContext> => {
    const runtime = runtimeSnapshotRef.current;
    const baseUrl = runtimeBaseUrl();

    let study: StudySnapshotPayload | null = null;
    try {
      const studyRes = await fetch(`${baseUrl}/api/study?limit=4`);
      if (studyRes.ok) {
        study = (await studyRes.json()) as StudySnapshotPayload;
      }
    } catch {
      // best effort
    }

    const blockedGateCount = study?.signals.blocked_gate_count ?? 0;
    const activeDriftCount = study?.signals.active_drift_count ?? 0;
    const queuePendingCount =
      study?.signals.queue_pending_count ?? runtime.catalog?.task_queue?.pending_count ?? 0;
    const truthGateBlocked =
      study?.signals.truth_gate_blocked
      ?? Boolean(runtime.simulation?.truth_state?.gate?.blocked ?? runtime.catalog?.truth_state?.gate?.blocked);

    let health: AutopilotHealth = "green";
    const healthReasons: string[] = [];
    if (!runtime.isConnected) {
      health = "red";
      healthReasons.push("runtime websocket disconnected");
    } else if (blockedGateCount >= 4 || activeDriftCount >= 8) {
      health = "red";
      healthReasons.push("high drift or blocked gate pressure");
    } else if (blockedGateCount >= 2 || activeDriftCount >= 4 || queuePendingCount >= 8) {
      health = "yellow";
      healthReasons.push("moderate pressure; run reduced-cost actions");
    }

    return {
      isConnected: runtime.isConnected,
      blockedGateCount,
      activeDriftCount,
      queuePendingCount,
      truthGateBlocked,
      health,
      healthReasons,
      permissions: autopilotPermissionsRef.current,
    };
  }, []);

  const hypothesizeAutopilotIntent = useCallback(
    async (ctx: AutopilotSenseContext): Promise<IntentHypothesis> => {
      if (autopilotDirectiveRef.current) {
        const goal = directiveToGoal(autopilotDirectiveRef.current);
        autopilotDirectiveRef.current = null;
        return {
          goal,
          confidence: 0.99,
          rationale: "user supplied directive",
        };
      }

      if (!ctx.isConnected) {
        return {
          goal: "restore-connectivity",
          confidence: 0.98,
          rationale: "runtime stream disconnected",
        };
      }
      if (ctx.blockedGateCount > 0 || ctx.truthGateBlocked) {
        return {
          goal: "clear-gates",
          confidence: clampValue(0.82 + ctx.blockedGateCount * 0.03, 0, 0.98),
          alternatives: [
            { goal: "scan-drift", confidence: 0.79 },
            { goal: "reduce-queue", confidence: 0.74 },
          ],
        };
      }
      if (ctx.queuePendingCount > 3) {
        return {
          goal: "reduce-queue",
          confidence: clampValue(0.77 + ctx.queuePendingCount * 0.015, 0, 0.94),
          alternatives: [{ goal: "scan-drift", confidence: 0.68 }],
        };
      }
      if (ctx.activeDriftCount > 0) {
        return {
          goal: "scan-drift",
          confidence: clampValue(0.76 + ctx.activeDriftCount * 0.02, 0, 0.9),
        };
      }
      return {
        goal: "maintain-observability",
        confidence: 0.64,
        alternatives: [
          { goal: "reduce-queue", confidence: 0.59 },
          { goal: "scan-drift", confidence: 0.57 },
        ],
      };
    },
    [],
  );

  const planAutopilotAction = useCallback(
    async (
      ctx: AutopilotSenseContext,
      goal: string,
    ): Promise<PlannedAction<AutopilotSenseContext>> => {
      const lastAction = autopilotLastActionRef.current;
      const isFreshAction = (actionId: string, maxAgeMs: number): boolean => {
        if (!lastAction) {
          return true;
        }
        if (lastAction.id !== actionId) {
          return true;
        }
        return Date.now() - lastAction.ts > maxAgeMs;
      };

      if (goal === "restore-connectivity") {
        return {
          id: "autopilot.wait-runtime",
          label: "wait runtime recovery",
          goal,
          risk: 0.1,
          cost: 0.05,
          requiredPerms: [],
          run: async () => ({ ok: false, summary: "runtime disconnected" }),
        };
      }

      if (goal === "clear-gates") {
        if (ctx.blockedGateCount >= 2 && isFreshAction("autopilot.push-truth-dry-run", 15000)) {
          return {
            id: "autopilot.push-truth-dry-run",
            label: "push truth dry-run",
            goal,
            risk: 0.58,
            cost: 0.42,
            requiredPerms: ["runtime.read", "truth.push.dry-run"],
            run: runAutopilotPushTruthDryRun,
          };
        }
        return {
          id: "autopilot.drift-scan",
          label: "drift scan",
          goal,
          risk: 0.22,
          cost: 0.18,
          requiredPerms: ["runtime.read"],
          run: runAutopilotDriftScan,
        };
      }

      if (goal === "reduce-queue") {
        return {
          id: "autopilot.study-snapshot",
          label: "study snapshot",
          goal,
          risk: 0.24,
          cost: 0.22,
          requiredPerms: ["runtime.read"],
          run: runAutopilotStudySnapshot,
        };
      }

      if (goal === "scan-drift") {
        return {
          id: "autopilot.drift-scan",
          label: "drift scan",
          goal,
          risk: 0.2,
          cost: 0.18,
          requiredPerms: ["runtime.read"],
          run: runAutopilotDriftScan,
        };
      }

      if (!isFreshAction("autopilot.study-snapshot", 20000)) {
        return {
          id: "autopilot.idle",
          label: "idle hold",
          goal,
          risk: 0.02,
          cost: 0.01,
          requiredPerms: [],
          run: async () => ({ ok: true, summary: "idle cadence hold" }),
        };
      }

      return {
        id: "autopilot.study-snapshot",
        label: "study snapshot",
        goal,
        risk: 0.19,
        cost: 0.2,
        requiredPerms: ["runtime.read"],
        run: runAutopilotStudySnapshot,
      };
    },
    [runAutopilotDriftScan, runAutopilotPushTruthDryRun, runAutopilotStudySnapshot],
  );

  const gateAutopilotAction = useCallback(
    (
      ctx: AutopilotSenseContext,
      hyp: IntentHypothesis,
      action: PlannedAction<AutopilotSenseContext>,
    ): GateVerdict<AutopilotSenseContext> => {
      if (ctx.health === "red") {
        return {
          ok: false,
          ask: {
            gate: "health",
            reason: `I paused because runtime health is red (${ctx.healthReasons.join("; ") || "unstable"}).`,
            need: "Should I keep waiting, or do you want me to pause autopilot?",
            options: ["keep waiting", "pause autopilot", "run /study now"],
            urgency: "high",
            context: {
              connected: ctx.isConnected,
              blocked_gates: ctx.blockedGateCount,
              active_drifts: ctx.activeDriftCount,
            },
          },
        };
      }

      if (hyp.confidence < AUTOPILOT_C_MIN) {
        return {
          ok: false,
          ask: {
            gate: "confidence",
            reason: `I tried to infer the next goal but confidence is ${hyp.confidence.toFixed(2)} (< ${AUTOPILOT_C_MIN}).`,
            need: "Pick the next move so I can continue.",
            options: ["run /study now", "run /drift", "pause autopilot"],
            urgency: "low",
            context: {
              inferred_goal: hyp.goal,
              alternatives: hyp.alternatives ?? [],
            },
          },
        };
      }

      const missingPerms = action.requiredPerms.filter((permission) => !ctx.permissions[permission]);
      if (missingPerms.length > 0) {
        const permission = missingPerms[0];
        return {
          ok: false,
          ask: {
            gate: "permission",
            reason: `I selected '${action.label}', but permission '${permission}' is missing.`,
            need: `Grant ${permission} so I can continue?`,
            options: ["grant permission", "deny", "pause autopilot"],
            urgency: "med",
            context: {
              permission,
              action_id: action.id,
            },
          },
        };
      }

      if (action.risk > AUTOPILOT_R_MAX) {
        const riskPermission = `risk:${action.id}`;
        if (!ctx.permissions[riskPermission]) {
          return {
            ok: false,
            ask: {
              gate: "risk",
              reason: `I can run '${action.label}', but risk ${action.risk.toFixed(2)} is above ${AUTOPILOT_R_MAX.toFixed(2)}.`,
              need: `Approve this higher-risk step (${action.label})?`,
              options: ["approve step", "skip", "pause autopilot"],
              urgency: "med",
              context: {
                permission: riskPermission,
                action_id: action.id,
                risk: action.risk,
              },
            },
          };
        }
      }

      return { ok: true, action };
    },
    [],
  );

  const handleAutopilotActionEvent = useCallback((event: AutopilotActionEvent) => {
    setAutopilotEvents((prev) => [event, ...prev].slice(0, 8));
    setAutopilotSummary(`${event.intent}: ${event.summary}`);
    if (event.result !== "skipped") {
      autopilotLastActionRef.current = {
        id: event.actionId,
        ts: Date.now(),
      };
    }
  }, []);

  const handleAutopilotAsk = useCallback((ask: AskPayload) => {
    autopilotPendingAskRef.current = ask;
    setAutopilotStatus("waiting");
    setAutopilotSummary(`${ask.gate || "unknown"} gate: ${ask.need}`);
  }, []);

  useEffect(() => {
    const autopilot = new Autopilot<AutopilotSenseContext>({
      sense: senseAutopilotContext,
      hypothesize: hypothesizeAutopilotIntent,
      plan: planAutopilotAction,
      gate: gateAutopilotAction,
      onActionEvent: handleAutopilotActionEvent,
      onAsk: handleAutopilotAsk,
      onTickError: () => {
        setAutopilotSummary("tick error; waiting for next cycle");
      },
      tickDelayMs: 5000,
    });

    autopilotRef.current = autopilot;

    if (autopilotEnabled) {
      autopilot.start();
    }

    return () => {
      autopilot.stop();
      if (autopilotRef.current === autopilot) {
        autopilotRef.current = null;
      }
    };
  }, [
    autopilotEnabled,
    gateAutopilotAction,
    handleAutopilotActionEvent,
    handleAutopilotAsk,
    hypothesizeAutopilotIntent,
    planAutopilotAction,
    senseAutopilotContext,
  ]);

  const handleAutopilotUserInput = useCallback(
    (text: string): boolean => {
      const autopilot = autopilotRef.current;
      const pendingAsk = autopilotPendingAskRef.current;
      if (!autopilot || !pendingAsk || !autopilot.isWaitingForInput()) {
        return false;
      }

      const normalized = text.trim().toLowerCase();
      const pauseRequested =
        normalized.includes("pause autopilot")
        || normalized.includes("disable autopilot")
        || normalized === "/autopilot off";

      if (pauseRequested) {
        autopilot.stop();
        setAutopilotEnabled(false);
        autopilotPendingAskRef.current = null;
        setAutopilotStatus("stopped");
        setAutopilotSummary("paused by user");
        emitSystemMessage("autopilot paused by user");
        return true;
      }

      const permission =
        typeof pendingAsk.context?.permission === "string" ? pendingAsk.context.permission : null;
      if (permission && isAffirmativeResponse(text)) {
        setAutopilotPermissions((prev) => ({
          ...prev,
          [permission]: true,
        }));
        emitSystemMessage(`autopilot permission granted: ${permission}`);
      } else if (permission && normalized.includes("deny")) {
        emitSystemMessage(`autopilot permission denied: ${permission}`);
      } else {
        autopilotDirectiveRef.current = text;
      }

      autopilotPendingAskRef.current = null;
      setAutopilotStatus("running");
      setAutopilotSummary("resumed");
      autopilot.resume();
      return true;
    },
    [emitSystemMessage],
  );

  const toggleAutopilot = useCallback(() => {
    setAutopilotEnabled((prev) => {
      const next = !prev;
      setAutopilotStatus(next ? "running" : "stopped");
      setAutopilotSummary(next ? "running" : "disabled");
      return next;
    });
  }, []);

  return {
    autopilotEnabled,
    autopilotStatus,
    autopilotSummary,
    autopilotEvents,
    handleAutopilotUserInput,
    toggleAutopilot,
  };
}
