import { useState, useCallback, useEffect, useMemo, useRef, lazy, Suspense, type CSSProperties } from "react";
import { motion, type PanInfo } from "framer-motion";
import { useWorldState } from "./hooks/useWorldState";
import { OVERLAY_VIEW_OPTIONS, SimulationCanvas } from "./components/Simulation/Canvas";
import { ChatPanel } from "./components/Panels/Chat";
import { PresenceCallDeck } from "./components/Panels/PresenceCallDeck";
import { ProjectionLedgerPanel } from "./components/Panels/ProjectionLedgerPanel";
import {
  Autopilot,
  type AskPayload,
  type AutopilotActionEvent,
  type AutopilotActionResult,
  type GateVerdict,
  type IntentHypothesis,
  type PlannedAction,
} from "./autopilot";
import type {
  CouncilApiResponse,
  DriftScanPayload,
  StudySnapshotPayload,
  TaskQueueSnapshot,
  UIPerspective,
  UIProjectionBundle,
  UIProjectionElementState,
  WorldInteractionResponse,
} from "./types";

const VitalsPanel = lazy(() =>
  import("./components/Panels/Vitals").then((module) => ({ default: module.VitalsPanel })),
);
const CatalogPanel = lazy(() =>
  import("./components/Panels/Catalog").then((module) => ({ default: module.CatalogPanel })),
);
const OmniPanel = lazy(() =>
  import("./components/Panels/Omni").then((module) => ({ default: module.OmniPanel })),
);
const MythWorldPanel = lazy(() =>
  import("./components/Panels/MythWorld").then((module) => ({ default: module.MythWorldPanel })),
);
const WebGraphWeaverPanel = lazy(() =>
  import("./components/Panels/WebGraphWeaverPanel").then((module) => ({
    default: module.WebGraphWeaverPanel,
  })),
);
const InspirationAtlasPanel = lazy(() =>
  import("./components/Panels/InspirationAtlasPanel").then((module) => ({
    default: module.InspirationAtlasPanel,
  })),
);
const StabilityObservatoryPanel = lazy(() =>
  import("./components/Panels/StabilityObservatoryPanel").then((module) => ({
    default: module.StabilityObservatoryPanel,
  })),
);

interface OverlayApi {
  pulseAt?: (x: number, y: number, power: number) => void;
  singAll?: () => void;
}

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

interface UiToast {
  id: number;
  title: string;
  body: string;
}

const AUTOPILOT_C_MIN = 0.72;
const AUTOPILOT_R_MAX = 0.45;
const PROJECTION_GRID_COLUMNS = 12;
const PROJECTION_GRID_ROWS = 24;

const DEFAULT_AUTOPILOT_PERMISSIONS: Record<string, boolean> = {
  "runtime.read": true,
  "truth.push.dry-run": false,
};

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
    normalized === "yes" ||
    normalized === "y" ||
    normalized.includes("grant") ||
    normalized.includes("allow") ||
    normalized.includes("approve")
  );
}

function runtimeBaseUrl(): string {
  return window.location.port === "5173" ? "http://127.0.0.1:8787" : "";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function projectionOpacity(raw: number | undefined, floor = 0.9): number {
  const normalized = clamp(typeof raw === "number" ? raw : 1, 0, 1);
  return floor + normalized * (1 - floor);
}

function DeferredPanelPlaceholder({ title }: { title: string }) {
  return (
    <div className="rounded-xl border border-[var(--line)] bg-[rgba(45,46,39,0.82)] px-4 py-5">
      <p className="text-sm font-semibold text-ink">{title}</p>
      <p className="text-xs text-muted mt-1">warming up panel...</p>
    </div>
  );
}

export default function App() {
  const [uiPerspective, setUiPerspective] = useState<UIPerspective>("hybrid");
  const { catalog, simulation, projection, isConnected } = useWorldState(uiPerspective);

  const [overlayApi, setOverlayApi] = useState<OverlayApi | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [voiceInputMeta, setVoiceInputMeta] = useState("voice input idle / 音声入力待機");
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [worldInteraction, setWorldInteraction] = useState<WorldInteractionResponse | null>(null);
  const [interactingPersonId, setInteractingPersonId] = useState<string | null>(null);
  const [deferredPanelsReady, setDeferredPanelsReady] = useState(false);
  const [isWideViewport, setIsWideViewport] = useState(
    () => window.matchMedia("(min-width: 1280px)").matches,
  );
  const [autopilotEnabled, setAutopilotEnabled] = useState(true);
  const [autopilotStatus, setAutopilotStatus] = useState<"running" | "waiting" | "stopped">("stopped");
  const [autopilotSummary, setAutopilotSummary] = useState("booting");
  const [autopilotEvents, setAutopilotEvents] = useState<AutopilotActionEvent[]>([]);
  const [autopilotPermissions, setAutopilotPermissions] = useState<Record<string, boolean>>(
    DEFAULT_AUTOPILOT_PERMISSIONS,
  );
  const [uiToasts, setUiToasts] = useState<UiToast[]>([]);

  const autopilotRef = useRef<Autopilot<AutopilotSenseContext> | null>(null);
  const autopilotPendingAskRef = useRef<AskPayload | null>(null);
  const autopilotDirectiveRef = useRef<string | null>(null);
  const autopilotPermissionsRef = useRef(autopilotPermissions);
  const autopilotLastActionRef = useRef<{ id: string; ts: number } | null>(null);
  const runtimeSnapshotRef = useRef({ catalog, simulation, isConnected });
  const toastSeqRef = useRef(0);
  const toastTimeoutsRef = useRef<Map<number, number>>(new Map());
  const gridContainerRef = useRef<HTMLDivElement>(null);

  const [layoutOverrides, setLayoutOverrides] = useState<Record<string, { x: number; y: number; w: number; h: number }>>({});
  const [isEditMode, setIsEditMode] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDeferredPanelsReady(true);
    }, 220);
    return () => {
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    const query = window.matchMedia("(min-width: 1280px)");
    const handleViewportChange = (event: MediaQueryListEvent) => {
      setIsWideViewport(event.matches);
    };
    setIsWideViewport(query.matches);
    query.addEventListener("change", handleViewportChange);
    return () => {
      query.removeEventListener("change", handleViewportChange);
    };
  }, []);

  useEffect(() => {
    runtimeSnapshotRef.current = { catalog, simulation, isConnected };
  }, [catalog, isConnected, simulation]);

  useEffect(() => {
    autopilotPermissionsRef.current = autopilotPermissions;
  }, [autopilotPermissions]);

  const dismissToast = useCallback((id: number) => {
    const timeoutId = toastTimeoutsRef.current.get(id);
    if (timeoutId !== undefined) {
      window.clearTimeout(timeoutId);
      toastTimeoutsRef.current.delete(id);
    }
    setUiToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  useEffect(() => {
    const handler: EventListener = (event) => {
      const customEvent = event as CustomEvent<{ title?: unknown; body?: unknown }>;
      const title =
        typeof customEvent.detail?.title === "string" ? customEvent.detail.title.trim() : "Notice";
      const body =
        typeof customEvent.detail?.body === "string" ? customEvent.detail.body.trim() : "";
      if (!body) {
        return;
      }

      const id = Date.now() + toastSeqRef.current;
      toastSeqRef.current += 1;
      setUiToasts((prev) => [{ id, title: title || "Notice", body }, ...prev].slice(0, 4));

      const timeoutId = window.setTimeout(() => {
        setUiToasts((prev) => prev.filter((toast) => toast.id !== id));
        toastTimeoutsRef.current.delete(id);
      }, 5200);
      toastTimeoutsRef.current.set(id, timeoutId);
    };

    window.addEventListener("ui:toast", handler);
    return () => {
      window.removeEventListener("ui:toast", handler);
      toastTimeoutsRef.current.forEach((timeoutId) => {
        window.clearTimeout(timeoutId);
      });
      toastTimeoutsRef.current.clear();
    };
  }, []);

  const handleRecord = useCallback(async () => {
    if (isRecording) {
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "audio/webm" });
        setRecordedBlob(blob);
        setVoiceInputMeta(`voice captured / 音声取得: ${Math.round(blob.size / 1024)}KB`);
        stream.getTracks().forEach((track) => {
          track.stop();
        });
        setIsRecording(false);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setVoiceInputMeta("recording voice / 録音中");

      window.setTimeout(() => {
        if (mediaRecorder.state === "recording") {
          mediaRecorder.stop();
        }
      }, 8000);
    } catch {
      setVoiceInputMeta("mic permission denied / マイク許可なし");
    }
  }, [isRecording]);

  const handleTranscribe = useCallback(async (): Promise<string | undefined> => {
    if (!recordedBlob) {
      return undefined;
    }

    const buffer = await recordedBlob.arrayBuffer();
    let binary = "";
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i += 1) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = btoa(binary);

    try {
      const baseUrl = runtimeBaseUrl();
      const response = await fetch(`${baseUrl}/api/transcribe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio_base64: base64, mime: recordedBlob.type }),
      });
      const payload = (await response.json()) as { ok?: boolean; text?: string; error?: string };
      if (payload.ok) {
        const text = String(payload.text ?? "");
        setVoiceInputMeta(`transcribed: ${text}`);
        return text;
      }
      setVoiceInputMeta(`error: ${String(payload.error ?? "unknown")}`);
      return undefined;
    } catch {
      setVoiceInputMeta("transcribe failed");
      return undefined;
    }
  }, [recordedBlob]);

  const handleSendVoice = useCallback(async () => {
    const text = await handleTranscribe();
    if (!text) {
      return;
    }

    window.dispatchEvent(
      new CustomEvent("chat-message", {
        detail: { role: "user", text },
      }),
    );

    const baseUrl = runtimeBaseUrl();
    try {
      const response = await fetch(`${baseUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", text }] }),
      });
      const payload = (await response.json()) as { reply?: string };
      const reply = String(payload.reply ?? "");

      window.dispatchEvent(
        new CustomEvent("chat-message", {
          detail: { role: "assistant", text: reply },
        }),
      );

      if (reply.includes("[[PULSE]]")) {
        overlayApi?.pulseAt?.(0.5, 0.5, 1);
      }
      if (reply.includes("[[SING]]")) {
        overlayApi?.singAll?.();
      }
    } catch {
      window.dispatchEvent(
        new CustomEvent("chat-message", {
          detail: { role: "system", text: "voice chat request failed" },
        }),
      );
    }
  }, [handleTranscribe, overlayApi]);

  const emitSystemMessage = useCallback((text: string) => {
    window.dispatchEvent(
      new CustomEvent("chat-message", {
        detail: {
          role: "system",
          text,
        },
      }),
    );
  }, []);

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
      study?.signals.truth_gate_blocked ??
      Boolean(runtime.simulation?.truth_state?.gate?.blocked ?? runtime.catalog?.truth_state?.gate?.blocked);

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
          confidence: clamp(0.82 + ctx.blockedGateCount * 0.03, 0, 0.98),
          alternatives: [
            { goal: "scan-drift", confidence: 0.79 },
            { goal: "reduce-queue", confidence: 0.74 },
          ],
        };
      }
      if (ctx.queuePendingCount > 3) {
        return {
          goal: "reduce-queue",
          confidence: clamp(0.77 + ctx.queuePendingCount * 0.015, 0, 0.94),
          alternatives: [{ goal: "scan-drift", confidence: 0.68 }],
        };
      }
      if (ctx.activeDriftCount > 0) {
        return {
          goal: "scan-drift",
          confidence: clamp(0.76 + ctx.activeDriftCount * 0.02, 0, 0.9),
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
      setAutopilotStatus("running");
      setAutopilotSummary("running");
    } else {
      setAutopilotStatus("stopped");
      setAutopilotSummary("disabled");
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
        normalized.includes("pause autopilot") ||
        normalized.includes("disable autopilot") ||
        normalized === "/autopilot off";

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
    setAutopilotEnabled((prev) => !prev);
  }, []);

  const handleLedgerCommand = useCallback(
    async (text: string): Promise<boolean> => {
      const trimmed = text.trim();
      if (!trimmed.toLowerCase().startsWith("/ledger")) {
        return false;
      }

      const payloadText = trimmed.replace(/^\/ledger\s*/i, "");
      const utterances = payloadText
        ? payloadText
            .split("|")
            .map((row) => row.trim())
            .filter((row) => row.length > 0)
        : [];

      try {
        const baseUrl = runtimeBaseUrl();
        const response = await fetch(`${baseUrl}/api/eta-mu-ledger`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ utterances }),
        });
        const payload = (await response.json()) as { jsonl?: string };
        const body = payload?.jsonl ? payload.jsonl.trim() : "(no utterances)";
        emitSystemMessage(`eta/mu ledger\n${body}`);
      } catch {
        emitSystemMessage("eta/mu ledger failed");
      }
      return true;
    },
    [emitSystemMessage],
  );

  const handlePresenceSayCommand = useCallback(
    async (text: string): Promise<boolean> => {
      const trimmed = text.trim();
      if (!trimmed.toLowerCase().startsWith("/say")) {
        return false;
      }

      const args = trimmed.replace(/^\/say\s*/i, "");
      const [presenceIdRaw, ...rest] = args.split(/\s+/).filter((token) => token.length > 0);
      const presence_id = presenceIdRaw || "witness_thread";
      const messageText = rest.join(" ");

      try {
        const baseUrl = runtimeBaseUrl();
        const response = await fetch(`${baseUrl}/api/presence/say`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            presence_id,
            text: messageText,
          }),
        });
        const payload = (await response.json()) as {
          presence_name?: { en?: string };
          rendered_text?: string;
          say_intent?: { facts?: unknown[]; asks?: unknown[]; repairs?: unknown[] };
        };
        emitSystemMessage(
          `${payload?.presence_name?.en || presence_id} / say\n${payload?.rendered_text || "(no render)"}\n` +
            `facts=${payload?.say_intent?.facts?.length || 0} asks=${payload?.say_intent?.asks?.length || 0} repairs=${payload?.say_intent?.repairs?.length || 0}`,
        );
      } catch {
        emitSystemMessage("presence say failed");
      }
      return true;
    },
    [emitSystemMessage],
  );

  const handleDriftCommand = useCallback(
    async (text: string): Promise<boolean> => {
      const trimmed = text.trim();
      if (trimmed.toLowerCase() !== "/drift") {
        return false;
      }

      try {
        const baseUrl = runtimeBaseUrl();
        const response = await fetch(`${baseUrl}/api/drift/scan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        const payload = (await response.json()) as {
          active_drifts?: unknown[];
          blocked_gates?: unknown[];
        };
        const drifts = Array.isArray(payload?.active_drifts) ? payload.active_drifts.length : 0;
        const blocked = Array.isArray(payload?.blocked_gates) ? payload.blocked_gates.length : 0;
        emitSystemMessage(`drift scan\nactive_drifts=${drifts} blocked_gates=${blocked}`);
      } catch {
        emitSystemMessage("drift scan failed");
      }
      return true;
    },
    [emitSystemMessage],
  );

  const handlePushTruthDryRunCommand = useCallback(
    async (text: string): Promise<boolean> => {
      const trimmed = text.trim().toLowerCase();
      if (trimmed !== "/push-truth --dry-run") {
        return false;
      }

      try {
        const baseUrl = runtimeBaseUrl();
        const response = await fetch(`${baseUrl}/api/push-truth/dry-run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        const payload = (await response.json()) as {
          gate?: { blocked?: boolean };
          needs?: string[];
        };
        const blocked = payload?.gate?.blocked ? "blocked" : "pass";
        const needs = Array.isArray(payload?.needs) ? payload.needs.join(", ") : "";
        emitSystemMessage(`push-truth dry-run\ngate=${blocked}\nneeds=${needs || "(none)"}`);
      } catch {
        emitSystemMessage("push-truth dry-run failed");
      }
      return true;
    },
    [emitSystemMessage],
  );

  const handleStudyCommand = useCallback(
    async (text: string): Promise<boolean> => {
      const raw = text.trim();
      const trimmed = raw.toLowerCase();
      const exportPrefix = "/study export";
      if (trimmed === exportPrefix || trimmed.startsWith(`${exportPrefix} `)) {
        const label = raw.slice(exportPrefix.length).trim();
        try {
          const baseUrl = runtimeBaseUrl();
          const response = await fetch(`${baseUrl}/api/study/export`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              label: label || "chat-export",
              include_truth: true,
              refs: ["chat:/study export"],
            }),
          });
          if (!response.ok) {
            throw new Error(`study export failed: ${response.status}`);
          }
          const payload = (await response.json()) as {
            ok?: boolean;
            event?: { id?: string; ts?: string; label?: string };
            history?: { count?: number; path?: string };
          };
          const eventId = String(payload.event?.id || "(unknown)");
          const historyCount = Number(payload.history?.count ?? 0);
          emitSystemMessage(
            `study export\nid=${eventId}\nlabel=${String(payload.event?.label || label || "chat-export")}\nhistory=${historyCount}`,
          );
        } catch {
          emitSystemMessage("study export failed");
        }
        return true;
      }

      if (trimmed !== "/study" && trimmed !== "/study now") {
        return false;
      }

      const baseUrl = runtimeBaseUrl();
      try {
        const studyResponse = await fetch(`${baseUrl}/api/study?limit=6`);
        if (studyResponse.ok) {
          const study = (await studyResponse.json()) as StudySnapshotPayload;
          const signals = study.signals;
          const topDecision = study.council?.decisions?.[0];
          const topDecisionLine = topDecision
            ? `top_decision=${topDecision.status} id=${topDecision.id} source=${String(topDecision.resource?.source_rel_path || "(unknown)")}`
            : "top_decision=(none)";
          const gateReasons = (study.drift?.blocked_gates ?? [])
            .map((row) => row.reason)
            .slice(0, 4)
            .join(", ");
          const warningLine = (study.warnings ?? [])
            .slice(0, 3)
            .map((row) => `${row.code}:${row.message}`)
            .join(" | ");

          emitSystemMessage(
            [
              "study snapshot",
              `stability=${Math.round(study.stability.score * 100)}% (${study.stability.label})`,
              `truth_gate=${signals.truth_gate_blocked ? "blocked" : "clear"}`,
              `blocked_gates=${signals.blocked_gate_count} active_drifts=${signals.active_drift_count}`,
              `queue_pending=${signals.queue_pending_count} queue_events=${signals.queue_event_count}`,
              `council_pending=${signals.council_pending_count} approved=${signals.council_approved_count} decisions=${signals.council_decision_count}`,
              topDecisionLine,
              `gate_reasons=${gateReasons || "(none)"}`,
              `runtime_receipts_within_vault=${String(study.runtime.receipts_path_within_vault)}`,
              `warnings=${warningLine || "(none)"}`,
            ].join("\n"),
          );
          return true;
        }

        if (studyResponse.status !== 404) {
          throw new Error(`study fetch failed: /api/study status=${studyResponse.status}`);
        }

        const [councilRes, queueRes, driftRes] = await Promise.all([
          fetch(`${baseUrl}/api/council?limit=6`),
          fetch(`${baseUrl}/api/task/queue`),
          fetch(`${baseUrl}/api/drift/scan`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          }),
        ]);

        if (!councilRes.ok || !queueRes.ok || !driftRes.ok) {
          throw new Error(
            `study fetch failed: council=${councilRes.status} queue=${queueRes.status} drift=${driftRes.status}`,
          );
        }

        const councilPayload = (await councilRes.json()) as CouncilApiResponse;
        const queuePayload = (await queueRes.json()) as { ok: boolean; queue: TaskQueueSnapshot };
        const driftPayload = (await driftRes.json()) as DriftScanPayload;

        const council = councilPayload.council;
        const queue = queuePayload.queue;
        const blocked = driftPayload.blocked_gates.length;
        const drifts = driftPayload.active_drifts.length;
        const pending = queue.pending_count;
        const pendingCouncil = council.pending_count;
        const truthBlocked = Boolean(
          simulation?.truth_state?.gate?.blocked ?? catalog?.truth_state?.gate?.blocked,
        );

        const blockedPenalty = Math.min(0.34, (blocked / 4) * 0.34);
        const driftPenalty = Math.min(0.18, (drifts / 8) * 0.18);
        const queuePenalty = Math.min(0.2, (pending / 8) * 0.2);
        const councilPenalty = Math.min(0.16, (pendingCouncil / 5) * 0.16);
        const truthPenalty = truthBlocked ? 0.12 : 0;
        const stabilityScore = Math.max(
          0,
          Math.min(1, 1 - blockedPenalty - driftPenalty - queuePenalty - councilPenalty - truthPenalty),
        );

        const topDecision = council.decisions?.[0];
        const topDecisionLine = topDecision
          ? `top_decision=${topDecision.status} id=${topDecision.id} source=${String(topDecision.resource?.source_rel_path || "(unknown)")}`
          : "top_decision=(none)";
        const gateReasons = driftPayload.blocked_gates
          .map((row) => row.reason)
          .slice(0, 4)
          .join(", ");

        emitSystemMessage(
          [
            "study snapshot",
            `stability=${Math.round(stabilityScore * 100)}%`,
            `truth_gate=${truthBlocked ? "blocked" : "clear"}`,
            `blocked_gates=${blocked} active_drifts=${drifts}`,
            `queue_pending=${pending} queue_events=${queue.event_count}`,
            `council_pending=${pendingCouncil} approved=${council.approved_count} decisions=${council.decision_count}`,
            topDecisionLine,
            `gate_reasons=${gateReasons || "(none)"}`,
            "runtime_receipts_within_vault=(unknown:legacy-mode)",
          ].join("\n"),
        );
      } catch {
        emitSystemMessage("study snapshot failed");
      }
      return true;
    },
    [catalog?.truth_state?.gate?.blocked, emitSystemMessage, simulation?.truth_state?.gate?.blocked],
  );

  const handleChatCommand = useCallback(
    async (text: string): Promise<boolean> => {
      if (await handleLedgerCommand(text)) {
        return true;
      }
      if (await handlePresenceSayCommand(text)) {
        return true;
      }
      if (await handleDriftCommand(text)) {
        return true;
      }
      if (await handlePushTruthDryRunCommand(text)) {
        return true;
      }
      if (await handleStudyCommand(text)) {
        return true;
      }
      return false;
    },
    [
      handleDriftCommand,
      handleLedgerCommand,
      handlePresenceSayCommand,
      handlePushTruthDryRunCommand,
      handleStudyCommand,
    ],
  );

  const handleWorldInteract = useCallback(async (personId: string, action: "speak" | "pray" | "sing") => {
    setInteractingPersonId(personId);
    try {
      const baseUrl = runtimeBaseUrl();
      const response = await fetch(`${baseUrl}/api/world/interact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ person_id: personId, action }),
      });
      const payload = (await response.json()) as WorldInteractionResponse;
      setWorldInteraction(payload);

      if (payload?.ok) {
        window.dispatchEvent(
          new CustomEvent("chat-message", {
            detail: {
              role: "assistant",
              text: `${payload.line_en}\n${payload.line_ja}`,
            },
          }),
        );
      }
    } catch {
      setWorldInteraction({
        ok: false,
        line_en: "Interaction failed. The field is unstable.",
        line_ja: "対話に失敗。場が不安定です。",
      });
    } finally {
      setInteractingPersonId(null);
    }
  }, []);

  const activeProjection: UIProjectionBundle | null =
    projection ?? simulation?.projection ?? catalog?.ui_projection ?? null;

  const projectionStateByElement = useMemo(() => {
    const map = new Map<string, UIProjectionElementState>();
    if (!activeProjection) {
      return map;
    }
    activeProjection.states.forEach((state) => {
      map.set(state.element_id, state);
    });
    return map;
  }, [activeProjection]);
  
  const projectionLayoutRects = useMemo(
    () => activeProjection?.layout?.rects ?? {},
    [activeProjection?.layout?.rects],
  );

  const mergedLayoutRects = useMemo(() => ({
    ...projectionLayoutRects,
    ...layoutOverrides,
  }), [projectionLayoutRects, layoutOverrides]);

  const projectionDensitySignalFor = useCallback(
    (state: UIProjectionElementState | undefined): number => {
      if (!state) {
        return 0.42;
      }
      const minArea = clamp(activeProjection?.layout?.clamps?.min_area ?? 0.1, 0.05, 1);
      const maxArea = Math.max(
        minArea + 0.001,
        clamp(activeProjection?.layout?.clamps?.max_area ?? 0.36, minArea + 0.001, 1),
      );
      const areaSignal = clamp((state.area - minArea) / (maxArea - minArea), 0, 1);
      const prioritySignal = clamp(state.priority, 0, 1);
      return clamp(areaSignal * 0.72 + prioritySignal * 0.28, 0, 1);
    },
    [activeProjection?.layout?.clamps?.max_area, activeProjection?.layout?.clamps?.min_area],
  );

  const projectionStyleFor = useCallback(
    (elementId: string, fallbackSpan = 12) => {
      const state = projectionStateByElement.get(elementId);
      const densitySignal = projectionDensitySignalFor(state);
      const baseStyle = {
        opacity: state ? projectionOpacity(state.opacity, 0.9) : 1,
        transform: state ? `scale(${(1 + clamp(state.pulse, 0, 1) * 0.014).toFixed(3)})` : undefined,
        transformOrigin: "center top",
      } as const;

      if (!isWideViewport) {
        return {
          ...baseStyle,
          transition: "transform 260ms ease, opacity 220ms ease",
        } as const;
      }

      const rect = mergedLayoutRects[elementId];
      if (rect) {
        const colStart = clamp(
          Math.floor(clamp(rect.x, 0, 0.98) * PROJECTION_GRID_COLUMNS) + 1,
          1,
          PROJECTION_GRID_COLUMNS,
        );
        const rowStart = clamp(
          Math.floor(clamp(rect.y, 0, 0.98) * PROJECTION_GRID_ROWS) + 1,
          1,
          PROJECTION_GRID_ROWS,
        );
        const maxColSpan = Math.max(1, PROJECTION_GRID_COLUMNS - colStart + 1);
        const maxRowSpan = Math.max(1, PROJECTION_GRID_ROWS - rowStart + 1);
        const colSpan = clamp(
          Math.round(clamp(rect.w, 0.05, 1) * PROJECTION_GRID_COLUMNS),
          1,
          maxColSpan,
        );
        const rowSpan = clamp(
          Math.round(clamp(rect.h, 0.05, 1) * PROJECTION_GRID_ROWS),
          1,
          maxRowSpan,
        );

        return {
          ...baseStyle,
          gridColumn: `${colStart} / span ${colSpan}`,
          gridRow: `${rowStart} / span ${rowSpan}`,
          transition:
            "grid-column 220ms ease, grid-row 220ms ease, transform 260ms ease, opacity 220ms ease",
        } as const;
      }

      const baseColSpan = clamp(fallbackSpan, 2, 12);
      const minColSpan = baseColSpan >= 10 ? 4 : baseColSpan >= 6 ? 3 : 2;
      const colSpan = clamp(
        Math.round(baseColSpan * (0.44 + densitySignal * 0.28)),
        minColSpan,
        baseColSpan,
      );
      const baseRowSpan = baseColSpan >= 10 ? 4 : baseColSpan >= 6 ? 3 : 2;
      const minRowSpan = baseRowSpan >= 4 ? 2 : 1;
      const rowSpan = clamp(
        Math.round(baseRowSpan * (0.52 + densitySignal * 0.28)),
        minRowSpan,
        baseRowSpan,
      );

      return {
        ...baseStyle,
        gridColumn: `span ${colSpan} / span ${colSpan}`,
        gridRow: `span ${rowSpan} / span ${rowSpan}`,
        transition:
          "grid-column 220ms ease, grid-row 220ms ease, transform 260ms ease, opacity 220ms ease",
      } as const;
    },
    [isWideViewport, projectionDensitySignalFor, mergedLayoutRects, projectionStateByElement],
  );

  const handleDragEnd = useCallback((_: MouseEvent | TouchEvent | PointerEvent, info: PanInfo, elementId: string) => {
    if (!gridContainerRef.current) return;
    const containerRect = gridContainerRef.current.getBoundingClientRect();
    const x = (info.point.x - containerRect.left) / containerRect.width;
    const y = (info.point.y - containerRect.top) / containerRect.height;
    
    // We update position but keep dimensions unless resized (resize logic separate)
    // Note: this simple drag updates x/y based on top-left.
    // Ideally we snap? x/y are normalized 0..1.
    // The projectionStyleFor logic snaps to grid cells.
    
    setLayoutOverrides(prev => ({
      ...prev,
      [elementId]: {
        ...prev[elementId],
        x: clamp(x, 0, 0.95),
        y: clamp(y, 0, 0.95),
        w: prev[elementId]?.w ?? 0.25, // preserve or default
        h: prev[elementId]?.h ?? 0.2,
      }
    }));
  }, []);

  const simulationMapState = projectionStateByElement.get("nexus.ui.simulation_map");
  const simulationCanvasHeight = useMemo(() => {
    return Math.round(300 + projectionDensitySignalFor(simulationMapState) * 120);
  }, [projectionDensitySignalFor, simulationMapState]);
  const dedicatedOverlayViews = useMemo(
    () => OVERLAY_VIEW_OPTIONS.filter((option) => option.id !== "omni"),
    [],
  );

  const projectionPerspective = activeProjection?.perspective ?? uiPerspective;
  const projectionOptions =
    activeProjection?.perspectives ??
    catalog?.ui_perspectives ?? [
      {
        id: "hybrid",
        symbol: "perspective.hybrid",
        name: "Hybrid",
        merge: "hybrid",
        description: "Wallclock ordering with causal overlays.",
        default: true,
      },
      {
        id: "causal-time",
        symbol: "perspective.causal-time",
        name: "Causal Time",
        merge: "causal-time",
        description: "Prioritize causal links over wallclock sequence.",
        default: false,
      },
      {
        id: "swimlanes",
        symbol: "perspective.swimlanes",
        name: "Swimlanes",
        merge: "swimlanes",
        description: "Parallel lanes with threaded causality.",
        default: false,
      },
    ];

  const activeChatLens = activeProjection?.chat_sessions?.[0] ?? null;
  const chatLensState = projectionStateByElement.get("nexus.ui.chat.witness_thread") ?? null;
  const latestAutopilotEvent = autopilotEvents[0] ?? null;

  const panelConfigs = useMemo(() => [
    {
      id: "nexus.ui.command_center",
      fallbackSpan: 12,
      render: () => <PresenceCallDeck catalog={catalog} simulation={simulation} />,
    },
    {
      id: "nexus.ui.simulation_map",
      fallbackSpan: 12,
      className: "card everything-dashboard-card !mt-0 relative overflow-hidden",
      render: () => (
        <>
          <div className="everything-dashboard-beam" />
          <div className="everything-dashboard-header mb-6 px-1">
            <div className="flex flex-col xl:flex-row xl:items-end xl:justify-between gap-2 mb-3">
              <div>
                <p className="everything-dashboard-overline mb-1">Simulation Core</p>
                <h2 className="text-2xl md:text-3xl font-bold border-none pb-0 leading-tight">Everything Dashboard</h2>
              </div>
              <p className="text-xs text-muted font-mono bg-[rgba(0,0,0,0.2)] rounded px-2 py-1">
                file nodes <code>{Number(catalog?.file_graph?.stats?.file_count ?? 0)}</code> | crawler nodes
                <code>{Number(catalog?.crawler_graph?.stats?.crawler_count ?? 0)}</code> | artifacts
                <code>
                  {Number(catalog?.counts?.audio ?? 0) +
                    Number(catalog?.counts?.image ?? 0) +
                    Number(catalog?.counts?.video ?? 0)}
                </code>
              </p>
            </div>
          </div>

          <div className="everything-dashboard-canvas-wrap">
            <SimulationCanvas
              simulation={simulation}
              catalog={catalog}
              onOverlayInit={(api) => setOverlayApi(api)}
              height={simulationCanvasHeight}
            />
          </div>
        </>
      ),
    },
    {
      id: "nexus.ui.dedicated_views",
      fallbackSpan: 12,
      render: () => (
        <div className="mt-0 rounded-xl border border-[var(--line)] bg-[rgba(14,22,28,0.58)] p-3 h-full">
          <p className="text-[11px] uppercase tracking-[0.12em] text-[#9ec7dd]">Dedicated World Views</p>
          <p className="text-xs text-muted mt-1">Each overlay lane rendered as its own live viewport.</p>
          <div className="mt-3 grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
            {dedicatedOverlayViews.map((view) => (
              <section key={view.id} className="rounded-lg border border-[rgba(126,166,192,0.32)] bg-[rgba(10,18,28,0.72)] p-2">
                <div className="mb-2">
                  <p className="text-sm font-semibold text-[#e5f3ff]">{view.label}</p>
                  <p className="text-[11px] text-[#9fc4dd]">{view.description}</p>
                </div>
                <SimulationCanvas
                  simulation={simulation}
                  catalog={catalog}
                  height={180}
                  defaultOverlayView={view.id}
                  overlayViewLocked
                  compactHud
                  interactive={false}
                />
              </section>
            ))}
          </div>
        </div>
      ),
    },
    {
      id: "nexus.ui.chat.witness_thread",
      fallbackSpan: 6,
      render: () => (
        <div className="space-y-4 h-full">
          <div className="rounded-xl border border-[var(--line)] bg-[rgba(45,46,39,0.88)] px-4 py-3 text-sm text-muted">
            Communication mode active: sound production controls are retired. Use Presence Call Deck for
            WebRTC communication and this lane for text/voice messaging.
          </div>

          {chatLensState ? (
            <p className="text-xs text-muted font-mono">
              chat-lens mass <code>{chatLensState.mass.toFixed(2)}</code> | priority
              <code>{chatLensState.priority.toFixed(2)}</code> | reason
              <code>{chatLensState.explain.dominant_field}</code>
            </p>
          ) : null}

          <div
            style={{
              opacity: chatLensState ? projectionOpacity(chatLensState.opacity, 0.92) : 1,
              transform: chatLensState
                ? `scale(${(1 + chatLensState.pulse * 0.012).toFixed(3)})`
                : undefined,
              transformOrigin: "center top",
              transition: "transform 200ms ease, opacity 200ms ease",
            }}
          >
            <ChatPanel
              onSend={(text) => {
                if (handleAutopilotUserInput(text)) {
                  return;
                }
                setIsThinking(true);
                (async () => {
                  const consumed = await handleChatCommand(text);
                  if (consumed) {
                    return;
                  }

                  const baseUrl = runtimeBaseUrl();
                  const response = await fetch(`${baseUrl}/api/chat`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ messages: [{ role: "user", text }] }),
                  });
                  const payload = (await response.json()) as { reply?: string };
                  const reply = String(payload.reply ?? "");
                  window.dispatchEvent(
                    new CustomEvent("chat-message", {
                      detail: { role: "assistant", text: reply },
                    }),
                  );
                  if (reply.includes("[[PULSE]]")) {
                    overlayApi?.pulseAt?.(0.5, 0.5, 1.0);
                  }
                  if (reply.includes("[[SING]]")) {
                    overlayApi?.singAll?.();
                  }
                })()
                  .catch(() => {
                    window.dispatchEvent(
                      new CustomEvent("chat-message", {
                        detail: { role: "system", text: "chat request failed" },
                      }),
                    );
                  })
                  .finally(() => {
                    setIsThinking(false);
                  });
              }}
              onRecord={handleRecord}
              onTranscribe={handleTranscribe}
              onSendVoice={handleSendVoice}
              isRecording={isRecording}
              isThinking={isThinking}
              voiceInputMeta={voiceInputMeta}
            />
          </div>
        </div>
      ),
    },
    {
      id: "nexus.ui.web_graph_weaver",
      fallbackSpan: 6,
      render: () => deferredPanelsReady ? (
        <Suspense fallback={<DeferredPanelPlaceholder title="Web Graph Weaver" />}>
          <WebGraphWeaverPanel />
        </Suspense>
      ) : (
        <DeferredPanelPlaceholder title="Web Graph Weaver" />
      ),
    },
    {
      id: "nexus.ui.inspiration_atlas",
      fallbackSpan: 6,
      render: () => deferredPanelsReady ? (
        <Suspense fallback={<DeferredPanelPlaceholder title="Inspiration Atlas" />}>
          <InspirationAtlasPanel simulation={simulation} />
        </Suspense>
      ) : (
        <DeferredPanelPlaceholder title="Inspiration Atlas" />
      ),
    },
    {
      id: "nexus.ui.entity_vitals",
      fallbackSpan: 6,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#a6e22e] opacity-60" />
          <h2 className="text-3xl font-bold mb-2">Entity Vitals / 実体バイタル</h2>
          <p className="text-muted mb-6">Live telemetry from the canonical named forms.</p>
          <div className="max-h-[62rem] overflow-y-auto pr-1">
            {deferredPanelsReady ? (
              <Suspense fallback={<DeferredPanelPlaceholder title="Entity Vitals" />}>
                <VitalsPanel
                  entities={simulation?.entities}
                  catalog={catalog}
                  presenceDynamics={simulation?.presence_dynamics}
                />
              </Suspense>
            ) : (
              <DeferredPanelPlaceholder title="Entity Vitals" />
            )}
          </div>
        </>
      ),
    },
    {
      id: "nexus.ui.projection_ledger",
      fallbackSpan: 6,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#66d9ef] opacity-70" />
          <h2 className="text-2xl font-bold mb-2">Projection Ledger / 映台帳</h2>
          <p className="text-muted mb-4">Sub-panels expose routing and control data for every known box.</p>
          <div className="max-h-[74rem] overflow-y-auto pr-1">
            <ProjectionLedgerPanel projection={activeProjection} />
          </div>
        </>
      ),
    },
    {
      id: "nexus.ui.autopilot_ledger",
      fallbackSpan: 6,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#fd971f] opacity-70" />
          <h2 className="text-2xl font-bold mb-2">Autopilot Ledger / 自動操縦台帳</h2>
          <p className="text-muted mb-4">
            Replay stream of intent, confidence, risk, permissions, and result.
          </p>
          <div className="space-y-2 max-h-[26rem] overflow-y-auto pr-1">
            {autopilotEvents.length === 0 ? (
              <p className="text-xs text-muted">No autopilot events yet.</p>
            ) : (
              autopilotEvents.map((event, index) => (
                <div
                  key={`${event.ts}-${event.actionId}-${index}`}
                  className="border border-[var(--line)] rounded-lg bg-[rgba(45,46,39,0.86)] p-2"
                >
                  <p className="text-xs font-semibold text-ink">
                    <code>{event.intent}</code>{" -> "}<code>{event.actionId}</code>
                  </p>
                  <p className="text-[11px] text-muted font-mono">
                    confidence {event.confidence.toFixed(2)} | risk {event.risk.toFixed(2)} | result
                    <code>{event.result}</code>
                    {event.gate ? (
                      <>
                        {" "}| gate <code>{event.gate}</code>
                      </>
                    ) : null}
                  </p>
                  <p className="text-[11px] text-muted font-mono">
                    perms {event.perms.length > 0 ? event.perms.join(", ") : "(none)"}
                  </p>
                  <p className="text-[11px] text-muted">{event.summary}</p>
                </div>
              ))
            )}
          </div>
        </>
      ),
    },
    {
      id: "nexus.ui.stability_observatory",
      fallbackSpan: 6,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#66d9ef] opacity-70" />
          <h2 className="text-2xl font-bold mb-2">Stability Observatory / 安定観測</h2>
          <p className="text-muted mb-4">
            Evidence-first view for study mode: council, gates, queue, and drift movement.
          </p>
          {deferredPanelsReady ? (
            <Suspense fallback={<DeferredPanelPlaceholder title="Stability Observatory" />}>
              <StabilityObservatoryPanel catalog={catalog} simulation={simulation} />
            </Suspense>
          ) : (
            <DeferredPanelPlaceholder title="Stability Observatory" />
          )}
        </>
      ),
    },
    {
      id: "nexus.ui.omni_archive",
      fallbackSpan: 8,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#ae81ff] opacity-65" />
          <h2 className="text-3xl font-bold mb-2">Omni Panel / 全感覚パネル</h2>
          <p className="text-muted mb-6">Receipt River, Mage of Receipts, and other cover entities.</p>
          {deferredPanelsReady ? (
            <Suspense fallback={<DeferredPanelPlaceholder title="Omni Archive" />}>
              <OmniPanel catalog={catalog} />
            </Suspense>
          ) : (
            <DeferredPanelPlaceholder title="Omni Archive" />
          )}
          <div className="mt-8">
            <h3 className="text-2xl font-bold mb-4">Vault Artifacts / 遺物録</h3>
            {deferredPanelsReady ? (
              <Suspense fallback={<DeferredPanelPlaceholder title="Vault Artifacts" />}>
                <CatalogPanel catalog={catalog} />
              </Suspense>
            ) : (
              <DeferredPanelPlaceholder title="Vault Artifacts" />
            )}
          </div>
        </>
      ),
    },
    {
      id: "nexus.ui.myth_commons",
      fallbackSpan: 4,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#fd971f] opacity-70" />
          <h2 className="text-3xl font-bold mb-2">Myth Commons / 神話共同体</h2>
          <p className="text-muted mb-6">People sing, pray to the Presences, and keep writing the myth.</p>
          {deferredPanelsReady ? (
            <Suspense fallback={<DeferredPanelPlaceholder title="Myth Commons" />}>
              <MythWorldPanel
                simulation={simulation}
                interaction={worldInteraction}
                interactingPersonId={interactingPersonId}
                onInteract={handleWorldInteract}
              />
            </Suspense>
          ) : (
            <DeferredPanelPlaceholder title="Myth Commons" />
          )}
        </>
      ),
    },
  ], [
    activeProjection,
    autopilotEvents,
    catalog,
    chatLensState,
    dedicatedOverlayViews,
    deferredPanelsReady,
    handleAutopilotUserInput,
    handleChatCommand,
    handleRecord,
    handleSendVoice,
    handleTranscribe,
    handleWorldInteract,
    interactingPersonId,
    isRecording,
    isThinking,
    overlayApi,
    simulation,
    simulationCanvasHeight,
    voiceInputMeta,
    worldInteraction
  ]);

  const sortedPanels = useMemo(() => {
    return panelConfigs
      .map((config) => {
        const state = projectionStateByElement.get(config.id);
        const priority = state?.priority ?? 0.1; // Default low priority if unknown
        const style = projectionStyleFor(config.id, config.fallbackSpan);
        return { ...config, priority, style };
      })
      .sort((a, b) => b.priority - a.priority); // Sort DESC
  }, [panelConfigs, projectionStateByElement, projectionStyleFor]);

  return (
    <main className="max-w-[1920px] mx-auto px-2 py-2 md:px-4 md:py-4 pb-20 transition-colors">
      <header className="mb-4 border-b border-line pb-3 flex flex-col gap-2">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold tracking-tight text-ink flex items-center gap-3">
             <span className="opacity-50">ημ</span>
             <span>eta-mu world daemon</span>
          </h1>
          <div className="flex items-center gap-4">
            <p className="text-muted text-xs font-mono hidden md:block">
              Part <code>{catalog?.part_roots?.[0]?.split("/").pop() || "?"}</code>
            </p>
            {!isConnected ? (
              <span className="text-[#f92672] font-bold text-xs animate-pulse">● Disconnected</span>
            ) : (
              <span className="text-[#a6e22e] font-bold text-xs flex items-center gap-2">● Connected</span>
            )}
          </div>
        </div>
        
        <div className="grid gap-2 lg:grid-cols-[1fr_auto] lg:items-center">
          <div className="text-[10px] text-muted space-y-0.5 font-mono opacity-70">
            <div className="flex flex-wrap gap-x-3 gap-y-1">
               <span>perspective: <code>{projectionPerspective}</code></span>
               <span>autopilot: <code>{autopilotEnabled ? autopilotStatus : "stopped"}</code></span>
               <span className="opacity-80">note: <code>{autopilotSummary}</code></span>
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {activeChatLens ? (
                <span>chat-lens: <code>{activeChatLens.presence}</code> ({activeChatLens.status})</span>
              ) : null}
              {latestAutopilotEvent ? (
                 <span>last: <code>{latestAutopilotEvent.actionId}</code> ({latestAutopilotEvent.result})</span>
              ) : null}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={toggleAutopilot}
              className={`border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                autopilotEnabled
                  ? "bg-[rgba(166,226,46,0.16)] border-[rgba(166,226,46,0.48)] text-[#a6e22e]"
                  : "bg-[rgba(249,38,114,0.16)] border-[rgba(249,38,114,0.48)] text-[#f92672]"
              }`}
            >
              {autopilotEnabled ? "Autopilot On" : "Autopilot Off"}
            </button>
            {projectionOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setUiPerspective(option.id as UIPerspective)}
                className={`border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                  projectionPerspective === option.id
                    ? "bg-[rgba(102,217,239,0.2)] text-[#66d9ef] border-[rgba(102,217,239,0.7)]"
                    : "bg-[rgba(39,40,34,0.78)] text-[var(--ink)] border-[var(--line)] hover:bg-[rgba(55,56,48,0.92)]"
                }`}
                title={option.description}
              >
                {option.name}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="flex justify-end px-2 mb-2 sticky top-2 z-50">
        <button
          type="button"
          onClick={() => setIsEditMode(!isEditMode)}
          className={`text-sm font-bold px-4 py-2 rounded-lg shadow-lg transition-all duration-200 
            ${isEditMode 
              ? "bg-[#ae81ff] text-white ring-2 ring-white/20 scale-105" 
              : "bg-[rgba(45,46,39,0.9)] text-[#ae81ff] border border-[#ae81ff]/40 hover:bg-[#ae81ff]/10 hover:border-[#ae81ff]"
            }`}
        >
          {isEditMode ? "Done Editing" : "Edit Layout"}
        </button>
      </div>

      <div 
        ref={gridContainerRef}
        className="grid grid-cols-1 xl:grid-cols-12 xl:grid-flow-dense gap-3 items-start xl:auto-rows-[minmax(2.5rem,auto)] relative"
      >
        {sortedPanels.map((panel) => (
          <motion.section
            key={panel.id}
            className={`${panel.className ?? ""} ${isEditMode ? "cursor-grab active:cursor-grabbing ring-2 ring-[#ae81ff] shadow-[0_0_15px_rgba(174,129,255,0.3)] z-10" : ""}`}
            style={panel.style as CSSProperties}
            drag={isEditMode}
            dragMomentum={false}
            dragElastic={0.1}
            whileHover={isEditMode ? { scale: 1.02, zIndex: 20 } : undefined}
            whileTap={isEditMode ? { scale: 1.05, zIndex: 30, cursor: "grabbing" } : undefined}
            onDragEnd={(_, info) => handleDragEnd(_, info, panel.id)}
            layout
          >
            {panel.render()}
            {isEditMode && (
               <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 rounded text-[10px] text-[#ae81ff] font-mono pointer-events-none">
                 DRAG ME
               </div>
            )}
          </motion.section>
        ))}
      </div>

      {uiToasts.length > 0 ? (
        <div className="fixed bottom-4 right-4 z-[80] flex w-[min(92vw,360px)] flex-col gap-2">
          {uiToasts.map((toast) => (
            <div
              key={toast.id}
              className="rounded-lg border border-[rgba(102,217,239,0.45)] bg-[rgba(12,23,31,0.94)] px-3 py-2 shadow-[0_8px_24px_rgba(0,0,0,0.45)]"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[#9ec7dd]">
                    {toast.title}
                  </p>
                  <p className="text-sm text-[#e9f6ff] mt-1">{toast.body}</p>
                </div>
                <button
                  type="button"
                  onClick={() => dismissToast(toast.id)}
                  className="text-xs text-[#9ec7dd] hover:text-white transition-colors"
                >
                  dismiss
                </button>
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </main>
  );
}
