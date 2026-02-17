import {
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  lazy,
  Suspense,
  type ReactNode,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { motion, type PanInfo } from "framer-motion";
import { useWorldState } from "./hooks/useWorldState";
import { OVERLAY_VIEW_OPTIONS, SimulationCanvas, type OverlayViewId } from "./components/Simulation/Canvas";
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
  EntityManifestItem,
  FileGraphConceptPresence,
  FileGraphNode,
  NamedFieldItem,
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
const CORE_CAMERA_ZOOM_MIN = 0.6;
const CORE_CAMERA_ZOOM_MAX = 1.8;
const CORE_CAMERA_PITCH_MIN = -36;
const CORE_CAMERA_PITCH_MAX = 36;
const CORE_CAMERA_YAW_MIN = -52;
const CORE_CAMERA_YAW_MAX = 52;
const CORE_CAMERA_X_LIMIT = 860;
const CORE_CAMERA_Y_LIMIT = 560;
const CORE_CAMERA_Z_MIN = -520;
const CORE_CAMERA_Z_MAX = 460;
const CORE_FLIGHT_BASE_SPEED = 230;
const CORE_FLIGHT_SPEED_MIN = 0.55;
const CORE_FLIGHT_SPEED_MAX = 2.4;

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

function isTextEntryTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  if (target.isContentEditable) {
    return true;
  }
  const tagName = target.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
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

type PanelAnchorKind = "node" | "cluster" | "region";
type PanelPreferredSide = "left" | "right" | "top" | "bottom";
type PanelWorldSize = "s" | "m" | "l" | "xl";

interface PanelConfig {
  id: string;
  fallbackSpan: number;
  className?: string;
  anchorKind?: PanelAnchorKind;
  anchorId?: string;
  worldSize?: PanelWorldSize;
  pinnedByDefault?: boolean;
  render: () => ReactNode;
}

interface WorldAnchorTarget {
  kind: PanelAnchorKind;
  id: string;
  label: string;
  x: number;
  y: number;
  radius: number;
  hue: number;
  confidence: number;
  presenceSignature: Record<string, number>;
}

interface WorldPanelLayoutEntry {
  id: string;
  panel: PanelConfig & { priority: number; depth: number };
  anchor: WorldAnchorTarget;
  anchorScreenX: number;
  anchorScreenY: number;
  side: PanelPreferredSide;
  x: number;
  y: number;
  width: number;
  height: number;
  tetherX: number;
  tetherY: number;
  glow: number;
  collapse: boolean;
}

interface PanelAnchorPreset {
  kind: PanelAnchorKind;
  worldSize: PanelWorldSize;
  pinnedByDefault?: boolean;
  anchorId?: string;
}

const MAX_WORLD_PANELS_VISIBLE = 8;
const WORLD_PANEL_MARGIN = 16;

const PANEL_ANCHOR_PRESETS: Record<string, PanelAnchorPreset> = {
  "nexus.ui.command_center": {
    kind: "node",
    worldSize: "xl",
    pinnedByDefault: true,
    anchorId: "anchor_registry",
  },
  "nexus.ui.chat.witness_thread": {
    kind: "node",
    worldSize: "l",
    pinnedByDefault: true,
    anchorId: "witness_thread",
  },
  "nexus.ui.web_graph_weaver": {
    kind: "cluster",
    worldSize: "m",
  },
  "nexus.ui.inspiration_atlas": {
    kind: "cluster",
    worldSize: "m",
  },
  "nexus.ui.entity_vitals": {
    kind: "node",
    worldSize: "m",
  },
  "nexus.ui.projection_ledger": {
    kind: "region",
    worldSize: "m",
    pinnedByDefault: true,
  },
  "nexus.ui.autopilot_ledger": {
    kind: "region",
    worldSize: "m",
  },
  "nexus.ui.stability_observatory": {
    kind: "region",
    worldSize: "l",
    pinnedByDefault: true,
    anchorId: "gates_of_truth",
  },
  "nexus.ui.omni_archive": {
    kind: "cluster",
    worldSize: "l",
  },
  "nexus.ui.myth_commons": {
    kind: "region",
    worldSize: "m",
  },
  "nexus.ui.dedicated_views": {
    kind: "region",
    worldSize: "xl",
    anchorId: "anchor_registry",
  },
};

function defaultPinnedPanelMap(panelIds: string[]): Record<string, boolean> {
  const pinned: Record<string, boolean> = {};
  panelIds.forEach((panelId) => {
    pinned[panelId] = Boolean(PANEL_ANCHOR_PRESETS[panelId]?.pinnedByDefault);
  });
  return pinned;
}

function normalizeUnit(raw: number | undefined | null, fallback = 0.5): number {
  if (typeof raw !== "number" || Number.isNaN(raw)) {
    return fallback;
  }
  return clamp(raw, 0, 1);
}

function panelSizeForWorld(
  worldSize: PanelWorldSize,
  priority: number,
  zoom: number,
  speedNorm: number,
): { width: number; height: number; collapse: boolean } {
  const bySize: Record<PanelWorldSize, { w: number; h: number }> = {
    s: { w: 220, h: 150 },
    m: { w: 286, h: 196 },
    l: { w: 342, h: 232 },
    xl: { w: 418, h: 282 },
  };
  const base = bySize[worldSize];
  const zoomScale = clamp(0.9 + ((zoom - 1) * 0.42), 0.76, 1.28);
  const priorityScale = 0.9 + (priority * 0.34);
  const motionScale = 1 - (speedNorm * 0.22);
  const width = clamp(Math.round(base.w * zoomScale * priorityScale * motionScale), 188, 560);
  const height = clamp(Math.round(base.h * zoomScale * (0.94 + priority * 0.26) * motionScale), 126, 420);
  const collapse = speedNorm > 0.62;
  return { width: collapse ? Math.round(width * 0.58) : width, height: collapse ? 56 : height, collapse };
}

function preferredSideForAnchor(
  panelId: string,
  px: number,
  py: number,
  viewportWidth: number,
  viewportHeight: number,
  sideByPanel: Map<string, PanelPreferredSide>,
): PanelPreferredSide {
  const dx = px - (viewportWidth / 2);
  const dy = py - (viewportHeight / 2);
  const axisDominant = Math.abs(dx) > Math.abs(dy) * 1.2;
  const suggested: PanelPreferredSide = axisDominant
    ? (dx < 0 ? "left" : "right")
    : (dy < 0 ? "top" : "bottom");
  const previous = sideByPanel.get(panelId);
  if (!previous) {
    sideByPanel.set(panelId, suggested);
    return suggested;
  }
  const shouldFlip =
    (previous === "left" && dx > 88)
    || (previous === "right" && dx < -88)
    || (previous === "top" && dy > 64)
    || (previous === "bottom" && dy < -64);
  if (shouldFlip) {
    sideByPanel.set(panelId, suggested);
    return suggested;
  }
  return previous;
}

function anchorOffsetForSide(side: PanelPreferredSide): { x: number; y: number } {
  if (side === "left") {
    return { x: -34, y: -8 };
  }
  if (side === "right") {
    return { x: 34, y: -8 };
  }
  if (side === "top") {
    return { x: 0, y: -32 };
  }
  return { x: 0, y: 32 };
}

function overlapAmount(a: WorldPanelLayoutEntry, b: WorldPanelLayoutEntry): { x: number; y: number } | null {
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;
  const overlapX = Math.min(ax2, bx2) - Math.max(a.x, b.x);
  const overlapY = Math.min(ay2, by2) - Math.max(a.y, b.y);
  if (overlapX <= 0 || overlapY <= 0) {
    return null;
  }
  return { x: overlapX, y: overlapY };
}

function containsAnchorNoCoverZone(panel: WorldPanelLayoutEntry, radius = 28): boolean {
  const cx = panel.anchorScreenX;
  const cy = panel.anchorScreenY;
  return (
    cx >= panel.x - radius
    && cx <= panel.x + panel.width + radius
    && cy >= panel.y - radius
    && cy <= panel.y + panel.height + radius
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
  const panelSideRef = useRef<Map<string, PanelPreferredSide>>(new Map());
  const panelScreenRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const cameraFlightRef = useRef<number | null>(null);
  const coreDragRef = useRef<{
    active: boolean;
    pointerId: number;
    mode: "orbit" | "pan";
    startX: number;
    startY: number;
    startPitch: number;
    startYaw: number;
    startCamX: number;
    startCamY: number;
  } | null>(null);
  const coreFlightKeysRef = useRef<Record<string, boolean>>({
    w: false,
    a: false,
    s: false,
    d: false,
    r: false,
    f: false,
    shift: false,
  });
  const coreFlightVelocityRef = useRef({ x: 0, y: 0, z: 0 });

  const [layoutOverrides, setLayoutOverrides] = useState<Record<string, { x: number; y: number; w: number; h: number }>>({});
  const [panelScreenBiases, setPanelScreenBiases] = useState<Record<string, { x: number; y: number }>>({});
  const [selectedPanelId, setSelectedPanelId] = useState<string | null>(null);
  const [hoveredPanelId, setHoveredPanelId] = useState<string | null>(null);
  const [pinnedPanels, setPinnedPanels] = useState<Record<string, boolean>>(() =>
    defaultPinnedPanelMap(Object.keys(PANEL_ANCHOR_PRESETS)),
  );
  const [isEditMode, setIsEditMode] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth);
  const [viewportHeight, setViewportHeight] = useState(() => window.innerHeight);
  const [coreCameraZoom, setCoreCameraZoom] = useState(1);
  const [coreCameraPitch, setCoreCameraPitch] = useState(10);
  const [coreCameraYaw, setCoreCameraYaw] = useState(-12);
  const [coreCameraPosition, setCoreCameraPosition] = useState({ x: 0, y: 0, z: 0 });
  const [coreOverlayView, setCoreOverlayView] = useState<OverlayViewId>("omni");
  const [coreFlightEnabled, setCoreFlightEnabled] = useState(true);
  const [coreFlightSpeed, setCoreFlightSpeed] = useState(1);

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
    const handleResize = () => {
      setViewportWidth(window.innerWidth);
      setViewportHeight(window.innerHeight);
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
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

  const nudgeCoreZoom = useCallback((delta: number) => {
    setCoreCameraZoom((prev) => clamp(prev + delta, CORE_CAMERA_ZOOM_MIN, CORE_CAMERA_ZOOM_MAX));
  }, []);

  const nudgeCorePitch = useCallback((delta: number) => {
    setCoreCameraPitch((prev) => clamp(prev + delta, CORE_CAMERA_PITCH_MIN, CORE_CAMERA_PITCH_MAX));
  }, []);

  const nudgeCoreYaw = useCallback((delta: number) => {
    setCoreCameraYaw((prev) => clamp(prev + delta, CORE_CAMERA_YAW_MIN, CORE_CAMERA_YAW_MAX));
  }, []);

  const toggleCoreFlight = useCallback(() => {
    setCoreFlightEnabled((prev) => !prev);
  }, []);

  const nudgeCoreFlightSpeed = useCallback((delta: number) => {
    setCoreFlightSpeed((prev) => clamp(prev + delta, CORE_FLIGHT_SPEED_MIN, CORE_FLIGHT_SPEED_MAX));
  }, []);

  const togglePanelPin = useCallback((panelId: string) => {
    setPinnedPanels((prev) => ({
      ...prev,
      [panelId]: !prev[panelId],
    }));
  }, []);

  const stopCameraFlight = useCallback(() => {
    if (cameraFlightRef.current !== null) {
      window.cancelAnimationFrame(cameraFlightRef.current);
      cameraFlightRef.current = null;
    }
  }, []);

  const resetCoreCamera = useCallback(() => {
    stopCameraFlight();
    setCoreCameraZoom(1);
    setCoreCameraPitch(10);
    setCoreCameraYaw(-12);
    setCoreCameraPosition({ x: 0, y: 0, z: 0 });
    coreFlightVelocityRef.current = { x: 0, y: 0, z: 0 };
  }, [stopCameraFlight]);

  const flyCameraToAnchor = useCallback((anchor: WorldAnchorTarget) => {
    stopCameraFlight();
    const start = {
      x: coreCameraPosition.x,
      y: coreCameraPosition.y,
      z: coreCameraPosition.z,
      yaw: coreCameraYaw,
      pitch: coreCameraPitch,
      zoom: coreCameraZoom,
    };
    const target = {
      x: clamp((0.5 - anchor.x) * 640, -CORE_CAMERA_X_LIMIT, CORE_CAMERA_X_LIMIT),
      y: clamp((0.5 - anchor.y) * 520, -CORE_CAMERA_Y_LIMIT, CORE_CAMERA_Y_LIMIT),
      z: clamp(
        anchor.kind === "node" ? 180 : anchor.kind === "cluster" ? 40 : -120,
        CORE_CAMERA_Z_MIN,
        CORE_CAMERA_Z_MAX,
      ),
      yaw: clamp((anchor.x - 0.5) * 68, CORE_CAMERA_YAW_MIN, CORE_CAMERA_YAW_MAX),
      pitch: clamp((0.5 - anchor.y) * 52, CORE_CAMERA_PITCH_MIN, CORE_CAMERA_PITCH_MAX),
      zoom: clamp(anchor.kind === "node" ? 1.18 : anchor.kind === "cluster" ? 1.06 : 0.94, CORE_CAMERA_ZOOM_MIN, CORE_CAMERA_ZOOM_MAX),
    };
    const startTs = performance.now();
    const durationMs = 760;
    const ease = (t: number) => 1 - ((1 - t) ** 3);

    const tick = (ts: number) => {
      const elapsed = ts - startTs;
      const t = clamp(elapsed / durationMs, 0, 1);
      const mix = ease(t);
      setCoreCameraPosition({
        x: start.x + ((target.x - start.x) * mix),
        y: start.y + ((target.y - start.y) * mix),
        z: start.z + ((target.z - start.z) * mix),
      });
      setCoreCameraYaw(start.yaw + ((target.yaw - start.yaw) * mix));
      setCoreCameraPitch(start.pitch + ((target.pitch - start.pitch) * mix));
      setCoreCameraZoom(start.zoom + ((target.zoom - start.zoom) * mix));
      if (t >= 1) {
        cameraFlightRef.current = null;
        return;
      }
      cameraFlightRef.current = window.requestAnimationFrame(tick);
    };
    cameraFlightRef.current = window.requestAnimationFrame(tick);
  }, [coreCameraPitch, coreCameraPosition.x, coreCameraPosition.y, coreCameraPosition.z, coreCameraYaw, coreCameraZoom, stopCameraFlight]);

  useEffect(() => {
    return () => {
      stopCameraFlight();
    };
  }, [stopCameraFlight]);

  const coreCameraTransform = useMemo(
    () =>
      `perspective(1800px) translate3d(${coreCameraPosition.x.toFixed(1)}px, ${coreCameraPosition.y.toFixed(1)}px, ${coreCameraPosition.z.toFixed(1)}px) rotateX(${coreCameraPitch.toFixed(2)}deg) rotateY(${coreCameraYaw.toFixed(2)}deg) scale(${coreCameraZoom.toFixed(3)})`,
    [coreCameraPitch, coreCameraPosition, coreCameraYaw, coreCameraZoom],
  );

  const handleCorePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (isTextEntryTarget(event.target)) {
        return;
      }
      const mode = event.shiftKey || event.button === 1 ? "pan" : "orbit";
      coreDragRef.current = {
        active: true,
        pointerId: event.pointerId,
        mode,
        startX: event.clientX,
        startY: event.clientY,
        startPitch: coreCameraPitch,
        startYaw: coreCameraYaw,
        startCamX: coreCameraPosition.x,
        startCamY: coreCameraPosition.y,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [coreCameraPitch, coreCameraPosition.x, coreCameraPosition.y, coreCameraYaw],
  );

  const handleCorePointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = coreDragRef.current;
    if (!drag || !drag.active || drag.pointerId !== event.pointerId) {
      return;
    }
    const dx = event.clientX - drag.startX;
    const dy = event.clientY - drag.startY;
    if (drag.mode === "pan") {
      setCoreCameraPosition((prev) => ({
        x: clamp(drag.startCamX + (dx * 0.9), -CORE_CAMERA_X_LIMIT, CORE_CAMERA_X_LIMIT),
        y: clamp(drag.startCamY + (dy * 0.9), -CORE_CAMERA_Y_LIMIT, CORE_CAMERA_Y_LIMIT),
        z: prev.z,
      }));
      return;
    }
    setCoreCameraYaw(clamp(drag.startYaw + (dx * 0.08), CORE_CAMERA_YAW_MIN, CORE_CAMERA_YAW_MAX));
    setCoreCameraPitch(clamp(drag.startPitch + (dy * 0.08), CORE_CAMERA_PITCH_MIN, CORE_CAMERA_PITCH_MAX));
  }, []);

  const handleCorePointerUp = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = coreDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    coreDragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
  }, []);

  const handleCoreWheel = useCallback((event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (event.shiftKey) {
      const speedDelta = event.deltaY < 0 ? 0.08 : -0.08;
      setCoreFlightSpeed((prev) => clamp(prev + speedDelta, CORE_FLIGHT_SPEED_MIN, CORE_FLIGHT_SPEED_MAX));
      return;
    }
    const delta = event.deltaY < 0 ? 0.06 : -0.06;
    setCoreCameraZoom((prev) => clamp(prev + delta, CORE_CAMERA_ZOOM_MIN, CORE_CAMERA_ZOOM_MAX));
  }, []);

  useEffect(() => {
    const keyFromEvent = (event: KeyboardEvent): string | null => {
      const key = event.key.toLowerCase();
      if (key === "w" || key === "a" || key === "s" || key === "d" || key === "r" || key === "f") {
        return key;
      }
      if (key === "shift") {
        return "shift";
      }
      return null;
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (isTextEntryTarget(event.target)) {
        return;
      }
      const mapped = keyFromEvent(event);
      if (!mapped) {
        return;
      }
      coreFlightKeysRef.current[mapped] = true;
      if (coreFlightEnabled) {
        event.preventDefault();
      }
    };

    const onKeyUp = (event: KeyboardEvent) => {
      const mapped = keyFromEvent(event);
      if (!mapped) {
        return;
      }
      coreFlightKeysRef.current[mapped] = false;
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [coreFlightEnabled]);

  useEffect(() => {
    if (!coreFlightEnabled) {
      coreFlightKeysRef.current = {
        w: false,
        a: false,
        s: false,
        d: false,
        r: false,
        f: false,
        shift: false,
      };
      coreFlightVelocityRef.current = { x: 0, y: 0, z: 0 };
      return;
    }

    let rafId = 0;
    let lastTs = performance.now();

    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - lastTs) / 1000);
      lastTs = now;
      const keys = coreFlightKeysRef.current;
      const yawRadians = (coreCameraYaw * Math.PI) / 180;
      const boost = keys.shift ? 2.2 : 1;
      const accel = CORE_FLIGHT_BASE_SPEED * coreFlightSpeed * boost;

      const strafe = (keys.d ? 1 : 0) - (keys.a ? 1 : 0);
      const climb = (keys.f ? 1 : 0) - (keys.r ? 1 : 0);
      const thrust = (keys.w ? 1 : 0) - (keys.s ? 1 : 0);

      const forwardX = Math.sin(yawRadians);
      const forwardZ = Math.cos(yawRadians);
      const rightX = Math.cos(yawRadians);
      const rightZ = -Math.sin(yawRadians);

      const velocity = coreFlightVelocityRef.current;
      velocity.x = (velocity.x * 0.88) + ((forwardX * thrust + rightX * strafe) * accel * dt);
      velocity.y = (velocity.y * 0.88) + (climb * accel * dt);
      velocity.z = (velocity.z * 0.88) + ((forwardZ * thrust + rightZ * strafe) * accel * dt);

      setCoreCameraPosition((prev) => ({
        x: clamp(prev.x + velocity.x, -CORE_CAMERA_X_LIMIT, CORE_CAMERA_X_LIMIT),
        y: clamp(prev.y + velocity.y, -CORE_CAMERA_Y_LIMIT, CORE_CAMERA_Y_LIMIT),
        z: clamp(prev.z + velocity.z, CORE_CAMERA_Z_MIN, CORE_CAMERA_Z_MAX),
      }));

      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, [coreCameraYaw, coreFlightEnabled, coreFlightSpeed]);

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

  const projectionElementById = useMemo(() => {
    const map = new Map<string, { presence?: string; binds_to?: string[]; kind?: string }>();
    activeProjection?.elements.forEach((element) => {
      map.set(element.id, {
        presence: element.presence,
        binds_to: element.binds_to,
        kind: element.kind,
      });
    });
    return map;
  }, [activeProjection?.elements]);

  const presenceAnchors = useMemo(() => {
    const map = new Map<string, WorldAnchorTarget>();
    (catalog?.entity_manifest ?? []).forEach((item: EntityManifestItem) => {
      const id = String(item.id || "").trim();
      if (!id) {
        return;
      }
      const x = normalizeUnit(item.x, Number.NaN);
      const y = normalizeUnit(item.y, Number.NaN);
      if (Number.isNaN(x) || Number.isNaN(y)) {
        return;
      }
      map.set(id, {
        kind: "node",
        id,
        label: item.en || id,
        x,
        y,
        hue: Number(item.hue ?? 210),
        radius: 0.08,
        confidence: 1,
        presenceSignature: { [id]: 1 },
      });
    });
    if (!map.has("anchor_registry")) {
      map.set("anchor_registry", {
        kind: "node",
        id: "anchor_registry",
        label: "Anchor Registry",
        x: 0.5,
        y: 0.5,
        hue: 184,
        radius: 0.08,
        confidence: 0.6,
        presenceSignature: { anchor_registry: 1 },
      });
    }
    map.set("particle_field", {
      kind: "node",
      id: "particle_field",
      label: "Particle Field",
      x: 0.5,
      y: 0.5,
      hue: 204,
      radius: 0.08,
      confidence: 0.52,
      presenceSignature: { particle_field: 1 },
    });
    return map;
  }, [catalog?.entity_manifest]);

  const fieldRegionAnchors = useMemo(() => {
    const map = new Map<string, WorldAnchorTarget>();
    const pushNode = (node: { id?: string; field?: string; label?: string; x?: number; y?: number; hue?: number }) => {
      const fieldKey = String(node.field || "").trim();
      const nodeId = String(node.id || "").trim();
      const regionKey = fieldKey || nodeId.replace(/^field:/, "");
      if (!regionKey) {
        return;
      }
      const x = normalizeUnit(node.x, Number.NaN);
      const y = normalizeUnit(node.y, Number.NaN);
      if (Number.isNaN(x) || Number.isNaN(y)) {
        return;
      }
      const region: WorldAnchorTarget = {
        kind: "region",
        id: regionKey,
        label: String(node.label || regionKey),
        x,
        y,
        radius: 0.16,
        hue: Number(node.hue ?? 196),
        confidence: 0.72,
        presenceSignature: {
          [regionKey]: 1,
          ...(fieldKey ? { [`field:${fieldKey}`]: 1 } : {}),
        },
      };
      map.set(regionKey, region);
      if (fieldKey) {
        map.set(fieldKey, region);
      }
    };
    (catalog?.file_graph?.field_nodes ?? []).forEach((node) => {
      pushNode(node);
    });
    (catalog?.crawler_graph?.field_nodes ?? []).forEach((node) => {
      pushNode(node);
    });
    return map;
  }, [catalog?.crawler_graph?.field_nodes, catalog?.file_graph?.field_nodes]);

  const namedRegionAnchors = useMemo(() => {
    const map = new Map<string, WorldAnchorTarget>();
    (catalog?.named_fields ?? []).forEach((field: NamedFieldItem) => {
      const id = String(field.id || "").trim();
      if (!id) {
        return;
      }
      const x = normalizeUnit(field.x, Number.NaN);
      const y = normalizeUnit(field.y, Number.NaN);
      if (Number.isNaN(x) || Number.isNaN(y)) {
        return;
      }
      map.set(id, {
        kind: "region",
        id,
        label: String(field.en || field.ja || id),
        x,
        y,
        radius: 0.2,
        hue: Number(field.hue ?? 202),
        confidence: 0.8,
        presenceSignature: { [id]: 1 },
      });
    });
    return map;
  }, [catalog?.named_fields]);

  const fileNodeById = useMemo(() => {
    const map = new Map<string, FileGraphNode>();
    const nodes = simulation?.file_graph?.file_nodes ?? catalog?.file_graph?.file_nodes ?? [];
    nodes.forEach((node) => {
      const id = String(node.id || "").trim();
      if (!id) {
        return;
      }
      map.set(id, node);
    });
    return map;
  }, [catalog?.file_graph?.file_nodes, simulation?.file_graph?.file_nodes]);

  const clusterAnchors = useMemo(() => {
    const map = new Map<string, WorldAnchorTarget>();
    const clusters: FileGraphConceptPresence[] =
      simulation?.file_graph?.concept_presences
      ?? catalog?.file_graph?.concept_presences
      ?? [];
    clusters.forEach((cluster) => {
      const clusterId = String(cluster.id || cluster.cluster_id || "").trim();
      if (!clusterId) {
        return;
      }
      const x = normalizeUnit(cluster.x, Number.NaN);
      const y = normalizeUnit(cluster.y, Number.NaN);
      if (Number.isNaN(x) || Number.isNaN(y)) {
        return;
      }
      const signature: Record<string, number> = {};
      const createdBy = String(cluster.created_by || "").trim();
      if (createdBy) {
        signature[createdBy] = clamp(Number(cluster.cohesion ?? 0.52), 0.12, 1);
      }
      const fieldScores = new Map<string, number>();
      (cluster.members ?? []).forEach((memberId) => {
        const node = fileNodeById.get(String(memberId));
        const dominantField = String(node?.dominant_field ?? "").trim();
        if (!dominantField) {
          return;
        }
        fieldScores.set(dominantField, (fieldScores.get(dominantField) ?? 0) + 1);
      });
      fieldScores.forEach((value, key) => {
        signature[`field:${key}`] = value;
      });

      let signatureTotal = 0;
      Object.values(signature).forEach((value) => {
        signatureTotal += value;
      });
      if (signatureTotal > 0) {
        Object.keys(signature).forEach((key) => {
          signature[key] = signature[key] / signatureTotal;
        });
      }

      const clusterAnchor: WorldAnchorTarget = {
        kind: "cluster",
        id: clusterId,
        label: String(cluster.label || cluster.label_ja || clusterId),
        x,
        y,
        radius: clamp(0.08 + (Number(cluster.file_count ?? 0) * 0.0028) + (Number(cluster.cohesion ?? 0.2) * 0.14), 0.1, 0.24),
        hue: Number(cluster.hue ?? 276),
        confidence: clamp(Number(cluster.cohesion ?? 0.5), 0.16, 1),
        presenceSignature: signature,
      };
      map.set(clusterId, clusterAnchor);
      const legacyClusterId = String(cluster.cluster_id || "").trim();
      if (legacyClusterId) {
        map.set(legacyClusterId, clusterAnchor);
      }
    });
    return map;
  }, [catalog?.file_graph?.concept_presences, fileNodeById, simulation?.file_graph?.concept_presences]);

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

  const panelConfigs = useMemo<PanelConfig[]>(() => [
    {
      id: "nexus.ui.command_center",
      fallbackSpan: 12,
      render: () => <PresenceCallDeck catalog={catalog} simulation={simulation} />,
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
    voiceInputMeta,
    worldInteraction
  ]);

  const sortedPanels = useMemo(() => {
    return panelConfigs
      .filter((config) => config.id !== "nexus.ui.simulation_map")
      .map((config) => {
        const state = projectionStateByElement.get(config.id);
        const preset = PANEL_ANCHOR_PRESETS[config.id];
        const priority = state?.priority ?? 0.1;
        const style = projectionStyleFor(config.id, config.fallbackSpan);
        const depth = Math.round(clamp(priority, 0, 1) * 160) + 24;
        return {
          ...config,
          anchorKind: config.anchorKind ?? preset?.kind ?? "node",
          anchorId: config.anchorId ?? preset?.anchorId,
          worldSize: config.worldSize ?? preset?.worldSize ?? "m",
          pinnedByDefault: config.pinnedByDefault ?? preset?.pinnedByDefault ?? false,
          priority,
          style,
          depth,
        };
      })
      .sort((a, b) => b.priority - a.priority);
  }, [panelConfigs, projectionStateByElement, projectionStyleFor]);

  const panelAnchorById = useMemo(() => {
    const map = new Map<string, WorldAnchorTarget>();
    const uniqueClusters = Array.from(clusterAnchors.values()).filter(
      (anchor, index, list) => list.findIndex((entry) => entry.id === anchor.id) === index,
    );
    const regionSource = new Map<string, WorldAnchorTarget>();
    namedRegionAnchors.forEach((anchor, key) => {
      regionSource.set(key, anchor);
    });
    fieldRegionAnchors.forEach((anchor, key) => {
      if (!regionSource.has(key)) {
        regionSource.set(key, anchor);
      }
    });

    const uniqueRegions = Array.from(regionSource.values()).filter(
      (anchor, index, list) => list.findIndex((entry) => entry.id === anchor.id) === index,
    );

    const nearestTo = (x: number, y: number, pool: WorldAnchorTarget[]) => {
      if (pool.length === 0) {
        return null;
      }
      let best = pool[0];
      let bestDistance = Number.POSITIVE_INFINITY;
      pool.forEach((anchor) => {
        const distance = Math.hypot(anchor.x - x, anchor.y - y);
        if (distance < bestDistance) {
          best = anchor;
          bestDistance = distance;
        }
      });
      return best;
    };

    sortedPanels.forEach((panel) => {
      const element = projectionElementById.get(panel.id);
      const state = projectionStateByElement.get(panel.id);
      const dominantField = String(state?.explain?.dominant_field ?? "").trim();
      const preferredPresence =
        panel.anchorId
        ?? element?.presence
        ?? (panel.id === "nexus.ui.dedicated_views" ? "anchor_registry" : "particle_field");
      const nodeAnchor =
        presenceAnchors.get(preferredPresence)
        ?? presenceAnchors.get("anchor_registry")
        ?? {
          kind: "node",
          id: "anchor_registry",
          label: "Anchor Registry",
          x: 0.5,
          y: 0.5,
          radius: 0.08,
          hue: 188,
          confidence: 0.4,
          presenceSignature: { anchor_registry: 1 },
        };

      const regionAnchorByField = dominantField ? fieldRegionAnchors.get(dominantField) : null;
      const regionAnchorByPresence = namedRegionAnchors.get(preferredPresence) ?? fieldRegionAnchors.get(preferredPresence);
      const nearestRegion = nearestTo(nodeAnchor.x, nodeAnchor.y, uniqueRegions);
      const regionAnchor =
        (panel.anchorId ? namedRegionAnchors.get(panel.anchorId) ?? fieldRegionAnchors.get(panel.anchorId) : null)
        ?? regionAnchorByField
        ?? regionAnchorByPresence
        ?? nearestRegion
        ?? nodeAnchor;

      const nearestCluster = (() => {
        if (uniqueClusters.length === 0) {
          return null;
        }
        let best: WorldAnchorTarget | null = null;
        let bestScore = Number.NEGATIVE_INFINITY;
        uniqueClusters.forEach((cluster) => {
          const distance = Math.hypot(cluster.x - nodeAnchor.x, cluster.y - nodeAnchor.y);
          const distanceScore = 1 - clamp(distance / 0.88, 0, 1);
          const presenceScore = cluster.presenceSignature[preferredPresence] ?? 0;
          const fieldScore = dominantField ? (cluster.presenceSignature[`field:${dominantField}`] ?? 0) : 0;
          const score = (distanceScore * 0.62) + (presenceScore * 0.26) + (fieldScore * 0.12);
          if (score > bestScore) {
            best = cluster;
            bestScore = score;
          }
        });
        return best;
      })();

      let resolvedAnchor: WorldAnchorTarget;
      if (panel.anchorKind === "cluster") {
        resolvedAnchor =
          (panel.anchorId ? clusterAnchors.get(panel.anchorId) : null)
          ?? nearestCluster
          ?? regionAnchor
          ?? nodeAnchor;
      } else if (panel.anchorKind === "region") {
        resolvedAnchor = regionAnchor ?? nodeAnchor;
      } else {
        resolvedAnchor = nodeAnchor;
      }

      map.set(panel.id, {
        ...resolvedAnchor,
        confidence: clamp(
          (resolvedAnchor.confidence * 0.78)
          + (panel.priority * 0.22),
          0.1,
          1,
        ),
      });
    });
    return map;
  }, [
    clusterAnchors,
    fieldRegionAnchors,
    namedRegionAnchors,
    presenceAnchors,
    projectionElementById,
    projectionStateByElement,
    sortedPanels,
  ]);

  const visiblePanelIds = useMemo(() => {
    const selected = new Set<string>();
    const include = (panelId: string | null | undefined) => {
      if (!panelId) {
        return;
      }
      if (!sortedPanels.some((panel) => panel.id === panelId)) {
        return;
      }
      selected.add(panelId);
    };
    include(selectedPanelId);
    include(hoveredPanelId);
    sortedPanels.forEach((panel) => {
      if (pinnedPanels[panel.id]) {
        selected.add(panel.id);
      }
    });
    sortedPanels.forEach((panel) => {
      if (selected.size < MAX_WORLD_PANELS_VISIBLE) {
        selected.add(panel.id);
      }
    });
    return Array.from(selected).slice(0, MAX_WORLD_PANELS_VISIBLE);
  }, [hoveredPanelId, pinnedPanels, selectedPanelId, sortedPanels]);

  const worldPanelLayout = useMemo<WorldPanelLayoutEntry[]>(() => {
    if (!isWideViewport) {
      return [];
    }
    const panelsById = new Map(sortedPanels.map((panel) => [panel.id, panel]));
    const velocity = coreFlightVelocityRef.current;
    const speedNorm = clamp(Math.hypot(velocity.x, velocity.y, velocity.z) / 26, 0, 1);
    const stageTop = 118;
    const stageBottom = viewportHeight - 14;
    const stageHeight = Math.max(120, stageBottom - stageTop);
    const centerX = viewportWidth / 2;
    const centerY = stageTop + (stageHeight / 2);
    const yaw = (coreCameraYaw * Math.PI / 180) * 0.72;
    const pitch = (coreCameraPitch * Math.PI / 180) * 0.68;
    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);
    const cosPitch = Math.cos(pitch);
    const sinPitch = Math.sin(pitch);

    const projectAnchor = (anchor: WorldAnchorTarget) => {
      const zBase = anchor.kind === "node" ? 0.62 : anchor.kind === "cluster" ? 0.24 : -0.14;
      let wx = (anchor.x - 0.5) * 2.25;
      let wy = (anchor.y - 0.5) * 1.86;
      let wz = zBase;

      wx -= coreCameraPosition.x / 660;
      wy -= coreCameraPosition.y / 560;
      wz -= coreCameraPosition.z / 920;

      const x1 = (wx * cosYaw) - (wz * sinYaw);
      const z1 = (wx * sinYaw) + (wz * cosYaw);
      const y1 = (wy * cosPitch) - (z1 * sinPitch);
      const z2 = (wy * sinPitch) + (z1 * cosPitch);
      const perspective = clamp(1 / (1 + (z2 * 0.7)), 0.46, 1.9);

      return {
        x: centerX + (x1 * viewportWidth * 0.34 * perspective * coreCameraZoom),
        y: centerY + (y1 * stageHeight * 0.47 * perspective * coreCameraZoom),
        perspective,
      };
    };

    const entries: WorldPanelLayoutEntry[] = [];
    visiblePanelIds.forEach((panelId) => {
      const panel = panelsById.get(panelId);
      if (!panel) {
        return;
      }
      const anchor = panelAnchorById.get(panelId);
      if (!anchor) {
        return;
      }
      const projected = projectAnchor(anchor);
      const side = preferredSideForAnchor(
        panelId,
        projected.x,
        projected.y,
        viewportWidth,
        viewportHeight,
        panelSideRef.current,
      );
      const sideOffset = anchorOffsetForSide(side);
      const size = panelSizeForWorld(panel.worldSize ?? "m", panel.priority, coreCameraZoom, speedNorm);
      const bias = panelScreenBiases[panelId] ?? { x: 0, y: 0 };

      let x = projected.x + sideOffset.x + bias.x;
      let y = projected.y + sideOffset.y + bias.y;
      if (side === "left") {
        x -= size.width;
        y -= size.height * 0.5;
      } else if (side === "right") {
        y -= size.height * 0.5;
      } else if (side === "top") {
        x -= size.width * 0.5;
        y -= size.height;
      } else {
        x -= size.width * 0.5;
      }

      const glow = selectedPanelId === panelId
        ? 0.96
        : hoveredPanelId === panelId
          ? 0.88
          : pinnedPanels[panelId]
            ? 0.72
            : clamp(0.44 + (panel.priority * 0.38), 0.4, 0.78);

      entries.push({
        id: panel.id,
        panel,
        anchor,
        anchorScreenX: projected.x,
        anchorScreenY: projected.y,
        side,
        x,
        y,
        width: size.width,
        height: size.height,
        tetherX: projected.x,
        tetherY: projected.y,
        glow,
        collapse: size.collapse,
      });
    });

    const clampEntry = (entry: WorldPanelLayoutEntry) => {
      entry.x = clamp(entry.x, WORLD_PANEL_MARGIN, viewportWidth - entry.width - WORLD_PANEL_MARGIN);
      entry.y = clamp(entry.y, stageTop, stageBottom - entry.height);
    };

    const updateTether = (entry: WorldPanelLayoutEntry) => {
      if (entry.side === "left") {
        entry.tetherX = entry.x + entry.width;
        entry.tetherY = clamp(entry.anchorScreenY, entry.y + 14, entry.y + entry.height - 14);
      } else if (entry.side === "right") {
        entry.tetherX = entry.x;
        entry.tetherY = clamp(entry.anchorScreenY, entry.y + 14, entry.y + entry.height - 14);
      } else if (entry.side === "top") {
        entry.tetherX = clamp(entry.anchorScreenX, entry.x + 14, entry.x + entry.width - 14);
        entry.tetherY = entry.y + entry.height;
      } else {
        entry.tetherX = clamp(entry.anchorScreenX, entry.x + 14, entry.x + entry.width - 14);
        entry.tetherY = entry.y;
      }
    };

    for (let pass = 0; pass < 6; pass += 1) {
      for (let i = 0; i < entries.length; i += 1) {
        for (let j = i + 1; j < entries.length; j += 1) {
          const a = entries[i];
          const b = entries[j];
          const overlap = overlapAmount(a, b);
          if (!overlap) {
            continue;
          }
          if (overlap.x < overlap.y) {
            const push = (overlap.x / 2) + 2;
            const direction = (a.x + (a.width / 2)) <= (b.x + (b.width / 2)) ? -1 : 1;
            a.x += direction * push;
            b.x -= direction * push;
          } else {
            const push = (overlap.y / 2) + 2;
            const direction = (a.y + (a.height / 2)) <= (b.y + (b.height / 2)) ? -1 : 1;
            a.y += direction * push;
            b.y -= direction * push;
          }
        }
      }
      entries.forEach((entry) => {
        if (containsAnchorNoCoverZone(entry, Math.max(20, entry.anchor.radius * Math.min(viewportWidth, stageHeight)))) {
          const centerRectX = entry.x + (entry.width / 2);
          const centerRectY = entry.y + (entry.height / 2);
          const dx = centerRectX - entry.anchorScreenX;
          const dy = centerRectY - entry.anchorScreenY;
          const distance = Math.hypot(dx, dy) || 0.0001;
          const push = 7 + (pass * 1.5);
          entry.x += (dx / distance) * push;
          entry.y += (dy / distance) * push;
        }
        clampEntry(entry);
      });
    }

    const smoothAlpha = clamp(0.26 - (speedNorm * 0.16), 0.09, 0.26);
    entries.forEach((entry) => {
      const previous = panelScreenRef.current.get(entry.id);
      if (previous) {
        entry.x = previous.x + ((entry.x - previous.x) * smoothAlpha);
        entry.y = previous.y + ((entry.y - previous.y) * smoothAlpha);
      }
      panelScreenRef.current.set(entry.id, { x: entry.x, y: entry.y });
      clampEntry(entry);
      updateTether(entry);
    });

    return entries;
  }, [
    coreCameraPitch,
    coreCameraPosition,
    coreCameraYaw,
    coreCameraZoom,
    hoveredPanelId,
    isWideViewport,
    panelAnchorById,
    panelScreenBiases,
    pinnedPanels,
    selectedPanelId,
    sortedPanels,
    viewportHeight,
    viewportWidth,
    visiblePanelIds,
  ]);

  const overflowPanels = useMemo(() => {
    const visible = new Set(visiblePanelIds);
    return sortedPanels.filter((panel) => !visible.has(panel.id)).slice(0, 6);
  }, [sortedPanels, visiblePanelIds]);

  const galaxyLayerStyles = useMemo(() => {
    const driftX = coreCameraPosition.x;
    const driftY = coreCameraPosition.y;
    const driftZ = coreCameraPosition.z;
    return {
      far: {
        transform: `translate3d(${((-driftX * 0.07) + (coreCameraYaw * 1.4)).toFixed(1)}px, ${((-driftY * 0.05) + (coreCameraPitch * 1.3)).toFixed(1)}px, ${(driftZ * 0.04).toFixed(1)}px) scale(${(1 + (driftZ * 0.00018)).toFixed(3)})`,
      },
      mid: {
        transform: `translate3d(${((-driftX * 0.14) + (coreCameraYaw * 2.2)).toFixed(1)}px, ${((-driftY * 0.11) + (coreCameraPitch * 1.8)).toFixed(1)}px, ${(driftZ * 0.08).toFixed(1)}px) scale(${(1.04 + (driftZ * 0.00022)).toFixed(3)})`,
      },
      near: {
        transform: `translate3d(${((-driftX * 0.22) + (coreCameraYaw * 3.1)).toFixed(1)}px, ${((-driftY * 0.18) + (coreCameraPitch * 2.4)).toFixed(1)}px, ${(driftZ * 0.14).toFixed(1)}px) scale(${(1.1 + (driftZ * 0.00032)).toFixed(3)})`,
      },
    };
  }, [coreCameraPitch, coreCameraPosition, coreCameraYaw]);

  return (
    <>
      <div
        className="simulation-core-backdrop"
        onPointerDown={handleCorePointerDown}
        onPointerMove={handleCorePointerMove}
        onPointerUp={handleCorePointerUp}
        onPointerCancel={handleCorePointerUp}
        onWheel={handleCoreWheel}
      >
        <div className="simulation-galaxy-layer simulation-galaxy-layer-far" style={galaxyLayerStyles.far} />
        <div className="simulation-galaxy-layer simulation-galaxy-layer-mid" style={galaxyLayerStyles.mid} />
        <div className="simulation-galaxy-layer simulation-galaxy-layer-near" style={galaxyLayerStyles.near} />
        <div className="simulation-core-stage" style={{ transform: coreCameraTransform }}>
          <SimulationCanvas
            simulation={simulation}
            catalog={catalog}
            onOverlayInit={(api) => setOverlayApi(api)}
            height={viewportHeight}
            defaultOverlayView={coreOverlayView}
            overlayViewLocked
            compactHud
            interactive={false}
            backgroundMode
            className="simulation-core-canvas"
          />
        </div>
        <p className="simulation-core-hint">
          drag orbit • shift+drag pan • wheel zoom • wasd strafe/drive • r/f rise/fall
        </p>
        <div className="simulation-core-vignette" />
      </div>

      <main className="relative z-20 max-w-[1920px] mx-auto px-2 py-2 md:px-4 md:py-4 pb-20 transition-colors">
        <header className="mb-4 border-b border-[rgba(166,205,235,0.25)] pb-3 flex flex-col gap-2 bg-[rgba(8,14,22,0.42)] backdrop-blur-[2px] rounded-xl px-3">
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
                <span>
                  core-camera: <code>{coreCameraZoom.toFixed(2)}x</code> / pitch
                  <code>{coreCameraPitch.toFixed(0)}deg</code> / yaw
                  <code>{coreCameraYaw.toFixed(0)}deg</code> / xyz
                  <code>{coreCameraPosition.x.toFixed(0)}</code>,
                  <code>{coreCameraPosition.y.toFixed(0)}</code>,
                  <code>{coreCameraPosition.z.toFixed(0)}</code>
                </span>
                <span>
                  flight: <code>{coreFlightEnabled ? "armed" : "paused"}</code> speed
                  <code>{coreFlightSpeed.toFixed(2)}x</code>
                </span>
                {activeChatLens ? (
                  <span>chat-lens: <code>{activeChatLens.presence}</code> ({activeChatLens.status})</span>
                ) : null}
                {latestAutopilotEvent ? (
                  <span>last: <code>{latestAutopilotEvent.actionId}</code> ({latestAutopilotEvent.result})</span>
                ) : null}
              </div>
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2">
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

              <button
                type="button"
                onClick={toggleCoreFlight}
                className={`border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                  coreFlightEnabled
                    ? "bg-[rgba(122,214,255,0.18)] border-[rgba(122,214,255,0.52)] text-[#9de3ff]"
                    : "bg-[rgba(180,180,180,0.14)] border-[rgba(182,182,182,0.34)] text-[#d2d7de]"
                }`}
              >
                {coreFlightEnabled ? "Flight Armed" : "Flight Paused"}
              </button>

              <div className="flex items-center gap-1 border rounded px-1 py-0.5 text-[10px] bg-[rgba(10,22,34,0.68)] border-[rgba(120,178,221,0.35)]">
                <button type="button" onClick={() => nudgeCoreFlightSpeed(-0.12)} className="px-1 text-[#bdd9f2]">thrust-</button>
                <button type="button" onClick={() => nudgeCoreFlightSpeed(0.12)} className="px-1 text-[#9ed6f8]">thrust+</button>
              </div>

              <select
                value={coreOverlayView}
                onChange={(event) => setCoreOverlayView(event.target.value as OverlayViewId)}
                className="border rounded px-2 py-0.5 text-[10px] font-semibold bg-[rgba(10,22,34,0.74)] text-[#9dd5f8] border-[rgba(120,178,221,0.4)]"
                title="simulation-core overlay lane"
              >
                {OVERLAY_VIEW_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    core:{option.label}
                  </option>
                ))}
              </select>

              <div className="flex items-center gap-1 border rounded px-1 py-0.5 text-[10px] bg-[rgba(10,22,34,0.68)] border-[rgba(120,178,221,0.35)]">
                <button type="button" onClick={() => nudgeCoreZoom(-0.08)} className="px-1 text-[#9ed6f8]">-</button>
                <button type="button" onClick={() => nudgeCoreZoom(0.08)} className="px-1 text-[#9ed6f8]">+</button>
                <button type="button" onClick={() => nudgeCorePitch(4)} className="px-1 text-[#a8e6bf]">tilt+</button>
                <button type="button" onClick={() => nudgeCorePitch(-4)} className="px-1 text-[#a8e6bf]">tilt-</button>
                <button type="button" onClick={() => nudgeCoreYaw(-4)} className="px-1 text-[#f5c18a]">yaw-</button>
                <button type="button" onClick={() => nudgeCoreYaw(4)} className="px-1 text-[#f5c18a]">yaw+</button>
                <button type="button" onClick={resetCoreCamera} className="px-1 text-[#f3d9b8]">reset</button>
              </div>

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
            className={`text-sm font-bold px-4 py-2 rounded-lg shadow-lg transition-all duration-200 ${
              isEditMode
                ? "bg-[#ae81ff] text-white ring-2 ring-white/20 scale-105"
                : "bg-[rgba(45,46,39,0.9)] text-[#ae81ff] border border-[#ae81ff]/40 hover:bg-[#ae81ff]/10 hover:border-[#ae81ff]"
            }`}
          >
            {isEditMode ? "Done Editing" : "Edit Layout"}
          </button>
        </div>

        {isWideViewport ? (
          <section className="world-panel-stage" aria-label="world anchored panels">
            <svg className="world-panel-tethers" viewBox={`0 0 ${viewportWidth} ${viewportHeight}`} preserveAspectRatio="none">
              <title>World panel tethers</title>
              {worldPanelLayout.map((entry) => {
                const controlOffset = entry.side === "left" || entry.side === "right" ? 70 : 44;
                const cx1 = entry.anchorScreenX + ((entry.side === "left" || entry.side === "right")
                  ? (entry.side === "left" ? -controlOffset : controlOffset)
                  : 0);
                const cy1 = entry.anchorScreenY + ((entry.side === "top" || entry.side === "bottom")
                  ? (entry.side === "top" ? -controlOffset : controlOffset)
                  : 0);
                const cx2 = entry.tetherX + ((entry.side === "left" || entry.side === "right")
                  ? (entry.side === "left" ? controlOffset : -controlOffset)
                  : 0);
                const cy2 = entry.tetherY + ((entry.side === "top" || entry.side === "bottom")
                  ? (entry.side === "top" ? controlOffset : -controlOffset)
                  : 0);
                const curvePath = `M ${entry.anchorScreenX.toFixed(1)} ${entry.anchorScreenY.toFixed(1)} C ${cx1.toFixed(1)} ${cy1.toFixed(1)}, ${cx2.toFixed(1)} ${cy2.toFixed(1)}, ${entry.tetherX.toFixed(1)} ${entry.tetherY.toFixed(1)}`;
                const glowAlpha = clamp(entry.glow, 0.2, 1);
                const haloRadius = Math.max(8, entry.anchor.radius * Math.min(viewportWidth, viewportHeight) * 0.22);
                return (
                  <g key={`tether-${entry.id}`}>
                    <circle
                      cx={entry.anchorScreenX}
                      cy={entry.anchorScreenY}
                      r={haloRadius * 1.8}
                      fill={`hsla(${entry.anchor.hue}, 86%, 64%, ${0.11 + (glowAlpha * 0.18)})`}
                    />
                    <circle
                      cx={entry.anchorScreenX}
                      cy={entry.anchorScreenY}
                      r={haloRadius}
                      fill={`hsla(${entry.anchor.hue}, 92%, 74%, ${0.28 + (glowAlpha * 0.28)})`}
                      stroke={`hsla(${entry.anchor.hue}, 96%, 84%, ${0.48 + (glowAlpha * 0.34)})`}
                      strokeWidth={1.2}
                    />
                    <path
                      d={curvePath}
                      stroke={`hsla(${entry.anchor.hue}, 92%, 72%, ${0.24 + (glowAlpha * 0.44)})`}
                      strokeWidth={1.1 + (glowAlpha * 1.4)}
                      strokeDasharray={entry.collapse ? "4 8" : "6 6"}
                      fill="none"
                    />
                  </g>
                );
              })}
            </svg>

            {worldPanelLayout.map((entry) => {
              const panelTitle = entry.panel.id.split(".").slice(-1)[0].replace(/_/g, " ");
              const isPinned = Boolean(pinnedPanels[entry.id]);
              const isSelected = selectedPanelId === entry.id;
              return (
                <motion.section
                  key={entry.id}
                  className={`floating-overlay-panel world-panel-shell ${isSelected ? "world-panel-selected" : ""} ${entry.collapse ? "world-panel-collapsed" : ""}`}
                  style={{
                    left: entry.x,
                    top: entry.y,
                    width: entry.width,
                    height: entry.height,
                    zIndex: Math.round(24 + (entry.glow * 18) + (isSelected ? 24 : 0)),
                    opacity: clamp(0.54 + (entry.glow * 0.54), 0.5, 1),
                  }}
                  drag={isEditMode}
                  dragMomentum={false}
                  dragElastic={0.08}
                  onHoverStart={() => setHoveredPanelId(entry.id)}
                  onHoverEnd={() => setHoveredPanelId((current) => (current === entry.id ? null : current))}
                  onClick={() => setSelectedPanelId(entry.id)}
                  onDoubleClick={() => flyCameraToAnchor(entry.anchor)}
                  onDragEnd={(_, info) => {
                    setPanelScreenBiases((prev) => {
                      const current = prev[entry.id] ?? { x: 0, y: 0 };
                      return {
                        ...prev,
                        [entry.id]: {
                          x: clamp(current.x + info.offset.x, -520, 520),
                          y: clamp(current.y + info.offset.y, -420, 420),
                        },
                      };
                    });
                  }}
                >
                  <header className="world-panel-header">
                    <div>
                      <p className="world-panel-title">{panelTitle}</p>
                      <p className="world-panel-subtitle">
                        {entry.anchor.kind}:{entry.anchor.label}
                      </p>
                    </div>
                    <div className="world-panel-actions">
                      <button
                        type="button"
                        className="world-panel-action"
                        onClick={(event) => {
                          event.stopPropagation();
                          togglePanelPin(entry.id);
                        }}
                        title={isPinned ? "unpin panel" : "pin panel"}
                      >
                        {isPinned ? "pinned" : "pin"}
                      </button>
                      <button
                        type="button"
                        className="world-panel-action"
                        onClick={(event) => {
                          event.stopPropagation();
                          flyCameraToAnchor(entry.anchor);
                        }}
                        title="fly camera to anchor"
                      >
                        inspect
                      </button>
                    </div>
                  </header>

                  {entry.collapse ? (
                    <div className="world-panel-collapsed-body">
                      <p>
                        moving at velocity <code>{coreFlightSpeed.toFixed(2)}x</code>
                      </p>
                      <p>
                        anchor confidence <code>{Math.round(entry.anchor.confidence * 100)}%</code>
                      </p>
                    </div>
                  ) : (
                    <div className="world-panel-body">{entry.panel.render()}</div>
                  )}

                  {isEditMode ? (
                    <div className="world-panel-edit-tag">drag bias</div>
                  ) : null}
                </motion.section>
              );
            })}

            {overflowPanels.length > 0 ? (
              <aside className="world-panel-chip-rack" aria-label="deferred panels">
                {overflowPanels.map((panel) => {
                  const label = panel.id.split(".").slice(-1)[0].replace(/_/g, " ");
                  const anchor = panelAnchorById.get(panel.id);
                  return (
                    <button
                      key={`chip-${panel.id}`}
                      type="button"
                      className="world-panel-chip"
                      onClick={() => setSelectedPanelId(panel.id)}
                      onDoubleClick={() => {
                        if (anchor) {
                          flyCameraToAnchor(anchor);
                        }
                      }}
                    >
                      {label}
                    </button>
                  );
                })}
              </aside>
            ) : null}
          </section>
        ) : (
          <div
            ref={gridContainerRef}
            className="overlay-constellation grid grid-cols-1 gap-3 items-start"
          >
            {sortedPanels.map((panel) => (
              <motion.section
                key={panel.id}
                className={`${panel.className ?? "card relative overflow-hidden"} floating-overlay-panel ${isEditMode ? "cursor-grab active:cursor-grabbing ring-2 ring-[#ae81ff] shadow-[0_0_15px_rgba(174,129,255,0.3)] z-10" : ""}`}
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
        )}

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
    </>
  );
}
