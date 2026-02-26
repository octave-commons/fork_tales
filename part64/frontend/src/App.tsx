// SPDX-License-Identifier: GPL-3.0-or-later
// This file is part of Fork Tales.
// Copyright (C) 2024-2025 Fork Tales Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

import {
  useState,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  lazy,
  Suspense,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { type PanInfo } from "framer-motion";
import { useAutopilotController } from "./hooks/useAutopilotController";
import { useWorldState } from "./hooks/useWorldState";
import {
  OVERLAY_VIEW_OPTIONS,
  SimulationCanvas,
  type NexusInteractionEvent,
  type OverlayViewId,
} from "./components/Simulation/Canvas";
import { CoreBackdrop } from "./components/App/CoreBackdrop";
import { CoreControlPanel, type MouseDaimonTuning } from "./components/App/CoreControlPanel";
import { CoreLayerManagerOverlay } from "./components/App/CoreLayerManagerOverlay";
import { WorldPanelsViewport } from "./components/App/WorldPanelsViewport";
import { MusePresencePanel } from "./components/Panels/MusePresencePanel";
import { ProjectionLedgerPanel } from "./components/Panels/ProjectionLedgerPanel";
import {
  CORE_CAMERA_PITCH_MAX,
  CORE_CAMERA_PITCH_MIN,
  CORE_CAMERA_X_LIMIT,
  CORE_CAMERA_YAW_MAX,
  CORE_CAMERA_YAW_MIN,
  CORE_CAMERA_Y_LIMIT,
  CORE_CAMERA_Z_MAX,
  CORE_CAMERA_Z_MIN,
  CORE_CAMERA_ZOOM_MAX,
  CORE_CAMERA_ZOOM_MIN,
  CORE_FLIGHT_BASE_SPEED,
  CORE_FLIGHT_SPEED_MAX,
  CORE_FLIGHT_SPEED_MIN,
  CORE_LAYER_OPTIONS,
  CORE_ORBIT_PERIOD_SECONDS,
  CORE_ORBIT_RADIUS_X,
  CORE_ORBIT_RADIUS_Y,
  CORE_ORBIT_RADIUS_Z,
  CORE_ORBIT_SPEED_MAX,
  CORE_ORBIT_SPEED_MIN,
  CORE_SIM_LAYER_DEPTH_MAX,
  CORE_SIM_LAYER_DEPTH_MIN,
  CORE_SIM_GRAPH_NODE_SMOOTHING_MAX,
  CORE_SIM_GRAPH_NODE_SMOOTHING_MIN,
  CORE_SIM_GRAPH_NODE_STEP_SCALE_MAX,
  CORE_SIM_GRAPH_NODE_STEP_SCALE_MIN,
  CORE_SIM_MOTION_SPEED_MAX,
  CORE_SIM_MOTION_SPEED_MIN,
  CORE_SIM_MOUSE_INFLUENCE_MAX,
  CORE_SIM_MOUSE_INFLUENCE_MIN,
  CORE_SIM_PARTICLE_DENSITY_MAX,
  CORE_SIM_PARTICLE_DENSITY_MIN,
  CORE_SIM_PARTICLE_SCALE_MAX,
  CORE_SIM_PARTICLE_SCALE_MIN,
  CORE_VISUAL_BRIGHTNESS_MAX,
  CORE_VISUAL_BRIGHTNESS_MIN,
  CORE_VISUAL_CONTRAST_MAX,
  CORE_VISUAL_CONTRAST_MIN,
  CORE_VISUAL_HUE_MAX,
  CORE_VISUAL_HUE_MIN,
  CORE_VISUAL_SATURATION_MAX,
  CORE_VISUAL_SATURATION_MIN,
  CORE_VISUAL_VIGNETTE_MAX,
  CORE_VISUAL_VIGNETTE_MIN,
  CORE_VISUAL_WASH_MAX,
  CORE_VISUAL_WASH_MIN,
  DEFAULT_CORE_LAYER_VISIBILITY,
  DEFAULT_CORE_SIMULATION_TUNING,
  DEFAULT_CORE_VISUAL_TUNING,
  HIGH_VISIBILITY_CORE_VISUAL_TUNING,
  type CoreLayerId,
  type CoreSimulationTuning,
  type CoreVisualTuning,
} from "./app/coreSimulationConfig";
import {
  normalizeMusePresenceId,
  normalizeMuseWorkspaceContext,
  sameStringArray,
} from "./app/museWorkspace";
import {
  PANEL_ANCHOR_PRESETS,
  WORLD_PANEL_MARGIN,
  containsAnchorNoCoverZone,
  defaultPinnedPanelMap,
  normalizeUnit,
  overlapAmount,
  panelSizeForWorld,
  preferredSideForAnchor,
  type PanelConfig,
  type PanelPreferredSide,
  type PanelWindowState,
  type WorldAnchorTarget,
  type WorldPanelNexusEntry,
  type WorldPanelLayoutEntry,
} from "./app/worldPanelLayout";
import { runtimeApiUrl, runtimeBaseUrl } from "./runtime/endpoints";
import type {
  ChatMessage,
  CouncilApiResponse,
  DriftScanPayload,
  EntityManifestItem,
  FileGraphConceptPresence,
  FileGraphNode,
  MuseEvent,
  MuseWorkspaceContext,
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
const RuntimeConfigPanel = lazy(() =>
  import("./components/Panels/RuntimeConfigPanel").then((module) => ({
    default: module.RuntimeConfigPanel,
  })),
);
const DaimoiPresencePanel = lazy(() =>
  import("./components/Panels/DaimoiPresencePanel").then((module) => ({
    default: module.DaimoiPresencePanel,
  })),
);
const WorldLogPanel = lazy(() =>
  import("./components/Panels/WorldLogPanel").then((module) => ({
    default: module.WorldLogPanel,
  })),
);

interface OverlayApi {
  pulseAt?: (x: number, y: number, power: number, target?: string) => void;
  singAll?: () => void;
  getAnchorRatio?: (kind: string, targetId: string) => { x: number; y: number; kind: string; label?: string } | null;
  projectRatioToClient?: (xRatio: number, yRatio: number) => { x: number; y: number; w: number; h: number };
  interactAt?: (
    xRatio: number,
    yRatio: number,
    options?: { openWorldscreen?: boolean },
  ) => { hitNode: boolean; openedWorldscreen: boolean; target: string; xRatio: number; yRatio: number };
  interactClientAt?: (
    clientX: number,
    clientY: number,
    options?: { openWorldscreen?: boolean },
  ) => { hitNode: boolean; openedWorldscreen: boolean; target: string; xRatio: number; yRatio: number };
}

interface UiToast {
  id: number;
  title: string;
  body: string;
}

interface UserPresenceInputPayload {
  kind: string;
  target: string;
  message?: string;
  xRatio?: number;
  yRatio?: number;
  embedDaimoi?: boolean;
  meta?: Record<string, unknown>;
}

type ParticleDisposition = "neutral" | "role-bound";

interface RankedPanel extends PanelConfig {
  priority: number;
  depth: number;
  councilScore: number;
  councilBoost: number;
  councilReason: string;
  presenceId: string;
  presenceLabel: string;
  presenceLabelJa: string;
  presenceRole: string;
  particleDisposition: ParticleDisposition;
  particleCount: number;
  toolHints: string[];
}

const PRESENCE_OPERATIONAL_ROLE_BY_ID: Record<string, string> = {
  witness_thread: "crawl-routing",
  keeper_of_receipts: "file-analysis",
  mage_of_receipts: "image-captioning",
  anchor_registry: "council-orchestration",
  gates_of_truth: "compliance-gating",
  view_lens_keeper: "camera-guidance",
  health_sentinel_gpu1: "compute-scheduler",
  health_sentinel_gpu0: "compute-scheduler",
  health_sentinel_npu0: "compute-scheduler",
  health_sentinel_cpu: "compute-scheduler",
};

const PANEL_TOOL_HINTS: Record<string, string[]> = {
  "nexus.ui.command_center": ["call", "say", "webrtc"],
  "nexus.ui.chat.witness_thread": ["ledger", "lineage", "particles"],
  "nexus.ui.chat.chaos": ["chat", "nearby", "pin"],
  "nexus.ui.chat.stability": ["chat", "nearby", "pin"],
  "nexus.ui.chat.symmetry": ["chat", "nearby", "pin"],
  "nexus.ui.web_graph_weaver": ["crawl", "queue", "graph"],
  "nexus.ui.inspiration_atlas": ["search", "curate", "seed"],
  "nexus.ui.entity_vitals": ["vitals", "telemetry", "watch"],
  "nexus.ui.projection_ledger": ["projection", "trace", "audit"],
  "nexus.ui.autopilot_ledger": ["autopilot", "risk", "gates"],
  "nexus.ui.world_log": ["receipts", "events", "review"],
  "nexus.ui.stability_observatory": ["study", "drift", "council"],
  "nexus.ui.runtime_config": ["config", "constants", "tuning"],
  "nexus.ui.daimoi_presence": ["daimoi", "presence", "focus"],
  "nexus.ui.omni_archive": ["catalog", "memories", "artifacts"],
  "nexus.ui.myth_commons": ["interact", "pray", "speak"],
  "nexus.ui.dedicated_views": ["overlay", "focus", "monitor"],
  "nexus.ui.glass_viewport": ["glass", "camera", "pan"],
};

const COUNCIL_BOOST_STORAGE_KEY = "eta_mu.council_boosts.v1";
const TERTIARY_PIN_STORAGE_KEY = "eta_mu.tertiary_pin.v1";
const MUSE_WORKSPACE_STORAGE_KEY = "eta_mu.muse_workspace.v1";
const INTERFACE_OPACITY_STORAGE_KEY = "eta_mu.interface_opacity.v2";
const GLASS_VIEWPORT_PANEL_ID = "nexus.ui.glass_viewport";
const RUNTIME_CONFIG_PANEL_ID = "nexus.ui.runtime_config";
const INTERFACE_OPACITY_MIN = 0.72;
const INTERFACE_OPACITY_MAX = 1;
const DEFAULT_INTERFACE_TRANSPARENCY_PERCENT = 0;
const DEFAULT_INTERFACE_OPACITY = 1 - (DEFAULT_INTERFACE_TRANSPARENCY_PERCENT / 100);
const FIXED_MUSE_PRESENCES = [
  {
    id: "nexus.ui.chat.witness_thread",
    presenceId: "witness_thread",
    label: "Witness Thread",
  },
  {
    id: "nexus.ui.chat.chaos",
    presenceId: "chaos",
    label: "Chaos",
  },
  {
    id: "nexus.ui.chat.stability",
    presenceId: "stability",
    label: "Stability",
  },
  {
    id: "nexus.ui.chat.symmetry",
    presenceId: "symmetry",
    label: "Symmetry",
  },
] as const;

const APP_WORKSPACE_NORMALIZE_OPTIONS = {
  maxPinnedFileNodeIds: 48,
  maxSearchQueryLength: 180,
  maxPinnedNexusSummaries: 24,
} as const;

const USER_PRESENCE_BATCH_IDLE_FLUSH_MS = 2400;
const USER_PRESENCE_BATCH_MAX_WINDOW_MS = 60_000;
const USER_PRESENCE_BATCH_MAX_EVENTS = 36;

function isGlassPrimaryPanelId(panelId: string): boolean {
  return panelId === GLASS_VIEWPORT_PANEL_ID || panelId === "nexus.ui.dedicated_views";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function resolveEventElement(target: EventTarget | null): Element | null {
  if (target instanceof Element) {
    return target;
  }
  if (target instanceof Node) {
    return target.parentElement;
  }
  return null;
}

function isTextEntryTarget(target: EventTarget | null): boolean {
  const element = resolveEventElement(target);
  if (!(element instanceof HTMLElement)) {
    return false;
  }
  if (element.isContentEditable) {
    return true;
  }
  const tagName = element.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

function isCorePointerBlockedTarget(target: EventTarget | null): boolean {
  const element = resolveEventElement(target);
  if (!element) {
    return false;
  }
  if (isTextEntryTarget(element)) {
    return true;
  }
  return Boolean(
    element.closest(
      "button, a, [role='button'], [data-core-pointer='block'], [data-panel-interactive='true']",
    ),
  );
}

function shouldRouteWheelToCore(target: EventTarget | null, deltaY = 0): boolean {
  const element = resolveEventElement(target);
  if (!element) {
    return true;
  }
  if (
    element.closest(
      "input, textarea, select, option, [contenteditable='true'], [role='slider'], [data-core-wheel='block']",
    )
  ) {
    return false;
  }

  if (element.closest("button, a, [role='button'], [data-panel-interactive='true']")) {
    return false;
  }

  const panelBody = element.closest(".world-panel-body");
  if (panelBody instanceof HTMLElement) {
    const maxScrollTop = Math.max(0, panelBody.scrollHeight - panelBody.clientHeight);
    if (maxScrollTop > 1) {
      const atTop = panelBody.scrollTop <= 1;
      const atBottom = panelBody.scrollTop >= maxScrollTop - 1;
      if ((deltaY < 0 && !atTop) || (deltaY > 0 && !atBottom) || Math.abs(deltaY) < 0.5) {
        return false;
      }
    }
  }

  return true;
}

function projectionOpacity(raw: number | undefined, floor = 0.9): number {
  const normalized = clamp(typeof raw === "number" ? raw : 1, 0, 1);
  return floor + normalized * (1 - floor);
}

function stableUnitHash(seed: string): number {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function toMuseSlug(raw: string): string {
  const cleaned = String(raw || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_\s-]+/g, "")
    .replace(/[\s-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!cleaned) {
    return "";
  }
  if (/^[0-9]/.test(cleaned)) {
    return `muse_${cleaned}`;
  }
  return cleaned;
}

function resolveRuntimeMediaUrl(rawUrl: string): string {
  const trimmed = String(rawUrl || "").trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  if (trimmed.startsWith("/")) {
    return runtimeApiUrl(trimmed);
  }
  return runtimeApiUrl(`/${trimmed.replace(/^\.+\//, "")}`);
}

function normalizeDeviceUtilization(raw: unknown): number {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  const scaled = value > 1 ? value / 100 : value;
  return clamp(scaled, 0, 1);
}

function buildDeviceSurroundingNodes(simulation: {
  presence_dynamics?: {
    resource_heartbeat?: {
      devices?: Record<string, { utilization?: number } | undefined>;
    };
  };
} | null): Array<Record<string, unknown>> {
  const devices = simulation?.presence_dynamics?.resource_heartbeat?.devices;
  if (!devices || typeof devices !== "object") {
    return [];
  }
  return Object.entries(devices)
    .map(([deviceId, payload]) => {
      const utilization = normalizeDeviceUtilization(payload?.utilization);
      return {
        id: `device:${deviceId}`,
        kind: "device",
        label: deviceId,
        text: `${deviceId} utilization ${Math.round(utilization * 100)}%`,
        utilization,
        visibility: "public",
      };
    })
    .slice(0, 6);
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
  const { catalog, simulation, projection, museEvents, isConnected } = useWorldState(uiPerspective);

  const [overlayApi, setOverlayApi] = useState<OverlayApi | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [voiceInputMeta, setVoiceInputMeta] = useState("voice input idle / 音声入力待機");
  const [activeMusePresenceId, setActiveMusePresenceId] = useState("witness_thread");
  const [museForgeLabel, setMuseForgeLabel] = useState("");
  const [museForgeBusy, setMuseForgeBusy] = useState(false);
  const [museWorkspaceContexts, setMuseWorkspaceContexts] = useState<Record<string, MuseWorkspaceContext>>(() => {
    if (typeof window === "undefined") {
      return {};
    }
    try {
      const raw = window.localStorage.getItem(MUSE_WORKSPACE_STORAGE_KEY);
      if (!raw) {
        return {};
      }
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      const normalized: Record<string, MuseWorkspaceContext> = {};
      Object.entries(parsed).forEach(([presenceId, value]) => {
        const normalizedPresence = normalizeMusePresenceId(String(presenceId || ""));
        if (!normalizedPresence || !value || typeof value !== "object") {
          return;
        }
        const workspace = normalizeMuseWorkspaceContext(
          value as Partial<MuseWorkspaceContext>,
          APP_WORKSPACE_NORMALIZE_OPTIONS,
        );
        if (workspace.pinnedFileNodeIds.length <= 0 && workspace.searchQuery.trim().length <= 0) {
          return;
        }
        normalized[normalizedPresence] = workspace;
      });
      return normalized;
    } catch {
      return {};
    }
  });
  const [museWorkspaceBindings, setMuseWorkspaceBindings] = useState<Record<string, string[]>>(() => {
    const seeded: Record<string, string[]> = {};
    Object.entries(museWorkspaceContexts).forEach(([presenceId, workspace]) => {
      if (workspace.pinnedFileNodeIds.length > 0) {
        seeded[presenceId] = workspace.pinnedFileNodeIds;
      }
    });
    return seeded;
  });
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [worldInteraction, setWorldInteraction] = useState<WorldInteractionResponse | null>(null);
  const [interactingPersonId, setInteractingPersonId] = useState<string | null>(null);
  const [deferredPanelsReady, setDeferredPanelsReady] = useState(false);
  const [uiToasts, setUiToasts] = useState<UiToast[]>([]);

  const toastSeqRef = useRef(0);
  const toastTimeoutsRef = useRef<Map<number, number>>(new Map());
  const panelSideRef = useRef<Map<string, PanelPreferredSide>>(new Map());
  const panelScreenRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const panelWorldScaleRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const cameraFlightRef = useRef<number | null>(null);
  const coreDragRef = useRef<{
    active: boolean;
    pointerId: number;
    mode: "orbit" | "pan";
    startX: number;
    startY: number;
    lastX: number;
    lastY: number;
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
  const corePointerFrameRef = useRef<number | null>(null);
  const corePointerPendingRef = useRef({ dx: 0, dy: 0 });
  const lastNudgePulseTsRef = useRef(0);
  const processedMuseEventSeqRef = useRef(0);
  const museAudioElementRef = useRef<HTMLAudioElement | null>(null);

  const [panelWorldBiases, setPanelWorldBiases] = useState<Record<string, { x: number; y: number }>>({});
  const [panelWindowStates, setPanelWindowStates] = useState<Record<string, PanelWindowState>>({});
  const [panelCouncilBoosts, setPanelCouncilBoosts] = useState<Record<string, number>>(() => {
    if (typeof window === "undefined") {
      return {};
    }
    try {
      const raw = window.localStorage.getItem(COUNCIL_BOOST_STORAGE_KEY);
      if (!raw) {
        return {};
      }
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      const normalized: Record<string, number> = {};
      Object.entries(parsed).forEach(([key, value]) => {
        const score = Number(value);
        if (!Number.isFinite(score) || score === 0) {
          return;
        }
        normalized[key] = clamp(score, -6, 8);
      });
      return normalized;
    } catch {
      return {};
    }
  });
  const [selectedPanelId, setSelectedPanelId] = useState<string | null>(null);
  const [hoveredPanelId, setHoveredPanelId] = useState<string | null>(null);
  const [tertiaryPinnedPanelId, setTertiaryPinnedPanelId] = useState<string | null>(() => {
    if (typeof window === "undefined") {
      return null;
    }
    const raw = window.localStorage.getItem(TERTIARY_PIN_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const clean = raw.trim();
    return clean.length > 0 ? clean : null;
  });
  const [pinnedPanels, setPinnedPanels] = useState<Record<string, boolean>>(() =>
    defaultPinnedPanelMap(Object.keys(PANEL_ANCHOR_PRESETS)),
  );
  const [isEditMode, setIsEditMode] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth);
  const [viewportHeight, setViewportHeight] = useState(() => window.innerHeight);
  const [coreCameraZoom, setCoreCameraZoom] = useState(1);
  const [coreCameraPitch, setCoreCameraPitch] = useState(0);
  const [coreCameraYaw, setCoreCameraYaw] = useState(0);
  const deferredCoreCameraZoom = useDeferredValue(coreCameraZoom);
  const deferredCoreCameraPitch = useDeferredValue(coreCameraPitch);
  const deferredCoreCameraYaw = useDeferredValue(coreCameraYaw);
  const [coreCameraPosition, setCoreCameraPosition] = useState({ x: 0, y: 0, z: 0 });
  const [coreOverlayView, setCoreOverlayView] = useState<OverlayViewId>("omni");
  const [coreFlightEnabled, setCoreFlightEnabled] = useState(true);
  const [coreFlightSpeed, setCoreFlightSpeed] = useState(1);
  const [coreOrbitEnabled, setCoreOrbitEnabled] = useState(false);
  const [coreOrbitSpeed, setCoreOrbitSpeed] = useState(0.58);
  const [coreOrbitPhase, setCoreOrbitPhase] = useState(0);
  const [coreSimulationTuning, setCoreSimulationTuning] = useState<CoreSimulationTuning>(DEFAULT_CORE_SIMULATION_TUNING);
  const deferredCoreSimulationTuning = useDeferredValue(coreSimulationTuning);
  const [coreVisualTuning, setCoreVisualTuning] = useState<CoreVisualTuning>(DEFAULT_CORE_VISUAL_TUNING);
  const [mouseDaimonTuning, setMouseDaimonTuning] = useState<MouseDaimonTuning>({
    enabled: true,
    message: "witness",
    mode: "push",
    radius: 0.18,
    strength: 0.42,
  });
  const userPresenceHoverThrottleRef = useRef<Record<string, number>>({});
  const userPresenceGenericEmitMsRef = useRef(0);
  const userPresenceInputBatchRef = useRef<Record<string, unknown>[]>([]);
  const userPresenceInputBatchStartedMsRef = useRef(0);
  const userPresenceInputBatchLastMsRef = useRef(0);
  const userPresenceInputBatchIdleTimerRef = useRef<number | null>(null);
  const userPresenceInputBatchWindowTimerRef = useRef<number | null>(null);
  const [coreLayerVisibility, setCoreLayerVisibility] = useState<Record<CoreLayerId, boolean>>(DEFAULT_CORE_LAYER_VISIBILITY);
  const [coreLayerManagerOpen, setCoreLayerManagerOpen] = useState(true);
  const [interfaceOpacity, setInterfaceOpacity] = useState(() => {
    if (typeof window === "undefined") {
      return DEFAULT_INTERFACE_OPACITY;
    }
    const stored = window.localStorage.getItem(INTERFACE_OPACITY_STORAGE_KEY);
    if (stored === null) {
      return DEFAULT_INTERFACE_OPACITY;
    }
    const raw = Number(stored);
    if (!Number.isFinite(raw)) {
      return DEFAULT_INTERFACE_OPACITY;
    }
    return clamp(raw, INTERFACE_OPACITY_MIN, INTERFACE_OPACITY_MAX);
  });

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDeferredPanelsReady(true);
    }, 220);
    return () => {
      window.clearTimeout(timer);
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
    if (typeof window === "undefined") {
      return;
    }
    if (Object.keys(panelCouncilBoosts).length <= 0) {
      window.localStorage.removeItem(COUNCIL_BOOST_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(
      COUNCIL_BOOST_STORAGE_KEY,
      JSON.stringify(panelCouncilBoosts),
    );
  }, [panelCouncilBoosts]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (!tertiaryPinnedPanelId) {
      window.localStorage.removeItem(TERTIARY_PIN_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(TERTIARY_PIN_STORAGE_KEY, tertiaryPinnedPanelId);
  }, [tertiaryPinnedPanelId]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const payload: Record<string, MuseWorkspaceContext> = {};
    Object.entries(museWorkspaceContexts).forEach(([presenceId, workspace]) => {
      const normalizedPresence = normalizeMusePresenceId(presenceId);
      if (!normalizedPresence) {
        return;
      }
      const normalizedWorkspace = normalizeMuseWorkspaceContext(
        workspace,
        APP_WORKSPACE_NORMALIZE_OPTIONS,
      );
      if (
        normalizedWorkspace.pinnedFileNodeIds.length <= 0
        && normalizedWorkspace.searchQuery.trim().length <= 0
      ) {
        return;
      }
      payload[normalizedPresence] = normalizedWorkspace;
    });
    if (Object.keys(payload).length <= 0) {
      window.localStorage.removeItem(MUSE_WORKSPACE_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(MUSE_WORKSPACE_STORAGE_KEY, JSON.stringify(payload));
  }, [museWorkspaceContexts]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    if (Math.abs(interfaceOpacity - DEFAULT_INTERFACE_OPACITY) < 0.0001) {
      window.localStorage.removeItem(INTERFACE_OPACITY_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(INTERFACE_OPACITY_STORAGE_KEY, interfaceOpacity.toFixed(3));
  }, [interfaceOpacity]);

  const dismissToast = useCallback((id: number) => {
    const timeoutId = toastTimeoutsRef.current.get(id);
    if (timeoutId !== undefined) {
      window.clearTimeout(timeoutId);
      toastTimeoutsRef.current.delete(id);
    }
    setUiToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  useEffect(() => {
    const toastTimeouts = toastTimeoutsRef.current;
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
        toastTimeouts.delete(id);
      }, 5200);
      toastTimeouts.set(id, timeoutId);
    };

    window.addEventListener("ui:toast", handler);
    return () => {
      window.removeEventListener("ui:toast", handler);
      toastTimeouts.forEach((timeoutId) => {
        window.clearTimeout(timeoutId);
      });
      toastTimeouts.clear();
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

  const emitChatMessage = useCallback((message: ChatMessage) => {
    window.dispatchEvent(
      new CustomEvent("chat-message", {
        detail: message,
      }),
    );
  }, []);

  const emitUiToast = useCallback((title: string, body: string) => {
    const cleanTitle = String(title || "Notice").trim() || "Notice";
    const cleanBody = String(body || "").trim();
    if (!cleanBody) {
      return;
    }
    window.dispatchEvent(
      new CustomEvent("ui:toast", {
        detail: {
          title: cleanTitle,
          body: cleanBody,
        },
      }),
    );
  }, []);

  const playMuseAudio = useCallback(async (rawUrl: string, label: string): Promise<boolean> => {
    const resolvedUrl = resolveRuntimeMediaUrl(rawUrl);
    if (!resolvedUrl) {
      return false;
    }
    try {
      const audio = museAudioElementRef.current ?? new Audio();
      museAudioElementRef.current = audio;
      audio.src = resolvedUrl;
      audio.currentTime = 0;
      await audio.play();
      emitUiToast("Muse Audio", `playing ${label || "selected track"}`);
      return true;
    } catch {
      emitUiToast("Muse Audio Ready", `queued ${label || "track"} (${resolvedUrl})`);
      return false;
    }
  }, [emitUiToast]);

  const openMuseImage = useCallback((rawUrl: string, label: string): boolean => {
    const resolvedUrl = resolveRuntimeMediaUrl(rawUrl);
    if (!resolvedUrl) {
      return false;
    }
    try {
      const opened = window.open(resolvedUrl, "_blank", "noopener,noreferrer");
      if (opened) {
        emitUiToast("Muse Image", `opened ${label || "selected image"}`);
        return true;
      }
      emitUiToast("Muse Image Ready", `image selected: ${resolvedUrl}`);
      return false;
    } catch {
      emitUiToast("Muse Image Ready", `image selected: ${resolvedUrl}`);
      return false;
    }
  }, [emitUiToast]);

  const handleCreateMuse = useCallback(async () => {
    const label = String(museForgeLabel || "").trim();
    if (!label || museForgeBusy) {
      return;
    }
    const museId = toMuseSlug(label);
    if (!museId) {
      emitUiToast("Muse Create Failed", "Provide a valid muse label.");
      return;
    }

    const anchor = {
      x: clamp(0.14 + (stableUnitHash(`${museId}|x`) * 0.72), 0.08, 0.92),
      y: clamp(0.16 + (stableUnitHash(`${museId}|y`) * 0.68), 0.08, 0.92),
      zoom: 1,
      kind: "ui-meta-create",
    };

    setMuseForgeBusy(true);
    try {
      const baseUrl = runtimeBaseUrl();
      const response = await fetch(`${baseUrl}/api/muse/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          muse_id: museId,
          label,
          anchor,
          user_intent_id: `ui-create:${museId}`,
        }),
      });
      const payload = (await response.json()) as {
        ok?: boolean;
        error?: string;
        muse?: { id?: string; label?: string };
      };
      if (!response.ok || !payload?.ok) {
        throw new Error(String(payload?.error || `http_${response.status}`));
      }

      const createdId = String(payload?.muse?.id || museId).trim() || museId;
      const createdLabel = String(payload?.muse?.label || label).trim() || label;
      setActiveMusePresenceId(createdId);
      setMuseForgeLabel("");
      emitUiToast("Muse Created", `${createdLabel} is online as ${createdId}`);
      emitChatMessage({
        role: "assistant",
        text: `muse created\nid=${createdId}\nlabel=${createdLabel}`,
        meta: {
          channel: "command",
          source: "meta:/api/muse/create",
          presenceId: createdId,
          presenceName: createdLabel,
        },
      });
    } catch (error) {
      const reason = error instanceof Error ? error.message : "unknown_error";
      emitUiToast("Muse Create Failed", reason);
      emitChatMessage({
        role: "assistant",
        text: `muse create failed\nreason=${reason}`,
        meta: {
          channel: "command",
          source: "meta:/api/muse/create",
          presenceId: activeMusePresenceId,
        },
      });
    } finally {
      setMuseForgeBusy(false);
    }
  }, [activeMusePresenceId, emitChatMessage, emitUiToast, museForgeBusy, museForgeLabel]);

  const buildMuseSurroundingNodes = useCallback((
    musePresenceId: string,
    workspace: MuseWorkspaceContext | null = null,
  ): Array<Record<string, unknown>> => {
    const normalizedMuse = normalizeMusePresenceId(musePresenceId || "witness_thread") || "witness_thread";
    const graphNodes = (
      simulation?.file_graph?.file_nodes
      ?? catalog?.file_graph?.file_nodes
      ?? []
    ).filter((row): row is FileGraphNode => Boolean(row));

    const nodeById = new Map<string, FileGraphNode>();
    graphNodes.forEach((row) => {
      const id = String(row.id || "").trim();
      if (id && !nodeById.has(id)) {
        nodeById.set(id, row);
      }
      const nodeId = String(row.node_id || "").trim();
      if (nodeId && !nodeById.has(nodeId)) {
        nodeById.set(nodeId, row);
      }
    });

    const pinnedIds = (
      workspace?.pinnedFileNodeIds
      ?? museWorkspaceBindings[normalizedMuse]
      ?? []
    )
      .map((item) => String(item || "").trim())
      .filter((item, index, all) => item.length > 0 && all.indexOf(item) === index)
      .slice(0, 16);

    const buildLibraryUrl = (relativePath: string): string | undefined => {
      const cleanPath = String(relativePath || "").trim().replace(/^\/+/, "");
      if (!cleanPath) {
        return undefined;
      }
      const encodedPath = cleanPath
        .split("/")
        .map((segment) => encodeURIComponent(segment))
        .join("/");
      return `/library/${encodedPath}`;
    };

    const resolveNodeUrl = (row: FileGraphNode | undefined): string | undefined => {
      const nodeUrl = String(row?.url ?? "").trim();
      if (nodeUrl) {
        return nodeUrl;
      }

      const archiveUrl = String(row?.archive_url ?? "").trim();
      if (archiveUrl) {
        return archiveUrl;
      }

      const archiveRelPath = String(
        row?.archive_rel_path ?? row?.archived_rel_path ?? "",
      ).trim();
      const archiveRelUrl = buildLibraryUrl(archiveRelPath);
      if (archiveRelUrl) {
        return archiveRelUrl;
      }

      const sourceRelPath = String(row?.source_rel_path ?? "").trim();
      return buildLibraryUrl(sourceRelPath);
    };

    const pinnedRows = pinnedIds.map((nodeId) => {
      const row = nodeById.get(nodeId);
      const baseSeed = stableUnitHash(`${normalizedMuse}|${nodeId}`);
      const x = clamp(Number(row?.x ?? (0.22 + (baseSeed * 0.56))), 0, 1);
      const y = clamp(Number(row?.y ?? (0.24 + (stableUnitHash(`${normalizedMuse}|${nodeId}|y`) * 0.52))), 0, 1);
      const label = String(row?.source_rel_path ?? row?.label ?? row?.name ?? nodeId).trim() || nodeId;
      const text = String(row?.summary ?? row?.text_excerpt ?? label).trim() || label;
      const sourceRelPath = String(row?.source_rel_path ?? "").trim();
      return {
        id: nodeId,
        kind: String(row?.kind ?? "resource"),
        label,
        text,
        x,
        y,
        visibility: "private",
        tags: [normalizedMuse, "workspace-pin"],
        source_rel_path: sourceRelPath || undefined,
        url: resolveNodeUrl(row),
      };
    });

    const nearbyRows = graphNodes
      .filter((row) => {
        const dominantPresence = normalizeMusePresenceId(String(row.dominant_presence ?? ""));
        const conceptPresence = normalizeMusePresenceId(String(row.concept_presence_id ?? ""));
        if (pinnedIds.includes(String(row.id || row.node_id || "").trim())) {
          return false;
        }
        return dominantPresence === normalizedMuse || conceptPresence === normalizedMuse;
      })
      .sort((left, right) => {
        const rightScore = Number(right.importance ?? 0) + Number(right.embed_layer_count ?? 0) * 0.18;
        const leftScore = Number(left.importance ?? 0) + Number(left.embed_layer_count ?? 0) * 0.18;
        return rightScore - leftScore;
      })
      .slice(0, 8)
      .map((row) => {
        const sourceRelPath = String(row.source_rel_path ?? "").trim();
        return {
          id: String(row.id ?? row.node_id ?? "").trim(),
          kind: String(row.kind ?? "resource"),
          label: String(row.source_rel_path ?? row.label ?? row.id ?? "resource").trim(),
          text: String(row.summary ?? row.text_excerpt ?? row.label ?? row.id ?? "").trim(),
          x: clamp(Number(row.x ?? 0.5), 0, 1),
          y: clamp(Number(row.y ?? 0.5), 0, 1),
          visibility: "public",
          tags: [normalizedMuse, String(row.dominant_field ?? "field")],
          source_rel_path: sourceRelPath || undefined,
          url: resolveNodeUrl(row),
        };
      })
      .filter((row) => row.id.length > 0);

    return [...pinnedRows, ...nearbyRows, ...buildDeviceSurroundingNodes(simulation)].slice(0, 36);
  }, [catalog?.file_graph?.file_nodes, museWorkspaceBindings, simulation]);

  const emitWitnessChatReply = useCallback((
    payload: {
      reply?: unknown;
      mode?: unknown;
      model?: unknown;
      trace?: unknown;
      fallback?: unknown;
      turn_id?: unknown;
      muse?: unknown;
      manifest?: unknown;
      daimoi?: unknown;
      field_deltas?: unknown;
      gpu_claim?: unknown;
      tool_results?: unknown;
      media_actions?: unknown;
      audio_actions?: unknown;
    },
    source: string,
    requestedMusePresenceId?: string,
  ) => {
    const reply = String(payload.reply ?? "").trim();
    const mode = String(payload.mode ?? "canonical").trim() || "canonical";
    const model = String(payload.model ?? "").trim() || undefined;

    const trace =
      payload.trace && typeof payload.trace === "object"
        ? (payload.trace as Record<string, unknown>)
        : null;
    const overlayTags = Array.isArray(trace?.overlay_tags)
      ? trace.overlay_tags
          .map((item) => String(item || "").trim())
          .filter((item) => item.length > 0)
      : [];
    const failures = Array.isArray(trace?.failures)
      ? trace.failures.filter((row) => row && typeof row === "object") as Array<Record<string, unknown>>
      : [];
    const entities = Array.isArray(trace?.entities)
      ? trace.entities.filter((row) => row && typeof row === "object") as Array<Record<string, unknown>>
      : [];
    const muse = payload.muse && typeof payload.muse === "object"
      ? payload.muse as Record<string, unknown>
      : null;
    const tracePresenceId = String(
      muse?.id
      ?? entities[0]?.presence_id
      ?? requestedMusePresenceId
      ?? activeMusePresenceId
      ?? "witness_thread",
    ).trim()
      || "witness_thread";
    const tracePresenceName = String(
      muse?.label
      ?? entities[0]?.presence_en
      ?? tracePresenceId,
    ).trim() || tracePresenceId;

    const manifest = payload.manifest && typeof payload.manifest === "object"
      ? payload.manifest as Record<string, unknown>
      : null;
    const explicitSelected = Array.isArray(manifest?.explicit_selected)
      ? manifest.explicit_selected
      : [];
    const surroundSelected = Array.isArray(manifest?.surround_selected)
      ? manifest.surround_selected
      : [];
    const daimoiRows = Array.isArray(payload.daimoi)
      ? payload.daimoi.filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object"))
      : [];
    const fieldDeltas = Array.isArray(payload.field_deltas)
      ? payload.field_deltas.filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object"))
      : [];
    const toolRows = Array.isArray(payload.tool_results)
      ? payload.tool_results.filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object"))
      : [];
    const explicitMediaActions = Array.isArray(payload.media_actions)
      ? payload.media_actions.filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object"))
      : [];
    const audioActions = Array.isArray(payload.audio_actions)
      ? payload.audio_actions.filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object"))
      : [];
    const mediaActions = explicitMediaActions.length > 0 ? explicitMediaActions : audioActions;
    const gpuClaim = payload.gpu_claim && typeof payload.gpu_claim === "object"
      ? payload.gpu_claim as Record<string, unknown>
      : null;
    const gpuStatus = String(gpuClaim?.status ?? "").trim();

    const fallback = typeof payload.fallback === "boolean"
      ? payload.fallback
      : (mode !== "ollama" || failures.length > 0);
    const safeReply = reply || "[empty witness reply]";
    emitChatMessage({
      role: "assistant",
      text: safeReply,
      meta: {
        channel: "llm",
        model,
        fallback,
        source,
        presenceId: tracePresenceId,
        presenceName: tracePresenceName,
      },
    });

    if (fallback) {
      const failureCodes = failures
        .slice(0, 3)
        .map((row) => {
          const presence = String(row.presence_id ?? "presence").trim() || "presence";
          const code = String(row.error_code ?? "fallback").trim() || "fallback";
          return `${presence}:${code}`;
        })
        .join(", ");
      emitChatMessage({
        role: "assistant",
        text: `muse fallback active (mode=${mode}${failureCodes ? `, failures=${failureCodes}` : ""}).`,
        meta: {
          channel: "command",
          source,
          presenceId: tracePresenceId,
          presenceName: tracePresenceName,
        },
      });
    }

    if (manifest || daimoiRows.length > 0 || fieldDeltas.length > 0 || gpuStatus || toolRows.length > 0) {
      const turnId = String(payload.turn_id ?? "").trim() || "(none)";
      emitChatMessage({
        role: "assistant",
        text: [
          "muse turn signal",
          `turn_id=${turnId}`,
          `explicit=${explicitSelected.length}`,
          `surrounding=${surroundSelected.length}`,
          `daimoi=${daimoiRows.length}`,
          `field_deltas=${fieldDeltas.length}`,
          `gpu=${gpuStatus || "released"}`,
          `tools=${toolRows.length}`,
          `media=${mediaActions.length}`,
        ].join("\n"),
        meta: {
          channel: "command",
          source: `${source}:turn`,
          presenceId: tracePresenceId,
          presenceName: tracePresenceName,
        },
      });
    }

    fieldDeltas.slice(0, 8).forEach((delta, index) => {
      const x = clamp(Number(delta.x ?? (0.18 + (stableUnitHash(`${tracePresenceId}|${index}`) * 0.64))), 0, 1);
      const y = clamp(Number(delta.y ?? (0.16 + (stableUnitHash(`${tracePresenceId}|${index}|y`) * 0.68))), 0, 1);
      const intensity = clamp(Number(delta.intensity ?? 0.48), 0, 1);
      overlayApi?.pulseAt?.(x, y, 0.32 + (intensity * 0.94), tracePresenceId);
    });

    if (fieldDeltas.length === 0) {
      daimoiRows.slice(0, 6).forEach((row, index) => {
        const x = clamp(Number(row.x ?? (0.2 + (stableUnitHash(`${tracePresenceId}|daimon|${index}`) * 0.6))), 0, 1);
        const y = clamp(Number(row.y ?? (0.18 + (stableUnitHash(`${tracePresenceId}|daimon|${index}|y`) * 0.62))), 0, 1);
        const energy = clamp(Number(row.energy ?? 0.44), 0, 1);
        overlayApi?.pulseAt?.(x, y, 0.3 + (energy * 0.82), tracePresenceId);
      });
    }

    if (gpuStatus === "granted") {
      overlayApi?.singAll?.();
      const device = String(gpuClaim?.device ?? "gpu").trim() || "gpu";
      emitUiToast("Muse GPU Claim", `${tracePresenceName} claimed ${device}`);
    }
    if (toolRows.length > 0) {
      const toolNames = toolRows
        .slice(0, 3)
        .map((row) => String(row.tool ?? "tool").trim() || "tool")
        .join(", ");
      emitUiToast("Muse Tools", `${tracePresenceName} ran ${toolNames}`);
    }

    if (mediaActions.length > 0) {
      const firstAction = mediaActions[0];
      const status = String(firstAction.status ?? "unknown").trim() || "unknown";
      const mediaKind = String(firstAction.media_kind ?? "media").trim() || "media";
      const selectedLabel = String(firstAction.selected_label ?? "media target").trim() || "media target";
      const selectedUrl = String(firstAction.selected_url ?? firstAction.url ?? "").trim();
      emitChatMessage({
        role: "assistant",
        text: [
          "muse media action",
          `kind=${mediaKind}`,
          `status=${status}`,
          `label=${selectedLabel}`,
          `url=${selectedUrl || "(none)"}`,
        ].join("\n"),
        meta: {
          channel: "command",
          source: `${source}:media`,
          presenceId: tracePresenceId,
          presenceName: tracePresenceName,
        },
      });
    }

    const requestedMedia = mediaActions.find(
      (row) => String(row.status ?? "").trim() === "requested",
    );
    if (requestedMedia) {
      const mediaKind = String(requestedMedia.media_kind ?? "audio").trim().toLowerCase();
      const selectedUrl = String(requestedMedia.selected_url ?? requestedMedia.url ?? "").trim();
      const selectedLabel = String(requestedMedia.selected_label ?? "media target").trim() || "media target";
      if (selectedUrl) {
        if (mediaKind === "image") {
          openMuseImage(selectedUrl, selectedLabel);
        } else {
          void playMuseAudio(selectedUrl, selectedLabel);
        }
      }
    }

    if (safeReply.includes("[[PULSE]]") || overlayTags.includes("[[PULSE]]")) {
      overlayApi?.pulseAt?.(0.5, 0.5, 1);
    }
    if (safeReply.includes("[[SING]]") || overlayTags.includes("[[SING]]")) {
      overlayApi?.singAll?.();
    }
  }, [activeMusePresenceId, emitChatMessage, emitUiToast, openMuseImage, overlayApi, playMuseAudio]);

  useEffect(() => {
    if (!Array.isArray(museEvents) || museEvents.length <= 0) {
      return;
    }
    const freshEvents = museEvents.filter(
      (row: MuseEvent) => Number(row.seq ?? 0) > processedMuseEventSeqRef.current,
    );
    if (freshEvents.length <= 0) {
      return;
    }

    let maxSeq = processedMuseEventSeqRef.current;
    freshEvents.forEach((eventRow: MuseEvent) => {
      const seq = Number(eventRow.seq ?? 0);
      if (Number.isFinite(seq)) {
        maxSeq = Math.max(maxSeq, seq);
      }
      const kind = String(eventRow.kind ?? "").trim();
      const museId = String(eventRow.muse_id ?? "").trim() || "witness_thread";
      const payload = eventRow.payload && typeof eventRow.payload === "object"
        ? eventRow.payload as Record<string, unknown>
        : {};

      if (kind === "field.delta.applied") {
        const x = clamp(0.16 + (stableUnitHash(`${museId}|${eventRow.turn_id}|${eventRow.seq}`) * 0.68), 0, 1);
        const y = clamp(0.14 + (stableUnitHash(`${museId}|${eventRow.turn_id}|${eventRow.seq}|y`) * 0.72), 0, 1);
        overlayApi?.pulseAt?.(x, y, 0.56, museId);
      }

      if (kind === "muse.gpu.claim.granted") {
        overlayApi?.singAll?.();
        const device = String(payload.device ?? "gpu").trim() || "gpu";
        emitUiToast("Muse GPU Claim", `${museId} claimed ${device}`);
      }

      if (kind === "audio.play.requested") {
        const label = String(payload.label ?? "audio track").trim() || "audio track";
        const target = String(payload.target_node_id ?? "audio").trim() || "audio";
        emitUiToast("Muse Audio Request", `${museId} requested ${label} (${target})`);
      }

      if (kind === "image.open.requested") {
        const label = String(payload.label ?? "image").trim() || "image";
        const target = String(payload.target_node_id ?? "image").trim() || "image";
        emitUiToast("Muse Image Request", `${museId} requested ${label} (${target})`);
      }

      if (kind === "muse.turn.completed" && normalizeMusePresenceId(museId) === normalizeMusePresenceId(activeMusePresenceId)) {
        const deltaCount = Number(payload.field_deltas ?? 0);
        const daimonCount = Number(payload.daimoi ?? 0);
        emitUiToast(
          "Muse Turn Complete",
          `${museId} emitted ${daimonCount} daimoi and ${deltaCount} field deltas`,
        );
      }

      if (kind === "muse.rate_limited" && normalizeMusePresenceId(museId) === normalizeMusePresenceId(activeMusePresenceId)) {
        emitUiToast("Muse Rate Limit", `${museId} hit turn budget; wait a moment.`);
      }
      if (kind === "muse.rejected" && normalizeMusePresenceId(museId) === normalizeMusePresenceId(activeMusePresenceId)) {
        const reason = String(payload.reason ?? "rejected").trim() || "rejected";
        emitUiToast("Muse Rejected", `${museId} blocked turn (${reason})`);
      }
    });

    processedMuseEventSeqRef.current = maxSeq;
  }, [activeMusePresenceId, emitUiToast, museEvents, overlayApi]);

  const handleSendVoice = useCallback(async (musePresenceId: string, workspace: MuseWorkspaceContext) => {
    const text = await handleTranscribe();
    if (!text) {
      return;
    }

    const resolvedMusePresenceId = String(musePresenceId || activeMusePresenceId || "witness_thread").trim()
      || "witness_thread";
    const surroundingNodes = buildMuseSurroundingNodes(resolvedMusePresenceId, workspace);
    emitChatMessage({
      role: "user",
      text,
      meta: {
        channel: "llm",
        source: "voice",
        presenceId: resolvedMusePresenceId,
      },
    });

    const baseUrl = runtimeBaseUrl();
    try {
      const response = await fetch(`${baseUrl}/api/muse/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          muse_id: resolvedMusePresenceId || "witness_thread",
          multi_entity: true,
          presence_ids: [resolvedMusePresenceId || "witness_thread"],
          text,
          mode: "stochastic",
          token_budget: 2048,
          graph_revision: simulation?.timestamp || catalog?.generated_at || "",
          surrounding_nodes: surroundingNodes,
        }),
      });
      if (!response.ok) {
        throw new Error(`muse request failed (${response.status})`);
      }
      const payload = (await response.json()) as {
        reply?: unknown;
        mode?: unknown;
        model?: unknown;
        trace?: unknown;
      };
      emitWitnessChatReply(payload, "voice:/api/muse/message", resolvedMusePresenceId);
    } catch {
      emitChatMessage({
        role: "assistant",
        text: "voice chat request failed",
        meta: {
          channel: "command",
          source: "voice:/api/muse/message",
          presenceId: resolvedMusePresenceId,
        },
      });
    }
  }, [
    activeMusePresenceId,
    buildMuseSurroundingNodes,
    catalog,
    emitChatMessage,
    emitWitnessChatReply,
    handleTranscribe,
    simulation,
  ]);

  const emitSystemMessage = useCallback((text: string) => {
    emitChatMessage({
      role: "assistant",
      text,
      meta: {
        channel: "command",
        source: "muse:system",
        presenceId: activeMusePresenceId,
      },
    });
  }, [activeMusePresenceId, emitChatMessage]);

  const handleOverlayInit = useCallback((api: unknown) => {
    setOverlayApi(api as OverlayApi);
  }, []);

  const handleMuseWorkspaceBindingsChange = useCallback((presenceId: string, fileNodeIds: string[]) => {
    const normalizedPresence = normalizeMusePresenceId(String(presenceId || "").trim() || "witness_thread");
    const normalizedIds = fileNodeIds
      .map((item) => String(item || "").trim())
      .filter((item, index, all) => item.length > 0 && all.indexOf(item) === index)
      .slice(0, 48);
    setMuseWorkspaceBindings((prev) => {
      const prevIds = prev[normalizedPresence] ?? [];
      if (prevIds.length === normalizedIds.length && prevIds.every((id, index) => id === normalizedIds[index])) {
        return prev;
      }
      return {
        ...prev,
        [normalizedPresence]: normalizedIds,
      };
    });
    setMuseWorkspaceContexts((prev) => {
      const currentWorkspace = normalizeMuseWorkspaceContext(
        prev[normalizedPresence],
        APP_WORKSPACE_NORMALIZE_OPTIONS,
      );
      if (sameStringArray(currentWorkspace.pinnedFileNodeIds, normalizedIds)) {
        return prev;
      }
      return {
        ...prev,
        [normalizedPresence]: {
          ...currentWorkspace,
          pinnedFileNodeIds: normalizedIds,
        },
      };
    });

    const baseUrl = runtimeBaseUrl();
    void fetch(`${baseUrl}/api/muse/sync-pins`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        muse_id: normalizedPresence,
        pinned_node_ids: normalizedIds,
        reason: "ui.workspace.sync",
      }),
    }).catch(() => {
      // keep local state even if backend sync fails
    });
  }, []);

  const handleMuseWorkspaceContextChange = useCallback((
    presenceId: string,
    workspace: MuseWorkspaceContext,
  ) => {
    const normalizedPresence = normalizeMusePresenceId(String(presenceId || "").trim() || "witness_thread");
    const normalizedWorkspace = normalizeMuseWorkspaceContext(
      workspace,
      APP_WORKSPACE_NORMALIZE_OPTIONS,
    );
    setMuseWorkspaceContexts((prev) => {
      const currentWorkspace = normalizeMuseWorkspaceContext(
        prev[normalizedPresence],
        APP_WORKSPACE_NORMALIZE_OPTIONS,
      );
      const pinnedUnchanged = sameStringArray(
        currentWorkspace.pinnedFileNodeIds,
        normalizedWorkspace.pinnedFileNodeIds,
      );
      const searchUnchanged = currentWorkspace.searchQuery === normalizedWorkspace.searchQuery;
      const summariesUnchanged = sameStringArray(
        currentWorkspace.pinnedNexusSummaries,
        normalizedWorkspace.pinnedNexusSummaries,
      );
      if (pinnedUnchanged && searchUnchanged && summariesUnchanged) {
        return prev;
      }
      return {
        ...prev,
        [normalizedPresence]: normalizedWorkspace,
      };
    });
    setMuseWorkspaceBindings((prev) => {
      const prevIds = prev[normalizedPresence] ?? [];
      if (sameStringArray(prevIds, normalizedWorkspace.pinnedFileNodeIds)) {
        return prev;
      }
      return {
        ...prev,
        [normalizedPresence]: normalizedWorkspace.pinnedFileNodeIds,
      };
    });
  }, []);

  const {
    autopilotEnabled,
    autopilotStatus,
    autopilotSummary,
    autopilotEvents,
    handleAutopilotUserInput,
    toggleAutopilot,
  } = useAutopilotController({ catalog, simulation, isConnected, emitSystemMessage });

  const nudgeCoreZoom = useCallback((delta: number) => {
    setCoreCameraZoom((prev) => clamp(prev + delta, CORE_CAMERA_ZOOM_MIN, CORE_CAMERA_ZOOM_MAX));
  }, []);

  const toggleCoreFlight = useCallback(() => {
    setCoreFlightEnabled((prev) => !prev);
  }, []);

  const nudgeCoreFlightSpeed = useCallback((delta: number) => {
    setCoreFlightSpeed((prev) => clamp(prev + delta, CORE_FLIGHT_SPEED_MIN, CORE_FLIGHT_SPEED_MAX));
  }, []);

  const toggleCoreOrbit = useCallback(() => {
    setCoreOrbitEnabled((prev) => !prev);
  }, []);

  const nudgeCoreOrbitSpeed = useCallback((delta: number) => {
    setCoreOrbitSpeed((prev) => clamp(prev + delta, CORE_ORBIT_SPEED_MIN, CORE_ORBIT_SPEED_MAX));
  }, []);

  const setCoreSimulationDial = useCallback((dial: keyof CoreSimulationTuning, value: number) => {
    setCoreSimulationTuning((prev) => {
      if (dial === "particleDensity") {
        return {
          ...prev,
          particleDensity: clamp(value, CORE_SIM_PARTICLE_DENSITY_MIN, CORE_SIM_PARTICLE_DENSITY_MAX),
        };
      }
      if (dial === "particleScale") {
        return {
          ...prev,
          particleScale: clamp(value, CORE_SIM_PARTICLE_SCALE_MIN, CORE_SIM_PARTICLE_SCALE_MAX),
        };
      }
      if (dial === "mouseInfluence") {
        return {
          ...prev,
          mouseInfluence: clamp(value, CORE_SIM_MOUSE_INFLUENCE_MIN, CORE_SIM_MOUSE_INFLUENCE_MAX),
        };
      }
      if (dial === "layerDepth") {
        return {
          ...prev,
          layerDepth: clamp(value, CORE_SIM_LAYER_DEPTH_MIN, CORE_SIM_LAYER_DEPTH_MAX),
        };
      }
      if (dial === "graphNodeSmoothness") {
        return {
          ...prev,
          graphNodeSmoothness: clamp(
            value,
            CORE_SIM_GRAPH_NODE_SMOOTHING_MIN,
            CORE_SIM_GRAPH_NODE_SMOOTHING_MAX,
          ),
        };
      }
      if (dial === "graphNodeStepScale") {
        return {
          ...prev,
          graphNodeStepScale: clamp(
            value,
            CORE_SIM_GRAPH_NODE_STEP_SCALE_MIN,
            CORE_SIM_GRAPH_NODE_STEP_SCALE_MAX,
          ),
        };
      }
      if (dial === "motionSpeed") {
        return {
          ...prev,
          motionSpeed: clamp(value, CORE_SIM_MOTION_SPEED_MIN, CORE_SIM_MOTION_SPEED_MAX),
        };
      }
      return {
        ...prev,
      };
    });
  }, []);

  const resetCoreSimulationTuning = useCallback(() => {
    setCoreSimulationTuning(DEFAULT_CORE_SIMULATION_TUNING);
  }, []);

  const updateMouseDaimonTuning = useCallback((partial: Partial<MouseDaimonTuning>) => {
    setMouseDaimonTuning((prev) => ({ ...prev, ...partial }));
  }, []);

  const clearUserPresenceInputBatchTimers = useCallback(() => {
    if (userPresenceInputBatchIdleTimerRef.current !== null) {
      window.clearTimeout(userPresenceInputBatchIdleTimerRef.current);
      userPresenceInputBatchIdleTimerRef.current = null;
    }
    if (userPresenceInputBatchWindowTimerRef.current !== null) {
      window.clearTimeout(userPresenceInputBatchWindowTimerRef.current);
      userPresenceInputBatchWindowTimerRef.current = null;
    }
  }, []);

  const flushUserPresenceInputBatch = useCallback((reason: string) => {
    if (userPresenceInputBatchRef.current.length <= 0) {
      clearUserPresenceInputBatchTimers();
      return;
    }

    const events = userPresenceInputBatchRef.current.slice(0, USER_PRESENCE_BATCH_MAX_EVENTS);
    userPresenceInputBatchRef.current = [];
    userPresenceInputBatchStartedMsRef.current = 0;
    userPresenceInputBatchLastMsRef.current = 0;
    clearUserPresenceInputBatchTimers();

    const baseUrl = runtimeBaseUrl();
    void fetch(`${baseUrl}/api/presence/user/input`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        events,
        flush_reason: reason,
        flushed_at_ms: Date.now(),
      }),
      keepalive: true,
    }).catch(() => undefined);
  }, [clearUserPresenceInputBatchTimers]);

  const scheduleUserPresenceInputBatchFlush = useCallback(() => {
    if (userPresenceInputBatchRef.current.length <= 0) {
      clearUserPresenceInputBatchTimers();
      return;
    }

    const nowMs = Date.now();
    const startedMs = userPresenceInputBatchStartedMsRef.current || nowMs;
    const lastMs = userPresenceInputBatchLastMsRef.current || nowMs;
    const idleDelayMs = Math.max(0, USER_PRESENCE_BATCH_IDLE_FLUSH_MS - (nowMs - lastMs));
    const maxWindowDelayMs = Math.max(0, USER_PRESENCE_BATCH_MAX_WINDOW_MS - (nowMs - startedMs));

    clearUserPresenceInputBatchTimers();
    userPresenceInputBatchIdleTimerRef.current = window.setTimeout(() => {
      flushUserPresenceInputBatch("idle_window");
    }, idleDelayMs);
    userPresenceInputBatchWindowTimerRef.current = window.setTimeout(() => {
      flushUserPresenceInputBatch("max_window");
    }, maxWindowDelayMs);
  }, [clearUserPresenceInputBatchTimers, flushUserPresenceInputBatch]);

  const emitUserPresenceInput = useCallback((payload: UserPresenceInputPayload) => {
    const kind = String(payload.kind || "input").trim().toLowerCase() || "input";
    const target = String(payload.target || "simulation").trim() || "simulation";
    const nowMs = Date.now();
    const eventRow: Record<string, unknown> = {
      kind,
      target,
      message: String(payload.message || "").trim(),
      embed_daimoi: Boolean(payload.embedDaimoi),
      meta: payload.meta && typeof payload.meta === "object" ? payload.meta : {},
      ts_client_ms: nowMs,
    };
    if (typeof payload.xRatio === "number" && Number.isFinite(payload.xRatio)) {
      eventRow.x_ratio = Math.max(0, Math.min(1, payload.xRatio));
    }
    if (typeof payload.yRatio === "number" && Number.isFinite(payload.yRatio)) {
      eventRow.y_ratio = Math.max(0, Math.min(1, payload.yRatio));
    }

    if (userPresenceInputBatchRef.current.length <= 0) {
      userPresenceInputBatchStartedMsRef.current = nowMs;
    }
    userPresenceInputBatchRef.current.push(eventRow);
    userPresenceInputBatchLastMsRef.current = nowMs;

    if (userPresenceInputBatchRef.current.length >= USER_PRESENCE_BATCH_MAX_EVENTS) {
      flushUserPresenceInputBatch("max_events");
      return;
    }
    scheduleUserPresenceInputBatchFlush();
  }, [flushUserPresenceInputBatch, scheduleUserPresenceInputBatchFlush]);

  const handleUserPresenceInput = useCallback((payload: UserPresenceInputPayload) => {
    emitUserPresenceInput(payload);
  }, [emitUserPresenceInput]);

  useEffect(() => {
    const onPageHide = () => {
      flushUserPresenceInputBatch("pagehide");
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        flushUserPresenceInputBatch("hidden");
      }
    };

    window.addEventListener("pagehide", onPageHide);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("pagehide", onPageHide);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      flushUserPresenceInputBatch("unmount");
      clearUserPresenceInputBatchTimers();
    };
  }, [clearUserPresenceInputBatchTimers, flushUserPresenceInputBatch]);

  useEffect(() => {
    const describeInteractiveTarget = (element: Element): string => {
      const target = element as HTMLElement;
      const aria = (target.getAttribute("aria-label") || "").trim();
      const title = (target.getAttribute("title") || "").trim();
      const id = target.id.trim();
      const dataPanel = (target.getAttribute("data-panel-id") || "").trim();
      const text = (target.textContent || "").trim().replace(/\s+/g, " ").slice(0, 64);
      const tag = target.tagName.toLowerCase();
      const identity = aria || title || text || dataPanel || id || tag;
      return `${tag}:${identity}`;
    };

    const onPointerOver = (event: Event) => {
      const element = (event.target as Element | null)?.closest(
        "button, [role='button'], input, select, textarea, a",
      );
      if (!element) {
        return;
      }
      const target = describeInteractiveTarget(element);
      const nowMs = Date.now();
      const lastMs = userPresenceHoverThrottleRef.current[target] ?? 0;
      if ((nowMs - lastMs) < 500) {
        return;
      }
      userPresenceHoverThrottleRef.current[target] = nowMs;
      emitUserPresenceInput({
        kind: "hover",
        target,
        message: `mouse hover over ${target}`,
        embedDaimoi: true,
        meta: {
          source: "ui-control",
        },
      });
    };

    const onClick = (event: Event) => {
      const element = (event.target as Element | null)?.closest(
        "button, [role='button'], input, select, textarea, a",
      );
      if (!element) {
        return;
      }
      const target = describeInteractiveTarget(element);
      emitUserPresenceInput({
        kind: "click",
        target,
        message: `click ${target}`,
        embedDaimoi: true,
        meta: {
          source: "ui-control",
        },
      });
    };

    const onInput = (event: Event) => {
      const element = (event.target as Element | null)?.closest("input, textarea, select");
      if (!element) {
        return;
      }
      const target = describeInteractiveTarget(element);
      emitUserPresenceInput({
        kind: "input",
        target,
        message: `input change on ${target}`,
        embedDaimoi: true,
        meta: {
          source: "ui-control",
        },
      });
    };

    const onKeyDown = (event: KeyboardEvent) => {
      const nowMs = Date.now();
      if ((nowMs - userPresenceGenericEmitMsRef.current) < 120) {
        return;
      }
      userPresenceGenericEmitMsRef.current = nowMs;
      const active = document.activeElement as HTMLElement | null;
      const target = active ? describeInteractiveTarget(active) : "keyboard:window";
      emitUserPresenceInput({
        kind: "keydown",
        target,
        message: `key ${event.key} on ${target}`,
        embedDaimoi: true,
        meta: {
          source: "keyboard",
          key: event.key,
        },
      });
    };

    window.addEventListener("pointerover", onPointerOver, true);
    window.addEventListener("click", onClick, true);
    window.addEventListener("input", onInput, true);
    window.addEventListener("keydown", onKeyDown, true);

    return () => {
      window.removeEventListener("pointerover", onPointerOver, true);
      window.removeEventListener("click", onClick, true);
      window.removeEventListener("input", onInput, true);
      window.removeEventListener("keydown", onKeyDown, true);
    };
  }, [emitUserPresenceInput]);

  const setCoreVisualDial = useCallback((dial: keyof CoreVisualTuning, value: number) => {
    setCoreVisualTuning((prev) => {
      if (dial === "brightness") {
        return {
          ...prev,
          brightness: clamp(value, CORE_VISUAL_BRIGHTNESS_MIN, CORE_VISUAL_BRIGHTNESS_MAX),
        };
      }
      if (dial === "contrast") {
        return {
          ...prev,
          contrast: clamp(value, CORE_VISUAL_CONTRAST_MIN, CORE_VISUAL_CONTRAST_MAX),
        };
      }
      if (dial === "saturation") {
        return {
          ...prev,
          saturation: clamp(value, CORE_VISUAL_SATURATION_MIN, CORE_VISUAL_SATURATION_MAX),
        };
      }
      if (dial === "hueRotate") {
        return {
          ...prev,
          hueRotate: clamp(value, CORE_VISUAL_HUE_MIN, CORE_VISUAL_HUE_MAX),
        };
      }
      if (dial === "backgroundWash") {
        return {
          ...prev,
          backgroundWash: clamp(value, CORE_VISUAL_WASH_MIN, CORE_VISUAL_WASH_MAX),
        };
      }
      return {
        ...prev,
        vignette: clamp(value, CORE_VISUAL_VIGNETTE_MIN, CORE_VISUAL_VIGNETTE_MAX),
      };
    });
  }, []);

  const resetCoreVisualTuning = useCallback(() => {
    setCoreVisualTuning(DEFAULT_CORE_VISUAL_TUNING);
  }, []);

  const setInterfaceOpacityDial = useCallback((value: number) => {
    setInterfaceOpacity(clamp(value, INTERFACE_OPACITY_MIN, INTERFACE_OPACITY_MAX));
  }, []);

  const resetInterfaceOpacity = useCallback(() => {
    setInterfaceOpacity(DEFAULT_INTERFACE_OPACITY);
  }, []);

  const boostCoreVisibility = useCallback(() => {
    setCoreVisualTuning(HIGH_VISIBILITY_CORE_VISUAL_TUNING);
  }, []);

  const applyCoreLayerPreset = useCallback((nextView: OverlayViewId) => {
    const normalizedView: OverlayViewId = nextView === "crawler-graph" ? "file-graph" : nextView;
    const nexusGraphView = normalizedView === "file-graph";
    setCoreOverlayView(normalizedView);
    if (nextView === "omni") {
      setCoreLayerVisibility({ ...DEFAULT_CORE_LAYER_VISIBILITY });
      return;
    }
    setCoreLayerVisibility({
      presence: normalizedView === "presence" || nexusGraphView,
      "file-impact": normalizedView === "file-impact",
      "file-graph": nexusGraphView,
      "true-graph": normalizedView === "true-graph" || nexusGraphView,
      "truth-gate": normalizedView === "truth-gate",
      logic: normalizedView === "logic",
      "pain-field": normalizedView === "pain-field",
    });
  }, []);

  const setCoreLayerEnabled = useCallback((layerId: CoreLayerId, enabled: boolean) => {
    setCoreLayerVisibility((prev) => ({
      ...prev,
      [layerId]: enabled,
    }));
  }, []);

  const setAllCoreLayers = useCallback((enabled: boolean) => {
    setCoreLayerVisibility({
      presence: enabled,
      "file-impact": enabled,
      "file-graph": enabled,
      "true-graph": enabled,
      "truth-gate": enabled,
      logic: enabled,
      "pain-field": enabled,
    });
    setCoreOverlayView(enabled ? "omni" : "presence");
  }, []);

  const activeCoreLayerCount = useMemo(
    () => CORE_LAYER_OPTIONS.reduce((count, option) => count + (coreLayerVisibility[option.id] ? 1 : 0), 0),
    [coreLayerVisibility],
  );

  const togglePanelPin = useCallback((panelId: string) => {
    if (panelId === GLASS_VIEWPORT_PANEL_ID) {
      setPinnedPanels((prev) => ({
        ...prev,
        [panelId]: true,
      }));
      return;
    }
    setPinnedPanels((prev) => ({
      ...prev,
      [panelId]: !prev[panelId],
    }));
  }, []);

  const adjustPanelCouncilRank = useCallback((panelId: string, delta: number) => {
    if (!panelId || !Number.isFinite(delta) || delta === 0) {
      return;
    }
    setPanelCouncilBoosts((prev) => {
      const current = prev[panelId] ?? 0;
      const next = clamp(current + delta, -6, 8);
      if (next === 0) {
        if (current === 0) {
          return prev;
        }
        const nextState = { ...prev };
        delete nextState[panelId];
        return nextState;
      }
      return {
        ...prev,
        [panelId]: next,
      };
    });
    setSelectedPanelId(panelId);
    setPanelWindowStates((prev) => ({
      ...prev,
      [panelId]: {
        open: true,
        minimized: false,
      },
    }));
  }, []);

  const pinPanelToTertiary = useCallback((panelId: string) => {
    const id = panelId.trim();
    if (!id) {
      return;
    }
    setTertiaryPinnedPanelId((prev) => (prev === id ? null : id));
    setPanelWindowStates((prev) => ({
      ...prev,
      [id]: {
        open: true,
        minimized: false,
      },
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
    if (corePointerFrameRef.current !== null) {
      window.cancelAnimationFrame(corePointerFrameRef.current);
      corePointerFrameRef.current = null;
    }
    corePointerPendingRef.current = { dx: 0, dy: 0 };
    setCoreCameraZoom(1);
    setCoreCameraPitch(0);
    setCoreCameraYaw(0);
    setCoreCameraPosition({ x: 0, y: 0, z: 0 });
    coreFlightVelocityRef.current = { x: 0, y: 0, z: 0 };
  }, [stopCameraFlight]);

  const nudgeCameraPan = useCallback((
    xRatioDelta: number,
    yRatioDelta: number,
    sourcePanelId?: string,
  ) => {
    stopCameraFlight();
    const dx = clamp(Number.isFinite(xRatioDelta) ? xRatioDelta : 0, -0.28, 0.28);
    const dy = clamp(Number.isFinite(yRatioDelta) ? yRatioDelta : 0, -0.28, 0.28);
    if (Math.abs(dx) < 1e-6 && Math.abs(dy) < 1e-6) {
      return;
    }

    setCoreCameraPosition((prev) => ({
      x: clamp(prev.x + (dx * 228), -CORE_CAMERA_X_LIMIT, CORE_CAMERA_X_LIMIT),
      y: clamp(prev.y + (dy * 176), -CORE_CAMERA_Y_LIMIT, CORE_CAMERA_Y_LIMIT),
      z: prev.z,
    }));
    setCoreCameraYaw((prev) =>
      clamp(prev + (dx * 7.2), CORE_CAMERA_YAW_MIN, CORE_CAMERA_YAW_MAX),
    );
    setCoreCameraPitch((prev) =>
      clamp(prev + (dy * 6.4), CORE_CAMERA_PITCH_MIN, CORE_CAMERA_PITCH_MAX),
    );

    const now = performance.now();
    if (now - lastNudgePulseTsRef.current >= 48) {
      const pulseX = clamp(0.5 + (dx * 0.92), 0.08, 0.92);
      const pulseY = clamp(0.5 + (dy * 0.92), 0.08, 0.92);
      overlayApi?.pulseAt?.(pulseX, pulseY, 0.74, sourcePanelId ?? "view_lens_keeper");
      lastNudgePulseTsRef.current = now;
    }
  }, [overlayApi, stopCameraFlight]);

  const resolveOverlayAnchorRatio = useCallback(
    (anchor: WorldAnchorTarget, panelAnchorId?: string): { x: number; y: number; label?: string } | null => {
      const getAnchorRatio = overlayApi?.getAnchorRatio;
      if (!getAnchorRatio) {
        return null;
      }

      const candidateIds = Array.from(
        new Set([
          String(panelAnchorId ?? "").trim(),
          String(anchor.id ?? "").trim(),
          String(anchor.label ?? "").trim(),
        ].filter((value) => value.length > 0)),
      );
      if (candidateIds.length === 0) {
        return null;
      }

      const candidateKinds = Array.from(new Set([anchor.kind, "presence", "region", "cluster", "node"]));
      for (const candidateId of candidateIds) {
        for (const kind of candidateKinds) {
          const found = getAnchorRatio(kind, candidateId);
          if (!found) {
            continue;
          }
          return {
            x: clamp(Number(found.x ?? 0.5), 0, 1),
            y: clamp(Number(found.y ?? 0.5), 0, 1),
            label: typeof found.label === "string" ? found.label : undefined,
          };
        }
      }

      return null;
    },
    [overlayApi],
  );

  const flyCameraToRatios = useCallback((
    anchorX: number,
    anchorY: number,
    anchorKind: WorldAnchorTarget["kind"],
    focusCenterX = 0.5,
    focusCenterY = 0.5,
  ) => {
    stopCameraFlight();
    const normalizedAnchorX = clamp(anchorX, 0, 1);
    const normalizedAnchorY = clamp(anchorY, 0, 1);
    const normalizedFocusX = clamp(focusCenterX, 0.04, 0.96);
    const normalizedFocusY = clamp(focusCenterY, 0.04, 0.96);
    const start = {
      x: coreCameraPosition.x,
      y: coreCameraPosition.y,
      z: coreCameraPosition.z,
      yaw: coreCameraYaw,
      pitch: coreCameraPitch,
      zoom: coreCameraZoom,
    };
    const target = {
      x: clamp((normalizedFocusX - normalizedAnchorX) * 640, -CORE_CAMERA_X_LIMIT, CORE_CAMERA_X_LIMIT),
      y: clamp((normalizedFocusY - normalizedAnchorY) * 520, -CORE_CAMERA_Y_LIMIT, CORE_CAMERA_Y_LIMIT),
      z: clamp(
        anchorKind === "node" ? 180 : anchorKind === "cluster" ? 40 : -120,
        CORE_CAMERA_Z_MIN,
        CORE_CAMERA_Z_MAX,
      ),
      yaw: clamp((normalizedAnchorX - normalizedFocusX) * 68, CORE_CAMERA_YAW_MIN, CORE_CAMERA_YAW_MAX),
      pitch: clamp((normalizedFocusY - normalizedAnchorY) * 52, CORE_CAMERA_PITCH_MIN, CORE_CAMERA_PITCH_MAX),
      zoom: clamp(anchorKind === "node" ? 1.18 : anchorKind === "cluster" ? 1.06 : 0.94, CORE_CAMERA_ZOOM_MIN, CORE_CAMERA_ZOOM_MAX),
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

  const flyCameraToAnchor = useCallback((anchor: WorldAnchorTarget) => {
    const overlayAnchor = resolveOverlayAnchorRatio(anchor);
    const anchorX = overlayAnchor?.x ?? anchor.x;
    const anchorY = overlayAnchor?.y ?? anchor.y;

    if (overlayAnchor) {
      overlayApi?.pulseAt?.(overlayAnchor.x, overlayAnchor.y, 1.12, anchor.id);
    }
    flyCameraToRatios(anchorX, anchorY, anchor.kind);
  }, [flyCameraToRatios, overlayApi, resolveOverlayAnchorRatio]);

  useEffect(() => {
    return () => {
      stopCameraFlight();
    };
  }, [stopCameraFlight]);

  useEffect(() => {
    return () => {
      if (corePointerFrameRef.current !== null) {
        window.cancelAnimationFrame(corePointerFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!coreOrbitEnabled) {
      setCoreOrbitPhase(0);
      return;
    }

    let rafId = 0;
    const startTs = performance.now();
    let lastEmitTs = startTs;
    const frameIntervalMs = 1000 / 30;
    const angularVelocity = ((Math.PI * 2) / CORE_ORBIT_PERIOD_SECONDS) * coreOrbitSpeed;

    const tick = (ts: number) => {
      if (ts - lastEmitTs >= frameIntervalMs) {
        const elapsedSeconds = (ts - startTs) / 1000;
        setCoreOrbitPhase(elapsedSeconds * angularVelocity);
        lastEmitTs = ts;
      }
      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, [coreOrbitEnabled, coreOrbitSpeed]);

  const coreOrbitOffset = useMemo(() => {
    if (!coreOrbitEnabled) {
      return { x: 0, y: 0, z: 0 };
    }
    return {
      x: Math.cos(coreOrbitPhase) * CORE_ORBIT_RADIUS_X,
      y: Math.sin((coreOrbitPhase * 0.63) + 0.42) * CORE_ORBIT_RADIUS_Y,
      z: Math.sin(coreOrbitPhase) * CORE_ORBIT_RADIUS_Z,
    };
  }, [coreOrbitEnabled, coreOrbitPhase]);

  const coreRenderedCameraPosition = useMemo(
    () => ({
      x: clamp(coreCameraPosition.x + coreOrbitOffset.x, -CORE_CAMERA_X_LIMIT, CORE_CAMERA_X_LIMIT),
      y: clamp(coreCameraPosition.y + coreOrbitOffset.y, -CORE_CAMERA_Y_LIMIT, CORE_CAMERA_Y_LIMIT),
      z: clamp(coreCameraPosition.z + coreOrbitOffset.z, CORE_CAMERA_Z_MIN, CORE_CAMERA_Z_MAX),
    }),
    [coreCameraPosition, coreOrbitOffset],
  );
  const deferredCoreRenderedCameraPosition = useDeferredValue(coreRenderedCameraPosition);

  const coreCameraTransform = useMemo(
    () =>
      `perspective(1800px) translate3d(${coreRenderedCameraPosition.x.toFixed(1)}px, ${coreRenderedCameraPosition.y.toFixed(1)}px, ${coreRenderedCameraPosition.z.toFixed(1)}px) rotateX(${coreCameraPitch.toFixed(2)}deg) rotateY(${coreCameraYaw.toFixed(2)}deg) scale(${coreCameraZoom.toFixed(3)})`,
    [coreCameraPitch, coreCameraYaw, coreCameraZoom, coreRenderedCameraPosition],
  );

  const coreSimulationFilter = useMemo(
    () =>
      `saturate(${coreVisualTuning.saturation.toFixed(3)}) contrast(${coreVisualTuning.contrast.toFixed(3)}) brightness(${coreVisualTuning.brightness.toFixed(3)}) hue-rotate(${coreVisualTuning.hueRotate.toFixed(1)}deg)`,
    [coreVisualTuning],
  );

  const handleCorePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (isCorePointerBlockedTarget(event.target)) {
        return;
      }
      event.preventDefault();
      stopCameraFlight();
      const mode = "pan";
      coreDragRef.current = {
        active: true,
        pointerId: event.pointerId,
        mode,
        startX: event.clientX,
        startY: event.clientY,
        lastX: event.clientX,
        lastY: event.clientY,
        startPitch: coreCameraPitch,
        startYaw: coreCameraYaw,
        startCamX: coreCameraPosition.x,
        startCamY: coreCameraPosition.y,
      };
      corePointerPendingRef.current = { dx: 0, dy: 0 };
      if (corePointerFrameRef.current !== null) {
        window.cancelAnimationFrame(corePointerFrameRef.current);
        corePointerFrameRef.current = null;
      }
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [coreCameraPitch, coreCameraPosition.x, coreCameraPosition.y, coreCameraYaw, stopCameraFlight],
  );

  const flushCorePointerPending = useCallback(() => {
    const pending = corePointerPendingRef.current;
    corePointerPendingRef.current = { dx: 0, dy: 0 };
    const activeDrag = coreDragRef.current;
    if (!activeDrag || !activeDrag.active) {
      return;
    }
    if (Math.abs(pending.dx) < 0.01 && Math.abs(pending.dy) < 0.01) {
      return;
    }
    if (activeDrag.mode === "pan") {
      setCoreCameraPosition((prev) => ({
        x: clamp(prev.x + (pending.dx * 1.45), -CORE_CAMERA_X_LIMIT, CORE_CAMERA_X_LIMIT),
        y: clamp(prev.y + (pending.dy * 1.45), -CORE_CAMERA_Y_LIMIT, CORE_CAMERA_Y_LIMIT),
        z: prev.z,
      }));
      return;
    }
    setCoreCameraYaw((prev) =>
      clamp(prev + (pending.dx * 0.08), CORE_CAMERA_YAW_MIN, CORE_CAMERA_YAW_MAX),
    );
    setCoreCameraPitch((prev) =>
      clamp(prev + (pending.dy * 0.08), CORE_CAMERA_PITCH_MIN, CORE_CAMERA_PITCH_MAX),
    );
  }, []);

  const scheduleCorePointerFlush = useCallback(() => {
    if (corePointerFrameRef.current !== null) {
      return;
    }
    corePointerFrameRef.current = window.requestAnimationFrame(() => {
      corePointerFrameRef.current = null;
      flushCorePointerPending();
    });
  }, [flushCorePointerPending]);

  const handleCorePointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = coreDragRef.current;
    if (!drag || !drag.active || drag.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    const dx = event.clientX - drag.lastX;
    const dy = event.clientY - drag.lastY;
    drag.lastX = event.clientX;
    drag.lastY = event.clientY;
    if (Math.abs(dx) < 0.1 && Math.abs(dy) < 0.1) {
      return;
    }
    corePointerPendingRef.current.dx += dx;
    corePointerPendingRef.current.dy += dy;
    if (corePointerFrameRef.current !== null) {
      return;
    }
    flushCorePointerPending();
    scheduleCorePointerFlush();
  }, [flushCorePointerPending, scheduleCorePointerFlush]);

  const handleCorePointerUp = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = coreDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    if (corePointerFrameRef.current !== null) {
      window.cancelAnimationFrame(corePointerFrameRef.current);
      corePointerFrameRef.current = null;
    }
    flushCorePointerPending();
    corePointerPendingRef.current = { dx: 0, dy: 0 };
    coreDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, [flushCorePointerPending]);

  const applyCoreWheelDelta = useCallback((deltaY: number, shiftKey: boolean) => {
    if (shiftKey) {
      const speedDelta = deltaY < 0 ? 0.08 : -0.08;
      setCoreFlightSpeed((prev) => clamp(prev + speedDelta, CORE_FLIGHT_SPEED_MIN, CORE_FLIGHT_SPEED_MAX));
      return;
    }
    const delta = deltaY < 0 ? 0.06 : -0.06;
    setCoreCameraZoom((prev) => clamp(prev + delta, CORE_CAMERA_ZOOM_MIN, CORE_CAMERA_ZOOM_MAX));
  }, []);

  const handleCoreWheel = useCallback((event: ReactWheelEvent<HTMLDivElement>) => {
    if (event.defaultPrevented) {
      return;
    }
    if (!shouldRouteWheelToCore(event.target, event.deltaY)) {
      return;
    }
    event.preventDefault();
    applyCoreWheelDelta(event.deltaY, event.shiftKey);
  }, [applyCoreWheelDelta]);

  useEffect(() => {
    const onGlobalWheel = (event: WheelEvent) => {
      if (event.defaultPrevented) {
        return;
      }
      if (!shouldRouteWheelToCore(event.target, event.deltaY)) {
        return;
      }
      event.preventDefault();
      applyCoreWheelDelta(event.deltaY, event.shiftKey);
    };
    window.addEventListener("wheel", onGlobalWheel, { passive: false, capture: true });
    return () => {
      window.removeEventListener("wheel", onGlobalWheel, true);
    };
  }, [applyCoreWheelDelta]);

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
    let lastEmitTs = lastTs;
    let pendingX = 0;
    let pendingY = 0;
    let pendingZ = 0;
    const emitIntervalMs = 1000 / 45;

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

      pendingX += velocity.x;
      pendingY += velocity.y;
      pendingZ += velocity.z;

      if (now - lastEmitTs >= emitIntervalMs) {
        setCoreCameraPosition((prev) => ({
          x: clamp(prev.x + pendingX, -CORE_CAMERA_X_LIMIT, CORE_CAMERA_X_LIMIT),
          y: clamp(prev.y + pendingY, -CORE_CAMERA_Y_LIMIT, CORE_CAMERA_Y_LIMIT),
          z: clamp(prev.z + pendingZ, CORE_CAMERA_Z_MIN, CORE_CAMERA_Z_MAX),
        }));
        pendingX = 0;
        pendingY = 0;
        pendingZ = 0;
        lastEmitTs = now;
      }

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
    async (text: string, fallbackMusePresenceId = activeMusePresenceId): Promise<boolean> => {
      const trimmed = text.trim();
      if (!trimmed.toLowerCase().startsWith("/say")) {
        return false;
      }

      const args = trimmed.replace(/^\/say\s*/i, "");
      const [presenceIdRaw, ...rest] = args.split(/\s+/).filter((token) => token.length > 0);
      const presence_id = presenceIdRaw || fallbackMusePresenceId || "witness_thread";
      const messageText = rest.join(" ");
      const surroundingNodes = buildMuseSurroundingNodes(presence_id, null);

      try {
        const baseUrl = runtimeBaseUrl();
        const response = await fetch(`${baseUrl}/api/muse/message`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            muse_id: presence_id,
            text: messageText,
            mode: "deterministic",
            token_budget: 1024,
            graph_revision: simulation?.timestamp || catalog?.generated_at || "",
            surrounding_nodes: surroundingNodes,
          }),
        });
        if (!response.ok) {
          throw new Error(`muse say request failed (${response.status})`);
        }
        const payload = (await response.json()) as {
          ok?: boolean;
          turn_id?: string;
          muse?: { label?: string };
          reply?: string;
          mode?: string;
          model?: string;
          fallback?: boolean;
          daimoi?: unknown[];
          field_deltas?: unknown[];
          gpu_claim?: Record<string, unknown>;
          tool_results?: unknown[];
          manifest?: {
            explicit_selected?: unknown[];
            surround_selected?: unknown[];
          };
        };
        emitWitnessChatReply(payload, "command:/say", presence_id);
        emitSystemMessage(
          `${payload?.muse?.label || presence_id} / muse turn ${payload?.turn_id || ""}\n${payload?.reply || "(no reply)"}\n` +
            `explicit=${payload?.manifest?.explicit_selected?.length || 0} surrounding=${payload?.manifest?.surround_selected?.length || 0}`,
        );
      } catch {
        emitSystemMessage("muse say failed");
      }
      return true;
    },
    [
      activeMusePresenceId,
      buildMuseSurroundingNodes,
      catalog?.generated_at,
      emitSystemMessage,
      emitWitnessChatReply,
      simulation?.timestamp,
    ],
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
    async (text: string, musePresenceId = activeMusePresenceId): Promise<boolean> => {
      if (await handleLedgerCommand(text)) {
        return true;
      }
      if (await handlePresenceSayCommand(text, musePresenceId)) {
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
      activeMusePresenceId,
      handleDriftCommand,
      handleLedgerCommand,
      handlePresenceSayCommand,
      handlePushTruthDryRunCommand,
      handleStudyCommand,
    ],
  );

  const handleMuseWorkspaceSend = useCallback((text: string, musePresenceId: string, workspace: MuseWorkspaceContext) => {
    const resolvedMusePresenceId = String(musePresenceId || activeMusePresenceId || "witness_thread").trim()
      || "witness_thread";
    setActiveMusePresenceId(resolvedMusePresenceId);
    if (handleAutopilotUserInput(text)) {
      return;
    }
    setIsThinking(true);
    (async () => {
      const consumed = await handleChatCommand(text, resolvedMusePresenceId);
      if (consumed) {
        return;
      }

      const baseUrl = runtimeBaseUrl();
      const surroundingNodes = buildMuseSurroundingNodes(resolvedMusePresenceId, workspace);
      const response = await fetch(`${baseUrl}/api/muse/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          muse_id: resolvedMusePresenceId || "witness_thread",
          multi_entity: true,
          presence_ids: [resolvedMusePresenceId || "witness_thread"],
          text,
          mode: "stochastic",
          token_budget: 2048,
          graph_revision: simulation?.timestamp || catalog?.generated_at || "",
          surrounding_nodes: surroundingNodes,
        }),
      });
      if (!response.ok) {
        throw new Error(`muse request failed (${response.status})`);
      }
      const payload = (await response.json()) as {
        reply?: unknown;
        mode?: unknown;
        model?: unknown;
        trace?: unknown;
      };
      emitWitnessChatReply(payload, "chat:/api/muse/message", resolvedMusePresenceId);
    })()
      .catch(() => {
        emitChatMessage({
          role: "assistant",
          text: "muse request failed",
          meta: {
            channel: "command",
            source: "chat:/api/muse/message",
            presenceId: resolvedMusePresenceId,
          },
        });
      })
      .finally(() => {
        setIsThinking(false);
      });
  }, [
    activeMusePresenceId,
    catalog,
    emitChatMessage,
    emitWitnessChatReply,
    handleAutopilotUserInput,
    buildMuseSurroundingNodes,
    handleChatCommand,
    simulation,
  ]);

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
    const elements = Array.isArray(activeProjection?.elements) ? activeProjection.elements : [];
    elements.forEach((element) => {
      map.set(element.id, {
        presence: element.presence,
        binds_to: element.binds_to,
        kind: element.kind,
      });
    });
    return map;
  }, [activeProjection?.elements]);

  const presenceManifestById = useMemo(() => {
    const map = new Map<string, { en: string; ja: string }>();
    (catalog?.entity_manifest ?? []).forEach((entry) => {
      const id = String(entry?.id ?? "").trim();
      if (!id) {
        return;
      }
      const en = String(entry?.en ?? id).trim() || id;
      const ja = String(entry?.ja ?? "").trim();
      map.set(id, { en, ja });
    });
    return map;
  }, [catalog?.entity_manifest]);

  const particleCountsByPresence = useMemo(() => {
    const byPresence: Record<string, number> = {};
    const rows = simulation?.presence_dynamics?.field_particles ?? simulation?.field_particles ?? [];
    for (const row of rows) {
      const presenceId = String(row?.presence_id ?? "").trim();
      if (!presenceId) {
        continue;
      }
      byPresence[presenceId] = (byPresence[presenceId] ?? 0) + 1;
    }
    return byPresence;
  }, [simulation?.field_particles, simulation?.presence_dynamics?.field_particles]);

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
    const states = Array.isArray(activeProjection.states) ? activeProjection.states : [];
    states.forEach((state) => {
      map.set(state.element_id, state);
    });
    return map;
  }, [activeProjection]);
  
  const handleWorldPanelDragEnd = useCallback((panelId: string, info: PanInfo) => {
    const panelScale = panelWorldScaleRef.current.get(panelId);
    const fallbackPixelsPerWorldX = Math.max(140, viewportWidth * 0.34 * deferredCoreCameraZoom);
    const fallbackPixelsPerWorldY = Math.max(110, Math.max(160, viewportHeight - 126) * 0.47 * deferredCoreCameraZoom);
    const pixelsPerWorldX = Math.max(90, panelScale?.x ?? fallbackPixelsPerWorldX);
    const pixelsPerWorldY = Math.max(74, panelScale?.y ?? fallbackPixelsPerWorldY);
    const worldDeltaX = info.offset.x / pixelsPerWorldX;
    const worldDeltaY = info.offset.y / pixelsPerWorldY;

    setPanelWorldBiases((prev) => {
      const current = prev[panelId] ?? { x: 0, y: 0 };
      return {
        ...prev,
        [panelId]: {
          x: clamp(current.x + worldDeltaX, -1.24, 1.24),
          y: clamp(current.y + worldDeltaY, -1.02, 1.02),
        },
      };
    });
  }, [deferredCoreCameraZoom, viewportHeight, viewportWidth]);

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
  const latestAutopilotEvent = autopilotEvents[0] ?? null;
  const museRuntimeSnapshot = catalog?.muse_runtime ?? null;
  const museForgePreviewId = toMuseSlug(museForgeLabel);

  const panelConfigs = useMemo<PanelConfig[]>(() => [
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
                  particleDensity={deferredCoreSimulationTuning.particleDensity}
                  particleScale={deferredCoreSimulationTuning.particleScale}
                  motionSpeed={deferredCoreSimulationTuning.motionSpeed}
                  mouseInfluence={deferredCoreSimulationTuning.mouseInfluence}
                  layerDepth={deferredCoreSimulationTuning.layerDepth}
                  graphNodeSmoothness={deferredCoreSimulationTuning.graphNodeSmoothness}
                  graphNodeStepScale={deferredCoreSimulationTuning.graphNodeStepScale}
                  museWorkspaceBindings={museWorkspaceBindings}
                />
              </section>
            ))}
          </div>
        </div>
      ),
    },
    {
      id: GLASS_VIEWPORT_PANEL_ID,
      fallbackSpan: 12,
      anchorKind: "region",
      anchorId: "view_lens_keeper",
      worldSize: "xl",
      pinnedByDefault: true,
      render: () => (
        <div className="mt-0 rounded-xl border border-[rgba(131,188,227,0.34)] bg-[rgba(8,20,31,0.7)] p-3 h-full">
          <p className="text-[11px] uppercase tracking-[0.12em] text-[#a6d6f5]">
            Glass Viewport Presence
          </p>
          <p className="text-xs text-[#cfe6f7] mt-1">
            This lane is managed through transparent glass mode for camera guidance and gentle map panning.
          </p>
          <p className="text-[11px] text-[#9ec7dd] mt-2">
            Use the glass controls to let the view-lens keeper guide what you see in the simulation.
          </p>
        </div>
      ),
    },
    ...FIXED_MUSE_PRESENCES.map((muse) => {
      const panelId = muse.id;
      const musePresenceId = muse.presenceId;
      const panelState = projectionStateByElement.get(panelId) ?? null;
      const panelSession =
        activeProjection?.chat_sessions?.find(
          (session) => normalizeMusePresenceId(String(session.presence ?? "")) === normalizeMusePresenceId(musePresenceId),
        )
        ?? null;
      const boundCount = museWorkspaceBindings[normalizeMusePresenceId(musePresenceId)]?.length ?? 0;

      return {
        id: panelId,
        fallbackSpan: 4,
        anchorKind: "node" as const,
        anchorId: musePresenceId,
        worldSize: "m" as const,
        render: () => (
          <div
            style={{
              opacity: panelState ? projectionOpacity(panelState.opacity, 0.92) : 1,
              transform: panelState
                ? `scale(${(1 + panelState.pulse * 0.01).toFixed(3)})`
                : undefined,
              transformOrigin: "center top",
              transition: "transform 200ms ease, opacity 200ms ease",
            }}
          >
            <MusePresencePanel
              museId={musePresenceId}
              onSend={handleMuseWorkspaceSend}
              onRecord={handleRecord}
              onTranscribe={handleTranscribe}
              onSendVoice={handleSendVoice}
              isRecording={isRecording}
              isThinking={isThinking}
              voiceInputMeta={voiceInputMeta}
              catalog={catalog}
              simulation={simulation}
              workspaceContext={
                museWorkspaceContexts[normalizeMusePresenceId(musePresenceId)]
                ?? null
              }
              onWorkspaceContextChange={handleMuseWorkspaceContextChange}
              onWorkspaceBindingsChange={handleMuseWorkspaceBindingsChange}
              chatLensState={panelState}
              activeChatSession={panelSession}
              activeMusePresenceId={activeMusePresenceId}
              onMusePresenceChange={setActiveMusePresenceId}
            />
            <p className="mt-2 text-[10px] text-[#8db3ca]">
              workspace binds <code>{boundCount}</code>
            </p>
          </div>
        ),
      } satisfies PanelConfig;
    }),
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
      id: "nexus.ui.world_log",
      fallbackSpan: 6,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#a6e22e] opacity-70" />
          <h2 className="text-2xl font-bold mb-2">World Log / 世界記録</h2>
          <p className="text-muted mb-4">
            Live timeline for receipts, eta-mu ingest, pending inbox files, presence account updates, and commentary events.
          </p>
          {deferredPanelsReady ? (
            <Suspense fallback={<DeferredPanelPlaceholder title="World Log" />}>
              <WorldLogPanel catalog={catalog} />
            </Suspense>
          ) : (
            <DeferredPanelPlaceholder title="World Log" />
          )}
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
      id: "nexus.ui.runtime_config",
      fallbackSpan: 6,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#ae81ff] opacity-70" />
          <h2 className="text-2xl font-bold mb-2">Runtime Config / 実行設定</h2>
          <p className="text-muted mb-4">
            Inspect live numeric constants exposed by <code>/api/config</code> for simulation and runtime tuning.
          </p>
          {deferredPanelsReady ? (
            <Suspense fallback={<DeferredPanelPlaceholder title="Runtime Config" />}>
              <RuntimeConfigPanel />
            </Suspense>
          ) : (
            <DeferredPanelPlaceholder title="Runtime Config" />
          )}
        </>
      ),
    },
    {
      id: "nexus.ui.daimoi_presence",
      fallbackSpan: 6,
      className: "card relative overflow-hidden",
      render: () => (
        <>
          <div className="absolute top-0 left-0 w-1 h-full bg-[#89c6eb] opacity-70" />
          <h2 className="text-2xl font-bold mb-2">Daimoi Presence Deck / 代網存在甲板</h2>
          <p className="text-muted mb-4">
            Probabilistic daimoi and presence distributions with direct camera focus controls.
          </p>
          {deferredPanelsReady ? (
            <Suspense fallback={<DeferredPanelPlaceholder title="Daimoi Presence" />}>
              <DaimoiPresencePanel
                catalog={catalog}
                simulation={simulation}
                onFocusAnchor={flyCameraToAnchor}
                onEmitUserInput={handleUserPresenceInput}
              />
            </Suspense>
          ) : (
            <DeferredPanelPlaceholder title="Daimoi Presence" />
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
    activeMusePresenceId,
    activeProjection,
    autopilotEvents,
    catalog,
    deferredCoreSimulationTuning,
    dedicatedOverlayViews,
    deferredPanelsReady,
    flyCameraToAnchor,
    handleUserPresenceInput,
    handleMuseWorkspaceBindingsChange,
    handleMuseWorkspaceContextChange,
    handleMuseWorkspaceSend,
    handleRecord,
    handleSendVoice,
    handleTranscribe,
    handleWorldInteract,
    interactingPersonId,
    isRecording,
    isThinking,
    museWorkspaceBindings,
    museWorkspaceContexts,
    projectionStateByElement,
    simulation,
    voiceInputMeta,
    worldInteraction,
  ]);

  const sortedPanels = useMemo<RankedPanel[]>(() => {
    const panelDrafts = panelConfigs
      .filter((config) => config.id !== "nexus.ui.simulation_map")
      .map((config) => {
        const state = projectionStateByElement.get(config.id);
        const element = projectionElementById.get(config.id);
        const preset = PANEL_ANCHOR_PRESETS[config.id];
        const priority = state?.priority ?? 0.1;
        return {
          config,
          state,
          element,
          preset,
          priority,
        };
      });

    const topOperationalPriority = panelDrafts.reduce((max, draft) => {
      if (isGlassPrimaryPanelId(draft.config.id)) {
        return max;
      }
      return Math.max(max, draft.priority);
    }, 0);

    const lowPriorityCycle = topOperationalPriority < 0.62;

    return panelDrafts
      .map((draft) => {
        const { config, state, element, preset, priority } = draft;
        const councilBoost = panelCouncilBoosts[config.id] ?? 0;

        const rawPresenceId = String(
          element?.presence
          ?? config.anchorId
          ?? preset?.anchorId
          ?? "particle_field",
        ).trim();
        const presenceId = rawPresenceId || "particle_field";
        const presenceMeta = presenceManifestById.get(presenceId);
        const presenceLabel = presenceMeta?.en ?? presenceId.replace(/[_-]+/g, " ");
        const presenceLabelJa = presenceMeta?.ja ?? "";
        const presenceRole = PRESENCE_OPERATIONAL_ROLE_BY_ID[presenceId] ?? "neutral";
        const particleDisposition: ParticleDisposition =
          presenceRole === "neutral" ? "neutral" : "role-bound";
        const particleCount = particleCountsByPresence[presenceId] ?? 0;
        const glassPanel = isGlassPrimaryPanelId(config.id);
        const glassPreferenceBoost = glassPanel
          ? (presenceRole === "camera-guidance" ? 0.14 : 0.08)
            + (lowPriorityCycle ? 0.19 : 0)
            + (particleCount <= 2 ? 0.03 : 0)
          : 0;
        const councilScore = clamp(priority + glassPreferenceBoost + (councilBoost * 0.11), 0, 2);
        const depth = Math.round(clamp(councilScore, 0, 1) * 160) + 24;
        const toolHints = PANEL_TOOL_HINTS[config.id] ?? ["inspect", "focus", "act"];
        const baseCouncilReason = String(
          state?.explain?.reason_en ?? "Council rank follows live field and presence signal.",
        );
        const councilReason = glassPanel && glassPreferenceBoost > 0
          ? `${baseCouncilReason} Glass lane preferred${lowPriorityCycle ? " during low-priority cycle." : "."}`
          : baseCouncilReason;

        return {
          ...config,
          anchorKind: config.anchorKind ?? preset?.kind ?? "node",
          anchorId: config.anchorId ?? preset?.anchorId,
          worldSize: config.worldSize ?? preset?.worldSize ?? "m",
          pinnedByDefault: config.pinnedByDefault ?? preset?.pinnedByDefault ?? false,
          priority,
          depth,
          councilScore,
          councilBoost,
          councilReason,
          presenceId,
          presenceLabel,
          presenceLabelJa,
          presenceRole,
          particleDisposition,
          particleCount,
          toolHints,
        };
      })
      .sort((left, right) => {
        if (right.councilScore !== left.councilScore) {
          return right.councilScore - left.councilScore;
        }
        if (isGlassPrimaryPanelId(right.id) !== isGlassPrimaryPanelId(left.id)) {
          return Number(isGlassPrimaryPanelId(right.id)) - Number(isGlassPrimaryPanelId(left.id));
        }
        return right.priority - left.priority;
      });
  }, [
    panelConfigs,
    panelCouncilBoosts,
    particleCountsByPresence,
    presenceManifestById,
    projectionElementById,
    projectionStateByElement,
  ]);

  const panelWindowStateById = useMemo<Record<string, PanelWindowState>>(() => {
    const stateById: Record<string, PanelWindowState> = {};
    sortedPanels.forEach((panel, index) => {
      const existing = panelWindowStates[panel.id];
      if (existing) {
        stateById[panel.id] = existing;
        return;
      }
      stateById[panel.id] = {
        open: Boolean(panel.pinnedByDefault || index < 3),
        minimized: false,
      };
    });
    return stateById;
  }, [panelWindowStates, sortedPanels]);

  const activatePanelWindow = useCallback((panelId: string) => {
    const current = panelWindowStateById[panelId] ?? { open: true, minimized: false };
    if (!current.open || current.minimized) {
      setPanelWindowStates((prev) => ({
        ...prev,
        [panelId]: {
          open: true,
          minimized: false,
        },
      }));
    }
    setSelectedPanelId(panelId);
  }, [panelWindowStateById]);

  const minimizePanelWindow = useCallback((panelId: string) => {
    if (panelId === GLASS_VIEWPORT_PANEL_ID) {
      setPanelWindowStates((prev) => ({
        ...prev,
        [panelId]: {
          open: true,
          minimized: false,
        },
      }));
      return;
    }
    setPanelWindowStates((prev) => ({
      ...prev,
      [panelId]: {
        open: true,
        minimized: true,
      },
    }));
    setSelectedPanelId((prev) => (prev === panelId ? null : prev));
    setHoveredPanelId((prev) => (prev === panelId ? null : prev));
  }, []);

  const closePanelWindow = useCallback((panelId: string) => {
    if (panelId === GLASS_VIEWPORT_PANEL_ID) {
      setPanelWindowStates((prev) => ({
        ...prev,
        [panelId]: {
          open: true,
          minimized: false,
        },
      }));
      setPinnedPanels((prev) => ({
        ...prev,
        [panelId]: true,
      }));
      return;
    }
    setPanelWindowStates((prev) => ({
      ...prev,
      [panelId]: {
        open: false,
        minimized: false,
      },
    }));
    setSelectedPanelId((prev) => (prev === panelId ? null : prev));
    setHoveredPanelId((prev) => (prev === panelId ? null : prev));
  }, []);

  const openRuntimeConfigPanel = useCallback(() => {
    setPanelWindowStates((prev) => ({
      ...prev,
      [RUNTIME_CONFIG_PANEL_ID]: {
        open: true,
        minimized: false,
      },
    }));
    setPinnedPanels((prev) => ({
      ...prev,
      [RUNTIME_CONFIG_PANEL_ID]: true,
    }));
    setSelectedPanelId(RUNTIME_CONFIG_PANEL_ID);
    setHoveredPanelId(null);
  }, []);

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

  const panelStateSpaceBiases = useMemo(() => {
    const byPanel: Record<string, { x: number; y: number }> = {};
    const dynamics = simulation?.presence_dynamics;
    const impactRows = Array.isArray(dynamics?.presence_impacts) ? dynamics.presence_impacts : [];
    if (impactRows.length === 0) {
      return byPanel;
    }

    const anchorLookup = new Map<string, WorldAnchorTarget>();
    const indexAnchor = (id: string, anchor: WorldAnchorTarget) => {
      if (!id || anchorLookup.has(id)) {
        return;
      }
      anchorLookup.set(id, anchor);
    };
    presenceAnchors.forEach((anchor, id) => {
      indexAnchor(id, anchor);
    });
    namedRegionAnchors.forEach((anchor, id) => {
      indexAnchor(id, anchor);
    });
    fieldRegionAnchors.forEach((anchor, id) => {
      indexAnchor(id, anchor);
    });
    clusterAnchors.forEach((anchor, id) => {
      indexAnchor(id, anchor);
    });

    const impactById = new Map<string, number>();
    let centroidX = 0;
    let centroidY = 0;
    let centroidWeight = 0;

    impactRows.forEach((row) => {
      const impactId = String(row.id ?? "").trim();
      if (!impactId) {
        return;
      }
      const affectedBy = row.affected_by ?? {};
      const affects = row.affects ?? {};
      const intensity = clamp(
        (clamp(Number(affectedBy.clicks ?? 0), 0, 1) * 0.44)
        + (clamp(Number(affectedBy.files ?? 0), 0, 1) * 0.31)
        + (clamp(Number(affects.world ?? 0), 0, 1) * 0.25),
        0,
        1,
      );
      if (intensity <= 0.02) {
        return;
      }
      impactById.set(impactId, intensity);
      const anchor = anchorLookup.get(impactId);
      if (!anchor) {
        return;
      }
      centroidX += anchor.x * intensity;
      centroidY += anchor.y * intensity;
      centroidWeight += intensity;
    });

    if (impactById.size === 0) {
      return byPanel;
    }

    const centroid = centroidWeight > 0
      ? {
          x: centroidX / centroidWeight,
          y: centroidY / centroidWeight,
        }
      : { x: 0.5, y: 0.5 };

    const flowRate = clamp(Number(dynamics?.river_flow?.rate ?? 0), 0, 1);
    const turbulence = clamp(Number(dynamics?.river_flow?.turbulence ?? 0), 0, 1);
    const clickPressure = clamp(Number(dynamics?.click_events ?? 0) / 16, 0, 1);
    const filePressure = clamp(Number(dynamics?.file_events ?? 0) / 18, 0, 1);
    const timestampMillis = Date.parse(String(simulation?.timestamp ?? ""));
    const timeSeed = Number.isFinite(timestampMillis)
      ? timestampMillis / 1000
      : Number(simulation?.world?.tick ?? 0);

    sortedPanels.forEach((panel) => {
      const anchor = panelAnchorById.get(panel.id);
      if (!anchor) {
        return;
      }

      let coupling = 0;
      Object.entries(anchor.presenceSignature ?? {}).forEach(([signatureId, rawWeight]) => {
        const weight = clamp(Number(rawWeight), 0, 1);
        if (weight <= 0) {
          return;
        }
        const normalizedId = signatureId.replace(/^field:/, "");
        const impact = impactById.get(signatureId) ?? impactById.get(normalizedId) ?? 0;
        coupling += weight * impact;
      });
      coupling += (impactById.get(anchor.id) ?? 0) * 0.35;

      const projectionState = projectionStateByElement.get(panel.id);
      const projectionPresenceSignal = clamp(Number(projectionState?.explain?.presence_signal ?? 0), 0, 1);
      const pulseSignal = clamp(Number(projectionState?.pulse ?? 0), 0, 1);
      const magnitude = clamp(
        (coupling * (0.12 + (projectionPresenceSignal * 0.18)))
        + (pulseSignal * 0.03)
        + ((clickPressure + filePressure) * 0.02),
        0,
        0.28,
      );
      if (magnitude <= 0.0006) {
        return;
      }

      const driftX = centroid.x - anchor.x;
      const driftY = centroid.y - anchor.y;
      const phase = timeSeed * (0.46 + (flowRate * 0.34)) + (stableUnitHash(panel.id) * Math.PI * 2);
      const swirl = 0.008 + (turbulence * 0.022);
      byPanel[panel.id] = {
        x: clamp((driftX * magnitude * 0.92) + (Math.cos(phase) * swirl), -0.34, 0.34),
        y: clamp((driftY * magnitude * 0.92) + (Math.sin(phase * 1.1) * swirl), -0.28, 0.28),
      };
    });

    return byPanel;
  }, [
    clusterAnchors,
    fieldRegionAnchors,
    namedRegionAnchors,
    panelAnchorById,
    presenceAnchors,
    projectionStateByElement,
    simulation?.presence_dynamics,
    simulation?.timestamp,
    simulation?.world?.tick,
    sortedPanels,
  ]);

  const openPanelIds = useMemo(() => {
    return sortedPanels
      .filter((panel) => {
        const windowState = panelWindowStateById[panel.id] ?? { open: true, minimized: false };
        return windowState.open && !windowState.minimized;
      })
      .map((panel) => panel.id);
  }, [panelWindowStateById, sortedPanels]);

  const visiblePanelIds = useMemo(() => {
    const ordered = [...openPanelIds];
    const bringToFront = (panelId: string | null | undefined) => {
      if (!panelId) {
        return;
      }
      const index = ordered.indexOf(panelId);
      if (index <= 0) {
        return;
      }
      ordered.splice(index, 1);
      ordered.unshift(panelId);
    };

    bringToFront(selectedPanelId);
    bringToFront(hoveredPanelId);
    sortedPanels.forEach((panel) => {
      if (pinnedPanels[panel.id]) {
        bringToFront(panel.id);
      }
    });

    return ordered;
  }, [hoveredPanelId, openPanelIds, pinnedPanels, selectedPanelId, sortedPanels]);

  const worldPanelLayout = useMemo<WorldPanelLayoutEntry[]>(() => {
    const panelsById = new Map(sortedPanels.map((panel) => [panel.id, panel]));
    const velocity = coreFlightVelocityRef.current;
    const speedNorm = clamp(Math.hypot(velocity.x, velocity.y, velocity.z) / 26, 0, 1);
    const stageTop = viewportHeight < 860 ? 104 : 118;
    const stageBottom = Math.max(stageTop + 132, viewportHeight - 14);
    const stageHeight = Math.max(120, stageBottom - stageTop);
    const centerX = viewportWidth / 2;
    const centerY = stageTop + (stageHeight / 2);
    const yaw = (deferredCoreCameraYaw * Math.PI / 180) * 0.72;
    const pitch = (deferredCoreCameraPitch * Math.PI / 180) * 0.68;
    const cosYaw = Math.cos(yaw);
    const sinYaw = Math.sin(yaw);
    const cosPitch = Math.cos(pitch);
    const sinPitch = Math.sin(pitch);
    const cameraOffsetX = deferredCoreRenderedCameraPosition.x / 660;
    const cameraOffsetY = deferredCoreRenderedCameraPosition.y / 560;
    const cameraOffsetZ = deferredCoreRenderedCameraPosition.z / 920;

    const anchorToWorldPoint = (anchor: WorldAnchorTarget) => ({
      x: (anchor.x - 0.5) * 2.25,
      y: (anchor.y - 0.5) * 1.86,
      z: anchor.kind === "node" ? 0.62 : anchor.kind === "cluster" ? 0.24 : -0.14,
    });

    const projectWorldPoint = (worldX: number, worldY: number, worldZ: number) => {
      const wx = worldX - cameraOffsetX;
      const wy = worldY - cameraOffsetY;
      const wz = worldZ - cameraOffsetZ;

      const x1 = (wx * cosYaw) - (wz * sinYaw);
      const z1 = (wx * sinYaw) + (wz * cosYaw);
      const y1 = (wy * cosPitch) - (z1 * sinPitch);
      const z2 = (wy * sinPitch) + (z1 * cosPitch);
      const perspective = clamp(1 / (1 + (z2 * 0.7)), 0.46, 1.9);

      return {
        x: centerX + (x1 * viewportWidth * 0.34 * perspective * deferredCoreCameraZoom),
        y: centerY + (y1 * stageHeight * 0.47 * perspective * deferredCoreCameraZoom),
        perspective,
      };
    };

    const entries: WorldPanelLayoutEntry[] = [];
    const trackedScaleIds = new Set<string>();
    visiblePanelIds.forEach((panelId) => {
      const panel = panelsById.get(panelId);
      if (!panel) {
        return;
      }
      const anchor = panelAnchorById.get(panelId);
      if (!anchor) {
        return;
      }
      const overlayAnchor = resolveOverlayAnchorRatio(anchor, panel.anchorId);
      const anchorWorld = overlayAnchor
        ? {
            x: (overlayAnchor.x - 0.5) * 2.25,
            y: (overlayAnchor.y - 0.5) * 1.86,
            z: anchor.kind === "node" ? 0.62 : anchor.kind === "cluster" ? 0.24 : -0.14,
          }
        : anchorToWorldPoint(anchor);
      const projected = overlayAnchor
        ? {
            x: overlayAnchor.x * viewportWidth,
            y: stageTop + (overlayAnchor.y * stageHeight),
            perspective: clamp(0.96 + ((deferredCoreCameraZoom - 1) * 0.18), 0.72, 1.34),
          }
        : projectWorldPoint(anchorWorld.x, anchorWorld.y, anchorWorld.z);
      const side = preferredSideForAnchor(
        panelId,
        projected.x,
        projected.y,
        viewportWidth,
        viewportHeight,
        panelSideRef.current,
      );
      const baseSize = panelSizeForWorld(panel.worldSize ?? "m", panel.priority, deferredCoreCameraZoom, speedNorm);
      const size = {
        width: Math.round(Math.min(baseSize.width, Math.max(176, viewportWidth - (WORLD_PANEL_MARGIN * 2)))),
        height: Math.round(Math.min(baseSize.height, Math.max(120, stageBottom - stageTop - 8))),
        collapse: baseSize.collapse,
      };
      const pixelsPerWorldX = Math.max(90, viewportWidth * 0.34 * projected.perspective * deferredCoreCameraZoom);
      const pixelsPerWorldY = Math.max(74, stageHeight * 0.47 * projected.perspective * deferredCoreCameraZoom);
      panelWorldScaleRef.current.set(panelId, { x: pixelsPerWorldX, y: pixelsPerWorldY });
      trackedScaleIds.add(panelId);

      const halfWorldWidth = (size.width / pixelsPerWorldX) * 0.5;
      const halfWorldHeight = (size.height / pixelsPerWorldY) * 0.5;
      const gapWorldX = Math.max(0.04, 22 / pixelsPerWorldX);
      const gapWorldY = Math.max(0.04, 18 / pixelsPerWorldY);

      let panelWorldX = anchorWorld.x;
      let panelWorldY = anchorWorld.y;
      if (side === "left") {
        panelWorldX -= halfWorldWidth + gapWorldX;
        panelWorldY -= gapWorldY * 0.34;
      } else if (side === "right") {
        panelWorldX += halfWorldWidth + gapWorldX;
        panelWorldY -= gapWorldY * 0.34;
      } else if (side === "top") {
        panelWorldY -= halfWorldHeight + gapWorldY;
      } else {
        panelWorldY += halfWorldHeight + gapWorldY;
      }

      const manualBias = panelWorldBiases[panelId] ?? { x: 0, y: 0 };
      const stateSpaceBias = panelStateSpaceBiases[panelId] ?? { x: 0, y: 0 };
      panelWorldX += manualBias.x + stateSpaceBias.x;
      panelWorldY += manualBias.y + stateSpaceBias.y;

      const panelScreen = projectWorldPoint(panelWorldX, panelWorldY, anchorWorld.z);
      const x = panelScreen.x - (size.width / 2);
      const y = panelScreen.y - (size.height / 2);

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
        panelWorldX,
        panelWorldY,
        panelWorldZ: anchorWorld.z,
        pixelsPerWorldX,
        pixelsPerWorldY,
        tetherX: projected.x,
        tetherY: projected.y,
        glow,
        collapse: size.collapse,
      });
    });

    panelWorldScaleRef.current.forEach((_scale, panelId) => {
      if (!trackedScaleIds.has(panelId)) {
        panelWorldScaleRef.current.delete(panelId);
      }
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
    deferredCoreCameraPitch,
    deferredCoreRenderedCameraPosition,
    deferredCoreCameraYaw,
    deferredCoreCameraZoom,
    hoveredPanelId,
    panelAnchorById,
    panelStateSpaceBiases,
    panelWorldBiases,
    pinnedPanels,
    resolveOverlayAnchorRatio,
    selectedPanelId,
    sortedPanels,
    viewportHeight,
    viewportWidth,
    visiblePanelIds,
  ]);

  const panelNexusLayout = useMemo<WorldPanelNexusEntry[]>(() => {
    const visibleEntryById = new Map(worldPanelLayout.map((entry) => [entry.id, entry]));
    const stageTop = viewportHeight < 860 ? 104 : 118;
    const stageBottom = Math.max(stageTop + 132, viewportHeight - 14);
    const stageHeight = Math.max(120, stageBottom - stageTop);

    return sortedPanels.flatMap((panel) => {
      const anchor = panelAnchorById.get(panel.id);
      if (!anchor) {
        return [];
      }
      const windowState = panelWindowStateById[panel.id] ?? { open: true, minimized: false };
      const visibleEntry = visibleEntryById.get(panel.id);
      const overlayAnchor = resolveOverlayAnchorRatio(anchor, panel.anchorId);
      const x = visibleEntry?.anchorScreenX
        ?? (overlayAnchor ? overlayAnchor.x * viewportWidth : anchor.x * viewportWidth);
      const y = visibleEntry?.anchorScreenY
        ?? (overlayAnchor ? stageTop + (overlayAnchor.y * stageHeight) : stageTop + (anchor.y * stageHeight));

      return [
        {
          panelId: panel.id,
          panelLabel: panel.id.split(".").slice(-1)[0].replace(/_/g, " "),
          anchor,
          x,
          y,
          hue: anchor.hue,
          confidence: anchor.confidence,
          open: windowState.open,
          minimized: windowState.minimized,
          selected: selectedPanelId === panel.id,
        },
      ];
    });
  }, [
    panelAnchorById,
    panelWindowStateById,
    resolveOverlayAnchorRatio,
    selectedPanelId,
    sortedPanels,
    viewportHeight,
    viewportWidth,
    worldPanelLayout,
  ]);

  const glassCenterRatio = useMemo(() => {
    const stageTop = viewportHeight < 860 ? 104 : 118;
    const stageBottom = Math.max(stageTop + 132, viewportHeight - 14);
    const stageHeight = Math.max(120, stageBottom - stageTop);
    const glassEntry = worldPanelLayout.find((entry) => entry.id === GLASS_VIEWPORT_PANEL_ID);
    if (!glassEntry) {
      return { x: 0.5, y: 0.5 };
    }
    return {
      x: clamp((glassEntry.x + (glassEntry.width / 2)) / Math.max(1, viewportWidth), 0.08, 0.92),
      y: clamp(((glassEntry.y + (glassEntry.height / 2)) - stageTop) / Math.max(1, stageHeight), 0.08, 0.92),
    };
  }, [viewportHeight, viewportWidth, worldPanelLayout]);

  const lastGlassClickRef = useRef<{ ts: number; x: number; y: number } | null>(null);

  const handleGlassInteractAt = useCallback((payload: {
    panelId: string;
    xRatio: number;
    yRatio: number;
    clientX?: number;
    clientY?: number;
  }) => {
    const anchorX = clamp(Number(payload.xRatio ?? 0.5), 0, 1);
    const anchorY = clamp(Number(payload.yRatio ?? 0.5), 0, 1);
    const hasClientPoint = Number.isFinite(payload.clientX) && Number.isFinite(payload.clientY);
    const result = hasClientPoint && overlayApi?.interactClientAt
      ? overlayApi.interactClientAt(Number(payload.clientX), Number(payload.clientY), { openWorldscreen: true })
      : overlayApi?.interactAt?.(anchorX, anchorY, { openWorldscreen: true });
    const resolvedX = clamp(Number(result?.xRatio ?? anchorX), 0, 1);
    const resolvedY = clamp(Number(result?.yRatio ?? anchorY), 0, 1);
    if (result?.hitNode) {
      return;
    }

    const now = performance.now();
    const last = lastGlassClickRef.current;
    if (last && (now - last.ts) < 320 && Math.abs(last.x - resolvedX) < 0.05 && Math.abs(last.y - resolvedY) < 0.05) {
      // Double click on glass -> fly camera
      flyCameraToRatios(resolvedX, resolvedY, "node", glassCenterRatio.x, glassCenterRatio.y);
      lastGlassClickRef.current = null;
      return;
    }
    lastGlassClickRef.current = { ts: now, x: resolvedX, y: resolvedY };

    overlayApi?.pulseAt?.(resolvedX, resolvedY, 0.88, result?.target ?? payload.panelId ?? "glass_click");
  }, [flyCameraToRatios, glassCenterRatio.x, glassCenterRatio.y, overlayApi]);

  const handleNexusInteraction = useCallback((event: NexusInteractionEvent) => {
    const anchorX = clamp(Number(event.xRatio ?? 0.5), 0, 1);
    const anchorY = clamp(Number(event.yRatio ?? 0.5), 0, 1);
    if (event.isDoubleTap) {
      flyCameraToRatios(anchorX, anchorY, "node", glassCenterRatio.x, glassCenterRatio.y);
    }
    overlayApi?.pulseAt?.(
      anchorX,
      anchorY,
      event.openWorldscreen ? 1.06 : 0.78,
      event.nodeId || event.label || "nexus",
    );
  }, [flyCameraToRatios, glassCenterRatio.x, glassCenterRatio.y, overlayApi]);

  const galaxyLayerStyles = useMemo(() => {
    const driftX = deferredCoreRenderedCameraPosition.x;
    const driftY = deferredCoreRenderedCameraPosition.y;
    const driftZ = deferredCoreRenderedCameraPosition.z;
    return {
      far: {
        transform: `translate3d(${((-driftX * 0.07) + (deferredCoreCameraYaw * 1.4)).toFixed(1)}px, ${((-driftY * 0.05) + (deferredCoreCameraPitch * 1.3)).toFixed(1)}px, ${(driftZ * 0.04).toFixed(1)}px) scale(${(1 + (driftZ * 0.00018)).toFixed(3)})`,
      },
      mid: {
        transform: `translate3d(${((-driftX * 0.14) + (deferredCoreCameraYaw * 2.2)).toFixed(1)}px, ${((-driftY * 0.11) + (deferredCoreCameraPitch * 1.8)).toFixed(1)}px, ${(driftZ * 0.08).toFixed(1)}px) scale(${(1.04 + (driftZ * 0.00022)).toFixed(3)})`,
      },
      near: {
        transform: `translate3d(${((-driftX * 0.22) + (deferredCoreCameraYaw * 3.1)).toFixed(1)}px, ${((-driftY * 0.18) + (deferredCoreCameraPitch * 2.4)).toFixed(1)}px, ${(driftZ * 0.14).toFixed(1)}px) scale(${(1.1 + (driftZ * 0.00032)).toFixed(3)})`,
      },
    };
  }, [deferredCoreCameraPitch, deferredCoreCameraYaw, deferredCoreRenderedCameraPosition]);

  return (
    <>
      <CoreBackdrop
        simulation={simulation}
        catalog={catalog}
        viewportHeight={viewportHeight}
        coreCameraTransform={coreCameraTransform}
        coreSimulationFilter={coreSimulationFilter}
        coreOverlayView={coreOverlayView}
        coreSimulationTuning={coreSimulationTuning}
        coreVisualTuning={coreVisualTuning}
        coreLayerVisibility={coreLayerVisibility}
        museWorkspaceBindings={museWorkspaceBindings}
        galaxyLayerStyles={galaxyLayerStyles}
        mouseDaimonTuning={mouseDaimonTuning}
        onUserPresenceInput={handleUserPresenceInput}
        onOverlayInit={handleOverlayInit}
        onNexusInteraction={handleNexusInteraction}
        glassCenterRatio={glassCenterRatio}
        onPointerDown={handleCorePointerDown}
        onPointerMove={handleCorePointerMove}
        onPointerUp={handleCorePointerUp}
        onWheel={handleCoreWheel}
      />

      <main
        className="relative z-20 w-full px-1 py-2 md:px-2 md:py-4 pb-20 lg:pr-[24rem] transition-colors pointer-events-none"
        style={{ opacity: interfaceOpacity }}
      >
        <header className="mb-4 border-b border-[rgba(166,205,235,0.28)] pb-3 flex flex-col gap-2 bg-[rgba(8,14,22,0.18)] rounded-xl px-3 shadow-[0_6px_16px_rgba(2,8,14,0.16)] pointer-events-auto">
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
        </header>

        <aside
          data-core-wheel="block"
          className="fixed inset-x-2 bottom-20 z-[74] max-h-[46vh] overflow-y-auto rounded-xl border border-[rgba(137,198,235,0.36)] bg-[linear-gradient(180deg,rgba(7,17,27,0.92),rgba(6,15,24,0.96))] p-3 shadow-[0_12px_30px_rgba(2,8,14,0.34)] pointer-events-auto lg:inset-x-auto lg:bottom-4 lg:right-2 lg:top-24 lg:w-[23rem] lg:max-h-[calc(100vh-8rem)]"
        >
          <div className="mb-2 flex items-center justify-between gap-2">
            <p className="text-[11px] uppercase tracking-[0.12em] text-[#a3d3ef]">Simulation Controls</p>
            <p className="text-[10px] text-[#beddf0]">ui opacity <code>{Math.round(interfaceOpacity * 100)}%</code></p>
          </div>

          <CoreControlPanel
            projectionPerspective={projectionPerspective}
            autopilotEnabled={autopilotEnabled}
            autopilotStatus={autopilotStatus}
            autopilotSummary={autopilotSummary}
            interfaceOpacity={interfaceOpacity}
            coreCameraZoom={coreCameraZoom}
            coreCameraPitch={coreCameraPitch}
            coreCameraYaw={coreCameraYaw}
            coreRenderedCameraPosition={coreRenderedCameraPosition}
            coreFlightEnabled={coreFlightEnabled}
            coreFlightSpeed={coreFlightSpeed}
            coreOrbitEnabled={coreOrbitEnabled}
            coreOrbitSpeed={coreOrbitSpeed}
            coreSimulationTuning={coreSimulationTuning}
            coreVisualTuning={coreVisualTuning}
            coreOverlayView={coreOverlayView}
            activeChatLens={activeChatLens}
            latestAutopilotEvent={latestAutopilotEvent}
            projectionOptions={projectionOptions}
            mouseDaimonTuning={mouseDaimonTuning}
            onToggleAutopilot={toggleAutopilot}
            onToggleCoreFlight={toggleCoreFlight}
            onToggleCoreOrbit={toggleCoreOrbit}
            onNudgeCoreFlightSpeed={nudgeCoreFlightSpeed}
            onNudgeCoreOrbitSpeed={nudgeCoreOrbitSpeed}
            onApplyCoreLayerPreset={applyCoreLayerPreset}
            onNudgeCoreZoom={nudgeCoreZoom}
            onResetCoreCamera={resetCoreCamera}
            onSelectPerspective={setUiPerspective}
            onSetInterfaceOpacity={setInterfaceOpacityDial}
            onResetInterfaceOpacity={resetInterfaceOpacity}
            onBoostCoreVisibility={boostCoreVisibility}
            onResetCoreVisualTuning={resetCoreVisualTuning}
            onSetCoreVisualDial={setCoreVisualDial}
            onResetCoreSimulationTuning={resetCoreSimulationTuning}
            onSetCoreSimulationDial={setCoreSimulationDial}
            onSetCoreOrbitSpeed={(value) => setCoreOrbitSpeed(clamp(value, CORE_ORBIT_SPEED_MIN, CORE_ORBIT_SPEED_MAX))}
            onSetMouseDaimonTuning={updateMouseDaimonTuning}
            onOpenRuntimeConfig={openRuntimeConfigPanel}
          />

          <div className="mt-3">
            <CoreLayerManagerOverlay
              inline
              activeLayerCount={activeCoreLayerCount}
              isOpen={coreLayerManagerOpen}
              layerVisibility={coreLayerVisibility}
              onToggleOpen={() => setCoreLayerManagerOpen((prev) => !prev)}
              onSetAllLayers={setAllCoreLayers}
              onSetLayerEnabled={setCoreLayerEnabled}
            />
          </div>

          <div className="mt-3 rounded-lg border border-[rgba(106,203,242,0.3)] bg-[rgba(8,19,29,0.38)] px-3 py-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-[11px] uppercase tracking-[0.12em] text-[#9ec7dd]">Muse Forge</p>
              <p className="text-[10px] text-[#b8d9ef]">
                runtime: <code>{museRuntimeSnapshot?.muse_count ?? 0}</code> muses | seq <code>{museRuntimeSnapshot?.event_seq ?? 0}</code>
              </p>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <input
                type="text"
                value={museForgeLabel}
                onChange={(event) => setMuseForgeLabel(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void handleCreateMuse();
                  }
                }}
                placeholder="create muse label (e.g. Archive Witness)"
                className="min-w-[220px] flex-1 rounded-md border border-[rgba(106,203,242,0.34)] bg-[rgba(10,23,34,0.86)] px-3 py-1.5 text-xs text-[#e6f5ff]"
              />
              <button
                type="button"
                disabled={museForgeBusy || !museForgePreviewId}
                onClick={() => {
                  void handleCreateMuse();
                }}
                className="rounded-md border border-[rgba(166,226,46,0.45)] bg-[rgba(166,226,46,0.16)] px-3 py-1.5 text-xs font-semibold text-[#e9ffd3] disabled:opacity-45"
              >
                {museForgeBusy ? "creating..." : "Create Muse"}
              </button>
            </div>
            <p className="mt-1 text-[10px] text-[#9fc4dd]">
              next id: <code>{museForgePreviewId || "(type label)"}</code>
            </p>
          </div>
        </aside>

        <WorldPanelsViewport
          viewportWidth={viewportWidth}
          viewportHeight={viewportHeight}
          worldPanelLayout={worldPanelLayout}
          panelNexusLayout={panelNexusLayout}
          sortedPanels={sortedPanels}
          panelWindowStateById={panelWindowStateById}
          tertiaryPinnedPanelId={tertiaryPinnedPanelId}
          pinnedPanels={pinnedPanels}
          selectedPanelId={selectedPanelId}
          isEditMode={isEditMode}
          coreFlightSpeed={coreFlightSpeed}
          onToggleEditMode={() => setIsEditMode((prev) => !prev)}
          onHoverPanel={setHoveredPanelId}
          onSelectPanel={activatePanelWindow}
          onTogglePanelPin={togglePanelPin}
          onActivatePanel={activatePanelWindow}
          onMinimizePanel={minimizePanelWindow}
          onClosePanel={closePanelWindow}
          onAdjustPanelCouncilRank={adjustPanelCouncilRank}
          onPinPanelToTertiary={pinPanelToTertiary}
          onFlyCameraToAnchor={flyCameraToAnchor}
          onGlassInteractAt={handleGlassInteractAt}
          onNudgeCameraPan={nudgeCameraPan}
          onWorldPanelDragEnd={handleWorldPanelDragEnd}
        />

        {uiToasts.length > 0 ? (
          <div className="fixed bottom-4 right-4 z-[80] pointer-events-none flex w-[min(92vw,360px)] flex-col gap-2">
            {uiToasts.map((toast) => (
              <div
                key={toast.id}
                className="pointer-events-auto rounded-lg border border-[rgba(102,217,239,0.45)] bg-[rgba(12,23,31,0.94)] px-3 py-2 shadow-[0_8px_24px_rgba(0,0,0,0.45)]"
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
