import {
  memo,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
  type ReactNode,
} from "react";
import type { PanInfo } from "framer-motion";
import type {
  PanelConfig,
  PanelWindowState,
  WorldAnchorTarget,
  WorldPanelNexusEntry,
  WorldPanelLayoutEntry,
} from "../../app/worldPanelLayout";
import { runtimeBaseUrl } from "../../runtime/endpoints";

interface SortedPanel extends PanelConfig {
  priority: number;
  depth: number;
  councilScore: number;
  councilBoost: number;
  councilReason: string;
  presenceId: string;
  presenceLabel: string;
  presenceLabelJa: string;
  presenceRole: string;
  particleDisposition: "neutral" | "role-bound";
  particleCount: number;
  toolHints: string[];
}

interface Props {
  viewportWidth: number;
  viewportHeight: number;
  worldPanelLayout: WorldPanelLayoutEntry[];
  panelNexusLayout: WorldPanelNexusEntry[];
  sortedPanels: SortedPanel[];
  panelWindowStateById: Record<string, PanelWindowState>;
  tertiaryPinnedPanelId: string | null;
  pinnedPanels: Record<string, boolean>;
  selectedPanelId: string | null;
  isEditMode: boolean;
  coreFlightSpeed: number;
  onToggleEditMode: () => void;
  onHoverPanel: (id: string | null) => void;
  onSelectPanel: (id: string) => void;
  onTogglePanelPin: (panelId: string) => void;
  onActivatePanel: (panelId: string) => void;
  onMinimizePanel: (panelId: string) => void;
  onClosePanel: (panelId: string) => void;
  onAdjustPanelCouncilRank: (panelId: string, delta: number) => void;
  onPinPanelToTertiary: (panelId: string) => void;
  onFlyCameraToAnchor: (anchor: WorldAnchorTarget) => void;
  onGlassInteractAt: (xRatio: number, yRatio: number) => void;
  onNudgeCameraPan: (xRatioDelta: number, yRatioDelta: number, sourcePanelId?: string) => void;
  onWorldPanelDragEnd: (panelId: string, info: PanInfo) => void;
}

interface PanelRenderTarget {
  render: () => ReactNode;
}

type FocusPaneMode = "panel" | "glass";
type OperationalState = "running" | "paused" | "blocked";
const GLASS_VIEWPORT_PANEL_ID = "nexus.ui.glass_viewport";
const PANEL_LABEL_CACHE = new Map<string, string>();
const GLASS_MIDDLE_PAN_GAIN = 2.8;
const GLASS_MIDDLE_PAN_CLAMP = 0.32;
const GLASS_TOUCH_PAN_GAIN = 1;
const GLASS_TOUCH_PAN_CLAMP = 0.14;

function panelLabelFromId(panelId: string): string {
  const cached = PANEL_LABEL_CACHE.get(panelId);
  if (cached) {
    return cached;
  }
  const label = panelId.split(".").slice(-1)[0].replace(/_/g, " ");
  PANEL_LABEL_CACHE.set(panelId, label);
  return label;
}

function roleLabel(value: string): string {
  return value.replace(/[_-]+/g, " ").trim() || "neutral";
}

function panelGlyph(presenceId: string, fallbackLabel: string): string {
  const cleanPresence = presenceId.trim();
  if (cleanPresence) {
    const token = cleanPresence.replace(/^health_sentinel_/, "").replace(/^presence\./, "");
    return token.slice(0, 2).toUpperCase();
  }
  return fallbackLabel.slice(0, 2).toUpperCase();
}

function clampRatio(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function isGlassInteractiveTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) {
    return false;
  }
  return Boolean(
    target.closest(
      "button, input, textarea, select, option, a, [role='button'], [data-panel-interactive='true']",
    ),
  );
}

function deriveOperationalState(councilReason: string, jobNote: string): OperationalState {
  const combined = `${councilReason} ${jobNote}`.toLowerCase();
  if (/blocked|deny|denied|forbidden|permission|backoff|skip|failed|error/.test(combined)) {
    return "blocked";
  }
  if (/wait|waiting|pending|pause|paused|idle|hold/.test(combined)) {
    return "paused";
  }
  return "running";
}

function summarizeOperationalReason(councilReason: string, jobNote: string): string {
  const source = jobNote.trim() || councilReason.trim() || "no detail";
  return source.replace(/\s+/g, " ").slice(0, 140);
}

function presenceRoleDescription(role: string): string {
  const normalized = role.trim().toLowerCase();
  if (normalized === "crawl-routing") {
    return "Routes crawler attention and node discovery order.";
  }
  if (normalized === "file-analysis") {
    return "Selects files for ingest and analysis cycles.";
  }
  if (normalized === "image-captioning") {
    return "Captions image evidence and proposes next actions.";
  }
  if (normalized === "council-orchestration") {
    return "Coordinates council rank decisions and panel ordering.";
  }
  if (normalized === "compliance-gating") {
    return "Checks gates and policy readiness before execution.";
  }
  if (normalized === "compute-scheduler") {
    return "Schedules compute lanes and runtime pressure response.";
  }
  if (normalized === "camera-guidance") {
    return "Keeps a glass-first camera lane and nudges map view gently.";
  }
  return "Neutral emitter: particles follow field influence and collisions.";
}

function presenceJobPrompt(panel: SortedPanel): string {
  const role = panel.presenceRole.trim().toLowerCase();
  if (role === "crawl-routing") {
    return "Prioritize the next crawl nodes and explain why they matter now.";
  }
  if (role === "file-analysis") {
    return "Choose which files to analyze next and state the evidence path.";
  }
  if (role === "image-captioning") {
    return "Caption current image evidence and include one concrete next action.";
  }
  if (role === "council-orchestration") {
    return "Re-rank active panels for the next operator cycle and explain ordering.";
  }
  if (role === "compliance-gating") {
    return "Report gate readiness and blocked reasons in concise form.";
  }
  if (role === "compute-scheduler") {
    return "Summarize compute pressure and recommend backend allocation now.";
  }
  if (role === "camera-guidance") {
    return "Describe what the operator should look at next and suggest one gentle pan cue.";
  }
  return "Report neutral field movement and suggest one useful operator action.";
}

function glassObservationPrompt(panel: SortedPanel): string {
  const role = panel.presenceRole.trim().toLowerCase();
  if (role === "crawl-routing") {
    return "Track frontier drift and watch where the crawl lane is pulling attention.";
  }
  if (role === "file-analysis") {
    return "Watch file clusters and identify which lane becomes the next ingestion target.";
  }
  if (role === "image-captioning") {
    return "Find the most visually active evidence region and hold it in view.";
  }
  if (role === "council-orchestration") {
    return "Observe rank pressure shifts and note which panel asks for escalation.";
  }
  if (role === "compliance-gating") {
    return "Watch gate-relevant regions and look for missing evidence signals.";
  }
  if (role === "compute-scheduler") {
    return "Watch pressure gradients and identify the hottest compute corridor.";
  }
  if (role === "camera-guidance") {
    return "Use the transparent viewport to track motion and steer where attention lands.";
  }
  return "Observe neutral field movement and where trajectories begin converging.";
}

function glassInteractionPrompt(panel: SortedPanel): string {
  const role = panel.presenceRole.trim().toLowerCase();
  if (role === "crawl-routing") {
    return "Use run job to commit the next crawl route after the camera lock stabilizes.";
  }
  if (role === "file-analysis") {
    return "Use run job to sync inbox analysis once the target cluster is centered.";
  }
  if (role === "image-captioning") {
    return "Use run job to caption what you are currently looking at.";
  }
  if (role === "council-orchestration") {
    return "Use move up/down after inspecting the active lane to steer council order.";
  }
  if (role === "compliance-gating") {
    return "Use run job to evaluate gate readiness for the focused scene.";
  }
  if (role === "compute-scheduler") {
    return "Use run job to rebalance compute for the region currently in frame.";
  }
  if (role === "camera-guidance") {
    return "Use gentle pan controls to shift the map view without abrupt camera jumps.";
  }
  return "Use run job to request one actionable next step from this focused view.";
}

const WorldPanelBody = memo(function WorldPanelBody({
  panel,
  collapse,
  coreFlightSpeed,
  anchorConfidence,
}: {
  panel: PanelRenderTarget;
  collapse: boolean;
  coreFlightSpeed: number;
  anchorConfidence: number;
}) {
  const renderedPanel = useMemo(() => panel.render(), [panel]);

  if (collapse) {
    return (
      <div className="world-panel-collapsed-body">
        <p>
          moving at velocity <code>{coreFlightSpeed.toFixed(2)}x</code>
        </p>
        <p>
          anchor confidence <code>{Math.round(anchorConfidence * 100)}%</code>
        </p>
      </div>
    );
  }

  return <div className="world-panel-body">{renderedPanel}</div>;
});

function WorldPanelsViewportInner({
  viewportWidth,
  viewportHeight,
  worldPanelLayout,
  panelNexusLayout,
  sortedPanels,
  panelWindowStateById,
  tertiaryPinnedPanelId,
  pinnedPanels,
  selectedPanelId,
  isEditMode,
  coreFlightSpeed,
  onToggleEditMode,
  onHoverPanel,
  onSelectPanel,
  onTogglePanelPin,
  onActivatePanel,
  onMinimizePanel,
  onClosePanel,
  onAdjustPanelCouncilRank,
  onPinPanelToTertiary,
  onFlyCameraToAnchor,
  onGlassInteractAt,
  onNudgeCameraPan,
  onWorldPanelDragEnd,
}: Props) {
  void viewportHeight;
  void onWorldPanelDragEnd;

  const [primarySplitRatio, setPrimarySplitRatio] = useState(1.55);
  const [tertiarySplitRatio, setTertiarySplitRatio] = useState(1);
  const [paneModeByPanelId, setPaneModeByPanelId] = useState<
    Record<string, FocusPaneMode>
  >({});
  const [jobBusyByPanelId, setJobBusyByPanelId] = useState<Record<string, boolean>>(
    {},
  );
  const [jobNoteByPanelId, setJobNoteByPanelId] = useState<Record<string, string>>(
    {},
  );
  const [trayPanelId, setTrayPanelId] = useState<string | null>(null);
  const [paneLocks, setPaneLocks] = useState<
    Partial<Record<"primary" | "secondary" | "tertiary", string>>
  >({});
  const [paneCountPreference, setPaneCountPreference] = useState<1 | 2 | 3>(3);
  const [detailsPanelId, setDetailsPanelId] = useState<string | null>(null);
  const [isPresenceRailCollapsed, setIsPresenceRailCollapsed] = useState(() => viewportWidth < 1460);
  const [presenceQuery, setPresenceQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<OperationalState | "all">("all");
  const [seamDrag, setSeamDrag] = useState<null | {
    seam: "primary" | "tertiary";
    startX: number;
    primaryRatio: number;
    tertiaryRatio: number;
  }>(null);
  const [glassDragPanelId, setGlassDragPanelId] = useState<string | null>(null);
  const seamDragRef = useRef<typeof seamDrag>(null);
  const seamDragFrameRef = useRef<number | null>(null);
  const seamDragMouseXRef = useRef<number | null>(null);
  const glassDragFrameRef = useRef<number | null>(null);
  const glassDragRef = useRef<null | {
    pointerId: number;
    panelId: string;
    paneRect: DOMRect;
    lastX: number;
    lastY: number;
    pendingDx: number;
    pendingDy: number;
    panGain: number;
    panClamp: number;
    moved: boolean;
  }>(null);
  const trayPanelIdRef = useRef<string | null>(trayPanelId);
  const iconButtonByPanelIdRef = useRef(new Map<string, HTMLButtonElement>());
  const iconPanelOrderRef = useRef<string[]>([]);
  const iconPanelIndexByIdRef = useRef(new Map<string, number>());
  const hoveredTrayIconIndexRef = useRef<number | null>(null);
  const hoveredTrayIconTargetRef = useRef<number | null>(null);
  const hoveredTrayIconFrameRef = useRef<number | null>(null);

  useEffect(() => {
    if (viewportWidth < 1120 && !isPresenceRailCollapsed) {
      setIsPresenceRailCollapsed(true);
    }
  }, [isPresenceRailCollapsed, viewportWidth]);

  useEffect(() => {
    seamDragRef.current = seamDrag;
  }, [seamDrag]);

  useEffect(() => {
    return () => {
      if (glassDragFrameRef.current !== null) {
        window.cancelAnimationFrame(glassDragFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!seamDrag) {
      return;
    }

    seamDragRef.current = seamDrag;
    const width = Math.max(viewportWidth, 860);

    const flushDrag = () => {
      seamDragFrameRef.current = null;
      const drag = seamDragRef.current;
      const mouseX = seamDragMouseXRef.current;
      if (!drag || mouseX === null) {
        return;
      }
      const deltaRatio = (mouseX - drag.startX) / width;
      if (drag.seam === "primary") {
        const next = clampRatio(drag.primaryRatio + (deltaRatio * 3.4), 1.05, 2.5);
        setPrimarySplitRatio((prev) => (Math.abs(prev - next) < 0.0001 ? prev : next));
        return;
      }
      const next = clampRatio(drag.tertiaryRatio - (deltaRatio * 2.8), 0.75, 1.45);
      setTertiarySplitRatio((prev) => (Math.abs(prev - next) < 0.0001 ? prev : next));
    };

    const onMouseMove = (event: MouseEvent) => {
      seamDragMouseXRef.current = event.clientX;
      if (seamDragFrameRef.current !== null) {
        return;
      }
      seamDragFrameRef.current = window.requestAnimationFrame(flushDrag);
    };

    const stopDragging = () => {
      seamDragRef.current = null;
      seamDragMouseXRef.current = null;
      if (seamDragFrameRef.current !== null) {
        window.cancelAnimationFrame(seamDragFrameRef.current);
        seamDragFrameRef.current = null;
      }
      setSeamDrag(null);
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", stopDragging);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", stopDragging);
      if (seamDragFrameRef.current !== null) {
        window.cancelAnimationFrame(seamDragFrameRef.current);
        seamDragFrameRef.current = null;
      }
    };
  }, [seamDrag, viewportWidth]);

  const applyHoveredTrayIconScales = useCallback((
    hoverIndex: number | null,
    trayIdOverride?: string | null,
  ) => {
    const panelOrder = iconPanelOrderRef.current;
    if (panelOrder.length <= 0) {
      return;
    }
    const trayId = trayIdOverride ?? trayPanelIdRef.current;
    panelOrder.forEach((panelId, index) => {
      const button = iconButtonByPanelIdRef.current.get(panelId);
      if (!button) {
        return;
      }
      const distance = hoverIndex === null ? 999 : Math.abs(index - hoverIndex);
      const proximity = hoverIndex === null ? 0 : Math.max(0, 1 - (distance / 3));
      const scale = 0.9 + (proximity * 0.58) + (trayId === panelId ? 0.2 : 0);
      button.style.setProperty("--unity-scale", scale.toFixed(3));
    });
  }, []);

  const applyHoveredTrayIconScalesDelta = useCallback((
    previousHoverIndex: number | null,
    nextHoverIndex: number | null,
    trayIdOverride?: string | null,
  ) => {
    const panelOrder = iconPanelOrderRef.current;
    if (panelOrder.length <= 0) {
      return;
    }
    const trayId = trayIdOverride ?? trayPanelIdRef.current;
    const applyIndexScale = (index: number) => {
      const panelId = panelOrder[index];
      const button = iconButtonByPanelIdRef.current.get(panelId);
      if (!button) {
        return;
      }
      const distance = nextHoverIndex === null ? 999 : Math.abs(index - nextHoverIndex);
      const proximity = nextHoverIndex === null ? 0 : Math.max(0, 1 - (distance / 3));
      const scale = 0.9 + (proximity * 0.58) + (trayId === panelId ? 0.2 : 0);
      button.style.setProperty("--unity-scale", scale.toFixed(3));
    };

    if (previousHoverIndex !== null) {
      for (let offset = -2; offset <= 2; offset += 1) {
        const index = previousHoverIndex + offset;
        if (index >= 0 && index < panelOrder.length) {
          applyIndexScale(index);
        }
      }
    }
    if (nextHoverIndex !== null) {
      for (let offset = -2; offset <= 2; offset += 1) {
        const index = nextHoverIndex + offset;
        if (index >= 0 && index < panelOrder.length) {
          applyIndexScale(index);
        }
      }
    }
  }, []);

  const applyTrayPanelScaleDelta = useCallback((
    previousTrayId: string | null,
    nextTrayId: string | null,
  ) => {
    if (previousTrayId === nextTrayId) {
      return;
    }
    const panelOrder = iconPanelOrderRef.current;
    const panelIndexById = iconPanelIndexByIdRef.current;
    if (panelOrder.length <= 0) {
      return;
    }
    const hoverIndex = hoveredTrayIconIndexRef.current;
    const applyPanelId = (panelId: string | null) => {
      if (!panelId) {
        return;
      }
      const index = panelIndexById.get(panelId);
      if (index === undefined) {
        return;
      }
      if (index < 0) {
        return;
      }
      const button = iconButtonByPanelIdRef.current.get(panelId);
      if (!button) {
        return;
      }
      const distance = hoverIndex === null ? 999 : Math.abs(index - hoverIndex);
      const proximity = hoverIndex === null ? 0 : Math.max(0, 1 - (distance / 3));
      const scale = 0.9 + (proximity * 0.58) + (nextTrayId === panelId ? 0.2 : 0);
      button.style.setProperty("--unity-scale", scale.toFixed(3));
    };
    applyPanelId(previousTrayId);
    applyPanelId(nextTrayId);
  }, []);

  const scheduleHoveredTrayIconIndex = useCallback((nextIndex: number | null) => {
    if (
      hoveredTrayIconTargetRef.current === nextIndex
      && hoveredTrayIconIndexRef.current === nextIndex
    ) {
      return;
    }
    hoveredTrayIconTargetRef.current = nextIndex;
    if (hoveredTrayIconFrameRef.current !== null) {
      return;
    }
    hoveredTrayIconFrameRef.current = window.requestAnimationFrame(() => {
      hoveredTrayIconFrameRef.current = null;
      const next = hoveredTrayIconTargetRef.current ?? null;
      const previous = hoveredTrayIconIndexRef.current;
      if (previous === next) {
        return;
      }
      hoveredTrayIconIndexRef.current = next;
      applyHoveredTrayIconScalesDelta(previous, next);
    });
  }, [applyHoveredTrayIconScalesDelta]);

  useEffect(() => {
    return () => {
      if (hoveredTrayIconFrameRef.current !== null) {
        window.cancelAnimationFrame(hoveredTrayIconFrameRef.current);
        hoveredTrayIconFrameRef.current = null;
      }
    };
  }, []);

  const paneModeForPanel = useCallback(
    (panelId: string): FocusPaneMode => {
      if (panelId === GLASS_VIEWPORT_PANEL_ID) {
        return "glass";
      }
      return paneModeByPanelId[panelId] ?? "panel";
    },
    [paneModeByPanelId],
  );

  const setPaneMode = useCallback((panelId: string, mode: FocusPaneMode) => {
    const id = panelId.trim();
    if (!id) {
      return;
    }
    if (id === GLASS_VIEWPORT_PANEL_ID) {
      setPaneModeByPanelId((prev) => {
        if (prev[id] === "glass") {
          return prev;
        }
        return {
          ...prev,
          [id]: "glass",
        };
      });
      return;
    }
    setPaneModeByPanelId((prev) => {
      const current = prev[id] ?? "panel";
      if (current === mode) {
        return prev;
      }
      return {
        ...prev,
        [id]: mode,
      };
    });
  }, []);

  const dispatchUiToast = useCallback((title: string, body: string) => {
    const text = body.trim();
    if (!text) {
      return;
    }
    window.dispatchEvent(
      new CustomEvent("ui:toast", {
        detail: {
          title,
          body: text,
        },
      }),
    );
  }, []);

  const requestPresenceJob = useCallback(async (
    path: string,
    payload: Record<string, unknown>,
  ): Promise<Record<string, unknown>> => {
    const base = runtimeBaseUrl();
    const response = await fetch(`${base}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data?.ok === false) {
      const details = String(data?.error ?? `request failed (${response.status})`);
      throw new Error(details);
    }
    return data as Record<string, unknown>;
  }, []);

  const runPresenceJob = useCallback(async (panel: SortedPanel) => {
    const panelId = panel.id;
    if (!panelId) {
      return;
    }
    setJobBusyByPanelId((prev) => ({ ...prev, [panelId]: true }));
    try {
      let summary = "job requested";
      if (panel.presenceRole === "crawl-routing") {
        await requestPresenceJob("/api/weaver/control", {
          action: "resume",
          seeds: [],
          max_depth: 2,
          max_nodes: 120,
          concurrency: 4,
          max_per_host: 2,
          entity_count: 12,
        });
        summary = "web crawl resume requested";
      } else if (panel.presenceRole === "file-analysis") {
        const payload = await requestPresenceJob("/api/eta-mu/sync", {});
        const syncStatus = String(payload?.sync_status ?? "ok");
        summary = `file analysis sync ${syncStatus}`;
      } else if (panel.presenceRole === "image-captioning") {
        const payload = await requestPresenceJob("/api/presence/say", {
          presence_id: panel.presenceId,
          text: presenceJobPrompt(panel),
        });
        summary = String(payload?.rendered_text ?? "caption planning requested")
          .replace(/\s+/g, " ")
          .slice(0, 160);
      } else if (panel.presenceRole === "compliance-gating") {
        const payload = await requestPresenceJob("/api/push-truth/dry-run", {});
        const gate = payload.gate;
        const blocked = Boolean(
          gate && typeof gate === "object" && (gate as Record<string, unknown>).blocked,
        );
        const needs = Array.isArray(payload?.needs)
          ? payload.needs.slice(0, 2).map((item: unknown) => String(item)).join(", ")
          : "";
        summary = blocked
          ? `gate blocked${needs ? `: ${needs}` : ""}`
          : "gate ready";
      } else if (panel.presenceRole === "camera-guidance") {
        const payload = await requestPresenceJob("/api/presence/say", {
          presence_id: panel.presenceId,
          text: presenceJobPrompt(panel),
        });
        summary = String(payload?.rendered_text ?? "camera cue ready")
          .replace(/\s+/g, " ")
          .slice(0, 160);
        onNudgeCameraPan(0.036, -0.02, panelId);
      } else {
        const payload = await requestPresenceJob("/api/presence/say", {
          presence_id: panel.presenceId,
          text: presenceJobPrompt(panel),
        });
        summary = String(payload?.rendered_text ?? "presence responded")
          .replace(/\s+/g, " ")
          .slice(0, 160);
      }
      setJobNoteByPanelId((prev) => ({ ...prev, [panelId]: summary }));
      dispatchUiToast(panel.presenceLabel, summary);
    } catch (error: unknown) {
      const details = error instanceof Error ? error.message : "job request failed";
      setJobNoteByPanelId((prev) => ({ ...prev, [panelId]: details }));
      dispatchUiToast(`${panel.presenceLabel} job`, details);
    } finally {
      setJobBusyByPanelId((prev) => ({ ...prev, [panelId]: false }));
    }
  }, [dispatchUiToast, onNudgeCameraPan, requestPresenceJob]);

  const guidePanelThroughGlass = useCallback((
    panel: SortedPanel,
    anchor: WorldAnchorTarget | null,
  ) => {
    const panelId = panel.id;
    if (!panelId) {
      return;
    }
    onActivatePanel(panelId);
    onSelectPanel(panelId);
    setPaneMode(panelId, "glass");
    if (anchor) {
      onFlyCameraToAnchor(anchor);
    }
    if (panel.id === GLASS_VIEWPORT_PANEL_ID || panel.presenceRole === "camera-guidance") {
      onNudgeCameraPan(0.032, -0.018, panelId);
    }
    const cue = anchor
      ? `camera moved to ${anchor.label}; ${glassInteractionPrompt(panel)}`
      : `glass interface active; ${glassInteractionPrompt(panel)}`;
    setJobNoteByPanelId((prev) => ({
      ...prev,
      [panelId]: cue,
    }));
    dispatchUiToast(panel.presenceLabel, cue);
  }, [
    dispatchUiToast,
    onActivatePanel,
    onFlyCameraToAnchor,
    onNudgeCameraPan,
    onSelectPanel,
    setPaneMode,
  ]);

  const nudgePanelView = useCallback((
    panel: SortedPanel,
    dx: number,
    dy: number,
    label: string,
  ) => {
    onNudgeCameraPan(dx, dy, panel.id);
    const note = `camera ${label}; ${glassInteractionPrompt(panel)}`;
    setJobNoteByPanelId((prev) => ({
      ...prev,
      [panel.id]: note,
    }));
    dispatchUiToast(panel.presenceLabel, note);
  }, [dispatchUiToast, onNudgeCameraPan]);

  const flushGlassPaneDrag = useCallback(() => {
    const drag = glassDragRef.current;
    if (!drag) {
      return;
    }
    const dx = clampRatio(drag.pendingDx * drag.panGain, -drag.panClamp, drag.panClamp);
    const dy = clampRatio(drag.pendingDy * drag.panGain, -drag.panClamp, drag.panClamp);
    drag.pendingDx = 0;
    drag.pendingDy = 0;
    if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) {
      return;
    }
    drag.moved = true;
    onNudgeCameraPan(dx, dy, drag.panelId);
  }, [onNudgeCameraPan]);

  const scheduleGlassPaneDragFlush = useCallback(() => {
    if (glassDragFrameRef.current !== null) {
      return;
    }
    glassDragFrameRef.current = window.requestAnimationFrame(() => {
      glassDragFrameRef.current = null;
      flushGlassPaneDrag();
    });
  }, [flushGlassPaneDrag]);

  const handleGlassPanePointerDown = useCallback((
    panelId: string,
    event: ReactPointerEvent<HTMLElement>,
  ) => {
    const interactiveTarget = isGlassInteractiveTarget(event.target);
    const isTouchPointer = event.pointerType === "touch";
    const isMiddleButtonPan = event.button === 1;
    const shouldStartPan = isMiddleButtonPan || isTouchPointer;

    // Middle-button drag or touch-drag pans the camera.
    if (shouldStartPan) {
      if (interactiveTarget) {
        return;
      }
      const paneRect = event.currentTarget.getBoundingClientRect();
      glassDragRef.current = {
        pointerId: event.pointerId,
        panelId,
        paneRect,
        lastX: event.clientX,
        lastY: event.clientY,
        pendingDx: 0,
        pendingDy: 0,
        panGain: isMiddleButtonPan ? GLASS_MIDDLE_PAN_GAIN : GLASS_TOUCH_PAN_GAIN,
        panClamp: isMiddleButtonPan ? GLASS_MIDDLE_PAN_CLAMP : GLASS_TOUCH_PAN_CLAMP,
        moved: false,
      };
      if (glassDragFrameRef.current !== null) {
        window.cancelAnimationFrame(glassDragFrameRef.current);
        glassDragFrameRef.current = null;
      }
      setGlassDragPanelId(panelId);
      event.currentTarget.setPointerCapture(event.pointerId);
      onActivatePanel(panelId);
      onSelectPanel(panelId);
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    // Left-click opens/focuses the nearest nexus target immediately.
    if (event.button === 0 && !interactiveTarget) {
      const paneRect = event.currentTarget.getBoundingClientRect();
      const xRatio = clampRatio((event.clientX - paneRect.left) / Math.max(1, paneRect.width), 0, 1);
      const yRatio = clampRatio((event.clientY - paneRect.top) / Math.max(1, paneRect.height), 0, 1);
      onActivatePanel(panelId);
      onSelectPanel(panelId);
      onGlassInteractAt(xRatio, yRatio);
      event.preventDefault();
      event.stopPropagation();
    }
  }, [onActivatePanel, onGlassInteractAt, onSelectPanel]);

  const handleGlassPanePointerMove = useCallback((event: ReactPointerEvent<HTMLElement>) => {
    const drag = glassDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    const dxPx = event.clientX - drag.lastX;
    const dyPx = event.clientY - drag.lastY;
    drag.lastX = event.clientX;
    drag.lastY = event.clientY;

    if (Math.abs(dxPx) < 0.25 && Math.abs(dyPx) < 0.25) {
      return;
    }

    const width = Math.max(220, drag.paneRect.width);
    const height = Math.max(180, drag.paneRect.height);
    drag.pendingDx += dxPx / width;
    drag.pendingDy += dyPx / height;
    if (glassDragFrameRef.current !== null) {
      event.preventDefault();
      return;
    }
    flushGlassPaneDrag();
    scheduleGlassPaneDragFlush();
    event.preventDefault();
    event.stopPropagation();
  }, [flushGlassPaneDrag, scheduleGlassPaneDragFlush]);

  const handleGlassPanePointerUp = useCallback((event: ReactPointerEvent<HTMLElement>) => {
    const drag = glassDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    // Handle releases for middle-button pan and touch pan.
    if (event.button !== 1 && event.pointerType !== "touch") {
      return;
    }
    if (glassDragFrameRef.current !== null) {
      window.cancelAnimationFrame(glassDragFrameRef.current);
      glassDragFrameRef.current = null;
    }
    const releasePanGain = event.pointerType === "touch" ? 2.5 : drag.panGain;
    const releasePanClamp = event.pointerType === "touch" ? 0.28 : drag.panClamp;
    // Middle mouse drag panning - increased speed
    if (Math.abs(drag.pendingDx) > 0.0005 || Math.abs(drag.pendingDy) > 0.0005) {
      const dx = clampRatio(drag.pendingDx * releasePanGain, -releasePanClamp, releasePanClamp);
      const dy = clampRatio(drag.pendingDy * releasePanGain, -releasePanClamp, releasePanClamp);
      drag.pendingDx = 0;
      drag.pendingDy = 0;
      if (Math.abs(dx) >= 0.001 || Math.abs(dy) >= 0.001) {
        drag.moved = true;
        onNudgeCameraPan(dx, dy, drag.panelId);
      }
    }

    glassDragRef.current = null;
    setGlassDragPanelId((prev) => (prev === drag.panelId ? null : prev));
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    event.preventDefault();
    event.stopPropagation();
  }, [onNudgeCameraPan]);

  const handleGlassPaneWheel = useCallback((
    panelId: string,
    event: ReactWheelEvent<HTMLElement>,
  ) => {
    if (isGlassInteractiveTarget(event.target)) {
      return;
    }
    const paneRect = event.currentTarget.getBoundingClientRect();
    const width = Math.max(220, paneRect.width);
    const height = Math.max(180, paneRect.height);
    const dx = clampRatio((-event.deltaX / width) * 3.2, -0.24, 0.24);
    const dy = clampRatio((-event.deltaY / height) * 3.2, -0.24, 0.24);
    if (Math.abs(dx) < 0.0006 && Math.abs(dy) < 0.0006) {
      return;
    }
    onNudgeCameraPan(dx, dy, panelId);
    event.preventDefault();
    event.stopPropagation();
  }, [onNudgeCameraPan]);

  const togglePanelGlassMode = useCallback((
    panel: SortedPanel,
    anchor: WorldAnchorTarget | null,
  ) => {
    if (panel.id === GLASS_VIEWPORT_PANEL_ID) {
      guidePanelThroughGlass(panel, anchor);
      return;
    }
    if (paneModeForPanel(panel.id) === "glass") {
      setPaneMode(panel.id, "panel");
      return;
    }
    guidePanelThroughGlass(panel, anchor);
  }, [guidePanelThroughGlass, paneModeForPanel, setPaneMode]);

  const panelById = useMemo(
    () => new Map(sortedPanels.map((panel) => [panel.id, panel])),
    [sortedPanels],
  );

  const rankByPanelId = useMemo(() => {
    const map = new Map<string, number>();
    sortedPanels.forEach((panel, index) => {
      map.set(panel.id, index + 1);
    });
    return map;
  }, [sortedPanels]);

  const layoutByPanelId = useMemo(
    () => new Map(worldPanelLayout.map((entry) => [entry.id, entry])),
    [worldPanelLayout],
  );

  const anchorByPanelId = useMemo(() => {
    const map = new Map<string, WorldAnchorTarget>();
    worldPanelLayout.forEach((entry) => {
      map.set(entry.id, entry.anchor);
    });
    panelNexusLayout.forEach((entry) => {
      if (!map.has(entry.panelId)) {
        map.set(entry.panelId, entry.anchor);
      }
    });
    return map;
  }, [panelNexusLayout, worldPanelLayout]);

  const rankedOpenIds = useMemo(() => {
    return sortedPanels
      .filter((panel) => {
        const state = panelWindowStateById[panel.id] ?? { open: true, minimized: false };
        return state.open && !state.minimized;
      })
      .map((panel) => panel.id);
  }, [panelWindowStateById, sortedPanels]);

  const focusCandidateIds = useMemo(
    () => rankedOpenIds.filter((panelId) => panelId !== trayPanelId),
    [rankedOpenIds, trayPanelId],
  );

  useEffect(() => {
    if (!trayPanelId) {
      return;
    }
    const state = panelWindowStateById[trayPanelId];
    if (!panelById.has(trayPanelId) || Boolean(state && (!state.open || state.minimized))) {
      setTrayPanelId(null);
    }
  }, [panelById, panelWindowStateById, trayPanelId]);

  useEffect(() => {
    const allowed = new Set(focusCandidateIds);
    setPaneLocks((prev) => {
      let touched = false;
      const next = { ...prev };
      (["primary", "secondary", "tertiary"] as const).forEach((kind) => {
        const locked = next[kind];
        if (locked && !allowed.has(locked)) {
          delete next[kind];
          touched = true;
        }
      });
      return touched ? next : prev;
    });
  }, [focusCandidateIds]);

  const sortedFocusIds = useMemo(() => {
    const openIdSet = new Set(focusCandidateIds);
    return sortedPanels
      .map((panel) => panel.id)
      .filter((panelId) => openIdSet.has(panelId));
  }, [focusCandidateIds, sortedPanels]);

  const isGlassForwardCandidate = useCallback((panelId: string): boolean => {
    if (panelId === GLASS_VIEWPORT_PANEL_ID || panelId === "nexus.ui.dedicated_views") {
      return true;
    }
    return paneModeForPanel(panelId) === "glass";
  }, [paneModeForPanel]);

  const setPaneLock = useCallback((
    paneKind: "primary" | "secondary" | "tertiary",
    panelId: string | null,
  ) => {
    setPaneLocks((prev) => {
      const next = { ...prev };
      if (!panelId) {
        if (!(paneKind in next)) {
          return prev;
        }
        delete next[paneKind];
        return next;
      }
      if (next[paneKind] === panelId) {
        return prev;
      }
      next[paneKind] = panelId;
      return next;
    });
  }, []);

  const clearPanelFromPaneLocks = useCallback((panelId: string) => {
    setPaneLocks((prev) => {
      let touched = false;
      const next = { ...prev };
      (["primary", "secondary", "tertiary"] as const).forEach((kind) => {
        if (next[kind] === panelId) {
          delete next[kind];
          touched = true;
        }
      });
      return touched ? next : prev;
    });
  }, []);

  const buildPaneAssignments = useCallback((
    locks: Partial<Record<"primary" | "secondary" | "tertiary", string>>,
  ) => {
    const queue = [...sortedFocusIds];
    const take = (panelId: string | null): string | null => {
      if (!panelId) {
        return null;
      }
      const index = queue.indexOf(panelId);
      if (index < 0) {
        return null;
      }
      const [match] = queue.splice(index, 1);
      return match ?? null;
    };

    const topOperationalScore = sortedFocusIds.reduce((max, panelId) => {
      if (isGlassForwardCandidate(panelId)) {
        return max;
      }
      const panel = panelById.get(panelId);
      return panel ? Math.max(max, panel.councilScore) : max;
    }, 0);
    const lowDemandCycle = topOperationalScore < 0.74;
    const glassPrimaryCandidate = sortedFocusIds.find((panelId) => {
      if (!isGlassForwardCandidate(panelId)) {
        return false;
      }
      const panel = panelById.get(panelId);
      if (!panel) {
        return false;
      }
      return lowDemandCycle || panel.councilScore >= topOperationalScore - 0.04;
    }) ?? null;

    const primary =
      take(locks.primary ?? null)
      ?? take(selectedPanelId && selectedPanelId !== trayPanelId ? selectedPanelId : null)
      ?? take(glassPrimaryCandidate)
      ?? queue.shift()
      ?? null;

    const secondary = take(locks.secondary ?? null) ?? queue.shift() ?? null;

    const tertiary =
      take(tertiaryPinnedPanelId)
      ?? take(locks.tertiary ?? null)
      ?? queue.shift()
      ?? null;

    return {
      primary,
      secondary,
      tertiary,
    };
  }, [
    isGlassForwardCandidate,
    panelById,
    selectedPanelId,
    sortedFocusIds,
    tertiaryPinnedPanelId,
    trayPanelId,
  ]);

  const paneAssignments = useMemo(
    () => buildPaneAssignments(paneLocks),
    [buildPaneAssignments, paneLocks],
  );

  const unlockedPaneAssignments = useMemo(
    () => buildPaneAssignments({}),
    [buildPaneAssignments],
  );

  const primaryPanelId = paneAssignments.primary;
  const secondaryPanelId = paneAssignments.secondary;
  const tertiaryPanelId = paneAssignments.tertiary;

  const primaryPanel = primaryPanelId ? panelById.get(primaryPanelId) ?? null : null;
  const primaryExpanded = Boolean(
    primaryPanel && primaryPanel.councilBoost >= 2 && focusCandidateIds.length > 1,
  );
  const primarySolo = Boolean(
    primaryPanel && primaryPanel.councilBoost >= 4 && focusCandidateIds.length > 2,
  );
  const maxPaneCount = viewportWidth >= 1420 ? 3 : viewportWidth >= 1100 ? 2 : 1;
  const paneCount = Math.min(paneCountPreference, maxPaneCount);
  const showSecondaryPane = paneCount >= 2;
  const showTertiaryPane = paneCount >= 3;
  const operationalMetaByPanelId = useMemo(() => {
    const map = new Map<string, { state: OperationalState; reason: string }>();
    sortedPanels.forEach((panel) => {
      const jobNote = jobNoteByPanelId[panel.id] ?? "";
      map.set(panel.id, {
        state: deriveOperationalState(panel.councilReason, jobNote),
        reason: summarizeOperationalReason(panel.councilReason, jobNote),
      });
    });
    return map;
  }, [jobNoteByPanelId, sortedPanels]);

  const deferredPresenceQuery = useDeferredValue(presenceQuery);
  const normalizedPresenceQuery = deferredPresenceQuery.trim().toLowerCase();
  const railSearchTextByPanelId = useMemo(() => {
    const map = new Map<string, string>();
    sortedPanels.forEach((panel) => {
      map.set(
        panel.id,
        [
          panel.presenceLabel,
          panel.presenceLabelJa,
          panel.presenceId,
          panel.id,
          panelLabelFromId(panel.id),
          panel.presenceRole,
        ]
          .join(" ")
          .toLowerCase(),
      );
    });
    return map;
  }, [sortedPanels]);

  const focusPaneIdSet = useMemo(() => {
    const ids = [primaryPanelId, secondaryPanelId, tertiaryPanelId].filter(
      (panelId): panelId is string => Boolean(panelId),
    );
    return new Set(ids);
  }, [primaryPanelId, secondaryPanelId, tertiaryPanelId]);

  const {
    panelStatusCounts,
    iconPanels,
    smartPilePanels,
  } = useMemo(() => {
    const counts: Record<OperationalState, number> = {
      running: 0,
      paused: 0,
      blocked: 0,
    };
    const nextIconPanels: SortedPanel[] = [];
    const nextSmartPilePanels: SortedPanel[] = [];
    const nextSmartPileFallbackPanels: SortedPanel[] = [];

    sortedPanels.forEach((panel) => {
      const status = operationalMetaByPanelId.get(panel.id)?.state ?? "running";
      counts[status] += 1;
      if (statusFilter !== "all" && status !== statusFilter) {
        return;
      }

      if (normalizedPresenceQuery) {
        const searchText = railSearchTextByPanelId.get(panel.id) ?? "";
        if (!searchText.includes(normalizedPresenceQuery)) {
          return;
        }
      }

      nextIconPanels.push(panel);

      if (panel.id !== trayPanelId) {
        nextSmartPileFallbackPanels.push(panel);
      }

      const panelState = panelWindowStateById[panel.id] ?? { open: true, minimized: false };
      if (!panelState.open || panelState.minimized) {
        return;
      }
      if (panel.id === trayPanelId || focusPaneIdSet.has(panel.id)) {
        return;
      }
      if (pinnedPanels[panel.id]) {
        return;
      }
      nextSmartPilePanels.push(panel);
    });

    const effectiveSmartPilePanels =
      nextSmartPilePanels.length > 0 ? nextSmartPilePanels : nextSmartPileFallbackPanels;

    return {
      panelStatusCounts: counts,
      iconPanels: nextIconPanels,
      smartPilePanels: effectiveSmartPilePanels,
    };
  }, [
    focusPaneIdSet,
    normalizedPresenceQuery,
    operationalMetaByPanelId,
    railSearchTextByPanelId,
    panelWindowStateById,
    pinnedPanels,
    sortedPanels,
    statusFilter,
    trayPanelId,
  ]);

  const smartPilePreviewPanels = useMemo(
    () => smartPilePanels.slice(0, 10),
    [smartPilePanels],
  );

  const iconRenderMetaByPanelId = useMemo(() => {
    const map = new Map<string, {
      anchor: WorldAnchorTarget | null;
      style: CSSProperties;
      rank: number;
      operationalState: OperationalState;
      operationalReason: string;
    }>();
    iconPanels.forEach((panel, index) => {
      const anchor = anchorByPanelId.get(panel.id) ?? null;
      const hue = Math.round(anchor?.hue ?? (180 + (index * 22)));
      const lensX = `${Math.round((anchor?.x ?? 0.5) * 100)}%`;
      const lensY = `${Math.round((anchor?.y ?? 0.5) * 100)}%`;
      const operationalMeta = operationalMetaByPanelId.get(panel.id);
      map.set(panel.id, {
        anchor,
        style: {
          "--unity-scale": "0.900",
          "--unity-hue": `${hue}`,
          "--unity-lens-x": lensX,
          "--unity-lens-y": lensY,
        } as CSSProperties,
        rank: Math.max(1, rankByPanelId.get(panel.id) ?? 1),
        operationalState: operationalMeta?.state ?? "running",
        operationalReason: operationalMeta?.reason ?? "no detail",
      });
    });
    return map;
  }, [anchorByPanelId, iconPanels, operationalMetaByPanelId, rankByPanelId]);

  useEffect(() => {
    const order = iconPanels.map((panel) => panel.id);
    iconPanelOrderRef.current = order;
    iconPanelIndexByIdRef.current = new Map(order.map((panelId, index) => [panelId, index]));
    applyHoveredTrayIconScales(hoveredTrayIconIndexRef.current);
  }, [applyHoveredTrayIconScales, iconPanels]);

  useEffect(() => {
    const previousTrayId = trayPanelIdRef.current;
    trayPanelIdRef.current = trayPanelId;
    applyTrayPanelScaleDelta(previousTrayId, trayPanelId);
  }, [applyTrayPanelScaleDelta, trayPanelId]);

  const movePanelToTray = useCallback((panelId: string) => {
    if (trayPanelId === panelId) {
      setTrayPanelId(null);
      return;
    }
    onActivatePanel(panelId);
    clearPanelFromPaneLocks(panelId);
    setTrayPanelId(panelId);
  }, [clearPanelFromPaneLocks, onActivatePanel, trayPanelId]);

  const focusTrayPanel = useCallback(() => {
    if (!trayPanelId) {
      return;
    }
    setTrayPanelId(null);
    onActivatePanel(trayPanelId);
    onSelectPanel(trayPanelId);
    onAdjustPanelCouncilRank(trayPanelId, 2);
  }, [onActivatePanel, onAdjustPanelCouncilRank, onSelectPanel, trayPanelId]);

  const dismissTrayPanel = useCallback(() => {
    setTrayPanelId(null);
  }, []);

  const gridTemplateColumns = useMemo(() => {
    if (viewportWidth < 1100) {
      return "minmax(0, 1fr)";
    }
    const primaryRatio = primarySolo
      ? Math.max(2.05, primarySplitRatio + 0.6)
      : primaryExpanded
        ? Math.max(1.8, primarySplitRatio + 0.25)
        : primarySplitRatio;
    if (showTertiaryPane) {
      return `minmax(0, ${primaryRatio.toFixed(2)}fr) 0.52rem minmax(0, 1fr) 0.52rem minmax(0, ${tertiarySplitRatio.toFixed(2)}fr)`;
    }
    return `minmax(0, ${primaryRatio.toFixed(2)}fr) 0.52rem minmax(0, 1fr)`;
  }, [
    primaryExpanded,
    primarySolo,
    primarySplitRatio,
    showTertiaryPane,
    tertiarySplitRatio,
    viewportWidth,
  ]);

  const startPrimarySeamDrag = useCallback((event: ReactMouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setSeamDrag({
      seam: "primary",
      startX: event.clientX,
      primaryRatio: primarySplitRatio,
      tertiaryRatio: tertiarySplitRatio,
    });
  }, [primarySplitRatio, tertiarySplitRatio]);

  const startTertiarySeamDrag = useCallback((event: ReactMouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setSeamDrag({
      seam: "tertiary",
      startX: event.clientX,
      primaryRatio: primarySplitRatio,
      tertiaryRatio: tertiarySplitRatio,
    });
  }, [primarySplitRatio, tertiarySplitRatio]);

  const renderFocusPane = (
    paneKind: "primary" | "secondary" | "tertiary",
    panelId: string | null,
  ): ReactNode => {
    if (!panelId) {
      return (
        <article className={`world-focus-pane world-focus-pane-${paneKind} world-focus-pane-empty`}>
          <header className="world-focus-pane-header">
            <div>
              <p className="world-focus-kicker">{paneKind} focus pane</p>
              <p className="world-focus-title">Glass lane awaiting presence</p>
            </div>
          </header>
          <div className="world-glass-pane world-glass-pane-empty-lane">
            <p className="world-glass-title">transparent standby viewport</p>
            <div className="world-glass-grid">
              <p className="world-glass-row">This lane remains glass until a presence is focused here.</p>
              <p className="world-glass-row">Council votes can move glass panes forward when simulation view is preferred.</p>
              <p className="world-glass-row">Use side task tray cards to move a presence into this lane.</p>
            </div>
          </div>
        </article>
      );
    }

    const panel = panelById.get(panelId);
    if (!panel) {
      return null;
    }
    const panelRank = Math.max(1, rankByPanelId.get(panelId) ?? 1);
    const layoutEntry = layoutByPanelId.get(panelId);
    const anchor = layoutEntry?.anchor ?? anchorByPanelId.get(panelId) ?? null;
    const panelState = panelWindowStateById[panelId] ?? { open: true, minimized: false };
    const paneMode = paneModeForPanel(panelId);
    const isPinned = Boolean(pinnedPanels[panelId]);
    const isTertiaryPinned = tertiaryPinnedPanelId === panelId;
    const isViewportKeeper =
      panel.id === GLASS_VIEWPORT_PANEL_ID || panel.presenceRole === "camera-guidance";
    const glassObservePrompt = glassObservationPrompt(panel);
    const glassInteractPrompt = glassInteractionPrompt(panel);
    const paneLocked = paneLocks[paneKind] === panelId;
    const unlockedPanelForPane = unlockedPaneAssignments[paneKind];
    const pendingShift = paneLocked && unlockedPanelForPane && unlockedPanelForPane !== panelId;
    const glassOverlayMode = paneMode === "glass";
    const preferredGlassPrimary = paneKind === "primary" && isGlassForwardCandidate(panelId);
    const jobBusy = Boolean(jobBusyByPanelId[panelId]);
    const jobNote = jobNoteByPanelId[panelId] ?? "";
    const anchorConfidence = anchor?.confidence ?? 0.5;
    const operationalMeta = operationalMetaByPanelId.get(panelId);
    const complianceState = operationalMeta?.state ?? "running";
    const complianceReason = operationalMeta?.reason ?? "no detail";
    const detailsOpen = detailsPanelId === panelId;
    const compactGlass = glassOverlayMode && !detailsOpen;
    const scoreDialRatio = clampRatio(0.5 + (Math.tanh(panel.councilScore * 0.45) * 0.5), 0, 1);
    const particleDialRatio = clampRatio(
      Math.log1p(Math.max(0, panel.particleCount)) / Math.log(241),
      0,
      1,
    );
    const boostDialRatio = clampRatio((panel.councilBoost + 4) / 8, 0, 1);
    const metaDials: Array<{
      id: string;
      label: string;
      valueLabel: string;
      ratio: number;
      hue: number;
    }> = [
      {
        id: "confidence",
        label: "confidence",
        valueLabel: `${Math.round(anchorConfidence * 100)}%`,
        ratio: clampRatio(anchorConfidence, 0, 1),
        hue: 196,
      },
      {
        id: "score",
        label: "score",
        valueLabel: panel.councilScore.toFixed(2),
        ratio: scoreDialRatio,
        hue: 158,
      },
      {
        id: "particles",
        label: "particles",
        valueLabel: `${panel.particleCount}`,
        ratio: particleDialRatio,
        hue: 34,
      },
    ];
    if (detailsOpen) {
      metaDials.push({
        id: "boost",
        label: "boost",
        valueLabel: panel.councilBoost >= 0 ? `+${panel.councilBoost}` : `${panel.councilBoost}`,
        ratio: boostDialRatio,
        hue: 208,
      });
    }

    return (
      <article
        className={`world-focus-pane world-focus-pane-${paneKind} ${selectedPanelId === panelId ? "world-focus-pane-selected" : ""} ${glassOverlayMode ? "world-focus-pane-glass" : ""} ${glassOverlayMode && glassDragPanelId === panelId ? "world-focus-pane-glass-dragging" : ""}`}
        onMouseEnter={() => {
          onHoverPanel(panelId);
          setPaneLock(paneKind, panelId);
        }}
        onMouseLeave={() => {
          onHoverPanel(null);
          setPaneLock(paneKind, null);
        }}
        onPointerDownCapture={glassOverlayMode ? (event) => handleGlassPanePointerDown(panelId, event) : undefined}
        onPointerMoveCapture={glassOverlayMode ? handleGlassPanePointerMove : undefined}
        onPointerUpCapture={glassOverlayMode ? handleGlassPanePointerUp : undefined}
        onPointerCancelCapture={glassOverlayMode ? handleGlassPanePointerUp : undefined}
        onLostPointerCapture={glassOverlayMode ? handleGlassPanePointerUp : undefined}
        onWheelCapture={glassOverlayMode ? (event) => handleGlassPaneWheel(panelId, event) : undefined}
      >
        <header className="world-focus-pane-header">
          <div className="min-w-0">
            <p className="world-focus-kicker">
              {paneKind} focus pane · rank {panelRank}
            </p>
            <p className="world-focus-title">{panelLabelFromId(panel.id)}</p>
            <p className="world-focus-presence">
              {panel.presenceLabel}
              {panel.presenceLabelJa ? <span> / {panel.presenceLabelJa}</span> : null}
            </p>
          </div>
          <div className="world-focus-status-stack">
            <span className="world-focus-status-pill">score {panel.councilScore.toFixed(2)}</span>
            <span className="world-focus-status-pill">boost {panel.councilBoost >= 0 ? `+${panel.councilBoost}` : panel.councilBoost}</span>
            <span
              className={`world-focus-status-pill ${
                panel.particleDisposition === "neutral"
                  ? "world-focus-status-pill-neutral"
                  : "world-focus-status-pill-role"
              }`}
            >
              particles {panel.particleCount} ({panel.particleDisposition})
            </span>
            <div className="world-focus-pane-modes" data-panel-interactive="true">
              {([1, 2, 3] as const).map((count) => (
                <button
                  key={`${paneKind}:mode:${count}`}
                  type="button"
                  className={`world-focus-pane-mode-btn ${paneCount === count ? "world-focus-pane-mode-btn-active" : ""}`}
                  onClick={() => setPaneCountPreference(count)}
                  disabled={count > maxPaneCount}
                  title={`show ${count} focus pane${count > 1 ? "s" : ""}`}
                >
                  {count}
                </button>
              ))}
            </div>
          </div>
        </header>

        {!compactGlass ? (
          <div className="world-focus-meta-row">
            <span className="world-focus-meta-chip">state {panelState.open ? (panelState.minimized ? "min" : "open") : "closed"}</span>
            <span className={`world-focus-meta-chip world-focus-meta-chip-state world-focus-meta-chip-state-${complianceState}`}>
              {complianceState}
            </span>
            {detailsOpen ? <span className="world-focus-meta-chip">presence {panel.presenceId}</span> : null}
            {detailsOpen ? <span className="world-focus-meta-chip">role {roleLabel(panel.presenceRole)}</span> : null}
            {detailsOpen ? <span className="world-focus-meta-chip">mode {paneMode}</span> : null}
            {detailsOpen && isPinned ? <span className="world-focus-meta-chip">pinned</span> : null}
            {detailsOpen && isTertiaryPinned ? <span className="world-focus-meta-chip">tertiary</span> : null}
          </div>
        ) : null}

        {!compactGlass ? (
          <div className="world-focus-dial-row">
            {metaDials.map((dial) => (
              <div key={`${panelId}:${dial.id}`} className="world-focus-dial">
                <span
                  className="world-focus-dial-face"
                  style={{
                    "--dial-hue": `${Math.round(dial.hue)}`,
                    "--dial-value": `${Math.round(clampRatio(dial.ratio, 0, 1) * 100)}%`,
                  } as CSSProperties}
                  aria-hidden="true"
                >
                  <span className="world-focus-dial-core">{dial.valueLabel}</span>
                </span>
                <span className="world-focus-dial-label">{dial.label}</span>
              </div>
            ))}
          </div>
        ) : null}

        {!compactGlass ? (
          <p className="world-focus-reason world-focus-reason-muted">
            {complianceState} · confidence {Math.round(anchorConfidence * 100)}% · {complianceReason}
          </p>
        ) : null}
        {detailsOpen ? (
          <>
            <p className="world-focus-reason world-focus-reason-muted">{presenceRoleDescription(panel.presenceRole)}</p>
            {preferredGlassPrimary ? (
              <p className="world-focus-reason">preferred simulation frame: glass overlay stays primary when demand is low.</p>
            ) : null}
            {paneLocked ? <p className="world-focus-reason">mouse lock: pane holds this card</p> : null}
            {pendingShift ? (
              <p className="world-focus-reason">
                pending rank on release: {panelLabelFromId(unlockedPanelForPane)} takes this pane
              </p>
            ) : null}
            {jobNote ? <p className="world-focus-reason">job: {jobNote}</p> : null}

            <div className="world-focus-tool-row">
              {panel.toolHints.slice(0, 6).map((hint) => (
                <span key={`${panel.id}:${hint}`} className="world-focus-tool-chip">
                  {hint}
                </span>
              ))}
            </div>
          </>
        ) : null}

        <div className="world-focus-actions" data-panel-interactive="true">
          {compactGlass ? (
            <>
              <button
                type="button"
                className="world-focus-action"
                title="guide camera toward this panel anchor"
                onClick={() => guidePanelThroughGlass(panel, anchor)}
              >
                cue
              </button>
              <button
                type="button"
                className="world-focus-action"
                title="run presence job now"
                onClick={() => {
                  void runPresenceJob(panel);
                }}
                disabled={jobBusy}
              >
                {jobBusy ? "..." : "job"}
              </button>
              <button
                type="button"
                className="world-focus-action"
                title="show situational controls"
                onClick={() => setDetailsPanelId(panelId)}
              >
                more
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className="world-focus-action"
                title="focus this panel"
                onClick={() => {
                  onActivatePanel(panelId);
                  onSelectPanel(panelId);
                }}
              >
                focus
              </button>
              <button
                type="button"
                className="world-focus-action"
                title="raise rank priority"
                onClick={() => onAdjustPanelCouncilRank(panelId, 1)}
              >
                +rank
              </button>
              <button
                type="button"
                className="world-focus-action"
                title="lower rank priority"
                onClick={() => onAdjustPanelCouncilRank(panelId, -1)}
              >
                -rank
              </button>
              <button
                type="button"
                className="world-focus-action"
                title="run presence job now"
                onClick={() => {
                  void runPresenceJob(panel);
                }}
                disabled={jobBusy}
              >
                {jobBusy ? "..." : "job"}
              </button>
              <button
                type="button"
                className="world-focus-action"
                title={detailsOpen ? "hide situational controls" : "show situational controls"}
                onClick={() => setDetailsPanelId((prev) => (prev === panelId ? null : panelId))}
              >
                {detailsOpen ? "less" : "more"}
              </button>

              {detailsOpen ? (
                <button
                  type="button"
                  className="world-focus-action"
                  title="move card into smart pile"
                  onClick={() => movePanelToTray(panelId)}
                >
                  pile
                </button>
              ) : null}
              {detailsOpen ? (
                <button
                  type="button"
                  className="world-focus-action"
                  title={paneMode === "glass" ? "switch back to panel view" : "switch to glass simulation view"}
                  onClick={() => togglePanelGlassMode(panel, anchor)}
                >
                  {panel.id === GLASS_VIEWPORT_PANEL_ID
                    ? "glass fixed"
                    : paneMode === "glass"
                      ? "panel"
                      : "glass"}
                </button>
              ) : null}
              {detailsOpen ? (
                <button
                  type="button"
                  className="world-focus-action"
                  title={isTertiaryPinned ? "remove tertiary pin" : "pin panel to tertiary lane"}
                  onClick={() => onPinPanelToTertiary(panelId)}
                >
                  {isTertiaryPinned ? "un-ter" : "ter"}
                </button>
              ) : null}
              {detailsOpen ? (
                <button
                  type="button"
                  className="world-focus-action"
                  title={isPinned ? "unpin from pinned set" : "pin panel in set"}
                  onClick={() => onTogglePanelPin(panelId)}
                  disabled={panel.id === GLASS_VIEWPORT_PANEL_ID}
                >
                  {panel.id === GLASS_VIEWPORT_PANEL_ID
                    ? "always"
                    : isPinned ? "unpin" : "pin"}
                </button>
              ) : null}
              {detailsOpen ? (
                <button
                  type="button"
                  className="world-focus-action"
                  title="guide camera toward this panel anchor"
                  onClick={() => guidePanelThroughGlass(panel, anchor)}
                >
                  guide
                </button>
              ) : null}
              {detailsOpen && anchor ? (
                <button
                  type="button"
                  className="world-focus-action"
                  title="inspect anchor in simulation"
                  onClick={() => onFlyCameraToAnchor(anchor)}
                >
                  look
                </button>
              ) : null}
              {detailsOpen && panel.id !== GLASS_VIEWPORT_PANEL_ID ? (
                <button
                  type="button"
                  className="world-focus-action"
                  title="minimize panel"
                  onClick={() => onMinimizePanel(panelId)}
                >
                  min
                </button>
              ) : null}
              {detailsOpen && panel.id !== GLASS_VIEWPORT_PANEL_ID ? (
                <button
                  type="button"
                  className="world-focus-action world-focus-action-close"
                  title="close panel"
                  onClick={() => onClosePanel(panelId)}
                >
                  x
                </button>
              ) : null}
            </>
          )}
        </div>

        {paneMode === "glass" ? (
            <div className="world-glass-pane">
            <p className="world-glass-title">glass viewport</p>
            <div className="world-glass-grid">
              <p className="world-glass-row">
                middle drag or touch drag to pan · trackpad two-finger scroll pans · click opens/focuses nexus
              </p>
              {detailsOpen ? (
                <>
                  <p className="world-glass-row">
                    panel {panelLabelFromId(panel.id)} · rank {panelRank}
                  </p>
                  <p className="world-glass-row">
                    particles {panel.particleCount} · {panel.particleDisposition}
                  </p>
                  <p className="world-glass-row">
                    presence {panel.presenceId} · role {roleLabel(panel.presenceRole)}
                  </p>
                  <p className="world-glass-row">observe: {glassObservePrompt}</p>
                  <p className="world-glass-row">interact: {glassInteractPrompt}</p>
                  {anchor ? (
                    <p className="world-glass-row">
                      anchor {anchor.label} @ {anchor.x.toFixed(2)}, {anchor.y.toFixed(2)}
                    </p>
                  ) : null}
                </>
              ) : null}
            </div>
            {detailsOpen ? (
              <div className="world-glass-actions" data-panel-interactive="true">
                <button
                  type="button"
                  onClick={() => guidePanelThroughGlass(panel, anchor)}
                >
                  camera cue
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onActivatePanel(panelId);
                    onSelectPanel(panelId);
                  }}
                >
                  focus panel
                </button>
                <button
                  type="button"
                  onClick={() => {
                    guidePanelThroughGlass(panel, anchor);
                    void runPresenceJob(panel);
                  }}
                  disabled={jobBusy}
                >
                  {jobBusy ? "running" : "guide + job"}
                </button>
              </div>
            ) : null}
            {detailsOpen && isViewportKeeper ? (
              <div className="world-glass-pan-pad" data-panel-interactive="true">
                <button
                  type="button"
                  onClick={() => nudgePanelView(panel, -0.11, 0, "panned left")}
                >
                  pan left
                </button>
                <button
                  type="button"
                  onClick={() => nudgePanelView(panel, 0.11, 0, "panned right")}
                >
                  pan right
                </button>
                <button
                  type="button"
                  onClick={() => nudgePanelView(panel, 0, -0.09, "panned up")}
                >
                  pan up
                </button>
                <button
                  type="button"
                  onClick={() => nudgePanelView(panel, 0, 0.09, "panned down")}
                >
                  pan down
                </button>
                <button
                  type="button"
                  onClick={() => {
                    guidePanelThroughGlass(panel, anchor);
                    nudgePanelView(panel, 0.05, -0.03, "drifted")
                  }}
                >
                  gentle drift
                </button>
              </div>
            ) : null}
            {detailsOpen ? (
              <div className="world-glass-tool-row">
                {panel.toolHints.slice(0, 4).map((hint) => (
                  <span key={`${panel.id}:glass:${hint}`} className="world-focus-tool-chip">
                    {hint}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        ) : (
          <WorldPanelBody
            panel={layoutEntry?.panel ?? panel}
            collapse={Boolean(layoutEntry?.collapse)}
            coreFlightSpeed={coreFlightSpeed}
            anchorConfidence={anchor?.confidence ?? 0.5}
          />
        )}
      </article>
    );
  };

  const renderTrayCard = (panel: SortedPanel): ReactNode => {
    const panelId = panel.id;
    const paneMode = paneModeForPanel(panelId);
    const rank = Math.max(1, rankByPanelId.get(panelId) ?? 1);
    const layoutEntry = layoutByPanelId.get(panelId);
    const anchor = layoutEntry?.anchor ?? anchorByPanelId.get(panelId) ?? null;
    const panelState = panelWindowStateById[panelId] ?? { open: true, minimized: false };
    const isViewportKeeper =
      panel.id === GLASS_VIEWPORT_PANEL_ID || panel.presenceRole === "camera-guidance";
    const jobBusy = Boolean(jobBusyByPanelId[panelId]);
    const jobNote = jobNoteByPanelId[panelId] ?? "";

    return (
      <article
        className="world-unity-tray-card"
        onMouseEnter={() => onHoverPanel(panelId)}
        onMouseLeave={() => onHoverPanel(null)}
      >
        <header className="world-pinned-window-header">
          <p className="world-smart-card-rank">#{rank}</p>
          <p className="world-smart-card-title">{panelLabelFromId(panel.id)}</p>
          <p className="world-smart-card-score">{panel.councilScore.toFixed(2)}</p>
        </header>

        <p className="world-smart-card-presence">
          {panel.presenceLabel} · {roleLabel(panel.presenceRole)}
        </p>
        <p className="world-smart-card-reason">{presenceRoleDescription(panel.presenceRole)}</p>
        <p className="world-smart-card-reason">
          state {panelState.open ? (panelState.minimized ? "min" : "open") : "closed"}
        </p>
        {jobNote ? <p className="world-smart-card-reason">job: {jobNote}</p> : null}

        <div className="world-smart-card-actions" data-panel-interactive="true">
          <button type="button" onClick={focusTrayPanel}>focus primary</button>
          <button type="button" onClick={() => onAdjustPanelCouncilRank(panelId, 1)}>move up</button>
          <button type="button" onClick={() => onAdjustPanelCouncilRank(panelId, -1)}>move down</button>
          <button
            type="button"
            onClick={() => togglePanelGlassMode(panel, anchor)}
          >
            {panel.id === GLASS_VIEWPORT_PANEL_ID
              ? "glass fixed"
              : paneMode === "glass" ? "panel" : "glass"}
          </button>
          <button
            type="button"
            onClick={() => {
              void runPresenceJob(panel);
            }}
            disabled={jobBusy}
          >
            {jobBusy ? "running" : "job"}
          </button>
          <button
            type="button"
            onClick={() => guidePanelThroughGlass(panel, anchor)}
          >
            guide
          </button>
          {anchor ? (
            <button
              type="button"
              onClick={() => onFlyCameraToAnchor(anchor)}
            >
              inspect
            </button>
          ) : null}
          <button type="button" onClick={dismissTrayPanel}>hide</button>
        </div>

        {paneMode === "glass" ? (
          <div className="world-pinned-glass">
            <p className="world-glass-row">
              particles {panel.particleCount} · {panel.particleDisposition}
            </p>
            <p className="world-glass-row">observe: {glassObservationPrompt(panel)}</p>
            <p className="world-glass-row">interact: {glassInteractionPrompt(panel)}</p>
            {anchor ? (
              <p className="world-glass-row">
                anchor {anchor.label} @ {anchor.x.toFixed(2)}, {anchor.y.toFixed(2)}
              </p>
            ) : null}
            {isViewportKeeper ? (
              <div className="world-glass-pan-pad" data-panel-interactive="true">
                <button type="button" onClick={() => nudgePanelView(panel, -0.11, 0, "panned left")}>pan left</button>
                <button type="button" onClick={() => nudgePanelView(panel, 0.11, 0, "panned right")}>pan right</button>
                <button type="button" onClick={() => nudgePanelView(panel, 0, -0.09, "panned up")}>pan up</button>
                <button type="button" onClick={() => nudgePanelView(panel, 0, 0.09, "panned down")}>pan down</button>
                <button
                  type="button"
                  onClick={() => {
                    guidePanelThroughGlass(panel, anchor);
                    nudgePanelView(panel, 0.05, -0.03, "drifted");
                  }}
                >
                  gentle drift
                </button>
              </div>
            ) : null}
          </div>
        ) : (
          <div className="world-pinned-window-body">
            <WorldPanelBody
              panel={layoutEntry?.panel ?? panel}
              collapse={Boolean(layoutEntry?.collapse)}
              coreFlightSpeed={coreFlightSpeed}
              anchorConfidence={anchor?.confidence ?? 0.5}
            />
          </div>
        )}
      </article>
    );
  };

  const renderSmartPileCard = (panel: SortedPanel): ReactNode => {
    const panelId = panel.id;
    const rank = Math.max(1, rankByPanelId.get(panelId) ?? 1);
    const panelState = panelWindowStateById[panelId] ?? { open: true, minimized: false };
    const panelStateLabel = panelState.open ? (panelState.minimized ? "min" : "open") : "closed";
    const anchor = anchorByPanelId.get(panelId) ?? null;
    const operationalMeta = operationalMetaByPanelId.get(panelId);
    const operationalState = operationalMeta?.state ?? "running";
    const operationalReason = operationalMeta?.reason ?? "no detail";

    return (
      <article key={`smart-pile-${panelId}`} className="world-smart-card world-smart-card-compact">
        <header className="world-smart-card-header">
          <p className="world-smart-card-rank">#{rank}</p>
          <p className="world-smart-card-title">{panelLabelFromId(panelId)}</p>
          <p className="world-smart-card-score">{panel.councilScore.toFixed(2)}</p>
        </header>
        <p className="world-smart-card-presence">{panel.presenceLabel}</p>
        <p className="world-smart-card-reason">
          <span className={`world-smart-card-status world-smart-card-status-${operationalState}`}>
            {operationalState}
          </span>
          {" "}· state {panelStateLabel}
        </p>
        <p className="world-smart-card-reason">{operationalReason}</p>
        <div className="world-smart-card-actions" data-panel-interactive="true">
          <button
            type="button"
            title="promote to primary lane"
            onClick={() => {
              onActivatePanel(panelId);
              onSelectPanel(panelId);
              onAdjustPanelCouncilRank(panelId, 1);
            }}
          >
            focus
          </button>
          <button type="button" title="raise rank" onClick={() => onAdjustPanelCouncilRank(panelId, 1)}>+rank</button>
          <button type="button" title="lower rank" onClick={() => onAdjustPanelCouncilRank(panelId, -1)}>-rank</button>
          <button type="button" title="pin to tertiary lane" onClick={() => onPinPanelToTertiary(panelId)}>ter</button>
          <button type="button" title="open as tray card" onClick={() => movePanelToTray(panelId)}>tray</button>
          {anchor ? (
            <button type="button" title="guide camera" onDoubleClick={() => guidePanelThroughGlass(panel, anchor)} onClick={() => onFlyCameraToAnchor(anchor)}>
              look
            </button>
          ) : null}
        </div>
      </article>
    );
  };

  return (
    <section className="world-council-root" aria-label="council ranked window manager">
      <header className="world-council-toolbar">
        <div>
          <p className="world-council-kicker">presence operating rail</p>
          <p className="world-council-title">
            one card per presence · drag vertical seams to resize lanes · double click seams to collapse
          </p>
        </div>
        <div className="world-council-toolbar-actions" data-panel-interactive="true">
          <div className="world-council-pane-toggle-group">
            {([1, 2, 3] as const).map((count) => (
              <button
                key={`toolbar-pane-count-${count}`}
                type="button"
                className={`world-council-pane-toggle ${paneCount === count ? "world-council-pane-toggle-active" : ""}`}
                onClick={() => setPaneCountPreference(count)}
                disabled={count > maxPaneCount}
                title={`show ${count} focus pane${count > 1 ? "s" : ""}`}
              >
                {count}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={onToggleEditMode}
            className={`world-council-edit-btn ${isEditMode ? "world-council-edit-btn-on" : ""}`}
          >
            {isEditMode ? "editing rank" : "edit rank"}
          </button>
        </div>
      </header>

      <div className="world-council-shell">
        <div
          className={`world-council-grid ${primaryExpanded ? "world-council-grid-primary-expanded" : ""} ${primarySolo ? "world-council-grid-primary-solo" : ""}`}
          style={viewportWidth >= 1100 ? { gridTemplateColumns } : undefined}
        >
          {renderFocusPane("primary", primaryPanelId)}
          {showSecondaryPane ? (
            <button
              type="button"
              className={`world-council-seam world-council-seam-primary ${seamDrag?.seam === "primary" ? "world-council-seam-active" : ""}`}
              onMouseDown={startPrimarySeamDrag}
              onDoubleClick={() => {
                if (paneCount > 1) {
                  setPaneCountPreference(1);
                  return;
                }
                setPaneCountPreference(maxPaneCount >= 2 ? 2 : 1);
              }}
              title="drag to resize primary split · double click to collapse/expand"
            >
              <span />
            </button>
          ) : null}
          {showSecondaryPane ? renderFocusPane("secondary", secondaryPanelId) : null}
          {showTertiaryPane ? (
            <button
              type="button"
              className={`world-council-seam world-council-seam-tertiary ${seamDrag?.seam === "tertiary" ? "world-council-seam-active" : ""}`}
              onMouseDown={startTertiarySeamDrag}
              onDoubleClick={() => {
                if (paneCount >= 3) {
                  setPaneCountPreference(2);
                  return;
                }
                if (maxPaneCount >= 3) {
                  setPaneCountPreference(3);
                }
              }}
              title="drag to resize tertiary split · double click to collapse/expand"
            >
              <span />
            </button>
          ) : null}
          {showTertiaryPane ? renderFocusPane("tertiary", tertiaryPanelId) : null}
        </div>

        <aside
          className={`world-unity-sidebar ${isPresenceRailCollapsed ? "world-unity-sidebar-collapsed" : ""}`}
          aria-label="presence task tray sidebar"
        >
          <header className="world-unity-sidebar-header">
            <div>
              <p>presence rail</p>
              <p>{iconPanels.length} shown · {sortedPanels.length} total</p>
            </div>
            <button
              type="button"
              className="world-unity-sidebar-toggle"
              onClick={() => setIsPresenceRailCollapsed((prev) => !prev)}
            >
              {isPresenceRailCollapsed ? "expand" : "collapse"}
            </button>
          </header>

          {!isPresenceRailCollapsed ? (
            <div className="world-unity-sidebar-filters" data-panel-interactive="true">
              <label className="world-unity-search-wrap">
                <span>find</span>
                <input
                  value={presenceQuery}
                  onChange={(event) => setPresenceQuery(event.target.value)}
                  className="world-unity-search-input"
                  placeholder="presence, role, panel"
                />
              </label>

              <div className="world-unity-filter-row">
                {([
                  { id: "all", label: "all", count: sortedPanels.length },
                  { id: "running", label: "run", count: panelStatusCounts.running },
                  { id: "paused", label: "pause", count: panelStatusCounts.paused },
                  { id: "blocked", label: "block", count: panelStatusCounts.blocked },
                ] as const).map((item) => (
                  <button
                    key={`status-filter-${item.id}`}
                    type="button"
                    className={`world-unity-filter-btn ${statusFilter === item.id ? "world-unity-filter-btn-active" : ""}`}
                    onClick={() => setStatusFilter(item.id)}
                  >
                    {item.label} {item.count}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="world-unity-sidebar-mini-summary">
              <span>run {panelStatusCounts.running}</span>
              <span>pause {panelStatusCounts.paused}</span>
              <span>block {panelStatusCounts.blocked}</span>
            </div>
          )}

          {!isPresenceRailCollapsed ? (
            <section className="world-smart-pile" aria-label="smart card pile">
              <header className="world-smart-pile-header">
                <p>smart card pile</p>
                <p>{smartPilePanels.length} cards</p>
              </header>
              <div className="world-smart-pile-list">
                {smartPilePreviewPanels.map((panel) => renderSmartPileCard(panel))}
                {smartPilePanels.length <= 0 ? (
                  <p className="world-pinned-empty">all cards are in focus lanes or hidden by active filters.</p>
                ) : null}
                {smartPilePanels.length > smartPilePreviewPanels.length ? (
                  <p className="world-pinned-empty">
                    showing top {smartPilePreviewPanels.length} of {smartPilePanels.length}; narrow filters for a tighter queue.
                  </p>
                ) : null}
              </div>
            </section>
          ) : null}

          <div className={`world-unity-icon-strip ${isPresenceRailCollapsed ? "world-unity-icon-strip-collapsed" : ""}`}>
            {iconPanels.map((panel, index) => {
              const iconMeta = iconRenderMetaByPanelId.get(panel.id);
              const anchor = iconMeta?.anchor ?? null;
              const style = iconMeta?.style ?? ({
                "--unity-scale": "0.900",
              } as CSSProperties);
              const trayOpen = trayPanelId === panel.id;
              const rank = iconMeta?.rank ?? Math.max(1, rankByPanelId.get(panel.id) ?? 1);
              const operationalState = iconMeta?.operationalState ?? "running";
              const operationalReason = iconMeta?.operationalReason ?? "no detail";

              return (
                <div key={`presence-icon-${panel.id}`} className="world-unity-icon-cluster">
                  <button
                    ref={(node) => {
                      const refs = iconButtonByPanelIdRef.current;
                      if (node) {
                        refs.set(panel.id, node);
                        return;
                      }
                      refs.delete(panel.id);
                    }}
                    type="button"
                    className={`world-unity-icon ${trayOpen ? "world-unity-icon-active" : ""} ${isPresenceRailCollapsed ? "world-unity-icon-collapsed" : ""}`}
                    style={style}
                    onMouseEnter={() => scheduleHoveredTrayIconIndex(index)}
                    onMouseLeave={() => scheduleHoveredTrayIconIndex(null)}
                    onFocus={() => scheduleHoveredTrayIconIndex(index)}
                    onBlur={() => scheduleHoveredTrayIconIndex(null)}
                    onClick={() => movePanelToTray(panel.id)}
                    onDoubleClick={() => guidePanelThroughGlass(panel, anchor)}
                    title={`${panel.presenceLabel} · ${operationalState} · rank ${rank} · ${operationalReason}`}
                  >
                    <span className="world-unity-icon-lens">
                      <span className="world-unity-icon-ping" />
                      <span className={`world-unity-icon-state world-unity-icon-state-${operationalState}`} />
                      <span className="world-unity-icon-glyph">{panelGlyph(panel.presenceId, panel.id)}</span>
                    </span>
                    <span className={`world-unity-icon-label ${isPresenceRailCollapsed ? "world-unity-icon-label-collapsed" : ""}`}>
                      {panel.presenceLabel}
                    </span>
                  </button>
                  {trayOpen && !isPresenceRailCollapsed ? (
                    <div className="world-unity-inline-card">
                      {renderTrayCard(panel)}
                    </div>
                  ) : null}
                </div>
              );
            })}
            {iconPanels.length <= 0 ? (
              <div className="world-task-tray-glass">
                <p className="world-glass-title">no presences match filters</p>
                <p className="world-glass-row">
                  {sortedPanels.length <= 0
                    ? "Presence baubles appear here when runtime presences are active."
                    : "Clear query/filter to restore the full rail list."}
                </p>
                {sortedPanels.length > 0 ? (
                  <button
                    type="button"
                    className="world-unity-filter-reset"
                    onClick={() => {
                      setPresenceQuery("");
                      setStatusFilter("all");
                    }}
                  >
                    reset filters
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        </aside>
      </div>

      {iconPanels.length > 0 && viewportWidth < 1100 ? (
        <footer className="world-unity-mobile-dock">
          {iconPanels.slice(0, 12).map((panel) => {
            const anchor = anchorByPanelId.get(panel.id) ?? null;
            const selected = selectedPanelId === panel.id || primaryPanelId === panel.id;
            return (
              <button
                key={`quick-dock-${panel.id}`}
                type="button"
                className={`world-unity-mobile-dock-item ${selected ? "world-unity-mobile-dock-item-selected" : ""}`}
                onClick={() => {
                  onActivatePanel(panel.id);
                  onSelectPanel(panel.id);
                }}
                onDoubleClick={() => {
                  guidePanelThroughGlass(panel, anchor);
                  onAdjustPanelCouncilRank(panel.id, 1);
                }}
                title={`${panel.presenceLabel} · click focus · double click glass + promote`}
              >
                <span className="world-unity-mobile-dock-core">{panelGlyph(panel.presenceId, panel.id)}</span>
                <span className="world-unity-mobile-dock-label">{panel.presenceLabel}</span>
              </button>
            );
          })}
        </footer>
      ) : null}
    </section>
  );
}

export const WorldPanelsViewport = memo(WorldPanelsViewportInner);
