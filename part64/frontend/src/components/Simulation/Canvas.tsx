import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import { createPortal } from "react-dom";
import type {
  SimulationState,
  Catalog,
  BackendFieldParticle,
} from "../../types";
import { runtimeBaseUrl } from "../../runtime/endpoints";
import { GalaxyModelDock } from "./GalaxyModelDock";

interface Props {
  simulation: SimulationState | null;
  catalog: Catalog | null;
  onOverlayInit?: (api: any) => void;
  onNexusInteraction?: (event: NexusInteractionEvent) => void;
  onUserPresenceInput?: (payload: {
    kind: string;
    target: string;
    message?: string;
    xRatio?: number;
    yRatio?: number;
    embedDaimoi?: boolean;
    meta?: Record<string, unknown>;
  }) => void;
  height?: number;
  defaultOverlayView?: OverlayViewId;
  overlayViewLocked?: boolean;
  compactHud?: boolean;
  interactive?: boolean;
  backgroundMode?: boolean;
  particleDensity?: number;
  particleScale?: number;
  motionSpeed?: number;
  mouseInfluence?: number;
  layerDepth?: number;
  graphNodeSmoothness?: number;
  graphNodeStepScale?: number;
  backgroundWash?: number;
  layerVisibility?: OverlayLayerVisibility;
  glassCenterRatio?: { x: number; y: number };
  museWorkspaceBindings?: Record<string, string[]>;
  className?: string;
  // Mouse daimon controls
  mouseDaimonEnabled?: boolean;
  mouseDaimonMessage?: string;
  mouseDaimonMode?: "push" | "pull" | "orbit" | "calm";
  mouseDaimonRadius?: number;
  mouseDaimonStrength?: number;
}

export interface NexusInteractionEvent {
  nodeId: string;
  nodeKind: "file" | "crawler" | "nexus";
  resourceKind: GraphNodeResourceKind;
  label: string;
  xRatio: number;
  yRatio: number;
  openWorldscreen: boolean;
  isDoubleTap?: boolean;
}

interface GraphWorldscreenState {
  nodeId: string;
  commentRef: string;
  url: string;
  imageRef?: string;
  label: string;
  nodeKind: "file" | "crawler" | "nexus";
  nodeTypeText?: string;
  nodeRoleText?: string;
  resourceKind: GraphNodeResourceKind;
  view: GraphWorldscreenView;
  subtitle: string;
  anchorRatioX?: number;
  anchorRatioY?: number;
  remoteFrameUrl?: string;
  encounteredAt?: string;
  sourceUrl?: string;
  domain?: string;
  titleText?: string;
  statusText?: string;
  contentTypeText?: string;
  complianceText?: string;
  discoveredAt?: string;
  fetchedAt?: string;
  summaryText?: string;
  tagsText?: string;
  labelsText?: string;
  projectionGroupId?: string;
  projectionConsolidatedCount?: number;
  projectionMemberManifest?: string[];
}

type GraphNodeResourceKind =
  | "text"
  | "image"
  | "audio"
  | "archive"
  | "blob"
  | "link"
  | "website"
  | "video"
  | "unknown";

type GraphWorldscreenView = "website" | "editor" | "video" | "metadata";
type GraphWorldscreenMode = "overview" | "conversation" | "stats";

interface EditorPreviewState {
  status: "idle" | "loading" | "ready" | "error";
  content: string;
  error: string;
  truncated: boolean;
}

interface WorldscreenPlacement {
  left: number;
  top: number;
  width: number;
  height: number;
  transformOrigin: string;
}

interface GraphNodeTitleOverlay {
  id: string;
  label: string;
  x: number;
  y: number;
  kind: "file" | "crawler" | "presence" | "nexus";
  isTrueGraph?: boolean;
  isProjectionOverflow?: boolean;
}

interface MusicNexusHotspot {
  id: string;
  label: string;
  x: number;
  y: number;
}

interface PresenceAccountEntry {
  presence_id: string;
  display_name: string;
  handle: string;
  avatar: string;
  bio: string;
  tags: string[];
}

interface ImageCommentEntry {
  id: string;
  image_ref: string;
  presence_id: string;
  comment: string;
  metadata: Record<string, unknown>;
  created_at: string;
  time: string;
}

type ComputeJobFilter = "all" | "llm" | "embedding" | "error" | "gpu" | "npu" | "cpu";

interface ComputeJobInsightRow {
  id: string;
  atText: string;
  tsMs: number;
  kind: string;
  op: string;
  backend: string;
  resource: string;
  emitterPresenceId: string;
  targetPresenceId: string;
  model: string;
  status: string;
  latencyMs: number | null;
  error: string;
}

interface ComputeJobInsightSummary {
  total: number;
  llm: number;
  embedding: number;
  ok: number;
  error: number;
  byResource: Record<string, number>;
  byBackend: Record<string, number>;
}

interface UserQueryEdgeRow {
  id: string;
  source: string;
  target: string;
  query: string;
  hits: number;
  life: number;
  strength: number;
}

interface HologramAudioVisualization {
  sourceUrl: string;
  durationSeconds: number;
  waveformBins: Float32Array;
  spectrogramBins: Float32Array[];
}

const COMPUTE_JOB_FILTER_OPTIONS: Array<{ id: ComputeJobFilter; label: string }> = [
  { id: "all", label: "all" },
  { id: "llm", label: "llm" },
  { id: "embedding", label: "embed" },
  { id: "error", label: "error" },
  { id: "gpu", label: "gpu" },
  { id: "npu", label: "npu" },
  { id: "cpu", label: "cpu" },
];

const HOLOGRAM_MODE_OPTIONS: Array<{ id: GraphWorldscreenMode; label: string }> = [
  { id: "overview", label: "overview" },
  { id: "conversation", label: "conversation" },
  { id: "stats", label: "stats" },
];

const EMPTY_COMPUTE_JOB_INSIGHTS = {
  rows: [] as ComputeJobInsightRow[],
  summary: {
    total: 0,
    llm: 0,
    embedding: 0,
    ok: 0,
    error: 0,
    byResource: {},
    byBackend: {},
  } as ComputeJobInsightSummary,
  filtered: [] as ComputeJobInsightRow[],
  gpuAvailability: 0,
  total180s: 0,
};

const EMPTY_RESOURCE_ECONOMY_HUD = {
  packets: 0,
  transfer: 0,
  actionPackets: 0,
  blockedPackets: 0,
  consumedTotal: 0,
  starvedPresences: 0,
};

const EMPTY_PARTICLE_LEGEND_STATS = {
  total: 0,
  primary: {
    chaos: 0,
    nexus: 0,
    smart: 0,
    resource: 0,
    transfer: 0,
    legacy: 0,
  },
  overlays: {
    transfer: 0,
    resource: 0,
  },
};

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  return value as Record<string, unknown>;
}

function normalizePresenceAccountEntries(payload: unknown): PresenceAccountEntry[] {
  const root = asRecord(payload);
  const entriesValue = root?.entries;
  if (!Array.isArray(entriesValue)) {
    return [];
  }
  const entries: PresenceAccountEntry[] = [];
  for (const entry of entriesValue) {
    const row = asRecord(entry);
    if (!row) {
      continue;
    }
    const presenceId = String(row.presence_id ?? "").trim();
    if (!presenceId) {
      continue;
    }
    const tagsValue = Array.isArray(row.tags) ? row.tags : [];
    const tags = tagsValue
      .map((item) => String(item ?? "").trim())
      .filter((item) => item.length > 0);
    entries.push({
      presence_id: presenceId,
      display_name: String(row.display_name ?? presenceId).trim() || presenceId,
      handle: String(row.handle ?? presenceId).trim() || presenceId,
      avatar: String(row.avatar ?? "").trim(),
      bio: String(row.bio ?? "").trim(),
      tags,
    });
  }
  return entries;
}

function normalizeImageCommentEntries(payload: unknown): ImageCommentEntry[] {
  const root = asRecord(payload);
  const entriesValue = root?.entries;
  if (!Array.isArray(entriesValue)) {
    return [];
  }
  const entries: ImageCommentEntry[] = [];
  for (const entry of entriesValue) {
    const row = asRecord(entry);
    if (!row) {
      continue;
    }
    const id = String(row.id ?? "").trim();
    const imageRef = String(row.image_ref ?? "").trim();
    const presenceId = String(row.presence_id ?? "").trim();
    const comment = String(row.comment ?? "").trim();
    if (!id || !imageRef || !presenceId || !comment) {
      continue;
    }
    entries.push({
      id,
      image_ref: imageRef,
      presence_id: presenceId,
      comment,
      metadata: asRecord(row.metadata) ?? {},
      created_at: String(row.created_at ?? row.time ?? "").trim(),
      time: String(row.time ?? row.created_at ?? "").trim(),
    });
  }
  return entries;
}

function worldscreenImageRef(worldscreen: GraphWorldscreenState): string {
  const preferred = String(worldscreen.imageRef ?? "").trim();
  if (preferred) {
    return preferred;
  }
  const source = String(worldscreen.sourceUrl ?? "").trim();
  if (source) {
    return source;
  }
  return String(worldscreen.url ?? "").trim();
}

function worldscreenCommentRef(worldscreen: GraphWorldscreenState): string {
  const explicit = String(worldscreen.commentRef ?? "").trim();
  if (explicit) {
    return explicit;
  }
  const imageRef = worldscreenImageRef(worldscreen);
  if (imageRef) {
    return imageRef;
  }
  const nodeId = String(worldscreen.nodeId ?? "").trim();
  if (nodeId) {
    return nodeId;
  }
  return String(worldscreen.url ?? "").trim();
}

function nexusCommentRefForNode(
  node: any,
  nodeKind: "file" | "crawler" | "nexus",
  worldscreenUrl: string,
): string {
  const candidates = [
    node?.comment_ref,
    node?.commentRef,
    node?.source_rel_path,
    node?.archive_rel_path,
    node?.archived_rel_path,
    node?.archive_member_path,
    node?.source_url,
    node?.url,
    node?.domain,
    node?.id,
    worldscreenUrl,
  ];

  for (const candidate of candidates) {
    const value = String(candidate ?? "").trim();
    if (!value) {
      continue;
    }
    if (nodeKind === "crawler" && value && !value.startsWith("crawler:")) {
      return `crawler:${value}`;
    }
    if (nodeKind === "file" && value && !value.startsWith("file:")) {
      return `file:${value}`;
    }
    if (nodeKind === "nexus" && value && !value.startsWith("nexus:")) {
      return `nexus:${value}`;
    }
    return value;
  }

  const fallbackId = String(node?.id ?? "").trim();
  if (fallbackId) {
    return `${nodeKind}:${fallbackId}`;
  }
  return "";
}

function extractHttpUrls(text: string): string[] {
  const matches = text.match(/https?:\/\/[^\s<>'")]+/gi) ?? [];
  return matches.filter((value, index, rows) => rows.indexOf(value) === index);
}

function isHttpUrlText(text: string): boolean {
  const value = text.trim();
  return /^https?:\/\//i.test(value);
}

function imageCommentParentId(entry: ImageCommentEntry): string {
  const metadata = asRecord(entry.metadata);
  return String(metadata?.parent_comment_id ?? "").trim();
}

interface FlattenedCommentEntry {
  entry: ImageCommentEntry;
  depth: number;
}

function flattenImageCommentThread(entries: ImageCommentEntry[]): FlattenedCommentEntry[] {
  type TreeNode = {
    entry: ImageCommentEntry;
    children: TreeNode[];
  };

  const rankById = new Map<string, number>();
  const sorted = [...entries].sort((left, right) => {
    const leftTs = Date.parse(left.created_at || left.time || "") || 0;
    const rightTs = Date.parse(right.created_at || right.time || "") || 0;
    if (leftTs !== rightTs) {
      return leftTs - rightTs;
    }
    return left.id.localeCompare(right.id);
  });
  sorted.forEach((entry, index) => {
    rankById.set(entry.id, index);
  });

  const nodeById = new Map<string, TreeNode>();
  for (const entry of sorted) {
    nodeById.set(entry.id, {
      entry,
      children: [],
    });
  }

  const roots: TreeNode[] = [];
  for (const entry of sorted) {
    const node = nodeById.get(entry.id);
    if (!node) {
      continue;
    }
    const parentId = imageCommentParentId(entry);
    if (!parentId || parentId === entry.id) {
      roots.push(node);
      continue;
    }
    const parentNode = nodeById.get(parentId);
    if (!parentNode) {
      roots.push(node);
      continue;
    }
    parentNode.children.push(node);
  }

  const orderNodes = (rows: TreeNode[]) => {
    rows.sort((left, right) => {
      const leftRank = rankById.get(left.entry.id) ?? Number.MAX_SAFE_INTEGER;
      const rightRank = rankById.get(right.entry.id) ?? Number.MAX_SAFE_INTEGER;
      return leftRank - rightRank;
    });
    for (const row of rows) {
      if (row.children.length > 1) {
        orderNodes(row.children);
      }
    }
  };

  orderNodes(roots);

  const flattened: FlattenedCommentEntry[] = [];
  const walk = (rows: TreeNode[], depth: number) => {
    for (const row of rows) {
      flattened.push({
        entry: row.entry,
        depth,
      });
      if (row.children.length > 0) {
        walk(row.children, depth + 1);
      }
    }
  };
  walk(roots, 0);
  return flattened;
}

function errorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error) {
    const message = error.message.trim();
    if (message) {
      return message;
    }
  }
  return fallback;
}

function bufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return window.btoa(binary);
}

async function fetchImagePayloadAsBase64(
  worldscreen: GraphWorldscreenState,
): Promise<{ base64: string; mime: string }> {
  const candidates = [String(worldscreen.url ?? "").trim(), String(worldscreen.remoteFrameUrl ?? "").trim()]
    .filter((value, index, rows) => value.length > 0 && rows.indexOf(value) === index);

  for (const candidate of candidates) {
    try {
      const response = await fetch(candidate, {
        method: "GET",
        credentials: "same-origin",
      });
      if (!response.ok) {
        continue;
      }
      const mime = String(response.headers.get("content-type") ?? "").trim().toLowerCase();
      if (mime && !mime.startsWith("image/")) {
        continue;
      }
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength <= 0) {
        continue;
      }
      return {
        base64: bufferToBase64(buffer),
        mime: mime || "image/png",
      };
    } catch {
      continue;
    }
  }

  throw new Error("unable to read image bytes for commentary");
}

function normalizePresenceKey(raw: string): string {
  return raw.trim().toLowerCase().replace(/[\s./:-]+/g, "_");
}

function canonicalPresenceId(raw: string): string {
  let value = String(raw || "").trim();
  if (!value) {
    return "";
  }
  if (value.startsWith("nexus:field:")) {
    value = value.slice("nexus:field:".length).trim();
  }
  if (value.startsWith("field:")) {
    return value.slice("field:".length).trim();
  }
  if (value.startsWith("presence:concept:")) {
    return value.slice("presence:concept:".length).trim();
  }
  if (value.startsWith("presence:")) {
    return value.slice("presence:".length).trim();
  }
  return value;
}

function stablePresenceRatio(seed: string, salt: number): number {
  const key = `${seed}|${salt}`;
  let hash = 2166136261;
  for (let index = 0; index < key.length; index += 1) {
    hash ^= key.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function presenceHueFromId(rawPresenceId: string): number {
  const presenceId = canonicalPresenceId(rawPresenceId);
  if (!presenceId) {
    return 198;
  }
  return Math.round(stablePresenceRatio(presenceId, 7) * 360) % 360;
}

function normalizePresenceAnchors(forms: Array<any>): Array<any> {
  if (forms.length <= 0) {
    return [];
  }

  const seen = new Set<string>();
  const positioned = forms
    .map((row, index) => {
      const presenceId = canonicalPresenceId(String(row?.id ?? ""));
      const fallbackAngle = stablePresenceRatio(presenceId || `presence-${index}`, 3) * Math.PI * 2;
      const fallbackRadius = 0.24 + (stablePresenceRatio(presenceId || `presence-${index}`, 11) * 0.4);
      const fallbackX = 0.5 + Math.cos(fallbackAngle) * fallbackRadius;
      const fallbackY = 0.5 + Math.sin(fallbackAngle) * fallbackRadius;
      const id = presenceId || String(row?.id ?? `presence-${index}`);
      if (!id || seen.has(id)) {
        return null;
      }
      seen.add(id);
      return {
        ...row,
        id,
        x: clamp01(Number(row?.x ?? fallbackX)),
        y: clamp01(Number(row?.y ?? fallbackY)),
        hue: Number.isFinite(Number(row?.hue))
          ? Number(row?.hue)
          : presenceHueFromId(presenceId || String(row?.id ?? "")),
      };
    })
    .filter((row): row is any => row !== null);

  const minX = 0.06;
  const maxX = 0.94;
  const minY = 0.08;
  const maxY = 0.92;

  return positioned.map((row) => {
    return {
      ...row,
      x: Math.min(maxX, Math.max(minX, Number(row.x ?? 0.5))),
      y: Math.min(maxY, Math.max(minY, Number(row.y ?? 0.5))),
    };
  });
}

function normalizeGraphNodePositionMap(payload: unknown): Map<string, { x: number; y: number }> {
  const map = new Map<string, { x: number; y: number }>();
  if (!payload) {
    return map;
  }

  const pushRow = (nodeIdRaw: unknown, rowValue: unknown) => {
    const nodeId = String(nodeIdRaw ?? "").trim();
    const row = asRecord(rowValue);
    if (!nodeId || !row) {
      return;
    }
    const x = Number(row.x);
    const y = Number(row.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return;
    }
    map.set(nodeId, { x: clamp01(x), y: clamp01(y) });
  };

  if (Array.isArray(payload)) {
    payload.forEach((entry) => {
      const row = asRecord(entry);
      if (!row) {
        return;
      }
      pushRow(row.id, row);
    });
    return map;
  }

  const root = asRecord(payload);
  if (!root) {
    return map;
  }
  Object.entries(root).forEach(([nodeId, rowValue]) => {
    pushRow(nodeId, rowValue);
  });
  return map;
}

function normalizePresenceAnchorPositionMap(payload: unknown): Map<string, { x: number; y: number }> {
  const map = new Map<string, { x: number; y: number }>();
  if (!payload) {
    return map;
  }

  const root = asRecord(payload);
  if (!root) {
    return map;
  }
  Object.entries(root).forEach(([presenceId, rowValue]) => {
    const row = asRecord(rowValue);
    if (!row) {
      return;
    }
    const x = Number(row.x);
    const y = Number(row.y);
    if (!presenceId || !Number.isFinite(x) || !Number.isFinite(y)) {
      return;
    }
    map.set(presenceId, { x: clamp01(x), y: clamp01(y) });
  });
  return map;
}

function shortPresenceIdLabel(raw: string): string {
  const value = raw.trim();
  if (!value) {
    return "presence";
  }
  const tail = value.includes(".") ? value.split(".").slice(-1)[0] : value;
  return tail.length > 18 ? `${tail.slice(0, 17)}~` : tail;
}

function normalizeWorkspaceBindingMap(
  raw: Record<string, string[]> | null | undefined,
): Record<string, string[]> {
  if (!raw || typeof raw !== "object") {
    return {};
  }
  const normalized: Record<string, string[]> = {};
  Object.entries(raw).forEach(([presenceId, nodeIds]) => {
    const key = normalizePresenceKey(String(presenceId || ""));
    if (!key || !Array.isArray(nodeIds)) {
      return;
    }
    const normalizedIds = nodeIds
      .map((item) => String(item || "").trim())
      .filter((item, index, all) => item.length > 0 && all.indexOf(item) === index)
      .slice(0, 36);
    if (normalizedIds.length > 0) {
      normalized[key] = normalizedIds;
    }
  });
  return normalized;
}

function normalizeComputeJobInsightRows(payload: unknown): ComputeJobInsightRow[] {
  if (!Array.isArray(payload)) {
    return [];
  }
  const rows: ComputeJobInsightRow[] = [];
  for (const item of payload) {
    const row = asRecord(item);
    if (!row) {
      continue;
    }
    const id = String(row.id ?? "").trim();
    if (!id) {
      continue;
    }

    const atText = String(row.at ?? "").trim();
    const tsNumber = Number(row.ts ?? 0);
    const tsMs = Number.isFinite(tsNumber) && tsNumber > 0
      ? (tsNumber > 10_000_000_000 ? tsNumber : tsNumber * 1000)
      : (atText ? Date.parse(atText) : 0);
    const backend = String(row.backend ?? "").trim().toLowerCase();
    const resourceRaw = String(row.resource ?? "").trim().toLowerCase();
    const inferredResource = resourceRaw
      || (backend.includes("npu")
        ? "npu"
        : backend.includes("gpu") || backend.includes("cuda") || backend.includes("torch") || backend.includes("vllm") || backend.includes("ollama")
          ? "gpu"
          : "cpu");
    const latencyRaw = row.latency_ms;
    let latencyMs: number | null = null;
    if (latencyRaw !== undefined && latencyRaw !== null) {
      const numeric = Number(latencyRaw);
      if (Number.isFinite(numeric)) {
        latencyMs = Math.max(0, numeric);
      }
    }

    rows.push({
      id,
      atText,
      tsMs: Number.isFinite(tsMs) && tsMs > 0 ? tsMs : 0,
      kind: String(row.kind ?? "").trim().toLowerCase() || "unknown",
      op: String(row.op ?? "").trim().toLowerCase(),
      backend,
      resource: inferredResource,
      emitterPresenceId: String(row.emitter_presence_id ?? "").trim(),
      targetPresenceId: String(row.target_presence_id ?? "").trim(),
      model: String(row.model ?? "").trim(),
      status: String(row.status ?? "").trim().toLowerCase() || "unknown",
      latencyMs,
      error: String(row.error ?? "").trim(),
    });
  }
  rows.sort((a, b) => b.tsMs - a.tsMs);
  return rows;
}

function normalizeUserQueryEdgeRows(payload: unknown): UserQueryEdgeRow[] {
  if (!Array.isArray(payload)) {
    return [];
  }
  const rows: UserQueryEdgeRow[] = [];
  for (let index = 0; index < payload.length; index += 1) {
    const row = asRecord(payload[index]);
    if (!row) {
      continue;
    }
    const id = String(row.id ?? `query-edge:${index}`).trim() || `query-edge:${index}`;
    const source = String(row.source ?? "").trim();
    const target = String(row.target ?? "").trim();
    if (!source || !target) {
      continue;
    }
    const hits = Math.max(1, Number(row.hits ?? 1));
    const lifeRaw = Number(row.life ?? 1);
    const strengthRaw = Number(row.strength ?? 0);
    rows.push({
      id,
      source,
      target,
      query: String(row.query ?? "").trim(),
      hits: Number.isFinite(hits) ? Math.max(1, Math.round(hits)) : 1,
      life: Number.isFinite(lifeRaw) ? clamp01(lifeRaw) : 1,
      strength: Number.isFinite(strengthRaw) ? clamp01(strengthRaw) : 0,
    });
  }
  return rows.slice(0, 96);
}

function summarizeComputeJobs(rows: ComputeJobInsightRow[]): ComputeJobInsightSummary {
  const summary: ComputeJobInsightSummary = {
    total: rows.length,
    llm: 0,
    embedding: 0,
    ok: 0,
    error: 0,
    byResource: {},
    byBackend: {},
  };
  for (const row of rows) {
    const kind = row.kind;
    if (kind === "llm") {
      summary.llm += 1;
    }
    if (kind.includes("embed")) {
      summary.embedding += 1;
    }
    if (row.status === "ok" || row.status === "success" || row.status === "cached") {
      summary.ok += 1;
    }
    if (row.status === "error" || row.status === "failed" || row.status === "timeout") {
      summary.error += 1;
    }
    const resourceKey = row.resource || "unknown";
    summary.byResource[resourceKey] = (summary.byResource[resourceKey] ?? 0) + 1;
    const backendKey = row.backend || "unknown";
    summary.byBackend[backendKey] = (summary.byBackend[backendKey] ?? 0) + 1;
  }
  return summary;
}

function computeJobAgeLabel(tsMs: number): string {
  if (!Number.isFinite(tsMs) || tsMs <= 0) {
    return "now";
  }
  const deltaMs = Math.max(0, Date.now() - tsMs);
  if (deltaMs < 1500) {
    return "now";
  }
  const seconds = Math.round(deltaMs / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.round(minutes / 60);
  return `${hours}h`;
}

function formatPlaybackClock(secondsInput: number): string {
  const seconds = Number.isFinite(secondsInput) ? Math.max(0, secondsInput) : 0;
  const whole = Math.floor(seconds + 0.0001);
  const minuteTotal = Math.floor(whole / 60);
  const sec = whole % 60;
  const hour = Math.floor(minuteTotal / 60);
  const min = minuteTotal % 60;
  if (hour > 0) {
    return `${hour}:${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  }
  return `${minuteTotal}:${String(sec).padStart(2, "0")}`;
}

function resolveWorldscreenMediaUrl(rawUrl: string): string {
  const value = String(rawUrl || "").trim();
  if (!value) {
    return "";
  }
  if (value.startsWith("http://") || value.startsWith("https://")) {
    return value;
  }
  const normalizedPath = value.startsWith("/") ? value : `/${value}`;
  const base = runtimeBaseUrl();
  return base ? `${base}${normalizedPath}` : normalizedPath;
}

function buildHannWindow(windowSize: number): Float32Array {
  const safeSize = Math.max(8, Math.floor(windowSize));
  const window = new Float32Array(safeSize);
  for (let index = 0; index < safeSize; index += 1) {
    window[index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / Math.max(1, safeSize - 1)));
  }
  return window;
}

function fftMagnitudes(windowedSamples: Float32Array): Float32Array {
  const sampleCount = windowedSamples.length;
  const real = new Float32Array(sampleCount);
  const imag = new Float32Array(sampleCount);
  real.set(windowedSamples);

  let swapIndex = 0;
  for (let index = 0; index < sampleCount; index += 1) {
    if (index < swapIndex) {
      const realTmp = real[index];
      const imagTmp = imag[index];
      real[index] = real[swapIndex];
      imag[index] = imag[swapIndex];
      real[swapIndex] = realTmp;
      imag[swapIndex] = imagTmp;
    }
    let bitMask = sampleCount >> 1;
    while (bitMask >= 1 && swapIndex >= bitMask) {
      swapIndex -= bitMask;
      bitMask >>= 1;
    }
    swapIndex += bitMask;
  }

  for (let step = 2; step <= sampleCount; step <<= 1) {
    const half = step >> 1;
    const stepAngle = (-2 * Math.PI) / step;
    const stepCos = Math.cos(stepAngle);
    const stepSin = Math.sin(stepAngle);
    for (let base = 0; base < sampleCount; base += step) {
      let twiddleReal = 1;
      let twiddleImag = 0;
      for (let offset = 0; offset < half; offset += 1) {
        const even = base + offset;
        const odd = even + half;
        const oddReal = real[odd];
        const oddImag = imag[odd];
        const tempReal = (twiddleReal * oddReal) - (twiddleImag * oddImag);
        const tempImag = (twiddleReal * oddImag) + (twiddleImag * oddReal);

        real[odd] = real[even] - tempReal;
        imag[odd] = imag[even] - tempImag;
        real[even] += tempReal;
        imag[even] += tempImag;

        const nextReal = (twiddleReal * stepCos) - (twiddleImag * stepSin);
        twiddleImag = (twiddleReal * stepSin) + (twiddleImag * stepCos);
        twiddleReal = nextReal;
      }
    }
  }

  const halfCount = sampleCount >> 1;
  const magnitudes = new Float32Array(halfCount);
  for (let index = 0; index < halfCount; index += 1) {
    const realValue = real[index];
    const imagValue = imag[index];
    magnitudes[index] = Math.sqrt((realValue * realValue) + (imagValue * imagValue));
  }
  return magnitudes;
}

function buildHologramAudioVisualization(
  audioBuffer: AudioBuffer,
  sourceUrl: string,
): HologramAudioVisualization {
  const sampleLength = Math.max(1, audioBuffer.length);
  const channelCount = Math.max(1, audioBuffer.numberOfChannels);
  const merged = new Float32Array(sampleLength);
  for (let channel = 0; channel < channelCount; channel += 1) {
    const channelData = audioBuffer.getChannelData(channel);
    for (let index = 0; index < sampleLength; index += 1) {
      merged[index] += channelData[index] / channelCount;
    }
  }

  const waveformBinCount = Math.max(900, Math.min(2400, Math.floor(sampleLength / 200)));
  const waveformBins = new Float32Array(waveformBinCount);
  const samplesPerWaveBin = Math.max(1, Math.floor(sampleLength / waveformBinCount));
  for (let bin = 0; bin < waveformBinCount; bin += 1) {
    const from = bin * samplesPerWaveBin;
    const to = Math.min(sampleLength, from + samplesPerWaveBin);
    let peak = 0;
    for (let cursor = from; cursor < to; cursor += 1) {
      const value = Math.abs(merged[cursor]);
      if (value > peak) {
        peak = value;
      }
    }
    waveformBins[bin] = peak;
  }

  const fftSize = 1024;
  const hopSize = 256;
  const bandCount = 96;
  const maxFrameCount = 360;
  const baseFrameCount = Math.max(1, Math.floor((sampleLength - fftSize) / hopSize) + 1);
  const frameStride = Math.max(1, Math.floor(baseFrameCount / maxFrameCount));
  const window = buildHannWindow(fftSize);
  const spectrogramBins: Float32Array[] = [];
  let maxBandValue = 1e-9;

  for (let frame = 0; frame < baseFrameCount; frame += frameStride) {
    const start = frame * hopSize;
    const windowed = new Float32Array(fftSize);
    for (let index = 0; index < fftSize; index += 1) {
      const sample = start + index < sampleLength ? merged[start + index] : 0;
      windowed[index] = sample * window[index];
    }
    const magnitudes = fftMagnitudes(windowed);
    const bands = new Float32Array(bandCount);
    for (let band = 0; band < bandCount; band += 1) {
      const startNorm = Math.pow(band / bandCount, 2.15);
      const endNorm = Math.pow((band + 1) / bandCount, 2.15);
      const low = Math.max(0, Math.floor(startNorm * (magnitudes.length - 1)));
      const high = Math.max(low + 1, Math.floor(endNorm * (magnitudes.length - 1)));
      let sum = 0;
      let count = 0;
      for (let magIndex = low; magIndex <= high && magIndex < magnitudes.length; magIndex += 1) {
        sum += magnitudes[magIndex];
        count += 1;
      }
      const energy = Math.log1p(sum / Math.max(1, count));
      bands[band] = energy;
      if (energy > maxBandValue) {
        maxBandValue = energy;
      }
    }
    spectrogramBins.push(bands);
  }

  const normalizeBy = Math.max(1e-6, maxBandValue);
  for (const bands of spectrogramBins) {
    for (let index = 0; index < bands.length; index += 1) {
      bands[index] = clamp01(bands[index] / normalizeBy);
    }
  }

  return {
    sourceUrl,
    durationSeconds: Math.max(0, audioBuffer.duration),
    waveformBins,
    spectrogramBins,
  };
}

function syncCanvasToDisplaySize(canvas: HTMLCanvasElement): { width: number; height: number; scale: number } {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return {
    width,
    height,
    scale: dpr,
  };
}

function drawHologramAudioBaseCanvas(
  canvas: HTMLCanvasElement,
  visualization: HologramAudioVisualization,
): void {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  const { width, height } = syncCanvasToDisplaySize(canvas);
  context.clearRect(0, 0, width, height);

  const topHeight = Math.max(40, Math.floor(height * 0.66));
  const waveHeight = Math.max(22, height - topHeight);
  const spectrogramFrames = visualization.spectrogramBins;
  const spectrogramImage = context.createImageData(width, topHeight);
  const pixelData = spectrogramImage.data;
  const frameCount = Math.max(1, spectrogramFrames.length);
  const bandCount = Math.max(1, spectrogramFrames[0]?.length ?? 0);

  for (let x = 0; x < width; x += 1) {
    const frameIndex = Math.min(frameCount - 1, Math.floor((x / Math.max(1, width - 1)) * (frameCount - 1)));
    const frameBins = spectrogramFrames[frameIndex] ?? spectrogramFrames[0];
    for (let y = 0; y < topHeight; y += 1) {
      const bandIndex = Math.min(
        bandCount - 1,
        Math.floor(((topHeight - 1 - y) / Math.max(1, topHeight - 1)) * (bandCount - 1)),
      );
      const energy = clamp01(Number(frameBins?.[bandIndex] ?? 0));
      const luminance = Math.pow(energy, 0.8);
      const r = Math.round(14 + (luminance * 235));
      const g = Math.round(30 + (Math.pow(luminance, 0.9) * 196));
      const b = Math.round(72 + (Math.pow(1 - luminance, 1.2) * 140));
      const pixelOffset = ((y * width) + x) * 4;
      pixelData[pixelOffset] = r;
      pixelData[pixelOffset + 1] = g;
      pixelData[pixelOffset + 2] = b;
      pixelData[pixelOffset + 3] = 255;
    }
  }

  context.putImageData(spectrogramImage, 0, 0);

  const waveTop = topHeight;
  context.fillStyle = "rgba(6, 16, 28, 0.96)";
  context.fillRect(0, waveTop, width, waveHeight);
  context.strokeStyle = "rgba(120, 188, 220, 0.42)";
  context.lineWidth = Math.max(1, width / 1100);
  context.beginPath();
  context.moveTo(0, waveTop + (waveHeight / 2));
  context.lineTo(width, waveTop + (waveHeight / 2));
  context.stroke();

  const waveBins = visualization.waveformBins;
  const waveBinCount = Math.max(1, waveBins.length);
  context.strokeStyle = "rgba(176, 236, 255, 0.92)";
  context.lineWidth = Math.max(1.2, width / 820);
  context.beginPath();
  for (let x = 0; x < width; x += 1) {
    const sampleIndex = Math.min(waveBinCount - 1, Math.floor((x / Math.max(1, width - 1)) * (waveBinCount - 1)));
    const amplitude = clamp01(Number(waveBins[sampleIndex] ?? 0));
    const y = waveTop + (waveHeight / 2) - (amplitude * waveHeight * 0.44);
    if (x === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }
  context.stroke();
}

function drawHologramAudioPlayhead(
  canvas: HTMLCanvasElement,
  ratioInput: number,
): void {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }
  const { width, height } = syncCanvasToDisplaySize(canvas);
  const ratio = clamp01(ratioInput);
  const x = ratio * width;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(255, 242, 204, 0.94)";
  context.lineWidth = Math.max(1.2, width / 1100);
  context.shadowColor = "rgba(255, 227, 160, 0.76)";
  context.shadowBlur = Math.max(4, width / 250);
  context.beginPath();
  context.moveTo(x, 0);
  context.lineTo(x, height);
  context.stroke();
  context.shadowBlur = 0;

  const markerRadius = Math.max(3.5, width / 220);
  context.fillStyle = "rgba(255, 236, 188, 0.94)";
  context.beginPath();
  context.arc(x, markerRadius + 2, markerRadius, 0, Math.PI * 2);
  context.fill();
}

export type OverlayViewId =
  | "omni"
  | "presence"
  | "file-impact"
  | "file-graph"
  | "true-graph"
  | "crawler-graph"
  | "truth-gate"
  | "logic"
  | "pain-field";

type OverlayLayerVisibility = Partial<Record<Exclude<OverlayViewId, "omni">, boolean>>;

interface OverlayViewOption {
  id: OverlayViewId;
  label: string;
  description: string;
}

export const OVERLAY_VIEW_OPTIONS: OverlayViewOption[] = [
  {
    id: "omni",
    label: "Omni",
    description: "All world overlays layered together.",
  },
  {
    id: "presence",
    label: "Presence",
    description: "Named-form field activity and live currents.",
  },
  {
    id: "file-impact",
    label: "File Impact",
    description: "Recent file drift and impacted presences.",
  },
  {
    id: "file-graph",
    label: "Nexus Graph",
    description: "Unified nexus topology from file and crawler sources.",
  },
  {
    id: "true-graph",
    label: "TrueGraph",
    description: "Static lineage graph from the uncompressed source ledger.",
  },
  {
    id: "truth-gate",
    label: "Truth Gate",
    description: "Gate readiness, claim status, and proof ring.",
  },
  {
    id: "logic",
    label: "Logic",
    description: "Logical graph joins, derivations, and blocks.",
  },
  {
    id: "pain-field",
    label: "Pain Field",
    description: "Failing tests and failure heat contours.",
  },
];

const VIDEO_EXTENSIONS = new Set(["mp4", "m4v", "mov", "webm", "mkv", "avi"]);
const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg", "avif", "heic"]);
const AUDIO_EXTENSIONS = new Set(["mp3", "wav", "ogg", "m4a", "flac", "aac", "opus"]);
const ARCHIVE_EXTENSIONS = new Set(["zip", "tar", "tgz", "gz", "bz2", "xz", "7z", "rar", "zst"]);
const TEXT_FILE_BASENAMES = new Set([
  "readme",
  "license",
  "makefile",
  "dockerfile",
  "procfile",
]);
const EDITOR_EXTENSIONS = new Set([
  "md",
  "mdx",
  "txt",
  "json",
  "jsonl",
  "ndjson",
  "csv",
  "tsv",
  "yaml",
  "yml",
  "toml",
  "ini",
  "cfg",
  "conf",
  "log",
  "lock",
  "sha1",
  "sha256",
  "lisp",
  "sexp",
  "js",
  "jsx",
  "ts",
  "tsx",
  "py",
  "html",
  "css",
  "c",
  "cc",
  "cpp",
  "h",
  "hpp",
  "go",
  "rs",
  "java",
  "sh",
  "bash",
  "zsh",
  "fish",
  "env",
  "gitignore",
  "gitattributes",
  "editorconfig",
  "npmrc",
  "pnpmfile",
  "sql",
  "proto",
  "graphql",
]);

function extensionFromPathLike(pathLike: string): string {
  const strippedQuery = pathLike.split("?")[0]?.split("#")[0] ?? "";
  const leaf = strippedQuery.split("/").pop() ?? "";
  const dot = leaf.lastIndexOf(".");
  if (dot < 0 || dot === leaf.length - 1) {
    return "";
  }
  return leaf.slice(dot + 1).toLowerCase();
}

function leafNameFromPathLike(pathLike: string): string {
  const strippedQuery = pathLike.split("?")[0]?.split("#")[0] ?? "";
  return (strippedQuery.split("/").pop() ?? "").trim().toLowerCase();
}

function sourcePathFromNode(node: any): string {
  return String(
    node?.source_rel_path
    || node?.archived_rel_path
    || node?.title
    || node?.domain
    || node?.url
    || node?.name
    || node?.label
    || node?.id
    || "",
  );
}

function pathLikeHasMp3Extension(pathLike: unknown): boolean {
  const pathText = String(pathLike ?? "").trim();
  if (!pathText) {
    return false;
  }
  if (extensionFromPathLike(pathText) === "mp3") {
    return true;
  }
  const query = pathText.split("?")[1] ?? "";
  if (!query) {
    return false;
  }
  const member = new URLSearchParams(query.split("#")[0] ?? "").get("member");
  if (!member) {
    return false;
  }
  return extensionFromPathLike(decodeURIComponent(member)) === "mp3";
}

function isMp3ContentType(value: unknown): boolean {
  const contentType = String(value ?? "").trim().toLowerCase();
  if (!contentType) {
    return false;
  }
  return (
    contentType.includes("audio/mpeg")
    || contentType.includes("audio/mp3")
    || contentType.includes("mpeg3")
  );
}

function classifyFileResourceKind(node: any): GraphNodeResourceKind {
  const kind = String(node?.kind ?? "").trim().toLowerCase();
  const sourcePath = sourcePathFromNode(node);
  const ext = extensionFromPathLike(sourcePath);
  const leafName = leafNameFromPathLike(sourcePath);
  const hasTextExcerpt = String(node?.text_excerpt ?? "").trim().length > 0;
  if (kind === "image" || IMAGE_EXTENSIONS.has(ext)) {
    return "image";
  }
  if (kind === "audio" || AUDIO_EXTENSIONS.has(ext)) {
    return "audio";
  }
  if (kind === "video" || VIDEO_EXTENSIONS.has(ext)) {
    return "video";
  }
  if (ARCHIVE_EXTENSIONS.has(ext)) {
    return "archive";
  }
  if (
    kind === "text"
    || EDITOR_EXTENSIONS.has(ext)
    || TEXT_FILE_BASENAMES.has(leafName)
    || hasTextExcerpt
  ) {
    return "text";
  }
  return "blob";
}

function classifyCrawlerResourceKind(node: any): GraphNodeResourceKind {
  const crawlerKind = String(node?.crawler_kind ?? "url").trim().toLowerCase();
  const contentType = String(node?.content_type ?? "").trim().toLowerCase();
  const ext = extensionFromPathLike(String(node?.url ?? ""));
  if (contentType.startsWith("image/") || IMAGE_EXTENSIONS.has(ext)) {
    return "image";
  }
  if (contentType.startsWith("audio/") || AUDIO_EXTENSIONS.has(ext)) {
    return "audio";
  }
  if (contentType.startsWith("video/") || VIDEO_EXTENSIONS.has(ext)) {
    return "video";
  }
  if (contentType.includes("zip") || ARCHIVE_EXTENSIONS.has(ext)) {
    return "archive";
  }
  if (crawlerKind === "domain") {
    return "website";
  }
  if (crawlerKind === "content") {
    return "website";
  }
  if (crawlerKind === "url") {
    return "link";
  }
  return "unknown";
}

function normalizeResourceKind(value: unknown): GraphNodeResourceKind | null {
  const normalized = String(value ?? "").trim().toLowerCase();
  switch (normalized) {
    case "text":
    case "image":
    case "audio":
    case "archive":
    case "blob":
    case "link":
    case "website":
    case "video":
    case "unknown":
      return normalized;
    default:
      return null;
  }
}

function resourceKindFromModality(value: unknown): GraphNodeResourceKind | null {
  const normalized = String(value ?? "").trim().toLowerCase();
  if (normalized === "web") {
    return "website";
  }
  if (normalized === "binary") {
    return "blob";
  }
  if (normalized === "archive") {
    return "archive";
  }
  return normalizeResourceKind(normalized);
}

function resourceKindForNode(node: any): GraphNodeResourceKind {
  const explicitResourceKind = normalizeResourceKind(node?.resource_kind);
  if (explicitResourceKind) {
    return explicitResourceKind;
  }
  const modalityResourceKind = resourceKindFromModality(node?.modality);
  if (modalityResourceKind) {
    return modalityResourceKind;
  }
  const nodeKind = String(node?.node_type ?? "file");
  if (nodeKind === "crawler") {
    return classifyCrawlerResourceKind(node);
  }
  return classifyFileResourceKind(node);
}

function isMusicNexusNode(node: any, resourceKind: GraphNodeResourceKind): boolean {
  const nodeType = String(node?.node_type ?? "").trim().toLowerCase();
  const kind = String(node?.kind ?? "").trim().toLowerCase();
  const audioClassified = (
    resourceKind === "audio"
    || nodeType === "audio"
    || kind === "audio"
    || kind === "music"
    || kind === "song"
    || kind === "midi"
  );
  if (!audioClassified) {
    return false;
  }

  const pathCandidates = [
    node?.archive_member_path,
    node?.archived_member_path,
    node?.member_path,
    node?.member_rel_path,
    node?.source_rel_path,
    node?.archive_rel_path,
    node?.archived_rel_path,
    node?.url,
    node?.source_url,
    node?.title,
    node?.name,
    node?.label,
    node?.id,
  ];
  for (const candidate of pathCandidates) {
    if (pathLikeHasMp3Extension(candidate)) {
      return true;
    }
  }

  return (
    isMp3ContentType(node?.content_type)
    || isMp3ContentType(node?.mime)
    || isMp3ContentType(node?.mime_type)
    || isMp3ContentType(node?.detected_content_type)
    || isMp3ContentType(node?.media_type)
    || isMp3ContentType(node?.resource_mime)
  );
}

function resourceKindLabel(resourceKind: GraphNodeResourceKind): string {
  switch (resourceKind) {
    case "text":
      return "TEXT";
    case "image":
      return "IMAGE";
    case "audio":
      return "AUDIO";
    case "archive":
      return "ARCHIVE";
    case "blob":
      return "BLOB";
    case "video":
      return "VIDEO";
    case "link":
      return "LINK";
    case "website":
      return "WEBSITE";
    default:
      return "RESOURCE";
  }
}

function isEditorResource(node: any, resourceKind: GraphNodeResourceKind): boolean {
  if (resourceKind === "text") {
    return true;
  }
  if (resourceKind !== "blob") {
    return false;
  }
  const fileKind = String(node?.kind ?? "").trim().toLowerCase();
  if (fileKind === "text") {
    return true;
  }
  const ext = extensionFromPathLike(sourcePathFromNode(node));
  return EDITOR_EXTENSIONS.has(ext);
}

function worldscreenViewForNode(
  node: any,
  nodeKind: "file" | "crawler" | "nexus",
  resourceKind: GraphNodeResourceKind,
): GraphWorldscreenView {
  if (nodeKind === "nexus") {
    return "metadata";
  }
  if (resourceKind === "image") {
    return "metadata";
  }
  if (resourceKind === "audio") {
    return "metadata";
  }
  if (resourceKind === "video") {
    return "video";
  }
  if (nodeKind === "file" && isEditorResource(node, resourceKind)) {
    return "editor";
  }
  return "website";
}

function worldscreenSubtitleForNode(
  node: any,
  nodeKind: "file" | "crawler" | "nexus",
  resourceKind: GraphNodeResourceKind,
): string {
  const kindText = resourceKindLabel(resourceKind);
  if (nodeKind === "nexus") {
    const nodeTypeText = String(node?.node_type ?? "nexus").trim().toLowerCase() || "nexus";
    const roleText = String(node?.kind ?? node?.presence_kind ?? "").trim().toLowerCase();
    return roleText ? `${kindText} · ${nodeTypeText}/${roleText}` : `${kindText} · ${nodeTypeText}`;
  }
  if (nodeKind === "crawler") {
    const domain = String(node?.domain ?? "").trim();
    return domain ? `${kindText} · ${shortPathLabel(domain)}` : kindText;
  }
  const pathText = sourcePathFromNode(node);
  return pathText ? `${kindText} · ${shortPathLabel(pathText)}` : kindText;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function clampValue(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function shortPathLabel(path: string): string {
  const trimmed = path.trim();
  if (!trimmed) {
    return "(unknown)";
  }
  const parts = trimmed.split("/");
  const leaf = parts[parts.length - 1] || trimmed;
  if (leaf.length <= 22) {
    return leaf;
  }
  return `${leaf.slice(0, 19)}...`;
}

function isViewCompactionBundleNode(node: any): boolean {
  if (!node || typeof node !== "object") {
    return false;
  }
  const nodeRole = String(node?.kind ?? node?.presence_kind ?? "").trim().toLowerCase();
  const semanticRole = String(node?.simulation_semantic_role ?? "").trim().toLowerCase();
  const truthScope = String(node?.truth_scope ?? "").trim().toLowerCase();
  const sourceRelPath = String(
    node?.source_rel_path
    ?? node?.archived_rel_path
    ?? node?.archive_rel_path
    ?? "",
  ).trim().toLowerCase();
  const projectionGroupId = String(node?.projection_group_id ?? "").trim();
  return Boolean(node?.projection_overflow)
    || Boolean(node?.consolidated)
    || Boolean(node?.semantic_bundle)
    || Boolean(node?.is_view_compaction_bundle)
    || nodeRole === "projection_overflow"
    || nodeRole === "view_compaction_bundle"
    || semanticRole === "view_compaction_aggregate"
    || truthScope === "excluded_projection_bundle"
    || projectionGroupId.length > 0
    || sourceRelPath.startsWith("_projection/")
    || sourceRelPath.startsWith("_consolidated/");
}

function isViewCompactionBundleEdge(edge: any): boolean {
  if (!edge || typeof edge !== "object") {
    return false;
  }
  const semanticRole = String(edge?.simulation_semantic_role ?? "").trim().toLowerCase();
  const truthScope = String(edge?.truth_scope ?? "").trim().toLowerCase();
  const projectionGroupId = String(edge?.projection_group_id ?? "").trim();
  const edgeId = String(edge?.id ?? "").trim();
  const sourceText = String(edge?.source ?? "").trim();
  const targetText = String(edge?.target ?? "").trim();
  return Boolean(edge?.projection_overflow)
    || Boolean(edge?.consolidated)
    || Boolean(edge?.semantic_bundle)
    || semanticRole === "view_compaction_aggregate"
    || truthScope === "excluded_projection_bundle"
    || projectionGroupId.length > 0
    || edgeId.includes("projection:")
    || sourceText.includes("projection:")
    || targetText.includes("projection:");
}

function isRemoteHttpUrl(url: string): boolean {
  const trimmed = url.trim().toLowerCase();
  return trimmed.startsWith("http://") || trimmed.startsWith("https://");
}

function timestampLabel(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    const millis = value > 10_000_000_000 ? value : value * 1000;
    const parsed = new Date(millis);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString();
    }
  }

  const text = String(value ?? "").trim();
  if (!text) {
    return "";
  }

  const numeric = Number(text);
  if (Number.isFinite(numeric) && numeric > 0) {
    const millis = numeric > 10_000_000_000 ? numeric : numeric * 1000;
    const parsed = new Date(millis);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString();
    }
  }

  const parsed = new Date(text);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toISOString();
  }
  return text;
}

function joinListValues(value: unknown): string {
  if (!Array.isArray(value)) {
    return "";
  }
  const flattened = value
    .map((item) => String(item ?? "").trim())
    .filter((item) => item.length > 0);
  return flattened.join(", ");
}

function resolveProjectionBundleManifest(node: any, fileGraph: any): { groupId: string; members: string[] } {
  const groupId = String(node?.projection_group_id ?? "").trim();
  if (!groupId || !fileGraph || typeof fileGraph !== "object") {
    return { groupId: "", members: [] };
  }
  const projection = (fileGraph as any)?.projection;
  const groups = Array.isArray(projection?.groups) ? projection.groups : [];
  const group = groups.find((row: any) => String(row?.id ?? "").trim() === groupId);
  if (!group || typeof group !== "object") {
    return { groupId, members: [] };
  }

  const sourceIds = Array.isArray(group.member_source_ids)
    ? group.member_source_ids.map((value: unknown) => String(value ?? "").trim()).filter((value: string) => value.length > 0)
    : [];
  if (sourceIds.length <= 0) {
    return { groupId, members: [] };
  }

  const fileNodes = Array.isArray((fileGraph as any)?.file_nodes) ? (fileGraph as any).file_nodes : [];
  const nodeById = new Map<string, any>();
  for (const fileNode of fileNodes) {
    const fileNodeId = String(fileNode?.id ?? "").trim();
    if (!fileNodeId || nodeById.has(fileNodeId)) {
      continue;
    }
    nodeById.set(fileNodeId, fileNode);
  }

  const members = sourceIds.map((sourceId: string) => {
    const sourceNode = nodeById.get(sourceId);
    const pathText = String(
      sourceNode?.source_rel_path
      ?? sourceNode?.archived_rel_path
      ?? sourceNode?.archive_rel_path
      ?? sourceNode?.label
      ?? sourceNode?.name
      ?? sourceId,
    ).trim();
    return pathText || sourceId;
  });
  return {
    groupId,
    members,
  };
}

function semanticWeightScaleForParticle(row: BackendFieldParticle): number {
  const textChars = Math.max(0, Number((row as any)?.semantic_text_chars ?? 0));
  const semanticMass = Math.max(0, Number((row as any)?.semantic_mass ?? row.mass ?? 0));
  const daimoiEnergy = Math.max(0, Number((row as any)?.daimoi_energy ?? 0));
  const messageProbability = clamp01(Number((row as any)?.message_probability ?? 0));
  const packageEntropy = Math.max(0, Number((row as any)?.package_entropy ?? 0));
  const textTerm = Math.log1p(textChars) * 0.22;
  const energyTerm = Math.log1p((daimoiEnergy * 2.2) + (messageProbability * 3.0)) * 0.28;
  const entropyTerm = packageEntropy * 0.06;
  const massTerm = semanticMass * 0.12;
  const scale = 0.9 + textTerm + energyTerm + entropyTerm + massTerm;
  return clampValue(scale, 0.78, 2.2);
}

function remoteFrameUrlForNode(
  node: any,
  worldscreenUrl: string,
  resourceKind: GraphNodeResourceKind,
): string {
  const candidates = [
    node?.crawler_frame_url,
    node?.crawler_snapshot_url,
    node?.snapshot_url,
    node?.preview_url,
    node?.thumbnail_url,
    node?.thumb_url,
    node?.image_url,
  ];

  for (const candidate of candidates) {
    const value = String(candidate ?? "").trim();
    if (isRemoteHttpUrl(value)) {
      return value;
    }
  }

  if (resourceKind === "image" && isRemoteHttpUrl(worldscreenUrl)) {
    return worldscreenUrl;
  }
  return "";
}

function worldscreenMetadataRows(worldscreen: GraphWorldscreenState): Array<{ key: string; value: string }> {
  const rows: Array<{ key: string; value: string }> = [
    { key: "resource", value: resourceKindLabel(worldscreen.resourceKind) },
    { key: "node-kind", value: String(worldscreen.nodeKind ?? "") },
    { key: "node-type", value: String(worldscreen.nodeTypeText ?? "") },
    { key: "node-role", value: String(worldscreen.nodeRoleText ?? "") },
    { key: "node-id", value: String(worldscreen.nodeId ?? "") },
    { key: "comment-ref", value: String(worldscreen.commentRef ?? "") },
    { key: "image-ref", value: String(worldscreen.imageRef ?? "") },
    { key: "url", value: worldscreen.url },
    { key: "domain", value: String(worldscreen.domain ?? "") },
    { key: "title", value: String(worldscreen.titleText ?? "") },
    { key: "status", value: String(worldscreen.statusText ?? "") },
    { key: "content-type", value: String(worldscreen.contentTypeText ?? "") },
    { key: "compliance", value: String(worldscreen.complianceText ?? "") },
    { key: "source", value: String(worldscreen.sourceUrl ?? "") },
    { key: "discovered", value: String(worldscreen.discoveredAt ?? "") },
    { key: "fetched", value: String(worldscreen.fetchedAt ?? "") },
    { key: "encountered", value: String(worldscreen.encounteredAt ?? "") },
    { key: "summary", value: String(worldscreen.summaryText ?? "") },
    { key: "tags", value: String(worldscreen.tagsText ?? "") },
    { key: "labels", value: String(worldscreen.labelsText ?? "") },
    { key: "projection-group", value: String(worldscreen.projectionGroupId ?? "") },
    {
      key: "bundle-members",
      value: worldscreen.projectionConsolidatedCount === undefined
        ? ""
        : String(Math.max(0, Math.floor(worldscreen.projectionConsolidatedCount))),
    },
  ];

  return rows.filter((row) => row.value.trim().length > 0);
}

function resolveWorldscreenPlacement(
  worldscreen: GraphWorldscreenState,
  container: HTMLDivElement | null,
  glassCenterRatio: { x: number; y: number } = { x: 0.5, y: 0.5 },
): WorldscreenPlacement {
  const fallbackWidth = typeof window !== "undefined"
    ? Math.max(620, Math.floor(window.innerWidth * 0.92))
    : 960;
  const fallbackHeight = typeof window !== "undefined"
    ? Math.max(420, Math.floor(window.innerHeight * 0.72))
    : 640;

  const containerWidth = Math.max(320, container?.clientWidth ?? fallbackWidth);
  const containerHeight = Math.max(220, container?.clientHeight ?? fallbackHeight);
  const width = clampValue(Math.round(Math.min(containerWidth * 0.92, 780)), 340, Math.max(340, containerWidth - 18));
  const height = clampValue(Math.round(Math.min(containerHeight * 0.68, 540)), 220, Math.max(220, containerHeight - 18));

  const margin = 10;
  const centerRatioX = clamp01(
    typeof glassCenterRatio?.x === "number"
      ? glassCenterRatio.x
      : (typeof worldscreen.anchorRatioX === "number" ? worldscreen.anchorRatioX : 0.5),
  );
  const centerRatioY = clamp01(
    typeof glassCenterRatio?.y === "number"
      ? glassCenterRatio.y
      : (typeof worldscreen.anchorRatioY === "number" ? worldscreen.anchorRatioY : 0.5),
  );
  const centerX = centerRatioX * containerWidth;
  const centerY = centerRatioY * containerHeight;

  let left = centerX - (width / 2);
  let top = centerY - (height / 2);

  left = clampValue(left, margin, Math.max(margin, containerWidth - width - margin));
  top = clampValue(top, margin, Math.max(margin, containerHeight - height - margin));

  const ratioWithin = (value: number, start: number, span: number): string => {
    if (value <= start) {
      return "0%";
    }
    const end = start + Math.max(1, span);
    if (value >= end) {
      return "100%";
    }
    const ratio = ((value - start) / Math.max(1, span)) * 100;
    return `${ratio.toFixed(2)}%`;
  };

  return {
    left,
    top,
    width,
    height,
    transformOrigin: `${ratioWithin(centerX, left, width)} ${ratioWithin(centerY, top, height)}`,
  };
}

function resolveWorldscreenUrl(
  openUrl: string,
  nodeKind: string,
  domain: string,
): string | null {
  const trimmedUrl = openUrl.trim();
  if (trimmedUrl.startsWith("http://") || trimmedUrl.startsWith("https://")) {
    return trimmedUrl;
  }
  if (trimmedUrl) {
    const normalizedPath = trimmedUrl.startsWith("/") ? trimmedUrl : `/${trimmedUrl}`;
    const base = runtimeBaseUrl();
    if (base) {
      return `${base}${normalizedPath}`;
    }
    return normalizedPath;
  }
  const domainText = domain.trim();
  if (nodeKind === "crawler" && domainText) {
    return `https://${domainText}`;
  }
  return null;
}

function normalizeLibraryRelativePath(pathLike: string): string {
  return pathLike
    .trim()
    .replace(/^\/+/, "")
    .split("/")
    .map((segment) => segment.trim())
    .filter((segment) => segment.length > 0)
    .join("/");
}

function encodeLibraryPath(pathLike: string): string {
  return normalizeLibraryRelativePath(pathLike)
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

function libraryUrlForPath(pathLike: string): string {
  const encodedPath = encodeLibraryPath(pathLike);
  if (!encodedPath) {
    return "";
  }
  return `/library/${encodedPath}`;
}

function libraryUrlForArchiveMember(archivePath: string, memberPath: string): string {
  const archiveUrl = libraryUrlForPath(archivePath);
  if (!archiveUrl) {
    return "";
  }
  const normalizedMember = normalizeLibraryRelativePath(memberPath);
  if (!normalizedMember) {
    return archiveUrl;
  }
  const memberParam = encodeURIComponent(normalizedMember);
  return `${archiveUrl}?member=${memberParam}`;
}

function openUrlForGraphNode(node: any, nodeKind: "file" | "crawler" | "nexus"): string {
  const directUrl = String(node?.url ?? "").trim();
  if (directUrl) {
    return directUrl;
  }

  if (nodeKind === "crawler") {
    return "";
  }

  const archiveMemberPath = String(node?.archive_member_path ?? "").trim();
  const archiveRelPath = String(
    node?.archive_rel_path || node?.archived_rel_path || "",
  ).trim();
  if (archiveRelPath && archiveMemberPath) {
    return libraryUrlForArchiveMember(archiveRelPath, archiveMemberPath);
  }

  const archiveUrl = String(node?.archive_url ?? "").trim();
  if (archiveUrl) {
    if (archiveMemberPath && !archiveUrl.includes("member=")) {
      const separator = archiveUrl.includes("?") ? "&" : "?";
      return `${archiveUrl}${separator}member=${encodeURIComponent(normalizeLibraryRelativePath(archiveMemberPath))}`;
    }
    return archiveUrl;
  }

  if (archiveRelPath) {
    return libraryUrlForPath(archiveRelPath);
  }

  const sourceRelPath = String(node?.source_rel_path ?? "").trim();
  if (sourceRelPath) {
    return libraryUrlForPath(sourceRelPath);
  }

  return "";
}

interface OverlayParticleFlags {
  particleMode: string;
  isCdbParticle: boolean;
  isChaosParticle: boolean;
  isStaticParticle: boolean;
  isNexusParticle: boolean;
  isSmartDaimoi: boolean;
  isResourceEmitter: boolean;
  isTransferParticle: boolean;
  routeNodeId: string;
  graphNodeId: string;
}

function resolveOverlayParticleFlags(row: BackendFieldParticle): OverlayParticleFlags {
  const rowId = String((row as any)?.id ?? "");
  const rowRecord = String((row as any)?.record ?? "");
  const rowSchema = String((row as any)?.schema_version ?? "");
  const particleMode = String((row as any)?.particle_mode ?? "").trim();
  const presenceRole = String((row as any)?.presence_role ?? "").trim().toLowerCase();
  const routeNodeId = String((row as any)?.route_node_id ?? "").trim();
  const graphNodeId = String((row as any)?.graph_node_id ?? "").trim();
  const topJob = String((row as any)?.top_job ?? "").trim();
  const isCdbParticle = rowId.startsWith("cdb:") || rowRecord.includes(".cdb.") || rowSchema.includes(".cdb.");
  const isChaosParticle = particleMode === "chaos-butterfly";
  const isStaticParticle = particleMode === "static-daimoi";
  const isNexusParticle = Boolean((row as any)?.is_nexus) || isStaticParticle || presenceRole === "nexus-passive";
  const isSmartDaimoi = !isNexusParticle;
  const isResourceEmitter = Boolean((row as any)?.resource_daimoi) || topJob === "emit_resource_packet";
  const isTransferParticle = (
    routeNodeId.length > 0
    && graphNodeId.length > 0
    && routeNodeId !== graphNodeId
  );
  return {
    particleMode,
    isCdbParticle,
    isChaosParticle,
    isStaticParticle,
    isNexusParticle,
    isSmartDaimoi,
    isResourceEmitter,
    isTransferParticle,
    routeNodeId,
    graphNodeId,
  };
}

export function SimulationCanvas({
  simulation,
  catalog,
  onOverlayInit,
  onNexusInteraction,
  onUserPresenceInput,
  height = 300,
  defaultOverlayView = "omni",
  overlayViewLocked = false,
  compactHud = false,
  interactive = true,
  backgroundMode = false,
  particleDensity = 1,
  particleScale = 1,
  motionSpeed = 1,
  mouseInfluence = 1,
  layerDepth = 1,
  graphNodeSmoothness = 1,
  graphNodeStepScale = 1,
  backgroundWash = 0.58,
  layerVisibility,
  glassCenterRatio = { x: 0.5, y: 0.5 },
  museWorkspaceBindings = {},
  className = "",
  mouseDaimonEnabled = true,
  mouseDaimonMessage = "witness",
  mouseDaimonMode = "push",
  mouseDaimonRadius = 0.18,
  mouseDaimonStrength = 0.42,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const metaRef = useRef<HTMLParagraphElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<SimulationState | null>(simulation);
  const catalogRef = useRef<Catalog | null>(catalog);
  const museWorkspaceBindingsRef = useRef<Record<string, string[]>>(
    normalizeWorkspaceBindingMap(museWorkspaceBindings),
  );
  const particleDensityRef = useRef(clampValue(particleDensity, 0.08, 1));
  const particleScaleRef = useRef(clampValue(particleScale, 0.5, 1.9));
  const motionSpeedRef = useRef(clampValue(motionSpeed, 0.25, 2.2));
  const mouseInfluenceRef = useRef(clampValue(mouseInfluence, 0, 2.5));
  const layerDepthRef = useRef(clampValue(layerDepth, 0.35, 2.2));
  const graphNodeSmoothnessRef = useRef(clampValue(graphNodeSmoothness, 0.5, 2.8));
  const graphNodeStepScaleRef = useRef(clampValue(graphNodeStepScale, 0.35, 2.8));
  const layerVisibilityRef = useRef(layerVisibility);
  const backgroundModeRef = useRef(backgroundMode);
  const backgroundWashRef = useRef(backgroundWash);
  const interactiveRef = useRef(interactive);
  const overlayViewRef = useRef<OverlayViewId>(defaultOverlayView);
  const onNexusInteractionRef = useRef(onNexusInteraction);
  const onOverlayInitRef = useRef(onOverlayInit);
  const onUserPresenceInputRef = useRef(onUserPresenceInput);
  const glassCenterRatioRef = useRef(glassCenterRatio);
  useEffect(() => {
    onUserPresenceInputRef.current = onUserPresenceInput;
  }, [onUserPresenceInput]);
  useEffect(() => {
    glassCenterRatioRef.current = glassCenterRatio;
  }, [glassCenterRatio]);
  const [worldscreen, setWorldscreen] = useState<GraphWorldscreenState | null>(null);
  const [worldscreenPinnedCenterRatio, setWorldscreenPinnedCenterRatio] = useState<{ x: number; y: number } | null>(null);
  const [worldscreenMode, setWorldscreenMode] = useState<GraphWorldscreenMode>("overview");
  const worldscreenAudioElementRef = useRef<HTMLAudioElement | null>(null);
  const worldscreenAudioBaseCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const worldscreenAudioPlayheadCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const worldscreenAudioSeekPointerIdRef = useRef<number | null>(null);
  const [worldscreenAudioViz, setWorldscreenAudioViz] = useState<HologramAudioVisualization | null>(null);
  const [worldscreenAudioVizStatus, setWorldscreenAudioVizStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [worldscreenAudioVizError, setWorldscreenAudioVizError] = useState("");
  const [worldscreenAudioClockText, setWorldscreenAudioClockText] = useState("0:00 / 0:00");
  const [editorPreview, setEditorPreview] = useState<EditorPreviewState>({
    status: "idle",
    content: "",
    error: "",
    truncated: false,
  });
  const [overlayView, setOverlayView] = useState<OverlayViewId>(defaultOverlayView);
  const [presenceAccounts, setPresenceAccounts] = useState<PresenceAccountEntry[]>([]);
  const [presenceAccountId, setPresenceAccountId] = useState("witness_thread");
  const [imageComments, setImageComments] = useState<ImageCommentEntry[]>([]);
  const [imageCommentPrompt, setImageCommentPrompt] = useState(
    "Describe the image evidence and one next action.",
  );
  const [imageCommentDraft, setImageCommentDraft] = useState("");
  const [imageCommentParentId, setImageCommentParentId] = useState("");
  const [imageCommentsLoading, setImageCommentsLoading] = useState(false);
  const [imageCommentBusy, setImageCommentBusy] = useState(false);
  const [imageCommentError, setImageCommentError] = useState("");
  const [modelDockOpen, setModelDockOpen] = useState(false);
  const [computeJobFilter, setComputeJobFilter] = useState<ComputeJobFilter>("all");
  const [computePanelCollapsed, setComputePanelCollapsed] = useState(false);
  const [musicNexusSpotlight, setMusicNexusSpotlight] = useState(false);
  const [musicNexusJumpLabel, setMusicNexusJumpLabel] = useState("");
  const [graphNodeTitleOverlays, setGraphNodeTitleOverlays] = useState<GraphNodeTitleOverlay[]>([]);
  const musicNexusSpotlightRef = useRef(musicNexusSpotlight);
  const musicNexusHotspotsRef = useRef<MusicNexusHotspot[]>([]);
  const musicNexusCycleRef = useRef(0);
  const interactAtRef = useRef<
    ((xRatio: number, yRatio: number, options?: { openWorldscreen?: boolean }) => {
      hitNode: boolean;
      openedWorldscreen: boolean;
      target: string;
      xRatio: number;
      yRatio: number;
    }) | null
  >(null);

  useEffect(() => {
    musicNexusSpotlightRef.current = musicNexusSpotlight;
  }, [musicNexusSpotlight]);

  // Mouse Daimon refs - synced from props (managed in CoreControlPanel)
  const mouseDaimonEnabledRef = useRef(mouseDaimonEnabled);
  const mouseDaimonMessageRef = useRef(mouseDaimonMessage);
  const mouseDaimonModeRef = useRef(mouseDaimonMode);
  const mouseDaimonRadiusRef = useRef(mouseDaimonRadius);
  const mouseDaimonStrengthRef = useRef(mouseDaimonStrength);

  useEffect(() => { mouseDaimonEnabledRef.current = mouseDaimonEnabled; }, [mouseDaimonEnabled]);
  useEffect(() => { mouseDaimonMessageRef.current = mouseDaimonMessage; }, [mouseDaimonMessage]);
  useEffect(() => { mouseDaimonModeRef.current = mouseDaimonMode; }, [mouseDaimonMode]);
  useEffect(() => { mouseDaimonRadiusRef.current = mouseDaimonRadius; }, [mouseDaimonRadius]);
  useEffect(() => { mouseDaimonStrengthRef.current = mouseDaimonStrength; }, [mouseDaimonStrength]);

  const lastNexusPointerTapRef = useRef<{ key: string; atMs: number } | null>(null);

  const renderParticleFieldRef = useRef(interactive || backgroundMode);
  const renderRichOverlayParticles = true;
  const renderOverlayWithWebgl = true;

  const resolveFieldParticleRows = useCallback((state: SimulationState | null): BackendFieldParticle[] => {
    const directRows = state?.presence_dynamics?.field_particles ?? state?.field_particles;
    return Array.isArray(directRows) ? (directRows as BackendFieldParticle[]) : [];
  }, []);

  const computeJobInsights = useMemo(() => {
    if (compactHud) {
      return EMPTY_COMPUTE_JOB_INSIGHTS;
    }
    const simulationRows = normalizeComputeJobInsightRows(simulation?.presence_dynamics?.compute_jobs);
    const runtimeRows = normalizeComputeJobInsightRows(catalog?.presence_runtime?.compute_jobs);
    const sourceRows = simulationRows.length > 0 ? simulationRows : runtimeRows;
    const dedupedById = new Map<string, ComputeJobInsightRow>();
    for (const row of sourceRows) {
      if (!dedupedById.has(row.id)) {
        dedupedById.set(row.id, row);
      }
    }
    const rows = Array.from(dedupedById.values()).sort((a, b) => b.tsMs - a.tsMs);
    const summary = summarizeComputeJobs(rows);
    const filtered = rows.filter((row) => {
      if (computeJobFilter === "all") {
        return true;
      }
      if (computeJobFilter === "llm") {
        return row.kind === "llm";
      }
      if (computeJobFilter === "embedding") {
        return row.kind.includes("embed");
      }
      if (computeJobFilter === "error") {
        return row.status === "error" || row.status === "failed" || row.status === "timeout";
      }
      if (computeJobFilter === "gpu" || computeJobFilter === "npu" || computeJobFilter === "cpu") {
        return row.resource === computeJobFilter;
      }
      return true;
    });

    const heartbeat = asRecord((catalog?.presence_runtime as Record<string, unknown> | undefined)?.resource_heartbeat);
    const devices = asRecord(heartbeat?.devices);
    const gpuRows = Object.entries(devices ?? {})
      .filter(([key]) => key.toLowerCase().startsWith("gpu"))
      .map(([, value]) => asRecord(value))
      .filter((row): row is Record<string, unknown> => Boolean(row));
    const gpuAvailability = gpuRows.length > 0
      ? clamp01(1 - (gpuRows.reduce((sum, row) => sum + clamp01(Number(row.utilization ?? 0) / 100), 0) / gpuRows.length))
      : 0;

    return {
      rows,
      summary,
      filtered,
      gpuAvailability,
      total180s: Number(simulation?.presence_dynamics?.compute_jobs_180s ?? catalog?.presence_runtime?.compute_jobs_180s ?? rows.length),
    };
  }, [catalog, compactHud, computeJobFilter, simulation]);

  const worldscreenAudioUrl = useMemo(() => {
    if (!worldscreen || worldscreen.resourceKind !== "audio") {
      return "";
    }
    const preferred = resolveWorldscreenMediaUrl(worldscreen.url);
    if (preferred) {
      return preferred;
    }
    return resolveWorldscreenMediaUrl(worldscreen.remoteFrameUrl || "");
  }, [worldscreen]);

  const seekWorldscreenAudioToRatio = useCallback((ratioInput: number) => {
    const audio = worldscreenAudioElementRef.current;
    if (!audio) {
      return;
    }
    const duration = Number.isFinite(audio.duration) && audio.duration > 0
      ? audio.duration
      : (worldscreenAudioViz?.durationSeconds ?? 0);
    if (!(duration > 0)) {
      return;
    }
    const ratio = clamp01(ratioInput);
    audio.currentTime = ratio * duration;
    setWorldscreenAudioClockText(
      `${formatPlaybackClock(audio.currentTime)} / ${formatPlaybackClock(duration)}`,
    );
  }, [worldscreenAudioViz?.durationSeconds]);

  const seekWorldscreenAudioFromClientX = useCallback((clientX: number) => {
    const canvas = worldscreenAudioPlayheadCanvasRef.current || worldscreenAudioBaseCanvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) {
      return;
    }
    const ratio = clamp01((clientX - rect.left) / rect.width);
    seekWorldscreenAudioToRatio(ratio);
  }, [seekWorldscreenAudioToRatio]);

  const loadPresenceAccounts = useCallback(
    async (signal?: AbortSignal): Promise<PresenceAccountEntry[]> => {
      const response = await fetch(`${runtimeBaseUrl()}/api/presence/accounts?limit=120`, {
        method: "GET",
        credentials: "same-origin",
        signal,
      });
      if (!response.ok) {
        throw new Error(`presence account fetch failed (${response.status})`);
      }
      const payload = await response.json();
      const entries = normalizePresenceAccountEntries(payload);
      setPresenceAccounts(entries);
      return entries;
    },
    [],
  );

  const loadImageComments = useCallback(
    async (imageRef: string, signal?: AbortSignal): Promise<ImageCommentEntry[]> => {
      const target = imageRef.trim();
      if (!target) {
        setImageComments([]);
        return [];
      }
      const query = encodeURIComponent(target);
      const response = await fetch(
        `${runtimeBaseUrl()}/api/image/comments?image_ref=${query}&limit=120`,
        {
          method: "GET",
          credentials: "same-origin",
          signal,
        },
      );
      if (!response.ok) {
        throw new Error(`image comment fetch failed (${response.status})`);
      }
      const payload = await response.json();
      const entries = normalizeImageCommentEntries(payload);
      setImageComments(entries);
      return entries;
    },
    [],
  );

  const ensurePresenceAccount = useCallback(
    async (presenceId: string): Promise<string> => {
      const chosen = presenceId.trim();
      if (!chosen) {
        throw new Error("presence account id is required");
      }
      const displayName = chosen.replace(/[_-]+/g, " ").trim() || chosen;
      const response = await fetch(`${runtimeBaseUrl()}/api/presence/accounts/upsert`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({
          presence_id: chosen,
          display_name: displayName,
          handle: chosen,
          avatar: "",
          bio: "",
          tags: ["nexus-commentary"],
        }),
      });
      const payload = asRecord(await response.json().catch(() => null));
      if (!response.ok || payload?.ok !== true) {
        throw new Error(String(payload?.error ?? `presence upsert failed (${response.status})`));
      }
      await loadPresenceAccounts();
      return chosen;
    },
    [loadPresenceAccounts],
  );

  const submitGeneratedImageCommentary = useCallback(async () => {
    if (!worldscreen || worldscreen.resourceKind !== "image") {
      return;
    }
    const commentRef = worldscreenCommentRef(worldscreen);
    if (!commentRef) {
      setImageCommentError("image reference unavailable for commentary");
      return;
    }
    setImageCommentBusy(true);
    setImageCommentError("");
    try {
      const presenceId = await ensurePresenceAccount(presenceAccountId);
      const imagePayload = await fetchImagePayloadAsBase64(worldscreen);
      const response = await fetch(`${runtimeBaseUrl()}/api/image/commentary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({
          image_base64: imagePayload.base64,
          image_ref: commentRef,
          mime: imagePayload.mime,
          presence_id: presenceId,
          prompt: imageCommentPrompt,
          persist: true,
        }),
      });
      const payload = asRecord(await response.json().catch(() => null));
      if (!response.ok || payload?.ok !== true) {
        throw new Error(String(payload?.error ?? `image commentary failed (${response.status})`));
      }
      const commentary = String(payload?.commentary ?? "").trim();
      if (commentary) {
        setImageCommentDraft(commentary);
      }
      await loadImageComments(commentRef);
    } catch (error: unknown) {
      setImageCommentError(errorMessage(error, "unable to generate image commentary"));
    } finally {
      setImageCommentBusy(false);
    }
  }, [
    ensurePresenceAccount,
    imageCommentPrompt,
    loadImageComments,
    presenceAccountId,
    worldscreen,
  ]);

  const submitManualImageComment = useCallback(async () => {
    if (!worldscreen) {
      return;
    }
    const commentRef = worldscreenCommentRef(worldscreen);
    if (!commentRef) {
      setImageCommentError("nexus reference unavailable for comment posting");
      return;
    }
    const commentText = imageCommentDraft.trim();
    if (!commentText) {
      setImageCommentError("comment text is empty");
      return;
    }

    setImageCommentBusy(true);
    setImageCommentError("");
    try {
      const presenceId = await ensurePresenceAccount(presenceAccountId);
      const response = await fetch(`${runtimeBaseUrl()}/api/image/comments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({
          image_ref: commentRef,
          presence_id: presenceId,
          comment: commentText,
          metadata: {
            source: "manual",
            nexus_ref: commentRef,
            node_id: worldscreen.nodeId,
            node_kind: worldscreen.nodeKind,
            resource_kind: worldscreen.resourceKind,
            target_url: worldscreen.url,
            compacted_into_nexus: true,
            true_graph_embed: true,
            parent_comment_id: imageCommentParentId || undefined,
          },
        }),
      });
      const payload = asRecord(await response.json().catch(() => null));
      if (!response.ok || payload?.ok !== true) {
        throw new Error(String(payload?.error ?? `image comment create failed (${response.status})`));
      }
      setImageCommentDraft("");
      setImageCommentParentId("");
      await loadImageComments(commentRef);
    } catch (error: unknown) {
      setImageCommentError(errorMessage(error, "unable to save image comment"));
    } finally {
      setImageCommentBusy(false);
    }
  }, [
    ensurePresenceAccount,
    imageCommentDraft,
    imageCommentParentId,
    loadImageComments,
    presenceAccountId,
    worldscreen,
  ]);

  const refreshNexusComments = useCallback((commentRef: string) => {
    const target = commentRef.trim();
    if (!target) {
      return;
    }
    setImageCommentsLoading(true);
    setImageCommentError("");
    void loadImageComments(target)
      .catch((error: unknown) => {
        setImageCommentError(errorMessage(error, "unable to refresh image comments"));
      })
      .finally(() => {
        setImageCommentsLoading(false);
      });
  }, [loadImageComments]);

  useEffect(() => {
    if (!overlayViewLocked) {
      return;
    }
    setOverlayView(defaultOverlayView);
  }, [defaultOverlayView, overlayViewLocked]);

  useEffect(() => {
    simulationRef.current = simulation;
  }, [simulation]);

  useEffect(() => {
    catalogRef.current = catalog;
  }, [catalog]);

  useEffect(() => {
    museWorkspaceBindingsRef.current = normalizeWorkspaceBindingMap(museWorkspaceBindings);
  }, [museWorkspaceBindings]);

  useEffect(() => {
    particleDensityRef.current = clampValue(particleDensity, 0.08, 1);
  }, [particleDensity]);

  useEffect(() => {
    particleScaleRef.current = clampValue(particleScale, 0.5, 1.9);
  }, [particleScale]);

  useEffect(() => {
    motionSpeedRef.current = clampValue(motionSpeed, 0.25, 2.2);
  }, [motionSpeed]);

  useEffect(() => {
    mouseInfluenceRef.current = clampValue(mouseInfluence, 0, 2.5);
  }, [mouseInfluence]);

  useEffect(() => {
    layerDepthRef.current = clampValue(layerDepth, 0.35, 2.2);
  }, [layerDepth]);

  useEffect(() => {
    graphNodeSmoothnessRef.current = clampValue(graphNodeSmoothness, 0.5, 2.8);
  }, [graphNodeSmoothness]);

  useEffect(() => {
    graphNodeStepScaleRef.current = clampValue(graphNodeStepScale, 0.35, 2.8);
  }, [graphNodeStepScale]);

  useEffect(() => {
    layerVisibilityRef.current = layerVisibility;
  }, [layerVisibility]);

  useEffect(() => {
    backgroundModeRef.current = backgroundMode;
  }, [backgroundMode]);

  useEffect(() => {
    backgroundWashRef.current = backgroundWash;
  }, [backgroundWash]);

  useEffect(() => {
    interactiveRef.current = interactive;
  }, [interactive]);

  useEffect(() => {
    if (interactive) {
      return;
    }
    setGraphNodeTitleOverlays((previous) => (previous.length > 0 ? [] : previous));
  }, [interactive]);

  useEffect(() => {
    renderParticleFieldRef.current = interactive || backgroundMode;
  }, [backgroundMode, interactive]);

  useEffect(() => {
    overlayViewRef.current = overlayView;
  }, [overlayView]);

  useEffect(() => {
    onNexusInteractionRef.current = onNexusInteraction;
  }, [onNexusInteraction]);

  useEffect(() => {
    onOverlayInitRef.current = onOverlayInit;
  }, [onOverlayInit]);

  useEffect(() => {
    if (!worldscreen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setWorldscreen(null);
        setWorldscreenPinnedCenterRatio(null);
        setWorldscreenMode("overview");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [worldscreen]);

  useEffect(() => {
    if (!worldscreenAudioUrl) {
      setWorldscreenAudioViz(null);
      setWorldscreenAudioVizStatus("idle");
      setWorldscreenAudioVizError("");
      setWorldscreenAudioClockText("0:00 / 0:00");
      return;
    }

    const controller = new AbortController();
    let active = true;
    setWorldscreenAudioVizStatus("loading");
    setWorldscreenAudioVizError("");
    setWorldscreenAudioClockText("0:00 / 0:00");

    void (async () => {
      let context: AudioContext | null = null;
      try {
        const response = await fetch(worldscreenAudioUrl, {
          method: "GET",
          credentials: "same-origin",
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`audio analysis fetch failed (${response.status})`);
        }
        const payload = await response.arrayBuffer();
        if (!active || controller.signal.aborted) {
          return;
        }

        context = new AudioContext();
        const decoded = await context.decodeAudioData(payload.slice(0));
        if (!active || controller.signal.aborted) {
          return;
        }

        const visualization = buildHologramAudioVisualization(
          decoded,
          worldscreenAudioUrl,
        );
        if (!active || controller.signal.aborted) {
          return;
        }
        setWorldscreenAudioViz(visualization);
        setWorldscreenAudioVizStatus("ready");
        setWorldscreenAudioClockText(
          `0:00 / ${formatPlaybackClock(visualization.durationSeconds)}`,
        );
      } catch (error: unknown) {
        if (!active || controller.signal.aborted) {
          return;
        }
        setWorldscreenAudioViz(null);
        setWorldscreenAudioVizStatus("error");
        setWorldscreenAudioVizError(
          errorMessage(error, "unable to compute waveform/spectrogram for this audio resource"),
        );
      } finally {
        if (context && context.state !== "closed") {
          void context.close().catch(() => {});
        }
      }
    })();

    return () => {
      active = false;
      controller.abort();
    };
  }, [worldscreenAudioUrl]);

  useEffect(() => {
    const canvas = worldscreenAudioBaseCanvasRef.current;
    if (!canvas || !worldscreenAudioViz || worldscreenAudioViz.sourceUrl !== worldscreenAudioUrl) {
      return;
    }

    const draw = () => {
      drawHologramAudioBaseCanvas(canvas, worldscreenAudioViz);
      drawHologramAudioPlayhead(worldscreenAudioPlayheadCanvasRef.current ?? canvas, 0);
    };

    draw();

    if (typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => {
      draw();
    });
    observer.observe(canvas);
    return () => {
      observer.disconnect();
    };
  }, [worldscreenAudioUrl, worldscreenAudioViz]);

  useEffect(() => {
    if (!worldscreenAudioUrl) {
      return;
    }

    const audio = worldscreenAudioElementRef.current;
    const playheadCanvas = worldscreenAudioPlayheadCanvasRef.current;
    if (!audio || !playheadCanvas) {
      return;
    }

    let rafId = 0;
    let lastLabelAt = 0;
    const tick = () => {
      const duration = Number.isFinite(audio.duration) && audio.duration > 0
        ? audio.duration
        : (worldscreenAudioViz?.durationSeconds ?? 0);
      const current = Number.isFinite(audio.currentTime) && audio.currentTime > 0
        ? audio.currentTime
        : 0;
      const ratio = duration > 0 ? clamp01(current / duration) : 0;
      drawHologramAudioPlayhead(playheadCanvas, ratio);

      const now = performance.now();
      if (now - lastLabelAt >= 120) {
        setWorldscreenAudioClockText(
          `${formatPlaybackClock(current)} / ${formatPlaybackClock(duration)}`,
        );
        lastLabelAt = now;
      }

      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => {
      if (rafId !== 0) {
        window.cancelAnimationFrame(rafId);
      }
    };
  }, [worldscreenAudioUrl, worldscreenAudioViz?.durationSeconds]);

  useEffect(() => {
    if (!worldscreen || worldscreen.view !== "editor") {
      setEditorPreview({
        status: "idle",
        content: "",
        error: "",
        truncated: false,
      });
      return;
    }

    const controller = new AbortController();
    let active = true;
    setEditorPreview({ status: "loading", content: "", error: "", truncated: false });

    fetch(worldscreen.url, {
      method: "GET",
      signal: controller.signal,
      credentials: "same-origin",
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`preview failed (${response.status})`);
        }
        const text = await response.text();
        if (!active) {
          return;
        }
        const maxChars = 36000;
        const truncated = text.length > maxChars;
        setEditorPreview({
          status: "ready",
          content: truncated ? text.slice(0, maxChars) : text,
          error: "",
          truncated,
        });
      })
      .catch((error: unknown) => {
        if (!active || controller.signal.aborted) {
          return;
        }
        const message = error instanceof Error ? error.message : "preview failed";
        if (/failed to fetch|networkerror/i.test(message)) {
          setWorldscreen((current) => {
            if (!current || current.view !== "editor" || current.url !== worldscreen.url) {
              return current;
            }
            return {
              ...current,
              view: "website",
            };
          });
          return;
        }
        setEditorPreview({
          status: "error",
          content: "",
          error: message,
          truncated: false,
        });
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [worldscreen]);

  useEffect(() => {
    if (!worldscreen) {
      setImageComments([]);
      setImageCommentError("");
      setImageCommentParentId("");
      setImageCommentsLoading(false);
      return;
    }

    const controller = new AbortController();
    const commentRef = worldscreenCommentRef(worldscreen);
    if (!commentRef) {
      setImageComments([]);
      setImageCommentsLoading(false);
      return () => {
        controller.abort();
      };
    }

    const preloadDelayMs = 120;
    setImageCommentParentId("");
    setImageCommentsLoading(true);
    setImageCommentError("");

    const timeoutId = window.setTimeout(() => {
      void Promise.all([
        loadPresenceAccounts(controller.signal),
        loadImageComments(commentRef, controller.signal),
      ])
        .then(([accounts]) => {
          setPresenceAccountId((current) => {
            const selected = current.trim();
            if (selected) {
              return selected;
            }
            return accounts[0]?.presence_id || "witness_thread";
          });
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) {
            return;
          }
          setImageCommentError(errorMessage(error, "unable to load image comments"));
        })
        .finally(() => {
          if (!controller.signal.aborted) {
            setImageCommentsLoading(false);
          }
        });
    }, preloadDelayMs);

    return () => {
      window.clearTimeout(timeoutId);
      controller.abort();
    };
  }, [loadImageComments, loadPresenceAccounts, worldscreen]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!renderParticleFieldRef.current) {
      return;
    }
    const gl = canvas.getContext("webgl", { alpha: false, antialias: true });
    if (!gl) return;

    const vertexSrc = `
      attribute vec3 aPos;
      attribute float aSize;
      attribute vec3 aColor;
      attribute float aSeed;
      uniform float uTime;
      uniform vec2 uMouse;
      uniform float uInfluence;
      uniform float uMotionRate;
      uniform float uParticleScale;
      uniform float uBloom;
      uniform mat4 uProjection;
      uniform mat4 uView;
      varying vec3 vColor;
      varying float vDepth;
      varying float vSpark;

      void main() {
        vec3 p = aPos;

        vec4 clip = uProjection * uView * vec4(p, 1.0);
        vec2 ndc = clip.xy / max(0.0001, clip.w);
        vec2 away = ndc - uMouse;
        float force = max(0.0, (1.0 - (length(away) * 2.25)) * uInfluence);

        gl_Position = clip;
        float perspective = clamp(1.62 / max(0.16, clip.w), 0.36, 2.95);
        float seedPulse = 1.35 + fract(aSeed * 71.0) * 1.55;
        float flicker = 0.74 + fract(aSeed * 96.0) * 0.26;
        float pointSize = ((aSize * seedPulse * perspective) + (force * 16.0)) * uParticleScale * (1.04 + flicker * 0.18);
        gl_PointSize = max(2.0, pointSize);
        vDepth = clamp((clip.z / clip.w) * 0.5 + 0.5, 0.0, 1.0);
        vSpark = flicker * (1.0 - vDepth);
        vColor = aColor + vec3(force * 0.3 + uBloom * 0.06);
      }
    `;

    const fragmentSrc = `
      precision mediump float;
      varying vec3 vColor;
      varying float vDepth;
      varying float vSpark;
      uniform vec3 uFogColor;
      uniform float uBloom;
      uniform float uTrailAlpha;

      void main() {
        vec2 c = gl_PointCoord - vec2(0.5, 0.5);
        float d = dot(c, c);
        if (d > 0.25) {
          discard;
        }
        float edge = smoothstep(0.25, 0.0, d);
        float core = smoothstep(0.09, 0.0, d);
        float ring = smoothstep(0.19, 0.11, sqrt(d));
        vec3 depthTint = mix(vColor, uFogColor, pow(vDepth, 1.5) * (0.58 + uBloom * 0.2));
        vec3 sparkle = vec3(vSpark * (0.05 + core * 0.22));
        vec3 color = depthTint
          + vec3(core * (0.58 + uBloom * 0.36))
          + vec3(ring * (0.1 + uBloom * 0.08))
          + sparkle;
        float alpha = clamp((edge * (0.74 + uBloom * 0.26)) + (core * (0.56 + uBloom * 0.24)), 0.0, 1.0) * uTrailAlpha;
        gl_FragColor = vec4(min(color, vec3(1.0)), alpha);
      }
    `;

    const compile = (type: number, source: string): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const perspective = (
      out: Float32Array,
      fovy: number,
      aspect: number,
      near: number,
      far: number,
    ) => {
      const f = 1.0 / Math.tan(fovy / 2.0);
      out[0] = f / aspect;
      out[1] = 0;
      out[2] = 0;
      out[3] = 0;
      out[4] = 0;
      out[5] = f;
      out[6] = 0;
      out[7] = 0;
      out[8] = 0;
      out[9] = 0;
      out[10] = (far + near) / (near - far);
      out[11] = -1;
      out[12] = 0;
      out[13] = 0;
      out[14] = (2 * far * near) / (near - far);
      out[15] = 0;
    };

    const lookAt = (
      out: Float32Array,
      eyeX: number,
      eyeY: number,
      eyeZ: number,
      targetX: number,
      targetY: number,
      targetZ: number,
      upX: number,
      upY: number,
      upZ: number,
    ) => {
      let z0 = eyeX - targetX;
      let z1 = eyeY - targetY;
      let z2 = eyeZ - targetZ;
      let len = Math.hypot(z0, z1, z2);
      if (len < 0.000001) {
        z2 = 1;
        len = 1;
      }
      z0 /= len;
      z1 /= len;
      z2 /= len;

      let x0 = (upY * z2) - (upZ * z1);
      let x1 = (upZ * z0) - (upX * z2);
      let x2 = (upX * z1) - (upY * z0);
      len = Math.hypot(x0, x1, x2);
      if (len < 0.000001) {
        x0 = 1;
        x1 = 0;
        x2 = 0;
        len = 1;
      }
      x0 /= len;
      x1 /= len;
      x2 /= len;

      const y0 = (z1 * x2) - (z2 * x1);
      const y1 = (z2 * x0) - (z0 * x2);
      const y2 = (z0 * x1) - (z1 * x0);

      out[0] = x0;
      out[1] = y0;
      out[2] = z0;
      out[3] = 0;
      out[4] = x1;
      out[5] = y1;
      out[6] = z1;
      out[7] = 0;
      out[8] = x2;
      out[9] = y2;
      out[10] = z2;
      out[11] = 0;
      out[12] = -((x0 * eyeX) + (x1 * eyeY) + (x2 * eyeZ));
      out[13] = -((y0 * eyeX) + (y1 * eyeY) + (y2 * eyeZ));
      out[14] = -((z0 * eyeX) + (z1 * eyeY) + (z2 * eyeZ));
      out[15] = 1;
    };

    const resolveDepth = (x: number, y: number, zHint: number, idx: number): number => {
      const explicit = Number(zHint);
      if (Number.isFinite(explicit)) {
        return Math.max(-1, Math.min(1, explicit));
      }
      const noise = Math.sin((x * 15.37) + (y * 27.91) + (idx * 0.913)) * 43758.5453123;
      const layer = (noise - Math.floor(noise)) * 2 - 1;
      const radial = Math.min(1, Math.hypot(x, y));
      const spread = 0.88 - (radial * 0.3);
      return layer * spread;
    };

    const vs = compile(gl.VERTEX_SHADER, vertexSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fragmentSrc);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      return;
    }

    const posBuffer = gl.createBuffer();
    const sizeBuffer = gl.createBuffer();
    const colorBuffer = gl.createBuffer();
    const seedBuffer = gl.createBuffer();
    if (!posBuffer || !sizeBuffer || !colorBuffer || !seedBuffer) {
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      return;
    }

    const locPos = gl.getAttribLocation(program, "aPos");
    const locSize = gl.getAttribLocation(program, "aSize");
    const locColor = gl.getAttribLocation(program, "aColor");
    const locSeed = gl.getAttribLocation(program, "aSeed");
    const locTime = gl.getUniformLocation(program, "uTime");
    const locMouse = gl.getUniformLocation(program, "uMouse");
    const locInfluence = gl.getUniformLocation(program, "uInfluence");
    const locMotionRate = gl.getUniformLocation(program, "uMotionRate");
    const locParticleScale = gl.getUniformLocation(program, "uParticleScale");
    const locBloom = gl.getUniformLocation(program, "uBloom");
    const locTrailAlpha = gl.getUniformLocation(program, "uTrailAlpha");
    const locFogColor = gl.getUniformLocation(program, "uFogColor");
    const locProjection = gl.getUniformLocation(program, "uProjection");
    const locView = gl.getUniformLocation(program, "uView");
    const activateProgram = gl.useProgram.bind(gl);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.disable(gl.DEPTH_TEST);
    gl.clearDepth(1);

    let count = 0;
    let capacity = 0;
    let currentPositions = new Float32Array(0);
    let targetPositions = new Float32Array(0);
    let currentSizes = new Float32Array(0);
    let targetSizes = new Float32Array(0);
    let currentColors = new Float32Array(0);
    let targetColors = new Float32Array(0);
    let currentSeeds = new Float32Array(0);
    let targetSeeds = new Float32Array(0);
    let lastTick = 0;
    let rafId = 0;
    let viewportWidth = 0;
    let viewportHeight = 0;
    let mouseX = 0;
    let mouseY = 0;
    let cameraMouseX = 0;
    let cameraMouseY = 0;
    let influence = 0;
    let scenePulse = 0.26;
    let scenePulseTarget = 0.26;
    let dynamicCount = 0;
    let ambientCount = 0;
    let ambientDirty = true;
    type TrailSnapshot = {
      positions: Float32Array;
      count: number;
    };
    const TRAIL_LAYER_COUNT = 8;
    const trailSnapshots: TrailSnapshot[] = [];
    const trailSnapshotPool: Float32Array[] = [];
    const projectionMatrix = new Float32Array(16);
    const viewMatrix = new Float32Array(16);

    const recycleTrailSnapshot = (positions: Float32Array) => {
      if (trailSnapshotPool.length >= TRAIL_LAYER_COUNT * 3) {
        return;
      }
      trailSnapshotPool.push(positions);
    };

    const acquireTrailSnapshot = (requiredLength: number): Float32Array => {
      for (let poolIndex = trailSnapshotPool.length - 1; poolIndex >= 0; poolIndex -= 1) {
        const candidate = trailSnapshotPool[poolIndex];
        if (candidate.length < requiredLength) {
          continue;
        }
        trailSnapshotPool.splice(poolIndex, 1);
        return candidate;
      }
      return new Float32Array(requiredLength);
    };

    const randomNoise = (seed: number): number => {
      const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453123;
      return value - Math.floor(value);
    };

    const rebuildAmbientLayer = () => {
      if (ambientCount <= 0) {
        return;
      }
      for (let i = 0; i < ambientCount; i += 1) {
        const n0 = randomNoise(i + 1.73);
        const n1 = randomNoise(i + 19.21);
        const n2 = randomNoise(i + 64.44);
        const n3 = randomNoise(i + 92.18);
        const n4 = randomNoise(i + 133.54);
        const theta = n0 * Math.PI * 2;
        const phi = Math.acos((n1 * 2) - 1);
        const radius = 0.66 + (n2 * n2) * 1.48;
        const x = Math.sin(phi) * Math.cos(theta) * radius;
        const y = Math.cos(phi) * radius * (0.72 + n3 * 0.46);
        const z = Math.sin(phi) * Math.sin(theta) * radius;
        targetPositions[i * 3] = x;
        targetPositions[(i * 3) + 1] = y;
        targetPositions[(i * 3) + 2] = z;
        targetSizes[i] = 0.72 + n4 * 1.55;
        targetColors[i * 3] = 0.12 + n2 * 0.2;
        targetColors[(i * 3) + 1] = 0.18 + n3 * 0.26;
        targetColors[(i * 3) + 2] = 0.26 + n4 * 0.34;
        targetSeeds[i] = ((i + 1) * 0.61803398875) % 1;
        currentPositions[i * 3] = targetPositions[i * 3];
        currentPositions[(i * 3) + 1] = targetPositions[(i * 3) + 1];
        currentPositions[(i * 3) + 2] = targetPositions[(i * 3) + 2];
        currentSizes[i] = targetSizes[i];
        currentColors[i * 3] = targetColors[i * 3];
        currentColors[(i * 3) + 1] = targetColors[(i * 3) + 1];
        currentColors[(i * 3) + 2] = targetColors[(i * 3) + 2];
        currentSeeds[i] = targetSeeds[i];
      }
    };

    const ensureCapacity = (pointCount: number) => {
      if (pointCount <= capacity) return;
      capacity = Math.max(pointCount, capacity * 2, 64);
      currentPositions = new Float32Array(capacity * 3);
      targetPositions = new Float32Array(capacity * 3);
      currentSizes = new Float32Array(capacity);
      targetSizes = new Float32Array(capacity);
      currentColors = new Float32Array(capacity * 3);
      targetColors = new Float32Array(capacity * 3);
      currentSeeds = new Float32Array(capacity);
      targetSeeds = new Float32Array(capacity);
      gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, currentPositions.byteLength, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, currentSizes.byteLength, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, currentColors.byteLength, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, seedBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, currentSeeds.byteLength, gl.DYNAMIC_DRAW);
    };

    const pushTrailSnapshot = () => {
      if (count <= 0) {
        while (trailSnapshots.length > 0) {
          const retired = trailSnapshots.pop();
          if (retired) {
            recycleTrailSnapshot(retired.positions);
          }
        }
        trailSnapshots.length = 0;
        return;
      }
      const requiredLength = count * 3;
      const snapshot = acquireTrailSnapshot(requiredLength);
      snapshot.set(targetPositions.subarray(0, requiredLength));
      trailSnapshots.push({
        positions: snapshot,
        count,
      });
      if (trailSnapshots.length > TRAIL_LAYER_COUNT) {
        const retired = trailSnapshots.shift();
        if (retired) {
          recycleTrailSnapshot(retired.positions);
        }
      }
    };

    const draw = (ts: number) => {
      const frameMs = lastTick > 0
        ? clampValue(ts - lastTick, 4, 48)
        : 16.6666667;
      lastTick = ts;
      const frameScale = frameMs / 16.6666667;
      const cameraLerp = 1 - Math.pow(1 - 0.08, frameScale);
      const scenePulseLerp = 1 - Math.pow(1 - 0.06, frameScale);
      for (let i = 0; i < count * 3; i += 1) {
        currentPositions[i] = targetPositions[i];
      }
      for (let i = 0; i < count; i += 1) {
        currentSizes[i] = targetSizes[i];
      }
      for (let i = 0; i < count * 3; i += 1) {
        currentColors[i] = targetColors[i];
      }
      for (let i = 0; i < count; i += 1) {
        currentSeeds[i] = targetSeeds[i];
      }
      cameraMouseX += (mouseX - cameraMouseX) * cameraLerp;
      cameraMouseY += (mouseY - cameraMouseY) * cameraLerp;
      scenePulse += (scenePulseTarget - scenePulse) * scenePulseLerp;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const nextWidth = Math.max(1, Math.floor(rect.width * dpr));
      const nextHeight = Math.max(1, Math.floor(rect.height * dpr));
      if (nextWidth !== viewportWidth || nextHeight !== viewportHeight) {
        viewportWidth = nextWidth;
        viewportHeight = nextHeight;
        canvas.width = viewportWidth;
        canvas.height = viewportHeight;
      }

      gl.viewport(0, 0, viewportWidth, viewportHeight);
      const fogR = 0.008 + scenePulse * 0.026;
      const fogG = 0.026 + scenePulse * 0.048;
      const fogB = 0.044 + scenePulse * 0.082;
      gl.clearColor(fogR, fogG, fogB, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      // Disable any previously enabled attributes to prevent state leakage
      gl.disableVertexAttribArray(0);
      gl.disableVertexAttribArray(1);
      gl.disableVertexAttribArray(2);
      gl.disableVertexAttribArray(3);

      if (count > 0 && locPos >= 0 && locSize >= 0 && locColor >= 0 && locSeed >= 0) {
        const motionRate = motionSpeedRef.current;
        const particlePointScale = particleScaleRef.current;
        const pointerInfluence = mouseInfluenceRef.current;
        const bloom = Math.min(1, 0.22 + scenePulse * 0.72 + influence * 0.14);
        const aspect = viewportWidth / Math.max(1, viewportHeight);
        perspective(projectionMatrix, (58 * Math.PI) / 180, aspect, 0.08, 24);

        const radius = (2.18 + Math.min(1.36, count / 1520)) - scenePulse * 0.24;
        const eyeX = cameraMouseX * (0.18 + scenePulse * 0.12);
        const eyeY = 0.26 + (cameraMouseY * 0.22);
        const eyeZ = radius;
        const targetX = cameraMouseX * 0.24;
        const targetY = cameraMouseY * 0.14;
        lookAt(viewMatrix, eyeX, eyeY, eyeZ, targetX, targetY, 0, 0, 1, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, currentPositions.subarray(0, count * 3));
        gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, currentSizes.subarray(0, count));
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, currentColors.subarray(0, count * 3));
        gl.bindBuffer(gl.ARRAY_BUFFER, seedBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, currentSeeds.subarray(0, count));

        activateProgram(program);
        gl.uniform1f(locTime, ts || 0);
        gl.uniform2f(locMouse, cameraMouseX, cameraMouseY);
        gl.uniform1f(locInfluence, influence * pointerInfluence);
        gl.uniform1f(locMotionRate, motionRate);
        gl.uniform1f(locParticleScale, particlePointScale);
        gl.uniform1f(locBloom, bloom);
        gl.uniform3f(locFogColor, fogR, fogG, fogB);
        gl.uniformMatrix4fv(locProjection, false, projectionMatrix);
        gl.uniformMatrix4fv(locView, false, viewMatrix);

        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.enableVertexAttribArray(locPos);
        gl.vertexAttribPointer(locPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
        gl.enableVertexAttribArray(locSize);
        gl.vertexAttribPointer(locSize, 1, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.enableVertexAttribArray(locColor);
        gl.vertexAttribPointer(locColor, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, seedBuffer);
        gl.enableVertexAttribArray(locSeed);
        gl.vertexAttribPointer(locSeed, 1, gl.FLOAT, false, 0, 0);

        if (trailSnapshots.length <= 0) {
          gl.uniform1f(locTrailAlpha, 1);
          gl.drawArrays(gl.POINTS, 0, count);
        } else {
          const totalLayers = trailSnapshots.length;
          for (let layerIndex = 0; layerIndex < totalLayers; layerIndex += 1) {
            const layer = trailSnapshots[layerIndex];
            const layerCount = Math.max(0, Math.min(count, layer.count));
            if (layerCount <= 0) {
              continue;
            }
            const layerProgress = (layerIndex + 1) / totalLayers;
            const layerAlpha = 0.12 + (layerProgress * 0.88);
            gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, layer.positions.subarray(0, layerCount * 3));
            gl.uniform1f(locTrailAlpha, layerAlpha);
            gl.drawArrays(gl.POINTS, 0, layerCount);
          }
        }

        // Disable attributes after drawing
        gl.disableVertexAttribArray(locPos);
        gl.disableVertexAttribArray(locSize);
        gl.disableVertexAttribArray(locColor);
        gl.disableVertexAttribArray(locSeed);
      }

      rafId = requestAnimationFrame(draw);
    };

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return;
      }
      mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseY = -((((e.clientY - rect.top) / rect.height) * 2) - 1);
      influence = Math.min(1.6, 0.68 + mouseInfluenceRef.current * 0.52);
    };
    window.addEventListener("mousemove", onMove);
    const decay = setInterval(() => { influence *= 0.95; }, 50);
    rafId = requestAnimationFrame(draw);

    (canvas as any).__updateSim = (state: SimulationState) => {
      const overlayRows = renderRichOverlayParticles
        ? (
            (Array.isArray(state.presence_dynamics?.field_particles)
              ? state.presence_dynamics?.field_particles
              : null)
            ??
            (Array.isArray(state.field_particles)
              ? state.field_particles
              : [])
          )
        : [];
      const hasOverlayParticles = overlayRows.length > 0;
      const sourceRows = hasOverlayParticles
        ? (renderOverlayWithWebgl ? overlayRows : [])
        : (Array.isArray(state.points) ? state.points : []);
      const sourceCount = sourceRows.length;
      const desiredAmbient = hasOverlayParticles
        ? 0
        : Math.max(72, Math.round(96 + particleDensityRef.current * 220));
      if (desiredAmbient !== ambientCount) {
        ambientCount = desiredAmbient;
        ambientDirty = true;
      }
      dynamicCount = sourceCount <= 0
        ? 0
        : Math.max(
          Math.min(sourceCount, 220),
          Math.max(1, Math.min(sourceCount, Math.round(sourceCount * particleDensityRef.current))),
        );
      count = ambientCount + dynamicCount;
      ensureCapacity(count);

      if (ambientDirty || lastTick === 0) {
        rebuildAmbientLayer();
        ambientDirty = false;
      }

      const dynamicOffset = ambientCount;
      for (let i = 0; i < dynamicCount; i += 1) {
        const sourceIndex = sourceCount <= dynamicCount
          ? i
          : Math.floor((i * sourceCount) / dynamicCount);
        const row = sourceRows[sourceIndex] as any;
        const x = clampValue(Number(row?.x ?? 0), -1, 1);
        const y = clampValue(Number(row?.y ?? 0), -1, 1);
        const size = clampValue(Number(row?.size ?? 1.5), 0.2, 4.8);
        const r = clamp01(Number(row?.r ?? 0.48));
        const g = clamp01(Number(row?.g ?? 0.64));
        const b = clamp01(Number(row?.b ?? 0.92));
        const zHint = Number(
          row?.z
          ?? row?.mass
          ?? row?.route_probability
          ?? row?.drift_score
          ?? Number.NaN,
        );
        const z = resolveDepth(x, y, zHint, sourceIndex);
        const writeIndex = dynamicOffset + i;
        targetPositions[writeIndex * 3] = x;
        targetPositions[(writeIndex * 3) + 1] = y;
        targetPositions[(writeIndex * 3) + 2] = z;
        targetSizes[writeIndex] = size;
        targetColors[writeIndex * 3] = r;
        targetColors[(writeIndex * 3) + 1] = g;
        targetColors[(writeIndex * 3) + 2] = b;
        const seed = ((sourceIndex + 1) * 0.754877666) % 1;
        targetSeeds[writeIndex] = seed;
        if (lastTick === 0) {
          currentPositions[writeIndex * 3] = targetPositions[writeIndex * 3];
          currentPositions[(writeIndex * 3) + 1] = targetPositions[(writeIndex * 3) + 1];
          currentPositions[(writeIndex * 3) + 2] = targetPositions[(writeIndex * 3) + 2];
          currentSizes[writeIndex] = targetSizes[writeIndex];
          currentColors[writeIndex * 3] = targetColors[writeIndex * 3];
          currentColors[(writeIndex * 3) + 1] = targetColors[(writeIndex * 3) + 1];
          currentColors[(writeIndex * 3) + 2] = targetColors[(writeIndex * 3) + 2];
          currentSeeds[writeIndex] = seed;
        }
      }

      pushTrailSnapshot();

      const flowRate = state.presence_dynamics?.river_flow?.rate;
      const forkTaxBalance = state.presence_dynamics?.fork_tax?.balance;
      const witnessContinuity = state.presence_dynamics?.witness_thread?.continuity_index;
      const witnessTrace = state.presence_dynamics?.witness_thread?.lineage?.[0]?.ref;
      const probabilisticSummary = state.presence_dynamics?.daimoi_probabilistic;
      const graphRuntimeSummary = probabilisticSummary?.graph_runtime;
      const nooiField = state.presence_dynamics?.nooi_field;
      const resourceDaimoiSummary = state.presence_dynamics?.resource_daimoi ?? probabilisticSummary?.resource_daimoi;
      const resourceConsumptionSummary = state.presence_dynamics?.resource_consumption ?? probabilisticSummary?.resource_consumption;
      const routeMean = Number(probabilisticSummary?.mean_route_probability ?? Number.NaN);
      const driftMean = Number(probabilisticSummary?.mean_drift_score ?? Number.NaN);
      const influenceMean = Number(probabilisticSummary?.mean_influence_power ?? Number.NaN);
      const priceMean = Number(graphRuntimeSummary?.price_mean ?? Number.NaN);
      const nooiCells = Number(nooiField?.active_cells ?? Number.NaN);
      const nooiInfluenceMean = Number(nooiField?.mean_influence ?? Number.NaN);
      const resourcePackets = Number(resourceDaimoiSummary?.delivered_packets ?? Number.NaN);
      const resourceTransfer = Number(resourceDaimoiSummary?.total_transfer ?? Number.NaN);
      const resourceConsumed = Number(resourceConsumptionSummary?.consumed_total ?? Number.NaN);
      const resourceBlocked = Number(resourceConsumptionSummary?.blocked_packets ?? Number.NaN);
      const resourceStarved = Number.isFinite(resourceBlocked)
        ? Math.max(0, Math.round(resourceBlocked))
        : Number.NaN;
      const truthState = state.truth_state ?? catalogRef.current?.truth_state;
      const truthClaim = truthState?.claim;
      const truthStatus = String(truthClaim?.status ?? "undecided");
      const truthKappa = Number(truthClaim?.kappa ?? 0);
      const flowNorm = clamp01(Number(flowRate ?? 0) / 2.8);
      const witnessNorm = clamp01(Number(witnessContinuity ?? 0));
      const truthNorm = truthState
        ? (truthStatus === "proved" ? 1 : truthStatus === "refuted" ? 0.34 : 0.58)
        : 0.46;
      const audioNorm = clamp01(Number(state.audio ?? 0) / Math.max(1, Number(state.total ?? 1)));
      const richnessNorm = clamp01(dynamicCount / 2200);
      scenePulseTarget = clampValue(
        0.18 + flowNorm * 0.24 + witnessNorm * 0.22 + truthNorm * 0.16 + audioNorm * 0.12 + richnessNorm * 0.12,
        0.14,
        1,
      );
      const fileGraph = state.file_graph ?? catalogRef.current?.file_graph;
      const inboxPending = fileGraph?.inbox?.pending_count;
      const particleLabel = hasOverlayParticles
        ? `field:${overlayRows.length}`
        : (sourceCount > dynamicCount ? `${dynamicCount}/${sourceCount}` : `${dynamicCount}`);
      const ambientLabel = ambientCount > 0 ? ` +${ambientCount} haze` : "";
      if (metaRef.current) {
        const inboxLabel = inboxPending !== undefined ? ` | inbox: ${inboxPending}` : "";
        const truthLabel = truthState
          ? ` | truth: ${truthStatus} k=${truthKappa.toFixed(2)}`
          : "";
        const graphRuntimeLabel = Number.isFinite(routeMean) || Number.isFinite(driftMean) || Number.isFinite(priceMean)
          ? (
              " | graph:" +
              `${Number.isFinite(routeMean) ? ` route ${Math.round(clamp01(routeMean) * 100)}%` : ""}` +
              `${Number.isFinite(driftMean) ? ` drift ${driftMean >= 0 ? "+" : ""}${clampValue(driftMean, -1, 1).toFixed(2)}` : ""}` +
              `${Number.isFinite(influenceMean) ? ` infl ${clamp01(Math.max(0, influenceMean)).toFixed(2)}` : ""}` +
              `${Number.isFinite(priceMean) ? ` price ${Math.max(0, priceMean).toFixed(2)}` : ""}`
            )
          : "";
        const nooiLabel = Number.isFinite(nooiCells) || Number.isFinite(nooiInfluenceMean)
          ? (
              " | nooi:" +
              `${Number.isFinite(nooiCells) ? ` cells ${Math.max(0, Math.round(nooiCells))}` : ""}` +
              `${Number.isFinite(nooiInfluenceMean) ? ` infl ${clamp01(Math.max(0, nooiInfluenceMean)).toFixed(2)}` : ""}`
            )
          : "";
        const resourceLabel = Number.isFinite(resourcePackets)
          || Number.isFinite(resourceTransfer)
          || Number.isFinite(resourceConsumed)
          || Number.isFinite(resourceBlocked)
          ? (
              " | resource:" +
              `${Number.isFinite(resourcePackets) ? ` packets ${Math.max(0, Math.round(resourcePackets))}` : ""}` +
              `${Number.isFinite(resourceTransfer) ? ` transfer ${Math.max(0, resourceTransfer).toFixed(2)}` : ""}` +
              `${Number.isFinite(resourceConsumed) ? ` consume ${Math.max(0, resourceConsumed).toFixed(2)}` : ""}` +
              `${Number.isFinite(resourceBlocked) ? ` blocked ${Math.max(0, Math.round(resourceBlocked))}` : ""}` +
              `${Number.isFinite(resourceStarved) ? ` starved ${Math.max(0, Math.round(resourceStarved))}` : ""}`
            )
          : "";
        if (flowRate !== undefined || forkTaxBalance !== undefined) {
          metaRef.current.textContent =
            "sim particles: " +
            particleLabel +
            ambientLabel +
            " | audio: " +
            state.audio +
            " | river: " +
            (flowRate ?? 0).toFixed(2) +
            " | fork-tax: " +
            Math.round(forkTaxBalance ?? 0) +
            " | witness: " +
            Math.round((witnessContinuity ?? 0) * 100) +
            "%" +
            (witnessTrace ? " [" + witnessTrace + "]" : "") +
            inboxLabel +
            truthLabel +
            graphRuntimeLabel +
            nooiLabel +
            resourceLabel;
        } else {
          metaRef.current.textContent =
            "sim particles: " +
            particleLabel +
            ambientLabel +
            " | audio: " +
            state.audio +
            inboxLabel +
            truthLabel +
            graphRuntimeLabel +
            nooiLabel +
            resourceLabel;
        }
      }
    };

    return () => {
      window.removeEventListener("mousemove", onMove);
      clearInterval(decay);
      cancelAnimationFrame(rafId);
      while (trailSnapshots.length > 0) {
        const retired = trailSnapshots.pop();
        if (retired) {
          recycleTrailSnapshot(retired.positions);
        }
      }
      trailSnapshotPool.length = 0;
      gl.deleteBuffer(posBuffer);
      gl.deleteBuffer(sizeBuffer);
      gl.deleteBuffer(colorBuffer);
      gl.deleteBuffer(seedBuffer);
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
    };
  }, []);

  useEffect(() => {
    if(simulation && canvasRef.current) (canvasRef.current as any).__updateSim?.(simulation);
  }, [simulation]);

  useEffect(() => {
    const density = particleDensity;
    if (!canvasRef.current || !simulationRef.current) {
      return;
    }
    if (!Number.isFinite(density)) {
      return;
    }
    (canvasRef.current as any).__updateSim?.(simulationRef.current);
  }, [particleDensity]);

  useEffect(() => {
    if (!renderOverlayWithWebgl) {
      return;
    }
    const canvas = overlayRef.current;
    if (!canvas) {
      return;
    }
    const gl = canvas.getContext("webgl", { alpha: true, antialias: true, premultipliedAlpha: true });
    if (!gl) {
      return;
    }

    const pointVertexShaderSource = `
      attribute vec2 aPos;
      attribute float aSize;
      attribute vec4 aColor;
      uniform vec2 uResolution;
      uniform float uAlphaScale;
      varying vec4 vColor;

      void main() {
        vec2 clip = vec2(
          (aPos.x / max(1.0, uResolution.x)) * 2.0 - 1.0,
          1.0 - (aPos.y / max(1.0, uResolution.y)) * 2.0
        );
        gl_Position = vec4(clip, 0.0, 1.0);
        gl_PointSize = aSize;
        vColor = vec4(aColor.rgb, clamp(aColor.a * uAlphaScale, 0.0, 1.0));
      }
    `;

    const pointFragmentShaderSource = `
      precision mediump float;
      varying vec4 vColor;

      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float distSq = dot(p, p);
        if (distSq > 1.0) {
          discard;
        }
        float alpha = (1.0 - distSq) * vColor.a;
        gl_FragColor = vec4(vColor.rgb, alpha);
      }
    `;

    const lineVertexShaderSource = `
      attribute vec2 aPos;
      attribute vec4 aColor;
      uniform vec2 uResolution;
      varying vec4 vColor;

      void main() {
        vec2 clip = vec2(
          (aPos.x / max(1.0, uResolution.x)) * 2.0 - 1.0,
          1.0 - (aPos.y / max(1.0, uResolution.y)) * 2.0
        );
        gl_Position = vec4(clip, 0.0, 1.0);
        vColor = aColor;
      }
    `;

    const lineFragmentShaderSource = `
      precision mediump float;
      varying vec4 vColor;

      void main() {
        gl_FragColor = vColor;
      }
    `;

    const compileShader = (type: number, source: string): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (!shader) {
        return null;
      }
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const createProgram = (vertexSource: string, fragmentSource: string): {
      program: WebGLProgram;
      vertexShader: WebGLShader;
      fragmentShader: WebGLShader;
    } | null => {
      const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
      const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
      if (!vertexShader || !fragmentShader) {
        if (vertexShader) {
          gl.deleteShader(vertexShader);
        }
        if (fragmentShader) {
          gl.deleteShader(fragmentShader);
        }
        return null;
      }
      const program = gl.createProgram();
      if (!program) {
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragmentShader);
        return null;
      }
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        gl.deleteProgram(program);
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragmentShader);
        return null;
      }
      return {
        program,
        vertexShader,
        fragmentShader,
      };
    };

    const pointProgramBundle = createProgram(pointVertexShaderSource, pointFragmentShaderSource);
    const lineProgramBundle = createProgram(lineVertexShaderSource, lineFragmentShaderSource);
    if (!pointProgramBundle || !lineProgramBundle) {
      if (pointProgramBundle) {
        gl.deleteProgram(pointProgramBundle.program);
        gl.deleteShader(pointProgramBundle.vertexShader);
        gl.deleteShader(pointProgramBundle.fragmentShader);
      }
      if (lineProgramBundle) {
        gl.deleteProgram(lineProgramBundle.program);
        gl.deleteShader(lineProgramBundle.vertexShader);
        gl.deleteShader(lineProgramBundle.fragmentShader);
      }
      return;
    }

    const pointProgram = pointProgramBundle.program;
    const lineProgram = lineProgramBundle.program;
    const pointBuffer = gl.createBuffer();
    const lineBuffer = gl.createBuffer();
    if (!pointBuffer || !lineBuffer) {
      gl.deleteBuffer(pointBuffer);
      gl.deleteBuffer(lineBuffer);
      gl.deleteProgram(pointProgramBundle.program);
      gl.deleteShader(pointProgramBundle.vertexShader);
      gl.deleteShader(pointProgramBundle.fragmentShader);
      gl.deleteProgram(lineProgramBundle.program);
      gl.deleteShader(lineProgramBundle.vertexShader);
      gl.deleteShader(lineProgramBundle.fragmentShader);
      return;
    }

    const pointLocPos = gl.getAttribLocation(pointProgram, "aPos");
    const pointLocSize = gl.getAttribLocation(pointProgram, "aSize");
    const pointLocColor = gl.getAttribLocation(pointProgram, "aColor");
    const pointLocResolution = gl.getUniformLocation(pointProgram, "uResolution");
    const pointLocAlphaScale = gl.getUniformLocation(pointProgram, "uAlphaScale");
    const lineLocPos = gl.getAttribLocation(lineProgram, "aPos");
    const lineLocColor = gl.getAttribLocation(lineProgram, "aColor");
    const lineLocResolution = gl.getUniformLocation(lineProgram, "uResolution");
    const activatePointProgram = gl.useProgram.bind(gl, pointProgram);
    const activateLineProgram = gl.useProgram.bind(gl, lineProgram);

    interface OverlayHotspot {
      id: string;
      kind: "presence" | "file" | "crawler" | "nexus";
      label: string;
      x: number;
      y: number;
      radius: number;
      radiusNorm: number;
      node?: any;
      nodeKind?: "file" | "crawler" | "nexus";
      nodeType?: string;
      resourceKind?: GraphNodeResourceKind;
      isMusicNexus?: boolean;
      isProjectionOverflow?: boolean;
      isTrueGraph?: boolean;
    }

    const fallbackNamedForms = [
      { id: "receipt_river", en: "Receipt River", ja: "領収書の川", hue: 212, x: 0.22, y: 0.38 },
      { id: "witness_thread", en: "Witness Thread", ja: "証人の糸", hue: 262, x: 0.63, y: 0.33 },
      { id: "fork_tax_canticle", en: "Fork Tax Canticle", ja: "フォーク税の聖歌", hue: 34, x: 0.44, y: 0.62 },
      { id: "mage_of_receipts", en: "Mage of Receipts", ja: "領収魔導師", hue: 286, x: 0.33, y: 0.71 },
      { id: "keeper_of_receipts", en: "Keeper of Receipts", ja: "領収書の番人", hue: 124, x: 0.57, y: 0.72 },
      { id: "anchor_registry", en: "Anchor Registry", ja: "錨台帳", hue: 184, x: 0.49, y: 0.5 },
      { id: "gates_of_truth", en: "Gates of Truth", ja: "真理の門", hue: 52, x: 0.76, y: 0.54 },
    ];

    const toRgbFromHue = (hue: number): [number, number, number] => {
      const h = ((hue % 360) + 360) % 360;
      const c = 0.72;
      const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
      const m = 0.16;
      let r = 0;
      let g = 0;
      let b = 0;
      if (h < 60) {
        r = c;
        g = x;
      } else if (h < 120) {
        r = x;
        g = c;
      } else if (h < 180) {
        g = c;
        b = x;
      } else if (h < 240) {
        g = x;
        b = c;
      } else if (h < 300) {
        r = x;
        b = c;
      } else {
        r = c;
        b = x;
      }
      return [r + m, g + m, b + m];
    };

    const toRatio = (value: number): number => {
      if (!Number.isFinite(value)) {
        return 0.5;
      }
      if (value >= 0 && value <= 1) {
        return value;
      }
      return clamp01((value + 1) * 0.5);
    };

    const resourceColor = (kind: GraphNodeResourceKind): [number, number, number] => {
      if (kind === "text") return [1.0, 0.84, 0.44];
      if (kind === "image") return [1.0, 0.53, 0.83];
      if (kind === "audio") return [0.5, 0.94, 1.0];
      if (kind === "video") return [1.0, 0.66, 0.43];
      if (kind === "website") return [0.62, 0.92, 0.73];
      if (kind === "link") return [0.55, 0.82, 1.0];
      if (kind === "archive") return [0.95, 0.68, 0.38];
      return [0.67, 0.74, 0.82];
    };

    const edgeColorByKind = (kind: string): [number, number, number, number] => {
      if (kind === "citation") return [1.0, 0.72, 0.42, 0.28];
      if (kind === "cross_reference") return [0.96, 0.53, 0.86, 0.26];
      if (kind === "paper_pdf") return [0.54, 0.9, 0.96, 0.24];
      if (kind === "canonical_redirect") return [0.86, 0.57, 0.96, 0.26];
      if (kind === "domain_membership") return [0.96, 0.8, 0.52, 0.2];
      return [0.54, 0.74, 0.92, 0.16];
    };

    const resolveNamedFormsForWebgl = () => {
      const manifestRows = Array.isArray(catalogRef.current?.entity_manifest)
        ? (catalogRef.current?.entity_manifest as Array<any>)
        : [];
      const baseRows = manifestRows.length > 0 ? manifestRows : fallbackNamedForms;
      const seen = new Set<string>();
      const rows = baseRows
        .map((raw: any) => {
          const id = canonicalPresenceId(String(raw?.id ?? "").trim());
          if (!id || seen.has(id)) {
            return null;
          }
          seen.add(id);
          return {
            id,
            en: String(raw?.en ?? raw?.label ?? shortPresenceIdLabel(id)).trim() || shortPresenceIdLabel(id),
            ja: String(raw?.ja ?? raw?.label_ja ?? "presence").trim() || "presence",
            x: clamp01(Number(raw?.x ?? 0.5)),
            y: clamp01(Number(raw?.y ?? 0.5)),
            hue: Number.isFinite(Number(raw?.hue)) ? Number(raw?.hue) : presenceHueFromId(id),
          };
        })
        .filter((row): row is { id: string; en: string; ja: string; x: number; y: number; hue: number } => row !== null);
      const simulationAnchors = normalizePresenceAnchorPositionMap(
        simulationRef.current?.presence_dynamics?.presence_anchor_positions,
      );
      const withBackendAnchors = rows.map((row) => {
        const backend = simulationAnchors.get(row.id);
        if (!backend) {
          return row;
        }
        return {
          ...row,
          x: backend.x,
          y: backend.y,
        };
      });
      const sourceRows = withBackendAnchors.length > 0 ? withBackendAnchors : fallbackNamedForms;
      return normalizePresenceAnchors(sourceRows as any);
    };

    let rafId = 0;
    let lastPaintTs = 0;
    let lastLowPowerSimulationTimestamp = "";
    let canvasWidth = 0;
    let canvasHeight = 0;
    const hotspots: OverlayHotspot[] = [];
    let userPresenceMouseEmitMs = 0;
    let pulse = { x: 0.5, y: 0.5, power: 0, atMs: 0, target: "particle_field" };
    let pointerField = { x: 0.5, y: 0.5, power: 0, inside: false };
    const PARTICLE_TRAIL_FRAME_COUNT = 12;
    const particleTrailFrames: number[][] = [];
    const particleTrailFramePool: number[][] = [];
    let lastTrailFrameKey = "";
    let lastNodeTitleOverlaySyncMs = 0;
    let graphNodePositionMapCacheKey = "";
    let graphNodePositionMap = new Map<string, { x: number; y: number }>();
    let graphNodeSmoothingPruneAtMs = 0;
    const smoothedGraphNodeStateById = new Map<string, { x: number; y: number; seenAtMs: number }>();
    let simulationTimestampCacheKey = "";
    let simulationTimestampMs = 0;
    let particleSmoothingPruneAtMs = 0;
    const smoothedParticleStateById = new Map<string, { x: number; y: number; seenAtMs: number }>();
    const smoothedParticleRows: BackendFieldParticle[] = [];
    const smoothedParticleX: number[] = [];
    const smoothedParticleY: number[] = [];
    const pointRows: number[] = [];
    const lineRows: number[] = [];
    const presencePointRowsCurrent: number[] = [];
    const namedFormRows: number[] = [];
    const livePresenceCentroids = new Map<string, { sumX: number; sumY: number; count: number }>();
    const graphNodeLookup = new Map<
      string,
      {
        x: number;
        y: number;
        node: any;
        nodeKind: "file" | "crawler" | "nexus";
        nodeType: string;
        isProjectionOverflow: boolean;
      }
    >();
    const daimoiFlowLanes = new Map<
      string,
      {
        sourceX: number;
        sourceY: number;
        targetX: number;
        targetY: number;
        count: number;
        score: number;
        seed: number;
        resourceTypeCounts: Record<string, number>;
      }
    >();
    const activeFlowLaneRows: Array<{
      sourceX: number;
      sourceY: number;
      targetX: number;
      targetY: number;
      count: number;
      score: number;
      seed: number;
      resourceTypeCounts: Record<string, number>;
    }> = [];
    const sortedPresenceCentroids: Array<[string, { sumX: number; sumY: number; count: number }]> = [];
    const seenNodeIds = new Set<string>();
    let uploadPointBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);
    let uploadLineBuffer: Float32Array<ArrayBufferLike> = new Float32Array(0);

    const ensureUploadBuffer = (
      buffer: Float32Array<ArrayBufferLike>,
      required: number,
    ): Float32Array<ArrayBufferLike> => {
      if (buffer.length >= required) {
        return buffer;
      }
      const nextSize = Math.max(required, Math.max(256, buffer.length * 2));
      return new Float32Array(nextSize);
    };

    const selectGraphNodeTitleOverlays = (
      sourceHotspots: OverlayHotspot[],
      options: {
        showFileLayer: boolean;
        showTrueGraphLayer: boolean;
        showCrawlerLayer: boolean;
        showPresenceLayer: boolean;
        interactiveEnabled: boolean;
      },
    ): GraphNodeTitleOverlay[] => {
      if (!options.interactiveEnabled) {
        return [];
      }
      const rows = sourceHotspots
        .filter((row) => (
          (row.kind === "file" && (options.showFileLayer || options.showTrueGraphLayer))
          || (row.kind === "crawler" && options.showCrawlerLayer)
          || (row.kind === "presence" && options.showPresenceLayer)
          || (row.kind === "nexus" && (options.showFileLayer || options.showTrueGraphLayer || options.showCrawlerLayer))
        ))
        .map((row) => {
          const importance = row.kind === "presence"
            ? clamp01((row.radius / 0.022) * 0.9)
            : clamp01(Number(row.node?.importance ?? (row.kind === "file" ? 0.36 : 0.28)));
          const kindBoost = row.kind === "file"
            ? 0.16
            : row.kind === "crawler"
              ? 0.08
              : row.kind === "nexus"
                ? 0.11
                : 0.05;
          const textBoost = row.resourceKind === "text" ? 0.08 : 0;
          const musicBoost = row.isMusicNexus ? 0.2 : 0;
          const score = clampValue(importance + kindBoost + textBoost + musicBoost, 0, 2);
          return {
            id: row.id,
            kind: row.kind,
            label: row.label,
            x: clamp01(row.x),
            y: clamp01(row.y),
            isTrueGraph: Boolean(row.isTrueGraph),
            isProjectionOverflow: Boolean(row.isProjectionOverflow),
            score,
          };
        })
        .sort((left, right) => right.score - left.score || left.label.length - right.label.length);

      const maxLabels = rows.length;
      const selected: Array<GraphNodeTitleOverlay & { halfWidth: number }> = [];
      for (let index = 0; index < rows.length && selected.length < maxLabels; index += 1) {
        const row = rows[index];
        const halfWidth = clampValue(0.04 + (row.label.length * 0.0028), 0.05, 0.16);
        const centerX = clampValue(row.x, 0.06, 0.94);
        const centerY = clampValue(row.y - (row.kind === "presence" ? 0.014 : 0.018), 0.07, 0.95);
        selected.push({
          id: row.id,
          label: row.label,
          x: centerX,
          y: centerY,
          kind: row.kind,
          isTrueGraph: row.isTrueGraph,
          isProjectionOverflow: row.isProjectionOverflow,
          halfWidth,
        });
      }

      return selected.map((row) => ({
        id: row.id,
        label: row.label,
        x: row.x,
        y: row.y,
        kind: row.kind,
        isTrueGraph: row.isTrueGraph,
        isProjectionOverflow: row.isProjectionOverflow,
      }));
    };

    const findNearestHotspotAt = (xRatio: number, yRatio: number): { row: OverlayHotspot | null; distance: number } => {
      let match: { row: OverlayHotspot; distance: number } | null = null;
      for (const hit of hotspots) {
        const distance = Math.hypot(xRatio - hit.x, yRatio - hit.y);
        const threshold = hit.kind === "presence"
          ? Math.max(0.012, hit.radiusNorm * 1.8)
          : hit.kind === "nexus"
            ? hit.radiusNorm * 1.42
            : hit.radiusNorm * 1.35;
        if (distance > threshold) {
          continue;
        }
        if (!match || distance < match.distance) {
          match = { row: hit, distance };
        }
      }
      return {
        row: match?.row ?? null,
        distance: match?.distance ?? Number.POSITIVE_INFINITY,
      };
    };

    const resolveInteractionHotspot = (xRatioInput: number, yRatioInput: number): {
      hit: OverlayHotspot | null;
      xRatio: number;
      yRatio: number;
    } => {
      const xRatio = clamp01(xRatioInput);
      const yRatio = clamp01(yRatioInput);
      const direct = findNearestHotspotAt(xRatio, yRatio);

      return {
        hit: direct.row,
        xRatio,
        yRatio,
      };
    };

    const shouldOpenWorldscreen = (nodeKind: "file" | "crawler" | "nexus", nodeId: string): boolean => {
      const key = `${nodeKind}:${nodeId}`;
      const nowMs = performance.now();
      const previous = lastNexusPointerTapRef.current;
      const isDoubleTap = Boolean(previous && previous.key === key && (nowMs - previous.atMs) <= 360);
      lastNexusPointerTapRef.current = { key, atMs: nowMs };
      return isDoubleTap;
    };

    const openWorldscreenForNode = (node: any, nodeKind: "file" | "crawler" | "nexus", xRatio: number, yRatio: number) => {
      const resourceKind = resourceKindForNode(node);
      const currentFileGraph = simulationRef.current?.file_graph ?? catalogRef.current?.file_graph;
      const projectionBundle = resolveProjectionBundleManifest(node, currentFileGraph);
      const graphNodeId = String(node?.id ?? "").trim();
      const label = shortPathLabel(
        String(
          node?.title
          || node?.domain
          || node?.source_rel_path
          || node?.archived_rel_path
          || node?.name
          || node?.label
          || graphNodeId
          || "node",
        ),
      );
      const openUrl = openUrlForGraphNode(node, nodeKind);
      const domain = String(node?.domain ?? "").trim();
      const worldscreenUrl = resolveWorldscreenUrl(openUrl, nodeKind, domain) ?? "";
      const frameUrl = remoteFrameUrlForNode(node, worldscreenUrl, resourceKind);
      const imageRef = String(
        node?.source_rel_path
        || node?.archive_rel_path
        || node?.archived_rel_path
        || node?.url
        || worldscreenUrl
        || graphNodeId,
      ).trim();
      const commentRef = nexusCommentRefForNode(node, nodeKind, worldscreenUrl) || imageRef;
      const pinnedCenter = {
        x: clamp01(Number(glassCenterRatioRef.current?.x ?? xRatio)),
        y: clamp01(Number(glassCenterRatioRef.current?.y ?? yRatio)),
      };
      setWorldscreenPinnedCenterRatio(pinnedCenter);
      setWorldscreenMode("overview");
      setWorldscreen({
        nodeId: graphNodeId,
        commentRef,
        url: worldscreenUrl,
        imageRef,
        label,
        nodeKind,
        nodeTypeText: String(node?.node_type ?? "").trim(),
        nodeRoleText: String(node?.kind ?? node?.presence_kind ?? "").trim(),
        resourceKind,
        anchorRatioX: clamp01(xRatio),
        anchorRatioY: clamp01(yRatio),
        view: worldscreenViewForNode(node, nodeKind, resourceKind),
        subtitle: worldscreenSubtitleForNode(node, nodeKind, resourceKind),
        remoteFrameUrl: resourceKind === "image" ? (frameUrl || worldscreenUrl) : frameUrl,
        encounteredAt: timestampLabel(node?.encountered_at ?? node?.encounteredAt ?? ""),
        sourceUrl: String(node?.source_url ?? node?.sourceUrl ?? "").trim(),
        domain,
        titleText: String(node?.title ?? "").trim(),
        statusText: String(node?.status ?? "").trim(),
        contentTypeText: String(node?.content_type ?? node?.contentType ?? "").trim(),
        complianceText: String(node?.compliance ?? "").trim(),
        discoveredAt: timestampLabel(node?.discovered_at ?? node?.discoveredAt ?? ""),
        fetchedAt: timestampLabel(node?.fetched_at ?? node?.fetchedAt ?? node?.last_seen ?? node?.lastSeen ?? ""),
        summaryText: String(node?.summary ?? node?.text_excerpt ?? "").trim(),
        tagsText: joinListValues(node?.tags),
        labelsText: joinListValues(node?.labels),
        projectionGroupId: projectionBundle.groupId,
        projectionConsolidatedCount: Number(node?.consolidated_count ?? projectionBundle.members.length),
        projectionMemberManifest: projectionBundle.members,
      });
    };

    const draw = (ts: number) => {
      const currentSimulation = simulationRef.current;
      const simulationTimestamp = String(currentSimulation?.timestamp ?? "").trim();
      const directRows = currentSimulation?.presence_dynamics?.field_particles ?? currentSimulation?.field_particles;
      const allFieldParticles = Array.isArray(directRows) ? (directRows as BackendFieldParticle[]) : [];
      const isBackgroundMode = backgroundModeRef.current;
      const isInteractive = interactiveRef.current;
      const currentOverlayView = overlayViewRef.current;
      const currentLayerVisibility = layerVisibilityRef.current;
      const isLowPowerOverlay = !isInteractive && !isBackgroundMode;
      const trailFrameCountLimit = isLowPowerOverlay ? 6 : PARTICLE_TRAIL_FRAME_COUNT;
      const baselineFrameMs = allFieldParticles.length > 1500 ? 34 : allFieldParticles.length > 900 ? 24 : 16;
      const lowPowerDataUnchanged = isLowPowerOverlay
        && simulationTimestamp.length > 0
        && simulationTimestamp === lastLowPowerSimulationTimestamp;
      const targetFrameMs = isLowPowerOverlay
        ? (lowPowerDataUnchanged ? 180 : Math.max(56, baselineFrameMs * 1.6))
        : baselineFrameMs;
      const rect = canvas.getBoundingClientRect();
      const hasArea = rect.width > 0.5 && rect.height > 0.5;
      const docEl = document.documentElement;
      const viewportW = window.innerWidth || docEl.clientWidth || rect.width;
      const viewportH = window.innerHeight || docEl.clientHeight || rect.height;
      const offscreen = hasArea
        && (rect.bottom <= 0 || rect.top >= viewportH || rect.right <= 0 || rect.left >= viewportW);
      const effectiveFrameMs = offscreen
        ? (isLowPowerOverlay ? 220 : 96)
        : targetFrameMs;
      if (lastPaintTs > 0 && (ts - lastPaintTs) < effectiveFrameMs) {
        rafId = requestAnimationFrame(draw);
        return;
      }
      lastPaintTs = ts;

      if (!hasArea || offscreen) {
        if (!isLowPowerOverlay) {
          lastLowPowerSimulationTimestamp = "";
        }
        rafId = requestAnimationFrame(draw);
        return;
      }

      const dprRaw = window.devicePixelRatio || 1;
      const dpr = isLowPowerOverlay ? Math.min(1.25, dprRaw) : dprRaw;
      const nextWidth = Math.max(1, Math.floor(rect.width * dpr));
      const nextHeight = Math.max(1, Math.floor(rect.height * dpr));
      const skipLowPowerRepaint = lowPowerDataUnchanged
        && nextWidth === canvasWidth
        && nextHeight === canvasHeight;
      if (skipLowPowerRepaint) {
        rafId = requestAnimationFrame(draw);
        return;
      }

      if (simulationTimestamp !== simulationTimestampCacheKey) {
        simulationTimestampCacheKey = simulationTimestamp;
        const parsedMs = Date.parse(simulationTimestamp);
        simulationTimestampMs = Number.isFinite(parsedMs) ? parsedMs : 0;
      }

      const nowWallMs = Date.now();
      const snapshotAgeMs = simulationTimestampMs > 0
        ? Math.max(0, nowWallMs - simulationTimestampMs)
        : 0;
      const extrapolationSeconds = Math.min(0.45, snapshotAgeMs * 0.001);
      const smoothingAlpha = clampValue(0.18 + (effectiveFrameMs / 120), 0.18, 0.72);
      smoothedParticleRows.length = 0;
      smoothedParticleX.length = 0;
      smoothedParticleY.length = 0;

      for (let index = 0; index < allFieldParticles.length; index += 1) {
        const row = allFieldParticles[index] as BackendFieldParticle;
        const rowId = String(row?.id ?? row?.presence_id ?? `particle:${index}`).trim() || `particle:${index}`;
        const baseX = toRatio(Number(row?.x ?? 0.5));
        const baseY = toRatio(Number(row?.y ?? 0.5));
        const vx = Number.isFinite(Number(row?.vx)) ? Number(row?.vx ?? 0) : 0;
        const vy = Number.isFinite(Number(row?.vy)) ? Number(row?.vy ?? 0) : 0;
        const projectedX = clamp01(baseX + (vx * extrapolationSeconds));
        const projectedY = clamp01(baseY + (vy * extrapolationSeconds));
        const previous = smoothedParticleStateById.get(rowId);

        let x = projectedX;
        let y = projectedY;
        if (previous) {
          const jumpMagnitude = Math.hypot(projectedX - previous.x, projectedY - previous.y);
          if (jumpMagnitude <= 0.48) {
            x = previous.x + (projectedX - previous.x) * smoothingAlpha;
            y = previous.y + (projectedY - previous.y) * smoothingAlpha;
          }
        }

        x = clamp01(x);
        y = clamp01(y);
        smoothedParticleStateById.set(rowId, {
          x,
          y,
          seenAtMs: nowWallMs,
        });
        smoothedParticleRows.push(row);
        smoothedParticleX.push(x);
        smoothedParticleY.push(y);
      }

      if (nowWallMs - particleSmoothingPruneAtMs >= 1200) {
        particleSmoothingPruneAtMs = nowWallMs;
        for (const [rowId, value] of smoothedParticleStateById.entries()) {
          if (nowWallMs - Number(value?.seenAtMs ?? 0) > 2200) {
            smoothedParticleStateById.delete(rowId);
          }
        }
      }

      const centroidStride = isLowPowerOverlay
        ? Math.max(1, Math.ceil(allFieldParticles.length / 880))
        : 1;
      livePresenceCentroids.clear();
      for (let index = 0; index < smoothedParticleRows.length; index += centroidStride) {
        const row = smoothedParticleRows[index];
        if (!row) {
          continue;
        }
        const presenceId = String(row?.presence_id ?? "").trim();
        if (!presenceId) {
          continue;
        }
        const xRatio = smoothedParticleX[index] ?? 0.5;
        const yRatio = smoothedParticleY[index] ?? 0.5;
        const current = livePresenceCentroids.get(presenceId) ?? { sumX: 0, sumY: 0, count: 0 };
        current.sumX += xRatio * centroidStride;
        current.sumY += yRatio * centroidStride;
        current.count += centroidStride;
        livePresenceCentroids.set(presenceId, current);
      }
      const renderFallbackManifestAnchors = livePresenceCentroids.size === 0;
      if (nextWidth !== canvasWidth || nextHeight !== canvasHeight) {
        canvasWidth = nextWidth;
        canvasHeight = nextHeight;
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
      }

      gl.viewport(0, 0, canvasWidth, canvasHeight);
      const washValue = isBackgroundMode
        ? Math.min(0.84, Math.max(0.28, backgroundWashRef.current + 0.08))
        : 0.54;
      gl.clearColor(0.02, 0.05, 0.09, washValue);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      pointRows.length = 0;
      lineRows.length = 0;
      presencePointRowsCurrent.length = 0;
      namedFormRows.length = 0;
      hotspots.length = 0;
      graphNodeLookup.clear();
      daimoiFlowLanes.clear();

      const addPoint = (
        xRatio: number,
        yRatio: number,
        sizePx: number,
        r: number,
        g: number,
        b: number,
        a: number,
      ) => {
        pointRows.push(
          xRatio * canvasWidth,
          yRatio * canvasHeight,
          Math.max(1.2, sizePx),
          clamp01(r),
          clamp01(g),
          clamp01(b),
          clamp01(a),
        );
      };

      const addLine = (
        x0: number,
        y0: number,
        x1: number,
        y1: number,
        r: number,
        g: number,
        b: number,
        a: number,
      ) => {
        const x0Px = x0 * canvasWidth;
        const y0Px = y0 * canvasHeight;
        const x1Px = x1 * canvasWidth;
        const y1Px = y1 * canvasHeight;
        lineRows.push(x0Px, y0Px, clamp01(r), clamp01(g), clamp01(b), clamp01(a));
        lineRows.push(x1Px, y1Px, clamp01(r), clamp01(g), clamp01(b), clamp01(a));
      };

      const showPresenceLayer = currentLayerVisibility?.presence ?? (
        currentOverlayView === "omni"
        || currentOverlayView === "presence"
        || currentOverlayView === "file-graph"
        || currentOverlayView === "true-graph"
        || currentOverlayView === "crawler-graph"
      );
      const showFileGraphLayer = currentLayerVisibility?.["file-graph"]
        ?? (!isBackgroundMode && (currentOverlayView === "omni" || currentOverlayView === "file-graph"));
      const showCrawlerGraphLayer = currentLayerVisibility?.["crawler-graph"]
        ?? (!isBackgroundMode && (currentOverlayView === "omni" || currentOverlayView === "crawler-graph"));
      const showTrueGraphLayer = currentLayerVisibility?.["true-graph"]
        ?? (!isBackgroundMode && (currentOverlayView === "omni" || currentOverlayView === "true-graph"));
      const showGraphFocusedView = showFileGraphLayer || showCrawlerGraphLayer || showTrueGraphLayer;
      const lockGraphToStaticLayout = showTrueGraphLayer && currentOverlayView === "true-graph";
      const showAmbientPresenceParticles = showPresenceLayer;
      const showRouteLaneTelemetry = showPresenceLayer || showGraphFocusedView;

      const namedForms = resolveNamedFormsForWebgl();
      const presenceAnchorById = new Map<string, { x: number; y: number }>();
      for (const form of namedForms) {
        const id = canonicalPresenceId(String((form as any)?.id ?? "").trim());
        const x = clamp01(Number((form as any)?.x ?? 0.5));
        const y = clamp01(Number((form as any)?.y ?? 0.5));
        if (id) {
          presenceAnchorById.set(id, { x, y });
        }
      }
      if (showAmbientPresenceParticles && renderFallbackManifestAnchors) {
        for (let index = 0; index < namedForms.length; index += 1) {
          const form = namedForms[index] as any;
          const pulseOffset = Math.sin((ts * 0.001 * 1.8) + index * 0.7) * 0.12 + 0.88;
          const [r, g, b] = toRgbFromHue(Number(form?.hue ?? 180));
          const x = clamp01(Number(form.x ?? 0.5));
          const y = clamp01(Number(form.y ?? 0.5));
          namedFormRows.push(
            x * canvasWidth,
            y * canvasHeight,
            Math.max(2.2, (13 + pulseOffset * 8.5) * dpr),
            clamp01(r),
            clamp01(g),
            clamp01(b),
            0.96,
          );
          hotspots.push({
            id: String(form.id ?? `presence-${index}`),
            kind: "presence",
            label: String(form.en ?? form.id ?? "presence"),
            x,
            y,
            radius: 0.03,
            radiusNorm: 0.03,
          });
        }
      }

      const fileGraph = currentSimulation?.file_graph ?? catalogRef.current?.file_graph;
      const crawlerGraph = currentSimulation?.crawler_graph ?? catalogRef.current?.crawler_graph;

      seenNodeIds.clear();
      const maxNodeCount = isLowPowerOverlay ? 760 : 1400;
      const graphPositionSourceKey = `${simulationTimestamp}|${String(currentSimulation?.presence_dynamics?.generated_at ?? "")}`;
      if (graphPositionSourceKey !== graphNodePositionMapCacheKey) {
        graphNodePositionMap = normalizeGraphNodePositionMap(
          currentSimulation?.presence_dynamics?.graph_node_positions,
        );
        graphNodePositionMapCacheKey = graphPositionSourceKey;
      }
      const graphNodeSmoothness = graphNodeSmoothnessRef.current;
      const graphNodeStepScale = graphNodeStepScaleRef.current;
      const graphNodeSmoothingAlphaBase = clampValue(
        (0.06 + (effectiveFrameMs / 320)) / graphNodeSmoothness,
        0.03,
        0.22,
      );
      const graphNodeSmoothingAlpha = clampValue(
        graphNodeSmoothingAlphaBase + clampValue(snapshotAgeMs / 1800, 0, 0.12 / graphNodeSmoothness),
        0.03,
        0.34,
      );
      const graphNodeSmoothingMaxStep = clampValue(
        (0.008 + (effectiveFrameMs / 1200)) * graphNodeStepScale,
        0.004,
        0.08,
      );
      let smoothedGraphNodeX = 0.5;
      let smoothedGraphNodeY = 0.5;
      const resolveSmoothedGraphNodePosition = (nodeId: string, targetXInput: number, targetYInput: number): void => {
        const targetX = clampValue(targetXInput, 0.012, 0.988);
        const targetY = clampValue(targetYInput, 0.012, 0.988);
        if (!nodeId) {
          smoothedGraphNodeX = targetX;
          smoothedGraphNodeY = targetY;
          return;
        }
        const previous = smoothedGraphNodeStateById.get(nodeId);
        let x = targetX;
        let y = targetY;
        if (previous) {
          const deltaX = targetX - previous.x;
          const deltaY = targetY - previous.y;
          const jumpMagnitude = Math.hypot(deltaX, deltaY);
          const adaptiveAlpha = clampValue(
            graphNodeSmoothingAlpha + clampValue(jumpMagnitude * 0.22, 0, 0.2),
            0.08,
            0.56,
          );
          let nextX = previous.x + deltaX * adaptiveAlpha;
          let nextY = previous.y + deltaY * adaptiveAlpha;

          const stepX = nextX - previous.x;
          const stepY = nextY - previous.y;
          const stepMagnitude = Math.hypot(stepX, stepY);
          const maxStep = jumpMagnitude > 0.84
            ? graphNodeSmoothingMaxStep * 4.5
            : jumpMagnitude > 0.52
              ? graphNodeSmoothingMaxStep * 2.6
              : graphNodeSmoothingMaxStep;
          if (stepMagnitude > maxStep && stepMagnitude > 1e-6) {
            const scale = maxStep / stepMagnitude;
            nextX = previous.x + stepX * scale;
            nextY = previous.y + stepY * scale;
          }
          x = nextX;
          y = nextY;
        }
        x = clampValue(x, 0.012, 0.988);
        y = clampValue(y, 0.012, 0.988);
        smoothedGraphNodeStateById.set(nodeId, {
          x,
          y,
          seenAtMs: nowWallMs,
        });
        smoothedGraphNodeX = x;
        smoothedGraphNodeY = y;
      };
      if (nowWallMs - graphNodeSmoothingPruneAtMs >= 1500) {
        graphNodeSmoothingPruneAtMs = nowWallMs;
        for (const [nodeId, value] of smoothedGraphNodeStateById.entries()) {
          if (nowWallMs - Number(value?.seenAtMs ?? 0) > 6400) {
            smoothedGraphNodeStateById.delete(nodeId);
          }
        }
      }
      const restrictGraphNodesToViewMap = !showTrueGraphLayer && showGraphFocusedView && graphNodePositionMap.size > 0;
      const spotlightMusicNexus = musicNexusSpotlightRef.current;
      const musicNodeIds = new Set<string>();
      const musicHotspots: MusicNexusHotspot[] = [];

      const ingestNodeRows = (nodes: any, sourceLayer: "file" | "crawler") => {
        if (!Array.isArray(nodes)) {
          return;
        }
        for (let index = 0; index < nodes.length && graphNodeLookup.size < maxNodeCount; index += 1) {
          const node = nodes[index] as any;
          const nodeId = String(node?.id ?? "").trim();
          if (!nodeId || seenNodeIds.has(nodeId)) {
            continue;
          }

          if (sourceLayer === "file" && !showFileGraphLayer && !showTrueGraphLayer) {
            continue;
          }
          if (sourceLayer === "crawler" && !showCrawlerGraphLayer) {
            continue;
          }
          seenNodeIds.add(nodeId);

          const rawType = String(node?.node_type ?? "").trim().toLowerCase();
          const hasCrawlerKind = String(node?.crawler_kind ?? "").trim().length > 0;
          const nodeRole = String(node?.kind ?? node?.presence_kind ?? "").trim().toLowerCase();
          const presenceKind = String(node?.presence_kind ?? "").trim().toLowerCase();
          const isPresenceNode = rawType === "presence"
            || nodeRole === "presence"
            || presenceKind === "presence"
            || nodeId.startsWith("presence:")
            || String(node?.node_id ?? "").trim().startsWith("presence:");
          const sourceRelPath = String(
            node?.source_rel_path
            ?? node?.archived_rel_path
            ?? node?.archive_rel_path
            ?? "",
          ).trim();
          const isProjectionOverflowNode = isViewCompactionBundleNode(node);
          const isCompactionArtifactNode = isProjectionOverflowNode;
          if (
            restrictGraphNodesToViewMap
            && !graphNodePositionMap.has(nodeId)
            && !isCompactionArtifactNode
            && !isPresenceNode
          ) {
            continue;
          }
          if (lockGraphToStaticLayout && isCompactionArtifactNode) {
            continue;
          }
          let nodeKind: "file" | "crawler" | "nexus" = (
            rawType === "crawler" || hasCrawlerKind
          )
            ? "crawler"
            : (rawType === "file" || rawType.length === 0)
              ? "file"
              : "nexus";
          let nodeType = rawType || (nodeKind === "crawler" ? "crawler" : "file");
          if (isProjectionOverflowNode) {
            nodeKind = "nexus";
            nodeType = "projection_overflow";
          }

          const baseX = toRatio(Number(node?.x ?? 0.5));
          const baseY = toRatio(Number(node?.y ?? 0.5));
          const backendNodePosition = graphNodePositionMap.get(nodeId);
          const presenceAnchorFallback = isPresenceNode
            ? (
              presenceAnchorById.get(canonicalPresenceId(nodeId))
              ?? presenceAnchorById.get(canonicalPresenceId(String(node?.presence_id ?? "")))
              ?? presenceAnchorById.get(canonicalPresenceId(String(node?.node_id ?? "")))
            )
            : undefined;
          const targetXRatio = clampValue(
            Number(backendNodePosition?.x ?? presenceAnchorFallback?.x ?? baseX),
            0.012,
            0.988,
          );
          const targetYRatio = clampValue(
            Number(backendNodePosition?.y ?? presenceAnchorFallback?.y ?? baseY),
            0.012,
            0.988,
          );
          let xRatio = targetXRatio;
          let yRatio = targetYRatio;
          if (lockGraphToStaticLayout) {
            xRatio = baseX;
            yRatio = baseY;
          } else {
            resolveSmoothedGraphNodePosition(nodeId, targetXRatio, targetYRatio);
            xRatio = smoothedGraphNodeX;
            yRatio = smoothedGraphNodeY;
          }
          const nodeLabelTextBase = nodeKind === "crawler"
            ? shortPathLabel(String(node?.title ?? node?.domain ?? node?.url ?? node?.label ?? nodeId))
            : nodeType === "tag"
              ? `#${shortPathLabel(String(node?.label ?? node?.tag ?? nodeId))}`
              : nodeType === "field"
                ? shortPathLabel(String(node?.label ?? node?.node_id ?? nodeId))
              : nodeType === "presence"
                ? shortPathLabel(String(node?.label ?? node?.node_id ?? nodeId))
                : shortPathLabel(String(sourceRelPath || node?.name || node?.label || nodeId));
          const nodeLabelText = isProjectionOverflowNode
            ? `[bundle] ${nodeLabelTextBase}`
            : nodeLabelTextBase;
          const resourceKind = isProjectionOverflowNode ? "unknown" : resourceKindForNode(node);
          const isTrueGraphNode = lockGraphToStaticLayout && sourceLayer === "file";
          const isMusicNode = !isProjectionOverflowNode && isMusicNexusNode(node, resourceKind);
          if (isMusicNode) {
            musicNodeIds.add(nodeId);
          }
          const importance = clamp01(Number(node?.importance ?? 0.35));
          let [r, g, b] = resourceColor(resourceKind);
          if (isProjectionOverflowNode) {
            [r, g, b] = [1.0, 0.74, 0.3];
          } else if (nodeKind === "nexus") {
            if (nodeType === "tag") {
              [r, g, b] = [0.73, 0.68, 0.98];
            } else if (nodeType === "field") {
              [r, g, b] = [0.44, 0.88, 0.94];
            } else if (nodeType === "presence") {
              [r, g, b] = [0.62, 0.92, 0.72];
            } else {
              [r, g, b] = [0.74, 0.8, 0.9];
            }
          }
          if (isMusicNode) {
            [r, g, b] = [0.42, 0.96, 1.0];
          }
          if (isTrueGraphNode) {
            if (isProjectionOverflowNode) {
              [r, g, b] = [0.98, 0.74, 0.36];
            } else if (nodeKind === "crawler") {
              [r, g, b] = [0.44, 0.78, 0.9];
            } else if (nodeKind === "nexus") {
              [r, g, b] = [0.58, 0.9, 0.98];
            } else {
              [r, g, b] = [0.5, 0.86, 0.95];
            }
          }
          const nodeSize = isProjectionOverflowNode
            ? 7.2 + importance * 5.6
            : nodeKind === "crawler"
              ? 4.2 + importance * 4.1
              : nodeKind === "nexus"
                ? 3.7 + importance * 3.4
                : 5.0 + importance * 4.2;
          const spotlightDim = spotlightMusicNexus && !isMusicNode ? 0.22 : 1;
          const trueGraphNodeAlphaScale = isTrueGraphNode ? 0.78 : 1;
          const trueGraphNodeSizeScale = isTrueGraphNode ? 0.94 : 1;
          const nodeAlpha = (isProjectionOverflowNode ? 0.96 : (nodeKind === "nexus" ? 0.72 : 0.82)) * spotlightDim * trueGraphNodeAlphaScale;
          const nodeScale = spotlightMusicNexus && isMusicNode ? 1.24 : (spotlightMusicNexus ? 0.82 : 1);
          addPoint(
            xRatio,
            yRatio,
            nodeSize * dpr * nodeScale * trueGraphNodeSizeScale,
            r,
            g,
            b,
            nodeAlpha,
          );
          if (isProjectionOverflowNode) {
            const ring = clampValue(0.008 + importance * 0.018, 0.008, 0.028);
            addLine(xRatio, yRatio - ring, xRatio + ring, yRatio, r, g, b, 0.58);
            addLine(xRatio + ring, yRatio, xRatio, yRatio + ring, r, g, b, 0.58);
            addLine(xRatio, yRatio + ring, xRatio - ring, yRatio, r, g, b, 0.58);
            addLine(xRatio - ring, yRatio, xRatio, yRatio - ring, r, g, b, 0.58);
          } else if (isMusicNode) {
            const ring = clampValue(0.006 + importance * 0.012, 0.006, 0.02);
            addLine(xRatio - ring, yRatio - ring, xRatio + ring, yRatio - ring, r, g, b, 0.66);
            addLine(xRatio + ring, yRatio - ring, xRatio + ring, yRatio + ring, r, g, b, 0.66);
            addLine(xRatio + ring, yRatio + ring, xRatio - ring, yRatio + ring, r, g, b, 0.66);
            addLine(xRatio - ring, yRatio + ring, xRatio - ring, yRatio - ring, r, g, b, 0.66);
          }
          graphNodeLookup.set(nodeId, {
            x: xRatio,
            y: yRatio,
            node,
            nodeKind,
            nodeType,
            isProjectionOverflow: isProjectionOverflowNode,
          });
          hotspots.push({
            id: nodeId,
            kind: nodeKind === "nexus" ? "nexus" : nodeKind,
            node,
            nodeKind,
            nodeType,
            resourceKind,
            isMusicNexus: isMusicNode,
            label: isMusicNode ? `[music] ${nodeLabelText}` : nodeLabelText,
            x: xRatio,
            y: yRatio,
            radius: nodeKind === "crawler" ? 0.022 : nodeKind === "nexus" ? 0.018 : 0.02,
            radiusNorm: nodeKind === "crawler" ? 0.022 : nodeKind === "nexus" ? 0.018 : 0.02,
            isProjectionOverflow: isProjectionOverflowNode,
            isTrueGraph: isTrueGraphNode,
          });
          if (isMusicNode) {
            musicHotspots.push({
              id: nodeId,
              label: nodeLabelText,
              x: xRatio,
              y: yRatio,
            });
          }
        }
      };

      const ingestFallbackGraphPositions = () => {
        if (!showGraphFocusedView || graphNodePositionMap.size <= 0) {
          return;
        }
        for (const [nodeId, nodePosition] of graphNodePositionMap.entries()) {
          if (!nodeId || graphNodeLookup.size >= maxNodeCount || graphNodeLookup.has(nodeId)) {
            continue;
          }
          const targetXRatio = clampValue(Number(nodePosition?.x ?? 0.5), 0.012, 0.988);
          const targetYRatio = clampValue(Number(nodePosition?.y ?? 0.5), 0.012, 0.988);
          resolveSmoothedGraphNodePosition(nodeId, targetXRatio, targetYRatio);
          const xRatio = smoothedGraphNodeX;
          const yRatio = smoothedGraphNodeY;
          const node = {
            id: nodeId,
            node_type: "nexus",
            label: nodeId,
          };
          const nodeSize = 3.6;
          addPoint(xRatio, yRatio, nodeSize * dpr, 0.74, 0.8, 0.9, 0.72);
          graphNodeLookup.set(nodeId, {
            x: xRatio,
            y: yRatio,
            node,
            nodeKind: "nexus",
            nodeType: "nexus",
            isProjectionOverflow: false,
          });
          hotspots.push({
            id: nodeId,
            kind: "nexus",
            node,
            nodeKind: "nexus",
            nodeType: "nexus",
            resourceKind: "text",
            label: shortPathLabel(nodeId),
            x: xRatio,
            y: yRatio,
            radius: 0.016,
            radiusNorm: 0.016,
          });
        }
      };

      ingestNodeRows(fileGraph?.nodes, "file");
      ingestNodeRows(fileGraph?.file_nodes, "file");
      ingestNodeRows(fileGraph?.crawler_nodes, "file");
      ingestNodeRows(crawlerGraph?.nodes, "crawler");
      ingestNodeRows(crawlerGraph?.crawler_nodes, "crawler");
      if (!lockGraphToStaticLayout) {
        ingestFallbackGraphPositions();
      }

      const fileEdges = (showFileGraphLayer || showTrueGraphLayer) && Array.isArray(fileGraph?.edges)
        ? fileGraph.edges
        : [];
      const crawlerEdges = showCrawlerGraphLayer && Array.isArray(crawlerGraph?.edges) ? crawlerGraph.edges : [];
      const maxEdgeCount = isLowPowerOverlay ? 1200 : 3200;

      const daimoiCounts: Record<string, number> = {};
      let totalResourceDaimoi = 0;
      const daimoiStride = isLowPowerOverlay
        ? Math.max(1, Math.ceil(allFieldParticles.length / 960))
        : 1;
      for (let i = 0; i < allFieldParticles.length; i += daimoiStride) {
        const row = allFieldParticles[i] as any;
        if (row.resource_daimoi) {
          const type = String(row.resource_type ?? "cpu");
          daimoiCounts[type] = (daimoiCounts[type] ?? 0) + 1;
          totalResourceDaimoi += daimoiStride;
        }
      }
      let dominantDaimoiType = "cpu";
      let maxCount = -1;
      Object.entries(daimoiCounts).forEach(([type, count]) => {
        if (count > maxCount) {
          maxCount = count;
          dominantDaimoiType = type;
        }
      });
      const getDaimoiColor = (type: string): [number, number, number] => {
        switch (type) {
          case "cpu": return [1.0, 0.55, 0.2];
          case "ram": return [0.2, 0.8, 1.0];
          case "disk": return [0.2, 0.9, 0.4];
          case "network": return [0.6, 0.4, 1.0];
          case "gpu": return [0.9, 0.2, 0.8];
          default: return [0.7, 0.7, 0.7];
        }
      };
      const [dr, dg, db] = getDaimoiColor(dominantDaimoiType);
      const strobePhase = (ts * 0.004);
      const strobe = (Math.sin(strobePhase) * 0.5) + 0.5;
      const flowIntensity = Math.min(1.0, totalResourceDaimoi / 20.0);

      let renderedEdgeCount = 0;
      const drawEdgeRows = (edges: any[]) => {
        for (let index = 0; index < edges.length && renderedEdgeCount < maxEdgeCount; index += 1) {
          const edge = edges[index] as any;
          const trueGraphEdgeMode = lockGraphToStaticLayout;
          if (trueGraphEdgeMode) {
            const isCompactionArtifactEdge = isViewCompactionBundleEdge(edge);
            if (isCompactionArtifactEdge) {
              continue;
            }
          }
          const sourceId = String(edge?.source ?? "").trim();
          const targetId = String(edge?.target ?? "").trim();
          if (!sourceId || !targetId || sourceId === targetId) {
            continue;
          }
          const source = graphNodeLookup.get(sourceId);
          const target = graphNodeLookup.get(targetId);
          if (!source || !target) {
            continue;
          }
          if (spotlightMusicNexus && !musicNodeIds.has(sourceId) && !musicNodeIds.has(targetId)) {
            continue;
          }
          const kind = String(edge?.kind ?? "").trim().toLowerCase();
          const [r, g, b, a] = edgeColorByKind(kind);

          const mix = trueGraphEdgeMode ? 0 : flowIntensity * strobe * 0.6;
          const fr = r * (1 - mix) + dr * mix;
          const fg = g * (1 - mix) + dg * mix;
          const fb = b * (1 - mix) + db * mix;
          const fa = trueGraphEdgeMode
            ? Math.max(0.08, a * 0.62)
            : Math.min(1.0, a + (flowIntensity * 0.3));

          addLine(source.x, source.y, target.x, target.y, fr, fg, fb, fa);
          renderedEdgeCount += 1;
        }
      };

      if (fileEdges.length > 0) {
        drawEdgeRows(fileEdges);
      }
      if (crawlerEdges.length > 0 && renderedEdgeCount < maxEdgeCount) {
        drawEdgeRows(crawlerEdges);
      }

      const transientQueryEdges = normalizeUserQueryEdgeRows(
        currentSimulation?.presence_dynamics?.user_query_transient_edges,
      );
      const promotedQueryEdges = normalizeUserQueryEdgeRows(
        currentSimulation?.presence_dynamics?.user_query_promoted_edges,
      );
      if ((showPresenceLayer || showGraphFocusedView) && (transientQueryEdges.length > 0 || promotedQueryEdges.length > 0)) {
        const userPresence = asRecord(currentSimulation?.presence_dynamics?.user_presence);
        const userAnchorX = clampValue(Number(userPresence?.anchor_x ?? 0.5), 0.02, 0.98);
        const userAnchorY = clampValue(Number(userPresence?.anchor_y ?? 0.72), 0.02, 0.98);
        const queryRows = [
          ...promotedQueryEdges.map((row) => ({ ...row, promoted: true })),
          ...transientQueryEdges.map((row) => ({ ...row, promoted: false })),
        ].slice(0, isLowPowerOverlay ? 28 : 72);

        const resolveTargetAnchor = (targetIdRaw: string): { x: number; y: number } | null => {
          const cleanTargetRaw = String(targetIdRaw || "").trim();
          const cleanTarget = canonicalPresenceId(cleanTargetRaw);
          if (!cleanTarget) {
            return null;
          }

          if (cleanTarget === "nexus") {
            const nexusNode = graphNodeLookup.get("nexus.user.cursor")
              ?? graphNodeLookup.get("nexus")
              ?? Array.from(graphNodeLookup.values()).find((row) => row.nodeKind === "nexus");
            if (nexusNode) {
              return { x: nexusNode.x, y: nexusNode.y };
            }
            return { x: 0.5, y: 0.5 };
          }

          const nodeRow = graphNodeLookup.get(cleanTargetRaw) ?? graphNodeLookup.get(cleanTarget);
          if (nodeRow) {
            return { x: nodeRow.x, y: nodeRow.y };
          }

          const presenceRow = presenceAnchorById.get(cleanTarget);
          if (presenceRow) {
            return { x: presenceRow.x, y: presenceRow.y };
          }
          return null;
        };

        for (let queryIndex = 0; queryIndex < queryRows.length; queryIndex += 1) {
          const row = queryRows[queryIndex];
          const targetAnchor = resolveTargetAnchor(row.target);
          if (!targetAnchor) {
            continue;
          }

          const seed = stablePresenceRatio(`${row.id}|${row.source}`, 61 + queryIndex);
          const angle = seed * Math.PI * 2;
          const radius = 0.012 + (stablePresenceRatio(row.source, 73) * (row.promoted ? 0.05 : 0.032));
          const sourceX = clampValue(userAnchorX + Math.cos(angle) * radius, 0.02, 0.98);
          const sourceY = clampValue(userAnchorY + Math.sin(angle) * radius, 0.02, 0.98);

          const intensity = row.promoted ? clampValue(0.45 + row.strength * 0.55, 0.45, 1.0) : clampValue(0.35 + row.life * 0.65, 0.35, 1.0);
          const r = row.promoted ? 1.0 : 0.52;
          const g = row.promoted ? 0.78 : 0.88;
          const b = row.promoted ? 0.42 : 1.0;
          const lineAlpha = row.promoted ? (0.2 + intensity * 0.44) : (0.1 + intensity * 0.3);
          const nodeAlpha = row.promoted ? 0.88 : 0.68;

          addLine(sourceX, sourceY, targetAnchor.x, targetAnchor.y, r, g, b, lineAlpha);
          addPoint(sourceX, sourceY, (2.0 + Math.min(6, row.hits) * 0.34) * dpr, r, g, b, clamp01(nodeAlpha - 0.1));
          addPoint(targetAnchor.x, targetAnchor.y, (2.6 + Math.min(7, row.hits) * 0.42) * dpr, r, g, b, nodeAlpha);
        }
      }

      if (!lockGraphToStaticLayout && (showFileGraphLayer || showTrueGraphLayer) && fileGraph) {
        const projectionGroups = Array.isArray((fileGraph as any)?.projection?.groups)
          ? (fileGraph as any).projection.groups
          : [];
        const projectionActive = Boolean((fileGraph as any)?.projection?.active);
        const fileNodes = Array.isArray((fileGraph as any)?.file_nodes) ? (fileGraph as any).file_nodes : [];
        const graphNodes = Array.isArray((fileGraph as any)?.nodes) ? (fileGraph as any).nodes : [];
        const fieldNodes = Array.isArray((fileGraph as any)?.field_nodes) ? (fileGraph as any).field_nodes : [];
        const tagNodes = Array.isArray((fileGraph as any)?.tag_nodes) ? (fileGraph as any).tag_nodes : [];
        const trueNodePositionById = new Map<string, { x: number; y: number }>();
        const registerProjectionNode = (node: any) => {
          const nodeId = String(node?.id ?? "").trim();
          if (!nodeId || trueNodePositionById.has(nodeId)) {
            return;
          }
          trueNodePositionById.set(nodeId, {
            x: clampValue(Number(node?.x ?? 0.5), 0.012, 0.988),
            y: clampValue(Number(node?.y ?? 0.5), 0.012, 0.988),
          });
        };
        graphNodes.forEach(registerProjectionNode);
        fieldNodes.forEach(registerProjectionNode);
        tagNodes.forEach(registerProjectionNode);
        fileNodes.forEach(registerProjectionNode);

        const overflowAnchorByGroupId = new Map<string, { x: number; y: number }>();
        for (const node of fileNodes) {
          const groupId = String(node?.projection_group_id ?? "").trim();
          if (!groupId || overflowAnchorByGroupId.has(groupId) || !isViewCompactionBundleNode(node)) {
            continue;
          }
          const id = String(node?.id ?? "").trim();
          const dynamic = graphNodeLookup.get(id);
          overflowAnchorByGroupId.set(groupId, {
            x: dynamic ? dynamic.x : clampValue(Number(node?.x ?? 0.5), 0.012, 0.988),
            y: dynamic ? dynamic.y : clampValue(Number(node?.y ?? 0.5), 0.012, 0.988),
          });
        }

        const resolveTrueNodeAnchor = (nodeId: string): { x: number; y: number } | null => {
          const cleanId = String(nodeId ?? "").trim();
          if (!cleanId) {
            return null;
          }
          const dynamic = graphNodeLookup.get(cleanId);
          if (dynamic) {
            return { x: dynamic.x, y: dynamic.y };
          }
          const staticNode = trueNodePositionById.get(cleanId);
          if (staticNode) {
            return staticNode;
          }
          const presenceRow = presenceAnchorById.get(cleanId);
          if (presenceRow) {
            return {
              x: clampValue(Number(presenceRow.x ?? 0.5), 0.012, 0.988),
              y: clampValue(Number(presenceRow.y ?? 0.5), 0.012, 0.988),
            };
          }
          return null;
        };

        const maxTrueGraphEdges = isLowPowerOverlay ? 320 : 1280;
        let trueGraphEdgeCount = 0;
        for (const group of projectionGroups) {
          if (trueGraphEdgeCount >= maxTrueGraphEdges) {
            break;
          }
          const groupId = String(group?.id ?? "").trim();
          if (!groupId || !projectionActive || !group?.surface_visible) {
            continue;
          }
          const sourceIds = Array.isArray(group?.member_source_ids)
            ? group.member_source_ids.map((value: unknown) => String(value ?? "").trim()).filter((value: string) => value.length > 0)
            : [];
          const targetIds = Array.isArray(group?.member_target_ids)
            ? group.member_target_ids.map((value: unknown) => String(value ?? "").trim()).filter((value: string) => value.length > 0)
            : [];
          if (sourceIds.length <= 0 && targetIds.length <= 0) {
            continue;
          }

          const anchor = overflowAnchorByGroupId.get(groupId);
          if (!anchor) {
            continue;
          }
          const memberEdgeCount = Math.max(1, Number(group?.member_edge_count ?? 1));
          const bridgeIntensity = clamp01(0.24 + (Math.log1p(memberEdgeCount) / 6.2));
          const sourceLimit = Math.min(sourceIds.length, isLowPowerOverlay ? 20 : 48);
          for (let sourceIndex = 0; sourceIndex < sourceLimit && trueGraphEdgeCount < maxTrueGraphEdges; sourceIndex += 1) {
            const sourceId = sourceIds[sourceIndex];
            const source = resolveTrueNodeAnchor(sourceId);
            if (!source) {
              continue;
            }
            addLine(
              source.x,
              source.y,
              anchor.x,
              anchor.y,
              1.0,
              0.84,
              0.44,
              0.16 + (bridgeIntensity * 0.32),
            );
            trueGraphEdgeCount += 1;
          }

          const targetLimit = Math.min(targetIds.length, isLowPowerOverlay ? 12 : 28);
          for (let targetIndex = 0; targetIndex < targetLimit && trueGraphEdgeCount < maxTrueGraphEdges; targetIndex += 1) {
            const targetId = targetIds[targetIndex];
            const target = resolveTrueNodeAnchor(targetId);
            if (!target) {
              continue;
            }
            addLine(
              anchor.x,
              anchor.y,
              target.x,
              target.y,
              0.56,
              0.92,
              1.0,
              0.12 + (bridgeIntensity * 0.24),
            );
            trueGraphEdgeCount += 1;
          }

          addPoint(
            anchor.x,
            anchor.y,
            (5.4 + Math.min(12, sourceIds.length + targetIds.length) * 0.18) * dpr,
            1.0,
            0.86,
            0.48,
            0.54,
          );
        }
      }


      if (showAmbientPresenceParticles || showRouteLaneTelemetry) {
        const particleRows = smoothedParticleRows;
        const maxParticleCount = Math.max(
          isLowPowerOverlay ? 180 : 240,
          Math.round((isLowPowerOverlay ? 980 : 2600) * particleDensityRef.current),
        );
        const step = Math.max(1, Math.ceil(particleRows.length / Math.max(1, maxParticleCount)));
        for (let index = 0; index < particleRows.length; index += step) {
          const row = particleRows[index] as any;
          if (!row) {
            continue;
          }
          const xRatio = smoothedParticleX[index] ?? 0.5;
          const yRatio = smoothedParticleY[index] ?? 0.5;
          if (showAmbientPresenceParticles) {
            const size = clampValue(Number(row?.size ?? 1.1), 0.3, 5.4);
            const semanticWeightScale = semanticWeightScaleForParticle(row as BackendFieldParticle);
            const scaledParticleSize = (size * 3.1 + 1.1) * semanticWeightScale * particleScaleRef.current * dpr;
            addPoint(
              xRatio,
              yRatio,
              scaledParticleSize,
              Number(row?.r ?? 0.58),
              Number(row?.g ?? 0.72),
              Number(row?.b ?? 0.92),
              0.66,
            );
            presencePointRowsCurrent.push(
              xRatio * canvasWidth,
              yRatio * canvasHeight,
              Math.max(1.2, scaledParticleSize),
              clamp01(Number(row?.r ?? 0.58)),
              clamp01(Number(row?.g ?? 0.72)),
              clamp01(Number(row?.b ?? 0.92)),
              0.66,
            );
          }

          const routeNodeId = showRouteLaneTelemetry ? String(row?.route_node_id ?? "").trim() : "";
          const graphNodeId = showRouteLaneTelemetry ? String(row?.graph_node_id ?? "").trim() : "";
          if (showRouteLaneTelemetry && routeNodeId && graphNodeId) {
            const source = graphNodeLookup.get(routeNodeId);
            const target = graphNodeLookup.get(graphNodeId);
            if (source && target) {
              const laneKey = `${routeNodeId}->${graphNodeId}`;
              const routeProbability = clamp01(Number(row?.route_probability ?? 0.34));
              const influencePower = clamp01(Number(row?.influence_power ?? 0.22));
              const laneWeight = clampValue(
                0.35 + (routeProbability * 0.45) + (influencePower * 0.2),
                0.18,
                1.2,
              );
              const resourceType = String(
                row?.resource_type
                  ?? row?.resource_consume_type
                  ?? row?.route_resource_focus
                  ?? dominantDaimoiType,
              ).trim().toLowerCase() || dominantDaimoiType;
              const existingLane = daimoiFlowLanes.get(laneKey);
              if (existingLane) {
                existingLane.count += 1;
                existingLane.score += laneWeight;
                existingLane.resourceTypeCounts[resourceType] = (existingLane.resourceTypeCounts[resourceType] ?? 0) + 1;
              } else {
                daimoiFlowLanes.set(laneKey, {
                  sourceX: source.x,
                  sourceY: source.y,
                  targetX: target.x,
                  targetY: target.y,
                  count: 1,
                  score: laneWeight,
                  seed: stablePresenceRatio(laneKey, 17),
                  resourceTypeCounts: {
                    [resourceType]: 1,
                  },
                });
              }
            }
          }
        }

        if (showPresenceLayer && livePresenceCentroids.size > 0) {
          sortedPresenceCentroids.length = 0;
          for (const entry of livePresenceCentroids.entries()) {
            sortedPresenceCentroids.push(entry);
          }
          sortedPresenceCentroids.sort((left, right) => right[1].count - left[1].count);
          const centroidLimit = Math.min(isLowPowerOverlay ? 96 : 180, sortedPresenceCentroids.length);
          for (let centroidIndex = 0; centroidIndex < centroidLimit; centroidIndex += 1) {
            const [presenceId, centroid] = sortedPresenceCentroids[centroidIndex];
            if (centroid.count <= 0) {
              continue;
            }
            hotspots.push({
              id: presenceId,
              kind: "presence",
              label: shortPresenceIdLabel(presenceId),
              x: clamp01(centroid.sumX / centroid.count),
              y: clamp01(centroid.sumY / centroid.count),
              radius: 0.022,
              radiusNorm: 0.022,
            });
          }
        }
      }

      if (daimoiFlowLanes.size > 0) {
        activeFlowLaneRows.length = 0;
        for (const lane of daimoiFlowLanes.values()) {
          activeFlowLaneRows.push(lane);
        }
        activeFlowLaneRows.sort((left, right) => right.score - left.score);
        const laneLimit = Math.min(isLowPowerOverlay ? 140 : 320, activeFlowLaneRows.length);

        for (let laneIndex = 0; laneIndex < laneLimit; laneIndex += 1) {
          const lane = activeFlowLaneRows[laneIndex];
          const normalizedCount = clamp01(lane.count / 6);
          const normalizedActivity = clamp01((lane.score / Math.max(1, lane.count)) * 0.9);
          const throughput = clamp01((normalizedCount * 0.72) + (normalizedActivity * 0.28));
          let dominantLaneResourceType = dominantDaimoiType;
          let dominantLaneResourceCount = -1;
          Object.entries(lane.resourceTypeCounts).forEach(([type, count]) => {
            const numericCount = Number(count);
            if (numericCount > dominantLaneResourceCount) {
              dominantLaneResourceType = type;
              dominantLaneResourceCount = numericCount;
            }
          });
          const [laneR, laneG, laneB] = getDaimoiColor(dominantLaneResourceType);
          const streamR = clamp01((laneR * 0.72) + 0.24);
          const streamG = clamp01((laneG * 0.72) + 0.24);
          const streamB = clamp01((laneB * 0.72) + 0.24);
          addLine(
            lane.sourceX,
            lane.sourceY,
            lane.targetX,
            lane.targetY,
            streamR,
            streamG,
            streamB,
            0.1 + (throughput * 0.42),
          );

          const laneDx = lane.targetX - lane.sourceX;
          const laneDy = lane.targetY - lane.sourceY;
          const laneLength = Math.hypot(laneDx, laneDy);
          if (laneLength > 0.0005) {
            const ux = laneDx / laneLength;
            const uy = laneDy / laneLength;
            if (throughput > 0.12) {
              const pulseCount = isLowPowerOverlay ? 1 : (throughput > 0.62 ? 2 : 1);
              const pulseTravelRate = 0.00008 + (throughput * 0.00014);
              const lanePhaseBase = (ts * pulseTravelRate) + (lane.seed * 11.0);
              for (let pulseIndex = 0; pulseIndex < pulseCount; pulseIndex += 1) {
                const pulsePhase = lanePhaseBase + (pulseIndex / Math.max(1, pulseCount));
                const pulseProgress = pulsePhase - Math.floor(pulsePhase);
                const laneProgress = clampValue(0.06 + (pulseProgress * 0.88), 0.04, 0.96);
                const pulseX = lane.sourceX + (laneDx * laneProgress);
                const pulseY = lane.sourceY + (laneDy * laneProgress);
                const pulseShimmer = 0.8 + (Math.sin((pulseProgress + lane.seed) * Math.PI * 2) * 0.2);
                const pulseAlpha = clamp01((0.24 + (throughput * 0.46)) * pulseShimmer);
                const pulseSize = (2.7 + (throughput * 4.8)) * dpr;
                addPoint(
                  pulseX,
                  pulseY,
                  pulseSize,
                  streamR,
                  streamG,
                  streamB,
                  pulseAlpha,
                );
                if (!isLowPowerOverlay && throughput > 0.34) {
                  const tailLen = 0.005 + (throughput * 0.01);
                  addLine(
                    pulseX - (ux * tailLen),
                    pulseY - (uy * tailLen),
                    pulseX,
                    pulseY,
                    streamR,
                    streamG,
                    streamB,
                    pulseAlpha * 0.68,
                  );
                }
              }
            }

            if (throughput <= 0.22) {
              continue;
            }
            const px = -uy;
            const py = ux;
            const arrowLen = 0.009 + (throughput * 0.018);
            const arrowWidth = arrowLen * 0.48;
            const tipX = lane.targetX;
            const tipY = lane.targetY;
            const baseX = tipX - (ux * arrowLen);
            const baseY = tipY - (uy * arrowLen);
            addLine(
              baseX + (px * arrowWidth),
              baseY + (py * arrowWidth),
              tipX,
              tipY,
              streamR,
              streamG,
              streamB,
              0.28 + (throughput * 0.5),
            );
            addLine(
              baseX - (px * arrowWidth),
              baseY - (py * arrowWidth),
              tipX,
              tipY,
              streamR,
              streamG,
              streamB,
              0.28 + (throughput * 0.5),
            );
          }
        }
      }

      const nowMs = performance.now();
      if (isInteractive && (nowMs - lastNodeTitleOverlaySyncMs) >= 140) {
        lastNodeTitleOverlaySyncMs = nowMs;
        const nextGraphNodeTitleOverlays = selectGraphNodeTitleOverlays(hotspots, {
          showFileLayer: showFileGraphLayer,
          showTrueGraphLayer: showTrueGraphLayer,
          showCrawlerLayer: showCrawlerGraphLayer,
          showPresenceLayer: showPresenceLayer,
          interactiveEnabled: isInteractive,
        });
        setGraphNodeTitleOverlays((previous) => {
          if (previous.length === nextGraphNodeTitleOverlays.length) {
            const unchanged = previous.every((row, index) => {
              const nextRow = nextGraphNodeTitleOverlays[index];
              if (!nextRow) {
                return false;
              }
              return (
                row.id === nextRow.id
                && row.label === nextRow.label
                && row.kind === nextRow.kind
                && Boolean(row.isTrueGraph) === Boolean(nextRow.isTrueGraph)
                && Boolean(row.isProjectionOverflow) === Boolean(nextRow.isProjectionOverflow)
                && Math.abs(row.x - nextRow.x) < 0.001
                && Math.abs(row.y - nextRow.y) < 0.001
              );
            });
            if (unchanged) {
              return previous;
            }
          }
          return nextGraphNodeTitleOverlays;
        });
      }

      if (pointerField.inside) {
        pointerField.power = Math.min(1.3, pointerField.power + 0.02);
      } else {
        pointerField.power *= 0.94;
      }
      if (pointerField.power > 0.02) {
        addPoint(pointerField.x, pointerField.y, (12 + pointerField.power * 20) * dpr, 0.72, 0.92, 1.0, 0.32);
      }

      const pulseAgeSec = pulse.atMs > 0 ? (performance.now() - pulse.atMs) * 0.001 : 99;
      if (pulse.power > 0.01 && pulseAgeSec < 1.5) {
        const fade = clamp01(1 - pulseAgeSec / 1.5);
        const radiusRatio = (0.02 + pulseAgeSec * 0.14) * Math.max(0.2, pulse.power);
        const segments = 28;
        for (let segment = 0; segment < segments; segment += 1) {
          const t0 = (segment / segments) * Math.PI * 2;
          const t1 = ((segment + 1) / segments) * Math.PI * 2;
          addLine(
            pulse.x + Math.cos(t0) * radiusRatio,
            pulse.y + Math.sin(t0) * radiusRatio,
            pulse.x + Math.cos(t1) * radiusRatio,
            pulse.y + Math.sin(t1) * radiusRatio,
            0.72,
            0.92,
            1.0,
            0.2 * fade,
          );
        }
      }

      // Disable any previously enabled attributes to prevent state leakage between programs
      gl.disableVertexAttribArray(0);
      gl.disableVertexAttribArray(1);
      gl.disableVertexAttribArray(2);
      gl.disableVertexAttribArray(3);

      if (lineRows.length > 0 && lineLocPos >= 0 && lineLocColor >= 0) {
        activateLineProgram();
        gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
        uploadLineBuffer = ensureUploadBuffer(uploadLineBuffer, lineRows.length);
        for (let index = 0; index < lineRows.length; index += 1) {
          uploadLineBuffer[index] = lineRows[index] ?? 0;
        }
        gl.bufferData(
          gl.ARRAY_BUFFER,
          uploadLineBuffer.subarray(0, lineRows.length),
          gl.DYNAMIC_DRAW,
        );
        gl.enableVertexAttribArray(lineLocPos);
        gl.vertexAttribPointer(lineLocPos, 2, gl.FLOAT, false, 6 * 4, 0);
        gl.enableVertexAttribArray(lineLocColor);
        gl.vertexAttribPointer(lineLocColor, 4, gl.FLOAT, false, 6 * 4, 2 * 4);
        if (lineLocResolution) {
          gl.uniform2f(lineLocResolution, canvasWidth, canvasHeight);
        }
        gl.drawArrays(gl.LINES, 0, lineRows.length / 6);
        // Disable line attributes after drawing
        gl.disableVertexAttribArray(lineLocPos);
        gl.disableVertexAttribArray(lineLocColor);
      }

      const drawPointCloud = (rows: number[], alphaScale: number) => {
        if (rows.length <= 0 || pointLocPos < 0 || pointLocSize < 0 || pointLocColor < 0) {
          return;
        }
        activatePointProgram();
        gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer);
        uploadPointBuffer = ensureUploadBuffer(uploadPointBuffer, rows.length);
        for (let index = 0; index < rows.length; index += 1) {
          uploadPointBuffer[index] = rows[index] ?? 0;
        }
        gl.bufferData(
          gl.ARRAY_BUFFER,
          uploadPointBuffer.subarray(0, rows.length),
          gl.DYNAMIC_DRAW,
        );
        gl.enableVertexAttribArray(pointLocPos);
        gl.vertexAttribPointer(pointLocPos, 2, gl.FLOAT, false, 7 * 4, 0);
        gl.enableVertexAttribArray(pointLocSize);
        gl.vertexAttribPointer(pointLocSize, 1, gl.FLOAT, false, 7 * 4, 2 * 4);
        gl.enableVertexAttribArray(pointLocColor);
        gl.vertexAttribPointer(pointLocColor, 4, gl.FLOAT, false, 7 * 4, 3 * 4);
        if (pointLocResolution) {
          gl.uniform2f(pointLocResolution, canvasWidth, canvasHeight);
        }
        if (pointLocAlphaScale) {
          gl.uniform1f(pointLocAlphaScale, clamp01(alphaScale));
        }
        gl.drawArrays(gl.POINTS, 0, rows.length / 7);
        gl.disableVertexAttribArray(pointLocPos);
        gl.disableVertexAttribArray(pointLocSize);
        gl.disableVertexAttribArray(pointLocColor);
      };

      if (particleTrailFrames.length > 0) {
        const totalTrailLayers = particleTrailFrames.length;
        for (let layerIndex = 0; layerIndex < totalTrailLayers; layerIndex += 1) {
          const layerRows = particleTrailFrames[layerIndex];
          const layerProgress = (layerIndex + 1) / totalTrailLayers;
          const layerAlpha = 0.1 + layerProgress * 0.9;
          drawPointCloud(layerRows, layerAlpha);
        }
      }
      drawPointCloud(pointRows, 1.0);

      if (namedFormRows.length > 0) {
        drawPointCloud(namedFormRows, 1.0);
      }

      if (showAmbientPresenceParticles && presencePointRowsCurrent.length > 0) {
        const frameKey = simulationTimestamp || String(Math.round(ts));
        if (frameKey !== lastTrailFrameKey) {
          const snapshot = particleTrailFramePool.pop() ?? [];
          snapshot.length = presencePointRowsCurrent.length;
          for (let index = 0; index < presencePointRowsCurrent.length; index += 1) {
            snapshot[index] = presencePointRowsCurrent[index] ?? 0;
          }
          particleTrailFrames.push(snapshot);
          if (particleTrailFrames.length > trailFrameCountLimit) {
            const retired = particleTrailFrames.shift();
            if (retired) {
              retired.length = 0;
              particleTrailFramePool.push(retired);
            }
          }
          lastTrailFrameKey = frameKey;
        }
      } else {
        while (particleTrailFrames.length > 0) {
          const retired = particleTrailFrames.pop();
          if (retired) {
            retired.length = 0;
            particleTrailFramePool.push(retired);
          }
        }
        lastTrailFrameKey = "";
      }

      if (metaRef.current) {
        metaRef.current.textContent = `webgl overlay particles:${allFieldParticles.length} nodes:${graphNodeLookup.size} hotspots:${hotspots.length}`;
      }
      musicNexusHotspotsRef.current = musicHotspots;

      if (isLowPowerOverlay) {
        lastLowPowerSimulationTimestamp = simulationTimestamp;
      } else {
        lastLowPowerSimulationTimestamp = "";
      }

      rafId = requestAnimationFrame(draw);
    };

    const pulseAt = (x: number, y: number, power: number, target = "particle_field") => {
      pulse = {
        x: clamp01(x),
        y: clamp01(y),
        power: clampValue(power, 0, 2),
        atMs: performance.now(),
        target,
      };
      const baseUrl = runtimeBaseUrl();
      fetch(`${baseUrl}/api/witness`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "touch", target }),
      }).catch(() => {});
    };

    const interactAt = (
      xRatioInput: number,
      yRatioInput: number,
      options?: { openWorldscreen?: boolean },
    ): { hitNode: boolean; openedWorldscreen: boolean; target: string; xRatio: number; yRatio: number } => {
      const resolved = resolveInteractionHotspot(xRatioInput, yRatioInput);
      const xRatio = resolved.xRatio;
      const yRatio = resolved.yRatio;
      const hit = resolved.hit;
      if (hit?.node && hit.nodeKind) {
        const nodeXRatio = clamp01(hit.x);
        const nodeYRatio = clamp01(hit.y);
        const isDoubleTap = shouldOpenWorldscreen(hit.nodeKind, hit.id);
        const openWorldscreen = Boolean(options?.openWorldscreen) || true; // Always open on click
        if (openWorldscreen) {
          openWorldscreenForNode(hit.node, hit.nodeKind, nodeXRatio, nodeYRatio);
        }
        const onNexusInteraction = onNexusInteractionRef.current;
        onNexusInteraction?.({
          nodeId: hit.id,
          nodeKind: hit.nodeKind,
          resourceKind: hit.resourceKind ?? "unknown",
          label: hit.label,
          xRatio: nodeXRatio,
          yRatio: nodeYRatio,
          openWorldscreen,
          isDoubleTap,
        });
        onUserPresenceInputRef.current?.({
          kind: "click",
          target: hit.id,
          message: `click graph node ${hit.id}`,
          xRatio: nodeXRatio,
          yRatio: nodeYRatio,
          embedDaimoi: true,
          meta: {
            source: "simulation-canvas",
            nodeKind: hit.nodeKind,
            renderer: "webgl",
          },
        });
        if (metaRef.current) {
          metaRef.current.textContent = openWorldscreen
            ? `hologram opened: ${hit.label}`
            : `focused node: ${hit.label}`;
        }
        pulseAt(nodeXRatio, nodeYRatio, 1.0, hit.id);
        return {
          hitNode: true,
          openedWorldscreen: openWorldscreen,
          target: hit.id,
          xRatio: nodeXRatio,
          yRatio: nodeYRatio,
        };
      }

      const target = hit?.id || "particle_field";
      const clickXRatio = hit ? clamp01(hit.x) : xRatio;
      const clickYRatio = hit ? clamp01(hit.y) : yRatio;
      onUserPresenceInputRef.current?.({
        kind: "click",
        target,
        message: `click simulation field ${target}`,
        xRatio: clickXRatio,
        yRatio: clickYRatio,
        embedDaimoi: true,
        meta: {
          source: "simulation-canvas",
          renderer: "webgl",
        },
      });
      pulseAt(clickXRatio, clickYRatio, 0.96, target);
      return {
        hitNode: false,
        openedWorldscreen: false,
        target,
        xRatio: clickXRatio,
        yRatio: clickYRatio,
      };
    };

    interactAtRef.current = interactAt;

    const api = {
      pulseAt,
      singAll: () => {},
      getAnchorRatio: (kind: string, targetId: string) => {
        const target = String(targetId ?? "").trim();
        if (!target) {
          return null;
        }
        if (kind === "node" || kind === "file" || kind === "crawler") {
          const match = hotspots.find((row) => (
            row.kind === "file"
            || row.kind === "crawler"
            || row.kind === "nexus"
          ) && row.id === target);
          if (!match) {
            return null;
          }
          return {
            x: match.x,
            y: match.y,
            kind: match.kind,
            label: match.label,
          };
        }
        const presence = hotspots.find((row) => row.kind === "presence" && row.id === target);
        if (!presence) {
          return null;
        }
        return {
          x: presence.x,
          y: presence.y,
          kind: "presence",
          label: presence.label,
        };
      },
      projectRatioToClient: (xRatio: number, yRatio: number) => {
        const rect = canvas.getBoundingClientRect();
        return {
          x: rect.left + clamp01(xRatio) * rect.width,
          y: rect.top + clamp01(yRatio) * rect.height,
          w: rect.width,
          h: rect.height,
        };
      },
      interactAt,
      interactClientAt: (
        clientX: number,
        clientY: number,
        options?: { openWorldscreen?: boolean },
      ) => {
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, rect.width);
        const height = Math.max(1, rect.height);
        const xRatio = clamp01((clientX - rect.left) / width);
        const yRatio = clamp01((clientY - rect.top) / height);
        return interactAt(xRatio, yRatio, options);
      },
    };

    const onPointerDown = (event: PointerEvent) => {
      if (!interactiveRef.current) {
        return;
      }
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return;
      }
      const xRatio = clamp01((event.clientX - rect.left) / rect.width);
      const yRatio = clamp01((event.clientY - rect.top) / rect.height);
      pointerField = {
        x: xRatio,
        y: yRatio,
        power: Math.max(pointerField.power, 0.24),
        inside: true,
      };

      interactAt(xRatio, yRatio);
    };

    const onPointerMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return;
      }
      pointerField = {
        x: clamp01((event.clientX - rect.left) / rect.width),
        y: clamp01((event.clientY - rect.top) / rect.height),
        power: Math.max(pointerField.power, 0.14),
        inside: true,
      };
      const nowMs = Date.now();
      if ((nowMs - userPresenceMouseEmitMs) >= 96) {
        userPresenceMouseEmitMs = nowMs;
        onUserPresenceInputRef.current?.({
          kind: "mouse_move",
          target: "simulation_canvas",
          message: "mouse move in simulation",
          xRatio: pointerField.x,
          yRatio: pointerField.y,
          embedDaimoi: false,
          meta: {
            source: "simulation-canvas",
            renderer: "webgl",
          },
        });
      }
    };

    const onPointerLeave = () => {
      pointerField = {
        ...pointerField,
        inside: false,
      };
    };

    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerleave", onPointerLeave);
    canvas.addEventListener("pointerdown", onPointerDown);
    onOverlayInitRef.current?.(api);
    rafId = requestAnimationFrame(draw);

    return () => {
      interactAtRef.current = null;
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerleave", onPointerLeave);
      canvas.removeEventListener("pointerdown", onPointerDown);
      cancelAnimationFrame(rafId);
      particleTrailFrames.length = 0;
      lastTrailFrameKey = "";
      gl.deleteBuffer(pointBuffer);
      gl.deleteBuffer(lineBuffer);
      gl.deleteProgram(pointProgramBundle.program);
      gl.deleteShader(pointProgramBundle.vertexShader);
      gl.deleteShader(pointProgramBundle.fragmentShader);
      gl.deleteProgram(lineProgramBundle.program);
      gl.deleteShader(lineProgramBundle.vertexShader);
      gl.deleteShader(lineProgramBundle.fragmentShader);
    };
  }, []);


  const activeOverlayView =
    OVERLAY_VIEW_OPTIONS.find((option) => option.id === overlayView) ?? OVERLAY_VIEW_OPTIONS[0];

  const containerClassName = backgroundMode
    ? `relative h-full w-full overflow-hidden ${className}`.trim()
    : `relative mt-3 border border-[rgba(36,31,26,0.16)] rounded-xl overflow-hidden bg-gradient-to-b from-[#0f1a1f] to-[#131b2a] ${className}`.trim();
  const canvasHeight: number | string = backgroundMode ? "100%" : height;
  const canvasPointerEvents = interactive ? "auto" : "none";
  const worldscreenCenterRatio = worldscreenPinnedCenterRatio ?? glassCenterRatio;
  const worldscreenPlacement = worldscreen
    ? resolveWorldscreenPlacement(worldscreen, containerRef.current, worldscreenCenterRatio)
    : null;
  const activeImageCommentRef = worldscreen ? worldscreenCommentRef(worldscreen) : "";
  const worldscreenConnector = useMemo(() => {
    if (!worldscreen || !worldscreenPlacement) {
      return null;
    }
    const container = containerRef.current;
    if (!container) {
      return null;
    }
    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
    const startX = clamp01(Number(worldscreen.anchorRatioX ?? 0.5));
    const startY = clamp01(Number(worldscreen.anchorRatioY ?? 0.5));
    const cardLeft = clamp01(worldscreenPlacement.left / width);
    const cardRight = clamp01((worldscreenPlacement.left + worldscreenPlacement.width) / width);
    const cardTop = clamp01(worldscreenPlacement.top / height);
    const cardBottom = clamp01((worldscreenPlacement.top + worldscreenPlacement.height) / height);
    const endX = clamp01(clampValue(startX, cardLeft, cardRight));
    const endY = clamp01(clampValue(startY, cardTop, cardBottom));
    return {
      startX,
      startY,
      endX,
      endY,
    };
  }, [worldscreen, worldscreenPlacement]);
  const worldscreenMetadataDetails = useMemo(() => {
    if (!worldscreen) {
      return [] as Array<{
        key: string;
        value: string;
        links: string[];
        isSingleLink: boolean;
      }>;
    }
    return worldscreenMetadataRows(worldscreen).map((row) => {
      const links = extractHttpUrls(row.value);
      const isSingleLink = links.length === 1
        && links[0] === row.value.trim()
        && isHttpUrlText(row.value);
      return {
        ...row,
        links,
        isSingleLink,
      };
    });
  }, [worldscreen]);
  const flattenCommentsEnabled = worldscreenMode !== "overview";
  const flattenedImageComments = useMemo(
    () => (flattenCommentsEnabled ? flattenImageCommentThread(imageComments) : []),
    [flattenCommentsEnabled, imageComments],
  );
  const commentEntryById = useMemo(() => {
    const map = new Map<string, ImageCommentEntry>();
    for (const entry of imageComments) {
      map.set(entry.id, entry);
    }
    return map;
  }, [imageComments]);
  const presenceDisplayNameById = useMemo(() => {
    const map = new Map<string, string>();
    for (const account of presenceAccounts) {
      map.set(account.presence_id, account.display_name || account.presence_id);
    }
    return map;
  }, [presenceAccounts]);
  const resolvePresenceName = useCallback((presenceId: string): string => {
    const normalized = presenceId.trim();
    if (!normalized) {
      return "unknown";
    }
    return presenceDisplayNameById.get(normalized) || normalized;
  }, [presenceDisplayNameById]);
  const imageCommentStats = useMemo(() => {
    const participantIds = new Set<string>();
    const rootCommentCount = flattenedImageComments.filter((row) => row.depth === 0).length;
    let deepestDepth = 0;
    let latestAt = "";
    for (const row of flattenedImageComments) {
      participantIds.add(row.entry.presence_id);
      if (row.depth > deepestDepth) {
        deepestDepth = row.depth;
      }
      const ts = String(row.entry.created_at || row.entry.time || "").trim();
      if (ts && ts > latestAt) {
        latestAt = ts;
      }
    }
    return {
      total: imageComments.length,
      participants: participantIds.size,
      rootCommentCount,
      deepestDepth,
      latestAt,
    };
  }, [flattenedImageComments, imageComments]);
  const activeFieldParticleRows = useMemo<BackendFieldParticle[]>(() => {
    return resolveFieldParticleRows(simulation);
  }, [resolveFieldParticleRows, simulation]);
  const queryTransientEdgeRows = useMemo(() => {
    return normalizeUserQueryEdgeRows(simulation?.presence_dynamics?.user_query_transient_edges);
  }, [simulation?.presence_dynamics?.user_query_transient_edges]);
  const queryPromotedEdgeRows = useMemo(() => {
    return normalizeUserQueryEdgeRows(simulation?.presence_dynamics?.user_query_promoted_edges);
  }, [simulation?.presence_dynamics?.user_query_promoted_edges]);
  const resourceEconomyHud = useMemo(() => {
    if (!interactive) {
      return EMPTY_RESOURCE_ECONOMY_HUD;
    }
    const dynamics = simulation?.presence_dynamics;
    const probabilistic = dynamics?.daimoi_probabilistic;
    const resourceSummary = dynamics?.resource_daimoi ?? probabilistic?.resource_daimoi;
    const consumptionSummary = dynamics?.resource_consumption ?? probabilistic?.resource_consumption;
    const packets = Math.max(0, Number(resourceSummary?.delivered_packets ?? 0));
    const transfer = Math.max(0, Number(resourceSummary?.total_transfer ?? 0));
    const actionPackets = Math.max(0, Number(consumptionSummary?.action_packets ?? 0));
    const blockedPackets = Math.max(0, Number(consumptionSummary?.blocked_packets ?? 0));
    const consumedTotal = Math.max(0, Number(consumptionSummary?.consumed_total ?? 0));
    const starvedPresences = Array.isArray(consumptionSummary?.starved_presences)
      ? consumptionSummary.starved_presences.length
      : 0;
    return {
      packets,
      transfer,
      actionPackets,
      blockedPackets,
      consumedTotal,
      starvedPresences,
    };
  }, [interactive, simulation]);
  const liveFieldParticleCount = activeFieldParticleRows.length;
  const fallbackPointCount = Array.isArray(simulation?.points) ? simulation.points.length : 0;
  const graphStructureHud = useMemo(() => {
    const truthGraph = simulation?.truth_graph as any;
    const viewGraph = simulation?.view_graph as any;
    const fileGraph = (simulation?.file_graph ?? catalog?.file_graph ?? null) as any;
    const hasFileGraph = Boolean(fileGraph && typeof fileGraph === "object");
    const projection = (fileGraph?.projection ?? viewGraph?.projection ?? null) as any;
    const projectionPolicy = (projection?.policy ?? null) as any;
    const fileNodes = Array.isArray(fileGraph?.file_nodes) ? fileGraph.file_nodes : [];
    const fileEdges = Array.isArray(fileGraph?.edges) ? fileGraph.edges : [];
    const projectionGroups = Array.isArray(projection?.groups) ? projection.groups : [];
    const overflowNodeCount = fileNodes.reduce((count: number, node: any) => {
      if (isViewCompactionBundleNode(node)) {
        return count + 1;
      }
      return count;
    }, 0);

    const truthNodeCount = Math.max(0, Number(truthGraph?.node_count ?? 0));
    const truthEdgeCount = Math.max(0, Number(truthGraph?.edge_count ?? 0));
    const viewNodeCount = Math.max(0, Number(viewGraph?.node_count ?? 0));
    const viewEdgeCount = Math.max(0, Number(viewGraph?.edge_count ?? 0));
    const projectionActive = Boolean(projection?.active ?? viewGraph?.projection?.active ?? false);
    const projectionMode = String(projection?.mode ?? viewGraph?.projection?.mode ?? "n/a");
    const projectionReason = String(projection?.reason ?? viewGraph?.projection?.reason ?? "n/a");
    const projectionBundleLedgerCount = Math.max(0, Number(viewGraph?.projection?.bundle_ledger_count ?? 0));
    const projectionCompactionDrive = Math.max(
      0,
      Math.min(
        1,
        Number(viewGraph?.projection?.compaction_drive ?? projectionPolicy?.compaction_drive ?? 0),
      ),
    );
    const projectionCpuPressure = Math.max(
      0,
      Math.min(
        1,
        Number(viewGraph?.projection?.cpu_pressure ?? projectionPolicy?.cpu_pressure ?? 0),
      ),
    );
    const projectionViewEdgePressure = Math.max(
      0,
      Math.min(
        1,
        Number(viewGraph?.projection?.view_edge_pressure ?? projectionPolicy?.view_edge_pressure ?? 0),
      ),
    );
    const projectionCpuUtilization = Math.max(
      0,
      Math.min(
        100,
        Number(viewGraph?.projection?.cpu_utilization ?? projectionPolicy?.cpu_utilization ?? 0),
      ),
    );
    const projectionEdgeThresholdBase = Math.max(
      0,
      Number(viewGraph?.projection?.edge_threshold_base ?? projection?.limits?.edge_threshold_base ?? 0),
    );
    const projectionEdgeThresholdEffective = Math.max(
      0,
      Number(viewGraph?.projection?.edge_threshold_effective ?? projection?.limits?.edge_threshold ?? 0),
    );
    const projectionEdgeCapBase = Math.max(
      0,
      Number(viewGraph?.projection?.edge_cap_base ?? projection?.limits?.edge_cap_base ?? 0),
    );
    const projectionEdgeCapEffective = Math.max(
      0,
      Number(viewGraph?.projection?.edge_cap_effective ?? projection?.limits?.edge_cap ?? 0),
    );
    const projectionCpuSentinelId = String(
      viewGraph?.projection?.cpu_sentinel_id ?? projectionPolicy?.presence_id ?? "",
    ).trim();
    const queryTransientEdgeCount = queryTransientEdgeRows.length;
    const queryPromotedEdgeCount = queryPromotedEdgeRows.length;
    const queryPeakHits = Math.max(
      0,
      ...queryTransientEdgeRows.map((row) => row.hits),
      ...queryPromotedEdgeRows.map((row) => row.hits),
    );
    const queryTopTarget = (
      queryTransientEdgeRows[0]?.target
      ?? queryPromotedEdgeRows[0]?.target
      ?? ""
    ).trim();
    const viewToTruthEdgeRatio = truthEdgeCount > 0 ? (viewEdgeCount / truthEdgeCount) : null;
    const hasContracts = (
      truthNodeCount > 0
      || truthEdgeCount > 0
      || viewNodeCount > 0
      || viewEdgeCount > 0
    );

    return {
      hasContracts,
      truthNodeCount,
      truthEdgeCount,
      viewNodeCount,
      viewEdgeCount,
      fileNodeCount: fileNodes.length,
      fileEdgeCount: fileEdges.length,
      hasFileGraph,
      projectionActive,
      projectionMode,
      projectionReason,
      projectionCompactionDrive,
      projectionCpuPressure,
      projectionViewEdgePressure,
      projectionCpuUtilization,
      projectionEdgeThresholdBase,
      projectionEdgeThresholdEffective,
      projectionEdgeCapBase,
      projectionEdgeCapEffective,
      projectionCpuSentinelId,
      projectionGroupCount: projectionGroups.length,
      projectionBundleLedgerCount,
      overflowNodeCount,
      viewToTruthEdgeRatio,
      queryTransientEdgeCount,
      queryPromotedEdgeCount,
      queryPeakHits,
      queryTopTarget,
    };
  }, [
    catalog?.file_graph,
    queryPromotedEdgeRows,
    queryTransientEdgeRows,
    simulation?.file_graph,
    simulation?.truth_graph,
    simulation?.view_graph,
  ]);
  const particleLegendStats = useMemo(() => {
    if (!interactive || activeFieldParticleRows.length <= 0) {
      return EMPTY_PARTICLE_LEGEND_STATS;
    }
    const primary = {
      chaos: 0,
      nexus: 0,
      smart: 0,
      resource: 0,
      transfer: 0,
      legacy: 0,
    };
    const overlays = {
      transfer: 0,
      resource: 0,
    };

    for (const row of activeFieldParticleRows) {
      const {
        isChaosParticle,
        isStaticParticle,
        isNexusParticle,
        isSmartDaimoi,
        isResourceEmitter,
        isTransferParticle,
      } = resolveOverlayParticleFlags(row);

      if (isTransferParticle) {
        overlays.transfer += 1;
      }
      if (isResourceEmitter) {
        overlays.resource += 1;
      }

      if (isChaosParticle) {
        primary.chaos += 1;
      } else if (isNexusParticle || isStaticParticle) {
        primary.nexus += 1;
      } else if (isResourceEmitter) {
        primary.resource += 1;
      } else if (isTransferParticle) {
        primary.transfer += 1;
      } else if (isSmartDaimoi) {
        primary.smart += 1;
      } else {
        primary.legacy += 1;
      }
    }

    return {
      total: activeFieldParticleRows.length,
      primary,
      overlays,
    };
  }, [activeFieldParticleRows, interactive]);
  const musicNexusNodeCount = useMemo(() => {
    const uniqueIds = new Set<string>();
    const sourceGraphs = [
      simulation?.file_graph,
      simulation?.crawler_graph,
      catalog?.file_graph,
      catalog?.crawler_graph,
    ];
    for (const graph of sourceGraphs) {
      if (!graph) {
        continue;
      }
      const graphRow = graph as any;
      const nodeRows = [graphRow.nodes, graphRow.file_nodes, graphRow.crawler_nodes];
      for (const rows of nodeRows) {
        if (!Array.isArray(rows)) {
          continue;
        }
        for (const row of rows) {
          const node = row as any;
          const nodeId = String(node?.id ?? "").trim();
          if (!nodeId || uniqueIds.has(nodeId)) {
            continue;
          }
          if (!isMusicNexusNode(node, resourceKindForNode(node))) {
            continue;
          }
          uniqueIds.add(nodeId);
        }
      }
    }
    return uniqueIds.size;
  }, [catalog, simulation]);
  const focusNextMusicNexus = useCallback(() => {
    const hotspots = musicNexusHotspotsRef.current;
    if (hotspots.length <= 0) {
      setMusicNexusJumpLabel("no mp3 nexus visible in current overlay");
      if (metaRef.current) {
        metaRef.current.textContent = "mp3 nexus jump unavailable";
      }
      return;
    }
    const nextIndex = musicNexusCycleRef.current % hotspots.length;
    const target = hotspots[nextIndex];
    musicNexusCycleRef.current = nextIndex + 1;
    setMusicNexusJumpLabel(target.label);
    interactAtRef.current?.(target.x, target.y, { openWorldscreen: true });
    if (metaRef.current) {
      metaRef.current.textContent = `mp3 nexus focused: ${target.label}`;
    }
  }, []);
  const overlayParticleModeActive = renderRichOverlayParticles && liveFieldParticleCount > 0;

  return (
    <div ref={containerRef} className={containerClassName}>
      <canvas ref={canvasRef} style={{ height: canvasHeight, pointerEvents: canvasPointerEvents }} className="block w-full" />
      <canvas ref={overlayRef} style={{ height: canvasHeight, pointerEvents: canvasPointerEvents }} className="absolute inset-0 w-full touch-none" />
      {interactive && graphNodeTitleOverlays.length > 0 ? (
        <div className="absolute inset-0 z-10 pointer-events-none overflow-hidden">
          {graphNodeTitleOverlays.map((row) => (
            <div
              key={`graph-node-title-${row.id}`}
              className={`absolute pointer-events-auto max-w-[15rem] -translate-x-1/2 -translate-y-full rounded border px-1.5 py-0.5 text-[10px] leading-4 whitespace-nowrap overflow-hidden text-ellipsis shadow-[0_8px_22px_rgba(0,0,0,0.32)] ${
                row.isTrueGraph && row.label.toLowerCase().startsWith("[bundle]")
                  ? "border-[rgba(255,142,220,0.9)] bg-[rgba(61,12,51,0.9)] text-[#ffe3ff]"
                  : row.isTrueGraph
                    ? "border-[rgba(230,129,255,0.82)] bg-[rgba(42,9,63,0.86)] text-[#f6d7ff]"
                  : row.label.toLowerCase().startsWith("[bundle]")
                  ? "border-[rgba(255,196,108,0.58)] bg-[rgba(58,36,10,0.72)] text-[#ffe7bf]"
                  : row.kind === "file"
                  ? "border-[rgba(152,216,255,0.44)] bg-[rgba(11,30,48,0.68)] text-[#dff3ff]"
                  : row.kind === "crawler"
                    ? "border-[rgba(186,210,164,0.38)] bg-[rgba(23,36,30,0.65)] text-[#d7f0dc]"
                    : row.kind === "nexus"
                      ? "border-[rgba(189,206,255,0.44)] bg-[rgba(19,28,52,0.68)] text-[#e5ecff]"
                      : "border-[rgba(178,190,232,0.44)] bg-[rgba(18,23,40,0.68)] text-[#e2e8ff]"
              }`}
              onPointerDown={(event) => {
                if (event.button !== 0) {
                  return;
                }
                event.preventDefault();
                event.stopPropagation();
                interactAtRef.current?.(row.x, row.y, { openWorldscreen: true });
              }}
              style={{
                left: `${(row.x * 100).toFixed(2)}%`,
                top: `${(row.y * 100).toFixed(2)}%`,
              }}
            >
              {row.label}
            </div>
          ))}
        </div>
      ) : null}
      {!compactHud ? (
        <div className="absolute top-2 right-2 z-10 w-[min(96%,31rem)] pointer-events-auto">
          <div className="rounded-md border border-[rgba(137,178,220,0.32)] bg-[rgba(9,22,36,0.72)] px-2 py-2 backdrop-blur-[2px]">
            <p className="text-[9px] uppercase tracking-[0.13em] text-[#a8d3f7]">view lanes</p>
            {!overlayViewLocked ? (
              <div className="mt-1 flex flex-wrap gap-1">
                {OVERLAY_VIEW_OPTIONS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => setOverlayView(option.id)}
                    className={`rounded-md border px-2 py-1 text-[10px] font-semibold transition-colors ${
                      overlayView === option.id
                        ? "border-[rgba(146,229,255,0.82)] bg-[rgba(66,170,214,0.34)] text-[#e8f8ff]"
                        : "border-[rgba(128,167,204,0.42)] bg-[rgba(15,34,54,0.62)] text-[#c4d7f0] hover:bg-[rgba(33,64,96,0.74)]"
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            ) : (
              <p className="mt-1 text-[10px] text-[#d3ebff]">lane locked: {activeOverlayView.label}</p>
            )}
            <p className="mt-1 text-[10px] text-[#bcd8ef]">{activeOverlayView.description}</p>
            <p className="mt-1 text-[10px] text-[#9fc7e3]">
              swarm mode braids nearby packets by owner + direction.
            </p>
            {interactive ? (
              <p className="mt-1 text-[10px] text-[#c4d7f0]">single tap centers nexus in glass lane · double tap opens hologram / 単タップで中心化・ダブルで起動</p>
            ) : null}
            <div className="mt-2 rounded border border-[rgba(122,198,228,0.32)] bg-[rgba(10,27,42,0.56)] px-2 py-1.5">
              <p className="text-[9px] uppercase tracking-[0.11em] text-[#9fd5f2]">mp3 nexus tools</p>
              <p className="mt-1 text-[10px] text-[#c8e4f5]">
                mp3 matches: {musicNexusNodeCount} · visible matches: {musicNexusHotspotsRef.current.length}
              </p>
              <div className="mt-1.5 flex flex-wrap gap-1">
                <button
                  type="button"
                  onClick={() => setMusicNexusSpotlight((previous) => !previous)}
                  className={`rounded border px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                    musicNexusSpotlight
                      ? "border-[rgba(128,236,255,0.82)] bg-[rgba(62,172,192,0.38)] text-[#ecfbff]"
                      : "border-[rgba(122,177,209,0.42)] bg-[rgba(12,33,52,0.62)] text-[#bed8ee] hover:bg-[rgba(26,58,85,0.74)]"
                  }`}
                >
                  {musicNexusSpotlight ? "mp3 spotlight on" : "mp3 spotlight off"}
                </button>
                <button
                  type="button"
                  onClick={focusNextMusicNexus}
                  className="rounded border border-[rgba(128,236,255,0.64)] bg-[rgba(25,94,116,0.38)] px-2 py-0.5 text-[10px] font-semibold text-[#dff7ff] hover:bg-[rgba(38,126,150,0.44)]"
                >
                  jump to next mp3 nexus
                </button>
              </div>
              {musicNexusJumpLabel ? (
                <p className="mt-1 text-[10px] text-[#a9d9ee]">focus: {musicNexusJumpLabel}</p>
              ) : null}
            </div>
          </div>
          <div className="mt-2 rounded-md border border-[rgba(126,214,247,0.34)] bg-[rgba(7,19,33,0.76)] px-2 py-2 shadow-[0_14px_30px_rgba(0,9,20,0.34)]">
            <div className="flex items-center justify-between gap-2">
              <p className="text-[9px] uppercase tracking-[0.13em] text-[#a9d8f2]">compute insight</p>
              <button
                type="button"
                onClick={() => setComputePanelCollapsed((prev) => !prev)}
                className="rounded border border-[rgba(136,193,226,0.4)] bg-[rgba(17,42,64,0.58)] px-2 py-0.5 text-[10px] font-semibold text-[#d7edff]"
              >
                {computePanelCollapsed ? "expand" : "collapse"}
              </button>
            </div>
            <p className="mt-1 text-[10px] text-[#bed9ee]">
              jobs 180s: {computeJobInsights.total180s} · window: {computeJobInsights.rows.length} · gpu idle est: {Math.round(computeJobInsights.gpuAvailability * 100)}%
            </p>
            {!computePanelCollapsed ? (
              <>
                <div className="mt-2 flex flex-wrap gap-1">
                  {COMPUTE_JOB_FILTER_OPTIONS.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => setComputeJobFilter(option.id)}
                      className={`rounded border px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                        computeJobFilter === option.id
                          ? "border-[rgba(144,227,255,0.78)] bg-[rgba(54,142,188,0.38)] text-[#e8f9ff]"
                          : "border-[rgba(120,170,206,0.38)] bg-[rgba(14,33,52,0.62)] text-[#bed8ee] hover:bg-[rgba(27,58,85,0.74)]"
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
                <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-[#c9e3f6]">
                  <p>llm: {computeJobInsights.summary.llm}</p>
                  <p>embed: {computeJobInsights.summary.embedding}</p>
                  <p>ok: {computeJobInsights.summary.ok}</p>
                  <p>error: {computeJobInsights.summary.error}</p>
                  <p>gpu: {computeJobInsights.summary.byResource.gpu ?? 0}</p>
                  <p>npu: {computeJobInsights.summary.byResource.npu ?? 0}</p>
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {Object.entries(computeJobInsights.summary.byBackend)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 4)
                    .map(([backend, count]) => (
                      <span
                        key={backend}
                        className="rounded border border-[rgba(116,172,205,0.34)] bg-[rgba(9,28,45,0.64)] px-1.5 py-0.5 text-[10px] text-[#b8d7ec]"
                      >
                        {backend}:{count}
                      </span>
                    ))}
                </div>
                <div className="mt-2 max-h-44 overflow-auto rounded border border-[rgba(114,177,214,0.3)] bg-[rgba(4,13,24,0.64)] p-1.5">
                  {computeJobInsights.filtered.length <= 0 ? (
                    <p className="text-[10px] text-[#99c0dc]">no compute jobs for selected filter</p>
                  ) : (
                    computeJobInsights.filtered.slice(0, 10).map((row) => (
                      <article
                        key={row.id}
                        className="mb-1.5 rounded border border-[rgba(94,149,183,0.26)] bg-[rgba(10,27,42,0.7)] px-1.5 py-1 last:mb-0"
                      >
                        <p className="text-[10px] text-[#d8efff]">
                          <span className="font-semibold">{row.kind}</span>
                          <span className="text-[#9fc4df]"> · </span>
                          <span>{row.op || "op"}</span>
                          <span className="text-[#9fc4df]"> · </span>
                          <span>{row.backend || "backend"}</span>
                          <span className="text-[#9fc4df]"> · </span>
                          <span>{row.resource || "resource"}</span>
                        </p>
                        <p className="text-[10px] text-[#a9cde7]">
                          {computeJobAgeLabel(row.tsMs)} ago · {row.status}
                          {row.latencyMs !== null ? ` · ${Math.round(row.latencyMs)}ms` : ""}
                          {row.model ? ` · ${row.model}` : ""}
                        </p>
                        {row.error ? (
                          <p className="text-[10px] text-[#ffcdae] line-clamp-2">{row.error}</p>
                        ) : null}
                      </article>
                    ))
                  )}
                </div>
              </>
            ) : null}
          </div>
        </div>
      ) : null}
      {!compactHud ? (
        <div className="absolute top-2 left-2 pointer-events-none rounded-md border border-[rgba(130,176,220,0.32)] bg-[rgba(8,20,33,0.62)] px-2 py-1">
          <p className="text-[9px] uppercase tracking-[0.11em] text-[#a8d3f7]">node legend</p>
          <p className="text-[10px]">
            <span className="text-[#ffd76a]">TEXT</span>
            <span className="text-[#c8dcf3]"> · </span>
            <span className="text-[#ff87d4]">IMAGE</span>
            <span className="text-[#c8dcf3]"> · </span>
            <span className="text-[#80f0ff]">AUDIO</span>
            <span className="text-[#c8dcf3]"> · </span>
            <span className="text-[#ffad63]">ARCHIVE</span>
            <span className="text-[#c8dcf3]"> · </span>
            <span className="text-[#9fb3c8]">BLOB</span>
          </p>
          <p className="text-[10px]">
            <span className="text-[#8ccfff]">LINK</span>
            <span className="text-[#c8dcf3]"> · </span>
            <span className="text-[#9ce9bb]">WEBSITE</span>
            <span className="text-[#c8dcf3]"> · </span>
            <span className="text-[#ff9f77]">VIDEO</span>
          </p>
          <p className="text-[10px] text-[#a7f2ff]">MUSIC NEXUS: cyan ring + [music] label</p>
          <p className="text-[10px] text-[#cfdcff]">NEXUS META: tag / field / presence nodes</p>
        </div>
      ) : null}
      {!compactHud ? (
        <div className="absolute bottom-2 left-2 text-xs text-[var(--muted)] pointer-events-none">
          <p ref={metaRef}>simulation stream active</p>
        </div>
      ) : null}
      {interactive ? (
        <div
          className={`absolute left-2 z-20 pointer-events-none rounded-md border border-[rgba(122,182,220,0.34)] bg-[rgba(6,16,28,0.72)] px-2 py-1.5 text-[10px] text-[#cde8ff] ${
            compactHud ? "bottom-10 max-w-[19rem]" : "bottom-12 max-w-[24rem]"
          }`}
        >
          <p className="uppercase tracking-[0.1em] text-[#d8b9ff]">graph contracts</p>
          {graphStructureHud.hasContracts ? (
            <>
              <p className="text-[#f0ccff]">
                truth n/e {graphStructureHud.truthNodeCount}/{graphStructureHud.truthEdgeCount}
              </p>
              <p className="text-[#e4beff]">
                view n/e {graphStructureHud.viewNodeCount}/{graphStructureHud.viewEdgeCount}
                {graphStructureHud.viewToTruthEdgeRatio !== null
                  ? ` · edge ratio ${(graphStructureHud.viewToTruthEdgeRatio).toFixed(2)}`
                  : ""}
              </p>
              <p className="text-[#c9def4]">
                projection {graphStructureHud.projectionActive ? "active" : "inactive"}
                {` · mode ${graphStructureHud.projectionMode}`}
              </p>
              <p className="text-[#c9def4]">
                reason {graphStructureHud.projectionReason}
                {` · groups ${graphStructureHud.projectionGroupCount}`}
                {` · ledgers ${graphStructureHud.projectionBundleLedgerCount}`}
              </p>
              <p className="text-[#c9def4]">
                drive {graphStructureHud.projectionCompactionDrive.toFixed(2)}
                {` · cpu ${graphStructureHud.projectionCpuUtilization.toFixed(1)}%`}
                {` · cpuP ${graphStructureHud.projectionCpuPressure.toFixed(2)}`}
                {` · edgeP ${graphStructureHud.projectionViewEdgePressure.toFixed(2)}`}
              </p>
              <p className="text-[#c9def4]">
                threshold {Math.round(graphStructureHud.projectionEdgeThresholdEffective)}/{Math.round(graphStructureHud.projectionEdgeThresholdBase)}
                {` · cap ${Math.round(graphStructureHud.projectionEdgeCapEffective)}/${Math.round(graphStructureHud.projectionEdgeCapBase)}`}
              </p>
              {graphStructureHud.projectionCpuSentinelId ? (
                <p className="text-[#d4baff]">sentinel {graphStructureHud.projectionCpuSentinelId}</p>
              ) : null}
              {graphStructureHud.hasFileGraph ? (
                <p className="text-[#c9def4]">
                  rendered file n/e {graphStructureHud.fileNodeCount}/{graphStructureHud.fileEdgeCount}
                  {` · bundle nodes ${graphStructureHud.overflowNodeCount}`}
                </p>
              ) : (
                <p className="text-[#9fbed7]">rendered file graph unavailable in current snapshot.</p>
              )}
              <p className="text-[#9ecbe8]">
                query edges transient/promoted {graphStructureHud.queryTransientEdgeCount}/{graphStructureHud.queryPromotedEdgeCount}
                {graphStructureHud.queryPeakHits > 0 ? ` · peak hits ${graphStructureHud.queryPeakHits}` : ""}
              </p>
              {graphStructureHud.queryTopTarget ? (
                <p className="text-[#9ecbe8]">top query target {graphStructureHud.queryTopTarget}</p>
              ) : null}
              <p className={graphStructureHud.projectionActive ? "text-[#ffd8b0]" : "text-[#a7c7df]"}>
                {graphStructureHud.projectionActive
                  ? "projection active: bundle nodes should be present when groups > 0."
                  : "projection inactive: compression bypassed, so bundle nodes can be zero."}
              </p>
            </>
          ) : (
            <p className="text-[#a7c7df]">truth/view contracts not present in current simulation snapshot.</p>
          )}
          <div className="my-1 border-t border-[rgba(117,158,190,0.3)]" />
          <p className="uppercase tracking-[0.1em] text-[#9fd2f3]">particle key</p>
          {overlayParticleModeActive ? (
            <>
              <p className="text-[#9ecbe8]">primary classes (one class per particle, n={particleLegendStats.total})</p>
              <p><span className="text-[#ffa8e8]">✦</span> chaos butterflies ({particleLegendStats.primary.chaos})</p>
              <p><span className="text-[#cce4ff]">◆</span> nexus particles (passive) ({particleLegendStats.primary.nexus})</p>
              <p><span className="text-[#ffdba1]">●</span> resource emitters / packets ({particleLegendStats.primary.resource})</p>
              <p><span className="text-[#7fe8ff]">●</span> transfer daimoi ({particleLegendStats.primary.transfer})</p>
              <p><span className="text-[#87b9df]">·</span> smart daimoi ({particleLegendStats.primary.smart}) · legacy points ({particleLegendStats.primary.legacy})</p>
              <p className="mt-1 text-[#9ecbe8]">
                stream signals: transfer ({particleLegendStats.overlays.transfer}) · resource ({particleLegendStats.overlays.resource})
              </p>
              <p className="text-[#9ecbe8]">ghost trails: smart daimoi keep short path memory for easier tracing.</p>
              <p className="text-[#9ecbe8]">flow lanes: backend simulation drives nexus routing telemetry.</p>
              <p className="text-[#9ecbe8]">all nexus node movement now comes from backend simulation deltas.</p>
              <p className="text-[#9ecbe8]">graph view: daimoi flow is shown as lane lines + pulses + arrowheads from backend state.</p>
              <p className="mt-1 text-[#ffd7aa]">
                economy: packets {resourceEconomyHud.packets} · actions {resourceEconomyHud.actionPackets} · blocked {resourceEconomyHud.blockedPackets}
              </p>
              <p className="text-[#ffbf9a]">
                transfer {resourceEconomyHud.transfer.toFixed(2)} · consumed {resourceEconomyHud.consumedTotal.toFixed(2)} · starved presences {resourceEconomyHud.starvedPresences}
              </p>
              <p className="mt-1 text-[#9ecbe8]">mode: field particles ({liveFieldParticleCount})</p>
            </>
          ) : (
            <>
              <p><span className="text-[#8fc8ff]">●</span> no field particles in current stream</p>
              <p className="text-[#9ecbe8]">source-of-truth mode keeps legacy point-cloud fallback disabled.</p>
              <p className="mt-1 text-[#9ecbe8]">mode: stream-only ({fallbackPointCount} legacy points omitted)</p>
            </>
          )}
        </div>
      ) : null}
      {interactive ? (
        <div className="absolute bottom-2 right-2 z-30 flex flex-col items-end gap-2 pointer-events-auto">
          {modelDockOpen ? <GalaxyModelDock onClose={() => setModelDockOpen(false)} /> : null}
          <button
            type="button"
            onClick={() => setModelDockOpen((prev) => !prev)}
            className="rounded-md border border-[rgba(132,200,239,0.5)] bg-[rgba(15,38,58,0.72)] px-2.5 py-1 text-[11px] font-semibold text-[#d7efff] shadow-[0_10px_24px_rgba(0,9,20,0.45)] hover:bg-[rgba(26,58,83,0.85)]"
          >
            {modelDockOpen ? "hide model dock" : "model dock"}
          </button>
        </div>
      ) : null}
      {interactive && worldscreen ? (() => {
        const overlay = (
          <div
            className="fixed inset-0 z-[92] pointer-events-auto"
            data-core-pointer="block"
            data-core-wheel="block"
          >
          {worldscreenConnector ? (
            <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
              <defs>
                <marker
                  id="worldscreen-connector-arrow"
                  markerWidth="8"
                  markerHeight="8"
                  refX="7"
                  refY="4"
                  orient="auto-start-reverse"
                >
                  <path d="M0,0 L8,4 L0,8 z" fill="rgba(158,226,255,0.9)" />
                </marker>
              </defs>
              <line
                x1={`${(worldscreenConnector.startX * 100).toFixed(2)}%`}
                y1={`${(worldscreenConnector.startY * 100).toFixed(2)}%`}
                x2={`${(worldscreenConnector.endX * 100).toFixed(2)}%`}
                y2={`${(worldscreenConnector.endY * 100).toFixed(2)}%`}
                stroke="rgba(158,226,255,0.76)"
                strokeWidth="2"
                strokeDasharray="5 4"
                markerEnd="url(#worldscreen-connector-arrow)"
              />
              <circle
                cx={`${(worldscreenConnector.startX * 100).toFixed(2)}%`}
                cy={`${(worldscreenConnector.startY * 100).toFixed(2)}%`}
                r="4"
                fill="rgba(120,205,255,0.9)"
                stroke="rgba(224,246,255,0.84)"
                strokeWidth="1"
              />
            </svg>
          ) : null}
          <section
            data-core-pointer="block"
            data-core-wheel="block"
            className="pointer-events-auto absolute rounded-2xl border border-[rgba(126,218,255,0.58)] bg-[linear-gradient(164deg,rgba(6,16,30,0.88),rgba(10,30,48,0.82),rgba(7,18,34,0.9))] backdrop-blur-[5px] shadow-[0_30px_90px_rgba(0,18,42,0.56)] overflow-hidden"
            style={
              worldscreenPlacement
                ? {
                    left: worldscreenPlacement.left,
                    top: worldscreenPlacement.top,
                    width: worldscreenPlacement.width,
                    height: worldscreenPlacement.height,
                    transform: "perspective(1300px) rotateX(5deg)",
                    transformOrigin: worldscreenPlacement.transformOrigin,
                    transition: "left 180ms ease-out, top 180ms ease-out, width 180ms ease-out, height 180ms ease-out, transform 220ms cubic-bezier(0.22,1,0.36,1)",
                    willChange: "left, top, transform",
                  }
                : undefined
            }
          >
            <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(transparent_0%,rgba(132,212,255,0.08)_48%,transparent_100%)] bg-[length:100%_3px] opacity-40" />
            <div className="pointer-events-none absolute -inset-8 bg-[radial-gradient(circle_at_75%_8%,rgba(74,220,255,0.26),transparent_42%),radial-gradient(circle_at_22%_88%,rgba(255,156,94,0.2),transparent_46%)]" />
            <header className="relative h-14 px-4 flex items-center justify-between border-b border-[rgba(132,196,244,0.35)] bg-[rgba(7,19,33,0.76)]">
              <div className="min-w-0">
                <p className="text-[10px] uppercase tracking-[0.14em] text-[#a9dbff]">hologram worldscreen / 投影スクリーン</p>
                <p className="text-sm text-[#ecf7ff] font-semibold truncate">
                  {worldscreen.label}
                  <span className="text-[11px] text-[#b9dcf7]"> ({worldscreen.nodeKind})</span>
                </p>
                <p className="text-[10px] text-[#9bc3df]">{worldscreen.subtitle}</p>
              </div>
              <div className="flex items-center gap-2 pl-3">
                <div className="flex items-center gap-1 rounded-md border border-[rgba(127,190,226,0.34)] bg-[rgba(10,28,45,0.48)] px-1 py-1">
                  {HOLOGRAM_MODE_OPTIONS.map((option) => (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => setWorldscreenMode(option.id)}
                      className={`rounded px-2 py-1 text-[10px] font-semibold transition-colors ${
                        worldscreenMode === option.id
                          ? "bg-[rgba(82,162,206,0.4)] text-[#ecf8ff]"
                          : "text-[#b3d4ea] hover:bg-[rgba(47,98,136,0.32)]"
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
                <span className="text-[10px] px-2 py-1 rounded-md border border-[rgba(142,223,255,0.44)] text-[#d8f3ff] bg-[rgba(33,95,132,0.24)]">
                  {resourceKindLabel(worldscreen.resourceKind)}
                </span>
                {String(worldscreen.url ?? "").trim().length > 0 ? (
                  <a
                    href={worldscreen.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs px-2.5 py-1 rounded-md border border-[rgba(145,190,240,0.42)] text-[#d4ebff] hover:bg-[rgba(72,119,170,0.2)]"
                  >
                    new tab
                  </a>
                ) : null}
                <button
                  type="button"
                  onPointerDown={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                  }}
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    setWorldscreen(null);
                    setWorldscreenPinnedCenterRatio(null);
                    setWorldscreenMode("overview");
                  }}
                  className="text-xs px-2.5 py-1 rounded-md border border-[rgba(245,200,171,0.45)] text-[#ffe6d2] hover:bg-[rgba(187,120,78,0.2)]"
                >
                  close
                </button>
              </div>
            </header>
            <div className="relative h-[calc(100%-3.5rem)] p-2 sm:p-3">
              {worldscreenMode === "overview" ? (
                <div className="h-full rounded-xl border border-[rgba(143,214,255,0.38)] bg-[linear-gradient(180deg,rgba(5,16,30,0.92),rgba(5,15,26,0.88))] overflow-hidden p-3 text-[#d9eeff]">
                  <div className="grid h-full gap-3 lg:grid-cols-[minmax(16rem,0.9fr)_minmax(0,1.1fr)]">
                    <aside className="overflow-auto pr-1">
                      <p className="text-[11px] uppercase tracking-[0.12em] text-[#9fd0ef]">
                        Remote resource metadata from crawler encounter
                      </p>
                      <div className="mt-2 grid gap-1.5 text-[11px] leading-5">
                        {worldscreenMetadataDetails.map((row) => (
                          <div key={`${row.key}:${row.value}`} className="grid grid-cols-[auto,1fr] gap-2">
                            <span className="text-[#87afcc] uppercase tracking-[0.08em]">{row.key}</span>
                            <div className="text-[#e2f3ff] break-all">
                              {row.isSingleLink ? (
                                <a
                                  href={row.value}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="underline decoration-[#78b6dd] underline-offset-2"
                                >
                                  {row.value}
                                </a>
                              ) : (
                                <span>{row.value}</span>
                              )}
                              {!row.isSingleLink && row.links.length > 0 ? (
                                <div className="mt-1 flex flex-wrap gap-1.5">
                                  {row.links.map((link) => (
                                    <a
                                      key={`${row.key}:${link}`}
                                      href={link}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="rounded border border-[rgba(113,181,220,0.35)] bg-[rgba(15,45,68,0.5)] px-1.5 py-0.5 text-[10px] text-[#d2edff]"
                                    >
                                      open link
                                    </a>
                                  ))}
                                </div>
                              ) : null}
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="mt-3 rounded border border-[rgba(112,174,207,0.32)] bg-[rgba(8,22,35,0.62)] p-2">
                        <p className="text-[10px] uppercase tracking-[0.1em] text-[#95c4e1]">conversation quick stats</p>
                        <p className="text-[11px] text-[#c9e7fb]">comments {imageCommentStats.total} · participants {imageCommentStats.participants}</p>
                        <p className="text-[11px] text-[#c9e7fb]">threads {imageCommentStats.rootCommentCount} · depth {imageCommentStats.deepestDepth}</p>
                      </div>
                      {worldscreen.projectionMemberManifest && worldscreen.projectionMemberManifest.length > 0 ? (
                        <div className="mt-3 rounded border border-[rgba(224,176,108,0.34)] bg-[rgba(46,30,10,0.45)] p-2">
                          <p className="text-[10px] uppercase tracking-[0.1em] text-[#ffd8a9]">bundle manifest</p>
                          <p className="text-[10px] text-[#f2d9bb]">
                            group <code>{worldscreen.projectionGroupId || "(unknown)"}</code> · files {worldscreen.projectionMemberManifest.length}
                          </p>
                          <div className="mt-1.5 max-h-40 overflow-auto rounded border border-[rgba(225,172,112,0.28)] bg-[rgba(24,15,5,0.56)] p-1.5">
                            {worldscreen.projectionMemberManifest.map((path, index) => (
                              <p key={`${worldscreen.projectionGroupId || "bundle"}:${path}`} className="text-[10px] leading-5 text-[#ffe8cd] break-all">
                                {index + 1}. {path}
                              </p>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </aside>

                    <section className="overflow-auto rounded-lg border border-[rgba(124,205,247,0.3)] bg-[rgba(6,17,29,0.56)] p-2">
                      {worldscreen.view === "video" ? (
                        <video
                          controls
                          autoPlay
                          src={worldscreen.url}
                          className="w-full max-h-[56vh] rounded-md bg-[rgba(4,9,16,0.86)]"
                        >
                          <track kind="captions" />
                        </video>
                      ) : null}

                      {worldscreen.view === "editor" ? (
                        <div className="min-h-[20rem] overflow-hidden rounded-md border border-[rgba(143,214,255,0.22)]">
                          {editorPreview.status === "loading" ? (
                            <div className="h-full min-h-[20rem] grid place-items-center text-sm text-[#b8e0ff]">loading file preview...</div>
                          ) : null}
                          {editorPreview.status === "error" ? (
                            <div className="h-full min-h-[20rem] grid place-items-center text-sm text-[#ffd6bb]">{editorPreview.error}</div>
                          ) : null}
                          {editorPreview.status === "ready" ? (
                            <pre className="h-full max-h-[56vh] overflow-auto px-3 py-2 text-[11px] leading-5 font-mono text-[#d9eeff]">
                              {editorPreview.content.split(/\r?\n/).slice(0, 220).map((line, index) => (
                                <div key={`${index}-${line.length}`} className="flex items-start gap-3">
                                  <span className="w-8 shrink-0 text-right text-[#6797ba]">{index + 1}</span>
                                  <span className="whitespace-pre-wrap break-all text-[#dbebff]">{line || " "}</span>
                                </div>
                              ))}
                              {editorPreview.truncated ? (
                                <p className="pt-2 text-[#9fc3df]">...preview truncated for performance</p>
                              ) : null}
                            </pre>
                          ) : null}
                        </div>
                      ) : null}

                      {worldscreen.view === "website" ? (
                        <iframe
                          title={`worldscreen-${worldscreen.label}`}
                          src={worldscreen.url}
                          className="w-full h-[56vh] rounded-md border border-[rgba(143,214,255,0.3)] bg-[#06101e]"
                          referrerPolicy="no-referrer"
                        />
                      ) : null}

                      {worldscreen.view === "metadata" ? (
                        <>
                          {worldscreen.resourceKind === "audio" ? (
                            <div className="rounded-md border border-[rgba(136,205,238,0.34)] bg-[rgba(5,15,28,0.84)] p-2">
                              <audio
                                ref={worldscreenAudioElementRef}
                                controls
                                preload="metadata"
                                src={worldscreenAudioUrl || worldscreen.url}
                                className="w-full"
                              >
                                <track kind="captions" />
                              </audio>
                              <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[10px] text-[#bcdcf1]">
                                <p className="uppercase tracking-[0.09em] text-[#a9d7f2]">mp3 waveform + spectrogram</p>
                                <p className="font-mono text-[#dff2ff]">{worldscreenAudioClockText}</p>
                              </div>
                              <div
                                data-core-pointer="block"
                                className="mt-2 relative h-56 w-full cursor-ew-resize overflow-hidden rounded border border-[rgba(126,190,226,0.34)] bg-[rgba(5,12,20,0.92)]"
                                onPointerDown={(event) => {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  worldscreenAudioSeekPointerIdRef.current = event.pointerId;
                                  event.currentTarget.setPointerCapture(event.pointerId);
                                  seekWorldscreenAudioFromClientX(event.clientX);
                                }}
                                onPointerMove={(event) => {
                                  if (worldscreenAudioSeekPointerIdRef.current !== event.pointerId) {
                                    return;
                                  }
                                  event.preventDefault();
                                  event.stopPropagation();
                                  seekWorldscreenAudioFromClientX(event.clientX);
                                }}
                                onPointerUp={(event) => {
                                  if (worldscreenAudioSeekPointerIdRef.current !== event.pointerId) {
                                    return;
                                  }
                                  event.preventDefault();
                                  event.stopPropagation();
                                  seekWorldscreenAudioFromClientX(event.clientX);
                                  worldscreenAudioSeekPointerIdRef.current = null;
                                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                                    event.currentTarget.releasePointerCapture(event.pointerId);
                                  }
                                }}
                                onPointerCancel={(event) => {
                                  if (worldscreenAudioSeekPointerIdRef.current === event.pointerId) {
                                    worldscreenAudioSeekPointerIdRef.current = null;
                                  }
                                  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                                    event.currentTarget.releasePointerCapture(event.pointerId);
                                  }
                                }}
                              >
                                <canvas
                                  ref={worldscreenAudioBaseCanvasRef}
                                  className="absolute inset-0 h-full w-full"
                                />
                                <canvas
                                  ref={worldscreenAudioPlayheadCanvasRef}
                                  className="pointer-events-none absolute inset-0 h-full w-full"
                                />
                              </div>
                              <p className="mt-1 text-[10px] text-[#9ec4dc]">
                                click or drag the visual to seek. the vertical line follows playback in real-time.
                              </p>
                              {worldscreenAudioVizStatus === "loading" ? (
                                <p className="mt-1 text-[10px] text-[#a9d8ee]">building waveform and spectrogram...</p>
                              ) : null}
                              {worldscreenAudioVizStatus === "error" ? (
                                <p className="mt-1 text-[10px] text-[#ffd8be]">{worldscreenAudioVizError}</p>
                              ) : null}
                            </div>
                          ) : worldscreen.resourceKind === "video" ? (
                            <video
                              controls
                              preload="metadata"
                              src={worldscreen.remoteFrameUrl || worldscreen.url}
                              className="w-full max-h-[52vh] rounded-md bg-[rgba(4,9,16,0.86)]"
                            >
                              <track kind="captions" />
                            </video>
                          ) : worldscreen.resourceKind === "image" || worldscreen.remoteFrameUrl ? (
                            <img
                              src={worldscreen.remoteFrameUrl || worldscreen.url}
                              alt={`crawler frame for ${worldscreen.label}`}
                              className="w-full max-h-[52vh] rounded-md object-contain bg-[rgba(4,9,16,0.86)]"
                              loading="lazy"
                            />
                          ) : (
                            <p className="text-[11px] text-[#9fc3df]">
                              No crawler frame is cached for this remote resource yet.
                            </p>
                          )}

                          {(worldscreen.resourceKind === "website" || worldscreen.resourceKind === "link") ? (
                            <div className="mt-2 flex flex-wrap gap-1.5">
                              <a
                                href={worldscreen.url}
                                target="_blank"
                                rel="noreferrer"
                                className="rounded border border-[rgba(113,181,220,0.35)] bg-[rgba(15,45,68,0.5)] px-2 py-0.5 text-[11px] text-[#d2edff]"
                              >
                                open website
                              </a>
                              {worldscreen.sourceUrl ? (
                                <a
                                  href={worldscreen.sourceUrl}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="rounded border border-[rgba(113,181,220,0.35)] bg-[rgba(15,45,68,0.5)] px-2 py-0.5 text-[11px] text-[#d2edff]"
                                >
                                  open source link
                                </a>
                              ) : null}
                            </div>
                          ) : null}
                        </>
                      ) : null}
                    </section>
                  </div>
                </div>
              ) : null}

              {worldscreenMode === "conversation" ? (
                <div className="h-full rounded-xl border border-[rgba(143,214,255,0.38)] bg-[linear-gradient(180deg,rgba(5,16,30,0.92),rgba(5,15,26,0.88))] overflow-hidden p-3 text-[#d9eeff]">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-[#9fd0ef]">true graph conversation</p>
                  <p className="mt-1 text-[10px] text-[#9ec4dd] break-all">
                    compact nexus ref: {activeImageCommentRef || "(unknown)"}
                  </p>

                  <div className="mt-2 grid gap-2 sm:grid-cols-2">
                    <div className="grid gap-1">
                      <span className="text-[10px] uppercase tracking-[0.08em] text-[#89b1cc]">presence account</span>
                      <input
                        value={presenceAccountId}
                        onChange={(event) => setPresenceAccountId(event.target.value)}
                        list="presence-account-options"
                        placeholder="witness_thread"
                        className="rounded border border-[rgba(140,196,231,0.38)] bg-[rgba(11,24,38,0.82)] px-2 py-1 text-[12px] text-[#def1ff] outline-none focus:border-[rgba(165,220,255,0.68)]"
                      />
                    </div>
                    {worldscreen.resourceKind === "image" ? (
                      <div className="grid gap-1">
                        <span className="text-[10px] uppercase tracking-[0.08em] text-[#89b1cc]">analysis prompt</span>
                        <input
                          value={imageCommentPrompt}
                          onChange={(event) => setImageCommentPrompt(event.target.value)}
                          placeholder="Describe the image evidence and one next action."
                          className="rounded border border-[rgba(140,196,231,0.38)] bg-[rgba(11,24,38,0.82)] px-2 py-1 text-[12px] text-[#def1ff] outline-none focus:border-[rgba(165,220,255,0.68)]"
                        />
                      </div>
                    ) : null}
                  </div>
                  <datalist id="presence-account-options">
                    {presenceAccounts.map((account) => (
                      <option key={account.presence_id} value={account.presence_id}>
                        {account.display_name}
                      </option>
                    ))}
                  </datalist>

                  {imageCommentParentId ? (
                    <p className="mt-2 text-[10px] text-[#a9d0ea]">
                      replying to {resolvePresenceName(commentEntryById.get(imageCommentParentId)?.presence_id ?? "")}
                      <button
                        type="button"
                        onClick={() => setImageCommentParentId("")}
                        className="ml-2 rounded border border-[rgba(140,194,226,0.38)] bg-[rgba(17,44,66,0.44)] px-1.5 py-0.5 text-[10px] text-[#d8edff]"
                      >
                        clear
                      </button>
                    </p>
                  ) : (
                    <p className="mt-2 text-[10px] text-[#8fb6d1]">posting a new root comment</p>
                  )}

                  <div className="mt-2 grid gap-1">
                    <span className="text-[10px] uppercase tracking-[0.08em] text-[#89b1cc]">comment draft</span>
                    <textarea
                      value={imageCommentDraft}
                      onChange={(event) => setImageCommentDraft(event.target.value)}
                      rows={3}
                      placeholder="Commentary appears here; edit before posting if needed."
                      className="rounded border border-[rgba(140,196,231,0.38)] bg-[rgba(11,24,38,0.82)] px-2 py-1 text-[12px] leading-5 text-[#def1ff] outline-none focus:border-[rgba(165,220,255,0.68)]"
                    />
                  </div>

                  <div className="mt-2 flex flex-wrap gap-2">
                    {worldscreen.resourceKind === "image" ? (
                      <button
                        type="button"
                        onClick={() => {
                          void submitGeneratedImageCommentary();
                        }}
                        disabled={imageCommentBusy}
                        className="rounded border border-[rgba(131,223,255,0.52)] bg-[rgba(40,113,148,0.34)] px-2.5 py-1 text-[11px] font-semibold text-[#e2f6ff] disabled:opacity-60"
                      >
                        {imageCommentBusy ? "analyzing..." : "analyze with qwen3-vl"}
                      </button>
                    ) : null}
                    <button
                      type="button"
                      onClick={() => {
                        void submitManualImageComment();
                      }}
                      disabled={imageCommentBusy || imageCommentDraft.trim().length === 0}
                      className="rounded border border-[rgba(154,206,247,0.48)] bg-[rgba(49,96,137,0.28)] px-2.5 py-1 text-[11px] font-semibold text-[#dff2ff] disabled:opacity-60"
                    >
                      post comment
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        refreshNexusComments(activeImageCommentRef);
                      }}
                      disabled={imageCommentBusy || !activeImageCommentRef}
                      className="rounded border border-[rgba(154,206,247,0.34)] bg-[rgba(20,60,94,0.24)] px-2.5 py-1 text-[11px] font-semibold text-[#cfe7fb] disabled:opacity-60"
                    >
                      refresh comments
                    </button>
                  </div>

                  {imageCommentError ? (
                    <p className="mt-2 text-[11px] text-[#ffd7be]">{imageCommentError}</p>
                  ) : null}

                  <div className="mt-2 rounded border border-[rgba(120,182,220,0.3)] bg-[rgba(4,12,21,0.62)] p-2 max-h-[calc(100%-17.2rem)] overflow-auto">
                    {imageCommentsLoading ? (
                      <p className="text-[11px] text-[#9bc2dd]">loading nexus comments...</p>
                    ) : null}
                    {!imageCommentsLoading && flattenedImageComments.length === 0 ? (
                      <p className="text-[11px] text-[#9bc2dd]">no comments yet for this nexus.</p>
                    ) : null}
                    {!imageCommentsLoading
                      ? flattenedImageComments.map(({ entry, depth }) => (
                          <article
                            key={entry.id}
                            className="pb-2 mb-2 border-b border-[rgba(108,164,199,0.24)] last:border-none last:pb-0 last:mb-0"
                            style={{ marginLeft: `${Math.min(depth * 18, 72)}px` }}
                          >
                            <p className="text-[10px] uppercase tracking-[0.08em] text-[#88b3d0]">
                              {resolvePresenceName(entry.presence_id)}
                              <span className="ml-1 text-[#6f95b1]">{timestampLabel(entry.created_at || entry.time)}</span>
                            </p>
                            <p className="text-[12px] leading-5 text-[#def2ff] whitespace-pre-wrap break-words">{entry.comment}</p>
                            <button
                              type="button"
                              onClick={() => setImageCommentParentId(entry.id)}
                              className="mt-1 rounded border border-[rgba(128,186,220,0.34)] bg-[rgba(13,39,58,0.54)] px-1.5 py-0.5 text-[10px] text-[#d3ebff]"
                            >
                              reply in-thread
                            </button>
                          </article>
                        ))
                      : null}
                  </div>
                </div>
              ) : null}

              {worldscreenMode === "stats" ? (
                <div className="h-full rounded-xl border border-[rgba(143,214,255,0.38)] bg-[linear-gradient(180deg,rgba(5,16,30,0.92),rgba(5,15,26,0.88))] overflow-auto p-3 text-[#d9eeff]">
                  <p className="text-[11px] uppercase tracking-[0.12em] text-[#9fd0ef]">nexus stats</p>
                  <div className="mt-2 grid gap-2 sm:grid-cols-2">
                    <div className="rounded border border-[rgba(112,174,207,0.32)] bg-[rgba(8,22,35,0.62)] p-2">
                      <p className="text-[10px] uppercase tracking-[0.1em] text-[#95c4e1]">resource</p>
                      <p className="text-[12px] text-[#e4f5ff]">{resourceKindLabel(worldscreen.resourceKind)} / {worldscreen.nodeKind}</p>
                      <p className="mt-1 text-[11px] text-[#b8d8ec] break-all">node: {worldscreen.nodeId || "(unknown)"}</p>
                      <p className="text-[11px] text-[#b8d8ec] break-all">ref: {activeImageCommentRef || "(unknown)"}</p>
                    </div>

                    <div className="rounded border border-[rgba(112,174,207,0.32)] bg-[rgba(8,22,35,0.62)] p-2">
                      <p className="text-[10px] uppercase tracking-[0.1em] text-[#95c4e1]">conversation</p>
                      <p className="text-[12px] text-[#e4f5ff]">comments: {imageCommentStats.total}</p>
                      <p className="text-[11px] text-[#b8d8ec]">participants: {imageCommentStats.participants}</p>
                      <p className="text-[11px] text-[#b8d8ec]">threads: {imageCommentStats.rootCommentCount}</p>
                      <p className="text-[11px] text-[#b8d8ec]">max depth: {imageCommentStats.deepestDepth}</p>
                    </div>
                  </div>

                  <div className="mt-3 rounded border border-[rgba(112,174,207,0.3)] bg-[rgba(8,22,35,0.52)] p-2 text-[11px] text-[#cde7fb]">
                    <p>discovered: {worldscreen.discoveredAt || "n/a"}</p>
                    <p>fetched: {worldscreen.fetchedAt || "n/a"}</p>
                    <p>encountered: {worldscreen.encounteredAt || "n/a"}</p>
                    <p>latest comment: {imageCommentStats.latestAt ? timestampLabel(imageCommentStats.latestAt) : "n/a"}</p>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
          </div>
        );
        if (typeof document !== "undefined") {
          return createPortal(overlay, document.body);
        }
        return overlay;
      })() : null}
    </div>
  );
}
