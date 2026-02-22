import { useRef, useEffect, useState, useCallback, useMemo } from "react";
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
  nodeKind: "file" | "crawler";
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
  nodeKind: "file" | "crawler";
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
  nodeKind: "file" | "crawler",
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
  const value = String(raw || "").trim();
  if (!value) {
    return "";
  }
  if (value.startsWith("field:")) {
    return value.slice("field:".length).trim();
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

function spreadPresenceAnchors(forms: Array<any>): Array<any> {
  if (forms.length <= 0) {
    return [];
  }

  const positioned = forms.map((row, index) => {
    const presenceId = canonicalPresenceId(String(row?.id ?? ""));
    const fallbackAngle = stablePresenceRatio(presenceId || `presence-${index}`, 3) * Math.PI * 2;
    const fallbackRadius = 0.24 + (stablePresenceRatio(presenceId || `presence-${index}`, 11) * 0.4);
    const fallbackX = 0.5 + Math.cos(fallbackAngle) * fallbackRadius;
    const fallbackY = 0.5 + Math.sin(fallbackAngle) * fallbackRadius;
    const baseX = clamp01(Number(row?.x ?? fallbackX));
    const baseY = clamp01(Number(row?.y ?? fallbackY));
    const expandedX = clamp01(0.5 + (baseX - 0.5) * 1.15);
    const expandedY = clamp01(0.5 + (baseY - 0.5) * 1.15);
    return {
      ...row,
      id: presenceId || String(row?.id ?? `presence-${index}`),
      x: expandedX,
      y: expandedY,
      spreadBaseX: expandedX,
      spreadBaseY: expandedY,
      hue: Number.isFinite(Number(row?.hue))
        ? Number(row?.hue)
        : presenceHueFromId(presenceId || String(row?.id ?? "")),
    };
  });

  const count = positioned.length;
  const minSpacing = count >= 26 ? 0.1 : count >= 18 ? 0.11 : 0.12;
  const minX = 0.06;
  const maxX = 0.94;
  const minY = 0.08;
  const maxY = 0.92;

  for (let iter = 0; iter < 20; iter += 1) {
    for (let left = 0; left < positioned.length; left += 1) {
      for (let right = left + 1; right < positioned.length; right += 1) {
        const leftRow = positioned[left];
        const rightRow = positioned[right];
        const dx = rightRow.x - leftRow.x;
        const dy = rightRow.y - leftRow.y;
        const distance = Math.hypot(dx, dy);
        if (distance >= minSpacing) {
          continue;
        }
        const unitX = distance > 0.0001
          ? dx / distance
          : Math.cos((left + 1) * 0.73 + (right + 1) * 0.29);
        const unitY = distance > 0.0001
          ? dy / distance
          : Math.sin((left + 1) * 0.61 + (right + 1) * 0.41);
        const push = (minSpacing - distance) * 0.5;
        leftRow.x = clamp01(leftRow.x - unitX * push);
        leftRow.y = clamp01(leftRow.y - unitY * push);
        rightRow.x = clamp01(rightRow.x + unitX * push);
        rightRow.y = clamp01(rightRow.y + unitY * push);
      }
    }
    for (const row of positioned) {
      row.x = Math.min(maxX, Math.max(minX, row.x + (row.spreadBaseX - row.x) * 0.02));
      row.y = Math.min(maxY, Math.max(minY, row.y + (row.spreadBaseY - row.y) * 0.02));
    }
  }

  return positioned.map((row) => {
    const cleaned = { ...row };
    delete (cleaned as any).spreadBaseX;
    delete (cleaned as any).spreadBaseY;
    return {
      ...cleaned,
      x: Math.min(maxX, Math.max(minX, Number(cleaned.x ?? 0.5))),
      y: Math.min(maxY, Math.max(minY, Number(cleaned.y ?? 0.5))),
    };
  });
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
        : backend.includes("gpu") || backend.includes("vllm") || backend.includes("ollama")
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

export type OverlayViewId =
  | "omni"
  | "presence"
  | "file-impact"
  | "file-graph"
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

function resourceKindForNode(node: any): GraphNodeResourceKind {
  const nodeKind = String(node?.node_type ?? "file");
  if (nodeKind === "crawler") {
    return classifyCrawlerResourceKind(node);
  }
  return classifyFileResourceKind(node);
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
  nodeKind: "file" | "crawler",
  resourceKind: GraphNodeResourceKind,
): GraphWorldscreenView {
  if (resourceKind === "image") {
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
  nodeKind: "file" | "crawler",
  resourceKind: GraphNodeResourceKind,
): string {
  const kindText = resourceKindLabel(resourceKind);
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

function openUrlForGraphNode(node: any, nodeKind: "file" | "crawler"): string {
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
  const worldscreenRef = useRef<GraphWorldscreenState | null>(null);
  useEffect(() => {
    worldscreenRef.current = worldscreen;
  }, [worldscreen]);
  const [worldscreenMode, setWorldscreenMode] = useState<GraphWorldscreenMode>("overview");
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
  }, [catalog, computeJobFilter, simulation]);

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
        setWorldscreenMode("overview");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [worldscreen]);

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

    const resolveDepth = (point: { x: number; y: number; size: number; z?: number }, idx: number): number => {
      const explicit = Number(point.z);
      if (Number.isFinite(explicit)) {
        return Math.max(-1, Math.min(1, explicit));
      }
      const noise = Math.sin((point.x * 15.37) + (point.y * 27.91) + (idx * 0.913)) * 43758.5453123;
      const layer = (noise - Math.floor(noise)) * 2 - 1;
      const radial = Math.min(1, Math.hypot(point.x, point.y));
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
    const projectionMatrix = new Float32Array(16);
    const viewMatrix = new Float32Array(16);

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
        trailSnapshots.length = 0;
        return;
      }
      const snapshot = new Float32Array(count * 3);
      snapshot.set(targetPositions.subarray(0, count * 3));
      trailSnapshots.push({
        positions: snapshot,
        count,
      });
      if (trailSnapshots.length > TRAIL_LAYER_COUNT) {
        trailSnapshots.shift();
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
      const points = hasOverlayParticles && renderOverlayWithWebgl
        ? overlayRows.map((row) => ({
            x: clampValue(Number((row as any)?.x ?? 0), -1, 1),
            y: clampValue(Number((row as any)?.y ?? 0), -1, 1),
            size: clampValue(Number((row as any)?.size ?? 1.5), 0.2, 4.8),
            r: clamp01(Number((row as any)?.r ?? 0.48)),
            g: clamp01(Number((row as any)?.g ?? 0.64)),
            b: clamp01(Number((row as any)?.b ?? 0.92)),
            z: clampValue(
              Number((row as any)?.mass ?? (row as any)?.route_probability ?? (row as any)?.drift_score ?? 0),
              -1,
              1,
            ),
          }))
        : (hasOverlayParticles ? [] : (state.points || []));
      const sourceCount = points.length;
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
        const p = points[sourceIndex];
        const z = resolveDepth(p, sourceIndex);
        const writeIndex = dynamicOffset + i;
        targetPositions[writeIndex * 3] = p.x;
        targetPositions[(writeIndex * 3) + 1] = p.y;
        targetPositions[(writeIndex * 3) + 2] = z;
        targetSizes[writeIndex] = p.size;
        targetColors[writeIndex * 3] = p.r;
        targetColors[(writeIndex * 3) + 1] = p.g;
        targetColors[(writeIndex * 3) + 2] = p.b;
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
      trailSnapshots.length = 0;
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
      kind: "presence" | "file" | "crawler";
      label: string;
      x: number;
      y: number;
      radius: number;
      node?: any;
      nodeKind?: "file" | "crawler";
      resourceKind?: GraphNodeResourceKind;
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
      return rows.length > 0 ? spreadPresenceAnchors(rows as any) : spreadPresenceAnchors(fallbackNamedForms as any);
    };

    let rafId = 0;
    let lastPaintTs = 0;
    let canvasWidth = 0;
    let canvasHeight = 0;
    let hotspots: OverlayHotspot[] = [];
    let userPresenceMouseEmitMs = 0;
    let pulse = { x: 0.5, y: 0.5, power: 0, atMs: 0, target: "particle_field" };
    let pointerField = { x: 0.5, y: 0.5, power: 0, inside: false };
    const PARTICLE_TRAIL_FRAME_COUNT = 12;
    const particleTrailFrames: number[][] = [];
    let lastTrailFrameKey = "";

    const findNearestHotspot = (xRatio: number, yRatio: number): OverlayHotspot | null => {
      let match: { row: OverlayHotspot; distance: number } | null = null;
      for (const row of hotspots) {
        const distance = Math.hypot(xRatio - row.x, yRatio - row.y);
        const threshold = row.radius * 1.8;
        if (distance > threshold) {
          continue;
        }
        if (!match || distance < match.distance) {
          match = { row, distance };
        }
      }
      return match?.row ?? null;
    };

    const shouldOpenWorldscreen = (nodeKind: "file" | "crawler", nodeId: string): boolean => {
      const key = `${nodeKind}:${nodeId}`;
      const nowMs = performance.now();
      const previous = lastNexusPointerTapRef.current;
      const isDoubleTap = Boolean(previous && previous.key === key && (nowMs - previous.atMs) <= 360);
      lastNexusPointerTapRef.current = { key, atMs: nowMs };
      return isDoubleTap;
    };

    const openWorldscreenForNode = (node: any, nodeKind: "file" | "crawler", xRatio: number, yRatio: number) => {
      const resourceKind = resourceKindForNode(node);
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
      const worldscreenUrl = resolveWorldscreenUrl(openUrl, nodeKind, domain);
      if (!worldscreenUrl) {
        return;
      }
      const frameUrl = remoteFrameUrlForNode(node, worldscreenUrl, resourceKind);
      const imageRef = String(
        node?.source_rel_path
        || node?.archive_rel_path
        || node?.archived_rel_path
        || node?.url
        || worldscreenUrl,
      ).trim();
      const commentRef = nexusCommentRefForNode(node, nodeKind, worldscreenUrl) || imageRef;
      setWorldscreenMode("overview");
      setWorldscreen({
        nodeId: graphNodeId,
        commentRef,
        url: worldscreenUrl,
        imageRef,
        label,
        nodeKind,
        resourceKind,
        anchorRatioX: clamp01(xRatio),
        anchorRatioY: clamp01(yRatio),
        view: resourceKind === "image"
          ? "metadata"
          : isRemoteHttpUrl(worldscreenUrl)
            ? "metadata"
            : worldscreenViewForNode(node, nodeKind, resourceKind),
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
      });
    };

    const draw = (ts: number) => {
      const currentSimulation = simulationRef.current;
      const simulationTimestamp = String(currentSimulation?.timestamp ?? "").trim();
      const allFieldParticles = (() => {
        const directRows = currentSimulation?.presence_dynamics?.field_particles ?? currentSimulation?.field_particles;
        return Array.isArray(directRows) ? (directRows as BackendFieldParticle[]) : [];
      })();
      const livePresenceCentroids = new Map<string, { sumX: number; sumY: number; count: number }>();
      for (let index = 0; index < allFieldParticles.length; index += 1) {
        const row = allFieldParticles[index] as any;
        const presenceId = String(row?.presence_id ?? "").trim();
        if (!presenceId) {
          continue;
        }
        const xRatio = toRatio(Number(row?.x ?? 0.5));
        const yRatio = toRatio(Number(row?.y ?? 0.5));
        const current = livePresenceCentroids.get(presenceId) ?? { sumX: 0, sumY: 0, count: 0 };
        current.sumX += xRatio;
        current.sumY += yRatio;
        current.count += 1;
        livePresenceCentroids.set(presenceId, current);
      }
      const renderFallbackManifestAnchors = livePresenceCentroids.size === 0;
      const isBackgroundMode = backgroundModeRef.current;
      const isInteractive = interactiveRef.current;
      const currentOverlayView = overlayViewRef.current;
      const currentLayerVisibility = layerVisibilityRef.current;
      const baselineFrameMs = allFieldParticles.length > 1500 ? 34 : allFieldParticles.length > 900 ? 24 : 16;
      const targetFrameMs = (!isInteractive && !isBackgroundMode)
        ? Math.max(34, baselineFrameMs)
        : baselineFrameMs;
      if (lastPaintTs > 0 && (ts - lastPaintTs) < targetFrameMs) {
        rafId = requestAnimationFrame(draw);
        return;
      }
      lastPaintTs = ts;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const nextWidth = Math.max(1, Math.floor(rect.width * dpr));
      const nextHeight = Math.max(1, Math.floor(rect.height * dpr));
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

      const pointRows: number[] = [];
      const lineRows: number[] = [];
      const presencePointRowsCurrent: number[] = [];
      hotspots = [];
      const graphNodeLookup = new Map<string, { x: number; y: number; node: any; nodeKind: "file" | "crawler" }>();

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
        || currentOverlayView === "crawler-graph"
      );
      const showFileGraphLayer = currentLayerVisibility?.["file-graph"]
        ?? (!isBackgroundMode && (currentOverlayView === "omni" || currentOverlayView === "file-graph"));
      const showCrawlerGraphLayer = currentLayerVisibility?.["crawler-graph"]
        ?? (!isBackgroundMode && (currentOverlayView === "omni" || currentOverlayView === "crawler-graph"));

      const namedForms = resolveNamedFormsForWebgl();
      const namedFormPoints: Array<{ x: number; y: number; sizePx: number; r: number; g: number; b: number; a: number }> = [];
      if (showPresenceLayer && renderFallbackManifestAnchors) {
        for (let index = 0; index < namedForms.length; index += 1) {
          const form = namedForms[index] as any;
          const pulseOffset = Math.sin((ts * 0.001 * 1.8) + index * 0.7) * 0.12 + 0.88;
          const [r, g, b] = toRgbFromHue(Number(form?.hue ?? 180));
          namedFormPoints.push({
            x: clamp01(Number(form.x ?? 0.5)),
            y: clamp01(Number(form.y ?? 0.5)),
            sizePx: (13 + pulseOffset * 8.5) * dpr,
            r,
            g,
            b,
            a: 0.96,
          });
          hotspots.push({
            id: String(form.id ?? `presence-${index}`),
            kind: "presence",
            label: String(form.en ?? form.id ?? "presence"),
            x: clamp01(Number(form.x ?? 0.5)),
            y: clamp01(Number(form.y ?? 0.5)),
            radius: 0.03,
          });
        }
      }

      const fileGraph = currentSimulation?.file_graph ?? catalogRef.current?.file_graph;
      const crawlerGraph = currentSimulation?.crawler_graph ?? catalogRef.current?.crawler_graph;

      const mergedNodeRows = [
        ...(Array.isArray(fileGraph?.nodes) ? fileGraph.nodes : []),
        ...(Array.isArray(fileGraph?.file_nodes) ? fileGraph.file_nodes : []),
        ...(Array.isArray(fileGraph?.crawler_nodes) ? fileGraph.crawler_nodes : []),
        ...(Array.isArray(crawlerGraph?.nodes) ? crawlerGraph.nodes : []),
        ...(Array.isArray(crawlerGraph?.crawler_nodes) ? crawlerGraph.crawler_nodes : []),
      ];

      const seenNodeIds = new Set<string>();
      const maxNodeCount = 1400;
      for (let index = 0; index < mergedNodeRows.length && graphNodeLookup.size < maxNodeCount; index += 1) {
        const node = mergedNodeRows[index] as any;
        const nodeId = String(node?.id ?? "").trim();
        if (!nodeId || seenNodeIds.has(nodeId)) {
          continue;
        }
        seenNodeIds.add(nodeId);
        const rawType = String(node?.node_type ?? "").trim().toLowerCase();
        const nodeKind: "file" | "crawler" = rawType === "crawler" || String(node?.crawler_kind ?? "").trim().length > 0
          ? "crawler"
          : "file";
        if (!showFileGraphLayer && nodeKind === "file") {
          continue;
        }
        if (!showCrawlerGraphLayer && nodeKind === "crawler") {
          continue;
        }
        const xRatio = toRatio(Number(node?.x ?? 0.5));
        const yRatio = toRatio(Number(node?.y ?? 0.5));
        const resourceKind = resourceKindForNode(node);
        const [r, g, b] = resourceColor(resourceKind);
        const importance = clamp01(Number(node?.importance ?? 0.35));
        const nodeSize = (nodeKind === "crawler" ? 4.4 : 5.1) + importance * 4.2;
        addPoint(xRatio, yRatio, nodeSize * dpr, r, g, b, nodeKind === "crawler" ? 0.84 : 0.78);
        graphNodeLookup.set(nodeId, { x: xRatio, y: yRatio, node, nodeKind });
        hotspots.push({
          id: nodeId,
          kind: nodeKind,
          node,
          nodeKind,
          resourceKind,
          label: shortPathLabel(String(node?.title ?? node?.domain ?? node?.label ?? nodeId)),
          x: xRatio,
          y: yRatio,
          radius: nodeKind === "crawler" ? 0.022 : 0.019,
        });
      }

      const fileEdges = showFileGraphLayer && Array.isArray(fileGraph?.edges) ? fileGraph.edges : [];
      const crawlerEdges = showCrawlerGraphLayer && Array.isArray(crawlerGraph?.edges) ? crawlerGraph.edges : [];
      const mergedEdges = [...fileEdges, ...crawlerEdges];
      const maxEdgeCount = 3200;

      const daimoiCounts: Record<string, number> = {};
      let totalResourceDaimoi = 0;
      for (let i = 0; i < allFieldParticles.length; i += 1) {
        const row = allFieldParticles[i] as any;
        if (row.resource_daimoi) {
          const type = String(row.resource_type ?? "cpu");
          daimoiCounts[type] = (daimoiCounts[type] ?? 0) + 1;
          totalResourceDaimoi += 1;
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

      for (let index = 0; index < mergedEdges.length && index < maxEdgeCount; index += 1) {
        const edge = mergedEdges[index] as any;
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
        const kind = String(edge?.kind ?? "").trim().toLowerCase();
        const [r, g, b, a] = edgeColorByKind(kind);
        
        const mix = flowIntensity * strobe * 0.6;
        const fr = r * (1 - mix) + dr * mix;
        const fg = g * (1 - mix) + dg * mix;
        const fb = b * (1 - mix) + db * mix;
        const fa = Math.min(1.0, a + (flowIntensity * 0.3));

        addLine(source.x, source.y, target.x, target.y, fr, fg, fb, fa);
      }


      if (showPresenceLayer) {
        const particleRows = allFieldParticles;
        const maxParticleCount = Math.max(240, Math.round(2600 * particleDensityRef.current));
        const step = Math.max(1, Math.ceil(particleRows.length / Math.max(1, maxParticleCount)));
        for (let index = 0; index < particleRows.length; index += step) {
          const row = particleRows[index] as any;
          const xRatio = toRatio(Number(row?.x ?? 0.5));
          const yRatio = toRatio(Number(row?.y ?? 0.5));
          const size = clampValue(Number(row?.size ?? 1.1), 0.3, 5.4);
          addPoint(
            xRatio,
            yRatio,
            (size * 3.1 + 1.1) * particleScaleRef.current * dpr,
            Number(row?.r ?? 0.58),
            Number(row?.g ?? 0.72),
            Number(row?.b ?? 0.92),
            0.66,
          );
          presencePointRowsCurrent.push(
            xRatio * canvasWidth,
            yRatio * canvasHeight,
            Math.max(1.2, (size * 3.1 + 1.1) * particleScaleRef.current * dpr),
            clamp01(Number(row?.r ?? 0.58)),
            clamp01(Number(row?.g ?? 0.72)),
            clamp01(Number(row?.b ?? 0.92)),
            0.66,
          );

          const routeNodeId = String(row?.route_node_id ?? "").trim();
          const graphNodeId = String(row?.graph_node_id ?? "").trim();
          if (routeNodeId && graphNodeId) {
            const source = graphNodeLookup.get(routeNodeId);
            const target = graphNodeLookup.get(graphNodeId);
            if (source && target) {
              addLine(source.x, source.y, target.x, target.y, 0.58, 0.88, 1.0, 0.14);
            }
          }
        }

        if (livePresenceCentroids.size > 0) {
          Array.from(livePresenceCentroids.entries())
            .sort((left, right) => right[1].count - left[1].count)
            .slice(0, 180)
            .forEach(([presenceId, centroid]) => {
              if (centroid.count <= 0) {
                return;
              }
              hotspots.push({
                id: presenceId,
                kind: "presence",
                label: shortPresenceIdLabel(presenceId),
                x: clamp01(centroid.sumX / centroid.count),
                y: clamp01(centroid.sumY / centroid.count),
                radius: 0.022,
              });
            });
        }
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
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lineRows), gl.DYNAMIC_DRAW);
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
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(rows), gl.DYNAMIC_DRAW);
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

      const currentWorldscreen = worldscreenRef.current;
      if (currentWorldscreen) {
        const targetId = String(currentWorldscreen.nodeId ?? "").trim();
        const target = graphNodeLookup.get(targetId);
        if (target) {
          const glass = glassCenterRatioRef.current;
          // Draw connecting line from node to lens center
          addLine(
            target.x,
            target.y,
            glass.x,
            glass.y,
            0.38,
            0.64,
            0.92,
            0.48,
          );
          // Highlight the node
          addPoint(
            target.x,
            target.y,
            16 * dpr,
            0.38,
            0.64,
            0.92,
            0.55,
          );
        }
      }

      if (namedFormPoints.length > 0) {
        const namedFormRows: number[] = [];
        for (let index = 0; index < namedFormPoints.length; index += 1) {
          const row = namedFormPoints[index] as any;
          namedFormRows.push(
            row.x * canvasWidth,
            row.y * canvasHeight,
            Math.max(2.2, row.sizePx),
            clamp01(row.r),
            clamp01(row.g),
            clamp01(row.b),
            clamp01(row.a),
          );
        }
        drawPointCloud(namedFormRows, 1.0);
      }

      if (showPresenceLayer && presencePointRowsCurrent.length > 0) {
        const frameKey = simulationTimestamp || String(Math.round(ts));
        if (frameKey !== lastTrailFrameKey) {
          particleTrailFrames.push(presencePointRowsCurrent.slice());
          if (particleTrailFrames.length > PARTICLE_TRAIL_FRAME_COUNT) {
            particleTrailFrames.shift();
          }
          lastTrailFrameKey = frameKey;
        }
      } else {
        particleTrailFrames.length = 0;
        lastTrailFrameKey = "";
      }

      if (metaRef.current) {
        metaRef.current.textContent = `webgl overlay particles:${allFieldParticles.length} nodes:${graphNodeLookup.size} hotspots:${hotspots.length}`;
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
    ): { hitNode: boolean; openedWorldscreen: boolean; target: string } => {
      const xRatio = clamp01(xRatioInput);
      const yRatio = clamp01(yRatioInput);
      const hit = findNearestHotspot(xRatio, yRatio);
      if (hit?.node && hit.nodeKind) {
        const isDoubleTap = shouldOpenWorldscreen(hit.nodeKind, hit.id);
        const openWorldscreen = Boolean(options?.openWorldscreen) || true; // Always open on click
        if (openWorldscreen) {
          openWorldscreenForNode(hit.node, hit.nodeKind, xRatio, yRatio);
        }
        onNexusInteractionRef.current?.({
          nodeId: hit.id,
          nodeKind: hit.nodeKind,
          resourceKind: hit.resourceKind ?? "unknown",
          label: hit.label,
          xRatio,
          yRatio,
          openWorldscreen,
          isDoubleTap,
        });
        onUserPresenceInputRef.current?.({
          kind: "click",
          target: hit.id,
          message: `click graph node ${hit.id}`,
          xRatio,
          yRatio,
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
            : `focused node: ${hit.label} (double tap to open hologram)`;
        }
        pulseAt(xRatio, yRatio, 1.0, hit.id);
        return {
          hitNode: true,
          openedWorldscreen: openWorldscreen,
          target: hit.id,
        };
      }

      const target = hit?.id || "particle_field";
      onUserPresenceInputRef.current?.({
        kind: "click",
        target,
        message: `click simulation field ${target}`,
        xRatio,
        yRatio,
        embedDaimoi: true,
        meta: {
          source: "simulation-canvas",
          renderer: "webgl",
        },
      });
      pulseAt(xRatio, yRatio, 0.96, target);
      return {
        hitNode: false,
        openedWorldscreen: false,
        target,
      };
    };

    const api = {
      pulseAt,
      singAll: () => {},
      getAnchorRatio: (kind: string, targetId: string) => {
        const target = String(targetId ?? "").trim();
        if (!target) {
          return null;
        }
        if (kind === "node" || kind === "file" || kind === "crawler") {
          const match = hotspots.find((row) => (row.kind === "file" || row.kind === "crawler") && row.id === target);
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
  const worldscreenPlacement = worldscreen
    ? resolveWorldscreenPlacement(worldscreen, containerRef.current, glassCenterRatio)
    : null;
  const activeImageCommentRef = worldscreen ? worldscreenCommentRef(worldscreen) : "";
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
  const resourceEconomyHud = useMemo(() => {
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
  }, [simulation]);
  const liveFieldParticleCount = activeFieldParticleRows.length;
  const fallbackPointCount = Array.isArray(simulation?.points) ? simulation.points.length : 0;
  const particleLegendStats = useMemo(() => {
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
  }, [activeFieldParticleRows]);
  const overlayParticleModeActive = renderRichOverlayParticles && liveFieldParticleCount > 0;

  return (
    <div ref={containerRef} className={containerClassName}>
      <canvas ref={canvasRef} style={{ height: canvasHeight, pointerEvents: canvasPointerEvents }} className="block w-full" />
      <canvas ref={overlayRef} style={{ height: canvasHeight, pointerEvents: canvasPointerEvents }} className="absolute inset-0 w-full touch-none" />
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
      {interactive && worldscreen ? (
        <div className="absolute inset-0 z-20 pointer-events-none">
          <section
            data-core-pointer="block"
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
                <a
                  href={worldscreen.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs px-2.5 py-1 rounded-md border border-[rgba(145,190,240,0.42)] text-[#d4ebff] hover:bg-[rgba(72,119,170,0.2)]"
                >
                  new tab
                </a>
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
                          {worldscreen.resourceKind === "video" ? (
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
      ) : null}
    </div>
  );
}
