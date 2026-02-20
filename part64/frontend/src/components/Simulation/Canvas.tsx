import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import type {
  SimulationState,
  Catalog,
  FileGraph,
  CrawlerGraph,
  TruthState,
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

type GraphNodeShape = "circle" | "square" | "diamond" | "triangle" | "hexagon";

interface GraphNodeVisualSpec {
  hue: number;
  saturation: number;
  value: number;
  shape: GraphNodeShape;
  liftBoost: number;
  glowBoost: number;
}

interface FileGraphRenderEdge {
  id: string;
  source: string;
  target: string;
  field: string;
  kind: string;
  weight: number;
}

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

function resolveParticlePresenceId(row: Partial<BackendFieldParticle> | null | undefined): string {
  const direct = String((row as any)?.presence_id ?? "").trim();
  if (direct) {
    return direct;
  }
  const owner = String((row as any)?.owner_presence_id ?? "").trim();
  if (owner) {
    return owner;
  }
  return String((row as any)?.owner ?? "").trim();
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

type PresenceSigilKind =
  | "bisected-circle"
  | "triangle-dot"
  | "crossed-ring"
  | "tri-spiral"
  | "broken-ring"
  | "double-notch";

type PresenceRingStyle = "solid" | "dashed" | "dotted" | "chain" | "double";

interface PresenceIdentitySignature {
  id: string;
  sigil: PresenceSigilKind;
  ringStyle: PresenceRingStyle;
  fieldLayerSignature: number;
  notchAngle: number;
  rotation: number;
}

const PRESENCE_SIGIL_SEQUENCE: PresenceSigilKind[] = [
  "bisected-circle",
  "triangle-dot",
  "crossed-ring",
  "tri-spiral",
  "broken-ring",
  "double-notch",
];

const PRESENCE_RING_STYLE_SEQUENCE: PresenceRingStyle[] = [
  "solid",
  "dashed",
  "dotted",
  "chain",
  "double",
];

function ringDashPatternForStyle(style: PresenceRingStyle): number[] {
  if (style === "dashed") {
    return [5.2, 3.6];
  }
  if (style === "dotted") {
    return [1.4, 2.8];
  }
  if (style === "chain") {
    return [8.5, 3.1, 2.2, 3.1];
  }
  return [];
}

function resolvePresenceIdentitySignature(rawPresenceId: string): PresenceIdentitySignature {
  const canonical = canonicalPresenceId(rawPresenceId);
  const baseId = canonical || normalizePresenceKey(rawPresenceId) || "presence";
  const sigilIndex = Math.floor(stablePresenceRatio(baseId, 53) * PRESENCE_SIGIL_SEQUENCE.length)
    % PRESENCE_SIGIL_SEQUENCE.length;
  const ringIndex = Math.floor(stablePresenceRatio(baseId, 59) * PRESENCE_RING_STYLE_SEQUENCE.length)
    % PRESENCE_RING_STYLE_SEQUENCE.length;
  const fieldLayerSignature = Math.floor(stablePresenceRatio(baseId, 61) * 8) % 8;
  const notchAngle = stablePresenceRatio(baseId, 67) * Math.PI * 2;
  const quarterTurns = Math.floor(stablePresenceRatio(baseId, 71) * 4) % 4;
  return {
    id: baseId,
    sigil: PRESENCE_SIGIL_SEQUENCE[sigilIndex],
    ringStyle: PRESENCE_RING_STYLE_SEQUENCE[ringIndex],
    fieldLayerSignature,
    notchAngle,
    rotation: quarterTurns * (Math.PI / 2),
  };
}

function drawPresenceSigilCore(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  radius: number,
  signature: PresenceIdentitySignature,
  options?: {
    strokeStyle?: string;
    fillStyle?: string;
    lineWidth?: number;
    includeOuterRing?: boolean;
    alpha?: number;
    compact?: boolean;
  },
): void {
  const sigilRadius = Math.max(1.15, radius);
  const strokeStyle = options?.strokeStyle ?? "rgba(228, 242, 255, 0.92)";
  const fillStyle = options?.fillStyle ?? "rgba(200, 220, 242, 0.14)";
  const lineWidth = Math.max(0.55, options?.lineWidth ?? (sigilRadius * 0.12));
  const includeOuterRing = options?.includeOuterRing ?? true;
  const compact = options?.compact ?? false;
  const alpha = Math.max(0, Math.min(1, options?.alpha ?? 1));
  const ringRadius = sigilRadius * (compact ? 0.88 : 1);

  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.strokeStyle = strokeStyle;
  ctx.fillStyle = fillStyle;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  if (includeOuterRing) {
    if (signature.sigil === "broken-ring") {
      const gap = compact ? 0.72 : 0.56;
      const start = signature.notchAngle + gap;
      const end = signature.notchAngle + (Math.PI * 2) - gap;
      ctx.beginPath();
      ctx.arc(cx, cy, ringRadius, start, end);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(cx, cy, ringRadius, 0, Math.PI * 2);
      ctx.stroke();
    }

    if (signature.sigil === "double-notch") {
      const inner = ringRadius * 0.72;
      ctx.beginPath();
      ctx.arc(cx, cy, inner, 0, Math.PI * 2);
      ctx.stroke();
      const nx = cx + Math.cos(signature.notchAngle) * ringRadius;
      const ny = cy + Math.sin(signature.notchAngle) * ringRadius;
      const tx = cx + Math.cos(signature.notchAngle) * inner;
      const ty = cy + Math.sin(signature.notchAngle) * inner;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(nx, ny);
      ctx.stroke();
    }
  }

  const inner = ringRadius * 0.68;
  if (signature.sigil === "bisected-circle") {
    ctx.beginPath();
    ctx.arc(cx, cy, inner, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, cy - inner);
    ctx.lineTo(cx, cy + inner);
    ctx.stroke();
    ctx.restore();
    return;
  }

  if (signature.sigil === "triangle-dot") {
    const angle = signature.rotation - (Math.PI / 2);
    ctx.beginPath();
    for (let i = 0; i < 3; i += 1) {
      const theta = angle + (i * Math.PI * 2) / 3;
      const px = cx + Math.cos(theta) * inner;
      const py = cy + Math.sin(theta) * inner;
      if (i === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.closePath();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(0.8, inner * 0.22), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return;
  }

  if (signature.sigil === "crossed-ring") {
    const spoke = inner * 0.95;
    ctx.beginPath();
    ctx.arc(cx, cy, inner, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - spoke, cy);
    ctx.lineTo(cx + spoke, cy);
    ctx.moveTo(cx, cy - spoke);
    ctx.lineTo(cx, cy + spoke);
    ctx.moveTo(cx - spoke * 0.72, cy - spoke * 0.72);
    ctx.lineTo(cx + spoke * 0.72, cy + spoke * 0.72);
    ctx.moveTo(cx + spoke * 0.72, cy - spoke * 0.72);
    ctx.lineTo(cx - spoke * 0.72, cy + spoke * 0.72);
    ctx.stroke();
    ctx.restore();
    return;
  }

  if (signature.sigil === "tri-spiral") {
    const armRadius = inner * 0.62;
    for (let arm = 0; arm < 3; arm += 1) {
      const angle = signature.rotation + arm * ((Math.PI * 2) / 3);
      const ox = cx + Math.cos(angle) * inner * 0.14;
      const oy = cy + Math.sin(angle) * inner * 0.14;
      ctx.beginPath();
      ctx.arc(
        ox,
        oy,
        armRadius,
        angle - 0.82,
        angle + 0.56,
      );
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  if (signature.sigil === "broken-ring") {
    const gap = compact ? 0.72 : 0.56;
    const start = signature.notchAngle + gap;
    const end = signature.notchAngle + (Math.PI * 2) - gap;
    ctx.beginPath();
    ctx.arc(cx, cy, inner, start, end);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - inner * 0.75, cy);
    ctx.lineTo(cx + inner * 0.75, cy);
    ctx.stroke();
    ctx.restore();
    return;
  }

  const cross = inner * 0.82;
  ctx.beginPath();
  ctx.moveTo(cx - cross, cy);
  ctx.lineTo(cx + cross, cy);
  ctx.moveTo(cx, cy - cross);
  ctx.lineTo(cx, cy + cross);
  ctx.stroke();
  ctx.restore();
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

function resourceDaimoiHue(resourceType: string): number {
  const key = String(resourceType || "").trim().toLowerCase();
  if (key === "cpu") return 36;
  if (key === "ram") return 182;
  if (key === "disk") return 122;
  if (key === "network") return 214;
  if (key === "gpu") return 284;
  if (key === "npu") return 332;
  return 48;
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

type FileNodeProvenanceKind = "workspace" | "archive" | "web" | "memory" | "synthetic";

function fileNodeProvenanceKind(node: any): FileNodeProvenanceKind {
  const sourceRelPath = String(node?.source_rel_path ?? "").trim();
  if (sourceRelPath) {
    return "workspace";
  }
  const archiveRelPath = String(node?.archived_rel_path ?? node?.archive_rel_path ?? "").trim();
  if (archiveRelPath) {
    return "archive";
  }
  const url = String(node?.url ?? node?.source_url ?? node?.sourceUrl ?? "").trim().toLowerCase();
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return "web";
  }
  const vecCollection = String(node?.vecstore_collection ?? "").trim();
  if (vecCollection) {
    return "memory";
  }
  return "synthetic";
}

function fileNodeProvenanceLabel(kind: FileNodeProvenanceKind): string {
  switch (kind) {
    case "workspace":
      return "workspace";
    case "archive":
      return "archive";
    case "web":
      return "web";
    case "memory":
      return "memory";
    default:
      return "derived";
  }
}

function fileNodeProvenanceHue(kind: FileNodeProvenanceKind): number {
  switch (kind) {
    case "workspace":
      return 154;
    case "archive":
      return 30;
    case "web":
      return 202;
    case "memory":
      return 278;
    default:
      return 56;
  }
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

type NexusVisualClass = "anchor" | "regular" | "relay" | "resource";

function firstFiniteNumber(...candidates: unknown[]): number | null {
  for (const candidate of candidates) {
    const numeric = Number(candidate);
    if (Number.isFinite(numeric)) {
      return numeric;
    }
  }
  return null;
}

function nexusWalletFillRatio(node: any): number {
  const credits = firstFiniteNumber(
    node?.wallet?.credits,
    node?.wallet_credits,
    node?.wallet_credit,
  );
  const spent = firstFiniteNumber(
    node?.wallet?.spent_total,
    node?.wallet_spent,
    node?.wallet?.spent,
  );
  if (credits !== null && spent !== null && (credits + spent) > 0.0001) {
    return clamp01(credits / (credits + spent));
  }
  if (credits !== null) {
    return clamp01(credits / Math.max(1, credits));
  }
  return clamp01(Number(node?.importance ?? 0.35));
}

function nexusSaturationRatio(node: any): number {
  const saturation = firstFiniteNumber(
    node?.node_saturation,
    node?.saturation,
    node?.capacity?.pressure,
  );
  if (saturation !== null) {
    return clamp01(saturation);
  }
  const cap = firstFiniteNumber(
    node?.capacity?.cap,
    node?.resource_cap,
    node?.cap,
  );
  const load = firstFiniteNumber(
    node?.capacity?.load,
    node?.resource_load,
    node?.load,
  );
  if (cap !== null && load !== null && cap > 0.0001) {
    return clamp01(load / cap);
  }
  return clamp01(Number(node?.importance ?? 0.2));
}

function nexusVisualClassForNode(node: any): NexusVisualClass {
  const id = String(node?.id ?? "").trim().toLowerCase();
  const role = String(node?.role ?? node?.kind ?? "").trim().toLowerCase();
  const relayFlag = Boolean(node?.relay) || Boolean(node?.is_relay) || role.includes("relay") || id.includes("relay");
  if (relayFlag) {
    return "relay";
  }
  const anchorFlag = role.includes("anchor") || id.includes("anchor_registry") || id.startsWith("anchor:");
  if (anchorFlag) {
    return "anchor";
  }
  const hasResourceSignals = (
    role.includes("resource")
    || firstFiniteNumber(node?.capacity?.cap, node?.resource_cap, node?.wallet?.credits) !== null
    || String(node?.resource_type ?? "").trim().length > 0
  );
  if (hasResourceSignals) {
    return "resource";
  }
  return "regular";
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

function resourceVisualSpec(resourceKind: GraphNodeResourceKind, fallbackHue: number): GraphNodeVisualSpec {
  switch (resourceKind) {
    case "text":
      return { hue: 50, saturation: 88, value: 98, shape: "square", liftBoost: 1.28, glowBoost: 1.22 };
    case "image":
      return { hue: 318, saturation: 82, value: 98, shape: "circle", liftBoost: 1.36, glowBoost: 1.3 };
    case "audio":
      return { hue: 186, saturation: 84, value: 98, shape: "diamond", liftBoost: 1.34, glowBoost: 1.28 };
    case "archive":
      return { hue: 28, saturation: 90, value: 98, shape: "hexagon", liftBoost: 1.3, glowBoost: 1.22 };
    case "blob":
      return { hue: 214, saturation: 34, value: 86, shape: "triangle", liftBoost: 1.04, glowBoost: 0.98 };
    case "video":
      return { hue: 8, saturation: 90, value: 100, shape: "hexagon", liftBoost: 1.62, glowBoost: 1.58 };
    case "link":
      return { hue: 200, saturation: 72, value: 96, shape: "triangle", liftBoost: 1.06, glowBoost: 1.04 };
    case "website":
      return { hue: 146, saturation: 70, value: 96, shape: "square", liftBoost: 1.1, glowBoost: 1.06 };
    default:
      return {
        hue: fallbackHue,
        saturation: 64,
        value: 92,
        shape: "circle",
        liftBoost: 1,
        glowBoost: 1,
      };
  }
}

function traceResourceShape(
  ctx: CanvasRenderingContext2D,
  shape: GraphNodeShape,
  cx: number,
  cy: number,
  radius: number,
): void {
  if (shape === "circle") {
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    return;
  }
  if (shape === "square") {
    const side = radius * 1.8;
    ctx.roundRect(cx - side / 2, cy - side / 2, side, side, radius * 0.36);
    return;
  }
  if (shape === "diamond") {
    ctx.moveTo(cx, cy - radius);
    ctx.lineTo(cx + radius, cy);
    ctx.lineTo(cx, cy + radius);
    ctx.lineTo(cx - radius, cy);
    ctx.closePath();
    return;
  }
  if (shape === "triangle") {
    ctx.moveTo(cx, cy - radius * 1.1);
    ctx.lineTo(cx + radius, cy + radius * 0.82);
    ctx.lineTo(cx - radius, cy + radius * 0.82);
    ctx.closePath();
    return;
  }
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - (Math.PI / 6);
    const px = cx + Math.cos(angle) * radius;
    const py = cy + Math.sin(angle) * radius;
    if (i === 0) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.closePath();
}

function fillResourceShape(
  ctx: CanvasRenderingContext2D,
  shape: GraphNodeShape,
  cx: number,
  cy: number,
  radius: number,
): void {
  ctx.beginPath();
  traceResourceShape(ctx, shape, cx, cy, radius);
  ctx.fill();
}

function strokeResourceShape(
  ctx: CanvasRenderingContext2D,
  shape: GraphNodeShape,
  cx: number,
  cy: number,
  radius: number,
): void {
  ctx.beginPath();
  traceResourceShape(ctx, shape, cx, cy, radius);
  ctx.stroke();
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function thinFileGraphEdgesForRendering(
  edges: any[],
  nodeById: Map<string, any>,
  fileNodeCount: number,
  selectedNodeId: string,
): FileGraphRenderEdge[] {
  const normalized: FileGraphRenderEdge[] = [];
  for (let index = 0; index < edges.length; index++) {
    const raw = edges[index] as any;
    const source = String(raw?.source ?? "").trim();
    const target = String(raw?.target ?? "").trim();
    if (!source || !target || source === target) {
      continue;
    }
    if (!nodeById.has(source) || !nodeById.has(target)) {
      continue;
    }
    normalized.push({
      id: String(raw?.id ?? `edge:${source}:${target}:${index}`),
      source,
      target,
      field: String(raw?.field ?? "").trim(),
      kind: String(raw?.kind ?? "relates").trim().toLowerCase() || "relates",
      weight: clamp01(Number(raw?.weight ?? 0.2)),
    });
  }

  if (normalized.length <= 340) {
    return normalized;
  }

  const targetDegree = new Map<string, number>();
  for (let index = 0; index < normalized.length; index++) {
    const target = normalized[index].target;
    targetDegree.set(target, (targetDegree.get(target) ?? 0) + 1);
  }

  const globalCap = Math.max(220, Math.min(860, Math.round(fileNodeCount * 1.55)));
  const selectedReserve = selectedNodeId
    ? Math.max(26, Math.min(120, Math.round(globalCap * 0.24)))
    : 0;
  const perSourceCap = 4;
  const perSourceCategorizeCap = 1;
  const perFieldHubCap = Math.max(26, Math.min(92, Math.round(fileNodeCount * 0.13)));
  const perConceptHubCap = Math.max(24, Math.min(120, Math.round(fileNodeCount * 0.16)));
  const tagMemberCap = Math.max(46, Math.min(240, Math.round(fileNodeCount * 0.5)));
  const tagPairCap = Math.max(18, Math.min(110, Math.round(fileNodeCount * 0.24)));

  const kindPriority = (kind: string): number => {
    if (kind === "spawns_presence") {
      return 1.22;
    }
    if (kind === "organized_by_presence") {
      return 1.06;
    }
    if (kind === "categorizes") {
      return 0.94;
    }
    if (kind === "labeled_as") {
      return 0.66;
    }
    if (kind === "relates_tag") {
      return 0.44;
    }
    return 0.58;
  };

  type ScoredEdge = {
    row: FileGraphRenderEdge;
    score: number;
    selected: boolean;
  };

  const scored: ScoredEdge[] = normalized.map((row) => {
    const sourceNode = nodeById.get(row.source);
    const targetNode = nodeById.get(row.target);
    const sourceImportance = clamp01(Number(sourceNode?.importance ?? 0.2));
    const targetImportance = clamp01(
      Number(
        targetNode?.importance
          ?? (String(targetNode?.presence_kind ?? "") === "concept" ? 0.72 : 0.22),
      ),
    );
    const isSelected =
      selectedNodeId.length > 0
      && (row.source === selectedNodeId || row.target === selectedNodeId);
    const selectedBoost = isSelected ? 2.9 : 0;
    const hubPenalty = row.kind === "categorizes"
      ? 1 / (1 + (Math.max(0, (targetDegree.get(row.target) ?? 1) - 1) * 0.018))
      : 1;
    const score =
      ((row.weight * 1.46)
      + kindPriority(row.kind)
      + (sourceImportance * 0.58)
      + (targetImportance * 0.28)
      + selectedBoost)
      * hubPenalty;
    return {
      row,
      score,
      selected: isSelected,
    };
  });

  scored.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    if (left.row.kind !== right.row.kind) {
      return left.row.kind.localeCompare(right.row.kind);
    }
    return left.row.id.localeCompare(right.row.id);
  });

  const picked: FileGraphRenderEdge[] = [];
  const seen = new Set<string>();
  const sourceCount = new Map<string, number>();
  const sourceCategorizeCount = new Map<string, number>();
  const fieldTargetCount = new Map<string, number>();
  const conceptTargetCount = new Map<string, number>();
  let pickedTagMemberEdges = 0;
  let pickedTagPairEdges = 0;

  const tryPick = (entry: ScoredEdge, forceSelected: boolean): boolean => {
    const row = entry.row;
    const key = `${row.source}|${row.target}|${row.kind}`;
    if (seen.has(key)) {
      return false;
    }

    if (!forceSelected && picked.length >= globalCap) {
      return false;
    }

    const sourceLimit = forceSelected ? perSourceCap + 4 : perSourceCap;
    const sourceHitCount = sourceCount.get(row.source) ?? 0;
    if (sourceHitCount >= sourceLimit) {
      return false;
    }

    if (row.kind === "categorizes") {
      const categorizeLimit = forceSelected
        ? perSourceCategorizeCap + 1
        : perSourceCategorizeCap;
      const categorizeHits = sourceCategorizeCount.get(row.source) ?? 0;
      if (categorizeHits >= categorizeLimit) {
        return false;
      }
      if (row.target.startsWith("field:")) {
        const fieldLimit = forceSelected ? perFieldHubCap + 18 : perFieldHubCap;
        const targetHits = fieldTargetCount.get(row.target) ?? 0;
        if (targetHits >= fieldLimit) {
          return false;
        }
      }
    }

    if (row.kind === "organized_by_presence" && row.target.startsWith("presence:concept:")) {
      const conceptLimit = forceSelected ? perConceptHubCap + 24 : perConceptHubCap;
      const conceptHits = conceptTargetCount.get(row.target) ?? 0;
      if (conceptHits >= conceptLimit) {
        return false;
      }
    }

    if (row.kind === "labeled_as") {
      const memberLimit = forceSelected ? tagMemberCap + 24 : tagMemberCap;
      if (pickedTagMemberEdges >= memberLimit) {
        return false;
      }
    }

    if (row.kind === "relates_tag") {
      const pairLimit = forceSelected ? tagPairCap + 12 : tagPairCap;
      if (pickedTagPairEdges >= pairLimit) {
        return false;
      }
    }

    seen.add(key);
    sourceCount.set(row.source, sourceHitCount + 1);
    if (row.kind === "categorizes") {
      sourceCategorizeCount.set(
        row.source,
        (sourceCategorizeCount.get(row.source) ?? 0) + 1,
      );
      if (row.target.startsWith("field:")) {
        fieldTargetCount.set(row.target, (fieldTargetCount.get(row.target) ?? 0) + 1);
      }
    }
    if (row.kind === "organized_by_presence" && row.target.startsWith("presence:concept:")) {
      conceptTargetCount.set(row.target, (conceptTargetCount.get(row.target) ?? 0) + 1);
    }
    if (row.kind === "labeled_as") {
      pickedTagMemberEdges += 1;
    }
    if (row.kind === "relates_tag") {
      pickedTagPairEdges += 1;
    }
    picked.push(row);
    return true;
  };

  if (selectedNodeId) {
    let reserved = 0;
    for (let index = 0; index < scored.length; index++) {
      const row = scored[index];
      if (!row.selected) {
        continue;
      }
      if (reserved >= selectedReserve) {
        break;
      }
      if (tryPick(row, true)) {
        reserved += 1;
      }
    }
  }

  for (let index = 0; index < scored.length; index++) {
    if (picked.length >= globalCap) {
      break;
    }
    tryPick(scored[index], false);
  }

  if (picked.length <= 0) {
    return normalized.slice(0, globalCap);
  }
  return picked;
}

function clampValue(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function layerHueByIndex(index: number): number {
  return (196 + index * 37) % 360;
}

function ratioFromMetric(value: number | undefined, fallback = 0.5): number {
  if (value === undefined || Number.isNaN(value)) {
    return fallback;
  }
  if (value > 1) {
    return clamp01(value / 100);
  }
  return clamp01(value);
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

interface OverlayGhostTrailPoint {
  xNorm: number;
  yNorm: number;
  atSec: number;
}

interface OverlayGhostTrailState {
  points: OverlayGhostTrailPoint[];
  seenAtSec: number;
  lastSampleAtSec: number;
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
  const onUserPresenceInputRef = useRef(onUserPresenceInput);
  const userPresenceMouseEmitMsRef = useRef(0);
  useEffect(() => {
    onUserPresenceInputRef.current = onUserPresenceInput;
  }, [onUserPresenceInput]);
  const [worldscreen, setWorldscreen] = useState<GraphWorldscreenState | null>(null);
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

  const renderParticleField = false;
  const renderRichOverlayParticles = true;

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
    if (!renderParticleField) {
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
        float alpha = clamp((edge * (0.74 + uBloom * 0.26)) + (core * (0.56 + uBloom * 0.24)), 0.0, 1.0);
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

    const draw = (ts: number) => {
      lastTick = ts;
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
      cameraMouseX += (mouseX - cameraMouseX) * 0.08;
      cameraMouseY += (mouseY - cameraMouseY) * 0.08;
      scenePulse += (scenePulseTarget - scenePulse) * 0.06;

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

      if (count > 0) {
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

        gl.drawArrays(gl.POINTS, 0, count);
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
      const points = hasOverlayParticles ? [] : (state.points || []);
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
    const canvas = overlayRef.current;
    if(!canvas) return;
    const ctx = canvas.getContext("2d");
    if(!ctx) return;

    let rafId = 0;
    let lastPaintTs = 0;
    let canvasWidth = 0;
    let canvasHeight = 0;
    const HEX_SIZE = 24;
    const fallbackNamedForms = [
        { id: "receipt_river", en: "Receipt River", ja: "領収書の川", hue: 212, x: 0.22, y: 0.38, freq: 196, type: "flow" },
        { id: "witness_thread", en: "Witness Thread", ja: "証人の糸", hue: 262, x: 0.63, y: 0.33, freq: 233, type: "network" },
        { id: "fork_tax_canticle", en: "Fork Tax Canticle", ja: "フォーク税の聖歌", hue: 34, x: 0.44, y: 0.62, freq: 277, type: "glitch" },
        { id: "mage_of_receipts", en: "Mage of Receipts", ja: "領収魔導師", hue: 286, x: 0.33, y: 0.71, freq: 311, type: "flow" },
        { id: "keeper_of_receipts", en: "Keeper of Receipts", ja: "領収書の番人", hue: 124, x: 0.57, y: 0.72, freq: 349, type: "geo" },
        { id: "anchor_registry", en: "Anchor Registry", ja: "錨台帳", hue: 184, x: 0.49, y: 0.5, freq: 392, type: "geo" },
        { id: "gates_of_truth", en: "Gates of Truth", ja: "真理の門", hue: 52, x: 0.76, y: 0.54, freq: 440, type: "portal" },
    ];

    let ripple = { x: 0.5, y: 0.5, power: 0, at: 0 };
    let pointerField = { x: 0.5, y: 0.5, power: 0, inside: false };
    let highlighted = -1;
    let selectedGraphNodeId = "";
    let selectedGraphNodeLabel = "";
    let graphNodeHits: Array<{
      node: any;
      x: number;
      y: number;
      radiusNorm: number;
      nodeKind: "file" | "crawler";
      resourceKind: GraphNodeResourceKind;
    }> = [];
    let particleTelemetryHits: Array<{
      x: number;
      y: number;
      radiusNorm: number;
      graphNodeId: string;
      routeNodeId: string;
    }> = [];
    const overlayMotionByParticleId = new Map<string, {
      xNorm: number;
      yNorm: number;
      vx: number;
      vy: number;
      seenAtSec: number;
    }>();
    const overlayGhostTrailByParticleId = new Map<string, OverlayGhostTrailState>();
    const ghostTrailMaxPoints = 18;
    const ghostTrailMaxAgeSec = 2.8;
    const ghostTrailSampleIntervalSec = 1 / 24;
    const ghostTrailMinStepNorm = 0.0018;
    const ghostTrailMotionFloor = 0.00045;
    const ghostTrailStaleAfterSec = 4.2;
    let overlayMotionLastFrameTs = 0;
    let overlayMotionDtSec = 1 / 60;
    let overlayMotionNowSec = 0;

    const resolveParticleRenderMotion = (
      row: BackendFieldParticle,
      fallbackId: string,
      isNexusParticle: boolean,
    ) => {
      const baseX = clamp01(Number(row?.x ?? 0.5));
      const baseY = clamp01(Number(row?.y ?? 0.5));
      const rawVx = Number((row as any)?.vx ?? 0);
      const rawVy = Number((row as any)?.vy ?? 0);
      const particleId = String(row?.id ?? "").trim() || fallbackId;
      const anchorBlend = isNexusParticle ? 0.2 : 0.1;
      const velocityBlend = 0.55;
      const velocityGain = isNexusParticle
        ? (6 + motionSpeedRef.current * 8)
        : (22 + motionSpeedRef.current * 20);

      let state = overlayMotionByParticleId.get(particleId);
      if (!state) {
        state = {
          xNorm: baseX,
          yNorm: baseY,
          vx: rawVx,
          vy: rawVy,
          seenAtSec: overlayMotionNowSec,
        };
        overlayMotionByParticleId.set(particleId, state);
      } else {
        state.xNorm += (baseX - state.xNorm) * anchorBlend;
        state.yNorm += (baseY - state.yNorm) * anchorBlend;
        state.vx += (rawVx - state.vx) * velocityBlend;
        state.vy += (rawVy - state.vy) * velocityBlend;
        state.xNorm = clamp01(state.xNorm + (state.vx * overlayMotionDtSec * velocityGain));
        state.yNorm = clamp01(state.yNorm + (state.vy * overlayMotionDtSec * velocityGain));

        if (state.xNorm <= 0.001 || state.xNorm >= 0.999) {
          state.vx *= -0.62;
        }
        if (state.yNorm <= 0.001 || state.yNorm >= 0.999) {
          state.vy *= -0.62;
        }
      }

      state.seenAtSec = overlayMotionNowSec;
      return {
        xNorm: state.xNorm,
        yNorm: state.yNorm,
        vx: state.vx,
        vy: state.vy,
        speed: Math.hypot(state.vx, state.vy),
      };
    };

    const drawDaimoiGhostTrail = (
      particleId: string,
      xNorm: number,
      yNorm: number,
      speed: number,
      colorBase: string,
      isTransferParticle: boolean,
      w: number,
      h: number,
    ) => {
      const key = particleId.trim();
      if (!key) {
        return;
      }

      let trail = overlayGhostTrailByParticleId.get(key);
      if (!trail) {
        trail = {
          points: [{ xNorm, yNorm, atSec: overlayMotionNowSec }],
          seenAtSec: overlayMotionNowSec,
          lastSampleAtSec: overlayMotionNowSec,
        };
        overlayGhostTrailByParticleId.set(key, trail);
      }
      trail.seenAtSec = overlayMotionNowSec;

      const points = trail.points;
      while (points.length > 0 && (overlayMotionNowSec - points[0].atSec) > ghostTrailMaxAgeSec) {
        points.shift();
      }
      if (points.length <= 0) {
        points.push({ xNorm, yNorm, atSec: overlayMotionNowSec });
        trail.lastSampleAtSec = overlayMotionNowSec;
      }

      if (speed >= ghostTrailMotionFloor) {
        const latest = points[points.length - 1];
        const moved = Math.hypot(xNorm - latest.xNorm, yNorm - latest.yNorm);
        const elapsed = overlayMotionNowSec - trail.lastSampleAtSec;
        if (moved >= ghostTrailMinStepNorm || elapsed >= ghostTrailSampleIntervalSec) {
          points.push({ xNorm, yNorm, atSec: overlayMotionNowSec });
          trail.lastSampleAtSec = overlayMotionNowSec;
        } else {
          latest.xNorm = xNorm;
          latest.yNorm = yNorm;
        }
      }

      if (points.length > ghostTrailMaxPoints) {
        points.splice(0, points.length - ghostTrailMaxPoints);
      }
      if (points.length < 2) {
        return;
      }

      ctx.save();
      ctx.globalCompositeOperation = "screen";
      ctx.lineCap = "round";
      for (let index = 1; index < points.length; index += 1) {
        const start = points[index - 1];
        const end = points[index];
        const ageSec = overlayMotionNowSec - end.atSec;
        if (ageSec > ghostTrailMaxAgeSec) {
          continue;
        }
        const ageFade = clamp01(1 - (ageSec / ghostTrailMaxAgeSec));
        const progress = index / (points.length - 1);
        const alpha = (isTransferParticle ? 0.22 : 0.18) * ageFade * (0.28 + progress * 0.88);
        if (alpha <= 0.01) {
          continue;
        }
        ctx.strokeStyle = `rgba(${colorBase}, ${alpha})`;
        ctx.lineWidth = (isTransferParticle ? 1.04 : 0.84) * (0.42 + progress * 0.86);
        ctx.beginPath();
        ctx.moveTo(start.xNorm * w, start.yNorm * h);
        ctx.lineTo(end.xNorm * w, end.yNorm * h);
        ctx.stroke();
      }
      ctx.restore();
    };

    const resolveNamedForms = (): Array<any> => {
        const fromCatalog = Array.isArray(catalogRef.current?.entity_manifest)
            ? (catalogRef.current?.entity_manifest as Array<any>)
            : [];
        const baseManifest = fromCatalog.length > 0 ? fromCatalog : fallbackNamedForms;
        const seen = new Set<string>();
        const normalized = baseManifest
            .map((raw: any) => {
                const id = canonicalPresenceId(String(raw?.id ?? ""));
                if (!id || seen.has(id)) {
                    return null;
                }
                seen.add(id);
                return {
                    ...raw,
                    id,
                    en: String(raw?.en ?? raw?.label ?? shortPresenceIdLabel(id)).trim() || shortPresenceIdLabel(id),
                    ja: String(raw?.ja ?? raw?.label_ja ?? "presence").trim() || "presence",
                    x: clamp01(Number(raw?.x ?? 0.5)),
                    y: clamp01(Number(raw?.y ?? 0.5)),
                    hue: Number.isFinite(Number(raw?.hue))
                        ? Number(raw?.hue)
                        : presenceHueFromId(id),
                };
            })
            .filter((row): row is any => row !== null);

        if (normalized.length <= 0) {
            return spreadPresenceAnchors(fallbackNamedForms);
        }
        return spreadPresenceAnchors(normalized);
    };

    const resolveFileGraph = (state: SimulationState | null): FileGraph | null => {
        const fromSimulation = state?.file_graph;
        const fromCatalog = catalogRef.current?.file_graph;
        const baseGraph = (
            fromSimulation && (Array.isArray(fromSimulation.file_nodes) || Array.isArray((fromSimulation as any).nodes))
                ? fromSimulation
                : (
                    fromCatalog && (Array.isArray(fromCatalog.file_nodes) || Array.isArray((fromCatalog as any).nodes))
                        ? fromCatalog
                        : null
                )
        );
        if (!baseGraph) {
            return null;
        }

        const crawlerGraph = resolveCrawlerGraph(state);
        const crawlerRows = Array.isArray((baseGraph as any)?.crawler_nodes)
            ? ((baseGraph as any).crawler_nodes as any[])
            : [];
        const externalCrawlerRows = Array.isArray(crawlerGraph?.crawler_nodes)
            ? (crawlerGraph?.crawler_nodes as any[])
            : [];
        const crawlerEdges = Array.isArray(crawlerGraph?.edges)
            ? (crawlerGraph?.edges as any[])
            : [];
        if (crawlerRows.length <= 0 && externalCrawlerRows.length <= 0 && crawlerEdges.length <= 0) {
            return baseGraph;
        }

        const fieldNodes = Array.isArray(baseGraph.field_nodes)
            ? baseGraph.field_nodes.filter((row) => row && typeof row === "object")
            : [];
        const tagNodes = Array.isArray((baseGraph as any).tag_nodes)
            ? ((baseGraph as any).tag_nodes as any[]).filter((row) => row && typeof row === "object")
            : [];
        const fileNodes = Array.isArray(baseGraph.file_nodes)
            ? baseGraph.file_nodes.filter((row) => row && typeof row === "object")
            : [];
        const nodesRaw = Array.isArray(baseGraph.nodes) && baseGraph.nodes.length > 0
            ? baseGraph.nodes
            : [...fieldNodes, ...tagNodes, ...fileNodes];
        const nodes = nodesRaw
            .filter((row) => row && typeof row === "object")
            .map((row: any) => ({ ...row }));
        const edges = Array.isArray(baseGraph.edges)
            ? baseGraph.edges.filter((row) => row && typeof row === "object").map((row: any) => ({ ...row }))
            : [];

        const nodeIdSet = new Set<string>();
        nodes.forEach((row: any) => {
            const id = String(row?.id ?? "").trim();
            if (id) {
                nodeIdSet.add(id);
            }
        });

        const fieldAliasByCrawlerTarget = new Map<string, string>();
        fieldNodes.forEach((row: any) => {
            const fieldId = String(row?.id ?? "").trim();
            const nodeId = String(row?.node_id ?? "").trim();
            const sourceTokens = [fieldId, nodeId];
            sourceTokens.forEach((token) => {
                if (!token) {
                    return;
                }
                const presenceId = token.startsWith("field:")
                    ? token.slice("field:".length).trim()
                    : token;
                if (!presenceId) {
                    return;
                }
                fieldAliasByCrawlerTarget.set(`crawler-field:${presenceId}`, fieldId || `field:${presenceId}`);
            });
        });

        const mergedCrawlerNodes: any[] = [];
        const crawlerSeen = new Set<string>();
        const addCrawlerNode = (raw: any) => {
            if (!raw || typeof raw !== "object") {
                return;
            }
            const id = String(raw?.id ?? "").trim();
            if (!id || crawlerSeen.has(id)) {
                return;
            }
            crawlerSeen.add(id);
            const normalized = {
                ...raw,
                id,
                node_type: "crawler",
                crawler_kind: String(raw?.crawler_kind ?? raw?.kind ?? "url").trim() || "url",
                kind: String(raw?.kind ?? raw?.crawler_kind ?? "url").trim() || "url",
                x: clamp01(Number(raw?.x ?? 0.5)),
                y: clamp01(Number(raw?.y ?? 0.5)),
            };
            mergedCrawlerNodes.push(normalized);
            if (!nodeIdSet.has(id)) {
                nodes.push(normalized);
                nodeIdSet.add(id);
            }
        };
        crawlerRows.forEach(addCrawlerNode);
        externalCrawlerRows.forEach(addCrawlerNode);

        const seenEdgeKeys = new Set<string>();
        edges.forEach((row: any) => {
            const source = String(row?.source ?? "").trim();
            const target = String(row?.target ?? "").trim();
            const kind = String(row?.kind ?? "").trim().toLowerCase();
            if (source && target) {
                seenEdgeKeys.add(`${source}|${target}|${kind}`);
            }
        });

        crawlerEdges.forEach((row: any, index) => {
            if (!row || typeof row !== "object") {
                return;
            }
            let source = String(row?.source ?? "").trim();
            let target = String(row?.target ?? "").trim();
            if (!source || !target) {
                return;
            }
            source = fieldAliasByCrawlerTarget.get(source) ?? source;
            target = fieldAliasByCrawlerTarget.get(target) ?? target;
            if (source.startsWith("crawler-field:")) {
                source = source.replace("crawler-field:", "field:");
            }
            if (target.startsWith("crawler-field:")) {
                target = target.replace("crawler-field:", "field:");
            }
            const kind = String(row?.kind ?? "hyperlink").trim().toLowerCase() || "hyperlink";
            if (!nodeIdSet.has(source) || !nodeIdSet.has(target) || source === target) {
                return;
            }
            const edgeKey = `${source}|${target}|${kind}`;
            if (seenEdgeKeys.has(edgeKey)) {
                return;
            }
            seenEdgeKeys.add(edgeKey);
            edges.push({
                ...row,
                id: String(row?.id ?? "").trim() || `nexus-crawler-edge:${index}`,
                source,
                target,
                kind,
                weight: clamp01(Number(row?.weight ?? 0.28)),
            });
        });

        const mergedStats = {
            ...(baseGraph.stats ?? {}),
            crawler_nexus_count: mergedCrawlerNodes.length,
            nexus_node_count: nodes.length,
            nexus_edge_count: edges.length,
        };
        return {
            ...baseGraph,
            nodes,
            edges,
            crawler_nodes: mergedCrawlerNodes,
            stats: mergedStats,
        };
    };

    const resolveCrawlerGraph = (state: SimulationState | null): CrawlerGraph | null => {
        const fromSimulation = state?.crawler_graph;
        if (fromSimulation && (Array.isArray(fromSimulation.crawler_nodes) || Array.isArray((fromSimulation as any).nodes))) {
            return fromSimulation;
        }
        const fromCatalog = catalogRef.current?.crawler_graph;
        if (fromCatalog && (Array.isArray(fromCatalog.crawler_nodes) || Array.isArray((fromCatalog as any).nodes))) {
            return fromCatalog;
        }
        return null;
    };

    const resolveTruthState = (state: SimulationState | null): TruthState | null => {
        const fromSimulation = state?.truth_state;
        if (fromSimulation && fromSimulation.record) {
            return fromSimulation;
        }
        const fromCatalog = catalogRef.current?.truth_state;
        if (fromCatalog && fromCatalog.record) {
            return fromCatalog;
        }
        return null;
    };

    const resolveLogicalGraph = (state: SimulationState | null) => {
        const fromSimulation = state?.logical_graph;
        if (fromSimulation && Array.isArray(fromSimulation.nodes)) {
            return fromSimulation;
        }
        const fromCatalog = catalogRef.current?.logical_graph;
        if (fromCatalog && Array.isArray(fromCatalog.nodes)) {
            return fromCatalog;
        }
        return null;
    };

    const resolvePainField = (state: SimulationState | null) => {
        const fromSimulation = state?.pain_field;
        if (fromSimulation && Array.isArray(fromSimulation.node_heat)) {
            return fromSimulation;
        }
        const fromCatalog = catalogRef.current?.pain_field;
        if (fromCatalog && Array.isArray(fromCatalog.node_heat)) {
            return fromCatalog;
        }
        return null;
    };

    const documentRangeFromImportance = (importance: number): number => {
        return 0.016 + clamp01(importance) * 0.064;
    };

    const nearestGraphNodeAt = (xRatio: number, yRatio: number) => {
        let match: { hit: (typeof graphNodeHits)[number]; distance: number } | null = null;
        for (const hit of graphNodeHits) {
            const dx = xRatio - hit.x;
            const dy = yRatio - hit.y;
            const distance = Math.hypot(dx, dy);
            if (distance > Math.max(0.012, hit.radiusNorm * 1.8)) {
                continue;
            }
            if (!match || distance < match.distance) {
                match = { hit, distance };
            }
        }
        return match?.hit ?? null;
    };

    const nearestParticleTelemetryAt = (xRatio: number, yRatio: number) => {
        let match: { hit: (typeof particleTelemetryHits)[number]; distance: number } | null = null;
        for (const hit of particleTelemetryHits) {
            const dx = xRatio - hit.x;
            const dy = yRatio - hit.y;
            const distance = Math.hypot(dx, dy);
            if (distance > Math.max(0.009, hit.radiusNorm * 1.35)) {
                continue;
            }
            if (!match || distance < match.distance) {
                match = { hit, distance };
            }
        }
        return match?.hit ?? null;
    };

    const resolveGraphNodeById = (
        state: SimulationState | null,
        nodeId: string,
    ): { node: any; nodeKind: "file" | "crawler" } | null => {
        const id = String(nodeId ?? "").trim();
        if (!id) {
            return null;
        }

        const fileGraph = resolveFileGraph(state);
        if (fileGraph) {
            const fileNode = [
                ...(Array.isArray(fileGraph.nodes) ? fileGraph.nodes : []),
                ...(Array.isArray(fileGraph.file_nodes) ? fileGraph.file_nodes : []),
                ...(Array.isArray(fileGraph.tag_nodes) ? fileGraph.tag_nodes : []),
            ].find((row: any) => String(row?.id ?? "").trim() === id);
            if (fileNode) {
                const nodeType = String((fileNode as any)?.node_type ?? "").trim().toLowerCase();
                return {
                    node: fileNode,
                    nodeKind: nodeType === "crawler" ? "crawler" : "file",
                };
            }
        }

        const crawlerGraph = resolveCrawlerGraph(state);
        if (crawlerGraph) {
            const crawlerNode = [
                ...(Array.isArray(crawlerGraph.nodes) ? crawlerGraph.nodes : []),
                ...(Array.isArray(crawlerGraph.crawler_nodes) ? crawlerGraph.crawler_nodes : []),
            ].find((row: any) => String(row?.id ?? "").trim() === id);
            if (crawlerNode) {
                return { node: crawlerNode, nodeKind: "crawler" };
            }
        }
        return null;
    };

    const nearestPresenceAt = (xRatio: number, yRatio: number, namedForms: Array<any>) => {
        let closestIndex = -1;
        let closestDistance = Number.POSITIVE_INFINITY;
        for (let i = 0; i < namedForms.length; i++) {
            const field = namedForms[i];
            const dx = Number(field?.x ?? 0) - xRatio;
            const dy = Number(field?.y ?? 0) - yRatio;
            const distance = Math.hypot(dx, dy);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestIndex = i;
            }
        }
        return { index: closestIndex, distance: closestDistance };
    };

    const drawNebula = (
        t: number,
        field: any,
        cx: number,
        cy: number,
        radius: number,
        hue: number,
        intensity: number,
        isHighlighted: boolean,
    ) => {
        const touchAge = Math.max(0, (performance.now() - ripple.at) / 1000);
        const touchDecay = Math.max(0, 1 - touchAge * 0.85);
        const localRadius = radius * (0.6 + (isHighlighted ? 0.4 : 0) + (touchDecay * ripple.power * 0.5) + (intensity * 0.2));
        const driftX = Math.sin(t * 0.33 + field.x * 0.4) * localRadius * 0.1;
        const driftY = Math.cos(t * 0.29 + field.y * 0.35) * localRadius * 0.1;
        const fog = ctx.createRadialGradient(cx+driftX, cy+driftY, localRadius*0.05, cx+driftX, cy+driftY, localRadius*0.8);
        const baseAlpha = isHighlighted ? 0.72 : 0.36;
        fog.addColorStop(0, "hsla(" + hue + ", 70%, 65%, " + (baseAlpha + intensity * 0.2) + ")");
        fog.addColorStop(0.5, "hsla(" + ((hue + 30) % 360) + ", 88%, 62%, " + ((baseAlpha * 0.66) + intensity * 0.16) + ")");
        fog.addColorStop(1, "hsla(" + ((hue + 30) % 360) + ", 80%, 55%, 0)");
        ctx.fillStyle = fog;
        ctx.beginPath();
        ctx.ellipse(cx+driftX, cy+driftY, localRadius, localRadius * 0.8, t * 0.1, 0, Math.PI * 2);
        ctx.fill();
    };

    interface DaimonPacketVisual {
        particleId: string;
        x: number;
        y: number;
        vx: number;
        vy: number;
        speed: number;
        particleRadius: number;
        intentHue: number;
        intentEnergy: number;
        mantleInfluence: number;
        instability: number;
        intentUnits: number;
        actionBlocked: boolean;
        ownerSignature: PresenceIdentitySignature;
        coreShape: "circle" | "diamond" | "square";
    }

    const resolveIntentHue = (row: BackendFieldParticle, fallbackHue: number): number => {
        const focus = String((row as any)?.route_resource_focus ?? "").trim().toLowerCase();
        if (focus) {
            return resourceDaimoiHue(focus);
        }
        const resourceType = String((row as any)?.resource_type ?? "").trim().toLowerCase();
        if (resourceType) {
            return resourceDaimoiHue(resourceType);
        }
        const consumeType = String((row as any)?.resource_consume_type ?? "").trim().toLowerCase();
        if (consumeType) {
            return resourceDaimoiHue(consumeType);
        }
        const topJob = String((row as any)?.top_job ?? "").trim().toLowerCase();
        if (topJob.includes("truth") || topJob.includes("proof")) {
            return 56;
        }
        if (topJob.includes("anchor")) {
            return 188;
        }
        if (topJob.includes("receipt") || topJob.includes("witness")) {
            return 206;
        }
        return fallbackHue;
    };

    const drawPacketCoreShape = (
        shape: "circle" | "diamond" | "square",
        x: number,
        y: number,
        radius: number,
    ) => {
        if (shape === "circle") {
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            return;
        }
        if (shape === "square") {
            const side = radius * 1.6;
            ctx.beginPath();
            ctx.roundRect(x - side / 2, y - side / 2, side, side, Math.max(0.3, radius * 0.34));
            return;
        }
        ctx.beginPath();
        ctx.moveTo(x, y - radius);
        ctx.lineTo(x + radius, y);
        ctx.lineTo(x, y + radius);
        ctx.lineTo(x - radius, y);
        ctx.closePath();
    };

    const drawNexusPacketSymbol = (
        x: number,
        y: number,
        radius: number,
        colorBase: string,
        relayMode: boolean,
    ) => {
        ctx.strokeStyle = `rgba(${colorBase}, ${relayMode ? 0.36 : 0.58})`;
        ctx.lineWidth = relayMode ? 0.82 : 1.05;
        if (relayMode) {
            ctx.setLineDash([2.2, 3.1]);
            ctx.lineDashOffset = -(overlayMotionNowSec * 28);
        }
        ctx.beginPath();
        ctx.arc(x, y, radius * 1.42, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = `rgba(${colorBase}, 0.86)`;
        drawPacketCoreShape("diamond", x, y, radius);
        ctx.fill();

        if (relayMode) {
            const trail = radius * 1.6;
            ctx.strokeStyle = `rgba(${colorBase}, 0.34)`;
            ctx.lineWidth = 0.72;
            ctx.beginPath();
            ctx.moveTo(x - trail, y + trail * 0.2);
            ctx.lineTo(x - trail * 0.28, y + trail * 0.62);
            ctx.stroke();
        }
    };

    const drawDaimonPacketSymbol = (
        packet: DaimonPacketVisual,
        w: number,
        h: number,
        t: number,
    ) => {
        const fallbackAngle = packet.ownerSignature.notchAngle + packet.ownerSignature.rotation;
        const directionAngle = packet.speed > 0.0001
            ? Math.atan2(packet.vy, packet.vx)
            : fallbackAngle;
        const dirX = Math.cos(directionAngle);
        const dirY = Math.sin(directionAngle);
        const normalX = -dirY;
        const normalY = dirX;

        const tailLength = clampValue(
            2.4 + packet.intentUnits * 0.95 + packet.speed * Math.max(w, h) * 0.42,
            2.8,
            28,
        );
        const curvature = packet.instability
            * (2.4 + packet.intentUnits * 0.35)
            * (stablePresenceRatio(packet.particleId, 79) > 0.5 ? 1 : -1);
        const tailStartX = packet.x - dirX * (packet.particleRadius * 0.34);
        const tailStartY = packet.y - dirY * (packet.particleRadius * 0.34);
        const tailEndX = packet.x - dirX * tailLength;
        const tailEndY = packet.y - dirY * tailLength;
        const controlX = (tailStartX + tailEndX) * 0.5 + normalX * curvature;
        const controlY = (tailStartY + tailEndY) * 0.5 + normalY * curvature;
        const tailAlpha = 0.18 + packet.intentEnergy * 0.48;
        const tailWidth = 0.62 + packet.intentEnergy * 1.14 + packet.mantleInfluence * 0.9;

        ctx.strokeStyle = `hsla(${packet.intentHue}, 88%, 74%, ${tailAlpha})`;
        ctx.lineWidth = tailWidth;
        ctx.beginPath();
        ctx.moveTo(tailStartX, tailStartY);
        ctx.quadraticCurveTo(controlX, controlY, tailEndX, tailEndY);
        ctx.stroke();

        for (let strand = -1; strand <= 1; strand += 1) {
            const strandOffset = strand * (0.4 + packet.mantleInfluence * 0.9);
            const sx = tailStartX + normalX * strandOffset;
            const sy = tailStartY + normalY * strandOffset;
            const ex = tailEndX + normalX * strandOffset;
            const ey = tailEndY + normalY * strandOffset;
            const cx = controlX + normalX * strandOffset * 0.4 + Math.sin(t * 2.1 + strand) * 0.6;
            const cy = controlY + normalY * strandOffset * 0.4 + Math.cos(t * 1.8 + strand) * 0.6;
            ctx.strokeStyle = `hsla(${packet.intentHue}, 84%, 80%, ${0.08 + packet.intentEnergy * 0.24})`;
            ctx.lineWidth = Math.max(0.32, tailWidth * 0.38);
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.quadraticCurveTo(cx, cy, ex, ey);
            ctx.stroke();
        }

        const haloRadius = packet.particleRadius * (1.32 + packet.mantleInfluence * 0.96);
        ctx.strokeStyle = `hsla(${packet.intentHue}, 92%, 76%, ${0.22 + packet.intentEnergy * 0.46})`;
        ctx.lineWidth = 0.44 + packet.mantleInfluence * 1.48;
        ctx.beginPath();
        ctx.arc(packet.x, packet.y, haloRadius, 0, Math.PI * 2);
        ctx.stroke();

        const glow = ctx.createRadialGradient(
            packet.x,
            packet.y,
            Math.max(0.45, packet.particleRadius * 0.25),
            packet.x,
            packet.y,
            packet.particleRadius * 2.1,
        );
        glow.addColorStop(0, `hsla(${packet.intentHue}, 96%, 88%, ${0.34 + packet.intentEnergy * 0.44})`);
        glow.addColorStop(0.58, `hsla(${(packet.intentHue + 24) % 360}, 92%, 66%, ${0.16 + packet.intentEnergy * 0.26})`);
        glow.addColorStop(1, "rgba(14, 22, 34, 0)");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(packet.x, packet.y, packet.particleRadius * 2.1, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `hsla(${packet.intentHue}, 96%, ${packet.actionBlocked ? 60 : 78}%, ${packet.actionBlocked ? 0.52 : 0.88})`;
        drawPacketCoreShape(packet.coreShape, packet.x, packet.y, packet.particleRadius * 1.08);
        ctx.fill();

        ctx.strokeStyle = `hsla(${packet.intentHue}, 84%, ${packet.actionBlocked ? 54 : 66}%, ${packet.actionBlocked ? 0.64 : 0.86})`;
        ctx.lineWidth = 0.52 + packet.intentEnergy * 0.7;
        drawPacketCoreShape(packet.coreShape, packet.x, packet.y, packet.particleRadius * 1.08);
        ctx.stroke();

        drawPresenceSigilCore(
            ctx,
            packet.x,
            packet.y,
            Math.max(1.05, packet.particleRadius * 0.54),
            packet.ownerSignature,
            {
                strokeStyle: "rgba(244, 250, 255, 0.94)",
                fillStyle: "rgba(226, 238, 252, 0.08)",
                lineWidth: Math.max(0.5, packet.particleRadius * 0.22),
                includeOuterRing: false,
                compact: true,
            },
        );

        if (packet.actionBlocked) {
            const markRadius = packet.particleRadius * 1.55;
            ctx.strokeStyle = "rgba(255, 132, 122, 0.8)";
            ctx.lineWidth = 0.7;
            ctx.beginPath();
            ctx.moveTo(packet.x - markRadius, packet.y - markRadius);
            ctx.lineTo(packet.x + markRadius, packet.y + markRadius);
            ctx.moveTo(packet.x + markRadius, packet.y - markRadius);
            ctx.lineTo(packet.x - markRadius, packet.y + markRadius);
            ctx.stroke();
        }
    };

    const drawDaimonRows = (
        t: number,
        rows: BackendFieldParticle[],
        w: number,
        h: number,
        globalAlpha: number,
        radiusScale: number,
        particlePrefix: string,
    ) => {
        if (rows.length <= 0) {
            return;
        }

        const swarmMode = particlePrefix === "unbound" && rows.length > 140;
        const swarmBuckets = new Map<string, {
            ownerId: string;
            ownerSignature: PresenceIdentitySignature;
            directionBucket: number;
            count: number;
            blockedCount: number;
            sumX: number;
            sumY: number;
            sumVx: number;
            sumVy: number;
            sumIntent: number;
            sumMantle: number;
            sumInstability: number;
            sumHue: number;
        }>();

        ctx.save();
        ctx.globalCompositeOperation = "lighter";
        ctx.globalAlpha = globalAlpha;

        for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
            const row = rows[rowIndex];
            const particleFallbackId = `${particlePrefix}:${rowIndex}`;
            const particleId = String(row?.id ?? "").trim() || particleFallbackId;
            const {
                isChaosParticle,
                isStaticParticle,
                isNexusParticle,
                isSmartDaimoi,
                isResourceEmitter,
                isTransferParticle,
                routeNodeId,
                graphNodeId,
            } = resolveOverlayParticleFlags(row);

            const particleMotion = resolveParticleRenderMotion(
                row,
                particleId,
                isNexusParticle || isStaticParticle,
            );
            const px = particleMotion.xNorm * w;
            const py = particleMotion.yNorm * h;
            const speed = particleMotion.speed;
            const resourceConsumeAmount = Math.max(0, Number((row as any)?.resource_consume_amount ?? 0));
            const actionBlocked = Boolean((row as any)?.resource_action_blocked);
            const isEconomyParticle = isResourceEmitter || resourceConsumeAmount > 0;
            const particleRadius = Math.max(
                isNexusParticle
                    ? 0.9
                    : (isTransferParticle || isEconomyParticle || isSmartDaimoi ? 1.14 : 0.74),
                clampValue(Number(row?.size ?? 1.0), 0.35, 2.2)
                    * radiusScale
                    * (isNexusParticle ? 0.88 : 1.16),
            );

            if (
                isTransferParticle
                && (routeNodeId.length > 0 || graphNodeId.length > 0)
                && particleTelemetryHits.length < 420
            ) {
                particleTelemetryHits.push({
                    x: clamp01(px / Math.max(1, w)),
                    y: clamp01(py / Math.max(1, h)),
                    radiusNorm: Math.max(0.004, (particleRadius * 2.2) / Math.max(w, h)),
                    graphNodeId,
                    routeNodeId,
                });
            }

            const red = Math.round(clamp01(Number(row?.r ?? 0.5)) * 255);
            const green = Math.round(clamp01(Number(row?.g ?? 0.5)) * 255);
            const blue = Math.round(clamp01(Number(row?.b ?? 0.5)) * 255);
            const colorBase = `${red},${green},${blue}`;

            if (isChaosParticle) {
                const wing = particleRadius * 1.36;
                ctx.strokeStyle = `rgba(${colorBase}, 0.92)`;
                ctx.lineWidth = 0.95;
                ctx.beginPath();
                ctx.moveTo(px - wing, py - wing);
                ctx.lineTo(px + wing, py + wing);
                ctx.moveTo(px + wing, py - wing);
                ctx.lineTo(px - wing, py + wing);
                ctx.stroke();
                continue;
            }

            if (isNexusParticle || isStaticParticle) {
                const relayMode = String((row as any)?.presence_role ?? "").trim().toLowerCase().includes("relay");
                drawNexusPacketSymbol(px, py, particleRadius * 1.14, colorBase, relayMode);
                continue;
            }

            if (!swarmMode && isSmartDaimoi) {
                drawDaimoiGhostTrail(
                    particleId,
                    particleMotion.xNorm,
                    particleMotion.yNorm,
                    speed,
                    colorBase,
                    isTransferParticle,
                    w,
                    h,
                );
            }

            const ownerPresenceId = canonicalPresenceId(
                String((row as any)?.owner_presence_id ?? (row as any)?.presence_id ?? ""),
            );
            const ownerSignature = resolvePresenceIdentitySignature(ownerPresenceId || particleId);
            const routeProbability = clamp01(Number((row as any)?.route_probability ?? (row as any)?.message_probability ?? 0.4));
            const influencePower = clamp01(Number((row as any)?.influence_power ?? 0.24));
            const intentEnergy = clamp01(0.16 + routeProbability * 0.45 + influencePower * 0.44);
            const mantleInfluence = clamp01(
                clamp01(Number((row as any)?.node_saturation ?? 0)) * 0.44
                + clamp01(Math.abs(Number((row as any)?.local_price ?? 0))) * 0.32
                + clamp01(Number((row as any)?.resource_availability ?? 0)) * 0.28,
            );
            const instability = clamp01(
                Math.abs(clampValue(Number((row as any)?.drift_score ?? 0), -1, 1)) * 0.72
                + clamp01(Number((row as any)?.package_entropy ?? 0)) * 0.56,
            );
            const intentUnits = Math.max(1, Math.round(clampValue(Number(row?.size ?? 1), 0.4, 2.8) * 2));
            const fallbackHue = presenceHueFromId(ownerPresenceId || particleId);
            const intentHue = resolveIntentHue(row, fallbackHue);
            const coreShape: "circle" | "diamond" | "square" = isTransferParticle
                ? "diamond"
                : (isEconomyParticle ? "square" : "circle");

            if (swarmMode) {
                const directionAngle = speed > 0.0001
                    ? Math.atan2(particleMotion.vy, particleMotion.vx)
                    : ownerSignature.rotation;
                const directionBucket = Math.floor(((directionAngle + Math.PI) / (Math.PI * 2)) * 12) % 12;
                const cellX = Math.max(0, Math.min(7, Math.floor(particleMotion.xNorm * 8)));
                const cellY = Math.max(0, Math.min(5, Math.floor(particleMotion.yNorm * 6)));
                const ownerKey = normalizePresenceKey(ownerPresenceId || ownerSignature.id);
                const key = `${ownerKey}|${directionBucket}|${cellX}|${cellY}`;
                const bucket = swarmBuckets.get(key) ?? {
                    ownerId: ownerPresenceId || ownerSignature.id,
                    ownerSignature,
                    directionBucket,
                    count: 0,
                    blockedCount: 0,
                    sumX: 0,
                    sumY: 0,
                    sumVx: 0,
                    sumVy: 0,
                    sumIntent: 0,
                    sumMantle: 0,
                    sumInstability: 0,
                    sumHue: 0,
                };
                bucket.count += 1;
                bucket.blockedCount += actionBlocked ? 1 : 0;
                bucket.sumX += px;
                bucket.sumY += py;
                bucket.sumVx += particleMotion.vx;
                bucket.sumVy += particleMotion.vy;
                bucket.sumIntent += intentEnergy;
                bucket.sumMantle += mantleInfluence;
                bucket.sumInstability += instability;
                bucket.sumHue += intentHue;
                swarmBuckets.set(key, bucket);
                continue;
            }

            drawDaimonPacketSymbol(
                {
                    particleId,
                    x: px,
                    y: py,
                    vx: particleMotion.vx,
                    vy: particleMotion.vy,
                    speed,
                    particleRadius,
                    intentHue,
                    intentEnergy,
                    mantleInfluence,
                    instability,
                    intentUnits,
                    actionBlocked,
                    ownerSignature,
                    coreShape,
                },
                w,
                h,
                t,
            );
        }

        if (swarmMode && swarmBuckets.size > 0) {
            const bundles = Array.from(swarmBuckets.values())
                .sort((left, right) => right.count - left.count)
                .slice(0, 120);
            for (let index = 0; index < bundles.length; index += 1) {
                const bundle = bundles[index];
                if (bundle.count <= 0) {
                    continue;
                }

                const cx = bundle.sumX / bundle.count;
                const cy = bundle.sumY / bundle.count;
                const avgVx = bundle.sumVx / bundle.count;
                const avgVy = bundle.sumVy / bundle.count;
                const avgSpeed = Math.hypot(avgVx, avgVy);
                const fallbackAngle = bundle.ownerSignature.rotation;
                const moveAngle = avgSpeed > 0.0001 ? Math.atan2(avgVy, avgVx) : fallbackAngle;
                const dirX = Math.cos(moveAngle);
                const dirY = Math.sin(moveAngle);
                const normalX = -dirY;
                const normalY = dirX;

                const countNorm = clamp01(bundle.count / 14);
                const intent = clamp01(bundle.sumIntent / bundle.count);
                const mantle = clamp01(bundle.sumMantle / bundle.count);
                const instability = clamp01(bundle.sumInstability / bundle.count);
                const hue = Math.round(bundle.sumHue / bundle.count) % 360;
                const blockedRatio = bundle.blockedCount / bundle.count;
                const ribbonWidth = 0.9 + countNorm * 3.6 + mantle * 1.2;
                const ribbonLength = clampValue(
                    10 + bundle.count * 1.9 + avgSpeed * Math.max(w, h) * 0.34,
                    12,
                    76,
                );
                const curve = instability
                    * (8 + countNorm * 11)
                    * ((bundle.directionBucket % 2 === 0) ? 1 : -1);
                const startX = cx - dirX * ribbonLength * 0.56;
                const startY = cy - dirY * ribbonLength * 0.56;
                const endX = cx + dirX * ribbonLength * 0.44;
                const endY = cy + dirY * ribbonLength * 0.44;
                const controlX = (startX + endX) * 0.5 + normalX * curve;
                const controlY = (startY + endY) * 0.5 + normalY * curve;

                ctx.strokeStyle = `hsla(${hue}, 90%, 72%, ${0.2 + intent * 0.44})`;
                ctx.lineWidth = ribbonWidth;
                ctx.beginPath();
                ctx.moveTo(startX, startY);
                ctx.quadraticCurveTo(controlX, controlY, endX, endY);
                ctx.stroke();

                for (let strand = -1; strand <= 1; strand += 1) {
                    const offset = strand * Math.max(0.42, ribbonWidth * 0.36);
                    const sx = startX + normalX * offset;
                    const sy = startY + normalY * offset;
                    const ex = endX + normalX * offset;
                    const ey = endY + normalY * offset;
                    const cx2 = controlX + normalX * offset * 0.52 + Math.sin((t * 1.9) + index + strand) * 0.9;
                    const cy2 = controlY + normalY * offset * 0.52 + Math.cos((t * 1.7) + index + strand) * 0.9;
                    ctx.strokeStyle = `hsla(${hue}, 86%, 82%, ${0.08 + intent * 0.2})`;
                    ctx.lineWidth = Math.max(0.34, ribbonWidth * 0.3);
                    ctx.beginPath();
                    ctx.moveTo(sx, sy);
                    ctx.quadraticCurveTo(cx2, cy2, ex, ey);
                    ctx.stroke();
                }

                drawPresenceSigilCore(
                    ctx,
                    endX,
                    endY,
                    Math.max(1.7, ribbonWidth * 0.56),
                    bundle.ownerSignature,
                    {
                        strokeStyle: "rgba(242, 250, 255, 0.92)",
                        fillStyle: "rgba(214, 232, 248, 0.06)",
                        lineWidth: Math.max(0.55, ribbonWidth * 0.2),
                        includeOuterRing: false,
                        compact: true,
                    },
                );

                if (blockedRatio > 0.32) {
                    const markRadius = Math.max(2.2, ribbonWidth * 0.9);
                    ctx.strokeStyle = `rgba(255, 136, 122, ${0.4 + blockedRatio * 0.5})`;
                    ctx.lineWidth = 0.72;
                    ctx.beginPath();
                    ctx.moveTo(cx - markRadius, cy - markRadius);
                    ctx.lineTo(cx + markRadius, cy + markRadius);
                    ctx.moveTo(cx + markRadius, cy - markRadius);
                    ctx.lineTo(cx - markRadius, cy + markRadius);
                    ctx.stroke();
                }
            }
        }

        ctx.restore();
    };

    const drawParticles = (
        t: number,
        field: any,
        radius: number,
        isHighlighted: boolean,
        fieldParticlesByPresence: Map<string, BackendFieldParticle[]>,
        w: number,
        h: number,
    ) => {
        const fieldId = String(field?.id ?? "").trim();
        const fieldKey = normalizePresenceKey(fieldId);
        const rows =
            fieldParticlesByPresence.get(fieldId)
            ?? fieldParticlesByPresence.get(fieldKey)
            ?? [];
        if (rows.length <= 0) {
            return;
        }
        drawDaimonRows(
            t,
            rows,
            w,
            h,
            isHighlighted ? 0.9 : 0.68,
            Math.max(0.1, radius / 62),
            fieldId || "presence",
        );
    };

    const drawUnboundParticles = (
        t: number,
        rows: BackendFieldParticle[],
        w: number,
        h: number,
    ) => {
        drawDaimonRows(
            t,
            rows,
            w,
            h,
            0.62,
            0.24,
            "unbound",
        );
    };

    const drawPresenceStatus = (
        cx: number,
        cy: number,
        radius: number,
        hue: number,
        entityState: any,
        presenceId: string,
        isHighlighted: boolean,
    ) => {
        const bpmRatio = clamp01((((entityState?.bpm || 78) - 60) / 80));
        const stabilityRatio = ratioFromMetric(entityState?.stability, 0.72);
        const resonanceRatio = ratioFromMetric(entityState?.resonance, 0.65);
        const ringRadius = radius * 1.08;
        const signature = resolvePresenceIdentitySignature(presenceId);

        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = "rgba(7, 14, 24, 0.84)";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.arc(cx, cy, ringRadius, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = isHighlighted ? "rgba(236, 247, 255, 0.9)" : "rgba(223, 238, 252, 0.72)";
        ctx.lineWidth = isHighlighted ? 1.75 : 1.25;
        const ringDash = ringDashPatternForStyle(signature.ringStyle);
        if (ringDash.length > 0) {
            ctx.setLineDash(ringDash);
            ctx.lineDashOffset = -(signature.notchAngle * 6.4);
        }
        ctx.beginPath();
        ctx.arc(cx, cy, ringRadius + 1.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (signature.ringStyle === "double") {
            ctx.strokeStyle = "rgba(207, 228, 246, 0.68)";
            ctx.lineWidth = 0.9;
            ctx.beginPath();
            ctx.arc(cx, cy, ringRadius + 4.4, 0, Math.PI * 2);
            ctx.stroke();
        }

        const layerAngle = -Math.PI / 2 + ((signature.fieldLayerSignature + 0.5) / 8) * Math.PI * 2;
        const markerInner = ringRadius + 4.9;
        const markerOuter = markerInner + 5.8;
        const markerX0 = cx + Math.cos(layerAngle) * markerInner;
        const markerY0 = cy + Math.sin(layerAngle) * markerInner;
        const markerX1 = cx + Math.cos(layerAngle) * markerOuter;
        const markerY1 = cy + Math.sin(layerAngle) * markerOuter;
        ctx.strokeStyle = "rgba(240, 250, 255, 0.88)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(markerX0, markerY0);
        ctx.lineTo(markerX1, markerY1);
        ctx.stroke();

        drawPresenceSigilCore(
            ctx,
            cx,
            cy,
            Math.max(5.2, radius * 0.42),
            signature,
            {
                strokeStyle: "rgba(230, 243, 255, 0.9)",
                fillStyle: "rgba(188, 214, 236, 0.14)",
                lineWidth: isHighlighted ? 1.18 : 0.96,
                includeOuterRing: true,
            },
        );

        const start = -Math.PI / 2;
        const rings = [
            { value: bpmRatio, color: "hsla(" + hue + ", 92%, 66%, 0.95)", width: 2.6, offset: 0 },
            { value: stabilityRatio, color: "hsla(" + ((hue + 130) % 360) + ", 88%, 62%, 0.95)", width: 2.1, offset: 4.5 },
            { value: resonanceRatio, color: "hsla(" + ((hue + 42) % 360) + ", 94%, 74%, 0.95)", width: 1.9, offset: 8 },
        ];

        rings.forEach((ring) => {
            ctx.strokeStyle = ring.color;
            ctx.lineWidth = ring.width;
            ctx.beginPath();
            ctx.arc(cx, cy, ringRadius + ring.offset, start, start + Math.PI * 2 * ring.value);
            ctx.stroke();
        });
        ctx.restore();

        return {
            bpm: Math.round(entityState?.bpm || 78),
            stabilityPct: Math.round(stabilityRatio * 100),
            resonancePct: Math.round(resonanceRatio * 100),
            fieldLayer: signature.fieldLayerSignature + 1,
        };
    };

    const drawEchoes = (t: number, w: number, h: number, state: SimulationState | null) => {
        if (!state?.echoes) return;
        state.echoes.forEach(echo => {
            const ex = echo.x * w; const ey = echo.y * h;
            const size = 4 + Math.sin(t * 2 + echo.hue) * 2;
            ctx.globalAlpha = echo.life * 0.45;
            ctx.fillStyle = "hsla(" + echo.hue + ", 80%, 72%, 0.62)";
            ctx.save();
            ctx.translate(ex, ey);
            ctx.rotate(t * 0.5 + echo.hue);
            ctx.fillRect(-size/2, -size/2, size, size);
            if (echo.life > 0.4) {
                ctx.rotate(-(t * 0.5 + echo.hue));
                ctx.fillStyle = "hsla(" + echo.hue + ", 45%, 88%, " + (echo.life * 0.7) + ")";
                ctx.font = "italic 9px serif";
                ctx.fillText(echo.text, 10, 0);
            }
            ctx.restore();
        });
    };

    const drawRiverFlow = (
        t: number,
        w: number,
        h: number,
        namedForms: Array<any>,
        state: SimulationState | null,
    ) => {
        const dynamics = state?.presence_dynamics;
        if (!dynamics) return;
        const river = namedForms.find((item: any) => item.id === "receipt_river");
        const anchor = namedForms.find((item: any) => item.id === "anchor_registry");
        const gates = namedForms.find((item: any) => item.id === "gates_of_truth");
        if (!river || !anchor || !gates) return;

        const flowRate = Math.max(0, Number(dynamics.river_flow?.rate ?? 0));
        const turbulence = Math.max(0, Number(dynamics.river_flow?.turbulence ?? 0));
        const flowNorm = Math.max(0, Math.min(1, flowRate / 12));
        const riverX = river.x * w;
        const riverY = river.y * h;
        const anchorX = anchor.x * w;
        const anchorY = anchor.y * h;
        const gatesX = gates.x * w;
        const gatesY = gates.y * h;

        ctx.save();
        ctx.globalCompositeOperation = "screen";
        ctx.globalAlpha = 0.45 + flowNorm * 0.3;
        for (let lane = 0; lane < 3; lane++) {
            const laneOffset = (lane - 1) * (10 + turbulence * 16);
            const hue = 198 + lane * 11;
            ctx.setLineDash([10 + lane * 3, 8 + lane * 2]);
            ctx.lineDashOffset = -((t * 110 * (0.6 + flowNorm)) + lane * 24);
            ctx.strokeStyle = `hsla(${hue}, 88%, ${62 + lane * 4}%, ${0.38 + flowNorm * 0.3})`;
            ctx.lineWidth = 1.4 + flowNorm * 1.6;
            ctx.beginPath();
            ctx.moveTo(riverX, riverY + laneOffset * 0.5);
            ctx.bezierCurveTo(
                riverX + (anchorX - riverX) * 0.34,
                riverY - 36 - laneOffset,
                riverX + (anchorX - riverX) * 0.72,
                anchorY + 24 + laneOffset,
                anchorX,
                anchorY,
            );
            ctx.bezierCurveTo(
                anchorX + (gatesX - anchorX) * 0.32,
                anchorY - 44 + laneOffset,
                anchorX + (gatesX - anchorX) * 0.72,
                gatesY + 22 - laneOffset,
                gatesX,
                gatesY,
            );
            ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = "rgba(176, 226, 255, 0.95)";
        ctx.font = "600 9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "left";
        ctx.fillText(
            `Receipt River / 領収書の川 ${flowRate.toFixed(2)} ${dynamics.river_flow?.unit ?? "m3/s"}`,
            Math.max(8, riverX - 52),
            Math.max(14, riverY - 18),
        );
        ctx.restore();
    };

    const drawWitnessThreadFlow = (
        t: number,
        w: number,
        h: number,
        namedForms: Array<any>,
        state: SimulationState | null,
    ) => {
        const dynamics = state?.presence_dynamics;
        const witnessState = dynamics?.witness_thread;
        if (!dynamics || !witnessState) return;

        const witness = namedForms.find((item: any) => item.id === "witness_thread");
        if (!witness) return;

        const witnessX = witness.x * w;
        const witnessY = witness.y * h;
        const continuity = clamp01(Number(witnessState.continuity_index ?? 0));

        const linkedIds = (witnessState.linked_presences ?? []).slice(0, 4);
        const linkedTargets = linkedIds
            .map((id) => namedForms.find((item: any) => item.id === id))
            .filter((item): item is any => Boolean(item));

        if (linkedTargets.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            linkedTargets.forEach((target, index) => {
                const tx = target.x * w;
                const ty = target.y * h;
                const sway = Math.sin((t * 1.7) + index * 0.8) * (6 + continuity * 14);
                const ctrl1x = witnessX + (tx - witnessX) * 0.34;
                const ctrl1y = witnessY - (18 + continuity * 28) + sway;
                const ctrl2x = witnessX + (tx - witnessX) * 0.72;
                const ctrl2y = ty + (14 + continuity * 18) - sway;

                const gradient = ctx.createLinearGradient(witnessX, witnessY, tx, ty);
                gradient.addColorStop(0, `hsla(256, 88%, 70%, ${0.34 + continuity * 0.4})`);
                gradient.addColorStop(0.45, `hsla(205, 92%, 74%, ${0.24 + continuity * 0.34})`);
                gradient.addColorStop(1, `hsla(172, 86%, 66%, ${0.2 + continuity * 0.24})`);

                ctx.setLineDash([8 + index * 2, 7 + index]);
                ctx.lineDashOffset = -((t * 54) + index * 18);
                ctx.strokeStyle = gradient;
                ctx.lineWidth = 1.1 + continuity * 1.7;
                ctx.beginPath();
                ctx.moveTo(witnessX, witnessY);
                ctx.bezierCurveTo(ctrl1x, ctrl1y, ctrl2x, ctrl2y, tx, ty);
                ctx.stroke();
            });
            ctx.setLineDash([]);
            ctx.restore();
        }

        const latestTrace = String(witnessState.lineage?.[0]?.ref ?? "");
        if (latestTrace) {
            ctx.save();
            ctx.globalCompositeOperation = "source-over";
            ctx.textAlign = "left";
            ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.fillStyle = "rgba(194, 226, 255, 0.92)";
            ctx.fillText(`Witness Thread continuity ${Math.round(continuity * 100)}%`, witnessX + 14, witnessY - 18);
            ctx.fillStyle = "rgba(160, 207, 248, 0.86)";
            ctx.fillText(`latest trace: ${latestTrace}`, witnessX + 14, witnessY - 8);
            ctx.restore();
        }
    };

    const drawFileInfluenceOverlay = (
        t: number,
        w: number,
        h: number,
        namedForms: Array<any>,
        state: SimulationState | null,
    ) => {
        const dynamics = state?.presence_dynamics;
        if (!dynamics) return;

        const impactRows = Array.isArray(dynamics.presence_impacts)
            ? dynamics.presence_impacts
            : [];
        const fileHeavyImpacts = [...impactRows]
            .filter((impact: any) => Number(impact?.affected_by?.files ?? 0) > 0.04)
            .sort(
                (a: any, b: any) =>
                    Number(b?.affected_by?.files ?? 0) - Number(a?.affected_by?.files ?? 0),
            )
            .slice(0, 6);

        if (fileHeavyImpacts.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            fileHeavyImpacts.forEach((impact: any, index) => {
                const target = namedForms.find((item: any) => item.id === impact.id);
                if (!target) return;
                const strength = clamp01(Number(impact?.affected_by?.files ?? 0));
                if (strength <= 0) return;

                const cx = target.x * w;
                const cy = target.y * h;
                const radius = 30 + (strength * 34) + Math.sin((t * 1.8) + index) * 2;
                ctx.setLineDash([10 + (index % 3) * 2, 8 + (index % 2) * 2]);
                ctx.lineDashOffset = -((t * 50) + index * 9);
                ctx.strokeStyle = `hsla(36, 96%, 68%, ${0.2 + strength * 0.42})`;
                ctx.lineWidth = 1.2 + strength * 2;
                ctx.beginPath();
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);
                ctx.stroke();
            });
            ctx.setLineDash([]);
            ctx.restore();
        }

        const paths = (dynamics.recent_file_paths ?? []).slice(0, 5);
        if (paths.length <= 0) {
            return;
        }

        const fileEvents = Number(dynamics.file_events ?? 0);
        const globalFilePulse = clamp01(fileEvents / 24);
        const leftRailX = w * 0.09;
        const topY = h * 0.16;
        const spanY = h * 0.64;
        const denominator = Math.max(1, paths.length - 1);

        ctx.save();
        ctx.globalCompositeOperation = "screen";

        for (let index = 0; index < paths.length; index++) {
            const filePath = String(paths[index] ?? "");
            const y =
                topY + (spanY * (index / denominator)) + Math.sin((t * 1.4) + index) * 5;
            const x = leftRailX + Math.cos((t * 1.15) + index) * 3;

            const impact = fileHeavyImpacts[index % Math.max(1, fileHeavyImpacts.length)];
            const target = impact
                ? namedForms.find((item: any) => item.id === impact.id)
                : null;
            const fallbackStrength = globalFilePulse > 0 ? globalFilePulse : 0.15;
            const strength = clamp01(Number(impact?.affected_by?.files ?? fallbackStrength));

            if (target) {
                const tx = target.x * w;
                const ty = target.y * h;
                const bend = (20 + strength * 38) * (index % 2 === 0 ? 1 : -1);
                ctx.setLineDash([7, 7]);
                ctx.lineDashOffset = -((t * 62) + index * 11);
                ctx.strokeStyle = `hsla(39, 92%, 67%, ${0.18 + strength * 0.34})`;
                ctx.lineWidth = 0.9 + strength * 1.6;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.bezierCurveTo(
                    x + (tx - x) * 0.3,
                    y - bend,
                    x + (tx - x) * 0.7,
                    ty + bend * 0.5,
                    tx,
                    ty,
                );
                ctx.stroke();
            }

            const beaconRadius = 4 + strength * 6;
            const glow = ctx.createRadialGradient(x, y, 1, x, y, beaconRadius * 2.2);
            glow.addColorStop(0, `rgba(255, 232, 186, ${0.7 + strength * 0.2})`);
            glow.addColorStop(0.55, `rgba(255, 182, 93, ${0.24 + strength * 0.3})`);
            glow.addColorStop(1, "rgba(96, 52, 20, 0)");
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.arc(x, y, beaconRadius * 2.2, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "rgba(255, 232, 190, 0.96)";
            ctx.beginPath();
            ctx.arc(x, y, beaconRadius, 0, Math.PI * 2);
            ctx.fill();

            ctx.globalCompositeOperation = "source-over";
            ctx.textAlign = "left";
            ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.fillStyle = "rgba(255, 227, 177, 0.96)";
            ctx.fillText(shortPathLabel(filePath), x + 10, y + 2);
            ctx.globalCompositeOperation = "screen";
        }

        ctx.setLineDash([]);
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "left";
        ctx.font = "600 9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(255, 218, 150, 0.95)";
        ctx.fillText("File Influence / ファイル影響", leftRailX - 6, Math.max(16, topY - 20));
        ctx.font = "500 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(240, 198, 144, 0.86)";
        ctx.fillText("recent file drift -> affected presence", leftRailX - 6, Math.max(24, topY - 10));
        ctx.restore();
    };

    const drawFileCategoryGraph = (
        t: number,
        w: number,
        h: number,
        state: SimulationState | null,
        namedForms: Array<any>,
    ) => {
        const graph = resolveFileGraph(state);
        if (!graph) {
            return;
        }

        const fieldNodes = Array.isArray(graph.field_nodes) ? graph.field_nodes : [];
        const fileNodes = Array.isArray(graph.file_nodes) ? graph.file_nodes : [];
        const crawlerNexusNodes = Array.isArray((graph as any).crawler_nodes)
            ? ((graph as any).crawler_nodes as any[])
            : [];
        const nexusNodes = [...fileNodes, ...crawlerNexusNodes];
        const graphNodes = Array.isArray(graph.nodes) && graph.nodes.length > 0
            ? graph.nodes
            : [...fieldNodes, ...nexusNodes];
        const tagNodes = Array.isArray((graph as any).tag_nodes)
            ? ((graph as any).tag_nodes as any[])
            : graphNodes.filter((node: any) => String(node?.node_type ?? "").toLowerCase() === "tag");
        const nodeById = new Map(graphNodes.map((node: any) => [String(node.id), node]));
        const edges = Array.isArray(graph.edges) ? graph.edges : [];
        const renderEdges = thinFileGraphEdgesForRendering(
            edges,
            nodeById,
            nexusNodes.length,
            selectedGraphNodeId,
        );
        let workspaceBindCount = 0;
        const resourceCounts: Record<string, number> = {};
        const provenanceCounts: Record<FileNodeProvenanceKind, number> = {
            workspace: 0,
            archive: 0,
            web: 0,
            memory: 0,
            synthetic: 0,
        };
        const graphPositionForNode = (node: any): { x: number; y: number } => ({
            x: clamp01(Number(node?.x ?? 0.5)),
            y: clamp01(Number(node?.y ?? 0.5)),
        });
        const embedLayers = Array.isArray((graph as any).embed_layers)
            ? ((graph as any).embed_layers as any[])
            : [];
        const activeEmbedLayers = embedLayers
            .filter((layer) => layer && layer.active !== false)
            .slice(0, 7);
        const layerDepth = layerDepthRef.current;
        const layerDepthNorm = clamp01((layerDepth - 0.4) / 1.5);
        const layerLaneTop = h * 0.16;
        const layerLaneStep = Math.max(20, Math.min(44, h * 0.08));
        const layerLaneX = w - Math.max(78, Math.min(178, w * 0.2));
        const layerLaneByKey = new Map<string, number>();
        activeEmbedLayers.forEach((layer, index) => {
            const layerId = String(layer.id ?? "").trim();
            const layerKey = String(layer.key ?? "").trim();
            if (layerId) {
                layerLaneByKey.set(layerId, index);
            }
            if (layerKey) {
                layerLaneByKey.set(layerKey, index);
            }
        });

        if (activeEmbedLayers.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            for (let index = 0; index < activeEmbedLayers.length; index++) {
                const layer = activeEmbedLayers[index] as any;
                const laneY = layerLaneTop + index * layerLaneStep;
                const hue = layerHueByIndex(index);
                const lineGlow = ctx.createLinearGradient(layerLaneX, laneY, w - 12, laneY);
                lineGlow.addColorStop(0, `hsla(${hue}, 84%, 68%, ${0.08 + layerDepthNorm * 0.16})`);
                lineGlow.addColorStop(1, `hsla(${(hue + 24) % 360}, 90%, 72%, ${0.24 + layerDepthNorm * 0.28})`);
                ctx.strokeStyle = lineGlow;
                ctx.lineWidth = 0.9 + layerDepthNorm * 1.4;
                ctx.beginPath();
                ctx.moveTo(layerLaneX, laneY);
                ctx.lineTo(w - 12, laneY);
                ctx.stroke();

                ctx.fillStyle = `hsla(${hue}, 95%, 82%, ${0.7 + layerDepthNorm * 0.2})`;
                ctx.beginPath();
                ctx.arc(layerLaneX, laneY, 2 + layerDepthNorm * 1.8, 0, Math.PI * 2);
                ctx.fill();

                ctx.globalCompositeOperation = "source-over";
                ctx.textAlign = "right";
                ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
                ctx.fillStyle = `hsla(${hue}, 88%, 84%, ${0.74 + layerDepthNorm * 0.16})`;
                ctx.fillText(
                    shortPathLabel(String(layer.label ?? layer.id ?? `layer-${index + 1}`)),
                    w - 14,
                    laneY - 4,
                );
                ctx.globalCompositeOperation = "screen";
            }
            ctx.restore();
        }

        if (renderEdges.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            ctx.lineCap = "round";
            for (let i = 0; i < renderEdges.length; i++) {
                const edge = renderEdges[i] as FileGraphRenderEdge;
                const src = nodeById.get(edge.source);
                const tgt = nodeById.get(edge.target);
                if (!src || !tgt) {
                    continue;
                }
                const srcPos = graphPositionForNode(src);
                const tgtPos = graphPositionForNode(tgt);
                const sx = srcPos.x * w;
                const sy = srcPos.y * h;
                const tx = tgtPos.x * w;
                const ty = tgtPos.y * h;
                const weight = clamp01(Number(edge.weight ?? 0.2));
                const hue = Number(src.hue ?? tgt.hue ?? 210);
                const dx = tx - sx;
                const dy = ty - sy;
                const distance = Math.hypot(dx, dy);
                const stretchNorm = clamp01(distance / Math.max(90, Math.min(w, h) * 0.52));
                const elasticity = clamp01((weight * 0.56) + (stretchNorm * 0.44));
                const normalX = distance > 0.0001 ? -dy / distance : 0;
                const normalY = distance > 0.0001 ? dx / distance : 1;
                const bendBase = (10 + stretchNorm * 24) * (i % 2 === 0 ? 1 : -1);
                const bend = bendBase + Math.sin((t * 1.28) + (i * 0.2)) * (4 + elasticity * 8);
                const controlX = (sx + tx) / 2 + normalX * bend;
                const controlY = (sy + ty) / 2 + normalY * bend;

                ctx.strokeStyle = `hsla(${hue}, 90%, 70%, ${0.08 + elasticity * 0.36})`;
                ctx.lineWidth = 0.58 + elasticity * 2.35;
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.quadraticCurveTo(controlX, controlY, tx, ty);
                ctx.stroke();

                ctx.strokeStyle = `hsla(${(hue + 18) % 360}, 88%, 80%, ${0.05 + elasticity * 0.26})`;
                ctx.lineWidth = Math.max(0.28, 0.36 + elasticity * 0.92);
                ctx.setLineDash([5 + elasticity * 8, 7 + (1 - elasticity) * 6]);
                ctx.lineDashOffset = -((t * 20) + i * 3);
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.quadraticCurveTo(controlX, controlY, tx, ty);
                ctx.stroke();
                ctx.setLineDash([]);
            }
            ctx.restore();
        }

        const workspaceBindingMap = museWorkspaceBindingsRef.current;
        const showWorkspaceBindingOverlay = false;
        if (showWorkspaceBindingOverlay && Object.keys(workspaceBindingMap).length > 0 && fileNodes.length > 0) {
            const presencePositionById = new Map<string, { x: number; y: number; hue: number }>();
            const filePositionById = new Map<string, { x: number; y: number; hue: number }>();

            for (const form of namedForms) {
                const presenceId = normalizePresenceKey(String((form as any)?.id ?? ""));
                if (!presenceId || presencePositionById.has(presenceId)) {
                    continue;
                }
                presencePositionById.set(presenceId, {
                    x: clamp01(Number((form as any)?.x ?? 0.5)),
                    y: clamp01(Number((form as any)?.y ?? 0.5)),
                    hue: Number((form as any)?.hue ?? 198),
                });
            }

            for (const field of fieldNodes as any[]) {
                const fx = clamp01(Number(field?.x ?? 0.5));
                const fy = clamp01(Number(field?.y ?? 0.5));
                const fh = Number(field?.hue ?? 198);
                const fieldId = String(field?.id ?? "").trim();
                const conceptFromFieldId = fieldId.startsWith("presence:concept:")
                    ? fieldId.slice("presence:concept:".length)
                    : "";
                const candidates = [
                    String(field?.dominant_presence ?? ""),
                    String(field?.concept_presence_id ?? ""),
                    String(field?.organized_by ?? ""),
                    conceptFromFieldId,
                ];
                for (const candidate of candidates) {
                    const normalizedPresence = normalizePresenceKey(candidate);
                    if (!normalizedPresence || presencePositionById.has(normalizedPresence)) {
                        continue;
                    }
                    presencePositionById.set(normalizedPresence, { x: fx, y: fy, hue: fh });
                }
            }

            for (const node of fileNodes as any[]) {
                const nx = clamp01(Number(node?.x ?? 0.5));
                const ny = clamp01(Number(node?.y ?? 0.5));
                const nh = Number(node?.hue ?? 210);
                const id = String(node?.id ?? "").trim();
                if (id && !filePositionById.has(id)) {
                    filePositionById.set(id, { x: nx, y: ny, hue: nh });
                }
                const nodeId = String(node?.node_id ?? "").trim();
                if (nodeId && !filePositionById.has(nodeId)) {
                    filePositionById.set(nodeId, { x: nx, y: ny, hue: nh });
                }
            }

            ctx.save();
            ctx.globalCompositeOperation = "screen";
            ctx.lineCap = "round";
            let presenceLane = 0;
            for (const [presenceId, nodeIds] of Object.entries(workspaceBindingMap)) {
                const source = presencePositionById.get(normalizePresenceKey(presenceId));
                if (!source || !Array.isArray(nodeIds) || nodeIds.length <= 0) {
                    continue;
                }
                const sourceX = source.x * w;
                const sourceY = source.y * h;
                const bindIds = nodeIds.slice(0, 24);
                for (let nodeIndex = 0; nodeIndex < bindIds.length; nodeIndex += 1) {
                    const target = filePositionById.get(String(bindIds[nodeIndex] ?? "").trim());
                    if (!target) {
                        continue;
                    }
                    workspaceBindCount += 1;
                    const targetX = target.x * w;
                    const targetY = target.y * h;
                    const hue = Number.isFinite(source.hue) ? source.hue : target.hue;
                    const alpha = Math.min(0.74, 0.22 + (bindIds.length * 0.01));
                    const bend = Math.sin((t * 1.7) + (presenceLane * 0.55) + (nodeIndex * 0.28)) * 7;
                    ctx.strokeStyle = `hsla(${hue}, 92%, 72%, ${alpha})`;
                    ctx.lineWidth = 0.7;
                    ctx.setLineDash([2.6, 4.8]);
                    ctx.lineDashOffset = -((t * 24) + presenceLane * 5 + nodeIndex);
                    ctx.beginPath();
                    ctx.moveTo(sourceX, sourceY);
                    ctx.quadraticCurveTo(
                        ((sourceX + targetX) / 2) + bend,
                        ((sourceY + targetY) / 2) - (bend * 0.42),
                        targetX,
                        targetY,
                    );
                    ctx.stroke();
                }
                presenceLane += 1;
            }
            ctx.setLineDash([]);
            ctx.restore();
        }

        if (fieldNodes.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "source-over";
            for (const field of fieldNodes as any[]) {
                const fx = clamp01(Number(field.x ?? 0.5)) * w;
                const fy = clamp01(Number(field.y ?? 0.5)) * h;
                const hue = Number(field.hue ?? 200);
                const presenceKind = String(field?.presence_kind ?? "").trim().toLowerCase();
                const isConceptPresence = presenceKind === "concept";
                const isOrganizerPresence = presenceKind === "organizer";
                const radius = isOrganizerPresence ? 10.5 : (isConceptPresence ? 5.3 : 8.5);
                ctx.strokeStyle = `hsla(${hue}, 88%, ${isConceptPresence ? 72 : 64}%, ${isConceptPresence ? 0.58 : 0.48})`;
                ctx.lineWidth = isOrganizerPresence ? 1.35 : (isConceptPresence ? 0.95 : 1.1);
                ctx.beginPath();
                ctx.arc(fx, fy, radius, 0, Math.PI * 2);
                ctx.stroke();

                if (isOrganizerPresence) {
                    ctx.setLineDash([3, 3]);
                    ctx.strokeStyle = `hsla(${hue}, 92%, 72%, 0.56)`;
                    ctx.lineWidth = 1.05;
                    ctx.beginPath();
                    ctx.arc(fx, fy, radius + 4.2, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                if (isConceptPresence) {
                    const glow = ctx.createRadialGradient(fx, fy, 0, fx, fy, radius * 2.6);
                    glow.addColorStop(0, `hsla(${hue}, 92%, 78%, 0.34)`);
                    glow.addColorStop(1, "rgba(12, 18, 30, 0)");
                    ctx.fillStyle = glow;
                    ctx.beginPath();
                    ctx.arc(fx, fy, radius * 2.6, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            ctx.restore();
        }

        if (tagNodes.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            for (let i = 0; i < tagNodes.length; i++) {
                const tag = tagNodes[i] as any;
                const nx = clamp01(Number(tag.x ?? 0.5));
                const ny = clamp01(Number(tag.y ?? 0.5));
                const px = nx * w;
                const py = ny * h;
                const memberCount = Number(tag.member_count ?? 1);
                const pulse = 0.5 + Math.sin((t * 2.1) + i * 0.33) * 0.5;
                const hue = Number(tag.hue ?? 38);
                const radius = 1.4 + Math.min(4, Math.log2(Math.max(2, memberCount + 1))) + pulse * 0.7;

                const halo = ctx.createRadialGradient(px, py, 0, px, py, radius * 2.8);
                halo.addColorStop(0, `hsla(${hue}, 94%, 76%, 0.7)`);
                halo.addColorStop(0.62, `hsla(${(hue + 34) % 360}, 86%, 60%, 0.22)`);
                halo.addColorStop(1, "rgba(16, 24, 38, 0)");
                ctx.fillStyle = halo;
                ctx.beginPath();
                ctx.arc(px, py, radius * 2.8, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = `hsla(${hue}, 92%, 78%, 0.76)`;
                ctx.lineWidth = 0.85;
                ctx.beginPath();
                ctx.moveTo(px, py - radius);
                ctx.lineTo(px + radius, py);
                ctx.lineTo(px, py + radius);
                ctx.lineTo(px - radius, py);
                ctx.closePath();
                ctx.stroke();

                graphNodeHits.push({
                    node: tag,
                    x: nx,
                    y: ny,
                    radiusNorm: (radius * 1.8) / Math.max(w, h),
                    nodeKind: "file",
                    resourceKind: "blob",
                });
            }
            ctx.restore();
        }

        if (nexusNodes.length > 0) {
            const showProvenanceOrbitDetail = nexusNodes.length <= 72;
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            for (let i = 0; i < nexusNodes.length; i++) {
                const node = nexusNodes[i] as any;
                const nx = clamp01(Number(node.x ?? 0.5));
                const ny = clamp01(Number(node.y ?? 0.5));
                const px = nx * w;
                const py = ny * h;
                const importance = clamp01(Number(node.importance ?? 0.2));
                const pulse = 0.5 + Math.sin((t * 3) + i * 0.33) * 0.5;
                const resourceKind = resourceKindForNode(node);
                const provenanceKind = fileNodeProvenanceKind(node);
                provenanceCounts[provenanceKind] += 1;
                resourceCounts[resourceKind] = (resourceCounts[resourceKind] ?? 0) + 1;
                const nodeType = String(node?.node_type ?? "file").trim().toLowerCase();
                const crawlerKind = String(node?.crawler_kind ?? "url").trim().toLowerCase();
                const fallbackHue = nodeType === "crawler"
                    ? (crawlerKind === "domain" ? 172 : (crawlerKind === "content" ? 26 : Number(node.hue ?? 205)))
                    : Number(node.hue ?? 210);
                const visual = resourceVisualSpec(resourceKind, fallbackHue);
                const nexusClass = nexusVisualClassForNode(node);
                let radius = (1.8 + importance * 3.2 + pulse * 0.9) * (resourceKind === "video" ? 1.08 : 1);
                if (nexusClass === "anchor") {
                    radius *= 1.24;
                } else if (nexusClass === "relay") {
                    radius *= 0.92;
                } else if (nexusClass === "resource") {
                    radius *= 1.08;
                }
                const isSelected = selectedGraphNodeId !== "" && selectedGraphNodeId === String(node.id ?? "");
                if (isSelected) {
                    radius += 2;
                }
                const hue = visual.hue;
                const lift = (1.9 + importance * 3.1) * visual.liftBoost;
                const depthY = py + lift;
                const walletFill = nexusWalletFillRatio(node);
                const saturationFill = nexusSaturationRatio(node);

                ctx.fillStyle = `hsla(${hue}, ${Math.round(visual.saturation * 0.9)}%, ${Math.round(visual.value * 0.44)}%, ${0.16 + (isSelected ? 0.1 : 0.02)})`;
                fillResourceShape(ctx, visual.shape, px, depthY, radius * 1.05);

                ctx.strokeStyle = `hsla(${hue}, 86%, 72%, ${0.16 + importance * 0.25})`;
                ctx.lineWidth = 0.45 + importance * 0.8;
                ctx.beginPath();
                ctx.moveTo(px, depthY - radius * 0.35);
                ctx.lineTo(px, py + radius * 0.35);
                ctx.stroke();

                const glowScale = nexusClass === "anchor" ? 2.8 : (nexusClass === "relay" ? 1.9 : 2.2);
                const glow = ctx.createRadialGradient(px, py, 0, px, py, radius * glowScale);
                glow.addColorStop(0, `hsla(${hue}, ${visual.saturation}%, ${visual.value}%, ${isSelected ? 0.88 : (0.48 * visual.glowBoost)})`);
                glow.addColorStop(0.62, `hsla(${(hue + 24) % 360}, ${Math.max(52, visual.saturation - 10)}%, ${Math.max(44, visual.value - 34)}%, ${isSelected ? 0.42 : 0.22})`);
                glow.addColorStop(1, "rgba(18, 26, 38, 0)");
                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(px, py, radius * glowScale, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = `hsla(${hue}, ${Math.min(96, visual.saturation + 6)}%, ${Math.min(98, visual.value + 2)}%, ${isSelected ? 0.98 : 0.9})`;
                fillResourceShape(ctx, visual.shape, px, py, radius);

                ctx.strokeStyle = `hsla(${hue}, ${Math.max(68, visual.saturation - 8)}%, ${Math.max(52, visual.value - 34)}%, ${isSelected ? 0.95 : 0.58})`;
                ctx.lineWidth = isSelected ? 1.35 : 0.9;
                strokeResourceShape(ctx, visual.shape, px, py, radius);

                ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
                fillResourceShape(ctx, visual.shape, px - radius * 0.26, py - radius * 0.28, Math.max(0.55, radius * 0.28));

                if (nexusClass === "anchor") {
                    const ringRadius = radius * 1.74;
                    ctx.strokeStyle = `hsla(${hue}, 86%, 78%, ${isSelected ? 0.92 : 0.74})`;
                    ctx.lineWidth = isSelected ? 1.34 : 1.04;
                    ctx.beginPath();
                    ctx.arc(px, py, ringRadius, 0, Math.PI * 2);
                    ctx.stroke();

                    ctx.strokeStyle = `hsla(${(hue + 36) % 360}, 82%, 70%, 0.82)`;
                    ctx.lineWidth = Math.max(0.9, radius * 0.24);
                    ctx.beginPath();
                    ctx.arc(px, py, ringRadius - radius * 0.42, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * walletFill);
                    ctx.stroke();

                    const anchorPresenceId = canonicalPresenceId(
                        String(node?.dominant_presence ?? node?.concept_presence_id ?? node?.presence_id ?? "anchor_registry"),
                    ) || "anchor_registry";
                    const anchorSignature = resolvePresenceIdentitySignature(anchorPresenceId);
                    drawPresenceSigilCore(
                        ctx,
                        px,
                        py,
                        Math.max(2.1, radius * 0.62),
                        anchorSignature,
                        {
                            strokeStyle: "rgba(238, 248, 255, 0.92)",
                            fillStyle: "rgba(212, 231, 247, 0.08)",
                            lineWidth: Math.max(0.68, radius * 0.15),
                            includeOuterRing: false,
                            compact: true,
                        },
                    );
                } else if (nexusClass === "relay") {
                    const relayRadius = radius * 1.55;
                    ctx.strokeStyle = `hsla(${hue}, 82%, 78%, 0.42)`;
                    ctx.lineWidth = 0.84;
                    ctx.setLineDash([3.2, 4.2]);
                    ctx.lineDashOffset = -((t * 24) + i * 3);
                    ctx.beginPath();
                    ctx.arc(px, py, relayRadius, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.setLineDash([]);

                    ctx.strokeStyle = `hsla(${hue}, 80%, 74%, 0.24)`;
                    ctx.lineWidth = 0.72;
                    ctx.beginPath();
                    ctx.moveTo(px - relayRadius * 1.1, py + relayRadius * 0.24);
                    ctx.lineTo(px - relayRadius * 0.26, py + relayRadius * 0.66);
                    ctx.stroke();
                } else if (nexusClass === "resource") {
                    const frameShape: GraphNodeShape = visual.shape === "circle"
                        ? "hexagon"
                        : (visual.shape === "diamond" ? "square" : visual.shape);
                    const frameRadius = radius * 1.44;
                    const capGlow = clamp01((saturationFill - 0.58) / 0.42);
                    ctx.strokeStyle = `hsla(${hue}, 86%, ${62 + capGlow * 16}%, ${0.58 + capGlow * 0.24})`;
                    ctx.lineWidth = 0.84 + capGlow * 1.1;
                    strokeResourceShape(ctx, frameShape, px, py, frameRadius);

                    if (capGlow > 0.02) {
                        const capHalo = ctx.createRadialGradient(px, py, 0, px, py, frameRadius * 2.1);
                        capHalo.addColorStop(0, `hsla(${hue}, 94%, 78%, ${0.2 + capGlow * 0.44})`);
                        capHalo.addColorStop(1, "rgba(14, 20, 30, 0)");
                        ctx.fillStyle = capHalo;
                        ctx.beginPath();
                        ctx.arc(px, py, frameRadius * 2.1, 0, Math.PI * 2);
                        ctx.fill();
                    }
                } else {
                    ctx.strokeStyle = `hsla(${hue}, 86%, 76%, ${isSelected ? 0.78 : 0.4})`;
                    ctx.lineWidth = 0.72;
                    ctx.beginPath();
                    ctx.arc(px, py, radius * 1.4, 0, Math.PI * 2);
                    ctx.stroke();
                }

                if (showProvenanceOrbitDetail || isSelected) {
                    const provenanceHue = fileNodeProvenanceHue(provenanceKind);
                    const provenanceOrbit = radius + 2.4 + importance * 2.2;
                    const provenanceAngle = (t * 0.9) + i * 0.41;
                    const provenanceX = px + Math.cos(provenanceAngle) * provenanceOrbit;
                    const provenanceY = py + Math.sin(provenanceAngle) * provenanceOrbit;
                    const provenanceDot = Math.max(0.8, 0.85 + importance * 0.8);
                    ctx.fillStyle = `hsla(${provenanceHue}, 92%, 80%, 0.92)`;
                    ctx.beginPath();
                    ctx.arc(provenanceX, provenanceY, provenanceDot, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.strokeStyle = `hsla(${provenanceHue}, 84%, 70%, 0.38)`;
                    ctx.lineWidth = 0.75;
                    ctx.setLineDash([2.5, 4]);
                    ctx.lineDashOffset = -((t * 28) + i * 0.7);
                    ctx.beginPath();
                    ctx.arc(px, py, provenanceOrbit, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }

                graphNodeHits.push({
                    node,
                    x: nx,
                    y: ny,
                    radiusNorm: (radius + lift * 0.42) / Math.max(w, h),
                    nodeKind: nodeType === "crawler" ? "crawler" : "file",
                    resourceKind,
                });

                const layerPoints = Array.isArray(node.embed_layer_points)
                    ? node.embed_layer_points
                    : [];
                for (let layerIdx = 0; layerIdx < layerPoints.length; layerIdx++) {
                    const layer = layerPoints[layerIdx] as any;
                    if (layer?.active === false) {
                        continue;
                    }
                    const lx = clamp01(Number(layer?.x ?? nx)) * w;
                    const ly = clamp01(Number(layer?.y ?? ny)) * h;
                    const layerHue = Number(layer?.hue ?? ((hue + 46 + layerIdx * 21) % 360));
                    const layerKey = String(layer?.id ?? layer?.key ?? "").trim();
                    const laneIndex = layerLaneByKey.get(layerKey);
                    const laneY = laneIndex === undefined
                        ? ly
                        : layerLaneTop + laneIndex * layerLaneStep;
                    const laneHue = laneIndex === undefined
                        ? layerHue
                        : layerHueByIndex(laneIndex);
                    const depthAlpha = 0.2 + layerDepthNorm * 0.56;
                    const layerRadius = Math.max(0.9, 0.85 + (importance * 0.85));

                    ctx.strokeStyle = `hsla(${layerHue}, 86%, 70%, ${depthAlpha})`;
                    ctx.lineWidth = 0.55 + layerDepthNorm * 0.62;
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(lx, ly);
                    ctx.stroke();

                    if (laneIndex !== undefined) {
                        ctx.strokeStyle = `hsla(${laneHue}, 92%, 72%, ${0.16 + layerDepthNorm * 0.34})`;
                        ctx.lineWidth = 0.45 + layerDepthNorm * 0.58;
                        ctx.setLineDash([4, 6]);
                        ctx.lineDashOffset = -((t * 36) + laneIndex * 4);
                        ctx.beginPath();
                        ctx.moveTo(lx, ly);
                        ctx.lineTo(layerLaneX, laneY);
                        ctx.stroke();
                        ctx.setLineDash([]);

                        ctx.fillStyle = `hsla(${laneHue}, 94%, 84%, ${0.42 + layerDepthNorm * 0.28})`;
                        ctx.beginPath();
                        ctx.arc(layerLaneX, laneY, 1.4 + layerDepthNorm * 1.2, 0, Math.PI * 2);
                        ctx.fill();
                    }

                    const layerGlow = ctx.createRadialGradient(lx, ly, 0, lx, ly, layerRadius * 2.8);
                    layerGlow.addColorStop(0, `hsla(${layerHue}, 92%, 76%, 0.76)`);
                    layerGlow.addColorStop(0.58, `hsla(${(layerHue + 28) % 360}, 84%, 58%, 0.32)`);
                    layerGlow.addColorStop(1, "rgba(10, 16, 28, 0)");
                    ctx.fillStyle = layerGlow;
                    ctx.beginPath();
                    ctx.arc(lx, ly, layerRadius * 2.8, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.fillStyle = `hsla(${layerHue}, 95%, 86%, 0.95)`;
                    ctx.beginPath();
                    ctx.arc(lx, ly, layerRadius, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            ctx.restore();
        }

        const topFieldRows = Object.entries(graph.stats?.field_counts ?? {})
            .sort((a, b) => Number(b[1]) - Number(a[1]))
            .slice(0, 4)
            .map(([field, count]) => `${field}:${count}`)
            .join("  ");
        const resourceRows = ["text", "image", "audio", "archive", "blob"]
            .map((kind) => `${kind}:${resourceCounts[kind] ?? 0}`)
            .join("  ");
        const provenanceRows = [
            `workspace:${provenanceCounts.workspace}`,
            `archive:${provenanceCounts.archive}`,
            `web:${provenanceCounts.web}`,
            `memory:${provenanceCounts.memory}`,
            `derived:${provenanceCounts.synthetic}`,
        ].join("  ");
        const activeLayerLabels = activeEmbedLayers
            .filter((layer) => layer && layer.active !== false)
            .slice(0, 3)
            .map((layer) => shortPathLabel(String(layer.label ?? layer.id ?? "")))
            .filter((value) => value.length > 0);
        const layerRows = activeLayerLabels.length > 0
            ? activeLayerLabels.join("  |  ")
            : "none";
        const conceptPresences = Array.isArray((graph as any).concept_presences)
            ? ((graph as any).concept_presences as any[])
            : [];
        const conceptRows = conceptPresences
            .slice(0, 2)
            .map((row) => shortPathLabel(String(row.label ?? row.id ?? "")))
            .filter((value) => value.length > 0)
            .join("  |  ");
        const conceptCount = Number((graph as any)?.stats?.concept_presence_count ?? conceptPresences.length ?? 0);
        const tagRows = tagNodes
            .slice(0, 3)
            .map((row) => shortPathLabel(String((row as any)?.label ?? (row as any)?.tag ?? (row as any)?.id ?? "")))
            .filter((value) => value.length > 0)
            .join("  |  ");
        const tagCount = Number((graph as any)?.stats?.tag_count ?? tagNodes.length ?? 0);
        const edgeRows = renderEdges.length >= edges.length
            ? `edges rendered: ${edges.length}`
            : `edges rendered: ${renderEdges.length}/${edges.length} (hub-capped)`;
        const inbox = graph.inbox;
        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "left";
        ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = inbox?.is_empty ? "rgba(158, 238, 194, 0.96)" : "rgba(255, 214, 140, 0.96)";
        const inboxText = inbox?.is_empty
            ? `ημ inbox: empty | knowledge: ${Number(graph.stats?.knowledge_entries ?? 0)}`
            : `ημ inbox: ${Number(inbox?.pending_count ?? 0)} pending`;
        ctx.fillText(inboxText, 10, 16);
        ctx.fillStyle = "rgba(182, 218, 250, 0.92)";
        ctx.fillText(`field categories: ${topFieldRows || "none"}`, 10, 27);
        ctx.fillStyle = "rgba(208, 228, 245, 0.88)";
        ctx.fillText(`resource kinds: ${resourceRows}`, 10, 51);
        ctx.fillStyle = "rgba(195, 229, 255, 0.86)";
        ctx.fillText(`provenance lanes: ${provenanceRows}`, 10, 62);
        ctx.fillStyle = "rgba(190, 223, 248, 0.84)";
        ctx.fillText(`embed layers: ${layerRows}`, 10, 73);
        ctx.fillStyle = "rgba(211, 232, 255, 0.82)";
        ctx.fillText(`concept presences: ${conceptCount}${conceptRows ? ` | ${conceptRows}` : ""}`, 10, 84);
        ctx.fillStyle = "rgba(214, 236, 255, 0.8)";
        ctx.fillText(`tags: ${tagCount}${tagRows ? ` | ${tagRows}` : ""}`, 10, 95);
        ctx.fillStyle = "rgba(202, 231, 255, 0.76)";
        ctx.fillText(edgeRows, 10, 106);
        ctx.fillStyle = "rgba(197, 235, 255, 0.74)";
        ctx.fillText(`muse binds: ${workspaceBindCount}`, 10, 117);
        ctx.restore();

        if (selectedGraphNodeId) {
            const selected = nexusNodes.find((row: any) => String(row.id) === selectedGraphNodeId)
                ?? tagNodes.find((row: any) => String(row.id) === selectedGraphNodeId);
            if (selected) {
                const sx = clamp01(Number(selected.x ?? 0.5)) * w;
                const sy = clamp01(Number(selected.y ?? 0.5)) * h;
                const selectedResourceKind = resourceKindForNode(selected);
                const selectedNodeType = String(selected?.node_type ?? "file").trim().toLowerCase();
                const provenanceKind = fileNodeProvenanceKind(selected);
                const provenanceLabel = fileNodeProvenanceLabel(provenanceKind);
                const label = shortPathLabel(
                    String(
                        selected.source_rel_path
                        || selected.archived_rel_path
                        || selected.name
                        || selected.label
                        || selected.id
                        || "",
                    ),
                );
                selectedGraphNodeLabel = label;
                ctx.save();
                ctx.globalCompositeOperation = "source-over";
                ctx.fillStyle = "rgba(8, 18, 29, 0.84)";
                ctx.strokeStyle = "rgba(164, 214, 255, 0.72)";
                ctx.lineWidth = 1;
                const boxX = Math.min(w - 208, sx + 10);
                const boxY = Math.max(10, sy - 56);
                ctx.beginPath();
                ctx.roundRect(boxX, boxY, 198, 50, 6);
                ctx.fill();
                ctx.stroke();
                ctx.fillStyle = "rgba(222, 241, 255, 0.96)";
                ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
                ctx.fillText(label, boxX + 6, boxY + 11);
                ctx.fillStyle = "rgba(180, 225, 255, 0.9)";
                ctx.fillText(`resource ${resourceKindLabel(selectedResourceKind)}`, boxX + 6, boxY + 21);
                ctx.fillText(
                    `field ${String(selected.dominant_field ?? "f6")} | layers ${Number(selected.embed_layer_count ?? 0)}`,
                    boxX + 6,
                    boxY + 30,
                );
                ctx.fillStyle = "rgba(170, 216, 247, 0.9)";
                ctx.fillText(
                    `origin ${provenanceLabel} | range ${Math.round(documentRangeFromImportance(Number(selected?.importance ?? 0.2)) * 100)}%`,
                    boxX + 6,
                    boxY + 39,
                );
                ctx.fillStyle = "rgba(165, 212, 248, 0.9)";
                const detailText = selectedNodeType === "tag"
                    ? `tag ${String(selected.tag ?? selected.node_id ?? "")}`
                    : (
                        selectedNodeType === "crawler"
                            ? `crawler ${String(selected.crawler_kind ?? "url")} | ${shortPathLabel(String(selected.domain ?? selected.url ?? ""))}`
                            : `concept ${String(selected.concept_presence_label ?? "unassigned")}`
                    );
                ctx.fillText(detailText, boxX + 6, boxY + 48);
                ctx.restore();
            }
        }
    };

    const drawCrawlerCategoryGraph = (
        t: number,
        w: number,
        h: number,
        state: SimulationState | null,
    ) => {
        const graph = resolveCrawlerGraph(state);
        if (!graph) {
            return;
        }

        const fieldNodes = Array.isArray(graph.field_nodes) ? graph.field_nodes : [];
        const crawlerNodes = Array.isArray(graph.crawler_nodes) ? graph.crawler_nodes : [];
        const graphNodes = Array.isArray(graph.nodes) && graph.nodes.length > 0
            ? graph.nodes
            : [...fieldNodes, ...crawlerNodes];
        const nodeById = new Map(graphNodes.map((node: any) => [String(node.id), node]));
        const edges = Array.isArray(graph.edges) ? graph.edges : [];

        if (edges.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            for (let i = 0; i < edges.length; i++) {
                const edge = edges[i] as any;
                const src = nodeById.get(String(edge.source));
                const tgt = nodeById.get(String(edge.target));
                if (!src || !tgt) {
                    continue;
                }
                const sx = clamp01(Number(src.x ?? 0.5)) * w;
                const sy = clamp01(Number(src.y ?? 0.5)) * h;
                const tx = clamp01(Number(tgt.x ?? 0.5)) * w;
                const ty = clamp01(Number(tgt.y ?? 0.5)) * h;
                const weight = clamp01(Number(edge.weight ?? 0.2));
                const kind = String(edge.kind ?? "");
                const hue = kind === "categorizes" ? 190 : 22;
                ctx.strokeStyle = `hsla(${hue}, 88%, 66%, ${0.04 + weight * 0.24})`;
                ctx.lineWidth = 0.35 + weight * 0.95;
                ctx.beginPath();
                const bend = Math.sin((t * 1.35) + (i * 0.12)) * 9;
                ctx.moveTo(sx, sy);
                ctx.quadraticCurveTo((sx + tx) / 2 - bend, (sy + ty) / 2 + bend * 0.5, tx, ty);
                ctx.stroke();
            }
            ctx.restore();
        }

        if (crawlerNodes.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            for (let i = 0; i < crawlerNodes.length; i++) {
                const node = crawlerNodes[i] as any;
                const nx = clamp01(Number(node.x ?? 0.5));
                const ny = clamp01(Number(node.y ?? 0.5));
                const px = nx * w;
                const py = ny * h;
                const importance = clamp01(Number(node.importance ?? 0.24));
                const pulse = 0.5 + Math.sin((t * 2.6) + i * 0.29) * 0.5;
                const resourceKind = classifyCrawlerResourceKind(node);
                const kind = String(node.crawler_kind ?? "url").toLowerCase();
                const fallbackHue = kind === "domain" ? 172 : (kind === "content" ? 26 : 205);
                const visual = resourceVisualSpec(resourceKind, fallbackHue);
                let radius = 1.2 + (importance * 2.5) + pulse * 0.8;
                const isSelected = selectedGraphNodeId !== "" && selectedGraphNodeId === String(node.id ?? "");
                if (isSelected) {
                    radius += 1.5;
                }
                const hue = visual.hue;
                const lift = (1.3 + importance * 2.2) * visual.liftBoost;
                const depthY = py + lift;

                ctx.fillStyle = `hsla(${hue}, ${Math.round(visual.saturation * 0.88)}%, ${Math.round(visual.value * 0.42)}%, ${0.12 + (isSelected ? 0.1 : 0.03)})`;
                fillResourceShape(ctx, visual.shape, px, depthY, radius * 1.02);

                const glow = ctx.createRadialGradient(px, py, 0, px, py, radius * 2.1);
                glow.addColorStop(0, `hsla(${hue}, ${visual.saturation}%, ${visual.value}%, ${isSelected ? 0.84 : 0.48})`);
                glow.addColorStop(0.65, `hsla(${(hue + 22) % 360}, ${Math.max(52, visual.saturation - 10)}%, ${Math.max(42, visual.value - 34)}%, ${isSelected ? 0.36 : 0.2})`);
                glow.addColorStop(1, "rgba(12, 18, 30, 0)");
                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(px, py, radius * 2.1, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = `hsla(${hue}, ${Math.min(96, visual.saturation + 4)}%, ${Math.min(98, visual.value + 1)}%, ${isSelected ? 0.98 : 0.88})`;
                fillResourceShape(ctx, visual.shape, px, py, radius);

                ctx.strokeStyle = `hsla(${hue}, ${Math.max(66, visual.saturation - 10)}%, ${Math.max(50, visual.value - 36)}%, ${isSelected ? 0.95 : 0.54})`;
                ctx.lineWidth = isSelected ? 1.2 : 0.85;
                strokeResourceShape(ctx, visual.shape, px, py, radius);

                graphNodeHits.push({
                    node,
                    x: nx,
                    y: ny,
                    radiusNorm: (radius + lift * 0.34) / Math.max(w, h),
                    nodeKind: "crawler",
                    resourceKind,
                });
            }
            ctx.restore();
        }

        const topFieldRows = Object.entries(graph.stats?.field_counts ?? {})
            .sort((a, b) => Number(b[1]) - Number(a[1]))
            .slice(0, 3)
            .map(([field, count]) => `${field}:${count}`)
            .join("  ");
        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "left";
        ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(157, 219, 255, 0.93)";
        ctx.fillText(
            `crawler nodes: ${Number(graph.stats?.crawler_count ?? 0)} | urls: ${Number(graph.stats?.url_nodes_total ?? 0)}`,
            10,
            40,
        );
        ctx.fillStyle = "rgba(142, 201, 246, 0.86)";
        ctx.fillText(`crawler fields: ${topFieldRows || "none"}`, 10, 51);
        ctx.restore();

        if (selectedGraphNodeId) {
            const selected = crawlerNodes.find((row: any) => String(row.id) === selectedGraphNodeId);
            if (!selected) {
                return;
            }
            const sx = clamp01(Number(selected.x ?? 0.5)) * w;
            const sy = clamp01(Number(selected.y ?? 0.5)) * h;
            const selectedResourceKind = classifyCrawlerResourceKind(selected);
            const label = shortPathLabel(
                String(
                    selected.title
                    || selected.domain
                    || selected.url
                    || selected.label
                    || selected.id
                    || "",
                ),
            );
            selectedGraphNodeLabel = label;
            const crawlerKind = String(selected.crawler_kind ?? "url");
            const domain = String(selected.domain ?? "");
            ctx.save();
            ctx.globalCompositeOperation = "source-over";
            ctx.fillStyle = "rgba(8, 16, 27, 0.86)";
            ctx.strokeStyle = "rgba(154, 207, 255, 0.74)";
            ctx.lineWidth = 1;
            const boxX = Math.min(w - 218, sx + 10);
            const boxY = Math.max(10, sy - 40);
            ctx.beginPath();
            ctx.roundRect(boxX, boxY, 208, 34, 5);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = "rgba(219, 239, 255, 0.96)";
            ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.fillText(label, boxX + 6, boxY + 11);
            ctx.fillStyle = "rgba(177, 224, 255, 0.9)";
            ctx.fillText(`${crawlerKind}${domain ? ` · ${shortPathLabel(domain)}` : ""}`, boxX + 6, boxY + 20);
            ctx.fillText(`resource ${resourceKindLabel(selectedResourceKind)} · field ${String(selected.dominant_field ?? "f2")}`, boxX + 6, boxY + 30);
            ctx.restore();
        }
    };

    const drawGhostSentinel = (t: number, w: number, h: number, state: SimulationState | null) => {
        const ghost = state?.presence_dynamics?.ghost;
        if (!ghost) return;

        const pulse = Math.max(0, Math.min(1, Number(ghost.auto_commit_pulse ?? 0)));
        const x = w * 0.88;
        const y = h * 0.14;
        const radius = 16 + pulse * 16 + Math.sin(t * 2.7) * 2;

        ctx.save();
        ctx.globalCompositeOperation = "screen";
        const aura = ctx.createRadialGradient(x, y, 2, x, y, radius * 1.8);
        aura.addColorStop(0, `rgba(208, 244, 255, ${0.56 + pulse * 0.28})`);
        aura.addColorStop(0.55, `rgba(96, 191, 255, ${0.24 + pulse * 0.16})`);
        aura.addColorStop(1, "rgba(32, 88, 120, 0)");
        ctx.fillStyle = aura;
        ctx.beginPath();
        ctx.arc(x, y, radius * 1.8, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = `rgba(186, 235, 255, ${0.64 + pulse * 0.24})`;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = "rgba(235, 248, 255, 0.95)";
        ctx.beginPath();
        ctx.arc(x, y, 3.6 + pulse * 1.8, 0, Math.PI * 2);
        ctx.fill();

        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "center";
        ctx.font = "600 9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(216, 238, 255, 0.95)";
        ctx.fillText(`${ghost.en} / ${ghost.ja}`, x, y + radius + 14);
        ctx.font = "500 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(166, 214, 255, 0.92)";
        ctx.fillText(`${ghost.status_ja} · pulse ${Math.round(pulse * 100)}%`, x, y + radius + 25);
        ctx.restore();
    };

    const drawGraphRuntimeDiagnostics = (
        w: number,
        h: number,
        state: SimulationState | null,
    ) => {
        const probabilistic = state?.presence_dynamics?.daimoi_probabilistic;
        const graphRuntime = probabilistic?.graph_runtime;
        if (!graphRuntime) {
            return;
        }

        const routeMean = clamp01(Number(probabilistic?.mean_route_probability ?? 0));
        const driftMean = clampValue(Number(probabilistic?.mean_drift_score ?? 0), -1, 1);
        const influenceMean = clamp01(Number(probabilistic?.mean_influence_power ?? 0));
        const edgeCostMean = Math.max(0, Number(graphRuntime?.edge_cost_mean ?? 0));
        const gravityMean = Math.max(0, Number(graphRuntime?.gravity_mean ?? 0));
        const priceMean = Math.max(0, Number(graphRuntime?.price_mean ?? 0));
        const nooiActiveCells = Math.max(0, Number(state?.presence_dynamics?.nooi_field?.active_cells ?? 0));
        const nodeCount = Math.max(0, Number(graphRuntime?.node_count ?? 0));
        const edgeCount = Math.max(0, Number(graphRuntime?.edge_count ?? 0));
        const topNode = Array.isArray(graphRuntime?.top_nodes)
            ? graphRuntime?.top_nodes?.[0]
            : undefined;
        const topNodeId = String((topNode as any)?.node_id ?? "").trim();

        const boxWidth = Math.min(332, Math.max(242, w * 0.31));
        const boxHeight = 44;
        const boxX = 10;
        const boxY = Math.max(8, h * 0.03);

        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = "rgba(8, 18, 30, 0.82)";
        ctx.strokeStyle = "rgba(122, 205, 255, 0.48)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 6);
        ctx.fill();
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(178, 232, 255, 0.95)";
        ctx.fillText(
            `graph route ${Math.round(routeMean * 100)}% · drift ${driftMean >= 0 ? "+" : ""}${driftMean.toFixed(2)} · infl ${influenceMean.toFixed(2)} · price ${priceMean.toFixed(2)}`,
            boxX + 8,
            boxY + 14,
        );

        ctx.fillStyle = "rgba(154, 214, 250, 0.9)";
        ctx.fillText(
            `cost ${edgeCostMean.toFixed(2)} · gravity ${gravityMean.toFixed(2)} · nooi ${nooiActiveCells} · nodes ${nodeCount} edges ${edgeCount}`,
            boxX + 8,
            boxY + 25,
        );

        if (topNodeId) {
            ctx.fillStyle = "rgba(126, 199, 246, 0.86)";
            ctx.fillText(`top gravity node: ${shortPathLabel(topNodeId)}`, boxX + 8, boxY + 36);
        }
        ctx.restore();
    };

    const drawMouseDaimon = (
        t: number,
        w: number,
        h: number,
        px: number,
        py: number,
        power: number,
        inside: boolean,
    ) => {
        const enabled = mouseDaimonEnabledRef.current;
        const message = mouseDaimonMessageRef.current;
        const mode = mouseDaimonModeRef.current;
        const radiusScale = mouseDaimonRadiusRef.current;
        const strength = mouseDaimonStrengthRef.current;

        if (!enabled || !inside || power < 0.02) {
            return;
        }

        const x = px * w;
        const y = py * h;
        const radius = Math.max(12, radiusScale * Math.min(w, h) * 0.5);
        const pulse = 0.5 + 0.5 * Math.sin(t * 3.2);

        ctx.save();

        // Mode-specific outer glow
        const modeColors: Record<string, { inner: string; outer: string; core: string }> = {
            push: { inner: "rgba(255, 180, 100, 0.72)", outer: "rgba(255, 120, 60, 0.28)", core: "#ffcc66" },
            pull: { inner: "rgba(100, 200, 255, 0.72)", outer: "rgba(60, 140, 255, 0.28)", core: "#88ddff" },
            orbit: { inner: "rgba(200, 150, 255, 0.72)", outer: "rgba(140, 100, 255, 0.28)", core: "#ccbbff" },
            calm: { inner: "rgba(150, 255, 180, 0.72)", outer: "rgba(100, 220, 140, 0.28)", core: "#aaffcc" },
        };
        const colors = modeColors[mode] || modeColors.push;

        // Influence zone gradient
        const influenceRadius = radius * (1.8 + strength * 1.2);
        const zoneGradient = ctx.createRadialGradient(x, y, 0, x, y, influenceRadius);
        zoneGradient.addColorStop(0, colors.inner);
        zoneGradient.addColorStop(0.4, colors.outer);
        zoneGradient.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.globalCompositeOperation = "screen";
        ctx.fillStyle = zoneGradient;
        ctx.beginPath();
        ctx.arc(x, y, influenceRadius, 0, Math.PI * 2);
        ctx.fill();

        // Core particle body
        ctx.globalCompositeOperation = "source-over";
        const coreGradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 0.6);
        coreGradient.addColorStop(0, "rgba(255, 255, 255, 0.95)");
        coreGradient.addColorStop(0.5, colors.inner);
        coreGradient.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = coreGradient;
        ctx.beginPath();
        ctx.arc(x, y, radius * 0.6 * (1 + pulse * 0.15), 0, Math.PI * 2);
        ctx.fill();

        // Orbiting ring for orbit mode
        if (mode === "orbit") {
            ctx.strokeStyle = `rgba(200, 180, 255, ${0.4 + pulse * 0.3})`;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 6]);
            ctx.lineDashOffset = -t * 28;
            ctx.beginPath();
            ctx.arc(x, y, radius * (1.2 + pulse * 0.2), 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Pulsing ring
        ctx.strokeStyle = colors.core;
        ctx.lineWidth = 1.2 + pulse * 0.6;
        ctx.globalAlpha = 0.5 + pulse * 0.4;
        ctx.beginPath();
        ctx.arc(x, y, radius * (0.9 + pulse * 0.25), 0, Math.PI * 2);
        ctx.stroke();

        // Mode indicator arrows
        ctx.globalAlpha = 0.7 + pulse * 0.3;
        ctx.strokeStyle = colors.core;
        ctx.lineWidth = 1.5;
        const arrowCount = mode === "calm" ? 4 : 6;
        for (let i = 0; i < arrowCount; i++) {
            const angle = (i / arrowCount) * Math.PI * 2 + t * (mode === "orbit" ? 1.2 : 0.3);
            const r1 = radius * 1.1;
            const r2 = radius * 1.4;
            const ax1 = x + Math.cos(angle) * r1;
            const ay1 = y + Math.sin(angle) * r1;
            const ax2 = x + Math.cos(angle) * r2;
            const ay2 = y + Math.sin(angle) * r2;

            if (mode === "push") {
                // Outward arrows
                ctx.beginPath();
                ctx.moveTo(ax1, ay1);
                ctx.lineTo(ax2, ay2);
                ctx.stroke();
                const headAngle = angle;
                const headLen = 4;
                ctx.beginPath();
                ctx.moveTo(ax2, ay2);
                ctx.lineTo(ax2 - Math.cos(headAngle - 0.4) * headLen, ay2 - Math.sin(headAngle - 0.4) * headLen);
                ctx.moveTo(ax2, ay2);
                ctx.lineTo(ax2 - Math.cos(headAngle + 0.4) * headLen, ay2 - Math.sin(headAngle + 0.4) * headLen);
                ctx.stroke();
            } else if (mode === "pull") {
                // Inward arrows
                ctx.beginPath();
                ctx.moveTo(ax2, ay2);
                ctx.lineTo(ax1, ay1);
                ctx.stroke();
                const headAngle = angle + Math.PI;
                const headLen = 4;
                ctx.beginPath();
                ctx.moveTo(ax1, ay1);
                ctx.lineTo(ax1 - Math.cos(headAngle - 0.4) * headLen, ay1 - Math.sin(headAngle - 0.4) * headLen);
                ctx.moveTo(ax1, ay1);
                ctx.lineTo(ax1 - Math.cos(headAngle + 0.4) * headLen, ay1 - Math.sin(headAngle + 0.4) * headLen);
                ctx.stroke();
            } else if (mode === "orbit") {
                // Tangent arrows (curved)
                const curveR = radius * 1.25;
                const curveStart = angle - 0.3;
                const curveEnd = angle + 0.3;
                ctx.beginPath();
                ctx.arc(x, y, curveR, curveStart, curveEnd);
                ctx.stroke();
            } else {
                // Calm: gentle dots
                ctx.beginPath();
                ctx.arc(ax2, ay2, 2 + pulse, 0, Math.PI * 2);
                ctx.fillStyle = colors.core;
                ctx.fill();
            }
        }

        // Message label
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "center";
        ctx.font = "600 9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
        ctx.shadowColor = "rgba(0, 0, 0, 0.6)";
        ctx.shadowBlur = 4;
        ctx.fillText(message, x, y + radius + 16);
        ctx.shadowBlur = 0;

        // Mode label
        ctx.font = "500 7px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = colors.core;
        ctx.fillText(`${mode} · r${(radiusScale * 100).toFixed(0)}% · s${(strength * 100).toFixed(0)}%`, x, y + radius + 28);

        ctx.restore();
    };

    const drawGraphDaimoiFlowOverlay = (
        t: number,
        w: number,
        h: number,
        namedForms: Array<any>,
        state: SimulationState | null,
    ) => {
        const fieldRows = resolveFieldParticleRows(state);
        if (fieldRows.length <= 0) {
            return;
        }

        const anchorByPresence = new Map<string, { x: number; y: number; hue: number }>();
        for (const form of namedForms) {
            const presenceId = canonicalPresenceId(String((form as any)?.id ?? ""));
            if (!presenceId) {
                continue;
            }
            const anchor = {
                x: clamp01(Number((form as any)?.x ?? 0.5)) * w,
                y: clamp01(Number((form as any)?.y ?? 0.5)) * h,
                hue: Number((form as any)?.hue ?? presenceHueFromId(presenceId)),
            };
            anchorByPresence.set(presenceId, anchor);
            anchorByPresence.set(normalizePresenceKey(presenceId), anchor);
        }

        const flowBuckets = new Map<
            string,
            {
                sourceId: string;
                targetId: string;
                weight: number;
                samples: number;
                resourceWeight: number;
            }
        >();

        for (const row of fieldRows) {
            if (!row || typeof row !== "object") {
                continue;
            }
            const sourcePresenceId = canonicalPresenceId(resolveParticlePresenceId(row));
            if (!sourcePresenceId) {
                continue;
            }

            const routeNodeId = String((row as any)?.route_node_id ?? "").trim();
            const routePresenceId = routeNodeId.startsWith("field:")
                ? canonicalPresenceId(routeNodeId.slice("field:".length))
                : "";
            const resourceTargetPresenceId = (row as any)?.resource_daimoi
                ? canonicalPresenceId(String((row as any)?.resource_target_presence_id ?? ""))
                : "";
            const targetPresenceId = routePresenceId || resourceTargetPresenceId;
            if (!targetPresenceId || targetPresenceId === sourcePresenceId) {
                continue;
            }

            const sourceAnchor = anchorByPresence.get(sourcePresenceId)
                ?? anchorByPresence.get(normalizePresenceKey(sourcePresenceId));
            const targetAnchor = anchorByPresence.get(targetPresenceId)
                ?? anchorByPresence.get(normalizePresenceKey(targetPresenceId));
            if (!sourceAnchor || !targetAnchor) {
                continue;
            }

            const routeProbability = clamp01(Number((row as any)?.route_probability ?? 0));
            const influencePower = clamp01(Number((row as any)?.influence_power ?? 0));
            const driftSignal = Math.abs(clampValue(Number((row as any)?.drift_score ?? 0), -1, 1));
            const resourceEmitNorm = (row as any)?.resource_daimoi
                ? clamp01(Number((row as any)?.resource_emit_amount ?? 0) * 24)
                : 0;
            const flowWeight =
                0.14
                + routeProbability * 0.56
                + influencePower * 0.34
                + driftSignal * 0.24
                + resourceEmitNorm * 0.78;
            const key = `${sourcePresenceId}|${targetPresenceId}`;
            const bucket = flowBuckets.get(key) ?? {
                sourceId: sourcePresenceId,
                targetId: targetPresenceId,
                weight: 0,
                samples: 0,
                resourceWeight: 0,
            };
            bucket.weight += flowWeight;
            bucket.samples += 1;
            bucket.resourceWeight += resourceEmitNorm;
            flowBuckets.set(key, bucket);
        }

        const flowRows = Array.from(flowBuckets.values())
            .sort((left, right) => right.weight - left.weight)
            .slice(0, 92);
        if (flowRows.length <= 0) {
            return;
        }

        const maxWeight = Math.max(
            0.0001,
            ...flowRows.map((row) => row.weight),
        );

        ctx.save();
        ctx.globalCompositeOperation = "screen";
        ctx.lineCap = "round";
        for (let index = 0; index < flowRows.length; index += 1) {
            const flow = flowRows[index];
            const sourceAnchor = anchorByPresence.get(flow.sourceId)
                ?? anchorByPresence.get(normalizePresenceKey(flow.sourceId));
            const targetAnchor = anchorByPresence.get(flow.targetId)
                ?? anchorByPresence.get(normalizePresenceKey(flow.targetId));
            if (!sourceAnchor || !targetAnchor) {
                continue;
            }

            const sx = sourceAnchor.x;
            const sy = sourceAnchor.y;
            const tx = targetAnchor.x;
            const ty = targetAnchor.y;
            const dx = tx - sx;
            const dy = ty - sy;
            const distance = Math.hypot(dx, dy);
            if (distance <= 0.0001) {
                continue;
            }

            const unitX = dx / distance;
            const unitY = dy / distance;
            const normalX = -unitY;
            const normalY = unitX;
            const flowNorm = clamp01(flow.weight / maxWeight);
            const bend = (8 + flowNorm * 20 + Math.sin(t * 0.9 + index * 0.23) * 6)
                * (index % 2 === 0 ? 1 : -1);
            const controlX = (sx + tx) * 0.5 + normalX * bend;
            const controlY = (sy + ty) * 0.5 + normalY * bend;
            const glowAlpha = clamp01(0.06 + flowNorm * 0.24);
            const sourceHue = Number(sourceAnchor.hue ?? 198);
            const targetHue = Number(targetAnchor.hue ?? 198);
            const gradient = ctx.createLinearGradient(sx, sy, tx, ty);
            gradient.addColorStop(0, `hsla(${sourceHue}, 90%, 72%, ${glowAlpha})`);
            gradient.addColorStop(1, `hsla(${targetHue}, 90%, 74%, ${glowAlpha})`);

            ctx.strokeStyle = gradient;
            ctx.lineWidth = 0.5 + flowNorm * (1.8 + flow.resourceWeight * 0.18);
            ctx.setLineDash([
                6 + flowNorm * 10,
                8 + (1 - flowNorm) * 8,
            ]);
            ctx.lineDashOffset = -((t * (26 + flowNorm * 32)) + index * 5);
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.quadraticCurveTo(controlX, controlY, tx, ty);
            ctx.stroke();

            const packetProgress = ((t * (0.18 + flowNorm * 0.24)) + index * 0.13) % 1;
            const inv = 1 - packetProgress;
            const packetX = (inv * inv * sx) + (2 * inv * packetProgress * controlX) + (packetProgress * packetProgress * tx);
            const packetY = (inv * inv * sy) + (2 * inv * packetProgress * controlY) + (packetProgress * packetProgress * ty);
            ctx.fillStyle = `hsla(${Math.round((sourceHue + targetHue) * 0.5)}, 96%, 82%, ${0.2 + flowNorm * 0.56})`;
            ctx.beginPath();
            ctx.arc(packetX, packetY, 1 + flowNorm * 2.3, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.setLineDash([]);
        ctx.restore();

        const topFlow = flowRows[0];
        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "left";
        ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(176, 233, 255, 0.94)";
        ctx.fillText(
            `graph daimoi lanes ${flowRows.length} · samples ${flowRows.reduce((sum, row) => sum + row.samples, 0)}`,
            10,
            h - 70,
        );
        ctx.fillStyle = "rgba(146, 213, 247, 0.88)";
        ctx.fillText(
            `top lane ${shortPresenceIdLabel(topFlow.sourceId)} -> ${shortPresenceIdLabel(topFlow.targetId)} · ${topFlow.weight.toFixed(1)}`,
            10,
            h - 58,
        );
        ctx.restore();
    };

    const drawResourceDaimoiOverlay = (
        t: number,
        w: number,
        h: number,
        namedForms: Array<any>,
        state: SimulationState | null,
    ) => {
        const dynamics = state?.presence_dynamics;
        if (!dynamics) {
            return;
        }
        const summary = dynamics.resource_daimoi ?? dynamics.daimoi_probabilistic?.resource_daimoi;
        const deliveredPackets = Math.max(0, Number(summary?.delivered_packets ?? 0));
        if (deliveredPackets <= 0) {
            return;
        }

        const directRows = Array.isArray(dynamics.field_particles) ? dynamics.field_particles : [];
        const fieldRows = directRows.length > 0 ? directRows : resolveFieldParticleRows(state);
        if (fieldRows.length <= 0) {
            return;
        }

        const anchorByPresence = new Map<string, { x: number; y: number }>();
        for (const form of namedForms) {
            const id = String((form as any)?.id ?? "").trim();
            if (!id) {
                continue;
            }
            const anchor = {
                x: clamp01(Number((form as any)?.x ?? 0.5)) * w,
                y: clamp01(Number((form as any)?.y ?? 0.5)) * h,
            };
            anchorByPresence.set(id, anchor);
            anchorByPresence.set(normalizePresenceKey(id), anchor);
        }

        const recipientTotals = new Map<string, number>();
        let rendered = 0;

        ctx.save();
        ctx.globalCompositeOperation = "screen";
        for (let index = 0; index < fieldRows.length; index += 1) {
            const row = fieldRows[index] as any;
            if (!row || !row.resource_daimoi) {
                continue;
            }
            const targetPresenceId = String(row.resource_target_presence_id ?? "").trim();
            if (!targetPresenceId) {
                continue;
            }

            const targetAnchor = anchorByPresence.get(targetPresenceId)
                ?? anchorByPresence.get(normalizePresenceKey(targetPresenceId));
            if (!targetAnchor) {
                continue;
            }

            const startX = clamp01(Number(row.x ?? 0.5)) * w;
            const startY = clamp01(Number(row.y ?? 0.5)) * h;
            const emitAmount = Math.max(0, Number(row.resource_emit_amount ?? 0));
            const emitNorm = clamp01(emitAmount * 22);
            const resourceHue = resourceDaimoiHue(String(row.resource_type ?? ""));
            const bend = Math.sin((t * 1.6) + (index * 0.24)) * (6 + emitNorm * 14);
            const midX = (startX + targetAnchor.x) * 0.5 + bend;
            const midY = (startY + targetAnchor.y) * 0.5 - bend * 0.55;

            ctx.strokeStyle = `hsla(${resourceHue}, 95%, 72%, ${0.2 + emitNorm * 0.55})`;
            ctx.lineWidth = 0.8 + emitNorm * 2.0;
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.quadraticCurveTo(midX, midY, targetAnchor.x, targetAnchor.y);
            ctx.stroke();

            const pulseRadius = 1.6 + (emitNorm * 3.0);
            ctx.fillStyle = `hsla(${resourceHue}, 98%, 80%, ${0.22 + emitNorm * 0.56})`;
            ctx.beginPath();
            ctx.arc(targetAnchor.x, targetAnchor.y, pulseRadius, 0, Math.PI * 2);
            ctx.fill();

            recipientTotals.set(
                targetPresenceId,
                (recipientTotals.get(targetPresenceId) ?? 0) + emitAmount,
            );

            rendered += 1;
            if (rendered >= 260) {
                break;
            }
        }
        ctx.restore();

        if (rendered <= 0) {
            return;
        }

        const topRecipient = Array.from(recipientTotals.entries())
            .sort((left, right) => right[1] - left[1])[0];
        const transferTotal = Math.max(0, Number(summary?.total_transfer ?? 0));

        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "left";
        ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(255, 238, 198, 0.96)";
        ctx.fillText(
            `resource daimoi packets ${rendered}/${deliveredPackets} · transfer ${transferTotal.toFixed(2)}`,
            10,
            h - 34,
        );
        if (topRecipient) {
            ctx.fillStyle = "rgba(255, 215, 158, 0.92)";
            ctx.fillText(
                `resource target ${shortPresenceIdLabel(topRecipient[0])} · +${topRecipient[1].toFixed(2)}`,
                10,
                h - 46,
            );
        }
        ctx.restore();
    };

    const drawNooiFieldOverlay = (
        t: number,
        w: number,
        h: number,
        state: SimulationState | null,
        daimoiLayerActive: boolean,
    ) => {
        const nooiField = state?.presence_dynamics?.nooi_field;
        const cells = Array.isArray(nooiField?.cells) ? nooiField.cells : [];
        if (cells.length <= 0) {
            return;
        }

        const cols = Math.max(1, Number(nooiField?.grid_cols ?? 1));
        const rows = Math.max(1, Number(nooiField?.grid_rows ?? 1));
        const cellW = w / cols;
        const cellH = h / rows;
        const vectorPeak = Math.max(0.0001, Number(nooiField?.vector_peak ?? 0.001));
        const pulse = 0.86 + Math.sin(t * 1.7) * 0.14;
        const grainGain = daimoiLayerActive ? 1 : 0.64;

        ctx.save();
        ctx.globalCompositeOperation = "screen";
        for (let index = 0; index < cells.length; index += 1) {
            const cell = cells[index] as any;
            const intensity = clamp01(Number(cell?.intensity ?? 0));
            if (intensity <= 0.01) {
                continue;
            }
            const influence = clamp01(Number(cell?.influence ?? intensity));
            const occupancyRatio = clamp01(Number(cell?.occupancy_ratio ?? 0));
            const cx = clamp01(Number(cell?.x ?? 0.5)) * w;
            const cy = clamp01(Number(cell?.y ?? 0.5)) * h;
            const vx = Number(cell?.vx ?? 0);
            const vy = Number(cell?.vy ?? 0);
            const vectorMagnitude = Math.hypot(vx, vy);
            const vectorNorm = clamp01(vectorMagnitude / vectorPeak);
            const dominant = String(cell?.dominant_presence_id ?? "").trim();
            const dominantHue = dominant.length > 0 ? presenceHueFromId(dominant) : 198;
            const heatRadius = Math.min(cellW, cellH) * (0.28 + influence * 1.46 + occupancyRatio * 0.52);
            const heat = ctx.createRadialGradient(cx, cy, 0, cx, cy, heatRadius * 2.2);
            heat.addColorStop(0, `hsla(${dominantHue}, 84%, 68%, ${(0.06 + intensity * 0.2 + occupancyRatio * 0.12) * pulse})`);
            heat.addColorStop(0.58, `hsla(${(dominantHue + 26) % 360}, 76%, 56%, ${(0.03 + influence * 0.14) * pulse})`);
            heat.addColorStop(1, "rgba(16, 22, 34, 0)");
            ctx.fillStyle = heat;
            ctx.beginPath();
            ctx.ellipse(cx, cy, heatRadius * 2.2, heatRadius * 1.5, 0, 0, Math.PI * 2);
            ctx.fill();

            if (vectorMagnitude > 1e-7) {
                const dirX = vx / vectorMagnitude;
                const dirY = vy / vectorMagnitude;
                const normalX = -dirY;
                const normalY = dirX;
                const streakCount = Math.max(
                    1,
                    Math.round(
                        (daimoiLayerActive ? 2 : 1)
                        + (vectorNorm + influence + occupancyRatio) * (daimoiLayerActive ? 3 : 1.5),
                    ),
                );
                const segmentLength = Math.min(cellW, cellH)
                    * (0.2 + vectorNorm * 0.84 + influence * 0.42)
                    * grainGain;
                for (let streak = 0; streak < streakCount; streak += 1) {
                    const lane = streakCount <= 1 ? 0 : (streak / (streakCount - 1)) - 0.5;
                    const sideOffset = lane * Math.min(cellW, cellH) * (0.58 + influence * 0.66);
                    const sway = Math.sin((t * 1.9) + index * 0.23 + streak * 0.71)
                        * (0.4 + vectorNorm * 1.8);
                    const originX = cx + normalX * sideOffset + dirX * sway;
                    const originY = cy + normalY * sideOffset + dirY * sway;
                    const startX = originX - dirX * segmentLength * 0.48;
                    const startY = originY - dirY * segmentLength * 0.48;
                    const endX = originX + dirX * segmentLength * 0.52;
                    const endY = originY + dirY * segmentLength * 0.52;

                    ctx.strokeStyle = `hsla(${dominantHue}, 88%, 78%, ${(0.06 + influence * 0.26 + vectorNorm * 0.18) * pulse})`;
                    ctx.lineWidth = 0.32 + (influence * 1.05 + vectorNorm * 0.72) * grainGain;
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(endX, endY);
                    ctx.stroke();
                }
            }
        }

        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "left";
        ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(176, 229, 255, 0.95)";
        ctx.fillText(
            `nooi wind ${cells.length} cells · influence ${Number(nooiField?.mean_influence ?? 0).toFixed(2)} · vpeak ${Number(nooiField?.vector_peak ?? 0).toFixed(3)} · daimoi ${daimoiLayerActive ? "linked" : "passive"}`,
            10,
            h - 10,
        );
        ctx.restore();
    };

    const drawTruthBindingOverlay = (
        t: number,
        w: number,
        h: number,
        namedForms: Array<any>,
        state: SimulationState | null,
    ) => {
        const truth = resolveTruthState(state);
        if (!truth) {
            return;
        }

        const gateAnchor = namedForms.find((item: any) => item.id === "gates_of_truth");
        const anchorX = (Number(gateAnchor?.x ?? 0.76)) * w;
        const anchorY = (Number(gateAnchor?.y ?? 0.54)) * h;
        const claim = truth.claim;
        const claims = Array.isArray(truth.claims) ? truth.claims : [];
        const guardPasses = Boolean(truth.guard?.passes);
        const gateBlocked = Boolean(truth.gate?.blocked);
        const kappa = clamp01(Number(claim?.kappa ?? 0));
        const claimStatus = String(claim?.status ?? "undecided");
        const statusHue = claimStatus === "proved" ? 144 : claimStatus === "refuted" ? 14 : 52;
        const ringHue = guardPasses ? 154 : (gateBlocked ? 18 : statusHue);
        const ringAlpha = guardPasses ? 0.84 : (gateBlocked ? 0.62 : 0.7);

        ctx.save();
        ctx.globalCompositeOperation = "screen";
        ctx.strokeStyle = `hsla(${ringHue}, 92%, 67%, ${ringAlpha})`;
        ctx.lineWidth = 1.8 + kappa * 2;
        ctx.setLineDash([10, 8]);
        ctx.lineDashOffset = -(t * 40);
        ctx.beginPath();
        ctx.arc(anchorX, anchorY, 18 + (kappa * 22), 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = `hsla(${ringHue}, 82%, 74%, ${0.24 + kappa * 0.32})`;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 7]);
        ctx.lineDashOffset = t * 30;
        ctx.beginPath();
        ctx.arc(anchorX, anchorY, 34 + (kappa * 24), 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        for (let i = 0; i < Math.min(4, claims.length); i++) {
            const row = claims[i] as any;
            const claimKappa = clamp01(Number(row?.kappa ?? 0));
            const rowStatus = String(row?.status ?? "undecided");
            const rowHue = rowStatus === "proved" ? 146 : (rowStatus === "refuted" ? 10 : 48);
            const orbit = 20 + (i * 7) + (claimKappa * 9);
            const angle = (t * (0.9 + (i * 0.18))) + (i * 1.2);
            const px = anchorX + Math.cos(angle) * orbit;
            const py = anchorY + Math.sin(angle) * orbit;
            ctx.fillStyle = `hsla(${rowHue}, 95%, 76%, ${0.7 + claimKappa * 0.25})`;
            ctx.beginPath();
            ctx.arc(px, py, 1.6 + claimKappa * 2.2, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();

        const reason = String((truth.gate?.reasons ?? [])[0] ?? "");
        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "right";
        ctx.font = "600 9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = guardPasses ? "rgba(172, 246, 200, 0.95)" : "rgba(255, 212, 160, 0.95)";
        ctx.fillText(
            `Truth gate: ${claimStatus} κ=${kappa.toFixed(2)} θ=${Number(truth.guard?.theta ?? 0).toFixed(2)}`,
            w - 10,
            64,
        );
        ctx.font = "500 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = gateBlocked ? "rgba(255, 181, 145, 0.92)" : "rgba(184, 228, 255, 0.9)";
        ctx.fillText(
            gateBlocked
                ? `gate blocked: ${reason || "needs proof"}`
                : `gate ready: ${String(truth.gate?.target ?? "push-truth")}`,
            w - 10,
            75,
        );
        ctx.restore();
    };

    const drawPainFieldOverlay = (
        t: number,
        w: number,
        h: number,
        state: SimulationState | null,
    ) => {
        const painField = resolvePainField(state);
        if (!painField) {
            return;
        }
        const nodeHeat = Array.isArray(painField.node_heat) ? painField.node_heat : [];
        const failures = Array.isArray(painField.failing_tests) ? painField.failing_tests : [];
        if (nodeHeat.length <= 0) {
            return;
        }

        ctx.save();
        ctx.globalCompositeOperation = "screen";
        const pulse = 0.84 + Math.sin(t * 1.8) * 0.16;
        for (let i = 0; i < nodeHeat.length; i++) {
            const heatRow = nodeHeat[i] as any;
            const heat = clamp01(Number(heatRow?.heat ?? 0));
            if (heat <= 0) {
                continue;
            }
            const x = clamp01(Number(heatRow?.x ?? 0.5)) * w;
            const y = clamp01(Number(heatRow?.y ?? 0.5)) * h;
            const radius = (30 + (heat * 90)) * pulse;
            const glow = ctx.createRadialGradient(x, y, radius * 0.08, x, y, radius);
            glow.addColorStop(0, `rgba(255, 86, 72, ${0.14 + heat * 0.4})`);
            glow.addColorStop(0.45, `rgba(255, 42, 18, ${0.08 + heat * 0.24})`);
            glow.addColorStop(1, "rgba(80, 10, 8, 0)");
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();

            if (heat > 0.55) {
                ctx.strokeStyle = `rgba(255, 132, 108, ${0.18 + heat * 0.42})`;
                ctx.lineWidth = 0.8 + heat * 1.6;
                ctx.setLineDash([8, 7]);
                ctx.lineDashOffset = -((t * 40) + i * 8);
                ctx.beginPath();
                ctx.arc(x, y, radius * 0.34, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }

        const debugTarget = (painField as any)?.debug as any;
        if (debugTarget && Boolean(debugTarget.grounded)) {
            const dx = clamp01(Number(debugTarget.x ?? 0.5)) * w;
            const dy = clamp01(Number(debugTarget.y ?? 0.5)) * h;
            const debugHeat = clamp01(Number(debugTarget.heat ?? 0));
            const markerRadius = 10 + (debugHeat * 14);
            ctx.strokeStyle = `rgba(255, 235, 190, ${0.6 + debugHeat * 0.3})`;
            ctx.lineWidth = 1.2 + debugHeat * 1.4;
            ctx.setLineDash([6, 4]);
            ctx.lineDashOffset = -(t * 30);
            ctx.beginPath();
            ctx.arc(dx, dy, markerRadius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.beginPath();
            ctx.moveTo(dx - markerRadius * 0.6, dy);
            ctx.lineTo(dx + markerRadius * 0.6, dy);
            ctx.moveTo(dx, dy - markerRadius * 0.6);
            ctx.lineTo(dx, dy + markerRadius * 0.6);
            ctx.stroke();
        }
        ctx.restore();

        const maxHeat = clamp01(Number(painField.max_heat ?? 0));
        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "left";
        ctx.font = "600 9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = failures.length > 0
            ? "rgba(255, 176, 158, 0.96)"
            : "rgba(206, 228, 243, 0.82)";
        ctx.fillText(
            `pain field: ${failures.length} failing tests | max heat ${maxHeat.toFixed(2)}`,
            10,
            h - 30,
        );
        if (failures.length > 0) {
            const topFailure = String(failures[0]?.name ?? "").trim();
            if (topFailure) {
                ctx.font = "500 8px ui-monospace, SFMono-Regular, Menlo, monospace";
                ctx.fillStyle = "rgba(255, 194, 180, 0.9)";
                ctx.fillText(`top failing: ${shortPathLabel(topFailure)}`, 10, h - 19);
            }
        }

        const debugLabelTarget = (painField as any)?.debug as any;
        const debugPath = String(debugLabelTarget?.path ?? debugLabelTarget?.label ?? "").trim();
        if (Boolean(debugLabelTarget?.grounded) && debugPath) {
            ctx.font = "500 8px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.fillStyle = "rgba(255, 221, 168, 0.94)";
            ctx.fillText(`DEBUG -> ${shortPathLabel(debugPath)}`, 10, h - 8);
        }
        ctx.restore();
    };

    const drawLogicalGraphOverlay = (
        t: number,
        w: number,
        h: number,
        state: SimulationState | null,
    ) => {
        const graph = resolveLogicalGraph(state);
        if (!graph) {
            return;
        }
        const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
        const edges = Array.isArray(graph.edges) ? graph.edges : [];
        if (nodes.length <= 0) {
            return;
        }

        const nodeById = new Map(nodes.map((node: any) => [String(node.id), node]));
        const kindColor = (kind: string) => {
            if (kind === "file") return "rgba(184, 230, 255, 0.6)";
            if (kind === "fact") return "rgba(255, 226, 132, 0.66)";
            if (kind === "rule") return "rgba(156, 255, 207, 0.64)";
            if (kind === "derivation") return "rgba(161, 210, 255, 0.64)";
            if (kind === "gate") return "rgba(184, 255, 192, 0.76)";
            if (kind === "contradiction") return "rgba(255, 145, 134, 0.8)";
            return "rgba(170, 198, 228, 0.55)";
        };

        if (edges.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            for (let i = 0; i < edges.length; i++) {
                const edge = edges[i] as any;
                const source = nodeById.get(String(edge.source));
                const target = nodeById.get(String(edge.target));
                if (!source || !target) {
                    continue;
                }
                const sx = clamp01(Number(source.x ?? 0.5)) * w;
                const sy = clamp01(Number(source.y ?? 0.5)) * h;
                const tx = clamp01(Number(target.x ?? 0.5)) * w;
                const ty = clamp01(Number(target.y ?? 0.5)) * h;
                const weight = clamp01(Number(edge.weight ?? 0.4));
                const kind = String(edge.kind ?? "");
                const color = kind === "blocks"
                    ? "rgba(255, 128, 112, 0.34)"
                    : kind === "proves"
                        ? "rgba(168, 226, 255, 0.3)"
                        : "rgba(176, 206, 235, 0.21)";
                const bend = Math.sin((t * 1.1) + i * 0.3) * (6 + weight * 9);
                ctx.strokeStyle = color;
                ctx.lineWidth = 0.35 + weight * 1.05;
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.quadraticCurveTo((sx + tx) / 2 + bend, (sy + ty) / 2 - bend * 0.35, tx, ty);
                ctx.stroke();
            }
            ctx.restore();
        }

        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        for (let i = 0; i < nodes.length; i++) {
            const node = nodes[i] as any;
            const kind = String(node.kind ?? "");
            const x = clamp01(Number(node.x ?? 0.5)) * w;
            const y = clamp01(Number(node.y ?? 0.5)) * h;
            const confidence = clamp01(Number(node.confidence ?? 0.6));
            const pulse = 0.85 + Math.sin((t * 2.1) + i * 0.41) * 0.15;
            const radius = 1.6 + confidence * 2.6 + (kind === "gate" ? 1.8 : 0);

            if (kind === "gate") {
                ctx.strokeStyle = kindColor(kind);
                ctx.lineWidth = 1.2;
                ctx.setLineDash([7, 6]);
                ctx.lineDashOffset = -(t * 35);
                ctx.beginPath();
                ctx.arc(x, y, radius * 2.2 * pulse, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);
            }

            if (kind === "contradiction") {
                ctx.fillStyle = kindColor(kind);
                ctx.beginPath();
                ctx.moveTo(x, y - radius * 1.35);
                ctx.lineTo(x + radius * 1.15, y + radius * 1.1);
                ctx.lineTo(x - radius * 1.15, y + radius * 1.1);
                ctx.closePath();
                ctx.fill();
                continue;
            }

            ctx.fillStyle = kindColor(kind);
            ctx.beginPath();
            ctx.arc(x, y, radius * pulse, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();

        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.textAlign = "right";
        ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "rgba(188, 219, 246, 0.9)";
        ctx.fillText(
            `logical graph nodes:${nodes.length} edges:${edges.length}`,
            w - 10,
            h - 16,
        );
        ctx.restore();
    };

    const draw = (ts: number) => {
        const currentSimulation = simulationRef.current;
        const allFieldParticleRows = resolveFieldParticleRows(currentSimulation);
        const fieldParticleCount = allFieldParticleRows.length;
        const targetFrameMs = fieldParticleCount > 1200
            ? 34
            : fieldParticleCount > 760
                ? 24
                : 16;
        if (lastPaintTs > 0 && (ts - lastPaintTs) < targetFrameMs) {
            rafId = requestAnimationFrame(draw);
            return;
        }
        lastPaintTs = ts;
        const namedForms = resolveNamedForms();
        const t = ts * 0.001;
        overlayMotionNowSec = t;
        overlayMotionDtSec = overlayMotionLastFrameTs > 0
            ? clampValue((ts - overlayMotionLastFrameTs) * 0.001, 1 / 144, 0.08)
            : (1 / 60);
        overlayMotionLastFrameTs = ts;
        graphNodeHits = [];
        particleTelemetryHits = [];
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
        const w = canvasWidth;
        const h = canvasHeight;
        ctx.clearRect(0, 0, w, h);
        const washValue = backgroundMode
            ? Math.min(0.82, Math.max(0.24, backgroundWash))
            : 0.46;
        ctx.fillStyle = `rgba(4, 10, 18, ${washValue})`;
        ctx.fillRect(0, 0, w, h);
        const vignette = ctx.createRadialGradient(w * 0.5, h * 0.48, Math.min(w, h) * 0.18, w * 0.5, h * 0.52, Math.max(w, h) * 0.78);
        const centerStop = backgroundMode
            ? Math.min(0.2, Math.max(0.05, (washValue - 0.24) * 0.42 + 0.08))
            : 0.08;
        const midStop = backgroundMode
            ? Math.min(0.46, Math.max(0.16, washValue - 0.1))
            : 0.28;
        const edgeStop = backgroundMode
            ? Math.min(0.76, Math.max(0.4, washValue + 0.12))
            : 0.62;
        vignette.addColorStop(0, `rgba(16, 32, 52, ${centerStop})`);
        vignette.addColorStop(0.5, `rgba(4, 10, 18, ${midStop})`);
        vignette.addColorStop(1, `rgba(2, 6, 12, ${edgeStop})`);
        ctx.fillStyle = vignette;
        ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = "screen";
        const audioCount = Math.max(0, currentSimulation?.audio || 0);
        const globalIntensity = Math.min(0.68, Math.log1p(audioCount) / 7.2);
        const pointerInfluence = mouseInfluenceRef.current;
        const pointerTargetPower = pointerField.inside
            ? (0.14 + pointerInfluence * 0.34)
            : 0;
        pointerField.power += (pointerTargetPower - pointerField.power) * 0.12;
        const pointerPower = Math.max(0, pointerField.power);
        const pointerX = pointerField.x * w;
        const pointerY = pointerField.y * h;

        if (pointerPower > 0.01) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            const influenceGlow = ctx.createRadialGradient(pointerX, pointerY, 0, pointerX, pointerY, Math.max(w, h) * (0.08 + pointerPower * 0.12));
            influenceGlow.addColorStop(0, `rgba(182, 238, 255, ${0.12 + pointerPower * 0.16})`);
            influenceGlow.addColorStop(0.45, `rgba(92, 188, 255, ${0.06 + pointerPower * 0.12})`);
            influenceGlow.addColorStop(1, "rgba(8, 16, 26, 0)");
            ctx.fillStyle = influenceGlow;
            ctx.beginPath();
            ctx.arc(pointerX, pointerY, Math.max(w, h) * (0.08 + pointerPower * 0.12), 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = `rgba(162, 226, 255, ${0.18 + pointerPower * 0.28})`;
            ctx.lineWidth = 0.8 + pointerPower * 1.4;
            ctx.setLineDash([8, 9]);
            ctx.lineDashOffset = -(t * 44);
            ctx.beginPath();
            ctx.arc(pointerX, pointerY, 18 + pointerPower * 46, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();
        }

        // Intent Grid
        ctx.save();
        ctx.strokeStyle = "rgba(100, 200, 255, 0.03)";
        ctx.lineWidth = 0.5;
        const gridSize = 40 * dpr;
        for(let x = 0; x < w; x += gridSize) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        for(let y = 0; y < h; y += gridSize) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }
        ctx.restore();

        const coreGlow = ctx.createRadialGradient(w/2, h/2, 0, w/2, h/2, 150 * (1 + globalIntensity));
        coreGlow.addColorStop(0, "rgba(255, 255, 200, " + (0.075 * globalIntensity) + ")");
        coreGlow.addColorStop(0.5, "rgba(100, 200, 255, " + (0.04 * globalIntensity) + ")");
        coreGlow.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = coreGlow;
        ctx.fillRect(0, 0, w, h);

        const hasLayerVisibility = layerVisibility !== undefined;
        const backgroundOmni = !hasLayerVisibility && backgroundMode && overlayView === "omni";
        const showPresenceLayer = layerVisibility?.presence ?? (
            overlayView === "omni"
            || overlayView === "presence"
            || overlayView === "file-graph"
            || overlayView === "crawler-graph"
        );
        const showFileImpactLayer = layerVisibility?.["file-impact"] ?? (!backgroundOmni && (overlayView === "omni" || overlayView === "file-impact"));
        const showFileGraphLayer = layerVisibility?.["file-graph"] ?? (!backgroundOmni && (overlayView === "omni" || overlayView === "file-graph"));
        const showCrawlerGraphLayer = layerVisibility?.["crawler-graph"] ?? (!backgroundOmni && (overlayView === "omni" || overlayView === "crawler-graph"));
        const showTruthGateLayer = layerVisibility?.["truth-gate"] ?? (overlayView === "omni" || overlayView === "truth-gate");
        const showLogicalLayer = layerVisibility?.logic ?? (!backgroundOmni && (overlayView === "omni" || overlayView === "logic"));
        const showPainFieldLayer = layerVisibility?.["pain-field"] ?? (!backgroundOmni && (overlayView === "omni" || overlayView === "pain-field"));
        const fileGraphData = resolveFileGraph(currentSimulation);
        const crawlerGraphData = resolveCrawlerGraph(currentSimulation);
        const hasFileGraphData = (
            (Array.isArray(fileGraphData?.nodes) ? fileGraphData.nodes.length : 0)
            + (Array.isArray(fileGraphData?.file_nodes) ? fileGraphData.file_nodes.length : 0)
        ) > 0;
        const hasCrawlerGraphData = (
            (Array.isArray(crawlerGraphData?.nodes) ? crawlerGraphData.nodes.length : 0)
            + (Array.isArray(crawlerGraphData?.crawler_nodes) ? crawlerGraphData.crawler_nodes.length : 0)
        ) > 0;
        const drawFileGraphLayer = showFileGraphLayer && hasFileGraphData;
        const drawCrawlerGraphLayer = showCrawlerGraphLayer && !drawFileGraphLayer && hasCrawlerGraphData;
        const pointerNearest = nearestPresenceAt(pointerField.x, pointerField.y, namedForms);
        const pointerHighlighted = pointerPower > 0.09 && pointerNearest.distance <= 0.18
            ? pointerNearest.index
            : -1;
        const activeLayerCount =
            (showPresenceLayer ? 1 : 0)
            + (showFileImpactLayer ? 1 : 0)
            + (drawFileGraphLayer ? 1 : 0)
            + (drawCrawlerGraphLayer ? 1 : 0)
            + (showTruthGateLayer ? 1 : 0)
            + (showLogicalLayer ? 1 : 0)
            + (showPainFieldLayer ? 1 : 0);
        const denseLayerMix = activeLayerCount >= 6 ? 0.6 : (activeLayerCount >= 4 ? 0.78 : 1);
        const resourceDaimoiSummary = currentSimulation?.presence_dynamics?.resource_daimoi
            ?? currentSimulation?.presence_dynamics?.daimoi_probabilistic?.resource_daimoi;
        const resourceConsumptionSummary = currentSimulation?.presence_dynamics?.resource_consumption
            ?? currentSimulation?.presence_dynamics?.daimoi_probabilistic?.resource_consumption;
        const resourcePacketCount = Math.max(0, Number(resourceDaimoiSummary?.delivered_packets ?? 0));
        const resourceActionPacketCount = Math.max(0, Number(resourceConsumptionSummary?.action_packets ?? 0));
        const resourceBlockedPacketCount = Math.max(0, Number(resourceConsumptionSummary?.blocked_packets ?? 0));
        const economyOverlayIntensity = clamp01(
            (resourcePacketCount / 320)
            + (resourceActionPacketCount / 420)
            + (resourceBlockedPacketCount / 260),
        );
        const graphLayerMix = denseLayerMix * (renderRichOverlayParticles
            ? (0.62 - economyOverlayIntensity * 0.18)
            : 1);
        const showGraphDaimoiOverlay = renderRichOverlayParticles
            && !drawFileGraphLayer
            && !drawCrawlerGraphLayer
            && showPresenceLayer
            && (resourcePacketCount > 0 || resourceActionPacketCount > 0);
        const showResourceDaimoiOverlay = renderRichOverlayParticles
            && showPresenceLayer
            && (resourcePacketCount > 0 || resourceActionPacketCount > 0);
        const showNooiFieldOverlay = renderRichOverlayParticles
            && showLogicalLayer
            && (Number(currentSimulation?.presence_dynamics?.nooi_field?.active_cells ?? 0) > 0)
            && economyOverlayIntensity < 0.82;
        const showPresenceNodes =
            showPresenceLayer || showFileImpactLayer || drawFileGraphLayer || drawCrawlerGraphLayer || showTruthGateLayer;
        const fieldParticlesByPresence = new Map<string, BackendFieldParticle[]>();
        const namedFormPresenceKeys = new Set<string>();
        const unboundFieldParticles: BackendFieldParticle[] = [];
        for (const form of namedForms) {
            const formId = canonicalPresenceId(String((form as any)?.id ?? "").trim());
            if (!formId) {
                continue;
            }
            namedFormPresenceKeys.add(formId);
            namedFormPresenceKeys.add(normalizePresenceKey(formId));
        }
        if (showPresenceNodes && renderRichOverlayParticles) {
            if (allFieldParticleRows.length > 0) {
                for (const row of allFieldParticleRows) {
                    if (!row || typeof row !== "object") continue;
                    const particle = row as BackendFieldParticle;
                    const presenceId = resolveParticlePresenceId(particle);
                    if (!presenceId) continue;
                    const canonicalId = canonicalPresenceId(presenceId);
                    const normalizedId = normalizePresenceKey(canonicalId || presenceId);
                    const isBound =
                        namedFormPresenceKeys.has(presenceId)
                        || (canonicalId.length > 0 && namedFormPresenceKeys.has(canonicalId))
                        || namedFormPresenceKeys.has(normalizedId);
                    if (!isBound) {
                        unboundFieldParticles.push(particle);
                        continue;
                    }
                    const aliasKeys = [presenceId, canonicalId, normalizedId]
                        .filter((value, idx, arr) => value.length > 0 && arr.indexOf(value) === idx);
                    for (const key of aliasKeys) {
                        let bucket = fieldParticlesByPresence.get(key);
                        if (!bucket) {
                            bucket = [];
                            fieldParticlesByPresence.set(key, bucket);
                        }
                        bucket.push(particle);
                    }
                }
            }
        }

        if (overlayMotionByParticleId.size > 0) {
            const staleAfterSec = 4.0;
            for (const [particleId, motionState] of overlayMotionByParticleId.entries()) {
                if ((overlayMotionNowSec - motionState.seenAtSec) > staleAfterSec) {
                    overlayMotionByParticleId.delete(particleId);
                }
            }
        }
        if (overlayGhostTrailByParticleId.size > 0) {
            for (const [particleId, trailState] of overlayGhostTrailByParticleId.entries()) {
                if ((overlayMotionNowSec - trailState.seenAtSec) > ghostTrailStaleAfterSec) {
                    overlayGhostTrailByParticleId.delete(particleId);
                }
            }
        }

        if (showPresenceLayer) {
            drawEchoes(t, w, h, currentSimulation);
            drawRiverFlow(t, w, h, namedForms, currentSimulation);
            drawWitnessThreadFlow(t, w, h, namedForms, currentSimulation);
            drawGhostSentinel(t, w, h, currentSimulation);
            drawGraphRuntimeDiagnostics(w, h, currentSimulation);
            if (showGraphDaimoiOverlay) {
                ctx.save();
                ctx.globalAlpha = 0.48;
                drawGraphDaimoiFlowOverlay(t, w, h, namedForms, currentSimulation);
                ctx.restore();
            }
        }
        if (showFileImpactLayer) {
            ctx.save();
            ctx.globalAlpha = denseLayerMix;
            drawFileInfluenceOverlay(t, w, h, namedForms, currentSimulation);
            ctx.restore();
        }
        if (showLogicalLayer) {
            ctx.save();
            ctx.globalAlpha = denseLayerMix;
            drawLogicalGraphOverlay(t, w, h, currentSimulation);
            ctx.restore();
        }
        if (drawFileGraphLayer) {
            ctx.save();
            ctx.globalAlpha = graphLayerMix;
            drawFileCategoryGraph(t, w, h, currentSimulation, namedForms);
            ctx.restore();
        }
        if (drawCrawlerGraphLayer) {
            ctx.save();
            ctx.globalAlpha = graphLayerMix;
            drawCrawlerCategoryGraph(t, w, h, currentSimulation);
            ctx.restore();
        }
        if (showTruthGateLayer) {
            drawTruthBindingOverlay(t, w, h, namedForms, currentSimulation);
        }
        if (showPainFieldLayer) {
            ctx.save();
            ctx.globalAlpha = denseLayerMix;
            drawPainFieldOverlay(t, w, h, currentSimulation);
            ctx.restore();
        }

        if (showPresenceNodes) {
            for(let i=0; i<namedForms.length; i++) {
            const f: any = namedForms[i];
            const cx = f.x * w; const cy = f.y * h;
            const radiusBase = (HEX_SIZE * 2.2);
            if(f.id === "core_pulse") {
                const coreRadius = radiusBase * (1.5 + Math.sin(t * 4) * 0.2);
                const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreRadius);
                g.addColorStop(0, "rgba(255, 255, 255, 0.8)");
                g.addColorStop(0.2, "rgba(255, 200, 100, 0.4)");
                g.addColorStop(1, "rgba(255, 100, 0, 0)");
                ctx.fillStyle = g;
                ctx.beginPath(); ctx.arc(cx, cy, coreRadius, 0, Math.PI*2); ctx.fill();
            }
            const entityState = currentSimulation?.entities?.find(e => e.id === f.id);
            const intensityRaw = entityState ? (entityState.bpm - 60) / 40 : globalIntensity;
            const intensity = Math.max(0, Math.min(1, intensityRaw));
            const isHighlighted = highlighted === i || pointerHighlighted === i;
            const telemetry = drawPresenceStatus(
                cx,
                cy,
                radiusBase,
                f.hue,
                entityState,
                String(f.id ?? ""),
                isHighlighted,
            );
            drawNebula(t, f, cx, cy, radiusBase, f.hue, intensity, isHighlighted);
            if (renderRichOverlayParticles) {
                drawParticles(t, f, radiusBase, isHighlighted, fieldParticlesByPresence, w, h);
            }
            const isBottomHalf = f.y > 0.7;
            const labelY = isBottomHalf ? cy - radiusBase * 1.2 : cy + radiusBase * 1.2;
            ctx.save();
            ctx.globalCompositeOperation = "source-over";
            ctx.textAlign = "center";
            ctx.font = "600 12px serif";
            const enW = ctx.measureText(f.en).width;
            ctx.font = "500 10px sans-serif";
            const jaW = ctx.measureText(f.ja).width;
            const metricLine = "BPM " + telemetry.bpm + "  STB " + telemetry.stabilityPct + "%  RES " + telemetry.resonancePct + "%  LYR " + telemetry.fieldLayer;
            const metricW = ctx.measureText(metricLine).width;
            const boxW = Math.max(enW, jaW, metricW) + 18;
            const boxH = 44;
            ctx.shadowBlur = isHighlighted ? 16 : 10;
            ctx.shadowColor = "hsla(" + f.hue + ", 85%, 62%, 0.42)";
            ctx.fillStyle = "rgba(6, 14, 24, " + (isHighlighted ? 0.93 : 0.82) + ")";
            ctx.beginPath();
            ctx.roundRect(cx - boxW/2, isBottomHalf ? labelY - boxH : labelY, boxW, boxH, 6);
            ctx.fill();
            ctx.strokeStyle = "hsla(" + f.hue + ", 70%, 60%, " + (isHighlighted ? 0.8 : 0.4) + ")";
            ctx.lineWidth = isHighlighted ? 1.5 : 1;
            ctx.stroke();
            const textCenterY = isBottomHalf ? labelY - boxH/2 : labelY + boxH/2;
            ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
            ctx.fillText(f.en, cx, textCenterY - 2);
            ctx.fillStyle = "rgba(200, 220, 255, 0.8)";
            ctx.font = "500 9px sans-serif";
            ctx.fillText(f.ja, cx, textCenterY + 9);
            ctx.fillStyle = "rgba(160, 226, 255, 0.96)";
            ctx.font = "600 8px ui-monospace, SFMono-Regular, Menlo, monospace";
            ctx.fillText(metricLine, cx, textCenterY + 20);
            ctx.restore();
            }

            if (showResourceDaimoiOverlay) {
                ctx.save();
                ctx.globalAlpha = 0.64;
                drawResourceDaimoiOverlay(t, w, h, namedForms, currentSimulation);
                ctx.restore();
            }
            if (showNooiFieldOverlay) {
                ctx.save();
                ctx.globalAlpha = 0.34;
                drawNooiFieldOverlay(
                    t,
                    w,
                    h,
                    currentSimulation,
                    fieldParticleCount > 0 || showResourceDaimoiOverlay,
                );
                ctx.restore();
            }

            if (renderRichOverlayParticles && unboundFieldParticles.length > 0) {
                drawUnboundParticles(t, unboundFieldParticles, w, h);
            }

        }

        // Draw mouse daimon last so it's on top
        drawMouseDaimon(t, w, h, pointerField.x, pointerField.y, pointerPower, pointerField.inside);

        rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);
    const api = {
        pulseAt: (x: number, y: number, power: number, target = "particle_field") => { 
            ripple = { x, y, power, at: performance.now() }; 
            highlighted = -1;
            const baseUrl = runtimeBaseUrl();
            fetch(baseUrl + "/api/witness", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ type: "touch", target })
            }).catch(() => {});
        },
        singAll: () => {},
        getAnchorRatio: (kind: string, targetId: string) => {
            const id = String(targetId ?? "").trim();
            if (!id) return null;

            if (kind === "node" || kind === "file" || kind === "crawler") {
                for (const hit of graphNodeHits) {
                    const hitId = String(hit.node?.id ?? "");
                    if (hitId && hitId === id) {
                        return {
                            x: hit.x,
                            y: hit.y,
                            kind: hit.nodeKind,
                            label: shortPathLabel(
                                String(
                                    hit.node?.title
                                    || hit.node?.domain
                                    || hit.node?.source_rel_path
                                    || hit.node?.archived_rel_path
                                    || hit.node?.name
                                    || hit.node?.label
                                    || hit.node?.id
                                    || "",
                                ),
                            ),
                        };
                    }
                }
                return null;
            }

            const forms = resolveNamedForms();
            const found = forms.find((f: any) => String(f?.id ?? "").trim() === id);
            if (!found) {
                return null;
            }
            return {
                x: clamp01(Number(found.x ?? 0.5)),
                y: clamp01(Number(found.y ?? 0.5)),
                kind: "presence",
                label: String(found.en ?? found.ja ?? found.id ?? id),
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
    };

    const shouldOpenWorldscreen = (nodeKind: "file" | "crawler", nodeId: string): boolean => {
        const key = `${nodeKind}:${nodeId}`;
        const now = performance.now();
        const previousTap = lastNexusPointerTapRef.current;
        const isDoubleTap = Boolean(
            previousTap
            && previousTap.key === key
            && (now - previousTap.atMs) <= 360,
        );
        lastNexusPointerTapRef.current = {
            key,
            atMs: now,
        };
        return isDoubleTap;
    };

    const activateGraphNode = (
        node: any,
        nodeKind: "file" | "crawler",
        xRatio: number,
        yRatio: number,
        openWorldscreen: boolean,
        selectionSource: "graph" | "particle" = "graph",
    ) => {
        const resourceKind = resourceKindForNode(node);
        const graphNodeId = String(node?.id ?? "").trim();
        selectedGraphNodeId = graphNodeId;
        selectedGraphNodeLabel = shortPathLabel(
            String(
                node?.title
                || node?.domain
                || node?.source_rel_path
                || node?.archived_rel_path
                || node?.name
                || node?.label
                || node?.id
                || "",
            ),
        );
        const target = String(
            node?.domain
            || node?.title
            || node?.source_rel_path
            || node?.archived_rel_path
            || node?.name
            || node?.id
            || "particle_field",
        );
        const witnessPrefix = nodeKind === "crawler" ? "crawler" : "file";
        api.pulseAt(xRatio, yRatio, 1.0, `${witnessPrefix}:${target}`);
        if (metaRef.current) {
            const selectedLabel = selectionSource === "particle"
                ? "route node"
                : (nodeKind === "crawler" ? "crawler node" : "file node");
            metaRef.current.textContent = openWorldscreen
                ? `opening hologram: ${selectedGraphNodeLabel}`
                : `focused ${selectedLabel}: ${selectedGraphNodeLabel} [${resourceKindLabel(resourceKind)}] (double tap to open hologram)`;
        }

        onNexusInteraction?.({
            nodeId: graphNodeId || selectedGraphNodeLabel,
            nodeKind,
            resourceKind,
            label: selectedGraphNodeLabel,
            xRatio: clamp01(xRatio),
            yRatio: clamp01(yRatio),
            openWorldscreen,
        });

        if (!openWorldscreen) {
            return;
        }

        const openUrl = openUrlForGraphNode(
            node,
            nodeKind === "crawler" ? "crawler" : "file",
        );
        const domain = String(node?.domain ?? "").trim();
        const worldscreenUrl = resolveWorldscreenUrl(openUrl, nodeKind, domain);
        if (worldscreenUrl) {
            const worldscreenNodeKind: "file" | "crawler" = nodeKind === "crawler" ? "crawler" : "file";
            const isRemoteResource = isRemoteHttpUrl(worldscreenUrl);
            const frameUrlCandidate = remoteFrameUrlForNode(node, worldscreenUrl, resourceKind);
            const imageRef = String(
                node?.source_rel_path
                || node?.archive_rel_path
                || node?.archived_rel_path
                || node?.url
                || worldscreenUrl,
            ).trim();
            const commentRef = nexusCommentRefForNode(node, worldscreenNodeKind, worldscreenUrl) || imageRef;
            const graphGeneratedAt =
                nodeKind === "crawler"
                    ? String(resolveCrawlerGraph(simulationRef.current)?.generated_at ?? "")
                    : String(resolveFileGraph(simulationRef.current)?.generated_at ?? "");
            const discoveredAt = timestampLabel(node?.discovered_at ?? node?.discoveredAt ?? "");
            const fetchedAt = timestampLabel(node?.fetched_at ?? node?.fetchedAt ?? node?.last_seen ?? node?.lastSeen ?? "");
            const encounteredAt =
                fetchedAt
                || discoveredAt
                || timestampLabel(node?.encountered_at ?? node?.encounteredAt ?? graphGeneratedAt);
            setWorldscreenMode("overview");
            setWorldscreen({
                nodeId: graphNodeId,
                commentRef,
                url: worldscreenUrl,
                imageRef,
                label: selectedGraphNodeLabel,
                nodeKind: worldscreenNodeKind,
                resourceKind,
                anchorRatioX: clamp01(xRatio),
                anchorRatioY: clamp01(yRatio),
                view: resourceKind === "image"
                    ? "metadata"
                    : isRemoteResource
                        ? "metadata"
                        : worldscreenViewForNode(node, worldscreenNodeKind, resourceKind),
                subtitle: worldscreenSubtitleForNode(node, worldscreenNodeKind, resourceKind),
                remoteFrameUrl: resourceKind === "image"
                    ? (frameUrlCandidate || worldscreenUrl)
                    : isRemoteResource
                        ? frameUrlCandidate
                        : "",
                encounteredAt,
                sourceUrl: String(node?.source_url ?? node?.sourceUrl ?? "").trim(),
                domain,
                titleText: String(node?.title ?? "").trim(),
                statusText: String(node?.status ?? "").trim(),
                contentTypeText: String(node?.content_type ?? node?.contentType ?? "").trim(),
                complianceText: String(node?.compliance ?? "").trim(),
                discoveredAt,
                fetchedAt,
                summaryText: String(node?.summary ?? node?.text_excerpt ?? "").trim(),
                tagsText: joinListValues(node?.tags),
                labelsText: joinListValues(node?.labels),
            });
            if (metaRef.current) {
                metaRef.current.textContent = `hologram opened: ${selectedGraphNodeLabel}`;
            }
        }
    };

    const onPointerDown = (event: PointerEvent) => {
        const rect = canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const xRatio = clamp01((event.clientX - rect.left) / rect.width);
        const yRatio = clamp01((event.clientY - rect.top) / rect.height);
        pointerField = {
            x: xRatio,
            y: yRatio,
            power: Math.max(pointerField.power, 0.2 + mouseInfluenceRef.current * 0.26),
            inside: true,
        };

        const graphHit = nearestGraphNodeAt(xRatio, yRatio);
        if (graphHit) {
            event.preventDefault();
            event.stopPropagation();
            const graphNodeId = String(graphHit.node?.id ?? "").trim();
            const openWorldscreen = shouldOpenWorldscreen(
                graphHit.nodeKind,
                graphNodeId || graphHit.nodeKind,
            );
            activateGraphNode(
                graphHit.node,
                graphHit.nodeKind,
                graphHit.x,
                graphHit.y,
                openWorldscreen,
                "graph",
            );
            onUserPresenceInputRef.current?.({
                kind: "click",
                target: graphNodeId || "graph_node",
                message: `click graph node ${graphNodeId || "graph_node"}`,
                xRatio,
                yRatio,
                embedDaimoi: true,
                meta: {
                    source: "simulation-canvas",
                    nodeKind: graphHit.nodeKind,
                },
            });
            return;
        }

        const particleHit = nearestParticleTelemetryAt(xRatio, yRatio);
        if (particleHit) {
            const currentState = simulationRef.current;
            const routeTargetId = String(particleHit.routeNodeId ?? "").trim();
            const graphTargetId = String(particleHit.graphNodeId ?? "").trim();
            const resolvedNode =
                resolveGraphNodeById(currentState, routeTargetId)
                || resolveGraphNodeById(currentState, graphTargetId);
            if (resolvedNode) {
                event.preventDefault();
                event.stopPropagation();
                const graphNodeId = String(resolvedNode.node?.id ?? "").trim();
                const openWorldscreen = shouldOpenWorldscreen(
                    resolvedNode.nodeKind,
                    graphNodeId || resolvedNode.nodeKind,
                );
                activateGraphNode(
                    resolvedNode.node,
                    resolvedNode.nodeKind,
                    particleHit.x,
                    particleHit.y,
                    openWorldscreen,
                    "particle",
                );
                onUserPresenceInputRef.current?.({
                    kind: "click",
                    target: graphNodeId || "particle_node",
                    message: `click particle route ${graphNodeId || "particle_node"}`,
                    xRatio,
                    yRatio,
                    embedDaimoi: true,
                    meta: {
                        source: "simulation-canvas",
                        nodeKind: resolvedNode.nodeKind,
                    },
                });
                return;
            }
        }

        const namedForms = resolveNamedForms();
        const nearest = nearestPresenceAt(xRatio, yRatio, namedForms);
        const targetField = nearest.index >= 0 ? namedForms[nearest.index] : null;
        const hasTarget = targetField && nearest.distance <= 0.22;
        const targetId = hasTarget ? String(targetField.id ?? "particle_field") : "particle_field";
        selectedGraphNodeId = "";
        selectedGraphNodeLabel = "";
        if (hasTarget) {
            highlighted = nearest.index;
        }
        api.pulseAt(xRatio, yRatio, 1.0, targetId);
        onUserPresenceInputRef.current?.({
            kind: "click",
            target: targetId,
            message: `click simulation field ${targetId}`,
            xRatio,
            yRatio,
            embedDaimoi: true,
            meta: {
                source: "simulation-canvas",
            },
        });
    };

    const onPointerMove = (event: PointerEvent) => {
        const rect = canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) {
            return;
        }
        pointerField = {
            x: clamp01((event.clientX - rect.left) / rect.width),
            y: clamp01((event.clientY - rect.top) / rect.height),
            power: Math.max(pointerField.power, 0.16 + mouseInfluenceRef.current * 0.24),
            inside: true,
        };
        const nowMs = Date.now();
        if ((nowMs - userPresenceMouseEmitMsRef.current) >= 96) {
            userPresenceMouseEmitMsRef.current = nowMs;
            onUserPresenceInputRef.current?.({
                kind: "mouse_move",
                target: "simulation_canvas",
                message: "mouse move in simulation",
                xRatio: pointerField.x,
                yRatio: pointerField.y,
                embedDaimoi: false,
                meta: {
                    source: "simulation-canvas",
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
    if (interactive) {
      canvas.addEventListener("pointerdown", onPointerDown);
    }
    if (onOverlayInit) onOverlayInit(api);
    return () => {
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerleave", onPointerLeave);
        if (interactive) {
          canvas.removeEventListener("pointerdown", onPointerDown);
        }
        cancelAnimationFrame(rafId);
    };
  }, [
    backgroundMode,
    backgroundWash,
    interactive,
    layerVisibility,
    onNexusInteraction,
    onOverlayInit,
    overlayView,
    resolveFieldParticleRows,
  ]);

  const activeOverlayView =
    OVERLAY_VIEW_OPTIONS.find((option) => option.id === overlayView) ?? OVERLAY_VIEW_OPTIONS[0];

  const containerClassName = backgroundMode
    ? `relative h-full w-full overflow-hidden ${className}`.trim()
    : `relative mt-3 border border-[rgba(36,31,26,0.16)] rounded-xl overflow-hidden bg-gradient-to-b from-[#0f1a1f] to-[#131b2a] ${className}`.trim();
  const canvasHeight: number | string = backgroundMode ? "100%" : height;
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
      <canvas ref={canvasRef} style={{ height: canvasHeight }} className="block w-full" />
      <canvas ref={overlayRef} style={{ height: canvasHeight }} className="absolute inset-0 w-full touch-none" />
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
