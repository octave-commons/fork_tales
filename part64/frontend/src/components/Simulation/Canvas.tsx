import { useRef, useEffect, useState } from "react";
import type {
  SimulationState,
  Catalog,
  FileGraph,
  CrawlerGraph,
  TruthState,
} from "../../types";

interface Props {
  simulation: SimulationState | null;
  catalog: Catalog | null;
  onOverlayInit?: (api: any) => void;
  height?: number;
  defaultOverlayView?: OverlayViewId;
  overlayViewLocked?: boolean;
  compactHud?: boolean;
  interactive?: boolean;
}

interface GraphWorldscreenState {
  url: string;
  label: string;
  nodeKind: "file" | "crawler";
  resourceKind: GraphNodeResourceKind;
  view: GraphWorldscreenView;
  subtitle: string;
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

type GraphWorldscreenView = "website" | "editor" | "video";

type GraphNodeShape = "circle" | "square" | "diamond" | "triangle" | "hexagon";

interface GraphNodeVisualSpec {
  hue: number;
  saturation: number;
  value: number;
  shape: GraphNodeShape;
  liftBoost: number;
  glowBoost: number;
}

interface EditorPreviewState {
  status: "idle" | "loading" | "ready" | "error";
  content: string;
  error: string;
  truncated: boolean;
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
    label: "File Graph",
    description: "File categories, links, and embed layers.",
  },
  {
    id: "crawler-graph",
    label: "Crawler Graph",
    description: "Crawler topology and web-domain structure.",
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

function runtimeBaseUrl(): string {
  return window.location.port === "5173" ? "http://127.0.0.1:8787" : "";
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

export function SimulationCanvas({
  simulation,
  catalog,
  onOverlayInit,
  height = 300,
  defaultOverlayView = "omni",
  overlayViewLocked = false,
  compactHud = false,
  interactive = true,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const metaRef = useRef<HTMLParagraphElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<SimulationState | null>(simulation);
  const catalogRef = useRef<Catalog | null>(catalog);
  const [worldscreen, setWorldscreen] = useState<GraphWorldscreenState | null>(null);
  const [editorPreview, setEditorPreview] = useState<EditorPreviewState>({
    status: "idle",
    content: "",
    error: "",
    truncated: false,
  });
  const [overlayView, setOverlayView] = useState<OverlayViewId>(defaultOverlayView);

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
    if (!worldscreen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setWorldscreen(null);
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
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl", { alpha: false, antialias: true });
    if (!gl) return;

    const vertexSrc = `
      attribute vec2 aPos;
      attribute float aSize;
      attribute vec3 aColor;
      uniform float uTime;
      uniform vec2 uMouse;
      uniform float uInfluence;
      varying vec3 vColor;
      void main() {
        vec2 d = aPos - uMouse;
        float dist = length(d);
        float force = max(0.0, (1.0 - dist * 3.0) * uInfluence);
        vec2 offset = normalize(d) * force * 0.1;
        float wobble = sin((aPos.x * 4.0) + (aPos.y * 2.0) + (uTime * 0.0025)) * 0.02;
        gl_Position = vec4(aPos.x + offset.x, aPos.y + wobble + offset.y, 0.0, 1.0);
        gl_PointSize = aSize + force * 10.0;
        vColor = aColor + vec3(force * 0.5);
      }
    `;

    const fragmentSrc = "precision mediump float; varying vec3 vColor; void main() { vec2 c = gl_PointCoord - vec2(0.5, 0.5); float d = dot(c, c); if (d > 0.25) discard; float soft = smoothstep(0.25, 0.0, d); float core = smoothstep(0.08, 0.0, d); vec3 color = min(vec3(1.0), vColor + vec3(core * 0.45)); float alpha = min(1.0, soft * 0.9 + core * 0.45); gl_FragColor = vec4(color, alpha); }";

    const compile = (type: number, source: string) => {
        const shader = gl.createShader(type);
        if(!shader) return null;
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        return shader;
    };

    const vs = compile(gl.VERTEX_SHADER, vertexSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fragmentSrc);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    if(!program) return;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    const posBuffer = gl.createBuffer();
    const sizeBuffer = gl.createBuffer();
    const colorBuffer = gl.createBuffer();

    const locPos = gl.getAttribLocation(program, "aPos");
    const locSize = gl.getAttribLocation(program, "aSize");
    const locColor = gl.getAttribLocation(program, "aColor");
    const locTime = gl.getUniformLocation(program, "uTime");
    const locMouse = gl.getUniformLocation(program, "uMouse");
    const locInfluence = gl.getUniformLocation(program, "uInfluence");
    const setProgram = gl.useProgram.bind(gl);

    let count = 0;
    let capacity = 0;
    let currentPositions = new Float32Array(0);
    let targetPositions = new Float32Array(0);
    let currentSizes = new Float32Array(0);
    let targetSizes = new Float32Array(0);
    let currentColors = new Float32Array(0);
    let targetColors = new Float32Array(0);
    let lastTick = 0;
    let rafId = 0;
    let viewportWidth = 0;
    let viewportHeight = 0;
    let mouseX = 0;
    let mouseY = 0;
    let influence = 0;

    const tracerCanvas = document.createElement("canvas");
    tracerCanvas.style.position = "absolute";
    tracerCanvas.style.inset = "0";
    tracerCanvas.style.pointerEvents = "none";
    containerRef.current?.appendChild(tracerCanvas);
    const ctxTracer = tracerCanvas.getContext("2d");
    let tracers: Array<{x: number, y: number, prevX: number, prevY: number, life: number}> = [];

    const ensureCapacity = (pointCount: number) => {
        if (pointCount <= capacity) return;
        capacity = Math.max(pointCount, capacity * 2, 64);
        currentPositions = new Float32Array(capacity * 2);
        targetPositions = new Float32Array(capacity * 2);
        currentSizes = new Float32Array(capacity);
        targetSizes = new Float32Array(capacity);
        currentColors = new Float32Array(capacity * 3);
        targetColors = new Float32Array(capacity * 3);
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, currentPositions.byteLength, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, currentSizes.byteLength, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, currentColors.byteLength, gl.DYNAMIC_DRAW);
    };

    const draw = (ts: number) => {
        const delta = lastTick > 0 ? Math.min(64, ts - lastTick) : 16;
        lastTick = ts;
        const blend = Math.min(1, (delta / 1000) * 8);
        for(let i=0; i<count*2; i++) currentPositions[i] += (targetPositions[i] - currentPositions[i]) * blend;
        for(let i=0; i<count; i++) currentSizes[i] += (targetSizes[i] - currentSizes[i]) * blend;
        for(let i=0; i<count*3; i++) currentColors[i] += (targetColors[i] - currentColors[i]) * blend;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const nextWidth = Math.max(1, Math.floor(rect.width * dpr));
        const nextHeight = Math.max(1, Math.floor(rect.height * dpr));
        if (nextWidth !== viewportWidth || nextHeight !== viewportHeight) {
            viewportWidth = nextWidth;
            viewportHeight = nextHeight;
            canvas.width = viewportWidth;
            canvas.height = viewportHeight;
            tracerCanvas.width = viewportWidth;
            tracerCanvas.height = viewportHeight;
        }
        gl.viewport(0, 0, viewportWidth, viewportHeight);
        gl.clearColor(0.016, 0.024, 0.04, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        if(count > 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, currentPositions.subarray(0, count*2));
            gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, currentSizes.subarray(0, count));
            gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, currentColors.subarray(0, count*3));
            setProgram(program);
            gl.uniform1f(locTime, ts || 0);
            gl.uniform2f(locMouse, mouseX, mouseY);
            gl.uniform1f(locInfluence, influence);
            gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
            gl.enableVertexAttribArray(locPos);
            gl.vertexAttribPointer(locPos, 2, gl.FLOAT, false, 0, 0);
            gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
            gl.enableVertexAttribArray(locSize);
            gl.vertexAttribPointer(locSize, 1, gl.FLOAT, false, 0, 0);
            gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
            gl.enableVertexAttribArray(locColor);
            gl.vertexAttribPointer(locColor, 3, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.POINTS, 0, count);
        }

        if(ctxTracer) {
            ctxTracer.clearRect(0, 0, viewportWidth, viewportHeight);
            ctxTracer.globalCompositeOperation = "screen";
            ctxTracer.lineWidth = 1.25;
            tracers.forEach(t => {
                t.life -= 0.01;
                ctxTracer.strokeStyle = "rgba(210, 235, 255, " + (t.life * 0.42) + ")";
                ctxTracer.beginPath();
                ctxTracer.moveTo((t.x+1)/2 * viewportWidth, (1-(t.y+1)/2) * viewportHeight);
                ctxTracer.lineTo((t.prevX+1)/2 * viewportWidth, (1-(t.prevY+1)/2) * viewportHeight);
                ctxTracer.stroke();
            });
            tracers = tracers.filter(t => t.life > 0);
            if(count > 0 && Math.random() < 0.18) {
                const idx = Math.floor(Math.random() * count);
                tracers.push({
                    x: currentPositions[idx*2],
                    y: currentPositions[idx*2+1],
                    prevX: currentPositions[idx*2] - (Math.random()-0.5)*0.1,
                    prevY: currentPositions[idx*2+1] - (Math.random()-0.5)*0.1,
                    life: 1.0
                });
                if(tracers.length > 200) tracers.shift();
            }
        }
        rafId = requestAnimationFrame(draw);
    };

    const onMove = (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouseY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
        influence = 1.0;
    };
    window.addEventListener("mousemove", onMove);
    const decay = setInterval(() => { influence *= 0.95; }, 50);
    rafId = requestAnimationFrame(draw);

    (canvas as any).__updateSim = (state: SimulationState) => {
        const points = state.points || [];
        count = points.length;
        ensureCapacity(count);
        for(let i=0; i<count; i++) {
            const p = points[i];
            targetPositions[i*2] = p.x;
            targetPositions[i*2+1] = p.y;
            targetSizes[i] = p.size;
            targetColors[i*3] = p.r;
            targetColors[i*3+1] = p.g;
            targetColors[i*3+2] = p.b;
            if(lastTick === 0) {
                currentPositions[i*2] = targetPositions[i*2];
                currentPositions[i*2+1] = targetPositions[i*2+1];
                currentSizes[i] = targetSizes[i];
                currentColors[i*3] = targetColors[i*3];
                currentColors[i*3+1] = targetColors[i*3+1];
                currentColors[i*3+2] = targetColors[i*3+2];
            }
        }
        const flowRate = state.presence_dynamics?.river_flow?.rate;
        const forkTaxBalance = state.presence_dynamics?.fork_tax?.balance;
        const witnessContinuity = state.presence_dynamics?.witness_thread?.continuity_index;
        const witnessTrace = state.presence_dynamics?.witness_thread?.lineage?.[0]?.ref;
        const truthState = state.truth_state ?? catalogRef.current?.truth_state;
        const truthClaim = truthState?.claim;
        const truthStatus = String(truthClaim?.status ?? "undecided");
        const truthKappa = Number(truthClaim?.kappa ?? 0);
        const fileGraph = state.file_graph ?? catalogRef.current?.file_graph;
        const inboxPending = fileGraph?.inbox?.pending_count;
        if(metaRef.current) {
            const inboxLabel = inboxPending !== undefined ? ` | inbox: ${inboxPending}` : "";
            const truthLabel = truthState
                ? ` | truth: ${truthStatus} κ=${truthKappa.toFixed(2)}`
                : "";
            if (flowRate !== undefined || forkTaxBalance !== undefined) {
                metaRef.current.textContent =
                    "sim particles: " +
                    state.total +
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
                    truthLabel;
            } else {
                metaRef.current.textContent =
                    "sim particles: " +
                    state.total +
                    " | audio: " +
                    state.audio +
                    inboxLabel +
                    truthLabel;
            }
        }
    };

    return () => {
        window.removeEventListener("mousemove", onMove);
        clearInterval(decay);
        cancelAnimationFrame(rafId);
        tracerCanvas.remove();
    };
  }, []);

  useEffect(() => {
    if(simulation && canvasRef.current) (canvasRef.current as any).__updateSim?.(simulation);
  }, [simulation]);

  useEffect(() => {
    const canvas = overlayRef.current;
    if(!canvas) return;
    const ctx = canvas.getContext("2d");
    if(!ctx) return;

    let rafId = 0;
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

    const resolveNamedForms = () => catalogRef.current?.entity_manifest || fallbackNamedForms;

    const resolveFileGraph = (state: SimulationState | null): FileGraph | null => {
        const fromSimulation = state?.file_graph;
        if (fromSimulation && Array.isArray(fromSimulation.file_nodes)) {
            return fromSimulation;
        }
        const fromCatalog = catalogRef.current?.file_graph;
        if (fromCatalog && Array.isArray(fromCatalog.file_nodes)) {
            return fromCatalog;
        }
        return null;
    };

    const resolveCrawlerGraph = (state: SimulationState | null): CrawlerGraph | null => {
        const fromSimulation = state?.crawler_graph;
        if (fromSimulation && Array.isArray(fromSimulation.crawler_nodes)) {
            return fromSimulation;
        }
        const fromCatalog = catalogRef.current?.crawler_graph;
        if (fromCatalog && Array.isArray(fromCatalog.crawler_nodes)) {
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

    const nearestGraphNodeAt = (xRatio: number, yRatio: number) => {
        let match: { hit: (typeof graphNodeHits)[number]; distance: number } | null = null;
        for (const hit of graphNodeHits) {
            const dx = xRatio - hit.x;
            const dy = yRatio - hit.y;
            const distance = Math.hypot(dx, dy);
            if (distance > Math.max(0.008, hit.radiusNorm * 1.5)) {
                continue;
            }
            if (!match || distance < match.distance) {
                match = { hit, distance };
            }
        }
        return match?.hit ?? null;
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

    const drawParticles = (
        t: number,
        field: any,
        cx: number,
        cy: number,
        radius: number,
        hue: number,
        intensity: number,
        isHighlighted: boolean,
    ) => {
        const count = isHighlighted ? 44 : 20;
        ctx.save();
        ctx.globalCompositeOperation = "lighter";
        ctx.globalAlpha = isHighlighted ? 0.74 : 0.46;
        for(let i=0; i<count; i++) {
            const seed = i * 0.91 + field.x * 0.43 + field.y * 0.21;
            const angle = t * (0.3 + (i%5)*0.05 + intensity * 0.1) + seed;
            const orbit = radius * (0.2 + (i % 6) * 0.1);
            const wobble = Math.sin(t*1.2+seed) * radius * 0.05;
            const px = cx + Math.cos(angle)*(orbit+wobble);
            const py = cy + Math.sin(angle * 1.1)*(orbit * 0.6 + wobble * 0.5);
            const r = Math.max(0.5, 1.2 + Math.sin(seed+t*2) * 0.8 + intensity * 0.4);
            const pHue = i % 2 === 0 ? hue : (hue + 40) % 360;
            ctx.fillStyle = "hsla(" + pHue + ", 88%, 68%, 0.26)";
            ctx.beginPath();
            ctx.arc(px, py, r * 2.2, 0, Math.PI*2);
            ctx.fill();
            ctx.fillStyle = "hsla(" + pHue + ", 94%, 78%, 0.82)";
            ctx.beginPath();
            ctx.arc(px, py, r, 0, Math.PI*2);
            ctx.fill();
            ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
            ctx.beginPath();
            ctx.arc(px, py, Math.max(0.45, r * 0.36), 0, Math.PI*2);
            ctx.fill();
        }
        ctx.restore();
    };

    const drawPresenceStatus = (cx: number, cy: number, radius: number, hue: number, entityState: any) => {
        const bpmRatio = clamp01((((entityState?.bpm || 78) - 60) / 80));
        const stabilityRatio = ratioFromMetric(entityState?.stability, 0.72);
        const resonanceRatio = ratioFromMetric(entityState?.resonance, 0.65);
        const ringRadius = radius * 1.08;

        ctx.save();
        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = "rgba(7, 14, 24, 0.84)";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.arc(cx, cy, ringRadius, 0, Math.PI * 2);
        ctx.stroke();

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
    ) => {
        const graph = resolveFileGraph(state);
        if (!graph) {
            return;
        }

        const fieldNodes = Array.isArray(graph.field_nodes) ? graph.field_nodes : [];
        const fileNodes = Array.isArray(graph.file_nodes) ? graph.file_nodes : [];
        const graphNodes = Array.isArray(graph.nodes) && graph.nodes.length > 0
            ? graph.nodes
            : [...fieldNodes, ...fileNodes];
        const nodeById = new Map(graphNodes.map((node: any) => [String(node.id), node]));
        const edges = Array.isArray(graph.edges) ? graph.edges : [];
        const resourceCounts: Record<string, number> = {};

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
                const hue = Number(src.hue ?? tgt.hue ?? 210);
                ctx.strokeStyle = `hsla(${hue}, 92%, 68%, ${0.05 + weight * 0.26})`;
                ctx.lineWidth = 0.4 + weight * 1.1;
                ctx.beginPath();
                const bend = Math.sin((t * 1.2) + (i * 0.17)) * 8;
                ctx.moveTo(sx, sy);
                ctx.quadraticCurveTo((sx + tx) / 2 + bend, (sy + ty) / 2 - bend * 0.45, tx, ty);
                ctx.stroke();
            }
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

        if (fileNodes.length > 0) {
            ctx.save();
            ctx.globalCompositeOperation = "screen";
            for (let i = 0; i < fileNodes.length; i++) {
                const node = fileNodes[i] as any;
                const nx = clamp01(Number(node.x ?? 0.5));
                const ny = clamp01(Number(node.y ?? 0.5));
                const px = nx * w;
                const py = ny * h;
                const importance = clamp01(Number(node.importance ?? 0.2));
                const pulse = 0.5 + Math.sin((t * 3) + i * 0.33) * 0.5;
                const resourceKind = classifyFileResourceKind(node);
                resourceCounts[resourceKind] = (resourceCounts[resourceKind] ?? 0) + 1;
                const fallbackHue = Number(node.hue ?? 210);
                const visual = resourceVisualSpec(resourceKind, fallbackHue);
                let radius = (1.8 + importance * 3.2 + pulse * 0.9) * (resourceKind === "video" ? 1.08 : 1);
                const isSelected = selectedGraphNodeId !== "" && selectedGraphNodeId === String(node.id ?? "");
                if (isSelected) {
                    radius += 2;
                }
                const hue = visual.hue;
                const lift = (1.9 + importance * 3.1) * visual.liftBoost;
                const depthY = py + lift;

                ctx.fillStyle = `hsla(${hue}, ${Math.round(visual.saturation * 0.9)}%, ${Math.round(visual.value * 0.44)}%, ${0.16 + (isSelected ? 0.1 : 0.02)})`;
                fillResourceShape(ctx, visual.shape, px, depthY, radius * 1.05);

                ctx.strokeStyle = `hsla(${hue}, 86%, 72%, ${0.16 + importance * 0.25})`;
                ctx.lineWidth = 0.45 + importance * 0.8;
                ctx.beginPath();
                ctx.moveTo(px, depthY - radius * 0.35);
                ctx.lineTo(px, py + radius * 0.35);
                ctx.stroke();

                const glow = ctx.createRadialGradient(px, py, 0, px, py, radius * 2.2);
                glow.addColorStop(0, `hsla(${hue}, ${visual.saturation}%, ${visual.value}%, ${isSelected ? 0.88 : (0.48 * visual.glowBoost)})`);
                glow.addColorStop(0.62, `hsla(${(hue + 24) % 360}, ${Math.max(52, visual.saturation - 10)}%, ${Math.max(44, visual.value - 34)}%, ${isSelected ? 0.42 : 0.22})`);
                glow.addColorStop(1, "rgba(18, 26, 38, 0)");
                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(px, py, radius * 2.2, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = `hsla(${hue}, ${Math.min(96, visual.saturation + 6)}%, ${Math.min(98, visual.value + 2)}%, ${isSelected ? 0.98 : 0.9})`;
                fillResourceShape(ctx, visual.shape, px, py, radius);

                ctx.strokeStyle = `hsla(${hue}, ${Math.max(68, visual.saturation - 8)}%, ${Math.max(52, visual.value - 34)}%, ${isSelected ? 0.95 : 0.58})`;
                ctx.lineWidth = isSelected ? 1.35 : 0.9;
                strokeResourceShape(ctx, visual.shape, px, py, radius);

                ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
                fillResourceShape(ctx, visual.shape, px - radius * 0.26, py - radius * 0.28, Math.max(0.55, radius * 0.28));

                graphNodeHits.push({
                    node,
                    x: nx,
                    y: ny,
                    radiusNorm: (radius + lift * 0.42) / Math.max(w, h),
                    nodeKind: "file",
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
                    const layerRadius = Math.max(0.9, 0.85 + (importance * 0.85));

                    ctx.strokeStyle = `hsla(${layerHue}, 86%, 70%, 0.28)`;
                    ctx.lineWidth = 0.55;
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(lx, ly);
                    ctx.stroke();

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
        const embedLayers = Array.isArray((graph as any).embed_layers)
            ? ((graph as any).embed_layers as any[])
            : [];
        const activeLayerLabels = embedLayers
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
        ctx.fillText(`embed layers: ${layerRows}`, 10, 62);
        ctx.fillStyle = "rgba(211, 232, 255, 0.82)";
        ctx.fillText(`concept presences: ${conceptCount}${conceptRows ? ` | ${conceptRows}` : ""}`, 10, 73);
        ctx.restore();

        if (selectedGraphNodeId) {
            const selected = fileNodes.find((row: any) => String(row.id) === selectedGraphNodeId);
            if (selected) {
                const sx = clamp01(Number(selected.x ?? 0.5)) * w;
                const sy = clamp01(Number(selected.y ?? 0.5)) * h;
                const selectedResourceKind = classifyFileResourceKind(selected);
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
                const boxY = Math.max(10, sy - 48);
                ctx.beginPath();
                ctx.roundRect(boxX, boxY, 198, 42, 6);
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
                ctx.fillStyle = "rgba(165, 212, 248, 0.9)";
                ctx.fillText(
                    `concept ${String(selected.concept_presence_label ?? "unassigned")}`,
                    boxX + 6,
                    boxY + 39,
                );
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

    const drawButterflies = (t: number, w: number, h: number) => {
        const count = 12;
        ctx.save();
        for(let i=0; i<count; i++) {
            const seed = i * 1.5 + t * 0.2;
            const bx = (Math.sin(seed) * 0.4 + 0.5) * w;
            const by = (Math.cos(seed * 0.8) * 0.4 + 0.5) * h;
            const size = 6 + Math.sin(t * 10 + i) * 2;
            const wingSpan = 4 + size;
            ctx.fillStyle = "rgba(10, 5, 15, 0.9)";
            ctx.shadowBlur = 8;
            ctx.shadowColor = i % 2 === 0 ? "rgba(255, 0, 100, 0.32)" : "rgba(150, 0, 255, 0.3)";
            ctx.beginPath();
            ctx.moveTo(bx, by);
            const flap = Math.sin(t * 15 + i) * 5;
            ctx.lineTo(bx - wingSpan, by - 5 - flap);
            ctx.lineTo(bx - wingSpan, by + 5 + flap);
            ctx.fill();
            ctx.beginPath();
            ctx.moveTo(bx, by);
            ctx.lineTo(bx + wingSpan, by - 5 - flap);
            ctx.lineTo(bx + wingSpan, by + 5 + flap);
            ctx.fill();
        }
        ctx.restore();
    };

    const draw = (ts: number) => {
        const currentSimulation = simulationRef.current;
        const namedForms = resolveNamedForms();
        const t = ts * 0.001;
        graphNodeHits = [];
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
        ctx.fillStyle = "rgba(4, 10, 18, 0.46)";
        ctx.fillRect(0, 0, w, h);
        const vignette = ctx.createRadialGradient(w * 0.5, h * 0.48, Math.min(w, h) * 0.18, w * 0.5, h * 0.52, Math.max(w, h) * 0.78);
        vignette.addColorStop(0, "rgba(16, 32, 52, 0.08)");
        vignette.addColorStop(0.5, "rgba(4, 10, 18, 0.28)");
        vignette.addColorStop(1, "rgba(2, 6, 12, 0.62)");
        ctx.fillStyle = vignette;
        ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = "screen";
        const audioCount = Math.max(0, currentSimulation?.audio || 0);
        const globalIntensity = Math.min(0.68, Math.log1p(audioCount) / 7.2);

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

        const showPresenceLayer = overlayView === "omni" || overlayView === "presence";
        const showFileImpactLayer = overlayView === "omni" || overlayView === "file-impact";
        const showFileGraphLayer = overlayView === "omni" || overlayView === "file-graph";
        const showCrawlerGraphLayer = overlayView === "omni" || overlayView === "crawler-graph";
        const showTruthGateLayer = overlayView === "omni" || overlayView === "truth-gate";
        const showLogicalLayer = overlayView === "omni" || overlayView === "logic";
        const showPainFieldLayer = overlayView === "omni" || overlayView === "pain-field";
        const showPresenceNodes =
            showPresenceLayer || showFileImpactLayer || showTruthGateLayer;

        if (showPresenceLayer) {
            drawEchoes(t, w, h, currentSimulation);
            drawRiverFlow(t, w, h, namedForms, currentSimulation);
            drawWitnessThreadFlow(t, w, h, namedForms, currentSimulation);
            drawGhostSentinel(t, w, h, currentSimulation);
            drawButterflies(t, w, h);
        }
        if (showFileImpactLayer) {
            drawFileInfluenceOverlay(t, w, h, namedForms, currentSimulation);
        }
        if (showLogicalLayer) {
            drawLogicalGraphOverlay(t, w, h, currentSimulation);
        }
        if (showFileGraphLayer) {
            drawFileCategoryGraph(t, w, h, currentSimulation);
        }
        if (showCrawlerGraphLayer) {
            drawCrawlerCategoryGraph(t, w, h, currentSimulation);
        }
        if (showTruthGateLayer) {
            drawTruthBindingOverlay(t, w, h, namedForms, currentSimulation);
        }
        if (showPainFieldLayer) {
            drawPainFieldOverlay(t, w, h, currentSimulation);
        }

        if(showPresenceNodes && namedForms.length > 2) {
            const loopPoints = namedForms.map((f: any) => ({x: f.x * w, y: f.y * h}));
            ctx.save();
            ctx.globalAlpha = 0.04 + globalIntensity * 0.1;
            ctx.setLineDash([2, 10]);
            ctx.strokeStyle = "rgba(" + (90 + globalIntensity * 80) + ", 220, " + (165 + globalIntensity * 45) + ", 0.28)";
            ctx.lineWidth = 0.4 + globalIntensity * 0.7;
            for(let i=0; i<loopPoints.length; i++) {
                for(let j=i+1; j<loopPoints.length; j++) {
                    const d = Math.hypot(loopPoints[i].x - loopPoints[j].x, loopPoints[i].y - loopPoints[j].y);
                    const maxDist = w * (0.22 + globalIntensity * 0.11);
                    if(d < maxDist) {
                        ctx.beginPath(); ctx.moveTo(loopPoints[i].x, loopPoints[i].y);
                        ctx.lineTo(loopPoints[j].x, loopPoints[j].y); ctx.stroke();
                    }
                }
            }
            ctx.restore();
            loopPoints.push(loopPoints[0]);
            ctx.strokeStyle = "rgba(145, 200, 235, " + (0.08 + globalIntensity * 0.1) + ")";
            ctx.lineWidth = 0.45 + globalIntensity * 0.7;
            ctx.beginPath(); ctx.moveTo(loopPoints[0].x, loopPoints[0].y);
            for(let i=1; i<loopPoints.length; i++) {
                const start = loopPoints[i-1]; const end = loopPoints[i];
                const mx = (start.x + end.x)/2; const my = (start.y + end.y)/2;
                const bend = Math.sin(t * 0.5 + i * 0.8) * 20 * (1 + globalIntensity);
                ctx.quadraticCurveTo(mx+bend, my-bend*0.5, end.x, end.y);
            }
            ctx.stroke();
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
            const telemetry = drawPresenceStatus(cx, cy, radiusBase, f.hue, entityState);
            const isHighlighted = highlighted === i;
            drawNebula(t, f, cx, cy, radiusBase, f.hue, intensity, isHighlighted);
            drawParticles(t, f, cx, cy, radiusBase, f.hue, intensity, isHighlighted);
            const isBottomHalf = f.y > 0.7;
            const labelY = isBottomHalf ? cy - radiusBase * 1.2 : cy + radiusBase * 1.2;
            ctx.save();
            ctx.globalCompositeOperation = "source-over";
            ctx.textAlign = "center";
            ctx.font = "600 12px serif";
            const enW = ctx.measureText(f.en).width;
            ctx.font = "500 10px sans-serif";
            const jaW = ctx.measureText(f.ja).width;
            const metricLine = "BPM " + telemetry.bpm + "  STB " + telemetry.stabilityPct + "%  RES " + telemetry.resonancePct + "%";
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
        }
        rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);
    const api = {
        pulseAt: (x: number, y: number, power: number, target = "particle_field") => { 
            ripple = { x, y, power, at: performance.now() }; 
            highlighted = -1;
            const baseUrl = window.location.port === "5173" ? "http://127.0.0.1:8787" : "";
            fetch(baseUrl + "/api/witness", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ type: "touch", target })
            }).catch(() => {});
        },
        singAll: () => {}
    };

    const onPointerDown = (event: PointerEvent) => {
        const rect = canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const xRatio = clamp01((event.clientX - rect.left) / rect.width);
        const yRatio = clamp01((event.clientY - rect.top) / rect.height);

        const graphHit = nearestGraphNodeAt(xRatio, yRatio);
        if (graphHit) {
            const node = graphHit.node;
            const nodeKind = graphHit.nodeKind;
            const resourceKind = resourceKindForNode(node);
            selectedGraphNodeId = String(node?.id ?? "");
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
                const selectedLabel = nodeKind === "crawler" ? "crawler node" : "file node";
                metaRef.current.textContent = `selected ${selectedLabel}: ${selectedGraphNodeLabel} [${resourceKindLabel(resourceKind)}]`;
            }
            const openUrl = openUrlForGraphNode(
                node,
                nodeKind === "crawler" ? "crawler" : "file",
            );
            const domain = String(node?.domain ?? "").trim();
            const worldscreenUrl = resolveWorldscreenUrl(openUrl, nodeKind, domain);
            if (worldscreenUrl) {
                const worldscreenNodeKind: "file" | "crawler" = nodeKind === "crawler" ? "crawler" : "file";
                setWorldscreen({
                    url: worldscreenUrl,
                    label: selectedGraphNodeLabel,
                    nodeKind: worldscreenNodeKind,
                    resourceKind,
                    view: worldscreenViewForNode(node, worldscreenNodeKind, resourceKind),
                    subtitle: worldscreenSubtitleForNode(node, worldscreenNodeKind, resourceKind),
                });
                if (metaRef.current) {
                    metaRef.current.textContent = `hologram opened: ${selectedGraphNodeLabel}`;
                }
            }
            return;
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
    };

    if (interactive) {
      canvas.addEventListener("pointerdown", onPointerDown);
    }
    if (onOverlayInit) onOverlayInit(api);
    return () => {
        if (interactive) {
          canvas.removeEventListener("pointerdown", onPointerDown);
        }
        cancelAnimationFrame(rafId);
    };
  }, [interactive, onOverlayInit, overlayView]);

  const activeOverlayView =
    OVERLAY_VIEW_OPTIONS.find((option) => option.id === overlayView) ?? OVERLAY_VIEW_OPTIONS[0];

  return (
    <div ref={containerRef} className="relative mt-3 border border-[rgba(36,31,26,0.16)] rounded-xl overflow-hidden bg-gradient-to-b from-[#0f1a1f] to-[#131b2a]">
      <canvas ref={canvasRef} style={{ height }} className="block w-full" />
      <canvas ref={overlayRef} style={{ height }} className="absolute inset-0 w-full touch-none" />
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
            {interactive ? (
              <p className="mt-1 text-[10px] text-[#c4d7f0]">tap node for hologram pop-out / 節点でホログラム起動</p>
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
      {interactive && worldscreen ? (
        <div className="absolute inset-0 z-20 pointer-events-none">
          <section className="pointer-events-auto absolute right-2 bottom-2 sm:right-4 sm:bottom-4 w-[min(96%,780px)] h-[min(68vh,540px)] rounded-2xl border border-[rgba(126,218,255,0.58)] bg-[linear-gradient(164deg,rgba(6,16,30,0.88),rgba(10,30,48,0.82),rgba(7,18,34,0.9))] backdrop-blur-[5px] shadow-[0_30px_90px_rgba(0,18,42,0.56)] overflow-hidden [transform:perspective(1300px)_rotateX(5deg)]">
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
                  onClick={() => setWorldscreen(null)}
                  className="text-xs px-2.5 py-1 rounded-md border border-[rgba(245,200,171,0.45)] text-[#ffe6d2] hover:bg-[rgba(187,120,78,0.2)]"
                >
                  close
                </button>
              </div>
            </header>
            <div className="relative h-[calc(100%-3.5rem)] p-2 sm:p-3">
              {worldscreen.view === "video" ? (
                <div className="h-full rounded-xl border border-[rgba(143,214,255,0.38)] bg-[radial-gradient(circle_at_30%_18%,rgba(89,214,255,0.2),rgba(5,17,29,0.86)_62%)] p-2 shadow-[inset_0_0_28px_rgba(70,204,255,0.2)]">
                  <video
                    controls
                    autoPlay
                    src={worldscreen.url}
                    className="h-full w-full rounded-lg object-contain bg-[rgba(6,14,22,0.9)]"
                  >
                    <track kind="captions" />
                  </video>
                </div>
              ) : null}

              {worldscreen.view === "editor" ? (
                <div className="h-full rounded-xl border border-[rgba(143,214,255,0.38)] bg-[linear-gradient(180deg,rgba(5,16,30,0.92),rgba(5,15,26,0.88))] overflow-hidden">
                  {editorPreview.status === "loading" ? (
                    <div className="h-full grid place-items-center text-sm text-[#b8e0ff]">loading file preview...</div>
                  ) : null}
                  {editorPreview.status === "error" ? (
                    <div className="h-full grid place-items-center text-sm text-[#ffd6bb]">{editorPreview.error}</div>
                  ) : null}
                  {editorPreview.status === "ready" ? (
                    <pre className="h-full overflow-auto px-3 py-2 text-[11px] leading-5 font-mono text-[#d9eeff]">
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
                  className="w-full h-full rounded-xl border border-[rgba(143,214,255,0.3)] bg-[#06101e]"
                  referrerPolicy="no-referrer"
                />
              ) : null}
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}
