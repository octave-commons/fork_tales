import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RefreshCw, Search, SlidersHorizontal } from "lucide-react";
import { relativeTime } from "../../app/time";
import { runtimeApiUrl } from "../../runtime/endpoints";

type RuntimeConfigValue =
  | number
  | RuntimeConfigValue[]
  | { [key: string]: RuntimeConfigValue };

interface RuntimeConfigModulePayload {
  constants: Record<string, RuntimeConfigValue>;
  constant_count: number;
  numeric_leaf_count: number;
}

interface RuntimeConfigPayload {
  ok: boolean;
  record?: string;
  runtime_config_version?: number;
  generated_at?: string;
  available_modules?: string[];
  module_count?: number;
  constant_count?: number;
  numeric_leaf_count?: number;
  modules?: Record<string, RuntimeConfigModulePayload>;
  error?: string;
}

interface RuntimeConfigMutationPayload {
  ok: boolean;
  error?: string;
  detail?: string;
  reset_count?: number;
  previous?: unknown;
  current?: unknown;
}

interface SimulationBootstrapLayerRow {
  id?: string;
  label?: string;
  collection?: string;
  space_id?: string;
  model_name?: string;
  file_count?: number;
  reference_count?: number;
  active?: boolean;
}

interface SimulationBootstrapSelectionPayload {
  graph_surface?: string;
  projection_mode?: string;
  projection_reason?: string;
  embed_layer_count?: number;
  active_embed_layer_count?: number;
  selected_embed_layers?: SimulationBootstrapLayerRow[];
}

interface SimulationBootstrapCompressionPayload {
  before_edges?: number;
  after_edges?: number;
  collapsed_edges?: number;
  edge_reduction_ratio?: number;
  edge_cap?: number;
  edge_cap_utilization?: number;
  overflow_nodes?: number;
  overflow_edges?: number;
  group_count?: number;
  active?: boolean;
}

interface SimulationBootstrapGroupRefPayload {
  group_id?: string;
  kind?: string;
  target?: string;
  surface_visible?: boolean;
  reasons?: Record<string, unknown>;
}

interface SimulationBootstrapMissingFilePayload {
  id?: string;
  node_id?: string;
  name?: string;
  kind?: string;
  path?: string;
  source_rel_path?: string;
  archive_rel_path?: string;
  archived_rel_path?: string;
  reason?: string;
  projection_group_refs?: SimulationBootstrapGroupRefPayload[];
}

interface SimulationBootstrapGraphDiffPayload {
  truth_file_node_count?: number;
  view_file_node_count?: number;
  truth_file_nodes_missing_from_view_count?: number;
  truth_file_nodes_missing_from_view?: SimulationBootstrapMissingFilePayload[];
  truth_file_nodes_missing_from_view_truncated?: boolean;
  view_projection_overflow_node_count?: number;
  projection_group_count?: number;
  projection_surface_visible_group_count?: number;
  projection_hidden_group_count?: number;
  projection_group_member_source_count?: number;
  ingested_item_count?: number;
  ingested_items_missing_from_truth_graph_count?: number;
  ingested_items_missing_from_truth_graph?: Array<Record<string, unknown>>;
  ingested_items_missing_from_truth_graph_truncated?: boolean;
  compaction_mode?: string;
  view_graph_reconstructable_from_truth_graph?: boolean;
  notes?: string[];
}

interface SimulationBootstrapReportPayload {
  ok?: boolean;
  record?: string;
  generated_at?: string;
  perspective?: string;
  error?: string;
  detail?: string;
  failed_phase?: string;
  selection?: SimulationBootstrapSelectionPayload;
  compression?: SimulationBootstrapCompressionPayload;
  graph_diff?: SimulationBootstrapGraphDiffPayload;
  phase_ms?: Record<string, unknown>;
}

interface SimulationBootstrapJobPayload {
  status?: string;
  job_id?: string;
  started_at?: string;
  updated_at?: string;
  completed_at?: string;
  phase?: string;
  phase_started_at?: string;
  phase_detail?: Record<string, unknown>;
  error?: string;
  request?: Record<string, unknown>;
  report?: SimulationBootstrapReportPayload | null;
}

interface SimulationBootstrapStatusPayload {
  ok?: boolean;
  record?: string;
  generated_at?: string;
  error?: string;
  detail?: string;
  job?: SimulationBootstrapJobPayload;
  report?: SimulationBootstrapReportPayload | null;
}

interface SimulationBootstrapQueuePayload {
  ok?: boolean;
  record?: string;
  status?: string;
  error?: string;
  detail?: string;
  job?: SimulationBootstrapJobPayload;
}

interface CatalogStreamProbeSnapshot {
  status: "idle" | "running" | "completed" | "failed" | "aborted";
  started_at?: string;
  updated_at?: string;
  finished_at?: string;
  stage?: string;
  heartbeat_count?: number;
  elapsed_ms?: number;
  rows_seen?: number;
  done_ok?: boolean;
  error?: string;
  sections?: Record<string, unknown>;
}

interface RuntimeConfigLeaf {
  moduleName: string;
  constantKey: string;
  leafId: string;
  pathTokens: string[];
  pathLabel: string;
  value: number;
  searchable: string;
}

interface RuntimeConfigEntry {
  key: string;
  value: RuntimeConfigValue;
  leafCount: number;
  preview: string;
  searchable: string;
  leaves: RuntimeConfigLeaf[];
}

interface RuntimeConfigModuleView {
  moduleName: string;
  constantCount: number;
  numericLeafCount: number;
  entries: RuntimeConfigEntry[];
}

function isRuntimeConfigMap(
  value: RuntimeConfigValue,
): value is { [key: string]: RuntimeConfigValue } {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 0.0001) || abs >= 10000) {
    return value.toExponential(4);
  }
  const compact = value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  return compact || "0";
}

function parseNumericInput(raw: string): number | null {
  const value = Number(raw.trim());
  if (!Number.isFinite(value)) {
    return null;
  }
  return value;
}

function numberFromUnknown(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value !== "string") {
    return null;
  }
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function formatDurationMs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "n/a";
  }
  const safeValue = Math.max(0, value);
  if (safeValue < 1000) {
    return `${Math.round(safeValue)}ms`;
  }
  if (safeValue < 10000) {
    return `${(safeValue / 1000).toFixed(1)}s`;
  }
  return `${Math.round(safeValue / 1000)}s`;
}

function normalizePhaseLabel(raw: string | undefined): string {
  const value = String(raw || "").trim();
  if (!value) {
    return "idle";
  }
  return value.replace(/_/g, " ");
}

function isNumericToken(token: string): boolean {
  return /^-?\d+$/.test(token.trim());
}

function buildPathLabel(pathTokens: string[]): string {
  if (pathTokens.length <= 0) {
    return "";
  }
  let label = "";
  pathTokens.forEach((token) => {
    if (isNumericToken(token)) {
      label += `[${token}]`;
    } else {
      label += label ? `.${token}` : token;
    }
  });
  return label;
}

function buildLeafId(
  moduleName: string,
  constantKey: string,
  pathTokens: string[],
): string {
  const tail = pathTokens.join("/");
  return `${moduleName}::${constantKey}::${tail}`;
}

function flattenNumericLeaves(
  value: RuntimeConfigValue,
  options: {
    moduleName: string;
    constantKey: string;
    pathTokens?: string[];
  },
): RuntimeConfigLeaf[] {
  const moduleName = options.moduleName;
  const constantKey = options.constantKey;
  const pathTokens = options.pathTokens ?? [];
  if (typeof value === "number") {
    const pathLabel = buildPathLabel(pathTokens);
    const leafId = buildLeafId(moduleName, constantKey, pathTokens);
    return [
      {
        moduleName,
        constantKey,
        leafId,
        pathTokens,
        pathLabel,
        value,
        searchable: `${moduleName} ${constantKey} ${pathLabel} ${formatNumber(value)}`.toLowerCase(),
      },
    ];
  }

  if (Array.isArray(value)) {
    const leaves: RuntimeConfigLeaf[] = [];
    value.forEach((item, index) => {
      leaves.push(
        ...flattenNumericLeaves(item, {
          moduleName,
          constantKey,
          pathTokens: [...pathTokens, String(index)],
        }),
      );
    });
    return leaves;
  }

  if (isRuntimeConfigMap(value)) {
    const leaves: RuntimeConfigLeaf[] = [];
    Object.keys(value)
      .sort((left, right) => left.localeCompare(right))
      .forEach((key) => {
        leaves.push(
          ...flattenNumericLeaves(value[key], {
            moduleName,
            constantKey,
            pathTokens: [...pathTokens, key],
          }),
        );
      });
    return leaves;
  }

  return [];
}

function countNumericLeaves(value: RuntimeConfigValue): number {
  if (typeof value === "number") {
    return 1;
  }
  if (Array.isArray(value)) {
    let total = 0;
    value.forEach((item) => {
      total += countNumericLeaves(item);
    });
    return total;
  }
  if (!isRuntimeConfigMap(value)) {
    return 0;
  }
  let total = 0;
  Object.values(value).forEach((item) => {
    total += countNumericLeaves(item);
  });
  return total;
}

function previewRuntimeConfigValue(value: RuntimeConfigValue): string {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (Array.isArray(value)) {
    const preview = value
      .slice(0, 5)
      .map((item) => previewRuntimeConfigValue(item))
      .join(", ");
    const suffix = value.length > 5 ? `, +${value.length - 5}` : "";
    return `[${preview}${suffix}]`;
  }
  if (!isRuntimeConfigMap(value)) {
    return "{}";
  }
  const entries = Object.entries(value);
  const preview = entries
    .slice(0, 4)
    .map(([key, item]) => `${key}:${previewRuntimeConfigValue(item)}`)
    .join(", ");
  const suffix = entries.length > 4 ? `, +${entries.length - 4}` : "";
  return `{${preview}${suffix}}`;
}

function normalizeModuleFilter(raw: string, availableModules: string[]): string {
  if (!raw || raw === "all") {
    return "all";
  }
  return availableModules.includes(raw) ? raw : "all";
}

function numbersClose(left: number, right: number): boolean {
  const delta = Math.abs(left - right);
  const scale = Math.max(1, Math.abs(left), Math.abs(right));
  return delta <= (scale * 1e-8);
}

function leafSliderSpec(leaf: RuntimeConfigLeaf, draft: number | null): {
  min: number;
  max: number;
  step: number;
} {
  const center = draft ?? leaf.value;
  const signature = `${leaf.constantKey} ${leaf.pathLabel}`.toUpperCase();
  if (signature.includes("FRICTION")) {
    const clampedCenter = Math.max(0.0, Math.min(2.0, center));
    const span = Math.max(0.05, Math.abs(clampedCenter) * 0.18);
    let min = Math.max(0.0, clampedCenter - span);
    let max = Math.min(2.0, clampedCenter + span);
    if ((max - min) < 0.0002) {
      min = Math.max(0.0, clampedCenter - 0.01);
      max = Math.min(2.0, clampedCenter + 0.01);
    }
    return {
      min,
      max,
      step: clampedCenter >= 1.0 ? 0.001 : 0.0001,
    };
  }

  if (signature.includes("DAMPING")) {
    const clampedCenter = Math.max(0.0, Math.min(4.0, center));
    const span = Math.max(0.05, Math.abs(clampedCenter) * 0.2);
    let min = Math.max(0.0, clampedCenter - span);
    let max = Math.min(4.0, clampedCenter + span);
    if ((max - min) < 0.0002) {
      min = Math.max(0.0, clampedCenter - 0.01);
      max = Math.min(4.0, clampedCenter + 0.01);
    }
    return {
      min,
      max,
      step: clampedCenter >= 1.0 ? 0.001 : 0.0001,
    };
  }

  const current = leaf.value;
  const focus = Math.max(Math.abs(current), Math.abs(draft ?? current), 0.000001);
  const span = focus < 1 ? 1 : focus * 1.4;
  const min = center - span;
  const max = center + span;
  let step = 0.0001;
  if (focus >= 1000) {
    step = 5;
  } else if (focus >= 100) {
    step = 1;
  } else if (focus >= 10) {
    step = 0.1;
  } else if (focus >= 1) {
    step = 0.01;
  } else if (focus >= 0.1) {
    step = 0.001;
  }
  return { min, max, step };
}

async function postRuntimeConfigMutation(
  path: string,
  payload: Record<string, unknown>,
): Promise<RuntimeConfigMutationPayload> {
  const response = await fetch(runtimeApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = (await response.json()) as RuntimeConfigMutationPayload;
  if (!response.ok || data.ok !== true) {
    return {
      ok: false,
      error: String(data.error || `request failed (${response.status})`),
      detail: String(data.detail || ""),
    };
  }
  return data;
}

function mutationNumericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return null;
}

const BOOTSTRAP_PERSPECTIVE_OPTIONS = [
  { id: "hybrid", label: "Hybrid" },
  { id: "causal-time", label: "Causal Time" },
  { id: "swimlanes", label: "Swimlanes" },
] as const;

export function RuntimeConfigPanel() {
  const [payload, setPayload] = useState<RuntimeConfigPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [moduleFilter, setModuleFilter] = useState("all");
  const [draftByLeafId, setDraftByLeafId] = useState<Record<string, string>>({});
  const [mutationMessage, setMutationMessage] = useState("");
  const [mutationError, setMutationError] = useState("");
  const [activeMutationLeafId, setActiveMutationLeafId] = useState("");
  const [bulkMutating, setBulkMutating] = useState(false);
  const [bootstrapPayload, setBootstrapPayload] = useState<SimulationBootstrapStatusPayload | null>(null);
  const [bootstrapPerspective, setBootstrapPerspective] = useState<string>("hybrid");
  const [bootstrapSyncInbox, setBootstrapSyncInbox] = useState(false);
  const [bootstrapLoading, setBootstrapLoading] = useState(false);
  const [bootstrapQueueing, setBootstrapQueueing] = useState(false);
  const [bootstrapMessage, setBootstrapMessage] = useState("");
  const [bootstrapError, setBootstrapError] = useState("");
  const [catalogStreamProbe, setCatalogStreamProbe] = useState<CatalogStreamProbeSnapshot>({
    status: "idle",
    rows_seen: 0,
    heartbeat_count: 0,
  });
  const [catalogStreamLogRows, setCatalogStreamLogRows] = useState<string[]>([]);
  const refreshRequestSeqRef = useRef(0);
  const bootstrapRequestSeqRef = useRef(0);
  const catalogStreamControllerRef = useRef<AbortController | null>(null);
  const catalogStreamRequestSeqRef = useRef(0);

  const refreshBootstrap = useCallback(async (withSpinner = true) => {
    const requestSeq = bootstrapRequestSeqRef.current + 1;
    bootstrapRequestSeqRef.current = requestSeq;
    if (withSpinner) {
      setBootstrapLoading(true);
    }
    try {
      const response = await fetch(runtimeApiUrl("/api/simulation/bootstrap"));
      const data = (await response.json()) as SimulationBootstrapStatusPayload;
      if (!response.ok || data.ok !== true) {
        throw new Error(String(data.error || `bootstrap status failed (${response.status})`));
      }
      if (requestSeq !== bootstrapRequestSeqRef.current) {
        return;
      }
      setBootstrapPayload(data);
      setBootstrapError("");
    } catch (fetchError) {
      if (requestSeq !== bootstrapRequestSeqRef.current) {
        return;
      }
      const message = fetchError instanceof Error ? fetchError.message : "bootstrap status fetch failed";
      setBootstrapError(message);
    } finally {
      if (withSpinner && requestSeq === bootstrapRequestSeqRef.current) {
        setBootstrapLoading(false);
      }
    }
  }, []);

  const appendCatalogStreamLog = useCallback((line: string) => {
    const text = String(line || "").trim();
    if (!text) {
      return;
    }
    setCatalogStreamLogRows((previous) => {
      const next = [text, ...previous];
      return next.slice(0, 18);
    });
  }, []);

  const stopCatalogStreamProbe = useCallback(() => {
    const controller = catalogStreamControllerRef.current;
    if (controller) {
      controller.abort();
      catalogStreamControllerRef.current = null;
    }
    const nowIso = new Date().toISOString();
    setCatalogStreamProbe((previous) => {
      if (previous.status !== "running") {
        return previous;
      }
      return {
        ...previous,
        status: "aborted",
        updated_at: nowIso,
        finished_at: nowIso,
      };
    });
    appendCatalogStreamLog("catalog stream aborted");
  }, [appendCatalogStreamLog]);

  const startCatalogStreamProbe = useCallback(async () => {
    if (catalogStreamControllerRef.current) {
      catalogStreamControllerRef.current.abort();
      catalogStreamControllerRef.current = null;
    }
    const requestSeq = catalogStreamRequestSeqRef.current + 1;
    catalogStreamRequestSeqRef.current = requestSeq;
    const controller = new AbortController();
    catalogStreamControllerRef.current = controller;

    const startedAt = new Date().toISOString();
    setCatalogStreamLogRows([]);
    setCatalogStreamProbe({
      status: "running",
      started_at: startedAt,
      updated_at: startedAt,
      stage: "connecting",
      heartbeat_count: 0,
      elapsed_ms: 0,
      rows_seen: 0,
    });
    appendCatalogStreamLog(`catalog stream start perspective=${bootstrapPerspective}`);

    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    try {
      const response = await fetch(
        runtimeApiUrl(
          `/api/catalog/stream?perspective=${encodeURIComponent(bootstrapPerspective)}&trim=1&chunk_rows=128`,
        ),
        {
          signal: controller.signal,
        },
      );
      if (!response.ok || !response.body) {
        throw new Error(`catalog stream failed (${response.status})`);
      }

      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let rowsSeen = 0;

      const processPayloadRow = (row: Record<string, unknown>) => {
        if (requestSeq !== catalogStreamRequestSeqRef.current) {
          return;
        }
        rowsSeen += 1;
        const rowType = String(row.type || "").trim().toLowerCase();
        const nowIso = new Date().toISOString();
        const elapsedMs = numberFromUnknown(row.elapsed_ms);
        const heartbeatCount = numberFromUnknown(row.heartbeat_count);
        const stage = String(row.stage || "").trim();
        const doneOk = row.ok === true || String(row.ok || "").trim().toLowerCase() === "true";
        const sections = (row.sections && typeof row.sections === "object" && !Array.isArray(row.sections))
          ? row.sections as Record<string, unknown>
          : undefined;

        setCatalogStreamProbe((previous) => {
          const next: CatalogStreamProbeSnapshot = {
            ...previous,
            updated_at: nowIso,
            rows_seen: rowsSeen,
            elapsed_ms: elapsedMs === null ? previous.elapsed_ms : elapsedMs,
            heartbeat_count: heartbeatCount === null
              ? previous.heartbeat_count
              : Math.max(0, Math.floor(heartbeatCount)),
            stage: stage || previous.stage,
          };
          if (rowType === "done") {
            next.status = doneOk ? "completed" : "failed";
            next.done_ok = doneOk;
            next.finished_at = nowIso;
            next.sections = sections;
          } else if (rowType === "error") {
            next.status = "failed";
            next.error = String(row.error || "catalog_stream_error").trim() || "catalog_stream_error";
            next.finished_at = nowIso;
          }
          return next;
        });

        if (rowType === "start") {
          appendCatalogStreamLog("stream connected");
        } else if (rowType === "progress") {
          appendCatalogStreamLog(stage ? `stage ${normalizePhaseLabel(stage)}` : "stage update");
        } else if (rowType === "heartbeat") {
          const heartbeatValue = heartbeatCount === null ? 0 : Math.max(0, Math.floor(heartbeatCount));
          if (heartbeatValue <= 2 || heartbeatValue % 5 === 0) {
            appendCatalogStreamLog(
              `heartbeat ${heartbeatValue} · elapsed ${formatDurationMs(elapsedMs)}`,
            );
          }
        } else if (rowType === "done") {
          appendCatalogStreamLog(
            `stream done ${doneOk ? "ok" : "failed"} · rows ${rowsSeen}`,
          );
        } else if (rowType === "error") {
          appendCatalogStreamLog(`stream error ${String(row.error || "catalog_stream_error")}`);
        }
      };

      while (true) {
        const readResult = await reader.read();
        if (readResult.done) {
          break;
        }
        buffer += decoder.decode(readResult.value, { stream: true });

        let newlineIndex = buffer.indexOf("\n");
        while (newlineIndex >= 0) {
          const rawLine = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          if (rawLine) {
            try {
              const payload = JSON.parse(rawLine);
              if (payload && typeof payload === "object" && !Array.isArray(payload)) {
                processPayloadRow(payload as Record<string, unknown>);
              }
            } catch {
              // ignore malformed stream row
            }
          }
          newlineIndex = buffer.indexOf("\n");
        }
      }

      const finalChunk = decoder.decode().trim();
      if (finalChunk) {
        try {
          const payload = JSON.parse(finalChunk);
          if (payload && typeof payload === "object" && !Array.isArray(payload)) {
            processPayloadRow(payload as Record<string, unknown>);
          }
        } catch {
          // ignore malformed final row
        }
      }

      if (requestSeq === catalogStreamRequestSeqRef.current) {
        const nowIso = new Date().toISOString();
        setCatalogStreamProbe((previous) => {
          if (previous.status !== "running") {
            return previous;
          }
          return {
            ...previous,
            status: "aborted",
            updated_at: nowIso,
            finished_at: nowIso,
            error: previous.error || "catalog_stream_ended_without_done",
          };
        });
      }
    } catch (streamError) {
      if (requestSeq !== catalogStreamRequestSeqRef.current) {
        return;
      }
      const nowIso = new Date().toISOString();
      const abortLike = streamError instanceof DOMException
        ? streamError.name === "AbortError"
        : streamError instanceof Error
          ? streamError.name === "AbortError"
          : false;
      if (abortLike) {
        setCatalogStreamProbe((previous) => ({
          ...previous,
          status: previous.status === "completed" ? previous.status : "aborted",
          updated_at: nowIso,
          finished_at: nowIso,
        }));
        appendCatalogStreamLog("catalog stream aborted");
      } else {
        const message = streamError instanceof Error ? streamError.message : "catalog stream failed";
        setCatalogStreamProbe((previous) => ({
          ...previous,
          status: "failed",
          updated_at: nowIso,
          finished_at: nowIso,
          error: message,
        }));
        appendCatalogStreamLog(`catalog stream failed: ${message}`);
      }
    } finally {
      try {
        await reader?.cancel();
      } catch {
        // ignore cancel failures
      }
      reader?.releaseLock();
      if (requestSeq === catalogStreamRequestSeqRef.current) {
        catalogStreamControllerRef.current = null;
      }
    }
  }, [appendCatalogStreamLog, bootstrapPerspective]);

  const queueBootstrap = useCallback(async () => {
    setBootstrapQueueing(true);
    setBootstrapError("");
    setBootstrapMessage("");
    try {
      const response = await fetch(runtimeApiUrl("/api/simulation/bootstrap"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          perspective: bootstrapPerspective,
          sync_inbox: bootstrapSyncInbox,
          include_simulation: false,
          wait: false,
        }),
      });
      const data = (await response.json()) as SimulationBootstrapQueuePayload;
      if (!response.ok || data.ok !== true) {
        throw new Error(String(data.error || `bootstrap queue failed (${response.status})`));
      }
      const queueStatus = String(data.status || data.job?.status || "queued").trim().toLowerCase() || "queued";
      const queuedJobId = String(data.job?.job_id || "").trim();
      setBootstrapMessage(
        queuedJobId
          ? `bootstrap ${queueStatus} · ${queuedJobId}`
          : `bootstrap ${queueStatus}`,
      );
      await refreshBootstrap(false);
    } catch (queueError) {
      const message = queueError instanceof Error ? queueError.message : "bootstrap queue failed";
      setBootstrapError(message);
    } finally {
      setBootstrapQueueing(false);
    }
  }, [bootstrapPerspective, bootstrapSyncInbox, refreshBootstrap]);

  const refreshConfig = useCallback(async (withSpinner = true) => {
    const requestSeq = refreshRequestSeqRef.current + 1;
    refreshRequestSeqRef.current = requestSeq;
    if (withSpinner) {
      setLoading(true);
    }
    setError("");
    try {
      const response = await fetch(runtimeApiUrl("/api/config"));
      const data = (await response.json()) as RuntimeConfigPayload;
      if (!response.ok || data.ok !== true) {
        throw new Error(String(data.error || `config request failed (${response.status})`));
      }
      if (!data.modules || typeof data.modules !== "object") {
        throw new Error("invalid config payload");
      }
      if (requestSeq !== refreshRequestSeqRef.current) {
        return;
      }
      setPayload(data);
      const availableModules = Array.isArray(data.available_modules)
        ? data.available_modules.map((item) => String(item || "")).filter(Boolean)
        : [];
      setModuleFilter((previous) => normalizeModuleFilter(previous, availableModules));
    } catch (fetchError) {
      if (requestSeq !== refreshRequestSeqRef.current) {
        return;
      }
      const message = fetchError instanceof Error ? fetchError.message : "config fetch failed";
      setError(message);
    } finally {
      if (withSpinner && requestSeq === refreshRequestSeqRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void refreshConfig(true);
    const interval = window.setInterval(() => {
      if (bulkMutating || activeMutationLeafId.length > 0 || Object.keys(draftByLeafId).length > 0) {
        return;
      }
      void refreshConfig(false);
    }, 10000);
    return () => {
      window.clearInterval(interval);
    };
  }, [activeMutationLeafId, bulkMutating, draftByLeafId, refreshConfig]);

  useEffect(() => {
    void refreshBootstrap(true);
  }, [refreshBootstrap]);

  useEffect(() => {
    const jobStatus = String(bootstrapPayload?.job?.status || "").trim().toLowerCase();
    const intervalMs = jobStatus === "running" ? 1500 : 12000;
    const interval = window.setInterval(() => {
      void refreshBootstrap(false);
    }, intervalMs);
    return () => {
      window.clearInterval(interval);
    };
  }, [bootstrapPayload?.job?.status, refreshBootstrap]);

  useEffect(() => {
    return () => {
      catalogStreamControllerRef.current?.abort();
      catalogStreamControllerRef.current = null;
    };
  }, []);

  const availableModules = useMemo(() => {
    if (!payload?.available_modules || !Array.isArray(payload.available_modules)) {
      return [];
    }
    return payload.available_modules.map((item) => String(item || "")).filter(Boolean);
  }, [payload?.available_modules]);

  const normalizedModuleFilter = normalizeModuleFilter(moduleFilter, availableModules);
  const normalizedSearch = searchQuery.trim().toLowerCase();

  const moduleViews = useMemo<RuntimeConfigModuleView[]>(() => {
    const modules = payload?.modules;
    if (!modules || typeof modules !== "object") {
      return [];
    }

    return Object.entries(modules)
      .sort(([left], [right]) => left.localeCompare(right))
      .filter(([moduleName]) => normalizedModuleFilter === "all" || moduleName === normalizedModuleFilter)
      .map(([moduleName, modulePayload]) => {
        const constants = modulePayload?.constants ?? {};
        const entries: RuntimeConfigEntry[] = Object.entries(constants)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, value]) => {
            const preview = previewRuntimeConfigValue(value);
            const leaves = flattenNumericLeaves(value, {
              moduleName,
              constantKey: key,
            });
            const leafSearchBlob = leaves
              .slice(0, 64)
              .map((leaf) => `${leaf.pathLabel} ${formatNumber(leaf.value)}`)
              .join(" ");
            return {
              key,
              value,
              leafCount: countNumericLeaves(value),
              preview,
              searchable: `${moduleName} ${key} ${preview} ${leafSearchBlob}`.toLowerCase(),
              leaves,
            };
          })
          .filter((entry) => !normalizedSearch || entry.searchable.includes(normalizedSearch));

        return {
          moduleName,
          constantCount: Number(modulePayload?.constant_count ?? 0),
          numericLeafCount: Number(modulePayload?.numeric_leaf_count ?? 0),
          entries,
        };
      })
      .filter((moduleView) => moduleView.entries.length > 0 || !normalizedSearch);
  }, [normalizedModuleFilter, normalizedSearch, payload?.modules]);

  const leafById = useMemo(() => {
    const map = new Map<string, RuntimeConfigLeaf>();
    moduleViews.forEach((moduleView) => {
      moduleView.entries.forEach((entry) => {
        entry.leaves.forEach((leaf) => {
          map.set(leaf.leafId, leaf);
        });
      });
    });
    return map;
  }, [moduleViews]);

  const matchedConstantCount = useMemo(
    () => moduleViews.reduce((sum, moduleView) => sum + moduleView.entries.length, 0),
    [moduleViews],
  );

  const matchedLeafCount = useMemo(
    () => moduleViews.reduce(
      (sum, moduleView) => sum + moduleView.entries.reduce((entrySum, entry) => entrySum + entry.leaves.length, 0),
      0,
    ),
    [moduleViews],
  );

  const bootstrapJob = bootstrapPayload?.job ?? null;
  const bootstrapReport = useMemo<SimulationBootstrapReportPayload | null>(() => {
    const directReport = bootstrapPayload?.report;
    if (directReport && typeof directReport === "object") {
      return directReport;
    }
    const jobReport = bootstrapJob?.report;
    if (jobReport && typeof jobReport === "object") {
      return jobReport;
    }
    return null;
  }, [bootstrapJob?.report, bootstrapPayload?.report]);

  const bootstrapJobStatus = String(bootstrapJob?.status || "idle").trim().toLowerCase() || "idle";
  const bootstrapJobPhase = String(bootstrapJob?.phase || "").trim().toLowerCase();
  const bootstrapJobId = String(bootstrapJob?.job_id || "").trim();
  const bootstrapJobIdShort = bootstrapJobId.length > 24 ? `…${bootstrapJobId.slice(-16)}` : bootstrapJobId || "n/a";
  const bootstrapPhaseDetail = (
    bootstrapJob?.phase_detail && typeof bootstrapJob.phase_detail === "object" && !Array.isArray(bootstrapJob.phase_detail)
      ? bootstrapJob.phase_detail
      : {}
  ) as Record<string, unknown>;
  const bootstrapHeartbeatCount = numberFromUnknown(bootstrapPhaseDetail.heartbeat_count);
  const bootstrapPhaseElapsedMs = numberFromUnknown(bootstrapPhaseDetail.phase_elapsed_ms);
  const bootstrapSelection = bootstrapReport?.selection ?? null;
  const bootstrapCompression = bootstrapReport?.compression ?? null;
  const bootstrapGraphDiff = bootstrapReport?.graph_diff ?? null;
  const bootstrapPhaseMs = bootstrapReport?.phase_ms ?? {};
  const bootstrapCatalogMs = numberFromUnknown(bootstrapPhaseMs.catalog);
  const bootstrapSimulationMs = numberFromUnknown(bootstrapPhaseMs.simulation);
  const bootstrapCacheStoreMs = numberFromUnknown(bootstrapPhaseMs.cache_store);
  const bootstrapBeforeEdges = numberFromUnknown(bootstrapCompression?.before_edges);
  const bootstrapAfterEdges = numberFromUnknown(bootstrapCompression?.after_edges);
  const bootstrapCollapsedEdges = numberFromUnknown(bootstrapCompression?.collapsed_edges);
  const bootstrapLayerCount = numberFromUnknown(bootstrapSelection?.embed_layer_count);
  const bootstrapActiveLayerCount = numberFromUnknown(bootstrapSelection?.active_embed_layer_count);
  const bootstrapTruthNodeCount = numberFromUnknown(bootstrapGraphDiff?.truth_file_node_count);
  const bootstrapViewNodeCount = numberFromUnknown(bootstrapGraphDiff?.view_file_node_count);
  const bootstrapMissingTruthToViewCount = numberFromUnknown(
    bootstrapGraphDiff?.truth_file_nodes_missing_from_view_count,
  );
  const bootstrapOverflowNodeCount = numberFromUnknown(
    bootstrapGraphDiff?.view_projection_overflow_node_count,
  );
  const bootstrapCompactionMode = String(bootstrapGraphDiff?.compaction_mode || "").trim();
  const bootstrapMissingTruthToViewRows = Array.isArray(bootstrapGraphDiff?.truth_file_nodes_missing_from_view)
    ? bootstrapGraphDiff?.truth_file_nodes_missing_from_view
    : [];
  const bootstrapMissingIngestRows = Array.isArray(bootstrapGraphDiff?.ingested_items_missing_from_truth_graph)
    ? bootstrapGraphDiff?.ingested_items_missing_from_truth_graph
    : [];
  const bootstrapProjectionGroupCount = numberFromUnknown(bootstrapGraphDiff?.projection_group_count);
  const bootstrapProjectionHiddenGroupCount = numberFromUnknown(
    bootstrapGraphDiff?.projection_hidden_group_count,
  );
  const bootstrapProjectionVisibleGroupCount = numberFromUnknown(
    bootstrapGraphDiff?.projection_surface_visible_group_count,
  );
  const bootstrapIngestedItemCount = numberFromUnknown(bootstrapGraphDiff?.ingested_item_count);
  const bootstrapMissingIngestCount = numberFromUnknown(
    bootstrapGraphDiff?.ingested_items_missing_from_truth_graph_count,
  );
  const bootstrapGraphDiffNotes = Array.isArray(bootstrapGraphDiff?.notes)
    ? bootstrapGraphDiff.notes.map((row) => String(row || "").trim()).filter(Boolean)
    : [];
  const bootstrapIsRunning = bootstrapJobStatus === "running";
  const bootstrapCanQueue = !bootstrapQueueing && !bootstrapIsRunning;
  const bootstrapStatusTone = bootstrapJobStatus === "failed"
    ? "text-[#ffcfbf]"
    : bootstrapJobStatus === "completed"
      ? "text-[#b6f0c0]"
      : bootstrapJobStatus === "running"
        ? "text-[#9ec7dd]"
        : "text-ink";
  const catalogStreamStatus = String(catalogStreamProbe.status || "idle").trim().toLowerCase() || "idle";
  const catalogStreamStatusTone = catalogStreamStatus === "failed"
    ? "text-[#ffcfbf]"
    : catalogStreamStatus === "completed"
      ? "text-[#b6f0c0]"
      : catalogStreamStatus === "running"
        ? "text-[#9ec7dd]"
        : catalogStreamStatus === "aborted"
          ? "text-[#e7d6b4]"
          : "text-ink";
  const catalogStreamRowsSeen = numberFromUnknown(catalogStreamProbe.rows_seen);
  const catalogStreamHeartbeatCount = numberFromUnknown(catalogStreamProbe.heartbeat_count);
  const catalogStreamElapsedMs = numberFromUnknown(catalogStreamProbe.elapsed_ms);
  const catalogStreamIsRunning = catalogStreamStatus === "running";

  const editedLeafIds = useMemo(() => {
    const edited: string[] = [];
    Object.entries(draftByLeafId).forEach(([leafId, draftValue]) => {
      const leaf = leafById.get(leafId);
      if (!leaf) {
        return;
      }
      const parsed = parseNumericInput(draftValue);
      if (parsed === null) {
        return;
      }
      if (!numbersClose(parsed, leaf.value)) {
        edited.push(leafId);
      }
    });
    return edited;
  }, [draftByLeafId, leafById]);

  const setLeafDraft = useCallback((leafId: string, nextValue: number) => {
    setDraftByLeafId((previous) => ({
      ...previous,
      [leafId]: formatNumber(nextValue),
    }));
  }, []);

  const applyLeaf = useCallback(async (
    leaf: RuntimeConfigLeaf,
    nextValue: number,
  ) => {
    setActiveMutationLeafId(leaf.leafId);
    setMutationError("");
    setMutationMessage("");
    const result = await postRuntimeConfigMutation("/api/config/update", {
      module: leaf.moduleName,
      key: leaf.constantKey,
      path: leaf.pathTokens,
      value: nextValue,
    });
    if (!result.ok) {
      setMutationError(String(result.error || "update failed"));
      setActiveMutationLeafId("");
      return;
    }
    setDraftByLeafId((previous) => {
      const next = { ...previous };
      delete next[leaf.leafId];
      return next;
    });
    const currentValue = mutationNumericValue(result.current);
    setMutationMessage(
      currentValue === null
        ? `updated ${leaf.moduleName}.${leaf.constantKey}${leaf.pathLabel ? `.${leaf.pathLabel}` : ""}`
        : `updated ${leaf.moduleName}.${leaf.constantKey}${leaf.pathLabel ? `.${leaf.pathLabel}` : ""} -> ${formatNumber(currentValue)}`,
    );
    await refreshConfig(false);
    setActiveMutationLeafId("");
  }, [refreshConfig]);

  const resetLeaf = useCallback(async (leaf: RuntimeConfigLeaf) => {
    setActiveMutationLeafId(leaf.leafId);
    setMutationError("");
    setMutationMessage("");
    const result = await postRuntimeConfigMutation("/api/config/reset", {
      module: leaf.moduleName,
      key: leaf.constantKey,
      path: leaf.pathTokens,
    });
    if (!result.ok) {
      setMutationError(String(result.error || "reset failed"));
      setActiveMutationLeafId("");
      return;
    }
    setDraftByLeafId((previous) => {
      const next = { ...previous };
      delete next[leaf.leafId];
      return next;
    });
    setMutationMessage(`reset ${leaf.moduleName}.${leaf.constantKey}${leaf.pathLabel ? `.${leaf.pathLabel}` : ""}`);
    await refreshConfig(false);
    setActiveMutationLeafId("");
  }, [refreshConfig]);

  const applyEdited = useCallback(async () => {
    if (editedLeafIds.length <= 0) {
      return;
    }
    setBulkMutating(true);
    setMutationError("");
    setMutationMessage("");
    let applied = 0;
    for (const leafId of editedLeafIds) {
      const leaf = leafById.get(leafId);
      if (!leaf) {
        continue;
      }
      const parsed = parseNumericInput(draftByLeafId[leafId] ?? "");
      if (parsed === null) {
        continue;
      }
      const result = await postRuntimeConfigMutation("/api/config/update", {
        module: leaf.moduleName,
        key: leaf.constantKey,
        path: leaf.pathTokens,
        value: parsed,
      });
      if (!result.ok) {
        setMutationError(String(result.error || `update failed on ${leaf.constantKey}`));
        setBulkMutating(false);
        return;
      }
      applied += 1;
    }

    if (applied > 0) {
      setDraftByLeafId((previous) => {
        const next = { ...previous };
        editedLeafIds.forEach((leafId) => {
          delete next[leafId];
        });
        return next;
      });
    }
    setMutationMessage(`applied ${applied} edited values`);
    await refreshConfig(false);
    setBulkMutating(false);
  }, [draftByLeafId, editedLeafIds, leafById, refreshConfig]);

  const resetAll = useCallback(async () => {
    setBulkMutating(true);
    setMutationError("");
    setMutationMessage("");
    const result = await postRuntimeConfigMutation("/api/config/reset", {});
    if (!result.ok) {
      setMutationError(String(result.error || "reset all failed"));
      setBulkMutating(false);
      return;
    }
    setDraftByLeafId({});
    setMutationMessage(`reset ${Number(result.reset_count ?? 0)} values to defaults`);
    await refreshConfig(false);
    setBulkMutating(false);
  }, [refreshConfig]);

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-[rgba(174,129,255,0.34)] bg-[rgba(39,40,34,0.9)] p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-ink flex items-center gap-2">
              <SlidersHorizontal size={15} />
              Runtime Config Interface
            </p>
            <p className="text-xs text-muted mt-1">
              Live controls for numeric constants exposed by <code>/api/config</code>.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                void refreshConfig(true);
              }}
              className="border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(55,56,48,0.92)]"
            >
              <span className="inline-flex items-center gap-1.5">
                <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
                Refresh
              </span>
            </button>
            <button
              type="button"
              onClick={() => {
                void applyEdited();
              }}
              disabled={editedLeafIds.length === 0 || bulkMutating}
              className="border border-[var(--line)] rounded-md bg-[rgba(44,67,39,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(63,94,56,0.92)] disabled:opacity-50"
            >
              Apply Edited ({editedLeafIds.length})
            </button>
            <button
              type="button"
              onClick={() => {
                void resetAll();
              }}
              disabled={bulkMutating}
              className="border border-[var(--line)] rounded-md bg-[rgba(73,45,45,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(96,58,58,0.92)] disabled:opacity-50"
            >
              Reset Runtime Defaults
            </button>
          </div>
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-4">
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">modules</p>
            <p className="text-sm font-semibold text-ink">{payload?.module_count ?? 0}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">constants</p>
            <p className="text-sm font-semibold text-ink">{payload?.constant_count ?? 0}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">numeric leaves</p>
            <p className="text-sm font-semibold text-ink">{payload?.numeric_leaf_count ?? 0}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">matched leaves</p>
            <p className="text-sm font-semibold text-ink">{matchedLeafCount}</p>
          </div>
        </div>

        <div className="mt-3 rounded-md border border-[rgba(126,166,192,0.28)] bg-[rgba(16,22,27,0.74)] p-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold text-[#cfe7f8]">Simulation Bootstrap</p>
              <p className="text-[11px] text-[#9ec7dd] mt-1">
                Queue a from-scratch simulation rebuild and inspect live phase heartbeats.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <label className="text-[11px] text-muted" htmlFor="bootstrap-perspective-select">
                perspective
              </label>
              <select
                id="bootstrap-perspective-select"
                value={bootstrapPerspective}
                onChange={(event) => {
                  setBootstrapPerspective(event.currentTarget.value);
                }}
                className="border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.94)] px-2 py-1 text-xs text-ink"
              >
                {BOOTSTRAP_PERSPECTIVE_OPTIONS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
              <label className="inline-flex items-center gap-1 text-[11px] text-muted">
                <input
                  type="checkbox"
                  checked={bootstrapSyncInbox}
                  onChange={(event) => {
                    setBootstrapSyncInbox(event.currentTarget.checked);
                  }}
                />
                sync inbox
              </label>
              <button
                type="button"
                onClick={() => {
                  void queueBootstrap();
                }}
                disabled={!bootstrapCanQueue}
                className="border border-[rgba(126,196,156,0.5)] rounded-md bg-[rgba(35,70,50,0.9)] px-3 py-1.5 text-xs font-semibold text-[#def7ea] hover:bg-[rgba(49,96,69,0.92)] disabled:opacity-50"
              >
                {bootstrapQueueing ? "queueing..." : bootstrapIsRunning ? "bootstrap running" : "queue bootstrap"}
              </button>
              <button
                type="button"
                onClick={() => {
                  void refreshBootstrap(true);
                }}
                className="border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(55,56,48,0.92)]"
              >
                <span className="inline-flex items-center gap-1.5">
                  <RefreshCw size={12} className={bootstrapLoading ? "animate-spin" : ""} />
                  status
                </span>
              </button>
            </div>
          </div>

          <div className="mt-2 rounded-md border border-[rgba(126,166,192,0.24)] bg-[rgba(10,14,19,0.54)] p-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-[11px] text-[#cfe7f8] font-semibold">Catalog Stream Probe</p>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => {
                    void startCatalogStreamProbe();
                  }}
                  disabled={catalogStreamIsRunning}
                  className="border border-[rgba(126,196,156,0.5)] rounded-md bg-[rgba(35,70,50,0.9)] px-2.5 py-1 text-[11px] font-semibold text-[#def7ea] hover:bg-[rgba(49,96,69,0.92)] disabled:opacity-50"
                >
                  {catalogStreamIsRunning ? "streaming..." : "start stream"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    stopCatalogStreamProbe();
                  }}
                  disabled={!catalogStreamIsRunning}
                  className="border border-[rgba(202,150,134,0.48)] rounded-md bg-[rgba(70,42,39,0.88)] px-2.5 py-1 text-[11px] font-semibold text-[#ffe0d7] hover:bg-[rgba(96,56,52,0.9)] disabled:opacity-50"
                >
                  stop
                </button>
              </div>
            </div>
            <p className="text-[11px] text-muted mt-1">
              status <code className={catalogStreamStatusTone}>{catalogStreamStatus}</code>
              {catalogStreamProbe.stage ? (
                <>
                  {" "}| stage <code>{normalizePhaseLabel(catalogStreamProbe.stage)}</code>
                </>
              ) : null}
              {catalogStreamRowsSeen !== null ? (
                <>
                  {" "}| rows <code>{String(Math.max(0, Math.floor(catalogStreamRowsSeen)))}</code>
                </>
              ) : null}
              {catalogStreamHeartbeatCount !== null ? (
                <>
                  {" "}| heartbeat <code>{String(Math.max(0, Math.floor(catalogStreamHeartbeatCount)))}</code>
                </>
              ) : null}
              {catalogStreamElapsedMs !== null ? (
                <>
                  {" "}| elapsed <code>{formatDurationMs(catalogStreamElapsedMs)}</code>
                </>
              ) : null}
              {catalogStreamProbe.updated_at ? (
                <>
                  {" "}| updated <code>{relativeTime(catalogStreamProbe.updated_at)}</code>
                </>
              ) : null}
            </p>
            {catalogStreamProbe.error ? (
              <p className="text-[11px] text-[#ffcfbf] mt-1">
                error <code>{catalogStreamProbe.error}</code>
              </p>
            ) : null}
            {catalogStreamProbe.sections && Object.keys(catalogStreamProbe.sections).length > 0 ? (
              <p className="text-[11px] text-[#9ec7dd] mt-1 break-all">
                sections <code>{Object.keys(catalogStreamProbe.sections).sort().join(", ")}</code>
              </p>
            ) : null}
            {catalogStreamLogRows.length > 0 ? (
              <div className="mt-2 max-h-24 overflow-auto space-y-1">
                {catalogStreamLogRows.map((row, index) => (
                  <p key={`catalog-stream-log-${index}-${row}`} className="text-[10px] text-[#a9cee5]">{row}</p>
                ))}
              </div>
            ) : null}
          </div>

          <div className="mt-3 grid gap-2 sm:grid-cols-4">
            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted">job status</p>
              <p className={`text-sm font-semibold ${bootstrapStatusTone}`}>{bootstrapJobStatus || "idle"}</p>
              <p className="text-[11px] text-muted mt-1">
                id <code>{bootstrapJobIdShort}</code>
              </p>
            </div>
            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted">phase</p>
              <p className="text-sm font-semibold text-ink">{normalizePhaseLabel(bootstrapJobPhase)}</p>
              <p className="text-[11px] text-muted mt-1">
                heartbeat <code>{bootstrapHeartbeatCount === null ? "-" : String(Math.max(0, Math.floor(bootstrapHeartbeatCount)))}</code>
                {" "}· elapsed <code>{formatDurationMs(bootstrapPhaseElapsedMs)}</code>
              </p>
            </div>
            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted">selection</p>
              <p className="text-sm font-semibold text-ink">
                {String(bootstrapSelection?.graph_surface || "n/a")}
              </p>
              <p className="text-[11px] text-muted mt-1">
                layers <code>{bootstrapActiveLayerCount === null ? "0" : String(Math.max(0, Math.round(bootstrapActiveLayerCount)))}</code>
                /<code>{bootstrapLayerCount === null ? "0" : String(Math.max(0, Math.round(bootstrapLayerCount)))}</code>
              </p>
            </div>
            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted">compression</p>
              <p className="text-sm font-semibold text-ink">
                {bootstrapBeforeEdges === null ? "0" : String(Math.max(0, Math.round(bootstrapBeforeEdges)))}
                {" -> "}
                {bootstrapAfterEdges === null ? "0" : String(Math.max(0, Math.round(bootstrapAfterEdges)))}
              </p>
              <p className="text-[11px] text-muted mt-1">
                collapsed <code>{bootstrapCollapsedEdges === null ? "0" : String(Math.max(0, Math.round(bootstrapCollapsedEdges)))}</code>
              </p>
            </div>
          </div>

          <p className="text-[11px] text-muted mt-2">
            phase timings: catalog <code>{formatDurationMs(bootstrapCatalogMs)}</code>
            {" "}| simulation <code>{formatDurationMs(bootstrapSimulationMs)}</code>
            {" "}| cache store <code>{formatDurationMs(bootstrapCacheStoreMs)}</code>
            {bootstrapSelection?.projection_reason ? (
              <>
                {" "}| reason <code>{bootstrapSelection.projection_reason}</code>
              </>
            ) : null}
            {bootstrapReport?.failed_phase ? (
              <>
                {" "}| failed phase <code>{bootstrapReport.failed_phase}</code>
              </>
            ) : null}
            {bootstrapJob?.phase_started_at ? (
              <>
                {" "}| phase since <code>{relativeTime(bootstrapJob.phase_started_at)}</code>
              </>
            ) : null}
            {bootstrapJob?.updated_at ? (
              <>
                {" "}| updated <code>{relativeTime(bootstrapJob.updated_at)}</code>
              </>
            ) : null}
            {bootstrapReport?.generated_at ? (
              <>
                {" "}| report <code>{relativeTime(bootstrapReport.generated_at)}</code>
              </>
            ) : null}
          </p>

          <div className="mt-2 grid gap-2 sm:grid-cols-3">
            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.8)] px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted">truth to view files</p>
                <p className="text-sm font-semibold text-ink">
                  {bootstrapTruthNodeCount === null ? "0" : String(Math.max(0, Math.round(bootstrapTruthNodeCount)))}
                  {" to "}
                  {bootstrapViewNodeCount === null ? "0" : String(Math.max(0, Math.round(bootstrapViewNodeCount)))}
                </p>
              <p className="text-[11px] text-muted mt-1">
                missing in view <code>{bootstrapMissingTruthToViewCount === null ? "0" : String(Math.max(0, Math.round(bootstrapMissingTruthToViewCount)))}</code>
              </p>
            </div>
            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.8)] px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted">projection bundles</p>
              <p className="text-sm font-semibold text-ink">
                {bootstrapProjectionVisibleGroupCount === null ? "0" : String(Math.max(0, Math.round(bootstrapProjectionVisibleGroupCount)))}
                {" visible / "}
                {bootstrapProjectionGroupCount === null ? "0" : String(Math.max(0, Math.round(bootstrapProjectionGroupCount)))}
              </p>
              <p className="text-[11px] text-muted mt-1">
                hidden groups <code>{bootstrapProjectionHiddenGroupCount === null ? "0" : String(Math.max(0, Math.round(bootstrapProjectionHiddenGroupCount)))}</code>
                {" "}| overflow nodes <code>{bootstrapOverflowNodeCount === null ? "0" : String(Math.max(0, Math.round(bootstrapOverflowNodeCount)))}</code>
              </p>
            </div>
            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.8)] px-3 py-2">
              <p className="text-[10px] uppercase tracking-wide text-muted">ingested coverage</p>
              <p className="text-sm font-semibold text-ink">
                items <code>{bootstrapIngestedItemCount === null ? "0" : String(Math.max(0, Math.round(bootstrapIngestedItemCount)))}</code>
              </p>
              <p className="text-[11px] text-muted mt-1">
                not in truth graph <code>{bootstrapMissingIngestCount === null ? "0" : String(Math.max(0, Math.round(bootstrapMissingIngestCount)))}</code>
              </p>
            </div>
          </div>

          <p className="text-[11px] text-muted mt-2">
            compaction mode <code>{bootstrapCompactionMode ? normalizePhaseLabel(bootstrapCompactionMode) : "n/a"}</code>
            {bootstrapGraphDiff?.view_graph_reconstructable_from_truth_graph ? (
              <>
                {" "}| reconstruction <code>truth_to_view_derivable</code>
              </>
            ) : null}
            {bootstrapGraphDiff?.truth_file_nodes_missing_from_view_truncated ? (
              <>
                {" "}| missing list truncated
              </>
            ) : null}
            {bootstrapGraphDiff?.ingested_items_missing_from_truth_graph_truncated ? (
              <>
                {" "}| ingest list truncated
              </>
            ) : null}
          </p>

          {bootstrapGraphDiffNotes.length > 0 ? (
            <div className="mt-2 rounded-md border border-[rgba(126,166,192,0.24)] bg-[rgba(10,14,19,0.54)] px-2 py-1.5">
              {bootstrapGraphDiffNotes.slice(0, 4).map((note) => (
                <p key={`bootstrap-note-${note}`} className="text-[11px] text-[#a9cee5]">{note}</p>
              ))}
            </div>
          ) : null}

          <details className="mt-2 rounded-md border border-[rgba(126,166,192,0.24)] bg-[rgba(10,14,19,0.54)] px-2 py-1.5">
            <summary className="cursor-pointer text-[11px] text-[#cfe7f8]">
              files ingested in truth graph but missing from view graph ({bootstrapMissingTruthToViewRows.length})
            </summary>
            {bootstrapMissingTruthToViewRows.length <= 0 ? (
              <p className="mt-2 text-[11px] text-muted">No missing truth-graph file rows in current report.</p>
            ) : (
              <div className="mt-2 max-h-44 overflow-auto space-y-1">
                {bootstrapMissingTruthToViewRows.slice(0, 180).map((row) => {
                  const path = String(row?.path || row?.source_rel_path || row?.archive_rel_path || row?.id || "").trim() || "(unknown)";
                  const reason = String(row?.reason || "unknown").trim() || "unknown";
                  const refs = Array.isArray(row?.projection_group_refs)
                    ? row.projection_group_refs as SimulationBootstrapGroupRefPayload[]
                    : [];
                  const rowId = String(row?.id || row?.node_id || path || "row").trim() || "row";
                  return (
                    <div key={`bootstrap-missing-row-${rowId}`} className="rounded border border-[rgba(126,166,192,0.2)] bg-[rgba(13,20,26,0.7)] px-2 py-1">
                      <p className="text-[11px] text-[#d9ecff] break-all"><code>{path}</code></p>
                      <p className="text-[10px] text-[#9ec7dd]">
                        reason <code>{normalizePhaseLabel(reason)}</code>
                        {refs.length > 0 ? (
                          <>
                            {" "}| groups <code>{refs.map((ref) => String(ref.group_id || "").trim()).filter(Boolean).join(", ") || "n/a"}</code>
                          </>
                        ) : null}
                      </p>
                    </div>
                  );
                })}
              </div>
            )}
          </details>

          <details className="mt-2 rounded-md border border-[rgba(126,166,192,0.24)] bg-[rgba(10,14,19,0.54)] px-2 py-1.5">
            <summary className="cursor-pointer text-[11px] text-[#cfe7f8]">
              ingested catalog items missing from truth graph ({bootstrapMissingIngestRows.length})
            </summary>
            {bootstrapMissingIngestRows.length <= 0 ? (
              <p className="mt-2 text-[11px] text-muted">No ingest-vs-truth mismatches in current report.</p>
            ) : (
              <div className="mt-2 max-h-44 overflow-auto space-y-1">
                {bootstrapMissingIngestRows.slice(0, 180).map((row) => {
                  const path = String(row.path || row.rel_path || row.name || "").trim() || "(unknown)";
                  const reason = String(row.reason || "unknown").trim() || "unknown";
                  const kind = String(row.kind || "").trim();
                  const rowId = `${path}|${kind || "unknown"}`;
                  return (
                    <div key={`bootstrap-ingest-missing-row-${rowId}`} className="rounded border border-[rgba(126,166,192,0.2)] bg-[rgba(13,20,26,0.7)] px-2 py-1">
                      <p className="text-[11px] text-[#d9ecff] break-all"><code>{path}</code></p>
                      <p className="text-[10px] text-[#9ec7dd]">
                        reason <code>{normalizePhaseLabel(reason)}</code>
                        {kind ? (
                          <>
                            {" "}| kind <code>{kind}</code>
                          </>
                        ) : null}
                      </p>
                    </div>
                  );
                })}
              </div>
            )}
          </details>

          {bootstrapMessage ? <p className="text-[11px] text-[#b6f0c0] mt-2">{bootstrapMessage}</p> : null}
          {bootstrapError ? <p className="text-[11px] text-[#ffcfbf] mt-2">{bootstrapError}</p> : null}
          {bootstrapJob?.error ? <p className="text-[11px] text-[#ffcfbf] mt-2">{bootstrapJob.error}</p> : null}
          {bootstrapReport?.error ? <p className="text-[11px] text-[#ffcfbf] mt-2">{bootstrapReport.error}</p> : null}
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <label className="text-[11px] text-muted" htmlFor="runtime-config-module-filter">
            module
          </label>
          <select
            id="runtime-config-module-filter"
            value={normalizedModuleFilter}
            onChange={(event) => {
              setModuleFilter(event.currentTarget.value);
            }}
            className="border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.94)] px-2 py-1 text-xs text-ink"
          >
            <option value="all">all modules</option>
            {availableModules.map((moduleName) => (
              <option key={moduleName} value={moduleName}>
                {moduleName}
              </option>
            ))}
          </select>

          <div className="inline-flex items-center gap-1 border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.94)] px-2 py-1">
            <Search size={12} className="text-muted" />
            <input
              value={searchQuery}
              onChange={(event) => {
                setSearchQuery(event.currentTarget.value);
              }}
              placeholder="search constants and leaves"
              className="bg-transparent text-xs text-ink outline-none w-[20rem] max-w-[56vw]"
            />
          </div>
        </div>

        <p className="text-[11px] text-muted mt-2">
          matches <code>{matchedConstantCount}</code> constants · <code>{matchedLeafCount}</code> leaves
          {payload?.generated_at ? (
            <>
              {" "}| refreshed <code>{relativeTime(payload.generated_at)}</code>
            </>
          ) : null}
          {payload?.record ? (
            <>
              {" "}| record <code>{payload.record}</code>
            </>
          ) : null}
          {typeof payload?.runtime_config_version === "number" ? (
            <>
              {" "}| version <code>{payload.runtime_config_version}</code>
            </>
          ) : null}
        </p>

        {mutationMessage ? <p className="text-[11px] text-[#b6f0c0] mt-2">{mutationMessage}</p> : null}
        {mutationError ? <p className="text-[11px] text-[#ffcfbf] mt-2">{mutationError}</p> : null}
        {error ? <p className="text-[11px] text-[#ffcfbf] mt-2">{error}</p> : null}
      </div>

      <div className="space-y-2 max-h-[36rem] overflow-y-auto pr-1">
        {moduleViews.length === 0 ? (
          <p className="text-xs text-muted">No constants matched this filter yet.</p>
        ) : (
          moduleViews.map((moduleView) => (
            <section
              key={moduleView.moduleName}
              className="rounded-lg border border-[var(--line)] bg-[rgba(31,32,29,0.86)] p-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-semibold text-ink">
                  <code>{moduleView.moduleName}</code>
                </p>
                <p className="text-[11px] text-muted">
                  constants <code>{moduleView.constantCount}</code> | leaves <code>{moduleView.numericLeafCount}</code>
                </p>
              </div>

              <div className="mt-2 space-y-2">
                {moduleView.entries.length === 0 ? (
                  <p className="text-xs text-muted">No constants matched in this module.</p>
                ) : (
                  moduleView.entries.map((entry) => (
                    <details
                      key={`${moduleView.moduleName}:${entry.key}`}
                      className="rounded-md border border-[rgba(126,166,192,0.26)] bg-[rgba(18,20,18,0.72)] px-3 py-2"
                    >
                      <summary className="cursor-pointer list-none">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="text-xs font-semibold text-[#d9ecff]">
                            <code>{entry.key}</code>
                          </p>
                          <p className="text-[11px] text-[#9ec7dd]">
                            leaves <code>{entry.leafCount}</code> | {entry.preview}
                          </p>
                        </div>
                      </summary>

                      <div className="mt-2 space-y-2">
                        {entry.leaves.length === 0 ? (
                          <p className="text-[11px] text-muted">No numeric leaves found.</p>
                        ) : (
                          entry.leaves.map((leaf) => {
                            const draftText = draftByLeafId[leaf.leafId] ?? formatNumber(leaf.value);
                            const parsedDraft = parseNumericInput(draftText);
                            const liveValue = parsedDraft ?? leaf.value;
                            const dirty = parsedDraft !== null && !numbersClose(parsedDraft, leaf.value);
                            const slider = leafSliderSpec(leaf, parsedDraft);
                            const sliderValue = Math.max(
                              slider.min,
                              Math.min(slider.max, liveValue),
                            );
                            const displayRef = `${leaf.constantKey}${leaf.pathLabel ? `.${leaf.pathLabel}` : ""}`;
                            const canMutate = !bulkMutating && activeMutationLeafId.length === 0;
                            return (
                              <div
                                key={leaf.leafId}
                                className="rounded-md border border-[rgba(126,166,192,0.2)] bg-[rgba(12,16,20,0.66)] p-2"
                              >
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <p className="text-[11px] text-[#d3e8ff] font-semibold">
                                    <code>{displayRef}</code>
                                  </p>
                                  <p className="text-[11px] text-[#9ec7dd]">
                                    current <code>{formatNumber(leaf.value)}</code>
                                  </p>
                                </div>

                                <div className="mt-2 grid gap-2 md:grid-cols-[auto_1fr_auto_auto_auto_auto] items-center">
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setLeafDraft(leaf.leafId, sliderValue - slider.step);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    -
                                  </button>
                                  <input
                                    type="range"
                                    min={slider.min}
                                    max={slider.max}
                                    step={slider.step}
                                    value={sliderValue}
                                    onChange={(event) => {
                                      setLeafDraft(leaf.leafId, Number(event.currentTarget.value));
                                    }}
                                    className="w-full accent-[rgb(126,188,222)]"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setLeafDraft(leaf.leafId, sliderValue + slider.step);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    +
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setLeafDraft(leaf.leafId, sliderValue * 0.5);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    x0.5
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setLeafDraft(leaf.leafId, sliderValue * 2);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    x2
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setDraftByLeafId((previous) => {
                                        const next = { ...previous };
                                        delete next[leaf.leafId];
                                        return next;
                                      });
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    clear
                                  </button>
                                </div>

                                <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto_auto]">
                                  <input
                                    value={draftText}
                                    onChange={(event) => {
                                      setDraftByLeafId((previous) => ({
                                        ...previous,
                                        [leaf.leafId]: event.currentTarget.value,
                                      }));
                                    }}
                                    className="border border-[var(--line)] rounded-md bg-[rgba(26,29,31,0.94)] px-2 py-1 text-xs text-ink"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => {
                                      if (parsedDraft === null) {
                                        return;
                                      }
                                      void applyLeaf(leaf, parsedDraft);
                                    }}
                                    disabled={!dirty || parsedDraft === null || !canMutate}
                                    className="border border-[rgba(126,196,156,0.5)] rounded-md bg-[rgba(35,70,50,0.9)] px-3 py-1 text-xs font-semibold text-[#def7ea] hover:bg-[rgba(49,96,69,0.92)] disabled:opacity-50"
                                  >
                                    {activeMutationLeafId === leaf.leafId ? "applying..." : "apply"}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      void resetLeaf(leaf);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[rgba(202,150,134,0.48)] rounded-md bg-[rgba(70,42,39,0.88)] px-3 py-1 text-xs font-semibold text-[#ffe0d7] hover:bg-[rgba(96,56,52,0.9)] disabled:opacity-50"
                                  >
                                    {activeMutationLeafId === leaf.leafId ? "resetting..." : "reset"}
                                  </button>
                                </div>
                              </div>
                            );
                          })
                        )}

                        <details className="rounded-md border border-[rgba(126,166,192,0.2)] bg-[rgba(11,14,18,0.62)] px-2 py-1">
                          <summary className="cursor-pointer text-[11px] text-muted">raw constant json</summary>
                          <pre className="mt-1 text-[11px] text-[#c7e6ff] whitespace-pre-wrap break-all">
                            {JSON.stringify(entry.value, null, 2)}
                          </pre>
                        </details>
                      </div>
                    </details>
                  ))
                )}
              </div>
            </section>
          ))
        )}
      </div>
    </div>
  );
}
