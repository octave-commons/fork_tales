import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { runtimeApiUrl } from "../../runtime/endpoints";
import type { MuseEvent, SimulationState } from "../../types";

type ThreatRadarMode = "local" | "global";

interface ThreatRadarRuntimeState {
  last_status?: string;
  last_turn_id?: string;
  last_error?: string;
  last_run_at?: string;
  next_run_monotonic?: number;
  last_skipped_reason?: string;
}

interface ThreatRadarRuntime {
  enabled?: boolean;
  muse_id?: string;
  label?: string;
  interval_seconds?: number;
  token_budget?: number;
  state?: ThreatRadarRuntimeState;
}

interface ThreatRow {
  risk_score: number;
  risk_level: "critical" | "high" | "medium" | "low";
  repo?: string;
  domain?: string;
  kind: string;
  number?: number;
  title: string;
  canonical_url: string;
  state?: string;
  signals?: string[];
  labels?: string[];
  cves?: string[];
  deterministic_score?: number;
  llm_score?: number;
  llm_model?: string;
  provisional?: boolean;
  source_type?: string;
  threat_metrics?: {
    overall_score?: number;
    confidence?: number;
    severity?: number;
    immediacy?: number;
    impact?: number;
    exploitability?: number;
    credibility?: number;
    exposure?: number;
    novelty?: number;
    operational_risk?: number;
    rationale?: string;
  };
}

interface ThreatRadarScoring {
  mode?: string;
  llm_enabled?: boolean;
  llm_applied?: boolean;
  llm_model?: string;
  llm_error?: string;
}

interface WatchSourceRow {
  url: string;
  kind?: string;
  title?: string;
  source_type?: string;
  domain_id?: string;
}

interface ThreatRadarResult {
  count: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  hot_repos?: Array<{ repo: string; max_risk_score: number }>;
  hot_kinds?: Array<{ kind: string; max_risk_score: number }>;
  source_count?: number;
  sources?: WatchSourceRow[];
  scoring?: ThreatRadarScoring;
  threats: ThreatRow[];
}

interface ThreatRadarReport {
  ok: boolean;
  radar?: "local" | "global" | "github" | "hormuz" | "cyber";
  snapshot_hash?: string;
  not_modified?: boolean;
  runtime?: ThreatRadarRuntime;
  result?: ThreatRadarResult;
}

interface ThreatRadarPanelProps {
  museEvents?: MuseEvent[];
  simulation?: SimulationState | null;
  isConnected?: boolean;
  fixedRadarMode?: ThreatRadarMode;
  assignedMuseByMode?: Partial<Record<ThreatRadarMode, string>>;
  museChatPanel?: ReactNode;
}

interface ThreatRadarCacheEnvelope {
  version: 1;
  local?: ThreatRadarReport;
  global?: ThreatRadarReport;
}

interface ThreatConversationPayload {
  ok: boolean;
  markdown?: string;
  comment_count?: number;
  url?: string;
  error?: string;
}

interface MuseMessagePayload {
  ok: boolean;
  reply?: string;
  turn_id?: string;
  error?: string;
}

const DEFAULT_RESULT: ThreatRadarResult = {
  count: 0,
  critical_count: 0,
  high_count: 0,
  medium_count: 0,
  low_count: 0,
  hot_repos: [],
  hot_kinds: [],
  threats: [],
};

const REPORT_WINDOW_TICKS_GITHUB = 1440;
const REPORT_WINDOW_TICKS_HORMUZ = 10080;
const REPORT_LIMIT_GITHUB = 24;
const REPORT_LIMIT_HORMUZ = 48;
const CONVERSATION_MAX_COMMENTS = 80;
const CONVERSATION_MAX_MARKDOWN_CHARS = 80_000;
const GLOBAL_FEED_ASK_TOKEN_BUDGET = 1536;
const THREAT_RADAR_DEFAULT_MUSE_ID = "github_security_review";
const THREAT_RADAR_CACHE_KEY = "eta_mu.threat_radar.report_cache.v2";
const THREAT_RADAR_EVENT_REFRESH_DEBOUNCE_MS = 1200;
const THREAT_RADAR_STREAM_REFRESH_MIN_INTERVAL_MS = 2500;
const THREAT_RADAR_REPORT_TIMEOUT_MS = 12_000;
const THREAT_RADAR_TICK_TIMEOUT_MS = 10_000;
const DEFAULT_ASSIGNED_MUSE_BY_MODE: Record<ThreatRadarMode, string> = {
  local: THREAT_RADAR_DEFAULT_MUSE_ID,
  global: THREAT_RADAR_DEFAULT_MUSE_ID,
};

function normalizeRadarMode(value: unknown, fallback: ThreatRadarMode): ThreatRadarMode {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "local" || normalized === "github") {
    return "local";
  }
  if (normalized === "global") {
    return "global";
  }
  return fallback;
}

function readThreatRadarCacheEnvelope(): ThreatRadarCacheEnvelope | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(THREAT_RADAR_CACHE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as ThreatRadarCacheEnvelope;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function readCachedThreatRadarReport(mode: ThreatRadarMode): ThreatRadarReport | null {
  const cache = readThreatRadarCacheEnvelope();
  if (!cache) {
    return null;
  }
  const row = mode === "local" ? cache.local : cache.global;
  if (!row || typeof row !== "object") {
    return null;
  }
  return row;
}

function writeCachedThreatRadarReport(mode: ThreatRadarMode, report: ThreatRadarReport): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    const next: ThreatRadarCacheEnvelope = {
      version: 1,
      ...(readThreatRadarCacheEnvelope() ?? {}),
      [mode]: report,
    };
    window.localStorage.setItem(THREAT_RADAR_CACHE_KEY, JSON.stringify(next));
  } catch {
    // ignore storage failures
  }
}

function badgeClass(level: string): string {
  switch (level) {
    case "critical":
      return "border border-[#f87171] bg-[rgba(127,29,29,0.42)] text-[#fecaca]";
    case "high":
      return "border border-[#fb923c] bg-[rgba(124,45,18,0.35)] text-[#fdba74]";
    case "medium":
      return "border border-[#facc15] bg-[rgba(113,63,18,0.3)] text-[#fde68a]";
    default:
      return "border border-[#22c55e] bg-[rgba(22,101,52,0.3)] text-[#bbf7d0]";
  }
}

function shortSignal(signal: string): string {
  return signal.replaceAll("_", " ");
}

function museLabelFromId(museId: string): string {
  const normalized = String(museId || "").trim().toLowerCase();
  if (normalized === "witness_thread") {
    return "Witness Thread";
  }
  if (normalized === "chaos") {
    return "Chaos";
  }
  if (normalized === "github_security_review") {
    return "Witness Thread";
  }
  if (!normalized) {
    return "Muse";
  }
  return normalized
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (token) => token.toUpperCase());
}

function normalizedThreatToken(value: unknown): string {
  return String(value || "").trim().toLowerCase();
}

function isSeedPlaceholderThreatRow(row: ThreatRow): boolean {
  if (normalizedThreatToken(row.kind) === "global:watchlist_source") {
    return true;
  }
  const tokens = [
    ...(Array.isArray(row.signals) ? row.signals : []),
    ...(Array.isArray(row.labels) ? row.labels : []),
  ].map((value) => normalizedThreatToken(value));
  return tokens.includes("watchlist_seed") || tokens.includes("pending_fetch");
}

function sourceSignalLabel(sourceType: string, kind: string): string {
  const sourceTypeToken = normalizedThreatToken(sourceType).replaceAll("_", " ");
  if (sourceTypeToken) {
    return sourceTypeToken;
  }
  const kindToken = normalizedThreatToken(kind);
  if (kindToken.startsWith("feed:")) {
    return "rss feed";
  }
  if (kindToken.includes("dataset")) {
    return "dataset";
  }
  if (kindToken) {
    return kindToken.replaceAll(":", " ");
  }
  return "watch source";
}

function isGithubLikeUrl(url: string): boolean {
  const lowered = String(url || "").trim().toLowerCase();
  if (!lowered) {
    return false;
  }
  return (
    lowered.includes("github.com/")
    || lowered.includes("api.github.com/")
    || lowered.includes("githubusercontent.com/")
    || lowered.includes("githubassets.com/")
  );
}

function isGithubLikeThreatRow(row: ThreatRow): boolean {
  const kind = String(row.kind || "").trim().toLowerCase();
  if (kind.startsWith("github:")) {
    return true;
  }
  return isGithubLikeUrl(String(row.canonical_url || ""));
}

function threatRowKey(row: ThreatRow): string {
  const repo = String(row.repo || "").trim();
  const number = Math.max(0, Number(row.number || 0));
  const kind = String(row.kind || "").trim();
  const anchor = repo ? `${repo}#${number}` : (kind || "threat");
  return `${anchor}:${row.canonical_url || row.title}`;
}

function supportsConversation(row: ThreatRow): boolean {
  const repo = String(row.repo || "").trim();
  const number = Math.max(0, Number(row.number || 0));
  return (row.kind === "github:issue" || row.kind === "github:pr") && !!repo && number > 0;
}

export function ThreatRadarPanel({
  museEvents = [],
  simulation = null,
  isConnected = false,
  fixedRadarMode,
  assignedMuseByMode,
  museChatPanel,
}: ThreatRadarPanelProps) {
  const lockedRadarMode = fixedRadarMode
    ? normalizeRadarMode(fixedRadarMode, "local")
    : null;
  const initialRadarMode = lockedRadarMode ?? "local";
  const initialCachedReportRef = useRef<ThreatRadarReport | null>(
    readCachedThreatRadarReport(initialRadarMode),
  );
  const [radarMode, setRadarMode] = useState<ThreatRadarMode>(initialRadarMode);
  const [report, setReport] = useState<ThreatRadarReport | null>(initialCachedReportRef.current);
  const [loading, setLoading] = useState(initialCachedReportRef.current == null);
  const [refreshing, setRefreshing] = useState(false);
  const [forcingTick, setForcingTick] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [radarFilterInput, setRadarFilterInput] = useState("");
  const [radarFilter, setRadarFilter] = useState("");
  const [selectedThreat, setSelectedThreat] = useState<ThreatRow | null>(null);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [conversationMarkdown, setConversationMarkdown] = useState("");
  const [conversationCommentCount, setConversationCommentCount] = useState(0);
  const [conversationUrl, setConversationUrl] = useState("");
  const [conversationError, setConversationError] = useState<string | null>(null);
  const [globalAskInput, setGlobalAskInput] = useState("");
  const [globalAskLoading, setGlobalAskLoading] = useState(false);
  const [globalAskReply, setGlobalAskReply] = useState("");
  const [globalAskError, setGlobalAskError] = useState<string | null>(null);
  const [globalAskTurnId, setGlobalAskTurnId] = useState("");

  const activeMuseByMode = useMemo<Record<ThreatRadarMode, string>>(() => ({
    local: String(
      assignedMuseByMode?.local
      || DEFAULT_ASSIGNED_MUSE_BY_MODE.local,
    ).trim() || DEFAULT_ASSIGNED_MUSE_BY_MODE.local,
    global: String(
      assignedMuseByMode?.global
      || DEFAULT_ASSIGNED_MUSE_BY_MODE.global,
    ).trim() || DEFAULT_ASSIGNED_MUSE_BY_MODE.global,
  }), [assignedMuseByMode?.global, assignedMuseByMode?.local]);

  const criticalCountPrimedRef = useRef(false);
  const previousCriticalCountRef = useRef(0);
  const reportRequestTokenRef = useRef(0);
  const processedMuseEventSeqRef = useRef(0);
  const lastMuseEventRefreshAtRef = useRef(0);
  const streamRefreshInFlightRef = useRef(false);
  const lastStreamRefreshAtRef = useRef(0);
  const managedOverflowRef = useRef<HTMLElement | null>(null);
  const snapshotHashByModeRef = useRef<Record<ThreatRadarMode, string>>({
    local: String(initialCachedReportRef.current?.snapshot_hash || "").trim(),
    global: String(readCachedThreatRadarReport("global")?.snapshot_hash || "").trim(),
  });

  useEffect(() => {
    if (!museChatPanel) {
      return;
    }
    const root = managedOverflowRef.current;
    const panelBody = root?.closest(".world-panel-body") as HTMLElement | null;
    if (!panelBody) {
      return;
    }
    const previousDisplay = panelBody.style.display;
    const previousFlexDirection = panelBody.style.flexDirection;
    const previousOverflow = panelBody.style.overflow;
    const previousOverscrollBehavior = panelBody.style.overscrollBehavior;
    panelBody.style.display = "flex";
    panelBody.style.flexDirection = "column";
    panelBody.style.overflow = "hidden";
    panelBody.style.overscrollBehavior = "contain";
    return () => {
      panelBody.style.display = previousDisplay;
      panelBody.style.flexDirection = previousFlexDirection;
      panelBody.style.overflow = previousOverflow;
      panelBody.style.overscrollBehavior = previousOverscrollBehavior;
    };
  }, [museChatPanel]);

  useEffect(() => {
    if (!lockedRadarMode) {
      return;
    }
    setRadarMode(lockedRadarMode);
  }, [lockedRadarMode]);

  useEffect(() => {
    const snapshotHash = String(report?.snapshot_hash || "").trim();
    if (!snapshotHash) {
      return;
    }
    snapshotHashByModeRef.current[radarMode] = snapshotHash;
  }, [radarMode, report?.snapshot_hash]);

  const fetchReport = useCallback(async (
    mode: "initial" | "refresh" = "refresh",
    options: { sinceSnapshotHash?: string } = {},
  ) => {
    const requestToken = reportRequestTokenRef.current + 1;
    reportRequestTokenRef.current = requestToken;
    const requestedRadarMode = radarMode;

    if (mode === "initial") {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError(null);
    try {
      const reportWindowTicks = requestedRadarMode === "global"
        ? REPORT_WINDOW_TICKS_HORMUZ
        : REPORT_WINDOW_TICKS_GITHUB;
      const reportLimit = requestedRadarMode === "global"
        ? REPORT_LIMIT_HORMUZ
        : REPORT_LIMIT_GITHUB;
      const params = new URLSearchParams({
        window_ticks: String(reportWindowTicks),
        limit: String(reportLimit),
        radar: requestedRadarMode,
      });
      const sinceSnapshotHash = String(options.sinceSnapshotHash || "").trim();
      if (sinceSnapshotHash) {
        params.set("since_snapshot_hash", sinceSnapshotHash);
      }
      if (radarFilter) {
        if (requestedRadarMode === "local") {
          params.set("repo", radarFilter);
        } else {
          if (radarFilter.includes(":")) {
            params.set("kind", radarFilter);
          } else {
            const normalizedDomain = radarFilter
              .replace(/^https?:\/\//, "")
              .split("/")[0]
              .trim();
            if (normalizedDomain) {
              params.set("domain", normalizedDomain);
            }
          }
        }
      }
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => {
        controller.abort();
      }, THREAT_RADAR_REPORT_TIMEOUT_MS);
      let response: Response;
      try {
        response = await fetch(runtimeApiUrl(`/api/muse/threat-radar/report?${params.toString()}`), {
          signal: controller.signal,
        });
      } finally {
        window.clearTimeout(timeoutId);
      }
      if (!response.ok) {
        throw new Error(`threat radar report request failed (${response.status})`);
      }
      const payload = (await response.json()) as ThreatRadarReport;
      if (requestToken !== reportRequestTokenRef.current) {
        return;
      }
      const payloadRadar = normalizeRadarMode(payload.radar, requestedRadarMode);
      if (payloadRadar !== requestedRadarMode) {
        return;
      }
      if (payload.not_modified) {
        const payloadSnapshotHash = String(payload.snapshot_hash || "").trim();
        if (payloadSnapshotHash) {
          snapshotHashByModeRef.current[payloadRadar] = payloadSnapshotHash;
        }
        setReport((current) => {
          if (!current) {
            return current;
          }
          if (normalizeRadarMode(current.radar, requestedRadarMode) !== requestedRadarMode) {
            return current;
          }
          return {
            ...current,
            snapshot_hash: payloadSnapshotHash || current.snapshot_hash,
            runtime: payload.runtime || current.runtime,
            ok: true,
            not_modified: true,
          };
        });
        return;
      }
      const payloadSnapshotHash = String(payload.snapshot_hash || "").trim();
      if (payloadSnapshotHash) {
        snapshotHashByModeRef.current[payloadRadar] = payloadSnapshotHash;
      }
      setReport(payload);
      writeCachedThreatRadarReport(payloadRadar, payload);
    } catch (reason) {
      if (requestToken !== reportRequestTokenRef.current) {
        return;
      }
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
      const cached = readCachedThreatRadarReport(requestedRadarMode);
      setReport((current) => {
        if (
          current
          && normalizeRadarMode(current.radar, requestedRadarMode) === requestedRadarMode
        ) {
          return current;
        }
        if (cached) {
          return cached;
        }
        return {
          ok: false,
          radar: requestedRadarMode,
          runtime: current?.runtime,
          result: DEFAULT_RESULT,
        };
      });
    } finally {
      if (requestToken === reportRequestTokenRef.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [radarFilter, radarMode]);

  const loadConversation = useCallback(async (threat: ThreatRow) => {
    if (!supportsConversation(threat) || !threat.repo || Number(threat.number || 0) <= 0) {
      setConversationMarkdown("");
      setConversationCommentCount(0);
      setConversationUrl(threat.canonical_url || "");
      setConversationError("thread preview is available for issue and pull request rows");
      return;
    }

    setConversationLoading(true);
    setConversationError(null);
    try {
      const repo = String(threat.repo || "").trim();
      const number = Math.max(1, Number(threat.number || 0));
      const params = new URLSearchParams({
        repo,
        number: String(number),
        kind: threat.kind,
        max_comments: String(CONVERSATION_MAX_COMMENTS),
        max_markdown_chars: String(CONVERSATION_MAX_MARKDOWN_CHARS),
      });
      const response = await fetch(runtimeApiUrl(`/api/github/conversation?${params.toString()}`));
      const payload = (await response.json()) as ThreatConversationPayload;
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || `conversation request failed (${response.status})`);
      }
      setConversationMarkdown(String(payload.markdown || "").trim());
      setConversationCommentCount(Math.max(0, Number(payload.comment_count || 0)));
      setConversationUrl(String(payload.url || threat.canonical_url || ""));
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setConversationError(message);
      setConversationMarkdown("");
      setConversationCommentCount(0);
      setConversationUrl(threat.canonical_url || "");
    } finally {
      setConversationLoading(false);
    }
  }, []);

  const selectThreat = useCallback((threat: ThreatRow) => {
    setSelectedThreat(threat);
    setGlobalAskError(null);
    setGlobalAskReply("");
    setGlobalAskTurnId("");
    if (radarMode === "local") {
      loadConversation(threat).catch(() => {});
      return;
    }
    setConversationLoading(false);
    setConversationError(null);
    setConversationCommentCount(0);
    setConversationMarkdown("");
    setConversationUrl(threat.canonical_url || "");
  }, [loadConversation, radarMode]);

  const applyRadarFilter = useCallback(() => {
    setRadarFilter(radarFilterInput.trim().toLowerCase());
  }, [radarFilterInput]);

  const clearRadarFilter = useCallback(() => {
    setRadarFilterInput("");
    setRadarFilter("");
  }, []);

  const forceTick = useCallback(async () => {
    setForcingTick(true);
    try {
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => {
        controller.abort();
      }, THREAT_RADAR_TICK_TIMEOUT_MS);
      let response: Response;
      try {
        response = await fetch(runtimeApiUrl("/api/muse/threat-radar/tick?force=true"), {
          signal: controller.signal,
        });
      } finally {
        window.clearTimeout(timeoutId);
      }
      if (!response.ok) {
        throw new Error(`threat radar tick request failed (${response.status})`);
      }
      await fetchReport("refresh");
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
    } finally {
      setForcingTick(false);
    }
  }, [fetchReport]);

  useEffect(() => {
    fetchReport("initial").catch(() => {});
  }, [fetchReport]);

  const streamRadarCursor = useMemo(() => {
    const simulationTimestamp = String(simulation?.timestamp || "").trim();
    const crawlerGeneratedAt = String(simulation?.crawler_graph?.generated_at || "").trim();
    const crawlerNodes = Array.isArray(simulation?.crawler_graph?.crawler_nodes)
      ? simulation?.crawler_graph?.crawler_nodes.length
      : 0;
    const crawlerEdges = Array.isArray(simulation?.crawler_graph?.edges)
      ? simulation?.crawler_graph?.edges.length
      : 0;
    return [simulationTimestamp, crawlerGeneratedAt, String(crawlerNodes), String(crawlerEdges)].join("|");
  }, [
    simulation?.timestamp,
    simulation?.crawler_graph?.generated_at,
    simulation?.crawler_graph?.crawler_nodes,
    simulation?.crawler_graph?.edges,
  ]);

  useEffect(() => {
    if (!isConnected) {
      return;
    }
    if (!streamRadarCursor) {
      return;
    }
    const nowMs = Date.now();
    if (nowMs - lastStreamRefreshAtRef.current < THREAT_RADAR_STREAM_REFRESH_MIN_INTERVAL_MS) {
      return;
    }
    if (streamRefreshInFlightRef.current) {
      return;
    }

    lastStreamRefreshAtRef.current = nowMs;
    streamRefreshInFlightRef.current = true;
    fetchReport("refresh", {
      sinceSnapshotHash: String(snapshotHashByModeRef.current[radarMode] || "").trim(),
    })
      .catch(() => {})
      .finally(() => {
        streamRefreshInFlightRef.current = false;
      });
  }, [fetchReport, isConnected, radarMode, streamRadarCursor]);

  useEffect(() => {
    if (!Array.isArray(museEvents) || museEvents.length <= 0) {
      return;
    }
    const expectedMuseIds = new Set(
      [
        String(report?.runtime?.muse_id || "").trim(),
        String(activeMuseByMode[radarMode] || "").trim(),
        THREAT_RADAR_DEFAULT_MUSE_ID,
      ]
        .filter((row) => row.length > 0),
    );
    let nextSeq = processedMuseEventSeqRef.current;
    let shouldRefresh = false;
    for (const row of museEvents) {
      const seq = Math.max(0, Number(row.seq || 0));
      if (seq > nextSeq) {
        nextSeq = seq;
      }
      if (seq <= processedMuseEventSeqRef.current) {
        continue;
      }
      const museId = String(row.muse_id || "").trim();
      if (museId && !expectedMuseIds.has(museId)) {
        continue;
      }
      const kind = String(row.kind || "").trim().toLowerCase();
      const payload = (row.payload && typeof row.payload === "object") ? row.payload : {};
      const query = String((payload as Record<string, unknown>).query || "").trim().toLowerCase();
      const tool = String((payload as Record<string, unknown>).tool || "").trim().toLowerCase();
      const snapshotHash = String((payload as Record<string, unknown>).snapshot_hash || "").trim();

      if (
        snapshotHash
        && (query.includes("threat_radar") || tool.includes("threat_radar") || kind === "muse_job_completed")
      ) {
        const currentSnapshotHash = String(snapshotHashByModeRef.current[radarMode] || "").trim();
        if (snapshotHash === currentSnapshotHash) {
          continue;
        }
      }

      if (
        kind === "muse.turn.completed"
        || kind === "muse.response.generated"
        || kind === "muse.tool.result"
        || kind === "muse_job_completed"
      ) {
        shouldRefresh = true;
      }
    }
    processedMuseEventSeqRef.current = Math.max(processedMuseEventSeqRef.current, nextSeq);
    if (!shouldRefresh) {
      return;
    }
    const nowMs = Date.now();
    if (nowMs - lastMuseEventRefreshAtRef.current < THREAT_RADAR_EVENT_REFRESH_DEBOUNCE_MS) {
      return;
    }
    lastMuseEventRefreshAtRef.current = nowMs;
    fetchReport("refresh", {
      sinceSnapshotHash: String(snapshotHashByModeRef.current[radarMode] || "").trim(),
    }).catch(() => {});
  }, [activeMuseByMode, fetchReport, museEvents, radarMode, report?.runtime?.muse_id]);

  const result = report?.result ?? DEFAULT_RESULT;
  const runtime = report?.runtime;
  const runtimeState = runtime?.state;
  const scoring = result.scoring;
  const rawThreats = Array.isArray(result.threats) ? result.threats : [];
  const scopedThreats = useMemo(() => {
    if (radarMode === "local") {
      return rawThreats.filter((row) => isGithubLikeThreatRow(row));
    }
    return rawThreats
      .filter((row) => !isGithubLikeThreatRow(row))
      .filter((row) => !isSeedPlaceholderThreatRow(row));
  }, [radarMode, rawThreats]);
  const provisionalSeedThreats = useMemo(() => {
    if (radarMode !== "global") {
      return [] as ThreatRow[];
    }
    return rawThreats
      .filter((row) => !isGithubLikeThreatRow(row))
      .filter((row) => isSeedPlaceholderThreatRow(row));
  }, [radarMode, rawThreats]);
  const signalSourceThreats = useMemo(() => {
    if (
      radarMode === "global"
      && scopedThreats.length === 0
      && provisionalSeedThreats.length > 0
    ) {
      return provisionalSeedThreats;
    }
    return scopedThreats;
  }, [provisionalSeedThreats, radarMode, scopedThreats]);
  const levelCounts = useMemo(() => {
    const base = {
      count: 0,
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
    };
    for (const row of scopedThreats) {
      base.count += 1;
      const level = String(row.risk_level || "low").trim().toLowerCase();
      if (level === "critical") {
        base.critical += 1;
      } else if (level === "high") {
        base.high += 1;
      } else if (level === "medium") {
        base.medium += 1;
      } else {
        base.low += 1;
      }
    }
    return base;
  }, [scopedThreats]);
  const topThreats = useMemo(() => scopedThreats.slice(0, 12), [scopedThreats]);
  const watchSources = useMemo(() => {
    const rows = Array.isArray(result.sources) ? result.sources : [];
    return rows
      .map((row) => ({
        url: String(row.url || "").trim(),
        kind: String(row.kind || "").trim(),
        title: String(row.title || "").trim(),
        sourceType: String(row.source_type || "").trim().toLowerCase(),
      }))
      .filter((row) => row.url.length > 0)
      .slice(0, 12);
  }, [result.sources]);
  const sourceCount = Math.max(
    watchSources.length,
    Math.max(0, Number(result.source_count || 0)),
  );
  const sourceSignalRows = useMemo(() => {
    const countByLabel = new Map<string, number>();
    for (const row of watchSources) {
      const label = sourceSignalLabel(row.sourceType, row.kind || "");
      if (!label) {
        continue;
      }
      countByLabel.set(label, (countByLabel.get(label) || 0) + 1);
    }
    return [...countByLabel.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 10)
      .map(([label, count]) => ({ label, count }));
  }, [watchSources]);
  const signalRows = useMemo(() => {
    const scoreBySignal = new Map<string, number>();
    for (const row of signalSourceThreats) {
      const tokens = [
        ...(Array.isArray(row.signals) ? row.signals : []),
        ...(Array.isArray(row.labels) ? row.labels : []),
      ];
      for (const token of tokens) {
        const normalized = String(token || "").trim().toLowerCase();
        if (!normalized) {
          continue;
        }
        const next = (scoreBySignal.get(normalized) || 0) + 1;
        scoreBySignal.set(normalized, next);
      }
    }
    return [...scoreBySignal.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 8)
      .map(([signal, count]) => ({ signal, count }));
  }, [signalSourceThreats]);
  const hotRows = useMemo(() => {
    const scoreByScope = new Map<string, number>();
    for (const row of signalSourceThreats) {
      const scopeValue = radarMode === "local"
        ? String(row.repo || "").trim().toLowerCase()
        : String(row.domain || row.kind || "").trim().toLowerCase();
      if (!scopeValue) {
        continue;
      }
      const score = Math.max(0, Number(row.risk_score || 0));
      const previous = scoreByScope.get(scopeValue) ?? 0;
      if (score > previous) {
        scoreByScope.set(scopeValue, score);
      }
    }
    return [...scoreByScope.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 8)
      .map(([label, score]) => ({
        id: label,
        label,
        score,
      }));
  }, [radarMode, signalSourceThreats]);
  const selectedThreatKey = selectedThreat ? threatRowKey(selectedThreat) : "";

  useEffect(() => {
    if (topThreats.length === 0) {
      setSelectedThreat(null);
      setConversationMarkdown("");
      setConversationCommentCount(0);
      setConversationError(null);
      setConversationUrl("");
      setGlobalAskError(null);
      setGlobalAskReply("");
      setGlobalAskTurnId("");
      return;
    }
    if (
      selectedThreat
      && topThreats.some((row) => threatRowKey(row) === threatRowKey(selectedThreat))
    ) {
      return;
    }
    selectThreat(topThreats[0]);
  }, [selectedThreat, selectThreat, topThreats]);

  useEffect(() => {
    if (loading) {
      return;
    }
    const nextCriticalCount = Math.max(0, Number(levelCounts.critical || 0));
    const top = topThreats[0] || null;
    if (!criticalCountPrimedRef.current) {
      previousCriticalCountRef.current = nextCriticalCount;
      criticalCountPrimedRef.current = true;
      return;
    }
    if (nextCriticalCount > previousCriticalCountRef.current && top) {
      const threatRef = top.repo
        ? `${top.repo} #${Math.max(0, Number(top.number || 0))}`
        : "top threat";
      window.dispatchEvent(
        new CustomEvent("ui:toast", {
          detail: {
            title: "Threat Radar Critical",
            body: `${nextCriticalCount} critical threats active. Top candidate: ${threatRef}.`,
          },
        }),
      );
    }
    previousCriticalCountRef.current = nextCriticalCount;
  }, [levelCounts.critical, loading, topThreats]);

  const selectedThreatSupportsConversation = radarMode === "local" && selectedThreat
    ? supportsConversation(selectedThreat)
    : false;
  const activeMuseId = String(activeMuseByMode[radarMode] || "").trim();
  const activeMuseLabel = museLabelFromId(activeMuseId);
  const panelTitle = lockedRadarMode
    ? (radarMode === "local"
      ? `${activeMuseLabel} Muse / Local GitHub Security Radar`
      : `${activeMuseLabel} Muse / Global Geopolitical Radar`)
    : (radarMode === "local"
      ? "Threat Radar / Local Cyber View"
      : "Threat Radar / Global Geopolitical Feed");
  const panelDescription = lockedRadarMode
    ? (radarMode === "local"
      ? "GitHub security muse lane with threat scoring, thread context, and direct response actions."
      : "Global signal muse lane over geopolitical web evidence, refreshed from live simulation stream updates.")
    : (radarMode === "local"
      ? "Local software security view: PR, issue, advisory, and code-risk signals you can directly influence."
      : "Global signal aggregator over knowledge-graph web evidence, refreshed from the live simulation websocket stream. Raw feed and dataset endpoints stay in Web Weaver.");
  const filterPlaceholder = radarMode === "local"
    ? "filter repo (owner/name)"
    : "filter domain (ukmto.org) or kind (maritime:...)";
  const scopeLabel = radarMode === "local" ? "repo" : "domain/kind";
  const modeMuseLabel = radarMode === "local"
    ? `${activeMuseLabel} / GitHub Security Radar`
    : `${activeMuseLabel} / Global Geopolitical Radar`;
  const runtimeLabel = radarMode === "local"
    ? (lockedRadarMode ? modeMuseLabel : (runtime?.label || runtime?.muse_id || modeMuseLabel))
    : modeMuseLabel;
  const scoringLabel = String(scoring?.mode || "deterministic").replaceAll("_", " ");
  const llmStatusLabel = scoring?.llm_applied
    ? "llm active"
    : scoring?.llm_enabled
      ? "llm standby"
      : "deterministic";
  const lastRunLabel = runtimeState?.last_run_at
    ? new Date(runtimeState.last_run_at).toLocaleString()
    : "not yet";
  const contextHeading = radarMode === "local"
    ? "LOCAL SOURCE CONTEXT"
    : "GLOBAL SOURCE CONTEXT";
  const contextScopeValue = radarMode === "local"
    ? String(selectedThreat?.repo || radarFilter || "").trim()
    : String(selectedThreat?.domain || selectedThreat?.kind || radarFilter || "").trim();
  const compactHotHeading = radarMode === "local"
    ? "Hot Repos / Kinds"
    : "Hot Domains / Kinds";

  const askGlobalFeed = useCallback(async () => {
    if (radarMode !== "global") {
      return;
    }
    const question = globalAskInput.trim();
    if (!question) {
      setGlobalAskError("enter a question for the global feed");
      return;
    }
    if (!selectedThreat) {
      setGlobalAskError("select a global threat row first");
      return;
    }

    setGlobalAskLoading(true);
    setGlobalAskError(null);
    setGlobalAskReply("");
    setGlobalAskTurnId("");

    try {
      const signalTokens = (
        Array.isArray(selectedThreat.labels) ? selectedThreat.labels : selectedThreat.signals || []
      )
        .map((row) => String(row || "").trim())
        .filter((row) => row.length > 0)
        .slice(0, 6);
      const sourceHints = watchSources
        .slice(0, 4)
        .map((row) => row.title || row.kind || row.url)
        .filter((row) => row.length > 0)
        .join("; ");
      const prompt = [
        `Question: ${question}`,
        "",
        "Focused global threat:",
        `- title: ${selectedThreat.title || "(untitled)"}`,
        `- kind: ${selectedThreat.kind || "unknown"}`,
        `- domain: ${selectedThreat.domain || ""}`,
        `- risk: ${selectedThreat.risk_level} ${Math.max(0, Number(selectedThreat.risk_score || 0))}`,
        `- source: ${selectedThreat.canonical_url || ""}`,
        `- signals: ${signalTokens.length > 0 ? signalTokens.map(shortSignal).join(" | ") : "none"}`,
        ...(sourceHints ? [`- configured_sources: ${sourceHints}`] : []),
        "",
        "Reply in three parts: (1) short situation summary, (2) why this matters for cyber posture, (3) one immediate check.",
      ].join("\n");

      const response = await fetch(runtimeApiUrl("/api/muse/message"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          muse_id: activeMuseId || String(runtime?.muse_id || THREAT_RADAR_DEFAULT_MUSE_ID).trim() || THREAT_RADAR_DEFAULT_MUSE_ID,
          text: prompt,
          mode: "deterministic",
          token_budget: GLOBAL_FEED_ASK_TOKEN_BUDGET,
          graph_revision: String(report?.snapshot_hash || "").trim(),
          surrounding_nodes: [],
        }),
      });
      const payload = (await response.json()) as MuseMessagePayload;
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || `global feed ask failed (${response.status})`);
      }
      setGlobalAskReply(String(payload.reply || "").trim() || "(no reply returned)");
      setGlobalAskTurnId(String(payload.turn_id || "").trim());
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setGlobalAskError(message);
      setGlobalAskReply("");
      setGlobalAskTurnId("");
    } finally {
      setGlobalAskLoading(false);
    }
  }, [
    globalAskInput,
    radarMode,
    report?.snapshot_hash,
    runtime?.muse_id,
    activeMuseId,
    selectedThreat,
    watchSources,
  ]);

  if (museChatPanel) {
    return (
      <section ref={managedOverflowRef} className="threat-radar-muse-layout card relative flex h-full min-h-0 max-h-full flex-1 flex-col overflow-hidden">
        <div className="absolute top-0 left-0 h-full w-1 bg-[#ef4444] opacity-75" />
        <div className="grid min-h-0 max-h-full flex-1 grid-rows-[minmax(0,1fr)] gap-2 overflow-hidden lg:grid-cols-[minmax(14rem,1fr)_minmax(0,1.7fr)]">
          <section className="flex min-h-0 max-h-full flex-col overflow-hidden rounded-lg border border-[var(--line)] bg-[rgba(10,14,22,0.82)] p-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs uppercase tracking-[0.16em] text-[#f4b4b4]">{contextHeading}</p>
              <div className="flex flex-wrap items-center gap-1.5">
                {lockedRadarMode ? (
                  <p className="rounded-md border border-[rgba(128,186,224,0.48)] bg-[rgba(18,42,60,0.72)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#d8ecff]">
                    {radarMode} scope
                  </p>
                ) : (
                  <select
                    value={radarMode}
                    onChange={(event) => {
                      const nextMode = event.currentTarget.value === "local" ? "local" : "global";
                      reportRequestTokenRef.current += 1;
                      const cachedReport = readCachedThreatRadarReport(nextMode);
                      setRadarMode(nextMode);
                      setRadarFilterInput("");
                      setRadarFilter("");
                      setReport(cachedReport);
                      setLoading(cachedReport == null);
                      setSelectedThreat(null);
                      setConversationLoading(false);
                      setConversationError(null);
                      setConversationCommentCount(0);
                      setConversationMarkdown("");
                      setConversationUrl("");
                      setGlobalAskInput("");
                      setGlobalAskError(null);
                      setGlobalAskReply("");
                      setGlobalAskTurnId("");
                    }}
                    className="rounded-md border border-[var(--line)] bg-[rgba(17,19,26,0.9)] px-2 py-1 text-[11px] text-ink"
                  >
                    <option value="global">global</option>
                    <option value="local">local</option>
                  </select>
                )}
                <button
                  type="button"
                  onClick={() => {
                    forceTick().catch(() => {});
                  }}
                  disabled={forcingTick}
                  className="rounded-md border border-[var(--line)] bg-[rgba(32,35,44,0.9)] px-2.5 py-1 text-[11px] font-semibold text-ink hover:bg-[rgba(44,49,61,0.95)] disabled:opacity-55"
                >
                  {forcingTick ? "running..." : "run now"}
                </button>
              </div>
            </div>

            <p className="mt-1.5 overflow-hidden text-ellipsis whitespace-nowrap text-[10px] text-[#a9c2d6]">
              stream {isConnected ? "ws live" : "ws offline"} | last run {lastRunLabel}
              {contextScopeValue ? ` | focus ${contextScopeValue}` : ""}
            </p>
            {runtimeState?.last_error ? (
              <p className="mt-1.5 rounded border border-[#b91c1c] bg-[rgba(127,29,29,0.22)] px-2 py-1 text-[11px] text-[#fecaca]">
                runtime error: {runtimeState.last_error}
              </p>
            ) : null}
            {error ? (
              <p className="mt-1.5 rounded border border-[#b91c1c] bg-[rgba(127,29,29,0.22)] px-2 py-1 text-[11px] text-[#fecaca]">
                {error}
              </p>
            ) : null}

            <div className="mt-2 grid min-h-0 flex-1 gap-2 overflow-hidden [grid-template-rows:minmax(0,1.72fr)_minmax(0,0.74fr)_minmax(0,0.68fr)_minmax(0,0.56fr)]">
              <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-md border border-[rgba(116,168,207,0.45)] bg-[rgba(16,30,44,0.62)] p-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs uppercase tracking-[0.12em] text-[#f4b4b4]">Top Threats</p>
                  <p
                    className={`text-[10px] uppercase tracking-[0.08em] text-[#9cc4df] transition-opacity ${(loading || refreshing) ? "opacity-100" : "opacity-0"}`}
                  >
                    refreshing
                  </p>
                </div>
                <div className="mt-1.5 h-full min-h-0 flex-1 space-y-1.5 overflow-y-auto pr-1">
                  {topThreats.map((row) => {
                    const rowRepo = String(row.repo || "").trim();
                    const rowNumber = Math.max(0, Number(row.number || 0));
                    const rowIdentity = rowRepo ? `${rowRepo} #${rowNumber}` : (row.kind || "resource");
                    const signalTokens = (Array.isArray(row.signals) && row.signals.length > 0)
                      ? row.signals
                      : (Array.isArray(row.labels) ? row.labels : []);
                    return (
                      <article
                        key={threatRowKey(row)}
                        className={`rounded-md border px-2 py-1.5 ${selectedThreatKey === threatRowKey(row)
                          ? "border-[rgba(246,113,113,0.72)] bg-[rgba(70,33,38,0.72)]"
                          : "border-[var(--line)] bg-[rgba(35,39,49,0.84)]"}`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <p className="min-w-0 truncate text-xs font-mono text-[#dbe8f5]">{rowIdentity}</p>
                          <span
                            className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${badgeClass(row.risk_level)}`}
                          >
                            {row.risk_level} {row.risk_score}
                          </span>
                        </div>
                        <p className="mt-0.5 truncate text-xs text-ink">{row.title || "(untitled)"}</p>
                        <p className="mt-0.5 truncate text-[10px] text-muted">
                          {signalTokens.slice(0, 3).map(shortSignal).join(" | ") || "no explicit signals"}
                        </p>
                        <div className="mt-1.5 flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => {
                              selectThreat(row);
                            }}
                            className="rounded border border-[rgba(137,189,226,0.5)] bg-[rgba(27,55,78,0.72)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#d9efff]"
                          >
                            {radarMode === "local" && supportsConversation(row) ? "thread" : "details"}
                          </button>
                          {row.canonical_url ? (
                            <a
                              href={row.canonical_url}
                              target="_blank"
                              rel="noreferrer"
                              className="rounded border border-[rgba(165,190,213,0.44)] bg-[rgba(35,43,59,0.72)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#dbe8f5]"
                            >
                              open
                            </a>
                          ) : null}
                        </div>
                      </article>
                    );
                  })}
                  {topThreats.length === 0 && !loading ? (
                    <p className="text-xs text-muted">
                      {radarMode === "local"
                        ? "No GitHub threat candidates yet."
                        : provisionalSeedThreats.length > 0
                          ? `No classified global threat signals yet. ${provisionalSeedThreats.length} watch seeds are queued for crawl evidence.`
                          : "No global geopolitics rows yet. Keep crawl running and trigger refresh."}
                    </p>
                  ) : null}
                </div>
              </section>

              <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-md border border-[rgba(116,168,207,0.4)] bg-[rgba(18,28,40,0.55)] p-2">
                <p className="text-[10px] uppercase tracking-[0.1em] text-[#9cc4df]">{compactHotHeading}</p>
                <div className="mt-1.5 h-full min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
                  {hotRows.map((row) => (
                    <div
                      key={row.id}
                      className="flex items-center justify-between rounded bg-[rgba(16,22,31,0.72)] px-2 py-1"
                    >
                      <p className="truncate text-[11px] font-mono text-[#cfe6f8]">{row.label}</p>
                      <p className="text-[10px] text-[#fda4af]">risk {row.score}</p>
                    </div>
                  ))}
                  {hotRows.length === 0 ? (
                    <p className="text-xs text-muted">No hot entries in current window.</p>
                  ) : null}
                </div>
              </section>

              <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-md border border-[rgba(116,168,207,0.4)] bg-[rgba(18,28,40,0.55)] p-2">
                <p className="text-[10px] uppercase tracking-[0.1em] text-[#9cc4df]">signal mix</p>
                <div className="mt-1.5 h-full min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
                  {signalRows.map((row) => (
                    <div
                      key={row.signal}
                      className="flex items-center justify-between rounded bg-[rgba(16,22,31,0.72)] px-2 py-1"
                    >
                      <p className="truncate text-[11px] text-[#cfe6f8]">{shortSignal(row.signal)}</p>
                      <p className="text-[10px] text-[#9cc4df]">{row.count}</p>
                    </div>
                  ))}
                  {signalRows.length === 0 ? (
                    <p className="text-xs text-muted">No signal labels resolved yet.</p>
                  ) : null}
                </div>
              </section>

              <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-md border border-[rgba(116,168,207,0.45)] bg-[rgba(20,32,46,0.58)] p-2">
                <p className="text-[10px] uppercase tracking-[0.1em] text-[#9cc4df]">source signal inputs</p>
                <div className="mt-1.5 h-full min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
                  {sourceSignalRows.map((sourceRow) => (
                    <div
                      key={sourceRow.label}
                      className="flex items-center justify-between gap-2 rounded bg-[rgba(18,24,33,0.68)] px-2 py-1"
                    >
                      <p className="truncate text-[11px] text-[#cfe6f8]">{sourceRow.label}</p>
                      <p className="text-[10px] text-[#9cc4df]">{sourceRow.count}</p>
                    </div>
                  ))}
                  {sourceSignalRows.length === 0 ? (
                    <p className="text-xs text-muted">No source signal classes reported by this radar query.</p>
                  ) : null}
                </div>
              </section>
            </div>
          </section>

          <div className="flex min-h-0 max-h-full flex-col overflow-hidden">
            <div className="h-full min-h-0 w-full max-h-full overflow-hidden">{museChatPanel}</div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="card relative flex h-full flex-col overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full bg-[#ef4444] opacity-75" />
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="mb-0 border-0 pb-0 text-[1.08rem] font-bold leading-tight">{panelTitle}</h2>
          <p className="text-[11px] leading-snug text-muted">{panelDescription}</p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {lockedRadarMode ? (
            <p className="rounded-md border border-[rgba(128,186,224,0.48)] bg-[rgba(18,42,60,0.72)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#d8ecff]">
              {radarMode} scope
            </p>
          ) : (
            <select
              value={radarMode}
              onChange={(event) => {
                const nextMode = event.currentTarget.value === "local" ? "local" : "global";
                reportRequestTokenRef.current += 1;
                const cachedReport = readCachedThreatRadarReport(nextMode);
                setRadarMode(nextMode);
                setRadarFilterInput("");
                setRadarFilter("");
                setReport(cachedReport);
                setLoading(cachedReport == null);
                setSelectedThreat(null);
                setConversationLoading(false);
                setConversationError(null);
                setConversationCommentCount(0);
                setConversationMarkdown("");
                setConversationUrl("");
                setGlobalAskInput("");
                setGlobalAskError(null);
                setGlobalAskReply("");
                setGlobalAskTurnId("");
              }}
              className="rounded-md border border-[var(--line)] bg-[rgba(17,19,26,0.9)] px-2 py-1 text-[11px] text-ink"
            >
              <option value="global">global</option>
              <option value="local">local</option>
            </select>
          )}
          <button
            type="button"
            onClick={() => {
              forceTick().catch(() => {});
            }}
            disabled={forcingTick}
            className="rounded-md border border-[var(--line)] bg-[rgba(32,35,44,0.9)] px-2.5 py-1 text-[11px] font-semibold text-ink hover:bg-[rgba(44,49,61,0.95)] disabled:opacity-55"
          >
            {forcingTick ? "running..." : "run now"}
          </button>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <input
          value={radarFilterInput}
          onChange={(event) => {
            setRadarFilterInput(event.currentTarget.value);
          }}
          placeholder={filterPlaceholder}
          className="min-w-[12rem] flex-[1_1_14rem] rounded-md border border-[var(--line)] bg-[rgba(17,19,26,0.9)] px-2 py-1 text-[11px] text-ink outline-none placeholder:text-[#8ea4b8]"
        />
        <button
          type="button"
          onClick={applyRadarFilter}
          className="shrink-0 rounded-md border border-[rgba(121,178,222,0.55)] bg-[rgba(34,62,86,0.75)] px-2.5 py-1 text-[11px] font-semibold text-[#d8ebff]"
        >
          apply filter
        </button>
        <button
          type="button"
          onClick={clearRadarFilter}
          className="shrink-0 rounded-md border border-[var(--line)] bg-[rgba(40,44,56,0.88)] px-2.5 py-1 text-[11px] font-semibold text-ink"
        >
          clear
        </button>
      </div>

      {radarFilter ? <p className="mt-1 text-[11px] text-[#a8bfd3]">scope {scopeLabel}: <code>{radarFilter}</code></p> : null}

      <div
        className="mt-2 grid gap-1.5"
        style={{ gridTemplateColumns: "repeat(auto-fit, minmax(8rem, 1fr))" }}
      >
        <div className="rounded-md border border-[rgba(131,89,62,0.44)] bg-[rgba(26,18,16,0.88)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.08em] text-[#d2a279]">critical</p>
          <p className="text-[1.06rem] font-semibold text-[#ffe4bf] tabular-nums">{levelCounts.critical}</p>
          <p className="text-[10px] text-[#e0c7b1]">total {levelCounts.count}</p>
        </div>
        <div className="rounded-md border border-[rgba(131,89,62,0.44)] bg-[rgba(26,18,16,0.88)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.08em] text-[#d2a279]">high</p>
          <p className="text-[1.06rem] font-semibold text-[#ffe4bf] tabular-nums">{levelCounts.high}</p>
          <p className="text-[10px] text-[#e0c7b1]">medium {levelCounts.medium}</p>
        </div>
        <div className="rounded-md border border-[rgba(131,89,62,0.44)] bg-[rgba(26,18,16,0.88)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.08em] text-[#d2a279]">runtime</p>
          <p className="text-[1.06rem] font-semibold capitalize text-[#ffe4bf]">{runtimeState?.last_status || "idle"}</p>
          <p className="text-[10px] text-[#e0c7b1]">interval {Math.round(Number(runtime?.interval_seconds || 0))}s</p>
        </div>
        <div className="rounded-md border border-[rgba(131,89,62,0.44)] bg-[rgba(26,18,16,0.88)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.08em] text-[#d2a279]">muse</p>
          <p className="truncate text-[0.92rem] font-semibold text-[#ffe4bf]">{runtimeLabel}</p>
          <p className="truncate text-[10px] text-[#e0c7b1]">{scoringLabel} | {llmStatusLabel}</p>
        </div>
        <div className="rounded-md border border-[rgba(131,89,62,0.44)] bg-[rgba(26,18,16,0.88)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.08em] text-[#d2a279]">coverage</p>
          <p className="text-[1.06rem] font-semibold text-[#ffe4bf] tabular-nums">{sourceCount}</p>
          <p className="text-[10px] text-[#e0c7b1]">sources | low {levelCounts.low}</p>
        </div>
      </div>

      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[#a9c2d6]">
        <span>stream {isConnected ? "ws live" : "ws offline"}</span>
        <span>last run {lastRunLabel}</span>
        <span>signal families {signalRows.length}</span>
        {radarMode === "global" && scopedThreats.length === 0 && provisionalSeedThreats.length > 0 ? (
          <span>seed queue {provisionalSeedThreats.length}</span>
        ) : null}
        {runtimeState?.last_skipped_reason ? <span>skip {runtimeState.last_skipped_reason}</span> : null}
      </div>

      {runtimeState?.last_error ? (
        <p className="mt-3 rounded-md border border-[#b91c1c] bg-[rgba(127,29,29,0.22)] px-3 py-2 text-xs text-[#fecaca]">
          runtime error: {runtimeState.last_error}
        </p>
      ) : null}

      {error ? (
        <p className="mt-3 rounded-md border border-[#b91c1c] bg-[rgba(127,29,29,0.22)] px-3 py-2 text-xs text-[#fecaca]">
          {error}
        </p>
      ) : null}

      <div
        className="mt-2 grid gap-2"
        style={{ gridTemplateColumns: "repeat(auto-fit, minmax(17rem, 1fr))" }}
      >
        <section className="rounded-lg border border-[var(--line)] bg-[rgba(22,25,32,0.72)] p-2">
          <p className="text-xs uppercase tracking-[0.12em] text-[#f4b4b4]">
            {radarMode === "local" ? "Hot Repos" : "Hot Domains / Kinds"}
          </p>
          <div className="mt-1.5 space-y-1.5">
            {hotRows.map((row) => (
              <div
                key={row.id}
                className="flex items-center justify-between rounded-md bg-[rgba(35,39,49,0.84)] px-2 py-1"
              >
                <p className="text-xs font-mono text-[#e7eff7]">{row.label}</p>
                <p className="text-[11px] text-[#fda4af]">risk {row.score}</p>
              </div>
            ))}
            {hotRows.length === 0 ? (
              <p className="text-xs text-muted">No hot entries in current window.</p>
            ) : null}
          </div>

          <div className="mt-2 rounded-md border border-[rgba(116,168,207,0.4)] bg-[rgba(18,28,40,0.55)] p-2">
            <p className="text-[10px] uppercase tracking-[0.1em] text-[#9cc4df]">signal mix</p>
            <div className="mt-1.5 space-y-1.5">
              {signalRows.map((row) => (
                <div
                  key={row.signal}
                  className="flex items-center justify-between rounded bg-[rgba(16,22,31,0.72)] px-2 py-1"
                >
                  <p className="truncate text-[11px] text-[#cfe6f8]">{shortSignal(row.signal)}</p>
                  <p className="text-[10px] text-[#9cc4df]">{row.count}</p>
                </div>
              ))}
              {signalRows.length === 0 ? (
                <p className="text-xs text-muted">No signal labels resolved yet.</p>
              ) : null}
            </div>
          </div>

          <div className="mt-2 rounded-md border border-[rgba(116,168,207,0.45)] bg-[rgba(20,32,46,0.58)] p-2">
            <p className="text-[10px] uppercase tracking-[0.1em] text-[#9cc4df]">source signal inputs</p>
            <div className="mt-1.5 space-y-1 max-h-[10rem] overflow-y-auto pr-1">
              {sourceSignalRows.map((sourceRow) => (
                <div
                  key={sourceRow.label}
                  className="flex items-center justify-between gap-2 rounded bg-[rgba(18,24,33,0.68)] px-2 py-1"
                >
                  <p className="truncate text-[11px] text-[#cfe6f8]">{sourceRow.label}</p>
                  <p className="text-[10px] text-[#9cc4df]">{sourceRow.count}</p>
                </div>
              ))}
              {sourceSignalRows.length === 0 ? (
                <p className="text-xs text-muted">No source signal classes reported by this radar query.</p>
              ) : null}
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-[var(--line)] bg-[rgba(22,25,32,0.72)] p-2">
          <p className="text-xs uppercase tracking-[0.12em] text-[#f4b4b4]">Top Threats</p>
          <div className="mt-1.5 space-y-1.5 max-h-[22rem] overflow-y-auto pr-1">
            {topThreats.map((row) => {
              const rowRepo = String(row.repo || "").trim();
              const rowNumber = Math.max(0, Number(row.number || 0));
              const rowIdentity = rowRepo ? `${rowRepo} #${rowNumber}` : (row.kind || "resource");
              const signalTokens = (Array.isArray(row.signals) && row.signals.length > 0)
                ? row.signals
                : (Array.isArray(row.labels) ? row.labels : []);
              const cveTokens = Array.isArray(row.cves) ? row.cves : [];
              return (
                <article
                  key={threatRowKey(row)}
                  className={`rounded-md border px-2 py-1.5 ${selectedThreatKey === threatRowKey(row)
                    ? "border-[rgba(246,113,113,0.72)] bg-[rgba(70,33,38,0.72)]"
                    : "border-[var(--line)] bg-[rgba(35,39,49,0.84)]"}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-mono text-[#dbe8f5]">{rowIdentity}</p>
                    <span
                      className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${badgeClass(row.risk_level)}`}
                    >
                      {row.risk_level} {row.risk_score}
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs text-ink">{row.title || "(untitled)"}</p>
                  <p className="mt-0.5 text-[10px] text-muted">
                    {signalTokens.slice(0, 3).map(shortSignal).join(" | ") || "no explicit signals"}
                    {cveTokens.length > 0 ? ` | ${cveTokens.join(", ")}` : ""}
                    {Number.isFinite(Number(row.llm_score)) ? ` | llm ${Math.max(0, Number(row.llm_score || 0))}` : ""}
                  </p>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        selectThreat(row);
                      }}
                      className="rounded border border-[rgba(137,189,226,0.5)] bg-[rgba(27,55,78,0.72)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#d9efff]"
                    >
                      {radarMode === "local" && supportsConversation(row) ? "thread" : "details"}
                    </button>
                    {row.canonical_url ? (
                      <a
                        href={row.canonical_url}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded border border-[rgba(165,190,213,0.44)] bg-[rgba(35,43,59,0.72)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#dbe8f5]"
                      >
                        open
                      </a>
                    ) : null}
                  </div>
                </article>
              );
            })}

            {topThreats.length === 0 && !loading ? (
              <>
                <p className="text-xs text-muted">
                  {radarMode === "local"
                    ? "No GitHub threat candidates yet."
                    : provisionalSeedThreats.length > 0
                      ? `No classified global threat signals yet. ${provisionalSeedThreats.length} watch seeds are queued for crawl evidence.`
                      : "No global geopolitics rows yet. Keep crawl running and trigger refresh."}
                </p>
              </>
            ) : null}
            {loading || refreshing ? (
              <p className="text-xs text-muted">refreshing radar...</p>
            ) : null}
          </div>
        </section>
      </div>

      <section className="mt-2 min-h-[16rem] flex-1 rounded-lg border border-[var(--line)] bg-[rgba(10,14,22,0.82)] p-2 overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs uppercase tracking-[0.12em] text-[#f4b4b4]">
            {radarMode === "local" ? "Conversation Context" : "Global Source Context"}
          </p>
          {selectedThreat ? (
            <p className="text-[11px] font-mono text-[#c2d8ea]">
              {radarMode === "local"
                ? `${String(selectedThreat.repo || "")} #${Math.max(0, Number(selectedThreat.number || 0))}`
                : (selectedThreat.domain || selectedThreat.kind || "global:resource")}
            </p>
          ) : null}
        </div>

        {museChatPanel ? (
          <div className="mt-2 grid gap-2 xl:grid-cols-[minmax(14rem,1fr)_minmax(0,1.8fr)]">
            <div className="rounded-md border border-[rgba(116,168,207,0.45)] bg-[rgba(16,30,44,0.62)] p-2">
              <p className="text-[10px] uppercase tracking-[0.1em] text-[#9cc4df]">focused threat context</p>
              {selectedThreat == null ? (
                <p className="mt-1.5 text-xs text-muted">Select a threat row to enrich muse context.</p>
              ) : (
                <div className="mt-1.5 space-y-1.5 text-xs text-[#d8ebff]">
                  <p>{selectedThreat.title || "(untitled)"}</p>
                  <p className="text-[#b6ccde]">
                    signals {
                      ((Array.isArray(selectedThreat.labels) ? selectedThreat.labels : selectedThreat.signals || [])
                        .slice(0, 4)
                        .map(shortSignal)
                        .join(" | ")) || "none"
                    }
                  </p>
                  {selectedThreat.threat_metrics ? (
                    <p className="text-[#9dc4e2]">
                      llm overall {Math.round(Number(selectedThreat.threat_metrics.overall_score || 0))}
                      {` | confidence ${Math.round(Number(selectedThreat.threat_metrics.confidence || 0))}`}
                      {` | severity ${Math.round(Number(selectedThreat.threat_metrics.severity || 0))}`}
                    </p>
                  ) : null}
                  {selectedThreat.canonical_url ? (
                    <a
                      href={selectedThreat.canonical_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-block underline decoration-dotted text-[#b6ccde]"
                    >
                      source
                    </a>
                  ) : null}
                </div>
              )}
              <p className="mt-2 text-[11px] text-[#b9d2e4]">
                Pin nearby nexus in muse chat to keep stable context and let generation include semantically related rows.
              </p>
            </div>
            <div className="min-h-[18rem]">{museChatPanel}</div>
          </div>
        ) : (
          <>
            {selectedThreat == null ? (
              <p className="mt-1.5 text-xs text-muted">Select a threat row to inspect context.</p>
            ) : null}

            {radarMode === "global" && selectedThreat != null ? (
              <div className="mt-1.5 space-y-1.5 text-xs text-[#d8ebff]">
                <p>{selectedThreat.title || "(untitled)"}</p>
                <p className="text-[#b6ccde]">
                  signals {
                    ((Array.isArray(selectedThreat.labels) ? selectedThreat.labels : selectedThreat.signals || [])
                      .slice(0, 4)
                      .map(shortSignal)
                      .join(" | ")) || "none"
                  }
                </p>
                {selectedThreat.threat_metrics ? (
                  <p className="text-[#9dc4e2]">
                    llm overall {Math.round(Number(selectedThreat.threat_metrics.overall_score || 0))}
                    {` | confidence ${Math.round(Number(selectedThreat.threat_metrics.confidence || 0))}`}
                    {` | severity ${Math.round(Number(selectedThreat.threat_metrics.severity || 0))}`}
                  </p>
                ) : null}
                {selectedThreat.canonical_url ? (
                  <a
                    href={selectedThreat.canonical_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-block underline decoration-dotted text-[#b6ccde]"
                  >
                    source
                  </a>
                ) : null}

                <div className="rounded-md border border-[rgba(116,168,207,0.45)] bg-[rgba(16,30,44,0.62)] p-2">
                  <p className="text-[10px] uppercase tracking-[0.1em] text-[#9cc4df]">talk to this feed</p>
                  <div className="mt-1.5 flex items-center gap-1.5">
                    <input
                      value={globalAskInput}
                      onChange={(event) => {
                        setGlobalAskInput(event.currentTarget.value);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          askGlobalFeed().catch(() => {});
                        }
                      }}
                      placeholder="ask: what changed and why does it matter?"
                      className="min-w-[12rem] flex-1 rounded-md border border-[var(--line)] bg-[rgba(12,18,27,0.9)] px-2 py-1 text-[11px] text-ink outline-none placeholder:text-[#8ea4b8]"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        askGlobalFeed().catch(() => {});
                      }}
                      disabled={globalAskLoading}
                      className="rounded border border-[rgba(137,189,226,0.5)] bg-[rgba(27,55,78,0.72)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#d9efff] disabled:opacity-55"
                    >
                      {globalAskLoading ? "asking..." : "ask"}
                    </button>
                  </div>
                  {globalAskError ? (
                    <p className="mt-1.5 rounded border border-[#b91c1c] bg-[rgba(127,29,29,0.24)] px-2 py-1 text-[11px] text-[#fecaca]">
                      {globalAskError}
                    </p>
                  ) : null}
                  {globalAskReply ? (
                    <>
                      <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[10px] text-[#9dc4e2]">
                        {globalAskTurnId ? <span>turn {globalAskTurnId}</span> : null}
                        <span>mode deterministic</span>
                      </div>
                      <pre className="mt-1 max-h-[12rem] overflow-auto whitespace-pre-wrap break-words rounded border border-[rgba(133,180,217,0.34)] bg-[rgba(7,18,31,0.8)] px-2 py-1.5 text-[11px] leading-5 text-[#d8ebff]">
                        {globalAskReply}
                      </pre>
                    </>
                  ) : null}
                </div>
              </div>
            ) : null}

            {radarMode === "local" && selectedThreat != null && !selectedThreatSupportsConversation ? (
              <>
                {selectedThreat.threat_metrics ? (
                  <p className="mt-2 text-xs text-[#9dc4e2]">
                    llm overall {Math.round(Number(selectedThreat.threat_metrics.overall_score || 0))}
                    {` | confidence ${Math.round(Number(selectedThreat.threat_metrics.confidence || 0))}`}
                    {` | severity ${Math.round(Number(selectedThreat.threat_metrics.severity || 0))}`}
                  </p>
                ) : null}
                <p className="mt-2 text-xs text-muted">
                  Thread preview is available for issue and pull request rows. Use <code>open</code> for this item.
                </p>
              </>
            ) : null}

            {radarMode === "local" && selectedThreatSupportsConversation && conversationLoading ? (
              <p className="mt-2 text-xs text-muted">loading conversation chain...</p>
            ) : null}

            {radarMode === "local" && selectedThreatSupportsConversation && conversationError ? (
              <p className="mt-2 rounded-md border border-[#b91c1c] bg-[rgba(127,29,29,0.24)] px-2 py-1.5 text-xs text-[#fecaca]">
                {conversationError}
              </p>
            ) : null}

            {radarMode === "local" && selectedThreatSupportsConversation && !conversationLoading && !conversationError ? (
              <>
                {selectedThreat?.threat_metrics ? (
                  <p className="mt-2 text-xs text-[#9dc4e2]">
                    llm overall {Math.round(Number(selectedThreat.threat_metrics.overall_score || 0))}
                    {` | confidence ${Math.round(Number(selectedThreat.threat_metrics.confidence || 0))}`}
                    {` | severity ${Math.round(Number(selectedThreat.threat_metrics.severity || 0))}`}
                  </p>
                ) : null}
                <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-[#b6ccde]">
                  <span>comments {conversationCommentCount}</span>
                  {conversationUrl ? (
                    <a
                      href={conversationUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="underline decoration-dotted"
                    >
                      source
                    </a>
                  ) : null}
                </div>
                <pre className="mt-2 max-h-[24rem] overflow-auto whitespace-pre-wrap break-words rounded-md border border-[rgba(133,180,217,0.34)] bg-[rgba(7,18,31,0.8)] px-3 py-2 text-[11px] leading-5 text-[#d8ebff]">
                  {conversationMarkdown || "no conversation markdown returned"}
                </pre>
              </>
            ) : null}
          </>
        )}
      </section>
    </section>
  );
}
