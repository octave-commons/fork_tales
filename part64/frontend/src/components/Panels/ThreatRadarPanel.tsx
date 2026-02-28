import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { runtimeApiUrl } from "../../runtime/endpoints";

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
  repo: string;
  kind: string;
  number: number;
  title: string;
  canonical_url: string;
  state: string;
  signals: string[];
  cves: string[];
}

interface ThreatRadarResult {
  count: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  hot_repos: Array<{ repo: string; max_risk_score: number }>;
  threats: ThreatRow[];
}

interface ThreatRadarReport {
  ok: boolean;
  runtime?: ThreatRadarRuntime;
  result?: ThreatRadarResult;
}

interface ThreatConversationPayload {
  ok: boolean;
  markdown?: string;
  comment_count?: number;
  url?: string;
  error?: string;
}

const DEFAULT_RESULT: ThreatRadarResult = {
  count: 0,
  critical_count: 0,
  high_count: 0,
  medium_count: 0,
  low_count: 0,
  hot_repos: [],
  threats: [],
};

const REPORT_WINDOW_TICKS = 1440;
const REPORT_LIMIT = 16;
const CONVERSATION_MAX_COMMENTS = 80;
const CONVERSATION_MAX_MARKDOWN_CHARS = 80_000;

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

function threatRowKey(row: ThreatRow): string {
  return `${row.repo}#${row.number}:${row.canonical_url}`;
}

function supportsConversation(row: ThreatRow): boolean {
  return row.kind === "github:issue" || row.kind === "github:pr";
}

export function ThreatRadarPanel() {
  const [report, setReport] = useState<ThreatRadarReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [forcingTick, setForcingTick] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [repoFilterInput, setRepoFilterInput] = useState("");
  const [repoFilter, setRepoFilter] = useState("");
  const [selectedThreat, setSelectedThreat] = useState<ThreatRow | null>(null);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [conversationMarkdown, setConversationMarkdown] = useState("");
  const [conversationCommentCount, setConversationCommentCount] = useState(0);
  const [conversationUrl, setConversationUrl] = useState("");
  const [conversationError, setConversationError] = useState<string | null>(null);

  const criticalCountPrimedRef = useRef(false);
  const previousCriticalCountRef = useRef(0);

  const fetchReport = useCallback(async (mode: "initial" | "refresh" = "refresh") => {
    if (mode === "initial") {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError(null);
    try {
      const params = new URLSearchParams({
        window_ticks: String(REPORT_WINDOW_TICKS),
        limit: String(REPORT_LIMIT),
      });
      if (repoFilter) {
        params.set("repo", repoFilter);
      }
      const response = await fetch(runtimeApiUrl(`/api/muse/threat-radar/report?${params.toString()}`));
      if (!response.ok) {
        throw new Error(`threat radar report request failed (${response.status})`);
      }
      const payload = (await response.json()) as ThreatRadarReport;
      setReport(payload);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [repoFilter]);

  const loadConversation = useCallback(async (threat: ThreatRow) => {
    if (!supportsConversation(threat) || !threat.repo || threat.number <= 0) {
      setConversationMarkdown("");
      setConversationCommentCount(0);
      setConversationUrl(threat.canonical_url || "");
      setConversationError("thread preview is available for issue and pull request rows");
      return;
    }

    setConversationLoading(true);
    setConversationError(null);
    try {
      const params = new URLSearchParams({
        repo: threat.repo,
        number: String(Math.max(1, threat.number)),
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
    loadConversation(threat).catch(() => {});
  }, [loadConversation]);

  const applyRepoFilter = useCallback(() => {
    setRepoFilter(repoFilterInput.trim().toLowerCase());
  }, [repoFilterInput]);

  const clearRepoFilter = useCallback(() => {
    setRepoFilterInput("");
    setRepoFilter("");
  }, []);

  const forceTick = useCallback(async () => {
    setForcingTick(true);
    try {
      const response = await fetch(runtimeApiUrl("/api/muse/threat-radar/tick?force=true"));
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
    const timer = window.setInterval(() => {
      fetchReport("refresh").catch(() => {});
    }, 30000);
    return () => {
      window.clearInterval(timer);
    };
  }, [fetchReport]);

  const result = report?.result ?? DEFAULT_RESULT;
  const runtime = report?.runtime;
  const runtimeState = runtime?.state;
  const topThreats = useMemo(() => result.threats.slice(0, 8), [result.threats]);
  const selectedThreatKey = selectedThreat ? threatRowKey(selectedThreat) : "";

  useEffect(() => {
    if (topThreats.length === 0) {
      setSelectedThreat(null);
      setConversationMarkdown("");
      setConversationCommentCount(0);
      setConversationError(null);
      setConversationUrl("");
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
    const nextCriticalCount = Math.max(0, Number(result.critical_count || 0));
    const top = result.threats[0] || null;
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
  }, [loading, result.critical_count, result.threats]);

  const selectedThreatSupportsConversation = selectedThreat
    ? supportsConversation(selectedThreat)
    : false;

  return (
    <section className="card relative overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full bg-[#ef4444] opacity-75" />
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold mb-1">Threat Radar / GitHub Security Watch</h2>
          <p className="text-muted text-sm">
            Proactive ranking of security-relevant PR, issue, and advisory signals from the live GitHub graph.
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            forceTick().catch(() => {});
          }}
          disabled={forcingTick}
          className="rounded-md border border-[var(--line)] bg-[rgba(32,35,44,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(44,49,61,0.95)] disabled:opacity-55"
        >
          {forcingTick ? "running..." : "run now"}
        </button>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-[1fr_auto_auto]">
        <input
          value={repoFilterInput}
          onChange={(event) => {
            setRepoFilterInput(event.currentTarget.value);
          }}
          placeholder="filter repo (owner/name)"
          className="rounded-md border border-[var(--line)] bg-[rgba(17,19,26,0.9)] px-2.5 py-1.5 text-xs text-ink outline-none placeholder:text-[#8ea4b8]"
        />
        <button
          type="button"
          onClick={applyRepoFilter}
          className="rounded-md border border-[rgba(121,178,222,0.55)] bg-[rgba(34,62,86,0.75)] px-3 py-1.5 text-xs font-semibold text-[#d8ebff]"
        >
          apply filter
        </button>
        <button
          type="button"
          onClick={clearRepoFilter}
          className="rounded-md border border-[var(--line)] bg-[rgba(40,44,56,0.88)] px-3 py-1.5 text-xs font-semibold text-ink"
        >
          clear
        </button>
      </div>

      {repoFilter ? <p className="mt-1 text-[11px] text-[#a8bfd3]">scope: <code>{repoFilter}</code></p> : null}

      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="mindfuck-card">
          <p className="mindfuck-k">critical</p>
          <p className="mindfuck-v">{result.critical_count}</p>
          <p className="mindfuck-small">total {result.count}</p>
        </div>
        <div className="mindfuck-card">
          <p className="mindfuck-k">high</p>
          <p className="mindfuck-v">{result.high_count}</p>
          <p className="mindfuck-small">medium {result.medium_count}</p>
        </div>
        <div className="mindfuck-card">
          <p className="mindfuck-k">runtime</p>
          <p className="mindfuck-v capitalize">{runtimeState?.last_status || "idle"}</p>
          <p className="mindfuck-small">interval {Math.round(Number(runtime?.interval_seconds || 0))}s</p>
        </div>
        <div className="mindfuck-card">
          <p className="mindfuck-k">muse</p>
          <p className="mindfuck-v">{runtime?.label || runtime?.muse_id || "-"}</p>
          <p className="mindfuck-small">low {result.low_count}</p>
        </div>
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

      <div className="mt-5 grid gap-4 xl:grid-cols-2">
        <section className="rounded-lg border border-[var(--line)] bg-[rgba(22,25,32,0.72)] p-3">
          <p className="text-xs uppercase tracking-[0.12em] text-[#f4b4b4]">Hot Repos</p>
          <div className="mt-2 space-y-2">
            {result.hot_repos.slice(0, 8).map((row) => (
              <div
                key={row.repo}
                className="flex items-center justify-between rounded-md bg-[rgba(35,39,49,0.84)] px-2 py-1.5"
              >
                <p className="text-xs font-mono text-[#e7eff7]">{row.repo}</p>
                <p className="text-xs text-[#fda4af]">risk {row.max_risk_score}</p>
              </div>
            ))}
            {result.hot_repos.length === 0 ? (
              <p className="text-xs text-muted">No risky repos in current window.</p>
            ) : null}
          </div>
        </section>

        <section className="rounded-lg border border-[var(--line)] bg-[rgba(22,25,32,0.72)] p-3">
          <p className="text-xs uppercase tracking-[0.12em] text-[#f4b4b4]">Top Threats</p>
          <div className="mt-2 space-y-2 max-h-[19rem] overflow-y-auto pr-1">
            {topThreats.map((row) => (
              <article
                key={threatRowKey(row)}
                className={`rounded-md border p-2 ${selectedThreatKey === threatRowKey(row)
                  ? "border-[rgba(246,113,113,0.72)] bg-[rgba(70,33,38,0.72)]"
                  : "border-[var(--line)] bg-[rgba(35,39,49,0.84)]"}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-mono text-[#dbe8f5]">{row.repo} #{row.number}</p>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase ${badgeClass(row.risk_level)}`}
                  >
                    {row.risk_level} {row.risk_score}
                  </span>
                </div>
                <p className="mt-1 text-xs text-ink">{row.title || "(untitled)"}</p>
                <p className="mt-1 text-[11px] text-muted">
                  {row.signals.slice(0, 3).map(shortSignal).join(" | ") || "no explicit signals"}
                  {row.cves.length > 0 ? ` | ${row.cves.join(", ")}` : ""}
                </p>
                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      selectThreat(row);
                    }}
                    className="rounded border border-[rgba(137,189,226,0.5)] bg-[rgba(27,55,78,0.72)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#d9efff]"
                  >
                    {supportsConversation(row) ? "thread" : "details"}
                  </button>
                  {row.canonical_url ? (
                    <a
                      href={row.canonical_url}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded border border-[rgba(165,190,213,0.44)] bg-[rgba(35,43,59,0.72)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#dbe8f5]"
                    >
                      open
                    </a>
                  ) : null}
                </div>
              </article>
            ))}

            {topThreats.length === 0 && !loading ? (
              <p className="text-xs text-muted">No threat candidates yet.</p>
            ) : null}
            {loading || refreshing ? (
              <p className="text-xs text-muted">refreshing radar...</p>
            ) : null}
          </div>
        </section>
      </div>

      <section className="mt-4 rounded-lg border border-[var(--line)] bg-[rgba(10,14,22,0.82)] p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs uppercase tracking-[0.12em] text-[#f4b4b4]">Conversation Context</p>
          {selectedThreat ? (
            <p className="text-[11px] font-mono text-[#c2d8ea]">
              {selectedThreat.repo} #{selectedThreat.number}
            </p>
          ) : null}
        </div>

        {selectedThreat == null ? (
          <p className="mt-2 text-xs text-muted">Select a threat row to inspect conversation context.</p>
        ) : null}

        {selectedThreat != null && !selectedThreatSupportsConversation ? (
          <p className="mt-2 text-xs text-muted">
            Thread preview is available for issue and pull request rows. Use <code>open</code> for this item.
          </p>
        ) : null}

        {selectedThreatSupportsConversation && conversationLoading ? (
          <p className="mt-2 text-xs text-muted">loading conversation chain...</p>
        ) : null}

        {selectedThreatSupportsConversation && conversationError ? (
          <p className="mt-2 rounded-md border border-[#b91c1c] bg-[rgba(127,29,29,0.24)] px-2 py-1.5 text-xs text-[#fecaca]">
            {conversationError}
          </p>
        ) : null}

        {selectedThreatSupportsConversation && !conversationLoading && !conversationError ? (
          <>
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
      </section>
    </section>
  );
}
