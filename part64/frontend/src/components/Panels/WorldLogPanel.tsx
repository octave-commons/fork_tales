import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, RefreshCw, Workflow } from "lucide-react";
import { runtimeApiUrl } from "../../runtime/endpoints";
import type { Catalog, WorldLogEvent, WorldLogPayload } from "../../types";

interface Props {
  catalog: Catalog | null;
}

function relativeTime(isoText: string): string {
  const raw = String(isoText || "").trim();
  if (!raw) {
    return "n/a";
  }
  const parsed = Date.parse(raw);
  if (!Number.isFinite(parsed)) {
    return raw;
  }
  const seconds = Math.max(0, Math.round((Date.now() - parsed) / 1000));
  if (seconds < 45) {
    return `${seconds}s ago`;
  }
  if (seconds < 3600) {
    return `${Math.round(seconds / 60)}m ago`;
  }
  if (seconds < 86400) {
    return `${Math.round(seconds / 3600)}h ago`;
  }
  return `${Math.round(seconds / 86400)}d ago`;
}

function compactText(text: string, limit = 240): string {
  const value = String(text || "").trim();
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, Math.max(1, limit - 3))}...`;
}

function worldLogPreviewImageUrl(source: string, refs: string[]): string {
  if (source !== "nasa_gibs") {
    return "";
  }
  for (const ref of refs) {
    const value = String(ref || "").trim();
    if (!value) {
      continue;
    }
    if (!value.startsWith("https://") && !value.startsWith("http://")) {
      continue;
    }
    const lower = value.toLowerCase();
    if (/\.(png|jpg|jpeg|webp)(\?|$)/.test(lower)) {
      return value;
    }
  }
  return "";
}

function eventTone(kind: string): string {
  const lower = String(kind || "").toLowerCase();
  if (lower.includes("pending")) {
    return "text-[#fd971f]";
  }
  if (lower.includes("error") || lower.includes("fail") || lower.includes("reject")) {
    return "text-[#f92672]";
  }
  if (lower.includes("ingested") || lower.includes("recorded")) {
    return "text-[#a6e22e]";
  }
  return "text-[#66d9ef]";
}

export function WorldLogPanel({ catalog }: Props) {
  const [payload, setPayload] = useState<WorldLogPayload | null>(catalog?.world_log ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [syncStatus, setSyncStatus] = useState("");

  useEffect(() => {
    if (catalog?.world_log) {
      setPayload(catalog.world_log);
    }
  }, [catalog?.world_log]);

  const fetchWorldLog = useCallback(async (withSpinner = true) => {
    if (withSpinner) {
      setLoading(true);
    }
    setError("");
    try {
      const response = await fetch(runtimeApiUrl("/api/world/events?limit=180"));
      if (!response.ok) {
        throw new Error(`world log request failed (${response.status})`);
      }
      const data = (await response.json()) as WorldLogPayload;
      if (!data || !Array.isArray(data.events)) {
        throw new Error("invalid world log payload");
      }
      setPayload(data);
    } catch (fetchError) {
      const message = fetchError instanceof Error ? fetchError.message : "world log unavailable";
      setError(message);
    } finally {
      if (withSpinner) {
        setLoading(false);
      }
    }
  }, []);

  const triggerInboxSync = useCallback(async () => {
    setSyncStatus("scheduling inbox ingest...");
    try {
      const response = await fetch(runtimeApiUrl("/api/eta-mu/sync"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: true, wait: false }),
      });
      const data = (await response.json()) as {
        ok?: boolean;
        status?: string;
        error?: string;
      };
      if (!response.ok || data?.ok !== true) {
        throw new Error(String(data?.error || `eta-mu sync failed (${response.status})`));
      }
      setSyncStatus(`inbox sync ${String(data.status || "scheduled")}`);
      void fetchWorldLog(false);
    } catch (syncError) {
      const message = syncError instanceof Error ? syncError.message : "inbox sync failed";
      setSyncStatus(message);
    }
  }, [fetchWorldLog]);

  useEffect(() => {
    void fetchWorldLog(true);
    const interval = window.setInterval(() => {
      void fetchWorldLog(false);
    }, 5500);
    return () => {
      window.clearInterval(interval);
    };
  }, [fetchWorldLog]);

  const events = payload?.events ?? [];
  const topSources = useMemo(
    () => Object.entries(payload?.sources ?? {}).sort((left, right) => right[1] - left[1]).slice(0, 4),
    [payload?.sources],
  );
  const topKinds = useMemo(
    () => Object.entries(payload?.kinds ?? {}).sort((left, right) => right[1] - left[1]).slice(0, 4),
    [payload?.kinds],
  );

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-[rgba(102,217,239,0.3)] bg-[rgba(39,40,34,0.88)] p-3">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <p className="text-sm font-semibold text-ink">World Log Stream</p>
            <p className="text-xs text-muted mt-1">
              Append-only event feed anchored to embeddings and cross-event relations.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => {
                void fetchWorldLog(true);
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
                void triggerInboxSync();
              }}
              className="border border-[rgba(166,226,46,0.48)] rounded-md bg-[rgba(56,66,22,0.36)] px-3 py-1.5 text-xs font-semibold text-[#c7f06f] hover:bg-[rgba(56,66,22,0.54)]"
            >
              ingest .ημ now
            </button>
          </div>
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">events</p>
            <p className="text-sm font-semibold text-ink">{payload?.count ?? 0}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">pending inbox</p>
            <p className="text-sm font-semibold text-ink">{payload?.pending_inbox ?? 0}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">relation links</p>
            <p className="text-sm font-semibold text-ink">{payload?.relation_count ?? 0}</p>
          </div>
        </div>

        {topSources.length > 0 ? (
          <p className="mt-2 text-[11px] text-muted">
            <Activity size={12} className="inline mr-1" />
            sources {topSources.map(([key, value]) => `${key}:${value}`).join(" | ")}
          </p>
        ) : null}
        {topKinds.length > 0 ? (
          <p className="text-[11px] text-muted">
            <Workflow size={12} className="inline mr-1" />
            kinds {topKinds.map(([key, value]) => `${key}:${value}`).join(" | ")}
          </p>
        ) : null}
        {syncStatus ? <p className="text-[11px] text-[#c8e6ff] mt-1">{syncStatus}</p> : null}
        {error ? <p className="text-[11px] text-[#ffcfbf] mt-1">{error}</p> : null}
      </div>

      <div className="max-h-[32rem] overflow-y-auto pr-1 space-y-2">
        {events.length === 0 ? (
          <p className="text-xs text-muted">No world-log events yet.</p>
        ) : (
          events.map((event: WorldLogEvent) => {
            const relations = Array.isArray(event.relations) ? event.relations : [];
            const refs = Array.isArray(event.refs) ? event.refs : [];
            const previewImageUrl = worldLogPreviewImageUrl(event.source, refs);
            return (
              <article
                key={event.id}
                className="rounded-lg border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2"
              >
                <p className="text-sm font-semibold text-ink">{compactText(event.title, 140)}</p>
                <p className="text-[11px] text-muted font-mono mt-0.5">
                  <span className={eventTone(event.kind)}>{event.kind}</span>
                  {" | "}
                  {event.source}
                  {" | "}
                  {relativeTime(event.ts)}
                </p>
                {event.detail ? (
                  <p className="text-xs text-muted mt-1 whitespace-pre-wrap break-words">
                    {compactText(event.detail, 220)}
                  </p>
                ) : null}
                {previewImageUrl ? (
                  <div className="mt-2 overflow-hidden rounded-md border border-[rgba(102,217,239,0.26)] bg-[rgba(18,20,18,0.74)]">
                    <img
                      src={previewImageUrl}
                      alt={`${event.title} tile`}
                      loading="lazy"
                      referrerPolicy="no-referrer"
                      className="block h-24 w-full object-cover"
                    />
                  </div>
                ) : null}
                {refs.length > 0 ? (
                  <p className="text-[11px] text-[#9fd0ef] mt-1 break-all">
                    refs: {refs.slice(0, 3).join(" | ")}
                  </p>
                ) : null}
                {relations.length > 0 ? (
                  <p className="text-[11px] text-[#b9d5e8] mt-1">
                    related: {relations.slice(0, 3).map((row) => `${row.event_id.slice(0, 8)}(${row.score.toFixed(2)})`).join(" · ")}
                  </p>
                ) : null}
              </article>
            );
          })
        )}
      </div>
    </div>
  );
}
