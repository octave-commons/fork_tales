import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, Gauge, RefreshCw, ShieldAlert } from "lucide-react";
import { relativeTime } from "../../app/time";
import { runtimeApiUrl } from "../../runtime/endpoints";
import type {
  Catalog,
  CouncilDecision,
  CouncilSnapshot,
  DriftScanPayload,
  SimulationState,
  StudySnapshotPayload,
  StudyWarning,
  TaskQueueSnapshot,
} from "../../types";

interface Props {
  catalog: Catalog | null;
  simulation: SimulationState | null;
}

interface QueueApiResponse {
  ok: boolean;
  queue: TaskQueueSnapshot;
}

interface LegacyCouncilResponse {
  ok: boolean;
  council: CouncilSnapshot;
}

interface StudyExportResponse {
  ok: boolean;
  event?: {
    id?: string;
    ts?: string;
    label?: string;
  };
  history?: {
    count?: number;
    path?: string;
  };
}

interface DriftSeverityCount {
  high: number;
  medium: number;
  low: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function shortPath(pathValue: string | undefined): string {
  const source = String(pathValue || "").trim();
  if (!source) {
    return "(n/a)";
  }
  if (source.length <= 68) {
    return source;
  }
  return `...${source.slice(-65)}`;
}

function decisionStatusClass(status: string): string {
  if (status === "executed" || status === "approved") {
    return "text-[#a6e22e]";
  }
  if (status === "blocked") {
    return "text-[#fd971f]";
  }
  if (status === "error") {
    return "text-[#f92672]";
  }
  return "text-[#66d9ef]";
}

function resourceStatusClass(status: string): string {
  const normalized = status.trim().toLowerCase();
  if (normalized === "hot") {
    return "text-[#f92672]";
  }
  if (normalized === "watch") {
    return "text-[#fd971f]";
  }
  if (normalized === "ok") {
    return "text-[#a6e22e]";
  }
  return "text-muted";
}

function summarizeDriftSeverities(payload: DriftScanPayload | null): DriftSeverityCount {
  const summary: DriftSeverityCount = { high: 0, medium: 0, low: 0 };
  if (!payload) {
    return summary;
  }
  payload.active_drifts.forEach((drift) => {
    const severity = String(drift.severity || "").toLowerCase();
    if (severity === "high") {
      summary.high += 1;
      return;
    }
    if (severity === "medium") {
      summary.medium += 1;
      return;
    }
    summary.low += 1;
  });
  return summary;
}

function computeStabilityIndex(input: {
  blockedGateCount: number;
  activeDrifts: number;
  queuePending: number;
  councilPending: number;
  truthBlocked: boolean;
}): number {
  const blockedPenalty = clamp(input.blockedGateCount / 4, 0, 1) * 0.34;
  const driftPenalty = clamp(input.activeDrifts / 8, 0, 1) * 0.18;
  const queuePenalty = clamp(input.queuePending / 8, 0, 1) * 0.2;
  const councilPenalty = clamp(input.councilPending / 5, 0, 1) * 0.16;
  const truthPenalty = input.truthBlocked ? 0.12 : 0;
  const raw = 1 - blockedPenalty - driftPenalty - queuePenalty - councilPenalty - truthPenalty;
  return clamp(raw, 0, 1);
}

function stabilityLabel(score: number): string {
  if (score >= 0.8) {
    return "stable";
  }
  if (score >= 0.56) {
    return "watch";
  }
  return "unstable";
}

export function StabilityObservatoryPanel({ catalog, simulation }: Props) {
  const [study, setStudy] = useState<StudySnapshotPayload | null>(null);
  const [council, setCouncil] = useState<CouncilSnapshot | null>(null);
  const [queue, setQueue] = useState<TaskQueueSnapshot | null>(null);
  const [drift, setDrift] = useState<DriftScanPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [exportStatus, setExportStatus] = useState("");
  const [sourceMode, setSourceMode] = useState<"study-v1" | "legacy" | "">("");
  const [lastFetchedAt, setLastFetchedAt] = useState<string>("");

  const refreshStudySnapshot = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const studyRes = await fetch(runtimeApiUrl("/api/study?limit=10"));
      if (studyRes.ok) {
        const payload = (await studyRes.json()) as StudySnapshotPayload;
        setStudy(payload);
        setCouncil(payload.council ?? null);
        setQueue(payload.queue ?? null);
        setDrift(payload.drift ?? null);
        setSourceMode("study-v1");
        setLastFetchedAt(payload.generated_at || new Date().toISOString());
        return;
      }

      if (studyRes.status !== 404) {
        throw new Error(`/api/study failed: ${studyRes.status}`);
      }

      const [councilRes, queueRes, driftRes] = await Promise.all([
        fetch(runtimeApiUrl("/api/council?limit=10")),
        fetch(runtimeApiUrl("/api/task/queue")),
        fetch(runtimeApiUrl("/api/drift/scan"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        }),
      ]);

      if (!councilRes.ok) {
        throw new Error(`/api/council failed: ${councilRes.status}`);
      }
      if (!queueRes.ok) {
        throw new Error(`/api/task/queue failed: ${queueRes.status}`);
      }
      if (!driftRes.ok) {
        throw new Error(`/api/drift/scan failed: ${driftRes.status}`);
      }

      const councilPayload = (await councilRes.json()) as LegacyCouncilResponse;
      const queuePayload = (await queueRes.json()) as QueueApiResponse;
      const driftPayload = (await driftRes.json()) as DriftScanPayload;

      setStudy(null);
      setCouncil(councilPayload.council ?? null);
      setQueue(queuePayload.queue ?? null);
      setDrift(driftPayload);
      setSourceMode("legacy");
      setLastFetchedAt(new Date().toISOString());
    } catch (fetchError) {
      const message = fetchError instanceof Error ? fetchError.message : "study fetch failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshStudySnapshot();
    const interval = window.setInterval(() => {
      void refreshStudySnapshot();
    }, 6500);
    return () => {
      window.clearInterval(interval);
    };
  }, [refreshStudySnapshot]);

  const exportStudySnapshot = useCallback(async () => {
    setExportStatus("exporting evidence...");
    try {
      const response = await fetch(runtimeApiUrl("/api/study/export"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label: "ui-panel",
          include_truth: true,
          refs: ["frontend:StabilityObservatoryPanel"],
        }),
      });
      if (!response.ok) {
        throw new Error(`/api/study/export failed: ${response.status}`);
      }
      const payload = (await response.json()) as StudyExportResponse;
      const eventId = String(payload.event?.id || "").trim() || "(unknown)";
      const historyCount = Number(payload.history?.count ?? 0);
      setExportStatus(`evidence exported ${eventId} (history=${historyCount})`);
      void refreshStudySnapshot();
    } catch (exportError) {
      const message = exportError instanceof Error ? exportError.message : "study export failed";
      setExportStatus(`export failed: ${message}`);
    }
  }, [refreshStudySnapshot]);

  const councilData = study?.council ?? council ?? catalog?.council ?? null;
  const queueData = study?.queue ?? queue ?? catalog?.task_queue ?? null;
  const driftData = study?.drift ?? drift;

  const queuePendingCount =
    study?.signals?.queue_pending_count ?? queueData?.pending_count ?? 0;
  const councilPendingCount =
    study?.signals?.council_pending_count ?? councilData?.pending_count ?? 0;
  const blockedGateCount =
    study?.signals?.blocked_gate_count ?? driftData?.blocked_gates.length ?? 0;
  const activeDriftCount =
    study?.signals?.active_drift_count ?? driftData?.active_drifts.length ?? 0;
  const truthBlocked = Boolean(
    study?.signals?.truth_gate_blocked ??
      simulation?.truth_state?.gate?.blocked ??
      catalog?.truth_state?.gate?.blocked,
  );
  const runtimeResource = study?.runtime?.resource ?? catalog?.presence_runtime?.resource_heartbeat;
  const resourceHotCount =
    study?.signals?.resource_hot_count ?? runtimeResource?.hot_devices?.length ?? 0;
  const resourceLogErrorRatio =
    study?.signals?.resource_log_error_ratio ?? runtimeResource?.log_watch?.error_ratio ?? 0;
  const resourceCpuUtilization = runtimeResource?.devices?.cpu?.utilization ?? 0;
  const npuDevice = runtimeResource?.devices?.npu0;
  const npuStatus = String(npuDevice?.status || "n/a");
  const npuUtilization = typeof npuDevice?.utilization === "number" ? npuDevice.utilization : 0;
  const npuQueueDepth = typeof npuDevice?.queue_depth === "number" ? npuDevice.queue_depth : 0;
  const npuTemperature = typeof npuDevice?.temperature === "number" ? npuDevice.temperature : 0;
  const npuDeviceLabel = String(npuDevice?.device || "NPU").trim() || "NPU";
  const npuTemperatureLabel = npuTemperature > 0 ? `${npuTemperature.toFixed(1)}C` : "n/a";
  const resourceAutoEmbeddings =
    runtimeResource?.auto_backend?.embeddings_order?.join(" -> ") ?? "(n/a)";
  const resourceAutoText = runtimeResource?.auto_backend?.text_order?.join(" -> ") ?? "(n/a)";

  const stability =
    typeof study?.stability?.score === "number"
      ? clamp(study.stability.score, 0, 1)
      : computeStabilityIndex({
          blockedGateCount,
          activeDrifts: activeDriftCount,
          queuePending: queuePendingCount,
          councilPending: councilPendingCount,
          truthBlocked,
        });
  const stabilityLabelText = study?.stability?.label ?? stabilityLabel(stability);
  const stabilityPct = Math.round(stability * 100);
  const driftSeverities = summarizeDriftSeverities(driftData ?? null);
  const recentDecisions: CouncilDecision[] = (councilData?.decisions ?? []).slice(0, 6);
  const pendingTasks = (queueData?.pending ?? []).slice(0, 6);

  const warnings: StudyWarning[] = useMemo(() => {
    if (study?.warnings && study.warnings.length > 0) {
      return study.warnings;
    }
    const synthetic: StudyWarning[] = [];
    (driftData?.blocked_gates ?? []).forEach((gate) => {
      synthetic.push({
        code: "gate.blocked",
        severity: "high",
        message: `${gate.target}: ${gate.reason}`,
      });
    });
    if ((driftData?.open_questions?.unresolved_count ?? 0) > 0) {
      synthetic.push({
        code: "drift.open_questions",
        severity: "medium",
        message: `${driftData?.open_questions?.unresolved_count ?? 0} open questions unresolved`,
      });
    }
    return synthetic;
  }, [driftData?.blocked_gates, driftData?.open_questions?.unresolved_count, study?.warnings]);

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-[rgba(102,217,239,0.32)] bg-[rgba(39,40,34,0.88)] p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold flex items-center gap-2">
              <Gauge size={16} />
              Stability Observatory
            </p>
            <p className="text-xs text-muted mt-1">
              Live evidence feed for council, gates, drift pressure, queue load, resource heartbeat, and runtime alignment.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                void refreshStudySnapshot();
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
                void exportStudySnapshot();
              }}
              className="border border-[rgba(102,217,239,0.6)] rounded-md bg-[rgba(26,52,61,0.45)] px-3 py-1.5 text-xs font-semibold text-[#66d9ef] hover:bg-[rgba(26,52,61,0.6)]"
            >
              Export Evidence
            </button>
          </div>
        </div>

        <div className="mt-3 rounded-lg border border-[var(--line)] bg-[rgba(31,32,29,0.9)] p-3">
          <div className="flex items-baseline justify-between gap-2">
            <p className="text-xs uppercase tracking-wide text-muted">stability index</p>
            <p className="text-sm font-mono">
              <span className={stability >= 0.72 ? "text-[#a6e22e]" : stability >= 0.56 ? "text-[#fd971f]" : "text-[#f92672]"}>
                {stabilityPct}% ({stabilityLabelText})
              </span>
            </p>
          </div>
          <div className="mt-2 h-2 rounded-full overflow-hidden bg-[rgba(22,23,20,0.95)]">
            <div
              className="h-full transition-[width] duration-400"
              style={{
                width: `${stabilityPct}%`,
                background:
                  "linear-gradient(90deg, rgba(249,38,114,0.88), rgba(253,151,31,0.85), rgba(166,226,46,0.9))",
              }}
            />
          </div>
          <p className="text-[11px] text-muted mt-2">
            source <code>{sourceMode || "unknown"}</code> | last evidence refresh <code>{relativeTime(lastFetchedAt)}</code>
          </p>
          {study?.runtime ? (
            <p className="text-[11px] text-muted mt-1">
              receipts <code>{shortPath(study.runtime.receipts_path)}</code> | within vault <code>{String(study.runtime.receipts_path_within_vault)}</code>
            </p>
          ) : null}
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">blocked gates</p>
            <p className="text-sm font-semibold text-ink">{blockedGateCount}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">active drifts</p>
            <p className="text-sm font-semibold text-ink">{activeDriftCount}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">queue pending</p>
            <p className="text-sm font-semibold text-ink">{queuePendingCount}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">council pending</p>
            <p className="text-sm font-semibold text-ink">{councilPendingCount}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">resource hot</p>
            <p className="text-sm font-semibold text-ink">{resourceHotCount}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">cpu utilization</p>
            <p className="text-sm font-semibold text-ink">{Math.round(resourceCpuUtilization)}%</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2 sm:col-span-2 lg:col-span-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">npu lane</p>
            <p className="text-sm font-semibold text-ink">
              <span className={resourceStatusClass(npuStatus)}>{npuStatus}</span>
              <span className="text-muted"> - </span>
              {Math.round(npuUtilization)}% utilization
            </p>
            <p className="text-[11px] text-muted mt-1">
              queue <code>{npuQueueDepth}</code> | temp <code>{npuTemperatureLabel}</code> | device <code>{npuDeviceLabel}</code>
            </p>
          </div>
        </div>

        <div className="mt-3 grid gap-2 md:grid-cols-2">
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">truth gate</p>
            <p className="text-sm font-semibold flex items-center gap-1.5">
              {truthBlocked ? <ShieldAlert size={14} className="text-[#f92672]" /> : <CheckCircle2 size={14} className="text-[#a6e22e]" />}
              {truthBlocked ? "blocked" : "clear"}
            </p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">drift severities</p>
            <p className="text-sm font-mono text-ink">
              high {driftSeverities.high} | medium {driftSeverities.medium} | low {driftSeverities.low}
            </p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2 md:col-span-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">resource routing</p>
            <p className="text-[11px] text-muted mt-1">
              embeddings <code>{resourceAutoEmbeddings}</code>
            </p>
            <p className="text-[11px] text-muted mt-1">
              text <code>{resourceAutoText}</code> | log error ratio <code>{resourceLogErrorRatio.toFixed(3)}</code>
            </p>
          </div>
        </div>

        {error ? (
          <p className="mt-3 text-xs text-[#f92672]">study refresh error: {error}</p>
        ) : null}
        {exportStatus ? (
          <p className="mt-1 text-xs text-muted">{exportStatus}</p>
        ) : null}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-xl border border-[var(--line)] bg-[rgba(39,40,34,0.84)] p-3">
          <p className="text-xs uppercase tracking-wide text-muted">Council Decisions</p>
          <div className="mt-2 space-y-2 max-h-[21rem] overflow-auto pr-1">
            {recentDecisions.length === 0 ? (
              <p className="text-xs text-muted">No council decisions captured yet.</p>
            ) : (
              recentDecisions.map((decision) => {
                const tally = decision.council?.tally;
                const gateReasons = decision.gate?.reasons ?? [];
                const sourcePath = String(decision.resource?.source_rel_path ?? "");
                return (
                  <article
                    key={decision.id}
                    className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.86)] px-3 py-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[11px] font-mono text-muted truncate">{decision.id}</p>
                      <p className={`text-xs font-semibold ${decisionStatusClass(decision.status)}`}>
                        {decision.status}
                      </p>
                    </div>
                    <p className="text-[11px] text-muted mt-1 truncate">
                      source: <code>{sourcePath || "(unknown)"}</code>
                    </p>
                    <p className="text-[11px] text-muted">
                      votes: yes {tally?.yes ?? 0} / no {tally?.no ?? 0} / abstain {tally?.abstain ?? 0} / req {tally?.required_yes ?? 0}
                    </p>
                    <p className="text-[11px] text-muted">
                      action: <code>{decision.action?.result ?? "pending"}</code>
                      {decision.action?.services && decision.action.services.length > 0
                        ? ` (${decision.action.services.join(",")})`
                        : ""}
                    </p>
                    {gateReasons.length > 0 ? (
                      <p className="text-[11px] text-[#fd971f] truncate">gate: {gateReasons.join(", ")}</p>
                    ) : null}
                    <p className="text-[11px] text-muted">{relativeTime(decision.created_at)}</p>
                  </article>
                );
              })
            )}
          </div>
        </div>

        <div className="rounded-xl border border-[var(--line)] bg-[rgba(39,40,34,0.84)] p-3">
          <p className="text-xs uppercase tracking-wide text-muted">Queue and Gate Evidence</p>
          <div className="mt-2 space-y-2">
            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.86)] p-2">
              <p className="text-[11px] text-muted font-semibold">Warnings</p>
              {warnings.length === 0 ? (
                <p className="text-[11px] text-[#a6e22e] mt-1 inline-flex items-center gap-1">
                  <CheckCircle2 size={12} />
                  none
                </p>
              ) : (
                <div className="mt-1 space-y-1 max-h-[8rem] overflow-auto pr-1">
                  {warnings.slice(0, 8).map((warning, index) => (
                    <p key={`${warning.code}-${index}`} className="text-[11px] text-[#fd971f]">
                      <AlertTriangle size={11} className="inline-block mr-1" />
                      {warning.code}: {warning.message}
                    </p>
                  ))}
                </div>
              )}
            </div>

            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.86)] p-2">
              <p className="text-[11px] text-muted font-semibold">Pending tasks</p>
              {pendingTasks.length === 0 ? (
                <p className="text-[11px] text-[#a6e22e] mt-1">queue clear</p>
              ) : (
                <div className="mt-1 space-y-1 max-h-[10rem] overflow-auto pr-1">
                  {pendingTasks.map((task) => (
                    <p key={task.id} className="text-[11px] text-ink truncate">
                      <code>{task.kind}</code> :: {task.id}
                    </p>
                  ))}
                </div>
              )}
              <p className="text-[11px] text-muted mt-1">
                dedupe keys <code>{queueData?.dedupe_keys ?? 0}</code> | events <code>{queueData?.event_count ?? 0}</code>
              </p>
            </div>

            <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.86)] p-2">
              <p className="text-[11px] text-muted font-semibold">Interpretation</p>
              <p className="text-[11px] text-muted mt-1">
                Stability drops when gate blocks accumulate, queue pressure rises, resource lanes run hot, and council decisions wait without closure.
              </p>
              <p className="text-[11px] text-muted mt-1 inline-flex items-center gap-1">
                <ShieldAlert size={11} />
                Use <code>/study</code> in chat for a compact evidence digest.
              </p>
              <p className="text-[11px] text-muted mt-1 inline-flex items-center gap-1">
                <RefreshCw size={11} />
                Auto-refresh every 6.5s; manual refresh keeps snapshots explicit.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
