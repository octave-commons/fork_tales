import { useMemo, useState } from "react";
import type { WorldAnchorTarget } from "../../app/worldPanelLayout";
import type { BackendFieldParticle, Catalog, SimulationState } from "../../types";

interface Props {
  catalog: Catalog | null;
  simulation: SimulationState | null;
  onFocusAnchor: (anchor: WorldAnchorTarget) => void;
}

interface PresenceAggregate {
  presenceId: string;
  label: string;
  hue: number;
  count: number;
  meanMessageProbability: number;
  meanRouteProbability: number;
  meanDeflect: number;
  meanDiffuse: number;
  centroidX: number;
  centroidY: number;
  topJobs: Array<{ name: string; probability: number }>;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

function normalizeRows(simulation: SimulationState | null): BackendFieldParticle[] {
  const directRows = simulation?.presence_dynamics?.field_particles ?? simulation?.field_particles;
  return Array.isArray(directRows) ? directRows : [];
}

function formatPercent(value: number): string {
  return `${Math.round(clamp01(value) * 100)}%`;
}

function barColor(hue: number): string {
  return `linear-gradient(90deg, hsla(${hue}, 82%, 66%, 0.78), hsla(${(hue + 32) % 360}, 84%, 58%, 0.94))`;
}

export function DaimoiPresencePanel({ catalog, simulation, onFocusAnchor }: Props) {
  const [lastFocusLabel, setLastFocusLabel] = useState("");
  const fieldRows = useMemo(() => normalizeRows(simulation), [simulation]);

  const presenceMeta = useMemo(() => {
    const map = new Map<string, { label: string; hue: number }>();
    (catalog?.entity_manifest ?? []).forEach((row) => {
      const id = String(row?.id ?? "").trim();
      if (!id) {
        return;
      }
      map.set(id, {
        label: String(row?.en ?? id).trim() || id,
        hue: Number(row?.hue ?? 202),
      });
    });
    return map;
  }, [catalog?.entity_manifest]);

  const presenceRows = useMemo<PresenceAggregate[]>(() => {
    const bucket = new Map<string, {
      count: number;
      message: number;
      route: number;
      deflect: number;
      diffuse: number;
      x: number;
      y: number;
      jobTotals: Record<string, number>;
      jobSamples: number;
    }>();

    fieldRows.forEach((row) => {
      const presenceId = String(row.owner_presence_id ?? row.presence_id ?? "").trim() || "(unknown)";
      const current = bucket.get(presenceId) ?? {
        count: 0,
        message: 0,
        route: 0,
        deflect: 0,
        diffuse: 0,
        x: 0,
        y: 0,
        jobTotals: {},
        jobSamples: 0,
      };

      current.count += 1;
      current.x += clamp01(Number(row.x ?? 0.5));
      current.y += clamp01(Number(row.y ?? 0.5));
      current.message += clamp01(Number(row.message_probability ?? 0));
      current.route += clamp01(Number(row.route_probability ?? 0));
      current.deflect += clamp01(Number(row.action_probabilities?.deflect ?? 0));
      current.diffuse += clamp01(Number(row.action_probabilities?.diffuse ?? 0));

      const jobProbabilities = row.job_probabilities ?? {};
      const jobEntries = Object.entries(jobProbabilities)
        .map(([name, value]) => [String(name), clamp01(Number(value))] as const)
        .filter(([name, value]) => name.length > 0 && value > 0);
      if (jobEntries.length > 0) {
        current.jobSamples += 1;
        jobEntries.forEach(([name, value]) => {
          current.jobTotals[name] = (current.jobTotals[name] ?? 0) + value;
        });
      }

      bucket.set(presenceId, current);
    });

    return Array.from(bucket.entries())
      .map(([presenceId, row]) => {
        const meta = presenceMeta.get(presenceId);
        const divisor = Math.max(1, row.count);
        const jobDivisor = Math.max(1, row.jobSamples);
        const topJobs = Object.entries(row.jobTotals)
          .map(([name, value]) => ({
            name,
            probability: clamp01(value / jobDivisor),
          }))
          .sort((left, right) => right.probability - left.probability)
          .slice(0, 4);

        return {
          presenceId,
          label: meta?.label ?? presenceId,
          hue: Number(meta?.hue ?? 202),
          count: row.count,
          meanMessageProbability: row.message / divisor,
          meanRouteProbability: row.route / divisor,
          meanDeflect: row.deflect / divisor,
          meanDiffuse: row.diffuse / divisor,
          centroidX: row.x / divisor,
          centroidY: row.y / divisor,
          topJobs,
        };
      })
      .sort((left, right) => right.count - left.count)
      .slice(0, 10);
  }, [fieldRows, presenceMeta]);

  const globalSummary = simulation?.presence_dynamics?.daimoi_probabilistic;

  const topJobTriggers = useMemo(() => {
    const rows = Object.entries(globalSummary?.job_triggers ?? {})
      .map(([name, count]) => ({ name, count: Math.max(0, Number(count ?? 0)) }))
      .filter((row) => row.count > 0)
      .sort((left, right) => right.count - left.count)
      .slice(0, 5);
    const total = rows.reduce((sum, row) => sum + row.count, 0);
    return {
      rows: rows.map((row) => ({
        ...row,
        probability: total > 0 ? row.count / total : 0,
      })),
      total,
    };
  }, [globalSummary?.job_triggers]);

  const actionDistribution = useMemo(() => {
    const rows = [
      { key: "deflect", value: Math.max(0, Number(globalSummary?.deflects ?? 0)), hue: 188 },
      { key: "diffuse", value: Math.max(0, Number(globalSummary?.diffuses ?? 0)), hue: 26 },
      { key: "handoff", value: Math.max(0, Number(globalSummary?.handoffs ?? 0)), hue: 142 },
      { key: "delivery", value: Math.max(0, Number(globalSummary?.deliveries ?? 0)), hue: 206 },
    ];
    const total = rows.reduce((sum, row) => sum + row.value, 0);
    return {
      rows: rows.map((row) => ({
        ...row,
        probability: total > 0 ? row.value / total : 0,
      })),
      total,
    };
  }, [globalSummary?.deflects, globalSummary?.diffuses, globalSummary?.handoffs, globalSummary?.deliveries]);

  const highlightedDaimoi = useMemo(() => {
    return fieldRows
      .map((row) => ({
        id: String(row.id ?? "").trim(),
        presenceId: String(row.owner_presence_id ?? row.presence_id ?? "").trim() || "(unknown)",
        x: clamp01(Number(row.x ?? 0.5)),
        y: clamp01(Number(row.y ?? 0.5)),
        messageProbability: clamp01(Number(row.message_probability ?? 0)),
        routeProbability: clamp01(Number(row.route_probability ?? 0)),
        driftScore: clamp01(Number(row.drift_score ?? 0)),
      }))
      .filter((row) => row.id.length > 0)
      .sort((left, right) => {
        const rightScore = right.messageProbability + right.routeProbability + (right.driftScore * 0.6);
        const leftScore = left.messageProbability + left.routeProbability + (left.driftScore * 0.6);
        return rightScore - leftScore;
      })
      .slice(0, 12);
  }, [fieldRows]);

  const focusAnchor = (anchor: WorldAnchorTarget, label: string) => {
    setLastFocusLabel(label);
    onFocusAnchor(anchor);
  };

  if (!simulation) {
    return <p className="text-xs text-[#9fc4dd]">Waiting for simulation payload...</p>;
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <div className="rounded-lg border border-[rgba(118,184,222,0.36)] bg-[rgba(9,20,30,0.62)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.1em] text-[#8eb6cf]">active daimoi</p>
          <p className="text-sm font-semibold text-[#e4f4ff]">{Number(globalSummary?.active ?? fieldRows.length)}</p>
        </div>
        <div className="rounded-lg border border-[rgba(118,184,222,0.36)] bg-[rgba(9,20,30,0.62)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.1em] text-[#8eb6cf]">collisions</p>
          <p className="text-sm font-semibold text-[#e4f4ff]">{Number(globalSummary?.collisions ?? 0)}</p>
        </div>
        <div className="rounded-lg border border-[rgba(118,184,222,0.36)] bg-[rgba(9,20,30,0.62)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.1em] text-[#8eb6cf]">mean message</p>
          <p className="text-sm font-semibold text-[#e4f4ff]">{formatPercent(Number(globalSummary?.mean_message_probability ?? 0))}</p>
        </div>
        <div className="rounded-lg border border-[rgba(118,184,222,0.36)] bg-[rgba(9,20,30,0.62)] px-2 py-1.5">
          <p className="text-[10px] uppercase tracking-[0.1em] text-[#8eb6cf]">mean entropy</p>
          <p className="text-sm font-semibold text-[#e4f4ff]">{Number(globalSummary?.mean_package_entropy ?? 0).toFixed(3)}</p>
        </div>
      </div>

      {lastFocusLabel ? (
        <div className="rounded-lg border border-[rgba(139,209,244,0.4)] bg-[linear-gradient(90deg,rgba(10,25,36,0.82),rgba(8,30,39,0.74))] px-3 py-1.5 text-xs text-[#d8eeff]">
          focus locked {"->"} <span className="font-semibold text-[#f3fbff]">{lastFocusLabel}</span>
        </div>
      ) : null}

      <section className="rounded-lg border border-[rgba(108,184,228,0.32)] bg-[rgba(8,19,28,0.5)] p-3">
        <p className="text-[11px] uppercase tracking-[0.12em] text-[#9ec7dd]">Action distribution</p>
        <div className="mt-2 grid gap-2 md:grid-cols-2">
          {actionDistribution.rows.map((row) => (
            <div key={row.key}>
              <div className="flex items-center justify-between text-[11px] text-[#cfe6f7]">
                <span>{row.key}</span>
                <span className="font-mono">{formatPercent(row.probability)} ({row.value})</span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-[rgba(44,72,94,0.46)]">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${row.probability * 100}%`,
                    background: barColor(row.hue),
                  }}
                />
              </div>
            </div>
          ))}
        </div>
        <p className="mt-2 text-[10px] text-[#9ec7dd]">total actions: <code>{actionDistribution.total}</code></p>
      </section>

      <section className="rounded-lg border border-[rgba(108,184,228,0.32)] bg-[rgba(8,19,28,0.5)] p-3">
        <p className="text-[11px] uppercase tracking-[0.12em] text-[#9ec7dd]">Top job trigger probabilities</p>
        <div className="mt-2 space-y-2">
          {topJobTriggers.rows.length === 0 ? (
            <p className="text-xs text-[#9fc4dd]">No job trigger distribution available yet.</p>
          ) : topJobTriggers.rows.map((row) => (
            <div key={row.name}>
              <div className="flex items-center justify-between text-[11px] text-[#cfe6f7]">
                <span>{row.name}</span>
                <span className="font-mono">{formatPercent(row.probability)} ({row.count})</span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-[rgba(44,72,94,0.46)]">
                <div className="h-full rounded-full bg-[linear-gradient(90deg,rgba(115,208,255,0.8),rgba(80,246,201,0.86))]" style={{ width: `${row.probability * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-[rgba(108,184,228,0.32)] bg-[rgba(8,19,28,0.5)] p-3">
        <p className="text-[11px] uppercase tracking-[0.12em] text-[#9ec7dd]">Presence stats (click to zoom)</p>
        <div className="mt-2 space-y-2 max-h-[22rem] overflow-y-auto pr-1">
          {presenceRows.length === 0 ? (
            <p className="text-xs text-[#9fc4dd]">No presence particle rows yet.</p>
          ) : presenceRows.map((presence) => (
            <button
              key={presence.presenceId}
              type="button"
              onClick={() => focusAnchor(
                {
                  kind: "node",
                  id: presence.presenceId,
                  label: presence.label,
                  x: presence.centroidX,
                  y: presence.centroidY,
                  radius: 0.12,
                  hue: presence.hue,
                  confidence: 0.9,
                  presenceSignature: { [presence.presenceId]: 1 },
                },
                `presence ${presence.label}`,
              )}
              className="w-full rounded-md border border-[rgba(124,190,228,0.28)] bg-[rgba(11,26,36,0.65)] px-2.5 py-2 text-left hover:border-[rgba(143,224,255,0.55)]"
            >
              <div className="flex items-center justify-between text-xs text-[#e8f6ff]">
                <span className="font-semibold">{presence.label}</span>
                <span className="font-mono">{presence.count} particles</span>
              </div>
              <p className="mt-1 text-[11px] text-[#a6cce1]">
                message {formatPercent(presence.meanMessageProbability)} | route {formatPercent(presence.meanRouteProbability)} | deflect {formatPercent(presence.meanDeflect)} | diffuse {formatPercent(presence.meanDiffuse)}
              </p>
              {presence.topJobs.length > 0 ? (
                <div className="mt-1.5 space-y-1">
                  {presence.topJobs.map((job) => (
                    <div key={`${presence.presenceId}:${job.name}`}>
                      <div className="flex items-center justify-between text-[10px] text-[#bcdff3]">
                        <span>{job.name}</span>
                        <span className="font-mono">{formatPercent(job.probability)}</span>
                      </div>
                      <div className="mt-0.5 h-1 rounded-full bg-[rgba(44,72,94,0.46)]">
                        <div className="h-full rounded-full" style={{ width: `${job.probability * 100}%`, background: barColor(presence.hue) }} />
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </button>
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-[rgba(108,184,228,0.32)] bg-[rgba(8,19,28,0.5)] p-3">
        <p className="text-[11px] uppercase tracking-[0.12em] text-[#9ec7dd]">Daimoi highlights (click to zoom)</p>
        <div className="mt-2 grid gap-1.5 md:grid-cols-2">
          {highlightedDaimoi.length === 0 ? (
            <p className="text-xs text-[#9fc4dd]">No daimoi highlights available.</p>
          ) : highlightedDaimoi.map((row) => (
            <button
              key={row.id}
              type="button"
              onClick={() => focusAnchor(
                {
                  kind: "node",
                  id: row.id,
                  label: row.id,
                  x: row.x,
                  y: row.y,
                  radius: 0.08,
                  hue: 198,
                  confidence: 0.74,
                  presenceSignature: { [row.presenceId]: 1 },
                },
                `daimoi ${row.id}`,
              )}
              className="rounded-md border border-[rgba(124,190,228,0.26)] bg-[rgba(11,26,36,0.58)] px-2 py-1.5 text-left hover:border-[rgba(143,224,255,0.52)]"
            >
              <p className="text-[11px] font-semibold text-[#e5f4ff]">{row.id}</p>
              <p className="text-[10px] text-[#a8cee3]">
                {row.presenceId} | m {formatPercent(row.messageProbability)} | r {formatPercent(row.routeProbability)}
              </p>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
