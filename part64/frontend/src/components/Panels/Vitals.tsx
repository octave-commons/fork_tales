import type { EntityState, Catalog, PresenceDynamics } from "../../types";

interface Props {
  entities?: EntityState[];
  catalog?: Catalog | null;
  presenceDynamics?: PresenceDynamics | null;
}

export function VitalsPanel({ entities, catalog, presenceDynamics }: Props) {
  if (!entities || entities.length === 0) {
    return (
      <div className="text-muted text-sm p-4">
        No vitals signal / バイタル信号なし
      </div>
    );
  }

  const manifest = catalog?.entity_manifest || [];
  const manifestById = new Map(
    manifest
      .filter((item) => item && item.id)
      .map((item) => [String(item.id), item]),
  );
  const impactById = new Map(
    (presenceDynamics?.presence_impacts ?? []).map((impact) => [impact.id, impact]),
  );
  const witnessState = presenceDynamics?.witness_thread;
  const witnessContinuityPct = Math.round((witnessState?.continuity_index ?? 0) * 100);
  const growthGuard = presenceDynamics?.growth_guard;
  const growthPressurePct = Math.round(Number(growthGuard?.pressure?.blend ?? 0) * 100);
  const growthAction = growthGuard?.action;
  const witnessLinks = (witnessState?.linked_presences ?? []).map((presenceId) => {
    const item = manifestById.get(presenceId);
    if (!item) {
      return presenceId;
    }
    return `${String(item.en ?? presenceId)} / ${String(item.ja ?? "")}`;
  });

  return (
    <div className="mt-3 space-y-4">
      {witnessState && (
        <article className="border border-[rgba(102,217,239,0.34)] rounded-2xl p-4 bg-gradient-to-br from-[rgba(45,46,39,0.94)] via-[rgba(39,40,34,0.92)] to-[rgba(31,32,29,0.94)] shadow-sm">
          <div className="flex flex-col gap-3 lg:flex-row lg:justify-between lg:items-start">
            <div>
              <h3 className="text-xl font-semibold text-ink mb-1">{witnessState.en} / {witnessState.ja}</h3>
              <p className="text-sm text-muted">{witnessState.notes_en}</p>
              <p className="text-xs text-muted mt-1">{witnessState.notes_ja}</p>
            </div>
            <div className="grid grid-cols-3 gap-2 w-full lg:w-auto">
              <div className="rounded-lg border border-[var(--line)] bg-[rgba(39,40,34,0.88)] px-3 py-2">
                <p className="text-[10px] uppercase tracking-wide text-muted">continuity</p>
                <p className="font-mono font-semibold text-ink">{witnessContinuityPct}%</p>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[rgba(39,40,34,0.88)] px-3 py-2">
                <p className="text-[10px] uppercase tracking-wide text-muted">click pressure</p>
                <p className="font-mono font-semibold text-ink">{Math.round((witnessState.click_pressure ?? 0) * 100)}%</p>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[rgba(39,40,34,0.88)] px-3 py-2">
                <p className="text-[10px] uppercase tracking-wide text-muted">file pressure</p>
                <p className="font-mono font-semibold text-ink">{Math.round((witnessState.file_pressure ?? 0) * 100)}%</p>
              </div>
            </div>
          </div>

          <div className="mt-3 rounded-xl border border-[var(--line)] bg-[rgba(39,40,34,0.84)] p-3">
            <p className="text-[11px] uppercase tracking-wide text-muted">Continuity Line / 連続線</p>
            <div className="relative mt-2 h-3 rounded-full overflow-hidden bg-[rgba(31,32,29,0.95)]">
              <div
                className="absolute inset-y-0 left-0 rounded-full"
                style={{
                  width: `${witnessContinuityPct}%`,
                  background: "linear-gradient(90deg, rgba(102,217,239,0.8), rgba(166,226,46,0.72), rgba(174,129,255,0.72))",
                }}
              />
              <div
                className="absolute top-0 bottom-0 w-[2px] bg-white/90"
                style={{ left: `${witnessContinuityPct}%` }}
              />
            </div>
            <p className="text-xs text-muted mt-2">
              Linked presences: {witnessLinks.join(" -> ") || "(none)"}
            </p>
          </div>

          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <div className="rounded-xl border border-[var(--line)] bg-[rgba(39,40,34,0.84)] p-3">
              <p className="text-[11px] uppercase tracking-wide text-muted mb-2">Lineage / 来歴</p>
              <div className="space-y-2">
                {(witnessState.lineage ?? []).map((entry, index) => (
                  <div key={`${entry.kind}-${entry.ref}-${index}`} className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.82)] px-2 py-1.5">
                    <p className="text-[11px] text-[#66d9ef] font-mono">{entry.kind}{" -> "}{entry.ref}</p>
                    <p className="text-[11px] text-muted">{entry.why_ja}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-xl border border-[var(--line)] bg-[rgba(39,40,34,0.84)] p-3">
              <p className="text-[11px] uppercase tracking-wide text-muted mb-2">How To Read / 読み方</p>
              <ol className="space-y-1.5 text-xs text-ink list-decimal pl-4">
                <li>Tap the field map to bind a target into witness continuity.</li>
                <li>Check lineage rows to see what changed and where it landed.</li>
                <li>Use <code>/say witness_thread ...</code> to narrate why a trace matters.</li>
              </ol>
            </div>
          </div>
        </article>
      )}

      {growthGuard && (
        <article className="border border-[rgba(166,226,46,0.34)] rounded-2xl p-4 bg-gradient-to-br from-[rgba(41,48,32,0.94)] via-[rgba(37,43,31,0.92)] to-[rgba(31,36,29,0.94)] shadow-sm">
          <div className="flex flex-col gap-2 md:flex-row md:justify-between md:items-start">
            <div>
              <h3 className="text-lg font-semibold text-ink">Growth Guard / 増殖監視</h3>
              <p className="text-xs text-muted mt-1">
                mode={growthGuard.mode} pressure={growthPressurePct}% action={growthAction?.kind ?? "noop"}
              </p>
            </div>
            <div className="grid grid-cols-3 gap-2 w-full md:w-auto">
              <div className="rounded-lg border border-[var(--line)] bg-[rgba(39,40,34,0.88)] px-3 py-2">
                <p className="text-[10px] uppercase tracking-wide text-muted">collapsed files</p>
                <p className="font-mono font-semibold text-ink">{Number(growthAction?.collapsed_file_nodes ?? 0)}</p>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[rgba(39,40,34,0.88)] px-3 py-2">
                <p className="text-[10px] uppercase tracking-wide text-muted">collapsed edges</p>
                <p className="font-mono font-semibold text-ink">{Number(growthAction?.collapsed_edges ?? 0)}</p>
              </div>
              <div className="rounded-lg border border-[var(--line)] bg-[rgba(39,40,34,0.88)] px-3 py-2">
                <p className="text-[10px] uppercase tracking-wide text-muted">clusters</p>
                <p className="font-mono font-semibold text-ink">{Number(growthAction?.clusters ?? 0)}</p>
              </div>
            </div>
          </div>
        </article>
      )}

      <div className="grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-6">
        {entities.map((e) => {
          const meta = manifest.find((m) => m.id === e.id) || { id: e.id, en: e.id, ja: "", hue: 200 };
          const impact = impactById.get(e.id);
          const isWitnessCard = e.id === "witness_thread";
          return (
            <article
              key={e.id}
              className={`border-2 rounded-2xl p-5 bg-[rgba(45,46,39,0.9)] shadow-sm hover:shadow-md transition-all group ${
                isWitnessCard
                  ? "border-[rgba(102,217,239,0.48)] shadow-[0_0_0_1px_rgba(102,217,239,0.22)]"
                  : "border-[var(--line)]"
              }`}
            >
              <div className="flex justify-between items-center mb-4">
                <strong className="text-lg font-bold group-hover:text-[#66d9ef] transition-colors">{meta.en}</strong>
                <span className="text-sm text-muted font-medium bg-bg-0 px-2 py-1 rounded-md">{meta.ja}</span>
              </div>

              <div className="pt-3 space-y-2 border-t border-line">
                <div className="flex justify-between text-sm">
                  <span className="text-muted">Pulse / 脈拍</span>
                  <span className="font-mono font-bold text-[#66d9ef]">{e.bpm} BPM</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted">Stability / 安定性</span>
                  <span className="font-mono font-bold text-[#a6e22e]">{e.stability}%</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted">Resonance / 共鳴</span>
                  <span className="font-mono font-bold text-[#ae81ff]">{e.resonance}Hz</span>
                </div>

                {e.vitals && Object.entries(e.vitals).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-sm pt-1">
                    <span className="text-muted capitalize">{k.replace("_", " ")}</span>
                    <span className="font-mono text-ink">{v}</span>
                  </div>
                ))}

                {impact && (
                  <>
                    <div className="flex justify-between text-sm pt-1">
                      <span className="text-muted">File Influence / ファイル影響</span>
                      <span className="font-mono text-ink">{Math.round(impact.affected_by.files * 100)}%</span>
                    </div>
                    <div className="flex justify-between text-sm pt-1">
                      <span className="text-muted">Click Influence / クリック影響</span>
                      <span className="font-mono text-ink">{Math.round(impact.affected_by.clicks * 100)}%</span>
                    </div>
                    {typeof impact.affected_by.resource === "number" ? (
                      <div className="flex justify-between text-sm pt-1">
                        <span className="text-muted">Resource Influence / 資源影響</span>
                        <span className="font-mono text-ink">{Math.round(impact.affected_by.resource * 100)}%</span>
                      </div>
                    ) : null}
                    <div className="flex justify-between text-sm pt-1">
                      <span className="text-muted">World Effect / 場への波及</span>
                      <span className="font-mono text-ink">{Math.round(impact.affects.world * 100)}%</span>
                    </div>
                    <p className="text-xs text-muted pt-1">{impact.notes_ja}</p>
                  </>
                )}
              </div>

              <div className="mt-4 h-2 bg-bg-1 rounded-full overflow-hidden">
                <div
                  className="h-full transition-[width] duration-1000 ease-in-out"
                  style={{
                    width: `${e.stability}%`,
                    backgroundColor: `hsl(${meta.hue}, 70%, 60%)`,
                  }}
                />
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
