import { useMemo } from "react";
import { runtimeApiUrl } from "../../runtime/endpoints";
import type { SimulationState } from "../../types";

interface Props {
  simulation: SimulationState | null;
}

interface InspirationBoard {
  id: string;
  title: string;
  subtitle: string;
  relPath: string;
  anchorPresence: string;
  notes: string;
}

const INSPIRATION_BOARDS: InspirationBoard[] = [
  {
    id: "web-search-sync",
    title: "Web Search Sync",
    subtitle: "eta/mu -> Pi -> Breath loop",
    relPath: ".ημ/ChatGPT Image Feb 15, 2026, 04_20_01 PM.png",
    anchorPresence: "receipt_river",
    notes: "Drive crawl, compliance, and feedback visibility from one loop.",
  },
  {
    id: "part64-runtime-system",
    title: "Part 64 Runtime System",
    subtitle: "Nexus state map + open questions",
    relPath: ".ημ/ChatGPT Image Feb 15, 2026, 01_50_05 PM.png",
    anchorPresence: "anchor_registry",
    notes: "Keep flow legible: receipts, queue, drift, api, and meta-daimoi.",
  },
  {
    id: "inner-jam",
    title: "Operation Mindfuck :: Inner Jam",
    subtitle: "Council rhythm + artifact workstation",
    relPath: ".ημ/ChatGPT Image Feb 15, 2026, 11_20_58 AM.png",
    anchorPresence: "mage_of_receipts",
    notes: "Let presences shape attention while proofs stay inspectable.",
  },
];

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function libraryAssetUrl(relPath: string): string {
  const encoded = relPath
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return runtimeApiUrl(`/library/${encoded}`);
}

function presenceForce(simulation: SimulationState | null, presenceId: string): number {
  const impact = (simulation?.presence_dynamics?.presence_impacts ?? []).find(
    (row) => row.id === presenceId,
  );
  if (!impact) {
    return 0;
  }

  const files = clamp01(Number(impact.affected_by?.files ?? 0));
  const clicks = clamp01(Number(impact.affected_by?.clicks ?? 0));
  const world = clamp01(Number(impact.affects?.world ?? 0));
  const ledger = clamp01(Number(impact.affects?.ledger ?? 0));
  return clamp01(files * 0.28 + clicks * 0.22 + world * 0.3 + ledger * 0.2);
}

export function InspirationAtlasPanel({ simulation }: Props) {
  const riverFlowRatio = clamp01(
    Number(simulation?.presence_dynamics?.river_flow?.rate ?? 0) / 12,
  );
  const witnessContinuity = clamp01(
    Number(simulation?.presence_dynamics?.witness_thread?.continuity_index ?? 0),
  );
  const ghostPulse = clamp01(
    Number(simulation?.presence_dynamics?.ghost?.auto_commit_pulse ?? 0),
  );
  const forkTaxPaidRatio = clamp01(
    Number(simulation?.presence_dynamics?.fork_tax?.paid_ratio ?? 1),
  );

  const boardForces = useMemo(() => {
    return INSPIRATION_BOARDS.map((board) => {
      const anchorForce = presenceForce(simulation, board.anchorPresence);
      let force = 0;

      if (board.id === "web-search-sync") {
        force = riverFlowRatio * 0.5 + witnessContinuity * 0.2 + anchorForce * 0.3;
      } else if (board.id === "part64-runtime-system") {
        force = witnessContinuity * 0.4 + anchorForce * 0.4 + (1 - forkTaxPaidRatio) * 0.2;
      } else {
        force = ghostPulse * 0.4 + anchorForce * 0.4 + riverFlowRatio * 0.2;
      }

      return {
        ...board,
        force: clamp01(force),
      };
    });
  }, [forkTaxPaidRatio, ghostPulse, riverFlowRatio, simulation, witnessContinuity]);

  return (
    <section className="card inspiration-shell relative overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full bg-[#fd971f] opacity-70" />
      <h2 className="text-3xl font-bold mb-2">Inspiration Atlas / .eta-mu</h2>
      <p className="text-muted mb-5 text-sm">
        Visual references from <code>.ημ/</code> now feed UI emphasis: stronger field and presence
        force means larger card weight.
      </p>

      <div className="inspiration-grid">
        {boardForces.map((board) => {
          const forcePct = Math.round(board.force * 100);
          return (
            <article
              key={board.id}
              className="inspiration-card"
              style={{
                minHeight: `${240 + board.force * 120}px`,
                transform: `scale(${0.96 + board.force * 0.05})`,
              }}
            >
              <img
                src={libraryAssetUrl(board.relPath)}
                alt={`${board.title} inspiration board`}
                className="inspiration-image"
                loading="lazy"
              />
              <div className="inspiration-overlay" />
              <div className="inspiration-copy">
                <p className="inspiration-k">{board.subtitle}</p>
                <h3 className="inspiration-title">{board.title}</h3>
                <p className="inspiration-note">{board.notes}</p>
                <div className="inspiration-force-track">
                  <div
                    className="inspiration-force-fill"
                    style={{ width: `${forcePct}%` }}
                  />
                </div>
                <p className="inspiration-force-label">
                  force {forcePct}% · anchor {board.anchorPresence}
                </p>
              </div>
            </article>
          );
        })}
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4 text-xs">
        <div className="inspiration-signal">
          <span>river flow</span>
          <b>{Math.round(riverFlowRatio * 100)}%</b>
        </div>
        <div className="inspiration-signal">
          <span>witness continuity</span>
          <b>{Math.round(witnessContinuity * 100)}%</b>
        </div>
        <div className="inspiration-signal">
          <span>ghost pulse</span>
          <b>{Math.round(ghostPulse * 100)}%</b>
        </div>
        <div className="inspiration-signal">
          <span>fork tax paid</span>
          <b>{Math.round(forkTaxPaidRatio * 100)}%</b>
        </div>
      </div>
    </section>
  );
}
