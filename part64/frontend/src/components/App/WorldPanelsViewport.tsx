import { memo, useMemo, type ReactNode } from "react";
import type { PanInfo } from "framer-motion";
import type {
  PanelConfig,
  PanelWindowState,
  WorldAnchorTarget,
  WorldPanelNexusEntry,
  WorldPanelLayoutEntry,
} from "../../app/worldPanelLayout";

interface SortedPanel extends PanelConfig {
  priority: number;
  depth: number;
  councilScore: number;
  councilBoost: number;
  councilReason: string;
  presenceId: string;
  presenceLabel: string;
  presenceLabelJa: string;
  presenceRole: string;
  particleDisposition: "neutral" | "role-bound";
  particleCount: number;
  toolHints: string[];
}

interface Props {
  viewportWidth: number;
  viewportHeight: number;
  worldPanelLayout: WorldPanelLayoutEntry[];
  panelNexusLayout: WorldPanelNexusEntry[];
  sortedPanels: SortedPanel[];
  panelWindowStateById: Record<string, PanelWindowState>;
  tertiaryPinnedPanelId: string | null;
  pinnedPanels: Record<string, boolean>;
  selectedPanelId: string | null;
  isEditMode: boolean;
  coreFlightSpeed: number;
  onToggleEditMode: () => void;
  onHoverPanel: (id: string | null) => void;
  onSelectPanel: (id: string) => void;
  onTogglePanelPin: (panelId: string) => void;
  onActivatePanel: (panelId: string) => void;
  onMinimizePanel: (panelId: string) => void;
  onClosePanel: (panelId: string) => void;
  onAdjustPanelCouncilRank: (panelId: string, delta: number) => void;
  onPinPanelToTertiary: (panelId: string) => void;
  onFlyCameraToAnchor: (anchor: WorldAnchorTarget) => void;
  onWorldPanelDragEnd: (panelId: string, info: PanInfo) => void;
}

interface PanelRenderTarget {
  render: () => ReactNode;
}

function panelLabelFromId(panelId: string): string {
  return panelId.split(".").slice(-1)[0].replace(/_/g, " ");
}

function roleLabel(value: string): string {
  return value.replace(/[_-]+/g, " ").trim() || "neutral";
}

function panelGlyph(presenceId: string, fallbackLabel: string): string {
  const cleanPresence = presenceId.trim();
  if (cleanPresence) {
    const token = cleanPresence.replace(/^health_sentinel_/, "").replace(/^presence\./, "");
    return token.slice(0, 2).toUpperCase();
  }
  return fallbackLabel.slice(0, 2).toUpperCase();
}

const WorldPanelBody = memo(function WorldPanelBody({
  panel,
  collapse,
  coreFlightSpeed,
  anchorConfidence,
}: {
  panel: PanelRenderTarget;
  collapse: boolean;
  coreFlightSpeed: number;
  anchorConfidence: number;
}) {
  const renderedPanel = useMemo(() => panel.render(), [panel]);

  if (collapse) {
    return (
      <div className="world-panel-collapsed-body">
        <p>
          moving at velocity <code>{coreFlightSpeed.toFixed(2)}x</code>
        </p>
        <p>
          anchor confidence <code>{Math.round(anchorConfidence * 100)}%</code>
        </p>
      </div>
    );
  }

  return <div className="world-panel-body">{renderedPanel}</div>;
});

function WorldPanelsViewportInner({
  viewportWidth,
  viewportHeight,
  worldPanelLayout,
  panelNexusLayout,
  sortedPanels,
  panelWindowStateById,
  tertiaryPinnedPanelId,
  pinnedPanels,
  selectedPanelId,
  isEditMode,
  coreFlightSpeed,
  onToggleEditMode,
  onHoverPanel,
  onSelectPanel,
  onTogglePanelPin,
  onActivatePanel,
  onMinimizePanel,
  onClosePanel,
  onAdjustPanelCouncilRank,
  onPinPanelToTertiary,
  onFlyCameraToAnchor,
  onWorldPanelDragEnd,
}: Props) {
  void viewportWidth;
  void viewportHeight;
  void onWorldPanelDragEnd;

  const panelById = useMemo(
    () => new Map(sortedPanels.map((panel) => [panel.id, panel])),
    [sortedPanels],
  );

  const layoutByPanelId = useMemo(
    () => new Map(worldPanelLayout.map((entry) => [entry.id, entry])),
    [worldPanelLayout],
  );

  const anchorByPanelId = useMemo(() => {
    const map = new Map<string, WorldAnchorTarget>();
    worldPanelLayout.forEach((entry) => {
      map.set(entry.id, entry.anchor);
    });
    panelNexusLayout.forEach((entry) => {
      if (!map.has(entry.panelId)) {
        map.set(entry.panelId, entry.anchor);
      }
    });
    return map;
  }, [panelNexusLayout, worldPanelLayout]);

  const rankedOpenIds = useMemo(() => {
    return sortedPanels
      .filter((panel) => {
        const state = panelWindowStateById[panel.id] ?? { open: true, minimized: false };
        return state.open && !state.minimized;
      })
      .map((panel) => panel.id);
  }, [panelWindowStateById, sortedPanels]);

  const primaryPanelId = useMemo(() => {
    if (selectedPanelId && rankedOpenIds.includes(selectedPanelId)) {
      return selectedPanelId;
    }
    return rankedOpenIds[0] ?? null;
  }, [rankedOpenIds, selectedPanelId]);

  const tertiaryPinnedOpenId =
    tertiaryPinnedPanelId && rankedOpenIds.includes(tertiaryPinnedPanelId)
      ? tertiaryPinnedPanelId
      : null;

  const secondaryPanelId = useMemo(() => {
    return (
      rankedOpenIds.find(
        (panelId) => panelId !== primaryPanelId && panelId !== tertiaryPinnedOpenId,
      ) ?? null
    );
  }, [primaryPanelId, rankedOpenIds, tertiaryPinnedOpenId]);

  const tertiaryPanelId = useMemo(() => {
    if (tertiaryPinnedOpenId) {
      return tertiaryPinnedOpenId;
    }
    return (
      rankedOpenIds.find(
        (panelId) => panelId !== primaryPanelId && panelId !== secondaryPanelId,
      ) ?? null
    );
  }, [primaryPanelId, rankedOpenIds, secondaryPanelId, tertiaryPinnedOpenId]);

  const primaryPanel = primaryPanelId ? panelById.get(primaryPanelId) ?? null : null;
  const primaryExpanded = Boolean(primaryPanel && primaryPanel.councilBoost >= 2 && rankedOpenIds.length > 1);

  const dockItems = useMemo(
    () => sortedPanels.map((panel, index) => {
      const state = panelWindowStateById[panel.id] ?? { open: true, minimized: false };
      return {
        id: panel.id,
        label: panelLabelFromId(panel.id),
        presenceId: panel.presenceId,
        rank: index + 1,
        open: state.open,
        minimized: state.minimized,
        selected: selectedPanelId === panel.id,
      };
    }),
    [panelWindowStateById, selectedPanelId, sortedPanels],
  );

  const renderFocusPane = (
    paneKind: "primary" | "secondary" | "tertiary",
    panelId: string | null,
  ): ReactNode => {
    if (!panelId) {
      return (
        <article className={`world-focus-pane world-focus-pane-${paneKind} world-focus-pane-empty`}>
          <header className="world-focus-pane-header">
            <div>
              <p className="world-focus-kicker">{paneKind} focus pane</p>
              <p className="world-focus-title">Awaiting council focus</p>
            </div>
          </header>
          <div className="world-focus-empty-body">
            <p>No open panel assigned to this lane yet.</p>
          </div>
        </article>
      );
    }

    const panel = panelById.get(panelId);
    if (!panel) {
      return null;
    }
    const panelRank = Math.max(1, sortedPanels.findIndex((row) => row.id === panelId) + 1);
    const layoutEntry = layoutByPanelId.get(panelId);
    const anchor = layoutEntry?.anchor ?? anchorByPanelId.get(panelId) ?? null;
    const panelState = panelWindowStateById[panelId] ?? { open: true, minimized: false };
    const isPinned = Boolean(pinnedPanels[panelId]);
    const isTertiaryPinned = tertiaryPinnedPanelId === panelId;

    return (
      <article
        className={`world-focus-pane world-focus-pane-${paneKind} ${selectedPanelId === panelId ? "world-focus-pane-selected" : ""}`}
        onMouseEnter={() => onHoverPanel(panelId)}
        onMouseLeave={() => onHoverPanel(null)}
      >
        <header className="world-focus-pane-header">
          <div className="min-w-0">
            <p className="world-focus-kicker">
              {paneKind} focus pane · rank {panelRank}
            </p>
            <p className="world-focus-title">{panelLabelFromId(panel.id)}</p>
            <p className="world-focus-presence">
              {panel.presenceLabel}
              {panel.presenceLabelJa ? <span> / {panel.presenceLabelJa}</span> : null}
            </p>
          </div>
          <div className="world-focus-status-stack">
            <span className="world-focus-status-pill">score {panel.councilScore.toFixed(2)}</span>
            <span className="world-focus-status-pill">boost {panel.councilBoost >= 0 ? `+${panel.councilBoost}` : panel.councilBoost}</span>
            <span
              className={`world-focus-status-pill ${
                panel.particleDisposition === "neutral"
                  ? "world-focus-status-pill-neutral"
                  : "world-focus-status-pill-role"
              }`}
            >
              particles {panel.particleCount} ({panel.particleDisposition})
            </span>
          </div>
        </header>

        <div className="world-focus-meta-row">
          <span className="world-focus-meta-chip">presence {panel.presenceId}</span>
          <span className="world-focus-meta-chip">role {roleLabel(panel.presenceRole)}</span>
          <span className="world-focus-meta-chip">state {panelState.open ? (panelState.minimized ? "min" : "open") : "closed"}</span>
          {isPinned ? <span className="world-focus-meta-chip">pinned</span> : null}
          {isTertiaryPinned ? <span className="world-focus-meta-chip">tertiary</span> : null}
        </div>

        <p className="world-focus-reason">{panel.councilReason}</p>

        <div className="world-focus-tool-row">
          {panel.toolHints.slice(0, 6).map((hint) => (
            <span key={`${panel.id}:${hint}`} className="world-focus-tool-chip">
              {hint}
            </span>
          ))}
        </div>

        <div className="world-focus-actions" data-panel-interactive="true">
          <button
            type="button"
            className="world-focus-action"
            onClick={() => {
              onActivatePanel(panelId);
              onSelectPanel(panelId);
            }}
          >
            focus
          </button>
          <button
            type="button"
            className="world-focus-action"
            onClick={() => onAdjustPanelCouncilRank(panelId, 1)}
          >
            promote
          </button>
          <button
            type="button"
            className="world-focus-action"
            onClick={() => onAdjustPanelCouncilRank(panelId, -1)}
          >
            demote
          </button>
          <button
            type="button"
            className="world-focus-action"
            onClick={() => onPinPanelToTertiary(panelId)}
          >
            {isTertiaryPinned ? "unpin tertiary" : "pin tertiary"}
          </button>
          <button
            type="button"
            className="world-focus-action"
            onClick={() => onTogglePanelPin(panelId)}
          >
            {isPinned ? "unpin" : "pin"}
          </button>
          {anchor ? (
            <button
              type="button"
              className="world-focus-action"
              onClick={() => onFlyCameraToAnchor(anchor)}
            >
              inspect
            </button>
          ) : null}
          <button
            type="button"
            className="world-focus-action"
            onClick={() => onMinimizePanel(panelId)}
          >
            min
          </button>
          <button
            type="button"
            className="world-focus-action world-focus-action-close"
            onClick={() => onClosePanel(panelId)}
          >
            close
          </button>
        </div>

        <WorldPanelBody
          panel={layoutEntry?.panel ?? panel}
          collapse={Boolean(layoutEntry?.collapse)}
          coreFlightSpeed={coreFlightSpeed}
          anchorConfidence={anchor?.confidence ?? 0.5}
        />
      </article>
    );
  };

  return (
    <>
      <section className="world-council-root" aria-label="council ranked window manager">
        <header className="world-council-toolbar">
          <div>
            <p className="world-council-kicker">smart card pile</p>
            <p className="world-council-title">
              council-voted window order · each pane maps to one presence
            </p>
          </div>
          <button
            type="button"
            onClick={onToggleEditMode}
            className={`world-council-edit-btn ${isEditMode ? "world-council-edit-btn-on" : ""}`}
          >
            {isEditMode ? "editing rank" : "edit rank"}
          </button>
        </header>

        <div className={`world-council-grid ${primaryExpanded ? "world-council-grid-primary-expanded" : ""}`}>
          {renderFocusPane("primary", primaryPanelId)}
          {!primaryExpanded ? renderFocusPane("secondary", secondaryPanelId) : null}
          {renderFocusPane("tertiary", tertiaryPanelId)}

          <aside className="world-smart-pile" aria-label="council ranked panel pile">
            <header className="world-smart-pile-header">
              <p>smart card pile</p>
              <p>{sortedPanels.length} total windows</p>
            </header>

            <div className="world-smart-pile-list">
              {sortedPanels.map((panel, index) => {
                const panelState = panelWindowStateById[panel.id] ?? {
                  open: true,
                  minimized: false,
                };
                const rank = index + 1;
                const selected = selectedPanelId === panel.id;
                const tertiaryPinned = tertiaryPinnedPanelId === panel.id;
                const isPinned = Boolean(pinnedPanels[panel.id]);
                const boostLabel = panel.councilBoost >= 0 ? `+${panel.councilBoost}` : `${panel.councilBoost}`;

                return (
                  <article
                    key={panel.id}
                    className={`world-smart-card ${selected ? "world-smart-card-selected" : ""}`}
                    onMouseEnter={() => onHoverPanel(panel.id)}
                    onMouseLeave={() => onHoverPanel(null)}
                  >
                    <div className="world-smart-card-header">
                      <p className="world-smart-card-rank">#{rank}</p>
                      <p className="world-smart-card-title">{panelLabelFromId(panel.id)}</p>
                      <p className="world-smart-card-score">
                        {panel.councilScore.toFixed(2)} ({boostLabel})
                      </p>
                    </div>
                    <p className="world-smart-card-presence">
                      {panel.presenceLabel} · {roleLabel(panel.presenceRole)} · {panel.particleDisposition}
                    </p>
                    <p className="world-smart-card-particles">particles: {panel.particleCount}</p>
                    <p className="world-smart-card-reason">{panel.councilReason}</p>
                    <div className="world-smart-card-actions" data-panel-interactive="true">
                      <button type="button" onClick={() => onActivatePanel(panel.id)}>
                        {panelState.open ? (panelState.minimized ? "restore" : "focus") : "open"}
                      </button>
                      <button type="button" onClick={() => onAdjustPanelCouncilRank(panel.id, 1)}>+ vote</button>
                      <button type="button" onClick={() => onAdjustPanelCouncilRank(panel.id, -1)}>- vote</button>
                      <button type="button" onClick={() => onPinPanelToTertiary(panel.id)}>
                        {tertiaryPinned ? "unpin 3rd" : "pin 3rd"}
                      </button>
                      <button type="button" onClick={() => onTogglePanelPin(panel.id)}>
                        {isPinned ? "unpin" : "pin"}
                      </button>
                      <button type="button" onClick={() => onMinimizePanel(panel.id)}>min</button>
                      <button type="button" onClick={() => onClosePanel(panel.id)}>close</button>
                    </div>
                  </article>
                );
              })}
            </div>
          </aside>
        </div>
      </section>

      <nav className="world-orbital-dock" aria-label="presence app orbit dock">
        {dockItems.map((item) => (
          <button
            key={`dock-${item.id}`}
            type="button"
            className={`world-orbital-dock-item ${item.selected ? "world-orbital-dock-item-selected" : ""} ${!item.open ? "world-orbital-dock-item-closed" : ""} ${item.minimized ? "world-orbital-dock-item-min" : ""}`}
            onMouseEnter={() => onHoverPanel(item.id)}
            onMouseLeave={() => onHoverPanel(null)}
            onClick={() => onActivatePanel(item.id)}
            onDoubleClick={() => onPinPanelToTertiary(item.id)}
            title={`${item.label} (rank ${item.rank})`}
          >
            <span className="world-orbital-dock-rank">{item.rank}</span>
            <span className="world-orbital-dock-core">
              {panelGlyph(item.presenceId, item.label)}
            </span>
            <span className="world-orbital-dock-label">{item.label}</span>
          </button>
        ))}
      </nav>
    </>
  );
}

export const WorldPanelsViewport = memo(WorldPanelsViewportInner);
