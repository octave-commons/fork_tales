import { type ReactNode, Suspense, lazy } from "react";

import { MusePresencePanel } from "../components/Panels/MusePresencePanel";
import { ProjectionLedgerPanel } from "../components/Panels/ProjectionLedgerPanel";
import {
  OVERLAY_VIEW_OPTIONS,
  SimulationCanvas,
} from "../components/Simulation/Canvas";
import { FIXED_MUSE_PRESENCES, GLASS_VIEWPORT_PANEL_ID } from "./appShellConstants";
import { type UseAppPanelConfigsArgs } from "./appPanelConfigTypes";
import { projectionOpacity } from "./appShellUtils";
import { normalizeMusePresenceId } from "./museWorkspace";
import { type PanelConfig } from "./worldPanelLayout";

const VitalsPanel = lazy(() =>
  import("../components/Panels/Vitals").then((module) => ({ default: module.VitalsPanel })),
);
const CatalogPanel = lazy(() =>
  import("../components/Panels/Catalog").then((module) => ({ default: module.CatalogPanel })),
);
const OmniPanel = lazy(() =>
  import("../components/Panels/Omni").then((module) => ({ default: module.OmniPanel })),
);
const MythWorldPanel = lazy(() =>
  import("../components/Panels/MythWorld").then((module) => ({ default: module.MythWorldPanel })),
);
const WebGraphWeaverPanel = lazy(() =>
  import("../components/Panels/WebGraphWeaverPanel").then((module) => ({
    default: module.WebGraphWeaverPanel,
  })),
);
const InspirationAtlasPanel = lazy(() =>
  import("../components/Panels/InspirationAtlasPanel").then((module) => ({
    default: module.InspirationAtlasPanel,
  })),
);
const StabilityObservatoryPanel = lazy(() =>
  import("../components/Panels/StabilityObservatoryPanel").then((module) => ({
    default: module.StabilityObservatoryPanel,
  })),
);
const RuntimeConfigPanel = lazy(() =>
  import("../components/Panels/RuntimeConfigPanel").then((module) => ({
    default: module.RuntimeConfigPanel,
  })),
);
const DaimoiPresencePanel = lazy(() =>
  import("../components/Panels/DaimoiPresencePanel").then((module) => ({
    default: module.DaimoiPresencePanel,
  })),
);
const WorldLogPanel = lazy(() =>
  import("../components/Panels/WorldLogPanel").then((module) => ({
    default: module.WorldLogPanel,
  })),
);

type OverlayViewOption = (typeof OVERLAY_VIEW_OPTIONS)[number];

export interface BuildPanelConfigsArgs extends UseAppPanelConfigsArgs {
  dedicatedOverlayViews: OverlayViewOption[];
}

function renderDeferredPanelPlaceholder(title: string) {
  return (
    <div className="rounded-xl border border-[var(--line)] bg-[rgba(45,46,39,0.82)] px-4 py-5">
      <p className="text-sm font-semibold text-ink">{title}</p>
      <p className="text-xs text-muted mt-1">warming up panel...</p>
    </div>
  );
}

function renderDeferredPanel(ready: boolean, title: string, panel: ReactNode): ReactNode {
  return ready
    ? <Suspense fallback={renderDeferredPanelPlaceholder(title)}>{panel}</Suspense>
    : renderDeferredPanelPlaceholder(title);
}

function buildDedicatedViewsPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.dedicated_views",
    fallbackSpan: 12,
    render: () => (
      <div className="mt-0 rounded-xl border border-[var(--line)] bg-[rgba(14,22,28,0.58)] p-3 h-full">
        <p className="text-[11px] uppercase tracking-[0.12em] text-[#9ec7dd]">Dedicated World Views</p>
        <p className="text-xs text-muted mt-1">Each overlay lane rendered as its own live viewport.</p>
        <div className="mt-3 grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
          {args.dedicatedOverlayViews.map((view) => (
            <section
              key={view.id}
              className="rounded-lg border border-[rgba(126,166,192,0.32)] bg-[rgba(10,18,28,0.72)] p-2"
            >
              <div className="mb-2">
                <p className="text-sm font-semibold text-[#e5f3ff]">{view.label}</p>
                <p className="text-[11px] text-[#9fc4dd]">{view.description}</p>
              </div>
              <SimulationCanvas
                simulation={args.simulation}
                catalog={args.catalog}
                height={180}
                defaultOverlayView={view.id}
                overlayViewLocked
                compactHud
                interactive={false}
                particleDensity={args.deferredCoreSimulationTuning.particleDensity}
                particleScale={args.deferredCoreSimulationTuning.particleScale}
                motionSpeed={args.deferredCoreSimulationTuning.motionSpeed}
                mouseInfluence={args.deferredCoreSimulationTuning.mouseInfluence}
                layerDepth={args.deferredCoreSimulationTuning.layerDepth}
                graphNodeSmoothness={args.deferredCoreSimulationTuning.graphNodeSmoothness}
                graphNodeStepScale={args.deferredCoreSimulationTuning.graphNodeStepScale}
                museWorkspaceBindings={args.museWorkspaceBindings}
              />
            </section>
          ))}
        </div>
      </div>
    ),
  };
}

function buildGlassViewportPanel(): PanelConfig {
  return {
    id: GLASS_VIEWPORT_PANEL_ID,
    fallbackSpan: 12,
    anchorKind: "region",
    anchorId: "view_lens_keeper",
    worldSize: "xl",
    pinnedByDefault: true,
    render: () => (
      <div className="mt-0 rounded-xl border border-[rgba(131,188,227,0.34)] bg-[rgba(8,20,31,0.7)] p-3 h-full">
        <p className="text-[11px] uppercase tracking-[0.12em] text-[#a6d6f5]">Glass Viewport Presence</p>
        <p className="text-xs text-[#cfe6f7] mt-1">
          This lane is managed through transparent glass mode for camera guidance and gentle map panning.
        </p>
        <p className="text-[11px] text-[#9ec7dd] mt-2">
          Use the glass controls to let the view-lens keeper guide what you see in the simulation.
        </p>
      </div>
    ),
  };
}

function buildMusePresencePanels(args: BuildPanelConfigsArgs): PanelConfig[] {
  return FIXED_MUSE_PRESENCES.map((muse) => {
    const panelId = muse.id;
    const musePresenceId = muse.presenceId;
    const panelState = args.projectionStateByElement.get(panelId) ?? null;
    const panelSession =
      args.activeProjection?.chat_sessions?.find(
        (session) => normalizeMusePresenceId(String(session.presence ?? "")) === normalizeMusePresenceId(musePresenceId),
      )
      ?? null;
    const bindingKey = normalizeMusePresenceId(musePresenceId);
    const boundCount = args.museWorkspaceBindings[bindingKey]?.length ?? 0;

    return {
      id: panelId,
      fallbackSpan: 4,
      anchorKind: "node" as const,
      anchorId: musePresenceId,
      worldSize: "m" as const,
      render: () => (
        <div
          style={{
            opacity: panelState ? projectionOpacity(panelState.opacity, 0.92) : 1,
            transform: panelState
              ? `scale(${(1 + panelState.pulse * 0.01).toFixed(3)})`
              : undefined,
            transformOrigin: "center top",
            transition: "transform 200ms ease, opacity 200ms ease",
          }}
        >
          <MusePresencePanel
            museId={musePresenceId}
            onSend={args.handleMuseWorkspaceSend}
            onRecord={args.handleRecord}
            onTranscribe={args.handleTranscribe}
            onSendVoice={args.handleSendVoice}
            isRecording={args.isRecording}
            isThinking={args.isThinking}
            voiceInputMeta={args.voiceInputMeta}
            catalog={args.catalog}
            simulation={args.simulation}
            workspaceContext={args.museWorkspaceContexts[bindingKey] ?? null}
            onWorkspaceContextChange={args.handleMuseWorkspaceContextChange}
            onWorkspaceBindingsChange={args.handleMuseWorkspaceBindingsChange}
            chatLensState={panelState}
            activeChatSession={panelSession}
            activeMusePresenceId={args.activeMusePresenceId}
            onMusePresenceChange={args.setActiveMusePresenceId}
          />
          <p className="mt-2 text-[10px] text-[#8db3ca]">
            workspace binds <code>{boundCount}</code>
          </p>
        </div>
      ),
    } satisfies PanelConfig;
  });
}

function buildWebGraphWeaverPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.web_graph_weaver",
    fallbackSpan: 6,
    render: () => renderDeferredPanel(args.deferredPanelsReady, "Web Graph Weaver", <WebGraphWeaverPanel />),
  };
}

function buildInspirationAtlasPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.inspiration_atlas",
    fallbackSpan: 6,
    render: () => renderDeferredPanel(
      args.deferredPanelsReady,
      "Inspiration Atlas",
      <InspirationAtlasPanel simulation={args.simulation} />,
    ),
  };
}

function buildEntityVitalsPanel(args: BuildPanelConfigsArgs): PanelConfig {
  const vitalsProps = {
    catalog: args.catalog,
    presenceDynamics: args.simulation?.presence_dynamics ?? null,
    ...(args.simulation?.entities ? { entities: args.simulation.entities } : {}),
  };

  return {
    id: "nexus.ui.entity_vitals",
    fallbackSpan: 6,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#a6e22e] opacity-60" />
        <h2 className="text-3xl font-bold mb-2">Entity Vitals / 実体バイタル</h2>
        <p className="text-muted mb-6">Live telemetry from the canonical named forms.</p>
        <div className="max-h-[62rem] overflow-y-auto pr-1">
          {renderDeferredPanel(args.deferredPanelsReady, "Entity Vitals", <VitalsPanel {...vitalsProps} />)}
        </div>
      </>
    ),
  };
}

function buildProjectionLedgerPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.projection_ledger",
    fallbackSpan: 6,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#66d9ef] opacity-70" />
        <h2 className="text-2xl font-bold mb-2">Projection Ledger / 映台帳</h2>
        <p className="text-muted mb-4">Sub-panels expose routing and control data for every known box.</p>
        <div className="max-h-[74rem] overflow-y-auto pr-1">
          <ProjectionLedgerPanel projection={args.activeProjection} />
        </div>
      </>
    ),
  };
}

function buildAutopilotLedgerPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.autopilot_ledger",
    fallbackSpan: 6,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#fd971f] opacity-70" />
        <h2 className="text-2xl font-bold mb-2">Autopilot Ledger / 自動操縦台帳</h2>
        <p className="text-muted mb-4">Replay stream of intent, confidence, risk, permissions, and result.</p>
        <div className="space-y-2 max-h-[26rem] overflow-y-auto pr-1">
          {args.autopilotEvents.length === 0 ? (
            <p className="text-xs text-muted">No autopilot events yet.</p>
          ) : (
            args.autopilotEvents.map((event, index) => (
              <div
                key={`${event.ts}-${event.actionId}-${index}`}
                className="border border-[var(--line)] rounded-lg bg-[rgba(45,46,39,0.86)] p-2"
              >
                <p className="text-xs font-semibold text-ink">
                  <code>{event.intent}</code>{" -> "}<code>{event.actionId}</code>
                </p>
                <p className="text-[11px] text-muted font-mono">
                  confidence {event.confidence.toFixed(2)} | risk {event.risk.toFixed(2)} | result
                  <code>{event.result}</code>
                  {event.gate ? (
                    <>
                      {" "}| gate <code>{event.gate}</code>
                    </>
                  ) : null}
                </p>
                <p className="text-[11px] text-muted font-mono">
                  perms {event.perms.length > 0 ? event.perms.join(", ") : "(none)"}
                </p>
                <p className="text-[11px] text-muted">{event.summary}</p>
              </div>
            ))
          )}
        </div>
      </>
    ),
  };
}

function buildWorldLogPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.world_log",
    fallbackSpan: 6,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#a6e22e] opacity-70" />
        <h2 className="text-2xl font-bold mb-2">World Log / 世界記録</h2>
        <p className="text-muted mb-4">
          Live timeline for receipts, eta-mu ingest, pending inbox files, presence account updates, and commentary events.
        </p>
        {renderDeferredPanel(args.deferredPanelsReady, "World Log", <WorldLogPanel catalog={args.catalog} />)}
      </>
    ),
  };
}

function buildStabilityObservatoryPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.stability_observatory",
    fallbackSpan: 6,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#66d9ef] opacity-70" />
        <h2 className="text-2xl font-bold mb-2">Stability Observatory / 安定観測</h2>
        <p className="text-muted mb-4">Evidence-first view for study mode: council, gates, queue, and drift movement.</p>
        {renderDeferredPanel(
          args.deferredPanelsReady,
          "Stability Observatory",
          <StabilityObservatoryPanel catalog={args.catalog} simulation={args.simulation} />,
        )}
      </>
    ),
  };
}

function buildRuntimeConfigPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.runtime_config",
    fallbackSpan: 6,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#ae81ff] opacity-70" />
        <h2 className="text-2xl font-bold mb-2">Runtime Config / 実行設定</h2>
        <p className="text-muted mb-4">
          Inspect live numeric constants exposed by <code>/api/config</code> for simulation and runtime tuning.
        </p>
        {renderDeferredPanel(args.deferredPanelsReady, "Runtime Config", <RuntimeConfigPanel />)}
      </>
    ),
  };
}

function buildDaimoiPresencePanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.daimoi_presence",
    fallbackSpan: 6,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#89c6eb] opacity-70" />
        <h2 className="text-2xl font-bold mb-2">Daimoi Presence Deck / 代網存在甲板</h2>
        <p className="text-muted mb-4">
          Probabilistic daimoi and presence distributions with direct camera focus controls.
        </p>
        {renderDeferredPanel(
          args.deferredPanelsReady,
          "Daimoi Presence",
          <DaimoiPresencePanel
            catalog={args.catalog}
            simulation={args.simulation}
            onFocusAnchor={args.flyCameraToAnchor}
            onEmitUserInput={args.handleUserPresenceInput}
          />,
        )}
      </>
    ),
  };
}

function buildOmniArchivePanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.omni_archive",
    fallbackSpan: 8,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#ae81ff] opacity-65" />
        <h2 className="text-3xl font-bold mb-2">Omni Panel / 全感覚パネル</h2>
        <p className="text-muted mb-6">Receipt River, Mage of Receipts, and other cover entities.</p>
        {renderDeferredPanel(args.deferredPanelsReady, "Omni Archive", <OmniPanel catalog={args.catalog} />)}
        <div className="mt-8">
          <h3 className="text-2xl font-bold mb-4">Vault Artifacts / 遺物録</h3>
          {renderDeferredPanel(args.deferredPanelsReady, "Vault Artifacts", <CatalogPanel catalog={args.catalog} />)}
        </div>
      </>
    ),
  };
}

function buildMythCommonsPanel(args: BuildPanelConfigsArgs): PanelConfig {
  return {
    id: "nexus.ui.myth_commons",
    fallbackSpan: 4,
    className: "card relative overflow-hidden",
    render: () => (
      <>
        <div className="absolute top-0 left-0 w-1 h-full bg-[#fd971f] opacity-70" />
        <h2 className="text-3xl font-bold mb-2">Myth Commons / 神話共同体</h2>
        <p className="text-muted mb-6">People sing, pray to the Presences, and keep writing the myth.</p>
        {renderDeferredPanel(
          args.deferredPanelsReady,
          "Myth Commons",
          <MythWorldPanel
            simulation={args.simulation}
            interaction={args.worldInteraction}
            interactingPersonId={args.interactingPersonId}
            onInteract={args.handleWorldInteract}
          />,
        )}
      </>
    ),
  };
}

export function buildPanelConfigs(args: BuildPanelConfigsArgs): PanelConfig[] {
  return [
    buildDedicatedViewsPanel(args),
    buildGlassViewportPanel(),
    ...buildMusePresencePanels(args),
    buildWebGraphWeaverPanel(args),
    buildInspirationAtlasPanel(args),
    buildEntityVitalsPanel(args),
    buildProjectionLedgerPanel(args),
    buildAutopilotLedgerPanel(args),
    buildWorldLogPanel(args),
    buildStabilityObservatoryPanel(args),
    buildRuntimeConfigPanel(args),
    buildDaimoiPresencePanel(args),
    buildOmniArchivePanel(args),
    buildMythCommonsPanel(args),
  ];
}
