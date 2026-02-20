import type {
  CSSProperties,
  PointerEvent as ReactPointerEvent,
  WheelEvent as ReactWheelEvent,
} from "react";
import {
  SimulationCanvas,
  type NexusInteractionEvent,
  type OverlayViewId,
} from "../Simulation/Canvas";
import type { Catalog, SimulationState } from "../../types";
import type {
  CoreLayerId,
  CoreSimulationTuning,
  CoreVisualTuning,
} from "../../app/coreSimulationConfig";
import type { MouseDaimonTuning } from "./CoreControlPanel";

interface GalaxyLayerStyles {
  far: CSSProperties;
  mid: CSSProperties;
  near: CSSProperties;
}

interface Props {
  simulation: SimulationState | null;
  catalog: Catalog | null;
  viewportHeight: number;
  coreCameraTransform: string;
  coreSimulationFilter: string;
  coreOverlayView: OverlayViewId;
  coreSimulationTuning: CoreSimulationTuning;
  coreVisualTuning: CoreVisualTuning;
  coreLayerVisibility: Record<CoreLayerId, boolean>;
  museWorkspaceBindings: Record<string, string[]>;
  galaxyLayerStyles: GalaxyLayerStyles;
  mouseDaimonTuning: MouseDaimonTuning;
  onUserPresenceInput: (payload: {
    kind: string;
    target: string;
    message?: string;
    xRatio?: number;
    yRatio?: number;
    embedDaimoi?: boolean;
    meta?: Record<string, unknown>;
  }) => void;
  onOverlayInit: (api: unknown) => void;
  onNexusInteraction: (event: NexusInteractionEvent) => void;
  glassCenterRatio: { x: number; y: number };
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onPointerUp: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onWheel: (event: ReactWheelEvent<HTMLDivElement>) => void;
}

export function CoreBackdrop({
  simulation,
  catalog,
  viewportHeight,
  coreCameraTransform,
  coreSimulationFilter,
  coreOverlayView,
  coreSimulationTuning,
  coreVisualTuning,
  coreLayerVisibility,
  museWorkspaceBindings,
  galaxyLayerStyles,
  mouseDaimonTuning,
  onUserPresenceInput,
  onOverlayInit,
  onNexusInteraction,
  glassCenterRatio,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onWheel,
}: Props) {
  return (
    <div
      className="simulation-core-backdrop"
      onPointerDownCapture={onPointerDown}
      onPointerMoveCapture={onPointerMove}
      onPointerUpCapture={onPointerUp}
      onPointerCancelCapture={onPointerUp}
      onWheel={onWheel}
    >
      <div className="simulation-galaxy-layer simulation-galaxy-layer-far" style={galaxyLayerStyles.far} />
      <div className="simulation-galaxy-layer simulation-galaxy-layer-mid" style={galaxyLayerStyles.mid} />
      <div className="simulation-galaxy-layer simulation-galaxy-layer-near" style={galaxyLayerStyles.near} />
      <div className="simulation-core-stage" style={{ transform: coreCameraTransform, filter: coreSimulationFilter }}>
        <SimulationCanvas
          simulation={simulation}
          catalog={catalog}
          onOverlayInit={onOverlayInit}
          onNexusInteraction={onNexusInteraction}
          onUserPresenceInput={onUserPresenceInput}
          height={viewportHeight}
          defaultOverlayView={coreOverlayView}
          overlayViewLocked
          compactHud
          interactive
          backgroundMode
          glassCenterRatio={glassCenterRatio}
          particleDensity={coreSimulationTuning.particleDensity}
          particleScale={coreSimulationTuning.particleScale}
          motionSpeed={coreSimulationTuning.motionSpeed}
          mouseInfluence={coreSimulationTuning.mouseInfluence}
          layerDepth={coreSimulationTuning.layerDepth}
          backgroundWash={coreVisualTuning.backgroundWash}
          layerVisibility={coreLayerVisibility}
          museWorkspaceBindings={museWorkspaceBindings}
          mouseDaimonEnabled={mouseDaimonTuning.enabled}
          mouseDaimonMessage={mouseDaimonTuning.message}
          mouseDaimonMode={mouseDaimonTuning.mode}
          mouseDaimonRadius={mouseDaimonTuning.radius}
          mouseDaimonStrength={mouseDaimonTuning.strength}
          className="simulation-core-canvas"
        />
      </div>
      <p className="simulation-core-hint">drag pan • wheel zoom • wasd strafe/drive • r/f rise/fall • enable orbit for galaxy sweep</p>
      <div className="simulation-core-vignette" style={{ opacity: coreVisualTuning.vignette }} />
    </div>
  );
}
