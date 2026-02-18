import type {
  CSSProperties,
  PointerEvent as ReactPointerEvent,
  WheelEvent as ReactWheelEvent,
} from "react";
import { SimulationCanvas, type OverlayViewId } from "../Simulation/Canvas";
import type { Catalog, SimulationState } from "../../types";
import type {
  CoreLayerId,
  CoreSimulationTuning,
  CoreVisualTuning,
} from "../../app/coreSimulationConfig";

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
  galaxyLayerStyles: GalaxyLayerStyles;
  onOverlayInit: (api: unknown) => void;
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
  galaxyLayerStyles,
  onOverlayInit,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onWheel,
}: Props) {
  return (
    <div
      className="simulation-core-backdrop"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
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
          height={viewportHeight}
          defaultOverlayView={coreOverlayView}
          overlayViewLocked
          compactHud
          interactive
          backgroundMode
          particleDensity={coreSimulationTuning.particleDensity}
          particleScale={coreSimulationTuning.particleScale}
          motionSpeed={coreSimulationTuning.motionSpeed}
          mouseInfluence={coreSimulationTuning.mouseInfluence}
          layerDepth={coreSimulationTuning.layerDepth}
          backgroundWash={coreVisualTuning.backgroundWash}
          layerVisibility={coreLayerVisibility}
          className="simulation-core-canvas"
        />
      </div>
      <p className="simulation-core-hint">drag pan • wheel zoom • wasd strafe/drive • r/f rise/fall • enable orbit for galaxy sweep</p>
      <div className="simulation-core-vignette" style={{ opacity: coreVisualTuning.vignette }} />
    </div>
  );
}
