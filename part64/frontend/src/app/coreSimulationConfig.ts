export interface CoreVisualTuning {
  brightness: number;
  contrast: number;
  saturation: number;
  hueRotate: number;
  backgroundWash: number;
  vignette: number;
}

export interface CoreSimulationTuning {
  particleDensity: number;
  particleScale: number;
  motionSpeed: number;
  mouseInfluence: number;
  layerDepth: number;
}

export type CoreLayerId =
  | "presence"
  | "file-impact"
  | "file-graph"
  | "crawler-graph"
  | "truth-gate"
  | "logic"
  | "pain-field";

export interface CoreLayerOption {
  id: CoreLayerId;
  label: string;
}

export const CORE_LAYER_OPTIONS: CoreLayerOption[] = [
  { id: "presence", label: "Presence currents" },
  { id: "file-impact", label: "File influence" },
  { id: "file-graph", label: "File graph" },
  { id: "crawler-graph", label: "Crawler graph" },
  { id: "truth-gate", label: "Truth gate" },
  { id: "logic", label: "Logic graph" },
  { id: "pain-field", label: "Pain field" },
];

export const DEFAULT_CORE_LAYER_VISIBILITY: Record<CoreLayerId, boolean> = {
  presence: true,
  "file-impact": true,
  "file-graph": true,
  "crawler-graph": true,
  "truth-gate": true,
  logic: true,
  "pain-field": true,
};

export const CORE_CAMERA_ZOOM_MIN = 0.6;
export const CORE_CAMERA_ZOOM_MAX = 1.8;
export const CORE_CAMERA_PITCH_MIN = 0;
export const CORE_CAMERA_PITCH_MAX = 0;
export const CORE_CAMERA_YAW_MIN = 0;
export const CORE_CAMERA_YAW_MAX = 0;
export const CORE_CAMERA_X_LIMIT = 860;
export const CORE_CAMERA_Y_LIMIT = 560;
export const CORE_CAMERA_Z_MIN = -520;
export const CORE_CAMERA_Z_MAX = 460;

export const CORE_ORBIT_SPEED_MIN = 0.18;
export const CORE_ORBIT_SPEED_MAX = 1.35;
export const CORE_ORBIT_RADIUS_X = 220;
export const CORE_ORBIT_RADIUS_Y = 74;
export const CORE_ORBIT_RADIUS_Z = 180;
export const CORE_ORBIT_PERIOD_SECONDS = 96;

export const CORE_FLIGHT_BASE_SPEED = 230;
export const CORE_FLIGHT_SPEED_MIN = 0.55;
export const CORE_FLIGHT_SPEED_MAX = 2.4;

export const CORE_SIM_PARTICLE_DENSITY_MIN = 0.18;
export const CORE_SIM_PARTICLE_DENSITY_MAX = 1;
export const CORE_SIM_PARTICLE_SCALE_MIN = 0.55;
export const CORE_SIM_PARTICLE_SCALE_MAX = 1.7;
export const CORE_SIM_MOTION_SPEED_MIN = 0.35;
export const CORE_SIM_MOTION_SPEED_MAX = 1.85;
export const CORE_SIM_MOUSE_INFLUENCE_MIN = 0;
export const CORE_SIM_MOUSE_INFLUENCE_MAX = 1.8;
export const CORE_SIM_LAYER_DEPTH_MIN = 0.4;
export const CORE_SIM_LAYER_DEPTH_MAX = 1.9;

export const CORE_VISUAL_BRIGHTNESS_MIN = 0.55;
export const CORE_VISUAL_BRIGHTNESS_MAX = 1.8;
export const CORE_VISUAL_CONTRAST_MIN = 0.65;
export const CORE_VISUAL_CONTRAST_MAX = 1.45;
export const CORE_VISUAL_SATURATION_MIN = 0.45;
export const CORE_VISUAL_SATURATION_MAX = 1.8;
export const CORE_VISUAL_HUE_MIN = -85;
export const CORE_VISUAL_HUE_MAX = 85;
export const CORE_VISUAL_WASH_MIN = 0.24;
export const CORE_VISUAL_WASH_MAX = 0.82;
export const CORE_VISUAL_VIGNETTE_MIN = 0.3;
export const CORE_VISUAL_VIGNETTE_MAX = 1.28;

export const DEFAULT_CORE_VISUAL_TUNING: CoreVisualTuning = {
  brightness: 1.34,
  contrast: 1.1,
  saturation: 1.08,
  hueRotate: 0,
  backgroundWash: 0.34,
  vignette: 0.52,
};

export const HIGH_VISIBILITY_CORE_VISUAL_TUNING: CoreVisualTuning = {
  brightness: 1.5,
  contrast: 1.14,
  saturation: 1.12,
  hueRotate: 0,
  backgroundWash: 0.26,
  vignette: 0.36,
};

export const DEFAULT_CORE_SIMULATION_TUNING: CoreSimulationTuning = {
  particleDensity: 0.52,
  particleScale: 0.88,
  motionSpeed: 0.82,
  mouseInfluence: 1.15,
  layerDepth: 1.25,
};
