import { describe, expect, it } from "vitest";

import {
  CORE_CAMERA_PITCH_MAX,
  CORE_CAMERA_PITCH_MIN,
  CORE_CAMERA_YAW_MAX,
  CORE_CAMERA_YAW_MIN,
  CORE_CAMERA_ZOOM_MAX,
  CORE_CAMERA_ZOOM_MIN,
  CORE_FLIGHT_SPEED_MAX,
  CORE_FLIGHT_SPEED_MIN,
  CORE_LAYER_OPTIONS,
  CORE_ORBIT_SPEED_MAX,
  CORE_ORBIT_SPEED_MIN,
  CORE_SIM_GRAPH_NODE_SMOOTHING_MAX,
  CORE_SIM_GRAPH_NODE_SMOOTHING_MIN,
  CORE_SIM_GRAPH_NODE_STEP_SCALE_MAX,
  CORE_SIM_GRAPH_NODE_STEP_SCALE_MIN,
  CORE_SIM_LAYER_DEPTH_MAX,
  CORE_SIM_LAYER_DEPTH_MIN,
  CORE_SIM_MOTION_SPEED_MAX,
  CORE_SIM_MOTION_SPEED_MIN,
  CORE_SIM_MOUSE_INFLUENCE_MAX,
  CORE_SIM_MOUSE_INFLUENCE_MIN,
  CORE_SIM_PARTICLE_DENSITY_MAX,
  CORE_SIM_PARTICLE_DENSITY_MIN,
  CORE_SIM_PARTICLE_SCALE_MAX,
  CORE_SIM_PARTICLE_SCALE_MIN,
  CORE_VISUAL_BRIGHTNESS_MAX,
  CORE_VISUAL_BRIGHTNESS_MIN,
  CORE_VISUAL_CONTRAST_MAX,
  CORE_VISUAL_CONTRAST_MIN,
  CORE_VISUAL_HUE_MAX,
  CORE_VISUAL_HUE_MIN,
  CORE_VISUAL_SATURATION_MAX,
  CORE_VISUAL_SATURATION_MIN,
  CORE_VISUAL_VIGNETTE_MAX,
  CORE_VISUAL_VIGNETTE_MIN,
  CORE_VISUAL_WASH_MAX,
  CORE_VISUAL_WASH_MIN,
  DEFAULT_CORE_LAYER_VISIBILITY,
  DEFAULT_CORE_SIMULATION_TUNING,
  DEFAULT_CORE_VISUAL_TUNING,
  HIGH_VISIBILITY_CORE_VISUAL_TUNING,
} from "./coreSimulationConfig";

describe("coreSimulationConfig", () => {
  it("keeps layer options and default visibility aligned", () => {
    const optionIds = CORE_LAYER_OPTIONS.map((entry) => entry.id);

    expect(optionIds).toEqual([
      "presence",
      "file-impact",
      "file-graph",
      "true-graph",
      "truth-gate",
      "logic",
      "pain-field",
    ]);

    optionIds.forEach((id) => {
      expect(typeof DEFAULT_CORE_LAYER_VISIBILITY[id]).toBe("boolean");
    });
  });

  it("defines monotonic numeric bounds", () => {
    expect(CORE_CAMERA_ZOOM_MIN).toBeLessThan(CORE_CAMERA_ZOOM_MAX);
    expect(CORE_CAMERA_PITCH_MIN).toBeLessThanOrEqual(CORE_CAMERA_PITCH_MAX);
    expect(CORE_CAMERA_YAW_MIN).toBeLessThanOrEqual(CORE_CAMERA_YAW_MAX);
    expect(CORE_ORBIT_SPEED_MIN).toBeLessThan(CORE_ORBIT_SPEED_MAX);
    expect(CORE_FLIGHT_SPEED_MIN).toBeLessThan(CORE_FLIGHT_SPEED_MAX);

    expect(CORE_SIM_PARTICLE_DENSITY_MIN).toBeLessThan(CORE_SIM_PARTICLE_DENSITY_MAX);
    expect(CORE_SIM_PARTICLE_SCALE_MIN).toBeLessThan(CORE_SIM_PARTICLE_SCALE_MAX);
    expect(CORE_SIM_MOTION_SPEED_MIN).toBeLessThan(CORE_SIM_MOTION_SPEED_MAX);
    expect(CORE_SIM_MOUSE_INFLUENCE_MIN).toBeLessThan(CORE_SIM_MOUSE_INFLUENCE_MAX);
    expect(CORE_SIM_LAYER_DEPTH_MIN).toBeLessThan(CORE_SIM_LAYER_DEPTH_MAX);
    expect(CORE_SIM_GRAPH_NODE_SMOOTHING_MIN).toBeLessThan(CORE_SIM_GRAPH_NODE_SMOOTHING_MAX);
    expect(CORE_SIM_GRAPH_NODE_STEP_SCALE_MIN).toBeLessThan(CORE_SIM_GRAPH_NODE_STEP_SCALE_MAX);

    expect(CORE_VISUAL_BRIGHTNESS_MIN).toBeLessThan(CORE_VISUAL_BRIGHTNESS_MAX);
    expect(CORE_VISUAL_CONTRAST_MIN).toBeLessThan(CORE_VISUAL_CONTRAST_MAX);
    expect(CORE_VISUAL_SATURATION_MIN).toBeLessThan(CORE_VISUAL_SATURATION_MAX);
    expect(CORE_VISUAL_HUE_MIN).toBeLessThan(CORE_VISUAL_HUE_MAX);
    expect(CORE_VISUAL_WASH_MIN).toBeLessThan(CORE_VISUAL_WASH_MAX);
    expect(CORE_VISUAL_VIGNETTE_MIN).toBeLessThan(CORE_VISUAL_VIGNETTE_MAX);
  });

  it("keeps default simulation dials inside declared ranges", () => {
    expect(DEFAULT_CORE_SIMULATION_TUNING.particleDensity).toBeGreaterThanOrEqual(CORE_SIM_PARTICLE_DENSITY_MIN);
    expect(DEFAULT_CORE_SIMULATION_TUNING.particleDensity).toBeLessThanOrEqual(CORE_SIM_PARTICLE_DENSITY_MAX);
    expect(DEFAULT_CORE_SIMULATION_TUNING.particleScale).toBeGreaterThanOrEqual(CORE_SIM_PARTICLE_SCALE_MIN);
    expect(DEFAULT_CORE_SIMULATION_TUNING.particleScale).toBeLessThanOrEqual(CORE_SIM_PARTICLE_SCALE_MAX);
    expect(DEFAULT_CORE_SIMULATION_TUNING.motionSpeed).toBeGreaterThanOrEqual(CORE_SIM_MOTION_SPEED_MIN);
    expect(DEFAULT_CORE_SIMULATION_TUNING.motionSpeed).toBeLessThanOrEqual(CORE_SIM_MOTION_SPEED_MAX);
    expect(DEFAULT_CORE_SIMULATION_TUNING.mouseInfluence).toBeGreaterThanOrEqual(CORE_SIM_MOUSE_INFLUENCE_MIN);
    expect(DEFAULT_CORE_SIMULATION_TUNING.mouseInfluence).toBeLessThanOrEqual(CORE_SIM_MOUSE_INFLUENCE_MAX);
    expect(DEFAULT_CORE_SIMULATION_TUNING.layerDepth).toBeGreaterThanOrEqual(CORE_SIM_LAYER_DEPTH_MIN);
    expect(DEFAULT_CORE_SIMULATION_TUNING.layerDepth).toBeLessThanOrEqual(CORE_SIM_LAYER_DEPTH_MAX);
    expect(DEFAULT_CORE_SIMULATION_TUNING.graphNodeSmoothness).toBeGreaterThanOrEqual(CORE_SIM_GRAPH_NODE_SMOOTHING_MIN);
    expect(DEFAULT_CORE_SIMULATION_TUNING.graphNodeSmoothness).toBeLessThanOrEqual(CORE_SIM_GRAPH_NODE_SMOOTHING_MAX);
    expect(DEFAULT_CORE_SIMULATION_TUNING.graphNodeStepScale).toBeGreaterThanOrEqual(CORE_SIM_GRAPH_NODE_STEP_SCALE_MIN);
    expect(DEFAULT_CORE_SIMULATION_TUNING.graphNodeStepScale).toBeLessThanOrEqual(CORE_SIM_GRAPH_NODE_STEP_SCALE_MAX);
  });

  it("keeps visual tuning presets inside bounds", () => {
    const visualKeys = [
      "brightness",
      "contrast",
      "saturation",
      "hueRotate",
      "backgroundWash",
      "vignette",
    ] as const;

    const ranges = {
      brightness: [CORE_VISUAL_BRIGHTNESS_MIN, CORE_VISUAL_BRIGHTNESS_MAX],
      contrast: [CORE_VISUAL_CONTRAST_MIN, CORE_VISUAL_CONTRAST_MAX],
      saturation: [CORE_VISUAL_SATURATION_MIN, CORE_VISUAL_SATURATION_MAX],
      hueRotate: [CORE_VISUAL_HUE_MIN, CORE_VISUAL_HUE_MAX],
      backgroundWash: [CORE_VISUAL_WASH_MIN, CORE_VISUAL_WASH_MAX],
      vignette: [CORE_VISUAL_VIGNETTE_MIN, CORE_VISUAL_VIGNETTE_MAX],
    } as const;

    visualKeys.forEach((key) => {
      const [min, max] = ranges[key];
      expect(DEFAULT_CORE_VISUAL_TUNING[key]).toBeGreaterThanOrEqual(min);
      expect(DEFAULT_CORE_VISUAL_TUNING[key]).toBeLessThanOrEqual(max);
      expect(HIGH_VISIBILITY_CORE_VISUAL_TUNING[key]).toBeGreaterThanOrEqual(min);
      expect(HIGH_VISIBILITY_CORE_VISUAL_TUNING[key]).toBeLessThanOrEqual(max);
    });
  });
});
