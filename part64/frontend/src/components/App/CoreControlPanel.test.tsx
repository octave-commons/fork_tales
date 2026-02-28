/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../Simulation/Canvas", () => ({
  OVERLAY_VIEW_OPTIONS: [
    { id: "omni", label: "Omni" },
    { id: "presence", label: "Presence" },
    { id: "true-graph", label: "True Graph" },
  ],
}));

import { CoreControlPanel } from "./CoreControlPanel";

type CoreControlPanelProps = Parameters<typeof CoreControlPanel>[0];

function makeProps(overrides: Partial<CoreControlPanelProps> = {}): CoreControlPanelProps {
  return {
    projectionPerspective: "hybrid",
    autopilotEnabled: true,
    autopilotStatus: "running",
    autopilotSummary: "guard active",
    interfaceOpacity: 0.9,
    coreCameraZoom: 1.05,
    coreCameraPitch: 0,
    coreCameraYaw: 0,
    coreRenderedCameraPosition: { x: 12, y: -8, z: 33 },
    coreFlightEnabled: true,
    coreFlightSpeed: 1.1,
    coreOrbitEnabled: false,
    coreOrbitSpeed: 0.58,
    coreSimulationTuning: {
      particleDensity: 0.66,
      particleScale: 1.1,
      motionSpeed: 1.02,
      mouseInfluence: 0.88,
      layerDepth: 1,
      graphNodeSmoothness: 1.2,
      graphNodeStepScale: 0.95,
    },
    coreVisualTuning: {
      brightness: 1.3,
      contrast: 1.1,
      saturation: 1.06,
      hueRotate: 0,
      backgroundWash: 0.33,
      vignette: 0.52,
    },
    coreOverlayView: "omni",
    activeChatLens: { presence: "witness_thread", status: "listening" },
    latestAutopilotEvent: { actionId: "scan-drift", result: "ok" },
    projectionOptions: [
      { id: "hybrid", name: "Hybrid", description: "Default" },
      { id: "causal-time", name: "Causal", description: "Causal ordering" },
    ],
    mouseDaimonTuning: {
      enabled: true,
      message: "witness",
      mode: "push",
      radius: 0.2,
      strength: 0.4,
    },
    onToggleAutopilot: vi.fn(),
    onToggleCoreFlight: vi.fn(),
    onToggleCoreOrbit: vi.fn(),
    onNudgeCoreFlightSpeed: vi.fn(),
    onNudgeCoreOrbitSpeed: vi.fn(),
    onApplyCoreLayerPreset: vi.fn(),
    onNudgeCoreZoom: vi.fn(),
    onResetCoreCamera: vi.fn(),
    onSelectPerspective: vi.fn(),
    onSetInterfaceOpacity: vi.fn(),
    onResetInterfaceOpacity: vi.fn(),
    onBoostCoreVisibility: vi.fn(),
    onResetCoreVisualTuning: vi.fn(),
    onSetCoreVisualDial: vi.fn(),
    onResetCoreSimulationTuning: vi.fn(),
    onSetCoreSimulationDial: vi.fn(),
    onSetCoreOrbitSpeed: vi.fn(),
    onSetMouseDaimonTuning: vi.fn(),
    onOpenRuntimeConfig: vi.fn(),
    ...overrides,
  };
}

function rangeFromLabel(labelMatcher: RegExp): HTMLInputElement {
  const label = screen
    .getAllByText(labelMatcher)
    .map((node) => node.closest("label"))
    .find((candidate): candidate is HTMLLabelElement => candidate instanceof HTMLLabelElement);
  if (!label) {
    throw new Error(`range label not found: ${String(labelMatcher)}`);
  }
  const input = label.querySelector('input[type="range"]');
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`range input not found for: ${labelMatcher}`);
  }
  return input;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("CoreControlPanel", () => {
  it("routes top-level controls to callbacks", () => {
    const props = makeProps();
    render(<CoreControlPanel {...props} />);

    expect(screen.getByText(/chat-lens:/i)).toBeTruthy();
    expect(screen.getByText(/last:/i)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Autopilot On" }));
    fireEvent.click(screen.getByRole("button", { name: "Flight Armed" }));
    fireEvent.click(screen.getByRole("button", { name: "Orbit Off" }));
    fireEvent.click(screen.getByRole("button", { name: "thrust-" }));
    fireEvent.click(screen.getByRole("button", { name: "thrust+" }));
    fireEvent.click(screen.getByRole("button", { name: "orbit-" }));
    fireEvent.click(screen.getByRole("button", { name: "orbit+" }));
    fireEvent.click(screen.getByRole("button", { name: "reset" }));
    fireEvent.click(screen.getByRole("button", { name: "runtime config" }));

    fireEvent.change(screen.getByTitle("simulation-core layer preset"), {
      target: { value: "presence" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Causal" }));

    expect(props.onToggleAutopilot).toHaveBeenCalledTimes(1);
    expect(props.onToggleCoreFlight).toHaveBeenCalledTimes(1);
    expect(props.onToggleCoreOrbit).toHaveBeenCalledTimes(1);
    expect(props.onNudgeCoreFlightSpeed).toHaveBeenCalledWith(-0.12);
    expect(props.onNudgeCoreFlightSpeed).toHaveBeenCalledWith(0.12);
    expect(props.onNudgeCoreOrbitSpeed).toHaveBeenCalledWith(-0.08);
    expect(props.onNudgeCoreOrbitSpeed).toHaveBeenCalledWith(0.08);
    expect(props.onResetCoreCamera).toHaveBeenCalledTimes(1);
    expect(props.onOpenRuntimeConfig).toHaveBeenCalledTimes(1);
    expect(props.onApplyCoreLayerPreset).toHaveBeenCalledWith("presence");
    expect(props.onSelectPerspective).toHaveBeenCalledWith("causal-time");
  });

  it("exposes visual, simulation, and mouse daimon dial callbacks", () => {
    const props = makeProps({
      autopilotEnabled: false,
      coreOrbitEnabled: true,
      mouseDaimonTuning: {
        enabled: false,
        message: "alpha",
        mode: "push",
        radius: 0.24,
        strength: 0.45,
      },
      activeChatLens: null,
      latestAutopilotEvent: null,
    });
    render(<CoreControlPanel {...props} />);

    fireEvent.click(screen.getByRole("button", { name: "show dials" }));
    fireEvent.click(screen.getByRole("button", { name: "show sim" }));

    fireEvent.change(rangeFromLabel(/transparency/i), { target: { value: "0.84" } });
    fireEvent.click(screen.getByRole("button", { name: "reset ui" }));
    fireEvent.click(screen.getByRole("button", { name: "boost visibility" }));
    fireEvent.click(screen.getByRole("button", { name: "reset look" }));
    fireEvent.change(rangeFromLabel(/brightness/i), { target: { value: "1.44" } });

    fireEvent.click(screen.getByRole("button", { name: "reset sim" }));
    fireEvent.change(rangeFromLabel(/^particles\b/i), { target: { value: "0.72" } });
    fireEvent.change(rangeFromLabel(/orbit speed/i), { target: { value: "0.66" } });

    fireEvent.click(screen.getByRole("button", { name: "disabled" }));
    fireEvent.change(screen.getByPlaceholderText("witness"), { target: { value: "beta" } });
    fireEvent.click(screen.getByRole("button", { name: "orbit" }));
    fireEvent.change(rangeFromLabel(/radius/i), { target: { value: "0.3" } });
    fireEvent.change(rangeFromLabel(/strength/i), { target: { value: "0.6" } });

    expect(props.onSetInterfaceOpacity).toHaveBeenCalledWith(0.84);
    expect(props.onResetInterfaceOpacity).toHaveBeenCalledTimes(1);
    expect(props.onBoostCoreVisibility).toHaveBeenCalledTimes(1);
    expect(props.onResetCoreVisualTuning).toHaveBeenCalledTimes(1);
    expect(props.onSetCoreVisualDial).toHaveBeenCalledWith("brightness", 1.44);
    expect(props.onResetCoreSimulationTuning).toHaveBeenCalledTimes(1);
    expect(props.onSetCoreSimulationDial).toHaveBeenCalledWith("particleDensity", 0.72);
    expect(props.onSetCoreOrbitSpeed).toHaveBeenCalledWith(0.66);
    expect(props.onSetMouseDaimonTuning).toHaveBeenCalledWith({ enabled: true });
    expect(props.onSetMouseDaimonTuning).toHaveBeenCalledWith({ message: "beta" });
    expect(props.onSetMouseDaimonTuning).toHaveBeenCalledWith({ mode: "orbit" });
    expect(props.onSetMouseDaimonTuning).toHaveBeenCalledWith({ radius: 0.3 });
    expect(props.onSetMouseDaimonTuning).toHaveBeenCalledWith({ strength: 0.6 });
  });
});
