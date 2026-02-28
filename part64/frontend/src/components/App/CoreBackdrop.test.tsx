/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

let lastCanvasProps: Record<string, unknown> | null = null;

vi.mock("../Simulation/Canvas", () => ({
  SimulationCanvas: (props: Record<string, unknown>) => {
    lastCanvasProps = props;
    return <div data-testid="simulation-canvas-proxy" />;
  },
}));

import { CoreBackdrop } from "./CoreBackdrop";

type CoreBackdropProps = Parameters<typeof CoreBackdrop>[0];

function makeProps(overrides: Partial<CoreBackdropProps> = {}): CoreBackdropProps {
  return {
    simulation: null,
    catalog: null,
    viewportHeight: 420,
    coreCameraTransform: "translate3d(10px, 20px, 0)",
    coreSimulationFilter: "brightness(1.2)",
    coreOverlayView: "omni",
    coreSimulationTuning: {
      particleDensity: 0.7,
      particleScale: 1.2,
      motionSpeed: 1.1,
      mouseInfluence: 0.9,
      layerDepth: 1,
      graphNodeSmoothness: 1,
      graphNodeStepScale: 1,
    },
    coreVisualTuning: {
      brightness: 1.2,
      contrast: 1.1,
      saturation: 1,
      hueRotate: 0,
      backgroundWash: 0.3,
      vignette: 0.55,
    },
    coreLayerVisibility: {
      presence: true,
      "file-impact": false,
      "file-graph": true,
      "true-graph": true,
      "truth-gate": false,
      logic: false,
      "pain-field": false,
    },
    museWorkspaceBindings: { witness_thread: ["file:1"] },
    galaxyLayerStyles: {
      far: { opacity: 0.3 },
      mid: { opacity: 0.5 },
      near: { opacity: 0.8 },
    },
    mouseDaimonTuning: {
      enabled: true,
      message: "witness",
      mode: "orbit",
      radius: 0.5,
      strength: 0.7,
    },
    onUserPresenceInput: vi.fn(),
    onOverlayInit: vi.fn(),
    onNexusInteraction: vi.fn(),
    glassCenterRatio: { x: 0.5, y: 0.5 },
    onPointerDown: vi.fn(),
    onPointerMove: vi.fn(),
    onPointerUp: vi.fn(),
    onWheel: vi.fn(),
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  lastCanvasProps = null;
  vi.restoreAllMocks();
});

describe("CoreBackdrop", () => {
  it("passes core simulation and mouse daimon props to SimulationCanvas", () => {
    render(<CoreBackdrop {...makeProps()} />);

    expect(screen.getByTestId("simulation-canvas-proxy")).toBeTruthy();
    expect(lastCanvasProps).toBeTruthy();
    expect(lastCanvasProps?.interactive).toBe(true);
    expect(lastCanvasProps?.backgroundMode).toBe(true);
    expect(lastCanvasProps?.overlayViewLocked).toBe(true);
    expect(lastCanvasProps?.compactHud).toBe(true);
    expect(lastCanvasProps?.height).toBe(420);
    expect(lastCanvasProps?.defaultOverlayView).toBe("omni");
    expect(lastCanvasProps?.mouseDaimonEnabled).toBe(true);
    expect(lastCanvasProps?.mouseDaimonMode).toBe("orbit");
    expect(lastCanvasProps?.mouseDaimonMessage).toBe("witness");
    expect(screen.getByText(/drag pan/i)).toBeTruthy();
  });

  it("routes pointer and wheel events through backdrop handlers", () => {
    const onPointerDown = vi.fn();
    const onPointerMove = vi.fn();
    const onPointerUp = vi.fn();
    const onWheel = vi.fn();

    const { container } = render(
      <CoreBackdrop
        {...makeProps({
          onPointerDown,
          onPointerMove,
          onPointerUp,
          onWheel,
        })}
      />,
    );

    const root = container.querySelector(".simulation-core-backdrop");
    if (!(root instanceof HTMLDivElement)) {
      throw new Error("core backdrop root missing");
    }

    fireEvent.pointerDown(root, { clientX: 10, clientY: 20 });
    fireEvent.pointerMove(root, { clientX: 14, clientY: 24 });
    fireEvent.pointerUp(root, { clientX: 16, clientY: 28 });
    fireEvent.pointerCancel(root, { clientX: 16, clientY: 28 });
    fireEvent.wheel(root, { deltaY: 20 });

    expect(onPointerDown).toHaveBeenCalledTimes(1);
    expect(onPointerMove).toHaveBeenCalledTimes(1);
    expect(onPointerUp).toHaveBeenCalledTimes(2);
    expect(onWheel).toHaveBeenCalledTimes(1);
  });
});
