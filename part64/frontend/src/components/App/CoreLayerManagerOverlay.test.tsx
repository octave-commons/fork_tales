/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CORE_LAYER_OPTIONS,
  DEFAULT_CORE_LAYER_VISIBILITY,
} from "../../app/coreSimulationConfig";
import { CoreLayerManagerOverlay } from "./CoreLayerManagerOverlay";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("CoreLayerManagerOverlay", () => {
  it("renders collapsed mode and toggles visibility", () => {
    const onToggleOpen = vi.fn();

    render(
      <CoreLayerManagerOverlay
        activeLayerCount={3}
        isOpen={false}
        layerVisibility={DEFAULT_CORE_LAYER_VISIBILITY}
        onToggleOpen={onToggleOpen}
        onSetAllLayers={vi.fn()}
        onSetLayerEnabled={vi.fn()}
      />,
    );

    expect(screen.getByText("layers manager")).toBeTruthy();
    expect(
      screen.getByText(
        (_content, node) => node?.textContent === `active 3/${CORE_LAYER_OPTIONS.length}`,
      ),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "show" }));
    expect(onToggleOpen).toHaveBeenCalledTimes(1);
  });

  it("routes open-state layer actions", () => {
    const onSetAllLayers = vi.fn();
    const onSetLayerEnabled = vi.fn();

    render(
      <CoreLayerManagerOverlay
        activeLayerCount={5}
        inline
        isOpen
        layerVisibility={DEFAULT_CORE_LAYER_VISIBILITY}
        onToggleOpen={vi.fn()}
        onSetAllLayers={onSetAllLayers}
        onSetLayerEnabled={onSetLayerEnabled}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "all on" }));
    fireEvent.click(screen.getByRole("button", { name: "all off" }));

    const firstLayer = CORE_LAYER_OPTIONS[0];
    const firstCheckbox = screen.getByRole("checkbox", { name: firstLayer.label });
    fireEvent.click(firstCheckbox);

    expect(onSetAllLayers).toHaveBeenCalledWith(true);
    expect(onSetAllLayers).toHaveBeenCalledWith(false);
    expect(onSetLayerEnabled).toHaveBeenCalledWith(firstLayer.id, !DEFAULT_CORE_LAYER_VISIBILITY[firstLayer.id]);
  });
});
