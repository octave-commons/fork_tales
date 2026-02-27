/* @vitest-environment jsdom */

import { afterEach, describe, expect, it } from "vitest";

import {
  buildDeviceSurroundingNodes,
  clamp,
  isCorePointerBlockedTarget,
  isTextEntryTarget,
  normalizeDeviceUtilization,
  projectionOpacity,
  resolveRuntimeMediaUrl,
  shouldRouteWheelToCore,
  stableUnitHash,
  toMuseSlug,
} from "./appShellUtils";

afterEach(() => {
  delete window.__ETA_MU_RUNTIME__;
  document.body.innerHTML = "";
});

describe("appShellUtils", () => {
  it("clamps numbers into range", () => {
    expect(clamp(3, 0, 2)).toBe(2);
    expect(clamp(-1, 0, 2)).toBe(0);
    expect(clamp(1.5, 0, 2)).toBe(1.5);
  });

  it("recognizes text-entry targets", () => {
    const input = document.createElement("input");
    const editable = document.createElement("div");
    Object.defineProperty(editable, "isContentEditable", { value: true, configurable: true });

    expect(isTextEntryTarget(input)).toBe(true);
    expect(isTextEntryTarget(editable)).toBe(true);
    expect(isTextEntryTarget(document.createElement("button"))).toBe(false);
  });

  it("blocks pointer actions for interactive controls", () => {
    const button = document.createElement("button");
    const buttonText = document.createTextNode("tap");
    button.appendChild(buttonText);
    document.body.appendChild(button);

    expect(isCorePointerBlockedTarget(buttonText)).toBe(true);
    expect(isCorePointerBlockedTarget(document.createElement("div"))).toBe(false);
  });

  it("routes wheel events only when panel body cannot scroll", () => {
    const panelBody = document.createElement("div");
    panelBody.className = "world-panel-body";
    const target = document.createElement("span");
    panelBody.appendChild(target);
    document.body.appendChild(panelBody);

    Object.defineProperty(panelBody, "scrollHeight", { value: 400, configurable: true });
    Object.defineProperty(panelBody, "clientHeight", { value: 200, configurable: true });
    Object.defineProperty(panelBody, "scrollTop", { value: 100, configurable: true, writable: true });

    expect(shouldRouteWheelToCore(target, 6)).toBe(false);
    panelBody.scrollTop = 200;
    expect(shouldRouteWheelToCore(target, 6)).toBe(true);
  });

  it("normalizes projection opacity and device utilization", () => {
    expect(projectionOpacity(undefined, 0.5)).toBe(1);
    expect(projectionOpacity(0.25, 0.5)).toBe(0.625);

    expect(normalizeDeviceUtilization("bad")).toBe(0);
    expect(normalizeDeviceUtilization(73)).toBe(0.73);
    expect(normalizeDeviceUtilization(1.5)).toBe(0.015);
  });

  it("generates stable hashes and slugs", () => {
    const a = stableUnitHash("witness");
    const b = stableUnitHash("witness");
    const c = stableUnitHash("chaos");
    expect(a).toBe(b);
    expect(a).not.toBe(c);
    expect(a).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThanOrEqual(1);

    expect(toMuseSlug("  Archive Witness  ")).toBe("archive_witness");
    expect(toMuseSlug("33 bells")).toBe("muse_33_bells");
    expect(toMuseSlug("!!!")).toBe("");
  });

  it("resolves runtime media URLs against runtime bridge base", () => {
    window.__ETA_MU_RUNTIME__ = { worldBaseUrl: "https://example.test/sim/demo/" };

    expect(resolveRuntimeMediaUrl("https://cdn.example.test/file.mp3")).toBe("https://cdn.example.test/file.mp3");
    expect(resolveRuntimeMediaUrl("/library/file.mp3")).toBe("https://example.test/sim/demo/library/file.mp3");
    expect(resolveRuntimeMediaUrl("./library/file.mp3")).toBe("https://example.test/sim/demo/library/file.mp3");
    expect(resolveRuntimeMediaUrl("   ")).toBe("");
  });

  it("builds surrounding device node rows", () => {
    const rows = buildDeviceSurroundingNodes({
      presence_dynamics: {
        resource_heartbeat: {
          devices: {
            cpu: { utilization: 72 },
            gpu0: { utilization: 0.4 },
            npu0: {},
            disk: { utilization: -4 },
            ram: { utilization: 96 },
            net: { utilization: 10 },
            extra: { utilization: 25 },
          },
        },
      },
    });

    expect(rows).toHaveLength(6);
    expect(rows[0]).toMatchObject({ id: "device:cpu", kind: "device", label: "cpu", utilization: 0.72 });
    expect(rows[2]).toMatchObject({ id: "device:npu0", utilization: 0 });
    expect(rows[3]).toMatchObject({ id: "device:disk", utilization: 0 });
  });
});
