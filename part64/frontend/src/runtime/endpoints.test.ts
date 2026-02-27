/* @vitest-environment jsdom */

import { afterEach, describe, expect, it } from "vitest";

import {
  runtimeApiUrl,
  runtimeBaseUrl,
  runtimeWeaverBaseCandidates,
  runtimeWebSocketUrl,
} from "./endpoints";

const originalPathname = window.location.pathname;

afterEach(() => {
  delete window.__ETA_MU_RUNTIME__;
  window.history.replaceState({}, "", originalPathname);
});

describe("runtime endpoint helpers", () => {
  it("prefers runtime bridge base URL when present", () => {
    window.__ETA_MU_RUNTIME__ = { worldBaseUrl: " https://example.test/sim/demo/ " };

    expect(runtimeBaseUrl()).toBe("https://example.test/sim/demo");
    expect(runtimeApiUrl("api/catalog?limit=5#tail")).toBe("https://example.test/sim/demo/api/catalog?limit=5%23tail");
  });

  it("derives /sim/<id> prefix from browser location", () => {
    window.history.replaceState({}, "", "/sim/alpha/world");

    expect(runtimeBaseUrl()).toBe(`${window.location.origin}/sim/alpha`);
    expect(runtimeApiUrl("/api/catalog")).toBe(`${window.location.origin}/sim/alpha/api/catalog`);
  });

  it("maps runtime base to websocket URL protocol", () => {
    window.__ETA_MU_RUNTIME__ = { worldBaseUrl: "https://example.org/root" };

    expect(runtimeWebSocketUrl("/ws?stream=1#tail")).toBe("wss://example.org/root/ws?stream=1%23tail");
  });

  it("falls back to local websocket URL when runtime base is empty", () => {
    window.history.replaceState({}, "", "/");

    const wsUrl = runtimeWebSocketUrl("ws");
    expect(wsUrl).toContain("ws://");
    expect(wsUrl).toContain("/ws");
  });

  it("returns unique weaver candidate URLs", () => {
    window.__ETA_MU_RUNTIME__ = {
      worldBaseUrl: "http://127.0.0.1:8787/sim/demo",
      weaverBaseUrl: "http://127.0.0.1:8787/sim/demo/weaver/",
    };

    const candidates = runtimeWeaverBaseCandidates();
    const weaverPrefixed = "http://127.0.0.1:8787/sim/demo/weaver";
    expect(candidates).toContain(weaverPrefixed);
    expect(candidates.filter((value) => value === weaverPrefixed)).toHaveLength(1);
    expect(candidates).toContain("http://localhost:8793");
  });
});
