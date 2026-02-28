/* @vitest-environment jsdom */

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SimulationCanvas } from "./Canvas";
import type { Catalog, SimulationState } from "../../types";

vi.mock("./GalaxyModelDock", () => {
  return {
    GalaxyModelDock: ({ onClose }: { onClose: () => void }) => (
      <div>
        <p>mock model dock</p>
        <button type="button" onClick={onClose}>close mock dock</button>
      </div>
    ),
  };
});

function createSimulationFixture(): SimulationState {
  const nowSeconds = Math.floor(Date.now() / 1000);
  return {
    presence_dynamics: {
      field_particles: [
        {
          id: "particle-1",
          owner_presence_id: "witness_thread",
          presence_id: "witness_thread",
          x: 0.4,
          y: 0.5,
          particle_mode: "chaos-butterfly",
          route_node_id: "route:1",
          graph_node_id: "node:1",
          top_job: "emit_resource_packet",
          resource_daimoi: true,
        },
      ],
      compute_jobs: [
        {
          id: "job-gpu",
          ts: nowSeconds - 3,
          kind: "llm",
          op: "gpu-op",
          backend: "cuda",
          status: "ok",
          model: "qwen3",
        },
        {
          id: "job-cpu",
          ts: nowSeconds - 8,
          kind: "embedding",
          op: "cpu-op",
          backend: "cpu",
          status: "error",
          error: "cpu timeout",
        },
        {
          id: "job-npu",
          ts: nowSeconds - 5,
          kind: "llm",
          op: "npu-op",
          backend: "npu0",
          status: "cached",
        },
      ],
      compute_jobs_180s: 5,
      resource_daimoi: {
        delivered_packets: 4,
        total_transfer: 3.2,
      },
      resource_consumption: {
        action_packets: 2,
        blocked_packets: 1,
        consumed_total: 2.4,
        starved_presences: ["gates_of_truth"],
      },
      user_query_transient_edges: [
        {
          id: "q-t1",
          source: "user",
          target: "nexus",
          query: "status",
          hits: 4,
          life: 0.7,
          strength: 0.4,
        },
      ],
      user_query_promoted_edges: [
        {
          id: "q-p1",
          source: "user",
          target: "witness_thread",
          query: "summarize",
          hits: 6,
          life: 1,
          strength: 0.8,
        },
      ],
    },
    truth_graph: {
      node_count: 10,
      edge_count: 12,
    },
    view_graph: {
      node_count: 7,
      edge_count: 8,
      projection: {
        active: true,
        mode: "bundle",
        reason: "cpu_guard",
        bundle_ledger_count: 2,
        compaction_drive: 0.52,
        cpu_pressure: 0.37,
        view_edge_pressure: 0.41,
        cpu_utilization: 45,
        edge_threshold_base: 100,
        edge_threshold_effective: 80,
        edge_cap_base: 200,
        edge_cap_effective: 160,
        cpu_sentinel_id: "sentinel.cpu",
      },
    },
    file_graph: {
      file_nodes: [
        { id: "file:README.md" },
        { id: "bundle:1", projection_group_id: "grp-1" },
      ],
      edges: [{ id: "edge-1", source: "file:README.md", target: "bundle:1" }],
      projection: {
        active: true,
        mode: "bundle",
        reason: "cpu_guard",
        groups: [{ group_id: "grp-1" }],
        policy: {
          compaction_drive: 0.52,
          cpu_pressure: 0.37,
          view_edge_pressure: 0.41,
          cpu_utilization: 45,
          presence_id: "sentinel.cpu",
        },
        limits: {
          edge_threshold_base: 100,
          edge_threshold: 80,
          edge_cap_base: 200,
          edge_cap: 160,
        },
      },
    },
  } as unknown as SimulationState;
}

function createCatalogFixture(): Catalog {
  return {
    presence_runtime: {
      resource_heartbeat: {
        devices: {
          gpu0: { utilization: 40 },
          gpu1: { utilization: 50 },
        },
      },
    },
  } as unknown as Catalog;
}

function createWorldscreenSimulationFixture(node: Record<string, unknown>): SimulationState {
  const base = createSimulationFixture() as unknown as Record<string, unknown>;
  (base as any).file_graph = {
    file_nodes: [node],
    nodes: [node],
    edges: [],
  };
  (base as any).crawler_graph = {
    nodes: [],
    edges: [],
  };
  return base as unknown as SimulationState;
}

function mockResponse(body: unknown, status = 200): Response {
  const textBody = typeof body === "string" ? body : JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => textBody,
    arrayBuffer: async () => new TextEncoder().encode(textBody).buffer,
  } as Response;
}

type FetchCall = {
  url: string;
  method: string;
  body: Record<string, unknown> | null;
};

function setupCanvasFetchMock(seedComments: Array<Record<string, unknown>> = []) {
  const calls: FetchCall[] = [];
  const presenceAccounts: Array<Record<string, unknown>> = [
    {
      presence_id: "witness_thread",
      display_name: "Witness Thread",
      handle: "witness_thread",
      avatar: "",
      bio: "",
      tags: ["nexus-commentary"],
    },
  ];
  const commentsByImageRef = new Map<string, Array<Record<string, unknown>>>();
  seedComments.forEach((entry) => {
    const imageRef = String(entry.image_ref ?? "").trim();
    if (!imageRef) {
      return;
    }
    const rows = commentsByImageRef.get(imageRef) ?? [];
    rows.push(entry);
    commentsByImageRef.set(imageRef, rows);
  });

  const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" || input instanceof URL ? String(input) : input.url;
    const method = String(init?.method || "GET").toUpperCase();
    let parsedBody: Record<string, unknown> | null = null;
    if (init?.body && typeof init.body === "string") {
      try {
        parsedBody = JSON.parse(init.body) as Record<string, unknown>;
      } catch {
        parsedBody = null;
      }
    }

    calls.push({
      url,
      method,
      body: parsedBody,
    });

    if (url.includes("/api/witness")) {
      return mockResponse({ ok: true });
    }

    if (url.includes("/api/presence/accounts?")) {
      return mockResponse({ entries: presenceAccounts });
    }

    if (url.includes("/api/presence/accounts/upsert") && method === "POST") {
      const presenceId = String(parsedBody?.presence_id ?? "").trim();
      if (presenceId && !presenceAccounts.some((row) => String(row.presence_id) === presenceId)) {
        presenceAccounts.push({
          presence_id: presenceId,
          display_name: String(parsedBody?.display_name ?? presenceId).trim() || presenceId,
          handle: String(parsedBody?.handle ?? presenceId).trim() || presenceId,
          avatar: "",
          bio: "",
          tags: ["nexus-commentary"],
        });
      }
      return mockResponse({ ok: true });
    }

    if (url.includes("/api/image/comments?") && method === "GET") {
      const parsed = new URL(url, "http://localhost");
      const imageRef = String(parsed.searchParams.get("image_ref") ?? "").trim();
      return mockResponse({ entries: commentsByImageRef.get(imageRef) ?? [] });
    }

    if (url.includes("/api/image/comments") && method === "POST") {
      const imageRef = String(parsedBody?.image_ref ?? "").trim();
      const nextId = `comment-${(commentsByImageRef.get(imageRef)?.length ?? 0) + 1}`;
      const nextComment = {
        id: nextId,
        image_ref: imageRef,
        presence_id: String(parsedBody?.presence_id ?? "witness_thread"),
        comment: String(parsedBody?.comment ?? ""),
        metadata: (parsedBody?.metadata as Record<string, unknown> | undefined) ?? {},
        created_at: "2026-02-28T00:00:00Z",
        time: "2026-02-28T00:00:00Z",
      };
      const rows = commentsByImageRef.get(imageRef) ?? [];
      rows.push(nextComment);
      commentsByImageRef.set(imageRef, rows);
      return mockResponse({ ok: true, id: nextId });
    }

    if (url.includes("/api/image/commentary") && method === "POST") {
      return mockResponse({ ok: true, commentary: "generated commentary" });
    }

    if (url.includes("/library/")) {
      return mockResponse("# simulated file preview\nhello from fixture");
    }

    if (url === "" || url === "/") {
      return mockResponse("ok");
    }

    return mockResponse({ ok: true });
  });

  return {
    fetchSpy,
    calls,
  };
}

type RafController = {
  flush: (iterations?: number) => void;
};

function installRafController(): RafController {
  let nextId = 1;
  const callbacks = new Map<number, FrameRequestCallback>();

  vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback: FrameRequestCallback) => {
    const id = nextId;
    nextId += 1;
    callbacks.set(id, callback);
    return id;
  });

  vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id: number) => {
    callbacks.delete(id);
  });

  return {
    flush: (iterations = 1) => {
      for (let pass = 0; pass < iterations; pass += 1) {
        const pending = Array.from(callbacks.entries());
        callbacks.clear();
        pending.forEach(([, callback]) => {
          callback(performance.now());
        });
      }
    },
  };
}

function createWebGlContextMock(): WebGLRenderingContext {
  return {
    VERTEX_SHADER: 0x8B31,
    FRAGMENT_SHADER: 0x8B30,
    COMPILE_STATUS: 0x8B81,
    LINK_STATUS: 0x8B82,
    ARRAY_BUFFER: 0x8892,
    DYNAMIC_DRAW: 0x88E8,
    FLOAT: 0x1406,
    BLEND: 0x0BE2,
    DEPTH_TEST: 0x0B71,
    SRC_ALPHA: 0x0302,
    ONE: 1,
    ONE_MINUS_SRC_ALPHA: 0x0303,
    COLOR_BUFFER_BIT: 0x4000,
    DEPTH_BUFFER_BIT: 0x0100,
    POINTS: 0x0000,
    LINES: 0x0001,
    createShader: vi.fn(() => ({} as WebGLShader)),
    shaderSource: vi.fn(),
    compileShader: vi.fn(),
    getShaderParameter: vi.fn(() => true),
    deleteShader: vi.fn(),
    createProgram: vi.fn(() => ({} as WebGLProgram)),
    attachShader: vi.fn(),
    linkProgram: vi.fn(),
    getProgramParameter: vi.fn(() => true),
    deleteProgram: vi.fn(),
    createBuffer: vi.fn(() => ({} as WebGLBuffer)),
    deleteBuffer: vi.fn(),
    getAttribLocation: vi.fn(() => 0),
    getUniformLocation: vi.fn(() => ({} as WebGLUniformLocation)),
    useProgram: vi.fn(),
    enable: vi.fn(),
    disable: vi.fn(),
    blendFunc: vi.fn(),
    clearDepth: vi.fn(),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    bufferSubData: vi.fn(),
    viewport: vi.fn(),
    clearColor: vi.fn(),
    clear: vi.fn(),
    disableVertexAttribArray: vi.fn(),
    enableVertexAttribArray: vi.fn(),
    vertexAttribPointer: vi.fn(),
    uniform1f: vi.fn(),
    uniform2f: vi.fn(),
    uniform3f: vi.fn(),
    uniformMatrix4fv: vi.fn(),
    drawArrays: vi.fn(),
  } as unknown as WebGLRenderingContext;
}

function installInteractiveCanvasMocks(): RafController {
  const raf = installRafController();

  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(() => {
    return {
      x: 0,
      y: 0,
      width: 960,
      height: 540,
      left: 0,
      top: 0,
      right: 960,
      bottom: 540,
      toJSON: () => ({}),
    } as DOMRect;
  });

  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation((contextId: string) => {
    if (contextId === "webgl") {
      return createWebGlContextMock() as unknown as RenderingContext;
    }
    return null as unknown as RenderingContext | null;
  });

  return raf;
}

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => {
    return null as unknown as RenderingContext | null;
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("SimulationCanvas", () => {
  it("switches overlay lanes and toggles mp3 tools", async () => {
    render(
      <SimulationCanvas
        simulation={createSimulationFixture()}
        catalog={createCatalogFixture()}
      />,
    );

    expect(screen.getByText("view lanes")).toBeTruthy();
    expect(screen.getByText("All world overlays layered together.")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Presence" }));

    await waitFor(() => {
      expect(screen.getByText("Named-form field activity and live currents.")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "mp3 spotlight off" }));
    expect(screen.getByRole("button", { name: "mp3 spotlight on" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "jump to next mp3 nexus" }));

    await waitFor(() => {
      expect(screen.getByText("focus: no mp3 nexus visible in current overlay")).toBeTruthy();
      expect(screen.getByText("mp3 nexus jump unavailable")).toBeTruthy();
    });
  });

  it("renders compute insights and supports filtering/collapse", async () => {
    render(
      <SimulationCanvas
        simulation={createSimulationFixture()}
        catalog={createCatalogFixture()}
      />,
    );

    expect(screen.getByText(/jobs 180s: 5/)).toBeTruthy();
    expect(screen.getByText(/window: 3/)).toBeTruthy();
    expect(screen.getByText(/gpu-op/)).toBeTruthy();
    expect(screen.getByText(/cpu-op/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "gpu" }));

    await waitFor(() => {
      expect(screen.getByText(/gpu-op/)).toBeTruthy();
      expect(screen.queryByText(/cpu-op/)).toBeNull();
      expect(screen.queryByText(/npu-op/)).toBeNull();
    });

    fireEvent.click(screen.getByRole("button", { name: "collapse" }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "expand" })).toBeTruthy();
    });
  });

  it("shows graph-contract and particle telemetry summaries", () => {
    render(
      <SimulationCanvas
        simulation={createSimulationFixture()}
        catalog={createCatalogFixture()}
      />,
    );

    expect(screen.getByText("graph contracts")).toBeTruthy();
    expect(screen.getByText(/truth n\/e 10\/12/)).toBeTruthy();
    expect(screen.getByText(/view n\/e 7\/8/)).toBeTruthy();
    expect(screen.getAllByText(/projection active/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/query edges transient\/promoted 1\/1/)).toBeTruthy();
    expect(screen.getByText(/top query target nexus/)).toBeTruthy();

    expect(screen.getByText(/primary classes \(one class per particle, n=1\)/)).toBeTruthy();
    expect(screen.getByText(/chaos butterflies \(1\)/)).toBeTruthy();
    expect(screen.getByText(/stream signals: transfer \(1\) · resource \(1\)/)).toBeTruthy();
    expect(screen.getByText(/economy: packets 4 · actions 2 · blocked 1/)).toBeTruthy();
  });

  it("opens worldscreen via overlay API, switches modes, and closes on escape", async () => {
    const raf = installInteractiveCanvasMocks();
    const { fetchSpy } = setupCanvasFetchMock();
    let overlayApi: any = null;

    render(
      <SimulationCanvas
        simulation={createWorldscreenSimulationFixture({
          id: "file:guide.md",
          source_rel_path: "docs/guide.md",
          kind: "text",
          x: 0.34,
          y: 0.42,
        })}
        catalog={createCatalogFixture()}
        onOverlayInit={(api) => {
          overlayApi = api;
        }}
      />,
    );

    await waitFor(() => {
      expect(overlayApi).toBeTruthy();
    });

    act(() => {
      raf.flush(3);
    });

    await waitFor(() => {
      expect(overlayApi.getAnchorRatio("file", "file:guide.md")).toBeTruthy();
    });

    const anchor = overlayApi.getAnchorRatio("file", "file:guide.md");
    act(() => {
      overlayApi.interactAt(anchor.x, anchor.y, { openWorldscreen: true });
    });

    await waitFor(() => {
      expect(screen.getByText(/hologram worldscreen/i)).toBeTruthy();
      expect(screen.getByText("Remote resource metadata from crawler encounter")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "conversation" }));
    await waitFor(() => {
      expect(screen.getByText("true graph conversation")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "stats" }));
    await waitFor(() => {
      expect(screen.getByText("nexus stats")).toBeTruthy();
    });

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => {
      expect(screen.queryByText(/hologram worldscreen/i)).toBeNull();
    });

    expect(fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/witness"))).toBe(true);
  });

  it("loads and posts image comments in conversation mode", async () => {
    const raf = installInteractiveCanvasMocks();
    const { calls } = setupCanvasFetchMock();
    let overlayApi: any = null;

    render(
      <SimulationCanvas
        simulation={createWorldscreenSimulationFixture({
          id: "file:scene.png",
          source_rel_path: "images/scene.png",
          kind: "image",
          x: 0.58,
          y: 0.39,
        })}
        catalog={createCatalogFixture()}
        onOverlayInit={(api) => {
          overlayApi = api;
        }}
      />,
    );

    await waitFor(() => {
      expect(overlayApi).toBeTruthy();
    });

    act(() => {
      raf.flush(3);
    });

    await waitFor(() => {
      expect(overlayApi.getAnchorRatio("file", "file:scene.png")).toBeTruthy();
    });

    const anchor = overlayApi.getAnchorRatio("file", "file:scene.png");
    act(() => {
      overlayApi.interactAt(anchor.x, anchor.y, { openWorldscreen: true });
    });

    await waitFor(() => {
      expect(screen.getByText(/hologram worldscreen/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "conversation" }));

    await waitFor(() => {
      expect(screen.getByText("true graph conversation")).toBeTruthy();
      expect(screen.getByText("no comments yet for this nexus.")).toBeTruthy();
    });

    fireEvent.change(screen.getByPlaceholderText("Commentary appears here; edit before posting if needed."), {
      target: { value: "new follow-up comment" },
    });
    fireEvent.click(screen.getByRole("button", { name: "post comment" }));

    await waitFor(() => {
      expect(screen.getByText("new follow-up comment")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "reply in-thread" }));
    await waitFor(() => {
      expect(screen.getByText(/replying to/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "clear" }));
    await waitFor(() => {
      expect(screen.getByText("posting a new root comment")).toBeTruthy();
    });

    const refreshCountBefore = calls.filter((call) => {
      return call.url.includes("/api/image/comments?") && call.method === "GET";
    }).length;
    fireEvent.click(screen.getByRole("button", { name: "refresh comments" }));
    await waitFor(() => {
      const refreshCountAfter = calls.filter((call) => {
        return call.url.includes("/api/image/comments?") && call.method === "GET";
      }).length;
      expect(refreshCountAfter).toBeGreaterThan(refreshCountBefore);
    });

    const postCommentCall = calls.find((call) => {
      return call.url.includes("/api/image/comments") && call.method === "POST";
    });
    expect(postCommentCall).toBeTruthy();
    expect(postCommentCall?.body).toMatchObject({
      image_ref: "file:images/scene.png",
      presence_id: "witness_thread",
      comment: "new follow-up comment",
      metadata: expect.objectContaining({
        source: "manual",
        node_id: "file:scene.png",
      }),
    });

    expect(calls.some((call) => call.url.includes("/api/presence/accounts/upsert") && call.method === "POST")).toBe(true);
  });
});
