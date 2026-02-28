/* @vitest-environment jsdom */

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WebGraphWeaverPanel } from "./WebGraphWeaverPanel";

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  static CONNECTING = 0;

  static OPEN = 1;

  static CLOSING = 2;

  static CLOSED = 3;

  readonly url: string;

  readyState = MockWebSocket.CONNECTING;

  onopen: ((event: Event) => void) | null = null;

  onmessage: ((event: MessageEvent) => void) | null = null;

  onclose: ((event: CloseEvent) => void) | null = null;

  onerror: ((event: Event) => void) | null = null;

  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
  });

  send = vi.fn();

  constructor(url: string | URL) {
    this.url = String(url);
    MockWebSocket.instances.push(this);
  }

  emitOpen(): void {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  emitMessage(payload: unknown): void {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }
}

function entitiesEnvelope() {
  return {
    ok: true,
    enabled: true,
    paused: false,
    count: 2,
    activation_threshold: 1,
    node_cooldown_ms: 3000,
    max_requests_per_host: 2,
    entities: [
      {
        id: "entity-1",
        label: "crawler-1",
        state: "moving",
        current_url: "https://opencode.ai/",
        target_url: "https://opencode.ai/",
        progress: 0.5,
        visits: 3,
      },
      {
        id: "entity-2",
        label: "crawler-2",
        state: "idle",
        current_url: null,
        target_url: null,
        visits: 1,
      },
    ],
  };
}

function statusPayload(optOutDomains: string[], state: "running" | "paused" | "stopped" = "running") {
  return {
    state,
    user_agent: "fork-tales-test-agent",
    active_domains: ["opencode.ai", "openrouter.ai"],
    domain_distribution: {
      "opencode.ai": 6,
      "openrouter.ai": 3,
    },
    depth_histogram: {
      "0": 2,
      "1": 4,
      "2": 3,
    },
    opt_out_domains: optOutDomains,
    event_count: 12,
    metrics: {
      crawl_rate_nodes_per_sec: 1.23,
      frontier_size: 5,
      active_fetchers: 2,
      compliance_percent: 97.5,
      discovered: 12,
      fetched: 9,
      skipped: 1,
      robots_blocked: 1,
      duplicate_content: 0,
      errors: 0,
      average_fetch_ms: 88,
      citation_edges: 2,
      wiki_reference_edges: 1,
      cross_reference_edges: 1,
      paper_pdf_edges: 1,
      host_concurrency_waits: 0,
      cooldown_blocked: 0,
      interactions: 4,
      activation_enqueues: 2,
      entity_moves: 7,
      entity_visits: 5,
      llm_analysis_success: 3,
      llm_analysis_fail: 0,
    },
    config: {
      max_depth: 3,
      max_nodes: 2500,
      concurrency: 2,
      max_requests_per_host: 2,
      node_cooldown_ms: 3000,
      activation_threshold: 1,
      default_delay_ms: 200,
    },
    graph_counts: {
      nodes_total: 2,
      edges_total: 1,
      url_nodes_total: 1,
    },
    entities: entitiesEnvelope(),
  };
}

function graphPayload() {
  return {
    ok: true,
    graph: {
      nodes: [
        {
          id: "domain:opencode.ai",
          kind: "domain",
          label: "opencode.ai",
          domain: "opencode.ai",
        },
        {
          id: "url:https://opencode.ai/",
          kind: "url",
          label: "https://opencode.ai/",
          domain: "opencode.ai",
          url: "https://opencode.ai/",
          source_url: null,
          depth: 0,
          activation_potential: 0.77,
          interaction_count: 2,
          analysis_provider: "llm",
          analysis_summary: "entrypoint summary",
        },
      ],
      edges: [
        {
          id: "edge-1",
          kind: "domain_membership",
          source: "domain:opencode.ai",
          target: "url:https://opencode.ai/",
        },
      ],
      counts: {
        nodes_total: 2,
        edges_total: 1,
        url_nodes_total: 1,
      },
    },
  };
}

function setupFetchMock() {
  const graphDomainRequests: string[] = [];
  const optOutDomains = ["robots.example"];

  const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = String(init?.method || "GET").toUpperCase();

    if (url.includes("/api/weaver/entities/control") && method === "POST") {
      return mockJsonResponse({
        ok: true,
        status: {
          ...statusPayload(optOutDomains),
          entities: entitiesEnvelope(),
        },
      });
    }

    if (url.includes("/api/weaver/entities/interact") && method === "POST") {
      return mockJsonResponse({
        ok: true,
        status: statusPayload(optOutDomains),
      });
    }

    if (url.includes("/api/weaver/entities")) {
      return mockJsonResponse(entitiesEnvelope());
    }

    if (url.includes("/api/weaver/control") && method === "POST") {
      return mockJsonResponse({
        ok: true,
        status: statusPayload(optOutDomains, "running"),
      });
    }

    if (url.includes("/api/weaver/opt-out") && method === "POST") {
      const body = JSON.parse(String(init?.body || "{}"));
      if (body.domain) {
        optOutDomains.push(String(body.domain));
      }
      return mockJsonResponse({ ok: true });
    }

    if (url.includes("/api/weaver/status")) {
      return mockJsonResponse(statusPayload(optOutDomains));
    }

    if (url.includes("/api/weaver/graph")) {
      const parsed = new URL(url);
      graphDomainRequests.push(parsed.searchParams.get("domain") || "");
      return mockJsonResponse(graphPayload());
    }

    if (url.includes("/api/weaver/events")) {
      return mockJsonResponse({
        ok: true,
        events: [
          {
            event: "seed_added",
            timestamp: Date.now(),
            url: "https://opencode.ai/",
          },
        ],
      });
    }

    return mockJsonResponse({ ok: true });
  });

  return {
    fetchSpy,
    graphDomainRequests,
  };
}

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => {
    return null as unknown as RenderingContext | null;
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("WebGraphWeaverPanel", () => {
  it("bootstraps status, entities, graph, and events", async () => {
    const { fetchSpy } = setupFetchMock();

    render(<WebGraphWeaverPanel />);

    const ws = MockWebSocket.instances[0];
    expect(ws).toBeTruthy();
    act(() => {
      ws.emitOpen();
    });

    await waitFor(() => {
      expect(screen.getByText("Web Graph Weaver / Web Graph Weaver")).toBeTruthy();
      expect(screen.getByText(/ws online/)).toBeTruthy();
      expect(screen.getByText(/fork-tales-test-agent/)).toBeTruthy();
      expect(screen.getByText(/state: active/)).toBeTruthy();
      expect(screen.getByText(/seed_added/)).toBeTruthy();
      expect(screen.getByText(/robots\.example/)).toBeTruthy();
    });

    expect(fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/weaver/status"))).toBe(true);
    expect(fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/weaver/graph"))).toBe(true);
    expect(fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/weaver/events"))).toBe(true);
    expect(fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/weaver/entities"))).toBe(true);
  });

  it("posts crawl and entity controls with expected payloads", async () => {
    const { fetchSpy } = setupFetchMock();

    render(<WebGraphWeaverPanel />);

    await waitFor(() => {
      expect(screen.getByText("Start Crawl")).toBeTruthy();
    });

    fireEvent.click(screen.getByText("Start Crawl"));

    await waitFor(() => {
      expect(fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/weaver/control"))).toBe(true);
      expect(screen.getByText(/crawl_state/)).toBeTruthy();
    });

    const controlCall = fetchSpy.mock.calls.find(([url, init]) => {
      return String(url).includes("/api/weaver/control") && String((init as RequestInit | undefined)?.method || "GET").toUpperCase() === "POST";
    });
    const controlBody = JSON.parse(String((controlCall?.[1] as RequestInit | undefined)?.body || "{}"));
    expect(controlBody).toMatchObject({
      action: "start",
      max_depth: 3,
      max_nodes: 2500,
      concurrency: 2,
      max_per_host: 2,
      entity_count: 2,
    });
    expect(Array.isArray(controlBody.seeds)).toBe(true);
    expect(controlBody.seeds).toContain("https://opencode.ai/");

    fireEvent.click(screen.getByText("Apply Entity Config"));

    await waitFor(() => {
      expect(fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/weaver/entities/control"))).toBe(true);
      expect(screen.getByText(/entity_control/)).toBeTruthy();
    });

    const entityControlCall = fetchSpy.mock.calls.find(([url, init]) => {
      return String(url).includes("/api/weaver/entities/control") && String((init as RequestInit | undefined)?.method || "GET").toUpperCase() === "POST";
    });
    const entityControlBody = JSON.parse(String((entityControlCall?.[1] as RequestInit | undefined)?.body || "{}"));
    expect(entityControlBody).toMatchObject({
      action: "configure",
      count: 2,
      max_per_host: 2,
    });
  });

  it("applies opt-out and responds to websocket graph delta updates", async () => {
    const { fetchSpy, graphDomainRequests } = setupFetchMock();

    render(<WebGraphWeaverPanel />);

    const ws = MockWebSocket.instances[0];
    expect(ws).toBeTruthy();
    act(() => {
      ws.emitOpen();
    });

    await waitFor(() => {
      expect(screen.getByText("Add Opt-Out")).toBeTruthy();
    });

    fireEvent.change(screen.getByPlaceholderText("example.com"), {
      target: { value: "forbidden.example" },
    });
    fireEvent.click(screen.getByText("Add Opt-Out"));

    await waitFor(() => {
      expect(screen.getByText(/forbidden\.example/)).toBeTruthy();
    });

    const optOutCall = fetchSpy.mock.calls.find(([url, init]) => {
      return String(url).includes("/api/weaver/opt-out") && String((init as RequestInit | undefined)?.method || "GET").toUpperCase() === "POST";
    });
    const optOutBody = JSON.parse(String((optOutCall?.[1] as RequestInit | undefined)?.body || "{}"));
    expect(optOutBody).toMatchObject({
      domain: "forbidden.example",
    });

    const domainFilter = screen.getByDisplayValue("(all domains)");
    fireEvent.change(domainFilter, { target: { value: "opencode.ai" } });

    await waitFor(() => {
      expect(graphDomainRequests).toContain("opencode.ai");
    });

    act(() => {
      ws.emitMessage({
        event: "graph_delta",
        timestamp: Date.now(),
        nodes: [
          {
            id: "url:https://openrouter.ai/",
            kind: "url",
            label: "https://openrouter.ai/",
            url: "https://openrouter.ai/",
            domain: "openrouter.ai",
            depth: 1,
          },
        ],
        edges: [
          {
            id: "edge-2",
            kind: "hyperlink",
            source: "url:https://opencode.ai/",
            target: "url:https://openrouter.ai/",
          },
        ],
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/graph_delta/)).toBeTruthy();
    });
  });

  it("handles snapshot and entity_tick websocket updates", async () => {
    setupFetchMock();

    render(<WebGraphWeaverPanel />);

    const ws = MockWebSocket.instances[0];
    expect(ws).toBeTruthy();
    act(() => {
      ws.emitOpen();
      ws.emitMessage({
        event: "snapshot",
        status: statusPayload(["robots.example"], "paused"),
        entities: {
          ok: true,
          enabled: true,
          paused: true,
          node_cooldown_ms: 6000,
          max_requests_per_host: 3,
          entities: [
            {
              id: "entity-snapshot",
              label: "snapshot-entity",
              state: "cooldown",
              current_url: "https://openrouter.ai/",
              visits: 9,
            },
          ],
        },
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/state: paused/)).toBeTruthy();
      expect(screen.getByText(/count 1/)).toBeTruthy();
      expect(screen.getByText(/snapshot-entity/)).toBeTruthy();
    });

    act(() => {
      ws.emitMessage({
        event: "entity_tick",
        entities_enabled: false,
        entities_paused: false,
        activation_threshold: 1,
        node_cooldown_ms: 4000,
        max_requests_per_host: 2,
        entities: [],
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/state: disabled/)).toBeTruthy();
      expect(screen.getByText(/count 0/)).toBeTruthy();
    });
  });

  it("surfaces offline hint when bootstrap status request fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/weaver/status")) {
        throw new TypeError("Failed to fetch");
      }
      return mockJsonResponse({ ok: true, graph: { nodes: [], edges: [], counts: { nodes_total: 0, edges_total: 0, url_nodes_total: 0 } } });
    });

    render(<WebGraphWeaverPanel />);

    await waitFor(() => {
      expect(screen.getByText(/If Web Graph Weaver is not running/)).toBeTruthy();
    });
  });
});
