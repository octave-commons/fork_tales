import { expect, test, type Page, type Route } from "@playwright/test";

const mockProjection = {
  record: "eta_mu.ui_projection.bundle.v1",
  generated_at: "2026-02-20T19:45:00Z",
  perspective: "hybrid",
  perspectives: [
    {
      id: "hybrid",
      symbol: "perspective.hybrid",
      name: "Hybrid",
      merge: "hybrid",
      description: "Wallclock ordering with causal overlays.",
      default: true,
    },
  ],
  elements: [
    {
      record: "eta_mu.ui_projection.element.v1",
      id: "nexus.ui.daimoi_presence",
      kind: "panel",
      title: "Daimoi Presence Deck",
      binds_to: ["presence_dynamics", "daimoi_probabilistic"],
      field_bindings: {
        daimoi: 1,
      },
      presence: "witness_thread",
      tags: ["daimoi", "presence"],
      lane: "center",
      memory_scope: "session",
    },
  ],
  states: [
    {
      record: "eta_mu.ui_projection.element_state.v1",
      element_id: "nexus.ui.daimoi_presence",
      ts: Date.parse("2026-02-20T19:45:00Z"),
      mass: 0.74,
      priority: 0.98,
      area: 0.66,
      opacity: 0.97,
      pulse: 0.22,
      sources: ["presence_dynamics"],
      explain: {
        field_signal: 0.82,
        presence_signal: 0.93,
        queue_signal: 0.12,
        causal_signal: 0.2,
        dominant_field: "daimoi",
        dominant_level: 0.88,
        field_bindings: {
          daimoi: 1,
        },
        reason_en: "Daimoi lane prioritized for inspection.",
        reason_ja: "代網レーンを優先表示。",
        coherence_tension: 0.15,
      },
    },
  ],
  chat_sessions: [],
};

const mockCatalog = {
  generated_at: "2026-02-20T19:45:00Z",
  part_roots: ["/vaults/fork_tales/part64"],
  entity_manifest: [
    {
      id: "witness_thread",
      en: "Witness Thread",
      ja: "証人の糸",
      hue: 188,
      x: 0.54,
      y: 0.45,
    },
    {
      id: "chaos",
      en: "Chaos",
      ja: "混沌",
      hue: 26,
      x: 0.68,
      y: 0.58,
    },
  ],
  ui_projection: mockProjection,
  ui_perspectives: mockProjection.perspectives,
};

const mockFieldParticles = [
  {
    id: "dm-1",
    presence_id: "witness_thread",
    owner_presence_id: "witness_thread",
    x: 0.58,
    y: 0.46,
    size: 3,
    r: 120,
    g: 182,
    b: 255,
    message_probability: 0.8,
    route_probability: 0.72,
    action_probabilities: {
      deflect: 0.72,
      diffuse: 0.28,
    },
    job_probabilities: {
      route: 0.64,
      trace: 0.36,
    },
    drift_score: 0.43,
  },
  {
    id: "dm-2",
    presence_id: "witness_thread",
    owner_presence_id: "witness_thread",
    x: 0.62,
    y: 0.42,
    size: 3,
    r: 118,
    g: 176,
    b: 246,
    message_probability: 0.74,
    route_probability: 0.66,
    action_probabilities: {
      deflect: 0.68,
      diffuse: 0.32,
    },
    job_probabilities: {
      route: 0.58,
      review: 0.42,
    },
    drift_score: 0.37,
  },
  {
    id: "dm-3",
    presence_id: "chaos",
    owner_presence_id: "chaos",
    x: 0.33,
    y: 0.61,
    size: 3,
    r: 255,
    g: 162,
    b: 102,
    message_probability: 0.51,
    route_probability: 0.48,
    action_probabilities: {
      deflect: 0.31,
      diffuse: 0.69,
    },
    job_probabilities: {
      explore: 0.62,
      route: 0.38,
    },
    drift_score: 0.58,
  },
];

const mockSimulation = {
  timestamp: "2026-02-20T19:45:00Z",
  total: 3,
  audio: 0,
  image: 0,
  video: 0,
  points: [],
  field_particles: mockFieldParticles,
  presence_dynamics: {
    generated_at: "2026-02-20T19:45:00Z",
    click_events: 2,
    file_events: 1,
    field_particles: mockFieldParticles,
    recent_click_targets: [],
    recent_file_paths: [],
    presence_impacts: [],
    river_flow: {
      unit: "hz",
      rate: 0.18,
      turbulence: 0.1,
    },
    daimoi_probabilistic: {
      record: "eta_mu.daimoi_probabilistic.summary.v1",
      schema_version: "1",
      active: 3,
      spawned: 6,
      collisions: 4,
      deflects: 40,
      diffuses: 20,
      handoffs: 20,
      deliveries: 20,
      job_triggers: {
        route: 50,
        explore: 30,
        review: 20,
      },
      mean_package_entropy: 0.412,
      mean_message_probability: 0.683,
    },
  },
  projection: mockProjection,
  perspective: "hybrid",
};

async function fulfillJson(route: Route, payload: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(payload),
  });
}

async function installRuntimeMocks(page: Page): Promise<void> {
  await page.route("**/api/**", async (route) => {
    const requestUrl = new URL(route.request().url());
    const path = requestUrl.pathname;
    if (path.endsWith("/api/catalog")) {
      await fulfillJson(route, mockCatalog);
      return;
    }
    if (path.endsWith("/api/ui/projection")) {
      await fulfillJson(route, { projection: mockProjection });
      return;
    }
    if (path.endsWith("/api/study")) {
      await fulfillJson(route, { error: "not found" }, 404);
      return;
    }
    if (path.endsWith("/api/council")) {
      await fulfillJson(route, {
        ok: true,
        council: {
          pending_count: 0,
          approved_count: 0,
          decision_count: 0,
          decisions: [],
        },
      });
      return;
    }
    if (path.endsWith("/api/task/queue")) {
      await fulfillJson(route, {
        ok: true,
        queue: {
          queue_log: "mock",
          pending_count: 0,
          dedupe_keys: 0,
          event_count: 0,
          pending: [],
        },
      });
      return;
    }
    if (path.endsWith("/api/drift/scan")) {
      await fulfillJson(route, {
        active_drifts: [],
        blocked_gates: [],
      });
      return;
    }
    await fulfillJson(route, {});
  });

  await page.addInitScript(
    ({
      catalogPayload,
      simulationPayload,
    }: {
      catalogPayload: Record<string, unknown> & { ui_projection?: unknown };
      simulationPayload: Record<string, unknown> & { projection?: unknown };
    }) => {
      class MockWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSING = 2;
        static CLOSED = 3;

        readyState = MockWebSocket.CONNECTING;
        onopen: ((event: Event) => void) | null = null;
        onmessage: ((event: MessageEvent<string>) => void) | null = null;
        onclose: ((event: Event) => void) | null = null;
        onerror: ((event: Event) => void) | null = null;

        constructor(_url: string | URL) {
          window.setTimeout(() => {
            this.readyState = MockWebSocket.OPEN;
            if (typeof this.onopen === "function") {
              this.onopen(new Event("open"));
            }
            this.emit({ type: "catalog", catalog: catalogPayload, mix: null });
            this.emit({
              type: "simulation",
              simulation: simulationPayload,
              projection: simulationPayload?.projection ?? catalogPayload?.ui_projection ?? null,
            });
          }, 0);
        }

        emit(payload: unknown) {
          if (typeof this.onmessage === "function") {
            this.onmessage(new MessageEvent("message", { data: JSON.stringify(payload) }));
          }
        }

        send(_payload: unknown) {
          // no-op for deterministic tests
        }

        close() {
          this.readyState = MockWebSocket.CLOSED;
          if (typeof this.onclose === "function") {
            this.onclose(new Event("close"));
          }
        }
      }

      Object.defineProperty(window, "WebSocket", {
        configurable: true,
        writable: true,
        value: MockWebSocket,
      });
    },
    {
      catalogPayload: mockCatalog,
      simulationPayload: mockSimulation,
    },
  );
}

test.beforeEach(async ({ page }) => {
  await installRuntimeMocks(page);
});

test("renders normalized Daimoi action distributions", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByText("Daimoi Presence Deck / 代網存在甲板")).toBeVisible();
  await expect(page.getByText("Action distribution")).toBeVisible();
  await expect(page.getByText("40% (40)")).toBeVisible();
  await expect(page.getByText("20% (20)").first()).toBeVisible();
  const triggerSection = page.getByText("Top job trigger probabilities").locator("xpath=ancestor::section[1]");
  await expect(triggerSection).toBeVisible();
  await expect(triggerSection.getByText("route")).toBeVisible();
});

test("updates focus-lock label when selecting presence and daimoi", async ({ page }) => {
  await page.goto("/");

  const witnessPresenceButton = page.getByRole("button", {
    name: /Witness Thread/i,
  }).first();
  await expect(witnessPresenceButton).toBeVisible();
  await witnessPresenceButton.click();
  await expect(page.getByText("focus locked -> presence Witness Thread")).toBeVisible();

  const daimonButton = page.getByRole("button", {
    name: /dm-1/i,
  }).first();
  await expect(daimonButton).toBeVisible();
  await daimonButton.click();
  await expect(page.getByText("focus locked -> daimoi dm-1")).toBeVisible();
});
