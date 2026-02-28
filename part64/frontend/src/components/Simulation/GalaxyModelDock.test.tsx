/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GalaxyModelDock } from "./GalaxyModelDock";

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status >= 200 && status < 300 ? "OK" : "ERR",
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response;
}

type CallRecord = {
  url: string;
  method: string;
  body: unknown;
};

function setupGatewayFetchMock() {
  const calls: CallRecord[] = [];
  type CodexAccount = {
    id: string;
    provider: string;
    masked_key: string;
    status: string;
    requests: number;
    last_used_ts: number | null;
    key_cooldown_remaining: number | null;
    env_key: string;
    persisted: boolean;
  };
  let rotationMode: "balanced" | "sequential" = "balanced";
  let fairCycleEnabled = false;
  let accounts: CodexAccount[] = [
    {
      id: "acct-1",
      provider: "openai",
      masked_key: "sk-***1111",
      status: "active",
      requests: 4,
      last_used_ts: 1735689600,
      key_cooldown_remaining: 0,
      env_key: "OPENAI_API_KEY_1",
      persisted: true,
    },
  ];

  const snapshot = () => ({
    schema_version: "1",
    provider: "openai",
    rotation_mode: rotationMode,
    fair_cycle_enabled: fairCycleEnabled,
    credential_count: accounts.length,
    accounts,
  });

  const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = String(init?.method || "GET").toUpperCase();
    const parsedBody = init?.body ? JSON.parse(String(init.body)) : null;
    calls.push({
      url,
      method,
      body: parsedBody,
    });

    if (url.includes("/v1/gateway/config")) {
      return mockJsonResponse({
        enabled: true,
        default_mode: "smart",
        default_model: "octave-commons/promethean",
        modes: ["smart", "direct", "hardway"],
        hardware_options: ["gpu", "npu", "cpu"],
        field_routes: {
          code: { smart: "octave-commons/promethean" },
          general: { smart: "octave-commons/promethean" },
        },
      });
    }

    if (url.includes("/v1/models?enriched=false")) {
      return mockJsonResponse({
        data: [
          { id: "octave-commons/promethean" },
          { id: "openai/gpt-4o-mini" },
          { id: "anthropic/claude-3.5-sonnet" },
        ],
      });
    }

    if (url.includes("/v1/providers")) {
      return mockJsonResponse(["openai", "anthropic"]);
    }

    if (url.includes("/v1/quota-stats")) {
      return mockJsonResponse({
        providers: {
          openai: {
            credential_count: 2,
            active_count: 1,
            on_cooldown_count: 1,
            exhausted_count: 0,
            total_requests: 9,
            approx_cost: 1.25,
            credentials: [
              { identifier: "OPENAI_API_KEY_1", status: "active", requests: 4, tier: "paid" },
              { identifier: "OPENAI_API_KEY_2", status: "cooldown", requests: 5, tier: "paid" },
            ],
          },
        },
        summary: {
          total_credentials: 2,
          active_credentials: 1,
          exhausted_credentials: 0,
          total_requests: 9,
        },
      });
    }

    if (url.includes("/v1/openai/codex/accounts?provider=openai")) {
      return mockJsonResponse(snapshot());
    }

    if (url.includes("/v1/openai/codex/events?limit=20")) {
      return mockJsonResponse({
        events: [
          {
            id: "evt-1",
            ts: "2026-02-28T14:00:00Z",
            action: "accounts.sync",
            outcome: "ok",
          },
        ],
      });
    }

    if (url.includes("/v1/openai/codex/login") && method === "POST") {
      const nextIndex = accounts.length + 1;
      accounts = [
        ...accounts,
        {
          id: `acct-${nextIndex}`,
          provider: "openai",
          masked_key: "sk-***9999",
          status: "active",
          requests: 0,
          last_used_ts: null,
          key_cooldown_remaining: null,
          env_key: `OPENAI_API_KEY_${nextIndex}`,
          persisted: true,
        },
      ];
      return mockJsonResponse({
        ok: true,
        added: true,
        persisted: true,
        accounts: snapshot(),
        validation: {
          model_count: 5,
          codex_model_count: 2,
        },
      });
    }

    if (url.includes("/v1/openai/codex/cycle") && method === "POST") {
      const body = parsedBody as Record<string, unknown>;
      rotationMode = body.rotation_mode === "sequential" ? "sequential" : "balanced";
      fairCycleEnabled = Boolean(body.fair_cycle_enabled);
      return mockJsonResponse({
        ok: true,
        accounts: snapshot(),
      });
    }

    if (url.includes("/v1/openai/codex/accounts/") && method === "DELETE") {
      const accountId = decodeURIComponent(url.split("/v1/openai/codex/accounts/")[1].split("?")[0] || "");
      accounts = accounts.filter((entry) => entry.id !== accountId);
      return mockJsonResponse({
        ok: true,
        accounts: snapshot(),
      });
    }

    if (url.includes("/v1/gateway/route") && method === "POST") {
      const body = parsedBody as Record<string, unknown>;
      return mockJsonResponse({
        applied: true,
        mode: String((body.gateway as Record<string, unknown>)?.mode ?? "smart"),
        field: String((body.gateway as Record<string, unknown>)?.field ?? "code"),
        hardware: String((body.gateway as Record<string, unknown>)?.hardware ?? "gpu"),
        requested_model: String(body.model ?? "octave-commons/promethean"),
        resolved_model: "openai/gpt-4o-mini",
        resolved_model_public: "openai/gpt-4o-mini",
        reason: "balanced route",
        candidate_models: ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"],
        provider_available: true,
      });
    }

    if (url.includes("/v1/chat/completions") && method === "POST") {
      return mockJsonResponse({
        choices: [
          {
            message: {
              content: "Ready: route is healthy and quota pool is active.",
            },
          },
        ],
      });
    }

    return mockJsonResponse({ ok: true });
  });

  return {
    fetchSpy,
    calls,
  };
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("GalaxyModelDock", () => {
  it("loads gateway catalog and codex account pool on mount", async () => {
    const { calls } = setupGatewayFetchMock();

    render(<GalaxyModelDock onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText("catalog synced")).toBeTruthy();
      expect(screen.getByText("loaded 1 codex account")).toBeTruthy();
      expect(screen.getByText("openai codex account pool")).toBeTruthy();
      expect(screen.getAllByText("openai").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("catalog.sync").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("codex.accounts.sync").length).toBeGreaterThanOrEqual(1);
    });

    expect(calls.some((call) => call.url.includes("/v1/gateway/config"))).toBe(true);
    expect(calls.some((call) => call.url.includes("/v1/openai/codex/accounts?provider=openai"))).toBe(true);
  });

  it("previews routes and runs prompt through gateway", async () => {
    const { calls } = setupGatewayFetchMock();

    render(<GalaxyModelDock onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText("catalog synced")).toBeTruthy();
      expect(screen.getByRole("button", { name: "preview route" })).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: "preview route" }));

    await waitFor(() => {
      expect(screen.getByText("route decision")).toBeTruthy();
      expect(screen.getByText("balanced route")).toBeTruthy();
      expect(screen.getAllByText("openai/gpt-4o-mini").length).toBeGreaterThanOrEqual(1);
    });

    fireEvent.click(screen.getByRole("button", { name: "run prompt" }));

    await waitFor(() => {
      expect(screen.getByText("Ready: route is healthy and quota pool is active.")).toBeTruthy();
      expect(screen.getAllByText("prompt.run").length).toBeGreaterThanOrEqual(1);
    });

    const routeCalls = calls.filter((call) => call.url.includes("/v1/gateway/route") && call.method === "POST");
    expect(routeCalls.length).toBeGreaterThanOrEqual(2);
    const routeBody = routeCalls[0].body as Record<string, unknown>;
    expect(routeBody).toMatchObject({
      model: "octave-commons/promethean",
      stream: false,
    });
    expect((routeBody.gateway as Record<string, unknown>)?.mode).toBe("smart");
    expect((routeBody.gateway as Record<string, unknown>)?.field).toBe("code");
  });

  it("updates cycle strategy, adds account, and removes account", async () => {
    const { calls } = setupGatewayFetchMock();

    render(<GalaxyModelDock onClose={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText("loaded 1 codex account")).toBeTruthy();
    });

    fireEvent.change(screen.getByDisplayValue("balanced"), {
      target: { value: "sequential" },
    });
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: "apply cycle strategy" }));

    await waitFor(() => {
      expect(screen.getByText("cycle strategy updated")).toBeTruthy();
    });

    fireEvent.change(screen.getByPlaceholderText("Paste OpenAI API key (sk-...)"), {
      target: { value: "sk-test-123" },
    });
    fireEvent.click(screen.getByRole("button", { name: "login + add" }));

    await waitFor(() => {
      expect(screen.getByText(/account ready \(2 codex models detected\)/)).toBeTruthy();
      expect(screen.getByText("sk-***9999")).toBeTruthy();
    });

    const removeButtons = screen.getAllByRole("button", { name: "remove" });
    fireEvent.click(removeButtons[0]);

    await waitFor(() => {
      expect(screen.getByText("account removed")).toBeTruthy();
    });

    const cycleCall = calls.find((call) => call.url.includes("/v1/openai/codex/cycle") && call.method === "POST");
    expect(cycleCall).toBeTruthy();
    expect(cycleCall?.body).toMatchObject({
      rotation_mode: "sequential",
      fair_cycle_enabled: true,
      provider: "openai",
      persist: true,
    });

    const loginCall = calls.find((call) => call.url.includes("/v1/openai/codex/login") && call.method === "POST");
    expect(loginCall).toBeTruthy();
    expect(loginCall?.body).toMatchObject({
      provider: "openai",
      api_key: "sk-test-123",
      persist: true,
      validate_key: true,
    });

    const removeCall = calls.find((call) => call.url.includes("/v1/openai/codex/accounts/") && call.method === "DELETE");
    expect(removeCall).toBeTruthy();
  });
});
