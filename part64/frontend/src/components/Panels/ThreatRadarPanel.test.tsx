/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ThreatRadarPanel } from "./ThreatRadarPanel";
import type { SimulationState } from "../../types";

const THREAT_RADAR_CACHE_KEY = "eta_mu.threat_radar.report_cache.v2";

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function simulationPayload(timestamp: string, generatedAt: string): SimulationState {
  return {
    timestamp,
    total: 0,
    audio: 0,
    image: 0,
    video: 0,
    points: [],
    crawler_graph: {
      record: "eta-mu.crawler-graph.v1",
      generated_at: generatedAt,
      source: {
        endpoint: "ws://127.0.0.1:8787/ws",
        service: "simulation",
      },
      status: {},
      nodes: [],
      field_nodes: [],
      crawler_nodes: [],
      edges: [],
      stats: {
        field_count: 0,
        crawler_count: 0,
        edge_count: 0,
        kind_counts: {},
        field_counts: {},
        nodes_total: 0,
        edges_total: 0,
        url_nodes_total: 0,
      },
    },
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  window.localStorage.removeItem(THREAT_RADAR_CACHE_KEY);
});

describe("ThreatRadarPanel", () => {
  it("loads report data and conversation preview", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        const isLocal = url.includes("radar=local");
        return mockJsonResponse({
          ok: true,
          radar: isLocal ? "local" : "global",
          runtime: {
            label: isLocal ? "Local Cyber Threat Radar" : "Global Geopolitical Feed",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: isLocal
            ? {
              count: 1,
              critical_count: 1,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              hot_repos: [{ repo: "octocat/hello-world", max_risk_score: 12 }],
              threats: [
                {
                  risk_score: 12,
                  risk_level: "critical",
                  repo: "octocat/hello-world",
                  kind: "github:issue",
                  number: 7,
                  title: "critical vuln",
                  canonical_url: "https://github.com/octocat/hello-world/issues/7",
                  state: "open",
                  signals: ["references_cve", "open_state"],
                  cves: ["CVE-2026-0001"],
                },
              ],
            }
            : {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              source_count: 2,
              sources: [
                {
                  url: "https://www.ukmto.org/advisory/003-26",
                  kind: "maritime:ukmto_advisory",
                  title: "UKMTO Advisory 003-26",
                  source_type: "crawl",
                },
              ],
              threats: [],
            },
        });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({
          ok: true,
          comment_count: 2,
          url: "https://github.com/octocat/hello-world/issues/7",
          markdown: "## Conversation Chain\n\n- example line",
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<ThreatRadarPanel />);

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeTruthy();
    });

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "local" } });

    await waitFor(() => {
      expect(screen.getByText("Threat Radar / Local Cyber View")).toBeTruthy();
      expect(screen.getByText("critical vuln")).toBeTruthy();
      expect(screen.getByText(/Conversation Chain/)).toBeTruthy();
    });

    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/github/conversation")),
    ).toBe(true);

    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("radar=local")),
    ).toBe(true);

    const filterInput = screen.getByPlaceholderText("filter repo (owner/name)");
    fireEvent.change(filterInput, { target: { value: "octocat/hello-world" } });
    fireEvent.click(screen.getByText("apply filter"));

    await waitFor(() => {
      expect(
        fetchSpy.mock.calls.some(([url]) => String(url).includes("repo=octocat%2Fhello-world")),
      ).toBe(true);
    });
  });

  it("emits a ui toast when local critical count rises", async () => {
    let reportCall = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        const isLocal = url.includes("radar=local");
        if (!isLocal) {
          return mockJsonResponse({
            ok: true,
            radar: "global",
            runtime: { interval_seconds: 45, state: { last_status: "ok" } },
            result: {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              threats: [],
            },
          });
        }
        reportCall += 1;
        if (reportCall === 1) {
          return mockJsonResponse({
            ok: true,
            radar: "local",
            runtime: { interval_seconds: 45, state: { last_status: "ok" } },
            result: {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              hot_repos: [],
              threats: [],
            },
          });
        }
        return mockJsonResponse({
          ok: true,
          radar: "local",
          runtime: { interval_seconds: 45, state: { last_status: "ok" } },
          result: {
            count: 1,
            critical_count: 1,
            high_count: 0,
            medium_count: 0,
            low_count: 0,
            hot_repos: [{ repo: "octocat/hello-world", max_risk_score: 12 }],
            threats: [
              {
                risk_score: 12,
                risk_level: "critical",
                repo: "octocat/hello-world",
                kind: "github:issue",
                number: 7,
                title: "critical vuln",
                canonical_url: "https://github.com/octocat/hello-world/issues/7",
                state: "open",
                signals: ["references_cve"],
                cves: ["CVE-2026-0001"],
              },
            ],
          },
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({
          ok: true,
          comment_count: 1,
          url: "https://github.com/octocat/hello-world/issues/7",
          markdown: "conversation",
        });
      }
      return mockJsonResponse({ ok: true });
    });

    const dispatchSpy = vi.spyOn(window, "dispatchEvent");
    render(<ThreatRadarPanel />);

    await waitFor(() => {
      expect(screen.getByText("run now")).toBeTruthy();
    });

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "local" } });

    fireEvent.click(screen.getByText("run now"));

    await waitFor(() => {
      expect(
        dispatchSpy.mock.calls.some(([event]) => event instanceof CustomEvent && event.type === "ui:toast"),
      ).toBe(true);
    });
  });

  it("uses global radar and applies kind filter", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        const isGlobal = url.includes("radar=global");
        return mockJsonResponse({
          ok: true,
          radar: isGlobal ? "global" : "local",
          runtime: {
            label: isGlobal ? "Global Geopolitical Feed" : "Local Cyber Threat Radar",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: isGlobal
            ? {
              count: 1,
              critical_count: 0,
              high_count: 1,
              medium_count: 0,
              low_count: 0,
              hot_kinds: [{ kind: "maritime:ukmto_advisory", max_risk_score: 5 }],
              threats: [
                {
                  risk_score: 5,
                  risk_level: "high",
                  domain: "ukmto.org",
                  kind: "maritime:ukmto_advisory",
                  title: "Global advisory update",
                  canonical_url: "https://www.ukmto.org/advisory/003-26",
                  labels: ["military_activity", "electronic_interference"],
                },
              ],
            }
            : {
              count: 0,
              critical_count: 0,
              high_count: 0,
                medium_count: 0,
                low_count: 0,
                hot_repos: [{ repo: "octocat/hello-world", max_risk_score: 8 }],
                threats: [],
              },
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        throw new Error("conversation endpoint should not be called for global mode");
      }
      return mockJsonResponse({ ok: true });
    });

    render(<ThreatRadarPanel />);

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeTruthy();
    });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "global" } });

    await waitFor(() => {
      expect(
        fetchSpy.mock.calls.some(([url]) => String(url).includes("radar=global")),
      ).toBe(true);
    });

    await waitFor(() => {
      expect(screen.getByText("Threat Radar / Global Geopolitical Feed")).toBeTruthy();
      expect(screen.getAllByText("Global advisory update").length).toBeGreaterThan(0);
      expect(screen.getByText("Global Source Context")).toBeTruthy();
      expect(screen.getByText("source")).toBeTruthy();
    });

    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("radar=global")),
    ).toBe(true);

    const filterInput = screen.getByPlaceholderText("filter domain (ukmto.org) or kind (maritime:...)");
    fireEvent.change(filterInput, { target: { value: "maritime:ukmto_advisory" } });
    fireEvent.click(screen.getByText("apply filter"));

    await waitFor(() => {
      expect(
        fetchSpy.mock.calls.some(([url]) => {
          const value = String(url);
          return value.includes("radar=global") && value.includes("kind=maritime%3Aukmto_advisory");
        }),
      ).toBe(true);
    });
  });

  it("hides github-like rows in global mode even when kind is generic", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        const isGlobal = url.includes("radar=global");
        return mockJsonResponse({
          ok: true,
          radar: isGlobal ? "global" : "local",
          runtime: {
            label: isGlobal ? "Global Geopolitical Feed" : "Local Cyber Threat Radar",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: isGlobal
            ? {
              count: 2,
              critical_count: 1,
              high_count: 1,
              medium_count: 0,
              low_count: 0,
              threats: [
                {
                  risk_score: 9,
                  risk_level: "critical",
                  domain: "api.github.com",
                  kind: "web:article",
                  title: "github api leak row",
                  canonical_url: "https://api.github.com/repos/openai/codex/issues/9001",
                  labels: ["security"],
                },
                {
                  risk_score: 6,
                  risk_level: "high",
                  domain: "ukmto.org",
                  kind: "maritime:ukmto_advisory",
                  title: "Global advisory safe",
                  canonical_url: "https://www.ukmto.org/advisory/003-26",
                  labels: ["military_activity"],
                },
              ],
            }
            : {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              threats: [],
            },
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<ThreatRadarPanel />);

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeTruthy();
    });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "global" } });

    await waitFor(() => {
      expect(screen.getByText("Global advisory safe")).toBeTruthy();
    });
    expect(screen.queryByText("github api leak row")).toBeNull();
  });

  it("keeps github-like url rows in local mode even with non-github kind", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        const isLocal = url.includes("radar=local");
        return mockJsonResponse({
          ok: true,
          radar: isLocal ? "local" : "global",
          runtime: {
            label: isLocal ? "Local Cyber Threat Radar" : "Global Geopolitical Feed",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: isLocal
            ? {
              count: 2,
              critical_count: 1,
              high_count: 1,
              medium_count: 0,
              low_count: 0,
              threats: [
                {
                  risk_score: 9,
                  risk_level: "critical",
                  repo: "openai/codex",
                  kind: "web:article",
                  title: "local github api issue",
                  canonical_url: "https://api.github.com/repos/openai/codex/issues/9002",
                  labels: ["security"],
                },
                {
                  risk_score: 6,
                  risk_level: "high",
                  kind: "maritime:ukmto_advisory",
                  title: "should not appear in local",
                  canonical_url: "https://www.ukmto.org/advisory/003-27",
                  labels: ["military_activity"],
                },
              ],
            }
            : {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              threats: [],
            },
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<ThreatRadarPanel />);

    await waitFor(() => {
      expect(screen.getByText("local github api issue")).toBeTruthy();
    });
    expect(screen.queryByText("should not appear in local")).toBeNull();
  });

  it("shows source signal classes when global threats are empty", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        const isGlobal = url.includes("radar=global");
        return mockJsonResponse({
          ok: true,
          radar: isGlobal ? "global" : "local",
          runtime: {
            label: isGlobal ? "Global Geopolitical Feed" : "Local Cyber Threat Radar",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: isGlobal
            ? {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              hot_kinds: [],
              source_count: 2,
              sources: [
                {
                  url: "https://www.ukmto.org/advisory/003-26",
                  kind: "maritime:ukmto_advisory",
                  title: "UKMTO Advisory 003-26",
                  source_type: "crawl",
                },
                {
                  url: "https://www.reuters.com/world/middle-east/maritime-advisory-2026-03-01/",
                  kind: "maritime:news_report",
                  title: "Reuters Maritime Advisory",
                  source_type: "crawl",
                },
              ],
              threats: [],
            }
            : {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              hot_repos: [{ repo: "octocat/hello-world", max_risk_score: 8 }],
              threats: [],
            },
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<ThreatRadarPanel />);

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeTruthy();
    });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "global" } });

    await waitFor(() => {
      expect(screen.getByText("source signal inputs")).toBeTruthy();
      expect(screen.getByText("crawl")).toBeTruthy();
      expect(screen.queryByText("UKMTO Advisory 003-26 (crawl)")).toBeNull();
      expect(screen.queryByText("Reuters Maritime Advisory (crawl)")).toBeNull();
    });
  });

  it("can ask muse about selected global feed threat", async () => {
    let museRequestBody: Record<string, unknown> | null = null;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        const isGlobal = url.includes("radar=global");
        return mockJsonResponse({
          ok: true,
          radar: isGlobal ? "global" : "local",
          runtime: {
            muse_id: "github_security_review",
            label: isGlobal ? "Global Geopolitical Feed" : "Local Cyber Threat Radar",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: isGlobal
            ? {
              count: 1,
              critical_count: 0,
              high_count: 1,
              medium_count: 0,
              low_count: 0,
              source_count: 1,
              sources: [
                {
                  url: "https://www.ukmto.org/advisory/003-26",
                  kind: "maritime:ukmto_advisory",
                  title: "UKMTO Advisory 003-26",
                  source_type: "crawl",
                },
              ],
              threats: [
                {
                  risk_score: 7,
                  risk_level: "high",
                  domain: "ukmto.org",
                  kind: "maritime:ukmto_advisory",
                  title: "Global advisory ask target",
                  canonical_url: "https://www.ukmto.org/advisory/003-26",
                  labels: ["military_activity"],
                },
              ],
            }
            : {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              threats: [],
            },
        });
      }
      if (url.includes("/api/muse/message")) {
        const rawBody = String(init?.body || "{}");
        museRequestBody = JSON.parse(rawBody) as Record<string, unknown>;
        return mockJsonResponse({
          ok: true,
          reply: "Situation: shipping risk rising. Why: impacts regional posture. Check: verify route advisories now.",
          turn_id: "turn:feed-42",
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<ThreatRadarPanel />);

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeTruthy();
    });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "global" } });

    await waitFor(() => {
      expect(screen.getAllByText("Global advisory ask target").length).toBeGreaterThan(0);
      expect(screen.getByText("talk to this feed")).toBeTruthy();
    });

    fireEvent.change(screen.getByPlaceholderText("ask: what changed and why does it matter?"), {
      target: { value: "What changed in this feed?" },
    });
    fireEvent.click(screen.getByText("ask"));

    await waitFor(() => {
      expect(screen.getByText(/Situation: shipping risk rising/)).toBeTruthy();
      expect(screen.getByText("turn turn:feed-42")).toBeTruthy();
    });

    expect(museRequestBody).not.toBeNull();
    const museRequest = (museRequestBody || {}) as Record<string, unknown>;
    expect(String(museRequest.muse_id || "")).toBe("github_security_review");
    expect(String(museRequest.mode || "")).toBe("deterministic");
    expect(String(museRequest.text || "")).toContain("Global advisory ask target");
    expect(String(museRequest.text || "")).toContain("What changed in this feed?");
  });

  it("supports fixed global mode and routes ask requests to assigned chaos muse", async () => {
    let museRequestBody: Record<string, unknown> | null = null;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        return mockJsonResponse({
          ok: true,
          radar: "global",
          runtime: {
            muse_id: "github_security_review",
            label: "Global Geopolitical Feed",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: {
            count: 1,
            critical_count: 0,
            high_count: 1,
            medium_count: 0,
            low_count: 0,
            threats: [
              {
                risk_score: 7,
                risk_level: "high",
                domain: "ukmto.org",
                kind: "maritime:ukmto_advisory",
                title: "Chaos global target",
                canonical_url: "https://www.ukmto.org/advisory/009-26",
                labels: ["maritime_activity"],
              },
            ],
          },
        });
      }
      if (url.includes("/api/muse/message")) {
        museRequestBody = JSON.parse(String(init?.body || "{}")) as Record<string, unknown>;
        return mockJsonResponse({
          ok: true,
          reply: "Chaos reply",
          turn_id: "turn-chaos",
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    render(
      <ThreatRadarPanel
        fixedRadarMode="global"
        assignedMuseByMode={{
          local: "witness_thread",
          global: "chaos",
        }}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByRole("combobox")).toBeNull();
      expect(screen.getByText("Chaos Muse / Global Geopolitical Radar")).toBeTruthy();
      expect(screen.getByText("Chaos global target")).toBeTruthy();
    });

    fireEvent.click(screen.getByText("details"));

    await waitFor(() => {
      expect(screen.getByText("talk to this feed")).toBeTruthy();
    });

    fireEvent.change(screen.getByPlaceholderText("ask: what changed and why does it matter?"), {
      target: { value: "Give me one immediate check." },
    });
    fireEvent.click(screen.getByText("ask"));

    await waitFor(() => {
      expect(screen.getByText("Chaos reply")).toBeTruthy();
    });

    const museRequest = (museRequestBody || {}) as Record<string, unknown>;
    expect(String(museRequest.muse_id || "")).toBe("chaos");
  });

  it("renders compact source-context layout when muse chat panel is provided", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        return mockJsonResponse({
          ok: true,
          radar: "global",
          runtime: {
            muse_id: "chaos",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: {
            count: 1,
            critical_count: 1,
            high_count: 0,
            medium_count: 0,
            low_count: 0,
            source_count: 2,
            sources: [
              {
                url: "https://www.ukmto.org/advisory/003-26",
                kind: "maritime:ukmto_advisory",
                title: "UKMTO Advisory 003-26",
                source_type: "crawl",
              },
            ],
            threats: [
              {
                risk_score: 10,
                risk_level: "critical",
                domain: "ukmto.org",
                kind: "maritime:ukmto_advisory",
                title: "Chaos compact row",
                canonical_url: "https://www.ukmto.org/advisory/003-26",
                labels: ["maritime_activity", "electronic_interference"],
              },
            ],
          },
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    render(
      <ThreatRadarPanel
        fixedRadarMode="global"
        assignedMuseByMode={{
          local: "witness_thread",
          global: "chaos",
        }}
        museChatPanel={<div>muse lane body</div>}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("GLOBAL SOURCE CONTEXT")).toBeTruthy();
      expect(screen.getByText("Top Threats")).toBeTruthy();
      expect(screen.getByText("Chaos compact row")).toBeTruthy();
      expect(screen.getByText("muse lane body")).toBeTruthy();
    });

    expect(screen.queryByText("Chaos Muse / Global Geopolitical Radar")).toBeNull();
  });

  it("ignores stale local report responses after switching back to global", async () => {
    const localReportResolver: { resolve: (value: Response) => void } = {
      resolve: () => undefined,
    };
    const pendingLocalReport = new Promise<Response>((resolve) => {
      localReportResolver.resolve = resolve;
    });

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        if (url.includes("radar=local")) {
          return pendingLocalReport;
        }
        if (url.includes("radar=global")) {
          return mockJsonResponse({
            ok: true,
            radar: "global",
            runtime: {
              label: "Global Geopolitical Feed",
              interval_seconds: 45,
              state: { last_status: "ok" },
            },
            result: {
              count: 1,
              critical_count: 0,
              high_count: 1,
              medium_count: 0,
              low_count: 0,
              hot_kinds: [{ kind: "maritime:ukmto_advisory", max_risk_score: 5 }],
              threats: [
                {
                  risk_score: 5,
                  risk_level: "high",
                  domain: "ukmto.org",
                  kind: "maritime:ukmto_advisory",
                  title: "Global only advisory",
                  canonical_url: "https://www.ukmto.org/advisory/003-26",
                  labels: ["military_activity"],
                },
              ],
            },
          });
        }
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<ThreatRadarPanel />);

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeTruthy();
    });

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "local" } });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "global" } });

    await waitFor(() => {
      expect(screen.getAllByText("Global only advisory").length).toBeGreaterThan(0);
      expect(screen.getByText("Threat Radar / Global Geopolitical Feed")).toBeTruthy();
    });

    localReportResolver.resolve(
      mockJsonResponse({
        ok: true,
        radar: "local",
        runtime: {
          label: "Local Threat Radar",
          interval_seconds: 45,
          state: { last_status: "ok" },
        },
        result: {
          count: 1,
          critical_count: 0,
          high_count: 0,
          medium_count: 1,
          low_count: 0,
          hot_repos: [{ repo: "octocat/hello-world", max_risk_score: 6 }],
          threats: [
            {
              risk_score: 6,
              risk_level: "medium",
              kind: "github:issue",
              repo: "octocat/hello-world",
              number: 77,
              title: "stale github threat",
              canonical_url: "https://github.com/octocat/hello-world/issues/77",
              signals: ["security_label"],
            },
          ],
        },
      }),
    );

    await waitFor(() => {
      expect(screen.queryByText("stale github threat")).toBeNull();
      expect(screen.getAllByText("Global only advisory").length).toBeGreaterThan(0);
    });
  });

  it("refreshes report when threat radar muse websocket events arrive", async () => {
    let globalReportCall = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        if (url.includes("radar=local")) {
          return mockJsonResponse({
            ok: true,
            radar: "local",
            runtime: {
              muse_id: "github_security_review",
              label: "Local Cyber Threat Radar",
              interval_seconds: 45,
              state: { last_status: "idle" },
            },
            result: {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              threats: [],
            },
          });
        }
        globalReportCall += 1;
        if (globalReportCall <= 1) {
          return mockJsonResponse({
            ok: true,
            radar: "global",
            runtime: {
              muse_id: "github_security_review",
              label: "Global Geopolitical Feed",
              interval_seconds: 45,
              state: { last_status: "idle" },
            },
            result: {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              source_count: 1,
              sources: [
                {
                  url: "https://www.ukmto.org/advisory/003-26",
                  kind: "maritime:ukmto_advisory",
                  title: "UKMTO Advisory 003-26",
                  source_type: "crawl",
                },
              ],
              threats: [],
            },
          });
        }
        return mockJsonResponse({
          ok: true,
          radar: "global",
          runtime: {
            muse_id: "github_security_review",
            label: "Global Geopolitical Feed",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: {
            count: 1,
            critical_count: 0,
            high_count: 1,
            medium_count: 0,
            low_count: 0,
            source_count: 1,
            sources: [
              {
                url: "https://www.ukmto.org/advisory/003-27",
                kind: "maritime:ukmto_advisory",
                title: "UKMTO Advisory 003-27",
                source_type: "crawl",
              },
            ],
            threats: [
              {
                risk_score: 7,
                risk_level: "high",
                domain: "ukmto.org",
                kind: "maritime:ukmto_advisory",
                title: "ws refreshed advisory",
                canonical_url: "https://www.ukmto.org/advisory/003-27",
                labels: ["maritime_activity"],
              },
            ],
          },
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    const { rerender } = render(<ThreatRadarPanel museEvents={[]} />);

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeTruthy();
    });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "global" } });

    await waitFor(() => {
      expect(screen.getByText("Threat Radar / Global Geopolitical Feed")).toBeTruthy();
    });
    expect(screen.queryByText("ws refreshed advisory")).toBeNull();

    rerender(
      <ThreatRadarPanel
        museEvents={[
          {
            record: "eta-mu.muse-event.v1",
            schema_version: "muse.event.v1",
            event_id: "mev:42",
            seq: 42,
            kind: "muse.turn.completed",
            status: "ok",
            muse_id: "github_security_review",
            turn_id: "turn:42",
            ts: "2026-03-03T00:00:00Z",
            payload: {},
          },
        ]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("ws refreshed advisory")).toBeTruthy();
    });
  });

  it("refreshes global report from simulation websocket stream updates", async () => {
    let nowMs = 1_000_000;
    vi.spyOn(Date, "now").mockImplementation(() => nowMs);
    let globalReportCall = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        if (url.includes("radar=local")) {
          return mockJsonResponse({
            ok: true,
            radar: "local",
            runtime: {
              muse_id: "github_security_review",
              label: "Local Cyber Threat Radar",
              interval_seconds: 45,
              state: { last_status: "idle" },
            },
            result: {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              threats: [],
            },
          });
        }
        globalReportCall += 1;
        if (globalReportCall <= 1) {
          return mockJsonResponse({
            ok: true,
            radar: "global",
            runtime: {
              muse_id: "github_security_review",
              label: "Global Geopolitical Feed",
              interval_seconds: 45,
              state: { last_status: "idle" },
            },
            result: {
              count: 0,
              critical_count: 0,
              high_count: 0,
              medium_count: 0,
              low_count: 0,
              threats: [],
            },
          });
        }
        return mockJsonResponse({
          ok: true,
          radar: "global",
          runtime: {
            muse_id: "github_security_review",
            label: "Global Geopolitical Feed",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
          result: {
            count: 1,
            critical_count: 0,
            high_count: 1,
            medium_count: 0,
            low_count: 0,
            threats: [
              {
                risk_score: 6,
                risk_level: "high",
                domain: "ukmto.org",
                kind: "maritime:ukmto_advisory",
                title: "stream refreshed advisory",
                canonical_url: "https://www.ukmto.org/advisory/003-29",
                labels: ["maritime_activity"],
              },
            ],
          },
        });
      }
      if (url.includes("/api/muse/threat-radar/tick")) {
        return mockJsonResponse({ ok: true, status: "triggered" });
      }
      if (url.includes("/api/github/conversation")) {
        return mockJsonResponse({ ok: true, markdown: "", comment_count: 0 });
      }
      return mockJsonResponse({ ok: true });
    });

    const { rerender } = render(
      <ThreatRadarPanel
        museEvents={[]}
        simulation={simulationPayload("2026-03-04T16:00:00Z", "2026-03-04T16:00:00Z")}
        isConnected
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole("combobox")).toBeTruthy();
    });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "global" } });

    await waitFor(() => {
      expect(screen.getByText("Threat Radar / Global Geopolitical Feed")).toBeTruthy();
    });
    expect(screen.queryByText("stream refreshed advisory")).toBeNull();

    nowMs += 4000;
    rerender(
      <ThreatRadarPanel
        museEvents={[]}
        simulation={simulationPayload("2026-03-04T16:00:05Z", "2026-03-04T16:00:05Z")}
        isConnected
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("stream refreshed advisory")).toBeTruthy();
    });
  });
});
