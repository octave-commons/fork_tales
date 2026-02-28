/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ThreatRadarPanel } from "./ThreatRadarPanel";

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ThreatRadarPanel", () => {
  it("loads report data and conversation preview", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        return mockJsonResponse({
          ok: true,
          runtime: {
            label: "GitHub Threat Radar",
            interval_seconds: 45,
            state: { last_status: "ok" },
          },
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
                signals: ["references_cve", "open_state"],
                cves: ["CVE-2026-0001"],
              },
            ],
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
      expect(screen.getByText("GitHub Threat Radar")).toBeTruthy();
      expect(screen.getByText("critical vuln")).toBeTruthy();
      expect(screen.getByText(/Conversation Chain/)).toBeTruthy();
    });

    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/github/conversation")),
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

  it("emits a ui toast when critical count rises", async () => {
    let reportCall = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/muse/threat-radar/report")) {
        reportCall += 1;
        if (reportCall === 1) {
          return mockJsonResponse({
            ok: true,
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

    fireEvent.click(screen.getByText("run now"));

    await waitFor(() => {
      expect(
        dispatchSpy.mock.calls.some(([event]) => event instanceof CustomEvent && event.type === "ui:toast"),
      ).toBe(true);
    });
  });
});
