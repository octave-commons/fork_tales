/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WorldLogPanel } from "./WorldLogPanel";
import type { WorldLogEvent, WorldLogPayload } from "../../types";

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function makeEvent(overrides: Partial<WorldLogEvent> = {}): WorldLogEvent {
  return {
    id: "evt-1",
    ts: "2026-02-28T12:00:00Z",
    source: "nasa_gibs",
    kind: "ingested",
    status: "ok",
    title: "Daily Earth tile",
    detail: "Fetched imagery tile for atmosphere sampling.",
    refs: ["https://example.test/tile.png"],
    tags: ["earth"],
    relations: [
      {
        event_id: "evt-0",
        score: 0.71,
        kind: "semantic",
      },
    ],
    ...overrides,
  };
}

function makePayload(overrides: Partial<WorldLogPayload> = {}): WorldLogPayload {
  const events = (overrides.events ?? [makeEvent()]) as WorldLogEvent[];
  return {
    ok: true,
    record: "eta-mu.world-log.v1",
    generated_at: "2026-02-28T12:00:00Z",
    count: events.length,
    limit: 180,
    pending_inbox: 1,
    sources: { nasa_gibs: events.length },
    kinds: { ingested: events.length },
    relation_count: 1,
    events,
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("WorldLogPanel", () => {
  it("renders fetched events and refreshes the stream", async () => {
    const firstPayload = makePayload({
      events: [makeEvent({ id: "evt-1", title: "Initial event title" })],
    });
    const refreshedPayload = makePayload({
      events: [makeEvent({ id: "evt-2", title: "Refreshed event title" })],
      pending_inbox: 0,
      relation_count: 3,
    });

    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(firstPayload))
      .mockResolvedValueOnce(mockJsonResponse(refreshedPayload));

    render(<WorldLogPanel catalog={null} />);

    await waitFor(() => {
      expect(screen.getByText("Initial event title")).toBeTruthy();
    });

    const preview = screen.getByRole("img", { name: /tile$/i });
    expect(preview.getAttribute("src")).toBe("https://example.test/tile.png");
    expect(preview.getAttribute("loading")).toBe("lazy");
    expect(preview.getAttribute("referrerpolicy")).toBe("no-referrer");

    fireEvent.click(screen.getByRole("button", { name: /refresh/i }));

    await waitFor(() => {
      expect(screen.getByText("Refreshed event title")).toBeTruthy();
    });

    const worldEventCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).includes("/api/world/events?limit=180"),
    );
    expect(worldEventCalls.length).toBeGreaterThanOrEqual(2);
  });

  it("posts eta-mu sync requests and updates sync status", async () => {
    const initialPayload = makePayload({
      events: [makeEvent({ title: "Before sync" })],
    });
    const afterSyncPayload = makePayload({
      events: [makeEvent({ title: "After sync" })],
    });

    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(mockJsonResponse(initialPayload))
      .mockResolvedValueOnce(mockJsonResponse({ ok: true, status: "queued" }))
      .mockResolvedValueOnce(mockJsonResponse(afterSyncPayload));

    render(<WorldLogPanel catalog={null} />);

    await waitFor(() => {
      expect(screen.getByText("Before sync")).toBeTruthy();
    });

    fireEvent.click(screen.getByRole("button", { name: /ingest/i }));

    await waitFor(() => {
      expect(screen.getByText("inbox sync queued")).toBeTruthy();
    });

    const syncCall = fetchMock.mock.calls.find(([url]) =>
      String(url).includes("/api/eta-mu/sync"),
    );
    expect(syncCall).toBeTruthy();
    const syncOptions = (syncCall?.[1] ?? {}) as RequestInit;
    expect(syncOptions.method).toBe("POST");
    expect(syncOptions.headers).toEqual({ "Content-Type": "application/json" });
    expect(syncOptions.body).toBe(JSON.stringify({ force: true, wait: false }));

    await waitFor(() => {
      expect(screen.getByText("After sync")).toBeTruthy();
    });
  });

  it("does not render image previews for non nasa sources", async () => {
    const payload = makePayload({
      events: [
        makeEvent({
          source: "github_presence",
          title: "GitHub event",
          refs: ["https://example.test/should-not-render.jpg"],
        }),
      ],
      sources: { github_presence: 1 },
    });

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(mockJsonResponse(payload));

    render(<WorldLogPanel catalog={null} />);

    await waitFor(() => {
      expect(screen.getByText("GitHub event")).toBeTruthy();
    });

    expect(screen.queryByRole("img")).toBeNull();
  });

  it("surfaces request errors when world-log fetch fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(mockJsonResponse({ ok: false }, 503));

    render(<WorldLogPanel catalog={null} />);

    await waitFor(() => {
      expect(screen.getByText("world log request failed (503)")).toBeTruthy();
    });
  });
});
