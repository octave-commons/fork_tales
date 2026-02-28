/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PresenceCallDeck } from "./PresenceCallDeck";

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

describe("PresenceCallDeck", () => {
  it("uses fallback presences and sends transcript messages", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/presence/say")) {
        return mockJsonResponse({
          rendered_text: "Witness thread response.",
          presence_name: {
            en: "Witness Thread",
          },
        });
      }
      return mockJsonResponse({ ok: true });
    });

    render(<PresenceCallDeck catalog={null} simulation={null} />);

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Witness Thread" })).toBeTruthy();
      expect(screen.getByRole("option", { name: "Receipt River" })).toBeTruthy();
    });

    const composer = screen.getByPlaceholderText("Message Witness Thread...");
    fireEvent.change(composer, { target: { value: "hello from ui" } });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() => {
      expect(screen.getByText("hello from ui")).toBeTruthy();
      expect(screen.getByText("Witness thread response.")).toBeTruthy();
    });

    const sayCall = fetchSpy.mock.calls.find(([url]) => String(url).includes("/api/presence/say"));
    expect(sayCall).toBeTruthy();
    const sayBody = JSON.parse(String((sayCall?.[1] as RequestInit | undefined)?.body || "{}"));
    expect(sayBody).toMatchObject({
      presence_id: "witness_thread",
      text: "hello from ui",
    });
  });

  it("surfaces call-start and presence-request errors", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/presence/say")) {
        return mockJsonResponse({ ok: false, error: "request failed" }, 500);
      }
      return mockJsonResponse({ ok: true });
    });

    render(
      <PresenceCallDeck
        catalog={{
          entity_manifest: [
            {
              id: "gates_of_truth",
              en: "Gates of Truth",
              ja: "Gates of Truth",
              type: "presence",
            },
          ],
        } as never}
        simulation={null}
      />,
    );

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Gates of Truth" })).toBeTruthy();
    });

    fireEvent.click(screen.getByText("Start Call"));

    await waitFor(() => {
      expect(screen.getByText("Error")).toBeTruthy();
      expect(screen.getByText(/Call failed:/)).toBeTruthy();
    });

    const composer = screen.getByPlaceholderText("Message Gates of Truth...");
    fireEvent.change(composer, { target: { value: "status?" } });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() => {
      expect(screen.getByText(/Presence request failed:/)).toBeTruthy();
    });

    expect(screen.getAllByText(/presence\/say failed \(500\)/).length).toBeGreaterThanOrEqual(1);
  });
});
