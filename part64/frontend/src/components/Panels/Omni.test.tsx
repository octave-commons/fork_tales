/* @vitest-environment jsdom */

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { OmniPanel } from "./Omni";
import type { Catalog } from "../../types";

function makeCatalog(): Catalog {
  return {
    generated_at: "2026-02-28T00:00:00Z",
    part_roots: ["part64"],
    counts: {},
    canonical_terms: [],
    cover_fields: [
      {
        id: "cover-1",
        part: "64",
        display_name: { en: "Eta Mu Cover", ja: "エタミュー" },
        display_role: { en: "cover", ja: "表紙" },
        url: "https://example.test/cover.png",
        seed: "seed-1",
      },
    ],
  } as unknown as Catalog;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("OmniPanel", () => {
  it("renders covers and fetched memory echoes", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        memories: [
          {
            id: "m1",
            text: "recent cue",
            metadata: { timestamp: "2026-02-28T10:00:00Z" },
          },
        ],
      }),
    } as Response);

    render(<OmniPanel catalog={makeCatalog()} />);

    await waitFor(() => {
      expect(screen.getByText("Eta Mu Cover")).toBeTruthy();
      expect(screen.getByText("Memory Fragments / 記憶の断片 (Echoes)")).toBeTruthy();
      expect(screen.getByText("recent cue")).toBeTruthy();
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/memories");
    const image = screen.getByRole("img", { name: "Eta Mu Cover" });
    expect(image.getAttribute("src")).toBe("https://example.test/cover.png");
  });

  it("returns null when catalog is unavailable", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ memories: [] }),
    } as Response);

    const view = render(<OmniPanel catalog={null} />);
    expect(view.container.firstChild).toBeNull();
  });

  it("handles memory fetch errors without crashing", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network offline"));

    render(<OmniPanel catalog={makeCatalog()} />);

    await waitFor(() => {
      expect(screen.getByText("Eta Mu Cover")).toBeTruthy();
    });

    expect(screen.queryByText("Memory Fragments / 記憶の断片 (Echoes)")).toBeNull();
  });
});
