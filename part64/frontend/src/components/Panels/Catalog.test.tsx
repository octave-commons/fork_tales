/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CatalogPanel } from "./Catalog";
import type { Catalog } from "../../types";

function makeCatalog(itemCount = 26): Catalog {
  return {
    generated_at: "2026-02-28T00:00:00Z",
    part_roots: ["part64"],
    counts: {},
    canonical_terms: [],
    cover_fields: [],
    items: Array.from({ length: itemCount }, (_unused, index) => {
      const id = index + 1;
      const kind = id % 4 === 0 ? "file" : id % 3 === 0 ? "video" : id % 2 === 0 ? "audio" : "image";
      return {
        role: "artifact",
        kind,
        name: `item-${id}`,
        rel_path: `artifacts/item-${id}.dat`,
        url: `https://example.test/item-${id}`,
        bytes: 1200 + id,
        part: "64",
        display_name: { en: `Item ${id}`, ja: `項目${id}` },
        display_role: { en: "artifact", ja: "成果物" },
      };
    }),
  } as unknown as Catalog;
}

function makeZipCatalogPayload() {
  return {
    ok: true,
    generated_at: "2026-02-28T00:00:00Z",
    member_limit: 240,
    zip_count: 1,
    zips: [
      {
        id: "zip-1",
        name: "session.zip",
        rel_path: "archives/session.zip",
        url: "https://example.test/session.zip",
        bytes: 2048,
        mtime_utc: "2026-02-28T00:00:00Z",
        members_total: 3,
        files_total: 2,
        dirs_total: 1,
        uncompressed_bytes_total: 4096,
        compressed_bytes_total: 2048,
        compression_ratio: 0.5,
        members_truncated: false,
        type_counts: { image: 1, text: 1, dir: 1 },
        extension_counts: [
          { ext: ".png", count: 1 },
          { ext: ".md", count: 1 },
        ],
        top_level_entries: [{ name: "assets", count: 2 }],
        members: [
          {
            path: "assets/board.png",
            kind: "image",
            ext: ".png",
            depth: 2,
            is_dir: false,
            bytes: 1024,
            compressed_bytes: 512,
            url: "https://example.test/board.png",
          },
          {
            path: "assets/notes.md",
            kind: "text",
            ext: ".md",
            depth: 2,
            is_dir: false,
            bytes: 512,
            compressed_bytes: 256,
            url: "https://example.test/notes.md",
          },
          {
            path: "assets",
            kind: "dir",
            ext: "",
            depth: 1,
            is_dir: true,
            bytes: 0,
            compressed_bytes: 0,
            url: "",
          },
        ],
      },
    ],
  };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("CatalogPanel", () => {
  it("returns null when catalog payload is unavailable", () => {
    const view = render(<CatalogPanel catalog={null} />);
    expect(view.container.firstChild).toBeNull();
  });

  it("renders zip atlas, supports filtering, and expands artifact list", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => makeZipCatalogPayload(),
    } as Response);

    render(<CatalogPanel catalog={makeCatalog()} />);

    await waitFor(() => {
      expect(screen.getByText("Zip Atlas / 圧縮アーカイブ")).toBeTruthy();
      expect(screen.getByText("session.zip")).toBeTruthy();
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/zips?member_limit=240");
    expect(screen.getByText("Item 24")).toBeTruthy();
    expect(screen.queryByText("Item 26")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /Show 2 more/ }));
    expect(screen.getByText("Item 26")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Show less / 折りたたむ" })).toBeTruthy();

    fireEvent.change(screen.getByPlaceholderText("filter by zip name, path, or member"), {
      target: { value: "no-match-value" },
    });
    expect(screen.getByText("No zip archives matched this filter.")).toBeTruthy();
  });
});
