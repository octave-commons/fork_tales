import { describe, expect, it } from "vitest";
import {
  normalizeMusePresenceId,
  normalizeMuseWorkspaceContext,
  sameStringArray,
} from "./museWorkspace";

describe("normalizeMusePresenceId", () => {
  it("normalizes casing, whitespace, and hyphen separators", () => {
    expect(normalizeMusePresenceId(" Witness-Thread ")).toBe("witness_thread");
    expect(normalizeMusePresenceId("gates  of---truth")).toBe("gates_of_truth");
  });
});

describe("sameStringArray", () => {
  it("returns true for equal arrays", () => {
    expect(sameStringArray(["a", "b"], ["a", "b"])).toBe(true);
  });

  it("returns false for mismatched order or length", () => {
    expect(sameStringArray(["a", "b"], ["b", "a"])).toBe(false);
    expect(sameStringArray(["a"], ["a", "b"])).toBe(false);
  });
});

describe("normalizeMuseWorkspaceContext", () => {
  it("returns empty defaults for missing input", () => {
    expect(normalizeMuseWorkspaceContext(null)).toEqual({
      pinnedFileNodeIds: [],
      searchQuery: "",
      pinnedNexusSummaries: [],
    });
  });

  it("trims, deduplicates, and preserves search query text", () => {
    expect(
      normalizeMuseWorkspaceContext({
        pinnedFileNodeIds: [" node-a ", "", "node-b", "node-a", " node-c "],
        searchQuery: "  keep spacing  ",
        pinnedNexusSummaries: [" summary-1 ", "", "summary-2", "summary-1"],
      }),
    ).toEqual({
      pinnedFileNodeIds: ["node-a", "node-b", "node-c"],
      searchQuery: "  keep spacing  ",
      pinnedNexusSummaries: ["summary-1", "summary-2"],
    });
  });

  it("respects override limits", () => {
    expect(
      normalizeMuseWorkspaceContext(
        {
          pinnedFileNodeIds: ["a", "b", "c"],
          searchQuery: "abcdefgh",
          pinnedNexusSummaries: ["x", "y", "z"],
        },
        {
          maxPinnedFileNodeIds: 2,
          maxSearchQueryLength: 4,
          maxPinnedNexusSummaries: 1,
        },
      ),
    ).toEqual({
      pinnedFileNodeIds: ["a", "b"],
      searchQuery: "abcd",
      pinnedNexusSummaries: ["x"],
    });
  });
});
