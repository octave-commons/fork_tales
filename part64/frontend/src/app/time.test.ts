import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { relativeTime } from "./time";

describe("relativeTime", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-02-20T12:00:00.000Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns n/a for empty values", () => {
    expect(relativeTime(undefined)).toBe("n/a");
    expect(relativeTime("   ")).toBe("n/a");
  });

  it("returns the raw value when parsing fails", () => {
    expect(relativeTime("not-a-date")).toBe("not-a-date");
  });

  it("formats second, minute, hour, and day deltas", () => {
    expect(relativeTime("2026-02-20T11:59:30.000Z")).toBe("30s ago");
    expect(relativeTime("2026-02-20T11:58:00.000Z")).toBe("2m ago");
    expect(relativeTime("2026-02-20T09:00:00.000Z")).toBe("3h ago");
    expect(relativeTime("2026-02-16T12:00:00.000Z")).toBe("4d ago");
  });
});
