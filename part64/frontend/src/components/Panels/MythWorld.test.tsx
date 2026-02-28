/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { MythWorldPanel } from "./MythWorld";
import type { SimulationState, WorldInteractionResponse } from "../../types";

function makeSimulation(): SimulationState {
  return {
    world: {
      tick: 42,
      prayer_intensity: 0.71,
      people: [
        {
          id: "p-1",
          name: { en: "Aiko", ja: "アイコ" },
          role: { en: "Scribe", ja: "書記" },
          instrument: "koto",
          hymn_bpm: 108,
          prays_to: "witness_thread",
          prayer_intensity: 0.63,
        },
      ],
      songs: [
        {
          id: "song-1",
          title: { en: "Morning Thread", ja: "朝の糸" },
          bpm: 108,
          energy: 0.84,
        },
      ],
      books: [
        {
          id: "book-1",
          title: { en: "First Ledger", ja: "最初の台帳" },
          author: { en: "Keeper", ja: "番人" },
          excerpt: { en: "A witness line begins.", ja: "証人線が始まる。" },
        },
      ],
    },
  } as unknown as SimulationState;
}

function makeInteraction(): WorldInteractionResponse {
  return {
    ok: true,
    speaker: { en: "Aiko", ja: "アイコ" },
    presence: { name: { en: "Witness Thread", ja: "証人の糸" } },
    line_en: "Follow the chain from first signal.",
    line_ja: "最初の信号から連鎖を追って。",
  } as unknown as WorldInteractionResponse;
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("MythWorldPanel", () => {
  it("shows loading-state copy when world state is absent", () => {
    render(
      <MythWorldPanel
        simulation={null}
        interaction={null}
        interactingPersonId={null}
        onInteract={vi.fn()}
      />,
    );

    expect(screen.getByText("World pulse is gathering / 世界の脈動を収集中")).toBeTruthy();
  });

  it("renders world content and routes interactions", () => {
    const onInteract = vi.fn();

    render(
      <MythWorldPanel
        simulation={makeSimulation()}
        interaction={makeInteraction()}
        interactingPersonId={null}
        onInteract={onInteract}
      />,
    );

    expect(screen.getByText("Aiko")).toBeTruthy();
    expect(screen.getByText("koto · 108 BPM")).toBeTruthy();
    expect(screen.getByText("Songbook / 聖歌帳")).toBeTruthy();
    expect(screen.getByText("Library / 文庫")).toBeTruthy();
    expect(screen.getByText("Presence Dialogue / プレゼンス対話")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Speak / 話す" }));
    fireEvent.click(screen.getByRole("button", { name: "Pray / 祈る" }));
    fireEvent.click(screen.getByRole("button", { name: "Sing / 歌う" }));

    expect(onInteract).toHaveBeenNthCalledWith(1, "p-1", "speak");
    expect(onInteract).toHaveBeenNthCalledWith(2, "p-1", "pray");
    expect(onInteract).toHaveBeenNthCalledWith(3, "p-1", "sing");
  });

  it("disables interaction buttons for the active speaker", () => {
    render(
      <MythWorldPanel
        simulation={makeSimulation()}
        interaction={null}
        interactingPersonId="p-1"
        onInteract={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "Speak / 話す" }).getAttribute("disabled")).not.toBeNull();
    expect(screen.getByRole("button", { name: "Pray / 祈る" }).getAttribute("disabled")).not.toBeNull();
    expect(screen.getByRole("button", { name: "Sing / 歌う" }).getAttribute("disabled")).not.toBeNull();
  });
});
