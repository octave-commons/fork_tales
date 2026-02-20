/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPanel } from "./Chat";
import type { Catalog, MuseWorkspaceContext, SimulationState } from "../../types";

type ChatPanelProps = Parameters<typeof ChatPanel>[0];

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function createSimulationFixture(): SimulationState {
  return {
    file_graph: {
      file_nodes: [
        {
          id: "node-1",
          source_rel_path: "docs/node-1.md",
          dominant_presence: "anchor_registry",
          dominant_field: "f3",
          importance: 0.9,
          embed_layer_count: 1,
          embed_layer_points: [],
        },
        {
          id: "node-2",
          source_rel_path: "docs/node-2.md",
          dominant_presence: "anchor_registry",
          dominant_field: "f4",
          importance: 0.7,
          embed_layer_count: 1,
          embed_layer_points: [],
        },
      ],
    },
  } as unknown as SimulationState;
}

function makeProps(overrides: Partial<ChatPanelProps> = {}): ChatPanelProps {
  const defaultWorkspace: MuseWorkspaceContext = {
    pinnedFileNodeIds: [],
    searchQuery: "",
    pinnedNexusSummaries: [],
  };

  return {
    onSend: vi.fn(),
    onRecord: vi.fn(),
    onTranscribe: vi.fn(),
    onSendVoice: vi.fn(),
    isRecording: false,
    isThinking: false,
    voiceInputMeta: "",
    catalog: null as Catalog | null,
    simulation: createSimulationFixture(),
    fixedMusePresenceId: "anchor_registry",
    workspaceContext: defaultWorkspace,
    onWorkspaceContextChange: vi.fn(),
    onWorkspaceBindingsChange: vi.fn(),
    chatLensState: null,
    activeChatSession: null,
    ...overrides,
  };
}

beforeEach(() => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/api/study")) {
      return mockJsonResponse({
        ok: true,
        generated_at: "2026-02-20T12:00:00.000Z",
        signals: {
          blocked_gate_count: 0,
          active_drift_count: 0,
          queue_pending_count: 0,
        },
      });
    }
    if (url.includes("/api/witness/lineage")) {
      return mockJsonResponse({
        ok: true,
        generated_at: "2026-02-20T12:00:00.000Z",
        checkpoint: {
          branch: "main",
          upstream: "origin/main",
          ahead: 0,
          behind: 0,
        },
        working_tree: {
          dirty: false,
          staged: 0,
          unstaged: 0,
          untracked: 0,
        },
      });
    }
    return mockJsonResponse({ ok: true });
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function getChatInput(): HTMLTextAreaElement {
  const element = document.getElementById("chat-input");
  if (!(element instanceof HTMLTextAreaElement)) {
    throw new Error("chat input textarea not found");
  }
  return element;
}

describe("ChatPanel workspace sync", () => {
  it("normalizes external workspace context and propagates binding callbacks", async () => {
    const onWorkspaceContextChange = vi.fn();
    const onWorkspaceBindingsChange = vi.fn();
    const props = makeProps({
      onWorkspaceContextChange,
      onWorkspaceBindingsChange,
      workspaceContext: {
        pinnedFileNodeIds: [" node-1 ", "node-1", "node-2"],
        searchQuery: "  initial query  ",
        pinnedNexusSummaries: ["ignore-me"],
      },
    });

    const view = render(<ChatPanel {...props} />);

    await waitFor(() => {
      expect(onWorkspaceBindingsChange).toHaveBeenCalled();
      expect(onWorkspaceContextChange).toHaveBeenCalled();
    });

    expect(onWorkspaceBindingsChange.mock.calls.at(-1)).toEqual([
      "anchor_registry",
      ["node-1", "node-2"],
    ]);
    expect(onWorkspaceContextChange.mock.calls.at(-1)).toEqual([
      "anchor_registry",
      {
        pinnedFileNodeIds: ["node-1", "node-2"],
        searchQuery: "  initial query  ",
        pinnedNexusSummaries: [],
      },
    ]);

    view.rerender(
      <ChatPanel
        {...props}
        workspaceContext={{
          pinnedFileNodeIds: [" node-2 ", "node-3", "node-3"],
          searchQuery: "updated",
          pinnedNexusSummaries: ["ignored"],
        }}
      />,
    );

    await waitFor(() => {
      expect(onWorkspaceBindingsChange.mock.calls.at(-1)).toEqual([
        "anchor_registry",
        ["node-2", "node-3"],
      ]);
      expect(onWorkspaceContextChange.mock.calls.at(-1)).toEqual([
        "anchor_registry",
        {
          pinnedFileNodeIds: ["node-2", "node-3"],
          searchQuery: "updated",
          pinnedNexusSummaries: [],
        },
      ]);
    });
  });
});

describe("ChatPanel message routing", () => {
  it("routes ledger messages to /say and keeps llm messages raw", async () => {
    const onSend = vi.fn();
    const props = makeProps({
      onSend,
      workspaceContext: {
        pinnedFileNodeIds: ["node-1"],
        searchQuery: "topic",
        pinnedNexusSummaries: [],
      },
    });

    render(<ChatPanel {...props} />);
    const input = getChatInput();

    fireEvent.change(input, { target: { value: "map drift" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: false });

    await waitFor(() => {
      expect(onSend).toHaveBeenCalled();
    });

    expect(onSend.mock.calls[0]?.[0]).toBe("/say anchor_registry map drift");
    expect(onSend.mock.calls[0]?.[1]).toBe("anchor_registry");
    expect(onSend.mock.calls[0]?.[2]).toMatchObject({
      pinnedFileNodeIds: ["node-1"],
      searchQuery: "topic",
    });

    fireEvent.click(screen.getByRole("button", { name: /llm mode/i }));
    fireEvent.change(input, { target: { value: "direct prompt" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: false });

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledTimes(2);
    });

    expect(onSend.mock.calls[1]?.[0]).toBe("direct prompt");
    expect(onSend.mock.calls[1]?.[1]).toBe("anchor_registry");
  });
});
