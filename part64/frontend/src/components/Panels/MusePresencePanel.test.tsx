/* @vitest-environment jsdom */

import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MusePresencePanel } from "./MusePresencePanel";
import type { MuseWorkspaceContext, SimulationState } from "../../types";

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
          source_rel_path: "docs/muse-node.md",
          dominant_presence: "anchor_registry",
          dominant_field: "f6",
          importance: 0.95,
          embed_layer_count: 1,
          embed_layer_points: [],
        },
      ],
    },
  } as unknown as SimulationState;
}

function getChatInput(): HTMLTextAreaElement {
  const element = document.getElementById("chat-input");
  if (!(element instanceof HTMLTextAreaElement)) {
    throw new Error("chat input textarea not found");
  }
  return element;
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
      });
    }
    return mockJsonResponse({ ok: true });
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("MusePresencePanel workspace integration", () => {
  it("emits workspace binding callbacks for the fixed muse", async () => {
    const onWorkspaceBindingsChange = vi.fn();
    const onWorkspaceContextChange = vi.fn();
    const workspaceContext: MuseWorkspaceContext = {
      pinnedFileNodeIds: ["node-1"],
      searchQuery: "alpha",
      pinnedNexusSummaries: [],
    };

    render(
      <MusePresencePanel
        museId="anchor_registry"
        onSend={vi.fn()}
        onRecord={vi.fn()}
        onTranscribe={vi.fn()}
        onSendVoice={vi.fn()}
        isRecording={false}
        isThinking={false}
        voiceInputMeta=""
        catalog={null}
        simulation={createSimulationFixture()}
        workspaceContext={workspaceContext}
        onWorkspaceContextChange={onWorkspaceContextChange}
        onWorkspaceBindingsChange={onWorkspaceBindingsChange}
        chatLensState={null}
        activeChatSession={null}
      />,
    );

    await waitFor(() => {
      expect(onWorkspaceBindingsChange).toHaveBeenCalledWith("anchor_registry", ["node-1"]);
      expect(onWorkspaceContextChange).toHaveBeenCalledWith("anchor_registry", {
        pinnedFileNodeIds: ["node-1"],
        searchQuery: "alpha",
        pinnedNexusSummaries: [],
      });
    });
  });

  it("sends muse chat messages with fixed presence and workspace context", async () => {
    const onSend = vi.fn();

    render(
      <MusePresencePanel
        museId="anchor_registry"
        onSend={onSend}
        onRecord={vi.fn()}
        onTranscribe={vi.fn()}
        onSendVoice={vi.fn()}
        isRecording={false}
        isThinking={false}
        voiceInputMeta=""
        catalog={null}
        simulation={createSimulationFixture()}
        workspaceContext={{
          pinnedFileNodeIds: ["node-1"],
          searchQuery: "alpha",
          pinnedNexusSummaries: [],
        }}
        onWorkspaceContextChange={vi.fn()}
        onWorkspaceBindingsChange={vi.fn()}
        chatLensState={null}
        activeChatSession={null}
      />,
    );

    const input = getChatInput();
    fireEvent.change(input, { target: { value: "hello muse" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: false });

    await waitFor(() => {
      expect(onSend).toHaveBeenCalled();
    });

    expect(onSend.mock.calls.at(-1)?.[0]).toBe("hello muse");
    expect(onSend.mock.calls.at(-1)?.[1]).toBe("anchor_registry");
    expect(onSend.mock.calls.at(-1)?.[2]).toMatchObject({
      pinnedFileNodeIds: ["node-1"],
      searchQuery: "alpha",
    });
  });
});
