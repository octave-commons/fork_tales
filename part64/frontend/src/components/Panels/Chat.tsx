import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Activity,
  FileAudio,
  GitBranch,
  MessageSquare,
  Mic,
  RefreshCw,
  Send,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import {
  normalizeMusePresenceId,
  normalizeMuseWorkspaceContext,
  sameStringArray,
} from "../../app/museWorkspace";
import { relativeTime } from "../../app/time";
import { runtimeApiUrl } from "../../runtime/endpoints";
import type { AskPayload } from "../../autopilot";
import type {
  BackendFieldParticle,
  Catalog,
  ChatMessage,
  FileGraphNode,
  MuseWorkspaceContext,
  SimulationState,
  StudySnapshotPayload,
  UIProjectionChatSession,
  UIProjectionElementState,
  WorldPresence,
  WitnessLineagePayload,
  WitnessThreadState,
} from "../../types";

interface Props {
  onSend: (text: string, musePresenceId: string, workspace: MuseWorkspaceContext) => void;
  onRecord: () => void;
  onTranscribe: () => void;
  onSendVoice: (musePresenceId: string, workspace: MuseWorkspaceContext) => void;
  isRecording: boolean;
  isThinking: boolean;
  voiceInputMeta: string;
  catalog: Catalog | null;
  simulation: SimulationState | null;
  activeMusePresenceId?: string;
  onMusePresenceChange?: (presenceId: string) => void;
  fixedMusePresenceId?: string;
  workspaceContext?: MuseWorkspaceContext | null;
  onWorkspaceContextChange?: (musePresenceId: string, workspace: MuseWorkspaceContext) => void;
  onWorkspaceBindingsChange?: (musePresenceId: string, pinnedFileNodeIds: string[]) => void;
  chatLensState?: UIProjectionElementState | null;
  activeChatSession?: UIProjectionChatSession | null;
  minimalMuseView?: boolean;
}

type ChatChannel = "ledger" | "llm";

const AUTOPILOT_OPTION_LIMIT = 5;
const RUNTIME_REFRESH_MS = 6500;

const FALLBACK_MUSES: WorldPresence[] = [
  {
    id: "witness_thread",
    name: { en: "Witness Thread", ja: "証人の糸" },
    type: "presence",
  },
  {
    id: "anchor_registry",
    name: { en: "Anchor Registry", ja: "錨台帳" },
    type: "presence",
  },
  {
    id: "gates_of_truth",
    name: { en: "Gates of Truth", ja: "真理の門" },
    type: "presence",
  },
];

interface MusePresenceOption {
  id: string;
  en: string;
  ja: string;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

function formatContextValue(value: unknown): string {
  if (value === null || value === undefined) {
    return String(value);
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return "[unserializable]";
  }
}

function normalizeAskPayload(payload: AskPayload): AskPayload {
  const reason = String(payload.reason || "autopilot is blocked").trim() || "autopilot is blocked";
  const need = String(payload.need || "reply with a decision").trim();
  const options = (payload.options || [])
    .map((option) => option.trim())
    .filter((option, index, all) => option.length > 0 && all.indexOf(option) === index)
    .slice(0, AUTOPILOT_OPTION_LIMIT);

  return {
    reason,
    need,
    options,
    context: payload.context,
    urgency: payload.urgency,
    gate: payload.gate || "unknown",
  };
}

function askSystemMessage(payload: AskPayload): string {
  const gate = payload.gate || "unknown";
  const urgency = payload.urgency ? ` (${payload.urgency})` : "";
  return `autopilot blocked [${gate}]${urgency}\n${payload.reason}\nneed: ${payload.need}`;
}

function toPercentText(raw: number | undefined): string {
  return `${Math.round(clamp01(Number(raw ?? 0)) * 100)}%`;
}

function particleColor(particle: BackendFieldParticle): string {
  const red = Math.round(clamp01(Number(particle.r ?? 0)) * 255);
  const green = Math.round(clamp01(Number(particle.g ?? 0)) * 255);
  const blue = Math.round(clamp01(Number(particle.b ?? 0)) * 255);
  return `rgb(${red} ${green} ${blue})`;
}

function buildMusePresenceOptions(catalog: Catalog | null, simulation: SimulationState | null): MusePresenceOption[] {
  const runtimeMuses = Array.isArray(catalog?.muse_runtime?.muses)
    ? catalog.muse_runtime.muses
    : [];
  if (runtimeMuses.length > 0) {
    return runtimeMuses
      .map((row) => {
        const id = String(row?.id ?? "").trim();
        if (!id) {
          return null;
        }
        const label = String(row?.label ?? id).trim() || id;
        return {
          id,
          en: label,
          ja: "",
        } satisfies MusePresenceOption;
      })
      .filter((row): row is MusePresenceOption => row !== null);
  }

  const fromWorld = simulation?.world?.presences ?? [];
  if (fromWorld.length > 0) {
    const mapped = fromWorld
      .map((row) => {
        const id = String(row?.id ?? "").trim();
        if (!id) {
          return null;
        }
        return {
          id,
          en: String(row?.name?.en ?? id).trim() || id,
          ja: String(row?.name?.ja ?? "").trim(),
        } satisfies MusePresenceOption;
      })
      .filter((row): row is MusePresenceOption => row !== null);
    if (mapped.length > 0) {
      return mapped;
    }
  }

  const manifest = Array.isArray(catalog?.entity_manifest) ? catalog.entity_manifest : [];
  if (manifest.length > 0) {
    const mapped = manifest
      .map((row) => {
        const id = String(row?.id ?? "").trim();
        if (!id) {
          return null;
        }
        return {
          id,
          en: String(row?.en ?? id).trim() || id,
          ja: String(row?.ja ?? "").trim(),
        } satisfies MusePresenceOption;
      })
      .filter((row): row is MusePresenceOption => row !== null);
    if (mapped.length > 0) {
      return mapped;
    }
  }

  return FALLBACK_MUSES.map((row) => ({
    id: row.id,
    en: row.name.en,
    ja: row.name.ja,
  }));
}

function collectNearbyEmbedSummaries(
  catalog: Catalog | null,
  simulation: SimulationState | null,
  musePresenceId: string,
): string[] {
  const normalizedPresence = normalizeMusePresenceId(musePresenceId);
  const graph = simulation?.file_graph ?? catalog?.file_graph;
  const nodes: FileGraphNode[] = Array.isArray(graph?.file_nodes) ? graph.file_nodes : [];
  if (nodes.length === 0) {
    return [];
  }

  const rows = nodes
    .filter((node) => {
      const dominantPresence = normalizeMusePresenceId(String(node?.dominant_presence ?? ""));
      const conceptPresence = normalizeMusePresenceId(String(node?.concept_presence_id ?? ""));
      return dominantPresence === normalizedPresence || conceptPresence === normalizedPresence;
    })
    .sort((left, right) => {
      const rightWeight = Number(right?.importance ?? 0) + Number(right?.embed_layer_count ?? 0) * 0.15;
      const leftWeight = Number(left?.importance ?? 0) + Number(left?.embed_layer_count ?? 0) * 0.15;
      return rightWeight - leftWeight;
    })
    .slice(0, 6)
    .map((node) => {
      const label = String(node.source_rel_path ?? node.label ?? node.id ?? "item").trim();
      const field = String(node.dominant_field ?? "f6").trim();
      const layerCount = Number(node.embed_layer_count ?? 0);
      const embedIds = (Array.isArray(node.embed_layer_points) ? node.embed_layer_points : [])
        .flatMap((layer) => (Array.isArray(layer.embed_ids) ? layer.embed_ids : []))
        .map((item) => String(item || "").trim())
        .filter((item, index, all) => item.length > 0 && all.indexOf(item) === index)
        .slice(0, 2);
      const embedText = embedIds.length > 0 ? embedIds.join(",") : "none";
      return `${label} | ${field} | layers=${layerCount} | embeds=${embedText}`;
    });

  return rows;
}

function messageTone(message: ChatMessage): string {
  if (message.role === "user") {
    return "bg-[rgba(102,217,239,0.16)] border-[rgba(102,217,239,0.38)]";
  }
  if (message.role === "assistant") {
    return "bg-[rgba(166,226,46,0.12)] border-[rgba(166,226,46,0.34)]";
  }
  return "bg-[rgba(249,38,114,0.08)] border-[rgba(249,38,114,0.28)]";
}

export function ChatPanel({
  onSend,
  onRecord,
  onTranscribe,
  onSendVoice,
  isRecording,
  isThinking,
  voiceInputMeta,
  catalog,
  simulation,
  activeMusePresenceId,
  onMusePresenceChange,
  fixedMusePresenceId,
  workspaceContext,
  onWorkspaceContextChange,
  onWorkspaceBindingsChange,
  chatLensState,
  activeChatSession,
  minimalMuseView = false,
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [pendingAsk, setPendingAsk] = useState<AskPayload | null>(null);
  const [chatChannel, setChatChannel] = useState<ChatChannel>("ledger");
  const [searchQuery, setSearchQuery] = useState(() => normalizeMuseWorkspaceContext(workspaceContext).searchQuery);
  const [pinnedFileNodeIds, setPinnedFileNodeIds] = useState<string[]>(
    () => normalizeMuseWorkspaceContext(workspaceContext).pinnedFileNodeIds,
  );
  const [studySnapshot, setStudySnapshot] = useState<StudySnapshotPayload | null>(null);
  const [lineageSnapshot, setLineageSnapshot] = useState<WitnessLineagePayload | null>(null);
  const [runtimeError, setRuntimeError] = useState("");
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const selectedMuseSeed = fixedMusePresenceId ?? activeMusePresenceId ?? "witness_thread";

  const musePresenceOptions = useMemo(
    () => buildMusePresenceOptions(catalog, simulation),
    [catalog, simulation],
  );

  const resolvedMusePresenceId = useMemo(() => {
    if (fixedMusePresenceId) {
      return fixedMusePresenceId;
    }
    const normalizedSelected = normalizeMusePresenceId(selectedMuseSeed);
    const selected = musePresenceOptions.find(
      (row) => normalizeMusePresenceId(row.id) === normalizedSelected,
    );
    if (selected) {
      return selected.id;
    }
    return musePresenceOptions[0]?.id ?? "witness_thread";
  }, [fixedMusePresenceId, musePresenceOptions, selectedMuseSeed]);

  const activeMuse = useMemo(
    () =>
      musePresenceOptions.find(
        (row) => normalizeMusePresenceId(row.id) === normalizeMusePresenceId(resolvedMusePresenceId),
      )
      ?? null,
    [musePresenceOptions, resolvedMusePresenceId],
  );

  const nearbyEmbedSummaries = useMemo(
    () => collectNearbyEmbedSummaries(catalog, simulation, resolvedMusePresenceId),
    [catalog, resolvedMusePresenceId, simulation],
  );

  const fileNodes = useMemo(() => {
    const graph = simulation?.file_graph ?? catalog?.file_graph;
    const rows = Array.isArray(graph?.file_nodes) ? graph.file_nodes : [];
    return rows.filter((row): row is FileGraphNode => Boolean(row));
  }, [catalog?.file_graph, simulation?.file_graph]);

  const fileNodeById = useMemo(() => {
    const map = new Map<string, FileGraphNode>();
    fileNodes.forEach((row) => {
      const id = String(row.id ?? "").trim();
      if (id && !map.has(id)) {
        map.set(id, row);
      }
      const nodeId = String(row.node_id ?? "").trim();
      if (nodeId && !map.has(nodeId)) {
        map.set(nodeId, row);
      }
    });
    return map;
  }, [fileNodes]);

  const pinnedNexusSummaries = useMemo(() => {
    return pinnedFileNodeIds
      .map((nodeId) => fileNodeById.get(nodeId))
      .filter((row): row is FileGraphNode => Boolean(row))
      .map((row) => {
        const label = String(row.source_rel_path ?? row.label ?? row.id ?? "nexus").trim();
        const field = String(row.dominant_field ?? "f6").trim();
        const presence = String(row.dominant_presence ?? "").trim() || "nexus";
        return `${label} | field=${field} | presence=${presence}`;
      });
  }, [fileNodeById, pinnedFileNodeIds]);

  const filteredSearchNexus = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const normalizedMuseId = normalizeMusePresenceId(resolvedMusePresenceId);
    const rows = fileNodes
      .map((row) => {
        const searchable = [
          String(row.source_rel_path ?? ""),
          String(row.label ?? ""),
          String(row.name ?? ""),
          String(row.summary ?? ""),
          String(row.text_excerpt ?? ""),
          String(row.dominant_field ?? ""),
          String(row.dominant_presence ?? ""),
          ...(Array.isArray(row.tags) ? row.tags.map((item) => String(item)) : []),
          ...(Array.isArray(row.labels) ? row.labels.map((item) => String(item)) : []),
        ].join(" ").toLowerCase();

        const matches = query.length <= 0 || searchable.includes(query);
        const museAffinity =
          normalizeMusePresenceId(String(row.dominant_presence ?? "")) === normalizedMuseId
            ? 1
            : 0;
        const weight = (Number(row.importance ?? 0) * 0.75) + (museAffinity * 0.9);
        return { row, matches, weight };
      })
      .filter((item) => item.matches)
      .sort((left, right) => right.weight - left.weight)
      .slice(0, 14)
      .map((item) => item.row);
    return rows;
  }, [fileNodes, resolvedMusePresenceId, searchQuery]);

  const nearbyNexusRows = useMemo(() => {
    const normalizedMuseId = normalizeMusePresenceId(resolvedMusePresenceId);
    const scored = fileNodes
      .map((row) => {
        const dominantPresence = normalizeMusePresenceId(String(row.dominant_presence ?? ""));
        const conceptPresence = normalizeMusePresenceId(String(row.concept_presence_id ?? ""));
        const affinity = dominantPresence === normalizedMuseId || conceptPresence === normalizedMuseId ? 1 : 0;
        const score = (Number(row.importance ?? 0) * 0.78) + (Number(row.embed_layer_count ?? 0) * 0.09) + (affinity * 0.92);
        return { row, affinity, score };
      })
      .sort((left, right) => right.score - left.score);

    const localRows = scored
      .filter((item) => item.affinity > 0)
      .slice(0, 16)
      .map((item) => item.row);

    if (localRows.length > 0) {
      return localRows;
    }

    return scored.slice(0, 16).map((item) => item.row);
  }, [fileNodes, resolvedMusePresenceId]);

  const liveWorkspaceContext = useMemo<MuseWorkspaceContext>(
    () => ({
      pinnedFileNodeIds,
      searchQuery,
      pinnedNexusSummaries,
    }),
    [pinnedFileNodeIds, pinnedNexusSummaries, searchQuery],
  );

  const workspaceSyncContext = useMemo<MuseWorkspaceContext>(
    () => ({
      pinnedFileNodeIds,
      searchQuery,
      pinnedNexusSummaries: [],
    }),
    [pinnedFileNodeIds, searchQuery],
  );

  const normalizedExternalWorkspace = useMemo(
    () => normalizeMuseWorkspaceContext(workspaceContext),
    [workspaceContext],
  );

  useEffect(() => {
    if (!sameStringArray(pinnedFileNodeIds, normalizedExternalWorkspace.pinnedFileNodeIds)) {
      setPinnedFileNodeIds(normalizedExternalWorkspace.pinnedFileNodeIds);
    }
    if (searchQuery !== normalizedExternalWorkspace.searchQuery) {
      setSearchQuery(normalizedExternalWorkspace.searchQuery);
    }
  }, [normalizedExternalWorkspace, pinnedFileNodeIds, searchQuery]);

  useEffect(() => {
    const normalizedCurrent = normalizeMusePresenceId(selectedMuseSeed);
    const normalizedResolved = normalizeMusePresenceId(resolvedMusePresenceId);
    if (!fixedMusePresenceId && onMusePresenceChange && normalizedCurrent !== normalizedResolved) {
      onMusePresenceChange(resolvedMusePresenceId);
    }
  }, [fixedMusePresenceId, onMusePresenceChange, resolvedMusePresenceId, selectedMuseSeed]);

  useEffect(() => {
    onWorkspaceBindingsChange?.(resolvedMusePresenceId, pinnedFileNodeIds);
  }, [onWorkspaceBindingsChange, pinnedFileNodeIds, resolvedMusePresenceId]);

  useEffect(() => {
    onWorkspaceContextChange?.(resolvedMusePresenceId, workspaceSyncContext);
  }, [onWorkspaceContextChange, resolvedMusePresenceId, workspaceSyncContext]);

  useEffect(() => {
    if (minimalMuseView && chatChannel !== "llm") {
      setChatChannel("llm");
    }
  }, [chatChannel, minimalMuseView]);

  const activeMuseLabel = activeMuse
    ? `${activeMuse.en}${activeMuse.ja ? ` / ${activeMuse.ja}` : ""}`
    : resolvedMusePresenceId;

  const witnessState: WitnessThreadState | null =
    simulation?.presence_dynamics?.witness_thread ?? null;

  const witnessParticles = useMemo(() => {
    const rows = simulation?.presence_dynamics?.field_particles ?? simulation?.field_particles ?? [];
    return rows.filter(
      (row): row is BackendFieldParticle =>
        Boolean(row)
        && normalizeMusePresenceId(String(row.presence_id || "")) === normalizeMusePresenceId(resolvedMusePresenceId),
    );
  }, [resolvedMusePresenceId, simulation?.field_particles, simulation?.presence_dynamics?.field_particles]);

  const particleSamples = useMemo(() => {
    return [...witnessParticles]
      .sort((left, right) => Number(right.size ?? 0) - Number(left.size ?? 0))
      .slice(0, 6);
  }, [witnessParticles]);

  const refreshRuntimeLedger = useCallback(async (withSpinner = true) => {
    if (withSpinner) {
      setRuntimeLoading(true);
    }

    const errors: string[] = [];

    try {
      const studyResponse = await fetch(runtimeApiUrl("/api/study?limit=4"));
      if (!studyResponse.ok) {
        errors.push(`study(${studyResponse.status})`);
      } else {
        const payload = (await studyResponse.json()) as StudySnapshotPayload;
        if (payload?.ok) {
          setStudySnapshot(payload);
        } else {
          errors.push("study(invalid)");
        }
      }
    } catch {
      errors.push("study(unreachable)");
    }

    try {
      const lineageResponse = await fetch(runtimeApiUrl("/api/witness/lineage"));
      if (!lineageResponse.ok) {
        errors.push(`lineage(${lineageResponse.status})`);
      } else {
        const payload = (await lineageResponse.json()) as WitnessLineagePayload;
        if (payload?.ok) {
          setLineageSnapshot(payload);
        } else {
          errors.push("lineage(invalid)");
        }
      }
    } catch {
      errors.push("lineage(unreachable)");
    }

    setRuntimeError(errors.join(" | "));
    if (withSpinner) {
      setRuntimeLoading(false);
    }
  }, []);

  const sendUserMessage = useCallback(
    (text: string): boolean => {
      const trimmed = text.trim();
      if (!trimmed || isThinking) {
        return false;
      }

      const effectiveChannel: ChatChannel = minimalMuseView ? "llm" : chatChannel;
      const routedPresenceId = resolvedMusePresenceId || "witness_thread";
      const witnessFallbackRoute = `/say witness_thread ${trimmed}`;
      const routedLedgerCommand =
        routedPresenceId === "witness_thread"
          ? witnessFallbackRoute
          : `/say ${routedPresenceId} ${trimmed}`;

      const outbound =
        effectiveChannel === "ledger" && !trimmed.startsWith("/")
          ? routedLedgerCommand
          : trimmed;
      onSend(outbound, routedPresenceId, liveWorkspaceContext);

      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          text: trimmed,
          meta: {
            channel: effectiveChannel,
            source: outbound === trimmed ? "raw" : `auto:/say ${routedPresenceId}`,
            presenceId: routedPresenceId,
            presenceName: activeMuse?.en,
          },
        },
      ]);
      setInput("");
      setPendingAsk(null);
      return true;
    },
    [activeMuse?.en, chatChannel, isThinking, liveWorkspaceContext, minimalMuseView, onSend, resolvedMusePresenceId],
  );

  const handleSend = () => {
    sendUserMessage(input);
  };

  const handleAskOption = (option: string) => {
    if (sendUserMessage(option)) {
      return;
    }

    setInput(option);
    window.requestAnimationFrame(() => {
      if (!inputRef.current) {
        return;
      }
      inputRef.current.focus();
      const cursor = option.length;
      inputRef.current.setSelectionRange(cursor, cursor);
    });
  };

  useEffect(() => {
    if (messages.length >= 0 && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    void refreshRuntimeLedger(true);
    const interval = window.setInterval(() => {
      void refreshRuntimeLedger(false);
    }, RUNTIME_REFRESH_MS);
    return () => {
      window.clearInterval(interval);
    };
  }, [refreshRuntimeLedger]);

  useEffect(() => {
    const handler: EventListener = (event) => {
      const customEvent = event as CustomEvent<ChatMessage>;
      if (!customEvent.detail) {
        return;
      }
      const incomingPresenceId = normalizeMusePresenceId(String(customEvent.detail.meta?.presenceId ?? ""));
      const currentPresenceId = normalizeMusePresenceId(resolvedMusePresenceId);
      if (incomingPresenceId && incomingPresenceId !== currentPresenceId) {
        return;
      }
      if (!incomingPresenceId && currentPresenceId !== "witness_thread") {
        return;
      }
      setMessages((prev) => [...prev, customEvent.detail]);
    };
    window.addEventListener("chat-message", handler);
    return () => window.removeEventListener("chat-message", handler);
  }, [resolvedMusePresenceId]);

  useEffect(() => {
    const handler: EventListener = (event) => {
      const customEvent = event as CustomEvent<AskPayload>;
      if (!customEvent.detail) {
        return;
      }
      const normalized = normalizeAskPayload(customEvent.detail);
      if (!normalized.need) {
        return;
      }

      setPendingAsk(normalized);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: askSystemMessage(normalized),
          meta: {
            channel: "command",
            source: "autopilot:ask",
            presenceId: resolvedMusePresenceId,
            presenceName: activeMuse?.en,
          },
        },
      ]);

      window.requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    };

    window.addEventListener("autopilot:ask", handler);
    return () => window.removeEventListener("autopilot:ask", handler);
  }, [activeMuse?.en, resolvedMusePresenceId]);

  if (minimalMuseView) {
    return (
      <div
        id="chat-panel"
        className="space-y-3 rounded-xl border border-[rgba(102,217,239,0.36)] bg-[linear-gradient(165deg,rgba(10,16,24,0.94),rgba(14,12,21,0.9))] p-3"
      >
        <div className="rounded-md border border-[rgba(102,217,239,0.24)] bg-[rgba(10,19,28,0.62)] px-3 py-2">
          <p className="text-sm font-semibold text-ink">{activeMuseLabel}</p>
          <p className="text-[11px] text-[#b7d8ee] mt-1">
            Pinned nexus in this muse panel are always included in generation context.
          </p>
        </div>

        <div className="rounded-md border border-[rgba(102,217,239,0.24)] bg-[rgba(11,20,29,0.64)] p-3">
          <p className="text-xs font-semibold text-[#d4ecff]">Nearby Nexus</p>
          <div className="mt-2 grid gap-1.5 max-h-[190px] overflow-auto pr-1">
            {nearbyNexusRows.length <= 0 ? (
              <p className="text-xs text-muted">No nearby nexus rows for this muse.</p>
            ) : (
              nearbyNexusRows.map((row) => {
                const nodeId = String(row.id ?? row.node_id ?? "").trim();
                if (!nodeId) {
                  return null;
                }
                const label = String(row.source_rel_path ?? row.label ?? nodeId).trim() || nodeId;
                const isPinned = pinnedFileNodeIds.includes(nodeId);
                return (
                  <div
                    key={nodeId}
                    className="flex items-center justify-between gap-2 rounded-md border border-[rgba(102,217,239,0.18)] bg-[rgba(14,24,36,0.56)] px-2 py-1"
                  >
                    <p className="text-[11px] text-[#d4e9f8] break-all">{label}</p>
                    <button
                      type="button"
                      onClick={() => {
                        setPinnedFileNodeIds((prev) => {
                          if (prev.includes(nodeId)) {
                            return prev.filter((id) => id !== nodeId);
                          }
                          return [...prev, nodeId].slice(-24);
                        });
                      }}
                      className={`rounded px-2 py-0.5 text-[10px] ${isPinned
                        ? "border border-[rgba(249,38,114,0.45)] bg-[rgba(249,38,114,0.16)] text-[#ffd0e3]"
                        : "border border-[rgba(102,217,239,0.45)] bg-[rgba(102,217,239,0.14)] text-[#d8f2ff]"}`}
                    >
                      {isPinned ? "unpin" : "pin"}
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>

        <div className="rounded-md border border-[rgba(166,226,46,0.3)] bg-[rgba(13,24,18,0.64)] p-3">
          <p className="text-xs font-semibold text-[#def5d2]">Pinned Nexus</p>
          {pinnedFileNodeIds.length <= 0 ? (
            <p className="text-xs text-muted mt-2">Nothing pinned yet.</p>
          ) : (
            <div className="mt-2 flex flex-wrap gap-2">
              {pinnedFileNodeIds.map((nodeId) => {
                const node = fileNodeById.get(nodeId);
                const label = String(node?.source_rel_path ?? node?.label ?? nodeId).trim() || nodeId;
                return (
                  <button
                    key={nodeId}
                    type="button"
                    onClick={() => {
                      setPinnedFileNodeIds((prev) => prev.filter((id) => id !== nodeId));
                    }}
                    className="rounded-md border border-[rgba(166,226,46,0.38)] bg-[rgba(32,47,21,0.74)] px-2 py-1 text-[11px] text-[#ddf6c9]"
                    title="unpin nexus"
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="rounded-lg border border-[var(--line)] bg-[rgba(16,20,20,0.74)] p-3">
          <p className="text-sm font-semibold text-ink">Muse Chat</p>
          <p className="text-[11px] text-muted mt-1">Messages route to `/api/muse/message` with pinned nexus context.</p>
          <div
            ref={scrollRef}
            className="mt-2 border border-[var(--line)] rounded-lg bg-[rgba(31,32,29,0.86)] p-2 min-h-[130px] max-h-[260px] overflow-auto grid gap-2"
          >
            {messages.map((msg, index) => {
              const chips: string[] = [];
              if (msg.meta?.channel) {
                chips.push(msg.meta.channel);
              }
              if (msg.meta?.model) {
                chips.push(`model=${msg.meta.model}`);
              }
              if (msg.meta?.fallback) {
                chips.push("fallback");
              }
              if (msg.meta?.source) {
                chips.push(msg.meta.source);
              }

              return (
                <article
                  key={`${index}-${msg.role}`}
                  className={`border rounded-lg p-2 text-sm whitespace-pre-wrap leading-relaxed ${messageTone(msg)}`}
                >
                  <span className="font-bold text-xs opacity-75 block mb-1">
                    {msg.role === "user"
                      ? "operator / 操作者"
                      : msg.role === "assistant"
                        ? `${msg.meta?.presenceName || msg.meta?.presenceId || activeMuse?.en || "muse"} / ミューズ`
                        : "system"}
                  </span>
                  {msg.text}
                  {chips.length > 0 ? (
                    <p className="mt-1 text-[10px] text-muted font-mono">{chips.join(" | ")}</p>
                  ) : null}
                </article>
              );
            })}
            {isThinking ? (
              <div className="border border-dashed border-[rgba(102,217,239,0.44)] rounded-lg p-2 text-sm bg-[rgba(102,217,239,0.12)] animate-pulse text-[#66d9ef]">
                <Sparkles size={14} className="inline mr-1" />
                {activeMuse?.en || "muse"} is synthesizing from simulation state...
              </div>
            ) : null}
          </div>

          <div className="grid grid-cols-[1fr_auto] gap-2 mt-2">
            <textarea
              id="chat-input"
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={isThinking ? "Thinking... / 思考中..." : "Ask this muse..."}
              disabled={isThinking}
              className="w-full min-h-[74px] max-h-[180px] resize-y border border-[var(--line)] rounded-lg p-2 font-inherit bg-[rgba(39,40,34,0.9)] text-ink disabled:opacity-50"
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleSend();
                }
              }}
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={isThinking}
              className="self-end border border-[var(--line)] rounded-lg px-3 py-2 bg-[rgba(102,217,239,0.2)] hover:bg-[rgba(102,217,239,0.28)] transition-colors disabled:opacity-50"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
    );
  }

  const isWitnessMuse = normalizeMusePresenceId(resolvedMusePresenceId) === "witness_thread";
  const selectedImpact = (simulation?.presence_dynamics?.presence_impacts ?? []).find(
    (row) => normalizeMusePresenceId(String(row?.id ?? "")) === normalizeMusePresenceId(resolvedMusePresenceId),
  );

  const continuityPercent = isWitnessMuse
    ? toPercentText(witnessState?.continuity_index)
    : toPercentText(Number(selectedImpact?.affects?.ledger ?? 0));
  const clickPercent = isWitnessMuse
    ? toPercentText(witnessState?.click_pressure)
    : toPercentText(Number(selectedImpact?.affected_by?.clicks ?? 0));
  const filePercent = isWitnessMuse
    ? toPercentText(witnessState?.file_pressure)
    : toPercentText(Number(selectedImpact?.affected_by?.files ?? 0));
  const dominantField = String(chatLensState?.explain?.dominant_field || "(none)");
  const linkedPresenceText = isWitnessMuse
    ? (witnessState?.linked_presences ?? []).join(" · ") || "(none)"
    : selectedImpact?.notes_en || "(no linked presences reported for this muse)";
  const lineageRows = isWitnessMuse ? witnessState?.lineage ?? [] : [];
  const runtimeSignals = studySnapshot?.signals;
  const checkpoint = lineageSnapshot?.checkpoint;
  const treeState = lineageSnapshot?.working_tree;
  const ledgerTitle = isWitnessMuse
    ? "Witness Thread Ledger / 証人の糸 台帳"
    : "Muse Workspace Ledger / ミューズ作業台帳";

  return (
    <div
      id="chat-panel"
      className="space-y-3 rounded-xl border border-[rgba(102,217,239,0.36)] bg-[linear-gradient(165deg,rgba(12,18,26,0.94),rgba(18,14,22,0.9))] p-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-ink">{ledgerTitle}</p>
          <p className="text-xs text-muted mt-1">
            Muse channel is the single conversation lane for operator input, system needs, and state translation.
          </p>
          <p className="text-[11px] text-[#b8d9ef] mt-1">
            active muse <code>{activeMuseLabel}</code>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {fixedMusePresenceId ? (
            <div className="rounded-md border border-[rgba(102,217,239,0.3)] bg-[rgba(14,23,33,0.86)] px-3 py-2">
              <p className="text-[10px] uppercase tracking-[0.12em] text-[#a8c6db]">Muse Workspace</p>
              <p className="text-xs text-[#d9edff] mt-1">{activeMuseLabel}</p>
            </div>
          ) : (
            <label className="grid gap-1 text-[10px] uppercase tracking-[0.12em] text-[#a8c6db]">
              Muse Presence
              <select
                value={resolvedMusePresenceId}
                onChange={(event) => onMusePresenceChange?.(event.target.value)}
                className="min-w-[220px] rounded-md border border-[rgba(102,217,239,0.34)] bg-[rgba(14,23,33,0.9)] px-2 py-1.5 text-xs text-[#e5f3ff]"
              >
                {musePresenceOptions.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.en}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button
            type="button"
            onClick={() => {
              void refreshRuntimeLedger(true);
            }}
            className="border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(55,56,48,0.92)]"
          >
            <span className="inline-flex items-center gap-1.5">
              <RefreshCw size={14} className={runtimeLoading ? "animate-spin" : ""} />
              Refresh Ledger
            </span>
          </button>
        </div>
      </div>

      <div className="rounded-lg border border-[rgba(102,217,239,0.26)] bg-[rgba(8,18,28,0.64)] p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs font-semibold text-[#cfe9ff]">Muse Workspace Nexus Binds / 結び目</p>
          <p className="text-[11px] text-[#9dc6dd]">Pinned nexus are used as active conversational context.</p>
        </div>
        <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto] md:items-start">
          <input
            type="text"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="search files/resources to pin into this muse workspace"
            className="w-full rounded-md border border-[rgba(102,217,239,0.28)] bg-[rgba(13,24,36,0.88)] px-3 py-2 text-xs text-[#e5f3ff]"
          />
          <p className="text-[11px] text-[#9dc6dd]">
            pinned <code>{pinnedFileNodeIds.length}</code>
          </p>
        </div>

        {pinnedFileNodeIds.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-2">
            {pinnedFileNodeIds.map((nodeId) => {
              const node = fileNodeById.get(nodeId);
              const label = String(node?.source_rel_path ?? node?.label ?? nodeId).trim() || nodeId;
              return (
                <button
                  key={nodeId}
                  type="button"
                  onClick={() => {
                    setPinnedFileNodeIds((prev) => prev.filter((id) => id !== nodeId));
                  }}
                  className="rounded-md border border-[rgba(166,226,46,0.38)] bg-[rgba(32,47,21,0.74)] px-2 py-1 text-[11px] text-[#ddf6c9]"
                  title="unpin nexus"
                >
                  {label}
                </button>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-muted mt-2">No nexus pinned yet for this muse workspace.</p>
        )}

        <div className="mt-2 grid gap-1.5 max-h-[168px] overflow-auto pr-1">
          {filteredSearchNexus.length <= 0 ? (
            <p className="text-xs text-muted">No matching nexus rows.</p>
          ) : (
            filteredSearchNexus.map((row) => {
              const nodeId = String(row.id ?? row.node_id ?? "").trim();
              const label = String(row.source_rel_path ?? row.label ?? nodeId).trim() || nodeId;
              const isPinned = pinnedFileNodeIds.includes(nodeId);
              return (
                <div
                  key={nodeId}
                  className="flex items-center justify-between gap-2 rounded-md border border-[rgba(102,217,239,0.18)] bg-[rgba(14,24,36,0.52)] px-2 py-1"
                >
                  <p className="text-[11px] text-[#d4e9f8] break-all">{label}</p>
                  <button
                    type="button"
                    onClick={() => {
                      setPinnedFileNodeIds((prev) => {
                        if (prev.includes(nodeId)) {
                          return prev.filter((id) => id !== nodeId);
                        }
                        return [...prev, nodeId].slice(-24);
                      });
                    }}
                    className={`rounded px-2 py-0.5 text-[10px] ${isPinned
                      ? "border border-[rgba(249,38,114,0.45)] bg-[rgba(249,38,114,0.16)] text-[#ffd0e3]"
                      : "border border-[rgba(102,217,239,0.45)] bg-[rgba(102,217,239,0.14)] text-[#d8f2ff]"}`}
                  >
                    {isPinned ? "unpin" : "pin"}
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-md border border-[rgba(102,217,239,0.3)] bg-[rgba(13,22,34,0.7)] px-3 py-2">
          <p className="text-[10px] uppercase tracking-wide text-[#98c7e8]">continuity index</p>
          <p className="text-sm font-semibold text-[#d9f0ff]">{continuityPercent}</p>
          <div className="mt-1 h-1.5 overflow-hidden rounded bg-[rgba(102,217,239,0.18)]">
            <div
              className="h-full bg-[linear-gradient(90deg,rgba(102,217,239,0.92),rgba(166,226,46,0.86))]"
              style={{ width: continuityPercent }}
            />
          </div>
        </div>
        <div className="rounded-md border border-[rgba(174,129,255,0.28)] bg-[rgba(28,18,38,0.64)] px-3 py-2">
          <p className="text-[10px] uppercase tracking-wide text-[#c3aae9]">click pressure</p>
          <p className="text-sm font-semibold text-[#efe3ff]">{clickPercent}</p>
          <div className="mt-1 h-1.5 overflow-hidden rounded bg-[rgba(174,129,255,0.2)]">
            <div
              className="h-full bg-[linear-gradient(90deg,rgba(174,129,255,0.9),rgba(102,217,239,0.82))]"
              style={{ width: clickPercent }}
            />
          </div>
        </div>
        <div className="rounded-md border border-[rgba(253,151,31,0.32)] bg-[rgba(40,26,13,0.64)] px-3 py-2">
          <p className="text-[10px] uppercase tracking-wide text-[#f3c596]">file pressure</p>
          <p className="text-sm font-semibold text-[#ffe7ce]">{filePercent}</p>
          <div className="mt-1 h-1.5 overflow-hidden rounded bg-[rgba(253,151,31,0.2)]">
            <div
              className="h-full bg-[linear-gradient(90deg,rgba(253,151,31,0.94),rgba(249,38,114,0.84))]"
              style={{ width: filePercent }}
            />
          </div>
        </div>
        <div className="rounded-md border border-[rgba(166,226,46,0.3)] bg-[rgba(20,32,13,0.66)] px-3 py-2">
          <p className="text-[10px] uppercase tracking-wide text-[#bfe8a7]">muse particles</p>
          <p className="text-sm font-semibold text-[#e5ffdb]">{witnessParticles.length}</p>
          <p className="text-[11px] text-[#d2efc0] mt-1">
            linked {isWitnessMuse ? witnessState?.linked_presences?.length ?? 0 : "n/a"}
          </p>
        </div>
      </div>

      <div className="rounded-lg border border-[rgba(126,166,192,0.32)] bg-[rgba(12,18,26,0.68)] p-3">
        <p className="text-xs font-semibold text-[#d5ecff]">Lineage Ledger / 来歴</p>
        <p className="text-[11px] text-[#9ec7dd] mt-1 break-words">linked presences: {linkedPresenceText}</p>
        <div className="mt-2 grid gap-1.5">
          {lineageRows.length <= 0 ? (
            <p className="text-xs text-muted">
              {isWitnessMuse
                ? "No lineage rows yet; waiting for witness touch or file drift."
                : "Lineage ledger is currently witness-thread specific; this muse relies on nearby embedding context."}
            </p>
          ) : (
            lineageRows.slice(0, 8).map((entry, index) => (
              <article
                key={`${entry.kind}-${entry.ref}-${index}`}
                className="rounded-md border border-[rgba(102,217,239,0.2)] bg-[rgba(14,24,36,0.66)] px-2.5 py-2"
              >
                <p className="text-[11px] font-mono text-[#cae8ff]">
                  <span className="text-[#66d9ef]">{entry.kind}</span>
                  {" | "}
                  <span className="break-all">{entry.ref}</span>
                </p>
                <p className="text-[11px] text-[#9ec7dd] mt-1">{entry.why_en}</p>
                <p className="text-[11px] text-[#8cb6d0]">{entry.why_ja}</p>
              </article>
            ))
          )}
        </div>
        {witnessState?.notes_en ? (
          <p className="text-[11px] text-[#b8d9ef] mt-2">note: {witnessState.notes_en}</p>
        ) : null}
      </div>

      <div className="rounded-lg border border-[rgba(166,226,46,0.3)] bg-[rgba(11,22,16,0.68)] p-3">
        <p className="text-xs font-semibold text-[#dff4d5]">Particles Made Clear / 粒子明瞭化</p>
        <p className="text-[11px] text-[#b4d6bb] mt-1">
          Active muse particles are shown as explicit slices with deterministic order.
        </p>
        {witnessParticles.length <= 0 ? (
          <p className="text-xs text-muted mt-2">No active muse particles in this frame.</p>
        ) : (
          <>
            <div className="mt-2 grid grid-cols-12 gap-1 sm:grid-cols-16">
              {witnessParticles.slice(0, 64).map((particle, index) => {
                const width = `${Math.max(10, Math.min(100, Number(particle.size ?? 0.01) * 2100))}%`;
                return (
                  <div
                    key={`${particle.id}-${index}`}
                    className="h-3 rounded-sm border border-[rgba(248,248,242,0.16)]"
                    style={{
                      backgroundColor: particleColor(particle),
                      opacity: 0.45 + (clamp01(Number(particle.size ?? 0.01) * 12) * 0.5),
                      width,
                    }}
                    title={`x=${particle.x.toFixed(3)} y=${particle.y.toFixed(3)} size=${Number(particle.size ?? 0).toFixed(4)}`}
                  />
                );
              })}
            </div>
            <p className="mt-2 text-[11px] font-mono text-[#d4f2cd] break-words">
              top samples:{" "}
              {particleSamples
                .map((row) => `(${row.x.toFixed(3)},${row.y.toFixed(3)} s=${Number(row.size ?? 0).toFixed(4)})`)
                .join(" | ")}
            </p>
          </>
        )}
      </div>

      <div className="rounded-lg border border-[rgba(249,38,114,0.28)] bg-[rgba(28,13,22,0.62)] p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs font-semibold text-[#ffd7e8]">Checkpoint Cycle / チェックポイント</p>
          {lineageSnapshot?.generated_at ? (
            <p className="text-[11px] text-[#f5bcd4]">updated {relativeTime(lineageSnapshot.generated_at)}</p>
          ) : null}
        </div>
        <p className="text-[11px] text-[#f2c0d9] mt-1">
          <GitBranch size={12} className="inline mr-1" />
          branch <code>{checkpoint?.branch || "(unknown)"}</code> | upstream <code>{checkpoint?.upstream || "(missing)"}</code> | ahead <code>{String(checkpoint?.ahead ?? 0)}</code> | behind <code>{String(checkpoint?.behind ?? 0)}</code>
        </p>
        <p className="text-[11px] text-[#f2c0d9] mt-1">
          tree dirty=<code>{String(treeState?.dirty ?? false)}</code> staged=<code>{String(treeState?.staged ?? 0)}</code> unstaged=<code>{String(treeState?.unstaged ?? 0)}</code> untracked=<code>{String(treeState?.untracked ?? 0)}</code>
        </p>
        {lineageSnapshot?.latest_commit ? (
          <p className="text-[11px] text-[#f8d6e6] mt-1">latest <code>{lineageSnapshot.latest_commit}</code></p>
        ) : null}

        {lineageSnapshot?.continuity_drift?.active ? (
          <p className="text-[11px] text-[#ffc6db] mt-1">
            <ShieldAlert size={12} className="inline mr-1" />
            drift {lineageSnapshot.continuity_drift.code}: {lineageSnapshot.continuity_drift.message}
          </p>
        ) : (
          <p className="text-[11px] text-[#d6f0cf] mt-1">continuity drift clear</p>
        )}

        <p className="text-[11px] text-[#e6bdd2] mt-1">
          study blocked_gates=<code>{String(runtimeSignals?.blocked_gate_count ?? 0)}</code> active_drifts=<code>{String(runtimeSignals?.active_drift_count ?? 0)}</code> queue_pending=<code>{String(runtimeSignals?.queue_pending_count ?? 0)}</code>
        </p>
        {runtimeError ? <p className="text-[11px] text-[#ffd9c2] mt-1">runtime fetch warning: {runtimeError}</p> : null}
      </div>

      <div className="rounded-lg border border-[rgba(126,166,192,0.3)] bg-[rgba(10,17,24,0.7)] p-3">
        <p className="text-xs font-semibold text-[#cbe5f8]">Session Lens / 透過レンズ</p>
        <p className="text-[11px] text-[#9ec7dd] mt-1">
          lens mass <code>{chatLensState ? chatLensState.mass.toFixed(2) : "n/a"}</code> | priority <code>{chatLensState ? chatLensState.priority.toFixed(2) : "n/a"}</code> | dominant field <code>{dominantField}</code>
        </p>
        {activeChatSession ? (
          <p className="text-[11px] text-[#b6d7eb] mt-1">
            session <code>{activeChatSession.id}</code> | presence <code>{activeChatSession.presence}</code> | status <code>{activeChatSession.status}</code>
          </p>
        ) : (
          <p className="text-[11px] text-muted mt-1">no active projection chat session</p>
        )}
        <p className="text-[11px] text-[#afcfe2] mt-1">catalog refs <code>{catalog?.items?.length ?? 0}</code></p>
      </div>

      <div className="rounded-lg border border-[rgba(102,217,239,0.26)] bg-[rgba(10,22,30,0.62)] p-3">
        <p className="text-xs font-semibold text-[#d4ecff]">Nearby Embedded Context / 近傍埋め込み</p>
        <p className="text-[11px] text-[#9dc6dd] mt-1">
          Nearby embedding-bearing items are sent with muse state context for language synthesis.
        </p>
        {nearbyEmbedSummaries.length <= 0 ? (
          <p className="text-xs text-muted mt-2">No nearby embedding summaries for this muse presence yet.</p>
        ) : (
          <div className="mt-2 grid gap-1.5">
            {nearbyEmbedSummaries.map((line) => (
              <p
                key={line}
                className="rounded-md border border-[rgba(102,217,239,0.2)] bg-[rgba(14,24,36,0.6)] px-2 py-1 text-[11px] font-mono text-[#c8e4f6] break-all"
              >
                {line}
              </p>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-lg border border-[var(--line)] bg-[rgba(16,20,20,0.74)] p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-semibold text-ink">Muse Conversation / ミューズ対話</p>
          <div className="inline-flex rounded-md border border-[rgba(102,217,239,0.32)] overflow-hidden">
            <button
              type="button"
              onClick={() => setChatChannel("ledger")}
              className={`px-3 py-1 text-xs transition-colors ${chatChannel === "ledger" ? "bg-[rgba(102,217,239,0.2)] text-[#d8f2ff]" : "bg-[rgba(31,32,29,0.86)] text-muted hover:bg-[rgba(55,56,48,0.92)]"}`}
            >
              ledger mode
            </button>
            <button
              type="button"
              onClick={() => setChatChannel("llm")}
              className={`px-3 py-1 text-xs transition-colors ${chatChannel === "llm" ? "bg-[rgba(166,226,46,0.2)] text-[#e9ffd3]" : "bg-[rgba(31,32,29,0.86)] text-muted hover:bg-[rgba(55,56,48,0.92)]"}`}
            >
              llm mode
            </button>
          </div>
        </div>

        <p className="text-[11px] text-muted mt-1">
          {chatChannel === "ledger"
            ? `ledger mode auto-routes plain text into /say ${resolvedMusePresenceId} for deterministic evidence replies.`
            : "llm mode uses /api/muse/message and carries live muse state + nearby embedding context."}
        </p>

        <div
          ref={scrollRef}
          className="mt-2 border border-[var(--line)] rounded-lg bg-[rgba(31,32,29,0.86)] p-2 min-h-[130px] max-h-[260px] overflow-auto grid gap-2"
        >
          {messages.map((msg, index) => {
            const chips: string[] = [];
            if (msg.meta?.channel) {
              chips.push(msg.meta.channel);
            }
            if (msg.meta?.model) {
              chips.push(`model=${msg.meta.model}`);
            }
            if (msg.meta?.fallback) {
              chips.push("fallback");
            }
            if (msg.meta?.source) {
              chips.push(msg.meta.source);
            }

            return (
              <article
                key={`${index}-${msg.role}`}
                className={`border rounded-lg p-2 text-sm whitespace-pre-wrap leading-relaxed ${messageTone(msg)}`}
              >
                <span className="font-bold text-xs opacity-75 block mb-1">
                  {msg.role === "user"
                    ? "operator / 操作者"
                    : msg.role === "assistant"
                      ? `${msg.meta?.presenceName || msg.meta?.presenceId || activeMuse?.en || "muse"} / ミューズ`
                      : "system"}
                </span>
                {msg.text}
                {chips.length > 0 ? (
                  <p className="mt-1 text-[10px] text-muted font-mono">{chips.join(" | ")}</p>
                ) : null}
              </article>
            );
          })}
          {isThinking ? (
            <div className="border border-dashed border-[rgba(102,217,239,0.44)] rounded-lg p-2 text-sm bg-[rgba(102,217,239,0.12)] animate-pulse text-[#66d9ef]">
              <Sparkles size={14} className="inline mr-1" />
              {activeMuse?.en || "muse"} is synthesizing from simulation state... / ミューズが状態から合成中...
            </div>
          ) : null}
        </div>

        {pendingAsk ? (
          <div className="mt-2 border border-[rgba(249,38,114,0.5)] rounded-lg bg-[rgba(249,38,114,0.1)] p-2">
            <p className="text-xs font-semibold text-[#f92672]">Autopilot Request / 自動操縦の質問</p>
            <p className="text-[11px] text-muted mt-1">
              gate: <code>{pendingAsk.gate || "unknown"}</code>
              {pendingAsk.urgency ? (
                <>
                  {" "}| urgency: <code>{pendingAsk.urgency}</code>
                </>
              ) : null}
            </p>
            <p className="text-xs text-muted mt-1">{pendingAsk.reason}</p>
            <p className="text-sm text-ink mt-1">{pendingAsk.need}</p>
            {pendingAsk.context ? (
              <p className="text-[11px] text-muted mt-1 font-mono">
                {Object.entries(pendingAsk.context)
                  .slice(0, 3)
                  .map(([key, value]) => `${key}=${formatContextValue(value)}`)
                  .join(" | ")}
              </p>
            ) : null}
            {pendingAsk.options && pendingAsk.options.length > 0 ? (
              <div className="mt-2 flex flex-wrap gap-2">
                {pendingAsk.options.map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => handleAskOption(option)}
                    className="text-xs border border-[rgba(249,38,114,0.55)] rounded-md px-2 py-1 bg-[rgba(249,38,114,0.16)] hover:bg-[rgba(249,38,114,0.24)] transition-colors"
                  >
                    {option}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        <div className="grid grid-cols-[1fr_auto] gap-2 mt-2">
          <textarea
            id="chat-input"
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={
              isThinking
                ? "Thinking... / 思考中..."
                : chatChannel === "ledger"
                  ? `Describe the work you want ${activeMuse?.en || "the muse"} to route into system signals...`
                  : "Ask the muse for a state-derived language response..."
            }
            disabled={isThinking}
            className="w-full min-h-[74px] max-h-[180px] resize-y border border-[var(--line)] rounded-lg p-2 font-inherit bg-[rgba(39,40,34,0.9)] text-ink disabled:opacity-50"
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                handleSend();
              }
            }}
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={isThinking}
            className="self-end border border-[var(--line)] rounded-lg px-3 py-2 bg-[rgba(102,217,239,0.2)] hover:bg-[rgba(102,217,239,0.28)] transition-colors disabled:opacity-50"
          >
            <Send size={18} />
          </button>
        </div>

        <div className="flex gap-2 mt-2 flex-wrap">
          <button
            type="button"
            onClick={onRecord}
            className={`flex items-center gap-2 btn-voice px-3 py-2 rounded-lg border border-[var(--line)] ${isRecording ? "animate-pulse text-[#f92672]" : ""}`}
          >
            <Mic size={16} />
            <span className="text-xs">Record</span>
          </button>
          <button
            type="button"
            onClick={onTranscribe}
            className="flex items-center gap-2 btn-voice px-3 py-2 rounded-lg border border-[var(--line)]"
          >
            <FileAudio size={16} />
            <span className="text-xs">Transcribe</span>
          </button>
          <button
            type="button"
            onClick={() => onSendVoice(resolvedMusePresenceId, liveWorkspaceContext)}
            className="flex items-center gap-2 btn-voice px-3 py-2 rounded-lg border border-[var(--line)]"
          >
            <MessageSquare size={16} />
            <span className="text-xs">Send Voice</span>
          </button>
        </div>

        <p className="text-xs text-muted mt-1">{voiceInputMeta}</p>
        <p className="text-[10px] text-muted/80 mt-1">
          commands: <code>/ledger ...</code> <code>{`/say ${resolvedMusePresenceId} ...`}</code> <code>/study</code> <code>/drift</code> <code>/push-truth --dry-run</code>
        </p>
        <p className="text-[10px] text-muted/80 mt-1">
          <Activity size={11} className="inline mr-1" />
          Muse lane translates language into system signals, and turns live system state back into language.
        </p>
      </div>
    </div>
  );
}
