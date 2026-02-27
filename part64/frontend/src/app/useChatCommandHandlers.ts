import { useCallback } from "react";

import { runtimeBaseUrl } from "../runtime/endpoints";
import type {
  CouncilDecision,
  CouncilApiResponse,
  DriftScanPayload,
  MuseWorkspaceContext,
  StudySnapshotPayload,
  TaskQueueSnapshot,
} from "../types";

interface UseChatCommandHandlersArgs {
  activeMusePresenceId: string;
  catalogGeneratedAt?: string;
  catalogTruthGateBlocked?: boolean;
  simulationTimestamp?: string;
  simulationTruthGateBlocked?: boolean;
  buildMuseSurroundingNodes: (
    musePresenceId: string,
    workspace: MuseWorkspaceContext | null,
  ) => Array<Record<string, unknown>>;
  emitSystemMessage: (text: string) => void;
  emitWitnessChatReply: (
    payload: Record<string, unknown>,
    source: string,
    requestedMusePresenceId?: string,
  ) => void;
}

interface UseChatCommandHandlersResult {
  handleChatCommand: (text: string, musePresenceId?: string) => Promise<boolean>;
}

interface CommandContext {
  catalogGeneratedAt: string | undefined;
  catalogTruthGateBlocked: boolean | undefined;
  simulationTimestamp: string | undefined;
  simulationTruthGateBlocked: boolean | undefined;
  buildMuseSurroundingNodes: (
    musePresenceId: string,
    workspace: MuseWorkspaceContext | null,
  ) => Array<Record<string, unknown>>;
  emitSystemMessage: (text: string) => void;
  emitWitnessChatReply: (
    payload: Record<string, unknown>,
    source: string,
    requestedMusePresenceId?: string,
  ) => void;
}

interface LegacyStudySnapshot {
  blocked: number;
  drifts: number;
  pending: number;
  pendingCouncil: number;
  truthBlocked: boolean;
  queueEventCount: number;
  councilApprovedCount: number;
  councilDecisionCount: number;
  topDecisionLine: string;
  gateReasons: string;
}

interface SayCommandResponse {
  muse?: { label?: string };
  turn_id?: string;
  reply?: string;
  manifest?: {
    explicit_selected?: unknown[];
    surround_selected?: unknown[];
  };
}

const LEDGER_COMMAND = "/ledger";
const SAY_COMMAND = "/say";
const DRIFT_COMMAND = "/drift";
const PUSH_TRUTH_DRY_RUN_COMMAND = "/push-truth --dry-run";
const STUDY_COMMAND = "/study";
const STUDY_NOW_COMMAND = "/study now";
const STUDY_EXPORT_PREFIX = "/study export";

function runtimeUrl(path: string): string {
  return `${runtimeBaseUrl()}${path}`;
}

function startsWithCommand(text: string, command: string): boolean {
  return text.toLowerCase().startsWith(command);
}

function parseLedgerUtterances(trimmedText: string): string[] {
  const payloadText = trimmedText.replace(/^\/ledger\s*/i, "");
  if (!payloadText) {
    return [];
  }
  return payloadText
    .split("|")
    .map((row) => row.trim())
    .filter((row) => row.length > 0);
}

async function handleLedgerCommand(text: string, context: CommandContext): Promise<boolean> {
  const trimmed = text.trim();
  if (!startsWithCommand(trimmed, LEDGER_COMMAND)) {
    return false;
  }

  try {
    const response = await fetch(runtimeUrl("/api/eta-mu-ledger"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ utterances: parseLedgerUtterances(trimmed) }),
    });
    const payload = (await response.json()) as { jsonl?: string };
    const body = payload?.jsonl ? payload.jsonl.trim() : "(no utterances)";
    context.emitSystemMessage(`eta/mu ledger\n${body}`);
  } catch {
    context.emitSystemMessage("eta/mu ledger failed");
  }

  return true;
}

function parseSayCommand(trimmedText: string, fallbackMusePresenceId: string): {
  presenceId: string;
  messageText: string;
} {
  const args = trimmedText.replace(/^\/say\s*/i, "");
  const [presenceIdRaw, ...messageTokens] = args
    .split(/\s+/)
    .filter((token) => token.length > 0);
  return {
    presenceId: presenceIdRaw || fallbackMusePresenceId || "witness_thread",
    messageText: messageTokens.join(" "),
  };
}

async function handlePresenceSayCommand(
  text: string,
  fallbackMusePresenceId: string,
  context: CommandContext,
): Promise<boolean> {
  const trimmed = text.trim();
  if (!startsWithCommand(trimmed, SAY_COMMAND)) {
    return false;
  }

  const { presenceId, messageText } = parseSayCommand(trimmed, fallbackMusePresenceId);
  const surroundingNodes = context.buildMuseSurroundingNodes(presenceId, null);

  try {
    const payload = await requestSayCommand(
      presenceId,
      messageText,
      surroundingNodes,
      context.simulationTimestamp,
      context.catalogGeneratedAt,
    );
    context.emitWitnessChatReply(payload as Record<string, unknown>, "command:/say", presenceId);
    context.emitSystemMessage(buildSayCommandSystemMessage(payload, presenceId));
  } catch {
    context.emitSystemMessage("muse say failed");
  }

  return true;
}

function buildSayCommandSystemMessage(payload: SayCommandResponse, presenceId: string): string {
  const explicitCount = payload?.manifest?.explicit_selected?.length || 0;
  const surroundingCount = payload?.manifest?.surround_selected?.length || 0;
  return `${payload?.muse?.label || presenceId} / muse turn ${payload?.turn_id || ""}\n${payload?.reply || "(no reply)"}\n`
    + `explicit=${explicitCount} surrounding=${surroundingCount}`;
}

async function requestSayCommand(
  presenceId: string,
  messageText: string,
  surroundingNodes: Array<Record<string, unknown>>,
  simulationTimestamp: string | undefined,
  catalogGeneratedAt: string | undefined,
): Promise<SayCommandResponse> {
  const response = await fetch(runtimeUrl("/api/muse/message"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      muse_id: presenceId,
      text: messageText,
      mode: "deterministic",
      token_budget: 1024,
      graph_revision: simulationTimestamp || catalogGeneratedAt || "",
      surrounding_nodes: surroundingNodes,
    }),
  });
  if (!response.ok) {
    throw new Error(`muse say request failed (${response.status})`);
  }

  return (await response.json()) as SayCommandResponse;
}

async function handleDriftCommand(text: string, context: CommandContext): Promise<boolean> {
  const trimmed = text.trim().toLowerCase();
  if (trimmed !== DRIFT_COMMAND) {
    return false;
  }

  try {
    const response = await fetch(runtimeUrl("/api/drift/scan"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const payload = (await response.json()) as {
      active_drifts?: unknown[];
      blocked_gates?: unknown[];
    };
    const drifts = Array.isArray(payload?.active_drifts) ? payload.active_drifts.length : 0;
    const blocked = Array.isArray(payload?.blocked_gates) ? payload.blocked_gates.length : 0;
    context.emitSystemMessage(`drift scan\nactive_drifts=${drifts} blocked_gates=${blocked}`);
  } catch {
    context.emitSystemMessage("drift scan failed");
  }

  return true;
}

async function handlePushTruthDryRunCommand(text: string, context: CommandContext): Promise<boolean> {
  if (text.trim().toLowerCase() !== PUSH_TRUTH_DRY_RUN_COMMAND) {
    return false;
  }

  try {
    const response = await fetch(runtimeUrl("/api/push-truth/dry-run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const payload = (await response.json()) as {
      gate?: { blocked?: boolean };
      needs?: string[];
    };
    const blocked = payload?.gate?.blocked ? "blocked" : "pass";
    const needs = Array.isArray(payload?.needs) ? payload.needs.join(", ") : "";
    context.emitSystemMessage(`push-truth dry-run\ngate=${blocked}\nneeds=${needs || "(none)"}`);
  } catch {
    context.emitSystemMessage("push-truth dry-run failed");
  }

  return true;
}

function isStudyExportCommand(trimmedLower: string): boolean {
  return trimmedLower === STUDY_EXPORT_PREFIX || trimmedLower.startsWith(`${STUDY_EXPORT_PREFIX} `);
}

async function handleStudyExportCommand(rawText: string, context: CommandContext): Promise<boolean> {
  const trimmedLower = rawText.toLowerCase();
  if (!isStudyExportCommand(trimmedLower)) {
    return false;
  }

  const label = rawText.slice(STUDY_EXPORT_PREFIX.length).trim();
  try {
    const response = await fetch(runtimeUrl("/api/study/export"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        label: label || "chat-export",
        include_truth: true,
        refs: ["chat:/study export"],
      }),
    });
    if (!response.ok) {
      throw new Error(`study export failed: ${response.status}`);
    }
    const payload = (await response.json()) as {
      event?: { id?: string; label?: string };
      history?: { count?: number };
    };
    const eventId = String(payload.event?.id || "(unknown)");
    const historyCount = Number(payload.history?.count ?? 0);
    const resolvedLabel = String(payload.event?.label || label || "chat-export");
    context.emitSystemMessage(`study export\nid=${eventId}\nlabel=${resolvedLabel}\nhistory=${historyCount}`);
  } catch {
    context.emitSystemMessage("study export failed");
  }

  return true;
}

function isStudySnapshotCommand(trimmedLower: string): boolean {
  return trimmedLower === STUDY_COMMAND || trimmedLower === STUDY_NOW_COMMAND;
}

function buildTopDecisionLine(topDecision: CouncilDecision | undefined): string {
  if (!topDecision) {
    return "top_decision=(none)";
  }
  return `top_decision=${topDecision.status} id=${topDecision.id} source=${String(topDecision.resource?.source_rel_path || "(unknown)")}`;
}

function buildStudySnapshotMessage(study: StudySnapshotPayload): string {
  const gateReasons = (study.drift?.blocked_gates ?? [])
    .map((row) => row.reason)
    .slice(0, 4)
    .join(", ");
  const warningLine = (study.warnings ?? [])
    .slice(0, 3)
    .map((row) => `${row.code}:${row.message}`)
    .join(" | ");
  const signals = study.signals;
  const topDecisionLine = buildTopDecisionLine(study.council?.decisions?.[0]);

  return [
    "study snapshot",
    `stability=${Math.round(study.stability.score * 100)}% (${study.stability.label})`,
    `truth_gate=${signals.truth_gate_blocked ? "blocked" : "clear"}`,
    `blocked_gates=${signals.blocked_gate_count} active_drifts=${signals.active_drift_count}`,
    `queue_pending=${signals.queue_pending_count} queue_events=${signals.queue_event_count}`,
    `council_pending=${signals.council_pending_count} approved=${signals.council_approved_count} decisions=${signals.council_decision_count}`,
    topDecisionLine,
    `gate_reasons=${gateReasons || "(none)"}`,
    `runtime_receipts_within_vault=${String(study.runtime.receipts_path_within_vault)}`,
    `warnings=${warningLine || "(none)"}`,
  ].join("\n");
}

function computeLegacyStabilityScore(snapshot: LegacyStudySnapshot): number {
  const blockedPenalty = Math.min(0.34, (snapshot.blocked / 4) * 0.34);
  const driftPenalty = Math.min(0.18, (snapshot.drifts / 8) * 0.18);
  const queuePenalty = Math.min(0.2, (snapshot.pending / 8) * 0.2);
  const councilPenalty = Math.min(0.16, (snapshot.pendingCouncil / 5) * 0.16);
  const truthPenalty = snapshot.truthBlocked ? 0.12 : 0;
  return Math.max(0, Math.min(1, 1 - blockedPenalty - driftPenalty - queuePenalty - councilPenalty - truthPenalty));
}

function buildLegacyStudySnapshotMessage(snapshot: LegacyStudySnapshot): string {
  const stabilityScore = computeLegacyStabilityScore(snapshot);
  return [
    "study snapshot",
    `stability=${Math.round(stabilityScore * 100)}%`,
    `truth_gate=${snapshot.truthBlocked ? "blocked" : "clear"}`,
    `blocked_gates=${snapshot.blocked} active_drifts=${snapshot.drifts}`,
    `queue_pending=${snapshot.pending} queue_events=${snapshot.queueEventCount}`,
    `council_pending=${snapshot.pendingCouncil} approved=${snapshot.councilApprovedCount} decisions=${snapshot.councilDecisionCount}`,
    snapshot.topDecisionLine,
    `gate_reasons=${snapshot.gateReasons || "(none)"}`,
    "runtime_receipts_within_vault=(unknown:legacy-mode)",
  ].join("\n");
}

async function fetchLegacyStudySnapshot(context: CommandContext): Promise<LegacyStudySnapshot> {
  const [councilRes, queueRes, driftRes] = await Promise.all([
    fetch(runtimeUrl("/api/council?limit=6")),
    fetch(runtimeUrl("/api/task/queue")),
    fetch(runtimeUrl("/api/drift/scan"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    }),
  ]);

  if (!councilRes.ok || !queueRes.ok || !driftRes.ok) {
    throw new Error(
      `study fetch failed: council=${councilRes.status} queue=${queueRes.status} drift=${driftRes.status}`,
    );
  }

  const councilPayload = (await councilRes.json()) as CouncilApiResponse;
  const queuePayload = (await queueRes.json()) as { ok: boolean; queue: TaskQueueSnapshot };
  const driftPayload = (await driftRes.json()) as DriftScanPayload;
  const council = councilPayload.council;
  const queue = queuePayload.queue;
  const truthBlocked = Boolean(context.simulationTruthGateBlocked ?? context.catalogTruthGateBlocked);
  const topDecision = council.decisions?.[0];
  const topDecisionLine = topDecision
    ? `top_decision=${topDecision.status} id=${topDecision.id} source=${String(topDecision.resource?.source_rel_path || "(unknown)")}`
    : "top_decision=(none)";

  return {
    blocked: driftPayload.blocked_gates.length,
    drifts: driftPayload.active_drifts.length,
    pending: queue.pending_count,
    pendingCouncil: council.pending_count,
    truthBlocked,
    queueEventCount: queue.event_count,
    councilApprovedCount: council.approved_count,
    councilDecisionCount: council.decision_count,
    topDecisionLine,
    gateReasons: driftPayload.blocked_gates
      .map((row) => row.reason)
      .slice(0, 4)
      .join(", "),
  };
}

async function handleStudyCommand(text: string, context: CommandContext): Promise<boolean> {
  const raw = text.trim();
  if (await handleStudyExportCommand(raw, context)) {
    return true;
  }

  if (!isStudySnapshotCommand(raw.toLowerCase())) {
    return false;
  }

  try {
    const studyResponse = await fetch(runtimeUrl("/api/study?limit=6"));
    if (studyResponse.ok) {
      const study = (await studyResponse.json()) as StudySnapshotPayload;
      context.emitSystemMessage(buildStudySnapshotMessage(study));
      return true;
    }

    if (studyResponse.status !== 404) {
      throw new Error(`study fetch failed: /api/study status=${studyResponse.status}`);
    }

    const legacySnapshot = await fetchLegacyStudySnapshot(context);
    context.emitSystemMessage(buildLegacyStudySnapshotMessage(legacySnapshot));
  } catch {
    context.emitSystemMessage("study snapshot failed");
  }

  return true;
}

async function dispatchChatCommand(
  text: string,
  musePresenceId: string,
  context: CommandContext,
): Promise<boolean> {
  if (await handleLedgerCommand(text, context)) {
    return true;
  }
  if (await handlePresenceSayCommand(text, musePresenceId, context)) {
    return true;
  }
  if (await handleDriftCommand(text, context)) {
    return true;
  }
  if (await handlePushTruthDryRunCommand(text, context)) {
    return true;
  }
  return handleStudyCommand(text, context);
}

export function useChatCommandHandlers({
  activeMusePresenceId,
  catalogGeneratedAt,
  catalogTruthGateBlocked,
  simulationTimestamp,
  simulationTruthGateBlocked,
  buildMuseSurroundingNodes,
  emitSystemMessage,
  emitWitnessChatReply,
}: UseChatCommandHandlersArgs): UseChatCommandHandlersResult {
  const handleChatCommand = useCallback(
    async (text: string, musePresenceId = activeMusePresenceId): Promise<boolean> =>
      dispatchChatCommand(text, musePresenceId, {
        catalogGeneratedAt,
        catalogTruthGateBlocked,
        simulationTimestamp,
        simulationTruthGateBlocked,
        buildMuseSurroundingNodes,
        emitSystemMessage,
        emitWitnessChatReply,
      }),
    [
      activeMusePresenceId,
      buildMuseSurroundingNodes,
      catalogGeneratedAt,
      catalogTruthGateBlocked,
      emitSystemMessage,
      emitWitnessChatReply,
      simulationTimestamp,
      simulationTruthGateBlocked,
    ],
  );

  return {
    handleChatCommand,
  };
}
