import type { Plugin } from "@opencode-ai/plugin";

const DEFAULT_SIMULATION_WITNESS_URL = "http://127.0.0.1:8787/api/witness";
const DEFAULT_DEDUPE_WINDOW_MS = 2000;
const MAX_DEDUPE_KEYS = 512;

const IMPORTANT_EVENT_TYPES = new Set<string>([
  "session.created",
  "session.idle",
  "session.compacted",
  "session.error",
  "command.executed",
  "permission.updated",
  "permission.replied",
  "file.edited",
  "file.watcher.updated",
  "todo.updated",
  "vcs.branch.updated",
]);

const WATCHED_TOOLS = new Set<string>([
  "bash",
  "write",
  "edit",
  "apply_patch",
]);

type GenericEvent = {
  type: string;
  properties?: unknown;
};

type SimulationWitnessEvent = {
  type: string;
  target: string;
  origin: string;
  ts: string;
  details?: Record<string, unknown>;
};

function toRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function compact(value: unknown): string {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function clip(value: string, maxLen: number): string {
  if (value.length <= maxLen) {
    return value;
  }
  return `${value.slice(0, Math.max(0, maxLen - 1))}~`;
}

function asPositiveInt(value: unknown, fallback: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return parsed;
}

function isNonZeroNumber(value: unknown): boolean {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed !== 0;
}

function isToolFailure(output: { title: string; output: string; metadata: unknown }): boolean {
  const metadata = toRecord(output.metadata);
  if (!metadata) {
    const title = compact(output.title).toLowerCase();
    return title.includes("error") || title.includes("failed");
  }

  if (isNonZeroNumber(metadata.exitCode) || isNonZeroNumber(metadata.exit_code)) {
    return true;
  }
  if (metadata.ok === false || metadata.success === false) {
    return true;
  }

  const title = compact(output.title).toLowerCase();
  return title.includes("error") || title.includes("failed");
}

function normalizeRelativePath(
  pathValue: string,
  directory: string,
  worktree: string,
): string {
  const clean = compact(pathValue);
  if (!clean) {
    return "unknown";
  }

  const normalizedDirectory = directory.endsWith("/") ? directory : `${directory}/`;
  const normalizedWorktree = worktree.endsWith("/") ? worktree : `${worktree}/`;

  if (clean.startsWith(normalizedWorktree)) {
    return clean.slice(normalizedWorktree.length);
  }
  if (clean.startsWith(normalizedDirectory)) {
    return clean.slice(normalizedDirectory.length);
  }
  return clean;
}

function eventTarget(
  event: GenericEvent,
  directory: string,
  worktree: string,
): string | null {
  const props = toRecord(event.properties);
  if (!props) {
    return event.type;
  }

  switch (event.type) {
    case "session.created": {
      const info = toRecord(props.info);
      const sessionID = compact(info?.id);
      return sessionID ? `session:${sessionID}` : "session:created";
    }
    case "session.idle": {
      const sessionID = compact(props.sessionID);
      return sessionID ? `session:${sessionID}` : "session:idle";
    }
    case "session.compacted": {
      const sessionID = compact(props.sessionID);
      return sessionID ? `session:${sessionID}` : "session:compacted";
    }
    case "session.error": {
      const sessionID = compact(props.sessionID) || "unknown";
      const error = toRecord(props.error);
      const errorName = compact(error?.name);
      if (errorName) {
        return `session:${sessionID}:${clip(errorName, 48)}`;
      }
      return `session:${sessionID}:error`;
    }
    case "command.executed": {
      const command = compact(props.name);
      const args = compact(props.arguments);
      if (!command) {
        return null;
      }
      const joined = args ? `/${command} ${args}` : `/${command}`;
      return clip(joined, 160);
    }
    case "permission.updated": {
      const permissionType = compact(props.type) || "permission";
      const title = compact(props.title) || "requested";
      return clip(`${permissionType}:${title}`, 160);
    }
    case "permission.replied": {
      const response = compact(props.response) || "replied";
      const permissionID = compact(props.permissionID) || "unknown";
      return clip(`${response}:${permissionID}`, 160);
    }
    case "file.edited": {
      const filePath = compact(props.file);
      if (!filePath) {
        return null;
      }
      return normalizeRelativePath(filePath, directory, worktree);
    }
    case "file.watcher.updated": {
      const watcherEvent = compact(props.event) || "change";
      if (watcherEvent === "change") {
        return null;
      }
      const filePath = compact(props.file);
      if (!filePath) {
        return watcherEvent;
      }
      const relPath = normalizeRelativePath(filePath, directory, worktree);
      return `${watcherEvent}:${relPath}`;
    }
    case "todo.updated": {
      const todosRaw = Array.isArray(props.todos) ? props.todos : [];
      const counts = {
        pending: 0,
        in_progress: 0,
        completed: 0,
        cancelled: 0,
      };

      for (const row of todosRaw) {
        const todo = toRecord(row);
        if (!todo) {
          continue;
        }
        const status = compact(todo.status);
        if (status in counts) {
          counts[status as keyof typeof counts] += 1;
        }
      }

      const sessionID = compact(props.sessionID);
      const summary = `todos:${todosRaw.length} p:${counts.pending} i:${counts.in_progress} c:${counts.completed} x:${counts.cancelled}`;
      return sessionID ? `session:${sessionID} ${summary}` : summary;
    }
    case "vcs.branch.updated": {
      const branch = compact(props.branch) || "unknown";
      return `branch:${branch}`;
    }
    default:
      return event.type;
  }
}

function toolTarget(
  tool: string,
  args: unknown,
  resultTag: "ok" | "failed",
  directory: string,
  worktree: string,
): string {
  const details = toRecord(args);

  if (tool === "bash") {
    const rawCommand = compact(details?.command);
    const commandHead = clip(rawCommand.split(/\s+/)[0] || "", 48);
    return commandHead
      ? `bash:${resultTag}:${commandHead}`
      : `bash:${resultTag}`;
  }

  const filePath = compact(details?.filePath);
  if (filePath) {
    const relPath = normalizeRelativePath(filePath, directory, worktree);
    return `${tool}:${resultTag}:${relPath}`;
  }

  return `${tool}:${resultTag}`;
}

function canUseHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

export const SimulationEventLoggerPlugin: Plugin = async ({
  client,
  directory,
  worktree,
}) => {
  const configuredUrl =
    compact(process.env.OPENCODE_SIMULATION_WITNESS_URL) ||
    compact(process.env.OPENCODE_SIMULATION_URL) ||
    DEFAULT_SIMULATION_WITNESS_URL;
  const simulationWitnessUrl = canUseHttpUrl(configuredUrl)
    ? configuredUrl
    : DEFAULT_SIMULATION_WITNESS_URL;

  const dedupeWindowMs = asPositiveInt(
    process.env.OPENCODE_SIM_EVENT_DEDUPE_MS,
    DEFAULT_DEDUPE_WINDOW_MS,
  );
  const recentlySent = new Map<string, number>();

  const emit = (event: SimulationWitnessEvent): void => {
    const cleanType = clip(compact(event.type), 64);
    const cleanTarget = clip(compact(event.target), 220);
    if (!cleanType || !cleanTarget) {
      return;
    }

    const dedupeKey = `${cleanType}|${cleanTarget}`;
    const now = Date.now();
    const previousTs = recentlySent.get(dedupeKey) ?? 0;
    if (now - previousTs < dedupeWindowMs) {
      return;
    }
    recentlySent.set(dedupeKey, now);

    while (recentlySent.size > MAX_DEDUPE_KEYS) {
      const oldest = recentlySent.keys().next().value;
      if (!oldest) {
        break;
      }
      recentlySent.delete(oldest);
    }

    const body = JSON.stringify({
      ...event,
      type: cleanType,
      target: cleanTarget,
    });

    void fetch(simulationWitnessUrl, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body,
    }).catch(() => undefined);
  };

  await client.app
    .log({
      body: {
        service: "simulation-event-logger",
        level: "info",
        message: "Simulation event logger plugin initialized",
        extra: {
          simulation_witness_url: simulationWitnessUrl,
          dedupe_window_ms: dedupeWindowMs,
        },
      },
    })
    .catch(() => undefined);

  emit({
    type: "opencode.plugin.loaded",
    target: "simulation-event-logger",
    origin: "opencode.plugin.simulation-event-logger",
    ts: new Date().toISOString(),
  });

  return {
    event: async ({ event }) => {
      if (!IMPORTANT_EVENT_TYPES.has(event.type)) {
        return;
      }

      const target = eventTarget(event as GenericEvent, directory, worktree);
      if (!target) {
        return;
      }

      emit({
        type: `opencode.${event.type}`,
        target,
        origin: "opencode.plugin.simulation-event-logger",
        ts: new Date().toISOString(),
      });
    },

    "permission.ask": async (input, output) => {
      const permissionType = compact(input.type) || "permission";
      const title = compact(input.title) || "request";
      const status = compact(output.status) || "ask";
      emit({
        type: "opencode.permission.ask",
        target: clip(`${permissionType}:${title}:${status}`, 220),
        origin: "opencode.plugin.simulation-event-logger",
        ts: new Date().toISOString(),
      });
    },

    "tool.execute.after": async (input, output) => {
      const tool = compact(input.tool).toLowerCase();
      if (!tool) {
        return;
      }

      const failed = isToolFailure(output);
      if (!failed && !WATCHED_TOOLS.has(tool)) {
        return;
      }

      const resultTag = failed ? "failed" : "ok";
      emit({
        type: failed ? "opencode.tool.failed" : "opencode.tool.executed",
        target: toolTarget(tool, input.args, resultTag, directory, worktree),
        origin: "opencode.plugin.simulation-event-logger",
        ts: new Date().toISOString(),
      });
    },
  };
};
