import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";

type GatewayMode = "smart" | "direct" | "hardway";

interface GatewayConfig {
  enabled: boolean;
  default_mode: GatewayMode;
  default_model: string;
  modes: string[];
  hardware_options: string[];
  field_routes: Record<string, Record<string, string>>;
}

interface GatewayDecision {
  applied: boolean;
  mode: string;
  field: string;
  hardware: string | null;
  requested_model: string;
  resolved_model: string;
  resolved_model_public: string;
  reason: string;
  candidate_models: string[];
  provider_available: boolean;
}

interface CompletionContentChunk {
  type: string;
  text?: string;
}

interface CompletionResponse {
  choices?: Array<{
    message?: {
      content?: string | CompletionContentChunk[];
    };
  }>;
  error?: {
    message?: string;
  };
  detail?: unknown;
}

interface ModelListingResponse {
  data?: Array<{
    id?: string;
  }>;
}

interface QuotaCredentialEntry {
  identifier?: string;
  status?: string;
  requests?: number;
  approx_cost?: number | null;
  tier?: string;
}

interface QuotaProviderEntry {
  credential_count?: number;
  active_count?: number;
  on_cooldown_count?: number;
  exhausted_count?: number;
  total_requests?: number;
  approx_cost?: number | null;
  credentials?: QuotaCredentialEntry[];
}

interface QuotaSummary {
  total_credentials?: number;
  active_credentials?: number;
  exhausted_credentials?: number;
  total_requests?: number;
}

interface QuotaStatsResponse {
  providers?: Record<string, QuotaProviderEntry>;
  summary?: QuotaSummary;
}

interface GatewayPayload {
  model: string;
  stream: false;
  messages: Array<{ role: "user"; content: string }>;
  gateway: Record<string, unknown>;
}

interface CodexAccountEntry {
  id: string;
  provider: string;
  masked_key: string;
  status: string;
  requests: number;
  last_used_ts?: number | null;
  key_cooldown_remaining?: number | null;
  env_key?: string | null;
  persisted?: boolean;
}

interface CodexAccountsResponse {
  schema_version?: string;
  provider?: string;
  rotation_mode?: string;
  fair_cycle_enabled?: boolean;
  credential_count?: number;
  accounts?: CodexAccountEntry[];
}

interface CodexServerEventEntry {
  id?: string;
  ts?: string;
  action?: string;
  outcome?: string;
  detail?: string;
  meta?: Record<string, unknown>;
}

interface CodexEventsResponse {
  events?: CodexServerEventEntry[];
}

interface CodexMutationResponse {
  ok?: boolean;
  added?: boolean;
  persisted?: boolean;
  env_key?: string | null;
  accounts?: CodexAccountsResponse;
  validation?: {
    model_count?: number;
    codex_model_count?: number;
  };
}

interface DockEventEntry {
  id: string;
  ts: string;
  action: string;
  outcome: "started" | "ok" | "error";
  detail: string;
}

interface Props {
  onClose: () => void;
}

const DEFAULT_GATEWAY_BASE_URL =
  import.meta.env.VITE_PROXY_BASE_URL ?? "http://127.0.0.1:18000";
const STORAGE_PREFIX = "eta_mu.galaxy_model_dock";
const STORAGE_KEYS = {
  gatewayUrl: `${STORAGE_PREFIX}.gateway_url`,
  apiKey: `${STORAGE_PREFIX}.api_key`,
  selectedModel: `${STORAGE_PREFIX}.selected_model`,
  mode: `${STORAGE_PREFIX}.mode`,
  field: `${STORAGE_PREFIX}.field`,
  hardware: `${STORAGE_PREFIX}.hardware`,
  overrideModel: `${STORAGE_PREFIX}.override_model`,
};

function readStoredValue(key: string, fallback = ""): string {
  if (typeof window === "undefined") {
    return fallback;
  }
  try {
    const value = window.localStorage.getItem(key);
    if (value === null) {
      return fallback;
    }
    return value;
  } catch {
    return fallback;
  }
}

function storeValue(key: string, value: string): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(key, value);
  } catch {
    return;
  }
}

function parseAssistantContent(payload: CompletionResponse): string {
  const content = payload.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const textBlocks = content
      .filter((entry) => entry.type === "text" && typeof entry.text === "string")
      .map((entry) => entry.text ?? "");
    if (textBlocks.length > 0) {
      return textBlocks.join("\n\n");
    }
  }
  if (payload.error?.message) {
    return payload.error.message;
  }
  if (payload.detail) {
    return JSON.stringify(payload.detail, null, 2);
  }
  return "No assistant content returned.";
}

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    const message = error.message.trim();
    if (message) {
      return message;
    }
  }
  return "unexpected request error";
}

function toMoneyLabel(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return "-";
  }
  if (!Number.isFinite(value) || value <= 0) {
    return "-";
  }
  if (value < 0.01) {
    return `<$${value.toFixed(4)}`;
  }
  return `$${value.toFixed(2)}`;
}

function formatLastUsed(value: number | null | undefined): string {
  if (!value || !Number.isFinite(value)) {
    return "never";
  }
  try {
    return new Date(value * 1000).toISOString();
  } catch {
    return "never";
  }
}

function normalizeMode(value: string): GatewayMode {
  if (value === "direct" || value === "hardway") {
    return value;
  }
  return "smart";
}

export function GalaxyModelDock({ onClose }: Props) {
  const initialSyncDoneRef = useRef(false);
  const [gatewayUrl, setGatewayUrl] = useState(() =>
    readStoredValue(STORAGE_KEYS.gatewayUrl, DEFAULT_GATEWAY_BASE_URL),
  );
  const [proxyApiKey, setProxyApiKey] = useState(() => readStoredValue(STORAGE_KEYS.apiKey));
  const [selectedModel, setSelectedModel] = useState(() =>
    readStoredValue(STORAGE_KEYS.selectedModel),
  );
  const [mode, setMode] = useState<GatewayMode>(() =>
    normalizeMode(readStoredValue(STORAGE_KEYS.mode, "smart")),
  );
  const [field, setField] = useState(() => readStoredValue(STORAGE_KEYS.field, "code"));
  const [hardware, setHardware] = useState(() => readStoredValue(STORAGE_KEYS.hardware, "gpu"));
  const [manualOverrideModel, setManualOverrideModel] = useState(() =>
    readStoredValue(STORAGE_KEYS.overrideModel),
  );
  const [prompt, setPrompt] = useState(
    "Run a quick readiness check and report one model-routing recommendation.",
  );

  const [config, setConfig] = useState<GatewayConfig | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [providers, setProviders] = useState<string[]>([]);
  const [quotaStats, setQuotaStats] = useState<QuotaStatsResponse | null>(null);
  const [decision, setDecision] = useState<GatewayDecision | null>(null);
  const [responseText, setResponseText] = useState("");
  const [statusMessage, setStatusMessage] = useState("idle");
  const [busy, setBusy] = useState(false);
  const [events, setEvents] = useState<DockEventEntry[]>([]);

  const [codexAccounts, setCodexAccounts] = useState<CodexAccountEntry[]>([]);
  const [codexRotationMode, setCodexRotationMode] = useState<"balanced" | "sequential">(
    "balanced",
  );
  const [codexFairCycleEnabled, setCodexFairCycleEnabled] = useState(false);
  const [codexServerEvents, setCodexServerEvents] = useState<CodexServerEventEntry[]>([]);
  const [codexLoginKey, setCodexLoginKey] = useState("");
  const [codexStatus, setCodexStatus] = useState("accounts not loaded");
  const [codexBusy, setCodexBusy] = useState(false);

  const fieldOptions = useMemo(
    () => Object.keys(config?.field_routes ?? {}).sort(),
    [config],
  );
  const quotaRows = useMemo(
    () => Object.entries(quotaStats?.providers ?? {}).sort(([a], [b]) => a.localeCompare(b)),
    [quotaStats],
  );
  const activeModel = selectedModel.trim() || config?.default_model || "octave-commons/promethean";

  const appendEvent = useCallback(
    (action: string, outcome: DockEventEntry["outcome"], detail: string) => {
      const normalizedDetail = detail.trim() || "(no detail)";
      const entry: DockEventEntry = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
        ts: new Date().toISOString(),
        action,
        outcome,
        detail: normalizedDetail,
      };
      setEvents((current) => [entry, ...current].slice(0, 40));
    },
    [],
  );

  useEffect(() => {
    storeValue(STORAGE_KEYS.gatewayUrl, gatewayUrl);
  }, [gatewayUrl]);

  useEffect(() => {
    storeValue(STORAGE_KEYS.apiKey, proxyApiKey);
  }, [proxyApiKey]);

  useEffect(() => {
    storeValue(STORAGE_KEYS.selectedModel, selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    storeValue(STORAGE_KEYS.mode, mode);
  }, [mode]);

  useEffect(() => {
    storeValue(STORAGE_KEYS.field, field);
  }, [field]);

  useEffect(() => {
    storeValue(STORAGE_KEYS.hardware, hardware);
  }, [hardware]);

  useEffect(() => {
    storeValue(STORAGE_KEYS.overrideModel, manualOverrideModel);
  }, [manualOverrideModel]);

  const apiCall = useCallback(
    async <T,>(path: string, init?: RequestInit): Promise<T> => {
      const headers = new Headers(init?.headers ?? {});
      if (init?.body && !headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }
      if (proxyApiKey.trim()) {
        headers.set("Authorization", `Bearer ${proxyApiKey.trim()}`);
      }

      const response = await fetch(`${gatewayUrl}${path}`, {
        ...init,
        headers,
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`${response.status} ${response.statusText}: ${detail}`);
      }
      return (await response.json()) as T;
    },
    [gatewayUrl, proxyApiKey],
  );

  const syncCatalog = useCallback(async () => {
    setBusy(true);
    setStatusMessage("syncing proxy catalog...");
    appendEvent("catalog.sync", "started", `${gatewayUrl}/v1/gateway/config`);
    try {
      const [gatewayConfig, modelData, providerData, quotaData] = await Promise.all([
        apiCall<GatewayConfig>("/v1/gateway/config"),
        apiCall<ModelListingResponse>("/v1/models?enriched=false"),
        apiCall<string[]>("/v1/providers"),
        apiCall<QuotaStatsResponse>("/v1/quota-stats"),
      ]);

      setConfig(gatewayConfig);
      setMode((current) => {
        const candidate = normalizeMode(current);
        const supportsCurrent = gatewayConfig.modes.includes(candidate);
        if (supportsCurrent) {
          return candidate;
        }
        return normalizeMode(gatewayConfig.default_mode || "smart");
      });

      const nextFieldOptions = Object.keys(gatewayConfig.field_routes ?? {});
      setField((current) => {
        const normalized = current.trim();
        if (normalized && nextFieldOptions.includes(normalized)) {
          return normalized;
        }
        return nextFieldOptions[0] ?? "code";
      });

      const nextHardwareOptions = gatewayConfig.hardware_options ?? [];
      setHardware((current) => {
        const normalized = current.trim();
        if (normalized && nextHardwareOptions.includes(normalized)) {
          return normalized;
        }
        return nextHardwareOptions[0] ?? "gpu";
      });

      const catalog = Array.isArray(modelData.data)
        ? modelData.data
            .map((entry) => (typeof entry.id === "string" ? entry.id.trim() : ""))
            .filter((entry): entry is string => entry.length > 0)
        : [];
      setModels(catalog);

      setSelectedModel((current) => {
        const normalized = current.trim();
        if (normalized && catalog.includes(normalized)) {
          return normalized;
        }
        if (gatewayConfig.default_model) {
          return gatewayConfig.default_model;
        }
        return catalog[0] ?? normalized;
      });

      setProviders(Array.isArray(providerData) ? providerData : []);
      setQuotaStats(quotaData);
      setStatusMessage("catalog synced");
      appendEvent(
        "catalog.sync",
        "ok",
        `models=${catalog.length} providers=${Array.isArray(providerData) ? providerData.length : 0}`,
      );
    } catch (error: unknown) {
      const message = toErrorMessage(error);
      setStatusMessage(`sync failed: ${message}`);
      appendEvent("catalog.sync", "error", message);
    } finally {
      setBusy(false);
    }
  }, [apiCall, appendEvent, gatewayUrl]);

  useEffect(() => {
    if (initialSyncDoneRef.current) {
      return;
    }
    initialSyncDoneRef.current = true;
    void syncCatalog();
  }, [syncCatalog]);

  const applyCodexSnapshot = useCallback((snapshot: CodexAccountsResponse | undefined) => {
    if (!snapshot) {
      return;
    }
    const nextAccounts = Array.isArray(snapshot.accounts) ? snapshot.accounts : [];
    setCodexAccounts(nextAccounts);
    setCodexRotationMode(snapshot.rotation_mode === "sequential" ? "sequential" : "balanced");
    setCodexFairCycleEnabled(Boolean(snapshot.fair_cycle_enabled));
  }, []);

  const loadCodexServerEvents = useCallback(async () => {
    try {
      const payload = await apiCall<CodexEventsResponse>("/v1/openai/codex/events?limit=20");
      setCodexServerEvents(Array.isArray(payload.events) ? payload.events : []);
    } catch {
      setCodexServerEvents([]);
    }
  }, [apiCall]);

  const loadCodexAccounts = useCallback(async () => {
    setCodexBusy(true);
    setCodexStatus("syncing codex accounts...");
    appendEvent("codex.accounts.sync", "started", `${gatewayUrl}/v1/openai/codex/accounts`);
    try {
      const snapshot = await apiCall<CodexAccountsResponse>(
        "/v1/openai/codex/accounts?provider=openai",
      );
      applyCodexSnapshot(snapshot);
      const count = Array.isArray(snapshot.accounts) ? snapshot.accounts.length : 0;
      setCodexStatus(`loaded ${count} codex account${count === 1 ? "" : "s"}`);
      appendEvent("codex.accounts.sync", "ok", `accounts=${count}`);
      void loadCodexServerEvents();
    } catch (error: unknown) {
      const message = toErrorMessage(error);
      setCodexStatus(`codex accounts unavailable: ${message}`);
      appendEvent("codex.accounts.sync", "error", message);
      setCodexServerEvents([]);
    } finally {
      setCodexBusy(false);
    }
  }, [apiCall, appendEvent, applyCodexSnapshot, gatewayUrl, loadCodexServerEvents]);

  const loginCodexAccount = useCallback(async () => {
    const candidate = codexLoginKey.trim();
    if (!candidate) {
      setCodexStatus("enter an OpenAI API key first");
      return;
    }
    setCodexBusy(true);
    setCodexStatus("validating and adding codex account...");
    appendEvent("codex.login", "started", "validating OpenAI credential");
    try {
      const result = await apiCall<CodexMutationResponse>("/v1/openai/codex/login", {
        method: "POST",
        body: JSON.stringify({
          provider: "openai",
          api_key: candidate,
          persist: true,
          validate_key: true,
        }),
      });
      applyCodexSnapshot(result.accounts);
      setCodexLoginKey("");
      const codexCount = Number(result.validation?.codex_model_count ?? 0);
      const modelCount = Number(result.validation?.model_count ?? 0);
      setCodexStatus(
        codexCount > 0
          ? `account ready (${codexCount} codex models detected)`
          : `account ready (${modelCount} models detected)`,
      );
      appendEvent(
        "codex.login",
        "ok",
        `added=${Boolean(result.added)} persisted=${Boolean(result.persisted)}`,
      );
      void loadCodexServerEvents();
    } catch (error: unknown) {
      const message = toErrorMessage(error);
      setCodexStatus(`login failed: ${message}`);
      appendEvent("codex.login", "error", message);
    } finally {
      setCodexBusy(false);
    }
  }, [apiCall, appendEvent, applyCodexSnapshot, codexLoginKey, loadCodexServerEvents]);

  const removeCodexAccount = useCallback(
    async (accountId: string) => {
      if (!accountId.trim()) {
        return;
      }
      setCodexBusy(true);
      setCodexStatus("removing codex account...");
      appendEvent("codex.remove", "started", accountId);
      try {
        const result = await apiCall<CodexMutationResponse>(
          `/v1/openai/codex/accounts/${encodeURIComponent(accountId)}?provider=openai&persist=true`,
          {
            method: "DELETE",
          },
        );
        applyCodexSnapshot(result.accounts);
        setCodexStatus("account removed");
        appendEvent("codex.remove", "ok", accountId);
        void loadCodexServerEvents();
      } catch (error: unknown) {
        const message = toErrorMessage(error);
        setCodexStatus(`remove failed: ${message}`);
        appendEvent("codex.remove", "error", message);
      } finally {
        setCodexBusy(false);
      }
    },
    [apiCall, appendEvent, applyCodexSnapshot, loadCodexServerEvents],
  );

  const applyCodexCycleConfig = useCallback(async () => {
    setCodexBusy(true);
    setCodexStatus("updating cycle strategy...");
    appendEvent(
      "codex.cycle.update",
      "started",
      `mode=${codexRotationMode} fair_cycle=${String(codexFairCycleEnabled)}`,
    );
    try {
      const result = await apiCall<CodexMutationResponse>("/v1/openai/codex/cycle", {
        method: "POST",
        body: JSON.stringify({
          provider: "openai",
          rotation_mode: codexRotationMode,
          fair_cycle_enabled: codexFairCycleEnabled,
          persist: true,
        }),
      });
      applyCodexSnapshot(result.accounts);
      setCodexStatus("cycle strategy updated");
      appendEvent("codex.cycle.update", "ok", `mode=${codexRotationMode}`);
      void loadCodexServerEvents();
    } catch (error: unknown) {
      const message = toErrorMessage(error);
      setCodexStatus(`cycle update failed: ${message}`);
      appendEvent("codex.cycle.update", "error", message);
    } finally {
      setCodexBusy(false);
    }
  }, [
    apiCall,
    appendEvent,
    applyCodexSnapshot,
    codexFairCycleEnabled,
    codexRotationMode,
    loadCodexServerEvents,
  ]);

  useEffect(() => {
    void loadCodexAccounts();
  }, [loadCodexAccounts]);

  const createGatewayPayload = useCallback((): GatewayPayload => {
    const gateway: Record<string, unknown> = {
      mode,
      field,
    };

    if (hardware.trim()) {
      gateway.hardware = hardware.trim();
    }
    if (mode === "direct" && manualOverrideModel.trim()) {
      gateway.direct_model = manualOverrideModel.trim();
    }
    if (mode === "hardway" && manualOverrideModel.trim()) {
      gateway.hardway_model = manualOverrideModel.trim();
    }

    return {
      model: activeModel,
      stream: false,
      messages: [{ role: "user", content: prompt }],
      gateway,
    };
  }, [activeModel, field, hardware, manualOverrideModel, mode, prompt]);

  const previewRoute = useCallback(
    async (event: FormEvent) => {
      event.preventDefault();
      setBusy(true);
      setStatusMessage("computing route...");
      appendEvent("route.preview", "started", `mode=${mode} field=${field} model=${activeModel}`);
      try {
        const payload = createGatewayPayload();
        const route = await apiCall<GatewayDecision>("/v1/gateway/route", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        setDecision(route);
        setStatusMessage("route ready");
        appendEvent(
          "route.preview",
          "ok",
          `${route.requested_model} -> ${route.resolved_model_public || route.resolved_model}`,
        );
      } catch (error: unknown) {
        const message = toErrorMessage(error);
        setStatusMessage(`route error: ${message}`);
        appendEvent("route.preview", "error", message);
      } finally {
        setBusy(false);
      }
    },
    [activeModel, apiCall, appendEvent, createGatewayPayload, field, mode],
  );

  const runPrompt = useCallback(async () => {
    setBusy(true);
    setResponseText("");
    setStatusMessage("routing and invoking model...");
    appendEvent("prompt.run", "started", `mode=${mode} field=${field} model=${activeModel}`);
    try {
      const payload = createGatewayPayload();

      const route = await apiCall<GatewayDecision>("/v1/gateway/route", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setDecision(route);
      appendEvent(
        "prompt.route",
        "ok",
        `${route.requested_model} -> ${route.resolved_model_public || route.resolved_model}`,
      );

      const completion = await apiCall<CompletionResponse>("/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      setResponseText(parseAssistantContent(completion));
      setStatusMessage("prompt complete");
      appendEvent("prompt.run", "ok", `completion model=${route.resolved_model_public || route.resolved_model}`);
      void syncCatalog();
    } catch (error: unknown) {
      const message = toErrorMessage(error);
      setResponseText(message);
      setStatusMessage(`request failed: ${message}`);
      appendEvent("prompt.run", "error", message);
    } finally {
      setBusy(false);
    }
  }, [activeModel, apiCall, appendEvent, createGatewayPayload, field, mode, syncCatalog]);

  return (
    <section className="w-[min(96vw,31rem)] max-h-[72vh] overflow-hidden rounded-xl border border-[rgba(129,203,245,0.5)] bg-[linear-gradient(152deg,rgba(6,16,28,0.94),rgba(8,22,36,0.92),rgba(6,18,30,0.96))] shadow-[0_26px_70px_rgba(0,9,22,0.54)] backdrop-blur-[4px]">
      <header className="flex items-start justify-between gap-3 border-b border-[rgba(117,183,219,0.4)] px-3 py-2.5">
        <div>
          <p className="text-[10px] uppercase tracking-[0.13em] text-[#9fd4f5]">galaxy model dock</p>
          <p className="text-sm font-semibold text-[#e3f4ff]">Multi-account OAuth + model control</p>
          <p className="text-[10px] text-[#9abfd8]">Merged from proxy gateway interface for in-galaxy management.</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded border border-[rgba(154,198,226,0.48)] bg-[rgba(15,40,62,0.65)] px-2 py-1 text-[11px] font-semibold text-[#cfe8fb] hover:bg-[rgba(22,52,78,0.78)]"
        >
          close
        </button>
      </header>

      <div className="max-h-[calc(72vh-4.6rem)] overflow-auto px-3 py-2.5 text-[11px] text-[#d6ebfb]">
        <div className="grid gap-2">
          <label className="grid gap-1">
            <span className="uppercase tracking-[0.09em] text-[#93bfdc]">gateway url</span>
            <input
              value={gatewayUrl}
              onChange={(event) => setGatewayUrl(event.target.value)}
              placeholder="http://127.0.0.1:18000"
              className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
            />
          </label>
          <label className="grid gap-1">
            <span className="uppercase tracking-[0.09em] text-[#93bfdc]">proxy api key (optional)</span>
            <input
              value={proxyApiKey}
              onChange={(event) => setProxyApiKey(event.target.value)}
              type="password"
              placeholder="Bearer token"
              className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
            />
          </label>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                void syncCatalog();
              }}
              disabled={busy}
              className="rounded border border-[rgba(133,224,255,0.5)] bg-[rgba(43,120,155,0.34)] px-2.5 py-1 text-[11px] font-semibold text-[#e2f7ff] disabled:opacity-60"
            >
              {busy ? "syncing..." : "sync catalog"}
            </button>
            <span className="text-[10px] text-[#a0c6df]">{statusMessage}</span>
          </div>

          <div className="rounded border border-[rgba(111,173,209,0.35)] bg-[rgba(5,16,28,0.62)] p-2">
            <p className="text-[10px] uppercase tracking-[0.09em] text-[#8cb7d3]">event stream</p>
            <div className="mt-1.5 max-h-24 overflow-auto space-y-1">
              {events.length > 0 ? (
                events.slice(0, 8).map((entry) => (
                  <p key={entry.id} className="text-[10px] leading-4 text-[#cfe8fb]">
                    <span className="text-[#8eb7d2]">{entry.ts.slice(11, 19)}</span>
                    {" · "}
                    <span
                      className={
                        entry.outcome === "error"
                          ? "text-[#ffb99d]"
                          : entry.outcome === "ok"
                            ? "text-[#b8f4ce]"
                            : "text-[#9fd4f5]"
                      }
                    >
                      {entry.outcome}
                    </span>
                    {" · "}
                    <span className="text-[#dff3ff]">{entry.action}</span>
                    {" · "}
                    <span className="text-[#b7d6ea]">{entry.detail}</span>
                  </p>
                ))
              ) : (
                <p className="text-[10px] text-[#8fb5cf]">No events yet.</p>
              )}
            </div>
          </div>

          <div className="rounded border border-[rgba(118,188,228,0.35)] bg-[rgba(5,15,27,0.64)] p-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-[10px] uppercase tracking-[0.09em] text-[#9ed2f2]">
                openai codex account pool
              </p>
              <button
                type="button"
                onClick={() => {
                  void loadCodexAccounts();
                }}
                disabled={codexBusy}
                className="rounded border border-[rgba(135,206,245,0.48)] bg-[rgba(24,71,102,0.36)] px-2 py-0.5 text-[10px] font-semibold text-[#def4ff] disabled:opacity-60"
              >
                {codexBusy ? "working..." : "reload accounts"}
              </button>
            </div>

            <p className="mt-1 text-[10px] text-[#a8cbe1]">{codexStatus}</p>

            <div className="mt-2 grid gap-2 sm:grid-cols-[1fr_auto]">
              <input
                value={codexLoginKey}
                onChange={(event) => setCodexLoginKey(event.target.value)}
                type="password"
                placeholder="Paste OpenAI API key (sk-...)"
                className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
              />
              <button
                type="button"
                onClick={() => {
                  void loginCodexAccount();
                }}
                disabled={codexBusy}
                className="rounded border border-[rgba(129,219,255,0.55)] bg-[rgba(39,112,144,0.34)] px-2.5 py-1 text-[11px] font-semibold text-[#e3f6ff] disabled:opacity-60"
              >
                login + add
              </button>
            </div>

            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              <label className="grid gap-1">
                <span className="uppercase tracking-[0.08em] text-[#93bfdc]">rotation mode</span>
                <select
                  value={codexRotationMode}
                  onChange={(event) =>
                    setCodexRotationMode(event.target.value === "sequential" ? "sequential" : "balanced")
                  }
                  className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
                >
                  <option value="balanced">balanced</option>
                  <option value="sequential">sequential</option>
                </select>
              </label>

              <label className="flex items-center gap-2 rounded border border-[rgba(107,169,204,0.3)] bg-[rgba(6,18,30,0.6)] px-2 py-1.5 text-[11px] text-[#d9efff]">
                <input
                  type="checkbox"
                  checked={codexFairCycleEnabled}
                  onChange={(event) => setCodexFairCycleEnabled(event.target.checked)}
                />
                fair cycle (each account exhausts before reuse)
              </label>
            </div>

            <button
              type="button"
              onClick={() => {
                void applyCodexCycleConfig();
              }}
              disabled={codexBusy}
              className="mt-2 rounded border border-[rgba(160,221,177,0.48)] bg-[rgba(38,102,66,0.28)] px-2.5 py-1 text-[11px] font-semibold text-[#dcffe7] disabled:opacity-60"
            >
              apply cycle strategy
            </button>

            <div className="mt-2 max-h-28 overflow-auto space-y-1.5 rounded border border-[rgba(105,165,200,0.3)] bg-[rgba(4,14,24,0.6)] p-1.5">
              {codexAccounts.length > 0 ? (
                codexAccounts.map((account) => (
                  <article
                    key={account.id}
                    className="flex items-start justify-between gap-2 rounded border border-[rgba(98,158,191,0.32)] bg-[rgba(7,20,32,0.78)] px-2 py-1"
                  >
                    <div>
                      <p className="text-[11px] font-semibold text-[#def3ff]">{account.masked_key}</p>
                      <p className="text-[10px] text-[#8fb6d0]">
                        {account.status} · requests {account.requests}
                        {account.key_cooldown_remaining
                          ? ` · cooldown ${account.key_cooldown_remaining}s`
                          : ""}
                      </p>
                      <p className="text-[10px] text-[#87acc6]">last used {formatLastUsed(account.last_used_ts ?? null)}</p>
                      {account.env_key ? (
                        <p className="text-[10px] text-[#7fa3bc]">{account.env_key}</p>
                      ) : null}
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        void removeCodexAccount(account.id);
                      }}
                      disabled={codexBusy}
                      className="rounded border border-[rgba(242,162,148,0.5)] bg-[rgba(122,53,45,0.32)] px-2 py-0.5 text-[10px] font-semibold text-[#ffd9d1] disabled:opacity-60"
                    >
                      remove
                    </button>
                  </article>
                ))
              ) : (
                <p className="text-[10px] text-[#8cb3cd]">No OpenAI Codex accounts loaded yet.</p>
              )}
            </div>

            {codexServerEvents.length > 0 ? (
              <div className="mt-2 max-h-24 overflow-auto rounded border border-[rgba(109,173,210,0.28)] bg-[rgba(4,13,24,0.62)] p-1.5">
                {codexServerEvents.slice(0, 10).map((event, index) => (
                  <p key={`${event.id ?? "evt"}-${index}`} className="text-[10px] leading-4 text-[#cce8fb]">
                    <span className="text-[#90bad5]">{String(event.ts ?? "").slice(11, 19) || "--:--:--"}</span>
                    {" · "}
                    <span className="text-[#dff4ff]">{event.action ?? "event"}</span>
                    {" · "}
                    <span className="text-[#9ed3ee]">{event.outcome ?? "?"}</span>
                  </p>
                ))}
              </div>
            ) : null}
          </div>
        </div>

        <div className="mt-2 grid grid-cols-2 gap-2">
          <div className="rounded border border-[rgba(120,177,211,0.3)] bg-[rgba(4,13,22,0.62)] px-2 py-1.5">
            <p className="text-[10px] uppercase tracking-[0.08em] text-[#85b1cf]">providers</p>
            <p className="text-sm font-semibold text-[#dff2ff]">{providers.length}</p>
          </div>
          <div className="rounded border border-[rgba(120,177,211,0.3)] bg-[rgba(4,13,22,0.62)] px-2 py-1.5">
            <p className="text-[10px] uppercase tracking-[0.08em] text-[#85b1cf]">models</p>
            <p className="text-sm font-semibold text-[#dff2ff]">{models.length}</p>
          </div>
          <div className="rounded border border-[rgba(120,177,211,0.3)] bg-[rgba(4,13,22,0.62)] px-2 py-1.5">
            <p className="text-[10px] uppercase tracking-[0.08em] text-[#85b1cf]">credentials</p>
            <p className="text-sm font-semibold text-[#dff2ff]">{quotaStats?.summary?.total_credentials ?? 0}</p>
          </div>
          <div className="rounded border border-[rgba(120,177,211,0.3)] bg-[rgba(4,13,22,0.62)] px-2 py-1.5">
            <p className="text-[10px] uppercase tracking-[0.08em] text-[#85b1cf]">active creds</p>
            <p className="text-sm font-semibold text-[#dff2ff]">{quotaStats?.summary?.active_credentials ?? 0}</p>
          </div>
        </div>

        <form onSubmit={previewRoute} className="mt-3 grid gap-2">
          <label className="grid gap-1">
            <span className="uppercase tracking-[0.08em] text-[#93bfdc]">base model</span>
            <input
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
              placeholder="provider/model"
              list="galaxy-model-catalog"
              className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
            />
            <datalist id="galaxy-model-catalog">
              {models.map((entry) => (
                <option key={entry} value={entry} />
              ))}
            </datalist>
          </label>

          <div className="grid gap-2 sm:grid-cols-3">
            <label className="grid gap-1">
              <span className="uppercase tracking-[0.08em] text-[#93bfdc]">mode</span>
              <select
                value={mode}
                onChange={(event) => setMode(normalizeMode(event.target.value))}
                className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
              >
                {(config?.modes ?? ["smart", "direct", "hardway"]).map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-1">
              <span className="uppercase tracking-[0.08em] text-[#93bfdc]">field</span>
              <select
                value={field}
                onChange={(event) => setField(event.target.value)}
                className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
              >
                {(fieldOptions.length > 0 ? fieldOptions : ["general", "code", "train"]).map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-1">
              <span className="uppercase tracking-[0.08em] text-[#93bfdc]">hardware</span>
              <select
                value={hardware}
                onChange={(event) => setHardware(event.target.value)}
                className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
              >
                {(config?.hardware_options ?? ["gpu", "npu", "cpu"]).map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="grid gap-1">
            <span className="uppercase tracking-[0.08em] text-[#93bfdc]">direct/hardway override model</span>
            <input
              value={manualOverrideModel}
              onChange={(event) => setManualOverrideModel(event.target.value)}
              placeholder="provider/model"
              className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
            />
          </label>

          <label className="grid gap-1">
            <span className="uppercase tracking-[0.08em] text-[#93bfdc]">test prompt</span>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={3}
              className="rounded border border-[rgba(132,188,222,0.36)] bg-[rgba(8,23,35,0.82)] px-2 py-1 text-[12px] text-[#e3f4ff] outline-none focus:border-[rgba(164,218,248,0.72)]"
            />
          </label>

          <div className="flex flex-wrap gap-2">
            <button
              type="submit"
              disabled={busy}
              className="rounded border border-[rgba(129,219,255,0.55)] bg-[rgba(39,112,144,0.34)] px-2.5 py-1 text-[11px] font-semibold text-[#e3f6ff] disabled:opacity-60"
            >
              preview route
            </button>
            <button
              type="button"
              onClick={() => {
                void runPrompt();
              }}
              disabled={busy}
              className="rounded border border-[rgba(245,192,131,0.55)] bg-[rgba(153,94,45,0.28)] px-2.5 py-1 text-[11px] font-semibold text-[#ffe7d2] disabled:opacity-60"
            >
              run prompt
            </button>
          </div>
          <p className="text-[10px] text-[#8fb9d4]">active model: {activeModel}</p>
        </form>

        {decision ? (
          <div className="mt-3 rounded border border-[rgba(124,194,235,0.38)] bg-[rgba(4,15,25,0.62)] p-2">
            <p className="text-[10px] uppercase tracking-[0.1em] text-[#8cb9d6]">route decision</p>
            <div className="mt-1.5 grid gap-1 text-[11px] leading-5">
              <p><span className="text-[#83afcb]">requested:</span> <code>{decision.requested_model}</code></p>
              <p><span className="text-[#83afcb]">resolved:</span> <code>{decision.resolved_model_public}</code></p>
              <p><span className="text-[#83afcb]">internal:</span> <code>{decision.resolved_model}</code></p>
              <p><span className="text-[#83afcb]">reason:</span> {decision.reason}</p>
            </div>
            {decision.candidate_models.length > 0 ? (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {decision.candidate_models.map((entry) => (
                  <span
                    key={entry}
                    className="rounded-full border border-[rgba(136,196,230,0.42)] bg-[rgba(14,42,63,0.66)] px-2 py-0.5 text-[10px] text-[#d8eeff]"
                  >
                    {entry}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        {quotaRows.length > 0 ? (
          <div className="mt-3 rounded border border-[rgba(124,194,235,0.38)] bg-[rgba(4,15,25,0.62)] p-2">
            <p className="text-[10px] uppercase tracking-[0.1em] text-[#8cb9d6]">multi-account provider status</p>
            <div className="mt-2 grid gap-2">
              {quotaRows.map(([providerName, providerStats]) => {
                const credentials = providerStats.credentials ?? [];
                return (
                  <article
                    key={providerName}
                    className="rounded border border-[rgba(107,171,208,0.28)] bg-[rgba(7,23,36,0.74)] p-2"
                  >
                    <p className="text-[11px] font-semibold text-[#dff2ff]">{providerName}</p>
                    <p className="text-[10px] text-[#8fb8d2]">
                      creds {providerStats.credential_count ?? 0} · active {providerStats.active_count ?? 0}
                      {" · "}
                      cooldown {providerStats.on_cooldown_count ?? 0}
                      {" · "}
                      exhausted {providerStats.exhausted_count ?? 0}
                      {" · "}
                      cost {toMoneyLabel(providerStats.approx_cost)}
                    </p>
                    {credentials.length > 0 ? (
                      <div className="mt-1.5 grid gap-1 text-[10px] text-[#c8e4f8]">
                        {credentials.slice(0, 3).map((entry) => (
                          <p key={`${providerName}:${entry.identifier}`} className="break-all">
                            <span className="text-[#8fb8d2]">{entry.status ?? "unknown"}</span>
                            {" · "}
                            <span>{entry.identifier ?? "(credential)"}</span>
                            {entry.tier ? ` · tier ${entry.tier}` : ""}
                          </p>
                        ))}
                        {credentials.length > 3 ? (
                          <p className="text-[#86afc8]">+{credentials.length - 3} more credentials</p>
                        ) : null}
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          </div>
        ) : null}

        <div className="mt-3 rounded border border-[rgba(124,194,235,0.34)] bg-[rgba(4,13,23,0.66)] p-2">
          <p className="text-[10px] uppercase tracking-[0.1em] text-[#8cb9d6]">assistant output</p>
          <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-[#dff2ff]">
            {responseText || "no completion yet"}
          </pre>
        </div>
      </div>
    </section>
  );
}
