import { FormEvent, useEffect, useMemo, useState } from "react";

type GatewayConfig = {
  enabled: boolean;
  default_mode: "smart" | "direct" | "hardway";
  default_model: string;
  modes: string[];
  hardware_options: string[];
  field_routes: Record<string, Record<string, string>>;
};

type GatewayDecision = {
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
};

type CompletionResponse = {
  choices?: Array<{
    message?: {
      content?: string | Array<{ type: string; text?: string }>;
    };
  }>;
  error?: {
    message?: string;
  };
  detail?: unknown;
};

const DEFAULT_BASE_URL =
  import.meta.env.VITE_PROXY_BASE_URL ?? "http://localhost:18000";

function parseAssistantContent(payload: CompletionResponse): string {
  const content = payload.choices?.[0]?.message?.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const textBlocks = content
      .filter((entry) => entry.type === "text" && entry.text)
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
    return error.message;
  }
  return "Unexpected error";
}

export default function App() {
  const [gatewayUrl, setGatewayUrl] = useState(
    () => localStorage.getItem("promethean.gatewayUrl") ?? DEFAULT_BASE_URL,
  );
  const [proxyApiKey, setProxyApiKey] = useState(
    () => localStorage.getItem("promethean.proxyApiKey") ?? "",
  );

  const [config, setConfig] = useState<GatewayConfig | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [providers, setProviders] = useState<string[]>([]);

  const [mode, setMode] = useState<"smart" | "direct" | "hardway">("smart");
  const [field, setField] = useState("code");
  const [hardware, setHardware] = useState("gpu");
  const [selectedModel, setSelectedModel] = useState("octave-commons/promethean");
  const [manualOverrideModel, setManualOverrideModel] = useState("");
  const [prompt, setPrompt] = useState(
    "Design a training-ready model routing plan for a mixed GPU/NPU deployment.",
  );

  const [decision, setDecision] = useState<GatewayDecision | null>(null);
  const [responseText, setResponseText] = useState("");
  const [busy, setBusy] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Idle");

  const fieldOptions = useMemo(
    () => Object.keys(config?.field_routes ?? {}).sort(),
    [config],
  );

  const activeModel = selectedModel || config?.default_model || "octave-commons/promethean";

  useEffect(() => {
    localStorage.setItem("promethean.gatewayUrl", gatewayUrl);
  }, [gatewayUrl]);

  useEffect(() => {
    localStorage.setItem("promethean.proxyApiKey", proxyApiKey);
  }, [proxyApiKey]);

  async function apiCall(path: string, init?: RequestInit): Promise<any> {
    const headers = new Headers(init?.headers ?? {});
    headers.set("Content-Type", "application/json");
    if (proxyApiKey.trim()) {
      headers.set("Authorization", `Bearer ${proxyApiKey.trim()}`);
    }

    const result = await fetch(`${gatewayUrl}${path}`, {
      ...init,
      headers,
    });

    if (!result.ok) {
      const body = await result.text();
      throw new Error(`${result.status} ${result.statusText}: ${body}`);
    }

    return result.json();
  }

  async function loadCatalog() {
    setBusy(true);
    setStatusMessage("Syncing gateway catalog...");
    try {
      const [gatewayConfig, modelData, providerData] = await Promise.all([
        apiCall("/v1/gateway/config"),
        apiCall("/v1/models?enriched=false"),
        apiCall("/v1/providers"),
      ]);

      setConfig(gatewayConfig);
      setMode(gatewayConfig.default_mode ?? "smart");
      setField(Object.keys(gatewayConfig.field_routes ?? {})[0] ?? "general");

      const catalog = Array.isArray(modelData?.data)
        ? modelData.data
            .map((entry: { id?: string }) => entry.id)
            .filter((entry: string | undefined): entry is string => Boolean(entry))
        : [];

      setModels(catalog);
      setProviders(Array.isArray(providerData) ? providerData : []);

      if (catalog.length > 0) {
        setSelectedModel(catalog[0]);
      } else if (gatewayConfig.default_model) {
        setSelectedModel(gatewayConfig.default_model);
      }

      setStatusMessage("Catalog synced");
    } catch (error) {
      setStatusMessage(`Catalog sync failed: ${toErrorMessage(error)}`);
    } finally {
      setBusy(false);
    }
  }

  function createGatewayPayload() {
    const gatewayBlock: Record<string, unknown> = {
      mode,
      field,
    };

    if (hardware) {
      gatewayBlock.hardware = hardware;
    }

    if (mode === "direct" && manualOverrideModel.trim()) {
      gatewayBlock.direct_model = manualOverrideModel.trim();
    }

    if (mode === "hardway" && manualOverrideModel.trim()) {
      gatewayBlock.hardway_model = manualOverrideModel.trim();
    }

    return {
      model: activeModel,
      stream: false,
      messages: [{ role: "user", content: prompt }],
      gateway: gatewayBlock,
    };
  }

  async function previewRoute(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setStatusMessage("Computing route...");
    try {
      const payload = createGatewayPayload();
      const result = (await apiCall("/v1/gateway/route", {
        method: "POST",
        body: JSON.stringify(payload),
      })) as GatewayDecision;

      setDecision(result);
      setStatusMessage("Route computed");
    } catch (error) {
      setStatusMessage(`Route error: ${toErrorMessage(error)}`);
    } finally {
      setBusy(false);
    }
  }

  async function runPrompt() {
    setBusy(true);
    setResponseText("");
    setStatusMessage("Routing and invoking model...");

    try {
      const payload = createGatewayPayload();

      const routed = (await apiCall("/v1/gateway/route", {
        method: "POST",
        body: JSON.stringify(payload),
      })) as GatewayDecision;
      setDecision(routed);

      const completion = (await apiCall("/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify(payload),
      })) as CompletionResponse;

      setResponseText(parseAssistantContent(completion));
      setStatusMessage("Completed");
    } catch (error) {
      setStatusMessage(`Request failed: ${toErrorMessage(error)}`);
      setResponseText(toErrorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <div className="backdrop" />
      <header className="hero card rise-1">
        <div>
          <p className="eyebrow">Promethean Control Plane</p>
          <h1>Smart Gateway and Field Router</h1>
          <p className="subtitle">
            Route requests by field, hardware, and mode using provider/model format.
            Promethean namespace: <code>octave-commons/promethean</code>.
          </p>
        </div>
        <div className="status-pill">{statusMessage}</div>
      </header>

      <main className="layout">
        <section className="card panel rise-2">
          <h2>Connection</h2>
          <label>
            Gateway URL
            <input
              value={gatewayUrl}
              onChange={(event) => setGatewayUrl(event.target.value)}
              placeholder="http://localhost:18000"
            />
          </label>
          <label>
            Proxy API Key
            <input
              value={proxyApiKey}
              onChange={(event) => setProxyApiKey(event.target.value)}
              placeholder="Bearer token value"
              type="password"
            />
          </label>
          <button type="button" onClick={loadCatalog} disabled={busy}>
            Sync Catalog
          </button>

          <div className="mini-grid">
            <div>
              <span className="mini-label">Providers</span>
              <span className="mini-value">{providers.length}</span>
            </div>
            <div>
              <span className="mini-label">Models</span>
              <span className="mini-value">{models.length}</span>
            </div>
          </div>
        </section>

        <section className="card panel rise-3">
          <h2>Router Composer</h2>
          <form onSubmit={previewRoute}>
            <label>
              Base model
              <input
                list="model-catalog"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                placeholder="provider/model"
              />
              <datalist id="model-catalog">
                {models.map((model) => (
                  <option key={model} value={model} />
                ))}
              </datalist>
            </label>

            <label>
              Mode
              <select value={mode} onChange={(event) => setMode(event.target.value as any)}>
                {(config?.modes ?? ["smart", "direct", "hardway"]).map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Field
              <select value={field} onChange={(event) => setField(event.target.value)}>
                {(fieldOptions.length > 0 ? fieldOptions : ["general", "code", "train"]).map(
                  (entry) => (
                    <option key={entry} value={entry}>
                      {entry}
                    </option>
                  ),
                )}
              </select>
            </label>

            <label>
              Hardware target
              <select value={hardware} onChange={(event) => setHardware(event.target.value)}>
                {(config?.hardware_options ?? ["gpu", "npu", "openvino", "tensorflow", "cpu"]).map(
                  (entry) => (
                    <option key={entry} value={entry}>
                      {entry}
                    </option>
                  ),
                )}
              </select>
            </label>

            <label>
              Direct/hardway override model
              <input
                value={manualOverrideModel}
                onChange={(event) => setManualOverrideModel(event.target.value)}
                placeholder="provider/model"
              />
            </label>

            <label>
              Prompt
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                rows={5}
              />
            </label>

            <div className="button-row">
              <button type="submit" disabled={busy}>
                Preview Route
              </button>
              <button type="button" className="ghost" onClick={runPrompt} disabled={busy}>
                Run Prompt
              </button>
            </div>
          </form>
        </section>

        <section className="card panel rise-4">
          <h2>Live Decision</h2>
          {decision ? (
            <>
              <div className="decision-grid">
                <div>
                  <span>Requested</span>
                  <code>{decision.requested_model}</code>
                </div>
                <div>
                  <span>Resolved</span>
                  <code>{decision.resolved_model_public}</code>
                </div>
                <div>
                  <span>Internal</span>
                  <code>{decision.resolved_model}</code>
                </div>
                <div>
                  <span>Reason</span>
                  <p>{decision.reason}</p>
                </div>
              </div>

              <div className="chips">
                <span className="chip">mode: {decision.mode}</span>
                <span className="chip">field: {decision.field}</span>
                <span className="chip">
                  hardware: {decision.hardware ?? "auto"}
                </span>
                <span className={`chip ${decision.provider_available ? "ok" : "warn"}`}>
                  provider: {decision.provider_available ? "available" : "not loaded"}
                </span>
              </div>

              <h3>Candidate Models</h3>
              <div className="chips">
                {decision.candidate_models.map((entry) => (
                  <span key={entry} className="chip model-chip">
                    {entry}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <p className="muted">Run "Preview Route" to inspect the smart gateway decision.</p>
          )}
        </section>

        <section className="card panel response-panel rise-5">
          <h2>Assistant Output</h2>
          <pre>{responseText || "No response yet."}</pre>
        </section>
      </main>
    </div>
  );
}
