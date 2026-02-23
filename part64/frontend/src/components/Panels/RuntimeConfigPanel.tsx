import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RefreshCw, Search, SlidersHorizontal } from "lucide-react";
import { relativeTime } from "../../app/time";
import { runtimeApiUrl } from "../../runtime/endpoints";

type RuntimeConfigValue =
  | number
  | RuntimeConfigValue[]
  | { [key: string]: RuntimeConfigValue };

interface RuntimeConfigModulePayload {
  constants: Record<string, RuntimeConfigValue>;
  constant_count: number;
  numeric_leaf_count: number;
}

interface RuntimeConfigPayload {
  ok: boolean;
  record?: string;
  runtime_config_version?: number;
  generated_at?: string;
  available_modules?: string[];
  module_count?: number;
  constant_count?: number;
  numeric_leaf_count?: number;
  modules?: Record<string, RuntimeConfigModulePayload>;
  error?: string;
}

interface RuntimeConfigMutationPayload {
  ok: boolean;
  error?: string;
  detail?: string;
  reset_count?: number;
  previous?: unknown;
  current?: unknown;
}

interface RuntimeConfigLeaf {
  moduleName: string;
  constantKey: string;
  leafId: string;
  pathTokens: string[];
  pathLabel: string;
  value: number;
  searchable: string;
}

interface RuntimeConfigEntry {
  key: string;
  value: RuntimeConfigValue;
  leafCount: number;
  preview: string;
  searchable: string;
  leaves: RuntimeConfigLeaf[];
}

interface RuntimeConfigModuleView {
  moduleName: string;
  constantCount: number;
  numericLeafCount: number;
  entries: RuntimeConfigEntry[];
}

function isRuntimeConfigMap(
  value: RuntimeConfigValue,
): value is { [key: string]: RuntimeConfigValue } {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 0.0001) || abs >= 10000) {
    return value.toExponential(4);
  }
  const compact = value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  return compact || "0";
}

function parseNumericInput(raw: string): number | null {
  const value = Number(raw.trim());
  if (!Number.isFinite(value)) {
    return null;
  }
  return value;
}

function isNumericToken(token: string): boolean {
  return /^-?\d+$/.test(token.trim());
}

function buildPathLabel(pathTokens: string[]): string {
  if (pathTokens.length <= 0) {
    return "";
  }
  let label = "";
  pathTokens.forEach((token) => {
    if (isNumericToken(token)) {
      label += `[${token}]`;
    } else {
      label += label ? `.${token}` : token;
    }
  });
  return label;
}

function buildLeafId(
  moduleName: string,
  constantKey: string,
  pathTokens: string[],
): string {
  const tail = pathTokens.join("/");
  return `${moduleName}::${constantKey}::${tail}`;
}

function flattenNumericLeaves(
  value: RuntimeConfigValue,
  options: {
    moduleName: string;
    constantKey: string;
    pathTokens?: string[];
  },
): RuntimeConfigLeaf[] {
  const moduleName = options.moduleName;
  const constantKey = options.constantKey;
  const pathTokens = options.pathTokens ?? [];
  if (typeof value === "number") {
    const pathLabel = buildPathLabel(pathTokens);
    const leafId = buildLeafId(moduleName, constantKey, pathTokens);
    return [
      {
        moduleName,
        constantKey,
        leafId,
        pathTokens,
        pathLabel,
        value,
        searchable: `${moduleName} ${constantKey} ${pathLabel} ${formatNumber(value)}`.toLowerCase(),
      },
    ];
  }

  if (Array.isArray(value)) {
    const leaves: RuntimeConfigLeaf[] = [];
    value.forEach((item, index) => {
      leaves.push(
        ...flattenNumericLeaves(item, {
          moduleName,
          constantKey,
          pathTokens: [...pathTokens, String(index)],
        }),
      );
    });
    return leaves;
  }

  if (isRuntimeConfigMap(value)) {
    const leaves: RuntimeConfigLeaf[] = [];
    Object.keys(value)
      .sort((left, right) => left.localeCompare(right))
      .forEach((key) => {
        leaves.push(
          ...flattenNumericLeaves(value[key], {
            moduleName,
            constantKey,
            pathTokens: [...pathTokens, key],
          }),
        );
      });
    return leaves;
  }

  return [];
}

function countNumericLeaves(value: RuntimeConfigValue): number {
  if (typeof value === "number") {
    return 1;
  }
  if (Array.isArray(value)) {
    let total = 0;
    value.forEach((item) => {
      total += countNumericLeaves(item);
    });
    return total;
  }
  if (!isRuntimeConfigMap(value)) {
    return 0;
  }
  let total = 0;
  Object.values(value).forEach((item) => {
    total += countNumericLeaves(item);
  });
  return total;
}

function previewRuntimeConfigValue(value: RuntimeConfigValue): string {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (Array.isArray(value)) {
    const preview = value
      .slice(0, 5)
      .map((item) => previewRuntimeConfigValue(item))
      .join(", ");
    const suffix = value.length > 5 ? `, +${value.length - 5}` : "";
    return `[${preview}${suffix}]`;
  }
  if (!isRuntimeConfigMap(value)) {
    return "{}";
  }
  const entries = Object.entries(value);
  const preview = entries
    .slice(0, 4)
    .map(([key, item]) => `${key}:${previewRuntimeConfigValue(item)}`)
    .join(", ");
  const suffix = entries.length > 4 ? `, +${entries.length - 4}` : "";
  return `{${preview}${suffix}}`;
}

function normalizeModuleFilter(raw: string, availableModules: string[]): string {
  if (!raw || raw === "all") {
    return "all";
  }
  return availableModules.includes(raw) ? raw : "all";
}

function numbersClose(left: number, right: number): boolean {
  const delta = Math.abs(left - right);
  const scale = Math.max(1, Math.abs(left), Math.abs(right));
  return delta <= (scale * 1e-8);
}

function leafSliderSpec(leaf: RuntimeConfigLeaf, draft: number | null): {
  min: number;
  max: number;
  step: number;
} {
  const center = draft ?? leaf.value;
  const signature = `${leaf.constantKey} ${leaf.pathLabel}`.toUpperCase();
  if (signature.includes("FRICTION")) {
    const clampedCenter = Math.max(0.0, Math.min(2.0, center));
    const span = Math.max(0.05, Math.abs(clampedCenter) * 0.18);
    let min = Math.max(0.0, clampedCenter - span);
    let max = Math.min(2.0, clampedCenter + span);
    if ((max - min) < 0.0002) {
      min = Math.max(0.0, clampedCenter - 0.01);
      max = Math.min(2.0, clampedCenter + 0.01);
    }
    return {
      min,
      max,
      step: clampedCenter >= 1.0 ? 0.001 : 0.0001,
    };
  }

  if (signature.includes("DAMPING")) {
    const clampedCenter = Math.max(0.0, Math.min(4.0, center));
    const span = Math.max(0.05, Math.abs(clampedCenter) * 0.2);
    let min = Math.max(0.0, clampedCenter - span);
    let max = Math.min(4.0, clampedCenter + span);
    if ((max - min) < 0.0002) {
      min = Math.max(0.0, clampedCenter - 0.01);
      max = Math.min(4.0, clampedCenter + 0.01);
    }
    return {
      min,
      max,
      step: clampedCenter >= 1.0 ? 0.001 : 0.0001,
    };
  }

  const current = leaf.value;
  const focus = Math.max(Math.abs(current), Math.abs(draft ?? current), 0.000001);
  const span = focus < 1 ? 1 : focus * 1.4;
  const min = center - span;
  const max = center + span;
  let step = 0.0001;
  if (focus >= 1000) {
    step = 5;
  } else if (focus >= 100) {
    step = 1;
  } else if (focus >= 10) {
    step = 0.1;
  } else if (focus >= 1) {
    step = 0.01;
  } else if (focus >= 0.1) {
    step = 0.001;
  }
  return { min, max, step };
}

async function postRuntimeConfigMutation(
  path: string,
  payload: Record<string, unknown>,
): Promise<RuntimeConfigMutationPayload> {
  const response = await fetch(runtimeApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = (await response.json()) as RuntimeConfigMutationPayload;
  if (!response.ok || data.ok !== true) {
    return {
      ok: false,
      error: String(data.error || `request failed (${response.status})`),
      detail: String(data.detail || ""),
    };
  }
  return data;
}

function mutationNumericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return null;
}

export function RuntimeConfigPanel() {
  const [payload, setPayload] = useState<RuntimeConfigPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [moduleFilter, setModuleFilter] = useState("all");
  const [draftByLeafId, setDraftByLeafId] = useState<Record<string, string>>({});
  const [mutationMessage, setMutationMessage] = useState("");
  const [mutationError, setMutationError] = useState("");
  const [activeMutationLeafId, setActiveMutationLeafId] = useState("");
  const [bulkMutating, setBulkMutating] = useState(false);
  const refreshRequestSeqRef = useRef(0);

  const refreshConfig = useCallback(async (withSpinner = true) => {
    const requestSeq = refreshRequestSeqRef.current + 1;
    refreshRequestSeqRef.current = requestSeq;
    if (withSpinner) {
      setLoading(true);
    }
    setError("");
    try {
      const response = await fetch(runtimeApiUrl("/api/config"));
      const data = (await response.json()) as RuntimeConfigPayload;
      if (!response.ok || data.ok !== true) {
        throw new Error(String(data.error || `config request failed (${response.status})`));
      }
      if (!data.modules || typeof data.modules !== "object") {
        throw new Error("invalid config payload");
      }
      if (requestSeq !== refreshRequestSeqRef.current) {
        return;
      }
      setPayload(data);
      const availableModules = Array.isArray(data.available_modules)
        ? data.available_modules.map((item) => String(item || "")).filter(Boolean)
        : [];
      setModuleFilter((previous) => normalizeModuleFilter(previous, availableModules));
    } catch (fetchError) {
      if (requestSeq !== refreshRequestSeqRef.current) {
        return;
      }
      const message = fetchError instanceof Error ? fetchError.message : "config fetch failed";
      setError(message);
    } finally {
      if (withSpinner && requestSeq === refreshRequestSeqRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    void refreshConfig(true);
    const interval = window.setInterval(() => {
      if (bulkMutating || activeMutationLeafId.length > 0 || Object.keys(draftByLeafId).length > 0) {
        return;
      }
      void refreshConfig(false);
    }, 10000);
    return () => {
      window.clearInterval(interval);
    };
  }, [activeMutationLeafId, bulkMutating, draftByLeafId, refreshConfig]);

  const availableModules = useMemo(() => {
    if (!payload?.available_modules || !Array.isArray(payload.available_modules)) {
      return [];
    }
    return payload.available_modules.map((item) => String(item || "")).filter(Boolean);
  }, [payload?.available_modules]);

  const normalizedModuleFilter = normalizeModuleFilter(moduleFilter, availableModules);
  const normalizedSearch = searchQuery.trim().toLowerCase();

  const moduleViews = useMemo<RuntimeConfigModuleView[]>(() => {
    const modules = payload?.modules;
    if (!modules || typeof modules !== "object") {
      return [];
    }

    return Object.entries(modules)
      .sort(([left], [right]) => left.localeCompare(right))
      .filter(([moduleName]) => normalizedModuleFilter === "all" || moduleName === normalizedModuleFilter)
      .map(([moduleName, modulePayload]) => {
        const constants = modulePayload?.constants ?? {};
        const entries: RuntimeConfigEntry[] = Object.entries(constants)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, value]) => {
            const preview = previewRuntimeConfigValue(value);
            const leaves = flattenNumericLeaves(value, {
              moduleName,
              constantKey: key,
            });
            const leafSearchBlob = leaves
              .slice(0, 64)
              .map((leaf) => `${leaf.pathLabel} ${formatNumber(leaf.value)}`)
              .join(" ");
            return {
              key,
              value,
              leafCount: countNumericLeaves(value),
              preview,
              searchable: `${moduleName} ${key} ${preview} ${leafSearchBlob}`.toLowerCase(),
              leaves,
            };
          })
          .filter((entry) => !normalizedSearch || entry.searchable.includes(normalizedSearch));

        return {
          moduleName,
          constantCount: Number(modulePayload?.constant_count ?? 0),
          numericLeafCount: Number(modulePayload?.numeric_leaf_count ?? 0),
          entries,
        };
      })
      .filter((moduleView) => moduleView.entries.length > 0 || !normalizedSearch);
  }, [normalizedModuleFilter, normalizedSearch, payload?.modules]);

  const leafById = useMemo(() => {
    const map = new Map<string, RuntimeConfigLeaf>();
    moduleViews.forEach((moduleView) => {
      moduleView.entries.forEach((entry) => {
        entry.leaves.forEach((leaf) => {
          map.set(leaf.leafId, leaf);
        });
      });
    });
    return map;
  }, [moduleViews]);

  const matchedConstantCount = useMemo(
    () => moduleViews.reduce((sum, moduleView) => sum + moduleView.entries.length, 0),
    [moduleViews],
  );

  const matchedLeafCount = useMemo(
    () => moduleViews.reduce(
      (sum, moduleView) => sum + moduleView.entries.reduce((entrySum, entry) => entrySum + entry.leaves.length, 0),
      0,
    ),
    [moduleViews],
  );

  const editedLeafIds = useMemo(() => {
    const edited: string[] = [];
    Object.entries(draftByLeafId).forEach(([leafId, draftValue]) => {
      const leaf = leafById.get(leafId);
      if (!leaf) {
        return;
      }
      const parsed = parseNumericInput(draftValue);
      if (parsed === null) {
        return;
      }
      if (!numbersClose(parsed, leaf.value)) {
        edited.push(leafId);
      }
    });
    return edited;
  }, [draftByLeafId, leafById]);

  const setLeafDraft = useCallback((leafId: string, nextValue: number) => {
    setDraftByLeafId((previous) => ({
      ...previous,
      [leafId]: formatNumber(nextValue),
    }));
  }, []);

  const applyLeaf = useCallback(async (
    leaf: RuntimeConfigLeaf,
    nextValue: number,
  ) => {
    setActiveMutationLeafId(leaf.leafId);
    setMutationError("");
    setMutationMessage("");
    const result = await postRuntimeConfigMutation("/api/config/update", {
      module: leaf.moduleName,
      key: leaf.constantKey,
      path: leaf.pathTokens,
      value: nextValue,
    });
    if (!result.ok) {
      setMutationError(String(result.error || "update failed"));
      setActiveMutationLeafId("");
      return;
    }
    setDraftByLeafId((previous) => {
      const next = { ...previous };
      delete next[leaf.leafId];
      return next;
    });
    const currentValue = mutationNumericValue(result.current);
    setMutationMessage(
      currentValue === null
        ? `updated ${leaf.moduleName}.${leaf.constantKey}${leaf.pathLabel ? `.${leaf.pathLabel}` : ""}`
        : `updated ${leaf.moduleName}.${leaf.constantKey}${leaf.pathLabel ? `.${leaf.pathLabel}` : ""} -> ${formatNumber(currentValue)}`,
    );
    await refreshConfig(false);
    setActiveMutationLeafId("");
  }, [refreshConfig]);

  const resetLeaf = useCallback(async (leaf: RuntimeConfigLeaf) => {
    setActiveMutationLeafId(leaf.leafId);
    setMutationError("");
    setMutationMessage("");
    const result = await postRuntimeConfigMutation("/api/config/reset", {
      module: leaf.moduleName,
      key: leaf.constantKey,
      path: leaf.pathTokens,
    });
    if (!result.ok) {
      setMutationError(String(result.error || "reset failed"));
      setActiveMutationLeafId("");
      return;
    }
    setDraftByLeafId((previous) => {
      const next = { ...previous };
      delete next[leaf.leafId];
      return next;
    });
    setMutationMessage(`reset ${leaf.moduleName}.${leaf.constantKey}${leaf.pathLabel ? `.${leaf.pathLabel}` : ""}`);
    await refreshConfig(false);
    setActiveMutationLeafId("");
  }, [refreshConfig]);

  const applyEdited = useCallback(async () => {
    if (editedLeafIds.length <= 0) {
      return;
    }
    setBulkMutating(true);
    setMutationError("");
    setMutationMessage("");
    let applied = 0;
    for (const leafId of editedLeafIds) {
      const leaf = leafById.get(leafId);
      if (!leaf) {
        continue;
      }
      const parsed = parseNumericInput(draftByLeafId[leafId] ?? "");
      if (parsed === null) {
        continue;
      }
      const result = await postRuntimeConfigMutation("/api/config/update", {
        module: leaf.moduleName,
        key: leaf.constantKey,
        path: leaf.pathTokens,
        value: parsed,
      });
      if (!result.ok) {
        setMutationError(String(result.error || `update failed on ${leaf.constantKey}`));
        setBulkMutating(false);
        return;
      }
      applied += 1;
    }

    if (applied > 0) {
      setDraftByLeafId((previous) => {
        const next = { ...previous };
        editedLeafIds.forEach((leafId) => {
          delete next[leafId];
        });
        return next;
      });
    }
    setMutationMessage(`applied ${applied} edited values`);
    await refreshConfig(false);
    setBulkMutating(false);
  }, [draftByLeafId, editedLeafIds, leafById, refreshConfig]);

  const resetAll = useCallback(async () => {
    setBulkMutating(true);
    setMutationError("");
    setMutationMessage("");
    const result = await postRuntimeConfigMutation("/api/config/reset", {});
    if (!result.ok) {
      setMutationError(String(result.error || "reset all failed"));
      setBulkMutating(false);
      return;
    }
    setDraftByLeafId({});
    setMutationMessage(`reset ${Number(result.reset_count ?? 0)} values to defaults`);
    await refreshConfig(false);
    setBulkMutating(false);
  }, [refreshConfig]);

  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-[rgba(174,129,255,0.34)] bg-[rgba(39,40,34,0.9)] p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-ink flex items-center gap-2">
              <SlidersHorizontal size={15} />
              Runtime Config Interface
            </p>
            <p className="text-xs text-muted mt-1">
              Live controls for numeric constants exposed by <code>/api/config</code>.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => {
                void refreshConfig(true);
              }}
              className="border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(55,56,48,0.92)]"
            >
              <span className="inline-flex items-center gap-1.5">
                <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
                Refresh
              </span>
            </button>
            <button
              type="button"
              onClick={() => {
                void applyEdited();
              }}
              disabled={editedLeafIds.length === 0 || bulkMutating}
              className="border border-[var(--line)] rounded-md bg-[rgba(44,67,39,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(63,94,56,0.92)] disabled:opacity-50"
            >
              Apply Edited ({editedLeafIds.length})
            </button>
            <button
              type="button"
              onClick={() => {
                void resetAll();
              }}
              disabled={bulkMutating}
              className="border border-[var(--line)] rounded-md bg-[rgba(73,45,45,0.9)] px-3 py-1.5 text-xs font-semibold text-ink hover:bg-[rgba(96,58,58,0.92)] disabled:opacity-50"
            >
              Reset Runtime Defaults
            </button>
          </div>
        </div>

        <div className="mt-3 grid gap-2 sm:grid-cols-4">
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">modules</p>
            <p className="text-sm font-semibold text-ink">{payload?.module_count ?? 0}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">constants</p>
            <p className="text-sm font-semibold text-ink">{payload?.constant_count ?? 0}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">numeric leaves</p>
            <p className="text-sm font-semibold text-ink">{payload?.numeric_leaf_count ?? 0}</p>
          </div>
          <div className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.84)] px-3 py-2">
            <p className="text-[10px] uppercase tracking-wide text-muted">matched leaves</p>
            <p className="text-sm font-semibold text-ink">{matchedLeafCount}</p>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <label className="text-[11px] text-muted" htmlFor="runtime-config-module-filter">
            module
          </label>
          <select
            id="runtime-config-module-filter"
            value={normalizedModuleFilter}
            onChange={(event) => {
              setModuleFilter(event.currentTarget.value);
            }}
            className="border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.94)] px-2 py-1 text-xs text-ink"
          >
            <option value="all">all modules</option>
            {availableModules.map((moduleName) => (
              <option key={moduleName} value={moduleName}>
                {moduleName}
              </option>
            ))}
          </select>

          <div className="inline-flex items-center gap-1 border border-[var(--line)] rounded-md bg-[rgba(31,32,29,0.94)] px-2 py-1">
            <Search size={12} className="text-muted" />
            <input
              value={searchQuery}
              onChange={(event) => {
                setSearchQuery(event.currentTarget.value);
              }}
              placeholder="search constants and leaves"
              className="bg-transparent text-xs text-ink outline-none w-[20rem] max-w-[56vw]"
            />
          </div>
        </div>

        <p className="text-[11px] text-muted mt-2">
          matches <code>{matchedConstantCount}</code> constants · <code>{matchedLeafCount}</code> leaves
          {payload?.generated_at ? (
            <>
              {" "}| refreshed <code>{relativeTime(payload.generated_at)}</code>
            </>
          ) : null}
          {payload?.record ? (
            <>
              {" "}| record <code>{payload.record}</code>
            </>
          ) : null}
          {typeof payload?.runtime_config_version === "number" ? (
            <>
              {" "}| version <code>{payload.runtime_config_version}</code>
            </>
          ) : null}
        </p>

        {mutationMessage ? <p className="text-[11px] text-[#b6f0c0] mt-2">{mutationMessage}</p> : null}
        {mutationError ? <p className="text-[11px] text-[#ffcfbf] mt-2">{mutationError}</p> : null}
        {error ? <p className="text-[11px] text-[#ffcfbf] mt-2">{error}</p> : null}
      </div>

      <div className="space-y-2 max-h-[36rem] overflow-y-auto pr-1">
        {moduleViews.length === 0 ? (
          <p className="text-xs text-muted">No constants matched this filter yet.</p>
        ) : (
          moduleViews.map((moduleView) => (
            <section
              key={moduleView.moduleName}
              className="rounded-lg border border-[var(--line)] bg-[rgba(31,32,29,0.86)] p-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-semibold text-ink">
                  <code>{moduleView.moduleName}</code>
                </p>
                <p className="text-[11px] text-muted">
                  constants <code>{moduleView.constantCount}</code> | leaves <code>{moduleView.numericLeafCount}</code>
                </p>
              </div>

              <div className="mt-2 space-y-2">
                {moduleView.entries.length === 0 ? (
                  <p className="text-xs text-muted">No constants matched in this module.</p>
                ) : (
                  moduleView.entries.map((entry) => (
                    <details
                      key={`${moduleView.moduleName}:${entry.key}`}
                      className="rounded-md border border-[rgba(126,166,192,0.26)] bg-[rgba(18,20,18,0.72)] px-3 py-2"
                    >
                      <summary className="cursor-pointer list-none">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <p className="text-xs font-semibold text-[#d9ecff]">
                            <code>{entry.key}</code>
                          </p>
                          <p className="text-[11px] text-[#9ec7dd]">
                            leaves <code>{entry.leafCount}</code> | {entry.preview}
                          </p>
                        </div>
                      </summary>

                      <div className="mt-2 space-y-2">
                        {entry.leaves.length === 0 ? (
                          <p className="text-[11px] text-muted">No numeric leaves found.</p>
                        ) : (
                          entry.leaves.map((leaf) => {
                            const draftText = draftByLeafId[leaf.leafId] ?? formatNumber(leaf.value);
                            const parsedDraft = parseNumericInput(draftText);
                            const liveValue = parsedDraft ?? leaf.value;
                            const dirty = parsedDraft !== null && !numbersClose(parsedDraft, leaf.value);
                            const slider = leafSliderSpec(leaf, parsedDraft);
                            const sliderValue = Math.max(
                              slider.min,
                              Math.min(slider.max, liveValue),
                            );
                            const displayRef = `${leaf.constantKey}${leaf.pathLabel ? `.${leaf.pathLabel}` : ""}`;
                            const canMutate = !bulkMutating && activeMutationLeafId.length === 0;
                            return (
                              <div
                                key={leaf.leafId}
                                className="rounded-md border border-[rgba(126,166,192,0.2)] bg-[rgba(12,16,20,0.66)] p-2"
                              >
                                <div className="flex flex-wrap items-center justify-between gap-2">
                                  <p className="text-[11px] text-[#d3e8ff] font-semibold">
                                    <code>{displayRef}</code>
                                  </p>
                                  <p className="text-[11px] text-[#9ec7dd]">
                                    current <code>{formatNumber(leaf.value)}</code>
                                  </p>
                                </div>

                                <div className="mt-2 grid gap-2 md:grid-cols-[auto_1fr_auto_auto_auto_auto] items-center">
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setLeafDraft(leaf.leafId, sliderValue - slider.step);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    -
                                  </button>
                                  <input
                                    type="range"
                                    min={slider.min}
                                    max={slider.max}
                                    step={slider.step}
                                    value={sliderValue}
                                    onChange={(event) => {
                                      setLeafDraft(leaf.leafId, Number(event.currentTarget.value));
                                    }}
                                    className="w-full accent-[rgb(126,188,222)]"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setLeafDraft(leaf.leafId, sliderValue + slider.step);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    +
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setLeafDraft(leaf.leafId, sliderValue * 0.5);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    x0.5
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setLeafDraft(leaf.leafId, sliderValue * 2);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    x2
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setDraftByLeafId((previous) => {
                                        const next = { ...previous };
                                        delete next[leaf.leafId];
                                        return next;
                                      });
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[var(--line)] rounded px-2 py-1 text-xs text-ink hover:bg-[rgba(40,52,68,0.62)] disabled:opacity-50"
                                  >
                                    clear
                                  </button>
                                </div>

                                <div className="mt-2 grid gap-2 md:grid-cols-[1fr_auto_auto]">
                                  <input
                                    value={draftText}
                                    onChange={(event) => {
                                      setDraftByLeafId((previous) => ({
                                        ...previous,
                                        [leaf.leafId]: event.currentTarget.value,
                                      }));
                                    }}
                                    className="border border-[var(--line)] rounded-md bg-[rgba(26,29,31,0.94)] px-2 py-1 text-xs text-ink"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => {
                                      if (parsedDraft === null) {
                                        return;
                                      }
                                      void applyLeaf(leaf, parsedDraft);
                                    }}
                                    disabled={!dirty || parsedDraft === null || !canMutate}
                                    className="border border-[rgba(126,196,156,0.5)] rounded-md bg-[rgba(35,70,50,0.9)] px-3 py-1 text-xs font-semibold text-[#def7ea] hover:bg-[rgba(49,96,69,0.92)] disabled:opacity-50"
                                  >
                                    {activeMutationLeafId === leaf.leafId ? "applying..." : "apply"}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => {
                                      void resetLeaf(leaf);
                                    }}
                                    disabled={!canMutate}
                                    className="border border-[rgba(202,150,134,0.48)] rounded-md bg-[rgba(70,42,39,0.88)] px-3 py-1 text-xs font-semibold text-[#ffe0d7] hover:bg-[rgba(96,56,52,0.9)] disabled:opacity-50"
                                  >
                                    {activeMutationLeafId === leaf.leafId ? "resetting..." : "reset"}
                                  </button>
                                </div>
                              </div>
                            );
                          })
                        )}

                        <details className="rounded-md border border-[rgba(126,166,192,0.2)] bg-[rgba(11,14,18,0.62)] px-2 py-1">
                          <summary className="cursor-pointer text-[11px] text-muted">raw constant json</summary>
                          <pre className="mt-1 text-[11px] text-[#c7e6ff] whitespace-pre-wrap break-all">
                            {JSON.stringify(entry.value, null, 2)}
                          </pre>
                        </details>
                      </div>
                    </details>
                  ))
                )}
              </div>
            </section>
          ))
        )}
      </div>
    </div>
  );
}
