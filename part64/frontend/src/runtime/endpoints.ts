const DEV_WORLD_ORIGIN = "http://127.0.0.1:8787";
const DEV_WEAVER_ORIGIN = "http://127.0.0.1:8793";

interface RuntimeBridgeConfig {
  mode?: "electron" | "browser";
  worldBaseUrl?: string;
  weaverBaseUrl?: string;
}

declare global {
  interface Window {
    __ETA_MU_RUNTIME__?: RuntimeBridgeConfig;
  }
}

function uniqueValues(values: string[]): string[] {
  const seen = new Set<string>();
  const output: string[] = [];
  values.forEach((value) => {
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    output.push(trimmed);
  });
  return output;
}

function normalizeHttpOrigin(value: unknown): string {
  if (typeof value !== "string") {
    return "";
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return "";
  }
  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return "";
    }
    return parsed.origin;
  } catch {
    return "";
  }
}

function originWithPort(origin: string, port: string): string {
  try {
    const parsed = new URL(origin);
    parsed.port = port;
    parsed.pathname = "";
    parsed.search = "";
    parsed.hash = "";
    return parsed.origin;
  } catch {
    return "";
  }
}

function runtimeBridgeConfig(): RuntimeBridgeConfig {
  if (typeof window === "undefined") {
    return {};
  }
  return window.__ETA_MU_RUNTIME__ ?? {};
}

export function runtimeBaseUrl(): string {
  const bridgeBase = normalizeHttpOrigin(runtimeBridgeConfig().worldBaseUrl);
  if (bridgeBase) {
    return bridgeBase;
  }

  const envBase = normalizeHttpOrigin(import.meta.env.VITE_RUNTIME_BASE_URL);
  if (envBase) {
    return envBase;
  }

  if (typeof window === "undefined") {
    return DEV_WORLD_ORIGIN;
  }

  if (window.location.port === "5173") {
    return DEV_WORLD_ORIGIN;
  }

  if (window.location.protocol === "file:") {
    return DEV_WORLD_ORIGIN;
  }

  return "";
}

export function runtimeApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = runtimeBaseUrl();
  if (!base) {
    return normalizedPath;
  }
  return new URL(normalizedPath, `${base}/`).toString();
}

export function runtimeWebSocketUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = runtimeBaseUrl();
  if (base) {
    const wsBase = new URL(base);
    wsBase.protocol = wsBase.protocol === "https:" ? "wss:" : "ws:";
    return new URL(normalizedPath, wsBase).toString();
  }

  const protocol =
    typeof window !== "undefined" && window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = typeof window !== "undefined" ? window.location.host : "127.0.0.1:8787";
  return `${protocol}//${host}${normalizedPath}`;
}

export function runtimeWeaverBaseCandidates(): string[] {
  const bridgeWeaverBase = normalizeHttpOrigin(runtimeBridgeConfig().weaverBaseUrl);
  const envWeaverBase = normalizeHttpOrigin(import.meta.env.VITE_WEAVER_BASE_URL);
  const worldBase = runtimeBaseUrl();

  const inferredFromWorld = worldBase ? originWithPort(worldBase, "8793") : "";
  const inferredFromLocation =
    typeof window !== "undefined" && window.location.hostname
      ? `${window.location.protocol === "https:" ? "https" : "http"}://${window.location.hostname}:8793`
      : "";

  return uniqueValues([
    bridgeWeaverBase,
    envWeaverBase,
    inferredFromWorld,
    inferredFromLocation,
    DEV_WEAVER_ORIGIN,
    "http://localhost:8793",
  ]);
}

export {};
