import process from "node:process";
import { contextBridge } from "electron";

const DEFAULT_WORLD_RUNTIME_URL = "http://127.0.0.1:8787";
const DEFAULT_WEAVER_RUNTIME_URL = "http://127.0.0.1:8793";

function normalizeHttpOrigin(value, fallback) {
  const raw = String(value || "").trim();
  if (!raw) {
    return fallback;
  }
  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return fallback;
    }
    parsed.pathname = "";
    parsed.search = "";
    parsed.hash = "";
    return parsed.origin;
  } catch {
    return fallback;
  }
}

const worldBaseUrl = normalizeHttpOrigin(process.env.ETA_MU_WORLD_BASE_URL, DEFAULT_WORLD_RUNTIME_URL);
const weaverBaseUrl = normalizeHttpOrigin(process.env.ETA_MU_WEAVER_BASE_URL, DEFAULT_WEAVER_RUNTIME_URL);

contextBridge.exposeInMainWorld(
  "__ETA_MU_RUNTIME__",
  Object.freeze({
    mode: "electron",
    worldBaseUrl,
    weaverBaseUrl,
  }),
);
