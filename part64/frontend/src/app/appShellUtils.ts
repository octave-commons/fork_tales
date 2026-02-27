import { runtimeApiUrl } from "../runtime/endpoints";

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function resolveEventElement(target: EventTarget | null): Element | null {
  if (target instanceof Element) {
    return target;
  }
  if (target instanceof Node) {
    return target.parentElement;
  }
  return null;
}

export function isTextEntryTarget(target: EventTarget | null): boolean {
  const element = resolveEventElement(target);
  if (!(element instanceof HTMLElement)) {
    return false;
  }
  if (element.isContentEditable) {
    return true;
  }
  const tagName = element.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select";
}

export function isCorePointerBlockedTarget(target: EventTarget | null): boolean {
  const element = resolveEventElement(target);
  if (!element) {
    return false;
  }
  if (isTextEntryTarget(element)) {
    return true;
  }
  return Boolean(
    element.closest(
      "button, a, [role='button'], [data-core-pointer='block'], [data-panel-interactive='true']",
    ),
  );
}

export function shouldRouteWheelToCore(target: EventTarget | null, deltaY = 0): boolean {
  const element = resolveEventElement(target);
  if (!element) {
    return true;
  }
  if (
    element.closest(
      "input, textarea, select, option, [contenteditable='true'], [role='slider'], [data-core-wheel='block']",
    )
  ) {
    return false;
  }

  if (element.closest("button, a, [role='button'], [data-panel-interactive='true']")) {
    return false;
  }

  const panelBody = element.closest(".world-panel-body");
  if (panelBody instanceof HTMLElement) {
    const maxScrollTop = Math.max(0, panelBody.scrollHeight - panelBody.clientHeight);
    if (maxScrollTop > 1) {
      const atTop = panelBody.scrollTop <= 1;
      const atBottom = panelBody.scrollTop >= maxScrollTop - 1;
      if ((deltaY < 0 && !atTop) || (deltaY > 0 && !atBottom) || Math.abs(deltaY) < 0.5) {
        return false;
      }
    }
  }

  return true;
}

export function projectionOpacity(raw: number | undefined, floor = 0.9): number {
  const normalized = clamp(typeof raw === "number" ? raw : 1, 0, 1);
  return floor + normalized * (1 - floor);
}

export function stableUnitHash(seed: string): number {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

export function toMuseSlug(raw: string): string {
  const cleaned = String(raw || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_\s-]+/g, "")
    .replace(/[\s-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!cleaned) {
    return "";
  }
  if (/^[0-9]/.test(cleaned)) {
    return `muse_${cleaned}`;
  }
  return cleaned;
}

export function resolveRuntimeMediaUrl(rawUrl: string): string {
  const trimmed = String(rawUrl || "").trim();
  if (!trimmed) {
    return "";
  }
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  if (trimmed.startsWith("/")) {
    return runtimeApiUrl(trimmed);
  }
  return runtimeApiUrl(`/${trimmed.replace(/^\.+\//, "")}`);
}

export function normalizeDeviceUtilization(raw: unknown): number {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  const scaled = value > 1 ? value / 100 : value;
  return clamp(scaled, 0, 1);
}

interface DeviceSurroundingSimulation {
  presence_dynamics?: {
    resource_heartbeat?: {
      devices?: Record<string, { utilization?: number } | undefined>;
    };
  };
}

export function buildDeviceSurroundingNodes(simulation: DeviceSurroundingSimulation | null): Array<Record<string, unknown>> {
  const devices = simulation?.presence_dynamics?.resource_heartbeat?.devices;
  if (!devices || typeof devices !== "object") {
    return [];
  }
  return Object.entries(devices)
    .map(([deviceId, payload]) => {
      const utilization = normalizeDeviceUtilization(payload?.utilization);
      return {
        id: `device:${deviceId}`,
        kind: "device",
        label: deviceId,
        text: `${deviceId} utilization ${Math.round(utilization * 100)}%`,
        utilization,
        visibility: "public",
      };
    })
    .slice(0, 6);
}
