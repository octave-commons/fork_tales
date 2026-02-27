// SPDX-License-Identifier: GPL-3.0-or-later
// This file is part of Fork Tales.
// Copyright (C) 2024-2025 Fork Tales Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

function normalizeHttpBase(value: unknown): string {
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
    parsed.search = "";
    parsed.hash = "";
    const path = parsed.pathname.replace(/\/+$/, "");
    return path ? `${parsed.origin}${path}` : parsed.origin;
  } catch {
    return "";
  }
}

function joinBasePath(base: string, path: string): string {
  const parsed = new URL(base);
  const [pathWithHashRaw, queryPartRaw] = path.split("?", 2);
  const pathWithHash = pathWithHashRaw ?? "";
  const queryPart = queryPartRaw ?? "";
  const [cleanPathPartRaw, hashPartRaw] = pathWithHash.split("#", 2);
  const cleanPathPart = cleanPathPartRaw ?? "";
  const hashPart = hashPartRaw ?? "";
  const basePath = parsed.pathname.replace(/\/+$/, "");
  const normalizedPath = cleanPathPart.startsWith("/")
    ? cleanPathPart
    : `/${cleanPathPart}`;
  const normalizedTail = normalizedPath.replace(/^\/+/, "");
  if (!normalizedTail) {
    parsed.pathname = basePath || "/";
  } else if (basePath) {
    parsed.pathname = `${basePath}/${normalizedTail}`;
  } else {
    parsed.pathname = `/${normalizedTail}`;
  }
  parsed.search = queryPart.length > 0 ? `?${queryPart}` : "";
  parsed.hash = hashPart.length > 0 ? `#${hashPart}` : "";
  return parsed.toString();
}

function runtimeGatewayPrefixFromLocation(): string {
  if (typeof window === "undefined") {
    return "";
  }
  const match = /^\/sim\/([A-Za-z0-9._-]+)/.exec(window.location.pathname || "");
  if (!match || !match[1]) {
    return "";
  }
  return `/sim/${match[1]}`;
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
  const bridgeBase = normalizeHttpBase(runtimeBridgeConfig().worldBaseUrl);
  if (bridgeBase) {
    return bridgeBase;
  }

  const envBase = normalizeHttpBase(import.meta.env["VITE_RUNTIME_BASE_URL"]);
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

  const gatewayPrefix = runtimeGatewayPrefixFromLocation();
  if (gatewayPrefix) {
    return `${window.location.origin}${gatewayPrefix}`;
  }

  return "";
}

export function runtimeApiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = runtimeBaseUrl();
  if (!base) {
    return normalizedPath;
  }
  return joinBasePath(base, normalizedPath);
}

export function runtimeWebSocketUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const base = runtimeBaseUrl();
  if (base) {
    const wsBase = new URL(base);
    wsBase.protocol = wsBase.protocol === "https:" ? "wss:" : "ws:";
    return joinBasePath(wsBase.toString(), normalizedPath);
  }

  const protocol =
    typeof window !== "undefined" && window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = typeof window !== "undefined" ? window.location.host : "127.0.0.1:8787";
  return `${protocol}//${host}${normalizedPath}`;
}

export function runtimeWeaverBaseCandidates(): string[] {
  const bridgeWeaverBase = normalizeHttpBase(runtimeBridgeConfig().weaverBaseUrl);
  const envWeaverBase = normalizeHttpBase(import.meta.env["VITE_WEAVER_BASE_URL"]);
  const worldBase = runtimeBaseUrl();
  const prefixedWeaverBase = worldBase ? joinBasePath(worldBase, "/weaver") : "";

  const inferredFromWorld = worldBase ? originWithPort(worldBase, "8793") : "";
  const inferredFromLocation =
    typeof window !== "undefined" && window.location.hostname
      ? `${window.location.protocol === "https:" ? "https" : "http"}://${window.location.hostname}:8793`
      : "";

  return uniqueValues([
    prefixedWeaverBase,
    bridgeWeaverBase,
    envWeaverBase,
    inferredFromWorld,
    inferredFromLocation,
    DEV_WEAVER_ORIGIN,
    "http://localhost:8793",
  ]);
}

export {};
