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

import { useState, useEffect, useRef, useCallback } from 'react';
import { runtimeApiUrl, runtimeWebSocketUrl } from '../runtime/endpoints';
import type {
  Catalog,
  MuseEvent,
  SimulationState,
  MixMeta,
  UIProjectionBundle,
  UIPerspective,
} from '../types';

interface WorldState {
  catalog: Catalog | null;
  simulation: SimulationState | null;
  mixMeta: MixMeta | null;
  projection: UIProjectionBundle | null;
  museEvents: MuseEvent[];
  isConnected: boolean;
}

const WS_WIRE_ARRAY_SCHEMA = 'eta-mu.ws.arr.v1';
const WS_PACK_TAG_OBJECT = -1;
const WS_PACK_TAG_ARRAY = -2;
const WS_PACK_TAG_STRING = -3;
const WS_PACK_TAG_BOOL = -4;
const WS_PACK_TAG_NULL = -5;

type IncomingWsMessage = { type?: unknown; [key: string]: unknown };

type WsChunkAssembly = {
  chunkTotal: number;
  parts: string[];
  receivedCount: number;
  updatedAtMs: number;
};

const WS_CHUNK_TTL_MS = 15_000;
const WS_CHUNK_CLEANUP_INTERVAL_MS = 2_000;
const WS_CHUNK_MAX_ACTIVE = 96;
const CATALOG_STREAM_CHUNK_ROWS = 128;

type CatalogStreamSection =
  | 'items'
  | 'file_nodes'
  | 'file_edges'
  | 'file_embed_layers'
  | 'crawler_nodes'
  | 'crawler_edges';

const CATALOG_STREAM_SECTION_PATHS: Record<CatalogStreamSection, string[]> = {
  items: ['items'],
  file_nodes: ['file_graph', 'file_nodes'],
  file_edges: ['file_graph', 'edges'],
  file_embed_layers: ['file_graph', 'embed_layers'],
  crawler_nodes: ['crawler_graph', 'crawler_nodes'],
  crawler_edges: ['crawler_graph', 'edges'],
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function catalogStreamSectionPath(section: string): string[] | null {
  const clean = String(section || '').trim().toLowerCase() as CatalogStreamSection;
  return CATALOG_STREAM_SECTION_PATHS[clean] ?? null;
}

function catalogStreamArrayTarget(catalog: Record<string, unknown>, section: string): unknown[] | null {
  const path = catalogStreamSectionPath(section);
  if (!path || path.length <= 0) {
    return null;
  }

  let cursor: Record<string, unknown> = catalog;
  for (let index = 0; index < path.length - 1; index += 1) {
    const key = path[index];
    const nested = cursor[key];
    if (!isRecord(nested)) {
      const nextNested: Record<string, unknown> = {};
      cursor[key] = nextNested;
      cursor = nextNested;
    } else {
      cursor = nested;
    }
  }

  const leafKey = path[path.length - 1];
  const leafValue = cursor[leafKey];
  if (Array.isArray(leafValue)) {
    return leafValue;
  }
  const nextRows: unknown[] = [];
  cursor[leafKey] = nextRows;
  return nextRows;
}

function createCatalogStreamDraft(metaCatalog: unknown): Catalog | null {
  if (!isRecord(metaCatalog)) {
    return null;
  }
  const draft = JSON.parse(JSON.stringify(metaCatalog)) as Record<string, unknown>;
  (Object.keys(CATALOG_STREAM_SECTION_PATHS) as CatalogStreamSection[]).forEach((section) => {
    const target = catalogStreamArrayTarget(draft, section);
    if (target) {
      target.length = 0;
    }
  });
  return draft as unknown as Catalog;
}

function mergeCatalogStreamRows(
  draftCatalog: Catalog,
  section: string,
  offset: number,
  rows: unknown[],
): void {
  const target = catalogStreamArrayTarget(draftCatalog as unknown as Record<string, unknown>, section);
  if (!target || !Array.isArray(rows) || rows.length <= 0) {
    return;
  }
  const baseOffset = Number.isFinite(offset) ? Math.max(0, Math.floor(offset)) : 0;
  rows.forEach((row, index) => {
    target[baseOffset + index] = row;
  });
}

function mergeMuseEvents(previous: MuseEvent[], incoming: MuseEvent[]): MuseEvent[] {
  if (!Array.isArray(incoming) || incoming.length <= 0) {
    return previous;
  }

  const seen = new Set(previous.map((row) => String(row.event_id || '').trim()));
  const merged = [...previous];
  let changed = false;

  incoming.forEach((row) => {
    const id = String(row.event_id || '').trim();
    if (!id || seen.has(id)) {
      return;
    }
    seen.add(id);
    merged.push(row);
    changed = true;
  });

  if (!changed) {
    return previous;
  }

  merged.sort((left, right) => Number(left.seq ?? 0) - Number(right.seq ?? 0));
  return merged.slice(-320);
}

function mergeSimulationPatch(
  previous: SimulationState | null,
  patch: Partial<SimulationState>,
): SimulationState | null {
  if (!patch || typeof patch !== 'object') {
    return previous;
  }
  if (!previous) {
    if (patch.timestamp && patch.total !== undefined && patch.points) {
      return patch as SimulationState;
    }
    return previous;
  }

  const next: SimulationState = {
    ...previous,
    ...patch,
  };
  if (patch.presence_dynamics && previous.presence_dynamics) {
    const previousDynamics = previous.presence_dynamics;
    const patchDynamics = patch.presence_dynamics;
    next.presence_dynamics = {
      ...previousDynamics,
      ...patchDynamics,
    };

    const previousParticles = Array.isArray(previousDynamics.field_particles)
      ? previousDynamics.field_particles
      : [];
    const patchParticles = Array.isArray(patchDynamics.field_particles)
      ? patchDynamics.field_particles
      : [];
    if (patchParticles.length > 0 && previousParticles.length > 0) {
      const previousById = new Map<string, (typeof previousParticles)[number]>();
      previousParticles.forEach((row, index) => {
        if (!row || typeof row !== 'object') {
          return;
        }
        const rowRecord = row as (typeof previousParticles)[number];
        const rowId = String(rowRecord.id ?? rowRecord.presence_id ?? `particle:${index}`).trim();
        if (!rowId || previousById.has(rowId)) {
          return;
        }
        previousById.set(rowId, rowRecord);
      });

      next.presence_dynamics.field_particles = patchParticles.map((row, index) => {
        if (!row || typeof row !== 'object') {
          return row;
        }
        const rowRecord = row as (typeof patchParticles)[number];
        const rowId = String(rowRecord.id ?? rowRecord.presence_id ?? `particle:${index}`).trim();
        const previousRow = previousById.get(rowId);
        if (!previousRow) {
          return row;
        }
        return {
          ...previousRow,
          ...rowRecord,
        } as (typeof patchParticles)[number];
      });
    }
  }
  return next;
}

function decodePackedWsNode(node: unknown, keyTable: string[]): unknown {
  if (typeof node === 'number') {
    return node;
  }
  if (!Array.isArray(node) || node.length === 0) {
    return null;
  }

  const tag = Number(node[0]);
  if (!Number.isFinite(tag)) {
    return null;
  }

  if (tag === WS_PACK_TAG_NULL) {
    return null;
  }
  if (tag === WS_PACK_TAG_BOOL) {
    return Number(node[1] ?? 0) !== 0;
  }
  if (tag === WS_PACK_TAG_STRING) {
    return typeof node[1] === 'string' ? node[1] : String(node[1] ?? '');
  }
  if (tag === WS_PACK_TAG_ARRAY) {
    return node.slice(1).map((row) => decodePackedWsNode(row, keyTable));
  }
  if (tag !== WS_PACK_TAG_OBJECT) {
    return null;
  }

  const output: Record<string, unknown> = {};
  for (let index = 1; index + 1 < node.length; index += 2) {
    const keySlot = Number(node[index]);
    const keyName =
      Number.isInteger(keySlot) && keySlot >= 0 && keySlot < keyTable.length
        ? keyTable[keySlot]
        : '';
    if (!keyName) {
      continue;
    }
    output[keyName] = decodePackedWsNode(node[index + 1], keyTable);
  }
  return output;
}

function decodeWsMessage(raw: unknown): IncomingWsMessage | null {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    return raw as IncomingWsMessage;
  }
  if (!Array.isArray(raw) || raw.length < 3 || raw[0] !== WS_WIRE_ARRAY_SCHEMA) {
    return null;
  }
  const keyTable = Array.isArray(raw[1])
    ? raw[1].map((row) => String(row ?? '')).filter((row) => row.length > 0)
    : [];
  const decoded = decodePackedWsNode(raw[2], keyTable);
  if (!decoded || typeof decoded !== 'object' || Array.isArray(decoded)) {
    return null;
  }
  return decoded as IncomingWsMessage;
}

export function useWorldState(perspective: UIPerspective = 'hybrid') {
  const initialState: WorldState = {
    catalog: null,
    simulation: null,
    mixMeta: null,
    projection: null,
    museEvents: [],
    isConnected: false,
  };
  const [state, setState] = useState<WorldState>(initialState);
  const stateRef = useRef<WorldState>(initialState);

  const wsRef = useRef<WebSocket | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const projectionFetchTimeoutRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const simulationFallbackInFlightRef = useRef(false);
  const simulationFallbackTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const flushFrameRef = useRef<number | null>(null);
  const connectRef = useRef<(() => void) | null>(null);
  const shouldReconnectRef = useRef(true);
  const wsChunkAssembliesRef = useRef<Record<string, WsChunkAssembly>>({});
  const wsChunkCleanupAtMsRef = useRef(0);
  const pendingPatchRef = useRef<{
    catalog?: Catalog | null;
    simulation?: SimulationState | null;
    mixMeta?: MixMeta | null;
    projection?: UIProjectionBundle | null;
    museEventsAppend?: MuseEvent[];
  }>({});

  const enqueueStatePatch = useCallback(
    (patch: {
      catalog?: Catalog | null;
      simulation?: SimulationState | null;
      mixMeta?: MixMeta | null;
      projection?: UIProjectionBundle | null;
      museEventsAppend?: MuseEvent[];
    }) => {
      const { museEventsAppend, ...restPatch } = patch;
      const existing = pendingPatchRef.current;
      if (restPatch.catalog !== undefined) {
        existing.catalog = restPatch.catalog;
      }
      if (restPatch.simulation !== undefined) {
        existing.simulation = restPatch.simulation;
      }
      if (restPatch.mixMeta !== undefined) {
        existing.mixMeta = restPatch.mixMeta;
      }
      if (restPatch.projection !== undefined) {
        existing.projection = restPatch.projection;
      }
      if (Array.isArray(museEventsAppend) && museEventsAppend.length > 0) {
        if (existing.museEventsAppend) {
          existing.museEventsAppend.push(...museEventsAppend);
        } else {
          existing.museEventsAppend = [...museEventsAppend];
        }
      }
      if (flushFrameRef.current !== null) {
        return;
      }
      flushFrameRef.current = window.requestAnimationFrame(() => {
        flushFrameRef.current = null;
        const next = pendingPatchRef.current;
        pendingPatchRef.current = {};
        setState((prev) => {
          const nextMuseEvents = next.museEventsAppend
            ? mergeMuseEvents(prev.museEvents, next.museEventsAppend)
            : prev.museEvents;
          const nextState = {
            ...prev,
            ...(next.catalog !== undefined ? { catalog: next.catalog } : {}),
            ...(next.simulation !== undefined ? { simulation: next.simulation } : {}),
            ...(next.mixMeta !== undefined ? { mixMeta: next.mixMeta } : {}),
            ...(next.projection !== undefined ? { projection: next.projection } : {}),
            ...(nextMuseEvents !== prev.museEvents ? { museEvents: nextMuseEvents } : {}),
          };
          if (
            nextState.catalog === prev.catalog
            && nextState.simulation === prev.simulation
            && nextState.mixMeta === prev.mixMeta
            && nextState.projection === prev.projection
            && nextState.museEvents === prev.museEvents
            && nextState.isConnected === prev.isConnected
          ) {
            return prev;
          }
          stateRef.current = nextState;
          return nextState;
        });
      });
    },
    [],
  );

  const connect = useCallback(() => {
    const url = runtimeWebSocketUrl(
      `/ws?perspective=${encodeURIComponent(perspective)}&delta_stream=workers&wire=arr&simulation_payload=trimmed&particle_payload=lite&ws_chunk=0&catalog_events=0&skip_catalog_bootstrap=1`,
    );

    wsChunkAssembliesRef.current = {};
    wsChunkCleanupAtMsRef.current = 0;

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setState((prev) => {
        const next = { ...prev, isConnected: true };
        stateRef.current = next;
        return next;
      });
    };

    ws.onmessage = (event) => {
      const handleDecodedMessage = (msg: IncomingWsMessage) => {
        const msgType = String(msg.type ?? '').trim();
        if (msgType === 'catalog') {
          const catalogPayload = (msg.catalog ?? null) as Catalog | null;
          enqueueStatePatch({
            catalog: catalogPayload,
            mixMeta: (msg.mix ?? null) as MixMeta | null,
            ...(catalogPayload?.ui_projection
              ? { projection: catalogPayload.ui_projection }
              : {}),
          });
        } else if (msgType === 'muse_events') {
          const eventsPayload = msg.events;
          const incoming = Array.isArray(eventsPayload)
            ? eventsPayload.filter((row: unknown): row is MuseEvent => {
                if (!row || typeof row !== 'object') {
                  return false;
                }
                const eventId = String((row as MuseEvent).event_id ?? '').trim();
                const kind = String((row as MuseEvent).kind ?? '').trim();
                return eventId.length > 0 && kind.length > 0;
              })
            : [];
          if (incoming.length > 0) {
            enqueueStatePatch({ museEventsAppend: incoming });
          }
        } else if (msgType === 'simulation') {
          const simulationPayload = (msg.simulation ?? null) as SimulationState | null;
          const projectionPayload = (msg.projection ?? simulationPayload?.projection ?? null) as
            | UIProjectionBundle
            | null;
          enqueueStatePatch({
            simulation: simulationPayload,
            ...(projectionPayload ? { projection: projectionPayload } : {}),
          });
        } else if (msgType === 'simulation_delta') {
          const deltaPayload = msg.delta as { patch?: unknown } | undefined;
          const deltaPatch = deltaPayload?.patch;
          if (deltaPatch && typeof deltaPatch === 'object') {
            const nextSimulation = mergeSimulationPatch(
              pendingPatchRef.current.simulation ?? stateRef.current.simulation,
              deltaPatch as Partial<SimulationState>,
            );
            const projectionPatch = (
              deltaPatch as { projection?: UIProjectionBundle | null }
            ).projection;
            enqueueStatePatch({
              ...(nextSimulation ? { simulation: nextSimulation } : {}),
              ...(projectionPatch ? { projection: projectionPatch } : {}),
            });
          }
        }
      };

      try {
        const msg = decodeWsMessage(JSON.parse(event.data));
        if (!msg) {
          return;
        }
        const msgType = String(msg.type ?? '').trim();
        if (msgType === 'ws_chunk') {
          const chunkId = String(msg.chunk_id ?? '').trim();
          const chunkIndex = Number(msg.chunk_index ?? -1);
          const chunkTotal = Number(msg.chunk_total ?? 0);
          const chunkPayload = typeof msg.payload === 'string' ? msg.payload : '';
          if (
            chunkId.length <= 0
            || !Number.isInteger(chunkIndex)
            || !Number.isInteger(chunkTotal)
            || chunkIndex < 0
            || chunkTotal <= 0
            || chunkIndex >= chunkTotal
            || chunkTotal > 4096
            || chunkPayload.length <= 0
          ) {
            return;
          }

          const nowMs = Date.now();
          if (nowMs - wsChunkCleanupAtMsRef.current >= WS_CHUNK_CLEANUP_INTERVAL_MS) {
            wsChunkCleanupAtMsRef.current = nowMs;
            const activeRows = Object.entries(wsChunkAssembliesRef.current);
            activeRows.forEach(([id, row]) => {
              if (nowMs - row.updatedAtMs > WS_CHUNK_TTL_MS) {
                delete wsChunkAssembliesRef.current[id];
              }
            });
            const remaining = Object.entries(wsChunkAssembliesRef.current);
            if (remaining.length > WS_CHUNK_MAX_ACTIVE) {
              remaining
                .sort((left, right) => left[1].updatedAtMs - right[1].updatedAtMs)
                .slice(0, remaining.length - WS_CHUNK_MAX_ACTIVE)
                .forEach(([id]) => {
                  delete wsChunkAssembliesRef.current[id];
                });
            }
          }

          let assembly = wsChunkAssembliesRef.current[chunkId];
          if (!assembly || assembly.chunkTotal !== chunkTotal) {
            assembly = {
              chunkTotal,
              parts: new Array(chunkTotal).fill(''),
              receivedCount: 0,
              updatedAtMs: nowMs,
            };
            wsChunkAssembliesRef.current[chunkId] = assembly;
          }

          if (!assembly.parts[chunkIndex]) {
            assembly.parts[chunkIndex] = chunkPayload;
            assembly.receivedCount += 1;
          }
          assembly.updatedAtMs = nowMs;

          if (assembly.receivedCount < assembly.chunkTotal) {
            return;
          }

          delete wsChunkAssembliesRef.current[chunkId];
          const mergedPayload = assembly.parts.join('');
          if (mergedPayload.length <= 0) {
            return;
          }
          try {
            const mergedMessage = decodeWsMessage(JSON.parse(mergedPayload));
            if (mergedMessage) {
              handleDecodedMessage(mergedMessage);
            }
          } catch (chunkError) {
            console.warn('WS chunk decode error', chunkError);
          }
          return;
        }

        handleDecodedMessage(msg);
      } catch (err) {
        console.error('WS parse error', err);
      }
    };

    ws.onclose = () => {
      setState((prev) => {
        const next = { ...prev, isConnected: false };
        stateRef.current = next;
        return next;
      });
      if (!shouldReconnectRef.current) {
        return;
      }
      retryTimeoutRef.current = window.setTimeout(() => {
        connectRef.current?.();
      }, 3000);
    };

    wsRef.current = ws;
  }, [enqueueStatePatch, perspective]);

  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();
    const controller = new AbortController();

    // Initial fetch fallback
    const fetchInitial = async () => {
      try {
        const perspectiveQs = `perspective=${encodeURIComponent(perspective)}`;
        const fetchCatalogFromStream = async (): Promise<Catalog | null> => {
          const streamRes = await fetch(
            runtimeApiUrl(
              `/api/catalog/stream?${perspectiveQs}&trim=1&chunk_rows=${CATALOG_STREAM_CHUNK_ROWS}`,
            ),
            {
              signal: controller.signal,
            },
          );
          if (!streamRes.ok || !streamRes.body || typeof streamRes.body.getReader !== 'function') {
            return null;
          }

          const reader = streamRes.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          let draftCatalog: Catalog | null = null;
          let streamDoneOk = false;
          let streamError = '';

          const processLine = (rawLine: string) => {
            const line = String(rawLine || '').trim();
            if (!line) {
              return;
            }

            let payload: unknown = null;
            try {
              payload = JSON.parse(line);
            } catch {
              return;
            }
            if (!isRecord(payload)) {
              return;
            }

            const rowType = String(payload.type ?? '').trim().toLowerCase();
            if (rowType === 'meta') {
              draftCatalog = createCatalogStreamDraft(payload.catalog);
              return;
            }
            if (rowType === 'rows') {
              if (!draftCatalog) {
                return;
              }
              const section = String(payload.section ?? '').trim();
              const offsetRaw = Number(payload.offset ?? 0);
              const rows = Array.isArray(payload.rows) ? payload.rows : [];
              mergeCatalogStreamRows(draftCatalog, section, offsetRaw, rows);
              return;
            }
            if (rowType === 'error') {
              streamError = String(payload.error ?? 'catalog_stream_error').trim() || 'catalog_stream_error';
              return;
            }
            if (rowType === 'done') {
              streamDoneOk = payload.ok === true || String(payload.ok ?? '').trim().toLowerCase() === 'true';
            }
          };

          try {
            while (true) {
              const { value, done } = await reader.read();
              if (done) {
                break;
              }
              buffer += decoder.decode(value, { stream: true });

              let newlineIndex = buffer.indexOf('\n');
              while (newlineIndex >= 0) {
                const line = buffer.slice(0, newlineIndex);
                buffer = buffer.slice(newlineIndex + 1);
                processLine(line);
                if (streamError) {
                  break;
                }
                newlineIndex = buffer.indexOf('\n');
              }
              if (streamError) {
                break;
              }
            }

            buffer += decoder.decode();
            if (buffer.trim()) {
              processLine(buffer);
            }
          } finally {
            reader.releaseLock();
          }

          if (streamError) {
            throw new Error(streamError);
          }
          if (!streamDoneOk || !draftCatalog) {
            return null;
          }
          return draftCatalog;
        };

        const fetchCatalogLegacyJson = async (): Promise<Catalog | null> => {
          const catalogRes = await fetch(runtimeApiUrl(`/api/catalog?${perspectiveQs}`), {
            signal: controller.signal,
          });
          if (!catalogRes.ok) {
            return null;
          }
          return (await catalogRes.json()) as Catalog;
        };

        const fetchCatalog = async () => {
          let catalog: Catalog | null = null;
          try {
            catalog = await fetchCatalogFromStream();
          } catch (streamError) {
            console.warn('Initial catalog stream fetch failed', streamError);
          }
          if (!catalog) {
            catalog = await fetchCatalogLegacyJson();
          }
          if (!catalog) {
            return;
          }

          enqueueStatePatch({
            catalog,
            ...(catalog?.ui_projection ? { projection: catalog.ui_projection } : {}),
          });

          if (!catalog?.ui_projection) {
            projectionFetchTimeoutRef.current = window.setTimeout(async () => {
              try {
                const projectionRes = await fetch(runtimeApiUrl(`/api/ui/projection?${perspectiveQs}`), {
                  signal: controller.signal,
                });
                if (!projectionRes.ok) {
                  return;
                }
                const projectionPayload = await projectionRes.json();
                if (projectionPayload?.projection) {
                  enqueueStatePatch({ projection: projectionPayload.projection });
                }
              } catch {
                // projection fallback is optional
              }
            }, 120);
          }
        };

        const fetchSimulation = async () => {
          const simulationRes = await fetch(runtimeApiUrl(`/api/simulation?${perspectiveQs}&payload=trimmed`), {
            signal: controller.signal,
          });
          if (!simulationRes.ok) {
            return;
          }
          const simulation = await simulationRes.json();
          enqueueStatePatch({
            simulation,
            ...(simulation?.projection ? { projection: simulation.projection } : {}),
          });
        };

        void fetchCatalog().catch((error) => {
          if (error instanceof DOMException && error.name === 'AbortError') {
            return;
          }
          console.warn('Initial catalog fetch failed', error);
        });
        void fetchSimulation().catch((error) => {
          if (error instanceof DOMException && error.name === 'AbortError') {
            return;
          }
          console.warn('Initial simulation fetch failed', error);
        });
      } catch (e) {
        if (!(e instanceof DOMException && e.name === 'AbortError')) {
          console.warn('Initial fetch failed', e);
        }
      }
    };

    void fetchInitial();

    return () => {
      shouldReconnectRef.current = false;
      controller.abort();
      if (wsRef.current) wsRef.current.close();
      clearTimeout(retryTimeoutRef.current);
      clearTimeout(projectionFetchTimeoutRef.current);
      if (flushFrameRef.current !== null) {
        window.cancelAnimationFrame(flushFrameRef.current);
        flushFrameRef.current = null;
      }
      pendingPatchRef.current = {};
      wsChunkAssembliesRef.current = {};
      wsChunkCleanupAtMsRef.current = 0;
    };
  }, [connect, enqueueStatePatch, perspective]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const scheduleNext = (delayMs: number) => {
      clearTimeout(simulationFallbackTimerRef.current);
      simulationFallbackTimerRef.current = window.setTimeout(() => {
        void pollSimulationFallback();
      }, delayMs);
    };

    const pollSimulationFallback = async () => {
      if (cancelled || controller.signal.aborted) {
        return;
      }
      if (simulationFallbackInFlightRef.current) {
        scheduleNext(3500);
        return;
      }
      if (stateRef.current.simulation) {
        return;
      }

      simulationFallbackInFlightRef.current = true;
      try {
        const perspectiveQs = `perspective=${encodeURIComponent(perspective)}`;
        const simulationRes = await fetch(runtimeApiUrl(`/api/simulation?${perspectiveQs}&payload=trimmed`), {
          signal: controller.signal,
        });
        if (!simulationRes.ok) {
          scheduleNext(4000);
          return;
        }
        const simulation = await simulationRes.json();
        enqueueStatePatch({
          simulation,
          ...(simulation?.projection ? { projection: simulation.projection } : {}),
        });
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          scheduleNext(4500);
        }
      } finally {
        simulationFallbackInFlightRef.current = false;
      }
    };

    scheduleNext(1200);

    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(simulationFallbackTimerRef.current);
      simulationFallbackInFlightRef.current = false;
    };
  }, [enqueueStatePatch, perspective]);

  return state;
}
