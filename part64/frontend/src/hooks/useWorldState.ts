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
    next.presence_dynamics = {
      ...previous.presence_dynamics,
      ...patch.presence_dynamics,
    };
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
  const flushFrameRef = useRef<number | null>(null);
  const connectRef = useRef<(() => void) | null>(null);
  const shouldReconnectRef = useRef(true);
  const pendingPatchRef = useRef<{
    catalog?: Catalog | null;
    simulation?: SimulationState | null;
    mixMeta?: MixMeta | null;
    projection?: UIProjectionBundle | null;
  }>({});

  const enqueueStatePatch = useCallback(
    (patch: {
      catalog?: Catalog | null;
      simulation?: SimulationState | null;
      mixMeta?: MixMeta | null;
      projection?: UIProjectionBundle | null;
    }) => {
      pendingPatchRef.current = {
        ...pendingPatchRef.current,
        ...patch,
      };
      if (flushFrameRef.current !== null) {
        return;
      }
      flushFrameRef.current = window.requestAnimationFrame(() => {
        flushFrameRef.current = null;
        const next = pendingPatchRef.current;
        pendingPatchRef.current = {};
        setState((prev) => {
          const nextState = {
            ...prev,
            ...(next.catalog !== undefined ? { catalog: next.catalog } : {}),
            ...(next.simulation !== undefined ? { simulation: next.simulation } : {}),
            ...(next.mixMeta !== undefined ? { mixMeta: next.mixMeta } : {}),
            ...(next.projection !== undefined ? { projection: next.projection } : {}),
          };
          stateRef.current = nextState;
          return nextState;
        });
      });
    },
    [],
  );

  const connect = useCallback(() => {
    const url = runtimeWebSocketUrl(
      `/ws?perspective=${encodeURIComponent(perspective)}&delta_stream=workers&wire=json`,
    );

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setState((prev) => {
        const next = { ...prev, isConnected: true };
        stateRef.current = next;
        return next;
      });
    };

    ws.onmessage = (event) => {
      try {
        const msg = decodeWsMessage(JSON.parse(event.data));
        if (!msg) {
          return;
        }
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
            setState((prev) => {
              const seen = new Set(prev.museEvents.map((row) => String(row.event_id || '').trim()));
              const merged = [...prev.museEvents];
              incoming.forEach((row: MuseEvent) => {
                const id = String(row.event_id || '').trim();
                if (!id || seen.has(id)) {
                  return;
                }
                seen.add(id);
                merged.push(row);
              });
              merged.sort((left, right) => Number(left.seq ?? 0) - Number(right.seq ?? 0));
              const next = {
                ...prev,
                museEvents: merged.slice(-320),
              };
              stateRef.current = next;
              return next;
            });
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
        const fetchCatalog = async () => {
          const catalogRes = await fetch(runtimeApiUrl(`/api/catalog?${perspectiveQs}`), {
            signal: controller.signal,
          });
          if (!catalogRes.ok) {
            return;
          }
          const catalog = await catalogRes.json();
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
          const simulationRes = await fetch(runtimeApiUrl(`/api/simulation?${perspectiveQs}&compact=1`), {
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

        void fetchCatalog();
        void fetchSimulation();
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
    };
  }, [connect, enqueueStatePatch, perspective]);

  return state;
}
