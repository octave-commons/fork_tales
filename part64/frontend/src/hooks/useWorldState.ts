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

export function useWorldState(perspective: UIPerspective = 'hybrid') {
  const [state, setState] = useState<WorldState>({
    catalog: null,
    simulation: null,
    mixMeta: null,
    projection: null,
    museEvents: [],
    isConnected: false,
  });

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
        setState((prev) => ({
          ...prev,
          ...(next.catalog !== undefined ? { catalog: next.catalog } : {}),
          ...(next.simulation !== undefined ? { simulation: next.simulation } : {}),
          ...(next.mixMeta !== undefined ? { mixMeta: next.mixMeta } : {}),
          ...(next.projection !== undefined ? { projection: next.projection } : {}),
        }));
      });
    },
    [],
  );

  const connect = useCallback(() => {
    const url = runtimeWebSocketUrl(`/ws?perspective=${encodeURIComponent(perspective)}`);

    const ws = new WebSocket(url);

    ws.onopen = () => {
      setState(s => ({ ...s, isConnected: true }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'catalog') {
          enqueueStatePatch({
            catalog: msg.catalog,
            mixMeta: msg.mix,
            ...(msg.catalog?.ui_projection ? { projection: msg.catalog.ui_projection } : {}),
          });
        } else if (msg.type === 'muse_events') {
          const incoming = Array.isArray(msg.events)
            ? msg.events.filter((row: unknown): row is MuseEvent => {
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
              return {
                ...prev,
                museEvents: merged.slice(-320),
              };
            });
          }
        } else if (msg.type === 'simulation') {
          enqueueStatePatch({
            simulation: msg.simulation,
            ...(msg.projection ?? msg.simulation?.projection
              ? { projection: msg.projection ?? msg.simulation?.projection }
              : {}),
          });
        } else if (msg.type === 'simulation_delta') {
          const deltaPatch = msg?.delta?.patch;
          if (deltaPatch && typeof deltaPatch === 'object') {
            setState((prev) => {
              const nextSimulation = mergeSimulationPatch(
                prev.simulation,
                deltaPatch as Partial<SimulationState>,
              );
              const projectionPatch = (
                deltaPatch as { projection?: UIProjectionBundle | null }
              ).projection;
              return {
                ...prev,
                ...(nextSimulation ? { simulation: nextSimulation } : {}),
                ...(projectionPatch ? { projection: projectionPatch } : {}),
              };
            });
          }
        }
      } catch (err) {
        console.error('WS parse error', err);
      }
    };

    ws.onclose = () => {
      setState(s => ({ ...s, isConnected: false }));
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
        const res = await fetch(runtimeApiUrl(`/api/catalog?${perspectiveQs}`), {
          signal: controller.signal,
        });
        if (res.ok) {
          const catalog = await res.json();
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
        }
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
