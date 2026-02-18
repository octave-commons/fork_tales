import { useState, useEffect, useRef, useCallback } from 'react';
import { runtimeApiUrl, runtimeWebSocketUrl } from '../runtime/endpoints';
import type {
  Catalog,
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
  isConnected: boolean;
}

export function useWorldState(perspective: UIPerspective = 'hybrid') {
  const [state, setState] = useState<WorldState>({
    catalog: null,
    simulation: null,
    mixMeta: null,
    projection: null,
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
        } else if (msg.type === 'simulation') {
          enqueueStatePatch({
            simulation: msg.simulation,
            ...(msg.projection ?? msg.simulation?.projection
              ? { projection: msg.projection ?? msg.simulation?.projection }
              : {}),
          });
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
