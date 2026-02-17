import { useState, useEffect, useRef, useCallback } from 'react';
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
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    // When running in dev mode (port 5173), connect to 8787. In prod, relative path.
    const host = window.location.port === '5173' ? '127.0.0.1:8787' : window.location.host;
    const url = `${proto}://${host}/ws?perspective=${encodeURIComponent(perspective)}`;

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
      retryTimeoutRef.current = window.setTimeout(connect, 3000);
    };

    wsRef.current = ws;
  }, [enqueueStatePatch, perspective]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();
    const controller = new AbortController();

    // Initial fetch fallback
    const fetchInitial = async () => {
      try {
        const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
        const perspectiveQs = `perspective=${encodeURIComponent(perspective)}`;
        const res = await fetch(`${baseUrl}/api/catalog?${perspectiveQs}`, {
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
                const projectionRes = await fetch(
                  `${baseUrl}/api/ui/projection?${perspectiveQs}`,
                  { signal: controller.signal },
                );
                if (!projectionRes.ok) {
                  return;
                }
                const projectionPayload = await projectionRes.json();
                if (projectionPayload?.projection) {
                  enqueueStatePatch({ projection: projectionPayload.projection });
                }
              } catch (_projectionError) {
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
