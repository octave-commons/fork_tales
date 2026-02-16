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
          setState(s => ({
            ...s,
            catalog: msg.catalog,
            mixMeta: msg.mix,
            projection: msg.catalog?.ui_projection ?? s.projection,
          }));
        } else if (msg.type === 'simulation') {
          setState(s => ({
            ...s,
            simulation: msg.simulation,
            projection:
              msg.projection ??
              msg.simulation?.projection ??
              s.projection,
          }));
        }
      } catch (err) {
        console.error('WS parse error', err);
      }
    };

    ws.onclose = () => {
      setState(s => ({ ...s, isConnected: false }));
      retryTimeoutRef.current = window.setTimeout(connect, 3000);
    };

    wsRef.current = ws;
  }, [perspective]);

  useEffect(() => {
    connect();
    
    // Initial fetch fallback
    const fetchInitial = async () => {
        try {
            const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
            const perspectiveQs = `perspective=${encodeURIComponent(perspective)}`;
            const res = await fetch(`${baseUrl}/api/catalog?${perspectiveQs}`);
            if(res.ok) {
                const catalog = await res.json();
                setState(s => ({ ...s, catalog, projection: catalog?.ui_projection ?? s.projection }));
            }

            const projectionRes = await fetch(`${baseUrl}/api/ui/projection?${perspectiveQs}`);
            if (projectionRes.ok) {
                const projectionPayload = await projectionRes.json();
                setState(s => ({ ...s, projection: projectionPayload?.projection ?? s.projection }));
            }
        } catch(e) {
            console.warn("Initial fetch failed", e);
        }
    };
    fetchInitial();

    return () => {
      if (wsRef.current) wsRef.current.close();
      clearTimeout(retryTimeoutRef.current);
    };
  }, [connect, perspective]);

  return state;
}
