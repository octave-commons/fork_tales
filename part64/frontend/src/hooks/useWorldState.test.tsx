/* @vitest-environment jsdom */

import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useWorldState } from './useWorldState';

const PACK_SCHEMA = 'eta-mu.ws.arr.v1';
const PACK_TAG_OBJECT = -1;
const PACK_TAG_ARRAY = -2;
const PACK_TAG_STRING = -3;
const PACK_TAG_BOOL = -4;
const PACK_TAG_NULL = -5;

function mockJsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function mockNdjsonResponse(lines: unknown[], status = 200): Response {
  const payload = `${lines.map((row) => JSON.stringify(row)).join('\n')}\n`;
  const encoded = new TextEncoder().encode(payload);
  const chunkSize = Math.max(1, Math.floor(encoded.length / 3));
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let offset = 0; offset < encoded.length; offset += chunkSize) {
        controller.enqueue(encoded.slice(offset, offset + chunkSize));
      }
      controller.close();
    },
  });
  return {
    ok: status >= 200 && status < 300,
    status,
    body,
    json: async () => ({ ok: false }),
  } as Response;
}

function simulationFixture(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    timestamp: '2026-02-21T18:00:00Z',
    total: 1,
    audio: 0,
    image: 0,
    video: 0,
    points: [],
    presence_dynamics: {
      user_presence: { id: 'user-1' },
      resource_heartbeat: { devices: { cpu: { utilization: 12 } } },
      field_particles: [{ id: 'dm-1' }],
    },
    ...overrides,
  };
}

function packWsMessage(payload: Record<string, unknown>): unknown[] {
  const keyTable: string[] = [];
  const keySlots = new Map<string, number>();

  const encodeNode = (value: unknown): unknown => {
    if (value === null || value === undefined) {
      return [PACK_TAG_NULL];
    }
    if (typeof value === 'boolean') {
      return [PACK_TAG_BOOL, value ? 1 : 0];
    }
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : [PACK_TAG_STRING, '0'];
    }
    if (typeof value === 'string') {
      return [PACK_TAG_STRING, value];
    }
    if (Array.isArray(value)) {
      return [PACK_TAG_ARRAY, ...value.map((row) => encodeNode(row))];
    }
    if (typeof value === 'object') {
      const record = value as Record<string, unknown>;
      const rows: unknown[] = [PACK_TAG_OBJECT];
      Object.entries(record).forEach(([key, nested]) => {
        const knownSlot = keySlots.get(key);
        const slot = knownSlot ?? keyTable.length;
        if (knownSlot === undefined) {
          keySlots.set(key, slot);
          keyTable.push(key);
        }
        rows.push(slot);
        rows.push(encodeNode(nested));
      });
      return rows;
    }
    return [PACK_TAG_STRING, String(value)];
  };

  return [PACK_SCHEMA, keyTable, encodeNode(payload)];
}

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  static CONNECTING = 0;

  static OPEN = 1;

  static CLOSING = 2;

  static CLOSED = 3;

  readonly url: string;

  readyState = MockWebSocket.OPEN;

  onopen: ((event: Event) => void) | null = null;

  onmessage: ((event: MessageEvent) => void) | null = null;

  onclose: ((event: CloseEvent) => void) | null = null;

  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
  });

  send = vi.fn();

  constructor(url: string | URL) {
    this.url = String(url);
    MockWebSocket.instances.push(this);
  }

  emitOpen(): void {
    this.onopen?.(new Event('open'));
  }

  emitMessage(payload: unknown): void {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }
}

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal('WebSocket', MockWebSocket as unknown as typeof WebSocket);
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => mockJsonResponse({ ok: false }, 503)) as unknown as typeof fetch,
  );
  let rafId = 0;
  const rafTimers = new Map<number, ReturnType<typeof setTimeout>>();
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    rafId += 1;
    const id = rafId;
    const timer = setTimeout(() => {
      rafTimers.delete(id);
      cb(performance.now());
    }, 0);
    rafTimers.set(id, timer);
    return id;
  });
  vi.stubGlobal('cancelAnimationFrame', (id: number) => {
    const timer = rafTimers.get(id);
    if (timer) {
      clearTimeout(timer);
      rafTimers.delete(id);
    }
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('useWorldState websocket worker streams', () => {
  it('connects to worker-delta websocket stream mode', async () => {
    const { result } = renderHook(() => useWorldState('hybrid'));
    const ws = MockWebSocket.instances[0];

    expect(ws).toBeDefined();
    expect(ws.url).toContain(
      '/ws?perspective=hybrid&delta_stream=workers&wire=arr&simulation_payload=trimmed&particle_payload=lite&ws_chunk=0&catalog_events=0&skip_catalog_bootstrap=1',
    );

    act(() => {
      ws.emitOpen();
    });

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });

  it('merges multiple worker simulation deltas into one coherent state', async () => {
    const { result } = renderHook(() => useWorldState('hybrid'));
    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.emitOpen();
    });

    act(() => {
      ws.emitMessage({
        type: 'simulation',
        simulation: simulationFixture(),
      });
    });

    await waitFor(() => {
      expect(result.current.simulation?.timestamp).toBe('2026-02-21T18:00:00Z');
    });

    act(() => {
      ws.emitMessage({
        type: 'simulation_delta',
        stream: 'workers',
        worker_id: 'sim-resource',
        delta: {
          patch: {
            timestamp: '2026-02-21T18:00:01Z',
            presence_dynamics: {
              resource_heartbeat: { devices: { cpu: { utilization: 72 } } },
            },
          },
        },
      });
    });
    act(() => {
      ws.emitMessage({
        type: 'simulation_delta',
        stream: 'workers',
        worker_id: 'sim-particles',
        delta: {
          patch: {
            timestamp: '2026-02-21T18:00:01Z',
            presence_dynamics: {
              field_particles: [{ id: 'dm-2' }],
            },
          },
        },
      });
    });

    await waitFor(() => {
      const dynamics = result.current.simulation?.presence_dynamics as
        | Record<string, unknown>
        | undefined;
      expect(result.current.simulation?.timestamp).toBe('2026-02-21T18:00:01Z');
      expect(result.current.simulation?.presence_dynamics?.resource_heartbeat).toEqual({
        devices: { cpu: { utilization: 72 } },
      });
      expect(result.current.simulation?.presence_dynamics?.field_particles).toEqual([
        { id: 'dm-2' },
      ]);
      expect(dynamics?.user_presence).toEqual({ id: 'user-1' });
    });
  });

  it('applies projection stream delta patches', async () => {
    const { result } = renderHook(() => useWorldState('hybrid'));
    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.emitOpen();
    });

    act(() => {
      ws.emitMessage({
        type: 'simulation',
        simulation: simulationFixture(),
      });
    });

    act(() => {
      ws.emitMessage({
        type: 'simulation_delta',
        stream: 'projection',
        worker_id: 'sim-projection',
        delta: {
          patch: {
            timestamp: '2026-02-21T18:00:02Z',
            projection: {
              record: 'projection.v1',
              perspective: 'hybrid',
              ts: 173,
            },
          },
        },
      });
    });

    await waitFor(() => {
      expect(result.current.projection).toMatchObject({
        record: 'projection.v1',
        perspective: 'hybrid',
      });
      expect(result.current.simulation?.timestamp).toBe('2026-02-21T18:00:02Z');
    });
  });

  it('reassembles chunked websocket simulation payloads', async () => {
    const { result } = renderHook(() => useWorldState('hybrid'));
    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.emitOpen();
    });

    const chunkedPayload = {
      type: 'simulation',
      simulation: simulationFixture({
        timestamp: '2026-02-21T18:00:04Z',
        total: 3,
      }),
    };
    const payloadText = JSON.stringify(chunkedPayload);
    const chunkSize = Math.max(1, Math.floor(payloadText.length / 3));
    const chunks: string[] = [];
    for (let offset = 0; offset < payloadText.length; offset += chunkSize) {
      chunks.push(payloadText.slice(offset, offset + chunkSize));
    }
    const emitOrder = chunks.map((_, index) => index);
    if (emitOrder.length >= 2) {
      const first = emitOrder[0];
      emitOrder[0] = emitOrder[1];
      emitOrder[1] = first;
    }

    act(() => {
      emitOrder.forEach((chunkIndex) => {
        ws.emitMessage({
          type: 'ws_chunk',
          chunk_id: 'sim:chunk:1',
          chunk_index: chunkIndex,
          chunk_total: chunks.length,
          payload: chunks[chunkIndex],
        });
      });
    });

    await waitFor(() => {
      expect(result.current.simulation?.timestamp).toBe('2026-02-21T18:00:04Z');
      expect(result.current.simulation?.total).toBe(3);
    });
  });

  it('decodes packed array websocket messages', async () => {
    const { result } = renderHook(() => useWorldState('hybrid'));
    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.emitOpen();
    });

    act(() => {
      ws.emitMessage(
        packWsMessage({
          type: 'simulation',
          simulation: simulationFixture(),
        }),
      );
    });

    act(() => {
      ws.emitMessage(
        packWsMessage({
          type: 'simulation_delta',
          stream: 'workers',
          worker_id: 'sim-resource',
          delta: {
            patch: {
              timestamp: '2026-02-21T18:00:03Z',
              presence_dynamics: {
                resource_heartbeat: { devices: { cpu: { utilization: 91 } } },
              },
            },
          },
        }),
      );
    });

    await waitFor(() => {
      expect(result.current.simulation?.timestamp).toBe('2026-02-21T18:00:03Z');
      expect(result.current.simulation?.presence_dynamics?.resource_heartbeat).toEqual({
        devices: { cpu: { utilization: 91 } },
      });
    });
  });

  it('hydrates initial catalog from ndjson stream rows', async () => {
    const streamRows = [
      {
        type: 'start',
        ok: true,
      },
      {
        type: 'progress',
        stage: 'catalog_begin',
      },
      {
        type: 'meta',
        catalog: {
          generated_at: '2026-02-23T00:00:00Z',
          part_roots: [],
          counts: {},
          canonical_terms: [],
          cover_fields: [],
          ui_projection: {
            record: 'projection.v1',
            perspective: 'hybrid',
            ts: 12,
          },
          items: {
            streamed: true,
            section: 'items',
            count: 1,
          },
          file_graph: {
            file_nodes: {
              streamed: true,
              section: 'file_nodes',
              count: 1,
            },
            edges: {
              streamed: true,
              section: 'file_edges',
              count: 0,
            },
            embed_layers: {
              streamed: true,
              section: 'file_embed_layers',
              count: 0,
            },
          },
          crawler_graph: {
            crawler_nodes: {
              streamed: true,
              section: 'crawler_nodes',
              count: 0,
            },
            edges: {
              streamed: true,
              section: 'crawler_edges',
              count: 0,
            },
          },
        },
      },
      {
        type: 'rows',
        section: 'items',
        offset: 0,
        rows: [
          {
            part: 'part-64',
            name: 'demo.txt',
            role: 'unknown',
            display_name: { en: 'demo', ja: 'demo' },
            display_role: { en: 'unknown', ja: 'unknown' },
            kind: 'text',
            bytes: 8,
            mtime_utc: '2026-02-23T00:00:00Z',
            rel_path: 'artifacts/demo.txt',
            url: '/library/artifacts/demo.txt',
          },
        ],
      },
      {
        type: 'rows',
        section: 'file_nodes',
        offset: 0,
        rows: [{ id: 'file:1', name: 'demo.txt' }],
      },
      {
        type: 'done',
        ok: true,
      },
    ];

    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/catalog/stream?')) {
        return mockNdjsonResponse(streamRows, 200);
      }
      if (url.includes('/api/simulation?')) {
        return mockJsonResponse({ ok: false }, 503);
      }
      if (url.includes('/api/catalog?')) {
        return mockJsonResponse({ ok: false }, 503);
      }
      return mockJsonResponse({ ok: false }, 503);
    });
    vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch);

    const { result } = renderHook(() => useWorldState('hybrid'));

    await waitFor(() => {
      expect(result.current.catalog?.items?.length).toBe(1);
      expect(result.current.catalog?.items?.[0]?.rel_path).toBe('artifacts/demo.txt');
      expect(result.current.projection).toMatchObject({
        record: 'projection.v1',
        perspective: 'hybrid',
      });
    });
  });
});
