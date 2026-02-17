import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type CrawlState = "running" | "paused" | "stopped";

interface WeaverNode {
  id: string;
  kind: "url" | "domain" | "content";
  label: string;
  url?: string;
  domain?: string;
  depth?: number;
  status?: string;
  compliance?: string;
  content_type?: string | null;
  title?: string;
}

interface WeaverEdge {
  id: string;
  kind: "hyperlink" | "canonical_redirect" | "domain_membership" | "content_membership";
  source: string;
  target: string;
}

interface WeaverGraph {
  nodes: WeaverNode[];
  edges: WeaverEdge[];
  counts?: {
    nodes_total: number;
    edges_total: number;
    url_nodes_total: number;
  };
}

interface WeaverStatus {
  state: CrawlState;
  user_agent: string;
  active_domains: string[];
  domain_distribution: Record<string, number>;
  depth_histogram: Record<string, number>;
  opt_out_domains: string[];
  event_count: number;
  metrics: {
    crawl_rate_nodes_per_sec: number;
    frontier_size: number;
    active_fetchers: number;
    compliance_percent: number;
    discovered: number;
    fetched: number;
    skipped: number;
    robots_blocked: number;
    duplicate_content: number;
    errors: number;
    average_fetch_ms: number;
  };
  graph_counts: {
    nodes_total: number;
    edges_total: number;
    url_nodes_total: number;
  };
}

interface WeaverEvent {
  event: string;
  timestamp: number;
  url?: string;
  source?: string | null;
  depth?: number;
  reason?: string;
  domain?: string;
  status?: number;
  outbound_count?: number;
  duration_ms?: number;
  [key: string]: unknown;
}

interface GraphPoint {
  x: number;
  y: number;
}

const PANEL_EVENT_LIMIT = 220;
const MAX_RENDER_NODES = 1300;
const MAX_RENDER_EDGES = 2400;
const FRAME_INTERVAL_NORMAL_MS = 33;
const FRAME_INTERVAL_DENSE_MS = 50;

function uniqueStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const output: string[] = [];
  for (const value of values) {
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    output.push(trimmed);
  }
  return output;
}

function weaverHttpCandidates(): string[] {
  if (typeof window === "undefined") {
    return ["http://127.0.0.1:8793"];
  }
  const protocol = window.location.protocol === "https:" ? "https" : "http";
  const host = window.location.hostname || "127.0.0.1";
  const envCandidate = import.meta.env.VITE_WEAVER_BASE_URL || "";
  return uniqueStrings([
    envCandidate,
    `${protocol}://${host}:8793`,
    `${protocol}://127.0.0.1:8793`,
    `${protocol}://localhost:8793`,
  ]);
}

function wsUrlFromHttpBase(base: string): string {
  const parsed = new URL(base);
  parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
  parsed.pathname = "/ws";
  parsed.search = "";
  parsed.hash = "";
  return parsed.toString();
}

function weaverWsCandidates(): string[] {
  return uniqueStrings(weaverHttpCandidates().map((base) => wsUrlFromHttpBase(base)));
}

function withOfflineHint(message: string): string {
  const lower = message.toLowerCase();
  const looksOffline =
    lower.includes("networkerror") ||
    lower.includes("failed to fetch") ||
    lower.includes("unreachable") ||
    lower.includes("status request failed") ||
    lower.includes("events request failed") ||
    lower.includes("graph request failed");
  if (!looksOffline) {
    return message;
  }
  return `${message} If Web Graph Weaver is not running, start it in part64 code with 'npm run weaver' or run 'python -m code.world_pm2 start'.`;
}

function hashString(value: string): number {
  let h = 0;
  for (let i = 0; i < value.length; i += 1) {
    h = Math.imul(31, h) + value.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

function colorForDomain(domain: string): string {
  const hue = hashString(domain || "domain") % 360;
  return `hsl(${hue}, 76%, 62%)`;
}

function shortNodeLabel(node: WeaverNode): string {
  if (node.kind === "domain") {
    return node.domain || node.label;
  }
  if (node.kind === "content") {
    return String(node.content_type || node.label);
  }
  if (node.title && node.title.trim().length > 0) {
    return node.title.slice(0, 42);
  }
  const url = String(node.url || node.label || "");
  const parts = url.split("/");
  const last = parts[parts.length - 1] || parts[parts.length - 2] || url;
  if (last.length <= 36) {
    return last;
  }
  return `${last.slice(0, 33)}...`;
}

function shortText(value: string, max = 92): string {
  const trimmed = value.trim();
  if (trimmed.length <= max) {
    return trimmed;
  }
  return `${trimmed.slice(0, Math.max(8, max - 3))}...`;
}

function eventTargetValue(event: WeaverEvent): string {
  return String(event.url || event.domain || event.reason || "-");
}

function sampleByStride<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) {
    return items;
  }
  const stride = Math.ceil(items.length / limit);
  const sampled: T[] = [];
  for (let i = 0; i < items.length; i += stride) {
    sampled.push(items[i]);
    if (sampled.length >= limit) {
      break;
    }
  }
  return sampled;
}

function mergeGraph(
  prev: WeaverGraph,
  deltaNodes: WeaverNode[] = [],
  deltaEdges: WeaverEdge[] = [],
): WeaverGraph {
  const nodeMap = new Map(prev.nodes.map((node) => [node.id, node]));
  const edgeMap = new Map(prev.edges.map((edge) => [edge.id, edge]));

  for (const node of deltaNodes) {
    nodeMap.set(node.id, node);
  }
  for (const edge of deltaEdges) {
    edgeMap.set(edge.id, edge);
  }

  return {
    ...prev,
    nodes: [...nodeMap.values()],
    edges: [...edgeMap.values()],
    counts: {
      nodes_total: nodeMap.size,
      edges_total: edgeMap.size,
      url_nodes_total: [...nodeMap.values()].filter((node) => node.kind === "url").length,
    },
  };
}

function useGraphLayout(nodes: WeaverNode[]) {
  return useMemo(() => {
    const points = new Map<string, GraphPoint>();
    const domainNodes = nodes.filter((node) => node.kind === "domain");
    const urlNodes = nodes.filter((node) => node.kind === "url");
    const contentNodes = nodes.filter((node) => node.kind === "content");

    const domainAnchors = new Map<string, GraphPoint>();
    const domainCount = Math.max(1, domainNodes.length);
    const domainRadius = 280;
    domainNodes.forEach((node, index) => {
      const angle = (Math.PI * 2 * index) / domainCount;
      const x = Math.cos(angle) * domainRadius;
      const y = Math.sin(angle) * domainRadius * 0.74;
      points.set(node.id, { x, y });
      if (node.domain) {
        domainAnchors.set(node.domain, { x, y });
      }
    });

    const perDomainIndex = new Map<string, number>();
    urlNodes.forEach((node) => {
      const domain = node.domain || "unknown";
      const anchor = domainAnchors.get(domain) || { x: 0, y: 0 };
      const idx = perDomainIndex.get(domain) || 0;
      const depth = Number(node.depth || 0);
      const angle = idx * 0.72 + (hashString(node.id) % 628) / 100;
      const radial = 30 + depth * 28 + (idx % 9) * 11;
      const x = anchor.x + Math.cos(angle) * radial;
      const y = anchor.y + Math.sin(angle) * radial * 0.82;
      points.set(node.id, { x, y });
      perDomainIndex.set(domain, idx + 1);
    });

    const contentCount = Math.max(1, contentNodes.length);
    contentNodes.forEach((node, index) => {
      const x = -220 + (index * 440) / Math.max(1, contentCount - 1);
      const y = 320 + ((index % 2) * 26);
      points.set(node.id, { x, y });
    });

    return points;
  }, [nodes]);
}

export function WebGraphWeaverPanel() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{
    active: boolean;
    startX: number;
    startY: number;
    baseOffsetX: number;
    baseOffsetY: number;
  }>({
    active: false,
    startX: 0,
    startY: 0,
    baseOffsetX: 0,
    baseOffsetY: 0,
  });

  const statusRefreshRef = useRef(0);
  const reconnectTimerRef = useRef<number | null>(null);
  const wsCandidateIndexRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);

  const [status, setStatus] = useState<WeaverStatus | null>(null);
  const [graph, setGraph] = useState<WeaverGraph>({ nodes: [], edges: [] });
  const [events, setEvents] = useState<WeaverEvent[]>([]);
  const [connection, setConnection] = useState<"connecting" | "online" | "offline">("connecting");
  const [seedInput, setSeedInput] = useState("https://example.com/");
  const [maxDepth, setMaxDepth] = useState(3);
  const [maxNodes, setMaxNodes] = useState(2500);
  const [concurrency, setConcurrency] = useState(2);
  const [optOutInput, setOptOutInput] = useState("");
  const [domainFilter, setDomainFilter] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [activeWeaverBase, setActiveWeaverBase] = useState<string>(
    () => weaverHttpCandidates()[0] || "http://127.0.0.1:8793",
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [viewScale, setViewScale] = useState(1);
  const [viewOffsetX, setViewOffsetX] = useState(0);
  const [viewOffsetY, setViewOffsetY] = useState(0);

  const visibleGraph = useMemo(() => {
    if (!domainFilter) {
      return graph;
    }
    const visibleIds = new Set(
      graph.nodes
        .filter((node) => node.domain === domainFilter || (node.kind === "domain" && node.domain === domainFilter))
        .map((node) => node.id),
    );
    for (const edge of graph.edges) {
      if (visibleIds.has(edge.source) || visibleIds.has(edge.target)) {
        visibleIds.add(edge.source);
        visibleIds.add(edge.target);
      }
    }
    return {
      nodes: graph.nodes.filter((node) => visibleIds.has(node.id)),
      edges: graph.edges.filter(
        (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
      ),
      counts: graph.counts,
    };
  }, [domainFilter, graph]);

  const renderGraph = useMemo(() => {
    const dense =
      visibleGraph.nodes.length > MAX_RENDER_NODES ||
      visibleGraph.edges.length > MAX_RENDER_EDGES;

    if (!dense) {
      return {
        nodes: visibleGraph.nodes,
        edges: visibleGraph.edges,
        sampledNodes: false,
        sampledEdges: false,
      };
    }

    const visibleNodeById = new Map(visibleGraph.nodes.map((node) => [node.id, node]));

    const selectedEdgeRows = selectedNodeId
      ? visibleGraph.edges.filter(
          (edge) => edge.source === selectedNodeId || edge.target === selectedNodeId,
        )
      : [];

    const edgeMap = new Map<string, WeaverEdge>();
    for (const edge of sampleByStride(visibleGraph.edges, MAX_RENDER_EDGES)) {
      edgeMap.set(edge.id, edge);
    }
    for (const edge of selectedEdgeRows) {
      edgeMap.set(edge.id, edge);
    }

    const nodeMap = new Map<string, WeaverNode>();
    for (const node of sampleByStride(visibleGraph.nodes, MAX_RENDER_NODES)) {
      nodeMap.set(node.id, node);
    }
    if (selectedNodeId) {
      const selectedNode = visibleNodeById.get(selectedNodeId);
      if (selectedNode) {
        nodeMap.set(selectedNode.id, selectedNode);
      }
    }
    for (const edge of edgeMap.values()) {
      const sourceNode = visibleNodeById.get(edge.source);
      if (sourceNode) {
        nodeMap.set(sourceNode.id, sourceNode);
      }
      const targetNode = visibleNodeById.get(edge.target);
      if (targetNode) {
        nodeMap.set(targetNode.id, targetNode);
      }
    }

    const edges = [...edgeMap.values()].filter(
      (edge) => nodeMap.has(edge.source) && nodeMap.has(edge.target),
    );

    return {
      nodes: [...nodeMap.values()],
      edges,
      sampledNodes: nodeMap.size < visibleGraph.nodes.length,
      sampledEdges: edges.length < visibleGraph.edges.length,
    };
  }, [selectedNodeId, visibleGraph.edges, visibleGraph.nodes]);

  const layout = useGraphLayout(renderGraph.nodes);

  const appendEvent = useCallback((event: WeaverEvent) => {
    setEvents((prev) => {
      const next = [...prev, event];
      if (next.length > PANEL_EVENT_LIMIT) {
        next.splice(0, next.length - PANEL_EVENT_LIMIT);
      }
      return next;
    });
  }, []);

  const setFriendlyError = useCallback((error: unknown) => {
    const message = withOfflineHint(
      error instanceof Error ? error.message : String(error),
    );
    setErrorMessage(message);
  }, []);

  const requestWeaver = useCallback(
    async (path: string, init?: RequestInit): Promise<Response> => {
      const candidates = uniqueStrings([activeWeaverBase, ...weaverHttpCandidates()]);
      let lastError: unknown = null;
      for (const base of candidates) {
        try {
          const response = await fetch(`${base}${path}`, init);
          if (response.status === 404 || response.status === 502 || response.status === 503) {
            lastError = new Error(`HTTP ${response.status} from ${base}`);
            continue;
          }
          setActiveWeaverBase(base);
          return response;
        } catch (error) {
          lastError = error;
        }
      }
      const message =
        lastError instanceof Error ? lastError.message : "unknown network error";
      throw new Error(withOfflineHint(`Web Graph Weaver unreachable (${message})`));
    },
    [activeWeaverBase],
  );

  const refreshStatus = useCallback(async () => {
    const response = await requestWeaver("/api/weaver/status");
    if (!response.ok) {
      throw new Error(`status request failed (${response.status})`);
    }
    const payload = (await response.json()) as WeaverStatus;
    setStatus(payload);
  }, [requestWeaver]);

  const refreshGraph = useCallback(
    async (nextDomainFilter: string) => {
      const params = new URLSearchParams();
      if (nextDomainFilter) {
        params.set("domain", nextDomainFilter);
      }
      params.set("node_limit", "12000");
      params.set("edge_limit", "26000");
      const query = params.toString();
      const response = await requestWeaver(`/api/weaver/graph${query ? `?${query}` : ""}`);
      if (!response.ok) {
        throw new Error(`graph request failed (${response.status})`);
      }
      const payload = (await response.json()) as { ok: boolean; graph: WeaverGraph };
      setGraph(payload.graph || { nodes: [], edges: [] });
    },
    [requestWeaver],
  );

  const refreshEvents = useCallback(async () => {
    const response = await requestWeaver("/api/weaver/events?limit=120");
    if (!response.ok) {
      throw new Error(`events request failed (${response.status})`);
    }
    const payload = (await response.json()) as { ok: boolean; events: WeaverEvent[] };
    setEvents(Array.isArray(payload.events) ? payload.events : []);
  }, [requestWeaver]);

  const bootstrap = useCallback(async () => {
    setErrorMessage(null);
    await Promise.all([refreshStatus(), refreshGraph(domainFilter), refreshEvents()]);
  }, [domainFilter, refreshEvents, refreshGraph, refreshStatus]);

  const postControl = useCallback(
    async (action: "start" | "pause" | "resume" | "stop") => {
      setErrorMessage(null);
      const seeds = seedInput
        .split(/[\n,]/)
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);

      const response = await requestWeaver("/api/weaver/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          seeds,
          max_depth: maxDepth,
          max_nodes: maxNodes,
          concurrency,
        }),
      });
      const payload = await response.json();
      if (!response.ok || payload?.ok !== true) {
        throw new Error(payload?.error || `control action failed (${response.status})`);
      }
      if (payload.status) {
        setStatus(payload.status as WeaverStatus);
      }
      appendEvent({
        event: "crawl_state",
        timestamp: Date.now(),
        reason: action,
      });
    },
    [appendEvent, concurrency, maxDepth, maxNodes, requestWeaver, seedInput],
  );

  const addOptOutDomain = useCallback(async () => {
    const domain = optOutInput.trim();
    if (!domain) {
      return;
    }
    setErrorMessage(null);
    const response = await requestWeaver("/api/weaver/opt-out", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain }),
    });
    const payload = await response.json();
    if (!response.ok || payload?.ok !== true) {
      throw new Error(payload?.error || `opt-out request failed (${response.status})`);
    }
    setOptOutInput("");
    await refreshStatus();
  }, [optOutInput, refreshStatus, requestWeaver]);

  useEffect(() => {
    bootstrap().catch((error) => {
      const message = withOfflineHint(
        error instanceof Error ? error.message : String(error),
      );
      setErrorMessage(message);
      setConnection("offline");
    });
  }, [bootstrap]);

  useEffect(() => {
    let cancelled = false;
    const connect = () => {
      if (cancelled) {
        return;
      }
      setConnection("connecting");
      const wsCandidates = weaverWsCandidates();
      if (wsCandidates.length === 0) {
        setConnection("offline");
        return;
      }
      const wsUrl = wsCandidates[wsCandidateIndexRef.current % wsCandidates.length];
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) {
          return;
        }
        wsCandidateIndexRef.current = 0;
        setConnection("online");
      };

      ws.onmessage = (event) => {
        if (cancelled) {
          return;
        }
        try {
          const payload = JSON.parse(String(event.data)) as {
            event: string;
            status?: WeaverStatus;
            graph?: WeaverGraph;
            recent_events?: WeaverEvent[];
            nodes?: WeaverNode[];
            edges?: WeaverEdge[];
          } & WeaverEvent;

          if (payload.event === "snapshot") {
            if (payload.status) {
              setStatus(payload.status);
            }
            if (payload.graph) {
              setGraph(payload.graph);
            }
            if (Array.isArray(payload.recent_events)) {
              setEvents(payload.recent_events);
            }
            return;
          }

          if (payload.event === "graph_delta") {
            setGraph((prev) => mergeGraph(prev, payload.nodes || [], payload.edges || []));
          }

          appendEvent(payload);

          const now = Date.now();
          if (now - statusRefreshRef.current > 900) {
            statusRefreshRef.current = now;
            refreshStatus().catch(() => {});
          }
        } catch (_err) {
          // ignore malformed event
        }
      };

      ws.onclose = () => {
        if (cancelled) {
          return;
        }
        setConnection("offline");
        wsCandidateIndexRef.current += 1;
        reconnectTimerRef.current = window.setTimeout(connect, 2200);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [appendEvent, refreshStatus]);

  useEffect(() => {
    refreshGraph(domainFilter).catch((error) => {
      const message = withOfflineHint(
        error instanceof Error ? error.message : String(error),
      );
      setErrorMessage(message);
    });
  }, [domainFilter, refreshGraph]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    let rafId = 0;
    let width = 0;
    let height = 0;
    let lastPaintAt = 0;

    const draw = (timestamp: number) => {
      const denseMode =
        renderGraph.sampledNodes ||
        renderGraph.sampledEdges ||
        renderGraph.edges.length > 1800 ||
        renderGraph.nodes.length > 1000;
      const frameInterval = denseMode
        ? FRAME_INTERVAL_DENSE_MS
        : FRAME_INTERVAL_NORMAL_MS;
      if (timestamp - lastPaintAt < frameInterval) {
        rafId = window.requestAnimationFrame(draw);
        return;
      }
      lastPaintAt = timestamp;

      const rect = canvas.getBoundingClientRect();
      const baseDpr = window.devicePixelRatio || 1;
      const dpr = denseMode ? Math.min(baseDpr, 1.2) : Math.min(baseDpr, 2);
      const nextWidth = Math.max(1, Math.floor(rect.width * dpr));
      const nextHeight = Math.max(1, Math.floor(rect.height * dpr));
      if (nextWidth !== width || nextHeight !== height) {
        width = nextWidth;
        height = nextHeight;
        canvas.width = width;
        canvas.height = height;
      }

      context.clearRect(0, 0, width, height);
      context.fillStyle = "rgba(9, 16, 28, 0.96)";
      context.fillRect(0, 0, width, height);

      const centerX = width / 2 + viewOffsetX * dpr;
      const centerY = height / 2 + viewOffsetY * dpr;
      const scale = viewScale * dpr;

      const nodeById = new Map(renderGraph.nodes.map((node) => [node.id, node]));
      const highlightedEdgeIds = new Set<string>();
      if (selectedNodeId) {
        for (const edge of renderGraph.edges) {
          if (edge.source === selectedNodeId || edge.target === selectedNodeId) {
            highlightedEdgeIds.add(edge.id);
          }
        }
      }

      context.save();
      for (const edge of renderGraph.edges) {
        const source = layout.get(edge.source);
        const target = layout.get(edge.target);
        if (!source || !target) {
          continue;
        }

        const sx = centerX + source.x * scale;
        const sy = centerY + source.y * scale;
        const tx = centerX + target.x * scale;
        const ty = centerY + target.y * scale;

        const highlighted = highlightedEdgeIds.has(edge.id);
        const pulse = denseMode
          ? 0.5
          : 0.5 + Math.sin((timestamp / 640) + hashString(edge.id) * 0.0003) * 0.5;

        if (edge.kind === "hyperlink") {
          context.strokeStyle = highlighted
            ? `rgba(112, 232, 255, ${0.58 + pulse * 0.3})`
            : `rgba(90, 156, 206, ${0.12 + pulse * 0.14})`;
          context.lineWidth = highlighted ? 1.2 : 0.6;
          if (!denseMode) {
            context.setLineDash([5, 7]);
            context.lineDashOffset = -(timestamp / 40);
          } else {
            context.setLineDash([]);
          }
        } else if (edge.kind === "canonical_redirect") {
          context.strokeStyle = highlighted
            ? "rgba(252, 143, 255, 0.82)"
            : "rgba(186, 100, 208, 0.35)";
          context.lineWidth = highlighted ? 1.3 : 0.7;
          if (!denseMode) {
            context.setLineDash([3, 5]);
            context.lineDashOffset = timestamp / 30;
          } else {
            context.setLineDash([]);
          }
        } else if (edge.kind === "domain_membership") {
          context.strokeStyle = highlighted
            ? "rgba(255, 205, 133, 0.84)"
            : "rgba(214, 164, 93, 0.28)";
          context.lineWidth = highlighted ? 1.2 : 0.5;
          context.setLineDash([]);
        } else {
          context.strokeStyle = highlighted
            ? "rgba(164, 239, 158, 0.85)"
            : "rgba(120, 186, 126, 0.26)";
          context.lineWidth = highlighted ? 1.0 : 0.45;
          if (!denseMode) {
            context.setLineDash([2, 5]);
            context.lineDashOffset = -(timestamp / 45);
          } else {
            context.setLineDash([]);
          }
        }

        context.beginPath();
        context.moveTo(sx, sy);
        context.lineTo(tx, ty);
        context.stroke();
      }
      context.setLineDash([]);
      context.restore();

      for (const node of renderGraph.nodes) {
        const point = layout.get(node.id);
        if (!point) {
          continue;
        }
        const x = centerX + point.x * scale;
        const y = centerY + point.y * scale;

        let radius = denseMode ? 1.8 : 2.4;
        if (node.kind === "domain") {
          radius = denseMode ? 6.8 : 8;
        } else if (node.kind === "content") {
          radius = denseMode ? 4.2 : 5.2;
        } else {
          radius = (denseMode ? 1.9 : 2.6) + Number(node.depth || 0) * 0.35;
        }
        const selected = selectedNodeId === node.id;
        if (selected) {
          context.fillStyle = "rgba(255, 244, 184, 0.95)";
          context.beginPath();
          context.arc(x, y, radius + 4.5, 0, Math.PI * 2);
          context.fill();
        }

        if (node.kind === "domain") {
          context.fillStyle = colorForDomain(String(node.domain || node.label));
        } else if (node.kind === "content") {
          context.fillStyle = "rgba(116, 216, 144, 0.95)";
        } else if (node.compliance === "robots_blocked" || node.status === "blocked") {
          context.fillStyle = "rgba(255, 110, 110, 0.9)";
        } else {
          context.fillStyle = "rgba(129, 203, 255, 0.85)";
        }

        if (node.kind === "content") {
          context.beginPath();
          context.moveTo(x, y - radius);
          context.lineTo(x + radius, y);
          context.lineTo(x, y + radius);
          context.lineTo(x - radius, y);
          context.closePath();
          context.fill();
        } else {
          context.beginPath();
          context.arc(x, y, radius, 0, Math.PI * 2);
          context.fill();
        }
      }

      if (selectedNodeId) {
        const selected = nodeById.get(selectedNodeId);
        if (selected) {
          context.fillStyle = "rgba(217, 233, 255, 0.95)";
          context.font = `${11 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
          context.textAlign = "left";
          context.fillText(
            `${selected.kind.toUpperCase()} :: ${shortNodeLabel(selected)}`,
            16 * dpr,
            28 * dpr,
          );
        }
      }

      context.fillStyle = "rgba(172, 197, 227, 0.82)";
      context.font = `${10 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      context.textAlign = "left";
      context.fillText(
        `nodes ${renderGraph.nodes.length}/${visibleGraph.nodes.length} / edges ${renderGraph.edges.length}/${visibleGraph.edges.length} / zoom ${viewScale.toFixed(2)}x`,
        16 * dpr,
        (height / dpr - 14) * dpr,
      );

      if (renderGraph.sampledNodes || renderGraph.sampledEdges) {
        context.fillStyle = "rgba(255, 201, 132, 0.86)";
        context.fillText("dense graph mode: sampled render active", 16 * dpr, 16 * dpr);
      }

      rafId = window.requestAnimationFrame(draw);
    };

    rafId = window.requestAnimationFrame(draw);
    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, [layout, renderGraph.edges, renderGraph.nodes, renderGraph.sampledEdges, renderGraph.sampledNodes, selectedNodeId, viewOffsetX, viewOffsetY, viewScale, visibleGraph.edges.length, visibleGraph.nodes.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const delta = event.deltaY < 0 ? 1.08 : 0.92;
      setViewScale((prev) => clamp(prev * delta, 0.35, 3.4));
    };

    const onPointerDown = (event: PointerEvent) => {
      dragRef.current = {
        active: true,
        startX: event.clientX,
        startY: event.clientY,
        baseOffsetX: viewOffsetX,
        baseOffsetY: viewOffsetY,
      };
      canvas.setPointerCapture(event.pointerId);
    };

    const onPointerMove = (event: PointerEvent) => {
      if (!dragRef.current.active) {
        return;
      }
      const dx = event.clientX - dragRef.current.startX;
      const dy = event.clientY - dragRef.current.startY;
      setViewOffsetX(dragRef.current.baseOffsetX + dx);
      setViewOffsetY(dragRef.current.baseOffsetY + dy);
    };

    const onPointerUp = (event: PointerEvent) => {
      const wasDragging = dragRef.current.active;
      dragRef.current.active = false;
      canvas.releasePointerCapture(event.pointerId);

      if (!wasDragging) {
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const worldX =
        ((event.clientX - rect.left) * dpr - canvas.width / 2 - viewOffsetX * dpr) /
        (viewScale * dpr);
      const worldY =
        ((event.clientY - rect.top) * dpr - canvas.height / 2 - viewOffsetY * dpr) /
        (viewScale * dpr);

      let bestId: string | null = null;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (const node of renderGraph.nodes) {
        const point = layout.get(node.id);
        if (!point) {
          continue;
        }
        const distance = Math.hypot(point.x - worldX, point.y - worldY);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestId = node.id;
        }
      }

      if (bestDistance <= 16) {
        setSelectedNodeId(bestId);
      }
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);

    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
    };
  }, [layout, renderGraph.nodes, viewOffsetX, viewOffsetY, viewScale]);

  const domainOptions = useMemo(() => {
    const fromStatus = Object.keys(status?.domain_distribution || {});
    return fromStatus.sort((a, b) => a.localeCompare(b));
  }, [status?.domain_distribution]);

  return (
    <section className="card relative overflow-hidden">
      <div className="absolute top-0 left-0 w-1 h-full bg-orange-400 opacity-70" />
      <h2 className="text-3xl font-bold mb-1">Web Graph Weaver / Web Graph Weaver</h2>
      <p className="text-muted mb-5 text-sm">
        Ethical crawl instrumentation: discover, validate, fetch, parse, and map relationship growth.
      </p>

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5 mb-4">
        <div className="mindfuck-card">
          <p className="mindfuck-k">crawl state</p>
          <p className="mindfuck-v capitalize">{status?.state || "stopped"}</p>
          <p className="mindfuck-small">ws {connection}</p>
        </div>
        <div className="mindfuck-card">
          <p className="mindfuck-k">crawl rate</p>
          <p className="mindfuck-v">{status?.metrics?.crawl_rate_nodes_per_sec?.toFixed(2) || "0.00"}/s</p>
          <p className="mindfuck-small">frontier {status?.metrics?.frontier_size ?? 0}</p>
        </div>
        <div className="mindfuck-card">
          <p className="mindfuck-k">compliance</p>
          <p className="mindfuck-v">{status?.metrics?.compliance_percent?.toFixed(1) || "0.0"}%</p>
          <p className="mindfuck-small">robots blocked {status?.metrics?.robots_blocked ?? 0}</p>
        </div>
        <div className="mindfuck-card">
          <p className="mindfuck-k">graph</p>
          <p className="mindfuck-v">{status?.graph_counts?.nodes_total ?? 0}</p>
          <p className="mindfuck-small">edges {status?.graph_counts?.edges_total ?? 0}</p>
        </div>
        <div className="mindfuck-card">
          <p className="mindfuck-k">crawler load</p>
          <p className="mindfuck-v">{status?.metrics?.active_fetchers ?? 0}</p>
          <p className="mindfuck-small">avg {status?.metrics?.average_fetch_ms ?? 0}ms</p>
        </div>
      </div>

      <div className="grid gap-3 xl:grid-cols-[1.7fr_1fr]">
        <div className="space-y-3">
          <article className="mindfuck-panel">
            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-6">
              <label className="md:col-span-2 xl:col-span-3">
                <span className="mindfuck-k">Seed URLs (one per line or comma)</span>
                <textarea
                  value={seedInput}
                  onChange={(event) => setSeedInput(event.target.value)}
                  className="mindfuck-input min-h-[72px] mt-1"
                />
              </label>

              <label>
                <span className="mindfuck-k">max depth</span>
                <input
                  type="number"
                  min={0}
                  max={8}
                  value={maxDepth}
                  onChange={(event) => setMaxDepth(Number(event.target.value || 0))}
                  className="mindfuck-input mt-1"
                />
              </label>

              <label>
                <span className="mindfuck-k">max nodes</span>
                <input
                  type="number"
                  min={100}
                  max={50000}
                  value={maxNodes}
                  onChange={(event) => setMaxNodes(Number(event.target.value || 100))}
                  className="mindfuck-input mt-1"
                />
              </label>

              <label>
                <span className="mindfuck-k">parallel fetchers</span>
                <input
                  type="number"
                  min={1}
                  max={24}
                  value={concurrency}
                  onChange={(event) => setConcurrency(Number(event.target.value || 1))}
                  className="mindfuck-input mt-1"
                />
              </label>
            </div>

            <div className="mt-3 flex flex-wrap gap-2">
              <button type="button" className="mindfuck-action-btn" onClick={() => postControl("start").catch((err) => setFriendlyError(err))}>
                Start Crawl
              </button>
              <button type="button" className="mindfuck-action-btn" onClick={() => postControl("pause").catch((err) => setFriendlyError(err))}>
                Pause
              </button>
              <button type="button" className="mindfuck-action-btn" onClick={() => postControl("resume").catch((err) => setFriendlyError(err))}>
                Resume
              </button>
              <button type="button" className="mindfuck-action-btn" onClick={() => postControl("stop").catch((err) => setFriendlyError(err))}>
                Stop
              </button>
              <button type="button" className="mindfuck-action-btn" onClick={() => bootstrap().catch((err) => setFriendlyError(err))}>
                Refresh
              </button>
            </div>

            <div className="mt-3 grid gap-2 md:grid-cols-[1fr_auto]">
              <div>
                <span className="mindfuck-k">Opt-out domain</span>
                <input
                  value={optOutInput}
                  onChange={(event) => setOptOutInput(event.target.value)}
                  placeholder="example.com"
                  className="mindfuck-input mt-1"
                />
              </div>
              <button
                type="button"
                className="mindfuck-action-btn self-end"
                onClick={() => addOptOutDomain().catch((err) => setFriendlyError(err))}
              >
                Add Opt-Out
              </button>
            </div>

            <p className="mindfuck-small mt-2">user-agent: {status?.user_agent || "(loading)"}</p>
            <p className="mindfuck-small">endpoint: {activeWeaverBase}</p>
            {errorMessage && <p className="text-xs text-[#f92672] mt-2">{errorMessage}</p>}
          </article>

          <article className="mindfuck-panel">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
              <p className="mindfuck-subhead mb-0">Graph View</p>
              <div className="flex items-center gap-2">
                <span className="mindfuck-small">domain filter</span>
                <select
                  value={domainFilter}
                  onChange={(event) => setDomainFilter(event.target.value)}
                  className="mindfuck-input !py-1 !px-2"
                >
                  <option value="">(all domains)</option>
                  {domainOptions.map((domain) => (
                    <option key={domain} value={domain}>
                      {domain}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="relative rounded-xl overflow-hidden border border-[rgba(102,217,239,0.38)] bg-[rgba(24,25,20,0.96)]">
              <canvas ref={canvasRef} className="block w-full h-[420px]" />
              <div className="absolute top-2 left-2 text-[10px] text-[#66d9ef] bg-[rgba(31,32,29,0.78)] px-2 py-1 rounded">
                wheel = zoom, drag = pan, click node = highlight path
              </div>
            </div>
          </article>
        </div>

        <div className="space-y-3">
          <article className="mindfuck-panel">
            <h3 className="mindfuck-subhead">Crawl Status</h3>
            <ul className="space-y-1.5 text-xs text-ink">
              <li>- discovered: {status?.metrics?.discovered ?? 0}</li>
              <li>- fetched: {status?.metrics?.fetched ?? 0}</li>
              <li>- skipped: {status?.metrics?.skipped ?? 0}</li>
              <li>- robots blocked: {status?.metrics?.robots_blocked ?? 0}</li>
              <li>- duplicate content: {status?.metrics?.duplicate_content ?? 0}</li>
              <li>- errors: {status?.metrics?.errors ?? 0}</li>
            </ul>

            <div className="mt-3">
              <p className="mindfuck-k">active domains</p>
              <p className="mindfuck-small">{(status?.active_domains ?? []).slice(0, 6).join(", ") || "(none)"}</p>
            </div>

            <div className="mt-3">
              <p className="mindfuck-k">opt-out list</p>
              <p className="mindfuck-small">{(status?.opt_out_domains ?? []).join(", ") || "(none)"}</p>
            </div>
          </article>

          <article className="mindfuck-panel">
            <h3 className="mindfuck-subhead">Domain Distribution</h3>
            <div className="space-y-1 max-h-[150px] overflow-auto pr-1">
              {Object.entries(status?.domain_distribution || {})
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([domain, count]) => (
                  <div key={domain} className="flex items-center justify-between text-xs">
                    <span className="truncate mr-2" style={{ color: colorForDomain(domain) }}>
                      {domain}
                    </span>
                    <span className="font-mono text-[#e6db74]">{count}</span>
                  </div>
                ))}
            </div>

            <h3 className="mindfuck-subhead mt-3">Depth Histogram</h3>
            <div className="space-y-1">
              {Object.entries(status?.depth_histogram || {})
                .sort((a, b) => Number(a[0]) - Number(b[0]))
                .map(([depth, count]) => (
                  <div key={depth} className="flex items-center justify-between text-xs">
                    <span className="text-muted">depth {depth}</span>
                    <span className="font-mono text-[#e6db74]">{count}</span>
                  </div>
                ))}
            </div>
          </article>

          <article className="mindfuck-panel">
            <h3 className="mindfuck-subhead">Event Stream</h3>
            <div className="mindfuck-table-wrap max-h-[300px]">
              <table className="mindfuck-table">
                <thead>
                  <tr>
                    <th>ts</th>
                    <th>event</th>
                    <th>target</th>
                  </tr>
                </thead>
                <tbody>
                  {[...events].slice(-120).reverse().map((event, index) => (
                    <tr key={`${event.timestamp}-${event.event}-${index}`}>
                      <td className="font-mono">{new Date(event.timestamp).toLocaleTimeString()}</td>
                      <td>{event.event}</td>
                      <td className="font-mono">{shortText(eventTargetValue(event), 88)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </div>
      </div>
    </section>
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
