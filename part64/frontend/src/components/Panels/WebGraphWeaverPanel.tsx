import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { runtimeWeaverBaseCandidates } from "../../runtime/endpoints";

type CrawlState = "running" | "paused" | "stopped";

interface WeaverNode {
  id: string;
  kind: "url" | "domain" | "content";
  label: string;
  url?: string;
  domain?: string;
  depth?: number;
  source_url?: string | null;
  status?: string;
  compliance?: string;
  content_type?: string | null;
  title?: string;
  activation_potential?: number;
  interaction_count?: number;
  cooldown_until?: number;
  last_requested_at?: number;
  analysis_summary?: string;
  analysis_provider?: string;
  last_analyzed_at?: number;
}

interface WeaverEdge {
  id: string;
  kind:
    | "hyperlink"
    | "canonical_redirect"
    | "domain_membership"
    | "content_membership"
    | "citation"
    | "wiki_reference"
    | "cross_reference"
    | "paper_pdf";
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
    semantic_edges?: number;
    citation_edges?: number;
    wiki_reference_edges?: number;
    cross_reference_edges?: number;
    paper_pdf_edges?: number;
    host_concurrency_waits?: number;
    cooldown_blocked?: number;
    interactions?: number;
    activation_enqueues?: number;
    entity_moves?: number;
    entity_visits?: number;
    llm_analysis_success?: number;
    llm_analysis_fail?: number;
  };
  config?: {
    max_depth?: number;
    max_nodes?: number;
    concurrency?: number;
    max_requests_per_host?: number;
    node_cooldown_ms?: number;
    activation_threshold?: number;
    default_delay_ms?: number;
  };
  graph_counts: {
    nodes_total: number;
    edges_total: number;
    url_nodes_total: number;
  };
  knowledge?: {
    source_families?: {
      arxiv?: number;
      wikipedia?: number;
      web?: number;
    };
    node_kinds?: {
      arxiv_abs?: number;
      arxiv_pdf?: number;
      wikipedia_article?: number;
      web_url?: number;
    };
  };
  entities?: WeaverEntityStateEnvelope;
  llm?: {
    enabled?: boolean;
    model?: string;
    base_url?: string;
  };
}

interface WeaverEntityState {
  id: string;
  label: string;
  state: "idle" | "moving" | "visiting" | "cooldown";
  current_url?: string | null;
  from_url?: string | null;
  target_url?: string | null;
  progress?: number;
  visits?: number;
  last_visit_at?: number;
  next_available_at?: number;
}

interface WeaverEntityStateEnvelope {
  ok?: boolean;
  enabled?: boolean;
  paused?: boolean;
  count?: number;
  activation_threshold?: number;
  node_cooldown_ms?: number;
  max_requests_per_host?: number;
  entities?: WeaverEntityState[];
  llm?: {
    enabled?: boolean;
    model?: string;
    base_url?: string;
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
  return uniqueStrings(runtimeWeaverBaseCandidates());
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

function edgeLayoutPriority(kind: WeaverEdge["kind"]): number {
  switch (kind) {
    case "hyperlink":
      return 0;
    case "canonical_redirect":
      return 1;
    case "citation":
      return 2;
    case "wiki_reference":
      return 3;
    case "cross_reference":
      return 4;
    case "paper_pdf":
      return 5;
    default:
      return 9;
  }
}

function useGraphLayout(nodes: WeaverNode[], edges: WeaverEdge[]) {
  return useMemo(() => {
    const points = new Map<string, GraphPoint>();
    const domainNodes = nodes.filter((node) => node.kind === "domain");
    const urlNodes = nodes.filter((node) => node.kind === "url");
    const contentNodes = nodes.filter((node) => node.kind === "content");
    const urlNodeIds = new Set(urlNodes.map((node) => node.id));

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

    const parentById = new Map<string, string>();
    for (const node of urlNodes) {
      const sourceUrl = String(node.source_url || "").trim();
      if (!sourceUrl) {
        continue;
      }
      const sourceId = `url:${sourceUrl}`;
      if (sourceId !== node.id && urlNodeIds.has(sourceId)) {
        parentById.set(node.id, sourceId);
      }
    }

    const inboundByTarget = new Map<string, Array<{ sourceId: string; rank: number }>>();
    for (const edge of edges) {
      if (!urlNodeIds.has(edge.source) || !urlNodeIds.has(edge.target)) {
        continue;
      }
      if (edge.source === edge.target) {
        continue;
      }
      const rows = inboundByTarget.get(edge.target) || [];
      rows.push({
        sourceId: edge.source,
        rank: edgeLayoutPriority(edge.kind),
      });
      inboundByTarget.set(edge.target, rows);
    }

    for (const node of urlNodes) {
      if (parentById.has(node.id)) {
        continue;
      }
      const candidates = inboundByTarget.get(node.id);
      if (!candidates || candidates.length === 0) {
        continue;
      }
      candidates.sort((left, right) => {
        if (left.rank !== right.rank) {
          return left.rank - right.rank;
        }
        return left.sourceId.localeCompare(right.sourceId);
      });
      const chosen = candidates[0];
      if (chosen.sourceId && chosen.sourceId !== node.id) {
        parentById.set(node.id, chosen.sourceId);
      }
    }

    const childrenByParent = new Map<string, WeaverNode[]>();
    for (const node of urlNodes) {
      const parentId = parentById.get(node.id);
      if (!parentId) {
        continue;
      }
      const children = childrenByParent.get(parentId) || [];
      children.push(node);
      childrenByParent.set(parentId, children);
    }

    const roots = urlNodes
      .filter((node) => !parentById.has(node.id))
      .sort((left, right) => {
        const depthDelta = Number(left.depth || 0) - Number(right.depth || 0);
        if (depthDelta !== 0) {
          return depthDelta;
        }
        return left.id.localeCompare(right.id);
      });

    const rootIndexByDomain = new Map<string, number>();
    for (const node of roots) {
      const domain = node.domain || "unknown";
      const anchor = domainAnchors.get(domain) || { x: 0, y: 0 };
      const idx = rootIndexByDomain.get(domain) || 0;
      const depth = Number(node.depth || 0);
      const angle = idx * 0.86 + (hashString(node.id) % 628) / 100;
      const radial = 34 + depth * 26 + (idx % 7) * 13;
      const x = anchor.x + Math.cos(angle) * radial;
      const y = anchor.y + Math.sin(angle) * radial * 0.82;
      points.set(node.id, { x, y });
      rootIndexByDomain.set(domain, idx + 1);
    }

    const placedUrlIds = new Set(roots.map((node) => node.id));
    const queue = roots.map((node) => node.id);
    while (queue.length > 0) {
      const parentId = queue.shift();
      if (!parentId) {
        break;
      }
      const parentPoint = points.get(parentId);
      if (!parentPoint) {
        continue;
      }
      const children = [...(childrenByParent.get(parentId) || [])].sort((left, right) =>
        left.id.localeCompare(right.id),
      );
      if (children.length === 0) {
        continue;
      }

      const spread = Math.min(Math.PI * 1.35, 0.52 * Math.max(1, children.length - 1));
      const anchorAngle = (hashString(parentId) % 628) / 100;

      for (let index = 0; index < children.length; index += 1) {
        const child = children[index];
        if (placedUrlIds.has(child.id)) {
          continue;
        }
        const ratio = children.length <= 1 ? 0.5 : index / (children.length - 1);
        const angle = anchorAngle - spread / 2 + spread * ratio;
        const depth = Number(child.depth || 0);
        const radial = 26 + depth * 12 + (index % 3) * 7;
        const x = parentPoint.x + Math.cos(angle) * radial;
        const y = parentPoint.y + Math.sin(angle) * radial * 0.86;
        points.set(child.id, { x, y });
        placedUrlIds.add(child.id);
        queue.push(child.id);
      }
    }

    const fallbackIndexByDomain = new Map<string, number>();
    for (const node of urlNodes) {
      if (placedUrlIds.has(node.id)) {
        continue;
      }
      const domain = node.domain || "unknown";
      const anchor = domainAnchors.get(domain) || { x: 0, y: 0 };
      const idx = fallbackIndexByDomain.get(domain) || 0;
      const depth = Number(node.depth || 0);
      const angle = idx * 0.93 + (hashString(node.id) % 628) / 100;
      const radial = 48 + depth * 14 + (idx % 5) * 9;
      points.set(node.id, {
        x: anchor.x + Math.cos(angle) * radial,
        y: anchor.y + Math.sin(angle) * radial * 0.84,
      });
      fallbackIndexByDomain.set(domain, idx + 1);
      placedUrlIds.add(node.id);
    }

    const contentNodeIds = new Set(contentNodes.map((node) => node.id));
    const contentIndexByUrlNode = new Map<string, number>();
    for (const edge of edges) {
      if (edge.kind !== "content_membership") {
        continue;
      }
      let contentId = "";
      let urlId = "";
      if (contentNodeIds.has(edge.source) && urlNodeIds.has(edge.target)) {
        contentId = edge.source;
        urlId = edge.target;
      } else if (contentNodeIds.has(edge.target) && urlNodeIds.has(edge.source)) {
        contentId = edge.target;
        urlId = edge.source;
      }
      if (!contentId || points.has(contentId)) {
        continue;
      }
      const anchor = points.get(urlId);
      if (!anchor) {
        continue;
      }
      const idx = contentIndexByUrlNode.get(urlId) || 0;
      const angle = (hashString(contentId) % 628) / 100 + idx * 0.9;
      const radial = 18 + (idx % 4) * 10;
      points.set(contentId, {
        x: anchor.x + Math.cos(angle) * radial,
        y: anchor.y + Math.sin(angle) * radial * 0.8,
      });
      contentIndexByUrlNode.set(urlId, idx + 1);
    }

    const contentCount = Math.max(1, contentNodes.length);
    contentNodes.forEach((node, index) => {
      if (points.has(node.id)) {
        return;
      }
      const x = -220 + (index * 440) / Math.max(1, contentCount - 1);
      const y = 320 + ((index % 2) * 26);
      points.set(node.id, { x, y });
    });

    return points;
  }, [edges, nodes]);
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
  const [seedInput, setSeedInput] = useState(
    "https://arxiv.org/list/cs.AI/recent\nhttps://en.wikipedia.org/wiki/Artificial_intelligence",
  );
  const [maxDepth, setMaxDepth] = useState(3);
  const [maxNodes, setMaxNodes] = useState(2500);
  const [concurrency, setConcurrency] = useState(2);
  const [maxPerHost, setMaxPerHost] = useState(2);
  const [entityCount, setEntityCount] = useState(4);
  const [interactionDelta, setInteractionDelta] = useState(0.35);
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
  const [entityState, setEntityState] = useState<WeaverEntityStateEnvelope | null>(null);

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

  const layout = useGraphLayout(renderGraph.nodes, renderGraph.edges);

  const selectedNode = useMemo(
    () => graph.nodes.find((node) => node.id === selectedNodeId) || null,
    [graph.nodes, selectedNodeId],
  );

  const entityRows = useMemo(
    () => entityState?.entities || status?.entities?.entities || [],
    [entityState?.entities, status?.entities?.entities],
  );

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
    if (payload.entities) {
      setEntityState(payload.entities);
      if (typeof payload.entities.count === "number") {
        setEntityCount(payload.entities.count);
      }
    }
    const maxPerHostFromStatus = Number(payload.config?.max_requests_per_host);
    if (Number.isFinite(maxPerHostFromStatus) && maxPerHostFromStatus > 0) {
      setMaxPerHost(maxPerHostFromStatus);
    }
  }, [requestWeaver]);

  const refreshEntities = useCallback(async () => {
    const response = await requestWeaver("/api/weaver/entities");
    if (!response.ok) {
      throw new Error(`entities request failed (${response.status})`);
    }
    const payload = (await response.json()) as WeaverEntityStateEnvelope;
    setEntityState(payload);
    if (typeof payload.count === "number") {
      setEntityCount(payload.count);
    }
    const maxPerHostFromEntities = Number(payload.max_requests_per_host);
    if (Number.isFinite(maxPerHostFromEntities) && maxPerHostFromEntities > 0) {
      setMaxPerHost(maxPerHostFromEntities);
    }
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
    await Promise.all([
      refreshStatus(),
      refreshGraph(domainFilter),
      refreshEvents(),
      refreshEntities(),
    ]);
  }, [domainFilter, refreshEntities, refreshEvents, refreshGraph, refreshStatus]);

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
          max_per_host: maxPerHost,
          entity_count: entityCount,
        }),
      });
      const payload = await response.json();
      if (!response.ok || payload?.ok !== true) {
        throw new Error(payload?.error || `control action failed (${response.status})`);
      }
      if (payload.status) {
        setStatus(payload.status as WeaverStatus);
        const entitiesFromStatus = (payload.status as WeaverStatus).entities;
        if (entitiesFromStatus) {
          setEntityState(entitiesFromStatus);
        }
      }
      appendEvent({
        event: "crawl_state",
        timestamp: Date.now(),
        reason: action,
      });
    },
    [appendEvent, concurrency, entityCount, maxDepth, maxNodes, maxPerHost, requestWeaver, seedInput],
  );

  const postEntityControl = useCallback(
    async (action: "start" | "pause" | "resume" | "stop" | "configure") => {
      setErrorMessage(null);
      const response = await requestWeaver("/api/weaver/entities/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          count: entityCount,
          max_per_host: maxPerHost,
        }),
      });
      const payload = await response.json();
      if (!response.ok || payload?.ok !== true) {
        throw new Error(payload?.error || `entity control failed (${response.status})`);
      }
      if (payload.status) {
        setStatus(payload.status as WeaverStatus);
      }
      if (payload.status?.entities) {
        setEntityState(payload.status.entities as WeaverEntityStateEnvelope);
      }
      appendEvent({
        event: "entity_control",
        timestamp: Date.now(),
        reason: action,
      });
    },
    [appendEvent, entityCount, maxPerHost, requestWeaver],
  );

  const interactWithNode = useCallback(
    async (url: string) => {
      const normalized = url.trim();
      if (!normalized) {
        return;
      }
      setErrorMessage(null);
      const response = await requestWeaver("/api/weaver/entities/interact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: normalized,
          delta: interactionDelta,
          source: "frontend_manual_interaction",
        }),
      });
      const payload = await response.json();
      if (!response.ok || payload?.ok !== true) {
        throw new Error(payload?.error || `node interaction failed (${response.status})`);
      }
      if (payload.status) {
        setStatus(payload.status as WeaverStatus);
        if ((payload.status as WeaverStatus).entities) {
          setEntityState((payload.status as WeaverStatus).entities as WeaverEntityStateEnvelope);
        }
      }
      await refreshGraph(domainFilter);
    },
    [domainFilter, interactionDelta, refreshGraph, requestWeaver],
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
            entities?: WeaverEntityStateEnvelope;
            graph?: WeaverGraph;
            recent_events?: WeaverEvent[];
            nodes?: WeaverNode[];
            edges?: WeaverEdge[];
          } & WeaverEvent;

          if (payload.event === "snapshot") {
            if (payload.status) {
              setStatus(payload.status);
              if (payload.status.entities) {
                setEntityState(payload.status.entities);
              }
            }
            if (payload.entities) {
              setEntityState(payload.entities);
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

          if (payload.event === "entity_tick") {
            const nextEntities: WeaverEntityStateEnvelope = {
              ok: true,
              enabled: Boolean(payload.entities_enabled),
              paused: Boolean(payload.entities_paused),
              activation_threshold: Number(payload.activation_threshold || 1),
              node_cooldown_ms: Number(payload.node_cooldown_ms || 0),
              max_requests_per_host: Number(payload.max_requests_per_host || 0),
              entities: Array.isArray(payload.entities)
                ? (payload.entities as WeaverEntityState[])
                : [],
            };
            setEntityState(nextEntities);
          }

          appendEvent(payload);

          const now = Date.now();
          if (now - statusRefreshRef.current > 900) {
            statusRefreshRef.current = now;
            refreshStatus().catch(() => {});
          }
        } catch {
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
    const gl = canvas.getContext("webgl", { alpha: false, antialias: true });
    if (!gl) {
      return;
    }

    const lineVertexShaderSource = `
      attribute vec2 aPos;
      attribute vec4 aColor;
      uniform vec2 uResolution;
      varying vec4 vColor;

      void main() {
        vec2 clip = vec2(
          (aPos.x / max(1.0, uResolution.x)) * 2.0 - 1.0,
          1.0 - (aPos.y / max(1.0, uResolution.y)) * 2.0
        );
        gl_Position = vec4(clip, 0.0, 1.0);
        vColor = aColor;
      }
    `;

    const lineFragmentShaderSource = `
      precision mediump float;
      varying vec4 vColor;

      void main() {
        gl_FragColor = vColor;
      }
    `;

    const pointVertexShaderSource = `
      attribute vec2 aPos;
      attribute float aSize;
      attribute vec4 aColor;
      uniform vec2 uResolution;
      varying vec4 vColor;

      void main() {
        vec2 clip = vec2(
          (aPos.x / max(1.0, uResolution.x)) * 2.0 - 1.0,
          1.0 - (aPos.y / max(1.0, uResolution.y)) * 2.0
        );
        gl_Position = vec4(clip, 0.0, 1.0);
        gl_PointSize = aSize;
        vColor = aColor;
      }
    `;

    const pointFragmentShaderSource = `
      precision mediump float;
      varying vec4 vColor;

      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float d2 = dot(p, p);
        if (d2 > 1.0) {
          discard;
        }
        float alpha = (1.0 - d2) * vColor.a;
        gl_FragColor = vec4(vColor.rgb, alpha);
      }
    `;

    const compileShader = (type: number, source: string): WebGLShader | null => {
      const shader = gl.createShader(type);
      if (!shader) {
        return null;
      }
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const createProgram = (
      vertexSource: string,
      fragmentSource: string,
    ): { program: WebGLProgram; vertexShader: WebGLShader; fragmentShader: WebGLShader } | null => {
      const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
      const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
      if (!vertexShader || !fragmentShader) {
        if (vertexShader) {
          gl.deleteShader(vertexShader);
        }
        if (fragmentShader) {
          gl.deleteShader(fragmentShader);
        }
        return null;
      }
      const program = gl.createProgram();
      if (!program) {
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragmentShader);
        return null;
      }
      gl.attachShader(program, vertexShader);
      gl.attachShader(program, fragmentShader);
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        gl.deleteProgram(program);
        gl.deleteShader(vertexShader);
        gl.deleteShader(fragmentShader);
        return null;
      }
      return {
        program,
        vertexShader,
        fragmentShader,
      };
    };

    const lineProgramBundle = createProgram(lineVertexShaderSource, lineFragmentShaderSource);
    const pointProgramBundle = createProgram(pointVertexShaderSource, pointFragmentShaderSource);
    if (!lineProgramBundle || !pointProgramBundle) {
      if (lineProgramBundle) {
        gl.deleteProgram(lineProgramBundle.program);
        gl.deleteShader(lineProgramBundle.vertexShader);
        gl.deleteShader(lineProgramBundle.fragmentShader);
      }
      if (pointProgramBundle) {
        gl.deleteProgram(pointProgramBundle.program);
        gl.deleteShader(pointProgramBundle.vertexShader);
        gl.deleteShader(pointProgramBundle.fragmentShader);
      }
      return;
    }

    const lineBuffer = gl.createBuffer();
    const pointBuffer = gl.createBuffer();
    if (!lineBuffer || !pointBuffer) {
      if (lineBuffer) {
        gl.deleteBuffer(lineBuffer);
      }
      if (pointBuffer) {
        gl.deleteBuffer(pointBuffer);
      }
      gl.deleteProgram(lineProgramBundle.program);
      gl.deleteShader(lineProgramBundle.vertexShader);
      gl.deleteShader(lineProgramBundle.fragmentShader);
      gl.deleteProgram(pointProgramBundle.program);
      gl.deleteShader(pointProgramBundle.vertexShader);
      gl.deleteShader(pointProgramBundle.fragmentShader);
      return;
    }

    const lineLocPos = gl.getAttribLocation(lineProgramBundle.program, "aPos");
    const lineLocColor = gl.getAttribLocation(lineProgramBundle.program, "aColor");
    const lineLocResolution = gl.getUniformLocation(lineProgramBundle.program, "uResolution");

    const pointLocPos = gl.getAttribLocation(pointProgramBundle.program, "aPos");
    const pointLocSize = gl.getAttribLocation(pointProgramBundle.program, "aSize");
    const pointLocColor = gl.getAttribLocation(pointProgramBundle.program, "aColor");
    const pointLocResolution = gl.getUniformLocation(pointProgramBundle.program, "uResolution");

    const activateLineProgram = gl.useProgram.bind(gl, lineProgramBundle.program);
    const activatePointProgram = gl.useProgram.bind(gl, pointProgramBundle.program);

    let rafId = 0;
    let width = 0;
    let height = 0;
    let lastPaintAt = 0;

    const hueToRgb = (hue: number): [number, number, number] => {
      const h = ((hue % 360) + 360) % 360;
      const c = 0.7;
      const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
      const m = 0.16;
      let r = 0;
      let g = 0;
      let b = 0;
      if (h < 60) {
        r = c;
        g = x;
      } else if (h < 120) {
        r = x;
        g = c;
      } else if (h < 180) {
        g = c;
        b = x;
      } else if (h < 240) {
        g = x;
        b = c;
      } else if (h < 300) {
        r = x;
        b = c;
      } else {
        r = c;
        b = x;
      }
      return [r + m, g + m, b + m];
    };

    const addLine = (
      rows: number[],
      x0: number,
      y0: number,
      x1: number,
      y1: number,
      r: number,
      g: number,
      b: number,
      a: number,
    ) => {
      rows.push(x0, y0, r, g, b, a);
      rows.push(x1, y1, r, g, b, a);
    };

    const addPoint = (
      rows: number[],
      x: number,
      y: number,
      size: number,
      r: number,
      g: number,
      b: number,
      a: number,
    ) => {
      rows.push(x, y, size, r, g, b, a);
    };

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

      gl.viewport(0, 0, width, height);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.clearColor(0.035, 0.062, 0.11, 0.96);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Disable any previously enabled attributes to prevent state leakage between programs
      gl.disableVertexAttribArray(0);
      gl.disableVertexAttribArray(1);
      gl.disableVertexAttribArray(2);

      const centerX = width / 2 + viewOffsetX * dpr;
      const centerY = height / 2 + viewOffsetY * dpr;
      const scale = viewScale * dpr;

      const highlightedEdgeIds = new Set<string>();
      if (selectedNodeId) {
        for (const edge of renderGraph.edges) {
          if (edge.source === selectedNodeId || edge.target === selectedNodeId) {
            highlightedEdgeIds.add(edge.id);
          }
        }
      }

      const edgeRows: number[] = [];
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

        let r = 0.47;
        let g = 0.72;
        let b = 0.86;
        let a = 0.15 + pulse * 0.08;

        if (edge.kind === "hyperlink") {
          r = 0.41;
          g = 0.78;
          b = 0.9;
          a = highlighted ? (0.66 + pulse * 0.2) : (0.16 + pulse * 0.12);
        } else if (edge.kind === "canonical_redirect") {
          r = 0.78;
          g = 0.47;
          b = 0.9;
          a = highlighted ? 0.72 : 0.28;
        } else if (edge.kind === "domain_membership") {
          r = 0.95;
          g = 0.73;
          b = 0.42;
          a = highlighted ? 0.7 : 0.22;
        } else if (edge.kind === "citation") {
          r = 0.95;
          g = 0.62;
          b = 0.38;
          a = highlighted ? 0.78 : 0.34;
        } else if (edge.kind === "cross_reference") {
          r = 0.9;
          g = 0.48;
          b = 0.84;
          a = highlighted ? 0.82 : 0.38;
        } else if (edge.kind === "paper_pdf") {
          r = 0.46;
          g = 0.84;
          b = 0.9;
          a = highlighted ? 0.74 : 0.32;
        } else {
          r = 0.58;
          g = 0.85;
          b = 0.62;
          a = highlighted ? 0.68 : 0.24;
        }

        addLine(edgeRows, sx, sy, tx, ty, r, g, b, a);
      }

      if (edgeRows.length > 0 && lineLocPos >= 0 && lineLocColor >= 0) {
        activateLineProgram();
        gl.bindBuffer(gl.ARRAY_BUFFER, lineBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(edgeRows), gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(lineLocPos);
        gl.vertexAttribPointer(lineLocPos, 2, gl.FLOAT, false, 6 * 4, 0);
        gl.enableVertexAttribArray(lineLocColor);
        gl.vertexAttribPointer(lineLocColor, 4, gl.FLOAT, false, 6 * 4, 2 * 4);
        if (lineLocResolution) {
          gl.uniform2f(lineLocResolution, width, height);
        }
        gl.drawArrays(gl.LINES, 0, edgeRows.length / 6);
        // Disable line attributes after drawing
        gl.disableVertexAttribArray(lineLocPos);
        gl.disableVertexAttribArray(lineLocColor);
      }

      const pointRows: number[] = [];

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
          addPoint(pointRows, x, y, (radius + 4.8) * dpr, 1.0, 0.94, 0.72, 0.36);
        }

        let r = 0.5;
        let g = 0.78;
        let b = 0.95;

        if (node.kind === "domain") {
          const hue = hashString(String(node.domain || node.label)) % 360;
          const rgb = hueToRgb(hue);
          r = rgb[0];
          g = rgb[1];
          b = rgb[2];
        } else if (node.kind === "content") {
          r = 0.46;
          g = 0.84;
          b = 0.56;
        } else if (node.compliance === "robots_blocked" || node.status === "blocked") {
          r = 1.0;
          g = 0.44;
          b = 0.44;
        }

        addPoint(pointRows, x, y, radius * dpr, r, g, b, 0.9);
      }

      for (let i = 0; i < entityRows.length; i += 1) {
        const entity = entityRows[i];
        const fromUrl = String(entity.from_url || entity.current_url || "");
        const targetUrl = String(entity.target_url || entity.current_url || "");
        if (!targetUrl) {
          continue;
        }
        const fromPoint = layout.get(`url:${fromUrl}`) || layout.get(`url:${targetUrl}`);
        const targetPoint = layout.get(`url:${targetUrl}`) || fromPoint;
        if (!fromPoint || !targetPoint) {
          continue;
        }
        const progress = entity.state === "moving"
          ? clamp(Number(entity.progress || 0), 0, 1)
          : 1;
        const px = fromPoint.x + (targetPoint.x - fromPoint.x) * progress;
        const py = fromPoint.y + (targetPoint.y - fromPoint.y) * progress;
        const x = centerX + px * scale;
        const y = centerY + py * scale;

        const hue = (hashString(entity.id || String(i)) % 360);
        const [r, g, b] = hueToRgb(hue);
        addPoint(pointRows, x, y, (denseMode ? 2.6 : 4.2) * dpr, r, g, b, 0.96);
        if (!denseMode) {
          addPoint(pointRows, x, y, 8.2 * dpr, r, g, b, 0.3);
        }
      }

      if (pointRows.length > 0 && pointLocPos >= 0 && pointLocSize >= 0 && pointLocColor >= 0) {
        activatePointProgram();
        gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pointRows), gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(pointLocPos);
        gl.vertexAttribPointer(pointLocPos, 2, gl.FLOAT, false, 7 * 4, 0);
        gl.enableVertexAttribArray(pointLocSize);
        gl.vertexAttribPointer(pointLocSize, 1, gl.FLOAT, false, 7 * 4, 2 * 4);
        gl.enableVertexAttribArray(pointLocColor);
        gl.vertexAttribPointer(pointLocColor, 4, gl.FLOAT, false, 7 * 4, 3 * 4);
        if (pointLocResolution) {
          gl.uniform2f(pointLocResolution, width, height);
        }
        gl.drawArrays(gl.POINTS, 0, pointRows.length / 7);
        // Disable point attributes after drawing
        gl.disableVertexAttribArray(pointLocPos);
        gl.disableVertexAttribArray(pointLocSize);
        gl.disableVertexAttribArray(pointLocColor);
      }

      rafId = window.requestAnimationFrame(draw);
    };

    rafId = window.requestAnimationFrame(draw);
    return () => {
      window.cancelAnimationFrame(rafId);
      gl.deleteBuffer(lineBuffer);
      gl.deleteBuffer(pointBuffer);
      gl.deleteProgram(lineProgramBundle.program);
      gl.deleteShader(lineProgramBundle.vertexShader);
      gl.deleteShader(lineProgramBundle.fragmentShader);
      gl.deleteProgram(pointProgramBundle.program);
      gl.deleteShader(pointProgramBundle.vertexShader);
      gl.deleteShader(pointProgramBundle.fragmentShader);
    };
  }, [entityRows, layout, renderGraph.edges, renderGraph.nodes, renderGraph.sampledEdges, renderGraph.sampledNodes, selectedNodeId, viewOffsetX, viewOffsetY, viewScale]);

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
        Includes citation + PDF edges for arXiv and reference cross-links for Wikipedia.
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
            <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-8">
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

              <label>
                <span className="mindfuck-k">max per host</span>
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={maxPerHost}
                  onChange={(event) => setMaxPerHost(Number(event.target.value || 1))}
                  className="mindfuck-input mt-1"
                />
              </label>

              <label>
                <span className="mindfuck-k">entity walkers</span>
                <input
                  type="number"
                  min={0}
                  max={24}
                  value={entityCount}
                  onChange={(event) => setEntityCount(Number(event.target.value || 0))}
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
              <button type="button" className="mindfuck-action-btn" onClick={() => postEntityControl("start").catch((err) => setFriendlyError(err))}>
                Start Entities
              </button>
              <button type="button" className="mindfuck-action-btn" onClick={() => postEntityControl("pause").catch((err) => setFriendlyError(err))}>
                Pause Entities
              </button>
              <button type="button" className="mindfuck-action-btn" onClick={() => postEntityControl("resume").catch((err) => setFriendlyError(err))}>
                Resume Entities
              </button>
              <button type="button" className="mindfuck-action-btn" onClick={() => postEntityControl("configure").catch((err) => setFriendlyError(err))}>
                Apply Entity Config
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

            <div className="mt-3 grid gap-2 md:grid-cols-[1fr_auto]">
              <div>
                <span className="mindfuck-k">interaction delta</span>
                <input
                  type="number"
                  min={0.05}
                  max={2}
                  step={0.05}
                  value={interactionDelta}
                  onChange={(event) => setInteractionDelta(Number(event.target.value || 0.05))}
                  className="mindfuck-input mt-1"
                />
              </div>
              <button
                type="button"
                className="mindfuck-action-btn self-end"
                onClick={() => selectedNode?.url && interactWithNode(selectedNode.url).catch((err) => setFriendlyError(err))}
                disabled={!selectedNode?.url}
              >
                Interact Selected Node
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

            <div className="mt-3 rounded-lg border border-[rgba(102,217,239,0.28)] p-3 bg-[rgba(17,22,31,0.55)]">
              <p className="mindfuck-k">Selected node</p>
              {!selectedNode && <p className="mindfuck-small">(click a URL node in the graph)</p>}
              {selectedNode && (
                <div className="space-y-1 text-xs text-ink">
                  <p className="font-mono break-all">{selectedNode.url || selectedNode.label}</p>
                  <p className="text-muted">
                    - discovered from: {selectedNode.source_url ? shortText(selectedNode.source_url, 96) : "(seed/manual)"}
                  </p>
                  <p>- activation: {Number(selectedNode.activation_potential || 0).toFixed(3)}</p>
                  <p>- interactions: {selectedNode.interaction_count ?? 0}</p>
                  <p>
                    - cooldown until: {selectedNode.cooldown_until
                      ? new Date(selectedNode.cooldown_until).toLocaleTimeString()
                      : "ready"}
                  </p>
                  <p>- analysis provider: {selectedNode.analysis_provider || "(none yet)"}</p>
                  {selectedNode.analysis_summary && (
                    <p className="text-[11px] text-[#dceaf8] leading-snug">
                      {shortText(selectedNode.analysis_summary, 240)}
                    </p>
                  )}
                </div>
              )}
            </div>
          </article>
        </div>

        <div className="space-y-3">
          <article className="mindfuck-panel">
            <h3 className="mindfuck-subhead">Entity Walkers</h3>
            <p className="mindfuck-small">
              state: {entityState?.enabled ? (entityState?.paused ? "paused" : "active") : "disabled"}
              {" · "}
              count {entityRows.length}
            </p>
            <p className="mindfuck-small">
              cooldown {Math.round(Number(entityState?.node_cooldown_ms || status?.config?.node_cooldown_ms || 0) / 1000)}s
              {" · "}
              max/host {entityState?.max_requests_per_host ?? status?.config?.max_requests_per_host ?? maxPerHost}
            </p>
            <div className="mt-2 space-y-1 max-h-[170px] overflow-auto pr-1">
              {entityRows.length === 0 && <p className="mindfuck-small">(no entities)</p>}
              {entityRows.slice(0, 16).map((entity) => (
                <div key={entity.id} className="rounded border border-[rgba(102,217,239,0.25)] px-2 py-1.5 text-xs">
                  <p className="font-mono text-[#dceaf8]">
                    {entity.label} · {entity.state}
                  </p>
                  <p className="text-muted font-mono break-all">
                    {shortText(String(entity.current_url || entity.target_url || "(idle)"), 72)}
                  </p>
                  <p className="text-muted">visits {entity.visits ?? 0}</p>
                </div>
              ))}
            </div>
          </article>

          <article className="mindfuck-panel">
            <h3 className="mindfuck-subhead">Crawl Status</h3>
            <ul className="space-y-1.5 text-xs text-ink">
              <li>- discovered: {status?.metrics?.discovered ?? 0}</li>
              <li>- fetched: {status?.metrics?.fetched ?? 0}</li>
              <li>- skipped: {status?.metrics?.skipped ?? 0}</li>
              <li>- robots blocked: {status?.metrics?.robots_blocked ?? 0}</li>
              <li>- duplicate content: {status?.metrics?.duplicate_content ?? 0}</li>
              <li>- errors: {status?.metrics?.errors ?? 0}</li>
              <li>- citation edges: {status?.metrics?.citation_edges ?? 0}</li>
              <li>- wiki references: {status?.metrics?.wiki_reference_edges ?? 0}</li>
              <li>- cross references: {status?.metrics?.cross_reference_edges ?? 0}</li>
              <li>- paper pdf links: {status?.metrics?.paper_pdf_edges ?? 0}</li>
              <li>- host waits: {status?.metrics?.host_concurrency_waits ?? 0}</li>
              <li>- cooldown blocked: {status?.metrics?.cooldown_blocked ?? 0}</li>
              <li>- interactions: {status?.metrics?.interactions ?? 0}</li>
              <li>- activation enqueues: {status?.metrics?.activation_enqueues ?? 0}</li>
              <li>- entity moves: {status?.metrics?.entity_moves ?? 0}</li>
              <li>- entity visits: {status?.metrics?.entity_visits ?? 0}</li>
              <li>- llm analyzed: {status?.metrics?.llm_analysis_success ?? 0}</li>
              <li>- llm failed: {status?.metrics?.llm_analysis_fail ?? 0}</li>
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
