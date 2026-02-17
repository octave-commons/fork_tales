const fs = require("fs");
const http = require("http");
const path = require("path");
const crypto = require("crypto");
const { URL } = require("url");
const { WebSocketServer } = require("ws");

const HOST = process.env.WEAVER_HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.WEAVER_PORT || "8793", 10);
const MAX_EVENTS = Number.parseInt(process.env.WEAVER_MAX_EVENTS || "1200", 10);
const DEFAULT_MAX_DEPTH = Number.parseInt(process.env.WEAVER_MAX_DEPTH || "3", 10);
const DEFAULT_MAX_NODES = Number.parseInt(process.env.WEAVER_MAX_NODES || "10000", 10);
const DEFAULT_CONCURRENCY = Number.parseInt(
  process.env.WEAVER_CONCURRENCY || "2",
  10,
);
const DEFAULT_DELAY_MS = Number.parseInt(
  process.env.WEAVER_DEFAULT_DELAY_MS || "1200",
  10,
);
const FETCH_TIMEOUT_MS = Number.parseInt(
  process.env.WEAVER_FETCH_TIMEOUT_MS || "12000",
  10,
);
const ROBOTS_CACHE_TTL_MS = Number.parseInt(
  process.env.WEAVER_ROBOTS_CACHE_TTL_MS || String(60 * 60 * 1000),
  10,
);
const USER_AGENT =
  process.env.WEAVER_USER_AGENT ||
  `WebGraphWeaver/0.1 (+http://${HOST}:${PORT}/api/weaver/opt-out)`;

const PART_ROOT = path.join(__dirname, "..");
const WORLD_STATE_DIR = path.join(PART_ROOT, "world_state");
const EVENT_LOG_PATH = path.join(WORLD_STATE_DIR, "web_graph_weaver.events.jsonl");
const DELTA_LOG_PATH = path.join(
  WORLD_STATE_DIR,
  "web_graph_weaver.graph_delta.jsonl",
);
const SNAPSHOT_PATH = path.join(WORLD_STATE_DIR, "web_graph_weaver.snapshot.json");

function ensureWorldStateDir() {
  if (!fs.existsSync(WORLD_STATE_DIR)) {
    fs.mkdirSync(WORLD_STATE_DIR, { recursive: true });
  }
}

function nowMs() {
  return Date.now();
}

function appendJsonLine(filePath, payload) {
  const line = `${JSON.stringify(payload)}\n`;
  fs.appendFile(filePath, line, () => {});
}

function hashText(input) {
  return crypto.createHash("sha1").update(input).digest("hex");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function parseJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => {
      if (chunks.length === 0) {
        resolve({});
        return;
      }
      const bodyText = Buffer.concat(chunks).toString("utf-8");
      try {
        resolve(JSON.parse(bodyText));
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, statusCode, payload) {
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  res.end(JSON.stringify(payload));
}

function normalizeDomain(input) {
  const value = String(input || "").trim().toLowerCase();
  if (!value) {
    return "";
  }
  return value.replace(/^https?:\/\//, "").replace(/\/$/, "");
}

function normalizeUrl(raw, base) {
  try {
    const url = base ? new URL(raw, base) : new URL(raw);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return null;
    }
    url.hash = "";
    url.username = "";
    url.password = "";
    if (url.protocol === "http:" && url.port === "80") {
      url.port = "";
    }
    if (url.protocol === "https:" && url.port === "443") {
      url.port = "";
    }
    const cleanPath = url.pathname.replace(/\/+/g, "/");
    if (cleanPath.length > 1) {
      url.pathname = cleanPath.replace(/\/+$/, "");
    } else {
      url.pathname = "/";
    }
    const sortedParams = [...url.searchParams.entries()].sort((a, b) => {
      if (a[0] === b[0]) {
        return a[1].localeCompare(b[1]);
      }
      return a[0].localeCompare(b[0]);
    });
    url.search = "";
    for (const [key, value] of sortedParams) {
      url.searchParams.append(key, value);
    }
    return url.toString();
  } catch (_err) {
    return null;
  }
}

function parseContentType(contentTypeHeader) {
  if (!contentTypeHeader) {
    return "application/octet-stream";
  }
  return String(contentTypeHeader).split(";")[0].trim().toLowerCase() || "application/octet-stream";
}

function extractCanonicalHref(html) {
  const match = /<link[^>]+rel\s*=\s*["'][^"']*canonical[^"']*["'][^>]*href\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s"'>]+))/i.exec(
    html,
  );
  if (!match) {
    return "";
  }
  return String(match[1] || match[2] || match[3] || "").trim();
}

function extractTitle(html) {
  const match = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html);
  if (!match) {
    return "";
  }
  return match[1].replace(/\s+/g, " ").trim().slice(0, 160);
}

function extractLinks(html, baseUrl) {
  const links = [];
  const seen = new Set();
  const pushLink = (url, nofollow) => {
    if (!url || seen.has(url)) {
      return;
    }
    seen.add(url);
    links.push({ url, nofollow });
  };

  const pageNoFollow = /<meta[^>]+name\s*=\s*["']robots["'][^>]+content\s*=\s*["'][^"']*nofollow[^"']*["'][^>]*>/i.test(
    html,
  );
  const anchorPattern = /<a\s+[^>]*href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'`<>]+))[^>]*>/gi;
  while (true) {
    const match = anchorPattern.exec(html);
    if (match === null) {
      break;
    }
    const tag = match[0] || "";
    const href = String(match[1] || match[2] || match[3] || "").trim();
    if (!href) {
      continue;
    }
    const relMatch = /rel\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'`<>]+))/i.exec(tag);
    const relValue = String(relMatch?.[1] || relMatch?.[2] || relMatch?.[3] || "").toLowerCase();
    const nofollow = pageNoFollow || relValue.split(/\s+/).includes("nofollow");
    const normalized = normalizeUrl(href, baseUrl);
    if (!normalized) {
      continue;
    }
    pushLink(normalized, nofollow);
  }

  const resourcePattern = /<(?:link|script|img|source)\s+[^>]*(?:href|src)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'`<>]+))[^>]*>/gi;
  while (true) {
    const match = resourcePattern.exec(html);
    if (match === null) {
      break;
    }
    const href = String(match[1] || match[2] || match[3] || "").trim();
    if (!href) {
      continue;
    }
    const normalized = normalizeUrl(href, baseUrl);
    if (!normalized) {
      continue;
    }
    pushLink(normalized, false);
  }

  return links;
}

function getDepthHistogram(urlNodes) {
  const histogram = {};
  for (const node of urlNodes) {
    const depth = Number(node.depth || 0);
    histogram[depth] = (histogram[depth] || 0) + 1;
  }
  return histogram;
}

function getDomainDistribution(urlNodes) {
  const distribution = {};
  for (const node of urlNodes) {
    const domain = String(node.domain || "unknown");
    distribution[domain] = (distribution[domain] || 0) + 1;
  }
  return distribution;
}

function normalizeRobotsRule(rawRule) {
  const rule = String(rawRule || "").trim();
  if (!rule) {
    return "";
  }
  if (rule === "/") {
    return "/";
  }
  if (!rule.startsWith("/")) {
    return `/${rule}`;
  }
  return rule;
}

function robotsRuleMatches(pathname, rule) {
  if (!rule) {
    return false;
  }
  if (rule === "/") {
    return true;
  }
  let escaped = rule.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
  escaped = escaped.replace(/\\\*/g, ".*");
  if (escaped.endsWith("$")) {
    return new RegExp(`^${escaped}`).test(pathname);
  }
  return new RegExp(`^${escaped}`).test(pathname);
}

function evaluateRobots(pathname, allowRules, disallowRules) {
  let bestAllow = -1;
  let bestDisallow = -1;
  for (const rule of allowRules) {
    if (robotsRuleMatches(pathname, rule)) {
      bestAllow = Math.max(bestAllow, rule.length);
    }
  }
  for (const rule of disallowRules) {
    if (robotsRuleMatches(pathname, rule)) {
      bestDisallow = Math.max(bestDisallow, rule.length);
    }
  }
  if (bestAllow === -1 && bestDisallow === -1) {
    return true;
  }
  return bestAllow >= bestDisallow;
}

function parseRobotsTxt(text, userAgent) {
  const groups = [];
  let current = {
    agents: [],
    allow: [],
    disallow: [],
    crawlDelayMs: null,
  };

  const lines = String(text || "").split(/\r?\n/);
  const pushCurrent = () => {
    if (
      current.agents.length > 0 ||
      current.allow.length > 0 ||
      current.disallow.length > 0 ||
      current.crawlDelayMs !== null
    ) {
      groups.push(current);
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/#.*/, "").trim();
    if (!line) {
      continue;
    }
    const idx = line.indexOf(":");
    if (idx <= 0) {
      continue;
    }
    const key = line.slice(0, idx).trim().toLowerCase();
    const value = line.slice(idx + 1).trim();

    if (key === "user-agent") {
      if (
        current.agents.length > 0 &&
        (current.allow.length > 0 ||
          current.disallow.length > 0 ||
          current.crawlDelayMs !== null)
      ) {
        pushCurrent();
        current = {
          agents: [],
          allow: [],
          disallow: [],
          crawlDelayMs: null,
        };
      }
      current.agents.push(value.toLowerCase());
      continue;
    }

    if (key === "allow") {
      const rule = normalizeRobotsRule(value);
      if (rule) {
        current.allow.push(rule);
      }
      continue;
    }

    if (key === "disallow") {
      const rule = normalizeRobotsRule(value);
      if (rule) {
        current.disallow.push(rule);
      }
      continue;
    }

    if (key === "crawl-delay") {
      const seconds = Number.parseFloat(value);
      if (Number.isFinite(seconds) && seconds >= 0) {
        current.crawlDelayMs = Math.round(seconds * 1000);
      }
      continue;
    }
  }
  pushCurrent();

  const ua = userAgent.toLowerCase();
  const matchingGroups = groups.filter((group) =>
    group.agents.some((agent) => agent === "*" || ua.includes(agent)),
  );

  if (matchingGroups.length === 0) {
    return {
      allow: [],
      disallow: [],
      crawlDelayMs: null,
    };
  }

  const allow = [];
  const disallow = [];
  let crawlDelayMs = null;
  for (const group of matchingGroups) {
    allow.push(...group.allow);
    disallow.push(...group.disallow);
    if (crawlDelayMs === null && group.crawlDelayMs !== null) {
      crawlDelayMs = group.crawlDelayMs;
    }
  }

  return {
    allow,
    disallow,
    crawlDelayMs,
  };
}

class FrontierQueue {
  constructor() {
    this.items = [];
    this.enqueuedUrls = new Set();
  }

  get size() {
    return this.items.length;
  }

  has(url) {
    return this.enqueuedUrls.has(url);
  }

  clear() {
    this.items = [];
    this.enqueuedUrls.clear();
  }

  push(item) {
    if (!item || !item.url) {
      return false;
    }
    if (this.enqueuedUrls.has(item.url)) {
      return false;
    }
    this.items.push(item);
    this.enqueuedUrls.add(item.url);
    this.items.sort((a, b) => {
      if (a.readyAt !== b.readyAt) {
        return a.readyAt - b.readyAt;
      }
      return b.priority - a.priority;
    });
    return true;
  }

  popReady(now) {
    for (let i = 0; i < this.items.length; i += 1) {
      if (this.items[i].readyAt <= now) {
        const [item] = this.items.splice(i, 1);
        this.enqueuedUrls.delete(item.url);
        return item;
      }
    }
    return null;
  }
}

class GraphStore {
  constructor({ maxUrlNodes }) {
    this.maxUrlNodes = maxUrlNodes;
    this.nodes = new Map();
    this.edges = new Map();
    this.urlNodeCount = 0;
  }

  _makeNodeId(kind, value) {
    return `${kind}:${value}`;
  }

  _makeEdgeId(kind, source, target) {
    return `${kind}|${source}|${target}`;
  }

  _insertNode(node) {
    if (this.nodes.has(node.id)) {
      return null;
    }
    if (node.kind === "url" && this.urlNodeCount >= this.maxUrlNodes) {
      return { rejected: true };
    }
    this.nodes.set(node.id, node);
    if (node.kind === "url") {
      this.urlNodeCount += 1;
    }
    return node;
  }

  upsertDomain(domain) {
    const id = this._makeNodeId("domain", domain);
    const created = this._insertNode({
      id,
      kind: "domain",
      label: domain,
      domain,
      discovered_at: nowMs(),
    });
    return {
      id,
      created: created && !created.rejected ? created : null,
    };
  }

  upsertContentType(contentType) {
    const id = this._makeNodeId("content", contentType);
    const created = this._insertNode({
      id,
      kind: "content",
      label: contentType,
      content_type: contentType,
      discovered_at: nowMs(),
    });
    return {
      id,
      created: created && !created.rejected ? created : null,
    };
  }

  upsertUrl(url, depth, sourceUrl) {
    const parsed = new URL(url);
    const id = this._makeNodeId("url", url);
    if (this.nodes.has(id)) {
      return {
        id,
        created: null,
        rejected: false,
      };
    }
    const created = this._insertNode({
      id,
      kind: "url",
      label: url,
      url,
      domain: parsed.hostname,
      depth,
      status: "discovered",
      source_url: sourceUrl || null,
      discovered_at: nowMs(),
      fetched_at: null,
      compliance: "pending",
      content_type: null,
      content_hash: null,
      duplicate_of: null,
      title: "",
    });
    return {
      id,
      created: created && !created.rejected ? created : null,
      rejected: Boolean(created && created.rejected),
    };
  }

  setUrlStatus(url, patch) {
    const id = this._makeNodeId("url", url);
    const existing = this.nodes.get(id);
    if (!existing) {
      return;
    }
    this.nodes.set(id, {
      ...existing,
      ...patch,
    });
  }

  upsertEdge(kind, source, target, extra = {}) {
    const id = this._makeEdgeId(kind, source, target);
    if (this.edges.has(id)) {
      return null;
    }
    const edge = {
      id,
      kind,
      source,
      target,
      discovered_at: nowMs(),
      ...extra,
    };
    this.edges.set(id, edge);
    return edge;
  }

  getUrlNodes() {
    return [...this.nodes.values()].filter((node) => node.kind === "url");
  }

  toSnapshot({ domainFilter = "", nodeLimit = 5000, edgeLimit = 12000 } = {}) {
    let nodes = [...this.nodes.values()];
    let edges = [...this.edges.values()];

    const normalizedFilter = normalizeDomain(domainFilter);
    if (normalizedFilter) {
      const allowedNodeIds = new Set(
        nodes
          .filter((node) =>
            node.kind === "domain"
              ? node.domain === normalizedFilter
              : node.domain === normalizedFilter,
          )
          .map((node) => node.id),
      );
      for (const edge of edges) {
        if (allowedNodeIds.has(edge.source) || allowedNodeIds.has(edge.target)) {
          allowedNodeIds.add(edge.source);
          allowedNodeIds.add(edge.target);
        }
      }
      nodes = nodes.filter((node) => allowedNodeIds.has(node.id));
      edges = edges.filter(
        (edge) => allowedNodeIds.has(edge.source) && allowedNodeIds.has(edge.target),
      );
    }

    if (nodes.length > nodeLimit) {
      nodes = nodes.slice(nodes.length - nodeLimit);
    }
    if (edges.length > edgeLimit) {
      edges = edges.slice(edges.length - edgeLimit);
    }

    return {
      nodes,
      edges,
      counts: {
        nodes_total: this.nodes.size,
        edges_total: this.edges.size,
        url_nodes_total: this.urlNodeCount,
      },
    };
  }
}

class RobotsCache {
  constructor() {
    this.entries = new Map();
  }

  get(origin) {
    const row = this.entries.get(origin);
    if (!row) {
      return null;
    }
    if (nowMs() - row.fetchedAt > ROBOTS_CACHE_TTL_MS) {
      this.entries.delete(origin);
      return null;
    }
    return row.policy;
  }

  set(origin, policy) {
    this.entries.set(origin, {
      fetchedAt: nowMs(),
      policy,
    });
  }
}

class WebGraphWeaver {
  constructor() {
    this.frontier = new FrontierQueue();
    this.graph = new GraphStore({ maxUrlNodes: DEFAULT_MAX_NODES });
    this.robotsCache = new RobotsCache();
    this.visitedUrls = new Set();
    this.contentHashIndex = new Map();
    this.optOutDomains = new Set();
    this.recentEvents = [];

    this.running = false;
    this.paused = false;
    this.startedAtMs = null;
    this.activeWorkers = 0;
    this.currentConcurrency = DEFAULT_CONCURRENCY;
    this.currentMaxDepth = DEFAULT_MAX_DEPTH;
    this.currentMaxNodes = DEFAULT_MAX_NODES;
    this.defaultDelayMs = DEFAULT_DELAY_MS;

    this.stats = {
      discovered: 0,
      fetched: 0,
      skipped: 0,
      robots_blocked: 0,
      errors: 0,
      duplicates: 0,
      compliance_checks: 0,
      compliance_pass: 0,
      compliance_fail: 0,
      total_fetch_time_ms: 0,
    };

    this.domainState = new Map();

    this.broadcast = () => {};
    this.scheduler = setInterval(() => {
      this.tick();
    }, 120);
  }

  setBroadcast(fn) {
    this.broadcast = fn;
  }

  _emit(event, payload = {}) {
    const row = {
      event,
      timestamp: nowMs(),
      ...payload,
    };
    this.recentEvents.push(row);
    if (this.recentEvents.length > MAX_EVENTS) {
      this.recentEvents.splice(0, this.recentEvents.length - MAX_EVENTS);
    }
    appendJsonLine(EVENT_LOG_PATH, row);
    this.broadcast(row);
    return row;
  }

  _emitGraphDelta(reason, nodes, edges) {
    if ((!nodes || nodes.length === 0) && (!edges || edges.length === 0)) {
      return;
    }
    const payload = {
      reason,
      nodes: nodes || [],
      edges: edges || [],
    };
    appendJsonLine(DELTA_LOG_PATH, {
      timestamp: nowMs(),
      ...payload,
    });
    this._emit("graph_delta", payload);
  }

  _compliancePercent() {
    if (this.stats.compliance_checks <= 0) {
      return 100;
    }
    return Number(
      ((this.stats.compliance_pass / this.stats.compliance_checks) * 100).toFixed(2),
    );
  }

  _markCompliance(ok, details) {
    this.stats.compliance_checks += 1;
    if (ok) {
      this.stats.compliance_pass += 1;
    } else {
      this.stats.compliance_fail += 1;
    }
    this._emit("compliance_update", {
      compliance_percent: this._compliancePercent(),
      checks: this.stats.compliance_checks,
      pass: this.stats.compliance_pass,
      fail: this.stats.compliance_fail,
      ...details,
    });
  }

  _domainState(domain) {
    if (!this.domainState.has(domain)) {
      this.domainState.set(domain, {
        nextAllowedAt: 0,
        crawlDelayMs: this.defaultDelayMs,
        backoffMs: 0,
        active: 0,
        lastFetchedAt: 0,
      });
    }
    return this.domainState.get(domain);
  }

  _priorityFor(url, depth, source) {
    let score = 0;
    score += Math.max(0, 100 - depth * 18);
    if (!source) {
      score += 26;
    }
    const domain = new URL(url).hostname;
    const domainInfo = this._domainState(domain);
    if (domainInfo.lastFetchedAt <= 0) {
      score += 14;
    }
    score += Math.random() * 0.5;
    return score;
  }

  enqueueUrl(rawUrl, sourceUrl, depth, reason = "discovered") {
    const normalized = normalizeUrl(rawUrl, sourceUrl || undefined);
    if (!normalized) {
      return { ok: false, reason: "invalid_url" };
    }
    if (this.visitedUrls.has(normalized)) {
      return { ok: false, reason: "already_visited" };
    }
    if (this.frontier.has(normalized)) {
      return { ok: false, reason: "already_enqueued" };
    }
    if (depth > this.currentMaxDepth) {
      return { ok: false, reason: "max_depth" };
    }

    const nodeResult = this.graph.upsertUrl(normalized, depth, sourceUrl || null);
    if (nodeResult.rejected) {
      return { ok: false, reason: "max_nodes" };
    }

    const parsed = new URL(normalized);
    const domainResult = this.graph.upsertDomain(parsed.hostname);
    const membershipEdge = this.graph.upsertEdge(
      "domain_membership",
      nodeResult.id,
      domainResult.id,
    );

    const createdNodes = [];
    const createdEdges = [];
    if (nodeResult.created) {
      createdNodes.push(nodeResult.created);
    }
    if (domainResult.created) {
      createdNodes.push(domainResult.created);
    }
    if (membershipEdge) {
      createdEdges.push(membershipEdge);
    }

    if (sourceUrl) {
      const sourceId = `url:${sourceUrl}`;
      const targetId = `url:${normalized}`;
      const edge = this.graph.upsertEdge("hyperlink", sourceId, targetId);
      if (edge) {
        createdEdges.push(edge);
      }
    }

    this._emitGraphDelta(reason, createdNodes, createdEdges);

    const priority = this._priorityFor(normalized, depth, sourceUrl);
    const pushed = this.frontier.push({
      url: normalized,
      sourceUrl,
      depth,
      enqueuedAt: nowMs(),
      readyAt: nowMs(),
      priority,
    });
    if (!pushed) {
      return { ok: false, reason: "frontier_duplicate" };
    }

    this.stats.discovered += 1;
    this._emit("node_discovered", {
      url: normalized,
      source: sourceUrl || null,
      depth,
    });
    return { ok: true, url: normalized };
  }

  async _policyFor(urlObj) {
    const origin = urlObj.origin;
    const cached = this.robotsCache.get(origin);
    if (cached) {
      return cached;
    }

    const robotsUrl = `${origin}/robots.txt`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
      const response = await fetch(robotsUrl, {
        method: "GET",
        redirect: "follow",
        headers: {
          "User-Agent": USER_AGENT,
          Accept: "text/plain, */*",
        },
        signal: controller.signal,
      });
      const text = response.ok ? await response.text() : "";
      const policy = parseRobotsTxt(text, USER_AGENT);
      this.robotsCache.set(origin, policy);
      return policy;
    } catch (_err) {
      const policy = {
        allow: [],
        disallow: [],
        crawlDelayMs: null,
      };
      this.robotsCache.set(origin, policy);
      return policy;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async _processItem(item) {
    const startedAt = nowMs();
    const urlObj = new URL(item.url);
    const domain = urlObj.hostname;

    if (this.visitedUrls.has(item.url)) {
      this.stats.skipped += 1;
      this._emit("fetch_skipped", {
        url: item.url,
        depth: item.depth,
        reason: "already_visited",
      });
      return;
    }

    if (this.optOutDomains.has(domain)) {
      this.stats.skipped += 1;
      this.stats.robots_blocked += 1;
      this.graph.setUrlStatus(item.url, {
        status: "blocked",
        compliance: "opt_out",
      });
      this._emit("robots_blocked", {
        url: item.url,
        depth: item.depth,
        reason: "opt_out",
        domain,
      });
      this._markCompliance(false, {
        reason: "opt_out",
        url: item.url,
      });
      return;
    }

    const domainState = this._domainState(domain);
    const now = nowMs();
    if (domainState.nextAllowedAt > now) {
      const readyAt = domainState.nextAllowedAt;
      this.frontier.push({
        ...item,
        readyAt,
      });
      this.stats.skipped += 1;
      this._emit("fetch_skipped", {
        url: item.url,
        depth: item.depth,
        reason: "crawl_delay_wait",
        retry_in_ms: readyAt - now,
      });
      return;
    }

    const policy = await this._policyFor(urlObj);
    const policyDelay = policy.crawlDelayMs;
    const delayMs = policyDelay !== null ? Math.max(this.defaultDelayMs, policyDelay) : this.defaultDelayMs;
    domainState.crawlDelayMs = delayMs;

    const pathForPolicy = `${urlObj.pathname}${urlObj.search || ""}`;
    const allowed = evaluateRobots(pathForPolicy, policy.allow, policy.disallow);
    if (!allowed) {
      this.stats.skipped += 1;
      this.stats.robots_blocked += 1;
      this.graph.setUrlStatus(item.url, {
        status: "blocked",
        compliance: "robots_blocked",
      });
      this._emit("robots_blocked", {
        url: item.url,
        depth: item.depth,
        reason: "robots_disallow",
        domain,
      });
      this._markCompliance(false, {
        reason: "robots_disallow",
        url: item.url,
      });
      return;
    }

    this._markCompliance(true, {
      reason: "robots_allow",
      url: item.url,
    });

    this.visitedUrls.add(item.url);
    domainState.active += 1;
    domainState.nextAllowedAt = nowMs() + delayMs + domainState.backoffMs;

    this._emit("fetch_started", {
      url: item.url,
      depth: item.depth,
      domain,
    });

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

    try {
      const response = await fetch(item.url, {
        method: "GET",
        redirect: "follow",
        headers: {
          "User-Agent": USER_AGENT,
          Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        signal: controller.signal,
      });

      const finalUrl = normalizeUrl(response.url, undefined) || item.url;
      if (finalUrl !== item.url) {
        const redirectSourceId = `url:${item.url}`;
        const redirectTarget = this.graph.upsertUrl(finalUrl, item.depth, item.sourceUrl || null);
        const createdNodes = [];
        if (redirectTarget.created) {
          createdNodes.push(redirectTarget.created);
        }
        const redirectEdge = this.graph.upsertEdge(
          "canonical_redirect",
          redirectSourceId,
          `url:${finalUrl}`,
        );
        this._emitGraphDelta(
          "canonical_redirect",
          createdNodes,
          redirectEdge ? [redirectEdge] : [],
        );
      }

      if (response.status === 429 || response.status === 503) {
        domainState.backoffMs = clamp(
          Math.max(1500, domainState.backoffMs > 0 ? domainState.backoffMs * 2 : 2500),
          1500,
          120000,
        );
      } else {
        domainState.backoffMs = Math.floor(domainState.backoffMs * 0.5);
      }

      if (!response.ok) {
        this.stats.skipped += 1;
        this.graph.setUrlStatus(item.url, {
          status: "skipped",
          compliance: "http_skip",
          fetched_at: nowMs(),
        });
        this._emit("fetch_skipped", {
          url: item.url,
          depth: item.depth,
          reason: "http_status",
          status: response.status,
        });
        return;
      }

      const contentType = parseContentType(response.headers.get("content-type"));
      const rawText = await response.text();
      const bodyText = rawText.slice(0, 750000);
      const contentHash = hashText(bodyText);
      const knownUrl = this.contentHashIndex.get(contentHash);

      const title = contentType.includes("html") ? extractTitle(bodyText) : "";
      this.graph.setUrlStatus(item.url, {
        status: "fetched",
        compliance: "allowed",
        fetched_at: nowMs(),
        content_type: contentType,
        content_hash: contentHash,
        title,
      });

      const contentNode = this.graph.upsertContentType(contentType);
      const contentEdge = this.graph.upsertEdge(
        "content_membership",
        `url:${item.url}`,
        contentNode.id,
      );
      this._emitGraphDelta(
        "content_type",
        contentNode.created ? [contentNode.created] : [],
        contentEdge ? [contentEdge] : [],
      );

      if (knownUrl && knownUrl !== item.url) {
        this.stats.skipped += 1;
        this.stats.duplicates += 1;
        this.graph.setUrlStatus(item.url, {
          status: "duplicate",
          duplicate_of: knownUrl,
        });
        this._emit("fetch_skipped", {
          url: item.url,
          depth: item.depth,
          reason: "duplicate_content",
          duplicate_of: knownUrl,
        });
        return;
      }

      this.contentHashIndex.set(contentHash, item.url);

      let outboundCount = 0;
      if (contentType.includes("html")) {
        const canonicalHref = extractCanonicalHref(bodyText);
        if (canonicalHref) {
          const canonicalUrl = normalizeUrl(canonicalHref, item.url);
          if (canonicalUrl && canonicalUrl !== item.url) {
            const canonicalNode = this.graph.upsertUrl(
              canonicalUrl,
              item.depth,
              item.url,
            );
            const canonicalEdges = [];
            if (canonicalNode.created) {
              canonicalEdges.push(canonicalNode.created);
            }
            const canonicalEdge = this.graph.upsertEdge(
              "canonical_redirect",
              `url:${item.url}`,
              `url:${canonicalUrl}`,
            );
            this._emitGraphDelta(
              "canonical_link",
              canonicalNode.created ? [canonicalNode.created] : [],
              canonicalEdge ? [canonicalEdge] : [],
            );
          }
        }

        const links = extractLinks(bodyText, item.url);
        for (const link of links) {
          if (link.nofollow) {
            this.stats.skipped += 1;
            this._emit("fetch_skipped", {
              url: link.url,
              source: item.url,
              depth: item.depth + 1,
              reason: "nofollow",
            });
            continue;
          }

          if (item.depth + 1 > this.currentMaxDepth) {
            continue;
          }

          const enqueued = this.enqueueUrl(
            link.url,
            item.url,
            item.depth + 1,
            "hyperlink_discovered",
          );
          if (enqueued.ok) {
            outboundCount += 1;
          }
        }
      }

      this.stats.fetched += 1;
      this.stats.total_fetch_time_ms += nowMs() - startedAt;
      domainState.lastFetchedAt = nowMs();

      this._emit("fetch_completed", {
        url: item.url,
        depth: item.depth,
        status: response.status,
        content_type: contentType,
        outbound_count: outboundCount,
        duration_ms: nowMs() - startedAt,
      });
    } catch (err) {
      this.stats.errors += 1;
      this.graph.setUrlStatus(item.url, {
        status: "error",
        compliance: "error",
        error: String(err.message || err),
      });

      const nextBackoff = clamp(
        Math.max(1500, domainState.backoffMs > 0 ? domainState.backoffMs * 2 : 3000),
        1500,
        180000,
      );
      domainState.backoffMs = nextBackoff;

      this._emit("fetch_skipped", {
        url: item.url,
        depth: item.depth,
        reason: "fetch_error",
        error: String(err.message || err),
      });
    } finally {
      clearTimeout(timeoutId);
      domainState.active = Math.max(0, domainState.active - 1);
      if (this.graph.urlNodeCount >= this.currentMaxNodes) {
        this.running = false;
        this.paused = false;
        this._emit("crawl_state", {
          state: "stopped",
          reason: "max_nodes_reached",
        });
      }
      this._persistSnapshot();
    }
  }

  tick() {
    if (!this.running || this.paused) {
      return;
    }

    while (this.activeWorkers < this.currentConcurrency) {
      const item = this.frontier.popReady(nowMs());
      if (!item) {
        break;
      }
      this.activeWorkers += 1;
      this._processItem(item)
        .catch(() => {})
        .finally(() => {
          this.activeWorkers -= 1;
        });
    }
  }

  start({ seeds, maxDepth, maxNodes, concurrency }) {
    const seedList = Array.isArray(seeds) ? seeds : [];
    const normalizedSeeds = [];
    for (const seed of seedList) {
      const normalized = normalizeUrl(seed, undefined);
      if (normalized) {
        normalizedSeeds.push(normalized);
      }
    }
    if (normalizedSeeds.length === 0) {
      return {
        ok: false,
        error: "at least one valid seed URL is required",
      };
    }

    if (!this.running) {
      this.startedAtMs = nowMs();
    }
    this.running = true;
    this.paused = false;
    this.currentMaxDepth = clamp(
      Number.parseInt(String(maxDepth || DEFAULT_MAX_DEPTH), 10) || DEFAULT_MAX_DEPTH,
      0,
      8,
    );
    this.currentMaxNodes = clamp(
      Number.parseInt(String(maxNodes || DEFAULT_MAX_NODES), 10) || DEFAULT_MAX_NODES,
      100,
      50000,
    );
    this.currentConcurrency = clamp(
      Number.parseInt(String(concurrency || DEFAULT_CONCURRENCY), 10) || DEFAULT_CONCURRENCY,
      1,
      24,
    );
    this.graph.maxUrlNodes = this.currentMaxNodes;

    for (const seed of normalizedSeeds) {
      this.enqueueUrl(seed, null, 0, "seed");
    }

    this._emit("crawl_state", {
      state: "running",
      seeds: normalizedSeeds,
      max_depth: this.currentMaxDepth,
      max_nodes: this.currentMaxNodes,
      concurrency: this.currentConcurrency,
      user_agent: USER_AGENT,
    });
    this._persistSnapshot();
    return {
      ok: true,
      state: "running",
      seeds: normalizedSeeds,
    };
  }

  pause() {
    if (!this.running) {
      return { ok: false, error: "crawler is not running" };
    }
    this.paused = true;
    this._emit("crawl_state", {
      state: "paused",
    });
    this._persistSnapshot();
    return {
      ok: true,
      state: "paused",
    };
  }

  resume() {
    if (!this.running) {
      return { ok: false, error: "crawler is not running" };
    }
    this.paused = false;
    this._emit("crawl_state", {
      state: "running",
    });
    this._persistSnapshot();
    return {
      ok: true,
      state: "running",
    };
  }

  stop() {
    this.running = false;
    this.paused = false;
    this.frontier.clear();
    this._emit("crawl_state", {
      state: "stopped",
      reason: "manual_stop",
    });
    this._persistSnapshot();
    return {
      ok: true,
      state: "stopped",
    };
  }

  addOptOutDomain(domain) {
    const normalized = normalizeDomain(domain);
    if (!normalized) {
      return { ok: false, error: "domain is required" };
    }
    this.optOutDomains.add(normalized);
    this._emit("compliance_update", {
      reason: "opt_out_added",
      domain: normalized,
      compliance_percent: this._compliancePercent(),
    });
    this._persistSnapshot();
    return { ok: true, domain: normalized };
  }

  removeOptOutDomain(domain) {
    const normalized = normalizeDomain(domain);
    if (!normalized) {
      return { ok: false, error: "domain is required" };
    }
    const removed = this.optOutDomains.delete(normalized);
    this._emit("compliance_update", {
      reason: removed ? "opt_out_removed" : "opt_out_not_found",
      domain: normalized,
      compliance_percent: this._compliancePercent(),
    });
    this._persistSnapshot();
    return { ok: true, removed, domain: normalized };
  }

  status() {
    const snapshot = this.graph.toSnapshot({ nodeLimit: 12000, edgeLimit: 26000 });
    const urlNodes = snapshot.nodes.filter((node) => node.kind === "url");
    const elapsedSeconds = this.startedAtMs
      ? Math.max(1, Math.floor((nowMs() - this.startedAtMs) / 1000))
      : 1;
    const crawlRate = Number((this.stats.fetched / elapsedSeconds).toFixed(3));
    const domainDistribution = getDomainDistribution(urlNodes);
    const depthHistogram = getDepthHistogram(urlNodes);
    const activeDomains = [...this.domainState.entries()]
      .filter(([, state]) => state.active > 0)
      .map(([domain]) => domain)
      .slice(0, 20);

    return {
      ok: true,
      state: this.running ? (this.paused ? "paused" : "running") : "stopped",
      started_at: this.startedAtMs,
      user_agent: USER_AGENT,
      config: {
        max_depth: this.currentMaxDepth,
        max_nodes: this.currentMaxNodes,
        concurrency: this.currentConcurrency,
        default_delay_ms: this.defaultDelayMs,
        fetch_timeout_ms: FETCH_TIMEOUT_MS,
      },
      metrics: {
        crawl_rate_nodes_per_sec: crawlRate,
        frontier_size: this.frontier.size,
        active_fetchers: this.activeWorkers,
        compliance_percent: this._compliancePercent(),
        discovered: this.stats.discovered,
        fetched: this.stats.fetched,
        skipped: this.stats.skipped,
        robots_blocked: this.stats.robots_blocked,
        duplicate_content: this.stats.duplicates,
        errors: this.stats.errors,
        average_fetch_ms:
          this.stats.fetched > 0
            ? Number((this.stats.total_fetch_time_ms / this.stats.fetched).toFixed(1))
            : 0,
      },
      active_domains: activeDomains,
      domain_distribution: domainDistribution,
      depth_histogram: depthHistogram,
      opt_out_domains: [...this.optOutDomains].sort(),
      graph_counts: snapshot.counts,
      event_count: this.recentEvents.length,
      opt_out_endpoint: `/api/weaver/opt-out`,
    };
  }

  events(limit = 200) {
    const safeLimit = clamp(Number.parseInt(String(limit), 10) || 200, 1, 2000);
    return this.recentEvents.slice(-safeLimit);
  }

  graphSnapshot({ domainFilter = "", nodeLimit = 5000, edgeLimit = 12000 } = {}) {
    return this.graph.toSnapshot({
      domainFilter,
      nodeLimit: clamp(Number.parseInt(String(nodeLimit), 10) || 5000, 100, 20000),
      edgeLimit: clamp(Number.parseInt(String(edgeLimit), 10) || 12000, 200, 80000),
    });
  }

  _persistSnapshot() {
    const payload = {
      generated_at: new Date().toISOString(),
      status: this.status(),
      graph: this.graphSnapshot({ nodeLimit: 15000, edgeLimit: 40000 }),
    };
    fs.writeFile(SNAPSHOT_PATH, JSON.stringify(payload, null, 2), () => {});
  }
}

ensureWorldStateDir();
const weaver = new WebGraphWeaver();

const wsServer = new WebSocketServer({ noServer: true });
const wsClients = new Set();

function broadcastEvent(eventPayload) {
  const message = JSON.stringify(eventPayload);
  for (const ws of wsClients) {
    if (ws.readyState === ws.OPEN) {
      ws.send(message);
    }
  }
}

weaver.setBroadcast(broadcastEvent);

wsServer.on("connection", (socket) => {
  wsClients.add(socket);
  socket.send(
    JSON.stringify({
      event: "snapshot",
      timestamp: nowMs(),
      status: weaver.status(),
      graph: weaver.graphSnapshot({ nodeLimit: 5000, edgeLimit: 12000 }),
      recent_events: weaver.events(120),
    }),
  );

  socket.on("close", () => {
    wsClients.delete(socket);
  });
});

const server = http.createServer(async (req, res) => {
  if (!req.url) {
    sendJson(res, 400, { ok: false, error: "missing request URL" });
    return;
  }

  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    res.end();
    return;
  }

  const parsed = new URL(req.url, `http://${req.headers.host || `${HOST}:${PORT}`}`);
  const pathname = parsed.pathname;

  if (req.method === "GET" && pathname === "/") {
    sendJson(res, 200, {
      ok: true,
      service: "web-graph-weaver",
      version: "0.1.0",
      status_endpoint: "/api/weaver/status",
      websocket_endpoint: "/ws",
      note: "Ethical crawl instrumentation service",
    });
    return;
  }

  if (req.method === "GET" && pathname === "/healthz") {
    sendJson(res, 200, {
      ok: true,
      status: "healthy",
      timestamp: nowMs(),
    });
    return;
  }

  if (req.method === "GET" && pathname === "/api/weaver/status") {
    sendJson(res, 200, weaver.status());
    return;
  }

  if (req.method === "GET" && pathname === "/api/weaver/events") {
    const limit = parsed.searchParams.get("limit") || "200";
    sendJson(res, 200, {
      ok: true,
      events: weaver.events(limit),
    });
    return;
  }

  if (req.method === "GET" && pathname === "/api/weaver/graph") {
    const domain = parsed.searchParams.get("domain") || "";
    const nodeLimit = parsed.searchParams.get("node_limit") || "5000";
    const edgeLimit = parsed.searchParams.get("edge_limit") || "12000";
    sendJson(res, 200, {
      ok: true,
      graph: weaver.graphSnapshot({
        domainFilter: domain,
        nodeLimit,
        edgeLimit,
      }),
    });
    return;
  }

  if (req.method === "GET" && pathname === "/api/weaver/opt-out") {
    sendJson(res, 200, {
      ok: true,
      domains: [...weaver.optOutDomains].sort(),
      how_to_opt_out:
        "POST /api/weaver/opt-out with {\"domain\":\"example.com\"}",
    });
    return;
  }

  if (req.method === "POST" && pathname === "/api/weaver/seed") {
    try {
      const body = await parseJsonBody(req);
      const seeds = Array.isArray(body.seeds)
        ? body.seeds
        : body.seed
          ? [body.seed]
          : [];
      const accepted = [];
      const rejected = [];
      for (const seed of seeds) {
        const outcome = weaver.enqueueUrl(seed, null, 0, "manual_seed");
        if (outcome.ok) {
          accepted.push(outcome.url);
        } else {
          rejected.push({ seed, reason: outcome.reason });
        }
      }
      sendJson(res, 200, {
        ok: true,
        accepted,
        rejected,
      });
    } catch (err) {
      sendJson(res, 400, {
        ok: false,
        error: String(err.message || err),
      });
    }
    return;
  }

  if (req.method === "POST" && pathname === "/api/weaver/control") {
    try {
      const body = await parseJsonBody(req);
      const action = String(body.action || "").trim().toLowerCase();
      let result;
      if (action === "start") {
        result = weaver.start({
          seeds: body.seeds,
          maxDepth: body.max_depth,
          maxNodes: body.max_nodes,
          concurrency: body.concurrency,
        });
      } else if (action === "pause") {
        result = weaver.pause();
      } else if (action === "resume") {
        result = weaver.resume();
      } else if (action === "stop") {
        result = weaver.stop();
      } else {
        result = { ok: false, error: "unknown action" };
      }

      sendJson(res, result.ok ? 200 : 400, {
        ...result,
        status: weaver.status(),
      });
    } catch (err) {
      sendJson(res, 400, {
        ok: false,
        error: String(err.message || err),
      });
    }
    return;
  }

  if (req.method === "POST" && pathname === "/api/weaver/opt-out") {
    try {
      const body = await parseJsonBody(req);
      const result = weaver.addOptOutDomain(body.domain);
      sendJson(res, result.ok ? 200 : 400, result);
    } catch (err) {
      sendJson(res, 400, {
        ok: false,
        error: String(err.message || err),
      });
    }
    return;
  }

  if (req.method === "DELETE" && pathname === "/api/weaver/opt-out") {
    try {
      const body = await parseJsonBody(req);
      const result = weaver.removeOptOutDomain(body.domain);
      sendJson(res, result.ok ? 200 : 400, result);
    } catch (err) {
      sendJson(res, 400, {
        ok: false,
        error: String(err.message || err),
      });
    }
    return;
  }

  sendJson(res, 404, {
    ok: false,
    error: "not found",
  });
});

server.on("upgrade", (req, socket, head) => {
  if (!req.url) {
    socket.destroy();
    return;
  }
  const parsed = new URL(req.url, `http://${req.headers.host || `${HOST}:${PORT}`}`);
  if (parsed.pathname !== "/ws") {
    socket.destroy();
    return;
  }
  wsServer.handleUpgrade(req, socket, head, (ws) => {
    wsServer.emit("connection", ws, req);
  });
});

server.listen(PORT, HOST, () => {
  console.log(`[weaver] Web Graph Weaver listening on http://${HOST}:${PORT}`);
  console.log(`[weaver] User-Agent: ${USER_AGENT}`);
});
