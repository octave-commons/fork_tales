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
const MAX_SEMANTIC_REFERENCES_PER_PAGE = Number.parseInt(
  process.env.WEAVER_MAX_SEMANTIC_REFERENCES_PER_PAGE || "120",
  10,
);
const MAX_ARXIV_REFERENCES_PER_PAGE = Number.parseInt(
  process.env.WEAVER_MAX_ARXIV_REFERENCES_PER_PAGE || "70",
  10,
);
const MAX_WIKIPEDIA_REFERENCES_PER_PAGE = Number.parseInt(
  process.env.WEAVER_MAX_WIKIPEDIA_REFERENCES_PER_PAGE || "90",
  10,
);
const DEFAULT_MAX_REQUESTS_PER_HOST = Number.parseInt(
  process.env.WEAVER_MAX_REQUESTS_PER_HOST || "2",
  10,
);
const NODE_COOLDOWN_MS = Number.parseInt(
  process.env.WEAVER_NODE_COOLDOWN_MS || String(10 * 60 * 1000),
  10,
);
const CONTENT_HASH_INDEX_MAX_RAW = Number.parseInt(
  process.env.WEAVER_CONTENT_HASH_INDEX_MAX || "60000",
  10,
);
const CONTENT_HASH_INDEX_MAX =
  Number.isFinite(CONTENT_HASH_INDEX_MAX_RAW) && CONTENT_HASH_INDEX_MAX_RAW >= 2048
    ? CONTENT_HASH_INDEX_MAX_RAW
    : 60000;
const DOMAIN_STATE_MAX_RAW = Number.parseInt(
  process.env.WEAVER_DOMAIN_STATE_MAX || "4000",
  10,
);
const DOMAIN_STATE_MAX =
  Number.isFinite(DOMAIN_STATE_MAX_RAW) && DOMAIN_STATE_MAX_RAW >= 128
    ? DOMAIN_STATE_MAX_RAW
    : 4000;
const ACTIVATION_THRESHOLD = Number.parseFloat(
  process.env.WEAVER_ACTIVATION_THRESHOLD || "1.0",
);
const INTERACTION_ACTIVATION_DELTA = Number.parseFloat(
  process.env.WEAVER_INTERACTION_ACTIVATION_DELTA || "0.35",
);
const ENTITY_VISIT_ACTIVATION_DELTA = Number.parseFloat(
  process.env.WEAVER_ENTITY_VISIT_ACTIVATION_DELTA || "0.28",
);
const DEFAULT_ENTITY_COUNT = Number.parseInt(
  process.env.WEAVER_ENTITY_COUNT || "4",
  10,
);
const ENTITY_TICK_MS = Number.parseInt(
  process.env.WEAVER_ENTITY_TICK_MS || "750",
  10,
);
const ENTITY_MOVE_MIN_MS = Number.parseInt(
  process.env.WEAVER_ENTITY_MOVE_MIN_MS || "900",
  10,
);
const ENTITY_MOVE_MAX_MS = Number.parseInt(
  process.env.WEAVER_ENTITY_MOVE_MAX_MS || "2600",
  10,
);
const LLM_ENABLED = !["0", "false", "off"].includes(
  String(process.env.WEAVER_LLM_ENABLED || "0").trim().toLowerCase(),
);
const LLM_BASE_URL = String(
  process.env.WEAVER_LLM_BASE_URL || process.env.TEXT_GENERATION_BASE_URL || "http://127.0.0.1:18000/v1",
).trim();
const LLM_MODEL = String(process.env.WEAVER_LLM_MODEL || "qwen3-vl:2b-instruct").trim();
const LLM_TIMEOUT_MS = Number.parseInt(
  process.env.WEAVER_LLM_TIMEOUT_MS || "9000",
  10,
);
const LLM_TEXT_MAX_CHARS = Number.parseInt(
  process.env.WEAVER_LLM_TEXT_MAX_CHARS || "7000",
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

function safeDecodeURIComponent(value) {
  try {
    return decodeURIComponent(value);
  } catch (_err) {
    return value;
  }
}

function normalizeArxivId(rawId) {
  let value = String(rawId || "").trim();
  if (!value) {
    return "";
  }
  value = value.replace(/\.pdf$/i, "").replace(/\/+$/, "");
  value = value.replace(/v\d+$/i, "");
  value = safeDecodeURIComponent(value);
  if (!/^[a-z0-9._\-\/]+$/i.test(value)) {
    return "";
  }
  return value;
}

function isArxivHost(hostname) {
  const host = String(hostname || "").toLowerCase();
  return host === "arxiv.org" || host.endsWith(".arxiv.org");
}

function extractArxivIdFromUrl(rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    if (!isArxivHost(parsed.hostname)) {
      return "";
    }
    const path = parsed.pathname || "";
    const match = /^\/(abs|pdf)\/(.+)$/i.exec(path);
    if (!match) {
      return "";
    }
    const candidate = normalizeArxivId(match[2]);
    return candidate;
  } catch (_err) {
    return "";
  }
}

function canonicalArxivAbsUrlFromId(arxivId) {
  const normalized = normalizeArxivId(arxivId);
  if (!normalized) {
    return "";
  }
  return `https://arxiv.org/abs/${normalized}`;
}

function canonicalArxivPdfUrlFromId(arxivId) {
  const normalized = normalizeArxivId(arxivId);
  if (!normalized) {
    return "";
  }
  return `https://arxiv.org/pdf/${normalized}.pdf`;
}

function isArxivPdfUrl(rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    return isArxivHost(parsed.hostname) && /^\/pdf\//i.test(parsed.pathname || "");
  } catch (_err) {
    return false;
  }
}

function canonicalWikipediaArticleUrl(rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    const host = String(parsed.hostname || "").toLowerCase();
    if (host !== "wikipedia.org" && !host.endsWith(".wikipedia.org")) {
      return "";
    }
    if (!parsed.pathname.startsWith("/wiki/")) {
      return "";
    }
    const rawSlug = parsed.pathname.slice("/wiki/".length);
    const slug = safeDecodeURIComponent(rawSlug).trim().replace(/\s+/g, "_");
    if (!slug || slug.includes(":")) {
      return "";
    }
    const canonicalHost = host.replace(/\.m\.wikipedia\.org$/, ".wikipedia.org");
    const encodedSlug = encodeURIComponent(slug).replace(/%2F/g, "/");
    const canonical = `https://${canonicalHost}/wiki/${encodedSlug}`;
    return normalizeUrl(canonical, undefined) || "";
  } catch (_err) {
    return "";
  }
}

function extractWikipediaSlugFromUrl(rawUrl) {
  const canonical = canonicalWikipediaArticleUrl(rawUrl);
  if (!canonical) {
    return "";
  }
  try {
    const parsed = new URL(canonical);
    return safeDecodeURIComponent(parsed.pathname.slice("/wiki/".length));
  } catch (_err) {
    return "";
  }
}

function classifyKnowledgeUrl(rawUrl) {
  const arxivId = extractArxivIdFromUrl(rawUrl);
  if (arxivId) {
    return isArxivPdfUrl(rawUrl) ? "arxiv_pdf" : "arxiv_abs";
  }
  const wikiUrl = canonicalWikipediaArticleUrl(rawUrl);
  if (wikiUrl) {
    return "wikipedia_article";
  }
  return "other";
}

function inferKnowledgeMetadata(rawUrl) {
  const knowledgeKind = classifyKnowledgeUrl(rawUrl);
  if (knowledgeKind === "arxiv_abs" || knowledgeKind === "arxiv_pdf") {
    return {
      source_family: "arxiv",
      knowledge_kind: knowledgeKind,
      arxiv_id: extractArxivIdFromUrl(rawUrl) || null,
      wikipedia_slug: null,
    };
  }
  if (knowledgeKind === "wikipedia_article") {
    return {
      source_family: "wikipedia",
      knowledge_kind: knowledgeKind,
      arxiv_id: null,
      wikipedia_slug: extractWikipediaSlugFromUrl(rawUrl) || null,
    };
  }
  return {
    source_family: "web",
    knowledge_kind: "web_url",
    arxiv_id: null,
    wikipedia_slug: null,
  };
}

function parseContentType(contentTypeHeader) {
  if (!contentTypeHeader) {
    return "application/octet-stream";
  }
  return String(contentTypeHeader).split(";")[0].trim().toLowerCase() || "application/octet-stream";
}

function isTextLikeContentType(contentType) {
  const value = String(contentType || "").toLowerCase();
  return (
    value.startsWith("text/") ||
    value.includes("html") ||
    value.includes("xml") ||
    value.includes("json") ||
    value.includes("javascript") ||
    value.includes("xhtml") ||
    value.includes("svg")
  );
}

function decodeHtmlEntities(text) {
  return String(text || "")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'");
}

function extractReadableTextFromHtml(html) {
  const withoutScripts = String(html || "")
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ");
  const withoutTags = withoutScripts.replace(/<[^>]+>/g, " ");
  const decoded = decodeHtmlEntities(withoutTags);
  return decoded.replace(/\s+/g, " ").trim();
}

function fallbackTextSummary(text) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) {
    return "No readable text extracted.";
  }
  if (clean.length <= 260) {
    return clean;
  }
  return `${clean.slice(0, 257)}...`;
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

function extractAnchorLinks(html, baseUrl) {
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
  return links;
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

  for (const link of extractAnchorLinks(html, baseUrl)) {
    pushLink(link.url, link.nofollow);
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

function extractArxivMentionIds(text) {
  const ids = [];
  const seen = new Set();
  const mentionPattern = /\barxiv\s*:\s*([a-z\-]+\/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?\b/gi;
  while (true) {
    const match = mentionPattern.exec(text);
    if (match === null) {
      break;
    }
    const normalized = normalizeArxivId(match[1]);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    ids.push(normalized);
  }
  return ids;
}

function dedupeSemanticReferences(rows, maxItems = MAX_SEMANTIC_REFERENCES_PER_PAGE) {
  const output = [];
  const seen = new Set();
  for (const row of rows) {
    const normalized = normalizeUrl(row.url, undefined);
    if (!normalized) {
      continue;
    }
    const edgeKind = String(row.edge_kind || "").trim().toLowerCase();
    if (!edgeKind) {
      continue;
    }
    const key = `${edgeKind}|${normalized}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    output.push({
      url: normalized,
      edge_kind: edgeKind,
      reason: String(row.reason || "semantic_reference"),
      nofollow: Boolean(row.nofollow),
      enqueue: Boolean(row.enqueue),
    });
    if (output.length >= maxItems) {
      break;
    }
  }
  return output;
}

function extractSemanticReferences(sourceUrl, html) {
  const sourceKind = classifyKnowledgeUrl(sourceUrl);
  if (sourceKind === "other") {
    return {
      source_kind: "other",
      references: [],
    };
  }

  const references = [];
  const anchorLinks = extractAnchorLinks(html, sourceUrl);

  if (sourceKind === "arxiv_abs" || sourceKind === "arxiv_pdf") {
    const sourceArxivId = extractArxivIdFromUrl(sourceUrl);
    if (sourceArxivId) {
      references.push({
        url: canonicalArxivPdfUrlFromId(sourceArxivId),
        edge_kind: "paper_pdf",
        reason: "arxiv_pdf_asset",
        nofollow: false,
        enqueue: true,
      });
    }

    let arxivReferenceBudget = MAX_ARXIV_REFERENCES_PER_PAGE;
    for (const link of anchorLinks) {
      const targetArxivId = extractArxivIdFromUrl(link.url);
      if (targetArxivId) {
        if (sourceArxivId && targetArxivId === sourceArxivId) {
          if (isArxivPdfUrl(link.url)) {
            references.push({
              url: canonicalArxivPdfUrlFromId(targetArxivId),
              edge_kind: "paper_pdf",
              reason: "arxiv_pdf_asset",
              nofollow: link.nofollow,
              enqueue: !link.nofollow,
            });
          }
        } else if (arxivReferenceBudget > 0) {
          references.push({
            url: canonicalArxivAbsUrlFromId(targetArxivId),
            edge_kind: "citation",
            reason: "arxiv_link_citation",
            nofollow: link.nofollow,
            enqueue: !link.nofollow,
          });
          arxivReferenceBudget -= 1;
        }
      }

      const wikiTarget = canonicalWikipediaArticleUrl(link.url);
      if (wikiTarget) {
        references.push({
          url: wikiTarget,
          edge_kind: "cross_reference",
          reason: "arxiv_to_wikipedia",
          nofollow: link.nofollow,
          enqueue: !link.nofollow,
        });
      }
    }

    if (arxivReferenceBudget > 0) {
      for (const mentionId of extractArxivMentionIds(html)) {
        if (sourceArxivId && mentionId === sourceArxivId) {
          continue;
        }
        references.push({
          url: canonicalArxivAbsUrlFromId(mentionId),
          edge_kind: "citation",
          reason: "arxiv_text_citation",
          nofollow: false,
          enqueue: false,
        });
        arxivReferenceBudget -= 1;
        if (arxivReferenceBudget <= 0) {
          break;
        }
      }
    }
  }

  if (sourceKind === "wikipedia_article") {
    const sourceWiki = canonicalWikipediaArticleUrl(sourceUrl);
    let wikipediaBudget = MAX_WIKIPEDIA_REFERENCES_PER_PAGE;

    for (const link of anchorLinks) {
      const wikiTarget = canonicalWikipediaArticleUrl(link.url);
      if (wikiTarget && wikiTarget !== sourceWiki && wikipediaBudget > 0) {
        references.push({
          url: wikiTarget,
          edge_kind: "wiki_reference",
          reason: "wikipedia_internal_link",
          nofollow: link.nofollow,
          enqueue: !link.nofollow,
        });
        wikipediaBudget -= 1;
      }

      const targetArxivId = extractArxivIdFromUrl(link.url);
      if (targetArxivId) {
        references.push({
          url: canonicalArxivAbsUrlFromId(targetArxivId),
          edge_kind: "cross_reference",
          reason: "wikipedia_to_arxiv",
          nofollow: link.nofollow,
          enqueue: !link.nofollow,
        });
      }
    }
  }

  return {
    source_kind: sourceKind,
    references: dedupeSemanticReferences(references),
  };
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
    const knowledge = inferKnowledgeMetadata(url);
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
      source_family: knowledge.source_family,
      knowledge_kind: knowledge.knowledge_kind,
      arxiv_id: knowledge.arxiv_id,
      wikipedia_slug: knowledge.wikipedia_slug,
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

  getNodeById(id) {
    return this.nodes.get(id) || null;
  }

  getUrlNode(url) {
    return this.nodes.get(this._makeNodeId("url", url)) || null;
  }

  getOutgoingUrlEdges(url) {
    const source = this._makeNodeId("url", url);
    const rows = [];
    for (const edge of this.edges.values()) {
      if (edge.source !== source) {
        continue;
      }
      if (!String(edge.target || "").startsWith("url:")) {
        continue;
      }
      rows.push(edge);
    }
    return rows;
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
    this.contentHashIndex = new Map();
    this.maxContentHashIndex = CONTENT_HASH_INDEX_MAX;
    this.optOutDomains = new Set();
    this.recentEvents = [];
    this.inFlightUrls = new Set();
    this.analysisInFlight = new Set();

    this.running = false;
    this.paused = false;
    this.startedAtMs = null;
    this.activeWorkers = 0;
    this.currentConcurrency = DEFAULT_CONCURRENCY;
    this.currentMaxRequestsPerHost = DEFAULT_MAX_REQUESTS_PER_HOST;
    this.currentMaxDepth = DEFAULT_MAX_DEPTH;
    this.currentMaxNodes = DEFAULT_MAX_NODES;
    this.defaultDelayMs = DEFAULT_DELAY_MS;
    this.nodeCooldownMs = NODE_COOLDOWN_MS;
    this.activationThreshold = ACTIVATION_THRESHOLD;

    this.entitiesEnabled = true;
    this.entitiesPaused = false;
    this.entityCount = DEFAULT_ENTITY_COUNT;
    this.entities = [];
    this.lastEntityBroadcastAt = 0;

    this.stats = {
      discovered: 0,
      fetched: 0,
      skipped: 0,
      robots_blocked: 0,
      errors: 0,
      duplicates: 0,
      semantic_edges: 0,
      citation_edges: 0,
      wiki_reference_edges: 0,
      cross_reference_edges: 0,
      paper_pdf_edges: 0,
      host_concurrency_waits: 0,
      cooldown_blocked: 0,
      interactions: 0,
      activation_enqueues: 0,
      entity_moves: 0,
      entity_visits: 0,
      llm_analysis_success: 0,
      llm_analysis_fail: 0,
      compliance_checks: 0,
      compliance_pass: 0,
      compliance_fail: 0,
      total_fetch_time_ms: 0,
    };

    this.domainState = new Map();
    this.maxDomainStateEntries = DOMAIN_STATE_MAX;

    this.broadcast = () => {};
    this.scheduler = setInterval(() => {
      this.tick();
    }, 120);
    this.entityScheduler = setInterval(() => {
      this.entityTick();
    }, ENTITY_TICK_MS);

    this._bootstrapEntities();
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
      if (this.domainState.size >= this.maxDomainStateEntries) {
        this._pruneDomainState();
      }
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

  _pruneDomainState() {
    if (this.domainState.size <= this.maxDomainStateEntries) {
      return;
    }
    const trimTarget = Math.max(64, Math.floor(this.maxDomainStateEntries * 0.9));
    for (const [domain, state] of this.domainState.entries()) {
      if (this.domainState.size <= trimTarget) {
        break;
      }
      if (Number(state?.active || 0) > 0) {
        continue;
      }
      this.domainState.delete(domain);
    }
    if (this.domainState.size <= trimTarget) {
      return;
    }
    for (const domain of this.domainState.keys()) {
      if (this.domainState.size <= trimTarget) {
        break;
      }
      this.domainState.delete(domain);
    }
  }

  _pruneContentHashIndex() {
    if (this.contentHashIndex.size <= this.maxContentHashIndex) {
      return;
    }
    const trimTarget = Math.max(1024, Math.floor(this.maxContentHashIndex * 0.9));
    const removeCount = this.contentHashIndex.size - trimTarget;
    if (removeCount <= 0) {
      return;
    }
    let removed = 0;
    for (const hash of this.contentHashIndex.keys()) {
      this.contentHashIndex.delete(hash);
      removed += 1;
      if (removed >= removeCount) {
        break;
      }
    }
  }

  _recordSemanticEdgeStats(edgeKind) {
    if (edgeKind === "citation") {
      this.stats.citation_edges += 1;
      return;
    }
    if (edgeKind === "wiki_reference") {
      this.stats.wiki_reference_edges += 1;
      return;
    }
    if (edgeKind === "cross_reference") {
      this.stats.cross_reference_edges += 1;
      return;
    }
    if (edgeKind === "paper_pdf") {
      this.stats.paper_pdf_edges += 1;
    }
  }

  _cooldownRemainingMs(url, atMs = nowMs()) {
    const node = this.graph.getUrlNode(url);
    if (!node) {
      return 0;
    }
    const until = Number(node.cooldown_until || 0);
    if (!Number.isFinite(until) || until <= 0) {
      return 0;
    }
    return Math.max(0, until - atMs);
  }

  _bootstrapEntities() {
    const nextCount = clamp(
      Number.parseInt(String(this.entityCount || DEFAULT_ENTITY_COUNT), 10) || DEFAULT_ENTITY_COUNT,
      0,
      24,
    );
    const existing = new Map(this.entities.map((entity) => [entity.id, entity]));
    const rows = [];
    for (let i = 0; i < nextCount; i += 1) {
      const id = `entity:${i + 1}`;
      const prev = existing.get(id);
      rows.push({
        id,
        label: `crawler-${i + 1}`,
        state: prev?.state || "idle",
        current_url: prev?.current_url || null,
        from_url: prev?.from_url || null,
        target_url: prev?.target_url || null,
        progress: Number(prev?.progress || 0),
        move_started_at: Number(prev?.move_started_at || 0),
        move_eta_ms: Number(prev?.move_eta_ms || 0),
        visits: Number(prev?.visits || 0),
        last_visit_at: Number(prev?.last_visit_at || 0),
        next_available_at: Number(prev?.next_available_at || 0),
      });
    }
    this.entities = rows;
  }

  _entitySnapshot() {
    return this.entities.map((entity) => ({
      id: entity.id,
      label: entity.label,
      state: entity.state,
      current_url: entity.current_url,
      from_url: entity.from_url,
      target_url: entity.target_url,
      progress: Number(entity.progress || 0),
      visits: Number(entity.visits || 0),
      last_visit_at: Number(entity.last_visit_at || 0),
      next_available_at: Number(entity.next_available_at || 0),
    }));
  }

  _emitEntityTick(force = false) {
    const now = nowMs();
    if (!force && now - this.lastEntityBroadcastAt < 600) {
      return;
    }
    this.lastEntityBroadcastAt = now;
    this._emit("entity_tick", {
      entities_enabled: this.entitiesEnabled ? 1 : 0,
      entities_paused: this.entitiesPaused ? 1 : 0,
      entities: this._entitySnapshot(),
      activation_threshold: this.activationThreshold,
      node_cooldown_ms: this.nodeCooldownMs,
      max_requests_per_host: this.currentMaxRequestsPerHost,
    });
  }

  _activateNode(url, delta, source, allowEnqueue = true) {
    const normalized = normalizeUrl(url, undefined);
    if (!normalized) {
      return {
        ok: false,
        error: "invalid_url",
      };
    }

    const known = this.graph.upsertUrl(normalized, 0, null);
    if (known.rejected) {
      return {
        ok: false,
        error: "max_nodes",
      };
    }

    const now = nowMs();
    const node = this.graph.getUrlNode(normalized) || {};
    const previous = Number(node.activation_potential || 0);
    const interactionCount = Number(node.interaction_count || 0) + 1;
    const safeDelta = Number.isFinite(delta) ? delta : INTERACTION_ACTIVATION_DELTA;
    const nextPotential = clamp(previous + safeDelta, 0, 8);
    const cooldownRemainingMs = this._cooldownRemainingMs(normalized, now);

    this.graph.setUrlStatus(normalized, {
      activation_potential: Number(nextPotential.toFixed(4)),
      interaction_count: interactionCount,
      last_interacted_at: now,
      last_interaction_source: String(source || "unknown"),
    });

    let enqueued = false;
    let enqueueReason = "threshold_not_reached";
    if (allowEnqueue && nextPotential >= this.activationThreshold && cooldownRemainingMs <= 0) {
      const depth = Number(node.depth || 0);
      const outcome = this.enqueueUrl(normalized, node.source_url || null, depth, "activation_threshold");
      if (outcome.ok) {
        enqueued = true;
        enqueueReason = "activation_enqueued";
        this.stats.activation_enqueues += 1;
        const remainingPotential = Math.max(0, nextPotential - this.activationThreshold);
        this.graph.setUrlStatus(normalized, {
          activation_potential: Number(remainingPotential.toFixed(4)),
        });
      } else {
        enqueueReason = String(outcome.reason || "enqueue_rejected");
      }
    } else if (cooldownRemainingMs > 0) {
      enqueueReason = "cooldown_active";
    }

    this.stats.interactions += 1;
    const patchedNode = this.graph.getUrlNode(normalized) || {};
    this._emit("node_interacted", {
      url: normalized,
      source: String(source || "unknown"),
      delta: Number(safeDelta.toFixed(4)),
      activation_potential: Number(patchedNode.activation_potential || 0),
      interaction_count: Number(patchedNode.interaction_count || interactionCount),
      cooldown_remaining_ms: cooldownRemainingMs,
      enqueued: enqueued ? 1 : 0,
      enqueue_reason: enqueueReason,
    });

    return {
      ok: true,
      url: normalized,
      enqueued,
      enqueue_reason: enqueueReason,
      cooldown_remaining_ms: cooldownRemainingMs,
      activation_potential: Number(patchedNode.activation_potential || 0),
      interaction_count: Number(patchedNode.interaction_count || interactionCount),
    };
  }

  _setEntityTarget(entity, targetUrl, reason) {
    const now = nowMs();
    const eta = clamp(
      Math.floor(ENTITY_MOVE_MIN_MS + Math.random() * Math.max(1, ENTITY_MOVE_MAX_MS - ENTITY_MOVE_MIN_MS)),
      120,
      15000,
    );
    entity.from_url = entity.current_url || null;
    entity.target_url = targetUrl;
    entity.state = "moving";
    entity.progress = 0;
    entity.move_started_at = now;
    entity.move_eta_ms = eta;
    this.stats.entity_moves += 1;
    this._emit("entity_move", {
      entity_id: entity.id,
      from_url: entity.from_url,
      target_url: entity.target_url,
      reason: String(reason || "route"),
      eta_ms: eta,
    });
  }

  _candidateTargetsForEntity(entity) {
    const rows = [];
    const seen = new Set();
    const sourceUrl = entity.current_url;
    if (sourceUrl) {
      for (const edge of this.graph.getOutgoingUrlEdges(sourceUrl)) {
        const targetUrl = String(edge.target || "").startsWith("url:")
          ? String(edge.target).slice(4)
          : "";
        if (!targetUrl || seen.has(targetUrl)) {
          continue;
        }
        seen.add(targetUrl);
        const node = this.graph.getUrlNode(targetUrl);
        const activation = Number(node?.activation_potential || 0);
        const cooldownRemaining = this._cooldownRemainingMs(targetUrl);
        const inFlight = this.inFlightUrls.has(targetUrl) || this.frontier.has(targetUrl);
        if (cooldownRemaining > 0 || inFlight) {
          continue;
        }
        const score = 0.65 + activation + Math.random() * 0.35;
        rows.push({
          url: targetUrl,
          score,
          reason: String(edge.kind || "linked"),
        });
      }
    }

    if (rows.length === 0) {
      const urlNodes = this.graph.getUrlNodes();
      for (const node of urlNodes) {
        const targetUrl = String(node.url || "");
        if (!targetUrl || seen.has(targetUrl)) {
          continue;
        }
        const cooldownRemaining = this._cooldownRemainingMs(targetUrl);
        const inFlight = this.inFlightUrls.has(targetUrl) || this.frontier.has(targetUrl);
        if (cooldownRemaining > 0 || inFlight) {
          continue;
        }
        const activation = Number(node.activation_potential || 0);
        const fetchedBias = String(node.status || "") === "fetched" ? 0.1 : 0.22;
        rows.push({
          url: targetUrl,
          score: activation + fetchedBias + Math.random() * 0.16,
          reason: "known_url",
        });
      }
    }

    rows.sort((a, b) => b.score - a.score);
    return rows;
  }

  async _analyzeNodeText(url, source) {
    const node = this.graph.getUrlNode(url);
    if (!node) {
      return;
    }
    const textExcerpt = String(node.text_excerpt || "").trim();
    if (!textExcerpt) {
      return;
    }

    const now = nowMs();
    const lastAnalyzedAt = Number(node.last_analyzed_at || 0);
    if (lastAnalyzedAt > 0 && now - lastAnalyzedAt < this.nodeCooldownMs) {
      return;
    }

    if (!this.analysisInFlight) {
      this.analysisInFlight = new Set();
    }
    if (this.analysisInFlight.has(url)) {
      return;
    }
    this.analysisInFlight.add(url);

    const briefText = textExcerpt.slice(0, Math.max(1200, LLM_TEXT_MAX_CHARS));
    const fallbackSummary = fallbackTextSummary(briefText);

    this._emit("link_text_analysis_started", {
      url,
      source: String(source || "unknown"),
      model: LLM_MODEL,
      llm_enabled: LLM_ENABLED ? 1 : 0,
    });

    try {
      let summary = fallbackSummary;
      let provider = "heuristic";

      if (LLM_ENABLED) {
        const prompt = [
          "Summarize the page text in 2 concise bullets.",
          "Then provide one line: FocusIntent: <what should a graph crawler learn from this page>.",
          "Avoid markdown tables and keep under 600 characters.",
          "--- PAGE TEXT START ---",
          briefText,
          "--- PAGE TEXT END ---",
        ].join("\n");

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS);
        try {
          const baseUrl = LLM_BASE_URL.replace(/\/+$/, "");
          const endpoint = baseUrl.endsWith("/v1/chat/completions")
            ? baseUrl
            : baseUrl.endsWith("/v1")
              ? `${baseUrl}/chat/completions`
              : `${baseUrl}/v1/chat/completions`;
          const response = await fetch(endpoint, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              model: LLM_MODEL,
              messages: [{ role: "user", content: prompt }],
              stream: false,
              temperature: 0.2,
              max_tokens: 300,
            }),
            signal: controller.signal,
          });
          if (!response.ok) {
            throw new Error(`llm_http_${response.status}`);
          }
          const payload = await response.json();
          const candidate = String(
            payload?.choices?.[0]?.message?.content || payload?.choices?.[0]?.text || "",
          )
            .replace(/\s+/g, " ")
            .trim();
          if (candidate) {
            summary = candidate.slice(0, 700);
            provider = "openai-chat";
          }
        } finally {
          clearTimeout(timeoutId);
        }
      }

      this.graph.setUrlStatus(url, {
        analysis_summary: summary,
        analysis_provider: provider,
        analysis_model: LLM_MODEL,
        last_analyzed_at: nowMs(),
      });
      this.stats.llm_analysis_success += 1;
      this._emit("link_text_analyzed", {
        url,
        source: String(source || "unknown"),
        provider,
        model: LLM_MODEL,
        summary,
      });
    } catch (err) {
      this.stats.llm_analysis_fail += 1;
      this.graph.setUrlStatus(url, {
        analysis_summary: fallbackSummary,
        analysis_provider: "fallback",
        analysis_model: LLM_MODEL,
        last_analyzed_at: nowMs(),
      });
      this._emit("link_text_analysis_failed", {
        url,
        source: String(source || "unknown"),
        model: LLM_MODEL,
        error: String(err?.message || err),
      });
    } finally {
      this.analysisInFlight.delete(url);
    }
  }

  async _onEntityArrive(entity) {
    const url = entity.current_url;
    if (!url) {
      entity.state = "idle";
      return;
    }

    const interaction = this._activateNode(
      url,
      ENTITY_VISIT_ACTIVATION_DELTA,
      entity.id,
      true,
    );
    if (interaction.ok) {
      const node = this.graph.getUrlNode(url);
      if (node && node.text_excerpt) {
        this._analyzeNodeText(url, entity.id).catch(() => {});
      }
    }

    entity.visits += 1;
    entity.last_visit_at = nowMs();
    entity.state = "cooldown";
    entity.next_available_at = nowMs() + clamp(Math.floor(this.defaultDelayMs * 0.85), 400, 5000);
    this.stats.entity_visits += 1;
    this._emit("entity_visit", {
      entity_id: entity.id,
      url,
      visit_count: entity.visits,
      activation_potential: interaction.activation_potential || 0,
      interaction_count: interaction.interaction_count || 0,
    });
  }

  entityTick() {
    if (!this.running || this.paused || !this.entitiesEnabled || this.entitiesPaused) {
      return;
    }

    const now = nowMs();
    for (const entity of this.entities) {
      if (entity.state === "moving" && entity.target_url) {
        const elapsed = now - Number(entity.move_started_at || now);
        const eta = Math.max(1, Number(entity.move_eta_ms || 1));
        entity.progress = clamp(elapsed / eta, 0, 1);
        if (entity.progress >= 1) {
          entity.current_url = entity.target_url;
          entity.target_url = null;
          entity.progress = 1;
          entity.state = "visiting";
          this._emit("entity_arrived", {
            entity_id: entity.id,
            url: entity.current_url,
          });
          this._onEntityArrive(entity).catch(() => {
            entity.state = "idle";
          });
        }
        continue;
      }

      if (entity.state === "visiting") {
        continue;
      }

      if (entity.state === "cooldown") {
        if (now < Number(entity.next_available_at || 0)) {
          continue;
        }
        entity.state = "idle";
        entity.progress = 0;
      }

      const candidates = this._candidateTargetsForEntity(entity);
      if (candidates.length === 0) {
        continue;
      }
      const pick = candidates[0];
      this._setEntityTarget(entity, pick.url, pick.reason);
    }

    this._emitEntityTick(false);
  }

  entityStatus() {
    return {
      ok: true,
      enabled: this.entitiesEnabled,
      paused: this.entitiesPaused,
      count: this.entities.length,
      activation_threshold: this.activationThreshold,
      node_cooldown_ms: this.nodeCooldownMs,
      max_requests_per_host: this.currentMaxRequestsPerHost,
      llm: {
        enabled: LLM_ENABLED,
        base_url: LLM_BASE_URL,
        model: LLM_MODEL,
      },
      entities: this._entitySnapshot(),
    };
  }

  entityControl({ action, count, activationThreshold, nodeCooldownMs, maxPerHost } = {}) {
    const normalizedAction = String(action || "").trim().toLowerCase();
    if (!normalizedAction) {
      return { ok: false, error: "action is required" };
    }

    if (normalizedAction === "start") {
      this.entitiesEnabled = true;
      this.entitiesPaused = false;
    } else if (normalizedAction === "pause") {
      this.entitiesPaused = true;
    } else if (normalizedAction === "resume") {
      this.entitiesEnabled = true;
      this.entitiesPaused = false;
    } else if (normalizedAction === "stop") {
      this.entitiesEnabled = false;
      this.entitiesPaused = false;
      for (const entity of this.entities) {
        entity.state = "idle";
        entity.target_url = null;
        entity.progress = 0;
      }
    } else if (normalizedAction !== "configure") {
      return { ok: false, error: "unknown entity action" };
    }

    if (count !== undefined) {
      this.entityCount = clamp(Number.parseInt(String(count), 10) || this.entityCount, 0, 24);
      this._bootstrapEntities();
    }
    if (activationThreshold !== undefined) {
      const threshold = Number.parseFloat(String(activationThreshold));
      if (Number.isFinite(threshold)) {
        this.activationThreshold = clamp(threshold, 0.1, 8);
      }
    }
    if (nodeCooldownMs !== undefined) {
      const cooldown = Number.parseInt(String(nodeCooldownMs), 10);
      if (Number.isFinite(cooldown)) {
        this.nodeCooldownMs = clamp(cooldown, 15_000, 24 * 60 * 60 * 1000);
      }
    }
    if (maxPerHost !== undefined) {
      const maxHost = Number.parseInt(String(maxPerHost), 10);
      if (Number.isFinite(maxHost)) {
        this.currentMaxRequestsPerHost = clamp(maxHost, 1, 12);
      }
    }

    this._emit("entity_control", {
      action: normalizedAction,
      enabled: this.entitiesEnabled ? 1 : 0,
      paused: this.entitiesPaused ? 1 : 0,
      count: this.entities.length,
      activation_threshold: this.activationThreshold,
      node_cooldown_ms: this.nodeCooldownMs,
      max_requests_per_host: this.currentMaxRequestsPerHost,
    });
    this._emitEntityTick(true);
    this._persistSnapshot();
    return {
      ok: true,
      action: normalizedAction,
      status: this.entityStatus(),
    };
  }

  registerInteraction({ url, delta, source }) {
    const outcome = this._activateNode(
      url,
      Number.isFinite(Number(delta)) ? Number(delta) : INTERACTION_ACTIVATION_DELTA,
      source || "client",
      true,
    );
    if (!outcome.ok) {
      return outcome;
    }
    this._emitEntityTick(true);
    this._persistSnapshot();
    return {
      ok: true,
      interaction: outcome,
      status: this.entityStatus(),
    };
  }

  _ingestSemanticReferences(item, semanticReferences) {
    const createdNodes = [];
    const createdEdges = [];
    const nodeSeen = new Set();
    const edgeSeen = new Set();
    const kindCounts = {};
    let enqueued = 0;
    let createdSemanticEdges = 0;

    for (const reference of semanticReferences) {
      const edgeKind = String(reference.edge_kind || "semantic_reference").trim().toLowerCase();
      if (!edgeKind) {
        continue;
      }
      const targetUrl = normalizeUrl(reference.url, item.url);
      if (!targetUrl || targetUrl === item.url) {
        continue;
      }

      const targetResult = this.graph.upsertUrl(targetUrl, item.depth + 1, item.url);
      if (targetResult.rejected) {
        this.stats.skipped += 1;
        this._emit("fetch_skipped", {
          url: targetUrl,
          source: item.url,
          depth: item.depth + 1,
          reason: "max_nodes_semantic_reference",
          edge_kind: edgeKind,
        });
        continue;
      }

      if (targetResult.created && !nodeSeen.has(targetResult.created.id)) {
        nodeSeen.add(targetResult.created.id);
        createdNodes.push(targetResult.created);
      }

      const targetDomain = new URL(targetUrl).hostname;
      const domainResult = this.graph.upsertDomain(targetDomain);
      if (domainResult.created && !nodeSeen.has(domainResult.created.id)) {
        nodeSeen.add(domainResult.created.id);
        createdNodes.push(domainResult.created);
      }

      const membershipEdge = this.graph.upsertEdge(
        "domain_membership",
        targetResult.id,
        domainResult.id,
      );
      if (membershipEdge && !edgeSeen.has(membershipEdge.id)) {
        edgeSeen.add(membershipEdge.id);
        createdEdges.push(membershipEdge);
      }

      const semanticEdge = this.graph.upsertEdge(
        edgeKind,
        `url:${item.url}`,
        targetResult.id,
        {
          relation: String(reference.reason || "semantic_reference"),
          nofollow: reference.nofollow ? 1 : 0,
        },
      );
      if (semanticEdge && !edgeSeen.has(semanticEdge.id)) {
        edgeSeen.add(semanticEdge.id);
        createdEdges.push(semanticEdge);
        createdSemanticEdges += 1;
        kindCounts[edgeKind] = (kindCounts[edgeKind] || 0) + 1;
        this._recordSemanticEdgeStats(edgeKind);
      }

      if (reference.enqueue && !reference.nofollow && item.depth + 1 <= this.currentMaxDepth) {
        const outcome = this.enqueueUrl(
          targetUrl,
          item.url,
          item.depth + 1,
          String(reference.reason || "semantic_reference"),
        );
        if (outcome.ok) {
          enqueued += 1;
        }
      }
    }

    this.stats.semantic_edges += createdSemanticEdges;
    return {
      createdNodes,
      createdEdges,
      createdSemanticEdges,
      enqueued,
      kindCounts,
    };
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
    const cooldownRemainingMs = this._cooldownRemainingMs(normalized);
    if (cooldownRemainingMs > 0) {
      this.stats.cooldown_blocked += 1;
      return {
        ok: false,
        reason: "cooldown_active",
        retry_in_ms: cooldownRemainingMs,
      };
    }
    if (this.inFlightUrls.has(normalized)) {
      return { ok: false, reason: "in_flight" };
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

    this.graph.setUrlStatus(normalized, {
      status: "queued",
      queued_at: nowMs(),
      last_enqueue_reason: reason,
    });

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

    if (this.inFlightUrls.has(item.url)) {
      this.stats.skipped += 1;
      this._emit("fetch_skipped", {
        url: item.url,
        depth: item.depth,
        reason: "already_in_flight",
      });
      return;
    }

    const cooldownRemainingMs = this._cooldownRemainingMs(item.url);
    if (cooldownRemainingMs > 0) {
      this.stats.skipped += 1;
      this.stats.cooldown_blocked += 1;
      this._emit("fetch_skipped", {
        url: item.url,
        depth: item.depth,
        reason: "node_cooldown",
        retry_in_ms: cooldownRemainingMs,
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
    if (domainState.active >= this.currentMaxRequestsPerHost) {
      const retryAt = now + clamp(Math.floor(this.defaultDelayMs * 0.7), 220, 2200);
      this.frontier.push({
        ...item,
        readyAt: retryAt,
      });
      this.stats.skipped += 1;
      this.stats.host_concurrency_waits += 1;
      this._emit("fetch_skipped", {
        url: item.url,
        depth: item.depth,
        reason: "host_concurrency_wait",
        retry_in_ms: retryAt - now,
        host_active: domainState.active,
        host_limit: this.currentMaxRequestsPerHost,
      });
      return;
    }

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

    this.inFlightUrls.add(item.url);
    domainState.active += 1;
    domainState.nextAllowedAt = nowMs() + delayMs + domainState.backoffMs;
    this.graph.setUrlStatus(item.url, {
      status: "fetching",
      last_requested_at: nowMs(),
      cooldown_until: nowMs() + this.nodeCooldownMs,
    });

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
      let bodyText = "";
      let contentHash = "";
      if (isTextLikeContentType(contentType)) {
        const rawText = await response.text();
        bodyText = rawText.slice(0, 750000);
        contentHash = hashText(bodyText);
      } else {
        const rawBuffer = Buffer.from(await response.arrayBuffer());
        const digestBuffer =
          rawBuffer.length > 2_000_000 ? rawBuffer.subarray(0, 2_000_000) : rawBuffer;
        contentHash = hashText(digestBuffer);
      }
      const knownUrl = this.contentHashIndex.get(contentHash);

      const title = contentType.includes("html") ? extractTitle(bodyText) : "";
      const readableText = contentType.includes("html")
        ? extractReadableTextFromHtml(bodyText)
        : String(bodyText || "").replace(/\s+/g, " ").trim();
      const textExcerpt = readableText.slice(0, LLM_TEXT_MAX_CHARS);
      this.graph.setUrlStatus(item.url, {
        status: "fetched",
        compliance: "allowed",
        fetched_at: nowMs(),
        content_type: contentType,
        content_hash: contentHash,
        title,
        text_excerpt: textExcerpt,
        text_excerpt_hash: textExcerpt ? hashText(textExcerpt) : "",
        last_visited_at: nowMs(),
        cooldown_until: nowMs() + this.nodeCooldownMs,
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
      this._pruneContentHashIndex();

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

        const semantic = extractSemanticReferences(item.url, bodyText);
        if (semantic.references.length > 0) {
          const semanticOutcome = this._ingestSemanticReferences(item, semantic.references);
          outboundCount += semanticOutcome.enqueued;
          this._emitGraphDelta(
            "semantic_reference",
            semanticOutcome.createdNodes,
            semanticOutcome.createdEdges,
          );
          this._emit("reference_extracted", {
            url: item.url,
            depth: item.depth,
            source_kind: semantic.source_kind,
            discovered: semantic.references.length,
            created_edges: semanticOutcome.createdSemanticEdges,
            enqueued: semanticOutcome.enqueued,
            kind_counts: semanticOutcome.kindCounts,
          });
        }
      }

      this.stats.fetched += 1;
      this.stats.total_fetch_time_ms += nowMs() - startedAt;
      domainState.lastFetchedAt = nowMs();

      if (textExcerpt) {
        this._analyzeNodeText(item.url, "fetch_visit").catch(() => {});
      }

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
      this.inFlightUrls.delete(item.url);
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

  start({ seeds, maxDepth, maxNodes, concurrency, maxPerHost, entityCount }) {
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
    this.entitiesPaused = false;
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
    this.currentMaxRequestsPerHost = clamp(
      Number.parseInt(String(maxPerHost || DEFAULT_MAX_REQUESTS_PER_HOST), 10)
        || DEFAULT_MAX_REQUESTS_PER_HOST,
      1,
      12,
    );
    if (entityCount !== undefined) {
      this.entityCount = clamp(
        Number.parseInt(String(entityCount), 10) || this.entityCount,
        0,
        24,
      );
      this._bootstrapEntities();
      this._emitEntityTick(true);
    }
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
      max_requests_per_host: this.currentMaxRequestsPerHost,
      entities: this.entities.length,
      user_agent: USER_AGENT,
    });
    this._emitEntityTick(true);
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
    this.entitiesPaused = true;
    this._emit("crawl_state", {
      state: "paused",
    });
    this._emitEntityTick(true);
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
    this.entitiesPaused = false;
    this._emit("crawl_state", {
      state: "running",
    });
    this._emitEntityTick(true);
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
    this.inFlightUrls.clear();
    for (const entity of this.entities) {
      entity.state = "idle";
      entity.target_url = null;
      entity.progress = 0;
    }
    this._emit("crawl_state", {
      state: "stopped",
      reason: "manual_stop",
    });
    this._emitEntityTick(true);
    this._persistSnapshot();
    return {
      ok: true,
      state: "stopped",
    };
  }

  shutdown() {
    this.running = false;
    this.paused = false;
    this.frontier.clear();
    clearInterval(this.scheduler);
    clearInterval(this.entityScheduler);
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
    const sourceFamilyCounts = {
      arxiv: 0,
      wikipedia: 0,
      web: 0,
    };
    const knowledgeKindCounts = {
      arxiv_abs: 0,
      arxiv_pdf: 0,
      wikipedia_article: 0,
      web_url: 0,
    };
    for (const node of urlNodes) {
      const sourceFamily = String(node.source_family || "web");
      if (sourceFamily in sourceFamilyCounts) {
        sourceFamilyCounts[sourceFamily] += 1;
      } else {
        sourceFamilyCounts.web += 1;
      }

      const knowledgeKind = String(node.knowledge_kind || "web_url");
      if (knowledgeKind in knowledgeKindCounts) {
        knowledgeKindCounts[knowledgeKind] += 1;
      } else {
        knowledgeKindCounts.web_url += 1;
      }
    }

    return {
      ok: true,
      state: this.running ? (this.paused ? "paused" : "running") : "stopped",
      started_at: this.startedAtMs,
      user_agent: USER_AGENT,
      config: {
        max_depth: this.currentMaxDepth,
        max_nodes: this.currentMaxNodes,
        concurrency: this.currentConcurrency,
        max_requests_per_host: this.currentMaxRequestsPerHost,
        default_delay_ms: this.defaultDelayMs,
        node_cooldown_ms: this.nodeCooldownMs,
        activation_threshold: this.activationThreshold,
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
        semantic_edges: this.stats.semantic_edges,
        citation_edges: this.stats.citation_edges,
        wiki_reference_edges: this.stats.wiki_reference_edges,
        cross_reference_edges: this.stats.cross_reference_edges,
        paper_pdf_edges: this.stats.paper_pdf_edges,
        host_concurrency_waits: this.stats.host_concurrency_waits,
        cooldown_blocked: this.stats.cooldown_blocked,
        interactions: this.stats.interactions,
        activation_enqueues: this.stats.activation_enqueues,
        entity_moves: this.stats.entity_moves,
        entity_visits: this.stats.entity_visits,
        llm_analysis_success: this.stats.llm_analysis_success,
        llm_analysis_fail: this.stats.llm_analysis_fail,
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
      knowledge: {
        source_families: sourceFamilyCounts,
        node_kinds: knowledgeKindCounts,
      },
      entities: this.entityStatus(),
      llm: {
        enabled: LLM_ENABLED,
        base_url: LLM_BASE_URL,
        model: LLM_MODEL,
      },
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

function createWeaverServer() {
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
        entities: weaver.entityStatus(),
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
        version: "0.3.0",
        status_endpoint: "/api/weaver/status",
        websocket_endpoint: "/ws",
        note: "Ethical crawl instrumentation service with entity-driven arXiv/Wikipedia exploration",
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

    if (req.method === "GET" && pathname === "/api/weaver/entities") {
      sendJson(res, 200, weaver.entityStatus());
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
            maxPerHost: body.max_per_host,
            entityCount: body.entity_count,
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

    if (req.method === "POST" && pathname === "/api/weaver/entities/control") {
      try {
        const body = await parseJsonBody(req);
        const result = weaver.entityControl({
          action: body.action,
          count: body.count,
          activationThreshold: body.activation_threshold,
          nodeCooldownMs: body.node_cooldown_ms,
          maxPerHost: body.max_per_host,
        });
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

    if (req.method === "POST" && pathname === "/api/weaver/entities/interact") {
      try {
        const body = await parseJsonBody(req);
        const result = weaver.registerInteraction({
          url: body.url,
          delta: body.delta,
          source: body.source,
        });
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

  function close() {
    for (const ws of wsClients) {
      try {
        ws.close();
      } catch (_err) {
        // noop
      }
    }
    wsClients.clear();
    wsServer.close();
    weaver.shutdown();
    return new Promise((resolve) => {
      if (!server.listening) {
        resolve();
        return;
      }
      server.close(() => {
        resolve();
      });
    });
  }

  return {
    weaver,
    server,
    wsServer,
    wsClients,
    close,
  };
}

if (require.main === module) {
  const runtime = createWeaverServer();
  runtime.server.listen(PORT, HOST, () => {
    console.log(`[weaver] Web Graph Weaver listening on http://${HOST}:${PORT}`);
    console.log(`[weaver] User-Agent: ${USER_AGENT}`);
  });
}

module.exports = {
  normalizeUrl,
  extractArxivIdFromUrl,
  canonicalArxivAbsUrlFromId,
  canonicalArxivPdfUrlFromId,
  canonicalWikipediaArticleUrl,
  extractSemanticReferences,
  classifyKnowledgeUrl,
  FrontierQueue,
  GraphStore,
  WebGraphWeaver,
  createWeaverServer,
};
