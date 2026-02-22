import http from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const HOST = process.env.MOCK_WORLD_HOST || "127.0.0.1";
const PORT = Number(process.env.MOCK_WORLD_PORT || 8787);
const scriptDir = dirname(fileURLToPath(import.meta.url));
const distDir = join(scriptDir, "..", "dist");

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967295;
  };
}

const presenceIds = [
  "receipt_river",
  "witness_thread",
  "fork_tax_canticle",
  "mage_of_receipts",
  "keeper_of_receipts",
  "anchor_registry",
  "gates_of_truth",
];

const entityManifest = presenceIds.map((id, index) => {
  const angle = (index / Math.max(1, presenceIds.length)) * Math.PI * 2;
  return {
    id,
    en: id.replace(/_/g, " "),
    ja: "presence",
    x: clamp01(0.5 + Math.cos(angle) * 0.28),
    y: clamp01(0.5 + Math.sin(angle) * 0.24),
    hue: (index * 360) / presenceIds.length,
  };
});

function makeFileGraph() {
  const random = seededRandom(41);
  const fieldNodes = entityManifest.map((row) => ({
    id: `field:${row.id}`,
    node_id: `field:${row.id}`,
    node_type: "field",
    field: row.id,
    label: row.en,
    x: row.x,
    y: row.y,
    hue: row.hue,
    importance: 0.7,
  }));

  const fileNodes = [];
  for (let i = 0; i < 140; i += 1) {
    const field = entityManifest[i % entityManifest.length];
    const jitterX = (random() - 0.5) * 0.36;
    const jitterY = (random() - 0.5) * 0.28;
    fileNodes.push({
      id: `file:doc:${i}`,
      node_id: `file:doc:${i}`,
      node_type: "file",
      field: field.id,
      label: `doc-${i}`,
      source_rel_path: `notes/doc-${i}.md`,
      x: clamp01(field.x + jitterX),
      y: clamp01(field.y + jitterY),
      hue: (field.hue + 24) % 360,
      importance: 0.3 + random() * 0.7,
      kind: i % 5 === 0 ? "image" : "text",
      title: `Dummy doc ${i}`,
      status: "ok",
      tags: ["dummy", `field-${i % entityManifest.length}`],
    });
  }

  const crawlerNodes = [];
  for (let i = 0; i < 60; i += 1) {
    const parent = fileNodes[i * 2];
    crawlerNodes.push({
      id: `crawler:url:${i}`,
      node_id: `crawler:url:${i}`,
      node_type: "crawler",
      crawler_kind: "url",
      label: `https://dummy.example/${i}`,
      url: `https://dummy.example/${i}`,
      domain: "dummy.example",
      title: `Dummy page ${i}`,
      content_type: i % 7 === 0 ? "image/png" : "text/html",
      compliance: "ok",
      x: clamp01((parent?.x ?? 0.5) + (random() - 0.5) * 0.22),
      y: clamp01((parent?.y ?? 0.5) + (random() - 0.5) * 0.2),
      hue: 188,
      importance: 0.22 + random() * 0.64,
    });
  }

  const nodes = [...fieldNodes, ...fileNodes, ...crawlerNodes];
  const edges = [];
  let edgeId = 0;

  fileNodes.forEach((node, index) => {
    edges.push({
      id: `edge:${edgeId++}`,
      source: `field:${node.field}`,
      target: node.id,
      field: node.field,
      kind: "categorizes",
      weight: 0.5,
    });
    if (index > 0) {
      edges.push({
        id: `edge:${edgeId++}`,
        source: fileNodes[index - 1].id,
        target: node.id,
        field: node.field,
        kind: "hyperlink",
        weight: 0.2 + random() * 0.6,
      });
    }
    if (index % 4 === 0) {
      const crawlerIndex = Math.floor(index / 4) % crawlerNodes.length;
      edges.push({
        id: `edge:${edgeId++}`,
        source: node.id,
        target: crawlerNodes[crawlerIndex].id,
        field: node.field,
        kind: "citation",
        weight: 0.28 + random() * 0.5,
      });
    }
  });

  return {
    record: "dummy:file-graph",
    generated_at: new Date().toISOString(),
    inbox: {
      record: "dummy:inbox",
      path: "dummy",
      pending_count: 0,
      processed_count: fileNodes.length,
      failed_count: 0,
      is_empty: false,
      knowledge_entries: fileNodes.length,
      last_ingested_at: new Date().toISOString(),
      errors: [],
    },
    nodes,
    field_nodes: fieldNodes,
    file_nodes: fileNodes,
    crawler_nodes: crawlerNodes,
    edges,
    stats: {
      field_count: fieldNodes.length,
      file_count: fileNodes.length,
      edge_count: edges.length,
      kind_counts: {},
      field_counts: {},
      knowledge_entries: fileNodes.length,
    },
  };
}

function makeCrawlerGraph(fileGraph) {
  const crawlerNodes = Array.isArray(fileGraph.crawler_nodes) ? fileGraph.crawler_nodes : [];
  const fieldNodes = Array.isArray(fileGraph.field_nodes) ? fileGraph.field_nodes : [];
  const edges = [];
  let edgeIndex = 0;
  crawlerNodes.forEach((node, index) => {
    const field = fieldNodes[index % Math.max(1, fieldNodes.length)];
    if (field) {
      edges.push({
        id: `crawler-edge:${edgeIndex++}`,
        source: field.id,
        target: node.id,
        field: field.field || field.id,
        kind: "hyperlink",
        weight: 0.32,
      });
    }
    if (index > 0) {
      edges.push({
        id: `crawler-edge:${edgeIndex++}`,
        source: crawlerNodes[index - 1].id,
        target: node.id,
        field: "dummy",
        kind: "cross_reference",
        weight: 0.26,
      });
    }
  });
  return {
    record: "dummy:crawler-graph",
    generated_at: new Date().toISOString(),
    source: {
      endpoint: "dummy",
      service: "dummy",
    },
    status: {},
    nodes: [...fieldNodes, ...crawlerNodes],
    field_nodes: fieldNodes,
    crawler_nodes: crawlerNodes,
    edges,
    stats: {
      field_count: fieldNodes.length,
      crawler_count: crawlerNodes.length,
      edge_count: edges.length,
      kind_counts: {},
      field_counts: {},
      nodes_total: fieldNodes.length + crawlerNodes.length,
      edges_total: edges.length,
      url_nodes_total: crawlerNodes.length,
    },
  };
}

const fileGraph = makeFileGraph();
const crawlerGraph = makeCrawlerGraph(fileGraph);

const catalog = {
  generated_at: new Date().toISOString(),
  entity_manifest: entityManifest,
  file_graph: fileGraph,
  crawler_graph: crawlerGraph,
  presence_runtime: {
    compute_jobs: [],
    compute_jobs_180s: 0,
    devices: {},
  },
};

function buildSimulation() {
  const random = seededRandom(Math.floor(Date.now() / 1000));
  const particles = [];
  const points = [];
  for (let i = 0; i < 1400; i += 1) {
    const form = entityManifest[i % entityManifest.length];
    const angle = (i / 1400) * Math.PI * 8 + random() * 0.6;
    const orbit = 0.04 + ((i % 17) / 17) * 0.19;
    const x = clamp01(form.x + Math.cos(angle) * orbit + (random() - 0.5) * 0.03);
    const y = clamp01(form.y + Math.sin(angle) * orbit + (random() - 0.5) * 0.03);
    const hue = (form.hue + i * 0.6) % 360;
    const colorBand = (hue % 120) / 120;
    const r = 0.35 + colorBand * 0.5;
    const g = 0.42 + (1 - colorBand) * 0.45;
    const b = 0.5 + ((Math.sin(angle) + 1) * 0.25);
    const routeNode = fileGraph.file_nodes[i % fileGraph.file_nodes.length];
    const graphNode = fileGraph.crawler_nodes[i % fileGraph.crawler_nodes.length];
    particles.push({
      id: `particle:${i}`,
      presence_id: form.id,
      x,
      y,
      size: 0.6 + random() * 2.2,
      r,
      g,
      b,
      particle_mode: i % 19 === 0 ? "chaos-butterfly" : "role-bound",
      is_nexus: i % 37 === 0,
      route_node_id: routeNode.id,
      graph_node_id: graphNode.id,
      top_job: i % 23 === 0 ? "emit_resource_packet" : "observe",
      resource_daimoi: i % 23 === 0,
      resource_type: i % 2 === 0 ? "gpu" : "npu",
      route_probability: random(),
      drift_score: random() * 0.8,
    });
    points.push({
      x: x * 2 - 1,
      y: y * 2 - 1,
      z: random() * 2 - 1,
      size: 1.0 + random() * 2.4,
      r,
      g,
      b,
    });
  }

  return {
    timestamp: new Date().toISOString(),
    total: particles.length,
    audio: 220,
    image: 340,
    video: 80,
    points,
    field_particles: particles,
    presence_dynamics: {
      field_particles: particles,
      compute_jobs: [],
      compute_jobs_180s: 0,
      devices: {},
    },
    file_graph: fileGraph,
    crawler_graph: crawlerGraph,
    entities: entityManifest.map((row, index) => ({
      id: row.id,
      bpm: 72 + index * 6,
      stability: 0.45 + index * 0.04,
      resonance: 0.34 + index * 0.07,
      field_layer: index % 8,
    })),
  };
}

function sendJson(res, status, payload) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function setCorsHeaders(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

function contentTypeForPath(pathname) {
  const ext = extname(pathname).toLowerCase();
  if (ext === ".html") return "text/html; charset=utf-8";
  if (ext === ".js" || ext === ".mjs") return "application/javascript; charset=utf-8";
  if (ext === ".css") return "text/css; charset=utf-8";
  if (ext === ".svg") return "image/svg+xml";
  if (ext === ".json") return "application/json; charset=utf-8";
  if (ext === ".png") return "image/png";
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".webp") return "image/webp";
  if (ext === ".ico") return "image/x-icon";
  return "application/octet-stream";
}

async function serveStaticAsset(pathname, res) {
  const safePath = normalize(pathname).replace(/^\/+/, "");
  const target = safePath ? join(distDir, safePath) : join(distDir, "index.html");
  try {
    const body = await readFile(target);
    res.statusCode = 200;
    res.setHeader("Content-Type", contentTypeForPath(target));
    res.end(body);
    return true;
  } catch {
    return false;
  }
}

const server = http.createServer(async (req, res) => {
  setCorsHeaders(res);
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }

  const requestUrl = new URL(req.url || "/", `http://${HOST}:${PORT}`);
  const { pathname } = requestUrl;

  if (req.method === "GET" && pathname === "/api/catalog") {
    sendJson(res, 200, catalog);
    return;
  }

  if (req.method === "GET" && pathname === "/api/simulation") {
    sendJson(res, 200, buildSimulation());
    return;
  }

  if (req.method === "GET" && pathname === "/api/ui/projection") {
    sendJson(res, 200, { ok: true, projection: null });
    return;
  }

  if (req.method === "POST" && pathname === "/api/witness") {
    sendJson(res, 200, { ok: true });
    return;
  }

  if (req.method === "POST" && pathname === "/api/presence/user/input") {
    sendJson(res, 200, { ok: true, processed: 1 });
    return;
  }

  if (pathname === "/ws") {
    res.statusCode = 426;
    res.end("websocket not implemented in dummy server");
    return;
  }

  if (req.method === "GET") {
    const served = await serveStaticAsset(pathname, res);
    if (served) {
      return;
    }
    const fallback = await serveStaticAsset("/index.html", res);
    if (fallback) {
      return;
    }
  }

  sendJson(res, 404, { ok: false, error: `not found: ${pathname}` });
});

server.listen(PORT, HOST, () => {
  process.stdout.write(`dummy world server listening on http://${HOST}:${PORT}\n`);
});

const close = () => {
  server.close(() => process.exit(0));
};

process.on("SIGINT", close);
process.on("SIGTERM", close);
