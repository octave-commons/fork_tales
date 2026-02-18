#!/usr/bin/env node
/**
 * contracts/契_ημ_ingest_v1.mjs
 *
 * Executable Contract (契): policy + meters + verification + deterministic failure modes
 * Target: implement the .ημ ingest -> (vision/text normalize) -> embed -> registry pipeline
 *
 * Run:
 *   node contracts/契_ημ_ingest_v1.mjs --test
 *   node contracts/契_ημ_ingest_v1.mjs --demo
 *
 * Implement by wiring handlers in `HANDLERS` (vision.describe, embed.write, etc).
 */

import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

const GLYPH_CONTRACT = "契";
const CTX = Object.freeze({ self: "己", you: "汝", them: "彼", world: "世" });

// -----------------------------
// 0) Contract: data, not prose
// -----------------------------
export const CONTRACT_ημ_INGEST_V1 = Object.freeze({
  glyph: GLYPH_CONTRACT,
  id: "sha256:TBD_BY_IMPLEMENTATION", // compute from canonical serialization if you want
  ver: "0.1.0",
  title: "ημ ingest → describe/normalize → embed → registry (local-only)",
  parties: [
    { ctx: CTX.self, id: "cephalon" },
    { ctx: CTX.world, id: "fs" },
    { ctx: CTX.world, id: "vision.local" },
    { ctx: CTX.world, id: "embed.local" },
    { ctx: CTX.world, id: "vectorstore.chroma" },
  ],

  // Hard boundaries: only these roots are in-scope.
  scope: {
    roots: [".ημ", ".Π"],
    ingestRoot: ".ημ",
    outRoot: ".Π",
    registryPath: ".ημ/ημ_registry.jsonl",
    backlogDir: ".ημ/backlog",
    allowedExt: {
      text: [".txt", ".md", ".json", ".yaml", ".yml", ".csv"],
      image: [".png", ".jpg", ".jpeg", ".webp", ".gif"],
    },
  },

  // Permits: allowlist. If not explicitly allowed, it is denied.
  permits: [
    rule({ kind: "fs.walk", on: "fs", withinRoots: true }),
    rule({ kind: "fs.read", on: "fs", withinRoots: true }),
    rule({ kind: "fs.append", on: "fs", withinRoots: true }),
    rule({ kind: "fs.mkdir", on: "fs", withinRoots: true }),
    rule({ kind: "vision.describe", on: "vision.local" }),
    rule({ kind: "embed.write", on: "vectorstore.chroma" }),
  ],

  // Forbids: explicit deny rules (checked before permits).
  forbids: [
    // No public internet by default.
    rule({ kind: "net.http", on: "public-internet" }),
    // No filesystem access outside roots.
    rule({ kind: "fs.*", on: "fs", withinRoots: false }),
  ],

  // Cost model: enforced meters (rate + concurrency + cap).
  cost: {
    meters: {
      // limits per process (you can persist later)
      "vision.describe": meter({
        unit: "requests",
        perMs: 60_000,
        maxPerWindow: 6,
        burst: 3,
        maxConcurrent: 1,
        totalCap: 500,
      }),
      "embed.write": meter({
        unit: "requests",
        perMs: 60_000,
        maxPerWindow: 12,
        burst: 6,
        maxConcurrent: 2,
        totalCap: 5_000,
      }),
    },
  },

  // Verify: machine-checkable gates with deterministic fail directives.
  verify: [
    check("path-within-roots", {
      when: (act) => act.kind.startsWith("fs."),
      must: (act, st, c) => isWithinRoots(act.attrs?.path ?? "", c.scope.roots),
      onFail: fail("deny", "path_out_of_scope"),
    }),

    check("content-hash-required", {
      when: (act) => ["vision.describe", "embed.write"].includes(act.kind),
      must: (act) => typeof act.attrs?.contentSha256 === "string" && act.attrs.contentSha256.length === 64,
      onFail: fail("deny", "missing_content_sha256"),
    }),

    check("idempotent-registry", {
      when: (act) => act.kind === "embed.write",
      must: (act, st) => !st.registry.has(act.attrs.contentSha256),
      onFail: fail("defer", "already_processed"),
    }),

    check("rate-and-concurrency", {
      when: (act) => act.kind in CONTRACT_ημ_INGEST_V1.cost.meters,
      must: (act, st, c) => chargeMeter(st, c, act.kind),
      onFail: fail("defer", "meter_exceeded"),
    }),
  ],

  // Fails: deterministic fallbacks (never expand permissions).
  fails: {
    deferToBacklog: true,
  },

  // Pipeline: executable plan skeleton (handlers implement side-effects).
  pipeline: [
    step("scan", {
      act: { kind: "fs.walk", on: "fs", attrs: { path: ".ημ" } },
      out: "files",
    }),

    step("filter-allowed", {
      fn: (st, c) => {
        const allowed = new Set([...c.scope.allowedExt.text, ...c.scope.allowedExt.image]);
        const keep = st.files.filter((p) => allowed.has(path.extname(p).toLowerCase()));
        return { ...st, candidates: keep };
      },
    }),

    step("hash-and-route", {
      fn: async (st, c) => {
        const routed = [];
        for (const p of st.candidates) {
          const sha = await sha256File(p);
          const ext = path.extname(p).toLowerCase();
          const kind = c.scope.allowedExt.image.includes(ext) ? "image" : "text";
          routed.push({ path: p, sha, kind });
        }
        return { ...st, routed };
      },
    }),

    step("process-each", {
      fn: async (st, c, runtime) => {
        const results = [];
        for (const item of st.routed) {
          // idempotence at file level (optional): if already in registry, skip early
          if (st.registry.has(item.sha)) continue;

          if (item.kind === "image") {
            const desc = await runtime.call({
              kind: "vision.describe",
              who: { ctx: CTX.self, id: "cephalon" },
              on: { ctx: CTX.world, id: "vision.local" },
              attrs: { path: item.path, contentSha256: item.sha },
            });
            results.push({ ...item, textForEmbed: desc.text });
          } else {
            const raw = await runtime.call({
              kind: "fs.read",
              who: { ctx: CTX.self, id: "cephalon" },
              on: { ctx: CTX.world, id: "fs" },
              attrs: { path: item.path },
            });
            results.push({ ...item, textForEmbed: normalizeText(raw.text) });
          }
        }
        return { ...st, processed: results };
      },
    }),

    step("embed-and-record", {
      fn: async (st, c, runtime) => {
        await runtime.call({
          kind: "fs.mkdir",
          who: { ctx: CTX.self, id: "cephalon" },
          on: { ctx: CTX.world, id: "fs" },
          attrs: { path: c.scope.backlogDir },
        });

        for (const item of st.processed) {
          const act = {
            kind: "embed.write",
            who: { ctx: CTX.self, id: "cephalon" },
            on: { ctx: CTX.world, id: "vectorstore.chroma" },
            attrs: {
              contentSha256: item.sha,
              sourcePath: item.path,
              modality: item.kind,
              text: item.textForEmbed,
            },
          };

          const decision = runtime.enforce(act);
          if (!decision.ok) {
            if (decision.mode === "defer" && c.fails.deferToBacklog) {
              await deferToBacklog(c, act, decision);
              continue;
            }
            throw new Error(`Denied: ${decision.why}`);
          }

          await runtime.call(act);

          // append registry
          const rec = {
            sha256: item.sha,
            path: item.path,
            modality: item.kind,
            status: "embedded",
            time: new Date().toISOString(),
          };
          await runtime.call({
            kind: "fs.append",
            who: { ctx: CTX.self, id: "cephalon" },
            on: { ctx: CTX.world, id: "fs" },
            attrs: { path: c.scope.registryPath, text: JSON.stringify(rec) + "\n" },
          });

          st.registry.add(item.sha);
        }

        return st;
      },
    }),
  ],

  // Tests: “if it isn’t in verify, it isn’t enforceable”
  tests: [
    {
      name: "deny public internet",
      act: { kind: "net.http", on: { ctx: CTX.world, id: "public-internet" }, attrs: { url: "https://example.com" } },
      expectOk: false,
      expectWhy: "forbidden",
    },
    {
      name: "deny fs.read out of scope",
      act: { kind: "fs.read", on: { ctx: CTX.world, id: "fs" }, attrs: { path: "/etc/passwd" } },
      expectOk: false,
      expectWhy: "path_out_of_scope",
    },
    {
      name: "deny embed.write without contentSha256",
      act: { kind: "embed.write", on: { ctx: CTX.world, id: "vectorstore.chroma" }, attrs: { text: "hi" } },
      expectOk: false,
      expectWhy: "missing_content_sha256",
    },
    {
      name: "defer embed.write if already processed",
      setup: (st) => st.registry.add("a".repeat(64)),
      act: {
        kind: "embed.write",
        on: { ctx: CTX.world, id: "vectorstore.chroma" },
        attrs: { contentSha256: "a".repeat(64), text: "hi" },
      },
      expectOk: false,
      expectMode: "defer",
      expectWhy: "already_processed",
    },
  ],
});

// -----------------------------
// 1) Runtime: enforce + execute
// -----------------------------
export function makeRuntime(contract, handlers = {}) {
  const st = {
    // idempotence registry = set of sha256 strings
    registry: new Set(),
    // meters state: per meter id
    meters: initMeters(contract),
  };

  return {
    state: st,
    enforce: (act) => enforce(contract, st, act),
    call: async (act) => {
      const decision = enforce(contract, st, act);
      if (!decision.ok) {
        if (decision.mode === "defer" && contract.fails.deferToBacklog) {
          await deferToBacklog(contract, act, decision);
          return { deferred: true, decision };
        }
        throw new Error(`Denied: ${decision.why}`);
      }

      const h = handlers[act.kind];
      if (!h) throw new Error(`Handler not implemented for act.kind=${act.kind}`);
      return await h(act, st, contract);
    },
    runPipeline: async () => runPipeline(contract, st, handlers),
    loadRegistry: async () => loadRegistryIntoState(contract, st),
  };
}

function enforce(contract, st, act) {
  const normalized = normalizeAct(act);

  // 1) forbids first
  for (const r of contract.forbids) {
    if (matchRule(contract, normalized, r)) {
      return decision(false, "forbidden", { rule: r });
    }
  }

  // 2) must be permitted by allowlist
  let permitted = false;
  for (const r of contract.permits) {
    if (matchRule(contract, normalized, r)) {
      permitted = true;
      break;
    }
  }
  if (!permitted) return decision(false, "not_permitted", { kind: normalized.kind });

  // 3) verify checks (deterministic fails)
  for (const chk of contract.verify) {
    if (!chk.when(normalized, st, contract)) continue;
    const ok = chk.must(normalized, st, contract);
    if (!ok) return { ok: false, mode: chk.onFail.mode, why: chk.onFail.why, proof: { check: chk.id } };
  }

  return decision(true, "ok");
}

// -----------------------------
// 2) Pipeline executor
// -----------------------------
async function runPipeline(contract, st, handlers) {
  const runtime = makeRuntime(contract, handlers);
  runtime.state.registry = st.registry; // share
  runtime.state.meters = st.meters;

  // best-effort: load registry if present
  await runtime.loadRegistry().catch(() => {});

  let ctx = { registry: st.registry, meters: st.meters };
  for (const s of contract.pipeline) {
    if (s.act) {
      const res = await runtime.call(s.act);
      if (s.out) ctx[s.out] = res.files ?? res;
    } else if (s.fn) {
      ctx = await s.fn(ctx, contract, runtime);
    }
  }
  return ctx;
}

// -----------------------------
// 3) Handlers to implement
// -----------------------------
const HANDLERS = {
  // filesystem
  "fs.walk": async (act) => ({ files: await walkDir(act.attrs.path) }),
  "fs.read": async (act) => ({ text: await fs.readFile(act.attrs.path, "utf8") }),
  "fs.append": async (act) => ({ ok: true, bytes: (await fs.appendFile(act.attrs.path, act.attrs.text)).length }),
  "fs.mkdir": async (act) => (await fs.mkdir(act.attrs.path, { recursive: true }), { ok: true }),

  // TODO: implement these with your local vision + embedding stack
  "vision.describe": async () => {
    throw new Error("TODO: implement vision.describe (local vision model). Return { text: <description> }");
  },
  "embed.write": async () => {
    throw new Error("TODO: implement embed.write (vectorstore write). Use act.attrs.text + metadata.");
  },
};

// -----------------------------
// 4) Utilities: rules, checks, meters
// -----------------------------
function rule(x) {
  return Object.freeze({ ...x });
}

function step(id, x) {
  return Object.freeze({ id, ...x });
}

function check(id, { when, must, onFail }) {
  return Object.freeze({ id, when, must, onFail });
}

function fail(mode, why) {
  return Object.freeze({ mode, why });
}

function decision(ok, why, proof = null) {
  return { ok, why, proof };
}

function normalizeAct(act) {
  // allow both {kind,on,attrs} and full {kind,who,on,attrs}
  const on = act.on ?? { ctx: CTX.world, id: "unknown" };
  return {
    kind: String(act.kind ?? ""),
    who: act.who ?? { ctx: CTX.self, id: "unknown" },
    on,
    attrs: act.attrs ?? {},
  };
}

function matchRule(contract, act, r) {
  // kind matching supports wildcard suffix "fs.*"
  const kindOk =
    r.kind === act.kind ||
    (r.kind?.endsWith(".*") && act.kind.startsWith(r.kind.slice(0, -2)));

  if (!kindOk) return false;

  // "on" matching: either exact id match or omitted
  const onId = act.on?.id ?? "unknown";
  if (r.on && r.on !== onId) return false;

  // withinRoots constraint for fs.*
  if (r.kind?.startsWith("fs.") || r.kind === "fs.*") {
    const p = act.attrs?.path;
    if (typeof r.withinRoots === "boolean" && typeof p === "string") {
      const inRoots = isWithinRoots(p, contract.scope.roots);
      if (r.withinRoots !== inRoots) return false;
    }
  }
  return true;
}

function isWithinRoots(p, roots) {
  // conservative: only allow relative paths under given roots
  if (!p || typeof p !== "string") return false;
  if (path.isAbsolute(p)) return false;

  const norm = path.normalize(p).replace(/\\/g, "/");
  return roots.some((r) => {
    const rr = path.normalize(r).replace(/\\/g, "/").replace(/\/+$/, "");
    return norm === rr || norm.startsWith(rr + "/");
  });
}

function meter({ unit, perMs, maxPerWindow, burst, maxConcurrent, totalCap }) {
  return Object.freeze({ unit, perMs, maxPerWindow, burst, maxConcurrent, totalCap });
}

function initMeters(contract) {
  const now = Date.now();
  const meters = {};
  for (const [k, m] of Object.entries(contract.cost.meters)) {
    meters[k] = {
      spec: m,
      windowStartMs: now,
      usedInWindow: 0,
      concurrent: 0,
      totalUsed: 0,
    };
  }
  return meters;
}

/**
 * chargeMeter(st, contract, meterId) → boolean
 * Implements a simple fixed-window + burst + concurrency + cap.
 * NOTE: totalUsed is incremented on allow; decrement concurrency on handler completion is TODO
 * (Implementers can wrap handlers to manage concurrent accurately.)
 */
function chargeMeter(st, contract, meterId) {
  const m = st.meters[meterId];
  if (!m) return true;

  const now = Date.now();
  const spec = m.spec;

  // reset window
  if (now - m.windowStartMs >= spec.perMs) {
    m.windowStartMs = now;
    m.usedInWindow = 0;
  }

  if (m.totalUsed >= spec.totalCap) return false;
  if (m.concurrent >= spec.maxConcurrent) return false;

  const limit = spec.maxPerWindow + (spec.burst ?? 0);
  if (m.usedInWindow + 1 > limit) return false;

  m.usedInWindow += 1;
  m.totalUsed += 1;
  m.concurrent += 1;
  return true;
}

function normalizeText(s) {
  return String(s).replace(/\r\n/g, "\n").trim();
}

async function sha256File(p) {
  const buf = await fs.readFile(p);
  return crypto.createHash("sha256").update(buf).digest("hex");
}

async function walkDir(root) {
  const out = [];
  async function rec(dir) {
    const ents = await fs.readdir(dir, { withFileTypes: true });
    for (const e of ents) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) await rec(full);
      else out.push(full.replace(/\\/g, "/"));
    }
  }
  await rec(root);
  return out;
}

// -----------------------------
// 5) Registry + backlog
// -----------------------------
async function loadRegistryIntoState(contract, st) {
  const p = contract.scope.registryPath;
  if (!isWithinRoots(p, contract.scope.roots)) return;

  let data = "";
  try {
    data = await fs.readFile(p, "utf8");
  } catch {
    return;
  }
  for (const line of data.split("\n")) {
    if (!line.trim()) continue;
    try {
      const rec = JSON.parse(line);
      if (typeof rec.sha256 === "string" && rec.sha256.length === 64) st.registry.add(rec.sha256);
    } catch {
      // ignore bad lines
    }
  }
}

async function deferToBacklog(contract, act, decision) {
  const dir = contract.scope.backlogDir;
  if (!isWithinRoots(dir, contract.scope.roots)) return;

  await fs.mkdir(dir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const name = `${stamp}_${safeFileSlug(act.kind)}_${(act.attrs?.contentSha256 ?? "nohash").slice(0, 12)}.json`;
  const p = path.join(dir, name).replace(/\\/g, "/");

  const payload = {
    time: new Date().toISOString(),
    decision,
    act,
  };
  await fs.writeFile(p, JSON.stringify(payload, null, 2), "utf8");
}

function safeFileSlug(s) {
  return String(s).replace(/[^a-zA-Z0-9_-]+/g, "_");
}

// -----------------------------
// 6) CLI: tests + demo
// -----------------------------
async function runTests() {
  const rt = makeRuntime(CONTRACT_ημ_INGEST_V1, HANDLERS);
  for (const t of CONTRACT_ημ_INGEST_V1.tests) {
    if (typeof t.setup === "function") t.setup(rt.state);
    const d = rt.enforce(t.act);

    const okMatch = d.ok === t.expectOk;
    const whyMatch = t.expectWhy ? d.why === t.expectWhy : true;
    const modeMatch = t.expectMode ? d.mode === t.expectMode : true;

    if (!okMatch || !whyMatch || !modeMatch) {
      console.error("FAIL:", t.name, { got: d, expected: { ok: t.expectOk, why: t.expectWhy, mode: t.expectMode } });
      process.exitCode = 1;
    } else {
      console.log("PASS:", t.name);
    }
  }
}

async function runDemo() {
  const rt = makeRuntime(CONTRACT_ημ_INGEST_V1, HANDLERS);
  console.log("Demo: running pipeline. (Will throw until you implement vision.describe + embed.write)");
  await rt.runPipeline();
}

const argv = new Set(process.argv.slice(2));
if (argv.has("--test")) {
  await runTests();
} else if (argv.has("--demo")) {
  await runDemo();
} else if (import.meta.url === `file://${process.argv[1]}`) {
  console.log(`Usage:
  node ${process.argv[1]} --test
  node ${process.argv[1]} --demo`);
}
