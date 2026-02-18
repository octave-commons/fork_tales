import { lexicon, rules } from "./rules.mjs";

const SLOT_KEYS = ["owner", "deadline", "evidence", "options", "dod"];

function normalize(s) {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

function detectSlots(text) {
  const slots = Object.fromEntries(SLOT_KEYS.map((k) => [k, null]));
  const lines = text.split(/\r?\n/);

  for (const line of lines) {
    const m = line.match(/^\s*(Owner|Deadline|Evidence|Options|DoD)\s*:\s*(.+)\s*$/i);
    if (!m) continue;
    const key = m[1].toLowerCase();
    const val = m[2].trim();
    if (key === "dod") slots.dod = val;
    else slots[key] = val;
  }

  const has = {};
  for (const k of SLOT_KEYS) has[k] = !!slots[k];

  return { slots, has };
}

function containsAny(hay, phrases) {
  return phrases.some((p) => hay.includes(p));
}

function containsAll(hay, phrases) {
  return phrases.every((p) => hay.includes(p));
}

function detectObservation(textNorm) {
  const self = /\b(i|me|my|mine|i'm|i’ve|i've|i am)\b/.test(textNorm);
  const other = /\b(you|your|yours|they|them|their|he|she|him|her)\b/.test(textNorm);
  if (self && other) return "mixed";
  if (self) return "self";
  if (other) return "other";
  return "unknown";
}

function commitLevel(textNorm, has) {
  const commitmentVerbs = ["i will", "we will", "i'll", "we'll", "i commit", "we commit"];
  const impliedVerbs = ["we should", "i should", "let's", "we need to"];

  if (containsAny(textNorm, commitmentVerbs) && has.owner && (has.deadline || has.dod)) {
    return "explicit";
  }
  if (containsAny(textNorm, commitmentVerbs) || containsAny(textNorm, impliedVerbs)) {
    return "implied";
  }
  return "none";
}

function proposeRewrite(text, missing) {
  const needLines = [];
  for (const m of missing) {
    const label = m === "dod" ? "DoD" : m[0].toUpperCase() + m.slice(1);
    needLines.push(`${label}: TODO`);
  }
  if (!needLines.length) return null;
  return text + "\n\n" + needLines.join("\n");
}

export function analyzeUtterance({ id, text, source, ts, wantRewrite }) {
  const textNorm = normalize(text);

  const { slots, has } = detectSlots(text);
  const obs = detectObservation(textNorm);
  const commitment = commitLevel(textNorm, has);

  const hits = [];
  for (const [ftype, phrases] of Object.entries(lexicon)) {
    for (const p of phrases) {
      if (textNorm.includes(p)) hits.push({ ftype, phrase: p });
    }
  }

  const frames = new Set();
  const needs = new Set();
  const severities = [];
  const agencyDeltas = [];

  for (const r of rules) {
    const okAny = !r.when.any?.length || containsAny(textNorm, r.when.any);
    const okAll = !r.when.all?.length || containsAll(textNorm, r.when.all);

    let okNot = true;
    if (r.when.not?.missingSlots?.length) {
      okNot = r.when.not.missingSlots.every((k) => has[k]);
    }
    if (r.when.not?.hasDeadline === true) {
      okNot = okNot && !has.deadline;
    }
    if (r.when.not?.hasEvidence === true) {
      okNot = okNot && !has.evidence;
    }
    if (r.when.obs) {
      okNot = okNot && (obs === r.when.obs || obs === "mixed");
    }

    if (okAny && okAll && okNot) {
      frames.add(r.outcome.frame);
      (r.outcome.needs ?? []).forEach((n) => {
        needs.add(n);
      });
      if (typeof r.outcome.severity === "number") severities.push(r.outcome.severity);
      if (typeof r.outcome.agencyDelta === "number") agencyDeltas.push(r.outcome.agencyDelta);
    }
  }

  const sev = severities.reduce((a, b) => a + b, 0);
  const mu = agencyDeltas.reduce((a, b) => a + b, 0);

  const proofCount = SLOT_KEYS.filter((k) => has[k]).length;
  const etaGap = Math.max(0, sev - proofCount);

  const out = {
    id,
    ts,
    source,
    obs,
    commitment,
    slots,
    frames: [...frames],
    needs: [...needs],
    sev,
    mu,
    etaGap,
    hits
  };

  if (wantRewrite) {
    out.rewrite = proposeRewrite(text, out.needs);
  }
  return out;
}
