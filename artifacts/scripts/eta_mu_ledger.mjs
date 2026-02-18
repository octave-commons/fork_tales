#!/usr/bin/env node
import fs from "node:fs"

const CLAIM_CUE_RE = /\b(should|will|obvious|clearly|must|need to|going to|plan to|we should|we will|it's obvious|it is obvious)\b/iu
const COMMIT_RE = /\b[0-9a-f]{7,40}\b/giu
const URL_RE = /https?:\/\/[^\s)\]}>"']+/giu
const FILE_RE = /\b[./~]?[-\w]+(?:\/[-\w.]+)+\.[a-z0-9]{1,8}\b/giu
const PR_RE = /\b(?:PR|pr)\s*#\d+\b/gu

function uniq(values) {
  return [...new Set(values)]
}

export function detectArtifactRefs(utterance) {
  const refs = []
  for (const re of [URL_RE, FILE_RE, PR_RE, COMMIT_RE]) {
    re.lastIndex = 0
    for (const match of utterance.matchAll(re)) {
      refs.push(match[0])
    }
  }
  return uniq(refs)
}

export function analyzeUtterance(utterance, index, now = new Date()) {
  const text = utterance.trim()
  const artifactRefs = detectArtifactRefs(text)
  const etaClaim = CLAIM_CUE_RE.test(text)
  const muProof = artifactRefs.length > 0
  const classification = muProof ? "mu-proof" : etaClaim ? "eta-claim" : "neutral"
  return {
    ts: now.toISOString(),
    idx: index,
    utterance: text,
    classification,
    eta_claim: etaClaim,
    mu_proof: muProof,
    artifact_refs: artifactRefs,
  }
}

export function utterancesToJsonl(utterances, now = new Date()) {
  return utterances
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line, i) => JSON.stringify(analyzeUtterance(line, i + 1, now)))
    .join("\n")
}

function main() {
  const stdin = fs.readFileSync(0, "utf8")
  const argvUtterances = process.argv.slice(2)
  const lines = stdin.trim().length > 0 ? stdin.split(/\r?\n/u) : argvUtterances
  const output = utterancesToJsonl(lines)
  if (output.length > 0) {
    process.stdout.write(`${output}\n`)
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    main()
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error))
    process.exit(1)
  }
}
