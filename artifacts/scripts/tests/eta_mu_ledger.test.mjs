import test from "node:test"
import assert from "node:assert/strict"
import path from "node:path"
import { spawnSync } from "node:child_process"

import { analyzeUtterance, detectArtifactRefs, utterancesToJsonl } from "../eta_mu_ledger.mjs"

test("classifies eta claims without artifacts", () => {
  const row = analyzeUtterance("we should ship this soon", 1, new Date("2026-02-15T00:00:00.000Z"))
  assert.equal(row.classification, "eta-claim")
  assert.equal(row.eta_claim, true)
  assert.equal(row.mu_proof, false)
  assert.deepEqual(row.artifact_refs, [])
})

test("promotes to mu-proof when artifact refs exist", () => {
  const refs = detectArtifactRefs("done in artifacts/scripts/prompt_lisp.mjs commit abcdef1 PR #42")
  assert.deepEqual(refs, ["artifacts/scripts/prompt_lisp.mjs", "PR #42", "abcdef1"])

  const row = analyzeUtterance("we will deliver in artifacts/scripts/prompt_lisp.mjs", 2, new Date("2026-02-15T00:00:00.000Z"))
  assert.equal(row.classification, "mu-proof")
  assert.equal(row.eta_claim, true)
  assert.equal(row.mu_proof, true)
})

test("renders one JSONL line per non-empty utterance", () => {
  const jsonl = utterancesToJsonl(["first line", "", "second line"], new Date("2026-02-15T00:00:00.000Z"))
  const lines = jsonl.split("\n")
  assert.equal(lines.length, 2)
  assert.equal(JSON.parse(lines[0]).idx, 1)
  assert.equal(JSON.parse(lines[1]).idx, 2)
})

test("cli reads stdin and emits jsonl", () => {
  const script = path.resolve("artifacts/scripts/eta_mu_ledger.mjs")
  const run = spawnSync(process.execPath, [script], {
    input: "we should do this\nproof in artifacts/scripts/prompt_lisp.mjs\n",
    encoding: "utf8",
  })

  assert.equal(run.status, 0)
  const lines = run.stdout.trim().split("\n")
  assert.equal(lines.length, 2)

  const first = JSON.parse(lines[0])
  const second = JSON.parse(lines[1])
  assert.equal(first.classification, "eta-claim")
  assert.equal(second.classification, "mu-proof")
})
