import test from "node:test"
import assert from "node:assert/strict"
import fs from "node:fs/promises"
import os from "node:os"
import path from "node:path"

import {
  PromptLispError,
  parseSExpr,
  printSExpr,
  parseFragmentForms,
  render,
  compileObservationFactsFromText,
  compilePromptDb,
} from "../prompt_lisp.mjs"

test("parse -> print -> parse roundtrip is stable", () => {
  const source = `
; sample
(fragment
  (meta (id "x/1") (kind command) (name "route") (enabled true))
  (provides (command "/route"))
  (template (join "hello" (nl) (if (= (var "A") "1") "yes" "no"))))
`
  const first = parseSExpr(source, { filePath: "sample.oprompt" })
  const printed = first.map((form) => printSExpr(form)).join("\n")
  const second = parseSExpr(printed, { filePath: "sample.oprompt" })
  const reprinted = second.map((form) => printSExpr(form)).join("\n")
  assert.equal(reprinted, printed)
})

test("render is deterministic with same env", async () => {
  const form = parseSExpr(`(template (join "X=" (var "X") (nl) (if (present? (var "X")) "y" "n")))`, { filePath: "render.sexp" })[0]
  const env = { X: "abc", ARGUMENTS: "a b", SEED: "s1", SESSION_ID: "sess", PART_ID: "p1" }
  const one = await render(form, { filePath: path.resolve("render.sexp"), env })
  const two = await render(form, { filePath: path.resolve("render.sexp"), env })
  assert.equal(one.text, two.text)
  assert.equal(one.witness.outputBytes, two.witness.outputBytes)
})

test("include cycle raises E_INCLUDE_CYCLE", async () => {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "prompt-lisp-cycle-"))
  const a = path.join(tmp, "a.sexp")
  const b = path.join(tmp, "b.sexp")
  await fs.writeFile(a, `(template (include (path "b.sexp") (as text)))`, "utf8")
  await fs.writeFile(b, `(template (include (path "a.sexp") (as text)))`, "utf8")

  const form = parseSExpr(await fs.readFile(a, "utf8"), { filePath: a })[0]
  await assert.rejects(
    () => render(form, { filePath: a, includeRoots: [tmp] }),
    (error) => error instanceof PromptLispError && error.code === "E_INCLUDE_CYCLE",
  )
})

test("observation.v2 compiles to claim/evidence facts", () => {
  const src = `
(observation.v2
  (id "obs:1")
  (vantage (agent eta-mu.world-daemon))
  (subject (who (kind self) (id eta-mu.world-daemon)))
  (channel (kind internal))
  (claims
    (claim
      (id c:1)
      (scope state)
      (key "ws.connected")
      (val "true")
      (evidence-refs tool.trace.1))))
`
  const facts = compileObservationFactsFromText(src, { filePath: "obs.sexp" })
  assert.ok(facts.some((fact) => fact[0] === "obs" && fact[1] === "obs:1"))
  assert.ok(facts.some((fact) => fact[0] === "claim" && fact[1] === "c:1"))
  assert.ok(facts.some((fact) => fact[0] === "ev-kind" && fact[2] === "internal"))
})

test("compilePromptDb emits commands and skills from enabled fragments", async () => {
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "prompt-lisp-compile-"))
  const fragmentRoot = path.join(tmp, "fragments")
  const outCommands = path.join(tmp, "commands")
  const outSkills = path.join(tmp, "skills")
  await fs.mkdir(fragmentRoot, { recursive: true })

  const commandFragment = `(fragment
  (meta (id "cmd/x@v1") (kind command) (name "route") (enabled true))
  (provides (command "/route"))
  (opencode.command (description "route command") (agent "plan"))
  (template (join "Call " (var "ARGUMENTS"))))`

  const skillFragment = `(fragment
  (meta (id "skill/s1@v1") (kind skill) (name "intent-protocol") (enabled true))
  (provides (skill "intent-protocol"))
  (opencode.skill
    (summary "Intent summary")
    (content "## Skill body")))`

  const disabledFragment = `(fragment
  (meta (id "cmd/y@v1") (kind command) (name "disabled") (enabled false))
  (provides (command "/disabled"))
  (template "ignore"))`

  const payloadCommandFragment = `(fragment
  (meta (id "cmd/payload@v1") (kind command) (name "payload-cmd") (enabled true))
  (provides (command "/payload-cmd"))
  (opencode.command (description "payload command") (agent "plan"))
  (payload (prompt (meta (name "payload-cmd")) (notes "hello"))))`

  await fs.writeFile(path.join(fragmentRoot, "01_cmd.oprompt"), commandFragment, "utf8")
  await fs.writeFile(path.join(fragmentRoot, "02_skill.oprompt"), skillFragment, "utf8")
  await fs.writeFile(path.join(fragmentRoot, "03_disabled.oprompt"), disabledFragment, "utf8")
  await fs.writeFile(path.join(fragmentRoot, "04_payload_cmd.oprompt"), payloadCommandFragment, "utf8")

  const result = await compilePromptDb({ fragmentRoot, outCommands, outSkills, includeRoots: [tmp] })
  assert.equal(result.generated.commands, 2)
  assert.equal(result.generated.skills, 1)

  const routeMd = await fs.readFile(path.join(outCommands, "route.md"), "utf8")
  const payloadMd = await fs.readFile(path.join(outCommands, "payload-cmd.md"), "utf8")
  const skillMd = await fs.readFile(path.join(outSkills, "intent-protocol", "SKILL.md"), "utf8")

  assert.match(routeMd, /description: route command/)
  assert.match(routeMd, /Call \$ARGUMENTS/)
  assert.match(payloadMd, /description: payload command/)
  assert.match(payloadMd, /\(payload \(prompt \(meta \(name "payload-cmd"\)\) \(notes "hello"\)\)\)/)
  assert.match(skillMd, /name: intent-protocol/)
  assert.match(skillMd, /## Skill body/)
})

test("fragment parser extracts required metadata", () => {
  const src = `(fragment
  (meta (id "x") (kind skill) (name "n") (enabled true))
  (provides (skill "n"))
  (content "ok"))`
  const forms = parseSExpr(src, { filePath: "frag.oprompt" })
  const fragment = parseFragmentForms(forms, "frag.oprompt")
  assert.equal(fragment.meta.id, "x")
  assert.equal(fragment.meta.kind, "skill")
  assert.equal(fragment.meta.name, "n")
  assert.equal(fragment.meta.enabled, true)
})
