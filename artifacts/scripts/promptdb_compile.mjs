#!/usr/bin/env node
import { compilePromptDb } from "./prompt_lisp.mjs"

async function main() {
  const result = await compilePromptDb({
    fragmentRoot: ".opencode/promptdb/fragments",
    outCommands: ".opencode/commands",
    outSkills: ".opencode/skills",
    includeRoots: [process.cwd(), ".opencode"],
  })
  console.log(JSON.stringify(result, null, 2))
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
