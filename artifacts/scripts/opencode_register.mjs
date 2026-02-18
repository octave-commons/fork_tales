\
#!/usr/bin/env node
import fs from "node:fs"
import path from "node:path"
import crypto from "node:crypto"

const file = process.argv[2]
if (!file) throw new Error("usage: node scripts/opencode_register.mjs <artifact.sexp>")

const src = fs.readFileSync(file, "utf8")
const sha = crypto.createHash("sha256").update(src, "utf8").digest("hex")

const idMatch = src.match(/\(id\s+"([^"]+)"\)/)
const typeMatch = src.match(/\(kind\s+([a-z-]+)\)/)
const nameMatch = src.match(/\(name\s+"([^"]+)"\)/)

if (!typeMatch || !nameMatch) throw new Error("missing (kind ...) or (name ...)")

const type = typeMatch[1]
const name = nameMatch[1]
const id = idMatch?.[1] ?? `${type}:${name}@sha256:${sha}`

const dir = path.resolve(".opencode/promptdb/fragments")
fs.mkdirSync(dir, { recursive: true })

const out = path.join(dir, `${name.replace(/[^\w./-]/g, "_")}@${sha.slice(0, 16)}.oprompt`)
fs.writeFileSync(out, src.replace(/\(id\s+"[^"]+"\)/, `(id "${id}")`), "utf8")

console.log(id)
