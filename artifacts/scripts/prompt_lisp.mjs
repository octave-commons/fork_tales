#!/usr/bin/env node
import fs from "node:fs/promises"
import path from "node:path"
import crypto from "node:crypto"

const DEFAULT_LIMITS = {
  includeDepth: 16,
  includeFiles: 256,
  outputBytes: 256 * 1024,
}

const SYMBOL_RE = /^[\p{L}\p{N}_+\-*/?.:#!==]+$/u

function makeLoc(file, line, column) {
  return { file, line, column }
}

export class PromptLispError extends Error {
  constructor(code, message, details = {}) {
    super(message)
    this.name = "PromptLispError"
    this.code = code
    this.details = details
  }
}

function parseError(code, message, loc) {
  throw new PromptLispError(code, message, { loc })
}

function node(type, value, loc) {
  if (type === "list") return { type, items: value, loc }
  return { type, value, loc }
}

function isNode(value, type) {
  return Boolean(value && value.type === type)
}

function isSymbol(value, expected) {
  return isNode(value, "symbol") && (expected === undefined || value.value === expected)
}

function asList(value, code, message) {
  if (!isNode(value, "list")) {
    throw new PromptLispError(code, message, { loc: value?.loc })
  }
  return value
}

function asString(value, code, message) {
  if (!isNode(value, "string")) {
    throw new PromptLispError(code, message, { loc: value?.loc })
  }
  return value.value
}

function asSymbol(value, code, message) {
  if (!isNode(value, "symbol")) {
    throw new PromptLispError(code, message, { loc: value?.loc })
  }
  return value.value
}

function boolFromNode(value) {
  if (isNode(value, "symbol")) {
    if (value.value === "true") return true
    if (value.value === "false") return false
  }
  if (isNode(value, "string")) {
    if (value.value === "true") return true
    if (value.value === "false") return false
  }
  return undefined
}

export function parseSExpr(source, options = {}) {
  const file = options.filePath ?? "<memory>"
  const normalizedSource = source.startsWith("\\\n") || source.startsWith("\\\r\n") ? source.replace(/^\\\r?\n/u, "") : source
  const chars = Array.from(normalizedSource)
  const len = chars.length
  let i = 0
  let line = 1
  let column = 1

  function currentLoc() {
    return makeLoc(file, line, column)
  }

  function peek(offset = 0) {
    return chars[i + offset]
  }

  function advance() {
    const ch = chars[i++]
    if (ch === "\n") {
      line += 1
      column = 1
    } else {
      column += 1
    }
    return ch
  }

  function skipWhitespaceAndComments() {
    while (i < len) {
      const ch = peek()
      if (ch === ";") {
        while (i < len && peek() !== "\n") {
          advance()
        }
        continue
      }
      if (ch === " " || ch === "\t" || ch === "\r" || ch === "\n") {
        advance()
        continue
      }
      break
    }
  }

  function parseString() {
    const loc = currentLoc()
    advance()
    let out = ""
    while (i < len) {
      const ch = advance()
      if (ch === "\\") {
        if (i >= len) parseError("E_PARSE_STRING_ESCAPE", "unterminated string escape", loc)
        const esc = advance()
        if (esc === "n") out += "\n"
        else if (esc === "t") out += "\t"
        else if (esc === "r") out += "\r"
        else if (esc === '"') out += '"'
        else if (esc === "\\") out += "\\"
        else out += esc
        continue
      }
      if (ch === '"') return node("string", out, loc)
      out += ch
    }
    parseError("E_PARSE_STRING_UNTERM", "unterminated string", loc)
  }

  function parseSymbolOrNumber() {
    const loc = currentLoc()
    let out = ""
    while (i < len) {
      const ch = peek()
      if (!ch || ch === "(" || ch === ")" || ch === ";" || /\s/u.test(ch)) break
      out += advance()
    }
    if (!out) parseError("E_PARSE_TOKEN", "unexpected token", loc)
    if (/^-?\d+$/.test(out)) return node("number", Number.parseInt(out, 10), loc)
    if (/^-?(?:\d+\.\d+|\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?$/.test(out) || /^-?\d+[eE][+-]?\d+$/.test(out)) {
      return node("number", Number.parseFloat(out), loc)
    }
    if (!SYMBOL_RE.test(out)) {
      parseError("E_PARSE_SYMBOL", `invalid symbol '${out}'`, loc)
    }
    return node("symbol", out, loc)
  }

  function parseForm() {
    skipWhitespaceAndComments()
    if (i >= len) parseError("E_PARSE_EOF", "unexpected EOF", currentLoc())
    const ch = peek()
    if (ch === "(") {
      const loc = currentLoc()
      advance()
      const items = []
      while (true) {
        skipWhitespaceAndComments()
        if (i >= len) parseError("E_PARSE_LIST_UNTERM", "unterminated list", loc)
        if (peek() === ")") {
          advance()
          return node("list", items, loc)
        }
        items.push(parseForm())
      }
    }
    if (ch === ")") {
      parseError("E_PARSE_UNEXPECTED_CLOSE", "unexpected ')'", currentLoc())
    }
    if (ch === '"') return parseString()
    return parseSymbolOrNumber()
  }

  const forms = []
  while (true) {
    skipWhitespaceAndComments()
    if (i >= len) break
    forms.push(parseForm())
  }
  return forms
}

function escapeString(value) {
  return value.replaceAll("\\", "\\\\").replaceAll("\n", "\\n").replaceAll("\t", "\\t").replaceAll("\r", "\\r").replaceAll('"', '\\"')
}

export function printSExpr(form) {
  if (!form || typeof form !== "object") {
    throw new PromptLispError("E_PRINT_NODE", "invalid AST node")
  }
  if (form.type === "string") return `"${escapeString(form.value)}"`
  if (form.type === "number") return Number.isFinite(form.value) ? String(form.value) : "0"
  if (form.type === "symbol") return form.value
  if (form.type === "list") return `(${form.items.map((item) => printSExpr(item)).join(" ")})`
  throw new PromptLispError("E_PRINT_NODE", `unknown AST node type '${String(form.type)}'`)
}

function listField(listNode, key) {
  for (const item of listNode.items) {
    if (!isNode(item, "list") || item.items.length < 2) continue
    if (isSymbol(item.items[0], key)) return item
  }
  return null
}

function parseMeta(fragmentNode) {
  const metaNode = listField(fragmentNode, "meta")
  if (!metaNode) throw new PromptLispError("E_FRAGMENT_META", "fragment missing (meta ...) section", { loc: fragmentNode.loc })
  const idNode = listField(metaNode, "id")
  const kindNode = listField(metaNode, "kind")
  const nameNode = listField(metaNode, "name")
  const enabledNode = listField(metaNode, "enabled")
  if (!idNode || !kindNode || !nameNode) {
    throw new PromptLispError("E_FRAGMENT_META_REQUIRED", "fragment meta requires id/kind/name", { loc: metaNode.loc })
  }
  const id = asString(idNode.items[1], "E_FRAGMENT_META_ID", "meta id must be a string")
  const kind = asSymbol(kindNode.items[1], "E_FRAGMENT_META_KIND", "meta kind must be a symbol")
  const name = asString(nameNode.items[1], "E_FRAGMENT_META_NAME", "meta name must be a string")
  const enabled = enabledNode ? (boolFromNode(enabledNode.items[1]) ?? true) : true
  return { id, kind, name, enabled }
}

function parseProvides(fragmentNode) {
  const providesNode = listField(fragmentNode, "provides")
  const provides = []
  if (!providesNode) return provides
  for (const item of providesNode.items.slice(1)) {
    if (!isNode(item, "list") || item.items.length < 2) continue
    const type = asSymbol(item.items[0], "E_FRAGMENT_PROVIDES", "provides key must be symbol")
    const value = asString(item.items[1], "E_FRAGMENT_PROVIDES", "provides value must be string")
    provides.push({ type, value })
  }
  return provides
}

export function parseFragmentForms(forms, filePath = "<memory>") {
  if (forms.length !== 1) {
    throw new PromptLispError("E_FRAGMENT_FORM_COUNT", "fragment file must contain exactly one top-level form", { filePath })
  }
  const form = asList(forms[0], "E_FRAGMENT_FORM", "top-level form must be a list")
  if (!isSymbol(form.items[0], "fragment")) {
    throw new PromptLispError("E_FRAGMENT_HEAD", "top-level form must start with 'fragment'", { loc: form.loc, filePath })
  }
  const meta = parseMeta(form)
  const provides = parseProvides(form)
  const body = form.items.slice(1).filter((item) => !isNode(item, "list") || !isSymbol(item.items[0], "meta") && !isSymbol(item.items[0], "provides"))
  return {
    filePath,
    ast: form,
    meta,
    provides,
    body,
  }
}

function toText(value) {
  if (value === null || value === undefined) return ""
  if (typeof value === "string") return value
  if (typeof value === "number") return String(value)
  if (typeof value === "boolean") return value ? "true" : "false"
  if (Array.isArray(value)) return value.map((entry) => toText(entry)).join("")
  if (isNode(value)) return printSExpr(value)
  return String(value)
}

function setEnvValue(scope, name, value) {
  scope.set(name, toText(value))
}

function getEnvValue(scope, name) {
  if (scope.has(name)) return scope.get(name)
  return ""
}

function parseEnvNode(envNode) {
  const out = new Map()
  if (!envNode || !isNode(envNode, "list") || !isSymbol(envNode.items[0], "env")) return out
  for (const item of envNode.items.slice(1)) {
    if (!isNode(item, "list") || item.items.length !== 2) continue
    const key = asString(item.items[0], "E_RENDER_ENV_KEY", "env key must be string")
    const value = item.items[1]
    if (isNode(value, "string") || isNode(value, "number") || isNode(value, "symbol")) {
      out.set(key, String(value.value))
    }
  }
  return out
}

function normalizeIncludeRoots(includeRoots) {
  const roots = includeRoots && includeRoots.length ? includeRoots : [process.cwd(), path.resolve(".opencode")]
  return [...new Set(roots.map((root) => path.resolve(root)))]
}

function isWithinRoot(targetPath, rootPath) {
  const relative = path.relative(rootPath, targetPath)
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative))
}

function assertWithinRoots(resolvedPath, roots) {
  if (!roots.some((root) => isWithinRoot(resolvedPath, root))) {
    throw new PromptLispError("E_INCLUDE_DENIED", `include path '${resolvedPath}' is outside allowlist roots`, { resolvedPath, roots })
  }
}

function evalPredicate(expr, state) {
  if (isNode(expr, "list") && isSymbol(expr.items[0], "=")) {
    if (expr.items.length !== 3) throw new PromptLispError("E_TEMPLATE_ARITY", "(= ...) expects exactly 2 args", { loc: expr.loc })
    const left = toText(evalExpr(expr.items[1], state))
    const right = toText(evalExpr(expr.items[2], state))
    return left === right
  }
  if (isNode(expr, "list") && isSymbol(expr.items[0], "present?")) {
    if (expr.items.length !== 2) throw new PromptLispError("E_TEMPLATE_ARITY", "(present? ...) expects exactly 1 arg", { loc: expr.loc })
    const value = toText(evalExpr(expr.items[1], state))
    return value.trim().length > 0
  }
  if (isNode(expr, "symbol")) {
    const value = getEnvValue(state.scope, expr.value)
    return value.trim().length > 0
  }
  return Boolean(evalExpr(expr, state))
}

function escapeValue(mode, value) {
  const text = toText(value)
  if (mode === "none") return text
  if (mode === "json") {
    const raw = JSON.stringify(text)
    return raw.slice(1, -1)
  }
  if (mode === "shell") {
    return `'${text.replaceAll("'", `'\\''`)}'`
  }
  if (mode === "markdown") {
    return text.replace(/[\\`*_{}\[\]()#+\-.!|>]/g, "\\$&")
  }
  throw new PromptLispError("E_ESCAPE_MODE", `unsupported escape mode '${mode}'`)
}

async function includeRead(expr, state) {
  let includePath = null
  let asMode = "text"
  for (const arg of expr.items.slice(1)) {
    if (!isNode(arg, "list") || arg.items.length < 2 || !isNode(arg.items[0], "symbol")) continue
    if (arg.items[0].value === "path") includePath = asString(arg.items[1], "E_INCLUDE_PATH", "include path must be a string")
    if (arg.items[0].value === "as") asMode = asSymbol(arg.items[1], "E_INCLUDE_AS", "include as mode must be symbol")
  }
  if (!includePath) {
    throw new PromptLispError("E_INCLUDE_PATH", "(include ...) missing (path \"...\")", { loc: expr.loc })
  }
  if (!["raw", "text", "sexp"].includes(asMode)) {
    throw new PromptLispError("E_INCLUDE_AS", "include mode must be one of raw|text|sexp", { loc: expr.loc })
  }
  if (state.depth >= state.limits.includeDepth) {
    throw new PromptLispError("E_LIMIT_DEPTH", `include depth exceeded (${state.limits.includeDepth})`, { loc: expr.loc })
  }

  const parentDir = path.dirname(state.filePath)
  const absolute = path.isAbsolute(includePath) ? includePath : path.resolve(parentDir, includePath)
  assertWithinRoots(absolute, state.includeRoots)

  if (state.includeStack.has(absolute)) {
    throw new PromptLispError("E_INCLUDE_CYCLE", `include cycle detected at '${absolute}'`, { includePath: absolute })
  }
  if (state.includeCount + 1 > state.limits.includeFiles) {
    throw new PromptLispError("E_LIMIT_FILES", `include file limit exceeded (${state.limits.includeFiles})`, { includePath: absolute })
  }

  const buf = await fs.readFile(absolute)
  const text = buf.toString("utf8")
  state.includeCount += 1
  if (asMode === "raw") return text
  if (asMode === "text") return text
  const parsed = parseSExpr(text, { filePath: absolute })
  return parsed
}

function pushOutput(state, chunk) {
  state.output += chunk
  if (Buffer.byteLength(state.output, "utf8") > state.limits.outputBytes) {
    throw new PromptLispError("E_LIMIT_OUTPUT", `rendered output exceeded ${state.limits.outputBytes} bytes`, {})
  }
}

function forbidUnknownForm(expr) {
  const head = isNode(expr.items[0], "symbol") ? expr.items[0].value : "<non-symbol>"
  throw new PromptLispError("E_FORBIDDEN_FORM", `forbidden form '${head}' in template evaluation`, { loc: expr.loc })
}

function evalExpr(expr, state) {
  if (isNode(expr, "string") || isNode(expr, "number")) return expr.value
  if (isNode(expr, "symbol")) {
    if (state.scope.has(expr.value)) return getEnvValue(state.scope, expr.value)
    return expr.value
  }
  if (!isNode(expr, "list") || expr.items.length === 0) return ""
  if (!isNode(expr.items[0], "symbol")) forbidUnknownForm(expr)
  const head = expr.items[0].value

  if (head === "join") {
    return expr.items.slice(1).map((part) => toText(evalExpr(part, state))).join("")
  }
  if (head === "nl") {
    return "\n"
  }
  if (head === "if") {
    if (expr.items.length !== 4) throw new PromptLispError("E_TEMPLATE_ARITY", "(if ...) expects predicate, then, else", { loc: expr.loc })
    return evalPredicate(expr.items[1], state) ? evalExpr(expr.items[2], state) : evalExpr(expr.items[3], state)
  }
  if (head === "=") return evalPredicate(expr, state) ? "true" : "false"
  if (head === "present?") return evalPredicate(expr, state) ? "true" : "false"
  if (head === "var") {
    if (expr.items.length !== 2) throw new PromptLispError("E_TEMPLATE_ARITY", "(var ...) expects 1 arg", { loc: expr.loc })
    const key = asString(expr.items[1], "E_TEMPLATE_VAR", "(var ...) requires string name")
    return getEnvValue(state.scope, key)
  }
  if (head === "set") {
    if (expr.items.length !== 3) throw new PromptLispError("E_TEMPLATE_ARITY", "(set ...) expects 2 args", { loc: expr.loc })
    const key = asString(expr.items[1], "E_TEMPLATE_SET", "(set ...) requires string name")
    const value = evalExpr(expr.items[2], state)
    setEnvValue(state.scope, key, value)
    return ""
  }
  if (head === "escape") {
    if (expr.items.length !== 3) throw new PromptLispError("E_TEMPLATE_ARITY", "(escape ...) expects mode + expr", { loc: expr.loc })
    const mode = asSymbol(expr.items[1], "E_ESCAPE_MODE", "escape mode must be symbol")
    return escapeValue(mode, evalExpr(expr.items[2], state))
  }
  forbidUnknownForm(expr)
}

async function evalExprAsync(expr, state) {
  if (isNode(expr, "list") && expr.items.length > 0 && isSymbol(expr.items[0], "include")) {
    const includePathNode = listField(expr, "path")
    let includePath = null
    if (includePathNode?.items?.[1]) {
      includePath = asString(includePathNode.items[1], "E_INCLUDE_PATH", "include path must be string")
    }
    const parentDir = path.dirname(state.filePath)
    const absolute = path.isAbsolute(includePath ?? "") ? includePath : path.resolve(parentDir, includePath ?? "")
    const nextState = {
      ...state,
      depth: state.depth + 1,
      includeStack: new Set([...state.includeStack, absolute]),
    }
    const value = await includeRead(expr, nextState)
    return value
  }

  if (isNode(expr, "list") && expr.items.length > 0 && isSymbol(expr.items[0], "join")) {
    const parts = []
    for (const item of expr.items.slice(1)) {
      parts.push(toText(await evalExprAsync(item, state)))
    }
    return parts.join("")
  }
  if (isNode(expr, "list") && expr.items.length > 0 && isSymbol(expr.items[0], "if")) {
    if (expr.items.length !== 4) throw new PromptLispError("E_TEMPLATE_ARITY", "(if ...) expects predicate, then, else", { loc: expr.loc })
    const pass = evalPredicate(expr.items[1], state)
    return evalExprAsync(pass ? expr.items[2] : expr.items[3], state)
  }
  if (isNode(expr, "list") && expr.items.length > 0 && isSymbol(expr.items[0], "set")) {
    const key = asString(expr.items[1], "E_TEMPLATE_SET", "(set ...) requires string name")
    const value = await evalExprAsync(expr.items[2], state)
    setEnvValue(state.scope, key, value)
    return ""
  }
  if (isNode(expr, "list") && expr.items.length > 0 && isSymbol(expr.items[0], "escape")) {
    const mode = asSymbol(expr.items[1], "E_ESCAPE_MODE", "escape mode must be symbol")
    const value = await evalExprAsync(expr.items[2], state)
    return escapeValue(mode, value)
  }
  return evalExpr(expr, state)
}

function resolveRenderTarget(expr) {
  if (!isNode(expr, "list") || expr.items.length === 0) {
    throw new PromptLispError("E_RENDER_EXPR", "render entry must be a list")
  }
  if (isSymbol(expr.items[0], "render")) {
    if (expr.items.length < 2) throw new PromptLispError("E_RENDER_EXPR", "(render ...) missing expression", { loc: expr.loc })
    const envNode = expr.items.find((item) => isNode(item, "list") && isSymbol(item.items[0], "env"))
    return { target: expr.items[1], envNode }
  }
  return { target: expr, envNode: null }
}

export async function render(entryExpr, options = {}) {
  const limits = { ...DEFAULT_LIMITS, ...(options.limits ?? {}) }
  const includeRoots = normalizeIncludeRoots(options.includeRoots)
  const baseEnv = new Map(Object.entries(options.env ?? {}))
  const { target, envNode } = resolveRenderTarget(entryExpr)
  for (const [k, v] of parseEnvNode(envNode)) {
    baseEnv.set(k, v)
  }

  const runtimeEnv = {
    ARGUMENTS: "",
    ARGS: "",
    SEED: "",
    SESSION_ID: "",
    PART_ID: "",
    ...Object.fromEntries(baseEnv.entries()),
  }
  const scope = new Map(Object.entries(runtimeEnv))

  let templateNode = target
  if (!isNode(templateNode, "list") || !isSymbol(templateNode.items[0], "template")) {
    throw new PromptLispError("E_RENDER_TEMPLATE", "render target must be (template ...)", { loc: templateNode.loc })
  }

  const state = {
    scope,
    includeRoots,
    filePath: options.filePath ?? path.resolve("<memory>"),
    includeStack: new Set(),
    includeCount: 0,
    depth: 0,
    limits,
    output: "",
  }

  for (const part of templateNode.items.slice(1)) {
    const rendered = await evalExprAsync(part, state)
    pushOutput(state, toText(rendered))
  }

  const witness = {
    includeCount: state.includeCount,
    outputBytes: Buffer.byteLength(state.output, "utf8"),
    limits,
  }

  return {
    text: state.output,
    meta: {
      filePath: options.filePath ?? "<memory>",
      includeRoots,
    },
    witness,
  }
}

function extractBodyNode(fragment, key) {
  for (const item of fragment.body) {
    if (isNode(item, "list") && isSymbol(item.items[0], key)) return item
  }
  return null
}

function extractCommandMetadata(fragment) {
  const commandNode = extractBodyNode(fragment, "opencode.command")
  const description = commandNode ? asString(listField(commandNode, "description")?.items?.[1] ?? node("string", "PromptDB command", commandNode.loc), "E_COMMAND_DESCRIPTION", "command description must be string") : "PromptDB command"
  const agentNode = commandNode ? listField(commandNode, "agent") : null
  const agent = agentNode?.items?.[1] ? asString(agentNode.items[1], "E_COMMAND_AGENT", "command agent must be string") : undefined
  return { description, agent }
}

function extractCommandName(fragment) {
  const provideCommand = fragment.provides.find((item) => item.type === "command")?.value
  if (provideCommand) return provideCommand.replace(/^\//, "")
  return fragment.meta.name.replace(/^\//, "")
}

function extractSkillContent(fragment) {
  const skillNode = extractBodyNode(fragment, "opencode.skill")
  if (skillNode) {
    const summaryNode = listField(skillNode, "summary")
    const description = summaryNode?.items?.[1] ? asString(summaryNode.items[1], "E_SKILL_SUMMARY", "skill summary must be string") : "PromptDB skill generated from fragments"
    const contentNode = listField(skillNode, "content")
    const content = contentNode?.items?.[1] ? asString(contentNode.items[1], "E_SKILL_CONTENT", "skill content must be string") : ""
    return { description, content }
  }
  const contentNode = extractBodyNode(fragment, "content")
  if (contentNode?.items?.[1]) {
    return {
      description: "PromptDB skill generated from fragments",
      content: asString(contentNode.items[1], "E_SKILL_CONTENT", "skill content must be string"),
    }
  }
  return { description: "PromptDB skill generated from fragments", content: "" }
}

function formatFrontmatter(data) {
  const lines = ["---"]
  for (const [key, value] of Object.entries(data)) {
    if (value === undefined || value === null || value === "") continue
    const safe = String(value).includes(":") ? JSON.stringify(String(value)) : String(value)
    lines.push(`${key}: ${safe}`)
  }
  lines.push("---")
  return `${lines.join("\n")}\n`
}

export async function compilePromptDb(options = {}) {
  const fragmentRoot = path.resolve(options.fragmentRoot ?? ".opencode/promptdb/fragments")
  const outCommands = path.resolve(options.outCommands ?? ".opencode/commands")
  const outSkills = path.resolve(options.outSkills ?? ".opencode/skills")
  const includeRoots = normalizeIncludeRoots(options.includeRoots)

  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true })
    const files = []
    for (const entry of entries) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) files.push(...(await walk(full)))
      else if (/\.(oprompt|sexp|lisp)$/i.test(entry.name)) files.push(full)
    }
    return files
  }

  const files = (await walk(fragmentRoot)).sort((a, b) => a.localeCompare(b))
  const fragments = []
  for (const filePath of files) {
    const source = await fs.readFile(filePath, "utf8")
    const forms = parseSExpr(source, { filePath })
    const fragment = parseFragmentForms(forms, filePath)
    if (!fragment.meta.enabled) continue
    fragments.push(fragment)
  }

  await fs.mkdir(outCommands, { recursive: true })
  await fs.mkdir(outSkills, { recursive: true })

  let generatedCommands = 0
  let generatedSkills = 0

  for (const fragment of fragments) {
    const templateNode = extractBodyNode(fragment, "template")
    const payloadNode = extractBodyNode(fragment, "payload")

    if (fragment.meta.kind === "command" || fragment.provides.some((item) => item.type === "command")) {
      const { description, agent } = extractCommandMetadata(fragment)
      const name = extractCommandName(fragment)
      let body = ""
      if (templateNode) {
        const rendered = await render(templateNode, {
          filePath: fragment.filePath,
          includeRoots,
          env: {
            ARGUMENTS: "$ARGUMENTS",
            ARGS: "$ARGS",
            SEED: "$SEED",
            SESSION_ID: "$SESSION_ID",
            PART_ID: "$PART_ID",
          },
        })
        body = rendered.text.trim()
      } else if (payloadNode) {
        body = printSExpr(payloadNode).trim()
      }
      const frontmatter = formatFrontmatter({ description, agent })
      await fs.writeFile(path.join(outCommands, `${name}.md`), `${frontmatter}${body}\n`, "utf8")
      generatedCommands += 1
    }

    if (fragment.meta.kind === "skill") {
      const skillName = fragment.meta.name
      const { description, content } = extractSkillContent(fragment)
      const frontmatter = formatFrontmatter({
        name: skillName,
        description,
        compatibility: "opencode",
      })
      const skillDir = path.join(outSkills, skillName)
      await fs.mkdir(skillDir, { recursive: true })
      await fs.writeFile(path.join(skillDir, "SKILL.md"), `${frontmatter}\n${content.trim()}\n`, "utf8")
      generatedSkills += 1
    }
  }

  return {
    ok: true,
    scanned: files.length,
    generated: {
      commands: generatedCommands,
      skills: generatedSkills,
    },
  }
}

function stableObservationId(obsNode) {
  const idField = listField(obsNode, "id")
  if (idField?.items?.[1]) {
    if (isNode(idField.items[1], "string")) return idField.items[1].value
    if (isNode(idField.items[1], "symbol")) return idField.items[1].value
  }
  const hash = crypto.createHash("sha256").update(printSExpr(obsNode), "utf8").digest("hex")
  return `obs:${hash.slice(0, 16)}`
}

function findObservationForms(nodeValue, out = []) {
  if (!isNode(nodeValue, "list")) return out
  if (isSymbol(nodeValue.items[0], "observation.v2")) out.push(nodeValue)
  for (const child of nodeValue.items) {
    findObservationForms(child, out)
  }
  return out
}

function normalizeAtomValue(valueNode) {
  if (isNode(valueNode, "string")) return valueNode.value
  if (isNode(valueNode, "symbol")) return valueNode.value
  if (isNode(valueNode, "number")) return String(valueNode.value)
  return ""
}

export function observationToFacts(observationNode) {
  const obs = asList(observationNode, "E_OBS_FORM", "observation must be a list")
  if (!isSymbol(obs.items[0], "observation.v2")) {
    throw new PromptLispError("E_OBS_FORM", "observation form must start with observation.v2", { loc: obs.loc })
  }
  const obsId = stableObservationId(obs)
  const facts = []
  facts.push(["obs", obsId])

  const vantageNode = listField(obs, "vantage")
  const subjectNode = listField(obs, "subject")
  const channelNode = listField(obs, "channel")
  const claimsNode = listField(obs, "claims")

  let vantageAgent = ""
  if (vantageNode) {
    const agentNode = listField(vantageNode, "agent")
    if (agentNode?.items?.[1]) vantageAgent = normalizeAtomValue(agentNode.items[1])
  }
  if (vantageAgent) facts.push(["vantage", obsId, vantageAgent])

  let subjectId = ""
  let subjectKind = ""
  if (subjectNode) {
    const whoNode = listField(subjectNode, "who")
    if (whoNode) {
      const kindNode = listField(whoNode, "kind")
      const idNode = listField(whoNode, "id")
      if (kindNode?.items?.[1]) subjectKind = normalizeAtomValue(kindNode.items[1])
      if (idNode?.items?.[1]) subjectId = normalizeAtomValue(idNode.items[1])
    }
  }
  if (subjectId) facts.push(["subject", obsId, subjectId])
  if (subjectKind) facts.push(["subject-kind", obsId, subjectKind])

  let channelKind = ""
  if (channelNode) {
    const kindNode = listField(channelNode, "kind")
    if (kindNode?.items?.[1]) channelKind = normalizeAtomValue(kindNode.items[1])
  }
  if (channelKind) facts.push(["channel-kind", obsId, channelKind])

  let claimIndex = 0
  if (claimsNode) {
    for (const child of claimsNode.items.slice(1)) {
      if (!isNode(child, "list") || !isSymbol(child.items[0], "claim")) continue
      claimIndex += 1
      const cidField = listField(child, "id")
      const claimId = cidField?.items?.[1] ? normalizeAtomValue(cidField.items[1]) : `c:${obsId}:${claimIndex}`
      facts.push(["claim", claimId])
      facts.push(["in-obs", claimId, obsId])

      const scopeNode = listField(child, "scope")
      const keyNode = listField(child, "key")
      const valNode = listField(child, "val")
      const textNode = listField(child, "text")
      if (scopeNode?.items?.[1]) facts.push(["scope", claimId, normalizeAtomValue(scopeNode.items[1])])
      if (keyNode?.items?.[1]) facts.push(["key", claimId, normalizeAtomValue(keyNode.items[1])])
      if (valNode?.items?.[1]) facts.push(["val", claimId, normalizeAtomValue(valNode.items[1])])
      else if (textNode?.items?.[1]) facts.push(["val", claimId, normalizeAtomValue(textNode.items[1])])

      const evidenceRefsNode = listField(child, "evidence-refs")
      if (evidenceRefsNode) {
        let evIndex = 0
        for (const evRef of evidenceRefsNode.items.slice(1)) {
          evIndex += 1
          const evId = `ev:${claimId}:${evIndex}`
          facts.push(["ev", evId])
          facts.push(["ev-for", evId, claimId])
          const token = normalizeAtomValue(evRef)
          const kind = /tool|trace|telemetry|state|prompt/u.test(token) ? "internal" : "external"
          facts.push(["ev-kind", evId, kind])
        }
      }
    }
  }
  return facts
}

export function compileObservationFactsFromText(source, options = {}) {
  const forms = parseSExpr(source, { filePath: options.filePath ?? "<memory>" })
  const facts = []
  for (const form of forms) {
    for (const observation of findObservationForms(form)) {
      facts.push(...observationToFacts(observation))
    }
  }
  return facts
}
