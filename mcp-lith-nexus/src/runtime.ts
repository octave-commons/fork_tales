import path from "node:path"

export interface RuntimePaths {
  repoRoot: string
  pythonWorkdir: string
}

export interface HttpRuntimeConfig extends RuntimePaths {
  host: string
  port: number
  mcpPath: string
}

export function resolveRuntimePaths(): RuntimePaths {
  const repoRoot = path.resolve(process.env.LITH_NEXUS_REPO_ROOT ?? path.join(import.meta.dirname, "..", ".."))
  const pythonWorkdir = path.resolve(process.env.LITH_NEXUS_PYTHON_WORKDIR ?? path.join(repoRoot, "part64"))
  return { repoRoot, pythonWorkdir }
}

export function resolveHttpRuntimeConfig(): HttpRuntimeConfig {
  const { repoRoot, pythonWorkdir } = resolveRuntimePaths()
  const host = process.env.LITH_NEXUS_HTTP_HOST?.trim() || "127.0.0.1"
  const port = Number.parseInt(process.env.LITH_NEXUS_HTTP_PORT ?? "8794", 10) || 8794
  const rawPath = process.env.LITH_NEXUS_HTTP_PATH?.trim() || "/mcp"
  const mcpPath = rawPath.startsWith("/") ? rawPath : `/${rawPath}`
  return { repoRoot, pythonWorkdir, host, port, mcpPath }
}
