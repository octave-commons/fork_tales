const path = require("node:path");

const resolveNvidiaVisibleDevices = () => {
  const raw = String(process.env.NVIDIA_VISIBLE_DEVICES || "all").trim();
  const probe = raw.toLowerCase();
  if (!raw || probe === "void" || probe === "none" || probe === "no-dev-files") {
    return "all";
  }
  return raw;
};

const resolveOrtGpuCapiDir = () =>
  String(process.env.CDB_ORT_GPU_CAPI_DIR || "/opt/ort-gpu/onnxruntime/capi");

const resolveOrtGpuIncludeDir = () =>
  String(
    process.env.CDB_ORT_GPU_INCLUDE_DIR ||
      "/app/onnxruntime-linux-x64-1.22.0/include",
  );

const resolveLithNexusScript = () =>
  String(
    process.env.LITH_NEXUS_SCRIPT ||
      path.resolve(__dirname, "..", "mcp-lith-nexus", "dist", "http.js"),
  );

const resolveLithNexusRepoRoot = () =>
  String(process.env.LITH_NEXUS_REPO_ROOT || path.resolve(__dirname, ".."));

const resolveLithNexusPythonWorkdir = () =>
  String(process.env.LITH_NEXUS_PYTHON_WORKDIR || __dirname);

module.exports = {
  apps: [
    {
      name: "eta-mu-world",
      cwd: __dirname,
      script: "code/world_web.py",
      interpreter: "python3",
      args: "--host 0.0.0.0 --port 8787 --part-root /app --vault-root /vault",
      env: {
        PYTHONUNBUFFERED: "1",
        NVIDIA_VISIBLE_DEVICES: resolveNvidiaVisibleDevices(),
        CDB_EMBED_GPU_VISIBLE_DEVICES:
          process.env.CDB_EMBED_GPU_VISIBLE_DEVICES || resolveNvidiaVisibleDevices(),
        CDB_EMBED_GPU_CONTROLLER_PROFILE:
          process.env.CDB_EMBED_GPU_CONTROLLER_PROFILE || "energy",
        CDB_ORT_GPU_CAPI_DIR: resolveOrtGpuCapiDir(),
        CDB_ORT_GPU_INCLUDE_DIR: resolveOrtGpuIncludeDir(),
        THREAT_RADAR_LLM_ENABLED:
          process.env.THREAT_RADAR_LLM_ENABLED || "1",
        THREAT_RADAR_LLM_MODEL:
          process.env.THREAT_RADAR_LLM_MODEL || "qwen3-vl:4b-instruct",
        THREAT_RADAR_LLM_TIMEOUT_SEC:
          process.env.THREAT_RADAR_LLM_TIMEOUT_SEC || "3",
        THREAT_RADAR_LLM_MAX_ITEMS:
          process.env.THREAT_RADAR_LLM_MAX_ITEMS || "6",
        THREAT_RADAR_LLM_MAX_TOKENS:
          process.env.THREAT_RADAR_LLM_MAX_TOKENS || "768",
        THREAT_RADAR_CLASSIFIER_ENABLED:
          process.env.THREAT_RADAR_CLASSIFIER_ENABLED || "1",
        THREAT_RADAR_CLASSIFIER_VERSION:
          process.env.THREAT_RADAR_CLASSIFIER_VERSION || "github_linear_v1",
        TEXT_GENERATION_BACKEND:
          process.env.TEXT_GENERATION_BACKEND || "vllm",
        TEXT_GENERATION_BASE_URL:
          process.env.TEXT_GENERATION_BASE_URL || "http://127.0.0.1:8789/v1",
        TEXT_GENERATION_BEARER_TOKEN:
          process.env.TEXT_GENERATION_BEARER_TOKEN || "change-me-open-hax-proxy-token",
        TEXT_GENERATION_MODEL:
          process.env.TEXT_GENERATION_MODEL || "qwen3-vl:4b-instruct",
        TEXT_GENERATION_TIMEOUT_SEC:
          process.env.TEXT_GENERATION_TIMEOUT_SEC || "60",
        TEXT_GENERATION_AUTH_HEADER:
          process.env.TEXT_GENERATION_AUTH_HEADER || "",
        TEXT_GENERATION_API_KEY:
          process.env.TEXT_GENERATION_API_KEY || "",
        TEXT_GENERATION_API_KEY_HEADER:
          process.env.TEXT_GENERATION_API_KEY_HEADER || "X-API-Key",
        RUNTIME_CATALOG_SUBPROCESS_ENABLED:
          process.env.RUNTIME_CATALOG_SUBPROCESS_ENABLED || "1",
        RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS:
          process.env.RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS || "45",
        RUNTIME_CATALOG_CACHE_SECONDS:
          process.env.RUNTIME_CATALOG_CACHE_SECONDS || "25",
        RUNTIME_CATALOG_HTTP_CACHE_SECONDS:
          process.env.RUNTIME_CATALOG_HTTP_CACHE_SECONDS || "10",
        WORLD_WEB_TRANSPORT: process.env.WORLD_WEB_TRANSPORT || "legacy",
        WORLD_WEB_LEGACY_PORT: process.env.WORLD_WEB_LEGACY_PORT || "18787",
        WORLD_WEB_ASGI_LIMIT_CONCURRENCY:
          process.env.WORLD_WEB_ASGI_LIMIT_CONCURRENCY || "384",
        WORLD_WEB_ASGI_PROXY_TIMEOUT_SECONDS:
          process.env.WORLD_WEB_ASGI_PROXY_TIMEOUT_SECONDS || "120",
        WORLD_WEB_ASGI_PROXY_SIMULATION_FULL_TIMEOUT_SECONDS:
          process.env.WORLD_WEB_ASGI_PROXY_SIMULATION_FULL_TIMEOUT_SECONDS || "45",
        WORLD_WEB_ASGI_WS_PING_INTERVAL_SECONDS:
          process.env.WORLD_WEB_ASGI_WS_PING_INTERVAL_SECONDS || "0",
        WORLD_WEB_ASGI_WS_PING_TIMEOUT_SECONDS:
          process.env.WORLD_WEB_ASGI_WS_PING_TIMEOUT_SECONDS || "30",
        RUNTIME_WS_MAX_CLIENTS: process.env.RUNTIME_WS_MAX_CLIENTS || "24",
        SIM_TICK_SECONDS: process.env.SIM_TICK_SECONDS || "0.083333",
        SIMULATION_WS_USE_CACHED_SNAPSHOTS:
          process.env.SIMULATION_WS_USE_CACHED_SNAPSHOTS || "1",
        SIMULATION_WS_CACHE_REFRESH_SECONDS:
          process.env.SIMULATION_WS_CACHE_REFRESH_SECONDS || "0.5",
        SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS:
          process.env.SIMULATION_WS_FULL_SNAPSHOT_HEARTBEAT_SECONDS || "2.5",
        SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED:
          process.env.SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED || "1",
        SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS:
          process.env.SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS || "180",
        SIMULATION_HTTP_FULL_ASYNC_LOCK_TIMEOUT_SECONDS:
          process.env.SIMULATION_HTTP_FULL_ASYNC_LOCK_TIMEOUT_SECONDS || "18",
        SIMULATION_HTTP_FULL_ASYNC_MAX_RUNNING_SECONDS:
          process.env.SIMULATION_HTTP_FULL_ASYNC_MAX_RUNNING_SECONDS || "90",
        SIMULATION_HTTP_FULL_ASYNC_START_MIN_INTERVAL_SECONDS:
          process.env.SIMULATION_HTTP_FULL_ASYNC_START_MIN_INTERVAL_SECONDS || "5",
      },
      watch: ["code/world_web.py", "code/world_web/**/*.py", "code/world_pm2.py"],
      ignore_watch: ["world_state", "__pycache__", "*.log", ".git", "artifacts", "*.md"],
      autorestart: true,
      min_uptime: "10s",
      max_restarts: 10,
      restart_delay: 2000,
      out_file: "./world_state/pm2-world-out.log",
      error_file: "./world_state/pm2-world-error.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      merge_logs: true,
    },
    {
      name: "eta-mu-io",
      cwd: __dirname,
      script: "code/world_io.js",
      env: {
        WORLD_API: process.env.WORLD_API || "http://127.0.0.1:8787",
      },
      watch: ["code/world_io.js"],
      autorestart: true,
    },
    {
      name: "eta-mu-tts",
      cwd: __dirname,
      script: "code/tts_service.py",
      interpreter: "python3",
      env: {
        PYTHONUNBUFFERED: "1",
      },
      watch: false,
      autorestart: true,
    },
    {
      name: "web-graph-weaver",
      cwd: __dirname,
      script: "code/web_graph_weaver.js",
      env: {
        WEAVER_HOST: process.env.WEAVER_HOST || "127.0.0.1",
        WEAVER_PORT: process.env.WEAVER_PORT || "8793",
        WEAVER_CONCURRENCY: process.env.WEAVER_CONCURRENCY || "32",
        WEAVER_MAX_DEPTH: process.env.WEAVER_MAX_DEPTH || "12",
        WEAVER_MAX_NODES: process.env.WEAVER_MAX_NODES || "2000000",
        WEAVER_MAX_REQUESTS_PER_HOST: process.env.WEAVER_MAX_REQUESTS_PER_HOST || "64",
        WEAVER_ENTITY_COUNT: process.env.WEAVER_ENTITY_COUNT || "128",
      },
      watch: ["code/web_graph_weaver.js"],
      ignore_watch: ["world_state", "artifacts", "*.md", "__pycache__"],
      autorestart: true,
    },
    {
      name: "eta-mu-lith-nexus-mcp",
      cwd: __dirname,
      script: resolveLithNexusScript(),
      interpreter: "node",
      env: {
        PYTHONUNBUFFERED: "1",
        LITH_NEXUS_REPO_ROOT: resolveLithNexusRepoRoot(),
        LITH_NEXUS_PYTHON_WORKDIR: resolveLithNexusPythonWorkdir(),
        LITH_NEXUS_HTTP_HOST: process.env.LITH_NEXUS_HTTP_HOST || "127.0.0.1",
        LITH_NEXUS_HTTP_PORT: process.env.LITH_NEXUS_HTTP_PORT || "8794",
        LITH_NEXUS_HTTP_PATH: process.env.LITH_NEXUS_HTTP_PATH || "/mcp",
      },
      watch: [
        path.resolve(__dirname, "..", "mcp-lith-nexus", "dist", "**", "*.js"),
      ],
      ignore_watch: ["world_state", "artifacts", "*.md", "__pycache__"],
      autorestart: true,
      min_uptime: "5s",
      max_restarts: 10,
      restart_delay: 2000,
      out_file: "./world_state/pm2-mcp-out.log",
      error_file: "./world_state/pm2-mcp-error.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      merge_logs: true,
    }
  ],
};
