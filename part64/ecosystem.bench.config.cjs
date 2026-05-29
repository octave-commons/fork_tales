const path = require("node:path");

const WATCH_WORLD_PATHS = ["code/world_web.py", "code/world_web/**/*.py"];
const WATCH_IO_PATHS = ["code/world_io.js"];
const WATCH_TTS_PATHS = ["code/tts_service.py"];
const WATCH_WEAVER_PATHS = ["code/web_graph_weaver.js"];
const WATCH_MCP_PATHS = [
  path.resolve(__dirname, "..", "mcp-lith-nexus", "dist", "**", "*.js"),
];

const WATCH_ENABLED = ["1", "true", "yes", "on"].includes(
  String(process.env.PM2_WATCH_MODE || "0").trim().toLowerCase(),
);

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

const SHARED_IGNORE_WATCH = [
  "world_state",
  "node_modules",
  "__pycache__",
  ".pytest_cache",
  ".git",
  "*.pyc",
  "*.log",
  "*.bak",
  "*.md",
];

module.exports = {
  apps: [
    {
      name: "eta-mu-world",
      cwd: __dirname,
      script: "code/world_web.py",
      interpreter: "python3",
      args: "--host 0.0.0.0 --port 8787 --part-root /app --vault-root /workspace",
      env: {
        PYTHONUNBUFFERED: "1",
        DOCKER_SIMULATION_RESOURCE_WORKERS:
          process.env.DOCKER_SIMULATION_RESOURCE_WORKERS || "10",
        NVIDIA_VISIBLE_DEVICES: resolveNvidiaVisibleDevices(),
        CDB_EMBED_GPU_VISIBLE_DEVICES:
          process.env.CDB_EMBED_GPU_VISIBLE_DEVICES || resolveNvidiaVisibleDevices(),
        CDB_EMBED_GPU_CONTROLLER_PROFILE:
          process.env.CDB_EMBED_GPU_CONTROLLER_PROFILE || "energy",
        CDB_EMBED_GPU_SIDECAR_SPLIT_RATIO:
          process.env.CDB_EMBED_GPU_SIDECAR_SPLIT_RATIO || "0.0",
        CDB_EMBED_GPU_SIDECAR_SPLIT_HOT_RATIO:
          process.env.CDB_EMBED_GPU_SIDECAR_SPLIT_HOT_RATIO || "0.0",
        OPENVINO_EMBED_DEVICE: process.env.OPENVINO_EMBED_DEVICE || "NPU",
        CDB_EMBED_DEVICE: process.env.CDB_EMBED_DEVICE || "NPU",
        CDB_EMBED_AUTO_POLICY:
          process.env.CDB_EMBED_AUTO_POLICY || "adaptive-npu",
        CDB_ORT_GPU_CAPI_DIR: resolveOrtGpuCapiDir(),
        CDB_ORT_GPU_INCLUDE_DIR: resolveOrtGpuIncludeDir(),
        THREAT_RADAR_LLM_ENABLED:
          process.env.THREAT_RADAR_LLM_ENABLED || "0",
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
        OPENPLANNER_ENABLED: process.env.OPENPLANNER_ENABLED || "1",
        OPENPLANNER_URL:
          process.env.OPENPLANNER_URL || "http://127.0.0.1:7777",
        OPENPLANNER_API_KEY:
          process.env.OPENPLANNER_API_KEY || "change-me",
        OPENPLANNER_PROJECT:
          process.env.OPENPLANNER_PROJECT || "eta-mu",
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
        CDB_FORCE_WORKERS: process.env.CDB_FORCE_WORKERS || "2",
        CDB_COLLISION_WORKERS: process.env.CDB_COLLISION_WORKERS || "2",
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
      watch: WATCH_ENABLED ? WATCH_WORLD_PATHS : false,
      ignore_watch: SHARED_IGNORE_WATCH,
      watch_delay: 1000,
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
        ETA_MU_INBOX_ROOT: process.env.ETA_MU_INBOX_ROOT || "/workspace/.ημ",
        ETA_MU_SYNC_DEBOUNCE_MS: process.env.ETA_MU_SYNC_DEBOUNCE_MS || "1500",
      },
      watch: WATCH_ENABLED ? WATCH_IO_PATHS : false,
      ignore_watch: SHARED_IGNORE_WATCH,
      watch_delay: 1000,
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
      watch: WATCH_ENABLED ? WATCH_TTS_PATHS : false,
      ignore_watch: SHARED_IGNORE_WATCH,
      watch_delay: 1000,
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
      watch: WATCH_ENABLED ? WATCH_WEAVER_PATHS : false,
      ignore_watch: SHARED_IGNORE_WATCH,
      watch_delay: 1000,
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
        WORLD_WEB_HOST: process.env.WORLD_WEB_HOST || "127.0.0.1",
        WORLD_WEB_PORT: process.env.WORLD_WEB_PORT || "8787",
        LITH_NEXUS_HTTP_HOST: process.env.LITH_NEXUS_HTTP_HOST || "127.0.0.1",
        LITH_NEXUS_HTTP_PORT: process.env.LITH_NEXUS_HTTP_PORT || "8794",
        LITH_NEXUS_HTTP_PATH: process.env.LITH_NEXUS_HTTP_PATH || "/mcp",
      },
      watch: WATCH_ENABLED ? WATCH_MCP_PATHS : false,
      ignore_watch: SHARED_IGNORE_WATCH,
      watch_delay: 1000,
      autorestart: true,
      min_uptime: "5s",
      max_restarts: 10,
      restart_delay: 2000,
      out_file: "./world_state/pm2-mcp-out.log",
      error_file: "./world_state/pm2-mcp-error.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      merge_logs: true,
    },
  ],
};
