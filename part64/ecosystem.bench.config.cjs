const WATCH_WORLD_PATHS = ["code/world_web.py", "code/world_web/**/*.py"];
const WATCH_IO_PATHS = ["code/world_io.js"];
const WATCH_TTS_PATHS = ["code/tts_service.py"];
const WATCH_WEAVER_PATHS = ["code/web_graph_weaver.js"];

const WATCH_ENABLED = ["1", "true", "yes", "on"].includes(
  String(process.env.PM2_WATCH_MODE || "0").trim().toLowerCase(),
);

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
      args: "--host 0.0.0.0 --port 8787 --part-root /app --vault-root /vault",
      env: {
        PYTHONUNBUFFERED: "1",
        DOCKER_SIMULATION_RESOURCE_WORKERS:
          process.env.DOCKER_SIMULATION_RESOURCE_WORKERS || "10",
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
        WEAVER_CONCURRENCY: process.env.WEAVER_CONCURRENCY || "2",
        WEAVER_MAX_DEPTH: process.env.WEAVER_MAX_DEPTH || "3",
        WEAVER_MAX_NODES: process.env.WEAVER_MAX_NODES || "10000",
      },
      watch: WATCH_ENABLED ? WATCH_WEAVER_PATHS : false,
      ignore_watch: SHARED_IGNORE_WATCH,
      watch_delay: 1000,
      autorestart: true,
    },
  ],
};
