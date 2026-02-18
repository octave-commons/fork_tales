module.exports = {
  apps: [
    {
      name: "eta-mu-electron-client",
      cwd: __dirname,
      script: "npm",
      args: "run electron:start",
      interpreter: "none",
      autorestart: false,
      watch: false,
      env: {
        WORLD_RUNTIME_URL: process.env.WORLD_RUNTIME_URL || "http://127.0.0.1:8787",
        WEAVER_RUNTIME_URL: process.env.WEAVER_RUNTIME_URL || "http://127.0.0.1:8793",
        DISPLAY: process.env.DISPLAY || ":0",
      },
    },
  ],
};
