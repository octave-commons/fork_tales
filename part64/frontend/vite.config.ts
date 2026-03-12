import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const buildSourceMaps = process.env.VITE_BUILD_SOURCEMAP !== "false";
const devRuntimeProxyTarget = process.env.VITE_DEV_PROXY_TARGET || "http://127.0.0.1:8787";
const DEFAULT_DEV_SERVER_PORT = 5197;
const parsedDevServerPort = Number.parseInt(
  process.env.VITE_DEV_SERVER_PORT ?? `${DEFAULT_DEV_SERVER_PORT}`,
  10,
);
const devServerPort = Number.isFinite(parsedDevServerPort)
  ? parsedDevServerPort
  : DEFAULT_DEV_SERVER_PORT;

// https://vite.dev/config/
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: devServerPort,
    strictPort: true,
    proxy: {
      "/api": {
        target: devRuntimeProxyTarget,
        changeOrigin: true,
      },
      "/ws": {
        target: devRuntimeProxyTarget,
        changeOrigin: true,
        ws: true,
      },
      "/stream": {
        target: devRuntimeProxyTarget,
        changeOrigin: true,
      },
      "/weaver": {
        target: devRuntimeProxyTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    sourcemap: buildSourceMaps,
  },
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
    exclude: ["e2e/**", "node_modules/**", "dist/**"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "src/**/*.test.{ts,tsx}",
        "src/**/*.d.ts",
        "src/main.tsx",
        "src/types/**",
        "src/app/appShellTypes.ts",
      ],
      reporter: ["text", "html", "json-summary", "json"],
      reportsDirectory: "./coverage",
    },
  },
});
