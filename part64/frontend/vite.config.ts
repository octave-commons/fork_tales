import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const buildSourceMaps = process.env.VITE_BUILD_SOURCEMAP !== "false";

// https://vite.dev/config/
export default defineConfig({
  base: "./",
  plugins: [react()],
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
