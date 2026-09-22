import { defineConfig } from "vitest/config";
import piskel from "./piskel-plugin.mjs";

// The shell's build. `vite build` -> dist/, which the pipeline API serves at
// `/` (anim/api/app.py, ANIM_WEB_DIST). `vite` (dev) proxies the API routes
// to a running anim-api so the panel works against real jobs while editing.
export default defineConfig({
  plugins: [piskel()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    host: true,
    port: 5173,
    proxy: Object.fromEntries(
      ["/jobs", "/runs", "/settings", "/health", "/palette", "/curation", "/turnaround", "/export", "/concept", "/models", "/docs", "/openapi.json"].map((route) => [
        route,
        { target: "http://127.0.0.1:8720", changeOrigin: false },
      ]),
    ),
  },
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
  },
});
