import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "WORKFLOWS_");
  const backendOrigin = env.WORKFLOWS_ADMIN_API_ORIGIN ?? "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target: backendOrigin,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: "../src/workflows_mcp/static/admin",
      emptyOutDir: true,
    },
    test: {
      environment: "jsdom",
      globals: true,
      passWithNoTests: true,
    },
  };
});
