import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/workflows_mcp/static/admin",
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
    globals: true,
    passWithNoTests: true,
  },
});
