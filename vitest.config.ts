import { defineConfig } from "vitest/config";
import { resolve } from "node:path";

export default defineConfig({
  resolve: {
    alias: {
      "cloudflare:workers": resolve(__dirname, "src/cloudflare-workers-polyfill.ts"),
    },
  },
  test: {
    environment: "node",
    setupFiles: ["./src/test-setup.ts"],
    include: [
      // Node-only tests: native addons (duckdb) and disk fixture I/O (node:fs)
      "src/operators-conformance.test.ts",
      "src/conformance.test.ts",
      "src/parquet-fixture.test.ts",
      "src/manifest.test.ts",
      "src/wasm-engine.integration.test.ts",
      "src/query-do.integration.test.ts",
    ],
    hookTimeout: 300_000,
    testTimeout: 300_000,
  },
});
