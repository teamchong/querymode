import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
  test: {
    include: ["src/**/*.test.ts"],
    exclude: [
      // These require native Node addons (duckdb) or disk fixture I/O (node:fs)
      "src/operators-conformance.test.ts",
      "src/conformance.test.ts",
      "src/parquet-fixture.test.ts",
      "src/manifest.test.ts",
      "src/wasm-engine.integration.test.ts",
      "src/query-do.integration.test.ts",
      "node_modules/**",
    ],
    hookTimeout: 300_000,
    testTimeout: 300_000,
    poolOptions: {
      workers: {
        wrangler: { configPath: "./wrangler.toml" },
      },
    },
  },
});
