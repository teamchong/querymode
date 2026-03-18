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
      "src/sql/lexer.test.ts",
      "src/sql/parser.test.ts",
      "src/sql/compiler.test.ts",
      "src/sql/evaluator.test.ts",
      "src/sql/integration.test.ts",
      "src/convenience.test.ts",
      "src/pipe.test.ts",
      "src/partition-catalog.test.ts",
      "src/bucket.test.ts",
      "src/materialized-executor.test.ts",
      "src/descriptor-to-code.test.ts",
      "src/pg-wire/pg-wire.test.ts",
      "src/columnar.test.ts",
      "src/decode.test.ts",
      "src/footer.test.ts",
      "src/format.test.ts",
      "src/iceberg.test.ts",
      "src/merge.test.ts",
      "src/page-processor.test.ts",
      "src/parquet-decode.test.ts",
      "src/parquet.test.ts",
      "src/partial-agg.test.ts",
      "src/vip-cache.test.ts",
      "src/hnsw.test.ts",
      "src/readers/readers.test.ts",
    ],
    hookTimeout: 300_000,
    testTimeout: 300_000,
  },
});
