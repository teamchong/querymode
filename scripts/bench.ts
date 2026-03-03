#!/usr/bin/env npx tsx
/**
 * Benchmark QueryMode via wrangler dev (full DO stack).
 *
 * Measures end-to-end latency: HTTP request → Worker → Query DO → R2 → WASM → response
 *
 * Prerequisites:
 *   1. `pnpm dev` running on localhost:8787
 *   2. `npx tsx scripts/seed-local-r2.ts` to populate local R2
 *
 * Usage: npx tsx scripts/bench.ts
 */

const BASE_URL = process.env.WORKER_URL ?? "http://localhost:8787";
const WARMUP_RUNS = 3;
const BENCH_RUNS = 20;

interface BenchResult {
  name: string;
  runs: number;
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
  avg: number;
  /** Server-side metrics from last run */
  serverMs?: number;
  r2ReadMs?: number;
  wasmExecMs?: number;
  bytesRead?: number;
  pagesSkipped?: number;
  rowCount?: number;
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

async function runQuery(body: unknown): Promise<{ latencyMs: number; result: Record<string, unknown> }> {
  const start = performance.now();
  const resp = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const latencyMs = performance.now() - start;
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Query failed (${resp.status}): ${text}`);
  }
  const result = await resp.json() as Record<string, unknown>;
  return { latencyMs, result };
}

async function bench(name: string, query: unknown): Promise<BenchResult> {
  // Warmup
  for (let i = 0; i < WARMUP_RUNS; i++) {
    try { await runQuery(query); } catch { /* table may not exist */ }
  }

  const latencies: number[] = [];
  let lastResult: Record<string, unknown> = {};

  for (let i = 0; i < BENCH_RUNS; i++) {
    try {
      const { latencyMs, result } = await runQuery(query);
      latencies.push(latencyMs);
      lastResult = result;
    } catch (err) {
      console.error(`  ${name} run ${i} failed:`, err);
    }
  }

  if (latencies.length === 0) {
    return { name, runs: 0, p50: 0, p95: 0, p99: 0, min: 0, max: 0, avg: 0 };
  }

  latencies.sort((a, b) => a - b);
  const avg = latencies.reduce((s, v) => s + v, 0) / latencies.length;

  return {
    name,
    runs: latencies.length,
    p50: percentile(latencies, 50),
    p95: percentile(latencies, 95),
    p99: percentile(latencies, 99),
    min: latencies[0],
    max: latencies[latencies.length - 1],
    avg,
    serverMs: lastResult.durationMs as number | undefined,
    r2ReadMs: lastResult.r2ReadMs as number | undefined,
    wasmExecMs: lastResult.wasmExecMs as number | undefined,
    bytesRead: lastResult.bytesRead as number | undefined,
    pagesSkipped: lastResult.pagesSkipped as number | undefined,
    rowCount: lastResult.rowCount as number | undefined,
  };
}

function formatMs(ms: number): string {
  return ms < 1 ? `${(ms * 1000).toFixed(0)}µs` : `${ms.toFixed(1)}ms`;
}

function printResults(results: BenchResult[]): void {
  console.log("\n" + "=".repeat(100));
  console.log("QueryMode Benchmark Results (wrangler dev — full DO stack)");
  console.log("=".repeat(100));

  const header = [
    "Benchmark".padEnd(35),
    "p50".padStart(8),
    "p95".padStart(8),
    "p99".padStart(8),
    "min".padStart(8),
    "max".padStart(8),
    "avg".padStart(8),
    "rows".padStart(8),
    "bytes".padStart(10),
    "skip".padStart(5),
  ].join(" | ");

  console.log(header);
  console.log("-".repeat(100));

  for (const r of results) {
    if (r.runs === 0) {
      console.log(`${r.name.padEnd(35)} | SKIPPED (table not found)`);
      continue;
    }
    const row = [
      r.name.padEnd(35),
      formatMs(r.p50).padStart(8),
      formatMs(r.p95).padStart(8),
      formatMs(r.p99).padStart(8),
      formatMs(r.min).padStart(8),
      formatMs(r.max).padStart(8),
      formatMs(r.avg).padStart(8),
      String(r.rowCount ?? "-").padStart(8),
      r.bytesRead != null ? `${(r.bytesRead / 1024).toFixed(1)}KB` : "-".padStart(10),
      String(r.pagesSkipped ?? "-").padStart(5),
    ].join(" | ");
    console.log(row);
  }

  console.log("-".repeat(100));

  // Server-side breakdown for last run of each
  console.log("\nServer-side breakdown (last run):");
  for (const r of results) {
    if (r.runs === 0) continue;
    console.log(
      `  ${r.name}: total=${formatMs(r.serverMs ?? 0)} ` +
      `r2=${formatMs(r.r2ReadMs ?? 0)} wasm=${formatMs(r.wasmExecMs ?? 0)}`,
    );
  }
}

async function main(): Promise<void> {
  // Check Worker is running
  try {
    const health = await fetch(`${BASE_URL}/health`);
    if (!health.ok) throw new Error(`${health.status}`);
    console.log("Worker health:", await health.json());
  } catch {
    console.error(`Cannot reach Worker at ${BASE_URL}. Is 'pnpm dev' running?`);
    process.exit(1);
  }

  // Force Query DO to register with Master by hitting /tables first
  console.log("Registering Query DO with Master...");
  await fetch(`${BASE_URL}/tables`);
  // Small delay for registration to complete
  await new Promise(r => setTimeout(r, 500));

  // Refresh all known tables so Query DO gets the footer data
  const knownTables = [
    // Lance datasets
    "simple_int64.lance/", "simple_float64.lance/", "mixed_types.lance/",
    "large.lance/", "multiple_batches.lance/", "strings_various.lance/",
    "with_nulls.lance/", "basic_types.lance/", "vectors.lance/",
    "vector_search.lance/",
    // Parquet files
    "simple.parquet", "simple_plain.parquet", "simple_snappy.parquet",
    "benchmark_100k.parquet", "benchmark_100k_uncompressed.parquet",
  ];

  console.log("Refreshing tables via Master DO...");
  for (const r2Key of knownTables) {
    try {
      const resp = await fetch(`${BASE_URL}/refresh`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ r2Key }),
      });
      if (resp.ok) {
        const result = await resp.json() as Record<string, unknown>;
        console.log(`  ${result.refreshed ? "Refreshed" : "Skip"}: ${r2Key}`);
      } else {
        console.log(`  Skip: ${r2Key} (${resp.status})`);
      }
    } catch {
      console.log(`  Skip: ${r2Key} (error)`);
    }
  }

  // Wait for broadcasts to propagate
  await new Promise(r => setTimeout(r, 500));

  const tablesResp = await fetch(`${BASE_URL}/tables`);
  const { tables } = await tablesResp.json() as { tables: string[] };
  console.log("\nAvailable tables:", tables);

  if (tables.length === 0) {
    console.error("No tables found. Run `npx tsx scripts/seed-local-r2.ts` first.");
    process.exit(1);
  }

  const results: BenchResult[] = [];

  // --- Lance benchmarks ---

  // 1. Full scan (small table)
  results.push(await bench("lance: full scan int64", {
    table: "simple_int64",
    filters: [],
    projections: [],
  }));

  // 2. Full scan float64
  results.push(await bench("lance: full scan float64", {
    table: "simple_float64",
    filters: [],
    projections: [],
  }));

  // 3. Mixed types scan
  results.push(await bench("lance: mixed types scan", {
    table: "mixed_types",
    filters: [],
    projections: [],
  }));

  // 4. Filter pushdown (gt)
  results.push(await bench("lance: filter int64 > value", {
    table: "simple_int64",
    filters: [{ column: "value", op: "gt", value: 50 }],
    projections: [],
  }));

  // 5. Column projection
  results.push(await bench("lance: select 1 col from mixed", {
    table: "mixed_types",
    filters: [],
    projections: ["int_col"],
  }));

  // 6. Large dataset scan
  results.push(await bench("lance: large dataset scan", {
    table: "large",
    filters: [],
    projections: [],
  }));

  // 7. Large dataset with filter
  results.push(await bench("lance: large + filter", {
    table: "large",
    filters: [{ column: "value", op: "gt", value: 500 }],
    projections: [],
  }));

  // 8. Multiple batches (multi-fragment)
  results.push(await bench("lance: multi-batch scan", {
    table: "multiple_batches",
    filters: [],
    projections: [],
  }));

  // 9. Strings
  results.push(await bench("lance: string scan", {
    table: "strings_various",
    filters: [],
    projections: [],
  }));

  // 10. Nulls
  results.push(await bench("lance: nulls scan", {
    table: "with_nulls",
    filters: [],
    projections: [],
  }));

  // --- Parquet benchmarks ---

  // 11. Simple Parquet (plain encoding)
  results.push(await bench("parquet: plain encoding", {
    table: "simple_plain",
    filters: [],
    projections: [],
  }));

  // 12. Parquet with Snappy
  results.push(await bench("parquet: snappy compressed", {
    table: "simple_snappy",
    filters: [],
    projections: [],
  }));

  // 13. Parquet 100K rows
  results.push(await bench("parquet: 100k rows", {
    table: "benchmark_100k",
    filters: [],
    projections: [],
  }));

  // 14. Parquet 100K uncompressed
  results.push(await bench("parquet: 100k uncompressed", {
    table: "benchmark_100k_uncompressed",
    filters: [],
    projections: [],
  }));

  // 15. Parquet 100K with filter
  results.push(await bench("parquet: 100k + filter", {
    table: "benchmark_100k",
    filters: [{ column: "id", op: "gt", value: 50000 }],
    projections: [],
  }));

  // --- Health/diagnostics ---
  console.log("\n--- Query DO Diagnostics ---");
  const diag = await fetch(`${BASE_URL}/health?deep=true`);
  const diagResult = await diag.json();
  console.log(JSON.stringify(diagResult, null, 2));

  printResults(results);
}

main().catch(console.error);
