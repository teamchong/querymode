#!/usr/bin/env npx tsx
/**
 * Benchmark QueryMode via wrangler dev (full DO stack).
 *
 * Measures end-to-end latency: HTTP request → Worker → Query DO → R2 → WASM → response
 *
 * Prerequisites:
 *   1. `pnpm dev` running on localhost:8787
 *   2. `npx tsx scripts/generate-bench-data.ts` to generate large fixtures
 *   3. `npx tsx scripts/seed-local-r2.ts` to populate local R2
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
    throw new Error(`Query failed (${resp.status}): ${text.slice(0, 200)}`);
  }
  const result = await resp.json() as Record<string, unknown>;
  return { latencyMs, result };
}

async function bench(name: string, query: unknown): Promise<BenchResult> {
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
      if (i === 0) console.error(`  ${name}: ${err}`);
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
  console.log("\n" + "=".repeat(120));
  console.log("QueryMode Benchmark Results (wrangler dev — full DO stack)");
  console.log("=".repeat(120));

  const header = [
    "Benchmark".padEnd(40),
    "p50".padStart(8),
    "p95".padStart(8),
    "min".padStart(8),
    "max".padStart(8),
    "avg".padStart(8),
    "rows".padStart(10),
    "server".padStart(8),
    "r2".padStart(8),
    "wasm".padStart(8),
  ].join(" | ");

  console.log(header);
  console.log("-".repeat(120));

  for (const r of results) {
    if (r.runs === 0) {
      console.log(`${r.name.padEnd(40)} | SKIPPED`);
      continue;
    }
    const row = [
      r.name.padEnd(40),
      formatMs(r.p50).padStart(8),
      formatMs(r.p95).padStart(8),
      formatMs(r.min).padStart(8),
      formatMs(r.max).padStart(8),
      formatMs(r.avg).padStart(8),
      String(r.rowCount ?? "-").padStart(10),
      formatMs(r.serverMs ?? 0).padStart(8),
      formatMs(r.r2ReadMs ?? 0).padStart(8),
      formatMs(r.wasmExecMs ?? 0).padStart(8),
    ].join(" | ");
    console.log(row);
  }

  console.log("-".repeat(120));
}

async function main(): Promise<void> {
  try {
    const health = await fetch(`${BASE_URL}/health`);
    if (!health.ok) throw new Error(`${health.status}`);
    console.log("Worker health:", await health.json());
  } catch {
    console.error(`Cannot reach Worker at ${BASE_URL}. Is 'pnpm dev' running?`);
    process.exit(1);
  }

  // Force Query DO to register
  console.log("Registering Query DO...");
  await fetch(`${BASE_URL}/tables`);
  await new Promise(r => setTimeout(r, 500));

  // Refresh tables
  const knownTables = [
    "simple_int64.lance/", "simple_float64.lance/", "mixed_types.lance/",
    "large.lance/", "multiple_batches.lance/",
    "simple.parquet", "simple_plain.parquet", "simple_snappy.parquet",
    "benchmark_100k.parquet", "benchmark_100k_uncompressed.parquet",
    "bench_100k_3col.parquet", "bench_100k_numeric.parquet", "bench_1m_numeric.parquet",
    "bench_iceberg_100k",
  ];

  console.log("Refreshing tables...");
  for (const r2Key of knownTables) {
    try {
      const resp = await fetch(`${BASE_URL}/refresh`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ r2Key }),
      });
      if (resp.ok) {
        const r = await resp.json() as Record<string, unknown>;
        console.log(`  ${r.refreshed ? "OK" : "Skip"}: ${r2Key}`);
      } else {
        console.log(`  Skip: ${r2Key} (${resp.status})`);
      }
    } catch {
      console.log(`  Skip: ${r2Key}`);
    }
  }
  await new Promise(r => setTimeout(r, 500));

  // Pre-register Iceberg tables by triggering lazy-load via a query
  const icebergTables = ["bench_iceberg_100k"];
  for (const tbl of icebergTables) {
    try {
      const resp = await fetch(`${BASE_URL}/query`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ table: tbl, filters: [], projections: ["id"], limit: 1 }),
      });
      if (resp.ok) {
        console.log(`  Iceberg OK: ${tbl}`);
      } else {
        const text = await resp.text();
        console.log(`  Iceberg Skip: ${tbl} (${resp.status}: ${text.slice(0, 100)})`);
      }
    } catch {
      console.log(`  Iceberg Skip: ${tbl}`);
    }
  }
  await new Promise(r => setTimeout(r, 500));

  const tablesResp = await fetch(`${BASE_URL}/tables`);
  const { tables } = await tablesResp.json() as { tables: { name: string; totalRows: number }[] };
  console.log("\nAvailable tables:", tables.map(t => `${t.name}(${t.totalRows})`).join(", "));

  const results: BenchResult[] = [];

  // ============================================================
  // PARQUET BENCHMARKS (the main event)
  // ============================================================

  // 100K rows, 3 columns (id int64, value float64, category utf8)
  results.push(await bench("pq: 100k×3col full scan", {
    table: "bench_100k_3col", filters: [], projections: [],
  }));

  results.push(await bench("pq: 100k×3col select 2 cols", {
    table: "bench_100k_3col", filters: [], projections: ["id", "value"],
  }));

  results.push(await bench("pq: 100k×3col filter id>50000", {
    table: "bench_100k_3col",
    filters: [{ column: "id", op: "gt", value: 50000 }],
    projections: [],
  }));

  // 100K rows, 2 numeric columns
  results.push(await bench("pq: 100k×2col numeric scan", {
    table: "bench_100k_numeric", filters: [], projections: [],
  }));

  results.push(await bench("pq: 100k×2col filter id>90000", {
    table: "bench_100k_numeric",
    filters: [{ column: "id", op: "gt", value: 90000 }],
    projections: [],
  }));

  // 1M rows, 2 numeric columns
  results.push(await bench("pq: 1M×2col numeric scan", {
    table: "bench_1m_numeric", filters: [], projections: [],
  }));

  results.push(await bench("pq: 1M×2col filter id>900000", {
    table: "bench_1m_numeric",
    filters: [{ column: "id", op: "gt", value: 900000 }],
    projections: [],
  }));

  // Legacy fixture parquets
  results.push(await bench("pq: 100k fixture (compressed)", {
    table: "benchmark_100k", filters: [], projections: [],
  }));

  results.push(await bench("pq: 100k fixture (uncompressed)", {
    table: "benchmark_100k_uncompressed", filters: [], projections: [],
  }));

  results.push(await bench("pq: snappy 5 rows", {
    table: "simple_snappy", filters: [], projections: [],
  }));

  // ============================================================
  // LANCE BENCHMARKS
  // ============================================================

  results.push(await bench("lance: 1000 rows full scan", {
    table: "large", filters: [], projections: [],
  }));

  results.push(await bench("lance: 1000 rows filter >500", {
    table: "large",
    filters: [{ column: "value", op: "gt", value: 500 }],
    projections: [],
  }));

  results.push(await bench("lance: multi-batch (3 frags)", {
    table: "multiple_batches", filters: [], projections: [],
  }));

  results.push(await bench("lance: mixed types scan", {
    table: "mixed_types", filters: [], projections: [],
  }));

  // ============================================================
  // ICEBERG BENCHMARKS
  // ============================================================

  results.push(await bench("iceberg: 100k×2col scan", {
    table: "bench_iceberg_100k", filters: [], projections: [],
  }));

  results.push(await bench("iceberg: 100k×2col filter >50000", {
    table: "bench_iceberg_100k",
    filters: [{ column: "id", op: "gt", value: 50000 }],
    projections: [],
  }));

  // ============================================================
  // RESULTS
  // ============================================================

  printResults(results);
}

main().catch(console.error);
