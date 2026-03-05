/**
 * Head-to-head benchmarks: QueryMode (Miniflare, full DO stack) vs DuckDB (native Node).
 *
 * QueryMode runs on the real CF Worker runtime via wrangler dev:
 *   HTTP → Worker → Query DO → R2 → WASM decode → operators → response
 *
 * DuckDB runs natively in Node.js — no serialization, no HTTP, no Worker overhead.
 *
 * Prerequisites:
 *   1. `pnpm dev` running on localhost:8787
 *   2. `npx tsx scripts/generate-bench-data.ts`
 *   3. `npx tsx scripts/seed-local-r2.ts`
 *
 * Usage: pnpm bench:operators
 */

import { describe, bench, beforeAll, afterAll } from "vitest";
import duckdb from "duckdb";

// ---------------------------------------------------------------------------
// QueryMode (Miniflare) helpers
// ---------------------------------------------------------------------------

const BASE_URL = process.env.WORKER_URL ?? "http://localhost:8787";

async function qmQuery(body: unknown): Promise<Record<string, unknown>> {
  const resp = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`QueryMode ${resp.status}: ${(await resp.text()).slice(0, 200)}`);
  return resp.json() as Promise<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// DuckDB helpers
// ---------------------------------------------------------------------------

let db: duckdb.Database;
let con: duckdb.Connection;

function duckRun(sql: string): Promise<void> {
  return new Promise((resolve, reject) => {
    con.run(sql, (err: Error | null) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

function duckQuery(sql: string): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    con.all(sql, (err: Error | null, rows: Record<string, unknown>[]) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
}

// ---------------------------------------------------------------------------
// Setup: DuckDB in-memory tables matching the seeded R2 Parquet data
// ---------------------------------------------------------------------------

beforeAll(async () => {
  // Verify worker is reachable
  const health = await fetch(`${BASE_URL}/health`);
  if (!health.ok) throw new Error(`Worker not reachable at ${BASE_URL}. Is 'pnpm dev' running?`);

  // Warm up Query DO registration
  await fetch(`${BASE_URL}/tables`);

  db = new duckdb.Database(":memory:");
  con = new duckdb.Connection(db);

  // bench_1m_numeric: id (BIGINT), value (DOUBLE) — 1M rows, deterministic
  // Matches generate-bench-data.ts: ids 0..999999, values random * 100000
  // For fair comparison, use the same deterministic pattern
  await duckRun(`CREATE TABLE bench_1m AS SELECT i AS id, (i * 7 + 13) % 10000 AS value FROM generate_series(0, 999999) t(i)`);

  // bench_100k_3col: id (BIGINT), value (DOUBLE), category (VARCHAR) — 100K rows
  const cats = ["alpha", "beta", "gamma", "delta", "epsilon"];
  await duckRun(`
    CREATE TABLE bench_100k AS
    SELECT i AS id,
           (i * 7 + 13) % 1000 AS value,
           CASE i % 5
             WHEN 0 THEN '${cats[0]}'
             WHEN 1 THEN '${cats[1]}'
             WHEN 2 THEN '${cats[2]}'
             WHEN 3 THEN '${cats[3]}'
             ELSE '${cats[4]}'
           END AS category
    FROM generate_series(0, 99999) t(i)
  `);
}, 60_000);

afterAll(() => {
  con?.close();
  db?.close();
});

// ===================================================================
// 1. Full scan — 1M rows, 2 numeric columns
// ===================================================================

describe("Full scan 1M×2col numeric", () => {
  bench("QueryMode (Miniflare)", async () => {
    await qmQuery({
      table: "bench_1m_numeric",
      filters: [],
      projections: [],
    });
  }, { time: 30_000, warmupIterations: 2 });

  bench("DuckDB (native)", async () => {
    await duckQuery(`SELECT * FROM bench_1m`);
  }, { time: 30_000, warmupIterations: 2 });
});

// ===================================================================
// 2. Filter scan — 1M rows, filter id > 900000 (~10% selectivity)
// ===================================================================

describe("Filter scan 1M id>900000", () => {
  bench("QueryMode (Miniflare)", async () => {
    await qmQuery({
      table: "bench_1m_numeric",
      filters: [{ column: "id", op: "gt", value: 900000 }],
      projections: [],
    });
  }, { time: 30_000, warmupIterations: 2 });

  bench("DuckDB (native)", async () => {
    await duckQuery(`SELECT * FROM bench_1m WHERE id > 900000`);
  }, { time: 30_000, warmupIterations: 2 });
});

// ===================================================================
// 3. Aggregate SUM — 1M rows
// ===================================================================

describe("Aggregate SUM 1M", () => {
  bench("QueryMode (Miniflare)", async () => {
    await qmQuery({
      table: "bench_1m_numeric",
      filters: [],
      projections: [],
      aggregates: [{ fn: "sum", column: "value", alias: "total" }],
    });
  }, { time: 30_000, warmupIterations: 2 });

  bench("DuckDB (native)", async () => {
    await duckQuery(`SELECT SUM(value) as total FROM bench_1m`);
  }, { time: 30_000, warmupIterations: 2 });
});

// ===================================================================
// 4. Aggregate group by category — 100K×3col
// ===================================================================

describe("Aggregate group by category 100K", () => {
  bench("QueryMode (Miniflare)", async () => {
    await qmQuery({
      table: "bench_100k_3col",
      filters: [],
      projections: [],
      aggregates: [
        { fn: "sum", column: "value", alias: "sum_value" },
        { fn: "count", column: "id", alias: "cnt" },
      ],
      groupBy: ["category"],
    });
  }, { time: 30_000, warmupIterations: 2 });

  bench("DuckDB (native)", async () => {
    await duckQuery(
      `SELECT category, SUM(value) as sum_value, COUNT(id) as cnt
       FROM bench_100k GROUP BY category`,
    );
  }, { time: 30_000, warmupIterations: 2 });
});

// ===================================================================
// 5. Sort + Limit (TopK) — 1M rows, top 100
// ===================================================================

describe("TopK 100 from 1M", () => {
  bench("QueryMode (Miniflare)", async () => {
    await qmQuery({
      table: "bench_1m_numeric",
      filters: [],
      projections: [],
      sortColumn: "value",
      sortDirection: "desc",
      limit: 100,
    });
  }, { time: 30_000, warmupIterations: 2 });

  bench("DuckDB (native)", async () => {
    await duckQuery(`SELECT * FROM bench_1m ORDER BY value DESC LIMIT 100`);
  }, { time: 30_000, warmupIterations: 2 });
});

// ===================================================================
// 6. Column projection — 100K, select 1 of 3 columns
// ===================================================================

describe("Projection 100K select 1 col", () => {
  bench("QueryMode (Miniflare)", async () => {
    await qmQuery({
      table: "bench_100k_3col",
      filters: [],
      projections: ["id"],
    });
  }, { time: 30_000, warmupIterations: 2 });

  bench("DuckDB (native)", async () => {
    await duckQuery(`SELECT id FROM bench_100k`);
  }, { time: 30_000, warmupIterations: 2 });
});

// ===================================================================
// 7. Filter + Aggregate — 100K, filter + count
// ===================================================================

describe("Filter + Count 100K id>50000", () => {
  bench("QueryMode (Miniflare)", async () => {
    await qmQuery({
      table: "bench_100k_3col",
      filters: [{ column: "id", op: "gt", value: 50000 }],
      projections: [],
      aggregates: [{ fn: "count", column: "id", alias: "cnt" }],
    });
  }, { time: 30_000, warmupIterations: 2 });

  bench("DuckDB (native)", async () => {
    await duckQuery(`SELECT COUNT(id) as cnt FROM bench_100k WHERE id > 50000`);
  }, { time: 30_000, warmupIterations: 2 });
});
