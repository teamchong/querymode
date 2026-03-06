/**
 * Next.js / Vinext API Route — QueryMode as the data layer.
 *
 * This example shows how to use QueryMode in an API route to query
 * Parquet/Lance/CSV files sitting in your project directory. No database
 * needed — just files.
 *
 * Works with:
 *   - Next.js App Router (app/api/orders/route.ts)
 *   - Vinext (Cloudflare's Vite-based Next.js — same App Router API, deploys to Workers)
 *   - Any framework with Web Standard Request/Response (Hono, SvelteKit, etc.)
 *
 * Setup:
 *   1. Place data files in your project (e.g., ./data/orders.parquet)
 *   2. Copy this into app/api/orders/route.ts
 *   3. QueryMode runs in-process — no sidecar, no connection pool
 *
 * Vinext advantage: QueryMode already runs on Cloudflare Workers.
 * With Vinext, your API routes deploy to Workers too — same runtime,
 * zero cold-start penalty for the query engine.
 */

// ─── App Router example (app/api/orders/route.ts) ──────────────────────

import { QueryMode } from "../src/local.js";
// In a real project: import { QueryMode } from "querymode/local"

// Single instance — footer cache persists across requests
const qm = QueryMode.local();

/**
 * GET /api/orders?region=US&min_amount=100&sort=desc&limit=20
 *
 * Query Parquet files directly from an API route.
 * No database, no ORM, no SQL — just files and code.
 */
export async function GET(request: { url: string }) {
  const url = new URL(request.url);
  const region = url.searchParams.get("region");
  const minAmount = Number(url.searchParams.get("min_amount") ?? 0);
  const sort = (url.searchParams.get("sort") ?? "desc") as "asc" | "desc";
  const limit = Number(url.searchParams.get("limit") ?? 20);

  // Build query — each method returns a new immutable DataFrame
  let query = qm.table("./data/orders.parquet");

  if (region) {
    query = query.filter("region", "eq", region);
  }
  if (minAmount > 0) {
    query = query.filter("amount", "gte", minAmount);
  }

  const result = await query
    .select("id", "region", "amount", "category", "created_at")
    .sort("amount", sort)
    .limit(limit)
    .collect();

  return new Response(JSON.stringify({
    rows: result.rows,
    meta: {
      rowCount: result.rowCount,
      bytesRead: result.bytesRead,
      pagesSkipped: result.pagesSkipped,
      durationMs: result.durationMs,
    },
  }), {
    headers: { "Content-Type": "application/json" },
  });
}

// ─── Aggregation endpoint ──────────────────────────────────────────────

/**
 * POST /api/orders (body: { groupBy: "region" })
 *
 * Aggregation — same QueryMode instance, same cached footers.
 */
export async function POST(request: { json: () => Promise<{ groupBy?: string }> }) {
  const body = await request.json();
  const groupByCol = body.groupBy ?? "region";

  const result = await qm
    .table("./data/orders.parquet")
    .groupBy(groupByCol)
    .aggregate("sum", "amount", "total")
    .aggregate("count", "*", "order_count")
    .aggregate("avg", "amount", "avg_amount")
    .sort("total", "desc")
    .collect();

  return new Response(JSON.stringify(result.rows), {
    headers: { "Content-Type": "application/json" },
  });
}

// ─── Demo: works without any data files ────────────────────────────────

/**
 * For quick prototyping — QueryMode.demo() generates 1000 rows in-memory.
 * No Parquet files needed.
 */
async function demoEndpoint() {
  const demo = QueryMode.demo();

  // Describe the schema (zero scan cost)
  const schema = await demo.describe();
  console.log("Schema:", schema);
  // { columns: [{ name: "id", dtype: "unknown" }, ...], totalRows: 1000 }

  // Preview first 5 rows
  const preview = await demo.head(5);
  console.log("Preview:", preview);

  // Full query with progress reporting
  const result = await demo
    .filter("category", "eq", "Electronics")
    .sort("amount", "desc")
    .limit(50)
    .collect({
      onProgress: (p) => {
        console.log(`Progress: ${p.rowsCollected} rows collected`);
      },
    });

  return result;
}

// Run demo if executed directly
demoEndpoint().then(r => {
  console.log(`\nDemo result: ${r.rowCount} rows`);
  for (const row of r.rows.slice(0, 3)) {
    console.log(`  ${row.id} | ${row.category} | ${row.amount}`);
  }
}).catch(console.error);
