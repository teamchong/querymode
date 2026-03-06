/**
 * ML Scoring Pipeline — custom business logic runs INSIDE the query pipeline.
 *
 * This example shows how a scoring function runs between Filter and TopK
 * operators with zero data materialization. In DuckDB/Polars, UDFs serialize
 * data across the engine boundary; here, scoring runs in-process on raw rows.
 *
 * Pipeline: MockOperator(orders) → FilterOperator(active) → ComputedColumnOperator(score) → TopKOperator(20) → HashJoinOperator(users)
 */

import {
  FilterOperator,
  ComputedColumnOperator,
  TopKOperator,
  HashJoinOperator,
  drainPipeline,
  type Operator,
  type RowBatch,
} from "../src/operators.js";
import type { Row } from "../src/types.js";

// ─── Mock data source ───────────────────────────────────────────────────

class MockOperator implements Operator {
  private rows: Row[];
  private cursor = 0;
  private batchSize: number;
  constructor(rows: Row[], batchSize = 100) {
    this.rows = rows;
    this.batchSize = batchSize;
  }
  async next(): Promise<RowBatch | null> {
    if (this.cursor >= this.rows.length) return null;
    const batch = this.rows.slice(this.cursor, this.cursor + this.batchSize);
    this.cursor += this.batchSize;
    return batch;
  }
  async close() {}
}

// Generate 500 orders
const orders: Row[] = Array.from({ length: 500 }, (_, i) => ({
  order_id: i + 1,
  user_id: (i % 50) + 1,
  amount: ((i * 7 + 13) % 1000) + 1,
  status: i % 3 === 0 ? "active" : i % 3 === 1 ? "pending" : "cancelled",
  recency_days: (i * 13 + 7) % 365,
}));

// Generate 50 users
const users: Row[] = Array.from({ length: 50 }, (_, i) => ({
  user_id: i + 1,
  name: `User_${i + 1}`,
  tier: i % 5 === 0 ? "premium" : "standard",
}));

// ─── Custom scoring function ────────────────────────────────────────────

function computeScore(row: Row): unknown {
  const amount = row.amount as number;
  const recency = row.recency_days as number;
  // Score = amount weight (70%) + recency bonus (30%, inverse)
  return amount * 0.7 + (365 - recency) * 0.3;
}

// ─── Build pipeline ─────────────────────────────────────────────────────

async function main() {
  // 1. Source: stream orders
  const orderSource = new MockOperator(orders);

  // 2. Filter: only active orders
  const filtered = new FilterOperator(orderSource, [
    { column: "status", op: "eq", value: "active" },
  ]);

  // 3. Score: custom ML-style scoring runs IN the pipeline
  const scored = new ComputedColumnOperator(filtered, [
    { alias: "score", fn: computeScore },
  ]);

  // 4. TopK: keep top 20 by score (heap-based, O(K) memory)
  const top20 = new TopKOperator(scored, "score", true, 20);

  // 5. Join: enrich with user data (right side = build side)
  const userSource = new MockOperator(users);
  const enriched = new HashJoinOperator(
    top20, userSource, "user_id", "user_id", "inner",
  );

  // Pull results
  const rows = await drainPipeline(enriched);

  console.log(`Top ${rows.length} scored orders (enriched with user data):\n`);
  for (const row of rows.slice(0, 5)) {
    console.log(`  Order #${row.order_id} | ${row.name} (${row.tier}) | amount: ${row.amount} | score: ${(row.score as number).toFixed(1)}`);
  }
  console.log(`  ... (${rows.length} total)\n`);
  console.log("Key insight: scoring ran IN the pipeline — no data left the process.");
  console.log("DuckDB/Polars UDFs serialize data across the engine boundary; this doesn't.");
}

main().catch(console.error);
