/**
 * Custom Spill Backend — plug your own storage into sort and join operators.
 *
 * This example implements an InMemorySpillBackend that satisfies the SpillBackend
 * interface. It's used with ExternalSortOperator and HashJoinOperator at a tiny
 * 4KB memory budget to force spilling.
 *
 * DuckDB: disk-only spill. Polars: no spill at all.
 * QueryMode: any SpillBackend — R2, disk, memory, S3, you decide.
 */

import {
  ExternalSortOperator,
  HashJoinOperator,
  drainPipeline,
  type Operator,
  type RowBatch,
} from "../src/operators.js";
import type { SpillBackend } from "../src/r2-spill.js";
import type { Row } from "../src/types.js";

// ─── In-memory spill backend ────────────────────────────────────────────

class InMemorySpillBackend implements SpillBackend {
  private runs = new Map<string, Row[]>();
  private nextId = 0;
  bytesWritten = 0;
  bytesRead = 0;

  async writeRun(rows: Row[]): Promise<string> {
    const id = `mem-run-${this.nextId++}`;
    const copy = rows.map(r => ({ ...r }));
    this.runs.set(id, copy);
    const size = rows.length * 100; // rough estimate
    this.bytesWritten += size;
    return id;
  }

  async *streamRun(spillId: string): AsyncGenerator<Row> {
    const rows = this.runs.get(spillId);
    if (!rows) throw new Error(`Spill run not found: ${spillId}`);
    const size = rows.length * 100;
    this.bytesRead += size;
    for (const row of rows) yield row;
  }

  async cleanup(): Promise<void> {
    this.runs.clear();
  }
}

// ─── Mock data source ───────────────────────────────────────────────────

class MockOperator implements Operator {
  private rows: Row[];
  private cursor = 0;
  private batchSize: number;
  constructor(rows: Row[], batchSize = 50) {
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

// ─── Demo ───────────────────────────────────────────────────────────────

async function main() {
  console.log("Custom Spill Backend Demo\n");

  // Generate 200 orders and 100 users
  const orders: Row[] = Array.from({ length: 200 }, (_, i) => ({
    order_id: i + 1,
    user_id: (i % 100) + 1,
    amount: ((i * 7 + 13) % 1000) + 1,
  }));

  const users: Row[] = Array.from({ length: 100 }, (_, i) => ({
    user_id: i + 1,
    name: `User_${i + 1}`,
  }));

  const spill = new InMemorySpillBackend();
  const TINY_BUDGET = 4 * 1024; // 4KB — forces spilling

  // 1. External sort with custom spill backend
  console.log("1. External sort (4KB budget, forces spill to memory backend):");
  const sortSource = new MockOperator(orders);
  const sorted = new ExternalSortOperator(
    sortSource, "amount", true, 0, TINY_BUDGET, spill,
  );
  const sortedRows = await drainPipeline(sorted);
  console.log(`   Sorted ${sortedRows.length} rows by amount desc`);
  console.log(`   Spill stats: ${spill.bytesWritten} bytes written, ${spill.bytesRead} bytes read`);
  console.log(`   Top 3: ${sortedRows.slice(0, 3).map(r => r.amount).join(", ")}\n`);

  // Reset spill
  await spill.cleanup();
  spill.bytesWritten = 0;
  spill.bytesRead = 0;

  // 2. Hash join with custom spill backend
  console.log("2. Hash join (4KB budget, forces Grace hash partitioning):");
  const leftSource = new MockOperator(orders);
  const rightSource = new MockOperator(users);
  const joined = new HashJoinOperator(
    leftSource, rightSource, "user_id", "user_id", "inner",
    TINY_BUDGET, spill,
  );
  const joinedRows = await drainPipeline(joined);
  console.log(`   Joined ${joinedRows.length} rows`);
  console.log(`   Spill stats: ${spill.bytesWritten} bytes written, ${spill.bytesRead} bytes read`);
  console.log(`   Sample: order #${joinedRows[0]?.order_id} → ${joinedRows[0]?.name}\n`);

  await spill.cleanup();

  console.log("Key insight: spill storage is pluggable — R2, disk, memory, S3.");
  console.log("DuckDB: disk only. Polars: no spill at all.");
  console.log("QueryMode: implement SpillBackend and plug it in.");
}

main().catch(console.error);
