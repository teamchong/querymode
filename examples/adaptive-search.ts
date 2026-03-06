/**
 * Adaptive Vector Search — dynamically widen search when too few results match.
 *
 * Traditional query planners have a fixed execution plan. Here, the pipeline
 * is recomposed at runtime: if the initial distance threshold yields too few
 * results, we widen it and search again. Impossible with a fixed query planner.
 *
 * Pipeline (per iteration):
 *   MockOperator(vectors) → VectorTopKOperator(top-50) → FilterOperator(_distance < threshold)
 */

import {
  FilterOperator,
  TopKOperator,
  drainPipeline,
  type Operator,
  type RowBatch,
} from "../src/operators.js";
import { cosineDistance } from "../src/hnsw.js";
import type { Row } from "../src/types.js";

// ─── Mock vector data ───────────────────────────────────────────────────

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

/**
 * Custom operator: compute cosine distance for each row and add _distance column.
 * This is the kind of operator you can compose freely in QueryMode.
 */
class VectorDistanceOperator implements Operator {
  private upstream: Operator;
  private column: string;
  private queryVec: Float32Array;
  constructor(upstream: Operator, column: string, queryVec: Float32Array) {
    this.upstream = upstream;
    this.column = column;
    this.queryVec = queryVec;
  }
  async next(): Promise<RowBatch | null> {
    const batch = await this.upstream.next();
    if (!batch) return null;
    return batch.map(row => ({
      ...row,
      _distance: cosineDistance(row[this.column] as Float32Array, this.queryVec),
    }));
  }
  async close() { await this.upstream.close(); }
}

// Deterministic PRNG
let prngState = 12345;
function xorshift32(): number {
  prngState ^= prngState << 13;
  prngState ^= prngState >>> 17;
  prngState ^= prngState << 5;
  return (prngState >>> 0) / 0xFFFFFFFF;
}

// Generate 200 items with 8-dim embeddings
const DIM = 8;
const items: Row[] = Array.from({ length: 200 }, (_, i) => {
  const embedding = new Float32Array(DIM);
  for (let d = 0; d < DIM; d++) embedding[d] = xorshift32() * 2 - 1;
  // Normalize
  let norm = 0;
  for (let d = 0; d < DIM; d++) norm += embedding[d] * embedding[d];
  norm = Math.sqrt(norm);
  for (let d = 0; d < DIM; d++) embedding[d] /= norm;
  return { id: i + 1, label: `item_${i + 1}`, embedding };
});

// Query vector (normalized)
const queryVec = new Float32Array(DIM);
for (let d = 0; d < DIM; d++) queryVec[d] = xorshift32() * 2 - 1;
let qNorm = 0;
for (let d = 0; d < DIM; d++) qNorm += queryVec[d] * queryVec[d];
qNorm = Math.sqrt(qNorm);
for (let d = 0; d < DIM; d++) queryVec[d] /= qNorm;

// ─── Adaptive search ────────────────────────────────────────────────────

const MIN_RESULTS = 10;
const INITIAL_THRESHOLD = 0.3;
const WIDEN_STEP = 0.15;
const MAX_THRESHOLD = 1.5;

async function adaptiveSearch(): Promise<Row[]> {
  let threshold = INITIAL_THRESHOLD;

  while (threshold <= MAX_THRESHOLD) {
    console.log(`  Searching with distance threshold: ${threshold.toFixed(2)}`);

    // Recompose pipeline with current threshold
    const source = new MockOperator(items);
    const withDistance = new VectorDistanceOperator(source, "embedding", queryVec);
    const topK = new TopKOperator(withDistance, "_distance", false, 50);
    const filtered = new FilterOperator(topK, [
      { column: "_distance", op: "lt", value: threshold },
    ]);

    const results = await drainPipeline(filtered);
    console.log(`  → Found ${results.length} results`);

    if (results.length >= MIN_RESULTS) {
      return results;
    }

    // Widen threshold and recompose
    threshold += WIDEN_STEP;
    console.log(`  Too few results, widening threshold...\n`);
  }

  // Final pass with no distance filter — just top 50
  console.log("  Final pass: returning all top-50 results");
  const source = new MockOperator(items);
  const withDistance = new VectorDistanceOperator(source, "embedding", queryVec);
  const topK = new TopKOperator(withDistance, "_distance", false, 50);
  return drainPipeline(topK);
}

async function main() {
  console.log("Adaptive Vector Search\n");
  console.log("Pipeline is recomposed at runtime based on result quality.\n");

  const results = await adaptiveSearch();

  console.log(`\nFinal results: ${results.length} items`);
  for (const row of results.slice(0, 5)) {
    console.log(`  ${row.label} | distance: ${(row._distance as number).toFixed(4)}`);
  }
  console.log(`\nKey insight: the pipeline was dynamically recomposed based on result count.`);
  console.log("A fixed query planner cannot do this — the plan IS the execution.");
}

main().catch(console.error);
