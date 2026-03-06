#!/usr/bin/env npx tsx
/**
 * Local micro-benchmark — measures core code paths without wrangler dev.
 *
 * Tests the specific optimizations:
 *   1. Null bitmap decode (fast path for 0xFF/0x00 bytes)
 *   2. Coalesce range merging (autoCoalesceGap + coalesceRanges)
 *   3. canSkipPage (page-level filter pushdown)
 *   4. Operator pipeline (ScanOperator → FilterOperator → TopK)
 *   5. In-memory query via MaterializedExecutor
 *
 * Usage: npx tsx scripts/bench-local.ts
 */

import { decodePage, canSkipPage } from "../src/decode.js";
import { coalesceRanges, autoCoalesceGap, type Range } from "../src/coalesce.js";
import { QueryMode } from "../src/local.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function timeIt(name: string, fn: () => void, iterations = 1000): { name: string; totalMs: number; opsPerSec: number; avgUs: number } {
  // Warmup
  for (let i = 0; i < Math.min(50, iterations); i++) fn();

  const start = performance.now();
  for (let i = 0; i < iterations; i++) fn();
  const totalMs = performance.now() - start;
  const avgUs = (totalMs / iterations) * 1000;
  const opsPerSec = Math.round(iterations / (totalMs / 1000));
  return { name, totalMs, opsPerSec, avgUs };
}

async function timeItAsync(name: string, fn: () => Promise<void>, iterations = 100): Promise<{ name: string; totalMs: number; opsPerSec: number; avgUs: number }> {
  // Warmup
  for (let i = 0; i < Math.min(10, iterations); i++) await fn();

  const start = performance.now();
  for (let i = 0; i < iterations; i++) await fn();
  const totalMs = performance.now() - start;
  const avgUs = (totalMs / iterations) * 1000;
  const opsPerSec = Math.round(iterations / (totalMs / 1000));
  return { name, totalMs, opsPerSec, avgUs };
}

type BenchResult = { name: string; totalMs: number; opsPerSec: number; avgUs: number };

function printTable(results: BenchResult[]): void {
  console.log(
    "\n" +
    [
      "Benchmark".padEnd(55),
      "avg".padStart(12),
      "ops/sec".padStart(12),
      "total".padStart(10),
    ].join(" | ")
  );
  console.log("-".repeat(95));
  for (const r of results) {
    const avgStr = r.avgUs < 1000 ? `${r.avgUs.toFixed(1)}µs` : `${(r.avgUs / 1000).toFixed(2)}ms`;
    console.log(
      [
        r.name.padEnd(55),
        avgStr.padStart(12),
        r.opsPerSec.toLocaleString().padStart(12),
        `${r.totalMs.toFixed(0)}ms`.padStart(10),
      ].join(" | ")
    );
  }
  console.log("-".repeat(95));
}

// ---------------------------------------------------------------------------
// 1. Null bitmap decode
// ---------------------------------------------------------------------------

function benchBitmapDecode(): BenchResult[] {
  const results: BenchResult[] = [];
  const rowCount = 100_000;

  // All valid (0xFF bytes) — fast path should fly
  const allValid = new ArrayBuffer(8 + Math.ceil(rowCount / 8) + rowCount * 8);
  new Uint8Array(allValid, 0, Math.ceil(rowCount / 8)).fill(0xFF);
  // Write int64 data after bitmap
  const dv = new DataView(allValid, Math.ceil(rowCount / 8));
  for (let i = 0; i < rowCount; i++) dv.setBigInt64(i * 8, BigInt(i), true);
  results.push(timeIt("bitmap: 100K rows, all valid (0xFF fast path)", () => {
    decodePage(allValid, "int64", 0, rowCount);
  }, 200));

  // 50% null (alternating 0xAA bytes)
  const halfNull = new ArrayBuffer(Math.ceil(rowCount / 8) + rowCount * 8);
  new Uint8Array(halfNull, 0, Math.ceil(rowCount / 8)).fill(0xAA); // 10101010
  results.push(timeIt("bitmap: 100K rows, 50% null (bit-by-bit)", () => {
    decodePage(halfNull, "int64", rowCount / 2, rowCount);
  }, 200));

  // All null (0x00 bytes) — fast path should batch-add
  const allNull = new ArrayBuffer(Math.ceil(rowCount / 8) + rowCount * 8);
  new Uint8Array(allNull, 0, Math.ceil(rowCount / 8)).fill(0x00);
  results.push(timeIt("bitmap: 100K rows, all null (0x00 fast path)", () => {
    decodePage(allNull, "int64", rowCount, rowCount);
  }, 200));

  return results;
}

// ---------------------------------------------------------------------------
// 2. Coalesce ranges
// ---------------------------------------------------------------------------

function benchCoalesce(): BenchResult[] {
  const results: BenchResult[] = [];

  // Dense ranges (small gaps — should merge aggressively)
  const denseRanges: Range[] = [];
  for (let i = 0; i < 500; i++) {
    denseRanges.push({ column: `col${i % 5}`, offset: i * 8200, length: 8000 });
  }
  results.push(timeIt("coalesce: 500 dense ranges (200B gaps)", () => {
    const gap = autoCoalesceGap(denseRanges);
    coalesceRanges(denseRanges, gap);
  }, 5000));

  // Sparse ranges (large gaps — should keep separate)
  const sparseRanges: Range[] = [];
  for (let i = 0; i < 500; i++) {
    sparseRanges.push({ column: `col${i % 5}`, offset: i * 1_000_000, length: 8000 });
  }
  results.push(timeIt("coalesce: 500 sparse ranges (992KB gaps)", () => {
    const gap = autoCoalesceGap(sparseRanges);
    coalesceRanges(sparseRanges, gap);
  }, 5000));

  // Mixed — realistic scenario
  const mixedRanges: Range[] = [];
  for (let i = 0; i < 200; i++) {
    // Clustered in groups of 10
    const group = Math.floor(i / 10);
    const inGroup = i % 10;
    mixedRanges.push({ column: `col${i % 3}`, offset: group * 500_000 + inGroup * 10_000, length: 8000 });
  }
  results.push(timeIt("coalesce: 200 mixed ranges (clustered)", () => {
    const gap = autoCoalesceGap(mixedRanges);
    coalesceRanges(mixedRanges, gap);
  }, 5000));

  return results;
}

// ---------------------------------------------------------------------------
// 3. canSkipPage (page-level filter pushdown)
// ---------------------------------------------------------------------------

function benchCanSkipPage(): BenchResult[] {
  const results: BenchResult[] = [];
  const pages = Array.from({ length: 100 }, (_, i) => ({
    byteOffset: BigInt(i * 80000),
    byteLength: 80000,
    rowCount: 10000,
    minValue: i * 10000,
    maxValue: (i + 1) * 10000 - 1,
  }));

  const filters = [{ column: "id", op: "gt" as const, value: 500000 }];

  results.push(timeIt("canSkipPage: 100 pages × gt filter (50% skip)", () => {
    let skipped = 0;
    for (const page of pages) {
      if (canSkipPage(page, filters, "id")) skipped++;
    }
  }, 10000));

  const rangeFilters = [
    { column: "id", op: "gte" as const, value: 200000 },
    { column: "id", op: "lt" as const, value: 800000 },
  ];
  results.push(timeIt("canSkipPage: 100 pages × range filter (40% skip)", () => {
    let skipped = 0;
    for (const page of pages) {
      if (canSkipPage(page, rangeFilters, "id")) skipped++;
    }
  }, 10000));

  return results;
}

// ---------------------------------------------------------------------------
// 4. In-memory query pipeline (MaterializedExecutor via fromJSON)
// ---------------------------------------------------------------------------

async function benchInMemoryQuery(): Promise<BenchResult[]> {
  const results: BenchResult[] = [];

  // Generate 10K rows
  const data = Array.from({ length: 10_000 }, (_, i) => ({
    id: i,
    value: Math.random() * 1000,
    category: ["alpha", "beta", "gamma", "delta", "epsilon"][i % 5],
    region: ["us", "eu", "asia"][i % 3],
  }));

  const qm = QueryMode.fromJSON(data, "bench_data");

  // Full scan
  results.push(await timeItAsync("query: 10K full scan", async () => {
    await qm.select("id", "value", "category").collect();
  }, 500));

  // Filter + collect
  results.push(await timeItAsync("query: 10K filter id>5000 (50% sel)", async () => {
    await qm.filter("id", "gt", 5000).collect();
  }, 500));

  // Filter + sort + limit (TopK)
  results.push(await timeItAsync("query: 10K filter+sort+limit(10)", async () => {
    await qm.filter("id", "gt", 5000).sort("value", "desc").limit(10).collect();
  }, 500));

  // Projection only
  results.push(await timeItAsync("query: 10K project 1 of 4 cols", async () => {
    await qm.select("id").collect();
  }, 500));

  // 100K rows
  const bigData = Array.from({ length: 100_000 }, (_, i) => ({
    id: i,
    amount: Math.random() * 10000,
    category: ["A", "B", "C", "D", "E"][i % 5],
  }));
  const bigQm = QueryMode.fromJSON(bigData, "big_bench");

  results.push(await timeItAsync("query: 100K full scan", async () => {
    await bigQm.collect();
  }, 50));

  results.push(await timeItAsync("query: 100K filter+sort+limit(100)", async () => {
    await bigQm.filter("id", "gt", 50000).sort("amount", "desc").limit(100).collect();
  }, 100));

  results.push(await timeItAsync("query: 100K filter id>90000 (10% sel)", async () => {
    await bigQm.filter("id", "gt", 90000).collect();
  }, 100));

  return results;
}

// ---------------------------------------------------------------------------
// 5. Int64 decode (pure decode path)
// ---------------------------------------------------------------------------

function benchInt64Decode(): BenchResult[] {
  const results: BenchResult[] = [];

  // 100K int64 values, no nulls
  const rowCount = 100_000;
  const buf = new ArrayBuffer(rowCount * 8);
  const dv = new DataView(buf);
  for (let i = 0; i < rowCount; i++) dv.setBigInt64(i * 8, BigInt(i), true);

  results.push(timeIt("decode: 100K int64 values (no nulls)", () => {
    decodePage(buf, "int64", 0, rowCount);
  }, 200));

  // 100K float64 values
  const fbuf = new ArrayBuffer(rowCount * 8);
  const fdv = new DataView(fbuf);
  for (let i = 0; i < rowCount; i++) fdv.setFloat64(i * 8, i * 1.5, true);

  results.push(timeIt("decode: 100K float64 values (no nulls)", () => {
    decodePage(fbuf, "float64", 0, rowCount);
  }, 200));

  return results;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("QueryMode Local Micro-Benchmark");
  console.log(`Node ${process.version} | ${process.platform} ${process.arch}`);
  console.log("=".repeat(95));

  const allResults: BenchResult[] = [];

  console.log("\n## Bitmap Decode");
  allResults.push(...benchBitmapDecode());

  console.log("\n## Int64/Float64 Decode");
  allResults.push(...benchInt64Decode());

  console.log("\n## Coalesce Ranges");
  allResults.push(...benchCoalesce());

  console.log("\n## Page Skip (canSkipPage)");
  allResults.push(...benchCanSkipPage());

  console.log("\n## In-Memory Query Pipeline");
  allResults.push(...await benchInMemoryQuery());

  console.log("\n\n" + "=".repeat(95));
  console.log("FULL RESULTS");
  printTable(allResults);
}

main().catch(console.error);
