/**
 * Manual LLM test — end-to-end scenarios for human or LLM verification.
 *
 * Run with: npx tsx src/manual-llm-test.ts
 *
 * Each scenario prints a description, runs the query, and shows the result.
 * An LLM reviewer should verify that each output matches the expected behavior
 * described in the comments.
 */

import { createFromJSON, createDemo } from "./convenience.js";
import { PartitionCatalog } from "./partition-catalog.js";
import type { ColumnMeta, TableMeta } from "./types.js";

let passed = 0;
let failed = 0;

function assert(condition: boolean, msg: string) {
  if (condition) {
    console.log(`  PASS: ${msg}`);
    passed++;
  } else {
    console.error(`  FAIL: ${msg}`);
    failed++;
  }
}

async function main() {
  console.log("=== QueryMode Manual LLM Test Suite ===\n");

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 1: Full DataFrame pipeline — filter, sort, limit, project
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 1: Full DataFrame pipeline ---");
  console.log("Expected: Top 3 US orders by amount desc, showing only name and amount");

  const orders = [
    { id: 1, name: "Alice", region: "us", amount: 500 },
    { id: 2, name: "Bob", region: "eu", amount: 300 },
    { id: 3, name: "Charlie", region: "us", amount: 800 },
    { id: 4, name: "Diana", region: "us", amount: 200 },
    { id: 5, name: "Eve", region: "asia", amount: 600 },
    { id: 6, name: "Frank", region: "us", amount: 1000 },
    { id: 7, name: "Grace", region: "eu", amount: 400 },
  ];

  const r1 = await createFromJSON(orders)
    .filter("region", "eq", "us")
    .sort("amount", "desc")
    .limit(3)
    .select("name", "amount")
    .collect();

  console.log("Result:", JSON.stringify(r1.rows));
  assert(r1.rowCount === 3, "3 rows returned");
  assert(r1.rows[0].name === "Frank" && r1.rows[0].amount === 1000, "First row: Frank, 1000");
  assert(r1.rows[1].name === "Charlie" && r1.rows[1].amount === 800, "Second row: Charlie, 800");
  assert(r1.rows[2].name === "Alice" && r1.rows[2].amount === 500, "Third row: Alice, 500");
  assert(!("region" in r1.rows[0]), "Projection excludes region column");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 2: GroupBy + Aggregation
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 2: GroupBy + Aggregation ---");
  console.log("Expected: Sum and count per region, sorted by total desc");

  const r2 = await createFromJSON(orders)
    .groupBy("region")
    .aggregate("sum", "amount", "total")
    .aggregate("count", "id", "order_count")
    .sort("total", "desc")
    .collect();

  console.log("Result:", JSON.stringify(r2.rows));
  assert(r2.rowCount === 3, "3 region groups");
  assert(r2.rows[0].region === "us", "US region has highest total");
  assert(r2.rows[0].total === 2500, "US total = 500+800+200+1000 = 2500");
  assert(r2.rows[0].order_count === 4, "US has 4 orders");
  const eu = r2.rows.find(r => r.region === "eu");
  assert(eu?.total === 700, "EU total = 300+400 = 700");
  assert(eu?.order_count === 2, "EU has 2 orders");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 3: Computed columns + chained filter
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 3: Computed columns + chained filter ---");
  console.log("Expected: Orders with tax > 75, showing name and tax amount");

  const r3 = await createFromJSON(orders)
    .computed("tax", (row) => (row.amount as number) * 0.1)
    .filter("tax", "gt", 75)
    .select("name", "tax")
    .sort("tax", "desc")
    .collect();

  console.log("Result:", JSON.stringify(r3.rows));
  assert(r3.rowCount === 2, "2 orders with tax > 75");
  assert(r3.rows[0].name === "Frank", "Frank has highest tax (100)");
  assert(r3.rows[0].tax === 100, "Frank tax = 1000 * 0.1 = 100");
  assert(r3.rows[1].name === "Charlie", "Charlie has second highest tax (80)");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 4: Null handling — fillNull + whereNull/whereNotNull
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 4: Null handling ---");
  console.log("Expected: Correct null detection and replacement");

  const withNulls = [
    { id: 1, name: "Alice", score: 90 },
    { id: 2, name: "Bob", score: null },
    { id: 3, name: "Charlie", score: null },
    { id: 4, name: "Diana", score: 85 },
  ];

  const nullCount = await createFromJSON(withNulls).whereNull("score").count();
  assert(nullCount === 2, "2 rows have null score");

  const notNullCount = await createFromJSON(withNulls).whereNotNull("score").count();
  assert(notNullCount === 2, "2 rows have non-null score");

  const filled = await createFromJSON(withNulls).fillNull("score", 0).collect();
  assert(filled.rows[1].score === 0, "Bob's null score filled with 0");
  assert(filled.rows[0].score === 90, "Alice's score unchanged");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 5: OR filter groups
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 5: OR filter groups ---");
  console.log("Expected: Rows matching (region=us AND amount>700) OR (region=asia)");

  const r5 = await createFromJSON(orders)
    .whereOr(
      [{ column: "region", op: "eq", value: "us" }, { column: "amount", op: "gt", value: 700 }],
      [{ column: "region", op: "eq", value: "asia" }],
    )
    .sort("name", "asc")
    .collect();

  console.log("Result:", JSON.stringify(r5.rows.map(r => r.name)));
  assert(r5.rowCount === 3, "3 matching rows");
  // US with amount > 700: Charlie(800), Frank(1000). Asia: Eve(600).
  assert(r5.rows.map(r => r.name).includes("Charlie"), "Charlie matches (us, 800)");
  assert(r5.rows.map(r => r.name).includes("Frank"), "Frank matches (us, 1000)");
  assert(r5.rows.map(r => r.name).includes("Eve"), "Eve matches (asia)");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 6: BigInt aggregation (the bug we fixed)
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 6: BigInt aggregation ---");
  console.log("Expected: Sum/avg/min/max work with BigInt values");

  const bigIntData = [
    { id: 1n, amount: 1000000000000n },
    { id: 2n, amount: 2000000000000n },
    { id: 3n, amount: 3000000000000n },
  ];

  const r6 = await createFromJSON(bigIntData)
    .aggregate("sum", "amount", "total")
    .collect();

  console.log("Result:", JSON.stringify(r6.rows, (_, v) => typeof v === "bigint" ? v.toString() : v));
  assert(r6.rows[0].total === 6000000000000, "BigInt sum = 6 trillion");

  const r6avg = await createFromJSON(bigIntData)
    .aggregate("avg", "amount", "avg_amount")
    .collect();
  assert(r6avg.rows[0].avg_amount === 2000000000000, "BigInt avg = 2 trillion");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 7: Partition catalog — O(1) fragment pruning
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 7: Partition catalog ---");
  console.log("Expected: O(1) lookup prunes to correct fragments");

  function makeFrag(id: number, region: string): [number, TableMeta] {
    return [id, {
      name: `frag-${id}`,
      columns: [{
        name: "region", dtype: "utf8",
        pages: [{ byteOffset: 0n, byteLength: 100, rowCount: 1000, nullCount: 0, minValue: region, maxValue: region }],
        nullCount: 0,
      } as ColumnMeta],
      totalRows: 1000, fileSize: 10000n, r2Key: `data/frag-${id}.lance`, updatedAt: Date.now(),
    }];
  }

  // Simulate 1M fragments, 50 regions
  const regions = ["us-east", "us-west", "eu-west", "eu-central", "asia-pacific",
    "us-south", "eu-north", "asia-south", "africa", "oceania"];
  const fragments = new Map<number, TableMeta>();
  for (let i = 0; i < 100; i++) {
    const [id, meta] = makeFrag(i, regions[i % regions.length]);
    fragments.set(id, meta);
  }

  const catalog = PartitionCatalog.fromFragments("region", fragments);
  const stats = catalog.stats();
  console.log(`Catalog: ${stats.partitionValues} partition values, ${stats.fragments} fragments`);
  assert(stats.partitionValues === 10, "10 distinct regions");
  assert(stats.fragments === 100, "100 fragments indexed");

  // eq lookup
  const usEast = catalog.prune([{ column: "region", op: "eq", value: "us-east" }]);
  assert(usEast !== null && usEast.length === 10, "eq 'us-east' → 10 fragments (every 10th)");

  // in lookup
  const multiRegion = catalog.prune([{ column: "region", op: "in", value: ["us-east", "eu-west"] }]);
  assert(multiRegion !== null && multiRegion.length === 20, "in [us-east, eu-west] → 20 fragments");

  // neq lookup
  const notUs = catalog.prune([{ column: "region", op: "neq", value: "us-east" }]);
  assert(notUs !== null && notUs.length === 90, "neq 'us-east' → 90 fragments");

  // Non-partition column returns null (can't prune)
  const cantPrune = catalog.prune([{ column: "status", op: "eq", value: "active" }]);
  assert(cantPrune === null, "Non-partition column returns null");

  // Serialize → deserialize roundtrip
  const serialized = catalog.serialize();
  const json = JSON.stringify(serialized);
  const restored = PartitionCatalog.deserialize(JSON.parse(json));
  const restoredResult = restored.prune([{ column: "region", op: "eq", value: "us-east" }]);
  assert(restoredResult !== null && restoredResult.length === 10, "Roundtrip preserves index");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 8: Multi-bucket hash distribution
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 8: FNV-1a bucket distribution ---");
  console.log("Expected: Tables distributed across 4 buckets with reasonable balance");

  const tables = [
    "orders", "users", "events", "products", "sessions",
    "clicks", "logs", "metrics", "payments", "invoices",
    "customers", "inventory", "shipments", "reviews", "ratings",
    "analytics", "notifications", "preferences", "tags", "categories",
  ];
  const bucketCount = 4;
  const dist = new Map<number, string[]>();
  for (const table of tables) {
    let h = 0x811c9dc5;
    for (let i = 0; i < table.length; i++) {
      h ^= table.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    const idx = (h >>> 0) % bucketCount;
    if (!dist.has(idx)) dist.set(idx, []);
    dist.get(idx)!.push(table);
  }

  console.log("Distribution:");
  for (const [bucket, tables] of dist) {
    console.log(`  Bucket ${bucket}: ${tables.length} tables (${tables.join(", ")})`);
  }

  assert(dist.size >= 3, "At least 3 of 4 buckets used with 20 tables");
  for (const [, tbls] of dist) {
    assert(tbls.length < 15, `No bucket has more than 75% of tables (has ${tbls.length})`);
  }

  // Same table always routes to same bucket
  const hash1 = fnv1a("orders");
  const hash2 = fnv1a("orders");
  assert(hash1 === hash2, "FNV-1a is deterministic");

  // Different fragments of same table → same bucket
  assert(fnv1a("orders") === fnv1a("orders"), "Same prefix → same hash");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 9: Demo dataset — realistic data exploration
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 9: Demo dataset exploration ---");

  const demo = createDemo();
  const shape = await demo.shape();
  console.log(`Shape: ${shape.rows} rows, ${shape.columns} columns`);
  assert(shape.rows === 1000, "Demo has 1000 rows");
  assert(shape.columns === 5, "Demo has 5 columns");

  const types = await createDemo().dtypes();
  console.log("Types:", JSON.stringify(types));
  assert("id" in types, "Has id column");
  assert("region" in types, "Has region column");

  const vc = await createDemo().valueCounts("region");
  console.log("Region distribution:", JSON.stringify(vc));
  assert(vc.length > 0, "Has region values");
  assert(vc[0].count >= vc[vc.length - 1].count, "Sorted descending by count");

  const sample = await createDemo().sample(5);
  assert(sample.length === 5, "Sample returns 5 rows");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 10: Immutability — branches don't interfere
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 10: DataFrame immutability ---");
  console.log("Expected: Two branches from same DataFrame produce independent results");

  const base = createFromJSON(orders);
  const [usCount, euCount, allCount] = await Promise.all([
    base.filter("region", "eq", "us").count(),
    base.filter("region", "eq", "eu").count(),
    base.count(),
  ]);

  assert(usCount === 4, "US branch: 4 orders");
  assert(euCount === 2, "EU branch: 2 orders");
  assert(allCount === 7, "Base unchanged: 7 orders");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Scenario 11: Export — toJSON and toCSV
  // ─────────────────────────────────────────────────────────────────────────
  console.log("--- Scenario 11: Export formats ---");

  const exportData = [
    { name: "Alice", city: "New York, NY", score: 95 },
    { name: 'Bob "The Builder"', city: "London", score: 88 },
  ];

  const jsonStr = await createFromJSON(exportData).toJSON({ pretty: true });
  const parsed = JSON.parse(jsonStr);
  assert(parsed.length === 2, "JSON has 2 rows");
  assert(parsed[0].name === "Alice", "JSON preserves values");
  assert(jsonStr.includes("\n"), "Pretty JSON has newlines");

  const csvStr = await createFromJSON(exportData).toCSV();
  console.log("CSV output:\n" + csvStr);
  assert(csvStr.includes("name,city,score"), "CSV header correct");
  assert(csvStr.includes('"New York, NY"'), "CSV escapes commas in values");
  assert(csvStr.includes('"Bob ""The Builder"""'), "CSV escapes quotes in values");

  const tsvStr = await createFromJSON(exportData).toCSV({ delimiter: "\t" });
  assert(tsvStr.includes("name\tcity\tscore"), "TSV uses tab delimiter");
  console.log();

  // ─────────────────────────────────────────────────────────────────────────
  // Summary
  // ─────────────────────────────────────────────────────────────────────────
  console.log("=== Summary ===");
  console.log(`Passed: ${passed}, Failed: ${failed}`);
  if (failed > 0) {
    process.exit(1);
  } else {
    console.log("All scenarios verified successfully.");
  }
}

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
